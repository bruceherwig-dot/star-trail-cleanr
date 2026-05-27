"""
Trail repair — Star Bridge: per-trail sparse optical flow morph from N-1/N+1.

THE CORE PROBLEM
----------------
We need to fill in the pixels that a trail was covering in frame N. We can't just
copy pixels from a static background because the sky is not static — stars move in
arcs across the frame over the course of a multi-hour session. A star that is at
position (x, y) in frame N-1 will be at a slightly different position in frame N+1.
Copying neighbor pixels directly without accounting for that motion would smear every
star in the repaired region.

HOW STAR BRIDGE WORKS
----------------------
For each detected trail, Star Bridge measures the local star motion at that specific
region of the frame using sparse Lucas-Kanade optical flow:

1. Find bright corner features (stars) in the prev frame (N-1) within a padded
   bounding box around the trail. Mask out the trail pixels so the tracker only
   latches onto real stars, not the trail itself.

2. Track those star points forward to the next frame (N+1). Discard any tracked
   points with implausible displacements (too small = not moving stars, too large =
   tracking noise). The median displacement of the surviving points is the local
   star motion vector (dx, dy) from N-1 to N+1.

3. Shift the prev frame (N-1) FORWARD by half that motion (dx/2, dy/2), and shift
   the next frame (N+1) BACKWARD by half that motion (-dx/2, -dy/2). Both neighbors
   now have their stars aligned to where they would be in frame N.

4. Average the two shifted neighbors. Paste the averaged pixels into frame N at the
   trail mask locations only. The rest of frame N is untouched.

WHY BLACK FILL IS SAFE
-----------------------
When star tracking fails (too few stars, implausible displacements, first/last frame
with only one neighbor), the trail pixels are filled with pure black — zero in all
channels. This is safe because the final output is a lighten-max composite: each
pixel in the stack takes the brightest value across all frames. A zero pixel loses
to any real star pixel in any other frame. The hole becomes invisible in the final
image, even if the repair was imperfect.

LONG TRAIL SEGMENTATION
------------------------
Trails longer than MAX_SEG_LENGTH pixels are split into overlapping segments before
repair. This matters because star motion is not perfectly uniform across a wide-angle
frame — there is some field distortion and the motion vector at the left edge of a
3000px trail may differ from the vector at the right edge. Shorter segments each get
their own local motion estimate, producing more accurate repair than a single global
estimate for the whole trail.

FALLBACKS IN ORDER
------------------
1. Star Bridge (two neighbors, LK tracking succeeds)      -> best quality
2. Single-neighbor copy (first or last frame of batch)    -> good quality
3. Single-neighbor LK (tracking fails, one neighbor)      -> shifted neighbor, no average
4. Black fill (tracking fails, no usable neighbors)       -> invisible in lighten-max
"""
import math
import cv2
import numpy as np


# ── Tuning constants ──────────────────────────────────────────────────────────
# PAD: pixels of context around each trail bounding box used for star tracking.
# MIN_AREA: components smaller than this (px) are skipped -- likely noise.
# MAX_SEG_LENGTH: long components are split into segments of this max length.
# MIN_DISP/MAX_DISP: plausible star displacement range N-1 to N+1 in pixels.
# MIN_STARS: minimum tracked stars needed to trust the shift estimate.
# TRAIL_WARM_MARGIN/TRAIL_BRIGHT_THRESH: warm-pixel cleanup gate after warp.
# Log fields: seg.method reflects which of these paths fired per segment.

PAD            = 120   # pixels around each trail bbox for feature search
MIN_AREA       = 500   # skip tiny mask components (noise)
MAX_SEG_LENGTH = 500   # components longer than this are split for tighter repair
MIN_DISP  = 1.0   # minimum plausible star displacement N-1 to N+1 (px)
MAX_DISP  = 60.0  # maximum plausible star displacement N-1 to N+1 (px)
MIN_STARS = 5     # minimum tracked stars needed to trust the shift
TRAIL_WARM_MARGIN   = 20  # how much warmer than local sky R-B counts as a trail remnant
TRAIL_BRIGHT_THRESH = 50  # minimum R value to be considered trail-bright

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


# ── Image helpers ─────────────────────────────────────────────────────────────
# _to_8bit: converts 16-bit frames to 8-bit for Lucas-Kanade (LK requires uint8).
# _shift_image: applies a fractional-pixel translation via warpAffine with reflect
#   border padding so stars near the trail edge are not blacked out by the warp.

def _to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return (img / 257).astype(np.uint8)
    return img


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ── Star feature tracking (Lucas-Kanade sparse optical flow) ──────────────────
# Finds bright corners in the prev patch (masking trail pixels), tracks them to
# the next patch with LK pyramidal optical flow, and returns the median shift.
# Returns (dx, dy, success, n_stars_tracked). n_stars_tracked is the count of
# valid tracked points that passed the displacement plausibility gate.
# Failure reasons: too few feature points found, too few tracked, or all
# displacements outside [MIN_DISP, MAX_DISP]. Log field: seg.n_stars.

def _track_stars(prev: np.ndarray, nxt: np.ndarray, trail_mask=None):
    """Track stars from prev to nxt. Returns (dx, dy, success, n_stars_tracked).

    trail_mask: boolean array same shape as patch. When provided, those pixels
    are zeroed in the search image so the tracker ignores trail features and
    finds only stars.
    """
    g_prev = cv2.cvtColor(_to_8bit(prev), cv2.COLOR_BGR2GRAY)
    g_next = cv2.cvtColor(_to_8bit(nxt),  cv2.COLOR_BGR2GRAY)

    if trail_mask is not None and trail_mask.any():
        g_search = g_prev.copy()
        g_search[trail_mask] = 0
    else:
        g_search = g_prev

    pts = cv2.goodFeaturesToTrack(
        g_search, maxCorners=500, qualityLevel=0.005,
        minDistance=5, blockSize=7
    )
    if pts is None or len(pts) < MIN_STARS:
        return 0.0, 0.0, False, 0

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_next, pts, None, **_LK_PARAMS)
    good = (status.ravel() == 1)
    if good.sum() < MIN_STARS:
        return 0.0, 0.0, False, int(good.sum())

    disp = (pts1[good] - pts[good]).reshape(-1, 2)
    mag  = np.linalg.norm(disp, axis=1)
    valid = (mag >= MIN_DISP) & (mag <= MAX_DISP)
    if valid.sum() < MIN_STARS:
        return 0.0, 0.0, False, int(valid.sum())

    return (float(np.median(disp[valid, 0])),
            float(np.median(disp[valid, 1])),
            True,
            int(valid.sum()))


# ── Component splitting (long trails into sub-segments) ───────────────────────
# Very long trails (> MAX_SEG_LENGTH px) are split into equal-length segments
# along the major axis so star tracking has a smaller, more homogeneous patch.
# Each segment gets its own tracking + warp pass.
# Log field: comp.split_into = number of segments returned.

def _split_component(comp_full: np.ndarray) -> list:
    """Split a full-frame boolean component mask into sub-masks along the major axis.

    If the major axis length exceeds MAX_SEG_LENGTH, splits into
    ceil(length / MAX_SEG_LENGTH) equal rectangle segments. Each sub-mask
    contains only the original component pixels that fall inside that segment's
    bounding rectangle. Returns a list of uint8 masks.
    """
    ys, xs = np.where(comp_full)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    trail_len = float(max(rect[1]))

    if trail_len <= MAX_SEG_LENGTH:
        m = np.zeros(comp_full.shape, dtype=np.uint8)
        m[comp_full] = 255
        return [m]

    n_segs = math.ceil(trail_len / MAX_SEG_LENGTH)
    box = cv2.boxPoints(rect)
    e01 = np.linalg.norm(box[1] - box[0])
    e12 = np.linalg.norm(box[2] - box[1])
    if e01 >= e12:
        a0, a1, b0, b1 = box[0], box[1], box[3], box[2]
    else:
        a0, a1, b0, b1 = box[1], box[2], box[0], box[3]

    H, W = comp_full.shape
    result = []
    for si in range(n_segs):
        t0, t1 = si / n_segs, (si + 1) / n_segs
        corners = np.array([
            a0 + t0 * (a1 - a0), a0 + t1 * (a1 - a0),
            b0 + t1 * (b1 - b0), b0 + t0 * (b1 - b0),
        ], dtype=np.int32)
        seg = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(seg, [corners.reshape(-1, 1, 2)], 255)
        seg[~comp_full] = 0  # keep only actual trail pixels
        if seg.any():
            result.append(seg)

    return result if result else [np.uint8(comp_full) * 255]


def repair_frame(frame: np.ndarray, mask: np.ndarray,
                 frame_idx: int,
                 neighbor_frames: list,
                 neighbor_masks: list = None,
                 debug_out=None) -> np.ndarray:
    """Replace masked trail pixels using Star Bridge sparse-track morph repair.

    Args:
        frame: original image (uint8 or uint16)
        mask: binary uint8 mask (255=trail, 0=sky) for this frame
        frame_idx: index of this frame in neighbor_frames
        neighbor_frames: full list of frames (same dtype as frame)
        neighbor_masks: optional list of mask arrays aligned with neighbor_frames
            (None entries = no mask / assume clean). When provided, a neighbor
            is skipped for any component where its mask overlaps that component,
            since its pixels there are trail, not sky.
        debug_out (dict, optional): filled with a "components" list. Each entry
            has id, area, bbox, split_into, and a "segments" list. Each segment
            has tracking_ok, dx, dy, n_stars, method, still_trail_px,
            union_zeroed_px.
    Returns:
        Repaired copy of frame.
    """
    result = frame.copy()
    trail = mask > 0
    if not trail.any():
        return result

    if debug_out is not None:
        debug_out["components"] = []

    H, W = mask.shape[:2]
    N = len(neighbor_frames)

    prev_idx = frame_idx - 1 if frame_idx > 0 else None
    next_idx = frame_idx + 1 if frame_idx < N - 1 else None
    has_prev = prev_idx is not None
    has_next = next_idx is not None

    if not has_prev and not has_next:
        return result

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        trail.astype(np.uint8))

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue

        # ── Per-component setup ───────────────────────────────────────────────
        comp_area = int(stats[i, cv2.CC_STAT_AREA])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        comp_full = (labels == i)
        sub_masks = _split_component(comp_full)

        comp_dbg = None
        if debug_out is not None:
            comp_dbg = {
                "id": i,
                "area": comp_area,
                "bbox": [bx, by, bx + bw, by + bh],
                "split_into": len(sub_masks),
                "segments": [],
            }

        for seg_idx, sub_mask in enumerate(sub_masks):
            seg_info = {"seg": seg_idx} if comp_dbg is not None else None

            sub_ys, sub_xs = np.where(sub_mask)
            x0 = max(0, int(sub_xs.min()) - PAD)
            y0 = max(0, int(sub_ys.min()) - PAD)
            x1 = min(W, int(sub_xs.max()) + PAD)
            y1 = min(H, int(sub_ys.max()) + PAD)

            comp_mask = sub_mask[y0:y1, x0:x1] > 0

            # Always repair from all available neighbors.
            # The union mask below blacks out any pixel where a neighbor has trail,
            # so overlap areas are handled pixel-by-pixel after repair.
            use_prev = has_prev
            use_next = has_next

            if not use_prev and not use_next:
                result[y0:y1, x0:x1][comp_mask] = 0
                if seg_info is not None:
                    seg_info.update({"method": "black_fill_no_neighbors",
                                     "tracking_ok": False, "dx": 0.0, "dy": 0.0,
                                     "n_stars": 0, "still_trail_px": 0,
                                     "union_zeroed_px": 0})
                    comp_dbg["segments"].append(seg_info)
                continue

            _method = "unknown"
            _dx = 0.0
            _dy = 0.0
            _ok = False
            _n_stars = 0

            if use_prev and use_next:
                # ── Star feature tracking (Lucas-Kanade sparse optical flow) ──
                patch_prev = neighbor_frames[prev_idx][y0:y1, x0:x1]
                patch_next = neighbor_frames[next_idx][y0:y1, x0:x1]
                dx, dy, ok, n_stars = _track_stars(patch_prev, patch_next,
                                                   trail_mask=comp_mask)
                _dx, _dy, _ok, _n_stars = dx, dy, ok, n_stars

                if ok:
                    # ── Warp synthesis ────────────────────────────────────────
                    warped_prev = _shift_image(patch_prev,  dx / 2.0,  dy / 2.0)
                    warped_next = _shift_image(patch_next, -dx / 2.0, -dy / 2.0)
                    if neighbor_masks is None:
                        # No masks provided — fall back to color-based contamination check
                        _CONTAM_THRESH = 0.20
                        def _contam(patch):
                            px = patch[comp_mask].astype(np.int32)
                            return float(np.mean(
                                (px[:, 2] - px[:, 0] > TRAIL_WARM_MARGIN + 10) &
                                (px[:, 2] > TRAIL_BRIGHT_THRESH)
                            ))
                        cp = _contam(warped_prev)
                        cn = _contam(warped_next)
                        if cp <= _CONTAM_THRESH and cn <= _CONTAM_THRESH:
                            synth = ((warped_prev.astype(np.float32) +
                                      warped_next.astype(np.float32)) / 2.0).astype(frame.dtype)
                            _method = "blend"
                        elif cp <= cn:
                            synth = warped_prev.copy()
                            _method = "prev_only"
                        else:
                            synth = warped_next.copy()
                            _method = "next_only"
                    else:
                        # Per-pixel: use only the clean neighbor where one side has trail.
                        prev_c = (neighbor_masks[prev_idx][y0:y1, x0:x1] > 0
                                  if has_prev and neighbor_masks[prev_idx] is not None
                                  else np.zeros(comp_mask.shape, dtype=bool))
                        next_c = (neighbor_masks[next_idx][y0:y1, x0:x1] > 0
                                  if has_next and neighbor_masks[next_idx] is not None
                                  else np.zeros(comp_mask.shape, dtype=bool))
                        synth = ((warped_prev.astype(np.float32) +
                                  warped_next.astype(np.float32)) / 2.0).astype(frame.dtype)
                        use_next_only = comp_mask & prev_c & ~next_c
                        if use_next_only.any():
                            synth[use_next_only] = warped_next[use_next_only]
                        use_prev_only = comp_mask & next_c & ~prev_c
                        if use_prev_only.any():
                            synth[use_prev_only] = warped_prev[use_prev_only]
                        _method = "blend"
                else:
                    synth = np.zeros_like(frame[y0:y1, x0:x1])
                    _method = "black_fill"

            elif use_prev:
                synth = neighbor_frames[prev_idx][y0:y1, x0:x1].copy()
                _method = "prev_only"

            else:
                synth = neighbor_frames[next_idx][y0:y1, x0:x1].copy()
                _method = "next_only"

            result[y0:y1, x0:x1][comp_mask] = synth[comp_mask]

            # ── Warm-pixel cleanup (still-trail remnants after warp) ──────────
            bg_pixels = frame[y0:y1, x0:x1][~comp_mask].astype(np.int32)
            if len(bg_pixels) >= 10:
                bg_rb = float(np.median(bg_pixels[:, 2] - bg_pixels[:, 0]))
            else:
                bg_rb = 0.0
            warm_thresh = bg_rb + TRAIL_WARM_MARGIN

            filled = result[y0:y1, x0:x1].astype(np.int32)
            still_trail = (comp_mask &
                           (filled[..., 2] - filled[..., 0] > warm_thresh) &
                           (filled[..., 2] > TRAIL_BRIGHT_THRESH))
            result[y0:y1, x0:x1][still_trail] = 0
            _still_trail_px = int(still_trail.sum())

            # ── AND union mask: zero pixels contaminated in BOTH neighbors ────
            # Pixels in only one neighbor's trail are already repaired above.
            # Black is transparent in lighten-max stacks so the cost is zero.
            _union_zeroed_px = 0
            if neighbor_masks is not None:
                prev_c = (neighbor_masks[prev_idx][y0:y1, x0:x1] > 0
                          if has_prev and neighbor_masks[prev_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                next_c = (neighbor_masks[next_idx][y0:y1, x0:x1] > 0
                          if has_next and neighbor_masks[next_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                union_both = comp_mask & prev_c & next_c
                result[y0:y1, x0:x1][union_both] = 0
                _union_zeroed_px = int(union_both.sum())

            if seg_info is not None:
                seg_info.update({
                    "tracking_ok":     _ok,
                    "dx":              round(_dx, 2),
                    "dy":              round(_dy, 2),
                    "n_stars":         _n_stars,
                    "method":          _method,
                    "still_trail_px":  _still_trail_px,
                    "union_zeroed_px": _union_zeroed_px,
                })
                comp_dbg["segments"].append(seg_info)

        if debug_out is not None and comp_dbg is not None:
            debug_out["components"].append(comp_dbg)

    return result
