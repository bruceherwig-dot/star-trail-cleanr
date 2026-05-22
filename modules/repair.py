"""Trail repair — Star Bridge: per-trail sparse feature tracking morph from N-1/N+1.

For each trail component, bright star features are tracked from N-1 to N+1
using Lucas-Kanade sparse optical flow. The median displacement gives the
local star motion at that trail's location. N-1 is shifted forward by half
and N+1 backward by half, averaged to synthesize frame N without the trail.
Paste the synthetic pixels into frame N at masked locations only.

Single-neighbor fallback (first/last frame): copy that neighbor directly.
Tracking failure fallback: black fill (transparent in lighten-max stacks).
"""
import math
import cv2
import numpy as np


# --- Tuning constants -------------------------------------------------------
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


def _to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return (img / 257).astype(np.uint8)
    return img


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _track_stars(prev: np.ndarray, nxt: np.ndarray, trail_mask=None):
    """Track stars from prev to nxt. Returns (dx, dy, success).

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
        return 0.0, 0.0, False

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_next, pts, None, **_LK_PARAMS)
    good = (status.ravel() == 1)
    if good.sum() < MIN_STARS:
        return 0.0, 0.0, False

    disp = (pts1[good] - pts[good]).reshape(-1, 2)
    mag  = np.linalg.norm(disp, axis=1)
    valid = (mag >= MIN_DISP) & (mag <= MAX_DISP)
    if valid.sum() < MIN_STARS:
        return 0.0, 0.0, False

    return float(np.median(disp[valid, 0])), float(np.median(disp[valid, 1])), True


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
                 neighbor_masks: list = None) -> np.ndarray:
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
    Returns:
        Repaired copy of frame.
    """
    result = frame.copy()
    trail = mask > 0
    if not trail.any():
        return result

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

        comp_full = (labels == i)
        sub_masks = _split_component(comp_full)

        for sub_mask in sub_masks:
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
                continue

            if use_prev and use_next:
                patch_prev = neighbor_frames[prev_idx][y0:y1, x0:x1]
                patch_next = neighbor_frames[next_idx][y0:y1, x0:x1]
                dx, dy, ok = _track_stars(patch_prev, patch_next, trail_mask=comp_mask)
                if ok:
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
                        elif cp <= cn:
                            synth = warped_prev.copy()
                        else:
                            synth = warped_next.copy()
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
                else:
                    synth = np.zeros_like(frame[y0:y1, x0:x1])

            elif use_prev:
                synth = neighbor_frames[prev_idx][y0:y1, x0:x1].copy()

            else:
                synth = neighbor_frames[next_idx][y0:y1, x0:x1].copy()

            result[y0:y1, x0:x1][comp_mask] = synth[comp_mask]

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

            # AND union mask: zero only pixels contaminated in BOTH neighbors.
            # Pixels in only one neighbor's trail are repaired above from the clean side.
            # Black is transparent in lighten-max stacks so the cost is zero.
            if neighbor_masks is not None:
                prev_c = (neighbor_masks[prev_idx][y0:y1, x0:x1] > 0
                          if has_prev and neighbor_masks[prev_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                next_c = (neighbor_masks[next_idx][y0:y1, x0:x1] > 0
                          if has_next and neighbor_masks[next_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                result[y0:y1, x0:x1][comp_mask & prev_c & next_c] = 0

    return result
