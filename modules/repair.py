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

WHY THE GIVE-UP FILL IS SAFE
----------------------------
When there is no clean neighbor sky to borrow — star tracking fails, a crossing
where both neighbors carry the trail, or a borrowed pixel that still looks like
trail — the give-up pixels are painted with the local sky color plus matching grain
and a feathered seam (the "crayon" sky-fill, see _sky_fill below). Pure black is used
only as a last resort, when there is too little surrounding sky to sample. This is a
deliberate change from the old behavior, which always filled give-up pixels with pure
black. Black would be invisible in a deep lighten-max composite (each stacked pixel
takes the brightest value across all frames, so a zero loses to any real star), but it
shows as a visible hole in a single frame or a short stack — bad for timelapse — which
is why the sky-color fill replaced it.

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
1. Star Bridge (two neighbors, LK tracking succeeds)  -> shift both neighbors to
   frame N's moment and blend / per-pixel-select (best quality).
2. Two neighbors, LK tracking fails                   -> keep the original pixels
   (so stars in the fat part of the mask survive) and sky-color-fill just the
   bright streak.
3. Single neighbor (first or last frame of batch)     -> plain copy of that
   neighbor's patch, no tracking and no shift.
4. No usable neighbor / crossing where both neighbors are dirty / a borrowed pixel
   that still looks like trail -> sky-color give-up fill (local sky + grain),
   falling back to pure black only when there is too little sky to sample.
"""
import math
import time
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
MAX_SEG_LENGTH = 500   # reference segment length at full-frame resolution; scaled
                       # per-frame in _split_component (see _REF_FRAME_PX) so the
                       # chop point is a fixed fraction of the frame, not a hard px count
_REF_FRAME_PX  = 6000 * 4000  # reference resolution the 500px value is calibrated for
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
    """Down-convert a 16-bit image to 8-bit, pass an 8-bit image through unchanged.

    Lucas-Kanade tracking (OpenCV) only accepts 8-bit grayscale, so 16-bit frames
    must be scaled down before they go to the tracker. 257 is the exact divisor that
    maps the full 16-bit range (0..65535) onto the full 8-bit range (0..255), since
    65535 / 255 = 257. This is used only for the tracking math, not for the final
    repaired pixels, which stay at the frame's original bit depth.

    img: an image array of any dtype.
    Returns: the same array if already 8-bit, otherwise a freshly scaled uint8 copy.
    """
    if img.dtype == np.uint16:
        return (img / 257).astype(np.uint8)
    return img


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate an image by a fractional (sub-pixel) amount and return the result.

    Used by Star Bridge to nudge a neighbor frame so its stars line up with where
    they would sit in frame N. dx/dy may be fractional, so warpAffine with linear
    interpolation is used rather than a plain array roll. BORDER_REFLECT mirrors the
    image at the edges instead of filling with black, so stars sitting right at the
    edge of the search window are not wiped out by the shift.

    img: the image to move. dx, dy: pixels to shift right / down (may be negative).
    Returns: a new shifted image the same size and dtype as the input.
    """
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ── Sky-color fallback fill (replaces the old black fill) ──────────────────────
# Star Bridge still borrows real sky and stars from neighbor frames wherever a clean
# neighbor exists -- that path is unchanged. These helpers only handle the give-up
# pixels that used to be set to pure black (no clean neighbor at a crossing, tracking
# failed, or a borrowed pixel that still looked like trail). Black is invisible in a
# deep lighten-max stack but shows as a hole in a single frame and in short stacks --
# bad for timelapse. Instead we paint the local sky color + matching grain.

def _sky_sample(region, comp_mask):
    """Median sky color and per-channel grain from the window's background, with the
    brightest ~30% of background pixels (stars) dropped so they don't tint the fill.
    Returns (median[C], std[C]) or (None, None) when there isn't enough sky to sample."""
    bg = region[~comp_mask]
    if bg.shape[0] < 20:
        return None, None
    lum = bg.max(axis=1).astype(np.int32)
    thr = np.percentile(lum, 70)            # drop the brightest ~30% (stars); scale-free
    sky = bg[lum <= thr]
    if sky.shape[0] < 20:
        sky = bg
    return np.median(sky, axis=0), sky.std(axis=0)


SKY_FILL_FEATHER = 3.0  # seam softening (px) applied ONLY to the sky-fill patches.
                        # The Star Bridge borrow is real, motion-aligned neighbor sky and
                        # is not feathered (would needlessly soften borrowed stars).


def _sky_fill(region, target_mask, comp_mask, feather=SKY_FILL_FEATHER):
    """Paint target_mask pixels of region with local sky color + matched grain, in
    place, then feather the seam so the patch fades into the surrounding sky with no
    hard edge. region is the working window (a view into result). Returns pixels filled;
    falls back to black only if there is too little surrounding sky to sample."""
    k = int(target_mask.sum())
    if k == 0:
        return 0
    med, std = _sky_sample(region, comp_mask)
    if med is None:
        region[target_mask] = 0             # no sky to sample -> old black behavior
        return 0
    maxv = 65535 if region.dtype == np.uint16 else 255
    noise = np.random.normal(0.0, 1.0, (k, region.shape[2])) * (std + 1.0)
    region[target_mask] = np.clip(med + noise, 0, maxv).astype(region.dtype)
    # Feather only the seam: softly blend a thin band straddling the fill boundary
    # toward a blurred copy, so the patch edge fades into the sky instead of a hard
    # cut. The interior is already sky and the outside is real sky, so nothing but the
    # boundary band changes and no trail can bleed back in.
    if feather and feather > 0:
        t = target_mask.astype(np.uint8)
        ksz = max(3, int(round(feather)) * 2 + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        ring = (cv2.dilate(t, ker) > 0) & ~(cv2.erode(t, ker) > 0)
        if ring.any():
            blurred = cv2.GaussianBlur(region, (0, 0), feather).astype(np.float32)
            a = np.clip(cv2.GaussianBlur(ring.astype(np.float32), (0, 0), feather), 0.0, 1.0)[..., None]
            region[:] = np.clip(a * blurred + (1.0 - a) * region.astype(np.float32),
                                0, maxv).astype(region.dtype)
    return k


def _trail_streak(region, comp_mask):
    """Within comp_mask, return just the actual bright trail streak -- the large or
    elongated bright components -- leaving compact star blobs out, so stars sitting in
    the fat part of the mask can be kept. Used when there is no neighbor to borrow."""
    lum = region.max(axis=2).astype(np.int32)
    bg = lum[~comp_mask]
    if bg.size < 20:
        return comp_mask.copy()
    sky_l = float(np.median(bg)); sky_s = float(bg.std()) + 1.0
    bright = (comp_mask & (lum > sky_l + 2.0 * sky_s)).astype(np.uint8)
    if not bright.any():
        return np.zeros_like(comp_mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, 8)
    streak = np.zeros_like(comp_mask)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        # Treat a bright blob as trail if it is large (area >= 40 px) OR elongated
        # (longer side at least 3x the shorter). Compact, roughly-round blobs fail both
        # tests and are left out, so real stars caught inside a fat mask are preserved.
        if a >= 40 or max(w, h) / max(1, min(w, h)) >= 3.0:
            streak |= (lab == i)
    return streak


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

    # Find up to 500 bright corner features (stars). qualityLevel 0.005 is deliberately
    # low so faint stars still qualify; minDistance 5 keeps picks from clustering on one
    # bright star. Fewer than MIN_STARS features = not enough to trust a motion estimate.
    pts = cv2.goodFeaturesToTrack(
        g_search, maxCorners=500, qualityLevel=0.005,
        minDistance=5, blockSize=7
    )
    if pts is None or len(pts) < MIN_STARS:
        return 0.0, 0.0, False, 0

    # Track each feature from prev to next. status==1 marks the points LK could follow.
    pts1, status, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_next, pts, None, **_LK_PARAMS)
    good = (status.ravel() == 1)
    if good.sum() < MIN_STARS:
        return 0.0, 0.0, False, int(good.sum())

    # Keep only displacements in the plausible star-motion band. Too small = a point
    # that did not really move (noise, hot pixel); too large = a mistracked feature.
    disp = (pts1[good] - pts[good]).reshape(-1, 2)
    mag  = np.linalg.norm(disp, axis=1)
    valid = (mag >= MIN_DISP) & (mag <= MAX_DISP)
    if valid.sum() < MIN_STARS:
        return 0.0, 0.0, False, int(valid.sum())

    # Median (not mean) of the surviving displacements -> robust to any stray outlier
    # that slipped through the band. This is the local prev->next star motion vector.
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

    The split threshold is frame-relative, not a fixed pixel count: MAX_SEG_LENGTH
    (500px) is the value calibrated for a full-size 6000x4000 frame, and it is scaled
    by the square root of the frame's area relative to that reference. This keeps the
    chop point at a constant fraction of the frame at any resolution, so small frames
    are not over-chopped into too many tiny segments.

    If the major axis length exceeds the scaled threshold, splits into
    ceil(length / threshold) equal rectangle segments. Each sub-mask contains only
    the original component pixels that fall inside that segment's bounding rectangle.
    Returns a list of uint8 masks.
    """
    H, W = comp_full.shape
    seg_len_max = MAX_SEG_LENGTH * math.sqrt((H * W) / _REF_FRAME_PX)

    ys, xs = np.where(comp_full)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    trail_len = float(max(rect[1]))

    if trail_len <= seg_len_max:
        m = np.zeros(comp_full.shape, dtype=np.uint8)
        m[comp_full] = 255
        return [m]

    n_segs = math.ceil(trail_len / seg_len_max)
    box = cv2.boxPoints(rect)
    e01 = np.linalg.norm(box[1] - box[0])
    e12 = np.linalg.norm(box[2] - box[1])
    if e01 >= e12:
        a0, a1, b0, b1 = box[0], box[1], box[3], box[2]
    else:
        a0, a1, b0, b1 = box[1], box[2], box[0], box[3]

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
                 polygon_segs: list = None,
                 combine: str = "average",
                 debug_out=None, _timing_acc=None, _single_component=False) -> np.ndarray:
    """Replace masked trail pixels using Star Bridge sparse-track morph repair.

    Args:
        frame: original image (uint8 or uint16)
        mask: binary uint8 mask (255=trail, 0=sky) for this frame
        frame_idx: index of this frame in neighbor_frames
        neighbor_frames: full list of frames (same dtype as frame)
        neighbor_masks: optional list of mask arrays aligned with neighbor_frames
            (None entries = no mask / assume clean). When provided, the choice of
            which neighbor to trust is made per pixel: where exactly one neighbor's
            mask marks trail, the other (clean) neighbor's pixel is used; where both
            neighbors mark trail there is nothing clean to borrow, so those pixels
            are sky-color-filled. Both neighbors are still used wherever they are
            clean.
        polygon_segs: optional list of per-polygon binary masks (one per trail
            arm or polygon). When provided, each segment is repaired independently
            with its own Star Bridge pass instead of merging all polygons into one
            connected-components analysis. Crossing-split arms are separate entries
            so each narrow arm is repaired independently.
        combine: how to fill a gap where BOTH neighbor frames are clean. "average"
            (default — the unchanged shipped behavior) blends the two warped
            neighbors. "single_shift" uses only the previous neighbor, shifted into
            position, with a per-pixel fall back to the next neighbor where the
            previous one is itself covered by the trail. single_shift keeps the
            borrowed stars at full brightness (no averaging dilution, so faint
            stars don't drop out) at the cost of a small star-position offset. The
            per-pixel handling for a dirty neighbor is identical under both modes.
        debug_out (dict, optional): filled with a "components" list. Each entry
            has id, area, bbox, split_into, and a "segments" list. Each segment
            has tracking_ok, dx, dy, n_stars, method, still_trail_px,
            union_zeroed_px.
    Returns:
        Repaired copy of frame.
    """
    _tc = time.perf_counter()
    result = frame.copy()
    _copy_dt = time.perf_counter() - _tc
    trail = mask > 0
    if not trail.any():
        return result

    # Per-step timing accumulator (diagnostic only; written to debug_out["timing"]).
    # Created at the top-level call and threaded through the per-segment recursion.
    if _timing_acc is None and debug_out is not None:
        _timing_acc = {}

    def _addt(key, dt):
        if _timing_acc is not None:
            _timing_acc[key] = _timing_acc.get(key, 0.0) + dt

    _addt("copy_s", _copy_dt)

    # When polygon segments are provided, repair each arm independently.
    # Each recursive call processes one narrow polygon with its own Star Bridge
    # pass. No connectedComponentsWithStats -- each polygon is already one unit.
    if polygon_segs is not None and len(polygon_segs) > 0:
        for seg_mask in polygon_segs:
            if not (seg_mask > 0).any():
                continue
            result = repair_frame(result, seg_mask, frame_idx, neighbor_frames,
                                  neighbor_masks=neighbor_masks, combine=combine,
                                  _timing_acc=_timing_acc, _single_component=True)
        if debug_out is not None:
            debug_out["timing"] = {k: round(v, 3) for k, v in _timing_acc.items()}
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

    # Identify connected components. A polygon segment (the recursive per-arm
    # call) is a single filled polygon = exactly one component, so connected-
    # components is redundant there -- skip it and take the segment as the one
    # component. Pixel-identical to running CC, just far cheaper.
    if _single_component:
        _ts = time.perf_counter()
        ys0, xs0 = np.where(trail)
        _addt("cc_s", time.perf_counter() - _ts)
        if len(xs0) >= MIN_AREA:
            bx0, by0 = int(xs0.min()), int(ys0.min())
            _comp_iter = [(int(len(xs0)), bx0, by0,
                           int(xs0.max()) - bx0 + 1, int(ys0.max()) - by0 + 1, trail)]
        else:
            _comp_iter = []
    else:
        _ts = time.perf_counter()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            trail.astype(np.uint8))
        _addt("cc_s", time.perf_counter() - _ts)
        _comp_iter = [(int(stats[i, cv2.CC_STAT_AREA]),
                       int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                       int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
                       (labels == i))
                      for i in range(1, num_labels)
                      if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]

    for _ci, (comp_area, bx, by, bw, bh, comp_full) in enumerate(_comp_iter, start=1):
        # ── Per-component setup ───────────────────────────────────────────────
        _ts = time.perf_counter()
        sub_masks = _split_component(comp_full)
        _addt("split_s", time.perf_counter() - _ts)

        comp_dbg = None
        if debug_out is not None:
            comp_dbg = {
                "id": _ci,
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
            # The AND union block below sky-color-fills any pixel where BOTH neighbors
            # have the trail (nothing clean to borrow), so overlap areas are handled
            # pixel-by-pixel after repair.
            use_prev = has_prev
            use_next = has_next

            if not use_prev and not use_next:
                # No neighbor to borrow from. Keep every original pixel (so stars in the
                # fat part of the mask survive) and replace ONLY the actual bright streak
                # with local sky color instead of blacking the whole box. If the streak
                # can't be isolated, fall back to filling the whole component.
                _win = result[y0:y1, x0:x1]
                _streak = _trail_streak(frame[y0:y1, x0:x1], comp_mask)
                _target = _streak if _streak.any() else comp_mask
                _filled = _sky_fill(_win, _target, comp_mask)
                if seg_info is not None:
                    seg_info.update({"method": "sky_fill_no_neighbors",
                                     "tracking_ok": False, "dx": 0.0, "dy": 0.0,
                                     "n_stars": 0, "still_trail_px": 0,
                                     "union_zeroed_px": 0, "sky_filled_px": int(_filled)})
                    comp_dbg["segments"].append(seg_info)
                continue

            _method = "unknown"
            _dx = 0.0
            _dy = 0.0
            _ok = False
            _n_stars = 0

            # ── Crayon v2: the cleaner RAW neighbor, for the give-up cases ───────
            # When the bridge can't produce clean borrowed sky (tracking failed, a
            # borrowed pixel still reads as trail, or both neighbors are dirty) we no
            # longer synthesize a sky-color + grain patch (that speckled on smooth
            # twilight skies). Instead we paste the CLEANER real neighbor frame raw
            # (no shift, no brightness levelling). raw_clean = the per-pixel cleaner-
            # neighbor image for this window. It is None only when there is genuinely no
            # neighbor at all (degenerate single-frame run).
            _rp = neighbor_frames[prev_idx][y0:y1, x0:x1] if has_prev else None
            _rn = neighbor_frames[next_idx][y0:y1, x0:x1] if has_next else None
            _pdirty = ((neighbor_masks[prev_idx][y0:y1, x0:x1] > 0)
                       if (has_prev and neighbor_masks is not None and neighbor_masks[prev_idx] is not None)
                       else np.zeros(comp_mask.shape, dtype=bool))
            _ndirty = ((neighbor_masks[next_idx][y0:y1, x0:x1] > 0)
                       if (has_next and neighbor_masks is not None and neighbor_masks[next_idx] is not None)
                       else np.zeros(comp_mask.shape, dtype=bool))
            if _rp is not None and _rn is not None:
                raw_clean = _rp.copy()
                raw_clean[_pdirty & ~_ndirty] = _rn[_pdirty & ~_ndirty]   # prev dirty -> next
                _both = _pdirty & _ndirty                                  # both dirty -> less-dirty one
                if _both.any() and int((_ndirty & comp_mask).sum()) < int((_pdirty & comp_mask).sum()):
                    raw_clean[_both] = _rn[_both]
            elif _rp is not None:
                raw_clean = _rp.copy()
            elif _rn is not None:
                raw_clean = _rn.copy()
            else:
                raw_clean = None

            if use_prev and use_next:
                # ── Star feature tracking (Lucas-Kanade sparse optical flow) ──
                patch_prev = neighbor_frames[prev_idx][y0:y1, x0:x1]
                patch_next = neighbor_frames[next_idx][y0:y1, x0:x1]
                _ts = time.perf_counter()
                dx, dy, ok, n_stars = _track_stars(patch_prev, patch_next,
                                                   trail_mask=comp_mask)
                _addt("track_s", time.perf_counter() - _ts)
                _dx, _dy, _ok, _n_stars = dx, dy, ok, n_stars

                if ok:
                    # ── Warp synthesis ────────────────────────────────────────
                    # Meet in the middle: push prev forward by half the prev->next
                    # motion and pull next backward by half. Both neighbors now have
                    # their stars aligned to frame N's moment in time.
                    _ts = time.perf_counter()
                    warped_prev = _shift_image(patch_prev,  dx / 2.0,  dy / 2.0)
                    warped_next = _shift_image(patch_next, -dx / 2.0, -dy / 2.0)
                    _addt("warp_s", time.perf_counter() - _ts)
                    if neighbor_masks is None:
                        # No masks provided — fall back to color-based contamination check.
                        # A neighbor is "contaminated" if >20% of its trail-region pixels
                        # still look warm and bright (i.e. the trail also crosses here in
                        # that neighbor). Blend only when BOTH neighbors are clean;
                        # otherwise take whichever single neighbor is cleaner.
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
                            if combine == "single_shift":
                                # One neighbor, shifted into place. No averaging, so
                                # borrowed stars keep full brightness.
                                synth = warped_prev.copy()
                                _method = "single_shift"
                            else:
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
                        # Masks provided: decide per pixel which neighbor to trust.
                        # prev_c / next_c mark where each neighbor's own mask says
                        # "trail here" (so that neighbor's pixels are dirty there).
                        prev_c = (neighbor_masks[prev_idx][y0:y1, x0:x1] > 0
                                  if has_prev and neighbor_masks[prev_idx] is not None
                                  else np.zeros(comp_mask.shape, dtype=bool))
                        next_c = (neighbor_masks[next_idx][y0:y1, x0:x1] > 0
                                  if has_next and neighbor_masks[next_idx] is not None
                                  else np.zeros(comp_mask.shape, dtype=bool))
                        # Base fill where both neighbors are clean: averaged blend
                        # (default) or the previous neighbor alone (single_shift).
                        # Then override the pixels where exactly one neighbor is
                        # dirty: prev dirty & next clean -> take next only, and vice
                        # versa. Pixels dirty in both are left as-is here and get
                        # sky-filled later (see the AND union block). The per-pixel
                        # dirty handling is identical under both modes.
                        if combine == "single_shift":
                            synth = warped_prev.copy()
                        else:
                            synth = ((warped_prev.astype(np.float32) +
                                      warped_next.astype(np.float32)) / 2.0).astype(frame.dtype)
                        use_next_only = comp_mask & prev_c & ~next_c
                        if use_next_only.any():
                            synth[use_next_only] = warped_next[use_next_only]
                        use_prev_only = comp_mask & next_c & ~prev_c
                        if use_prev_only.any():
                            synth[use_prev_only] = warped_prev[use_prev_only]
                        _method = "single_shift" if combine == "single_shift" else "blend"
                else:
                    # Tracking failed (too few stars to measure the shift). Crayon v2:
                    # paste the cleaner real neighbor raw (no shift) instead of a
                    # synthetic sky-color fill. raw_clean is never None here (both
                    # neighbors exist).
                    synth = raw_clean.copy()
                    _method = "raw_clean_track_failed"

            elif use_prev:
                # Only the previous frame is available (e.g. the LAST frame of a
                # sequence has no next). Track the star motion between prev and THIS
                # frame and shift prev onto this frame's star positions, instead of
                # pasting it in flat. Falls back to a flat copy only if there are too
                # few stars to track reliably, so it never makes a frame worse.
                patch_prev = neighbor_frames[prev_idx][y0:y1, x0:x1]
                _dx, _dy, _ok, _n_stars = _track_stars(
                    patch_prev, frame[y0:y1, x0:x1], trail_mask=comp_mask)
                if _ok:
                    # prev's stars sit at p; in THIS frame they are at p+(dx,dy).
                    # Shift prev forward by (dx,dy) to land them on this frame.
                    synth = _shift_image(patch_prev, _dx, _dy)
                    _method = "prev_shift"
                else:
                    synth = patch_prev.copy()
                    _method = "prev_only"

            else:
                # Only the next frame is available (e.g. the FIRST frame of a sequence
                # -- like 143A8732 -- has no prev). Track the star motion between THIS
                # frame and next and shift next BACK onto this frame's positions.
                # Falls back to a flat copy only if tracking is unreliable.
                patch_next = neighbor_frames[next_idx][y0:y1, x0:x1]
                _dx, _dy, _ok, _n_stars = _track_stars(
                    frame[y0:y1, x0:x1], patch_next, trail_mask=comp_mask)
                if _ok:
                    # next's stars sit at p+(dx,dy) relative to THIS frame's p.
                    # Shift next back by (-dx,-dy) to land them on this frame.
                    synth = _shift_image(patch_next, -_dx, -_dy)
                    _method = "next_shift"
                else:
                    synth = patch_next.copy()
                    _method = "next_only"

            _tp = time.perf_counter()
            # ── Paste the borrowed neighbor sky at its native brightness ─────────
            # We deliberately do NOT brightness-level the patch to the surrounding
            # window. On a twilight gradient the window average spans the darker sky
            # above the warm horizon glow, so levelling dragged warm fills toward gray
            # (a visible gray rectangle: the neighbor sky there reads red ~200, but
            # levelling shipped red ~138). The neighbor frames are near-identical to this
            # one, so their sky already matches -- pasting it raw keeps the warm glow.
            # Levelling only ever helped a faint lighten on the first frame or two of a
            # run (sky brightness drifts most at the very start); that is 1-2 frames in
            # an 800-frame run, not worth graying every twilight fill.
            result[y0:y1, x0:x1][comp_mask] = synth[comp_mask]

            # ── Warm-pixel cleanup (still-trail remnants after warp) ──────────
            # Airplane/satellite trails read warm: red channel noticeably above blue.
            # Measure this region's own sky baseline (median red-minus-blue of the
            # surrounding background) so the gate adapts to warm-toned skies and
            # twilight gradients instead of using one fixed colour everywhere.
            bg_pixels = frame[y0:y1, x0:x1][~comp_mask].astype(np.int32)
            if len(bg_pixels) >= 10:
                bg_rb = float(np.median(bg_pixels[:, 2] - bg_pixels[:, 0]))
            else:
                bg_rb = 0.0
            warm_thresh = bg_rb + TRAIL_WARM_MARGIN

            # A repaired pixel is still trail if it is both warmer than the local sky
            # baseline (by TRAIL_WARM_MARGIN) AND bright enough in red. Channel index 2
            # is red, index 0 is blue (OpenCV stores images as BGR).
            filled = result[y0:y1, x0:x1].astype(np.int32)
            still_trail = (comp_mask &
                           (filled[..., 2] - filled[..., 0] > warm_thresh) &
                           (filled[..., 2] > TRAIL_BRIGHT_THRESH))
            # Borrowed pixel still looks like trail -> the neighbor we borrowed from
            # had the trail here too. Crayon v2: paste the cleaner raw neighbor (native
            # brightness) instead of a synthetic sky-color patch.
            if still_trail.any():
                if raw_clean is not None:
                    result[y0:y1, x0:x1][still_trail] = raw_clean[still_trail]
                else:
                    _sky_fill(result[y0:y1, x0:x1], still_trail, comp_mask)
            _still_trail_px = int(still_trail.sum())

            # ── AND union mask: BOTH neighbors have the trail here (the crossing) ──
            # Pixels in only one neighbor's trail are already repaired above. Where both
            # neighbors are dirty there is nothing clean to borrow; these are trail
            # pixels (no star to keep), so paint local sky instead of black.
            _union_zeroed_px = 0
            if neighbor_masks is not None:
                prev_c = (neighbor_masks[prev_idx][y0:y1, x0:x1] > 0
                          if has_prev and neighbor_masks[prev_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                next_c = (neighbor_masks[next_idx][y0:y1, x0:x1] > 0
                          if has_next and neighbor_masks[next_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                union_both = comp_mask & prev_c & next_c
                if union_both.any():
                    # Crayon v2: both neighbors dirty here -> paste the less-dirty raw
                    # neighbor (native brightness) instead of a synthetic fill.
                    if raw_clean is not None:
                        result[y0:y1, x0:x1][union_both] = raw_clean[union_both]
                    else:
                        _sky_fill(result[y0:y1, x0:x1], union_both, comp_mask)
                _union_zeroed_px = int(union_both.sum())
            _addt("paste_s", time.perf_counter() - _tp)

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

    if debug_out is not None and _timing_acc is not None:
        debug_out["timing"] = {k: round(v, 3) for k, v in _timing_acc.items()}
    return result
