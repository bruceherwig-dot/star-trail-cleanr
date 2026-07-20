"""
Trail repair (Star Bridge): fill a trail's pixels from neighbors N-1/N+1 -- slide the
sky to follow star motion, but keep the static foreground exactly in place.

THE CORE PROBLEM
----------------
We need to fill in the pixels that a trail was covering in frame N. We can't just
copy pixels from a static background because the sky is not static — stars move in
arcs across the frame over the course of a multi-hour session. A star that is at
position (x, y) in frame N-1 will be at a slightly different position in frame N+1.
Copying neighbor pixels directly without accounting for that motion would smear every
star in the repaired region.

HOW IT WORKS
------------
1. MEASURE the local star motion (dx,dy) from N-1 to N+1 with a cascade (see
   _detect_shift / _phase_shift / _track_stars): detect-and-measure votes on the common
   shift of the star streaks, phase correlation measures it on the whole patch, and the
   shift is trusted only when the two AGREE (or one is strongly confident). This works in
   low-feature regions where the old corner-tracking (Lucas-Kanade) failed.

2. ROUTE each gap pixel by MOTION, not brightness (the still-vs-moving split):
   - Where the two clean neighbors AGREE, the scene did not move between them
     (foreground, ground, still sky) -> keep it UNSHIFTED, so a horizon or foreground
     line stays exactly put (the slide would step it). Foreground-agnostic: a light rock
     or building is kept the same as a dark hill.
   - Where they DIFFER, something moved (a star) -> use the SLID neighbor so the star
     lands at its frame-N position. Shipped combine is "single_shift" (shift the
     colour-closest neighbor); "average" blends both shifted neighbors.
   - Small bright blobs the slide adds over a still background are protected as stars;
     larger bright regions (glow slid over foreground) are not.

3. LEVEL brightness. Each borrowed pixel is pre-levelled to this frame's local sky colour
   (per-source). A final ring-levelling absorbs the small frame-to-frame sky drift, but
   CAPPED to the drift actually measured per-source (plus a small margin), so it can never
   crush a fill to black or blow it out bright by matching a collar that is not the same
   content as the patch (a trail crossing dark foreground).

4. CLEAN UP isolated near-black dots left at sharp foreground edges (where the edge
   jittered between neighbors) by replacing them with their local median.

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
1. Two neighbors, shift measured -> still-vs-moving routed fill (best quality).
2. Two neighbors, shift NOT measured -> paste the colour-closest neighbor (raw_clean).
3. Single neighbor (first/last frame) -> shift that one neighbor onto frame N, or a
   plain levelled copy if the shift can't be measured. THEN edge-frame foreground
   protection: reach to the SECOND same-side neighbor and keep any static object (an
   unmasked trunk/rock) UNSHIFTED, so the single-neighbor slide doesn't nick its edge.
4. Crossing where both neighbors carry the trail -> paint local sky colour + grain (the
   "crayon" _sky_fill), falling back to black only when there is too little sky to sample.

NOTE: the old warm-pixel cleanup (overwrite "warm + bright-red" pixels as leftover trail)
is DISABLED -- a warm star read identically, so it deleted stars.
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
# TRAIL_WARM_MARGIN/TRAIL_BRIGHT_THRESH: warm+bright gate for the no-masks
#   contamination check (is the trail also crossing here in a neighbor?).
# Log fields: seg.method reflects which of these paths fired per segment.

PAD            = 120   # pixels around each trail bbox for feature search
MIN_AREA       = 500   # skip tiny mask components (noise)
MAX_SEG_LENGTH = 500   # reference segment length at full-frame resolution; scaled
                       # per-frame in _split_component (see _REF_FRAME_PX) so the
                       # chop point is a fixed fraction of the frame, not a hard px count
_REF_FRAME_PX  = 6000 * 4000  # reference resolution the 500px value is calibrated for
MIN_DISP  = 1.0   # minimum plausible star displacement N-1 to N+1 (px)
MAX_DISP  = 60.0  # maximum plausible star displacement N-1 to N+1 (px)
TRAIL_WARM_MARGIN   = 20  # how much warmer than local sky R-B counts as a trail remnant
TRAIL_BRIGHT_THRESH = 50  # minimum R value to be considered trail-bright
# Ring-levelling cap: the final ring-levelling nudges the filled patch to match a thin collar
# around the trail, to absorb small frame-to-frame SKY drift. Where the trail crosses dark
# foreground (branch, horizon, rock, building) that collar is NOT sky, and an uncapped match
# crushed the fill to BLACK (or blew it out bright). Cap the nudge to the drift we already
# MEASURED per-source (same spot, this frame vs neighbor) plus a small margin -- it adapts to
# each scene and refuses any larger match (a content mismatch, not drift), keeping the fill.
_RING_DRIFT_MARGIN   = 3   # levels the ring may add beyond the measured per-source drift
_RING_OFF_CAP        = 6   # fallback cap when no per-source drift was measurable
# Still-vs-moving routing: judge each gap pixel by MOTION, not brightness. Where the two clean
# neighbors agree, the scene did not move (ground/hill/trunk/still sky) -> keep it unshifted so
# the foreground stays exactly put; where they differ, a star moved -> keep the slid fill.
_STILL_ROUTING       = True  # master switch for the motion-based fill routing
_STILL_TOL           = 18    # neighbors agree within this (max-channel) = "still" (didn't move)
_STAR_MARGIN         = 15    # slid fill this much brighter than the still scene = an added star
_STAR_MAX_AREA       = 500   # a protected slid star blob is at most this big (px); larger bright
                             # regions are glow slid over foreground -> still routed to unshifted
_SPECKLE_DARK        = 25    # a patch pixel below this max-channel may be edge-speckle residue
_SPECKLE_MARGIN      = 30    # ...cleaned only if its local median is this much brighter (isolated)
# Collar for the ring-levelling step: the repaired patch is brightness-matched to
# the sky in a thin ring hugging the trail (dilate by 15 px, minus the trail
# itself). Local on purpose -- a whole-window match drags warm twilight fills gray.
_RING_KERNEL = np.ones((31, 31), np.uint8)
# The collar must measure SKY, not foreground. When a trail runs beside a trunk or ridge, the
# ring of pixels around it includes that dark foreground; an unfiltered median then reads dark,
# and the ring-levelling drags the whole fill down toward it -- a few levels dark across the
# entire patch (the subtle, capped cousin of the black-rectangle bug). So drop pixels darker
# than this fraction of the collar's own sky level (its 70th percentile) before measuring; only
# sky remains. Measured on IMG_2971: an unfiltered collar left the fill 4 levels dark; filtering
# lands it on the surrounding sky. Falls back to the unfiltered ring if too little sky is left.
_COLLAR_SKY_FRAC = 0.5
# Darken-foreground restore: a trail crossing dark STATIC foreground (a Joshua-tree spike,
# a branch, a rock, a rooftop) cannot be rebuilt by sliding a neighbor -- the neighbor's sky
# slides OVER the foreground and erases it, because the fill routes by star motion and the
# foreground has none to borrow. But on a fixed tripod that foreground is the SAME dark pixel
# in every frame, and the trail is bright, so the per-pixel MINIMUM (a darken blend) across a
# few neighbor frames is the true foreground with the bright trail rejected -- as long as one
# frame in the window is clear of that spot, which holds because the trail moves. Applied only
# to masked pixels darker than a fraction of the LOCAL sky (measured in the same collar the
# ring-levelling uses); those pixels are REPLACED outright with the darken value. A feather was
# tried and dropped -- blending the clean darken against the already-erased repair underneath
# reintroduced the mangled edge; a hard replace gives crisp spikes (verified on IMG_2971,
# 2026-07-06). Brighter/sky pixels are left to the Star Bridge slide so the moving stars stay
# correct. Even a +/-1 window recovers a spike (it is dark in every frame); the wider window is
# margin against a slow trail that lingers on a pixel for several frames.
_DARKEN_FOREGROUND   = True
_DARKEN_WINDOW       = 3     # reach +/-N neighbor frames for the darken(min); clamped at set ends
_DARKEN_FG_FRAC      = 0.72  # a masked pixel darker than this * local sky is dark foreground
# Reach-further-for-crossings: at a crossing BOTH immediate neighbors carry the trail, so N-1 and
# N+1 have no clean sky at that spot. But the trail is moving, so by N-2/N+2 (or a little further)
# it has usually cleared off, leaving real sky to borrow. Rather than paste a still-dirty neighbor,
# search outward for the first frame that is clean at each crossing pixel and borrow its sky,
# shifted by the per-frame star motion so the stars land on frame N. Measured on the Joshua Tree
# set: at IMG_3020 every crossing pixel is clean by +/-2; at IMG_3019 the rest clear by +/-3.
# Limited to the neighbor frames actually loaded (full +/-4 mid-batch, fewer at a batch edge).
_CROSS_REACH_ENABLED = True
_CROSS_REACH         = 4


# ── Image helpers ─────────────────────────────────────────────────────────────
# _to_8bit: converts 16-bit frames to 8-bit for the cascade tracker (needs uint8).
# _shift_image: applies a fractional-pixel translation via warpAffine with reflect
#   border padding so stars near the trail edge are not blacked out by the warp.

def _to_8bit(img: np.ndarray) -> np.ndarray:
    """Down-convert a 16-bit image to 8-bit, pass an 8-bit image through unchanged.

    The cascade tracker (OpenCV) only accepts 8-bit grayscale, so 16-bit frames
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


# ── Local star motion: the cascade ───────────────────────────────────────────
# Three helpers measure the prev->next star shift for a trail patch WITHOUT needing
# trackable corners (the old Lucas-Kanade tracker slid off streaky horizon stars and
# mis-placed them). _detect_shift finds the star streaks as blobs and votes on their
# common shift; _phase_shift reads the whole patch's frequency pattern (FFT, sub-pixel)
# and returns a confidence; _track_stars runs both and trusts the shift only when they
# agree (or one is strongly confident), else reports failure so the caller pastes the
# unshifted neighbor instead of sliding the wrong way. Log field: seg.n_stars.

def _detect_shift(gp, gn):
    """Local star motion by DETECTING the star streaks as objects and voting on their
    shared displacement. A long-exposure star is a smooth streak with no corner for
    Lucas-Kanade to grab, but it is a distinct bright blob whose centre moves by the
    inter-frame motion -- so even one clean star yields the shift, and many vote for a
    robust common shift. The lone trail streak is outvoted by the stars.

    gp, gn: 8-bit grayscale patches (prev, next). Returns ((dx,dy) or None, n_agreeing).
    """
    def streaks(g):
        sky = np.median(g)
        thr = max(sky + 22, np.percentile(g, 92))
        b = cv2.morphologyEx((g > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
        out = []
        for k in range(1, n):
            if st[k, 4] < 6 or st[k, 4] > 4000:
                continue
            ys, xs = np.where(lab == k)
            w = g[ys, xs].astype(np.float64)
            out.append((float((xs * w).sum() / w.sum()),
                        float((ys * w).sum() / w.sum()), float(w.sum())))
        return out
    sp, sn = streaks(gp), streaks(gn)
    cands = []
    for px, py, pw in sp:
        for nx, ny, nw in sn:
            dx, dy = nx - px, ny - py
            if MIN_DISP <= (dx * dx + dy * dy) ** 0.5 <= MAX_DISP:
                cands.append((dx, dy, min(pw, nw)))
    if not cands:
        return None, 0
    best = None
    for dx, dy, _ in cands:
        v = [c for c in cands if (c[0] - dx) ** 2 + (c[1] - dy) ** 2 <= 16]
        s = sum(c[2] for c in v)
        if best is None or s > best[0]:
            best = (s, np.average([c[0] for c in v], weights=[c[2] for c in v]),
                       np.average([c[1] for c in v], weights=[c[2] for c in v]), len(v))
    return (best[1], best[2]), best[3]


def _phase_shift(gp, gn):
    """Local star motion via phase correlation (FFT, sub-pixel). Reads the whole patch's
    frequency pattern rather than tracking points, so it stays accurate on streaky horizon
    stars where corner tracking returns garbage. Returns ((dx,dy) or None, response 0..~1).
    """
    h, w = gp.shape
    if h < 24 or w < 24:
        return None, 0.0
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(gp.astype(np.float32), gn.astype(np.float32), win)
    if not (MIN_DISP <= (dx * dx + dy * dy) ** 0.5 <= MAX_DISP):
        return None, resp
    return (dx, dy), resp


def _track_stars(prev: np.ndarray, nxt: np.ndarray, trail_mask=None):
    """Local star motion prev->next as (dx, dy, success, n_stars, cascade_tier). Cascade of two methods
    that do NOT need trackable corners (unlike the old Lucas-Kanade tracker, which slid off
    streaky horizon stars and mis-placed them): detect-and-measure and phase correlation.
    The shift is trusted only when the two AGREE (within a few px), or when one is strongly
    confident on its own; otherwise success is False and the caller falls back to a plain
    neighbor paste rather than sliding the wrong way. trail_mask is accepted for signature
    compatibility -- the detect vote ignores the lone trail streak, so it isn't masked here.
    """
    gp = cv2.cvtColor(_to_8bit(prev), cv2.COLOR_BGR2GRAY)
    gn = cv2.cvtColor(_to_8bit(nxt),  cv2.COLOR_BGR2GRAY)
    Sd, votes = _detect_shift(gp, gn)
    Sp, resp = _phase_shift(gp, gn)
    # The 5th return value names WHICH branch produced the shift, for the run log.
    if Sd is not None and Sp is not None and (Sd[0] - Sp[0]) ** 2 + (Sd[1] - Sp[1]) ** 2 <= 25:
        return (Sd[0] + Sp[0]) / 2.0, (Sd[1] + Sp[1]) / 2.0, True, int(votes), "agree"   # both agreed
    if Sp is not None and resp >= 0.6:
        return float(Sp[0]), float(Sp[1]), True, 1, "phase"                     # phase confident alone
    if Sd is not None and votes >= 4:
        return float(Sd[0]), float(Sd[1]), True, int(votes), "detect"           # enough star votes alone
    return 0.0, 0.0, False, int(votes), "fail"                                  # neither confident -> paste


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


def _darken_fill(patch_now, dmin, dmed, comp_mask, collar, maxv, dt):
    """Restore dark static foreground under a trail with a darken (min) replace.

    On a fixed tripod a spike/trunk/rock is the same dark pixel in every frame, and a
    trail is bright, so the per-pixel min across neighbor frames (`dmin`) is that
    foreground with the bright trail rejected. The masked pixels judged to be dark
    foreground are REPLACED with the min -- a hard replace, not a blend: mixing the clean
    darken with the already-erased repair underneath just reintroduced the mangled edge.
    Sky pixels are left untouched so the Star Bridge slide keeps the moving stars. Local
    sky is a high percentile (the 70th) of the collar's max-channel, which stays right even
    when the collar straddles dark foreground.

    The "is this dark foreground?" test uses the per-pixel MEDIAN across the window
    (`dmed`), NOT the min. A real spike is dark in almost every frame (a trail only crosses
    it briefly), so its median stays dark. A sky pixel is dark only in its single darkest
    frame -- its median sits at the true sky level -- so using the min to gate stamped the
    star-free sky floor over the whole trail and left every cleaned patch a few levels
    darker than the surrounding sky (a shipped v2.71 bug, proven on Greg Meyer Arizona,
    2026-07-19: the min gate fired on 54% of the repaired sky, the median gate on 7%, while
    keeping 93% of the Joshua-tree spike pixels the min gate restored). The REPLACEMENT
    still uses the min -- the crisp true foreground -- only the gate changed.

    patch_now: the already-repaired patch (BGR, this frame's dtype).
    dmin: per-pixel min of the neighbor window (the replacement value), same shape/dtype.
    dmed: per-pixel median of the same window (the foreground/sky gate), same shape/dtype.
    comp_mask: bool mask of the trail pixels in this window.
    collar: bool mask of the local sky ring (dilated trail minus the trail).
    maxv, dt: value ceiling and dtype (unused by the replace, kept for signature parity).
    Returns (repaired_patch, foreground_px_filled, local_sky_level or None).
    """
    cpx = patch_now[collar]
    if cpx.shape[0] < 20:
        return patch_now, 0, None
    # 70th percentile, NOT the median: a segment hugging a dark tree has a collar that is
    # mostly foreground, so the median collapses toward black (measured sky=5 next to a
    # Joshua tree), the threshold drops to ~zero and every spike is missed. The sky pixels
    # are the brighter part of even a tree-heavy collar, so a high percentile reads the true
    # sky level (~37) and the threshold lands right. Verified on IMG_2971, 2026-07-06.
    sky = float(np.percentile(cpx.max(axis=1), 70))
    if sky <= 0:
        return patch_now, 0, None
    # Gate on the MEDIAN across the window (typical value), not the darkest frame -- the min
    # of a sky pixel is the star-free sky floor and reads as false "dark foreground".
    fg = comp_mask & (dmed.max(axis=2).astype(np.float32) < _DARKEN_FG_FRAC * sky)
    if not fg.any():
        return patch_now, 0, sky
    out = patch_now.copy()
    out[fg] = dmin[fg]   # replace with the crisp true foreground (the min), not the median
    return out, int(fg.sum()), sky


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
            edge_still_px, fg_darken_px (dark static foreground pixels restored by
            the darken-min blend instead of being erased by the sky slide),
            union_zeroed_px, ring_off (the per-channel B,G,R brightness nudge
            the final ring-levelling step applied, or None if it could not
            measure), and base ("prev" or "next" -- which neighbor's sky colour
            sat closest to this frame's local collar and was borrowed from).
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
        if debug_out is not None:
            debug_out["components"] = []
        for _poly_i, seg_mask in enumerate(polygon_segs, start=1):
            if not (seg_mask > 0).any():
                continue
            # Thread a debug dict into each per-polygon pass and collect its one
            # component, so the run log records the fill method + cascade for EVERY
            # trail. This path used to drop all of it and log only timing (the reason
            # the logs couldn't say which method or cascade tier fired -- 2026-07-02).
            _sub_dbg = {} if debug_out is not None else None
            result = repair_frame(result, seg_mask, frame_idx, neighbor_frames,
                                  neighbor_masks=neighbor_masks, combine=combine,
                                  debug_out=_sub_dbg,
                                  _timing_acc=_timing_acc, _single_component=True)
            if _sub_dbg is not None:
                for _comp in _sub_dbg.get("components", []):
                    _comp["polygon"] = _poly_i   # which detected polygon/arm this was
                    debug_out["components"].append(_comp)
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
                    # _filled>0 = the crayon sky-fill painted; _filled==0 = the RARE pure-black
                    # fallback (no neighbor AND too little surrounding sky to sample a colour).
                    seg_info.update({"method": ("crayon_sky_no_neighbors" if _filled > 0
                                                else "black_no_sky"),
                                     "cascade": "no_neighbors",
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
            _cascade = "none"
            _edge_still_px = 0

            # ── Per-neighbor colour fit, measured on a local sky collar ──────────
            # The collar is a thin ring hugging the trail (minus every trail pixel
            # in the window). Medians on that collar tell us two things: (a) which
            # neighbor's sky colour sits closest to THIS frame right here, and
            # (b) the exact per-channel nudge that maps each neighbor's sky onto
            # this frame's. Fire glow, moving smoke and airglow make neighbors
            # genuinely differ frame to frame; borrowing without this produced
            # two-toned violet patches (071A7602 blue rectangle, see
            # known_problems). Every borrowed pixel below is pre-levelled with its
            # OWN source's nudge, and where a base neighbor must be chosen the
            # colour-closest one wins (Bruce's rule, 2026-06-10).
            _rp = neighbor_frames[prev_idx][y0:y1, x0:x1] if has_prev else None
            _rn = neighbor_frames[next_idx][y0:y1, x0:x1] if has_next else None
            _pdirty = ((neighbor_masks[prev_idx][y0:y1, x0:x1] > 0)
                       if (has_prev and neighbor_masks is not None and neighbor_masks[prev_idx] is not None)
                       else np.zeros(comp_mask.shape, dtype=bool))
            _ndirty = ((neighbor_masks[next_idx][y0:y1, x0:x1] > 0)
                       if (has_next and neighbor_masks is not None and neighbor_masks[next_idx] is not None)
                       else np.zeros(comp_mask.shape, dtype=bool))
            # Build the levelling collar: the ring of pixels hugging the trail (dilate minus the
            # trail), then KEEP ONLY THE SKY in it. A trail beside a trunk would otherwise measure
            # the trunk as if it were sky, and the levelling would drag the whole fill darker toward
            # it (see _COLLAR_SKY_FRAC). Drop pixels darker than half the collar's own sky level (its
            # 70th percentile); fall back to the full ring if too little sky is left to measure.
            _dilc = ((cv2.dilate(comp_mask.astype(np.uint8), _RING_KERNEL) > 0)
                     & ~trail[y0:y1, x0:x1])
            _wmax = frame[y0:y1, x0:x1].max(2)
            if int(_dilc.sum()) >= 20:
                _sl = np.percentile(_wmax[_dilc], 70)                 # the collar's own sky level
                _sky_only = _dilc & (_wmax > _COLLAR_SKY_FRAC * _sl)  # keep sky, drop dark foreground
                _collar = _sky_only if int(_sky_only.sum()) >= 20 else _dilc
            else:
                _collar = _dilc
            _maxv = 65535 if frame.dtype == np.uint16 else 255
            _off_p = _off_n = _cur_med = None
            _cur_sky = frame[y0:y1, x0:x1][_collar]
            if _cur_sky.shape[0] >= 20:
                _cur_med = np.median(_cur_sky, axis=0).astype(np.float32)
                if _rp is not None:
                    _ps = _rp[_collar & ~_pdirty]
                    if _ps.shape[0] >= 20:
                        _off_p = _cur_med - np.median(_ps, axis=0).astype(np.float32)
                if _rn is not None:
                    _ns = _rn[_collar & ~_ndirty]
                    if _ns.shape[0] >= 20:
                        _off_n = _cur_med - np.median(_ns, axis=0).astype(np.float32)
            # Which neighbor's sky is the closer colour match here?
            _next_closer = (_off_p is not None and _off_n is not None
                            and float(np.abs(_off_n).sum()) < float(np.abs(_off_p).sum()))

            def _level(img, off):
                """A borrowed window, nudged onto this frame's local sky colour."""
                if off is None:
                    return img.copy()
                return np.clip(img.astype(np.float32) + off,
                               0, _maxv).astype(frame.dtype)

            # ── Crayon v2: the colour-closest RAW neighbor, for the give-up cases ──
            # When the bridge can't produce clean borrowed sky (tracking failed, a
            # borrowed pixel still reads as trail, or both neighbors are dirty) we
            # paste the real neighbor frame, pre-levelled to this frame's local sky.
            # Base = the colour-closest neighbor; pixels where the base is dirty and
            # the other neighbor is clean come from the other (also pre-levelled, so
            # the patch cannot end up two-toned). raw_clean is None only when there
            # is genuinely no neighbor at all (degenerate single-frame run).
            if _rp is not None and _rn is not None:
                _rp_l = _level(_rp, _off_p)
                _rn_l = _level(_rn, _off_n)
                if _next_closer:
                    raw_clean = _rn_l.copy()
                    raw_clean[_ndirty & ~_pdirty] = _rp_l[_ndirty & ~_pdirty]
                else:
                    raw_clean = _rp_l.copy()
                    raw_clean[_pdirty & ~_ndirty] = _rn_l[_pdirty & ~_ndirty]
                # Pixels dirty in BOTH stay with the base: there is nothing clean
                # to borrow, and the base is already the closest colour.
            elif _rp is not None:
                raw_clean = _level(_rp, _off_p)
            elif _rn is not None:
                raw_clean = _level(_rn, _off_n)
            else:
                raw_clean = None

            if use_prev and use_next:
                # ── Local star motion: the cascade (detect-and-measure + phase correlation) ──
                patch_prev = neighbor_frames[prev_idx][y0:y1, x0:x1]
                patch_next = neighbor_frames[next_idx][y0:y1, x0:x1]
                _ts = time.perf_counter()
                dx, dy, ok, n_stars, tier = _track_stars(patch_prev, patch_next,
                                                         trail_mask=comp_mask)
                _addt("track_s", time.perf_counter() - _ts)
                _dx, _dy, _ok, _n_stars, _cascade = dx, dy, ok, n_stars, tier

                if ok:
                    # ── Warp synthesis ────────────────────────────────────────
                    # Meet in the middle: push prev forward by half the prev->next
                    # motion and pull next backward by half. Both neighbors now have
                    # their stars aligned to frame N's moment in time.
                    _ts = time.perf_counter()
                    warped_prev = _shift_image(patch_prev,  dx / 2.0,  dy / 2.0)
                    warped_next = _shift_image(patch_next, -dx / 2.0, -dy / 2.0)
                    _addt("warp_s", time.perf_counter() - _ts)
                    # Pre-level each warped neighbor onto this frame's local sky
                    # colour BEFORE any composition, so pixels sourced from
                    # different neighbors land on the same colour (no two-toned
                    # patches when the fire glow / smoke shifts between frames).
                    _wp = _level(warped_prev, _off_p)
                    _wn = _level(warped_next, _off_n)
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
                                # borrowed stars keep full brightness. The colour-
                                # closest neighbor is the one borrowed from.
                                synth = (_wn.copy() if _next_closer else _wp.copy())
                                _method = "single_shift"
                            else:
                                synth = ((_wp.astype(np.float32) +
                                          _wn.astype(np.float32)) / 2.0).astype(frame.dtype)
                                _method = "blend"
                        elif cp <= cn:
                            synth = _wp.copy()
                            _method = "prev_only"
                        else:
                            synth = _wn.copy()
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
                        # (default) or the colour-closest neighbor alone
                        # (single_shift). Then override the pixels where exactly one
                        # neighbor is dirty: prev dirty & next clean -> take next
                        # only, and vice versa. All sources are pre-levelled to this
                        # frame's local sky, so mixing sources cannot two-tone the
                        # patch. Pixels dirty in both are left as-is here and get
                        # filled later (see the AND union block).
                        if combine == "single_shift":
                            synth = (_wn.copy() if _next_closer else _wp.copy())
                        else:
                            synth = ((_wp.astype(np.float32) +
                                      _wn.astype(np.float32)) / 2.0).astype(frame.dtype)
                        use_next_only = comp_mask & prev_c & ~next_c
                        if use_next_only.any():
                            synth[use_next_only] = _wn[use_next_only]
                        use_prev_only = comp_mask & next_c & ~prev_c
                        if use_prev_only.any():
                            synth[use_prev_only] = _wp[use_prev_only]
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
                _dx, _dy, _ok, _n_stars, _cascade = _track_stars(
                    patch_prev, frame[y0:y1, x0:x1], trail_mask=comp_mask)
                if _ok:
                    # prev's stars sit at p; in THIS frame they are at p+(dx,dy).
                    # Shift prev forward by (dx,dy) to land them on this frame.
                    synth = _level(_shift_image(patch_prev, _dx, _dy), _off_p)
                    _method = "prev_shift"
                else:
                    synth = _level(patch_prev, _off_p)
                    _method = "prev_only"

            else:
                # Only the next frame is available (e.g. the FIRST frame of a sequence
                # -- like 143A8732 -- has no prev). Track the star motion between THIS
                # frame and next and shift next BACK onto this frame's positions.
                # Falls back to a flat copy only if tracking is unreliable.
                patch_next = neighbor_frames[next_idx][y0:y1, x0:x1]
                _dx, _dy, _ok, _n_stars, _cascade = _track_stars(
                    frame[y0:y1, x0:x1], patch_next, trail_mask=comp_mask)
                if _ok:
                    # next's stars sit at p+(dx,dy) relative to THIS frame's p.
                    # Shift next back by (-dx,-dy) to land them on this frame.
                    synth = _level(_shift_image(patch_next, -_dx, -_dy), _off_n)
                    _method = "next_shift"
                else:
                    synth = _level(patch_next, _off_n)
                    _method = "next_only"

            # ── Edge-frame foreground protection (the FIRST or LAST frame of the set) ──
            # An edge frame has only ONE neighbor, so the two-neighbor still-vs-moving routing
            # below cannot run. With no foreground mask, an unmasked STATIC object (a tree
            # trunk, a rock) then gets SLID by the single-neighbor shift, nicking its edge -- a
            # light notch that survives the lighten-max stack (found on IMG_2946, the first JT
            # frame, 2026-07-02). Fix: reach to the SECOND neighbor on the SAME side (N+1,N+2
            # for the first frame; N-1,N-2 for the last). By the second frame the trail has
            # moved off this spot, so both are clean here; a static object is identical in both,
            # so where they AGREE we keep the pixel UNSHIFTED from the nearer neighbor (the
            # object stays put) while moving sky still slides. Same star-protect rule as the
            # two-neighbor routing. Interior frames (both neighbors) skip this and use the block
            # below unchanged.
            if _STILL_ROUTING and (use_prev != use_next):     # exactly one neighbor = edge frame
                if use_next:
                    _near_idx, _far_idx, _near_off = next_idx, next_idx + 1, _off_n
                else:
                    _near_idx, _far_idx, _near_off = prev_idx, prev_idx - 1, _off_p
                if 0 <= _far_idx < N:
                    _near = neighbor_frames[_near_idx][y0:y1, x0:x1]
                    _far  = neighbor_frames[_far_idx][y0:y1, x0:x1]
                    _nd = ((neighbor_masks[_near_idx][y0:y1, x0:x1] > 0)
                           if (neighbor_masks is not None and neighbor_masks[_near_idx] is not None)
                           else np.zeros(comp_mask.shape, dtype=bool))
                    _fd = ((neighbor_masks[_far_idx][y0:y1, x0:x1] > 0)
                           if (neighbor_masks is not None and neighbor_masks[_far_idx] is not None)
                           else np.zeros(comp_mask.shape, dtype=bool))
                    _unshift = _level(_near, _near_off)
                    # Local sky brightness (the collar), to tell DARK foreground from sky.
                    _cpx = frame[y0:y1, x0:x1][_collar]
                    _sky_lvl = float(np.median(_cpx.max(axis=1))) if _cpx.shape[0] >= 20 else 128.0
                    # Static where BOTH same-side neighbors are clean AND agree in brightness.
                    _still_e = ((~_nd) & (~_fd) &
                                (np.abs(_near.max(axis=2).astype(np.int16) -
                                        _far.max(axis=2).astype(np.int16)) < _STILL_TOL))
                    # ALSO static: dark silhouette foreground in the near neighbor. Every user
                    # shoots on a fixed tripod, so opaque foreground (trunk, rock, building) is
                    # pixel-identical in every frame, and no trail or star is ever IN FRONT of
                    # it. A pixel far darker than the local sky is that silhouette even when a
                    # FAT trail mask claims the spot is dirty (the mask overhangs clean trunk --
                    # that veto is what left the poly-3 trunk notch: f2's mask covered the trunk
                    # edge, blocked the still test, and the slid fill nicked it). Dark = keep put.
                    _still_e |= (_near.max(axis=2).astype(np.int16) < 0.5 * _sky_lvl)
                    # Protect a slid star -- BUT only where it sits over SKY. A real star is
                    # never in front of the opaque trunk/rock, so a small bright patch over DARK
                    # foreground is sky slid over a foreground edge and MUST stay unshifted, not
                    # be rescued (that was the poly-3 notch: a bright sliver at a thin branch tip).
                    _extra_e = (_still_e
                                & ((synth.max(axis=2).astype(np.int16) -
                                    _unshift.max(axis=2).astype(np.int16)) > _STAR_MARGIN)
                                & (_unshift.max(axis=2).astype(np.int16) > 0.5 * _sky_lvl))
                    if _extra_e.any():
                        _nl, _lab, _st, _ = cv2.connectedComponentsWithStats(
                            _extra_e.astype(np.uint8), 8)
                        for _k in range(1, _nl):
                            if _st[_k, cv2.CC_STAT_AREA] <= _STAR_MAX_AREA:
                                _still_e[_lab == _k] = False
                    if _still_e.any():
                        synth[_still_e] = _unshift[_still_e]
                        _edge_still_px = int((_still_e & comp_mask).sum())

            # ── Still-vs-moving routing: where the two CLEAN neighbors AGREE, the scene did
            # not move between them (ground, hill, trunk, still sky) -- keep it UNSHIFTED so the
            # foreground stays exactly put (the slide would displace it, stepping a horizon line).
            # Where they differ, something moved (a star) -- keep the slid fill so it lands right.
            # Judges by motion, not brightness, so light foreground is kept like dark.
            if (_STILL_ROUTING and _rp is not None and _rn is not None
                    and raw_clean is not None):
                _still = ((~_pdirty) & (~_ndirty) &
                          (np.abs(_rp.max(axis=2).astype(np.int16) -
                                  _rn.max(axis=2).astype(np.int16)) < _STILL_TOL))
                # Protect slid stars: a small bright blob the slide adds over the still
                # background is a star at its correct (slid) position, so DON'T overwrite it
                # with the star-free unshifted scene. Large bright regions (glow slid over
                # foreground) are not stars and stay routed.
                _extra = (_still & ((synth.max(axis=2).astype(np.int16) -
                                     raw_clean.max(axis=2).astype(np.int16)) > _STAR_MARGIN))
                if _extra.any():
                    _nl, _lab, _st, _ = cv2.connectedComponentsWithStats(
                        _extra.astype(np.uint8), 8)
                    for _k in range(1, _nl):
                        if _st[_k, cv2.CC_STAT_AREA] <= _STAR_MAX_AREA:
                            _still[_lab == _k] = False        # keep this slid star
                if _still.any():
                    synth[_still] = raw_clean[_still]

            _tp = time.perf_counter()
            # Paste the borrowed neighbor sky as-is. Brightness is corrected AFTER
            # all the fills below, in one ring-levelling step (see end of this loop)
            # that measures the sky in a thin LOCAL collar around the trail -- never
            # against the whole window, which on a twilight gradient drags warm fills
            # toward gray (the neighbor sky read red ~200 but shipped ~138 gray).
            result[y0:y1, x0:x1][comp_mask] = synth[comp_mask]

            # ── Warm-pixel cleanup: DISABLED (it was deleting stars) ──────────
            # This step flagged "warm + bright-red" repaired pixels as leftover trail and
            # overwrote them. A warm/pink STAR reads identically, so wherever a trail crossed
            # a star it deleted the star. Measured on a full set: ~98% of what it removed was a
            # star the slide had placed correctly; only ~1.6% was genuine trail with no clean
            # source. Disabling it restores those stars -- the visible breaks in bright arcs,
            # and the black/holed notches in single frames that hurt timelapse. Genuine
            # both-dirty crossings are still handled by the AND-union block below.
            still_trail = np.zeros(comp_mask.shape, dtype=bool)
            _still_trail_px = 0

            # ── AND union mask: BOTH neighbors have the trail here (the crossing) ──
            # Pixels in only one neighbor's trail are already repaired above. Where both
            # neighbors are dirty there is nothing clean to borrow; these are trail
            # pixels (no star to keep), so paint local sky instead of black.
            _union_zeroed_px = 0
            _cross_reach_px = 0
            if neighbor_masks is not None:
                prev_c = (neighbor_masks[prev_idx][y0:y1, x0:x1] > 0
                          if has_prev and neighbor_masks[prev_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                next_c = (neighbor_masks[next_idx][y0:y1, x0:x1] > 0
                          if has_next and neighbor_masks[next_idx] is not None
                          else np.zeros(comp_mask.shape, dtype=bool))
                union_both = comp_mask & prev_c & next_c
                _union_zeroed_px = int(union_both.sum())
                if union_both.any():
                    # A crossing: both immediate neighbors carry the trail here, so N-1 and N+1 have
                    # no clean sky to borrow. Reach outward -- by N-2/N+2 (or a little further) the
                    # trail has usually moved off this spot -- for the first frame that is clean at
                    # each pixel, and borrow its sky shifted by the per-frame star motion (half the
                    # measured N-1->N+1 shift, times the distance) so the stars land on frame N.
                    _remaining = union_both.copy()
                    if _CROSS_REACH_ENABLED:
                        for _d in range(2, _CROSS_REACH + 1):
                            if not _remaining.any():
                                break
                            for _sgn in (1, -1):
                                _k = frame_idx + _sgn * _d
                                if not (0 <= _k < N):
                                    continue
                                _km = ((neighbor_masks[_k][y0:y1, x0:x1] > 0)
                                       if neighbor_masks[_k] is not None
                                       else np.zeros(comp_mask.shape, dtype=bool))
                                _clean = _remaining & ~_km          # frame _k is clean at these px
                                if not _clean.any():
                                    continue
                                _pk = neighbor_frames[_k][y0:y1, x0:x1]
                                # level frame _k onto this frame's local sky colour
                                _offk = None
                                if _cur_med is not None:
                                    _kc = _pk[_collar]
                                    if _kc.shape[0] >= 20:
                                        _offk = _cur_med - np.median(_kc, axis=0).astype(np.float32)
                                # shift _k's stars onto frame N (per-frame motion = the shift / 2)
                                _wk = (_shift_image(_pk, -_sgn * _d * (_dx / 2.0),
                                                         -_sgn * _d * (_dy / 2.0))
                                       if _ok else _pk)
                                _wk = _level(_wk, _offk)
                                result[y0:y1, x0:x1][_clean] = _wk[_clean]
                                _remaining &= ~_clean
                                _cross_reach_px += int(_clean.sum())
                    # Whatever no clean frame was found for falls back to the colour-closest paste
                    # (or crayon sky-fill) -- a genuinely persistent crossing.
                    if _remaining.any():
                        if raw_clean is not None:
                            result[y0:y1, x0:x1][_remaining] = raw_clean[_remaining]
                        else:
                            _sky_fill(result[y0:y1, x0:x1], _remaining, comp_mask)

            # ── Ring levelling: match the finished patch to the sky right beside it ──
            # Sky brightness drifts frame to frame (worst at a run's first and last
            # frames), so a borrowed patch can sit a few levels brighter or darker
            # than this frame's sky -- in a timelapse that flickers as a ghost
            # rectangle. Nudge the whole repaired patch, per colour channel, by the
            # median difference between the sky in a thin collar hugging the trail
            # and the patch itself. The collar is LOCAL (~15 px), so on a twilight
            # gradient it measures the warm glow next to the patch and warm fills
            # stay warm. Mid-run the difference measures ~0 and this is a no-op.
            # Collar pixels exclude every trail pixel in the window so nearby trails
            # cannot skew the sky measurement.
            _ring_off = None
            _ring_px = frame[y0:y1, x0:x1][_collar]
            _patch = result[y0:y1, x0:x1][comp_mask]
            if _ring_px.shape[0] >= 20 and _patch.shape[0] >= 20:
                _off = (np.median(_ring_px, axis=0).astype(np.float32) -
                        np.median(_patch, axis=0).astype(np.float32))
                # Cap the nudge to the frame-to-frame drift we ALREADY measured per-source
                # (the same spot, this frame vs the neighbor), plus a small margin. That drift is
                # the only legitimate correction and it ADAPTS to the scene (bright twilight vs
                # dark night). A large ring offset with ~0 measured drift means the collar is not
                # the same content as the patch -- a trail crossing dark foreground (branch,
                # horizon, rock, building) -- so it is refused and the already-correct fill is
                # kept: no black crush, no bright blow-out. _RING_OFF_CAP is the fallback only
                # when no per-source drift was measurable (too little collar to compare).
                _base_off = _off_n if _next_closer else _off_p
                if _base_off is None:
                    _base_off = _off_p if _off_p is not None else _off_n
                if _base_off is not None:
                    _cap = np.abs(_base_off) + _RING_DRIFT_MARGIN
                else:
                    _cap = float(_RING_OFF_CAP)
                _off = np.clip(_off, -_cap, _cap)
                result[y0:y1, x0:x1][comp_mask] = np.clip(
                    _patch.astype(np.float32) + _off, 0, _maxv).astype(frame.dtype)
                _ring_off = [round(float(v), 1) for v in _off]

            # ── Edge-speckle cleanup: isolated near-black dots left where a sharp foreground
            # edge jittered between neighbors (those pixels weren't "still", so they kept the dark
            # slid value). Where a patch pixel is near-black AND its local median is much brighter
            # (an isolated dot in bright surroundings), replace it with that median. Safe: a real
            # dark region medians dark (untouched); a star is bright (untouched). medianBlur ksize
            # 5 supports 8- and 16-bit, so it works on TIFFs too.
            _res = result[y0:y1, x0:x1]
            _rmax = _res.max(axis=2).astype(np.int16)
            _spk = comp_mask & (_rmax < _SPECKLE_DARK)
            if _spk.any():
                _med = cv2.medianBlur(_res, 5)
                _spk &= (_med.max(axis=2).astype(np.int16) - _rmax > _SPECKLE_MARGIN)
                if _spk.any():
                    _res[_spk] = _med[_spk]
            _addt("paste_s", time.perf_counter() - _tp)

            # ── Darken restore of dark static foreground (spikes, trunks, rocks) ──
            # Where the trail crosses dark foreground, the slide fill above borrows sky and
            # erases it. On a fixed tripod that foreground is the same dark pixel in every frame
            # and the trail is bright, so the per-pixel MIN across a +/-window of neighbors is the
            # true foreground with the trail rejected. _darken_fill replaces only the masked pixels
            # darker than a fraction of the local sky with that min (a hard replace, no feather);
            # brighter sky pixels keep the slide so moving stars stay put.
            _fg_darken_px, _fg_sky = 0, None
            if _DARKEN_FOREGROUND and N > 1:
                _w0 = max(0, frame_idx - _DARKEN_WINDOW)
                _w1 = min(N, frame_idx + _DARKEN_WINDOW + 1)
                if _w1 - _w0 >= 2:
                    _wstack = np.stack([neighbor_frames[_k][y0:y1, x0:x1]
                                        for _k in range(_w0, _w1)])
                    _dmin = np.min(_wstack, axis=0)                       # crisp foreground (the replace)
                    _dmed = np.median(_wstack, axis=0).astype(frame.dtype)  # typical value (the gate)
                    _res_d, _fg_darken_px, _fg_sky = _darken_fill(
                        result[y0:y1, x0:x1], _dmin, _dmed, comp_mask, _collar, _maxv, frame.dtype)
                    result[y0:y1, x0:x1] = _res_d

            if seg_info is not None:
                seg_info.update({
                    "tracking_ok":     _ok,
                    "cascade":         _cascade,
                    "dx":              round(_dx, 2),
                    "dy":              round(_dy, 2),
                    "n_stars":         _n_stars,
                    "method":          _method,
                    "still_trail_px":  _still_trail_px,
                    "edge_still_px":   _edge_still_px,
                    "fg_darken_px":    _fg_darken_px,
                    "union_zeroed_px": _union_zeroed_px,
                    "cross_reach_px":  _cross_reach_px,
                    "ring_off":        _ring_off,
                    "base":            "next" if _next_closer else "prev",
                })
                comp_dbg["segments"].append(seg_info)

        if debug_out is not None and comp_dbg is not None:
            debug_out["components"].append(comp_dbg)

    if debug_out is not None and _timing_acc is not None:
        debug_out["timing"] = {k: round(v, 3) for k, v in _timing_acc.items()}
    return result
