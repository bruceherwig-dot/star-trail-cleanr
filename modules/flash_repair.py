"""Single-frame flash finder for the finished star trail (sky half).

Plain English: a cosmic-ray hit or sensor glitch lights a few pixels in ONE frame
(sometimes two or three in a row) and nowhere else. In a lighten-max star trail that
one bright frame survives into the finished picture as a dot. This module finds them
by looking at every frame next to the frames before and after it.

HOW A FLASH IS TOLD FROM EVERYTHING ELSE (each filter removes ONE kind of error, and
each was added because a real photographer's picture showed that error):
  - a STAR moves, so it is compared with the neighbouring frames SHIFTED by the sky's
    measured motion, and ALSO unshifted (a stuck pixel does not move);
  - a PLANE or SATELLITE shows up as another flag a few pixels away in the next frames;
  - a LIGHTING EVENT (headlights, a flash of cloud) flags hundreds of spots in one frame;
  - a STREAK is long compared with the stars of this same sequence;
  - a SMOOTH blob is a star: a hit lands on the sensor and is never blurred by the lens,
    so a real flash has hard pixel edges. Judged against this sequence's own stars;
  - a TWINKLING star is dimmer, not gone, in the neighbouring frames.
Strongly coloured compact flashes skip the smooth / twinkling tests: no star is that
colour, and they were the specks photographers could see and wanted removed.

NOTHING HERE IS TUNED TO ONE CAMERA. Star size, star motion, noise and the colour of a
star's fringe differ enormously between cameras and lenses (measured on 31 sequences:
star colour fringe 18-255, star length 5-26 px, motion 0.03-19 px per frame), so the
star-relative numbers are measured from the sequence being cleaned. Where the code
cannot measure the sky's motion, or finds far too many flags, it DOES NOTHING and says
so, because removing nothing beats removing the wrong thing.

Research and tests behind every number: runs/overnight_2026_10_12 (ignored by git) and
memory/project/current/project_session_20261011.md.
"""
import collections
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# ── numbers that are properties of the physics, not of one camera ──────────────
MIN_PEAK = 100          # a hit saturates; a flash dimmer than this is not what we hunt
MAX_BLOB_AREA = 60      # px: bigger than this is not a speck (stars are 3-40 px)
MAX_BLOB_SIDE = 15      # px
SKY_PATCH_MAX = 256     # half-size of a motion-fit patch (clamped to the frame below)
MOTION_PAIR_STEP = 10   # re-measure the sky's motion every 10th frame pair; it drifts slowly
MAX_MOTION_PX = 40      # a measured shift bigger than this is a bad fit
MIN_MOTION_PATCHES = 6  # patches that must agree before a motion fit is believed
BURST_MIN = 10          # a frame with more flags than max(this, 5x typical) is a lighting event
BURST_FACTOR = 5
FLOOR_CANDIDATES = (40, 50, 60, 70, 80, 90, 100)   # contrast floors the sequence may pick from
COLOUR_MIN = 100        # colour split (max-min channel) that marks a hit, not a star
COLOUR_FLASH_MIN_CONTRAST = 60     # coloured flags this strong are also kept to FIND blinking spots
RESCUE_CONTRAST = 180   # a flash this bright is kept even when it scores as smooth
SHARP_MIN = 0.6         # keep a flag sharper than this share of the sequence's own stars
STILL_THERE = 0.2       # share of the flash still visible next door (noise subtracted) = a star
BOTH_SIDES_SIGMA = 3.0  # faint copy in BOTH neighbours, in noise-widths = a twinkling star
BAIL_FLAGS_PER_FRAME = 10          # more kept flags per frame than this means we are not looking at specks
MIN_FRAMES = 5
# blinking sensor defects: a pixel that flashes a strong colour in short bursts spread across the
# whole night (Glenn's blue dashes: 3-65 of 321 frames, often 2-6 in a row). A star's coloured
# fringe sits on a pixel for ONE stretch while the star passes; a defect comes back.
BASELINE_SAMPLES = 16       # frames sampled to learn each pixel's usual colour
BASELINE_CHROMA_MAX = 30    # a pixel whose usual colour split is above this is coloured scenery, not a defect
BLINK_MIN_BURSTS = 3        # separate bursts of flashes
BLINK_SPAN = 0.4            # ... spread over at least this share of the sequence
DEFECT_COLOUR_STRONG = 120
DEFECT_RING_COLOURED_MAX = 0.15
SPECK_TINT = 40
SPECK_TINT_HALO = 12
DEFECT_COLOUR_VISIBLE = 50  # a BLINKING DEFECT must also stand out in COLOUR from its surroundings in the
                            # finished picture. Brightness alone did not protect borrego's dense rainbow-fringed
                            # trails: every pixel there is coloured, the repair took the trails' own fringes for
                            # the defect and notched them. A real defect is a coloured point on a quieter ground.
REPAIR_CHROMA = 80          # in a repaired frame only pixels this coloured (plus 2 px of white core) change
REPAIR_ABOVE = 12           # ... and only if this far above the pixel's usual value
DEFECT_HALF = 5             # px: the window rebuilt around a blinking defect
# the GROUND (everything under the foreground mask) does not move, so no sky motion is needed:
# a pixel whose usual colour is neutral that shows a strong colour is a speck, for one frame
# or ten. Found by comparing with the pixel's OWN long-run colour, not with neighbour frames
# (a two-frame flash cannot beat both its neighbours).
GROUND_COLOUR_MIN = 80      # colour split that marks a speck on the ground
GROUND_BASE_MAX = 30        # ... where the pixel's usual colour split is below this
GROUND_HALF = 16            # px: a ground speck can be 15 px tall (a red/white streak)
GROUND_REACH = 8            # px a speck may grow from its strongly coloured pixels through connected bright ones
GROUND_BRIGHT_ABOVE = 45    # a connected pixel this far above its usual value belongs to the speck, coloured or white
GROUND_WEAK_COLOUR = 25     # ... or this weakly coloured and 12 above
GROUND_BASELINE_SAMPLES = 16
VISIBLE_MIN = 20            # a spot is only worth repairing if, in the FINISHED picture, it stands out this
                            # far above the brightest tenth of the pixels 3-7 px around it. A speck lost
                            # inside a bright trail cannot be seen, and touching it nicked the trail (Glenn:
                            # a solid trail came out broken into dashes).


def chan_max(img):
    """Per-pixel maximum over the colour channels (6 ms; numpy's .max(2) takes 158 ms
    on a 24 MP frame, with the identical result)."""
    b, g, r = cv2.split(img)
    return cv2.max(cv2.max(b, g), r)


# ── per-flag measurements ─────────────────────────────────────────────────────
def _chroma(img):
    """Per-pixel colour split (max channel minus min channel)."""
    b, g, r = cv2.split(img)
    return cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))


def usual_colour(n_frames, read_frame, size):
    """Each pixel's usual colour split: the MINIMUM over a spread of frames. A defect is
    on in a few percent of frames, so the minimum stays low where the pixel is neutral;
    streaming the minimum keeps memory at one frame (a median would need them all)."""
    base = None
    for i in np.unique(np.linspace(0, n_frames - 1, min(BASELINE_SAMPLES, n_frames)).astype(int)):
        img = read_frame(int(i))
        if img is None or img.shape[:2] != size:
            continue
        c = _chroma(img)
        base = c if base is None else cv2.min(base, c)
    return base


def _local_med_sigma(g, x, y):
    win = g[max(0, y - 14):y + 15, max(0, x - 14):x + 15].astype(np.float64)
    med = float(np.median(win))
    return med, max(1.0, 1.4826 * float(np.median(np.abs(win - med))))


def _excess(g, x, y):
    med, sg = _local_med_sigma(g, x, y)
    return float(g[max(0, y - 3):y + 4, max(0, x - 3):x + 4].max()) - med, sg


def _sharp_features(img, g, x, y):
    """(steepness, channel-shape mismatch, roughness) of the point at (x, y), plus the
    peak pixel found. These three separated 38 known flashes from ~7,900 stars at
    0.80-0.86 each; combined 0.87. Returns None near the frame edge."""
    h, w = g.shape
    if not (8 < x < w - 8 and 8 < y < h - 8):
        return None
    win = g[y - 2:y + 3, x - 2:x + 3]
    j = np.unravel_index(int(np.argmax(win)), win.shape)
    py, px = y - 2 + j[0], x - 2 + j[1]
    peak = float(g[py, px])
    nn = g[py - 1:py + 2, px - 1:px + 2].astype(np.float32)
    ring = g[py - 5:py + 6, px - 5:px + 6][[0, 10]].ravel()
    bg = float(np.median(ring))
    steep = (peak - (nn.sum() - peak) / 8.0) / max(1.0, peak - bg)
    ch = img[py - 1:py + 2, px - 1:px + 2].reshape(-1, 3).astype(float)
    cvar = float(np.std(ch / np.maximum(ch.max(0), 1.0), axis=0).mean())
    rough = (float(np.abs(np.diff(nn, 2, axis=0)).sum() + np.abs(np.diff(nn, 2, axis=1)).sum())
             / max(1.0, peak - bg))
    return (steep, cvar, rough), (px, py)


def _object_length(g, px, py):
    """Length in px of the whole bright object under the flash, measured at a LOW
    threshold so a streak shows its full length."""
    c = g[max(0, py - 60):py + 61, max(0, px - 60):px + 61].astype(np.float32)
    cx, cy = min(px, 60), min(py, 60)
    bg = float(np.median(c))
    peak = float(c[cy, cx])
    thr = bg + max(25.0, 0.25 * (peak - bg))
    _n, lab, st, _c = cv2.connectedComponentsWithStats((c >= thr).astype(np.uint8), 8)
    lbl = lab[cy, cx]
    return int(max(st[lbl, cv2.CC_STAT_WIDTH], st[lbl, cv2.CC_STAT_HEIGHT])) if lbl else 0


def _still_there(g0, gp, gn, x, y):
    """Largest share of this flash still visible in a neighbouring frame AFTER taking
    off 4 noise-widths (so a bright noisy sky does not read as 'still there')."""
    e0, _ = _excess(g0, x, y)
    if e0 <= 0:
        return 0.0
    best = 0.0
    for g in (gp, gn):
        e, sg = _excess(g, x, y)
        best = max(best, max(0.0, e - 4.0 * sg) / e0)
    return best


def _both_sides_sigma(gp, gn, x, y):
    out = []
    for g in (gp, gn):
        med, sg = _local_med_sigma(g, x, y)
        out.append((float(g[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()) - med) / sg)
    return out


# ── the sky's motion ──────────────────────────────────────────────────────────
def measure_motion(ga, gb, fg):
    """Affine map (2x3) taking a star's position in frame A to its position in frame B,
    from phase correlation on sky patches, or None when too few patches agree. Median
    blur first so single stuck pixels cannot pin the answer at zero."""
    h, w = ga.shape
    s = int(min(SKY_PATCH_MAX, min(h, w) // 6))
    if s < 24:
        return None
    win = cv2.createHanningWindow((2 * s, 2 * s), cv2.CV_32F)
    a = cv2.medianBlur(ga, 3)
    b = cv2.medianBlur(gb, 3)
    src, dst = [], []
    for gy in range(5):
        for gx in range(7):
            cx, cy = int((gx + .5) * w / 7), int((gy + .5) * h / 5)
            if cy - s < 0 or cy + s > h or cx - s < 0 or cx + s > w:
                continue
            if fg is not None and fg[cy - s:cy + s, cx - s:cx + s].mean() > 0.15:
                continue
            pa = a[cy - s:cy + s, cx - s:cx + s].astype(np.float32)
            pb = b[cy - s:cy + s, cx - s:cx + s].astype(np.float32)
            (dx, dy), resp = cv2.phaseCorrelate(pa, pb, win)
            if resp >= 0.3 and np.hypot(dx, dy) < MAX_MOTION_PX:
                src.append([cx, cy, 1.0])
                dst.append([cx + dx, cy + dy])
    if len(src) < MIN_MOTION_PATCHES:
        return None
    p, *_ = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)
    return p.T


# ── reference stars: what a star looks like in THIS sequence ───────────────────
class StarReference:
    """Sharpness, colour fringe and length of this sequence's own stars. Every
    star-relative cutoff comes from here, so nothing is a number from one camera."""

    def __init__(self, frames, fg):
        feats, chroma, lens = [], [], []
        for img in frames:
            g = chan_max(img)
            sky = g if fg is None else g[~fg]
            thr = max(MIN_PEAK, 0.5 * float(np.percentile(sky, 99.9)))
            m = (g >= thr).astype(np.uint8)
            if fg is not None:
                m &= (~fg).astype(np.uint8)
            n, _lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, min(n, 3000)):
                if 3 <= st[i, cv2.CC_STAT_AREA] <= MAX_BLOB_AREA:
                    lens.append(int(max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])))
                    f = _sharp_features(img, g, int(cen[i][0]), int(cen[i][1]))
                    if f is not None:
                        feats.append(f[0])
                        (px, py) = f[1]
                        w = img[py - 1:py + 2, px - 1:px + 2].reshape(-1, 3).astype(int)
                        chroma.append(int((w.max(1) - w.min(1)).max()))
        self.n = len(feats)
        self.feats = np.array(feats) if feats else np.zeros((0, 3))
        self.length_p95 = int(np.percentile(lens, 95)) if lens else 9
        self.chroma_p99 = int(np.percentile(chroma, 99)) if chroma else 0
        self.length_cut = max(15, 3 * self.length_p95)
        # if a star's own colour fringe already reaches the top of the scale, colour
        # cannot tell a hit from a star in this sequence: turn the colour exemption off
        self.colour_usable = self.chroma_p99 < 250

    def sharpness(self, f):
        """Share of this sequence's stars the point out-sharpens, averaged over three measures."""
        if self.n < 30:
            return 1.0          # too few stars to judge: do not condemn on a guess
        return float(np.mean([(self.feats[:, i] < f[i]).mean() for i in range(3)]))


class Flag:
    __slots__ = ("frame", "x", "y", "contrast", "half", "colour", "pix")

    def __init__(self, frame, x, y, contrast, half, colour, pix=None):
        self.frame, self.x, self.y = frame, x, y
        self.contrast, self.half, self.colour = contrast, half, colour
        self.pix = pix          # (ys, xs) of the flash's own pixels: ONLY these are repaired, never a star beside it

    def __repr__(self):
        return "Flag(frame=%d, x=%d, y=%d, contrast=%d, colour=%s)" % (
            self.frame, self.x, self.y, self.contrast, self.colour)


class FlashResult:
    """What the sky pass found. `flags` are the single-frame flashes to repair;
    `colour_flags` are strongly coloured candidates (any frame count) used later to
    find blinking defects. `bail` is None, or the plain-English reason nothing was done."""

    def __init__(self):
        self.flags = []
        self.colour_flags = []
        self.removed = collections.Counter()
        self.floor = None
        self.bail = None
        self.frames = 0
        self.burst_frames = 0
        self.motion_pairs_measured = 0
        self.motion_pairs_failed = 0
        self.raw_per_frame = []        # raw candidate count per frame, for the Star Log and for testing
        self.defect_candidates = []    # (frame, x, y): strongly coloured pixels where the usual colour is neutral
        self.star_reference = None


def find_flashes(n_frames, read_frame, fg_mask=None, progress=None, threads=4):
    """Run the sky pass over frames 0..n_frames-1.

    read_frame(i) -> BGR uint8 image or None. fg_mask: HxW bool (True = foreground,
    skipped here) or None. Every frame is read exactly ONCE and kept in a rolling cache
    (the previous version read each frame seven times and was five times slower). The
    only repeats are the few frames sampled up front to learn each pixel's usual colour.
    """
    res = FlashResult()
    res.frames = n_frames
    if n_frames < MIN_FRAMES:
        res.bail = "too few frames (%d) to compare each with its neighbours" % n_frames
        return res
    pool = ThreadPoolExecutor(max_workers=threads)
    futs = {}
    cache = {}

    def prefetch(i):
        if 0 <= i < n_frames and i not in futs:
            futs[i] = pool.submit(read_frame, i)

    def get(i):
        i = min(max(i, 0), n_frames - 1)
        if i not in cache:
            prefetch(i)
            img = futs[i].result()
            if img is not None:
                cache[i] = (img, chan_max(img))
            else:
                # an unreadable frame stands in as its neighbour: nothing can be flagged there
                cache[i] = cache.get(i - 1) or (get(i - 1) if i > 0 else None)
        return cache[i]

    first = get(0)
    if first is None:
        res.bail = "the first frame could not be read"
        return res
    H, W = first[1].shape
    fg = None
    if fg_mask is not None:
        fg = fg_mask if fg_mask.shape == (H, W) else cv2.resize(
            fg_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    fg = fg.astype(bool) if fg is not None else None
    sky_mask = np.ones((H, W), bool) if fg is None else ~fg

    ref_frames = []
    for i in (n_frames // 5, n_frames // 2, 4 * n_frames // 5):
        r = get(i)
        if r is not None:
            ref_frames.append(r[0])
    star = StarReference(ref_frames, fg)
    res.star_reference = star

    base_chroma = usual_colour(n_frames, read_frame, (H, W))
    k5 = np.ones((5, 5), np.uint8)
    raw_by = {}
    counts = []
    Mlast = None
    kept_all = []                    # flags that passed EVERY test, at the lowest floor; the sequence's floor picks later
    early_all = []                   # flags that passed the shape tests only: the stage the elbow rule was measured at
    colour_flags = []

    def process(k):
        lst = raw_by.get(k, [])
        if not lst:
            return
        typ = float(np.median(counts)) if counts else 0.0       # over ALL frames: a quiet sky is quiet
        if len(lst) > max(BURST_MIN, BURST_FACTOR * typ):
            res.burst_frames += 1
            res.removed["lighting event"] += len(lst)
            return
        cur, prev, nxt = get(k), get(k - 1), get(k + 1)
        if cur is None or prev is None or nxt is None:
            return
        cim, c = cur
        p, nx = prev[1], nxt[1]
        for (kk, x, y, dd, w, h, mot, asp, blob) in lst:
            reach = 40.0 + 4.0 * mot
            moved = False
            for q in (k - 2, k - 1, k + 1, k + 2):
                for o in raw_by.get(q, []):
                    if 3.0 < np.hypot(x - o[1], y - o[2]) <= reach:
                        moved = True
                        break
                if moved:
                    break
            if moved:
                res.removed["moving object"] += 1
                continue
            f = _sharp_features(cim, c, x, y)
            if f is None:
                continue
            feat, (px, py) = f
            wv = cim[py - 1:py + 2, px - 1:px + 2].reshape(-1, 3).astype(int)
            chroma = int((wv.max(1) - wv.min(1)).max())
            colour = bool(star.colour_usable and chroma >= COLOUR_MIN and asp < 1.6)
            if colour and dd >= COLOUR_FLASH_MIN_CONTRAST:
                colour_flags.append((k, px, py))
            if _object_length(c, px, py) > star.length_cut:
                res.removed["streak"] += 1
                continue
            if not (colour or star.sharpness(feat) >= SHARP_MIN or dd >= RESCUE_CONTRAST):
                res.removed["smooth (a star)"] += 1
                continue
            flag = Flag(k, px, py, dd, max(4, max(w, h) // 2 + 3), colour, blob)
            early_all.append(flag)
            if not colour:
                if _still_there(c, p, nx, px, py) >= STILL_THERE:
                    res.removed["still there next frame (a star)"] += 1
                    continue
                bs = _both_sides_sigma(p, nx, px, py)
                if bs[0] >= BOTH_SIDES_SIGMA and bs[1] >= BOTH_SIDES_SIGMA:
                    res.removed["twinkling star"] += 1
                    continue
            kept_all.append(flag)

    floor_detect = min(FLOOR_CANDIDATES)
    for i in range(2, 6):
        prefetch(i)
    pg = get(0)[1]
    cg = get(1)[1]
    pd = cv2.dilate(pg, k5)
    cd = cv2.dilate(cg, k5)
    for k in range(1, n_frames - 1):
        for j in range(k + 2, k + 6):
            prefetch(j)
        nxt = get(k + 1)
        ng = nxt[1]
        nd = cv2.dilate(ng, k5)
        if Mlast is None or k % MOTION_PAIR_STEP == 1:
            Mt = measure_motion(cg, ng, fg)
            if Mt is None:
                res.motion_pairs_failed += 1
            else:
                res.motion_pairs_measured += 1
                Mlast = Mt
            if Mlast is None and res.motion_pairs_failed >= 20:
                res.bail = "the sky's motion could not be measured, so stars cannot be told from flashes"
                break
        M = Mlast if Mlast is not None else np.array([[1, 0, 0], [0, 1, 0]], float)
        Minv = cv2.invertAffineTransform(M.astype(np.float32))
        if base_chroma is not None and star.colour_usable:
            cand = ((_chroma(get(k)[0]) >= COLOUR_MIN) & (base_chroma < BASELINE_CHROMA_MAX) & sky_mask).astype(np.uint8)
            ncc, _l, stc, cenc = cv2.connectedComponentsWithStats(cand, 8)
            for i in range(1, ncc):
                if stc[i, cv2.CC_STAT_AREA] <= MAX_BLOB_AREA:
                    res.defect_candidates.append((k, int(cenc[i][0]), int(cenc[i][1])))
        d = cg.astype(np.int16) - np.maximum(pd, nd).astype(np.int16)
        mask = ((d >= floor_detect) & (cg >= MIN_PEAK) & sky_mask).astype(np.uint8)
        nc, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        mot = float(np.hypot(*(M[:, 2] + M[:, :2] @ np.array([W / 2, H / 2]) - np.array([W / 2, H / 2]))))
        lst = []

        def around(g, pt, r=4):
            xx, yy = int(round(pt[0])), int(round(pt[1]))
            if 0 <= xx < W and 0 <= yy < H:
                return int(g[max(0, yy - r):yy + r + 1, max(0, xx - r):xx + r + 1].max())
            return 0

        for i in range(1, nc):
            area = st[i, cv2.CC_STAT_AREA]
            bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if area > MAX_BLOB_AREA or max(bw, bh) > MAX_BLOB_SIDE:
                continue
            x, y = int(cen[i][0]), int(cen[i][1])
            pk = int(cg[y, x])
            xn = M @ np.array([x, y, 1.0])
            xp = Minv @ np.array([x, y, 1.0])
            # the SHIFTED neighbours: where a star at (x, y) would be one frame later / earlier
            if pk - max(around(ng, xn), around(pg, xp)) < floor_detect:
                continue
            x0, y0 = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            sub = lab[y0:y0 + bh, x0:x0 + bw] == i
            yy, xx = np.nonzero(sub)
            asp = 1.0
            if len(xx) >= 3:
                (_a, _b), (rw, rh), _t = cv2.minAreaRect(np.c_[xx, yy].astype(np.float32))
                asp = max(rw, rh) / max(1.0, min(rw, rh))
            lst.append((k, x, y, int(d[y0:y0 + bh, x0:x0 + bw][sub].max()), int(bw), int(bh), mot, round(asp, 2),
                        (yy + y0, xx + x0)))
        raw_by[k] = lst
        counts.append(len(lst))
        if k >= 3:
            process(k - 2)
        for old in [q for q in cache if q < k - 4]:
            cache.pop(old, None)
            futs.pop(old, None)
        pg, cg, pd, cd = cg, ng, cd, nd
        if progress is not None and k % 100 == 0:
            progress(k, n_frames)
    pool.shutdown(wait=False)
    if res.bail:
        return res
    for kk in (n_frames - 3, n_frames - 2):
        if kk >= 1:
            process(kk)

    # ── the sequence picks its OWN contrast floor ("elbow" rule): the lowest floor at
    # which the flag rate is no more than twice the rate at 100. Counted BEFORE the
    # still-there / twinkling tests (which empty the low-contrast end on their own and
    # would flatten the curve): measured that way on four sets it gave 90, 70, 50 and 60
    # where one fixed number could not suit them all.
    res.raw_per_frame = counts
    nfr = max(1, n_frames - 2)
    rate = {f: sum(1 for fl in early_all if fl.contrast >= f) / nfr for f in FLOOR_CANDIDATES}
    ceiling = 2.0 * rate[100]
    chosen = 100
    for f in FLOOR_CANDIDATES:
        if rate[f] <= max(ceiling, 1.0 / nfr):
            chosen = f
            break
    res.floor = chosen
    res.flags = [fl for fl in kept_all if fl.contrast >= chosen]
    res.colour_flags = colour_flags
    per_frame = len(res.flags) / nfr
    if per_frame > BAIL_FLAGS_PER_FRAME:
        res.bail = ("%.1f flashes per frame is far more than specks: something other than "
                    "specks is being flagged, so nothing was repaired" % per_frame)
        res.flags = []
    return res



# ── blinking sensor defects ───────────────────────────────────────────────────
def _bursts(frames, gap):
    fr = sorted(frames)
    runs = 1
    for a, b in zip(fr, fr[1:]):
        if b - a > gap:
            runs += 1
    return runs


def find_blinking_defects(candidates, n_frames):
    """Pixels that flash a strong colour in separate bursts spread across the night.

    Measured on borrego_springs_1: 18,312 pixels flashed in 3+ frames, but only 529 did
    so in 3+ separate bursts spread over 40% of the night; the other 85% were the
    coloured fringes of stars crossing. A red speck that flashed in 11 of 1,559 frames
    (bursts at frames 77, 391, 532, 799, 897, 1387, 1448) passes easily. It is NOT found
    by asking whether a flash beats both neighbouring frames, because a defect often
    lasts two or three frames in a row.
    """
    gap = max(4, n_frames // 100)
    cell = collections.defaultdict(set)
    for k, x, y in candidates:
        cell[(x // 3, y // 3)].add(k)
    found = []
    for (cx, cy), frames in cell.items():
        if len(frames) < BLINK_MIN_BURSTS:
            continue
        if max(frames) - min(frames) < BLINK_SPAN * n_frames:
            continue
        if _bursts(frames, gap) >= BLINK_MIN_BURSTS:
            found.append((len(frames), cx * 3 + 1, cy * 3 + 1))
    found.sort(reverse=True)
    out = []
    for _n, x, y in found:                       # one spot per place: drop a neighbour within 4 px
        if all(abs(x - ox) > 4 or abs(y - oy) > 4 for ox, oy in out):
            out.append((x, y))
    return out


# ── the repair ────────────────────────────────────────────────────────────────
def _flash_mask(win, k, half):
    """Boolean (2h+1, 2h+1) mask of the pixels that ARE the flash in frame k (its own blob
    plus 1 px), or None when the blob pixels are unknown (then the whole window is used)."""
    plist = win["pix"].get(k)
    if not plist:
        return None
    n = 2 * half + 1
    m = np.zeros((n, n), np.uint8)
    for ys, xs in plist:
        ly, lx = np.asarray(ys) - (win["y"] - half), np.asarray(xs) - (win["x"] - half)
        ok = (ly >= 0) & (ly < n) & (lx >= 0) & (lx < n)
        m[ly[ok], lx[ok]] = 1
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


def _window_index(xs, ys, half, h, w):
    """Index arrays (n, 2*half+1, 2*half+1) for gathering square windows, clamped to the frame."""
    d = np.arange(-half, half + 1)
    yy = np.clip(np.asarray(ys)[:, None, None] + d[None, :, None], 0, h - 1)
    xx = np.clip(np.asarray(xs)[:, None, None] + d[None, None, :], 0, w - 1)
    return yy, xx


def _stands_out(gray, yy, xx, rebuilt, original, chroma=None):
    """Per window: does the speck being removed stand out in the finished picture?
    Takes the brightest pixel the repair would lower and compares it with the brightest
    tenth of the pixels 3-7 px around it."""
    H, W = gray.shape
    changed = (rebuilt < original).any(3)
    keep = np.zeros(len(yy), bool)
    for j in range(len(yy)):
        if not changed[j].any():
            continue
        vals = np.where(changed[j], (gray if chroma is None else chroma)[yy[j], xx[j]], -1)   # a defect is judged by its most COLOURED pixel
        a, b = np.unravel_index(int(np.argmax(vals)), vals.shape)
        wy = np.broadcast_to(yy[j], vals.shape)             # the index arrays are (w,1) and (1,w): broadcast to the window
        wx = np.broadcast_to(xx[j], vals.shape)
        py, px = int(wy[a, b]), int(wx[a, b])
        y0, y1, x0, x1 = max(0, py - 7), min(H, py + 8), max(0, px - 7), min(W, px + 8)
        win = gray[y0:y1, x0:x1]
        gy, gx = np.ogrid[y0:y1, x0:x1]
        d = np.maximum(np.abs(gy - py), np.abs(gx - px))
        ring = win[(d >= 3) & (d <= 7)]
        ok = ring.size > 0 and int(gray[py, px]) - int(np.percentile(ring, 90)) >= VISIBLE_MIN
        if chroma is not None and ring.size > 0:
            cring = chroma[y0:y1, x0:x1][(d >= 3) & (d <= 7)]
            # against the TYPICAL (median) pixel around it, not the most coloured tenth: a bright trail
            # beside a real dash would raise that bar out of reach (Glenn's dashes 3 and 4)
            excess = int(chroma[py, px]) - int(np.median(cring))
            # a blue dash is dark in grey, so a bright trail beside it fails the brightness test:
            # a very strong colour excess stands in for it
            ok = excess >= DEFECT_COLOUR_VISIBLE and (ok or excess >= DEFECT_COLOUR_STRONG)
            # a real dash sits in quiet sky (measured: 1-4% of the ring strongly coloured); the rainbow fringe of a
            # dense trail has 22-29% (borrego): that is a trail, not a speck, and is left alone
            ok = ok and float((cring >= REPAIR_CHROMA).mean()) < DEFECT_RING_COLOURED_MAX
        keep[j] = ok
    return keep


def repair_stack(stack, n_frames, read_frame, flags, defects, threads=4):
    """Return (new_stack, stats): `stack` with each single-frame flash and each blinking
    defect removed. The stack is whatever the caller built (plain or comet), so nothing
    is re-stacked; instead each flagged spot is REBUILT from the frames (the maximum of
    its window over the sequence, with the flash taken out) and the result is the
    DARKER of that and the stack, so a repair can only remove light, never add it.

      single flash: in the flash frame the window becomes the lower of that frame and the
        brighter of the frame before / after, so a star sitting in the window is kept;
      blinking defect: in every frame only the pixels carrying the speck's colour (plus 2 px
        of white core) go back to the pixel's own usual value (40th percentile over the
        sequence). NEVER 'every bright pixel in the window': on an overexposed ground that
        painted black squares into bright rock.
    """
    out = stack.copy()
    H, W = out.shape[:2]
    stats = {"flash_spots": 0, "defect_spots": 0, "pixels_changed": 0}
    by_cell = collections.OrderedDict()
    for fl in flags:
        key = (fl.x // 6, fl.y // 6)
        if key not in by_cell:
            by_cell[key] = {"x": fl.x, "y": fl.y, "half": fl.half, "frames": set(), "pix": {}}
        by_cell[key]["frames"].add(fl.frame)
        if fl.pix is not None:
            by_cell[key]["pix"].setdefault(fl.frame, []).append(fl.pix)
        by_cell[key]["half"] = max(by_cell[key]["half"], fl.half)
    groups = collections.defaultdict(list)                    # half -> list of windows
    for w in by_cell.values():
        groups[min(9, int(w["half"]))].append(w)
    stats["flash_spots"] = len(by_cell)
    dx = np.array([d[0] for d in defects], int)
    dy = np.array([d[1] for d in defects], int)
    stats["defect_spots"] = len(defects)
    if not by_cell and not len(defects):
        return out, stats

    gidx = {}
    for half, wins in groups.items():
        xs = [w["x"] for w in wins]
        ys = [w["y"] for w in wins]
        yy, xx = _window_index(xs, ys, half, H, W)
        gidx[half] = {"yy": yy, "xx": xx, "wins": wins,
                      "max": np.zeros(yy.shape + (3,), np.uint8),
                      "frames": [w["frames"] for w in wins], "half": half}
    if len(defects):
        dyy, dxx = _window_index(dx, dy, DEFECT_HALF, H, W)
        # each defect pixel's usual value: 40th percentile over a spread of frames
        samp = np.unique(np.linspace(0, n_frames - 1, min(60, n_frames)).astype(int))
        crops = []
        for i in samp:
            img = read_frame(int(i))
            if img is not None and img.shape[:2] == (H, W):
                crops.append(img[dyy, dxx])
        base = (np.percentile(np.stack(crops), 40, axis=0).astype(np.uint8)
                if crops else None)
        dmax = np.zeros(dyy.shape + (3,), np.uint8)
    else:
        base = None

    pool = ThreadPoolExecutor(max_workers=threads)
    futs = {}

    def prefetch(i):
        if 0 <= i < n_frames and i not in futs:
            futs[i] = pool.submit(read_frame, i)

    def frame(i):
        i = min(max(i, 0), n_frames - 1)
        prefetch(i)
        img = futs[i].result()
        return img if img is not None and img.shape[:2] == (H, W) else None

    for k in range(n_frames):
        for j in range(k + 1, k + 4):
            prefetch(j)
        cur = frame(k)
        if cur is None:
            continue
        for half, g in gidx.items():
            crops = cur[g["yy"], g["xx"]]
            hit = [i for i, fs in enumerate(g["frames"]) if k in fs]
            if hit:
                prv, nxt = frame(k - 1), frame(k + 1)
                if prv is not None and nxt is not None:
                    nb = np.maximum(prv[g["yy"][hit], g["xx"][hit]], nxt[g["yy"][hit], g["xx"][hit]])
                    crops = crops.copy()
                    low = np.minimum(crops[hit], nb)
                    for slot, i in enumerate(hit):
                        m = _flash_mask(g["wins"][i], k, half)
                        if m is None:
                            continue
                        sel = low[slot]
                        sel[~m] = crops[i][~m]                  # a star beside the flash keeps its pixels
                        low[slot] = sel
                    crops[hit] = low
            g["max"] = np.maximum(g["max"], crops)
        if base is not None:
            crops = cur[dyy, dxx]
            ch = crops.max(3).astype(np.int16) - crops.min(3).astype(np.int16)
            # the speck's own colour channel = the strongest channel of its most coloured centre pixel; a pixel
            # belongs to the speck only if it leans toward that channel (a mint-green trail touching a blue dash
            # does not, so it is not taken along with it)
            c0, c1 = DEFECT_HALF - 1, DEFECT_HALF + 2
            cc = ch[:, c0:c1, c0:c1].reshape(len(dx), -1)
            top = cc.argmax(1)
            cpx = crops[:, c0:c1, c0:c1].reshape(len(dx), -1, 3)[np.arange(len(dx)), top]
            cidx = cpx.argmax(1)
            sel = np.take_along_axis(crops.astype(np.int16), cidx[:, None, None, None].repeat(crops.shape[1], 1).repeat(crops.shape[2], 2), 3)[..., 0]
            green = crops[..., 1].astype(np.int16)
            other = np.where((cidx == 1)[:, None, None], np.maximum(crops[..., 0], crops[..., 2]).astype(np.int16), green)
            tint = sel - other
            sp = np.zeros((len(dx), 2 * DEFECT_HALF + 3, 2 * DEFECT_HALF + 3), bool)
            sp[:, 1:-1, 1:-1] = (ch >= REPAIR_CHROMA) & (tint >= SPECK_TINT)
            grown = sp.copy()
            for a in range(-1, 2):                      # 1 px of reach: the speck's own halo / white core
                for b in range(-1, 2):
                    grown |= np.roll(np.roll(sp, a, 1), b, 2)
            grown = grown[:, 1:-1, 1:-1] & (tint >= SPECK_TINT_HALO)
            above = crops.max(3).astype(np.int16) - base.max(3).astype(np.int16)
            bad = grown & (above >= REPAIR_ABOVE)
            # repair only in frames where the defect ITSELF shows its colour: a trail crossing it in other frames is left alone
            on = (sp[:, 1:-1, 1:-1][:, c0:c1, c0:c1] & (above[:, c0:c1, c0:c1] >= REPAIR_ABOVE)).any((1, 2))
            bad &= on[:, None, None]
            crops = crops.copy()
            crops[bad] = base[bad]
            dmax = np.maximum(dmax, crops)
        futs.pop(k - 2, None)
    pool.shutdown(wait=False)

    gray = chan_max(stack)
    stack_chroma = _chroma(stack)
    hidden = 0
    for g in gidx.values():
        orig = stack[g["yy"], g["xx"]]
        ok = _stands_out(gray, g["yy"], g["xx"], g["max"], orig)
        hidden += int((~ok).sum())
        np.minimum.at(out, (g["yy"][ok], g["xx"][ok]), g["max"][ok])
    if base is not None:
        orig = stack[dyy, dxx]
        ok = _stands_out(gray, dyy, dxx, dmax, orig, stack_chroma)
        hidden += int((~ok).sum())
        np.minimum.at(out, (dyy[ok], dxx[ok]), dmax[ok])
    stats["hidden_not_repaired"] = hidden
    stats["pixels_changed"] = int((out != stack).any(2).sum())
    return out, stats


# ── the ground ────────────────────────────────────────────────────────────────
class GroundResult:
    """Specks on the (static) ground. `bail` is None, or why nothing was done there."""

    def __init__(self):
        self.spots = []              # (x, y)
        self.burst_frames = set()    # frames where the whole ground changed (lighting events)
        self.candidates = 0
        self.flags_per_frame = 0.0
        self.bail = None


def find_ground_specks(n_frames, read_frame, fg_mask):
    """Find strongly coloured specks under the foreground mask.

    Guards, each from a real failure on a photographer's set:
      - LIGHTING-EVENT frames (Cheryl Hanscom Wilcox, overexposed pink rock: frame 7 alone
        gave 580 false flags while a normal frame had none) are skipped entirely;
      - if the ground STILL gives more than BAIL_FLAGS_PER_FRAME flags per frame, it is not
        speckled but coloured scenery (Juniper's warm lit trees gave ~100 per frame), so
        nothing is done there.
    """
    res = GroundResult()
    if fg_mask is None or not np.asarray(fg_mask).any():
        res.bail = "no foreground mask: there is no ground to look at"
        return res
    fg = np.asarray(fg_mask).astype(bool)
    first = read_frame(0)
    if first is None:
        res.bail = "the first frame could not be read"
        return res
    H, W = first.shape[:2]
    if fg.shape != (H, W):
        fg = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    y0 = max(0, int(np.where(fg.any(1))[0].min()) - 8)
    fgd = cv2.dilate(fg[y0:].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    samples = []
    for i in np.unique(np.linspace(0, n_frames - 1, min(GROUND_BASELINE_SAMPLES, n_frames)).astype(int)):
        img = read_frame(int(i))
        if img is not None and img.shape[:2] == (H, W):
            samples.append(_chroma(img[y0:]))
    if len(samples) < 3:
        res.bail = "too few readable frames to learn the ground's usual colour"
        return res
    base = np.median(np.stack(samples), axis=0).astype(np.uint8)
    del samples

    cand = []
    counts = collections.Counter()
    for k in range(n_frames):
        img = read_frame(k)
        if img is None or img.shape[:2] != (H, W):
            continue
        c = _chroma(img[y0:])
        m = ((c >= GROUND_COLOUR_MIN) & (base < GROUND_BASE_MAX) & fgd).astype(np.uint8)
        n, _lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] <= MAX_BLOB_AREA:
                cand.append((k, int(cen[i][0]), int(cen[i][1]) + y0))
                counts[k] += 1
    res.candidates = len(cand)
    typical = float(np.median([counts.get(k, 0) for k in range(n_frames)]))
    res.burst_frames = {k for k, v in counts.items() if v > max(BURST_MIN, BURST_FACTOR * typical)}
    cand = [c for c in cand if c[0] not in res.burst_frames]
    res.flags_per_frame = len(cand) / max(1, n_frames)
    if res.flags_per_frame > BAIL_FLAGS_PER_FRAME:
        res.bail = ("the ground gives %.1f coloured flags per frame: it is coloured scenery, "
                    "not specks, so nothing was repaired there" % res.flags_per_frame)
        return res
    cell = collections.defaultdict(list)
    for k, x, y in cand:
        cell[(x // 6, y // 6)].append((x, y))
    for pts in cell.values():
        x = int(np.median([p[0] for p in pts]))
        y = int(np.median([p[1] for p in pts]))
        if GROUND_HALF <= x < W - GROUND_HALF and GROUND_HALF <= y < H - GROUND_HALF:
            res.spots.append((x, y))
    return res


def repair_ground(stack, n_frames, read_frame, ground, trail_cut=25, threads=4):
    """Return (new_stack, stats) with the ground's specks removed.

    Only each frame's OWN speck-coloured pixels (plus GROUND_REACH px of white core) go
    back to the pixel's usual value. Rewriting the whole window painted black squares into
    an overexposed ground (the 'usual value' is the DARK quiet-frame ground, but a few lit
    frames make the stack bright), and replacing 'any bright pixel' made it worse.
    A spot sitting on a long bright line in the finished stack is a star trail crossing the
    ground's edge, not a speck: left alone (a yellow trail tip was cut once).
    """
    out = stack.copy()
    H, W = out.shape[:2]
    stats = {"ground_spots": 0, "on_a_trail": 0, "pixels_changed": 0}
    if ground is None or ground.bail or not ground.spots:
        return out, stats
    gray = chan_max(stack)
    spots = []
    for x, y in ground.spots:
        if _object_length(gray, x, y) > trail_cut:
            stats["on_a_trail"] += 1
        else:
            spots.append((x, y))
    stats["ground_spots"] = len(spots)
    if not spots:
        return out, stats
    sx = np.array([s_[0] for s_ in spots], int)
    sy = np.array([s_[1] for s_ in spots], int)
    yy, xx = _window_index(sx, sy, GROUND_HALF, H, W)
    skip = ground.burst_frames
    samp = [int(i) for i in np.unique(np.linspace(0, n_frames - 1, min(60, n_frames)).astype(int))
            if int(i) not in skip]
    crops = []
    for i in samp:
        img = read_frame(i)
        if img is not None and img.shape[:2] == (H, W):
            crops.append(img[yy, xx])
    if not crops:
        return out, stats
    base = np.percentile(np.stack(crops), 40, axis=0).astype(np.uint8)
    del crops
    wmax = np.zeros(yy.shape + (3,), np.uint8)
    pool = ThreadPoolExecutor(max_workers=threads)
    futs = {}
    for k in range(n_frames):
        for j in range(k, min(n_frames, k + 4)):
            if j not in futs:
                futs[j] = pool.submit(read_frame, j)
        img = futs[k].result()
        futs.pop(k - 1, None)
        if img is None or img.shape[:2] != (H, W):
            continue
        crop = img[yy, xx]
        if k in skip:
            wmax = np.maximum(wmax, crop)             # a lighting event: left exactly as it is
            continue
        ch = crop.max(3).astype(np.int16) - crop.min(3).astype(np.int16)
        above = crop.max(3).astype(np.int16) - base.max(3).astype(np.int16)
        seed = ch >= GROUND_COLOUR_MIN
        # the speck = the strongly coloured pixels plus the bright / weakly coloured pixels
        # CONNECTED to them (a 15 px red-and-white streak has pale ends). Growth stops at
        # GROUND_REACH px: it can never spread across a lit scene.
        member = ((above >= GROUND_BRIGHT_ABOVE) | ((ch >= GROUND_WEAK_COLOUR) & (above >= REPAIR_ABOVE)))
        bad = seed & (above >= REPAIR_ABOVE)
        for _ in range(GROUND_REACH):
            grown = bad.copy()
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    grown |= np.roll(np.roll(bad, a, 1), b, 2)
            bad = (grown & member) | bad
        crop = crop.copy()
        crop[bad] = base[bad]
        wmax = np.maximum(wmax, crop)
    pool.shutdown(wait=False)
    np.minimum.at(out, (yy, xx), wmax)
    stats["pixels_changed"] = int((out != stack).any(2).sum())
    return out, stats


# ── what to tell the user and the usage report ────────────────────────────────
MARKER_LINE = "FLASHFINDER:"
MARKER_COUNTS = "FLASHFINDER_COUNTS:"


def _n(count, one, many):
    return "%d %s" % (count, one if count == 1 else many)


def plain_summary(sky, repair_stats, ground, ground_stats):
    """Plain-English lines for the Star Log, plus a small dict of counts for the usage
    report. `sky` is the FlashResult (or None when that pass never ran), `repair_stats`
    the dict repair_stack returned, `ground` the GroundResult, `ground_stats` the dict
    repair_ground returned. Counts only: no names, paths or images ever go in here."""
    lines = []
    counts = {"flash_spots": 0, "defect_spots": 0, "ground_spots": 0, "pixels_changed": 0,
              "lighting_event_frames": 0, "left_alone": 0, "sky_stopped": False, "ground_stopped": False}
    repair_stats = repair_stats or {}
    ground_stats = ground_stats or {}

    if sky is None or sky.bail:
        counts["sky_stopped"] = True
        lines.append("Flash finder, sky: nothing was changed because %s." % (sky.bail if sky else "it did not run"))
    else:
        fl, df = repair_stats.get("flash_spots", 0), repair_stats.get("defect_spots", 0)
        counts["flash_spots"], counts["defect_spots"] = fl, df
        counts["lighting_event_frames"] = sky.burst_frames
        counts["left_alone"] += repair_stats.get("hidden_not_repaired", 0)
        counts["pixels_changed"] += repair_stats.get("pixels_changed", 0)
        if fl or df:
            lines.append("Flash finder, sky: removed %s and %s." % (
                _n(fl, "one-frame flash", "one-frame flashes"),
                _n(df, "blinking sensor defect", "blinking sensor defects")))
        else:
            lines.append("Flash finder, sky: no single-frame flashes or blinking defects found.")
        if sky.burst_frames:
            lines.append("Flash finder, sky: skipped %s where the whole picture changed at once (lighting events)."
                         % _n(sky.burst_frames, "frame", "frames"))
        if repair_stats.get("hidden_not_repaired", 0):
            lines.append("Flash finder, sky: left alone %s that would have damaged a nearby trail or were not visible."
                         % _n(repair_stats["hidden_not_repaired"], "spot", "spots"))

    if ground is None or ground.bail:
        counts["ground_stopped"] = True
        lines.append("Flash finder, ground: left alone because %s." % (ground.bail if ground else "it did not run"))
    else:
        g = ground_stats.get("ground_spots", 0)
        counts["ground_spots"] = g
        counts["left_alone"] += ground_stats.get("on_a_trail", 0)
        counts["pixels_changed"] += ground_stats.get("pixels_changed", 0)
        text = "Flash finder, ground: removed %s" % _n(g, "speck", "specks")
        if ground_stats.get("on_a_trail", 0):
            text += " and left %d alone that sat on a trail" % ground_stats["on_a_trail"]
        lines.append(text + ".")
    return lines, counts


def marker_lines(lines, counts):
    """The lines the engine prints so the app can pick them up (same idea as HOTPIX_SKIPPED:)."""
    import json
    return [MARKER_LINE + " " + t for t in lines] + [MARKER_COUNTS + " " + json.dumps(counts, sort_keys=True)]


def parse_marker(line):
    """('line', text) or ('counts', dict) for a printed marker line; None for anything else."""
    import json
    s = line.strip()
    if s.startswith(MARKER_COUNTS):
        try:
            return "counts", json.loads(s[len(MARKER_COUNTS):].strip())
        except ValueError:
            return None
    if s.startswith(MARKER_LINE):
        return "line", s[len(MARKER_LINE):].strip()
    return None


def run_on_folder(cleaned_dir, names, stack, fg_mask, read_path, say=print, threads=4):
    """The whole flash finder as the star trail step uses it: find single-frame flashes
    and blinking defects in the sky, find coloured specks on the ground, repair the
    finished `stack`, print the plain-English summary behind marker lines, and return
    the new stack. `names` are the frame filenames in the order the trail was built;
    `fg_mask` is the foreground mask as saved (uint8, 255 = ground) or None;
    `read_path(path)` reads one frame as BGR uint8. Never raises: a pass that hits an
    error leaves the stack as it was and says so in the summary."""
    import os
    paths = [os.path.join(cleaned_dir, n) for n in names]
    n = len(paths)

    def rd(i):
        try:
            return read_path(paths[i])
        except Exception:
            return None

    fg = None
    if fg_mask is not None and fg_mask.shape[:2] == stack.shape[:2]:
        fg = fg_mask > 127
    out = stack
    sky, rstats, ground, gstats = None, None, None, None

    try:
        sky = find_flashes(n, rd, fg, progress=lambda k, t: say("  finding flashes: %d/%d" % (k, t)))
        if not sky.bail:
            defects = find_blinking_defects(sky.defect_candidates, n)
            out, rstats = repair_stack(out, n, rd, sky.flags, defects, threads)
    except Exception as e:
        sky = FlashResult()
        sky.bail = "of an unexpected error (%s)" % type(e).__name__
        out = stack

    try:
        ground = find_ground_specks(n, rd, fg)
        if not ground.bail:
            out, gstats = repair_ground(out, n, rd, ground, threads=threads)
    except Exception as e:
        ground = GroundResult()
        ground.bail = "of an unexpected error (%s)" % type(e).__name__

    lines, counts = plain_summary(sky, rstats, ground, gstats)
    for m in marker_lines(lines, counts):
        say(m)
    return out
