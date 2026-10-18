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


def chan_max(img):
    """Per-pixel maximum over the colour channels (6 ms; numpy's .max(2) takes 158 ms
    on a 24 MP frame, with the identical result)."""
    b, g, r = cv2.split(img)
    return cv2.max(cv2.max(b, g), r)


# ── per-flag measurements ─────────────────────────────────────────────────────
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
    __slots__ = ("frame", "x", "y", "contrast", "half", "colour")

    def __init__(self, frame, x, y, contrast, half, colour):
        self.frame, self.x, self.y = frame, x, y
        self.contrast, self.half, self.colour = contrast, half, colour

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
        self.star_reference = None


def find_flashes(n_frames, read_frame, fg_mask=None, progress=None, threads=4):
    """Run the sky pass over frames 0..n_frames-1.

    read_frame(i) -> BGR uint8 image or None. fg_mask: HxW bool (True = foreground,
    skipped here) or None. Every frame is read exactly ONCE and kept in a rolling cache
    (the previous version read each frame seven times and was five times slower).
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
        for (kk, x, y, dd, w, h, mot, asp) in lst:
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
            flag = Flag(k, px, py, dd, max(4, max(w, h) // 2 + 3), colour)
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
            lst.append((k, x, y, int(d[lab == i].max()), int(bw), int(bh), mot, round(asp, 2)))
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

