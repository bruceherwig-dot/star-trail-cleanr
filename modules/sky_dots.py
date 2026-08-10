"""Sky colored-speck / hot-pixel removal for the finished star trail.

Plain English: after a star trail is stacked, tiny colored dots are left behind --
stuck sensor pixels, cosmic-ray hits, and Bayer demosaic defects. This module
finds them by three signatures that no star shares, then paints over each one on
the finished stack, taking the replacement from along the trail's own direction.

WHY THE FILL RUNS ON THE STACK AND NOT PER FRAME (2026-08-09). Until now a dot
sitting on or beside a trail took a second route: erase a 3x3 patch out of EVERY
frame and re-stack, on the theory that the trail would survive. It cannot. The
pixel is erased in every frame, so no frame is left to contribute the trail
there -- and the re-stacked pixel then holds ONE frame's sky value while every
pixel around it holds the brightest of hundreds of frames. The maximum of 300
noise samples is far above any single sample, so the spot came out as a hole
DARKER than the sky around it. Measured on Bruce's Perseid set: of 683 removals
with a trail running through, 171 (25%) ended more than 8 levels below their own
surroundings -- the black dots he circled. The open-sky route, which filled from
the finished stack, left 0 of 269 dark. So the per-frame route is gone: it never
did its job, it produced the only visible damage, and it was the sole reason this
option had to re-read every frame (now it reads ~40, for detection only).

How a dot is told apart from a star:
  - PERSISTENCE: a sensor defect sits at the exact same pixel in every frame; a
    real star drifts with Earth rotation, so it never repeats at one pixel.
  - POINT-not-LINE: a defect is a single peak brighter than its whole surrounding
    ring; a trail pixel sits on a ridge with bright neighbours along the line.
  - COLOR no star can have: pure saturated color even when dim, or green below
    both red and blue (magenta/purple), which no blackbody star can be.

Proven over six shots / five cameras (2026-07-13/14). Opt-in only -- it re-reads
every frame, so it is the slow path. A foreground guard (below) refuses to run on
an unmasked landscape so it can never eat the ground.
"""
import cv2
import numpy as np

from tools.hot_pixels_v2 import (
    build_hot_pixel_map_chromatic,
    build_hot_pixel_map_white,
)


class SkyDotsBail(Exception):
    """Raised when the foreground guard trips: the specks clumped into a solid
    shape (an unmasked landscape) rather than scattering across the sky. Removing
    them would erase part of the foreground, so the caller must fall back to the
    plain trail and tell the user."""


# ── the three per-dot tests (all read the finished plain stack `big`) ────────
#
# (_isolated() lived here: a dark-ring test that routed each dot to the composite
# fill or to the per-frame re-stack. Both dots are filled the same way now, so
# there is nothing left to route. See the header for why the per-frame route
# went away.)

def _saturated(big, cx, cy):
    """Relative color purity (max-min)/max over the 3x3 center. A pure red scores
    high whether bright or dim; neutral white scores ~0. High purity = not a star
    color = defect. Works for DIM colored dots, unlike an absolute threshold."""
    p = big[cy - 1:cy + 2, cx - 1:cx + 2].reshape(-1, 3).astype(float)
    mx = p.max(1)
    mn = p.min(1)
    return bool(((mx > 12) & ((mx - mn) / np.maximum(mx, 1) > 0.5)).any())


def _green_deficient(big, cx, cy):
    """True if green is below BOTH red and blue (magenta/purple). Green is always a
    star's middle channel, so green-below-both is impossible for a real star and
    proves a Bayer defect -- no isolation needed. BGR order: 0=B, 1=G, 2=R."""
    p = big[cy - 1:cy + 2, cx - 1:cx + 2].reshape(-1, 3).astype(int)
    B, G, R = p[:, 0], p[:, 1], p[:, 2]
    return bool((((np.minimum(B, R) - G) > 4) & (np.maximum(B, R) > 14)).any())


def _persistence_count(sample_frames, cx, cy, floor=40):
    """How many sampled frames have something bright at this exact pixel.

    THE test that separates a sensor defect from starlight. A stuck pixel is a
    property of the sensor: it lands on the same pixel in every single frame. A
    star is somewhere else in every frame -- its light passes through any given
    pixel once. Nothing about a dot's size, shape, colour or brightness in the
    finished stack can tell the two apart; only this can.
    """
    hits = 0
    for f in sample_frames:
        if f is None or cy >= f.shape[0] or cx >= f.shape[1]:
            continue
        if int(f[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].max()) > floor:
            hits += 1
    return hits


def _blob_candidates(big):
    """Every small point-like blob with defect-ish colour: the population that
    gets judged, and nothing more. Shape and colour NARROW the field; they decide
    nothing. A small sharp oddly-coloured point is, on that evidence alone, just
    as likely to be a star -- which is exactly the mistake this module has made
    twice. Only the frames can convict.

    Two brightness floors: the low one (32) finds faint dots, the high one (90)
    un-merges a bright dot fused to a faint trail.

    Returns (label image, centroids). Candidate k owns the pixels labelled k+1."""
    H, W = big.shape[:2]
    g = big.max(2).astype(np.int16)
    cand_lab = np.zeros((H, W), np.int32)
    pts = []
    for thr in (32, 90):
        n, lab, st, ce = cv2.connectedComponentsWithStats((g > thr).astype(np.uint8), 8)
        for i in range(1, n):
            if not (st[i, cv2.CC_STAT_AREA] <= 14
                    and max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) <= 5):
                continue
            cx, cy = int(ce[i][0]), int(ce[i][1])
            if not (8 < cx < W - 8 and 8 < cy < H - 8):
                continue
            # Clamped to the frame: the border guard above only reserves 8px, so
            # near an edge this window is cut short rather than running off (a
            # negative slice index would silently wrap).
            y0, y1 = max(0, cy - 6), min(H, cy + 7)
            x0, x1 = max(0, cx - 6), min(W, cx + 7)
            win = g[y0:y1, x0:x1]
            yy, xx = np.ogrid[y0:y1, x0:x1]
            d = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
            peak = int(g[cy, cx])
            r_near = (d >= 4) & (d <= 6)
            is_point = (not r_near.any()) or peak > int(win[r_near].max()) + 5
            if not (is_point and (_saturated(big, cx, cy)
                                  or _green_deficient(big, cx, cy) or peak > 42)):
                continue
            pts.append((cx, cy))
            x, y = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
            bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            sub = lab[y:y + bh, x:x + bw] == i
            cand_lab[y:y + bh, x:x + bw][sub] = len(pts)
    return cand_lab, pts


def _paint(cand_lab, ids, n_pts):
    """Turn a list of condemned candidate numbers into a mask, via a lookup table
    rather than one full-frame comparison per candidate."""
    lut = np.zeros(n_pts + 1, np.uint8)
    if ids:
        lut[np.array(ids, np.int32)] = 255
    return lut[cand_lab]


def _flecks(pts, peak, evidence):
    """Which candidates are one-off events -- cosmic rays, satellite glints, a
    single frame's noise -- rather than stars.

    The sky turns, so the light of a real star is at a KNOWN place one frame
    earlier and another known place one frame later. Follow it there: a star is
    nearly as bright at those two spots, a one-off event leaves empty sky at both.
    This is Bruce's own rule ("if it was a star, it would be in a trail") with the
    trail predicted from the sky's actual rotation, rather than assumed from a
    circle drawn around the dot on the finished stack -- where neighbouring trails
    make the question unanswerable.

    A dot too dim to read (peak below _FLECK_MIN_PEAK) is LEFT ALONE. It cannot be
    judged and the cost of guessing wrong is a mark on a real star.

    Measured on Bruce's Perseid sequence against 800 control points on real trails
    plus 12 leftover spots each known to appear in exactly one frame: this erases
    all 12, and 0 of 400 bright-trail and 3 of 400 faint-trail controls. The rule
    it replaced -- "was there any starlight within 10px in consecutive frames" --
    scored 0 of 800 controls but caught 0 of the 12, because in a field this
    crowded some trail is nearly always passing within 10px."""
    return [k + 1 for k in range(len(pts))
            if peak[k] > _FLECK_MIN_PEAK and evidence[k] < _FLECK_RATIO * peak[k]]


def _detect_map(big, sample_frames, run_map=None, cand=None, fleck_ids=None):
    """Build the full speck mask: the run's own stuck-pixel map, UNION the
    persistence detectors (chromatic + white, which need the sampled frames),
    UNION the point-and-color small-blob detector (which reads the finished stack
    `big` but must still be backed by persistence).

    ONLY WHAT CAN BE PROVEN IS ERASED (2026-08-09). Every fill leaves a mark: a
    star trail here is BEADED, and no fill can invent the bead pattern, so a
    patched spot loses about 44% of its texture and the eye finds it. Measured on
    Bruce's Perseid trail, the brightness of the patches was right and he could
    still see every one. That changes the economics completely -- a removal is
    only worth making when we can show the thing was a defect, because the cost of
    being wrong is a visible smudge on a real star.

    So the discriminator is persistence, and nothing else. A stuck pixel is a
    property of the sensor: same pixel, every frame. A star passes through any
    given pixel once and is gone.

    `run_map` is the strongest evidence available and it comes free: during the
    clean, every 20-frame batch builds its own stuck-pixel map and ORs it into
    hot_pixel_map.png, so a defect that only wakes up late in the night is still
    caught in its own batch. Its bar is strict -- bright in 80% of a batch's
    frames AND one channel more than 60 levels above the other two.

    WHAT WAS REMOVED HERE, AND WHY IT MUST NOT COME BACK: this pass used to also
    erase a dot for being ALONE -- brighter than everything within 4-6px, later
    14px. Both were measured against ground truth on Bruce's own frames and both
    failed badly. At 6px, every bead of every dotted trail read as a lone dot:
    38,878 pixels removed, 45 of them real defects (0.12%). At 14px it erased
    1,693 dots of which 22 were stuck pixels -- and the rule cannot work in
    principle, because his stars move only 0.2 to 5.8 px between frames, so a
    bead's neighbours are inside the very radius the test looks past. Colour,
    collinearity and structure-tensor direction were measured too and separate
    nothing (14% vs 13%, 85% vs 77%, 78% vs 75%). Do not re-add an isolation
    test; the leftover one-off flecks are a separate problem and need evidence
    the finished stack does not contain."""
    H, W = big.shape[:2]

    ch = build_hot_pixel_map_chromatic(
        sample_frames, center_threshold=22, inner_threshold=11, min_fraction=0.6)
    wh = build_hot_pixel_map_white(
        sample_frames, center_threshold=18, inner_threshold=9, min_fraction=0.6)
    hot = cv2.bitwise_or(ch, wh)

    cand_lab, pts = _blob_candidates(big) if cand is None else cand
    # THE decision, and the only one this pass can make on its own: do the frames
    # show this thing at the same pixel over and over? That is what a sensor
    # defect is. Nothing about a dot's size, shape, colour or brightness in the
    # finished stack can stand in for it -- every substitute has been measured
    # against ground truth and failed (see above).
    stuck = [k + 1 for k, (cx, cy) in enumerate(pts)
             if sample_frames and _persistence_count(sample_frames, cx, cy)
             >= max(3, int(0.6 * len(sample_frames)))]
    cr = _paint(cand_lab, stuck, len(pts))

    out = cv2.bitwise_or(hot, cr)
    if fleck_ids:
        out = cv2.bitwise_or(out, _paint(cand_lab, fleck_ids, len(pts)))
    if run_map is not None and run_map.shape[:2] == (H, W):
        # The run's own map, gathered 20 frames at a time while cleaning. Union,
        # not intersection: a defect it caught in one batch is a defect, whether
        # or not a 40-frame sample of the whole night happens to agree.
        out = cv2.bitwise_or(out, (run_map > 0).astype(np.uint8) * 255)
    return out


_BATCH = 20            # the clean judges 20 frames at a time; match it exactly
_FLECK_MIN_PEAK = 40   # too dim to judge: leave it alone
_FLECK_RATIO = 0.4     # the sky-predicted spot must beat this share of the dot's peak


def _fit_sky_motion(a, b, hw):
    """How far the sky moves between two consecutive frames, as one affine map.

    Phase correlation on a grid of patches, then a least-squares fit. The stars
    turn about the pole, which is an affine motion of the frame, so a single fit
    predicts the shift at any pixel -- 0.03 px near the pole, 6 px at the edge on
    Bruce's Perseid set. Returns a 3x2 matrix, or None when too few patches agree
    (cloud, a bare sky with nothing to lock onto, an unusable pair)."""
    H, W = hw
    S = 256
    src, dst = [], []
    for gy in range(4):
        for gx in range(6):
            cy, cx = int((gy + 0.5) * H / 4), int((gx + 0.5) * W / 6)
            if cy - S < 0 or cy + S > H or cx - S < 0 or cx + S > W:
                continue
            pa = a[cy - S:cy + S, cx - S:cx + S].astype(np.float32)
            pb = b[cy - S:cy + S, cx - S:cx + S].astype(np.float32)
            (dx, dy), resp = cv2.phaseCorrelate(pa, pb)
            if resp < 0.15 or np.hypot(dx, dy) > 40:
                continue
            src.append([cx, cy, 1.0])
            dst.append([cx + dx, cy + dy])
    if len(src) < 6:
        return None
    M, *_ = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)
    return M


def _sky_motion(paths, read_frame, hw):
    """Measure the sky's motion from three pairs spread across the night and only
    believe it if all three agree.

    They must, for a tripod on the ground under a turning sky. If they don't, the
    sequence has something else going on -- a bump, drifting cloud, a mount -- and
    the whole fleck test is then pointed at the wrong pixels. Returns None in that
    case, and the caller removes stuck pixels only. Removing nothing beats
    removing the wrong thing."""
    H, W = hw
    n = len(paths)
    fits = []
    for frac in (0.1, 0.5, 0.9):
        i = min(max(1, int(frac * n)), n - 2)
        a, b = read_frame(paths[i]), read_frame(paths[i + 1])
        if a is None or b is None:
            continue
        M = _fit_sky_motion(a.max(2), b.max(2), (H, W))
        if M is not None:
            fits.append(M)
    if len(fits) < 2:
        print("  sky motion could not be measured; skipping fleck removal",
              flush=True)
        return None
    # Compare what each fit predicts at the frame's corners and centre.
    probe = np.array([[W * 0.2, H * 0.2, 1.0], [W * 0.8, H * 0.2, 1.0],
                      [W * 0.5, H * 0.5, 1.0], [W * 0.2, H * 0.8, 1.0],
                      [W * 0.8, H * 0.8, 1.0]])
    preds = [probe @ M for M in fits]
    spread = max(float(np.abs(p - q).max()) for p in preds for q in preds)
    if spread > 1.5:
        print(f"  sky motion disagrees across the night by {spread:.1f}px; "
              f"skipping fleck removal", flush=True)
        return None
    M = sum(fits) / len(fits)
    step = np.hypot(*(probe @ M - probe[:, :2]).T)
    print(f"  sky motion measured: {step.min():.2f} to {step.max():.2f} px "
          f"between frames", flush=True)
    return M


def _scan_frames(paths, read_frame, hw, want_sample=40, pts=None, build_map=True,
                 motion=None):
    """Read every frame once and judge it in batches of 20, the way the clean
    does, rebuilding the stuck-pixel map for a sequence whose saved one is gone.

    Batching is the point, and it is Bruce's (2026-08-09): a defect that only
    wakes up partway through the night is present in nearly every frame of ITS
    OWN batch, and so clears the 80%-of-frames bar there, while across the whole
    night it looks occasional and is missed. Judging 300 frames as one pile is
    what let real specks survive; judging fifteen piles of twenty catches them.

    A spread of frames is kept on the way past for the other detectors, so the
    sequence is read once, not twice. Those frames are the same objects held in
    the current batch, so keeping them costs nothing extra.

    The same pass answers the fleck question for every candidate in `pts`, given
    the sky's motion: how bright the dot gets in its best frame, and how bright
    the sky-predicted spots are in the frames either side of that one. A star is
    nearly as bright where the sky carried it; a one-off event leaves empty sky
    at both.

    `build_map` False skips the per-batch stuck-pixel map (the expensive part)
    when the run already saved one -- the pass is then only about the flecks.

    Returns (stuck-pixel map or None, retained sample frames, peak, evidence)."""
    from modules.hot_pixels import build_hot_pixel_map

    H, W = hw
    n = len(paths)
    keep = set(np.linspace(0, n - 1, min(want_sample, n)).astype(int).tolist())
    acc = np.zeros((H, W), np.uint8) if build_map else None
    sample, batch = [], []

    pts = pts or []
    track = bool(pts) and motion is not None
    peak = np.zeros(len(pts), np.int16)
    ev_before = np.zeros(len(pts), np.int16)
    ev_after = np.zeros(len(pts), np.int16)
    if track:
        xs = np.array([p[0] for p in pts], np.float64)
        ys = np.array([p[1] for p in pts], np.float64)
        q = np.stack([xs, ys, np.ones_like(xs)], 1) @ motion
        dx, dy = q[:, 0] - xs, q[:, 1] - ys

        def _cl(v, hi):
            return np.clip(np.round(v), 3, hi - 4).astype(np.int32)

        xi, yi = _cl(xs, W), _cl(ys, H)                  # the dot itself
        px, py = _cl(xs - dx, W), _cl(ys - dy, H)        # where it was one frame back
        nx, ny = _cl(xs + dx, W), _cl(ys + dy, H)        # where it lands one frame on
        # The dot is read exactly; the predicted spots get 2px of slack for the
        # motion fit. Slack on the DOT is what let neighbouring trails explain
        # away real flecks -- measured, it halved the catch.
        k_dot = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_pred = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        prev_pred = np.zeros(len(pts), np.int16)
        just_peaked = np.zeros(len(pts), bool)

    for k, p in enumerate(paths):
        f = read_frame(p)
        if f is not None:
            if f.shape[:2] != (H, W):
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            if track:
                gm = f.max(2)
                d_dot = cv2.dilate(gm, k_dot)
                d_pred = cv2.dilate(gm, k_pred)
                v_dot = d_dot[yi, xi].astype(np.int16)
                v_prev = d_pred[py, px].astype(np.int16)
                v_next = d_pred[ny, nx].astype(np.int16)
                # A dot that peaked on the LAST frame: did the light arrive where
                # the sky would have carried it?
                ev_after[just_peaked] = np.maximum(ev_after[just_peaked],
                                                   v_next[just_peaked])
                brighter = v_dot > peak
                ev_before[brighter] = prev_pred[brighter]   # ... or come from there
                peak[brighter] = v_dot[brighter]
                just_peaked, prev_pred = brighter, v_prev
            if build_map:
                batch.append(f)
                if k in keep:
                    sample.append(f)
        if build_map and len(batch) >= _BATCH:
            acc = cv2.bitwise_or(acc, build_hot_pixel_map(batch))
            batch = []
        print(f"  finding specks: {k + 1}/{n}", flush=True)
    # A short tail batch is judged too, but only if there is enough of it to mean
    # anything -- "bright in 80% of 3 frames" is not evidence of anything.
    if build_map and len(batch) >= 5:
        acc = cv2.bitwise_or(acc, build_hot_pixel_map(batch))
    return acc, sample, peak, np.maximum(ev_before, ev_after)


def _foreground_guard(allmap, fg_mask, H, W):
    """Keep the run from eating an unmasked landscape.

    With a foreground mask: drop every speck that falls on the foreground and
    proceed with the sky ones (a painted mask is the trusted answer). Prints a
    flood check so the log shows the sky/foreground split.

    Without a mask: allowed only when the specks are SCATTERED across the sky (an
    all-sky frame, like a full-dome shot). If instead they clump into one solid
    region -- the signature of an unmasked foreground being flooded -- bail, so the
    caller falls back to the plain trail rather than erasing the ground.

    Returns the speck mask to actually remove. Raises SkyDotsBail on a clump."""
    if fg_mask is not None and fg_mask.shape[:2] == (H, W):
        nn, ll, ss, cc = cv2.connectedComponentsWithStats((allmap > 0).astype(np.uint8), 8)
        sky = fgd = 0
        for i in range(1, nn):
            cx, cy = int(cc[i][0]), int(cc[i][1])
            if 8 < cx < W - 8 and 8 < cy < H - 8:
                if fg_mask[cy, cx] > 127:
                    fgd += 1
                else:
                    sky += 1
        print(f"  speck flood check: {sky} in sky, {fgd} on foreground (masked out)", flush=True)
        return cv2.bitwise_and(allmap, (fg_mask <= 127).astype(np.uint8) * 255)

    # No mask -- only safe if the specks are scattered, not clumped.
    dil = cv2.dilate(allmap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n2, l2, s2, _ = cv2.connectedComponentsWithStats((dil > 0).astype(np.uint8), 8)
    largest = max((int(s2[i, cv2.CC_STAT_AREA]) for i in range(1, n2)), default=0)
    frac = largest / float(H * W)
    if frac > 0.01:
        raise SkyDotsBail(
            "The specks clumped into a solid shape, which usually means an "
            "unmasked foreground. Paint a rough foreground mask in the app and "
            "try again -- running now would erase part of your landscape.")
    return allmap


def _ring_colour(big, mask, cx, cy, lo=4, hi=9):
    """The colour of the sky immediately around a speck: the per-channel median of
    an annulus, skipping anything that is itself being removed. This is the FLOOR
    every fill is held to, which is what makes a hole impossible -- a filled pixel
    can never come out darker than the picture around it."""
    H, W = big.shape[:2]
    y0, y1 = max(0, cy - hi), min(H, cy + hi + 1)
    x0, x1 = max(0, cx - hi), min(W, cx + hi + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    sel = (d >= lo) & (mask[y0:y1, x0:x1] == 0)
    if not sel.any():
        return None
    return np.median(big[y0:y1, x0:x1][sel].reshape(-1, 3), axis=0)


def _fill_specks(big, mask):
    """Paint over every speck on the finished stack and return the result.

    Inpainting does the work: it grows the picture inward from the edge of each
    speck, following whatever structure runs into it, so a trail crossing the spot
    continues through and open sky stays sky.

    WHY NOT A SOLID COLOUR (2026-08-09). The first version of this measured the
    trail running through and painted the blob that one colour. It fixed the black
    dots and replaced them with pale ones: a flat slab of trail colour, several
    pixels across, against a background that is beaded and varying, reads as a
    smudge -- and picking a single colour means picking the trail's peak, so the
    slab came out brighter than the trail it was patching. Bruce spotted them on
    sight. A speck is a few pixels wide; the fill has to vary the way its
    surroundings vary, which is what inpainting is for and a constant never is.

    A safety pass follows: any spot left sitting below the picture around it is
    lifted to match. That guarantee is the whole reason this function exists --
    a hole is the one failure a viewer's eye goes straight to.

    Returns (filled image, speck count, number of spots the safety pass lifted).
    """
    # Navier-Stokes, not Telea: measured on a trail severed by a 5px patch it
    # carries 94 of the trail's 150 through the gap where Telea manages 79, and
    # it never bottoms out at sky level inside the patch the way Telea does.
    m8 = (mask > 0).astype(np.uint8)
    out = cv2.inpaint(big, m8, 3, cv2.INPAINT_NS)

    n, lab, st, ce = cv2.connectedComponentsWithStats(m8, 8)
    lifted = 0
    for i in range(1, n):
        cx, cy = int(ce[i][0]), int(ce[i][1])
        sky = _ring_colour(out, mask, cx, cy)
        if sky is None:
            continue
        x, y, bw, bh = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                        int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]))
        # Work inside this blob's own bounding box. Testing `lab == i` across the
        # whole frame instead would be a 30-megapixel comparison per speck, and
        # there are thousands of specks -- minutes of waiting for nothing.
        sub = lab[y:y + bh, x:x + bw] == i
        patch = out[y:y + bh, x:x + bw][sub]
        # EVERY filled pixel is held to the surrounding sky, not just the patch's
        # brightest one. Judging a patch by its brightest pixel let its middle sit
        # below the sky and still pass: 4 of 833 patches on Bruce's Perseid trail
        # came out as small dark spots that way. Per pixel, the promise holds by
        # construction rather than nearly holding.
        raised = np.maximum(patch, np.clip(sky, 0, 255).astype(np.uint8))
        if np.array_equal(raised, patch):
            continue
        out[y:y + bh, x:x + bw][sub] = raised
        lifted += 1
    return out, (n - 1), lifted


def remove_specks(cleaned_dir, names, big, fg_mask, read_frame, comet_tail=0,
                  run_map=None):
    """Remove sky hot pixels / colored specks and return a cleaned stack.

    Args:
        cleaned_dir: folder of cleaned frames.
        names: the frame filenames to stack, IN THE SAME ORDER the trail used
            (comet mode is order-dependent, so the caller passes the comet order).
        big: the finished stack (BGR uint8) -- both the detection reference and
            the image that gets painted. Whatever the caller built (plain or comet)
            is what comes back, dots gone; nothing is re-stacked.
        fg_mask: grayscale foreground mask (HxW uint8) or None.
        read_frame: callable(path) -> BGR uint8 image (robust reader from caller).
        comet_tail: accepted and ignored. It used to drive a comet-faded re-stack;
            since the fill happens on the finished stack, the caller's comet fade
            is preserved automatically and there is nothing to reproduce. Kept in
            the signature so the existing callsite keeps working.
        run_map: the run's accumulated stuck-pixel map (HxW uint8) or None. Built
            during the clean, one 20-frame batch at a time, so it sees defects a
            40-frame sample of the whole night can miss. None when the run did not
            write one -- it is only built when a foreground mask exists.

    Returns:
        cleaned stack (BGR uint8) -- the caller's own stack with the specks painted
        out.

    Raises:
        SkyDotsBail: the foreground guard tripped; caller falls back to plain.
    """
    import os
    H, W = big.shape[:2]
    paths = [os.path.join(cleaned_dir, n) for n in names]

    print("  finding hot pixels and colored specks...", flush=True)
    # How many frames may be held at once. Each is H x W x 3 bytes, and a 30
    # megapixel sequence is 90MB a frame, so on a small machine 40 of them plus
    # the batch is the difference between working and swapping. Keep a 4x margin;
    # without psutil, behave as before.
    want_sample = 40
    try:
        import psutil
        want_sample = max(8, min(40, int(psutil.virtual_memory().available
                                         // (H * W * 3 * 4))))
    except Exception:
        pass

    # Every dot that will be judged, found once and reused by both tests.
    cand = _blob_candidates(big)
    pts = cand[1]
    motion = _sky_motion(paths, read_frame, (H, W))

    if run_map is not None:
        print(f"  using the run's own hot-pixel map "
              f"({int((run_map > 0).sum())} px marked while cleaning), and reading "
              f"the frames to sort flecks from stars", flush=True)
        _, sample, peak, evidence = _scan_frames(
            paths, read_frame, (H, W), want_sample, pts=pts, build_map=False,
            motion=motion)
    else:
        # No saved map -- the run that cleaned these frames predates keeping it,
        # or had no foreground mask to build one. Rebuild it the same way the
        # clean does, 20 frames at a time. Costs a full pass over the sequence,
        # which is why this is the opt-in checkbox. The fleck evidence rides along
        # on the same pass.
        print("  no saved hot-pixel map for this folder; rebuilding it from the "
              "frames, 20 at a time", flush=True)
        run_map, sample, peak, evidence = _scan_frames(
            paths, read_frame, (H, W), want_sample, pts=pts, motion=motion)
        print(f"  rebuilt hot-pixel map: {int((run_map > 0).sum())} px marked",
              flush=True)

    fleck_ids = _flecks(pts, peak, evidence)
    print(f"  {len(pts)} dots judged: {len(fleck_ids)} are one-off flecks "
          f"(the sky's rotation does not account for them)", flush=True)
    allmap = _detect_map(big, sample, run_map=run_map, cand=cand,
                         fleck_ids=fleck_ids)
    allmap = _foreground_guard(allmap, fg_mask, H, W)

    # Grow each speck by a pixel so its demosaic fringe goes with it, then paint
    # them all out on the finished stack. One route now, no re-reading. The growth
    # is deliberately small: every pixel added is a pixel of real picture thrown
    # away, and an over-grown patch is visible even when the fill is perfect.
    allmap = cv2.dilate(allmap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out, n_specks, n_lifted = _fill_specks(big, allmap)
    print(f"  removing {n_specks} specks ({n_lifted} needed lifting to sky level)",
          flush=True)

    # Self-check on the FINISHED image, not on the intent: re-measure every spot
    # that was painted and count the ones sitting below the picture around them.
    # A hole is the one failure the user actually sees, and the fill is built so
    # this can only ever print 0. It reads the output, so a future edit that drops
    # the sky floor shows up here instead of in Bruce's Photoshop.
    nn, ll, ss, cc = cv2.connectedComponentsWithStats((allmap > 0).astype(np.uint8), 8)
    holes = 0
    for i in range(1, nn):
        cx, cy = int(cc[i][0]), int(cc[i][1])
        sky = _ring_colour(out, allmap, cx, cy)
        if sky is not None and int(out[cy, cx].max()) < int(sky.max()) - 8:
            holes += 1
    print(f"  speck removal done (spots left darker than their surroundings, "
          f"must be 0: {holes})", flush=True)
    return out
