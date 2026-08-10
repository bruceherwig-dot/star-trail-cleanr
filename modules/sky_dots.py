"""Sky colored-speck / hot-pixel removal for the finished star trail.

Plain English: after a star trail is stacked, tiny colored dots are left behind --
stuck sensor pixels, cosmic-ray hits, and Bayer demosaic defects. This module
finds them by three signatures that no star shares, then removes them two ways:

  - OPEN-SKY dots (nothing bright around them) are filled from the surrounding
    sky directly on the finished stack.
  - ON-TRAIL dots (sitting on a bright star trail) are removed from each frame and
    the trail is re-stacked, so the trail itself survives.

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

def _isolated(g, cx, cy, r=6, thr=42):
    """True if the ring around (cx, cy) is dark (max < thr): an open-sky dot with
    no star trail beside it, safe to fill on the composite. A dot next to a bright
    trail fails this and is routed to per-frame removal instead."""
    yy, xx = np.ogrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
    ring = ((np.abs(yy - cy) > 2) | (np.abs(xx - cx) > 2))
    return g[cy - r:cy + r + 1, cx - r:cx + r + 1][ring].max() < thr


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


def _detect_map(big, sample_frames):
    """Build the full speck mask: the persistence detectors (chromatic + white,
    which need the sampled frames) UNION the point-and-color small-blob detector
    (which reads only the finished stack `big`).

    The small-blob pass runs at two brightness floors (g>32 and g>90): the low
    floor catches faint dots; the high one un-merges a bright dot fused to a faint
    trail. Each candidate must be a local POINT (peak > its r=4..6 ring + 5) AND a
    real dot (saturated color, green-deficient, or bright > 42) AND persistent
    across the sampled frames.

    THAT LAST CONDITION IS THE POINT (added 2026-08-03). Without it this pass
    judged the finished stack alone, where a star bead and a stuck pixel look
    identical: it decided "isolated dot?" by checking only 4-6 pixels out, so in
    any sequence with a gap between exposures -- where each star draws a dotted
    line rather than a solid one -- every bead of every trail qualified.
    Measured on Bruce's Perseid set before the fix: 38,878 pixels removed, of
    which 45 were real fixed defects (0.12%), while 54 of the 99 genuine defects
    present were missed entirely. It was erasing starlight by the thousand."""
    H, W = big.shape[:2]
    g = big.max(2).astype(np.int16)

    ch = build_hot_pixel_map_chromatic(
        sample_frames, center_threshold=22, inner_threshold=11, min_fraction=0.6)
    wh = build_hot_pixel_map_white(
        sample_frames, center_threshold=18, inner_threshold=9, min_fraction=0.6)
    hot = cv2.bitwise_or(ch, wh)

    cr = np.zeros((H, W), np.uint8)
    for thr in (32, 90):
        n, lab, st, ce = cv2.connectedComponentsWithStats((g > thr).astype(np.uint8), 8)
        for i in range(1, n):
            if (st[i, cv2.CC_STAT_AREA] <= 14
                    and max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) <= 5):
                cx, cy = int(ce[i][0]), int(ce[i][1])
                if not (8 < cx < W - 8 and 8 < cy < H - 8):
                    continue
                yy, xx = np.ogrid[cy - 6:cy + 7, cx - 6:cx + 7]
                rg = (np.maximum(np.abs(yy - cy), np.abs(xx - cx)) >= 4)
                is_point = int(g[cy, cx]) > int(g[cy - 6:cy + 7, cx - 6:cx + 7][rg].max()) + 5
                if not (is_point and (_saturated(big, cx, cy)
                                      or _green_deficient(big, cx, cy)
                                      or int(g[cy, cx]) > 42)):
                    continue
                # Looks like a dot in the stack -- but so does every bead of a
                # dotted star trail. Only erase it if the frames agree it is
                # stuck to the sensor. Needs most of the sample: a star can
                # graze the same pixel in a couple of frames by chance.
                if sample_frames and _persistence_count(sample_frames, cx, cy) < \
                        max(3, int(0.6 * len(sample_frames))):
                    continue
                cr[lab == i] = 255
    return cv2.bitwise_or(hot, cr)


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


def remove_specks(cleaned_dir, names, big, fg_mask, read_frame, comet_tail=0):
    """Remove sky hot pixels / colored specks and return a cleaned stack.

    Args:
        cleaned_dir: folder of cleaned frames.
        names: the frame filenames to stack, IN THE SAME ORDER the trail used
            (comet mode is order-dependent, so the caller passes the comet order).
        big: the finished stack (BGR uint8) -- the detection reference. Not
            re-stacked; used only to find the dots.
        fg_mask: grayscale foreground mask (HxW uint8) or None.
        read_frame: callable(path) -> BGR uint8 image (robust reader from caller).
        comet_tail: 0 for a plain lighten-max re-stack (default). When > 0 (a
            fraction of the sequence, e.g. 0.5/0.75/1.0), the re-stack fades like the
            comet stacker -- dim the running stack before folding each frame -- so a
            comet trail keeps its fade instead of being flattened to a plain trail.

    Returns:
        cleaned stack (BGR uint8): lighten-max, or comet-faded when comet_tail > 0.

    Raises:
        SkyDotsBail: the foreground guard tripped; caller falls back to plain.
    """
    import os
    H, W = big.shape[:2]
    g = big.max(2).astype(np.int16)
    paths = [os.path.join(cleaned_dir, n) for n in names]

    # Sample ~40 evenly spaced frames for the persistence detectors.
    print("  finding hot pixels and colored specks...", flush=True)
    idx = sorted(set(np.linspace(0, len(paths) - 1, 40).astype(int).tolist()))
    sample = []
    for j, i in enumerate(idx):
        f = read_frame(paths[i])
        if f is not None:
            sample.append(f)
        print(f"  finding specks: {j + 1}/{len(idx)}", flush=True)

    allmap = _detect_map(big, sample)
    allmap = _foreground_guard(allmap, fg_mask, H, W)

    # Route each surviving speck: dark ring -> composite fill; bright ring -> per-frame.
    nn, ll, ss, cc = cv2.connectedComponentsWithStats((allmap > 0).astype(np.uint8), 8)
    ontrail_pts = []
    for i in range(1, nn):
        cx, cy = int(cc[i][0]), int(cc[i][1])
        if not (8 < cx < W - 8 and 8 < cy < H - 8):
            continue
        if not _isolated(g, cx, cy, thr=42):
            ontrail_pts.append((cx, cy))
    n_iso = (nn - 1) - len(ontrail_pts)
    print(f"  removing {n_iso} open-sky + {len(ontrail_pts)} on-trail specks", flush=True)

    # Re-stack, removing on-trail specks from each frame first (3x3 patch <- ring median).
    # Comet mode dims the running stack before folding each frame (matching
    # _comet_stack_fullres) so the tail keeps fading; plain mode is order-independent
    # lighten-max. `names` already arrives in the trail's order (reversed for a reversed
    # comet), so folding them here reproduces the same trail with the dots gone.
    n = len(paths)
    comet = float(comet_tail) > 0
    if comet:
        tail_frames = max(1, int(round(float(comet_tail) * n)))
        fade = 0.04 ** (1.0 / tail_frames)   # ~4% bright after tail_frames, as in _comet_stack_fullres
    facc = None                              # float32 accumulator for comet
    acc = np.zeros((H, W, 3), np.uint8)      # uint8 accumulator for plain lighten-max
    off = [(dx, dy) for dy in (-3, -2, 2, 3) for dx in (-3, -2, 2, 3)]
    for k, p in enumerate(paths):
        f = read_frame(p)
        if f is None:
            continue
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        for cx, cy in ontrail_pts:
            vals = [f[cy + dy, cx + dx] for dx, dy in off
                    if 0 <= cy + dy < H and 0 <= cx + dx < W]
            f[cy - 1:cy + 2, cx - 1:cx + 2] = np.median(np.array(vals), axis=0)
        if comet:
            ff = f.astype(np.float32)
            if facc is None:
                facc = ff
            else:
                facc *= fade
                np.maximum(facc, ff, out=facc)
        else:
            np.maximum(acc, f, out=acc)
        if k % 25 == 0:
            print(f"  cleaning specks: {k}/{n}", flush=True)
    print(f"  cleaning specks: {n}/{n}", flush=True)
    if comet and facc is not None:
        acc = np.clip(facc, 0, 255).astype(np.uint8)

    # Composite ring-median fill for the open-sky specks on the finished stack.
    isomask = np.zeros((H, W), np.uint8)
    for i in range(1, nn):
        cx, cy = int(cc[i][0]), int(cc[i][1])
        if 8 < cx < W - 8 and 8 < cy < H - 8 and _isolated(g, cx, cy, thr=42):
            isomask[ll == i] = 255
    isomask = cv2.dilate(isomask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    out = acc.copy()
    mn, mll, mss, _ = cv2.connectedComponentsWithStats((isomask > 0).astype(np.uint8), 8)
    for i in range(1, mn):
        x, y, bw, bh, _ = mss[i]
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
        subm = (mll[y0:y1, x0:x1] == i)
        ring = acc[y0:y1, x0:x1][~subm & (isomask[y0:y1, x0:x1] == 0)]
        if len(ring) > 0:
            out[y0:y1, x0:x1][subm] = np.median(ring.reshape(-1, 3), axis=0)

    # Self-check on the raw arrays: removal can only darken, never brighten.
    brighter = int((out.max(2).astype(int) > big.max(2).astype(int) + 3).sum())
    print(f"  speck removal done (pixels brightened anywhere, should be ~0: {brighter})", flush=True)
    return out
