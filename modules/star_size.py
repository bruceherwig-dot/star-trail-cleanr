"""How long and how fat are the stars in THIS set?

The detector's size filter used to throw away any detection under a fixed pixel area, which also threw
away real short trails (airplane dashes, 20-60 px long) the model had correctly found. The rule it should
follow instead: a detection is a trail if it is RED (aircraft light), LONGER than the longest star, or FATTER
than a star. Only something that is none of those is star-sized and can be dropped.

This module measures "the longest star" and "the fattest star" from the frames themselves. A star is a small
bright blob that shows up AGAIN in the next frame at about the same place (the sky only turns a few pixels
between frames); a plane or satellite dash has moved far by then, so it does not pair up. The measurement uses
the 95th percentile of the paired blobs, not the maximum: the top 1% is polluted by clumps of touching stars.

Lengths and widths are the ellipse-equivalent axis lengths (4 x the square root of the covariance
eigenvalues), the same definition skimage's regionprops uses in the detector's gate, so they compare directly.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree

_PARTNER_PX = 15       # a star re-appears within this many px in the next frame
_MIN_STARS = 500       # fewer paired stars than this -> no trustworthy measurement, caller falls back
_PCTL = 95
_MIN_BLOB_AREA = 3
_TOPHAT_SIZE = 31


def blob_axes(mask_bin, fg_mask=None, min_area=_MIN_BLOB_AREA):
    """Bright blobs in a binary image: returns (centroids Nx2 [x, y], long axis N, short axis N).
    Done with whole-image sums (no per-blob Python loop) so a 24 MP frame takes about a second."""
    m = (mask_bin > 0).astype(np.uint8)
    if fg_mask is not None:
        m[fg_mask > 127] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    cnt = np.bincount(ids, minlength=n).astype(np.float64)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    sx = np.bincount(ids, weights=x, minlength=n)
    sy = np.bincount(ids, weights=y, minlength=n)
    sxx = np.bincount(ids, weights=x * x, minlength=n)
    syy = np.bincount(ids, weights=y * y, minlength=n)
    sxy = np.bincount(ids, weights=x * y, minlength=n)
    c = np.maximum(cnt, 1.0)
    mx, my = sx / c, sy / c
    vxx = np.maximum(sxx / c - mx * mx, 0.0)
    vyy = np.maximum(syy / c - my * my, 0.0)
    vxy = sxy / c - mx * my
    tr = vxx + vyy
    det_term = np.sqrt(np.maximum((vxx - vyy) ** 2 / 4.0 + vxy * vxy, 0.0))
    l1 = tr / 2.0 + det_term
    l2 = np.maximum(tr / 2.0 - det_term, 0.0)
    keep = (np.arange(n) > 0) & (cnt >= min_area)
    return cents[keep], 4.0 * np.sqrt(l1[keep]), 4.0 * np.sqrt(l2[keep])


def _star_blobs(gray, fg_mask):
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (_TOPHAT_SIZE, _TOPHAT_SIZE)))
    sig = 1.4826 * float(np.median(np.abs(th.astype(np.float32) - float(np.median(th)))))
    return blob_axes(th > max(8.0, 6.0 * sig), fg_mask)


def _gray(img):
    if img.ndim == 2:
        g = img
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv2.convertScaleAbs(g, alpha=255.0 / max(float(g.max()), 1.0))
    return g


def measure_star_size(frames, fg_mask=None, max_pairs=2):
    """Star size for one set. frames: list of images (BGR or gray, in shooting order; None entries skipped).
    fg_mask: 255 = foreground (ignored). Returns {"length": px, "width": px, "stars": count} or None when
    there are too few paired stars to trust (the caller then keeps its old fixed-size rule). "shift" is the median
    distance a paired star moved between the two frames, in px."""
    usable = [i for i, f in enumerate(frames) if f is not None]
    pairs = [(a, b) for a, b in zip(usable, usable[1:]) if b == a + 1]
    if not pairs:
        return None
    picks = sorted({0, len(pairs) // 2})[:max_pairs]
    lens, wids, shifts = [], [], []
    for pi in picks:
        a, b = pairs[pi]
        ga, gb = _gray(frames[a]), _gray(frames[b])
        fg = fg_mask
        if fg is not None and fg.shape[:2] != ga.shape[:2]:
            fg = cv2.resize(fg, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_NEAREST)
        ca, la, wa = _star_blobs(ga, fg)
        cb, _, _ = _star_blobs(gb, fg)
        if len(ca) == 0 or len(cb) == 0:
            continue
        d, _ = cKDTree(cb).query(ca, distance_upper_bound=_PARTNER_PX)
        ok = np.isfinite(d)
        lens.append(la[ok])
        wids.append(wa[ok])
        shifts.append(d[ok])
    if not lens:
        return None
    lens = np.concatenate(lens)
    wids = np.concatenate(wids)
    shifts = np.concatenate(shifts)
    if len(lens) < _MIN_STARS:
        return None
    return {"length": float(np.percentile(lens, _PCTL)),
            "width": float(np.percentile(wids, _PCTL)),
            "stars": int(len(lens)),
            # How far a star moves between two frames in this set (median of the paired stars' distances), so
            # the mover rescue can look for a partner only as far as a star could have gone.
            "shift": float(np.median(shifts))}
