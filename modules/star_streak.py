"""Streaking-star filter (dev, off by default).

Long-exposure sequences (fewer/longer frames -- common when a timelapse is turned into a
star trail) make the stars smear into short LINES instead of dots. The trail detector, trained
on sharp-star sets, mistakes each streak for a trail and erases it.

This module measures how long the stars actually run in THIS set -- read off the frames
themselves, never a formula, so it tracks whatever exposure/lens/optics the shooter used -- and
drops any detected trail shorter than that ceiling. Exceptions:
  - RED (warm) detections are kept (aircraft nav light), reusing repair's warm test.
  - On sharp-star sets the measured ceiling is tiny, so nothing is dropped (self-disabling).

Enabled via astro_clean_v5 --streak-filter (dev checkbox in the GUI). Full rationale + the
measured Bombay Beach numbers: runs/experiments/2026_06_13_star_motion_filter/BUILD_PLAN.md.
"""
import numpy as np
import cv2

from modules.repair import TRAIL_WARM_MARGIN

_MIN_STREAKS = 40      # need at least this many star streaks before a measurement is trusted
_PCTL = 99             # ceiling = this percentile of star-streak lengths ...
_MARGIN = 1.25         # ... times this margin, to clear the longest (edge-bloated) stars
_TRAIL_CUT = 100.0     # streaks longer than this are trails/foreground -> excluded from star stats
_MIN_AREA = 4
_MIN_ASPECT = 1.8

MIN_TRAIL_PX = 150     # flat length floor (px): drop detections shorter than this unless red


def _long_axis(xs, ys):
    """Long-axis length (px) of a point cloud via its principal direction."""
    if len(xs) < 2:
        return 0.0, 0.0
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    t = pts @ vt[0]
    w = pts @ vt[1]
    return float(t.max() - t.min()), float(w.max() - w.min())


def _streak_lengths(gray, fg_mask=None):
    """Lengths of bright, elongated blobs (= star streaks). Foreground excluded."""
    thr = max(42, int(np.median(gray) + 3.5 * gray.std()))
    bw = (gray > thr).astype(np.uint8)
    if fg_mask is not None:
        bw[fg_mask > 127] = 0                      # fg_mask: 255 = foreground, 0 = sky
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bw)
    out = []
    for ci in range(1, n):
        if stats[ci, cv2.CC_STAT_AREA] < _MIN_AREA:
            continue
        x, y = stats[ci, cv2.CC_STAT_LEFT], stats[ci, cv2.CC_STAT_TOP]
        w, h = stats[ci, cv2.CC_STAT_WIDTH], stats[ci, cv2.CC_STAT_HEIGHT]
        ys, xs = np.nonzero(lbl[y:y + h, x:x + w] == ci)
        L, W = _long_axis(xs, ys)
        if L >= 4 and L / (W + 1e-6) >= _MIN_ASPECT:
            out.append(L)
    return out


def measure_ceiling(frame_imgs, fg_mask=None):
    """Per-set star-streak ceiling (px), or None if too few stars to trust the measurement."""
    lens = []
    for img in frame_imgs:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lens.extend(_streak_lengths(gray, fg_mask))
    lens = np.array([v for v in lens if v <= _TRAIL_CUT], dtype=np.float64)
    if len(lens) < _MIN_STREAKS:
        return None
    return float(np.percentile(lens, _PCTL) * _MARGIN)


def _mask_long_axis(mask):
    """Long-axis length (px) of a binary seg mask (cropped to its bbox first, for speed)."""
    m = (mask > 0).astype(np.uint8)
    rect = cv2.boundingRect(m)
    if rect[2] == 0 or rect[3] == 0:
        return 0.0
    x, y, w, h = rect
    ys, xs = np.nonzero(m[y:y + h, x:x + w])
    return _long_axis(xs, ys)[0]


def _is_red(mask, frame_bgr):
    """True if the seg's pixels read warm/red (aircraft nav light), per repair's warm test."""
    if frame_bgr is None or frame_bgr.ndim < 3:
        return False
    m = (mask > 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(m)
    if w == 0 or h == 0:
        return False
    sub_m = m[y:y + h, x:x + w] > 0
    px = frame_bgr[y:y + h, x:x + w][sub_m]
    if len(px) == 0:
        return False
    b = float(np.median(px[:, 0])); r = float(np.median(px[:, 2]))
    return (r - b) > TRAIL_WARM_MARGIN and r > 60


def filter_segs(segs, corners, frame_bgr, ceiling):
    """Drop detections shorter than `ceiling` (long-axis px) unless red (nav light). A flat
    length floor -- no per-set measurement -- so it is cheap and predictable. Returns
    (kept_segs, kept_corners, dropped_segs, n_red_kept), corners kept index-aligned."""
    kept_s, kept_c, dropped, n_red = [], [], [], 0
    for i, seg in enumerate(segs):
        corner = corners[i] if i < len(corners) else None
        if _mask_long_axis(seg) >= ceiling:
            kept_s.append(seg); kept_c.append(corner)
        elif _is_red(seg, frame_bgr):
            n_red += 1
            kept_s.append(seg); kept_c.append(corner)
        else:
            dropped.append(seg)
    return kept_s, kept_c, dropped, n_red
