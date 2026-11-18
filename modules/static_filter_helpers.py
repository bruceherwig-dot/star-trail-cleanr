"""Two helpers for the static-false-positive step, working only where the pixels are.

WHAT THEY ARE FOR. After the static filter decides some detections are fixed objects
(a roofline, a fence post), two jobs follow:
  * is_bright_trail_boxed: before a detection is thrown out as static, check whether its
    pixels are much brighter than the sky around them (a real bright trail, such as a
    plane's strobe, is kept even when it matches neighbouring frames).
  * drop_rejected_segments: take out of a frame's repair and export lists every outline
    that the static filter mostly erased (half or more of its pixels removed).

WHY THEY LOOK LIKE THIS. Both used to do their counting across the WHOLE photograph for
every single detection or outline (22 million pixels, several times each), though the
thing being judged covers a few thousand pixels. Measured 2026-11-18 on a 6-frame batch of
a trail-heavy set: 3.8 s dropping outlines (one call) and 1.9 s on the bright-trail check
(11 calls). Both now look only where the pixels are and give EXACTLY the same answers.
The original whole-frame versions are kept in tests/test_static_filter_helpers.py as the
definition, and the tests compare the two on many random cases.

Both are recorded in the run's timing table (static.bright_trail_veto and
batch.drop_rejected_polygons), with what they decided.
"""
import cv2
import numpy as np


def is_bright_trail_boxed(comp_crop, box, img_bgr, ratio_threshold, pad=12):
    """(is_bright, ratio): do the component's pixels stand out from the sky around them?

    comp_crop: True where the component is, covering the box (x1, y1, x2, y2) (inclusive
    corners, in full-frame pixels). Compares the 90th-percentile brightest-channel value
    inside the component with the median of a ring 10 pixels wide around it. Same numbers
    as the whole-frame version, found on a crop padded by `pad` (more than the ring needs),
    clipped at the frame edge so the edge behaves exactly as before.
    """
    if img_bgr is None:
        return False, 0.0
    x1, y1, x2, y2 = box
    H, W = img_bgr.shape[:2]
    ya, yb = max(0, y1 - pad), min(H, y2 + 1 + pad)
    xa, xb = max(0, x1 - pad), min(W, x2 + 1 + pad)
    canvas = np.zeros((yb - ya, xb - xa), dtype=bool)
    canvas[y1 - ya:y2 + 1 - ya, x1 - xa:x2 + 1 - xa] = comp_crop
    img = img_bgr[ya:yb, xa:xb]
    ys, xs = np.where(canvas)
    if len(ys) == 0:
        return False, 0.0
    inside = img[ys, xs].astype(np.float32)
    inside_bright = float(np.percentile(np.max(inside, axis=1), 90))
    m = canvas.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(m, kernel) > 0) & ~canvas
    ry, rx = np.where(ring)
    if len(ry) == 0:
        return False, 0.0
    surrounding = img[ry, rx].astype(np.float32)
    surround_med = float(np.median(np.max(surrounding, axis=1)))
    if surround_med < 1:
        return False, 0.0
    ratio = round(inside_bright / surround_med, 2)
    return ratio >= ratio_threshold, ratio


def drop_rejected_segments(segs, corners, removed):
    """Remove the outlines the static filter mostly erased.

    segs: one full-frame mask per outline; corners: the outline corner lists, in the same
    order (one may be missing at the end); removed: True where the static filter removed
    pixels. An outline is dropped when half or more of its pixels were removed.
    Returns (kept_segs, kept_corners, number_dropped).

    Only the area the filter removed can overlap an outline, so the overlap is counted in
    that area's own box; the outline's total size is a single fast count. Same counts and
    the same decision as checking the whole frame for every outline.
    """
    rows = np.flatnonzero(removed.any(axis=1))
    kept_s, kept_c, n_drop = [], [], 0
    if rows.size:
        cols = np.flatnonzero(removed.any(axis=0))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        removed_box = removed[r0:r1, c0:c1]
    for j, seg in enumerate(segs):
        drop = False
        if rows.size:
            if seg.dtype == np.uint8 and seg.ndim == 2:
                px = cv2.countNonZero(seg)
            else:
                px = int(np.count_nonzero(seg > 0))
            if px:
                overlap = int(np.count_nonzero((seg[r0:r1, c0:c1] > 0) & removed_box))
                drop = overlap * 2 >= px
        if drop:
            n_drop += 1
            continue
        kept_s.append(seg)
        if j < len(corners):
            kept_c.append(corners[j])
    return kept_s, kept_c, n_drop
