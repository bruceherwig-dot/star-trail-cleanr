"""
Crossing splitter — junction-disc approach for X, T, and V shaped trail blobs.

Replaces try_split() in trail_grouper.py. The old function is kept there but
no longer called. To revert: in trail_grouper.py swap the two split_crossing()
calls back to try_split() and remove the crossing_splitter import.

THE FIVE-PIECE MODEL
--------------------
When two trails cross (or meet), a single SAHI detection blob contains up to
five distinct regions:

  - Left arm   (trail A, left of junction)
  - Right arm  (trail A, right of junction)
  - Top arm    (trail B, above junction)
  - Bottom arm (trail B, below junction)
  - Junction   (the middle zone where both trails physically overlap)

The junction zone is what caused the old try_split() to fail: assigning those
pixels to either half inflated the minor axis of that half, and the quality
check rejected the split.

HOW THIS MODULE HANDLES IT
---------------------------
1. Find two dominant Hough angle clusters -- same detection logic as old code.
2. Find the junction point: intersection of the two cluster representative lines.
3. Try increasing disc radii (0, 10, 20 ... 80 px) around the junction until
   both split halves each contain at least one elongated trail arm.
4. Once found, dilate each clean arm back by that radius and clip to the
   original blob pixels. This fills the gap so each trail's mask is continuous
   and both masks overlap in the crossing zone -- no unrepaired gap is left.

This handles X, T, and V shapes with the same logic. For T and V the junction
lands at the endpoint of one trail (or a shared vertex) rather than the middle,
but the disc exclusion and dilation work identically.

Public API
----------
split_crossing(mask) -> list[ndarray]
    Returns [mask] unchanged if no valid split is found.
    Returns [mask_a, mask_b] if the blob cleanly separates into two trails.
    Same return contract as the old try_split() in trail_grouper.py.
"""

import cv2
import numpy as np
from skimage.measure import label as sklabel, regionprops as skregionprops

# ── Constants ──────────────────────────────────────────────────────────────────
# These mirror the equivalent values in trail_grouper.py.
# If trail_grouper constants change, update these to match.

_SPLIT_AREA_MIN     = 5000   # px  -- blobs smaller than this are never split
_HOUGH_THRESHOLD    = 20     # votes -- Hough threshold for large blobs
_HOUGH_MIN_LINE     = 25     # px  -- minimum Hough line length
_HOUGH_MAX_GAP      = 15     # px  -- maximum gap in a Hough line
_DBSCAN_EPS         = 5.0    # deg -- angular neighbourhood for DBSCAN clustering
_DBSCAN_MIN_SAMPLES = 2      # minimum lines per DBSCAN cluster
_MIN_SPLIT_ANGLE    = 10.0   # deg -- minimum angular separation between clusters
_MIN_AREA           = 1000   # px  -- minimum area for a valid trail arm
_MIN_ASPECT         = 2.0    # ratio -- minimum major/minor for a valid trail arm
_DISC_STEP          = 10     # px  -- radius increment when scanning for clean split
_DISC_MAX           = 80     # px  -- maximum exclusion radius to attempt


# ── Geometry helpers ───────────────────────────────────────────────────────────
# Self-contained copies of the helpers in trail_grouper.py.
# Duplicated here to keep this module independent (avoids circular imports).

def _line_angle_deg(L):
    x1, y1, x2, y2 = L
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _circular_mean_angle(angle_list):
    rads = np.radians([2.0 * a for a in angle_list])
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.degrees(mean_rad / 2.0) % 180.0)


def _angle_dist(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)


def _perp_dist(px, py, L):
    x1, y1, x2, y2 = L
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return float("inf")
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / np.sqrt(dx * dx + dy * dy)


def _line_intersect(L1, L2):
    x1, y1, x2, y2 = L1
    x3, y3, x4, y4 = L2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _dbscan_angles(angles, eps, min_samples):
    n = len(angles)
    labels  = [-1] * n
    visited = [False] * n
    cluster_id = 0

    def nbrs(i):
        return [j for j in range(n) if _angle_dist(angles[i], angles[j]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        hood = nbrs(i)
        if len(hood) < min_samples:
            continue
        labels[i] = cluster_id
        seed = list(hood)
        while seed:
            q = seed.pop()
            if not visited[q]:
                visited[q] = True
                q_hood = nbrs(q)
                if len(q_hood) >= min_samples:
                    seed.extend(q_hood)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels, cluster_id


# ── Shape validator ────────────────────────────────────────────────────────────

def _has_valid_arm(m):
    """Return True if mask contains at least one component shaped like a trail arm.

    Checks every connected component individually -- unlike detection_props in
    trail_grouper.py which only inspects the largest one. This is what allows the
    junction-disc approach to work: after disc exclusion the fat crossing-zone blob
    is gone and we only need one of the remaining arms to be valid.
    """
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < _MIN_AREA:
            continue
        comp = (lbl == i).astype(np.uint8) * 255
        rp = skregionprops(sklabel(comp))
        if not rp:
            continue
        p = max(rp, key=lambda x: x.area)
        if p.axis_minor_length < 1:
            continue
        if p.axis_major_length / p.axis_minor_length >= _MIN_ASPECT:
            return True
    return False


def _passes_shape_filter(m):
    """Return True if mask passes the same shape test detection_props applies in trail_grouper.

    Mirrors detection_props exactly: picks the largest component, checks area,
    aspect ratio, and pixel-density gate. All three must pass for split_crossing
    to accept the dilated output as a valid trail mask.
    """
    rp = skregionprops(sklabel(m))
    if not rp:
        return False
    p = max(rp, key=lambda x: x.area)
    if p.area < _MIN_AREA:
        return False
    minor = p.axis_minor_length
    if minor < 1:
        return False
    if p.axis_major_length / minor < _MIN_ASPECT:
        return False
    pixel_density = p.area / max(p.axis_major_length, 1)
    if minor > 50 and minor > 2 * pixel_density:
        return False
    return True


# ── Public API ─────────────────────────────────────────────────────────────────

def split_crossing(mask):
    """Split an X, T, or V shaped blob into two independent trail masks.

    Finds the junction point, tries increasing exclusion disc radii until both
    halves each have an elongated arm, then dilates each arm back into the
    crossing zone so both masks provide complete coverage with overlap.

    Returns [mask] unchanged if no valid split is found.
    Returns [mask_a, mask_b] if the blob splits cleanly.
    """
    area = int((mask > 0).sum())
    if area < _SPLIT_AREA_MIN:
        return [mask]

    blob_u8 = (mask > 0).astype(np.uint8) * 255

    # Pixel coords of the blob in full-frame space.
    ys, xs = np.where(blob_u8 > 0)
    row_lo, row_hi = int(ys.min()), int(ys.max())
    col_lo, col_hi = int(xs.min()), int(xs.max())

    # Run HoughLinesP on a tight crop instead of the full frame.
    # A 6000x4000 mask with a small trail blob wastes 99%+ of Hough scan time
    # on empty pixels. Cropping first gives ~100x speedup on large frames.
    blob_crop = blob_u8[row_lo:row_hi + 1, col_lo:col_hi + 1]
    local_threshold = 10 if area < 20000 else _HOUGH_THRESHOLD
    lines_crop = cv2.HoughLinesP(blob_crop, 1, np.pi / 180,
                                 threshold=local_threshold,
                                 minLineLength=_HOUGH_MIN_LINE,
                                 maxLineGap=_HOUGH_MAX_GAP)
    if lines_crop is None or len(lines_crop) < 2:
        return [mask]

    # Translate crop-local line coords to full-frame coords.
    lines = [[[L[0][0] + col_lo, L[0][1] + row_lo,
               L[0][2] + col_lo, L[0][3] + row_lo]]
             for L in lines_crop]

    angles = [_line_angle_deg(L[0]) for L in lines]
    labels, _ = _dbscan_angles(angles, _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES)
    unique = [l for l in set(labels) if l >= 0]
    if len(unique) < 2:
        return [mask]

    cluster_means = {cid: _circular_mean_angle(
        [angles[i] for i, l in enumerate(labels) if l == cid]
    ) for cid in unique}

    # Pick the pair with the largest angular separation.
    best_c1, best_c2, best_dist = unique[0], unique[1], 0.0
    for ci in range(len(unique)):
        for cj in range(ci + 1, len(unique)):
            d = _angle_dist(cluster_means[unique[ci]], cluster_means[unique[cj]])
            if d > best_dist:
                best_dist = d
                best_c1, best_c2 = unique[ci], unique[cj]

    if best_dist < _MIN_SPLIT_ANGLE:
        return [mask]

    def _longest(cid):
        clines = [lines[i][0] for i, l in enumerate(labels) if l == cid]
        return max(clines, key=lambda L: (L[2] - L[0]) ** 2 + (L[3] - L[1]) ** 2)

    rep1 = _longest(best_c1)
    rep2 = _longest(best_c2)

    # Junction anchor: intersection of the two rep lines, fallback to centroid
    # if the intersection falls outside the blob's bounding box.
    pt = _line_intersect(rep1, rep2)
    if pt is not None:
        ax, ay = pt
        if not (col_lo <= ax <= col_hi and row_lo <= ay <= row_hi):
            ax, ay = float(xs.mean()), float(ys.mean())
    else:
        ax, ay = float(xs.mean()), float(ys.mean())

    # Extended rep lines through the anchor at each cluster's mean angle.
    extent = float(max(mask.shape) + 100)

    def _anchor_line(angle_deg):
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad) * extent, np.sin(rad) * extent
        return [ax - dx, ay - dy, ax + dx, ay + dy]

    arep1 = _anchor_line(cluster_means[best_c1])
    arep2 = _anchor_line(cluster_means[best_c2])

    # Precompute perpendicular distances for all blob pixels at once.
    # The old code looped over each pixel in Python calling _perp_dist() --
    # for a 50k-pixel blob that's 50k function calls per disc radius.
    # Vectorizing with numpy eliminates that overhead entirely.
    x1_1, y1_1, x2_1, y2_1 = arep1
    x1_2, y1_2, x2_2, y2_2 = arep2
    dx1, dy1 = x2_1 - x1_1, y2_1 - y1_1
    dx2, dy2 = x2_2 - x1_2, y2_2 - y1_2
    len1 = float(np.hypot(dx1, dy1))
    len2 = float(np.hypot(dx2, dy2))
    xs_f = xs.astype(float)
    ys_f = ys.astype(float)
    d1_all = np.abs(dy1 * xs_f - dx1 * ys_f + x2_1 * y1_1 - y2_1 * x1_1) / len1
    d2_all = np.abs(dy2 * xs_f - dx2 * ys_f + x2_2 * y1_2 - y2_2 * x1_2) / len2

    crop_rows = row_hi - row_lo + 1
    crop_cols = col_hi - col_lo + 1

    found_a = found_b = None

    for disc_r in range(0, _DISC_MAX + _DISC_STEP, _DISC_STEP):
        if disc_r > 0:
            keep = (xs_f - ax) ** 2 + (ys_f - ay) ** 2 >= disc_r * disc_r
        else:
            keep = np.ones(len(xs_f), dtype=bool)

        yk = ys[keep] - row_lo
        xk = xs[keep] - col_lo
        side_a = d1_all[keep] <= d2_all[keep]

        mask_a_c = np.zeros((crop_rows, crop_cols), dtype=np.uint8)
        mask_b_c = np.zeros((crop_rows, crop_cols), dtype=np.uint8)
        mask_a_c[yk[side_a],  xk[side_a]]  = 255
        mask_b_c[yk[~side_a], xk[~side_a]] = 255

        if mask_a_c.max() == 0 or mask_b_c.max() == 0:
            continue

        if not _has_valid_arm(mask_a_c) or not _has_valid_arm(mask_b_c):
            continue

        if disc_r > 0:
            d = 2 * disc_r + 1
            struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            out_a_c = cv2.bitwise_and(cv2.dilate(mask_a_c, struct), blob_crop)
            out_b_c = cv2.bitwise_and(cv2.dilate(mask_b_c, struct), blob_crop)
        else:
            out_a_c = mask_a_c
            out_b_c = mask_b_c

        if _passes_shape_filter(out_a_c) and _passes_shape_filter(out_b_c):
            found_a = out_a_c
            found_b = out_b_c
            break

    if found_a is None:
        return [mask]

    # Embed crop-local results back into full-frame masks.
    out_a = np.zeros_like(mask)
    out_b = np.zeros_like(mask)
    out_a[row_lo:row_hi + 1, col_lo:col_hi + 1] = found_a
    out_b[row_lo:row_hi + 1, col_lo:col_hi + 1] = found_b
    return [out_a, out_b]
