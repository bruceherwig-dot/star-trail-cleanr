"""
Crossing splitter — Voronoi arm approach for X, T, Y and V shaped trail blobs.

Replaces try_split() in trail_grouper.py. The old function is kept there but
no longer called. To revert: in trail_grouper.py swap the two split_crossing()
calls back to try_split() and remove the crossing_splitter import.

THE FOUR-ARM MODEL
------------------
When two trails cross (or meet), a single SAHI detection blob contains up to
four arms radiating from a junction point:

  - Arm A+  (trail A, positive direction from junction)
  - Arm A-  (trail A, negative direction from junction)
  - Arm B+  (trail B, positive direction from junction)
  - Arm B-  (trail B, negative direction from junction)

Each pixel is assigned to exactly one arm using Voronoi (perpendicular distance
to each trail centerline) then split by sign of projection along the trail
direction. Junction pixels are naturally assigned to whichever arm they are
closest to -- no separate junction piece, no black fill, no overlap.

For T, Y, V shapes one or two arms are absent (too few pixels) and are skipped,
yielding 3 or 2 arms respectively.

HOW THIS MODULE HANDLES IT
---------------------------
1. Find two dominant Hough angle clusters.
2. Find the junction anchor: intersection of the two representative lines.
3. Disc sweep (validation only): try increasing exclusion radii until both
   Voronoi halves each contain an elongated arm. This confirms the split is
   real before committing to the four-arm output.
4. Output: assign ALL blob pixels (including junction zone) to their nearest
   arm via Voronoi + projection sign. Return up to four non-overlapping masks.
   Each arm is a continuous rectangular strip from its tip to the junction
   center -- no gaps, no overlaps, ready for independent Star Bridge repair.

Public API
----------
split_crossing(mask) -> list[ndarray]
    Returns [mask] unchanged if no valid split is found.
    Returns [arm1, arm2, ...] (2-4 masks) if the blob splits into trail arms.
    Each returned mask is a separate non-overlapping arm for independent repair.
"""

import cv2
import math
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
_MIN_SPLIT_ANGLE    = 25.0   # deg -- minimum angular separation between clusters
_MIN_AREA           = 1000   # px  -- minimum area for a valid trail arm
_MIN_ASPECT         = 2.0    # ratio -- minimum major/minor for a valid trail arm
_DISC_STEP          = 10     # px  -- radius increment when scanning for clean split
_DISC_MAX           = 80     # px  -- maximum exclusion radius to attempt
_MIN_ARM_FRACTION   = 0.12   # ratio -- smaller arm must be >= 12% of combined kept pixels
_REF_FRAME_PX       = 6000 * 4000   # reference resolution for normalizing area thresholds


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

def _has_valid_arm(m, area_scale=1.0):
    """Return True if mask contains at least one component shaped like a trail arm.

    Checks every connected component individually -- unlike detection_props in
    trail_grouper.py which only inspects the largest one. This is what allows the
    junction-disc approach to work: after disc exclusion the fat crossing-zone blob
    is gone and we only need one of the remaining arms to be valid.
    """
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < _MIN_AREA * area_scale:
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
    area_scale = (mask.shape[0] * mask.shape[1]) / _REF_FRAME_PX
    if area < _SPLIT_AREA_MIN * area_scale:
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

    # ── Disc sweep: validation only ───────────────────────────────────────────
    # Exclude increasing disc radii around the anchor until both Voronoi halves
    # each contain an elongated arm. This confirms the blob is a real crossing
    # before committing to the four-arm output. The disc radius is NOT used in
    # the output -- all pixels (including junction) are assigned via Voronoi.
    valid_split = False

    for disc_r in range(0, _DISC_MAX + _DISC_STEP, _DISC_STEP):
        if disc_r > 0:
            keep = (xs_f - ax) ** 2 + (ys_f - ay) ** 2 >= disc_r * disc_r
        else:
            keep = np.ones(len(xs_f), dtype=bool)

        yk = ys[keep] - row_lo
        xk = xs[keep] - col_lo
        side_a_k = d1_all[keep] <= d2_all[keep]

        mask_a_c = np.zeros((crop_rows, crop_cols), dtype=np.uint8)
        mask_b_c = np.zeros((crop_rows, crop_cols), dtype=np.uint8)
        mask_a_c[yk[side_a_k],  xk[side_a_k]]  = 255
        mask_b_c[yk[~side_a_k], xk[~side_a_k]] = 255

        if mask_a_c.max() == 0 or mask_b_c.max() == 0:
            continue

        n_a = int((mask_a_c > 0).sum())
        n_b = int((mask_b_c > 0).sum())
        if min(n_a, n_b) / max(n_a + n_b, 1) < _MIN_ARM_FRACTION:
            continue

        if _has_valid_arm(mask_a_c, area_scale) and _has_valid_arm(mask_b_c, area_scale):
            valid_split = True
            break

    if not valid_split:
        return [mask]

    # ── Four-arm Voronoi output ───────────────────────────────────────────────
    # Assign ALL blob pixels (including junction zone) to their nearest arm.
    # Each pixel goes to the trail whose centerline it is closest to (d1 vs d2),
    # then to the arm on its side of the junction (projection sign along trail).
    # Result: up to four non-overlapping masks, each a continuous tip-to-center
    # rectangular strip. Skips any arm with insufficient pixels (handles T/Y/V).
    side_a = d1_all <= d2_all
    udx1, udy1 = dx1 / len1, dy1 / len1
    udx2, udy2 = dx2 / len2, dy2 / len2
    proj_a = (xs_f - ax) * udx1 + (ys_f - ay) * udy1
    proj_b = (xs_f - ax) * udx2 + (ys_f - ay) * udy2

    arm_selections = [
        side_a  & (proj_a >= 0),   # trail A, positive direction
        side_a  & (proj_a <  0),   # trail A, negative direction
        ~side_a & (proj_b >= 0),   # trail B, positive direction
        ~side_a & (proj_b <  0),   # trail B, negative direction
    ]

    result = []
    for arm_sel in arm_selections:
        if int(arm_sel.sum()) < _MIN_AREA * area_scale:
            continue
        m_crop = np.zeros((crop_rows, crop_cols), dtype=np.uint8)
        m_crop[ys[arm_sel] - row_lo, xs[arm_sel] - col_lo] = 255
        m_full = np.zeros_like(mask)
        m_full[row_lo:row_hi + 1, col_lo:col_hi + 1] = m_crop
        result.append(m_full)

    return result if len(result) >= 2 else [mask]
