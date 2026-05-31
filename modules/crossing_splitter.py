"""
Crossing splitter -- spine + tips model for X, T, Y, V, and multi-crossing blobs.

SPINE + TIPS MODEL
------------------
When trails cross, one trail is the SPINE (longest, runs unbroken through the
junction including the junction zone). Everything else is TIPS -- the arms of
crossing trails that extend above and below the spine band. Each tip is a
separate connected component.

Examples:
  X crossing  -> 3 pieces: 1 spine + 2 tips (upper arm, lower arm)
  3-trail     -> 5 pieces: 1 spine + 4 tips (2 above, 2 below)

Key properties:
  - union(spine + all tips) == original SAHI blob. Full coverage, no lost pixels.
  - The spine is an unbroken, full-length trail mask (best for repair).
  - Each tip is a short arm (repaired independently with its own motion estimate).
  - The junction zone stays with the spine (no Voronoi artifacts at the junction).

TRIGGER: blob elongation (major/minor) <= threshold. A single trail is highly
elongated (8-20+) and skips. A crossing blob is squat (1-4) and enters.
Fully dimensionless, no pixel threshold.

HOW IT WORKS
------------
1. Elongation trigger (dimensionless).
2. Find the spine direction: Hough line segments grouped by angle (DBSCAN),
   pick the cluster with the most total line length.
3. Measure single-trail width: sample cross-sections along the spine at many
   positions, take the 20th percentile width (= width where only the spine
   is present, robust to junction-zone widening).
4. Spine mask: all blob pixels within (spine_width * 0.6) of the spine
   centerline. Captures the full spine trail plus the junction overlap zone.
5. Tip masks: all remaining blob pixels, split into connected components.
   Each component with sufficient area is a separate tip.
6. Validate: spine must be elongated, at least 1 valid tip required.

Public API
----------
split_crossing(mask) -> list[ndarray]
    Returns [mask] unchanged if no valid split is found.
    Returns [spine, tip1, tip2, ...] if the blob splits.
"""

import cv2
import math
import numpy as np
from skimage.measure import label as sklabel, regionprops as skregionprops


# -- Constants ----------------------------------------------------------------

_SPLIT_AREA_MIN     = 5000    # px (at ref resolution) -- blobs smaller skip
_ELONGATION_THRESH  = 4.5     # ratio -- blobs more elongated skip (already single trail)
_HOUGH_THRESHOLD    = 20      # votes
_HOUGH_MIN_LINE     = 25      # px -- minimum Hough line length
_HOUGH_MAX_GAP      = 15      # px -- maximum gap in a Hough line
_DBSCAN_EPS         = 5.0     # deg -- angular neighbourhood for DBSCAN
_DBSCAN_MIN_SAMPLES = 2       # minimum lines per DBSCAN cluster
_MIN_SPLIT_ANGLE    = 15.0    # deg -- below this, all clusters are near-parallel
_MIN_AREA           = 1000    # px (at ref resolution) -- minimum area for a valid output
_MIN_ASPECT         = 2.0     # ratio -- minimum major/minor for valid spine
_SPINE_BAND_FACTOR  = 0.6     # spine half-width = measured_width * this factor
_WIDTH_PERCENTILE   = 20      # percentile of cross-section widths for single-trail estimate
_N_WIDTH_SAMPLES    = 20      # number of cross-section samples along the spine
_END_FRACTION       = 0.15    # fraction of spine length at each end used for width
_REF_FRAME_PX       = 6000 * 4000  # reference resolution for normalizing area thresholds


# -- Geometry helpers ----------------------------------------------------------

def _line_angle_deg(L):
    """Angle of a line segment in degrees [0, 180)."""
    x1, y1, x2, y2 = L
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _circular_mean_angle(angle_list):
    """Mean of angles in [0, 180) handling wraparound."""
    rads = np.radians([2.0 * a for a in angle_list])
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.degrees(mean_rad / 2.0) % 180.0)


def _angle_dist(a, b):
    """Angular distance in [0, 90] between two angles in [0, 180)."""
    d = abs(a - b)
    return min(d, 180.0 - d)


def _line_length(L):
    """Length of a line segment."""
    x1, y1, x2, y2 = L
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _dbscan_angles(angles, eps, min_samples):
    """DBSCAN clustering on circular angle values."""
    n = len(angles)
    labels = [-1] * n
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


# -- Width measurement --------------------------------------------------------

def _measure_spine_width(xs_f, ys_f, along, perp, n_samples=20, pctl=20):
    """Measure the single-trail width by sampling cross-sections along the spine.

    Divides the along-spine range into n_samples bins, computes the perpendicular
    extent (max - min) in each bin, and returns the pctl-th percentile extent.
    At crossing points the extent is wider; the low percentile captures the
    narrower sections where only the spine is present.
    """
    t_min, t_max = float(along.min()), float(along.max())
    t_span = t_max - t_min
    if t_span < 5:
        return float(perp.max() - perp.min())

    edges = np.linspace(t_min, t_max, n_samples + 1)
    widths = []
    for i in range(n_samples):
        sel = (along >= edges[i]) & (along < edges[i + 1])
        if sel.sum() < 5:
            continue
        p_sel = perp[sel]
        widths.append(float(p_sel.max() - p_sel.min()))

    if not widths:
        return float(perp.max() - perp.min())

    return float(np.percentile(widths, pctl))


# -- Public API ----------------------------------------------------------------

def split_crossing(mask, frame_px=None):
    """Split a crossing blob into spine + tip masks.

    The spine is the longest trail, running unbroken through the junction
    (including the junction zone). Tips are the crossing trail arms that
    extend above and below the spine band. Full SAHI coverage is preserved:
    union(spine + tips) == original blob.

    frame_px: total pixels of the FULL frame, used to normalise the area
    thresholds. Pass it when `mask` is a cropped region so the thresholds match
    the full-frame behaviour. Defaults to the mask's own size (full-frame call).

    Returns [mask] unchanged if no valid split is found.
    Returns [spine, tip1, tip2, ...] if the blob splits.
    """
    area = int((mask > 0).sum())
    h, w = mask.shape[:2]
    area_scale = (frame_px if frame_px else (h * w)) / _REF_FRAME_PX
    min_px = int(_MIN_AREA * area_scale)
    if area < _SPLIT_AREA_MIN * area_scale:
        return [mask]

    # -- Elongation trigger (dimensionless) ------------------------------------
    blob_u8 = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(blob_u8 > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    major_len = max(rect[1])
    minor_len = min(rect[1])
    if minor_len < 1:
        return [mask]
    elongation = major_len / minor_len
    if elongation > _ELONGATION_THRESH:
        return [mask]  # already a single thin trail

    # -- Hough lines on tight crop ---------------------------------------------
    row_lo, row_hi = int(ys.min()), int(ys.max())
    col_lo, col_hi = int(xs.min()), int(xs.max())
    blob_crop = blob_u8[row_lo:row_hi + 1, col_lo:col_hi + 1]
    local_threshold = 10 if area < 20000 else _HOUGH_THRESHOLD
    lines_crop = cv2.HoughLinesP(blob_crop, 1, np.pi / 180,
                                 threshold=local_threshold,
                                 minLineLength=_HOUGH_MIN_LINE,
                                 maxLineGap=_HOUGH_MAX_GAP)
    if lines_crop is None or len(lines_crop) < 2:
        return [mask]

    # Translate crop-local to full-frame coords
    lines = [[[L[0][0] + col_lo, L[0][1] + row_lo,
               L[0][2] + col_lo, L[0][3] + row_lo]]
             for L in lines_crop]

    # -- Angle clustering ------------------------------------------------------
    angles = [_line_angle_deg(L[0]) for L in lines]
    labels, _ = _dbscan_angles(angles, _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES)
    unique = sorted(set(l for l in labels if l >= 0))
    if not unique:
        return [mask]

    cluster_means = {cid: _circular_mean_angle(
        [angles[i] for i, l in enumerate(labels) if l == cid]
    ) for cid in unique}

    # Need at least 2 distinct angle clusters to identify a crossing
    if len(unique) >= 2:
        best_dist = 0.0
        for ci in range(len(unique)):
            for cj in range(ci + 1, len(unique)):
                d = _angle_dist(cluster_means[unique[ci]], cluster_means[unique[cj]])
                if d > best_dist:
                    best_dist = d
        if best_dist < _MIN_SPLIT_ANGLE:
            return [mask]  # all near-parallel, not a crossing
    else:
        return [mask]  # single direction, not a crossing

    # -- Pick spine direction: cluster with the most total Hough line length ----
    cluster_lengths = {}
    for cid in unique:
        total = sum(_line_length(lines[i][0])
                    for i, l in enumerate(labels) if l == cid)
        cluster_lengths[cid] = total

    spine_cid = max(unique, key=lambda c: cluster_lengths[c])
    spine_angle = cluster_means[spine_cid]

    # -- Project all blob pixels onto spine coordinate system ------------------
    xs_f = xs.astype(float)
    ys_f = ys.astype(float)
    cx_blob = float(xs_f.mean())
    cy_blob = float(ys_f.mean())

    spine_rad = np.radians(spine_angle)
    along_dx, along_dy = np.cos(spine_rad), np.sin(spine_rad)
    perp_dx, perp_dy = -np.sin(spine_rad), np.cos(spine_rad)

    # along = projection onto spine direction (how far along the spine)
    along = (xs_f - cx_blob) * along_dx + (ys_f - cy_blob) * along_dy
    # perp = perpendicular distance from spine centerline (signed)
    perp = (xs_f - cx_blob) * perp_dx + (ys_f - cy_blob) * perp_dy

    # -- Measure single-trail width (robust, using low percentile) -------------
    spine_width = _measure_spine_width(
        xs_f, ys_f, along, perp,
        n_samples=_N_WIDTH_SAMPLES, pctl=_WIDTH_PERCENTILE)

    if spine_width < 3:
        return [mask]

    # Find the spine's center offset by sampling per-bin median perpendicular
    # positions along the spine. Each bin contributes one center estimate;
    # taking the overall median is robust to both X and T crossings. The old
    # end-pixel approach failed on T-shapes because one "end" IS the junction,
    # pulling the center way off.
    t_min, t_max = float(along.min()), float(along.max())
    t_span = t_max - t_min
    edges = np.linspace(t_min, t_max, _N_WIDTH_SAMPLES + 1)
    bin_centers = []
    for i in range(_N_WIDTH_SAMPLES):
        sel = (along >= edges[i]) & (along < edges[i + 1])
        if sel.sum() < 5:
            continue
        bin_centers.append(float(np.median(perp[sel])))
    p_center = float(np.median(bin_centers)) if bin_centers else 0.0

    # -- Build spine and tip masks ---------------------------------------------
    band_half = spine_width * _SPINE_BAND_FACTOR
    spine_sel = np.abs(perp - p_center) <= band_half

    # Spine mask
    spine_mask = np.zeros_like(mask)
    spine_mask[ys[spine_sel], xs[spine_sel]] = 255

    # Validate spine is elongated
    sp_rp = skregionprops(sklabel(spine_mask))
    if sp_rp:
        sp = max(sp_rp, key=lambda x: x.area)
        if sp.axis_minor_length > 0:
            sp_elong = sp.axis_major_length / sp.axis_minor_length
            if sp_elong < _MIN_ASPECT:
                return [mask]  # spine isn't trail-shaped, bail
    else:
        return [mask]

    # Tip masks: remaining pixels, split into connected components
    tip_all = np.zeros_like(mask)
    tip_sel = ~spine_sel
    if tip_sel.sum() < min_px:
        return [mask]  # nothing significant outside the spine
    tip_all[ys[tip_sel], xs[tip_sel]] = 255

    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        tip_all, connectivity=8)

    tips = []
    for cc_id in range(1, n_cc):
        cc_area = cc_stats[cc_id, cv2.CC_STAT_AREA]
        if cc_area < min_px:
            continue
        tip_mask = np.zeros_like(mask)
        tip_mask[cc_labels == cc_id] = 255
        tips.append(tip_mask)

    if not tips:
        return [mask]  # no valid tips

    # -- Verify full coverage --------------------------------------------------
    # Every pixel of the original blob must be in spine or a tip.
    # Tiny fragments that fell below min_px are reassigned to the spine.
    covered = spine_mask.copy()
    for t in tips:
        covered = cv2.bitwise_or(covered, t)
    uncovered = cv2.bitwise_and(blob_u8, cv2.bitwise_not(covered))
    if uncovered.any():
        # Reassign uncovered pixels to spine (they are small sub-threshold fragments)
        spine_mask = cv2.bitwise_or(spine_mask, uncovered)

    return [spine_mask] + tips
