"""
Trail grouper: crossing splitter + elongation filter + union-find grouper + polygon fitting.

Public API
----------
filter_masks(preds, h, w)
    Run crossing splitter + elongation filter on raw SAHI predictions.
    Returns a list of uint8 masks (255=trail) that passed all filters.
    Used by detect_trails.detect_frame to build the combined mask.

filter_masks_with_props(preds, h, w, sky_mask=None)
    Like filter_masks but returns (masks, det_list) in one pass.
    masks[i] and det_list[i] correspond to the same detection.
    Used by polymakr.py for full-chain detection → polygon fitting.

thicken_mask(mask, h, w, min_aspect=2.0)
    Convert a combined binary pixel mask into fit_polygon rectangles.
    Returns (polygon_list, thick_mask). Used by STC repair to replace the
    tight pixel mask with wider rectangles consistent with CVAT annotations.

group_detections(det_list)
    Union-find grouper with 4-gate check (angle, width-ratio, perp, tip-to-tip).
    Returns list-of-lists of detection indices.

fit_polygon(group_indices, det_list)
    Fit one 4-corner rectangle to a group of detections.
    Returns (corners_xy, width, u_avg).

detection_props(mask, min_aspect=None)
    Return region properties for one mask, or None if elongation filter fails.

try_split(mask)
    Crossing splitter: returns [mask] unchanged or [mask_a, mask_b].
"""

import cv2
import numpy as np
from skimage.measure import label as sklabel, regionprops as skregionprops

# ── Constants ─────────────────────────────────────────────────────────────────

CONF_THRESH         = 0.30
MAX_ANGLE_DEG       = 7.0
MIN_ASPECT          = 3.5
MIN_AREA            = 300

SPLIT_AREA_MIN      = 5000
HOUGH_THRESHOLD     = 20
HOUGH_MIN_LINE      = 25
HOUGH_MAX_GAP       = 15
DBSCAN_EPS          = 5.0
DBSCAN_MIN_SAMPLES  = 2
MIN_SPLIT_ANGLE_DEG = 15.0
SEAM_MARGIN         = 3.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pred_to_mask(pred, h, w):
    if pred.mask is None:
        return None
    seg = pred.mask.bool_mask
    if seg is None:
        return None
    arr = seg.astype(np.uint8) * 255
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    return arr


def _pca_aspect(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    cov = np.cov(pts.T)
    ev, _ = np.linalg.eigh(cov)
    return float(np.sqrt(ev[-1]) / max(0.5, np.sqrt(ev[0])))


def _line_angle_deg(L):
    x1, y1, x2, y2 = L
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _circular_mean_angle(angle_list):
    """Circular mean for undirected angles in [0, 180) degrees."""
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


# ── Crossing splitter ─────────────────────────────────────────────────────────

def try_split(mask):
    """
    Check mask for an X-shaped crossing blob and split it if found.

    Returns [mask] unchanged if no crossing is detected, or [mask_a, mask_b]
    if exactly two distinct trail directions are found.

    Adaptive Hough threshold: blobs with area < 20,000px use threshold=10;
    larger blobs use HOUGH_THRESHOLD (20). Smaller crossing blobs don't
    generate enough Hough votes at threshold=20 to fire the split.
    """
    area = int((mask > 0).sum())
    if area < SPLIT_AREA_MIN:
        return [mask]

    local_threshold = 10 if area < 20000 else HOUGH_THRESHOLD

    blob_u8 = (mask > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(blob_u8, 1, np.pi / 180,
                            threshold=local_threshold,
                            minLineLength=HOUGH_MIN_LINE,
                            maxLineGap=HOUGH_MAX_GAP)
    if lines is None or len(lines) < 2:
        return [mask]

    angles = [_line_angle_deg(L[0]) for L in lines]
    labels, _ = _dbscan_angles(angles, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    unique = [l for l in set(labels) if l >= 0]

    if len(unique) < 2:
        return [mask]

    # Circular mean per cluster; pick the pair with maximum angular separation.
    # Using circular mean (not linear) handles horizontal-trail lines that straddle
    # the 0°/180° boundary. Accepting >2 clusters handles crossing junctions that
    # generate Hough lines at many angles, chaining what should be 2 clusters into 3+.
    cluster_means = {cid: _circular_mean_angle(
        [angles[i] for i, l in enumerate(labels) if l == cid]
    ) for cid in unique}

    best_c1, best_c2, best_dist = unique[0], unique[1], 0.0
    for ci in range(len(unique)):
        for cj in range(ci + 1, len(unique)):
            d = _angle_dist(cluster_means[unique[ci]], cluster_means[unique[cj]])
            if d > best_dist:
                best_dist = d
                best_c1, best_c2 = unique[ci], unique[cj]

    if best_dist < MIN_SPLIT_ANGLE_DEG:
        return [mask]

    def _longest(cid):
        clines = [lines[i][0] for i, l in enumerate(labels) if l == cid]
        return max(clines, key=lambda L: (L[2] - L[0]) ** 2 + (L[3] - L[1]) ** 2)

    rep1 = _longest(best_c1)
    rep2 = _longest(best_c2)

    ys, xs = np.where(mask > 0)
    mask_a = np.zeros_like(mask)
    mask_b = np.zeros_like(mask)
    for px, py in zip(xs, ys):
        d1 = _perp_dist(px, py, rep1)
        d2 = _perp_dist(px, py, rep2)
        if abs(d1 - d2) < SEAM_MARGIN:
            continue
        if d1 <= d2:
            mask_a[py, px] = 255
        else:
            mask_b[py, px] = 255

    if mask_a.max() == 0 or mask_b.max() == 0:
        return [mask]

    def _arm_minor(m):
        rp = skregionprops(sklabel(m))
        return max(p.axis_minor_length for p in rp) if rp else 0

    if _arm_minor(mask_a) > 50 or _arm_minor(mask_b) > 50:
        return [mask]

    return [mask_a, mask_b]


# ── Elongation filter ─────────────────────────────────────────────────────────

def detection_props(mask, min_aspect=None):
    """
    Return region properties for one mask, or None if the elongation filter fails.

    min_aspect overrides MIN_ASPECT when provided (e.g. lower threshold for
    split arms that are known to be part of a crossing blob).
    """
    if min_aspect is None:
        min_aspect = MIN_ASPECT
    lbl = sklabel(mask)
    props = skregionprops(lbl)
    if not props:
        return None
    p = max(props, key=lambda x: x.area)
    minor = p.axis_minor_length
    if minor < 1 or p.area < MIN_AREA:
        return None
    if p.axis_major_length / minor < min_aspect:
        return None
    pixel_density = p.area / max(p.axis_major_length, 1)
    if minor > 50 and minor > 2 * pixel_density:
        return None
    _, vecs = np.linalg.eigh(p.inertia_tensor)
    u = vecs[:, 0]
    return {
        "centroid": np.array(p.centroid),
        "u":        u,
        "minor":    minor,
        "major":    p.axis_major_length,
        "area":     p.area,
        "coords":   np.column_stack(np.where(mask > 0)),
    }


# ── Public: filter masks ──────────────────────────────────────────────────────

def filter_masks(preds, h, w):
    """
    Run crossing splitter + elongation filter on raw SAHI predictions.

    Returns a list of uint8 masks (255=trail) that passed all filters.
    Caller unions them into one combined mask.
    """
    passing = []
    for pred in preds:
        m = _pred_to_mask(pred, h, w)
        if m is None:
            continue
        candidates = try_split(m)
        is_split = len(candidates) == 2
        for cm in candidates:
            if detection_props(cm, min_aspect=2.5 if is_split else None) is not None:
                passing.append(cm)
    return passing


def filter_masks_with_props(preds, h, w, sky_mask=None):
    """
    Like filter_masks, but returns (masks, det_list) in one pass.

    sky_mask (white=sky, black=foreground), if provided, is AND-ed with each
    prediction mask before the crossing splitter and elongation filter.
    """
    masks    = []
    det_list = []
    for pred in preds:
        m = _pred_to_mask(pred, h, w)
        if m is None:
            continue
        if sky_mask is not None:
            sm = (sky_mask if sky_mask.shape == (h, w)
                  else cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST))
            m = cv2.bitwise_and(m, sm)
            if m.max() == 0:
                continue
        candidates = try_split(m)
        is_split = len(candidates) == 2
        for cm in candidates:
            props = detection_props(cm, min_aspect=2.5 if is_split else None)
            if props is not None:
                masks.append(cm)
                det_list.append(props)
    return masks, det_list


# ── Union-find grouper ────────────────────────────────────────────────────────

def group_detections(det_list):
    """
    Group detections belonging to the same trail using union-find.

    All four gates must pass to connect a pair:
      1. Angle between major axes < MAX_ANGLE_DEG
      2. Width ratio < 3x (prevents grouping thin with fat)
      3. Perpendicular centroid distance < 2x max minor
      4. Two-stage gap: if masks overlap → always merge;
         otherwise tip-to-tip distance < 2.5x min minor

    Returns list-of-lists of detection indices (one list per trail group).
    """
    from scipy.spatial import KDTree

    n = len(det_list)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            di, dj = det_list[i], det_list[j]

            cos_sim = min(abs(float(np.dot(di["u"], dj["u"]))), 1.0)
            adiff = min(np.degrees(np.arccos(cos_sim)),
                        180.0 - np.degrees(np.arccos(cos_sim)))
            if adiff > MAX_ANGLE_DEG:
                continue

            if max(di["minor"], dj["minor"]) / max(min(di["minor"], dj["minor"]), 1) > 3.0:
                continue

            u_ref = di["u"] if np.dot(di["u"], dj["u"]) > 0 else -di["u"]
            diff = dj["centroid"] - di["centroid"]
            along = float(np.dot(diff, u_ref))
            perp = float(np.sqrt(max(float(np.dot(diff, diff)) - along ** 2, 0.0)))
            if perp > 1.2 * max(di["minor"], dj["minor"]):
                continue

            tree_i = KDTree(di["coords"])
            dists, _ = tree_i.query(dj["coords"], k=1)
            nearest_gap = float(np.min(dists))
            if nearest_gap > 0:
                def _pixel_tips(d):
                    t = d["coords"] @ d["u"]
                    return (d["coords"][np.argmax(t)].astype(float),
                            d["coords"][np.argmin(t)].astype(float))
                ti = _pixel_tips(di)
                tj = _pixel_tips(dj)
                tip_gap = min(
                    np.linalg.norm(ti[0] - tj[0]),
                    np.linalg.norm(ti[0] - tj[1]),
                    np.linalg.norm(ti[1] - tj[0]),
                    np.linalg.norm(ti[1] - tj[1]),
                )
                if tip_gap > 2.5 * min(di["minor"], dj["minor"]):
                    continue

            union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ── Polygon fitting ───────────────────────────────────────────────────────────

def thicken_mask(mask, h, w, min_aspect=2.0):
    """
    Convert a combined binary pixel mask into fit_polygon rectangles.

    Finds connected blobs, groups them by trail axis (same gates as
    group_detections), fits one rectangle per group, and returns
    (polygon_list, thick_mask).

    Blobs that fail the elongation filter or coverage check are kept as
    raw pixels in thick_mask so no detected trail is dropped from repair.
    """
    if mask is None or mask.max() == 0:
        return [], np.zeros((h, w), dtype=np.uint8)

    lbl      = sklabel(mask > 0)
    n_labels = int(lbl.max())

    det_list = []
    fallback = np.zeros((h, w), dtype=np.uint8)

    for comp_id in range(1, n_labels + 1):
        comp  = (lbl == comp_id).astype(np.uint8) * 255
        props = detection_props(comp, min_aspect=min_aspect)
        if props is None:
            fallback = np.maximum(fallback, comp)
        else:
            det_list.append(props)

    if not det_list:
        return [], np.maximum(fallback, mask)

    groups   = group_detections(det_list)
    thick    = fallback.copy()
    polys    = []

    union_det = np.zeros((h, w), dtype=np.uint8)
    for det in det_list:
        union_det[det["coords"][:, 0], det["coords"][:, 1]] = 255

    _MIN_COV = 0.30
    for grp in groups:
        corners, _, _ = fit_polygon(grp, det_list)
        pts    = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        poly_m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_m, [pts], 255)
        area   = max(int(poly_m.sum()) // 255, 1)
        olap   = int(cv2.bitwise_and(poly_m, union_det).sum()) // 255
        if olap / area >= _MIN_COV:
            thick = np.maximum(thick, poly_m)
            polys.append(corners)
        else:
            for idx in grp:
                d = det_list[idx]
                thick[d["coords"][:, 0], d["coords"][:, 1]] = 255

    return polys, thick


def fit_polygon(group_indices, det_list):
    """
    Fit one 4-corner rectangle to a group of detections.

    Returns (corners_xy, width, u_avg) where corners_xy is a list of
    (x, y) integer tuples suitable for cv2.polylines / cv2.fillPoly.
    """
    dets = [det_list[i] for i in group_indices]

    median_minor = float(np.median([d["minor"] for d in dets]))
    pruned = [d for d in dets if d["minor"] <= 2.0 * median_minor]
    if pruned:
        dets = pruned

    total_area = sum(d["area"] for d in dets)
    centroid   = sum(d["centroid"] * d["area"] for d in dets) / total_area

    u_sum = np.zeros(2)
    for d in dets:
        u = d["u"]
        if u_sum.dot(u) < 0:
            u = -u
        u_sum += u * d["area"]
    u_avg = u_sum / np.linalg.norm(u_sum)

    t_centroid = centroid @ u_avg
    tip_pad = max(d["minor"] for d in dets) / 2
    t_min = min(float((d["coords"] @ u_avg).min()) for d in dets) - tip_pad
    t_max = max(float((d["coords"] @ u_avg).max()) for d in dets) + tip_pad
    p_min = centroid + (t_min - t_centroid) * u_avg
    p_max = centroid + (t_max - t_centroid) * u_avg

    trail_length = t_max - t_min
    width = (float(np.median([
        min(d["minor"], d["area"] / max(d["major"], 1))
        for d in dets
    ])) * 2.0) + min(trail_length / 200.0, 40.0)

    u_perp   = np.array([-u_avg[1], u_avg[0]])
    half_w   = width / 2.0
    corners_rc = [
        p_min + half_w * u_perp,
        p_min - half_w * u_perp,
        p_max - half_w * u_perp,
        p_max + half_w * u_perp,
    ]
    corners_xy = [(int(c[1]), int(c[0])) for c in corners_rc]
    return corners_xy, width, u_avg
