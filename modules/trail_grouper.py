"""
Trail grouper — fragment splitter, union-find grouper, and polygon fitter.

THE PROBLEM THIS MODULE SOLVES
-------------------------------
SAHI tiled inference runs the YOLO model on each 640x640 tile independently. A
single physical trail that crosses three tiles comes back as three separate mask
blobs — one per tile. Left ungrouped, the pipeline would try to repair the same
trail three times using three overlapping masks, producing a patchy result and
inflating the trail count shown to the user.

This module fuses those fragments back into one coherent detection before repair.

STEP 1 — CROSSING SPLITTER (split_crossing)
--------------------------------------------
Sometimes a tile contains two trails that cross at an angle, and the model returns
them as one merged blob. split_crossing() in modules/crossing_splitter.py finds the
junction point where the two trails meet, excludes a disc of pixels around it to
isolate clean arms, then dilates the arms back to fill the gap. Handles X crossings,
T intersections, and V shapes. The old try_split() function is kept below but is
no longer called -- swap the import to revert.

STEP 1b — PARALLEL TRAIL SPLITTER (_try_split_parallel)
---------------------------------------------------------
Some tiles capture two parallel trails running side by side. The model merges them
into one fat blob (minor axis > ~65px). _try_split_parallel() samples perpendicular
cross-sections in the central 40% of the blob. If 2+ slices show two filled spans
separated by a gap >= 3px, the blob is split along the median gap position into two
independent detections. This fires only on fat blobs that the crossing splitter did
not split (single angle cluster). Applied in both filter paths after split_crossing.

STEP 2 — FILTER (filter_masks / filter_masks_with_props)
---------------------------------------------------------
Each candidate mask is tested against area and aspect ratio thresholds. Tiny blobs
(likely noise) and blobs that are too square (not trail-shaped) are discarded here.
The sky mask, if provided, zeroes out detections in the foreground zone.

STEP 3 — GROUPER (group_detections)
-------------------------------------
Union-find algorithm with a 4-gate check. Two detections are merged into the same
trail if they pass all four tests:
  - Angle gate: their principal axes point in the same direction (within MAX_ANGLE_DEG).
  - Width-ratio gate: their widths are within a factor of 2x (same physical object).
  - Perpendicular gate: they are close in the direction perpendicular to the trail
    axis (not two parallel but separate trails).
  - Tip-to-tip gate: the tip of one detection is close to the tip of the other
    (they are spatially adjacent, not just parallel at a distance).

STEP 4 — POLYGON FITTING (fit_polygon / fit_curved_group)
----------------------------------------------------------
Once a group of fragments is confirmed as one trail, fit_polygon() wraps the combined
pixel footprint in a minimal 4-corner rectangle aligned to the trail's principal axis.
For long trails that visibly curve across the frame (common with wide-angle lenses),
fit_curved_group() splits the trail into 600px-wide strips and fits a separate polygon
to each strip, producing a chain of rectangles that follows the arc.

These polygons are the unit used for CVAT annotation review and for the repair mask
passed to modules/repair.py.

Public API
----------
filter_masks(preds, h, w)
    Run crossing splitter + area filter on raw SAHI predictions.
    Returns a list of uint8 masks (255=trail) that passed all filters.
    Used by detect_trails.detect_frame to build the combined mask.

filter_masks_with_props(preds, h, w, sky_mask=None)
    Like filter_masks but returns (masks, det_list) in one pass.
    masks[i] and det_list[i] correspond to the same detection.
    Used by polymakr.py for full-chain detection → polygon fitting.

thicken_mask(mask, h, w)
    Convert a combined binary pixel mask into fit_polygon rectangles.
    Returns (polygon_list, thick_mask). Used by STC repair to replace the
    tight pixel mask with wider rectangles consistent with CVAT annotations.

group_detections(det_list)
    Union-find grouper with 4-gate check (angle, width-ratio, perp, tip-to-tip).
    Returns list-of-lists of detection indices.

fit_polygon(group_indices, det_list)
    Fit one 4-corner rectangle to a group of detections.
    Returns (corners_xy, width, u_avg).

detection_props(mask)
    Return region properties for one mask, or None if area filter fails.

try_split(mask)
    Crossing splitter: returns [mask] unchanged or [mask_a, mask_b].
"""

import cv2
import numpy as np
import math
import time
from skimage.measure import label as sklabel, regionprops as skregionprops
from modules.crossing_splitter import split_crossing

# ── Constants ─────────────────────────────────────────────────────────────────
# Thresholds used across filter, splitter, grouper, and polygon fitter.
# Changing any of these affects detection coverage — check the run log before tuning.

CONF_THRESH         = 0.30
MAX_ANGLE_DEG       = 7.0
MIN_AREA            = 1000
MIN_ASPECT          = 2.0

SPLIT_AREA_MIN      = 5000
HOUGH_THRESHOLD     = 20
HOUGH_MIN_LINE      = 25
HOUGH_MAX_GAP       = 15
DBSCAN_EPS          = 5.0
DBSCAN_MIN_SAMPLES  = 2
TIP_PAD_PX          = 0   # pixels added to each tip of the fitted polygon; negative trims
MIN_SPLIT_ANGLE_DEG = 10.0
SEAM_MARGIN         = 3.0

_REF_FRAME_PX = 6000 * 4000   # reference resolution for normalizing area thresholds

_CURVED_MIN_XSPAN         = 1500   # px -- group must span this wide to be curved
_CURVED_MIN_ANGLE_SPREAD  = 5.0    # degrees -- min angle variation to call it curved
_CURVED_STRIP_PX          = 600    # target strip width in px
_CURVED_STRIP_OVERLAP     = 0.15   # fraction of strip width shared with neighbors
_CURVED_TIP_PAD           = 120    # px added beyond pixel extent on outer trail tips


# ── Internal helpers ──────────────────────────────────────────────────────────
# Geometry, angle math, and per-pixel helpers used by the public functions.
# None of these write to the log — they are pure computation.

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


def _is_red_trail(mask, img):
    """Return True if pixels under mask are predominantly red (airplane nav light).

    img must be RGB (channel 0 = R).  Thresholds match filter_small_components.
    """
    pixels = img[mask > 0]
    if len(pixels) == 0:
        return False
    r_vals = pixels[:, 0].astype(float)
    top_mask = r_vals >= np.percentile(r_vals, 90)
    top_r = float(r_vals[top_mask].mean())
    top_g = float(pixels[top_mask, 1].mean())
    top_b = float(pixels[top_mask, 2].mean())
    return top_r > 80 and top_r > top_g * 1.4 and top_r > top_b * 1.4


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


# ── Crossing splitter ─────────────────────────────────────────────────────────
# Detects blobs where two trails cross and splits them into two separate masks.
# Uses Hough line detection + DBSCAN angle clustering to find two distinct directions.
# For same-angle blobs (parallel trail fusions), falls back to bimodal perp split.
# Input: one SAHI prediction mask. Output: [mask] unchanged, or [mask_a, mask_b].
# Log field: grouper.try_split_fired counts how many predictions were split.

def _try_bimodal_split(mask):
    """
    Split a blob that fuses two parallel trails by detecting a bimodal pixel
    distribution perpendicular to the trail direction.

    Projects all pixels onto the axis perpendicular to the blob's major axis,
    runs k-means with k=2 on the 1D projection, and splits if the two cluster
    means are well-separated relative to their spread (sep > 1.5x avg std and
    > 20px absolute). Both pieces must pass the elongation filter.

    Returns [mask] if no clean bimodal split is found, [mask_a, mask_b] if split.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return [mask]

    coords = np.stack([ys, xs], axis=1).astype(float)
    cov = np.cov(coords.T)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, np.argmax(vals)]
    u_perp = np.array([-u[1], u[0]])
    centroid = coords.mean(axis=0)
    perp_proj = np.dot(coords - centroid, u_perp)

    p_min, p_max = perp_proj.min(), perp_proj.max()
    c1 = p_min + (p_max - p_min) / 3.0
    c2 = p_min + 2.0 * (p_max - p_min) / 3.0
    for _ in range(30):
        labels = (np.abs(perp_proj - c2) < np.abs(perp_proj - c1)).astype(int)
        c1_n = perp_proj[labels == 0].mean() if (labels == 0).any() else c1
        c2_n = perp_proj[labels == 1].mean() if (labels == 1).any() else c2
        if abs(c1_n - c1) < 0.1 and abs(c2_n - c2) < 0.1:
            break
        c1, c2 = c1_n, c2_n

    sep = abs(c2 - c1)
    std1 = perp_proj[labels == 0].std() if (labels == 0).sum() > 5 else 999.0
    std2 = perp_proj[labels == 1].std() if (labels == 1).sum() > 5 else 999.0
    avg_std = (std1 + std2) / 2.0

    if sep < 1.5 * avg_std or sep < 20.0:
        return [mask]

    coords_int = coords.astype(int)
    mask_a = np.zeros_like(mask)
    mask_b = np.zeros_like(mask)
    mask_a[coords_int[labels == 0, 0], coords_int[labels == 0, 1]] = 255
    mask_b[coords_int[labels == 1, 0], coords_int[labels == 1, 1]] = 255

    if mask_a.max() == 0 or mask_b.max() == 0:
        return [mask]
    if detection_props(mask_a) is None or detection_props(mask_b) is None:
        return [mask]

    return [mask_a, mask_b]


def try_split(mask):
    """
    Check mask for a crossing blob and split it if two distinct trail
    directions are found.

    Returns [mask] unchanged if no crossing is detected, or [mask_a, mask_b]
    if exactly two distinct trail directions are found.

    Adaptive Hough threshold: blobs with area < 20,000px use threshold=10;
    larger blobs use HOUGH_THRESHOLD (20). Smaller crossing blobs don't
    generate enough Hough votes at threshold=20 to fire the split.
    """
    area = int((mask > 0).sum())
    area_scale = (mask.shape[0] * mask.shape[1]) / _REF_FRAME_PX
    if area < SPLIT_AREA_MIN * area_scale:
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
        return _try_bimodal_split(mask)

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
        return _try_bimodal_split(mask)

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

    # If either half would be dropped by the elongation filter, the split created
    # a fat non-trail blob from the overlap zone of two nearby parallel trails.
    # Revert to the original — a genuine crossing always produces two elongated halves.
    if detection_props(mask_a) is None or detection_props(mask_b) is None:
        return [mask]

    return [mask_a, mask_b]


# ── Elongation filter ─────────────────────────────────────────────────────────
# Rejects masks that are too round or too fat to be a trail. Gates: area >= MIN_AREA,
# aspect ratio (major/minor) >= MIN_ASPECT, and minor axis not wider than pixel density.
# Returns a props dict on pass, None on reject. Red nav lights bypass this gate.
# Log field: grouper.failed_elongation counts predictions rejected here.

def detection_props(mask, min_aspect=None, min_area=None):
    """Return region properties for one mask, or None if area/elongation filter fails.

    min_aspect overrides MIN_ASPECT (pass 0.0 to skip the AR gate entirely).
    min_area overrides MIN_AREA (pass 0 to skip the area gate entirely).
    """
    if min_aspect is None:
        min_aspect = MIN_ASPECT
    area_scale = (mask.shape[0] * mask.shape[1]) / _REF_FRAME_PX
    if min_area is None:
        min_area = MIN_AREA * area_scale
    # Crop to bounding box before running skimage regionprops.
    # sklabel/skregionprops scan the entire image array, so running on the
    # full 6000x4000 frame mask is ~100x slower than needed when the trail
    # blob occupies a small fraction of that space.
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    row_lo, row_hi = int(ys.min()), int(ys.max())
    col_lo, col_hi = int(xs.min()), int(xs.max())
    crop = mask[row_lo:row_hi + 1, col_lo:col_hi + 1]
    lbl = sklabel(crop)
    props = skregionprops(lbl)
    if not props:
        return None
    p = max(props, key=lambda x: x.area)
    minor = p.axis_minor_length
    if minor < 1 or p.area < min_area:
        return None
    if p.axis_major_length / minor < min_aspect:
        return None
    pixel_density = p.area / max(p.axis_major_length, 1)
    if minor > 50 * math.sqrt(area_scale) and minor > 1.6 * pixel_density:
        return None
    _, vecs = np.linalg.eigh(p.inertia_tensor)
    u = vecs[:, 0]
    return {
        "centroid": np.array([p.centroid[0] + row_lo, p.centroid[1] + col_lo]),
        "u":        u,
        "minor":    minor,
        "major":    p.axis_major_length,
        "area":     p.area,
        "coords":   np.column_stack((ys, xs)),
    }


def _try_split_parallel(mask):
    """Split a fat SAHI mask into two parallel trail masks when a perpendicular gap exists.

    Some SAHI predictions merge two close parallel trails into one fat blob. Samples
    10 perpendicular cross-sections in the central 40% of the blob. If >= 2 slices
    show two filled spans separated by a gap >= 3px (each span >= 10px), splits all
    blob pixels at the median gap position.

    Returns [mask] unchanged if the blob is not fat enough (minor < 65px at 24MP)
    or no consistent gap is found. Returns [mask_a, mask_b] on a successful split.
    """
    area_scale = (mask.shape[0] * mask.shape[1]) / _REF_FRAME_PX
    fat_threshold = 65.0 * math.sqrt(area_scale)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [mask]

    row_lo, row_hi = int(ys.min()), int(ys.max())
    col_lo, col_hi = int(xs.min()), int(xs.max())
    crop = (mask[row_lo:row_hi + 1, col_lo:col_hi + 1] > 0).astype(np.uint8) * 255
    lbl = sklabel(crop)
    props = skregionprops(lbl)
    if not props:
        return [mask]
    p = max(props, key=lambda x: x.area)
    minor = p.axis_minor_length
    major = p.axis_major_length
    if minor < fat_threshold or major < minor * 2.0:
        return [mask]

    _, vecs = np.linalg.eigh(p.inertia_tensor)
    u = vecs[:, 0]   # unit direction in (row, col) space: u[0]=row, u[1]=col
    cy_full = float(p.centroid[0]) + row_lo
    cx_full = float(p.centroid[1]) + col_lo

    half_span = int(minor) + 8
    sample_ts = np.linspace(-major * 0.20, major * 0.20, 10)

    gap_positions = []
    for t in sample_ts:
        row_c = cy_full + u[0] * t
        col_c = cx_full + u[1] * t
        profile = []
        for s in range(-half_span, half_span + 1):
            r = int(round(row_c - u[1] * s)) - row_lo
            c = int(round(col_c + u[0] * s)) - col_lo
            if 0 <= r < crop.shape[0] and 0 <= c < crop.shape[1]:
                profile.append(int(crop[r, c] > 0))
            else:
                profile.append(0)
        spans = []
        in_s = False
        for j, v in enumerate(profile):
            sc = j - half_span
            if v and not in_s:
                s_start = sc
                in_s = True
            elif not v and in_s:
                spans.append((s_start, sc - 1))
                in_s = False
        if in_s:
            spans.append((s_start, half_span))
        if len(spans) == 2:
            s0_w = spans[0][1] - spans[0][0] + 1
            s1_w = spans[1][1] - spans[1][0] + 1
            g_w = spans[1][0] - spans[0][1] - 1
            if g_w >= 3 and s0_w >= 10 and s1_w >= 10:
                gap_positions.append((spans[0][1] + spans[1][0]) / 2.0)

    if len(gap_positions) < 2:
        return [mask]

    split_off = float(np.median(gap_positions))
    drow = ys.astype(float) - cy_full
    dcol = xs.astype(float) - cx_full
    perp_off = drow * (-u[1]) + dcol * u[0]

    side_a = perp_off < split_off
    side_b = ~side_a
    min_px = int(MIN_AREA * area_scale)
    if int(side_a.sum()) < min_px or int(side_b.sum()) < min_px:
        return [mask]

    mask_a = np.zeros_like(mask)
    mask_b = np.zeros_like(mask)
    mask_a[ys[side_a], xs[side_a]] = 255
    mask_b[ys[side_b], xs[side_b]] = 255
    return [mask_a, mask_b]


# ── Public: filter masks ──────────────────────────────────────────────────────
# Entry points for the full filter pipeline: crossing splitter → elongation filter.
# filter_masks: used by detect_frame (simple combined mask, no polygon fitting).
# filter_masks_with_props: used by detect_frame_polygon (returns props for grouping).
# Both support debug_out to capture per-stage rejection counts for the run log.

def filter_masks(preds, h, w, img=None):
    """
    Run crossing splitter + elongation filter on raw SAHI predictions.

    Returns a list of uint8 masks (255=trail) that passed all filters.
    Caller unions them into one combined mask.

    img (RGB, optional): when provided, components that fail the elongation
    filter are kept anyway if their pixels are predominantly red (nav light).
    """
    passing = []
    for pred in preds:
        m = _pred_to_mask(pred, h, w)
        if m is None:
            continue
        candidates = split_crossing(m)
        expanded = []
        for cm in candidates:
            expanded.extend(_try_split_parallel(cm))
        for cm in expanded:
            if detection_props(cm) is not None:
                passing.append(cm)
            elif img is not None and _is_red_trail(cm, img):
                passing.append(cm)
    return passing


def filter_masks_with_props(preds, h, w, sky_mask=None, img=None, debug_out=None,
                            timing_out=None):
    """
    Like filter_masks, but returns (masks, det_list, edge_candidates) in one pass.

    sky_mask (white=sky, black=foreground), if provided, is AND-ed with each
    prediction mask before the crossing splitter and elongation filter.

    img (RGB, optional): when provided, components that fail the elongation
    filter are kept anyway if their pixels are predominantly red (nav light).

    debug_out (dict, optional): populated with per-stage rejection counts when
    provided. Keys: raw_pred_count, no_mask, sky_zeroed, try_split_fired,
    failed_elongation, kept_as_nav_light, passed, edge_candidate_count.

    edge_candidates: list of dicts {"mask", "u", "bbox"} for components that
    failed elongation but whose bounding box touches any image edge (within 20px).
    These are rescued by the pipeline if 2+ neighboring frames contain a mask
    component with matching slope.
    """
    masks           = []
    det_list        = []
    edge_candidates = []
    _ts = _te = _tec = 0.0
    if debug_out is not None:
        debug_out.update({
            "raw_pred_count":      len(preds),
            "no_mask":             0,
            "sky_zeroed":          0,
            "try_split_fired":     0,
            "crossing_arm_count":  0,
            "failed_elongation":   0,
            "kept_as_nav_light":   0,
            "passed":              0,
            "edge_candidate_count": 0,
        })
    for pred in preds:
        m = _pred_to_mask(pred, h, w)
        if m is None:
            if debug_out is not None:
                debug_out["no_mask"] += 1
            continue
        _pred_conf = float(getattr(getattr(pred, "score", None), "value", 0.0))
        if sky_mask is not None:
            sm = (sky_mask if sky_mask.shape == (h, w)
                  else cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST))
            m = cv2.bitwise_and(m, sm)
            if m.max() == 0:
                if debug_out is not None:
                    debug_out["sky_zeroed"] += 1
                continue
        _t0 = time.perf_counter()
        candidates = split_crossing(m)
        _ts += time.perf_counter() - _t0
        if len(candidates) > 1:
            if debug_out is not None:
                debug_out["try_split_fired"] += 1
                debug_out["crossing_arm_count"] += len(candidates)
        expanded = []
        for cm in candidates:
            expanded.extend(_try_split_parallel(cm))
        candidates = expanded
        for cm in candidates:
            _t0 = time.perf_counter()
            props = detection_props(cm)
            if props is None and img is not None and _is_red_trail(cm, img):
                props = detection_props(cm, min_aspect=0.0)
                if props is not None and debug_out is not None:
                    debug_out["kept_as_nav_light"] += 1
            _te += time.perf_counter() - _t0
            if props is not None:
                props["conf"] = _pred_conf
                masks.append(cm)
                det_list.append(props)
                if debug_out is not None:
                    debug_out["passed"] += 1
            else:
                if debug_out is not None:
                    debug_out["failed_elongation"] += 1
                # Detections clipped by the image boundary look stubby (low AR or
                # small area) but are real trails -- false positives at frame edges
                # are vanishingly rare.  Keep them unconditionally so the grouper
                # and repair pass treat them like any other detection.
                _t0 = time.perf_counter()
                px = np.where(cm > 0)
                if len(px[0]) > 0:
                    by1, by2 = int(px[0].min()), int(px[0].max())
                    bx1, bx2 = int(px[1].min()), int(px[1].max())
                    if by1 <= 19 or by2 >= h - 20 or bx1 <= 19 or bx2 >= w - 20:
                        edge_props = detection_props(cm, min_aspect=0.0, min_area=0)
                        if edge_props is not None:
                            edge_props["conf"] = _pred_conf
                            masks.append(cm)
                            det_list.append(edge_props)
                            if debug_out is not None:
                                debug_out["passed"] += 1
                _tec += time.perf_counter() - _t0
    if timing_out is not None:
        timing_out["try_split_s"] = _ts
        timing_out["elongation_s"] = _te
        timing_out["edge_cand_s"]  = _tec
    return masks, det_list, edge_candidates


# ── Union-find grouper ────────────────────────────────────────────────────────
# Connects detections that belong to the same physical trail using union-find.
# Five gates must ALL pass to merge a pair: angle, width-ratio, perp distance,
# tip-to-tip gap, and tip alignment. Returns list-of-lists of detection indices
# (one per trail). Log field: detect record group_count = number of groups.

def group_detections(det_list):
    """
    Group detections belonging to the same trail using union-find.

    All five gates must pass to connect a pair:
      1. Angle between major axes < MAX_ANGLE_DEG
      2. Width ratio < 3x (prevents grouping thin with fat)
      3. Perpendicular centroid distance < 0.9x max minor
      4. Two-stage gap: if masks overlap → always merge;
         otherwise tip-to-tip distance < 2.5x min minor
      5. Tip alignment: the tip-to-tip vector must be mostly along-trail
         (perpendicular component < 0.5x max minor); prevents staggered
         parallel trails from merging via nearby tips.

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
            if perp > 0.5 * max(di["minor"], dj["minor"]):
                continue

            tree_i = KDTree(di["coords"])
            dists, _ = tree_i.query(dj["coords"], k=1)
            nearest_gap = float(np.min(dists))
            if nearest_gap > 0:
                continue

            union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ── Polygon fitting ───────────────────────────────────────────────────────────
# Fits one tight 4-corner rectangle to each detection group.
# Width is derived from the median minor axis of the group members plus a
# length-scaled bonus. TIP_PAD_PX trims or extends each tip.
# Log field: detect record polygon_count = number of polygons filled onto the mask.

def thicken_mask(mask, h, w):
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
        props = detection_props(comp)
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
    all_dets = [det_list[i] for i in group_indices]

    median_minor = float(np.median([d["minor"] for d in all_dets]))
    pruned = [d for d in all_dets if d["minor"] <= 2.0 * median_minor]
    dets = pruned if pruned else all_dets

    all_coords_combined = np.vstack([d["coords"] for d in all_dets])
    centroid = all_coords_combined.mean(axis=0)

    u_sum = np.zeros(2)
    for d in dets:
        u = d["u"]
        if u_sum.dot(u) < 0:
            u = -u
        u_sum += u * d["area"]
    u_avg = u_sum / np.linalg.norm(u_sum)

    t_centroid = centroid @ u_avg
    tip_pad = TIP_PAD_PX
    t_min = min(float((d["coords"] @ u_avg).min()) for d in all_dets) - tip_pad
    t_max = max(float((d["coords"] @ u_avg).max()) for d in all_dets) + tip_pad
    p_min = centroid + (t_min - t_centroid) * u_avg
    p_max = centroid + (t_max - t_centroid) * u_avg

    trail_length = t_max - t_min
    width = (float(np.median([
        min(d["minor"], d["area"] / max(d["major"], 1))
        for d in dets
    ])) * 1.5) + min(trail_length / 200.0, 40.0)

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


def _group_angle_spread(group_indices, det_list):
    """Return angle spread (degrees) across detections in a group.
    Handles the 0/180 wrap: angles are unwrapped relative to the median."""
    angles = []
    for i in group_indices:
        d = det_list[i]
        a = float(np.degrees(np.arctan2(d["u"][0], d["u"][1]))) % 180
        angles.append(a)
    med = float(np.median(angles))
    unwrapped = []
    for a in angles:
        diff = a - med
        if diff > 90:
            diff -= 180
        elif diff < -90:
            diff += 180
        unwrapped.append(diff)
    return max(unwrapped) - min(unwrapped)


def _fit_poly_from_pixels(coords_rc):
    """PCA rectangle fit for a (N,2) (row,col) pixel array.
    Returns list of (x,y) corner tuples, or None if too few pixels."""
    if len(coords_rc) < 10:
        return None
    cov = np.cov(coords_rc.T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, np.argmax(evals)]
    u_perp = np.array([-u[1], u[0]])
    centroid = coords_rc.mean(axis=0)
    t = coords_rc @ u
    t_centroid = float(centroid @ u)
    t_min, t_max = float(t.min()), float(t.max())
    col_lo = float(coords_rc[:,1].min())
    col_hi = float(coords_rc[:,1].max())
    step = max((col_hi - col_lo) / 15, 1.0)
    widths = []
    c = col_lo
    while c < col_hi:
        rows = coords_rc[(coords_rc[:,1] >= c) & (coords_rc[:,1] < c + step), 0]
        if len(rows) > 2:
            widths.append(float(rows.max() - rows.min()))
        c += step
    half_w = (float(np.median(widths)) if widths else 20.0) * 0.75
    p_min = centroid + (t_min - t_centroid) * u
    p_max = centroid + (t_max - t_centroid) * u
    corners_rc = [
        p_min + half_w * u_perp,
        p_min - half_w * u_perp,
        p_max - half_w * u_perp,
        p_max + half_w * u_perp,
    ]
    return [(int(c[1]), int(c[0])) for c in corners_rc]


def fit_curved_group(group_indices, det_list):
    """Strip-based polygon fit for curved trails.

    Divides all group pixels into N x-column strips with overlap and fits one
    PCA rectangle per strip. Returns a list of corner lists (one per strip),
    each suitable for cv2.fillPoly.

    Called when a group's x-span exceeds _CURVED_MIN_XSPAN AND its angle
    spread exceeds _CURVED_MIN_ANGLE_SPREAD degrees.
    """
    all_coords = np.vstack([det_list[i]["coords"] for i in group_indices])
    x_min = int(all_coords[:,1].min())
    x_max = int(all_coords[:,1].max())
    x_span = x_max - x_min
    n_strips = max(2, round(x_span / _CURVED_STRIP_PX))
    x_lo = x_min - _CURVED_TIP_PAD
    x_hi = x_max + _CURVED_TIP_PAD
    total = x_hi - x_lo
    strip_w = total / n_strips
    overlap = strip_w * _CURVED_STRIP_OVERLAP
    result = []
    for s in range(n_strips):
        sx1 = x_lo + s * strip_w - (overlap if s > 0 else 0)
        sx2 = x_lo + (s + 1) * strip_w + (overlap if s < n_strips - 1 else 0)
        strip = all_coords[(all_coords[:,1] >= sx1) & (all_coords[:,1] < sx2)]
        corners = _fit_poly_from_pixels(strip)
        if corners is not None:
            result.append(corners)
    return result
