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
T intersections, and V shapes. The main SAHI filter path (filter_masks /
filter_masks_with_props) uses split_crossing(). The older try_split() function below
is still called by detect_trails.py's targeted single-tile re-inference pass
(_run_targeted_tile / _run_targeted_tile_rot90).

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
  - Width-ratio gate: their widths are within a factor of 3x (same physical object).
  - Perpendicular gate: they are close in the direction perpendicular to the trail
    axis (not two parallel but separate trails).
  - Contact gate: the two pixel sets must touch — their nearest-neighbor distance is
    exactly 0 (they overlap or abut, not just parallel at a distance).

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
    Returns (masks, det_list, edge_candidates) in one pass.
    masks[i] and det_list[i] correspond to the same detection. edge_candidates is
    currently always an empty list (see the function docstring).
    Used by polymakr.py for full-chain detection → polygon fitting.

thicken_mask(mask, h, w)
    Convert a combined binary pixel mask into fit_polygon rectangles by grouping
    blobs by trail axis and fitting one rectangle per group.
    Returns (polygon_list, thick_mask). Currently used only by diagnostic tools
    (tools/diag_*.py), not by the live repair path.

group_detections(det_list)
    Union-find grouper with 4-gate check (angle, width-ratio, perp, contact).
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
# --- Two independent mask-shape knobs (ratios, resolution-independent) ------
# Both scale off the trail's MEASURED thickness, so changing one does not move
# the other. No pixel constants, so they behave the same at any resolution.
#   THICKNESS (across the trail): mask thickness = MASK_THICKNESS_MULT x the
#     measured trail thickness. 1.2 leaves a 20% margin beyond the trail.
#   LENGTH (along the trail): each tip is pulled inward by TIP_TRIM_FRAC x the
#     measured thickness. 0.25 trims a little off each end; 0 disables.
MASK_THICKNESS_MULT = 1.2    # thickness knob
TIP_TRIM_FRAC       = 0.25   # length knob (trim per tip, x measured thickness)
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
    """Turn one SAHI prediction object into a full-frame uint8 mask.

    pred is a single SAHI/YOLO prediction. h, w are the target frame height
    and width. Returns a (h, w) uint8 array where trail pixels are 255 and the
    rest are 0, or None if the prediction carries no usable segmentation.

    SAHI stores the mask at the tile's own resolution, so if it does not already
    match the requested frame size it is nearest-neighbor resized up to (h, w).
    Nearest-neighbor keeps the mask strictly binary (no interpolated gray edges).

    THIS IS A HOT PATH: it runs once per detection per stage, and a busy frame
    carries 50+ detections, so on a 44MP frame a wasted pass here is tens of
    megabytes of memory traffic multiplied by fifty. It was the single largest
    cost in BOTH of the big detection stages until 2026-08-26. Two rules follow
    from that, and the comments in the body explain why each one matters:
      * take the OUTLINE from the prediction and fill it yourself; never ask
        SAHI for the rendered version
      * allocate once -- no astype-then-multiply, no intermediate copies
    Callers only ever test the result for non-zero, but it is kept at 0/255
    because OpenCV calls downstream expect a conventional binary image.
    """
    if pred.mask is None:
        return None

    # DRAW THE OUTLINE OURSELVES rather than asking SAHI for `.bool_mask`.
    #
    # Despite the name, that property returns FLOAT64. Its decoder allocates
    # `np.zeros([height, width])` with no dtype, fills the polygon into it, then
    # computes `.astype(bool)` and throws the result away without assigning it.
    # On a 44MP frame that is a 354 MB array where 44 MB would do, per call, per
    # detection -- and the property recomputes from scratch on every access, so
    # nothing is cached. Measured 2026-08-26 as the largest single cost in both
    # phantom pruning and polygon fitting.
    #
    # The geometry here is copied from SAHI's own decoder (round, then to int)
    # so the filled pixels are identical; only the container changes, and we
    # fill 255 directly instead of filling 1.0 and scaling afterwards.
    _m = pred.mask
    seg_poly = getattr(_m, "segmentation", None)
    if seg_poly:
        try:
            fh, fw = int(_m.full_shape[0]), int(_m.full_shape[1])
            arr = np.zeros((fh, fw), np.uint8)
            pts = [np.array(p).reshape(-1, 2).round().astype(int) for p in seg_poly]
            cv2.fillPoly(arr, pts, 255)
            if arr.shape != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
            return arr
        except Exception:
            pass      # anything unexpected: fall through to the library's own path

    seg = _m.bool_mask
    if seg is None:
        return None
    # A bool array is one byte per element, so a uint8 VIEW of it costs nothing
    # and already holds 0/1. Scaling that to 0/255 needs ONE full-frame
    # allocation, where astype-then-multiply needs two. This runs once per
    # detection per stage, and on a 44MP frame each avoided pass is 44 MB of
    # memory traffic: with 53 detections that is 2.3 GB of pointless copying per
    # frame. Measured 2026-08-26 as the single largest cost in both phantom
    # pruning and polygon fitting.
    base = (seg.view(np.uint8)
            if seg.dtype == np.bool_ and seg.flags["C_CONTIGUOUS"]
            else seg.astype(np.uint8))
    arr = base * 255
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    return arr


def _pca_aspect(mask):
    """Return the elongation (aspect ratio) of a blob via principal component analysis.

    mask is a binary image. Returns the ratio of the long axis to the short axis
    of the pixel cloud (a long thin trail gives a big number, a round blob gives
    near 1.0). Returns 0.0 if fewer than 5 lit pixels (too few to measure).

    Computes the covariance of the pixel coordinates, takes its eigenvalues
    (the squared spread along each principal axis), and returns sqrt(long)/sqrt(short).
    The 0.5 floor on the denominator avoids divide-by-zero on single-row blobs.

    Note: not referenced elsewhere in the project's current code path.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    cov = np.cov(pts.T)
    ev, _ = np.linalg.eigh(cov)
    return float(np.sqrt(ev[-1]) / max(0.5, np.sqrt(ev[0])))


def _line_angle_deg(L):
    """Return the orientation of a line segment in degrees, folded into [0, 180).

    L is (x1, y1, x2, y2). The result is an undirected angle: a line and the
    same line pointing the other way give the same value (that is the purpose of
    the % 180). Used to cluster Hough lines by direction in the crossing splitter.
    """
    x1, y1, x2, y2 = L
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _circular_mean_angle(angle_list):
    """Circular mean for undirected angles in [0, 180) degrees.

    angle_list is a list of orientations in degrees. Returns their average,
    handling the wrap-around at the 0/180 boundary (a plain numeric average of
    1 deg and 179 deg would give 90 deg, which is wrong; this returns ~0 deg).

    Trick: doubling each angle maps the [0,180) half-circle onto a full [0,360)
    circle, so the standard sin/cos vector-mean works, then the result is halved
    back. Used to get one representative direction per Hough angle cluster.
    """
    rads = np.radians([2.0 * a for a in angle_list])
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.degrees(mean_rad / 2.0) % 180.0)


def _angle_dist(a, b):
    """Smallest angular difference between two undirected angles in [0, 180).

    a, b are orientations in degrees. Returns a value in [0, 90]: because the
    angles are undirected, 10 deg and 170 deg are only 20 deg apart, not 160.
    """
    d = abs(a - b)
    return min(d, 180.0 - d)


def _perp_dist(px, py, L):
    """Perpendicular distance from point (px, py) to the infinite line through L.

    L is (x1, y1, x2, y2) defining the line. Returns the shortest (perpendicular)
    distance from the point to that line, or infinity if L is a degenerate
    zero-length segment. Used in try_split to assign each blob pixel to whichever
    of the two crossing arms it sits closest to.
    """
    x1, y1, x2, y2 = L
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return float("inf")
    # Standard point-to-line formula: |cross product of direction and offset|
    # divided by the line length.
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
    """Return the (x, y) intersection point of two infinite lines, or None if parallel.

    L1 and L2 are each (x1, y1, x2, y2). Returns the crossing point of the two
    lines extended infinitely, or None when they are parallel (denominator near
    zero). Note: this is a pure-geometry helper and is not referenced elsewhere
    in this module's current code path.
    """
    x1, y1, x2, y2 = L1
    x3, y3, x4, y4 = L2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _dbscan_angles(angles, eps, min_samples):
    """Cluster a list of angles by direction using a 1-D DBSCAN over angular distance.

    angles is a list of orientations in degrees. eps is the maximum angular
    distance (degrees) for two angles to be considered neighbors. min_samples is
    the minimum neighborhood size for a point to seed a cluster. Returns
    (labels, n_clusters): labels[i] is the cluster id of angles[i], or -1 for an
    unclustered/noise angle.

    This is a hand-rolled DBSCAN (no sklearn dependency) that uses _angle_dist as
    the metric, so it correctly treats angles near the 0/180 wrap as close. The
    crossing splitter uses it to discover how many distinct trail directions the
    Hough lines fall into (two directions = a crossing).
    """
    n = len(angles)
    labels  = [-1] * n
    visited = [False] * n
    cluster_id = 0

    def nbrs(i):
        """Indices of all angles within eps degrees of angle i (its neighborhood)."""
        return [j for j in range(n) if _angle_dist(angles[i], angles[j]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        hood = nbrs(i)
        if len(hood) < min_samples:
            # Too few neighbors to be a core point; leave as noise (label stays -1).
            continue
        labels[i] = cluster_id
        # Grow the cluster outward from this seed, absorbing any density-reachable
        # angle. seed acts as a work queue of points still to expand from.
        seed = list(hood)
        while seed:
            q = seed.pop()
            if not visited[q]:
                visited[q] = True
                q_hood = nbrs(q)
                if len(q_hood) >= min_samples:
                    # q is itself a core point, so its neighbors join the frontier.
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

    # Find the blob's major axis u via PCA, then u_perp is the across-trail axis.
    # Projecting every pixel onto u_perp collapses the blob to a 1-D distribution;
    # two parallel trails show up as two humps along that axis.
    coords = np.stack([ys, xs], axis=1).astype(float)
    cov = np.cov(coords.T)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, np.argmax(vals)]
    u_perp = np.array([-u[1], u[0]])
    centroid = coords.mean(axis=0)
    perp_proj = np.dot(coords - centroid, u_perp)

    # 1-D k-means with k=2 (Lloyd iterations) on the perpendicular projection.
    # Centers start at the 1/3 and 2/3 points of the spread; loop reassigns each
    # pixel to its nearer center, recomputes centers, and stops once they settle
    # (movement < 0.1px) or after 30 iterations.
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

    # Accept the split only if the two clusters are clearly separated: the gap
    # between centers must exceed both 1.5x their average spread and 20px.
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
        """Return the longest Hough line segment belonging to angle cluster cid.

        That longest segment is used as the representative direction for the arm,
        since a longer line is a more reliable estimate of the trail's axis.
        """
        clines = [lines[i][0] for i, l in enumerate(labels) if l == cid]
        return max(clines, key=lambda L: (L[2] - L[0]) ** 2 + (L[3] - L[1]) ** 2)

    rep1 = _longest(best_c1)
    rep2 = _longest(best_c2)

    # Assign each blob pixel to the closer of the two representative arm lines.
    # Pixels almost equidistant from both lines (within SEAM_MARGIN px of the
    # seam) are dropped, leaving a thin gap that keeps the two arms cleanly apart.
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
        """Largest minor-axis (across-trail width) among the blobs in mask m."""
        rp = skregionprops(sklabel(m))
        return max(p.axis_minor_length for p in rp) if rp else 0

    # A real crossing arm is thin. If either half is wider than 50px the "split"
    # actually carved the fat overlap of two parallel trails, so reject it.
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
    # Area gate: drop blobs that are too tiny to be a real trail.
    if minor < 1 or p.area < min_area:
        return None
    # Aspect gate: drop blobs that are too round (not elongated like a trail).
    if p.axis_major_length / minor < min_aspect:
        return None
    # Fat-blob gate: a true trail's pixels pack tightly along its length, so its
    # area-per-length (pixel_density) is close to its width. A blob that is both
    # absolutely wide (minor > 50px, scaled by resolution) AND much wider than its
    # density implies is a fat clump, not a trail. Reject it.
    pixel_density = p.area / max(p.axis_major_length, 1)
    if minor > 50 * math.sqrt(area_scale) and minor > 1.6 * pixel_density:
        return None
    # Trail direction u from the inertia tensor's smallest-eigenvalue eigenvector
    # (the axis of least spread is along the trail's length).
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


def _try_split_parallel(mask, frame_px=None):
    """Split a fat SAHI mask into two parallel trail masks when a perpendicular gap exists.

    Some SAHI predictions merge two close parallel trails into one fat blob. Samples
    10 perpendicular cross-sections in the central 40% of the blob. If >= 2 slices
    show two filled spans separated by a gap >= 3px (each span >= 10px), splits all
    blob pixels at the median gap position.

    frame_px: total pixels of the FULL frame, used to normalise the fat-blob
    threshold. Pass it when `mask` is a cropped region so the threshold matches the
    full-frame behaviour. Defaults to the mask's own size (full-frame call).

    Returns [mask] unchanged if the blob is not fat enough (minor < 65px at 24MP)
    or no consistent gap is found. Returns [mask_a, mask_b] on a successful split.
    """
    area_scale = (frame_px if frame_px else (mask.shape[0] * mask.shape[1])) / _REF_FRAME_PX
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

    # half_span = how far to look on each side when sampling across the trail.
    # sample_ts = 10 sampling positions along the central 40% of the blob length
    # (from -20% to +20% of major); the ends are skipped where trails taper.
    half_span = int(minor) + 8
    sample_ts = np.linspace(-major * 0.20, major * 0.20, 10)

    gap_positions = []
    for t in sample_ts:
        # Walk a line perpendicular to the trail at this position along its length
        # and record a 0/1 profile of which cross-section samples land on the blob.
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
        # Collapse the 0/1 profile into runs of consecutive lit samples ("spans").
        spans = []
        in_s = False
        for j, v in enumerate(profile):
            sc = j - half_span   # signed offset from the trail center
            if v and not in_s:
                s_start = sc
                in_s = True
            elif not v and in_s:
                spans.append((s_start, sc - 1))
                in_s = False
        if in_s:
            spans.append((s_start, half_span))
        # Exactly two lit spans with a clear empty gap between them (gap >= 3px,
        # each span >= 10px) means two parallel trails here. Record the gap's
        # midpoint as a candidate split position.
        if len(spans) == 2:
            s0_w = spans[0][1] - spans[0][0] + 1
            s1_w = spans[1][1] - spans[1][0] + 1
            g_w = spans[1][0] - spans[0][1] - 1
            if g_w >= 3 and s0_w >= 10 and s1_w >= 10:
                gap_positions.append((spans[0][1] + spans[1][0]) / 2.0)

    # Need the gap to show up in at least 2 of the 10 slices to trust it.
    if len(gap_positions) < 2:
        return [mask]

    # Split every blob pixel at the median gap offset, measured perpendicular to
    # the trail. perp_off is each pixel's signed across-trail distance from center.
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

    edge_candidates: currently always returned as an empty list (kept for
    signature/caller compatibility). Edge-touching components that fail the normal
    elongation filter are instead rescued in-line right here: re-tested with the
    aspect and area gates disabled (min_aspect=0.0, min_area=0) and, on pass, added
    directly to masks and det_list as ordinary detections.
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
# Four gates must ALL pass to merge a pair: angle, width-ratio, perpendicular
# offset, and contact (the two pixel sets must touch — nearest-neighbor distance
# of 0). Returns list-of-lists of detection indices (one per trail). Log field:
# detect record group_count = number of groups.

def group_detections(det_list):
    """
    Group detections belonging to the same trail using union-find.

    det_list is the list of detection-property dicts produced by detection_props
    (each has "u" direction, "centroid", "minor" width, and "coords" pixels).
    Returns a list-of-lists of detection indices, one inner list per trail group.

    WHAT THE CODE ACTUALLY CHECKS (read from the loop below). Two detections are
    merged only if they clear all FOUR gates in order:
      1. Angle: the angle between the two trail axes < MAX_ANGLE_DEG.
      2. Width ratio: wider minor / narrower minor <= 3x (keeps a thin trail from
         merging with a fat unrelated blob).
      3. Perpendicular offset: the across-trail component of the centroid-to-
         centroid vector <= 0.5x the larger minor (rejects parallel-but-separate
         trails sitting side by side).
      4. Contact: the two pixel sets must touch — nearest-neighbor distance must
         be exactly 0. Any positive gap rejects the merge.

    NOTE: the older five-gate description above the function (two-stage tip-to-tip
    gap + tip alignment, and the 0.9x figure) is stale; the four gates listed here
    are what the current code enforces.
    """
    from scipy.spatial import KDTree

    n = len(det_list)
    parent = list(range(n))   # union-find: parent[i] points toward i's group root

    def find(x):
        """Return the group-root index for detection x, with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        """Merge the groups containing detections a and b into one."""
        parent[find(a)] = find(b)

    # Test every unordered pair (i, j). A pair must clear ALL gates below to merge;
    # the first failing gate `continue`s and the pair is left in separate groups.
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = det_list[i], det_list[j]

            # Gate 1 — angle: the two trail axes must point the same way. abs()
            # makes the comparison direction-agnostic; clamp to 1.0 guards arccos.
            cos_sim = min(abs(float(np.dot(di["u"], dj["u"]))), 1.0)
            adiff = min(np.degrees(np.arccos(cos_sim)),
                        180.0 - np.degrees(np.arccos(cos_sim)))
            if adiff > MAX_ANGLE_DEG:
                continue

            # Gate 2 — width ratio: widths within 3x, so a thin trail is not
            # merged with a fat unrelated blob.
            if max(di["minor"], dj["minor"]) / max(min(di["minor"], dj["minor"]), 1) > 3.0:
                continue

            # Gate 3 — perpendicular offset: decompose the centroid-to-centroid
            # vector into along-trail and across-trail parts; reject if the two
            # detections are offset sideways (parallel but separate trails).
            u_ref = di["u"] if np.dot(di["u"], dj["u"]) > 0 else -di["u"]
            diff = dj["centroid"] - di["centroid"]
            along = float(np.dot(diff, u_ref))
            perp = float(np.sqrt(max(float(np.dot(diff, diff)) - along ** 2, 0.0)))
            if perp > 0.5 * max(di["minor"], dj["minor"]):
                continue

            # Gate 4 — contact: the two pixel sets must actually touch. Nearest
            # neighbor distance of 0 means they overlap/abut; any gap rejects.
            tree_i = KDTree(di["coords"])
            dists, _ = tree_i.query(dj["coords"], k=1)
            nearest_gap = float(np.min(dists))
            if nearest_gap > 0:
                continue

            union(i, j)

    # Collect detections sharing a root into one list per group.
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ── Polygon fitting ───────────────────────────────────────────────────────────
# Fits one tight 4-corner rectangle to each detection group.
# Width is derived from the median minor axis of the group members plus a
# length-scaled bonus. TIP_TRIM_FRAC pulls each tip inward (length knob).
# Log field: detect record polygon_count = number of polygons filled onto the mask.

def thicken_mask(mask, h, w):
    """
    Convert a combined binary pixel mask into fit_polygon rectangles.

    Finds connected blobs, groups them by trail axis (same gates as
    group_detections), fits one rectangle per group, and returns
    (polygon_list, thick_mask).

    Blobs that fail the elongation filter or coverage check are kept as
    raw pixels in thick_mask so no detected trail is dropped.

    NOTE: not called by the live pipeline (the active detection path
    detect_frame_polygon goes filter_masks_with_props → group_detections →
    fit_polygon/fit_curved_group directly). This function is referenced only by
    the diagnostic scripts under tools/diag_*.py.
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

    # union_det = all detection pixels combined, used to measure how much of each
    # fitted rectangle actually sits over real detected pixels.
    union_det = np.zeros((h, w), dtype=np.uint8)
    for det in det_list:
        union_det[det["coords"][:, 0], det["coords"][:, 1]] = 255

    _MIN_COV = 0.30   # rectangle must overlap detected pixels by at least 30%
    for grp in groups:
        corners, _, _ = fit_polygon(grp, det_list)
        pts    = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        poly_m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_m, [pts], 255)
        area   = max(int(poly_m.sum()) // 255, 1)
        olap   = int(cv2.bitwise_and(poly_m, union_det).sum()) // 255
        if olap / area >= _MIN_COV:
            # Rectangle is a good fit: use the clean polygon.
            thick = np.maximum(thick, poly_m)
            polys.append(corners)
        else:
            # Rectangle is mostly empty (e.g. a sharply curved trail the straight
            # fit ballooned over): fall back to painting the raw detected pixels.
            for idx in grp:
                d = det_list[idx]
                thick[d["coords"][:, 0], d["coords"][:, 1]] = 255

    return polys, thick


def fit_polygon(group_indices, det_list):
    """
    Fit one 4-corner rectangle to a group of detections.

    group_indices are the indices (into det_list) of the fragments that
    group_detections decided belong to the same trail. det_list is the full
    detection-property list. Returns (corners_xy, width, u_avg):
      - corners_xy: four (x, y) integer tuples for cv2.polylines / cv2.fillPoly
      - width: the rectangle's thickness in pixels
      - u_avg: the averaged trail-direction unit vector (row, col)

    The rectangle is aligned to the trail's averaged principal axis, spans the
    full pixel extent along that axis (minus a small tip trim), and is sized
    across the trail by the measured thickness times the MASK_THICKNESS_MULT knob.
    """
    all_dets = [det_list[i] for i in group_indices]

    # Drop unusually fat fragments (minor > 2x the group median) when averaging
    # the direction and thickness, so one blobby member cannot bias the fit.
    # The full set (all_dets) is still used for the along-trail extent below.
    median_minor = float(np.median([d["minor"] for d in all_dets]))
    pruned = [d for d in all_dets if d["minor"] <= 2.0 * median_minor]
    dets = pruned if pruned else all_dets

    all_coords_combined = np.vstack([d["coords"] for d in all_dets])
    centroid = all_coords_combined.mean(axis=0)

    # Area-weighted average direction. Flip each member's u to point the same way
    # as the running sum first (axes are undirected, so +u and -u are equivalent).
    u_sum = np.zeros(2)
    for d in dets:
        u = d["u"]
        if u_sum.dot(u) < 0:
            u = -u
        u_sum += u * d["area"]
    u_avg = u_sum / np.linalg.norm(u_sum)

    # Along-trail extent: project all pixels onto u_avg and take the min/max,
    # so the rectangle reaches the true tips of the longest fragment in the group.
    t_centroid = centroid @ u_avg
    raw_min = min(float((d["coords"] @ u_avg).min()) for d in all_dets)
    raw_max = max(float((d["coords"] @ u_avg).max()) for d in all_dets)
    trail_length = raw_max - raw_min

    # Measured trail thickness (across the trail), before any inflation.
    measured_thick = float(np.median([
        min(d["minor"], d["area"] / max(d["major"], 1))
        for d in dets
    ]))

    # THICKNESS knob: mask thickness = MASK_THICKNESS_MULT x measured thickness,
    # plus a small length-scaled margin.
    width = measured_thick * MASK_THICKNESS_MULT + min(trail_length / 200.0, 20.0)

    # LENGTH knob: pull each tip inward by TIP_TRIM_FRAC x measured thickness.
    # Short-trail guard: never trim more than 0.25x the trail's own length per
    # tip, so a stubby trail cannot be eaten into an inverted polygon.
    tip_trim = min(TIP_TRIM_FRAC * measured_thick, 0.25 * trail_length)
    t_min = raw_min + tip_trim
    t_max = raw_max - tip_trim
    p_min = centroid + (t_min - t_centroid) * u_avg
    p_max = centroid + (t_max - t_centroid) * u_avg

    # Build the four corners: from each trimmed tip (p_min, p_max) step half the
    # width out to each side along u_perp (the across-trail axis). corners are in
    # (row, col); swap to (x, y) = (col, row) for OpenCV at the end.
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
    # Thickness measured PERPENDICULAR to this strip's own direction (u_perp),
    # NOT as the vertical row-span the old code used. Row-span over-measures any
    # sloped or curved trail because it includes the trail's own rise across the
    # slice (a steep diagonal looked far thicker than it really is). The
    # perpendicular spread is the true across-the-trail width at any angle, the
    # same honest measure the straight fitter uses. A robust 2.5-97.5 percentile
    # band ignores a few stray pixels. Scaled by the shared thickness knob (half
    # here, since half_w is a half-width).
    perp = (coords_rc - centroid) @ u_perp
    thickness = float(np.percentile(perp, 97.5) - np.percentile(perp, 2.5))
    half_w = (thickness if thickness > 0 else 20.0) * (MASK_THICKNESS_MULT / 2.0)
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
    # Split along the x (column) axis into ~_CURVED_STRIP_PX-wide strips, at least 2.
    # The trail tips are padded out by _CURVED_TIP_PAD so the end rectangles fully
    # cover the trail ends rather than stopping at the last detected pixel.
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
        # Each strip is widened by `overlap` into its neighbors (except at the two
        # outer ends) so adjacent rectangles overlap and leave no gap on the curve.
        sx1 = x_lo + s * strip_w - (overlap if s > 0 else 0)
        sx2 = x_lo + (s + 1) * strip_w + (overlap if s < n_strips - 1 else 0)
        strip = all_coords[(all_coords[:,1] >= sx1) & (all_coords[:,1] < sx2)]
        corners = _fit_poly_from_pixels(strip)
        if corners is not None:
            result.append(corners)
    return result
