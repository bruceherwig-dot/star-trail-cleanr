"""
Crossing splitter -- spine + tips model for X, T, Y, V, and multi-crossing blobs.

WHERE THIS FITS IN THE APP
--------------------------
Star Trail CleanR detects airplane/satellite trails with a YOLO+SAHI model.
When two (or more) trails cross, the detector often returns the crossing as ONE
fat blob mask instead of separate trails. A single blob is hard to repair well,
because the "Star Bridge" repair borrows clean sky along each trail's own motion
direction, and a crossing has two directions. This module takes such a blob and
splits it back into the individual trails (a long "spine" plus short crossing
"tips") so each piece can be repaired with its own motion estimate. If a blob
isn't actually a crossing, it's returned unchanged.

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
# The two px-AREA thresholds below (_SPLIT_AREA_MIN, _MIN_AREA) are quoted "at
# ref resolution" (a 6000x4000 = 24MP frame). At runtime only those two are
# scaled by the actual frame size (see `area_scale` in `split_crossing`) so the
# same physical trail behaves the same on a small JPEG and a full TIFF. The
# Hough px thresholds (_HOUGH_MIN_LINE, _HOUGH_MAX_GAP) and the dimensionless
# ratio/angle/percentile constants are NOT scaled.

_SPLIT_AREA_MIN     = 5000    # px (at ref resolution) -- blobs smaller skip
_ELONGATION_THRESH  = 4.5     # ratio -- blobs more elongated skip (already single trail)
_HOUGH_THRESHOLD    = 20      # votes
_HOUGH_MIN_LINE     = 25      # px -- minimum Hough line length
_HOUGH_MAX_GAP      = 15      # px -- maximum gap in a Hough line
_DBSCAN_EPS         = 5.0     # deg -- angular neighbourhood for DBSCAN
_DBSCAN_MIN_SAMPLES = 2       # minimum lines per DBSCAN cluster
_MIN_SPLIT_ANGLE    = 15.0    # deg -- below this, all clusters are near-parallel
_MIN_SECOND_DIR_FRAC = 0.15   # a second direction only counts as crossing evidence
                              # when it carries >= this fraction of the TOTAL Hough
                              # line length. A dashed/dotted trail band manufactures
                              # fake off-angle traces (dot-to-dot diagonals, the
                              # frame-edge clip), but those carry only 1-2% of the
                              # evidence; a real crossing's second trail carries
                              # ~40%+ (143A8819: 57/43 split vs GoPro dashed trails
                              # 93-97% one-direction). Measured 2026-06-11.
_MIN_AREA           = 1000    # px (at ref resolution) -- minimum area for a valid output
_MIN_ASPECT         = 2.0     # ratio -- minimum major/minor for valid spine
_SPINE_BAND_FACTOR  = 0.6     # spine half-width = measured_width * this factor
_WIDTH_PERCENTILE   = 20      # percentile of cross-section widths for single-trail estimate
_N_WIDTH_SAMPLES    = 20      # number of cross-section samples along the spine
_END_FRACTION       = 0.15    # px fraction -- UNUSED legacy constant from the old end-pixel width approach; width is now measured by binning the whole spine in _measure_spine_width
_REF_FRAME_PX       = 6000 * 4000  # reference resolution for normalizing area thresholds


# -- Geometry helpers ----------------------------------------------------------

def _line_angle_deg(L):
    """Angle of a line segment in degrees, folded into the range [0, 180).

    A line has no direction (a segment pointing up-left is the same line as one
    pointing down-right), so the angle is taken modulo 180 to treat those as
    equal. Used to group Hough line segments by orientation.

    L: a 4-tuple/list (x1, y1, x2, y2) of the segment's two endpoints.
    Returns the orientation angle in degrees, 0 <= angle < 180.
    """
    x1, y1, x2, y2 = L
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _circular_mean_angle(angle_list):
    """Average a set of orientation angles, handling the wraparound at 0/180.

    A plain numeric average is wrong for angles: e.g. the mean of 1 deg and
    179 deg should be 0 deg (they almost coincide), not 90 deg. This doubles
    each angle so the [0,180) orientation space maps onto a full [0,360) circle,
    averages the unit vectors there, then halves back. Used to get one
    representative angle per DBSCAN cluster of Hough lines.

    angle_list: a list of angles in degrees, each in [0, 180).
    Returns the circular-mean angle in degrees, in [0, 180).
    """
    rads = np.radians([2.0 * a for a in angle_list])
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.degrees(mean_rad / 2.0) % 180.0)


def _angle_dist(a, b):
    """Smallest angle between two orientations, in [0, 90].

    Because orientations wrap at 180, the distance between e.g. 10 deg and
    170 deg is 20 deg, not 160 deg. The result caps at 90 (perpendicular is the
    most two undirected lines can differ). Used to decide whether two Hough
    clusters point in genuinely different directions (a crossing) and as the
    neighbourhood test inside the DBSCAN clustering.

    a, b: angles in degrees, each in [0, 180).
    Returns the wrap-aware angular separation in degrees, in [0, 90].
    """
    d = abs(a - b)
    return min(d, 180.0 - d)


def _line_length(L):
    """Euclidean length of a line segment.

    L: a 4-tuple/list (x1, y1, x2, y2) of the segment's two endpoints.
    Returns the straight-line distance between the endpoints, in pixels.
    Used to weight each angle cluster by how much trail length it represents.
    """
    x1, y1, x2, y2 = L
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _dbscan_angles(angles, eps, min_samples):
    """Group a list of orientation angles into clusters using DBSCAN.

    This is a hand-rolled, from-scratch DBSCAN that runs on 1-D angle values
    using the wrap-aware ``_angle_dist`` as its distance metric (so it correctly
    treats angles near 0 and near 180 as close). It exists to find how many
    distinct trail directions are present in a blob: each cluster is one
    candidate trail orientation.

    angles: list of angles in degrees, each in [0, 180).
    eps: angular neighbourhood radius in degrees -- two angles within eps are
        considered neighbours.
    min_samples: minimum neighbours (including self) an angle needs to be a
        "core" point that can grow a cluster.

    Returns (labels, cluster_count) where ``labels[i]`` is the cluster id for
    angle i (or -1 for noise/unclustered), and ``cluster_count`` is the number
    of clusters formed.
    """
    n = len(angles)
    labels = [-1] * n        # -1 means "not yet assigned to any cluster" (noise)
    visited = [False] * n
    cluster_id = 0

    # Neighbours of angle i: every angle within eps degrees of it (incl. itself).
    def nbrs(i):
        return [j for j in range(n) if _angle_dist(angles[i], angles[j]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        hood = nbrs(i)
        # Too few neighbours -> not a core point; leave as noise for now (it may
        # still be pulled into a cluster later as a border point).
        if len(hood) < min_samples:
            continue
        # Start a new cluster from this core point and flood-fill outward.
        labels[i] = cluster_id
        seed = list(hood)
        while seed:
            q = seed.pop()
            if not visited[q]:
                visited[q] = True
                q_hood = nbrs(q)
                # Only core points expand the cluster further.
                if len(q_hood) >= min_samples:
                    seed.extend(q_hood)
            # Absorb q as a border point if it isn't already claimed.
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

    The goal is to recover how wide ONE trail is, even though the blob bulges
    where trails overlap. Picking a low percentile (rather than the mean or max)
    is the trick: most cross-sections contain only the spine and are narrow, so
    the 20th percentile lands on a spine-only section and ignores the fat
    junction zone.

    xs_f, ys_f: float pixel coordinates of the blob (currently unused inside the
        body -- the measurement works entirely from the projected ``along``/
        ``perp`` arrays -- but kept in the signature for clarity/symmetry).
    along: each pixel's distance projected along the spine direction.
    perp: each pixel's signed perpendicular distance from the spine centerline.
    n_samples: number of bins to slice the spine into along its length.
    pctl: which percentile of the per-bin widths to return (low = spine-only).

    Returns the estimated single-trail width in pixels.
    """
    t_min, t_max = float(along.min()), float(along.max())
    t_span = t_max - t_min
    # Spine too short to slice meaningfully -> fall back to the full perp extent.
    if t_span < 5:
        return float(perp.max() - perp.min())

    edges = np.linspace(t_min, t_max, n_samples + 1)
    widths = []
    for i in range(n_samples):
        # Pixels whose along-position falls in this bin.
        sel = (along >= edges[i]) & (along < edges[i + 1])
        if sel.sum() < 5:
            continue  # too few pixels in this slice to trust its width
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
    # Scale all px thresholds to this frame's resolution relative to the 24MP
    # reference. frame_px lets the caller pass the FULL-frame size when `mask`
    # is only a crop, so a cropped call behaves like a full-frame call.
    area_scale = (frame_px if frame_px else (h * w)) / _REF_FRAME_PX
    min_px = int(_MIN_AREA * area_scale)
    if area < _SPLIT_AREA_MIN * area_scale:
        return [mask]  # too small to bother splitting

    # -- Elongation trigger (dimensionless) ------------------------------------
    # Fit the tightest rotated rectangle around the blob. A lone trail is long
    # and thin (high major/minor ratio); a crossing blob is squat. Only squat
    # blobs proceed -- thin ones are already a single trail and need no split.
    blob_u8 = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(blob_u8 > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    major_len = max(rect[1])      # rect[1] is (width, height) of the box
    minor_len = min(rect[1])
    if minor_len < 1:
        return [mask]  # degenerate (zero-thickness) box, can't form a ratio
    elongation = major_len / minor_len
    if elongation > _ELONGATION_THRESH:
        return [mask]  # already a single thin trail

    # -- Hough lines on tight crop ---------------------------------------------
    # Crop to the blob's bounding box first so Hough only sees blob pixels and
    # runs fast. A lower vote threshold is used for small blobs (fewer pixels
    # means fewer votes are available, so 20 votes may be unreachable).
    row_lo, row_hi = int(ys.min()), int(ys.max())
    col_lo, col_hi = int(xs.min()), int(xs.max())
    blob_crop = blob_u8[row_lo:row_hi + 1, col_lo:col_hi + 1]
    local_threshold = 10 if area < 20000 else _HOUGH_THRESHOLD
    lines_crop = cv2.HoughLinesP(blob_crop, 1, np.pi / 180,
                                 threshold=local_threshold,
                                 minLineLength=_HOUGH_MIN_LINE,
                                 maxLineGap=_HOUGH_MAX_GAP)
    if lines_crop is None or len(lines_crop) < 2:
        return [mask]  # need at least 2 line segments to detect a crossing

    # Hough returned coords relative to the crop; shift them back to full-frame
    # coordinates by adding the crop's top-left corner offset.
    lines = [[[L[0][0] + col_lo, L[0][1] + row_lo,
               L[0][2] + col_lo, L[0][3] + row_lo]]
             for L in lines_crop]

    # -- Angle clustering ------------------------------------------------------
    # Group the line segments by orientation. Each cluster is one candidate
    # trail direction; a true crossing produces 2+ clusters at different angles.
    angles = [_line_angle_deg(L[0]) for L in lines]
    labels, _ = _dbscan_angles(angles, _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES)
    unique = sorted(set(l for l in labels if l >= 0))  # drop noise (-1)
    if not unique:
        return [mask]  # no coherent direction found

    # One representative angle per cluster.
    cluster_means = {cid: _circular_mean_angle(
        [angles[i] for i, l in enumerate(labels) if l == cid]
    ) for cid in unique}

    if len(unique) < 2:
        return [mask]  # single direction, not a crossing

    # -- Pick spine direction: cluster with the most total Hough line length ----
    # The spine is the dominant (longest) trail. "Most total line length" is a
    # better proxy for the dominant trail than "most segments", because a long
    # straight trail yields a few long segments while noise yields many short
    # ones.
    cluster_lengths = {}
    for cid in unique:
        total = sum(_line_length(lines[i][0])
                    for i, l in enumerate(labels) if l == cid)
        cluster_lengths[cid] = total

    spine_cid = max(unique, key=lambda c: cluster_lengths[c])
    spine_angle = cluster_means[spine_cid]

    # -- Crossing-evidence gate: a WEIGHTY second direction ---------------------
    # A crossing means another trail runs through the spine: it must sit at a
    # real angle to the spine (>= _MIN_SPLIT_ANGLE) AND carry real line evidence
    # (>= _MIN_SECOND_DIR_FRAC of the total). Dashed/dotted trail bands produce
    # off-angle clusters from dot-to-dot traces and the frame-edge clip, but
    # those carry only 1-2% of the evidence -- a real crossing trail carries
    # ~40%+. Without the weight test, a handful of noise traces split (or
    # rescued) single dashed trails (GoPro frames 77/365/536/370).
    total_len = sum(cluster_lengths.values())
    has_weighty_second = any(
        cid != spine_cid
        and _angle_dist(cluster_means[cid], spine_angle) >= _MIN_SPLIT_ANGLE
        and cluster_lengths[cid] >= _MIN_SECOND_DIR_FRAC * total_len
        for cid in unique)
    if not has_weighty_second:
        return [mask]  # no real second direction, not a crossing

    # -- Project all blob pixels onto spine coordinate system ------------------
    # Build a local axis system aligned to the spine: "along" runs down the
    # spine, "perp" runs across it. Every blob pixel gets an (along, perp)
    # coordinate measured from the blob centroid. This makes the spine a
    # horizontal band (small |perp|) and the crossing arms stick out (large
    # |perp|), which is what the band-split below relies on.
    xs_f = xs.astype(float)
    ys_f = ys.astype(float)
    cx_blob = float(xs_f.mean())
    cy_blob = float(ys_f.mean())

    spine_rad = np.radians(spine_angle)
    along_dx, along_dy = np.cos(spine_rad), np.sin(spine_rad)   # unit vector along spine
    perp_dx, perp_dy = -np.sin(spine_rad), np.cos(spine_rad)    # unit vector across spine

    # along = projection onto spine direction (how far along the spine)
    along = (xs_f - cx_blob) * along_dx + (ys_f - cy_blob) * along_dy
    # perp = perpendicular distance from spine centerline (signed)
    perp = (xs_f - cx_blob) * perp_dx + (ys_f - cy_blob) * perp_dy

    # -- Measure single-trail width (robust, using low percentile) -------------
    spine_width = _measure_spine_width(
        xs_f, ys_f, along, perp,
        n_samples=_N_WIDTH_SAMPLES, pctl=_WIDTH_PERCENTILE)

    if spine_width < 3:
        return [mask]  # implausibly thin measurement, don't trust the split

    # Find the spine's center offset by sampling per-bin median perpendicular
    # positions along the spine. Each bin contributes one center estimate;
    # taking the overall median is robust to both X and T crossings. The old
    # end-pixel approach failed on T-shapes because one "end" IS the junction,
    # pulling the center way off.
    t_min, t_max = float(along.min()), float(along.max())
    t_span = t_max - t_min      # (computed for symmetry with _measure_spine_width; not gated on here)
    edges = np.linspace(t_min, t_max, _N_WIDTH_SAMPLES + 1)
    bin_centers = []
    for i in range(_N_WIDTH_SAMPLES):
        sel = (along >= edges[i]) & (along < edges[i + 1])
        if sel.sum() < 5:
            continue
        # Median perp of this slice = where the spine sits in this slice.
        bin_centers.append(float(np.median(perp[sel])))
    # Median of the per-slice centers = the spine's overall perp offset.
    p_center = float(np.median(bin_centers)) if bin_centers else 0.0

    # -- Build spine and tip masks ---------------------------------------------
    # Spine = every blob pixel within band_half of the spine centerline. The 0.6
    # factor keeps the band a bit narrower than the full measured width so the
    # arms separate cleanly while still capturing the junction overlap zone.
    band_half = spine_width * _SPINE_BAND_FACTOR
    spine_sel = np.abs(perp - p_center) <= band_half

    # Paint the selected pixels into a full-size mask (255 = spine).
    spine_mask = np.zeros_like(mask)
    spine_mask[ys[spine_sel], xs[spine_sel]] = 255

    # Validate spine is elongated. If the chosen band came out blobby rather
    # than trail-shaped, the direction estimate was wrong -- abandon the split
    # and return the original blob untouched. Uses the largest connected
    # component of the spine mask (the band can fragment).
    sp_rp = skregionprops(sklabel(spine_mask))
    if sp_rp:
        sp = max(sp_rp, key=lambda x: x.area)
        if sp.axis_minor_length > 0:
            sp_elong = sp.axis_major_length / sp.axis_minor_length
            if sp_elong < _MIN_ASPECT:
                return [mask]  # spine isn't trail-shaped, bail
    else:
        return [mask]  # spine mask is empty, bail

    # Tip masks: every blob pixel NOT in the spine band. These are the crossing
    # arms. Split them into separate connected components so each arm becomes
    # its own tip mask (repaired independently downstream).
    tip_all = np.zeros_like(mask)
    tip_sel = ~spine_sel
    if tip_sel.sum() < min_px:
        return [mask]  # nothing significant outside the spine
    tip_all[ys[tip_sel], xs[tip_sel]] = 255

    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        tip_all, connectivity=8)

    # cc id 0 is the background, so iterate from 1. Drop sub-threshold
    # fragments; only arms with enough area become real tips.
    tips = []
    for cc_id in range(1, n_cc):
        cc_area = cc_stats[cc_id, cv2.CC_STAT_AREA]
        if cc_area < min_px:
            continue
        tip_mask = np.zeros_like(mask)
        tip_mask[cc_labels == cc_id] = 255
        tips.append(tip_mask)

    if not tips:
        return [mask]  # no valid tips -> treat as a single trail after all

    # -- Verify full coverage --------------------------------------------------
    # Every pixel of the original blob must end up in the spine or a tip, so the
    # split loses no detected trail pixels (union(spine + tips) == blob). The
    # only pixels that can leak out are the sub-threshold tip fragments dropped
    # above; fold them back into the spine.
    covered = spine_mask.copy()
    for t in tips:
        covered = cv2.bitwise_or(covered, t)          # union of spine + all tips
    # uncovered = blob pixels not in any output mask (blob AND NOT covered).
    uncovered = cv2.bitwise_and(blob_u8, cv2.bitwise_not(covered))
    if uncovered.any():
        # Reassign uncovered pixels to spine (they are small sub-threshold fragments)
        spine_mask = cv2.bitwise_or(spine_mask, uncovered)

    # -- Post-split sanity: the pieces must actually point different ways ------
    # A wide DOTTED trail band can manufacture a fake second Hough direction:
    # short diagonal traces from a dot in one row to a dot in the next, or the
    # straight artificial edge where the frame border clips the mask. That fake
    # cluster passes the entry angle gate and the band-split then carves a
    # single trail into stacked near-parallel strips (GoPro frames 77/365/536,
    # known_problems false_crossing_split entries -- four overlapping polygons
    # on one dotted trail). A real crossing produces pieces whose own long-axis
    # directions differ by at least the entry gate's angle; if every piece
    # points the same way, the "crossing" was noise -- cancel the split and
    # return the blob whole.
    piece_angles = []
    for pm in [spine_mask] + tips:
        pys, pxs = np.where(pm > 0)
        if len(pxs) < min_px:
            continue
        prect = cv2.minAreaRect(
            np.column_stack([pxs, pys]).astype(np.float32).reshape(-1, 1, 2))
        (pw, ph), ptheta = prect[1], prect[2]
        if min(pw, ph) <= 0:
            continue
        # Angle of the LONG side, folded to [0, 180) like _line_angle_deg.
        piece_angles.append((ptheta if pw >= ph else ptheta + 90.0) % 180.0)
    if len(piece_angles) < 2:
        return [mask]  # fewer than two measurable pieces: nothing to separate
    widest_piece_angle = max(_angle_dist(a, b)
                             for i, a in enumerate(piece_angles)
                             for b in piece_angles[i + 1:])
    if widest_piece_angle < _MIN_SPLIT_ANGLE:
        return [mask]  # all pieces near-parallel: one trail, not a crossing

    return [spine_mask] + tips


# Crossing-evidence rescue: maximum fill fraction for a believable tangle. Real
# crossing trails are thin lines through a mostly-empty box (the 143A8819 tangle
# filled 20% of its box); flooding false positives are solid (50%+). Keeps the
# rescue from ever protecting a flood.
_EVIDENCE_MAX_FILL = 0.45


def has_crossing_evidence(mask, frame_px=None):
    """Is this blob a believable multi-trail crossing, even if it can't be split?

    Used as a RESCUE check by the detection pipeline: when split_crossing()
    gives up on a blob (multi-touch tangles defeat the spine+tips construction),
    the blob used to fall through to the aspect gate, which deletes squat shapes
    as flood false positives -- killing every real trail inside the tangle (the
    143A8819 case: the model masked all four trails at 0.81 confidence and the
    pipeline dropped the lot). This function re-runs the splitter's own ENTRY
    evidence -- enough area, 2+ coherent Hough line directions separated by a
    real crossing angle -- plus a fill-fraction cap that floods can't pass
    (trails are thin lines in a mostly-empty box; floods are solid). A True
    result means "keep this blob whole and let the repair handle it": the Star
    Bridge repair borrows sky by tracking stars, so it cleans a multi-trail
    tangle from the union mask without needing the trails separated.

    Returns True when the blob shows genuine crossing evidence.
    """
    area = int((mask > 0).sum())
    h, w = mask.shape[:2]
    area_scale = (frame_px if frame_px else (h * w)) / _REF_FRAME_PX
    if area < _SPLIT_AREA_MIN * area_scale:
        return False                    # too small to be a multi-trail tangle
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    major_len = max(rect[1])
    minor_len = max(1e-6, min(rect[1]))
    fill = area / max(1.0, major_len * minor_len)
    if fill > _EVIDENCE_MAX_FILL:
        return False                    # solid blob = flood-like, not thin trails
    blob_u8 = (mask > 0).astype(np.uint8) * 255
    crop = blob_u8[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    local_threshold = 10 if area < 20000 else _HOUGH_THRESHOLD
    lines = cv2.HoughLinesP(crop, 1, np.pi / 180,
                            threshold=local_threshold,
                            minLineLength=_HOUGH_MIN_LINE,
                            maxLineGap=_HOUGH_MAX_GAP)
    if lines is None or len(lines) < 2:
        return False                    # no coherent line structure at all
    angles = [_line_angle_deg(L[0]) for L in lines]
    labels, _ = _dbscan_angles(angles, _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES)
    unique = sorted(set(l for l in labels if l >= 0))
    if len(unique) < 2:
        return False                    # one direction = a single trail, not a crossing
    means = {cid: _circular_mean_angle(
        [angles[i] for i, l in enumerate(labels) if l == cid]) for cid in unique}
    # Same weighty-second-direction gate as split_crossing: the rescue must not
    # protect a single dashed trail whose dot-to-dot noise traces mimic a second
    # direction (GoPro 370: 93% one direction, fake second at 2%; the real
    # 143A8819 tangle reads 57/43). The second direction must sit at a real
    # angle to the dominant one AND carry real line evidence.
    lengths = {cid: sum(_line_length(lines[i][0])
                        for i, l in enumerate(labels) if l == cid)
               for cid in unique}
    total_len = sum(lengths.values())
    dom = max(unique, key=lambda c: lengths[c])
    return any(
        cid != dom
        and _angle_dist(means[cid], means[dom]) >= _MIN_SPLIT_ANGLE
        and lengths[cid] >= _MIN_SECOND_DIR_FRAC * total_len
        for cid in unique)
