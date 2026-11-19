"""Slope-match merge: post-process YOLO/SAHI mask output so a single trail that
crosses tile boundaries comes through as ONE polygon instead of multiple
disconnected pieces.

Plain-English picture: the detector looks at the photo in overlapping 640x640
tiles. A long airplane or satellite trail that runs across a tile boundary can
get found twice — once as a piece in the left tile and once as a piece in the
right tile. The usual de-duplication step (SAHI's NMS) only catches duplicates
whose bounding boxes overlap a lot, and two end-to-end pieces of one straight
trail barely overlap, so both survive as separate detections. The repair step
would then treat one airplane as two trails. This module's job is to glue those
pieces back together: it looks at every pair of detected trail masks and, when a
pair really looks like two halves of the SAME straight streak, it unions them
into a single mask.

How it decides a pair belongs together (all four tests must pass):
  1. Same direction — the two pieces point the same way (slopes agree within a
     few degrees).
  2. Collinear — one piece's center sits very close to the straight line the
     other piece lies along (they're end-to-end, not side-by-side).
  3. Right-sized gap — their nearest tips are close but not on top of each
     other (NMS already handles fully-overlapping pairs).
  4. Truly touching — the two masks literally share some lit pixels, which is
     the fingerprint of one continuous physical trail rather than two unrelated
     streaks that happen to line up.
Both pieces must also be long and thin (elongated), since trails are.

This file does pure geometry on binary masks (NumPy/OpenCV). It does not run the
neural net, read images, or write files — it just takes the list of detected
masks for one frame and returns a (usually shorter) list with the split pieces
combined. The single public entry point is merge().

Validated 2026-04-25 across 963 frames of the Greg Meyer Brightened dataset.
Algorithm and threshold derivation in
runs/active/big_box_2026_04_25/NOTES.md.
"""
import math
from typing import List, Tuple

import cv2
import numpy as np


# Tuning constants — derived empirically through 8 iterations on the Greg
# Meyer test set. Change here, not at call sites.
ANGLE_TOL_DEG = 3.0          # principal-axis slopes must agree within this
PERP_DIST_PX = 25.0          # one mask's center must lie this close to the
                              # other's principal line
MIN_GAP_PX = 5.0             # nearest endpoints must be at least this far apart
                              # (NMS already handles fully-overlapping pairs)
MAX_GAP_PX = 150.0           # but not more than this (safety bound)
ASPECT_MIN = 2.0             # both pieces must be elongated (length / thickness)
MIN_MASK_OVERLAP_PX = 10     # the masks must literally share at least this many
                              # pixels — the contiguous-trail fingerprint


def _mask_metrics(mask: np.ndarray):
    """Measure the shape and orientation of one trail mask.

    Runs PCA (principal component analysis) on the lit pixels to find the
    direction the streak points and how long/thin it is. Think of fitting an
    oriented rectangle around the blob: the long edge gives the trail's
    direction and length, the short edge gives its thickness.

    Input:
        mask: a 2D binary image where lit pixels (> 0) are the trail and 0 is
            sky. Same full-image size as every other mask in the frame.

    Returns:
        A dict describing the blob's geometry (see keys below), or None if the
        mask has fewer than 3 lit pixels — too few to fit a meaningful axis, so
        the caller skips it for merging.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    # Stack the lit-pixel coordinates as (x, y) points and center them on their
    # mean so PCA measures spread, not position.
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mu = coords.mean(axis=0)
    centered = coords - mu
    # Covariance of the centered points; its eigenvectors are the blob's axes.
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    # Sort axes by spread, largest first, so column 0 is the long (principal)
    # axis and column 1 is the short (perpendicular) axis.
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    principal = evecs[:, 0]
    perp = evecs[:, 1]
    # Project every pixel onto each axis; the spread along an axis is that
    # dimension's extent. Long axis -> length, short axis -> thickness.
    proj_p = centered @ principal
    proj_q = centered @ perp
    length = float(proj_p.max() - proj_p.min())
    thick = float(proj_q.max() - proj_q.min())
    return {
        "mu": mu,                                              # center point (x, y)
        "ang": math.degrees(math.atan2(principal[1], principal[0])),  # direction in degrees
        "length": length,                                      # extent along the long axis
        "thick": max(thick, 1.0),                              # extent across (min 1 to avoid /0)
        "aspect": length / max(thick, 1.0),                    # how elongated: length / thickness
        "p_lo": mu + proj_p.min() * principal,                 # one tip of the streak (x, y)
        "p_hi": mu + proj_p.max() * principal,                 # the other tip (x, y)
    }


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two line orientations, in degrees (0..90).

    Inputs a and b are angles in degrees. Trail direction is a line, not an
    arrow: pointing "up" and "down" is the same streak, and 179 deg is nearly
    the same as 1 deg. The modulo-180 wrap folds the difference into the 0..90
    range so those near-opposite slopes correctly read as a tiny difference.
    """
    return abs((a - b + 90) % 180 - 90)


def _perp_distance(point, ref_center, angle_deg: float) -> float:
    """How far `point` sits off the infinite line through `ref_center`.

    The line passes through ref_center at orientation angle_deg. This returns
    the perpendicular (sideways) distance from `point` to that line, in pixels.
    Used to test collinearity: two pieces of one straight trail have a tiny
    perpendicular distance; pieces of two side-by-side trails do not.

    (nx, ny) is the unit vector perpendicular to the line, so the dot product
    of (point - ref_center) with it is exactly that sideways offset.
    """
    ang = math.radians(angle_deg)
    nx, ny = -math.sin(ang), math.cos(ang)
    return abs((point[0] - ref_center[0]) * nx + (point[1] - ref_center[1]) * ny)


def _mask_intersection_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Count pixels that are lit in BOTH masks (their shared overlap area).

    Inputs are two full-image binary masks. Returns the number of pixels where
    both are on. This is the "do they literally touch" fingerprint that
    distinguishes two pieces of one real trail from two unrelated streaks that
    merely happen to line up.

    Speed note: full-image masks are mostly empty, so comparing them whole would
    be wasteful. Instead it first finds the rectangle where the two masks'
    bounding boxes overlap and counts only inside that small window. If the
    bounding boxes don't overlap at all, the answer is trivially 0.
    """
    ya, xa = np.where(mask_a > 0)
    yb, xb = np.where(mask_b > 0)
    if len(xa) == 0 or len(xb) == 0:
        return 0
    # Bounding box of the overlap region: top-left is the max of each mask's
    # top-left, bottom-right is the min of each mask's bottom-right.
    rx0, ry0 = max(xa.min(), xb.min()), max(ya.min(), yb.min())
    rx1, ry1 = min(xa.max(), xb.max()), min(ya.max(), yb.max())
    if rx1 < rx0 or ry1 < ry0:
        # Bounding boxes are disjoint, so the masks cannot share any pixel.
        return 0
    # Crop both masks to that shared window and count where both are lit.
    sub_a = mask_a[ry0:ry1 + 1, rx0:rx1 + 1]
    sub_b = mask_b[ry0:ry1 + 1, rx0:rx1 + 1]
    return int(np.logical_and(sub_a > 0, sub_b > 0).sum())


def _pair_score(item_a, item_b) -> float:
    """Decide whether two trail pieces should merge, and how good a match it is.

    Each input is a working item: a dict with the mask ("mask") and its measured
    geometry ("met", from _mask_metrics). Applies the four merge tests in order
    (cheap geometry checks first, the costly pixel-overlap check last) and
    returns one of:
        None  -> the pair fails a test; do NOT merge.
        float -> a score where LOWER is a better/tighter match. The caller uses
                 this to merge the single best pair first when several qualify.

    The tests, in the order checked:
      1. Both pieces elongated enough (aspect >= ASPECT_MIN) — trails are thin.
      2. Directions agree within ANGLE_TOL_DEG.
      3. They are collinear: b's center lies within PERP_DIST_PX of a's line.
      4. The nearest tips are between MIN_GAP_PX and MAX_GAP_PX apart — close
         but not on top of each other (NMS already removes near-duplicates).
      5. The masks physically overlap by at least MIN_MASK_OVERLAP_PX pixels.
    """
    a = item_a["met"]
    b = item_b["met"]
    # Test 1: both must be long-and-thin; squat blobs are not trails.
    if a["aspect"] < ASPECT_MIN or b["aspect"] < ASPECT_MIN:
        return None
    # Test 2: must point the same direction.
    da = _angle_diff(a["ang"], b["ang"])
    if da > ANGLE_TOL_DEG:
        return None
    # Test 3: b's center must sit on (close to) a's line — end-to-end, not
    # side-by-side.
    d_perp = _perp_distance(b["mu"], a["mu"], a["ang"])
    if d_perp > PERP_DIST_PX:
        return None
    # Test 4: find the closest pair of tips across the two pieces. Each piece has
    # two endpoints (p_lo, p_hi); try all four tip-to-tip combinations and take
    # the shortest. That shortest distance is the "gap" between the pieces.
    candidates = [
        (a["p_lo"], b["p_lo"]),
        (a["p_lo"], b["p_hi"]),
        (a["p_hi"], b["p_lo"]),
        (a["p_hi"], b["p_hi"]),
    ]
    best = min(candidates, key=lambda c: float(np.linalg.norm(np.array(c[0]) - np.array(c[1]))))
    gap = float(np.linalg.norm(np.array(best[0]) - np.array(best[1])))
    if gap < MIN_GAP_PX or gap > MAX_GAP_PX:
        return None
    # Test 5 (most expensive, done last): the masks must literally share pixels.
    overlap = _mask_intersection_count(item_a["mask"], item_b["mask"])
    if overlap < MIN_MASK_OVERLAP_PX:
        return None
    # Combined score: angle dominates; perpendicular offset and gap are scaled
    # down (/10 and /200) so they only break ties between similarly-aligned
    # pairs. Lower is a better match.
    return da + d_perp / 10 + gap / 200


def merge(masks: List[np.ndarray]) -> List[np.ndarray]:
    """Iteratively union pairs of masks that look like pieces of one trail.

    Args:
        masks: list of 2D binary uint8 arrays (255 = trail, 0 = sky), all the
            same shape (full-image size).

    Returns:
        New list of binary uint8 arrays (same shape) where pieces of the same
        trail have been combined. Length is <= input length. If the input is
        empty or has fewer than two valid masks, the input is returned as-is.
    """
    # Nothing to merge with fewer than two masks.
    if len(masks) < 2:
        return list(masks)

    # Wrap each mask with its measured geometry so we don't recompute it on every
    # comparison. Masks too small to score (met is None) are carried through
    # untouched and never considered for merging.
    items = []
    for m in masks:
        met = _mask_metrics(m)
        if met is None:
            # mask too small to score — keep it but don't try to merge it
            items.append({"mask": m, "met": None})
        else:
            items.append({"mask": m, "met": met})

    # Greedy loop: repeatedly find the single best-scoring mergeable pair, union
    # it, and re-measure the result. Each pass merges at most one pair, so a
    # trail split into three tiles collapses over successive passes (A+B, then
    # AB+C). Stops when no remaining pair qualifies.
    while True:
        best = None
        best_score = float("inf")
        # Scan every unordered pair (j starts at i+1) for the best score.
        for i in range(len(items)):
            if items[i]["met"] is None:
                continue
            for j in range(i + 1, len(items)):
                if items[j]["met"] is None:
                    continue
                s = _pair_score(items[i], items[j])
                if s is not None and s < best_score:
                    best_score = s
                    best = (i, j)
        if best is None:
            break
        i, j = best
        # Union the two masks (pixel-wise max), drop the originals, and append
        # the combined piece with freshly measured geometry so it can merge
        # again in a later pass.
        merged_mask = np.maximum(items[i]["mask"], items[j]["mask"])
        new_met = _mask_metrics(merged_mask)
        items = [it for k, it in enumerate(items) if k not in (i, j)]
        items.append({"mask": merged_mask, "met": new_met})

    return [it["mask"] for it in items]
