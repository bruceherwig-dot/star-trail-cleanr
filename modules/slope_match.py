"""Slope-match merge: post-process YOLO/SAHI mask output so a single trail that
crosses tile boundaries comes through as ONE polygon instead of multiple
disconnected pieces.

After SAHI's NMS step, two pieces of the same long thin trail in adjacent tiles
often survive as separate detections because their bounding boxes don't overlap
enough for NMS to recognize them as duplicates. This module identifies pairs
that are pieces of one physical trail (matching slope, near-collinear, masks
sharing actual pixels) and unions them.

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
    """Principal-axis geometry of a binary mask via PCA on the foreground pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mu = coords.mean(axis=0)
    centered = coords - mu
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    principal = evecs[:, 0]
    perp = evecs[:, 1]
    proj_p = centered @ principal
    proj_q = centered @ perp
    length = float(proj_p.max() - proj_p.min())
    thick = float(proj_q.max() - proj_q.min())
    return {
        "mu": mu,
        "ang": math.degrees(math.atan2(principal[1], principal[0])),
        "length": length,
        "thick": max(thick, 1.0),
        "aspect": length / max(thick, 1.0),
        "p_lo": mu + proj_p.min() * principal,
        "p_hi": mu + proj_p.max() * principal,
    }


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 90) % 180 - 90)


def _perp_distance(point, ref_center, angle_deg: float) -> float:
    ang = math.radians(angle_deg)
    nx, ny = -math.sin(ang), math.cos(ang)
    return abs((point[0] - ref_center[0]) * nx + (point[1] - ref_center[1]) * ny)


def _mask_intersection_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Count pixels where both masks are on. Computed inside the bbox
    intersection only so it is fast on full-image masks."""
    ya, xa = np.where(mask_a > 0)
    yb, xb = np.where(mask_b > 0)
    if len(xa) == 0 or len(xb) == 0:
        return 0
    rx0, ry0 = max(xa.min(), xb.min()), max(ya.min(), yb.min())
    rx1, ry1 = min(xa.max(), xb.max()), min(ya.max(), yb.max())
    if rx1 < rx0 or ry1 < ry0:
        return 0
    sub_a = mask_a[ry0:ry1 + 1, rx0:rx1 + 1]
    sub_b = mask_b[ry0:ry1 + 1, rx0:rx1 + 1]
    return int(np.logical_and(sub_a > 0, sub_b > 0).sum())


def _pair_score(item_a, item_b) -> float:
    """Score (lower = better) for whether the pair should merge, or None if not."""
    a = item_a["met"]
    b = item_b["met"]
    if a["aspect"] < ASPECT_MIN or b["aspect"] < ASPECT_MIN:
        return None
    da = _angle_diff(a["ang"], b["ang"])
    if da > ANGLE_TOL_DEG:
        return None
    d_perp = _perp_distance(b["mu"], a["mu"], a["ang"])
    if d_perp > PERP_DIST_PX:
        return None
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
    overlap = _mask_intersection_count(item_a["mask"], item_b["mask"])
    if overlap < MIN_MASK_OVERLAP_PX:
        return None
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
    if len(masks) < 2:
        return list(masks)

    items = []
    for m in masks:
        met = _mask_metrics(m)
        if met is None:
            # mask too small to score — keep it but don't try to merge it
            items.append({"mask": m, "met": None})
        else:
            items.append({"mask": m, "met": met})

    while True:
        best = None
        best_score = float("inf")
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
        merged_mask = np.maximum(items[i]["mask"], items[j]["mask"])
        new_met = _mask_metrics(merged_mask)
        items = [it for k, it in enumerate(items) if k not in (i, j)]
        items.append({"mask": merged_mask, "met": new_met})

    return [it["mask"] for it in items]
