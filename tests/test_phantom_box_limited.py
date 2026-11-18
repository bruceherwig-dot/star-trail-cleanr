"""Phantom pruning works inside each detection's own box and must give the same answers.

Measured 2026-11-18: merging the 949 detection masks of a 20-frame batch and finding
each one's edges swept the whole frame twice per detection (4.3 of 11 s). The box now
comes from the outline points. The old whole-frame expressions are kept here as the
definition: the box must hold every lit pixel, and the union must match pixel for pixel.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.detect_pipeline import _pred_box, _union_of_masks   # noqa: E402
from modules.trail_grouper import _pred_to_mask                   # noqa: E402


def _pred(polys, h, w):
    seg = [list(np.asarray(p, float).ravel()) for p in polys]
    return SimpleNamespace(mask=SimpleNamespace(segmentation=seg, full_shape=(h, w), bool_mask=None))


def _random_pred(rng, h, w):
    polys = []
    for _ in range(int(rng.integers(1, 3))):
        n = int(rng.integers(3, 8))
        cx, cy = rng.uniform(-30, w + 30), rng.uniform(-30, h + 30)
        r = rng.uniform(1, 90)
        ang = np.sort(rng.uniform(0, 6.28, n))
        pts = np.stack([cx + r * np.cos(ang) * rng.uniform(.2, 1), cy + r * np.sin(ang) * rng.uniform(.2, 1)], 1)
        polys.append(pts + rng.uniform(-.49, .49, pts.shape))
    return _pred(polys, h, w)


def test_the_box_always_holds_every_lit_pixel():
    for seed in range(300):
        rng = np.random.default_rng(seed)
        h, w = int(rng.integers(20, 260)), int(rng.integers(20, 260))
        p = _random_pred(rng, h, w)
        m = _pred_to_mask(p, h, w)
        r0, r1, c0, c1 = _pred_box(p, h, w)
        total = int((m > 0).sum())
        inside = int((m[r0:r1, c0:c1] > 0).sum()) if r1 > r0 and c1 > c0 else 0
        assert inside == total, (seed, total, inside)


def test_union_with_boxes_matches_the_whole_frame_union():
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        h, w = int(rng.integers(20, 300)), int(rng.integers(20, 300))
        preds = [_random_pred(rng, h, w) for _ in range(int(rng.integers(0, 25)))]
        masks = [_pred_to_mask(p, h, w) for p in preds]
        old = np.zeros((h, w), np.uint8)
        for m in masks:
            old[m > 0] = 1
        new = _union_of_masks(masks, h, w, [_pred_box(p, h, w) for p in preds])
        assert np.array_equal(old, new), seed


def test_no_outline_or_a_resized_mask_gives_the_whole_frame():
    h, w = 50, 60
    no_outline = SimpleNamespace(mask=SimpleNamespace(segmentation=None, full_shape=(h, w)))
    assert _pred_box(no_outline, h, w) == (0, h, 0, w)
    resized = _pred([[(1, 1), (10, 1), (10, 10)]], h * 2, w * 2)
    assert _pred_box(resized, h, w) == (0, h, 0, w)
    assert _pred_box(object(), h, w) == (0, h, 0, w)


def test_a_detection_made_in_a_tile_carries_that_tile_as_its_box():
    from modules.detect_pipeline import _SyntheticPred
    h, w = 100, 120
    m = np.zeros((h, w), bool)
    m[30:40, 50:90] = True
    p = _SyntheticPred(m, 0.5, box=(20, 70, 40, 100))
    r0, r1, c0, c1 = _pred_box(p, h, w)
    assert (r0, r1, c0, c1) == (20, 70, 40, 100)
    assert int((m[r0:r1, c0:c1]).sum()) == int(m.sum())
    # a box running past the frame edge is clipped to the frame
    q = _SyntheticPred(m, 0.5, box=(80, 140, 100, 180))
    assert _pred_box(q, h, w) == (80, 100, 100, 120)
    # no box known: the whole frame, always safe
    assert _pred_box(_SyntheticPred(m, 0.5), h, w) == (0, h, 0, w)
    masks = [_pred_to_mask_from_bool(m)]
    old = (m > 0).astype(np.uint8)
    assert np.array_equal(_union_of_masks(masks, h, w, [(20, 70, 40, 100)]), old)


def _pred_to_mask_from_bool(m):
    return m.astype(np.uint8) * 255
