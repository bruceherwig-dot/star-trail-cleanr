"""The polygon de-duplication must keep exactly the polygons and masks it kept
when it worked on whole-frame masks.

Measured 2026-11-17: two whole-frame lines (a compare and a sum, once per
polygon) took 6.3 of this function's 7.2 seconds on 6 busy RAW frames. It now
measures inside each polygon's own box. The old version is kept in the module as
_polys_and_segs_deduped_reference; the two must agree on every polygon and every
pixel, because this step decides which trails get repaired.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.detect_pipeline import (_polys_and_segs_deduped,             # noqa: E402
                                     _polys_and_segs_deduped_reference)


def _rect(cx, cy, length, width, angle_deg):
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-d[1], d[0]])
    c = np.array([cx, cy], float)
    pts = [c - d * length / 2 - n * width / 2, c + d * length / 2 - n * width / 2,
           c + d * length / 2 + n * width / 2, c - d * length / 2 + n * width / 2]
    return np.round(np.array(pts)).astype(np.int32)


def _same(h, w, polys):
    ref_p, ref_s = _polys_and_segs_deduped_reference(polys, h, w)
    new_p, new_s = _polys_and_segs_deduped(polys, h, w)
    assert len(ref_p) == len(new_p) and len(ref_s) == len(new_s)
    for a, b in zip(ref_p, new_p):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(ref_s, new_s):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return len(ref_p), len(polys)


def test_identical_on_many_random_polygon_sets():
    merged_somewhere = False
    for seed in range(40):
        rng = np.random.default_rng(seed)
        h, w = int(rng.integers(200, 520)), int(rng.integers(200, 620))
        polys = []
        for _ in range(int(rng.integers(0, 14))):
            cx, cy = float(rng.uniform(-30, w + 30)), float(rng.uniform(-30, h + 30))   # some hang off the frame
            length, width = float(rng.uniform(4, 380)), float(rng.uniform(2, 40))
            ang = float(rng.uniform(0, 180))
            polys.append(_rect(cx, cy, length, width, ang))
            if rng.random() < 0.5:          # a redundant fragment sitting inside it, same angle
                polys.append(_rect(cx, cy, length * rng.uniform(0.3, 0.9), width * rng.uniform(0.3, 0.9),
                                   ang + float(rng.uniform(-5, 5))))
        kept, total = _same(h, w, polys)
        merged_somewhere |= kept < total
    assert merged_somewhere, "the random sets stopped containing mergeable fragments"


def test_duplicates_and_contained_fragments_are_merged_the_same_way():
    big = _rect(200, 150, 300, 20, 30)
    inner = _rect(200, 150, 200, 8, 30)
    kept, total = _same(300, 400, [inner, big, inner.copy()])
    assert kept == 1 and total == 3


def test_polygons_hanging_off_every_edge_and_fully_outside():
    polys = [_rect(0, 0, 100, 20, 45), _rect(399, 0, 100, 20, 135), _rect(0, 299, 100, 20, 135),
             _rect(399, 299, 100, 20, 45), _rect(-500, -500, 50, 10, 0), _rect(900, 900, 50, 10, 0),
             _rect(200, 150, 80, 10, 10)]
    _same(300, 400, polys)


def test_tiny_and_degenerate_polygons():
    polys = [np.array([[10, 10], [10, 10], [10, 10], [10, 10]], np.int32),          # a point
             np.array([[20, 20], [60, 20], [100, 20], [140, 20]], np.int32),        # a line
             np.array([[30, 30], [31, 30], [31, 31], [30, 31]], np.int32),          # 2x2
             _rect(120, 120, 60, 12, 20)]
    _same(200, 260, polys)


def test_zero_one_and_two_polygons():
    _same(100, 120, [])
    _same(100, 120, [_rect(50, 50, 40, 8, 0)])
    _same(100, 120, [_rect(50, 50, 40, 8, 0), _rect(50, 50, 60, 12, 0)])


def test_no_whole_frame_compare_or_sum_left_in_the_new_version():
    src = (REPO / "modules" / "detect_pipeline.py").read_text(encoding="utf-8")
    a = src.index("def _polys_and_segs_deduped(polygons: list, h: int, w: int):")
    b = src.index("def _log_thin_events(log, events):")
    body = src[a:b]
    assert "bmasks" not in body
    assert "[s > 0 for s in segs]" not in body
