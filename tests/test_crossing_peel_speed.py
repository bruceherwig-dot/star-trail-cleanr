"""The two speedups in the crossing peel / tangle check must give exactly the old answers.

1. _best_line works all 180 angles in one table; _best_line_reference (the original loop) is the
   definition, and the two must agree on the angle, the offset and every member point.
2. has_crossing_evidence remembers its last answer for an identical blob; it must never return a
   stale answer when the blob, its contents or the frame size differ.
Measured 2026-11-18 on 6 busy frames: line search 0.98 s, repeat tangle checks about 0.2 s.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import crossing_peel                                          # noqa: E402
from modules.crossing_peel import _best_line, _best_line_reference         # noqa: E402
from modules.crossing_splitter import has_crossing_evidence, _has_crossing_evidence_uncached   # noqa: E402


def _points(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 1500))
    kind = seed % 4
    if kind == 0:
        return rng.uniform(0, 300, (n, 2))
    if kind == 1:
        t = rng.uniform(0, 200, n)
        return np.column_stack([t, 0.3 * t + rng.normal(0, 1.5, n)])
    if kind == 2:
        return rng.integers(0, 60, (n, 2)).astype(float)          # many exact ties
    t = rng.uniform(0, 150, n)
    a = np.column_stack([t, 0.5 * t])
    b = np.column_stack([t, 150 - t])
    return np.vstack([a, b]) + rng.normal(0, 0.8, (2 * n, 2))      # a crossing


def test_fast_line_search_matches_the_one_by_one_search():
    for seed in range(300):
        rng = np.random.default_rng(10_000 + seed)
        w = float(rng.uniform(1, 40))
        pts = _points(seed)
        a, b = _best_line(pts, w), _best_line_reference(pts, w)
        assert a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2]), seed


def test_a_single_point_and_a_straight_row_agree_too():
    for pts in (np.array([[5.0, 7.0]]), np.column_stack([np.arange(40.0), np.full(40, 3.0)]),
                np.column_stack([np.full(40, 3.0), np.arange(40.0)])):
        for w in (1.0, 4.0, 25.0):
            a, b = _best_line(pts, w), _best_line_reference(pts, w)
            assert a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2])


def test_very_big_blobs_use_the_one_by_one_search():
    old = crossing_peel._FAST_SEARCH_MAX_CELLS
    try:
        crossing_peel._FAST_SEARCH_MAX_CELLS = 10        # forces the fallback
        pts = _points(3)
        a, b = _best_line(pts, 5.0), _best_line_reference(pts, 5.0)
        assert a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2])
    finally:
        crossing_peel._FAST_SEARCH_MAX_CELLS = old


def _blob(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(120, 400)), int(rng.integers(120, 400))
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 5))):
        cv2.line(m, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                 (int(rng.integers(0, w)), int(rng.integers(0, h))), 255, int(rng.integers(2, 14)))
    return m


def test_evidence_answers_match_the_uncached_check_on_a_stream_of_blobs():
    blobs = [_blob(s) for s in range(60)]
    stream = []
    for b in blobs:
        stream += [b, b, b.copy()]            # repeats, as the peel guard then the pipeline ask
    for i, m in enumerate(stream):
        fp = 24_000_000 if i % 5 else None
        assert has_crossing_evidence(m, frame_px=fp) == _has_crossing_evidence_uncached(m, frame_px=fp), i


def test_evidence_does_not_reuse_an_answer_for_a_changed_blob_or_frame_size():
    m = _blob(7)
    first = has_crossing_evidence(m, frame_px=1_000_000)
    # same blob, different frame size: the size floor changes, so the cache must not answer
    assert has_crossing_evidence(m, frame_px=1_000_000_000) == _has_crossing_evidence_uncached(m, frame_px=1_000_000_000)
    assert has_crossing_evidence(m, frame_px=1_000_000) == first
    # the same array edited in place must not get the old answer
    m2 = m.copy()
    has_crossing_evidence(m2, frame_px=24_000_000)
    m2[:] = 0
    assert has_crossing_evidence(m2, frame_px=24_000_000) == _has_crossing_evidence_uncached(m2, frame_px=24_000_000)
