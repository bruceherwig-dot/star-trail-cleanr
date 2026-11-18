"""The star-shift vote must return exactly what the all-pairs Python loop returned.

Measured 2026-11-17: that loop (every candidate scanned against every candidate)
was about half of the star tracking time on 8 JPG frames. Neighbours are now found
with one array comparison; the weights are still summed with Python's own sum in
the same order. The old loop is kept here as the definition.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import repair as R                                        # noqa: E402


def _old_vote(sp, sn):
    cands = []
    for px, py, pw in sp:
        for nx, ny, nw in sn:
            dx, dy = nx - px, ny - py
            if R.MIN_DISP <= (dx * dx + dy * dy) ** 0.5 <= R.MAX_DISP:
                cands.append((dx, dy, min(pw, nw)))
    if not cands:
        return None, 0
    best = None
    for dx, dy, _ in cands:
        v = [c for c in cands if (c[0] - dx) ** 2 + (c[1] - dy) ** 2 <= 16]
        s = sum(c[2] for c in v)
        if best is None or s > best[0]:
            best = (s, np.average([c[0] for c in v], weights=[c[2] for c in v]),
                    np.average([c[1] for c in v], weights=[c[2] for c in v]), len(v))
    return (best[1], best[2]), best[3]


def _new_vote(sp, sn):
    real = R._find_star_streaks
    R._find_star_streaks = lambda g: g          # hand the star lists straight in
    try:
        return R._detect_shift(sp, sn)
    finally:
        R._find_star_streaks = real


def _same(sp, sn):
    a, b = _old_vote(sp, sn), _new_vote(sp, sn)
    assert a[1] == b[1]
    if a[0] is None:
        assert b[0] is None
    else:
        assert a[0][0] == b[0][0] and a[0][1] == b[0][1], (a, b)


def test_identical_on_many_random_star_sets():
    for seed in range(120):
        r = np.random.default_rng(seed)
        k = int(r.integers(1, 41))
        sp = [(float(r.uniform(0, 400)), float(r.uniform(0, 400)), float(r.uniform(100, 5000))) for _ in range(k)]
        sh = (float(r.uniform(-30, 30)), float(r.uniform(-30, 30)))
        sn = [(x + sh[0] + float(r.normal(0, 1.5)), y + sh[1] + float(r.normal(0, 1.5)),
               w * float(r.uniform(0.8, 1.2))) for x, y, w in sp]
        sn += [(float(r.uniform(0, 400)), float(r.uniform(0, 400)), float(r.uniform(100, 5000)))
               for _ in range(int(r.integers(0, 8)))]
        _same(sp, sn)


def test_ties_and_exact_boundaries():
    # Two equally good clusters: the first must win, as before.
    sp = [(10.0, 10.0, 1000.0), (200.0, 200.0, 1000.0)]
    sn = [(15.0, 10.0, 1000.0), (205.0, 200.0, 1000.0)]
    _same(sp, sn)
    # Candidates exactly 4 px apart, then a hair over.
    sp = [(0.0, 0.0, 500.0), (4.0, 0.0, 500.0), (8.0, 0.0, 500.0)]
    sn = [(10.0, 0.0, 500.0), (14.0, 0.0, 500.0), (18.0, 0.0, 500.0)]
    _same(sp, sn)
    sn = [(10.0, 0.0, 500.0), (14.000001, 0.0, 500.0), (18.0, 0.0, 500.0)]
    _same(sp, sn)
    # 3-4-5 triangles sit exactly on the squared-distance boundary of 16 (4 and 0, or 0 and 4).
    sp = [(0.0, 0.0, 700.0), (0.0, 4.0, 700.0), (4.0, 0.0, 900.0)]
    sn = [(20.0, 5.0, 700.0), (20.0, 9.0, 700.0), (24.0, 5.0, 900.0)]
    _same(sp, sn)


def test_none_one_and_identical_candidates():
    _same([], [])
    _same([(5.0, 5.0, 100.0)], [])
    _same([(5.0, 5.0, 100.0)], [(5.0, 5.0, 100.0)])        # shift of 0: below MIN_DISP
    _same([(5.0, 5.0, 100.0)], [(12.0, 9.0, 100.0)])       # one candidate
    _same([(5.0, 5.0, 100.0)] * 3, [(12.0, 9.0, 100.0)] * 3)   # identical candidates
    _same([(0.0, 0.0, 100.0)], [(500.0, 500.0, 100.0)])    # beyond MAX_DISP


def test_the_vote_loop_is_gone_from_the_engine_source():
    src = (REPO / "modules" / "repair.py").read_text(encoding="utf-8")
    assert "v = [c for c in cands if (c[0] - dx) ** 2" not in src
    assert "np.nonzero(np.abs(d2 - 16.0) < 1e-6)" in src
