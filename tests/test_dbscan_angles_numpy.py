"""The crossing splitter's angle clustering must give the same clusters as before.

Measured 2026-11-18: its one-pair-at-a-time neighbour search was 3.7 of the 5 s that
split_crossing took on a busy 6-frame set. The neighbour search is now done with numpy
using the same arithmetic. The original is kept as _dbscan_angles_reference and the
labels must match on every random angle list, including angles near the 0/180 wrap.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.crossing_splitter import (_dbscan_angles, _dbscan_angles_reference,   # noqa: E402
                                       _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES)


def test_labels_match_on_random_angle_lists():
    for seed in range(1500):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(0, 80))
        kind = seed % 4
        if kind == 0:
            angles = [float(a) for a in rng.uniform(0, 180, n)]
        elif kind == 1:            # a few tight directions, some noise
            centres = rng.uniform(0, 180, int(rng.integers(1, 4)))
            angles = [float((rng.choice(centres) + rng.normal(0, 3)) % 180) for _ in range(n)]
        elif kind == 2:            # crowded around the wrap at 0 and 180
            angles = [float((rng.normal(0, 6)) % 180) for _ in range(n)]
        else:                      # whole degrees and halves, lots of exact ties at the edge of eps
            angles = [float(rng.integers(0, 360)) / 2.0 for _ in range(n)]
        for eps, ms in ((_DBSCAN_EPS, _DBSCAN_MIN_SAMPLES), (2.0, 2), (7.5, 3), (0.0, 1)):
            assert _dbscan_angles(angles, eps, ms) == _dbscan_angles_reference(angles, eps, ms), (seed, eps, ms)


def test_empty_and_single_inputs():
    for a in ([], [10.0], [10.0, 10.0], [0.0, 179.9]):
        assert _dbscan_angles(a, _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES) == \
            _dbscan_angles_reference(a, _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES)
