"""Speck removal must work when the run's saved stuck-pixel map is present.

Regression (2026-09-29): with a saved map the frame scan kept no sample frames, the
persistence detectors returned None, and combining them crashed with OpenCV's
"Sizes of input arguments do not match" on every masked run.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules.sky_dots import remove_specks


def test_remove_specks_with_saved_map():
    rng = np.random.default_rng(1)
    H, W = 200, 240
    with tempfile.TemporaryDirectory() as d:
        names = []
        for i in range(8):
            f = np.full((H, W, 3), 15, np.uint8)
            f += rng.integers(0, 3, size=f.shape, dtype=np.uint8)
            f[80, 100] = (250, 30, 250)          # same pixel every frame: a defect
            f[20 + i * 20, 50 + i * 10] = 200    # a moving star
            n = f"f{i:02d}.png"
            cv2.imwrite(str(Path(d) / n), f)
            names.append(n)
        big = np.max([cv2.imread(str(Path(d) / n)) for n in names], axis=0)
        run_map = np.zeros((H, W), np.uint8)
        run_map[80, 100] = 255
        out = remove_specks(d, names, big, None, lambda p: cv2.imread(p),
                            run_map=run_map)
        assert out.shape == big.shape
        assert int(out[80, 100].max()) < 100, "planted defect was not removed"
