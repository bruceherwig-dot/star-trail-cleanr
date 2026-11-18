"""The channel-maximum helper must give exactly what .max(axis=2) gave.

Repair takes the brightest channel of each pixel a dozen times per trail piece.
numpy's reduction over a short last axis is slow (1.16 ms against 0.086 ms for
the maximum of the three planes, on a 170,000-pixel patch, measured 2026-11-17).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.repair import _chan_max                                   # noqa: E402


def test_same_values_and_dtype_on_8_and_16_bit_patches():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        h, w = int(rng.integers(1, 120)), int(rng.integers(1, 120))
        for dt, top in ((np.uint8, 256), (np.uint16, 65536)):
            a = rng.integers(0, top, (h, w, 3), dtype=dt)
            ref, got = a.max(axis=2), _chan_max(a)
            assert ref.dtype == got.dtype and ref.shape == got.shape and np.array_equal(ref, got)
            assert np.array_equal(ref.astype(np.int16), got.astype(np.int16))


def test_non_contiguous_views_and_tiny_patches():
    rng = np.random.default_rng(1)
    big = rng.integers(0, 65536, (80, 90, 3), dtype=np.uint16)
    view = big[7:60, 11:70]
    assert np.array_equal(view.max(axis=2), _chan_max(view))
    one = rng.integers(0, 256, (1, 1, 3), dtype=np.uint8)
    assert np.array_equal(one.max(axis=2), _chan_max(one))


def test_other_channel_counts_keep_the_original_call():
    rng = np.random.default_rng(2)
    for c in (1, 2, 4):
        a = rng.integers(0, 256, (20, 30, c), dtype=np.uint8)
        assert np.array_equal(a.max(axis=2), _chan_max(a))


def test_repair_has_no_raw_channel_max_left():
    """Only the helper's own fallback may call .max(axis=2); every other line must use _chan_max."""
    src = (REPO / "modules" / "repair.py").read_text(encoding="utf-8")
    offenders = [ln for ln in src.splitlines()
                 if (".max(axis=2)" in ln or ".max(2)" in ln)
                 and not ln.lstrip().startswith(('"""', '#'))
                 and "return a.max(axis=2)" not in ln
                 and "the same values as a.max(axis=2)" not in ln]
    assert not offenders, f"a slow channel maximum crept back into repair.py: {offenders}"
