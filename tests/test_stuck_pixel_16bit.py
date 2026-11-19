"""The 16-bit ground stuck-pixel repaint must paint exactly what the old step painted.

Until 2026-11-17 the engine read each frame off the disk again for this step
(a second RAW decode per frame). It now paints on the frame already in memory.
The old formula is kept here, fed from an array instead of a file, as the
definition of "the same pixels".
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.hot_pixels import repaint_ground_stuck_pixels_16bit      # noqa: E402


def _old_step(orig_from_disk, dilated):
    orig = orig_from_disk.copy()
    orig8 = (orig >> 8).astype(np.uint8)
    rep8 = cv2.inpaint(orig8, dilated, 3, cv2.INPAINT_NS)
    rep16 = rep8.astype(np.uint16) * 257
    orig[dilated > 0] = rep16[dilated > 0]
    return orig


def _frame(seed, h=90, w=120):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)


def _spots(seed, h=90, w=120, n=6):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        y, x = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))
        m[y, x] = 255
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))


def test_same_pixels_as_the_old_step_on_many_frames():
    for seed in range(20):
        f, d = _frame(seed), _spots(seed + 100)
        assert np.array_equal(_old_step(f, d), repaint_ground_stuck_pixels_16bit(f, d)), seed


def test_spots_touching_the_frame_edge_and_a_full_mask():
    f = _frame(7)
    edge = np.zeros(f.shape[:2], np.uint8)
    edge[0:15, 0:15] = 255
    edge[-15:, -15:] = 255
    for d in (edge, np.full(f.shape[:2], 255, np.uint8)):
        assert np.array_equal(_old_step(f, d), repaint_ground_stuck_pixels_16bit(f, d))


def test_the_frame_passed_in_is_not_changed():
    f = _frame(3)
    before = f.copy()
    out = repaint_ground_stuck_pixels_16bit(f, _spots(4))
    assert np.array_equal(f, before)
    assert out is not f and out.dtype == np.uint16


def test_pixels_outside_the_spots_are_untouched():
    f, d = _frame(11), _spots(12)
    out = repaint_ground_stuck_pixels_16bit(f, d)
    assert np.array_equal(out[d == 0], f[d == 0])
    assert not np.array_equal(out[d > 0], f[d > 0])


def test_a_frame_brought_up_from_8_bit_is_painted_not_blacked_out():
    """A mixed batch: the loader turns an 8-bit frame into 16-bit (x257). The old
    step read the 8-bit file back and wrote zeros into every spot."""
    rng = np.random.default_rng(5)
    f8 = rng.integers(30, 200, (80, 100, 3), dtype=np.uint8)
    f16 = f8.astype(np.uint16) * 257
    d = np.zeros((80, 100), np.uint8)
    d[38:43, 48:53] = 255
    out = repaint_ground_stuck_pixels_16bit(f16, d)
    assert out.dtype == np.uint16
    assert (out[d > 0] > 0).all(), "the spots came out black"


def test_engine_paints_in_memory_and_no_longer_rereads_frames():
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert "repaint_ground_stuck_pixels_16bit(frames[i], dilated)" in src
    assert "orig = robust_imread(frame_files[i]" not in src
