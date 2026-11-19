"""The darken restore must give exactly what it gave when it worked on the whole crop.

Measured 2026-11-17: the per-pixel median across the neighbor windows took 5.5 s
of 21.7 s of repair on 6 RAW frames, but the restore only reads it at the trail's
own pixels (1 to 8 percent of the crop). _darken_fill_windows works those out
only there. The original _darken_fill, fed the whole-crop median and minimum, is
the definition: the two must agree on every pixel.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.repair import _darken_fill, _darken_fill_windows           # noqa: E402


def _whole_crop(patch, wins, comp, collar):
    stack = np.stack(wins)
    dmin = np.min(stack, axis=0)
    dmed = np.median(stack, axis=0).astype(patch.dtype)
    maxv = 65535 if patch.dtype == np.uint16 else 255
    return _darken_fill(patch, dmin, dmed, comp, collar, maxv, patch.dtype)


def _case(seed, dtype, k, h=70, w=90, dark_fg=True):
    rng = np.random.default_rng(seed)
    top = 255 if dtype == np.uint8 else 65535
    sky = int(top * 0.16)
    wins = [np.clip(rng.normal(sky, top * 0.02, (h, w, 3)), 0, top).astype(dtype) for _ in range(k)]
    if dark_fg:                                   # dark static ground in every window
        for wn in wins:
            wn[20:40, 30:50] = int(top * 0.02)
    cm = np.zeros((h, w), np.uint8)
    cv2.line(cm, (5, 25), (85, 35), 1, 5)
    comp = cm.astype(bool)
    for wn in wins[: max(1, k // 2)]:             # a bright trail crossing in some windows
        wn[comp] = int(top * 0.9)
    ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool) & ~comp
    patch = wins[len(wins) // 2].copy()
    patch[comp] = int(top * 0.17)                 # the sky slide already painted over the trail
    return patch, wins, comp, ring


def test_identical_on_many_cases_8_and_16_bit_and_every_window_size():
    changed = 0
    for seed in range(40):
        dtype = np.uint8 if seed % 2 == 0 else np.uint16
        k = 2 + seed % 6                          # 2 to 7 windows, as at a sequence edge
        patch, wins, comp, ring = _case(seed, dtype, k)
        a = _whole_crop(patch, wins, comp, ring)
        b = _darken_fill_windows(patch, wins, comp, ring, patch.dtype)
        assert a[1] == b[1] and a[2] == b[2], f"counts differ in case {seed}"
        assert np.array_equal(a[0], b[0]), f"pixels differ in case {seed}"
        changed += a[1]
    assert changed > 200, "the cases stopped triggering the restore"


def test_no_dark_ground_returns_the_patch_untouched():
    patch, wins, comp, ring = _case(5, np.uint8, 5, dark_fg=False)
    a = _whole_crop(patch, wins, comp, ring)
    b = _darken_fill_windows(patch, wins, comp, ring, patch.dtype)
    assert a[1] == b[1] == 0 and b[0] is patch and a[2] == b[2]


def test_cheap_exits_match_empty_mask_tiny_ring_and_zero_sky():
    patch, wins, comp, ring = _case(7, np.uint8, 5)
    empty = np.zeros_like(comp)
    a = _whole_crop(patch, wins, empty, ring)
    b = _darken_fill_windows(patch, wins, empty, ring, patch.dtype)
    assert (a[1], a[2]) == (b[1], b[2]) and np.array_equal(a[0], b[0])
    tiny = np.zeros_like(ring)
    tiny[0, :10] = True
    a = _whole_crop(patch, wins, comp, tiny)
    b = _darken_fill_windows(patch, wins, comp, tiny, patch.dtype)
    assert a[1] == b[1] == 0
    assert a[2] is None and b[2] is None
    black = [np.zeros_like(w) for w in wins]
    zp = np.zeros_like(patch)
    a = _whole_crop(zp, black, comp, ring)
    b = _darken_fill_windows(zp, black, comp, ring, zp.dtype)
    assert (a[1], a[2]) == (b[1], b[2])


def test_the_input_patch_is_not_changed():
    patch, wins, comp, ring = _case(11, np.uint16, 7)
    before = patch.copy()
    out = _darken_fill_windows(patch, wins, comp, ring, patch.dtype)
    assert np.array_equal(patch, before)
    assert out[1] > 0 and out[0] is not patch


def test_repair_uses_the_masked_version_and_no_whole_crop_median():
    src = (REPO / "modules" / "repair.py").read_text(encoding="utf-8")
    assert "_darken_fill_windows(" in src
    assert "np.median(_wstack, axis=0)" not in src
