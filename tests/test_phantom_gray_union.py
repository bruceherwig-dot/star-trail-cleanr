"""Phantom pruning's grey image and detection union must be exactly what they were.

Measured 2026-11-17: the grey conversion (image.max(2)) took 1.4 s and the
per-detection union (union[m > 0] = 1) 1.9 s of the 8.3 s the stage took on 9
frames. Both now use faster routes; the old expressions are kept here as the
definition and the results must match on every pixel.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.detect_pipeline import _max_channel_gray, _union_of_masks   # noqa: E402


def _old_gray(img):
    return img.max(2).astype(np.uint8) if img.ndim == 3 else img.astype(np.uint8)


def _old_union(masks, h, w):
    u = np.zeros((h, w), np.uint8)
    for m in masks:
        if m is not None:
            u[m > 0] = 1
    return u


def test_grey_matches_on_random_8_bit_colour_frames():
    for seed in range(15):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (int(rng.integers(5, 200)), int(rng.integers(5, 200)), 3), dtype=np.uint8)
        a, b = _old_gray(img), _max_channel_gray(img)
        assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape and np.array_equal(a, b)


def test_other_frame_types_keep_the_original_path():
    rng = np.random.default_rng(3)
    for img in (rng.integers(0, 65536, (40, 50, 3), dtype=np.uint16),     # 16-bit: max then wrap to 8-bit
                rng.integers(0, 256, (40, 50), dtype=np.uint8),           # one channel
                rng.integers(0, 256, (40, 50, 4), dtype=np.uint8),        # four channels
                rng.integers(0, 256, (40, 50, 1), dtype=np.uint8)):
        a, b = _old_gray(img), _max_channel_gray(img)
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def test_non_contiguous_input_is_fine():
    rng = np.random.default_rng(4)
    big = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    view = big[5:50, 10:70]
    assert np.array_equal(_old_gray(view), _max_channel_gray(view))


def test_union_matches_on_random_mask_sets():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        h, w = int(rng.integers(20, 300)), int(rng.integers(20, 300))
        masks = []
        for _ in range(int(rng.integers(0, 25))):
            if rng.random() < 0.15:
                masks.append(None)
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.line(m, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                     (int(rng.integers(0, w)), int(rng.integers(0, h))), 255, int(rng.integers(1, 9)))
            masks.append(m)
        a, b = _old_union(masks, h, w), _union_of_masks(masks, h, w)
        assert a.dtype == b.dtype == np.uint8 and np.array_equal(a, b), seed


def test_union_handles_empty_all_none_and_full_masks():
    h, w = 30, 40
    assert np.array_equal(_union_of_masks([], h, w), _old_union([], h, w))
    assert np.array_equal(_union_of_masks([None, None], h, w), _old_union([None, None], h, w))
    full = np.full((h, w), 255, np.uint8)
    assert np.array_equal(_union_of_masks([full], h, w), _old_union([full], h, w))
    odd = np.zeros((h, w), np.uint8)
    odd[3, 4] = 1          # a non-255 non-zero value still counts
    odd[5, 6] = 128
    assert np.array_equal(_union_of_masks([odd], h, w), _old_union([odd], h, w))


def test_the_stage_uses_the_helpers_and_no_per_detection_whole_frame_compare():
    src = (REPO / "modules" / "detect_pipeline.py").read_text(encoding="utf-8")
    a = src.index("def stage_prune_phantoms(")
    b = src.index("    sky = _phantom_local_sky(gray)", a)
    head = src[a:b]
    assert "_max_channel_gray(state.image)" in head
    assert "_union_of_masks(pred_masks, h, w)" in head
    assert "union[m > 0] = 1" not in head
