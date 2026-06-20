"""Smoke tests for the dev streaking-star filter (modules/star_streak.py).

Guards the core behavior: short detections are dropped as star streaks, long trails survive,
and a short RED detection (nav light) is kept. Structural only -- visual quality is Bruce's eye.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from modules.star_streak import filter_segs, measure_ceiling, _mask_long_axis


def _line_mask(h, w, p0, p1, thick=3):
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, p0, p1, 255, thick)
    return m


def test_short_white_dropped_long_kept():
    h, w = 220, 420
    short = _line_mask(h, w, (50, 100), (78, 100))     # ~28px
    long_ = _line_mask(h, w, (200, 30), (320, 170))    # ~184px
    frame = np.full((h, w, 3), 30, np.uint8)           # dim gray, not red
    assert _mask_long_axis(short) < 50
    assert _mask_long_axis(long_) > 50
    kept, kept_c, dropped, n_red = filter_segs([short, long_], [None, None], frame, ceiling=50)
    assert len(kept) == 1 and len(dropped) == 1 and n_red == 0
    assert np.array_equal(kept[0], long_)


def test_short_red_kept_as_nav_light():
    h, w = 220, 420
    short = _line_mask(h, w, (50, 100), (78, 100))     # ~28px, below ceiling
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :, 2] = 200                               # red channel high (BGR) -> warm
    kept, kept_c, dropped, n_red = filter_segs([short], [None], frame, ceiling=50)
    assert len(kept) == 1 and n_red == 1 and len(dropped) == 0


def test_measure_ceiling_none_when_too_few_stars():
    blanks = [np.zeros((200, 300, 3), np.uint8) for _ in range(3)]
    assert measure_ceiling(blanks) is None


def test_measure_ceiling_value_with_many_streaks():
    # A grid of ~120 short bright streaks on a dark frame -> a measurable ceiling.
    img = np.zeros((600, 800, 3), np.uint8)
    for r in range(40, 560, 45):
        for c in range(40, 760, 60):
            cv2.line(img, (c, r), (c + 18, r + 18), (200, 200, 200), 2)   # ~25px streaks
    ceiling = measure_ceiling([img, img, img])
    assert ceiling is not None
    assert 10 < ceiling < 80      # in the plausible streak range, not absurd


def test_flat_150_floor_drops_short_keeps_long():
    from modules.star_streak import MIN_TRAIL_PX
    assert MIN_TRAIL_PX == 150
    h, w = 320, 520
    shortish = _line_mask(h, w, (50, 150), (170, 150))    # ~120px, under 150
    long_ = _line_mask(h, w, (50, 220), (290, 220))       # ~240px, over 150
    frame = np.full((h, w, 3), 30, np.uint8)              # dim gray, not red
    kept, kept_c, dropped, n_red = filter_segs([shortish, long_], [None, None], frame, ceiling=MIN_TRAIL_PX)
    assert len(dropped) == 1 and len(kept) == 1
    assert np.array_equal(kept[0], long_)
