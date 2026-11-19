"""Guards against the sub-640 frame crash (Sentry 7527007045).

A user ran a folder of 800x533 downsized previews; the tiler computed a
negative tile start (533 - 640 = -107), which crashed the worker pasting a
640x640 mask into a 107-row slot. Two layers protect against this:
  1. The app blocks frames whose shorter side is under MIN_FRAME_SHORT_SIDE.
  2. The tiler never emits a negative start, so the worker can't crash even
     if a small frame reaches it some other way.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.frame_list import frame_too_small, MIN_FRAME_SHORT_SIDE
from modules.detect_pipeline import _tile_starts


def test_threshold_is_1280():
    assert MIN_FRAME_SHORT_SIDE == 1280


def test_downsized_previews_are_blocked():
    # The actual crashing frame and common preview sizes.
    assert frame_too_small(800, 533)      # the Sentry crash frame
    assert frame_too_small(1024, 683)
    assert frame_too_small(1920, 1280) is False   # short side exactly 1280 = OK
    assert frame_too_small(1280, 720)     # short side 720 < 1280 = too small


def test_real_camera_frames_pass():
    for w, h in [(6000, 4000), (8256, 5504), (5472, 3648), (4096, 2731), (1280, 1280)]:
        assert frame_too_small(w, h) is False, (w, h)


def test_tile_starts_never_negative():
    # Every dimension, including sub-640, must yield only non-negative starts.
    for size in [50, 107, 533, 639, 640, 800, 1280, 2731, 4096, 8256]:
        starts = _tile_starts(size, 640, 512)
        assert starts, size
        assert all(s >= 0 for s in starts), (size, starts)


def test_tile_starts_unchanged_for_normal_sizes():
    # The fix must not alter tiling for any frame >= tile size.
    assert _tile_starts(4096, 640, 512) == [0, 512, 1024, 1536, 2048, 2560, 3072, 3456]
    assert _tile_starts(2731, 640, 512) == [0, 512, 1024, 1536, 2048, 2091]
    assert _tile_starts(640, 640, 512) == [0]
