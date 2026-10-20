"""The ground is not scenery-painted: textured static ground must not be treated as stuck pixels.

King of Wings (ISO 10000, 2026-10-19): the ground's texture flagged 16,707
"stuck" spots, each painted over with a smooth blob in every frame, which the
star-trail stack turned into dark dots. A real defect on a quiet ground must
still be repaired.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.hot_pixels import build_hot_pixel_map, ground_looks_like_scenery


def _mask(h, w, ground_from_row):
    m = np.zeros((h, w), dtype=np.uint8)
    m[ground_from_row:] = 255
    return m


def test_noisy_static_ground_is_treated_as_scenery():
    """Same coloured speckle in every frame on the ground, quiet sky: must be left alone."""
    rng = np.random.default_rng(3)
    h, w = 600, 800
    ground = rng.integers(0, 130, size=(h // 3, w, 3), dtype=np.uint8)   # identical every frame
    frames = []
    for _ in range(6):
        sky = rng.integers(8, 14, size=(h - h // 3, w, 3), dtype=np.uint8)
        frames.append(np.vstack([sky, ground]))
    hot = build_hot_pixel_map(frames)
    fg = _mask(h, w, h - h // 3)
    scenery, flags, g_dens, s_dens = ground_looks_like_scenery(hot, fg)
    assert flags > 500, f"test setup did not make the ground look stuck ({flags} flags)"
    assert scenery, f"noisy ground was not recognised as scenery ({flags} flags, {g_dens:.0f}/Mpx vs sky {s_dens:.0f}/Mpx)"


def test_few_real_defects_on_ground_are_still_repaired():
    hot = np.zeros((600, 800), dtype=np.uint8)
    fg = _mask(600, 800, 400)
    for y, x in [(420, 100), (450, 300), (500, 700)]:
        hot[y, x] = 255
    scenery, flags, _, _ = ground_looks_like_scenery(hot, fg)
    assert not scenery and flags == 3


def test_defects_spread_evenly_over_sky_and_ground_are_still_repaired():
    """A sensor with many real defects has them everywhere, so the ground is not special."""
    rng = np.random.default_rng(5)
    h, w = 1000, 2000
    hot = (rng.random((h, w)) < 0.001).astype(np.uint8) * 255
    fg = _mask(h, w, 500)
    scenery, flags, g_dens, s_dens = ground_looks_like_scenery(hot, fg)
    assert flags > 500
    assert not scenery, f"evenly spread defects wrongly called scenery ({g_dens:.0f} vs {s_dens:.0f}/Mpx)"


def test_no_ground_or_empty_map_is_not_scenery():
    fg = np.zeros((100, 100), dtype=np.uint8)
    hot = np.full((100, 100), 255, dtype=np.uint8)
    assert not ground_looks_like_scenery(hot, fg)[0]
    assert not ground_looks_like_scenery(np.zeros((100, 100), np.uint8), _mask(100, 100, 50))[0]
