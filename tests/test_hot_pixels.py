"""Hot-pixel detector finds planted defects and inpaints them."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.hot_pixels import build_hot_pixel_map, fix_hot_pixels


def _synthetic_frames(n=5, shape=(128, 128), hot_coord=(40, 60), background=15):
    """Generate N BGR frames with a dark sky background and a single hot red pixel at hot_coord.
    Frames are identical except for low random noise, so the hot pixel is the only temporally-consistent bright spot."""
    frames = []
    rng = np.random.default_rng(42)
    y, x = hot_coord
    for i in range(n):
        f = np.full((*shape, 3), background, dtype=np.uint8)
        f += rng.integers(0, 3, size=f.shape, dtype=np.uint8)
        f[y, x, 2] = 250
        f[y, x, 1] = 30
        f[y, x, 0] = 30
        frames.append(f)
    return frames


def test_build_hot_pixel_map_finds_planted_defect():
    frames = _synthetic_frames()
    mask = build_hot_pixel_map(frames)
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert mask[40, 60] > 0, "planted red hot pixel not detected"


def test_build_hot_pixel_map_clean_frames_have_empty_mask():
    rng = np.random.default_rng(7)
    frames = [rng.integers(10, 25, size=(128, 128, 3), dtype=np.uint8) for _ in range(5)]
    mask = build_hot_pixel_map(frames)
    hit_count = int((mask > 0).sum())
    assert hit_count < 5, f"false positives on clean frames: {hit_count}"


def test_fix_hot_pixels_repairs_planted_defect():
    frames = _synthetic_frames()
    repaired = fix_hot_pixels(frames)
    assert len(repaired) == len(frames)
    for f in repaired:
        red_at_defect = int(f[40, 60, 2])
        assert red_at_defect < 100, f"hot pixel not repaired, red={red_at_defect}"
