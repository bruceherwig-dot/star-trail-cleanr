"""The first or last frame of a set must not copy the neighbour's own plane trail.

Found 2026-11-04 on Joshua Tree frame 3033, the last frame of its set. A plane trail sat in nearly
the same place in frames 3032 and 3033. A normal frame has two neighbours, and when both carry the
trail the repair reaches further out for a clean frame. An edge frame has one neighbour, that test
needed two, so it never fired and 3032's masked trail was pasted straight into 3033 (a pixel at 143
where the original peaked at 96).

The rule these tests pin: where the one neighbour's own mask says "trail", the edge frame must borrow
from a cleaner frame further out on the same side, or fill with local sky, never paste the trail back.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.repair import repair_frame  # noqa: E402

H, W = 240, 320
SKY = 40


def _sky(seed):
    rng = np.random.default_rng(seed)
    f = np.full((H, W, 3), SKY, np.float32) + rng.normal(0, 2, (H, W, 3))
    return np.clip(f, 0, 255).astype(np.uint8)


def _trail_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (40, 60), (280, 150), 255, 9)
    return m


def _with_trail(seed, mask):
    f = _sky(seed)
    f[mask > 0] = (200, 210, 250)
    return f


def _run(trail_in):
    """A 4-frame set. trail_in = which of the frames 0..3 carry the trail. Frame 3 (the last) is repaired."""
    mask = _trail_mask()
    frames = [_with_trail(i, mask) if i in trail_in else _sky(i) for i in range(4)]
    masks = [mask if i in trail_in else None for i in range(4)]
    out = repair_frame(frames[3], mask, 3, frames, neighbor_masks=masks, polygon_segs=[mask])
    return out, mask


def _level_in_mask(out, mask):
    return float(out[mask > 0].max(axis=1).mean())


def test_the_neighbours_trail_is_not_pasted_into_the_last_frame():
    """Frames 2 and 3 carry the trail, frame 1 is clean: the repair must reach back to frame 1."""
    out, mask = _run({2, 3})
    assert _level_in_mask(out, mask) < SKY + 25, "the last frame copied the neighbour's trail"


def test_with_no_clean_frame_the_fill_is_sky_not_the_trail():
    """Every earlier frame carries the trail: nothing clean to borrow, so local sky, not the trail."""
    out, mask = _run({0, 1, 2, 3})
    assert _level_in_mask(out, mask) < SKY + 25, "the last frame copied the neighbour's trail"


def test_the_first_frame_is_protected_the_same_way():
    mask = _trail_mask()
    trail_in = {0, 1}
    frames = [_with_trail(i, mask) if i in trail_in else _sky(i) for i in range(4)]
    masks = [mask if i in trail_in else None for i in range(4)]
    out = repair_frame(frames[0], mask, 0, frames, neighbor_masks=masks, polygon_segs=[mask])
    assert _level_in_mask(out, mask) < SKY + 25, "the first frame copied the neighbour's trail"


def test_a_clean_neighbour_is_still_borrowed_as_before():
    """The neighbour does not carry the trail here: nothing changes, the result is that neighbour's sky."""
    out, mask = _run({3})
    assert abs(_level_in_mask(out, mask) - SKY) < 8
