"""The still-vs-moving routing must report how much it held in place.

This is the protection that keeps a static scene -- a wall, a horizon, a trunk -- from
being displaced by the star slide. It is what stands between a fixed building and a
copy of itself a few pixels over.

Until 2026-09-06 it counted nothing. The run log carried a field called `still_trail_px`
that belonged to the disabled warm-pixel scrub and was hard-wired to zero, so a log from
a run with visible cloning looked identical to a log from a clean one. Diagnosing the
UofR Memorial Chapel report meant reading the source instead of reading the log.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.repair import repair_frame

H = W = 120


def _scene(star_x):
    """Sky, a static bright wall down the middle, and one star that moves."""
    f = np.full((H, W, 3), 40, np.uint8)
    f[:, 55:65] = 200                     # the wall: identical in every frame
    f[20:24, star_x:star_x + 4] = 255     # the star: somewhere different each frame
    return f


def _segments(dbg):
    return [s for c in dbg.get("components", []) for s in c.get("segments", [])]


def test_the_counter_reports_what_the_routing_held():
    frames = [_scene(10), _scene(30), _scene(50)]
    mask = np.zeros((H, W), np.uint8)
    mask[40:70, 30:90] = 255              # a trail crossing the wall
    clean = np.zeros((H, W), np.uint8)

    dbg = {}
    repair_frame(frames[1], mask, 1, frames,
                 neighbor_masks=[clean, mask, clean], debug_out=dbg)

    segs = _segments(dbg)
    assert segs, "the repair should have logged at least one segment"
    assert all("still_held_px" in s for s in segs), \
        "every repaired segment must report still_held_px"
    assert any(s["still_held_px"] > 0 for s in segs), (
        "both neighbours agree across a static wall, so the routing must hold pixels "
        "in place and say how many")


def test_the_dead_counter_is_gone():
    """`still_trail_px` was a permanent zero from a disabled feature. It read like proof
    the routing never fires, and it cost real diagnosis time. It must not come back."""
    frames = [_scene(10), _scene(30), _scene(50)]
    mask = np.zeros((H, W), np.uint8)
    mask[40:70, 30:90] = 255
    clean = np.zeros((H, W), np.uint8)

    dbg = {}
    repair_frame(frames[1], mask, 1, frames,
                 neighbor_masks=[clean, mask, clean], debug_out=dbg)

    for s in _segments(dbg):
        assert "still_trail_px" not in s, (
            "still_trail_px belonged to the disabled warm-pixel scrub and was always 0; "
            "use still_held_px, which counts the still routing itself")

    src = (REPO / "modules" / "run_logger.py").read_text(encoding="utf-8")
    assert "still_held_px" in src, "the run log legend must explain the real counter"
    assert '"still_trail_px"' not in src, \
        "the legend must not still document the dead field as a log key"
