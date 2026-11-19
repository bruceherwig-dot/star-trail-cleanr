"""The large-square fix drops a fat detection when thin pieces already cover it. Only pieces pointing a
DIFFERENT way may count (the arms of a crossing). A long plane line's own pieces point the same way and used to
"cover" the whole glowing line, so its middle was left in the picture (Joshua Tree 80 sec, frame 2963).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.trail_grouper import detection_props, thin_fat_detections  # noqa: E402

H, W = 4000, 6000          # a 24 MP frame, so the pixel sizes below are the real ones


def _det(draw):
    m = np.zeros((H, W), np.uint8)
    draw(m)
    p = detection_props(m, min_aspect=0.0, min_area=0)
    assert p is not None
    return p


def _rect(x0, y0, x1, y1):
    return lambda m: cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)


def _line(p0, p1, thick):
    return lambda m: cv2.line(m, p0, p1, 255, thick)


def test_a_long_fat_line_covered_only_by_its_own_pieces_is_kept():
    fat = _det(_rect(1000, 2000, 1800, 2120))                    # 800 x 120: just over the fat cutoff (100)
    own = [_det(_rect(1000 + i * 160, 2030, 1150 + i * 160, 2090)) for i in range(5)]   # thin (60), same direction, cover most of it
    out, stats = thin_fat_detections([fat] + own, H, W)
    assert stats["dropped_covered"] == 0, stats
    assert len(out) >= len(own) + 1        # the fat line is still represented (kept, or cut into a band)


def test_a_fat_junction_covered_by_crossing_arms_is_still_dropped():
    fat = _det(_rect(2000, 1900, 2300, 2150))                    # a 300 x 250 junction blob
    arm1 = _det(_line((1500, 1500), (2800, 2600), 40))           # two thin arms crossing through it
    arm2 = _det(_line((1500, 2600), (2800, 1500), 40))
    out, stats = thin_fat_detections([fat, arm1, arm2], H, W)
    # The junction blob goes away one way or another: dropped as redundant, dropped as covered, or cut thin.
    assert stats["dropped_redundant"] + stats["dropped_covered"] + stats["thinned"] >= 1, stats
    assert all(o is not fat for o in out)
