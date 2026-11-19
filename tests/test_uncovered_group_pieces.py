"""A long piece of a detection that its fitted polygon leaves out must get a polygon of its own.

Found 2026-11-04 on Joshua Tree frame 3030. The model drew one outline around a steep diagonal and a
bright streak joined to it by a thin neck. A group gets ONE fitted polygon, along its dominant
direction, so the lobe lay outside it: 8,147 outlined pixels, 462 reached the cleaning mask, nothing
was logged. uncovered_pieces() finds long pieces no polygon covers; ordinary outline slack is ignored.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.trail_grouper import uncovered_pieces  # noqa: E402

H, W = 600, 700


def _coords(mask):
    ys, xs = np.nonzero(mask)
    return np.column_stack([ys, xs])


def _thick(mask, p0, p1, width):
    cv2.line(mask, p0, p1, 1, width)


def _poly_around(p0, p1, half):
    """A rectangle (corner list) of the given half-width around a straight line."""
    a = np.array(p0, float)
    b = np.array(p1, float)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]]) * half
    return [(a + n).round().astype(int).tolist(), (b + n).round().astype(int).tolist(),
            (b - n).round().astype(int).tolist(), (a - n).round().astype(int).tolist()]


def test_a_side_lobe_outside_the_fitted_polygon_gets_its_own_box():
    blob = np.zeros((H, W), np.uint8)
    _thick(blob, (100, 560), (420, 40), 30)          # the long diagonal
    _thick(blob, (300, 330), (600, 250), 34)         # a bright streak off to the side, joined by overlap
    polygon = _poly_around((100, 560), (420, 40), 20)    # one polygon along the dominant direction
    boxes = uncovered_pieces(_coords(blob), [polygon], H, W)
    assert boxes, "the side lobe was left outside every polygon"
    cov = np.zeros((H, W), np.uint8)
    for b in boxes + [polygon]:
        cv2.fillPoly(cov, [np.array(b, np.int32).reshape(-1, 1, 2)], 1)
    line = np.zeros((H, W), np.uint8)
    cv2.line(line, (330, 330), (570, 255), 1, 1)
    assert cov[line > 0].mean() > 0.9, "the streak inside the lobe is still uncovered"


def test_a_group_its_polygon_already_covers_gets_nothing_extra():
    blob = np.zeros((H, W), np.uint8)
    _thick(blob, (100, 560), (420, 40), 30)
    polygon = _poly_around((100, 560), (420, 40), 20)
    assert uncovered_pieces(_coords(blob), [polygon], H, W) == []


def test_thin_outline_slack_is_not_turned_into_boxes():
    """A polygon a little narrower than the outline leaves only fringe, which must be ignored."""
    blob = np.zeros((H, W), np.uint8)
    _thick(blob, (100, 560), (420, 40), 34)
    polygon = _poly_around((100, 560), (420, 40), 14)    # slack of a few px each side
    assert uncovered_pieces(_coords(blob), [polygon], H, W) == []


def test_nothing_to_do_without_polygons():
    blob = np.zeros((H, W), np.uint8)
    _thick(blob, (100, 560), (420, 40), 30)
    assert uncovered_pieces(_coords(blob), [], H, W) == []
