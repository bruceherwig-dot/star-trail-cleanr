"""Smoke tests for the large-square fix in tools/masks_to_labelme.py.

The old route draws ONE box around every touching blob of a mask, so two crossing
trails became one huge box. The fixed route keeps one shape per app polygon (the
<frame>_polys.json the app writes beside each mask). tools/ is not tracked by git,
so if the tool is not on this machine the tests quietly do nothing.
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
TOOL = REPO / "tools" / "masks_to_labelme.py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

H, W = 1000, 1400
# two thin trails crossing in an X (corner lists, as the app saves them)
TRAIL_A = [[100, 300], [1200, 700], [1190, 745], [90, 345]]
TRAIL_B = [[100, 750], [1200, 250], [1190, 205], [90, 705]]


def _short_side(points):
    return min(cv2.minAreaRect(np.array(points, np.float32).reshape(-1, 1, 2))[1])


def test_crossing_trails_stay_two_thin_boxes():
    if not TOOL.exists():
        return
    import masks_to_labelme as m
    mask = np.zeros((H, W), np.uint8)
    for t in (TRAIL_A, TRAIL_B):
        cv2.fillPoly(mask, [np.array(t, np.int32)], 255)
    old = m.mask_to_shapes(mask)
    assert len(old) == 1 and _short_side(old[0]["points"]) > 200, "setup: the old route should fuse the X"
    new = m.shapes_from_polygons([TRAIL_A, TRAIL_B], H, W)
    assert len(new) == 2, "one shape per app polygon expected, got %d" % len(new)
    assert all(_short_side(s["points"]) < 80 for s in new), "a fixed-route box is still fat"


def test_shapes_are_clipped_to_the_image():
    if not TOOL.exists():
        return
    import masks_to_labelme as m
    shapes = m.shapes_from_polygons([[[-50, 100], [400, 100], [400, 140], [-50, 140]]], H, W)
    assert len(shapes) == 1
    xs = [p[0] for p in shapes[0]["points"]]
    assert min(xs) >= 0 and max(xs) <= W - 1


def test_load_app_polygons_uses_a_matching_file_only():
    if not TOOL.exists():
        return
    import masks_to_labelme as m
    with tempfile.TemporaryDirectory() as t:
        mask_path = Path(t) / "f1.png"
        cv2.imwrite(str(mask_path), np.zeros((H, W), np.uint8))
        assert m.load_app_polygons(mask_path, H, W) is None, "no polygon file means fall back"
        pj = Path(t) / "f1_polys.json"
        pj.write_text(json.dumps({"width": W, "height": H,
                                  "polygons": [{"id": 0, "corners": TRAIL_A}]}))
        assert m.load_app_polygons(mask_path, H, W) == [TRAIL_A]
        assert m.load_app_polygons(mask_path, H + 1, W) is None, "a size mismatch must fall back"
        pj.write_text("not json")
        assert m.load_app_polygons(mask_path, H, W) is None, "a broken file must fall back"
