"""Smoke tests for the large-square fix (thin_fat_detections in modules/trail_grouper.py).

Where two trails cross, the model's junction detection is an X of thin lines that
reads as one wide blob; boxing it gives a big square. These tests build that
shape synthetically and check the fix cuts it into thin bands, drops a stubby
junction piece that other trails already cover, and never touches a long thin
trail. No model and no image files needed; runs in a fraction of a second.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from skimage.measure import label, regionprops  # noqa: E402
from modules.trail_grouper import thin_fat_detections  # noqa: E402

H, W = 3000, 4000          # about 12 MP; the fat cutoff scales to about 71 px here


def _det(draw):
    """Detection dict like the pipeline builds. detection_props() itself rejects fat blobs
    (the pipeline keeps them when its fat-blob gate is off), so build it directly."""
    m = np.zeros((H, W), np.uint8)
    draw(m)
    ys, xs = np.where(m > 0)
    rp = max(regionprops(label(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])),
             key=lambda r: r.area)
    _, vecs = np.linalg.eigh(rp.inertia_tensor)
    return {"centroid": np.array([rp.centroid[0] + ys.min(), rp.centroid[1] + xs.min()]),
            "u": vecs[:, 0], "minor": rp.axis_minor_length, "major": rp.axis_major_length,
            "area": rp.area, "coords": np.column_stack((ys, xs)), "conf": 0.9}


def _line(p1, p2, thick):
    return lambda m: cv2.line(m, p1, p2, 255, thick)


def _x_blob(m):
    cv2.line(m, (1000, 1000), (1700, 1400), 255, 28)   # one arm of the X
    cv2.line(m, (1000, 1400), (1700, 1000), 255, 28)   # the other arm


def _short_side(d):
    return min(cv2.minAreaRect(d["coords"][:, ::-1].astype(np.float32))[1])


def test_x_blob_is_cut_into_two_thin_bands():
    d = _det(_x_blob)
    assert d["minor"] > 71, "test setup: the X must read as a fat blob (minor %.0f)" % d["minor"]
    out, stats = thin_fat_detections([d], H, W)
    assert stats["thinned"] == 1, stats
    assert len(out) == 2, "an X should become two bands, got %d" % len(out)
    for b in out:
        assert b["minor"] < 71, "band still fat (minor %.0f)" % b["minor"]
        assert b["major"] > 500, "band lost its length (major %.0f)" % b["major"]


def test_long_thin_trails_are_never_touched():
    a = _det(_line((200, 600), (1900, 700), 30))
    b = _det(_line((900, 300), (1300, 1100), 30))      # crosses a, points a different way
    out, stats = thin_fat_detections([a, b], H, W)
    assert len(out) == 2 and out[0] is a and out[1] is b and sum(stats.values()) == 0, stats


def test_stubby_piece_covered_by_other_trails_is_dropped():
    a = _det(_line((200, 1000), (1800, 1000), 30))
    b = _det(_line((600, 400), (1200, 1600), 30))
    # a short, wide piece sitting on the junction, pointing between the two
    stub = _det(lambda m: cv2.ellipse(m, (900, 1000), (130, 55), 20, 0, 360, 255, -1))
    assert stub["minor"] > 1.5 * 30 and stub["major"] / stub["minor"] < 4
    out, stats = thin_fat_detections([a, b, stub], H, W)
    assert stats["dropped_redundant"] == 1, stats
    assert any(x is a for x in out) and any(x is b for x in out), "a real trail was dropped"
    assert not any(x is stub for x in out), "the junction piece was kept"


def test_fat_blob_with_no_neighbours_is_not_silently_deleted():
    d = _det(_x_blob)
    out, _ = thin_fat_detections([d], H, W)
    assert len(out) >= 1, "a lone fat blob must become bands, not vanish"


def test_empty_and_clean_lists_pass_through():
    assert thin_fat_detections([], H, W)[0] == []
    a = _det(_line((200, 600), (1900, 700), 30))
    out, stats = thin_fat_detections([a], H, W)
    assert len(out) == 1 and out[0] is a and sum(stats.values()) == 0
