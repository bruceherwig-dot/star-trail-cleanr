"""image_size must report UPRIGHT dimensions so the pre-run scan can tell a
portrait frame from a landscape one. A sideways shot stores the same pixel
dimensions as a level shot and only diverges once rotated upright, which is why
mixed-orientation folders used to slip past the pre-run check and fail mid-run.
"""
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image
from modules.io_safe import image_size


def _make_jpeg(path, w, h, orientation):
    """Write a w x h JPEG carrying the given EXIF Orientation tag."""
    img = Image.new("RGB", (w, h), (10, 20, 30))
    ex = img.getexif()
    ex[0x0112] = orientation
    img.save(path, exif=ex.tobytes())


def test_level_frame_keeps_stored_size():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "level.jpg")
        _make_jpeg(p, 40, 20, 1)
        assert image_size(p) == (40, 20)


def test_rotated_90cw_reports_upright_portrait():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "port6.jpg")
        _make_jpeg(p, 40, 20, 6)  # stored landscape, upright is portrait
        assert image_size(p) == (20, 40)


def test_rotated_90ccw_reports_upright_portrait():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "port8.jpg")
        _make_jpeg(p, 40, 20, 8)
        assert image_size(p) == (20, 40)


def test_180_rotation_does_not_swap():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "flip180.jpg")
        _make_jpeg(p, 40, 20, 3)  # 180 deg keeps width/height
        assert image_size(p) == (40, 20)


def test_portrait_and_landscape_now_differ():
    """The core of the bug fix: two frames with identical stored dimensions but
    opposite orientation must report different sizes."""
    with tempfile.TemporaryDirectory() as d:
        land = os.path.join(d, "land.jpg")
        port = os.path.join(d, "port.jpg")
        _make_jpeg(land, 40, 20, 1)
        _make_jpeg(port, 40, 20, 6)
        assert image_size(land) != image_size(port)
