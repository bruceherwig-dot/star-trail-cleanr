"""The Star Trail tab's Frame Range: cut frames off either end, never below 20.

The numbers a person sees count every shot found; the automatic first-3-and-last-3
skip is not part of them and must stay applied underneath, always.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import make_share_clip as msc


def _names(n):
    return [f"f{i:03d}.jpg" for i in range(n)]


def test_no_trim_leaves_the_list_alone():
    names = _names(100)
    assert msc._apply_trim(names, 0, 0) == names


def test_trim_cuts_the_requested_frames_from_each_end():
    names = _names(100)
    kept = msc._apply_trim(names, 10, 5)
    assert kept == names[10:95]
    assert len(kept) == 85


def test_the_automatic_test_shot_skip_is_still_applied_underneath():
    """_list_frames already dropped the first/last 3, so a trim of 0 must not
    bring them back, and a trim counts from what is left."""
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        for i in range(30):
            Image.new("RGB", (4, 4)).save(Path(d) / f"IMG_{i:03d}.jpg")
        listed = msc._list_frames(d)
    assert len(listed) == 30 - msc.SKIP_FIRST - msc.SKIP_LAST
    assert listed[0] == "IMG_003.jpg" and listed[-1] == "IMG_026.jpg"
    kept = msc._apply_trim(listed, 2, 0)
    assert kept[0] == "IMG_005.jpg"


def test_a_trim_that_would_leave_fewer_than_20_is_refused():
    """Judged on the count the person sees: shots listed plus the hidden 3+3."""
    names = _names(52)           # a person sees 58
    # 58 - 38 = 20 is allowed; 58 - 39 = 19 is not.
    assert len(msc._apply_trim(names, 20, 18)) == 52 - 38
    assert msc._apply_trim(names, 20, 19) == names


def test_the_slider_stops_at_the_floor_and_cannot_cross():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import star_trail_cleanr as S
    sl = S.FrameRangeSlider(total=100, min_keep=20)
    sl.set_range(50, 50)
    assert sl.used() == 20, "both grips together must stop at the floor"
    assert (sl.start(), sl.end()) == (50, 30)
    sl.set_range(0, 0)
    assert sl.used() == 100
    sl.set_range(-5, -5)
    assert (sl.start(), sl.end()) == (0, 0), "grips never go below 0"


def test_the_trim_reaches_the_program_as_command_line_options():
    import subprocess
    out = subprocess.run([sys.executable, str(REPO / "make_share_clip.py"), "--help"],
                         capture_output=True, text=True).stdout
    assert "--trim-start" in out and "--trim-end" in out
