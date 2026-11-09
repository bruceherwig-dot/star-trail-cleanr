"""Step 4 Frame Range: leave out frames at either end of the sequence.

Background, 2026-11-09 (Katrina Brown folder): a stray 20 mm foreground shot,
IMG_9254, sat in front of a 16 mm sequence. Step 4 had no way to skip it, and the
mask editor opened on it (first file alphabetically), so the mask was painted on the
wrong horizon and ~400 px of real sky went unsearched. The Frame Range bar fixes
the first half; the mask editor now opens on the first frame the run will use.

Also pinned here: the run must NOT add the start offset a second time when it hands
the engine the frame manifest (which is already cut to the chosen range).
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SRC = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")


def _frames(n):
    return [f"/x/IMG_{i:04d}.jpg" for i in range(n)]


def test_cut_from_the_start_skips_the_stray_first_frame():
    from modules.frame_list import run_selection
    fr = _frames(10)
    assert run_selection(fr, cut_start=1)[0] == "/x/IMG_0001.jpg"
    assert len(run_selection(fr, cut_start=1)) == 9


def test_cut_from_the_end_and_both_ends():
    from modules.frame_list import run_selection
    fr = _frames(10)
    assert run_selection(fr, cut_end=2)[-1] == "/x/IMG_0007.jpg"
    both = run_selection(fr, cut_start=1, cut_end=2)
    assert both[0] == "/x/IMG_0001.jpg" and both[-1] == "/x/IMG_0007.jpg"
    assert len(both) == 7


def test_the_number_counts_from_the_start_grip():
    """'20 frames from the middle': the cap is applied AFTER the bar."""
    from modules.frame_list import run_selection
    fr = _frames(100)
    sel = run_selection(fr, cut_start=40, limit="20")
    assert len(sel) == 20
    assert sel[0] == "/x/IMG_0040.jpg" and sel[-1] == "/x/IMG_0059.jpg"


def test_all_frames_and_blank_mean_no_cap():
    from modules.frame_list import run_selection
    fr = _frames(30)
    assert len(run_selection(fr, limit="All Frames")) == 30
    assert len(run_selection(fr, limit="")) == 30
    assert len(run_selection(fr, limit="junk")) == 30


def test_a_trim_that_would_leave_under_three_frames_is_ignored():
    from modules.frame_list import trim_from_end, run_selection
    fr = _frames(5)
    assert trim_from_end(fr, 3) == fr          # would leave 2
    assert len(trim_from_end(fr, 2)) == 3      # leaves exactly 3: allowed
    assert run_selection(fr, cut_end=4) == fr


def test_ordered_frame_list_matches_the_runs_order_and_drops_twins():
    """Folder order = number-aware names, one file per frame."""
    from modules.frame_list import ordered_frame_list
    import cv2
    import numpy as np
    d = Path(tempfile.mkdtemp())
    for name in ("IMG_10.jpg", "IMG_2.jpg", "IMG_1.jpg", "IMG_1.tif"):
        cv2.imwrite(str(d / name), np.zeros((8, 8, 3), np.uint8))
    got = [Path(p).name for p in ordered_frame_list(str(d), prefer_raw=False)]
    assert got[:3] == ["IMG_1.tif", "IMG_2.jpg", "IMG_10.jpg"] or \
        got[:3] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"], got
    assert len(got) == 3, f"the JPG/TIFF twin of IMG_1 was not removed: {got}"


def test_empty_folder_gives_an_empty_list():
    from modules.frame_list import ordered_frame_list
    assert ordered_frame_list(tempfile.mkdtemp()) == []


def test_run_does_not_add_the_start_offset_twice():
    """The manifest is cut to the range before the engine reads it, so the engine's
    --start must NOT include frame_start when a manifest exists."""
    assert 'abs_start = (start if getattr(self, "_frame_manifest", None)' in SRC, (
        "the batch start offset no longer skips frame_start when the manifest is "
        "in use -- every frame before the chosen start would be skipped twice")


def test_the_gui_hands_the_bar_values_to_the_run():
    assert "frame_start = self._step4_range.start()" in SRC
    assert "trim_end = self._step4_range.end()" in SRC
    assert "trim_end=trim_end," in SRC


def test_the_old_dev_boxes_are_gone():
    assert "_dev_start_frame" not in SRC and "_dev_end_frame" not in SRC, (
        "a leftover reference to the retired dev Start/End boxes would crash a run")


def test_trim_is_applied_before_the_manifest_is_written():
    """Same rule as test_manifest_matches_plan: the manifest must describe the
    trimmed list, or the engine and the batch plan disagree about frame 0."""
    trim = SRC.find("frames = trim_from_end(frames, self.trim_end)")
    write = SRC.find("write_manifest(_mf.name, frames)")
    assert 0 < trim < write


def test_mask_editor_opens_on_the_first_frame_the_run_will_use():
    assert "self._mask_window.load_frames(frames, _open_at)" in SRC
    assert "self._mask_window.load_frames(frames, 0)" not in SRC
