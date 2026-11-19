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


def _qt_app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _Pt:
    """The little bit of a mouse event the bar reads: where the pointer is."""
    def __init__(self, x):
        self._x = x

    def position(self):
        from PySide6.QtCore import QPointF
        return QPointF(self._x, 0)


def _x_for(bar, frames):
    return frames / bar._total * bar._span() + bar._GRIP_W / 2


def test_slide_mode_moves_the_whole_window_with_the_left_grip():
    """Step 4: with 100 frames chosen, dragging the left grip must slide the
    100-frame window (right grip travelling with it), never squeeze it."""
    _qt_app()
    import star_trail_cleanr as S
    bar = S.FrameRangeSlider(708, 3)
    bar.resize(1000, 36)
    bar.set_slide(True)
    bar.set_range(142, 708 - 142 - 100)
    bar._drag = "start"
    bar.mouseMoveEvent(_Pt(_x_for(bar, 300)))
    assert bar.start() == 300 and bar.used() == 100 and bar.end() == 308
    bar.mouseMoveEvent(_Pt(_x_for(bar, 5000)))          # dragged past the end
    assert bar.start() == 608 and bar.end() == 0 and bar.used() == 100


def test_without_slide_mode_the_grips_move_independently():
    """The Star Trail and Timelapse tabs (and Step 4 on All Frames) are unchanged."""
    _qt_app()
    import star_trail_cleanr as S
    bar = S.FrameRangeSlider(572, 20)
    bar.resize(1000, 36)
    bar.set_range(10, 26)
    bar._drag = "start"
    bar.mouseMoveEvent(_Pt(_x_for(bar, 50)))
    assert bar.start() == 50 and bar.end() == 26 and bar.used() == 496


def test_dropdown_and_bar_are_one_selection():
    """The number in the dropdown puts the right grip that many frames after the
    left; dragging the right grip changes the number; All Frames sends it to the
    far end. Built on the real window; the saved dropdown value is put back."""
    import os, time
    _qt_app()
    sys.path.insert(0, str(REPO / "tests"))
    import cv2
    import numpy as np
    import qt_offscreen
    import star_trail_cleanr as S
    from PySide6.QtWidgets import QApplication
    d = Path(tempfile.mkdtemp())
    for i in range(40):
        cv2.imwrite(str(d / f"IMG_{9254 + i:04d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    saved = S.SETTINGS.value("frame_limit", "20")
    # Closing a MainWindow saves its size into the person's real settings, and an
    # offscreen window is 1x1 -- so put the saved window size back afterwards.
    saved_geo = S.SETTINGS.value("window_geometry")
    app = QApplication.instance()
    w = S.MainWindow()
    try:
        w._frame_limit.setCurrentText("10")
        w._folder_input.setText(str(d))
        w._update_frame_count()
        t = time.time()
        while not w._step4_ordered and time.time() - t < 15:
            app.processEvents(); time.sleep(0.05)
        bar = w._step4_range
        assert (bar.start(), bar.end()) == (0, 30)           # 10 frames from the front
        w._frame_limit.setCurrentText("15")
        assert (bar.start(), bar.end()) == (0, 25)           # number -> right grip
        bar.set_range(0, 20)                                 # right grip dragged
        assert w._frame_limit.currentText() == "20"          # grip -> number
        bar.set_range(0, 0)
        assert w._frame_limit.currentText() == "All Frames"  # far end = All Frames
        w._frame_limit.setCurrentText("10")
        bar.set_range(5, 25)                                 # left grip slid
        assert w._frame_limit.currentText() == "10"          # number untouched
        assert "IMG_9259.jpg to IMG_9268.jpg" in w._step4_names.text()
        w._frame_limit.setCurrentText("All Frames")
        assert (bar.start(), bar.end()) == (5, 0)            # keeps the start, opens the end
    finally:
        qt_offscreen.join_qt_threads(w, app)
        S.SETTINGS.setValue("frame_limit", saved)
        if saved_geo is None:
            S.SETTINGS.remove("window_geometry")
        else:
            S.SETTINGS.setValue("window_geometry", saved_geo)


def test_step4_row_says_skip_first_and_skip_last():
    _qt_app()
    import star_trail_cleanr as S
    row = S.FrameRangeRow(100, 708, 3, skip_words=True)
    row.slider.set_range(158, 0)
    assert row._lo.text() == "Skip first 158" and row._hi.text() == "Skip last 0"
    old = S.FrameRangeRow(90, 572, 20)                       # the other tabs
    old.slider.set_range(10, 0)
    assert old._lo.text() == "-10" and old._hi.text() == "0"


def test_step4_bar_has_no_count_box_and_other_tabs_keep_theirs():
    """The count box sat over the right grip on Step 4, which says the number on the
    line under the bar instead. The Star Trail and Timelapse tabs keep the box."""
    _qt_app()
    import star_trail_cleanr as S
    assert S.FrameRangeRow(100, 708, 3, skip_words=True).slider._show_count is False
    assert S.FrameRangeRow(90, 572, 20).slider._show_count is True


def test_window_grows_when_the_frame_range_row_appears():
    """2026-11-09: the window's minimum height was measured at first show, before
    the folder's frames loaded and the Frame Range row and file line appeared. The
    page then outgrew the window and the bottom buttons were clipped. The window
    must re-measure when the row appears (and only ever grow)."""
    import time
    _qt_app()
    sys.path.insert(0, str(REPO / "tests"))
    import cv2
    import numpy as np
    import qt_offscreen
    import star_trail_cleanr as S
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication

    class _Tall:                       # a screen tall enough that the 90% cap never bites
        def availableGeometry(self):
            return QRect(0, 0, 3000, 4000)

    d = Path(tempfile.mkdtemp())
    for i in range(30):
        cv2.imwrite(str(d / f"IMG_{i:04d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    saved = S.SETTINGS.value("frame_limit", "20")
    saved_geo = S.SETTINGS.value("window_geometry")
    app = QApplication.instance()
    real_screen = S.QApplication.primaryScreen
    S.QApplication.primaryScreen = staticmethod(lambda: _Tall())
    w = S.MainWindow()
    try:
        w.show()
        t = time.time()
        while time.time() - t < 1.0:                 # let the first-show measure run
            app.processEvents(); time.sleep(0.02)
        # The window restores the person's last folder, so the row may already be
        # showing. Start from the state the bug needs: no folder, row hidden, and
        # the height measured in that state, exactly as at first show.
        w._folder_input.setText("")
        w._update_frame_count()
        for _ in range(20):
            app.processEvents(); time.sleep(0.02)
        assert w._step4_row.isHidden()
        w._lock_min_height()
        before = w.minimumHeight()
        w._folder_input.setText(str(d))
        w._update_frame_count()
        t = time.time()
        while time.time() - t < 15 and not (w._step4_ordered and not w._step4_row.isHidden()):
            app.processEvents(); time.sleep(0.05)
        t = time.time()
        while time.time() - t < 1.0:                 # let the re-measure run
            app.processEvents(); time.sleep(0.02)
        assert not w._step4_row.isHidden(), "the Frame Range row never appeared"
        after = w.minimumHeight()
        assert after > before, (before, after,
            "the window did not re-measure when the Frame Range row appeared")
        assert w.height() >= after
    finally:
        S.QApplication.primaryScreen = real_screen
        qt_offscreen.join_qt_threads(w, app)
        S.SETTINGS.setValue("frame_limit", saved)
        if saved_geo is None:
            S.SETTINGS.remove("window_geometry")
        else:
            S.SETTINGS.setValue("window_geometry", saved_geo)
