"""16-bit TIFF output is greyed out when every source frame is a JPEG.

WHY THE OPTION IS USELESS THERE. A JPEG holds 8 bits per channel and cannot hold
more. Asked for 16-bit output the engine simply scales those values up by 257 to
fill the range, so the result is a much larger file containing exactly the same
picture. Measured on a real 5472x3648 frame from Bruce's Joshua Tree sequence
(2026-08-30): 120 MB, exactly 256 distinct levels, and not one value in 59.8
million that 8 bits could not have held. That held on five frames, all of which
had real repairs in them, so even Star Bridge's blending stays on the 8-bit grid.
Roughly thirty times the disk for nothing, with nothing on screen saying so.

TIFF 8-BIT IS NOT TOUCHED. It earns its place from JPEG sources by not
re-compressing on the way out -- about half a level of difference on a real frame.

These tests drive the REAL QComboBox the window uses, not a stand-in, because the
point is that the actual Qt calls work. They build no MainWindow and start no
threads, so they stay fast and cannot leave a live thread at exit.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GUI = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")


def _combo():
    from PySide6.QtWidgets import QApplication, QComboBox
    QApplication.instance() or QApplication([])
    c = QComboBox()
    c.addItems(["JPG", "TIFF 8-bit", "TIFF 16-bit"])
    return c


class _Stub:
    """Only what _sync_format_options touches -- no window, no threads."""
    def __init__(self):
        import star_trail_cleanr as S
        self._format_combo = _combo()
        self._format_auto_change = False
        self._format_user_choice = None
        self._EIGHT_BIT_ONLY_EXTS = S.MainWindow._EIGHT_BIT_ONLY_EXTS
        self._sync = lambda folder: S.MainWindow._sync_format_options(self, folder)


def _folder(names):
    d = Path(tempfile.mkdtemp())
    for n in names:
        (d / n).write_bytes(b"x")           # nothing is opened; the name decides
    return str(d)


def _enabled(combo, text):
    from PySide6.QtCore import Qt
    item = combo.model().item(combo.findText(text))
    return bool(item.flags() & Qt.ItemIsEnabled)


def test_an_all_jpeg_folder_greys_out_16_bit():
    s = _Stub()
    s._sync(_folder(["a.jpg", "b.JPG", "c.jpeg"]))
    assert not _enabled(s._format_combo, "TIFF 16-bit"), (
        "16-bit is still offered for JPEG sources, where it costs about thirty "
        "times the disk and adds nothing")


def test_eight_bit_tiff_and_jpg_stay_available():
    """8-bit TIFF is the useful choice for JPEG sources -- it avoids
    re-compressing on the way out. Greying it out too would be wrong."""
    s = _Stub()
    s._sync(_folder(["a.jpg", "b.jpg"]))
    assert _enabled(s._format_combo, "TIFF 8-bit")
    assert _enabled(s._format_combo, "JPG")


def test_a_raw_folder_keeps_every_option():
    s = _Stub()
    s._sync(_folder(["a.cr2", "b.cr2"]))
    assert _enabled(s._format_combo, "TIFF 16-bit")


def test_one_raw_among_the_jpegs_is_enough_to_keep_16_bit():
    """A single frame that can carry real depth means 16-bit output would
    preserve something. Only an ENTIRELY 8-bit folder disables it."""
    s = _Stub()
    s._sync(_folder([f"f{i}.jpg" for i in range(20)] + ["odd.nef"]))
    assert _enabled(s._format_combo, "TIFF 16-bit"), (
        "one RAW frame's depth would be thrown away silently")


def test_a_tiff_folder_keeps_16_bit():
    """A TIFF may be either depth and the extension cannot tell us which, so it
    must never be assumed 8-bit."""
    s = _Stub()
    s._sync(_folder(["a.tif", "b.tif"]))
    assert _enabled(s._format_combo, "TIFF 16-bit")


def test_an_empty_or_missing_folder_proves_nothing():
    """Nothing is known yet, so nothing should be taken away."""
    s = _Stub()
    s._sync(_folder([]))
    assert _enabled(s._format_combo, "TIFF 16-bit")
    s._sync("/no/such/folder/anywhere")
    assert _enabled(s._format_combo, "TIFF 16-bit")
    s._sync("")
    assert _enabled(s._format_combo, "TIFF 16-bit")


def test_non_image_files_are_ignored():
    """A folder of JPEGs with a text file beside them is still a JPEG folder."""
    s = _Stub()
    s._sync(_folder(["a.jpg", "b.jpg", "notes.txt", "log.jsonl"]))
    assert not _enabled(s._format_combo, "TIFF 16-bit")


# ── the dropdown must never be left sitting on a disabled choice ────────────

def test_a_user_on_16_bit_is_moved_to_8_bit_tiff_not_to_jpg():
    """They asked for a TIFF. They should still get one."""
    s = _Stub()
    s._format_combo.setCurrentText("TIFF 16-bit")
    s._sync(_folder(["a.jpg"]))
    assert s._format_combo.currentText() == "TIFF 8-bit", (
        f"left on {s._format_combo.currentText()!r}")


def test_their_real_choice_comes_back_with_a_raw_folder():
    """Being moved off 16-bit for one folder must not cost them the setting."""
    s = _Stub()
    s._format_combo.setCurrentText("TIFF 16-bit")
    s._sync(_folder(["a.jpg"]))
    s._sync(_folder(["a.cr2"]))
    assert s._format_combo.currentText() == "TIFF 16-bit", (
        "their 16-bit preference was lost after one JPEG folder")


def test_a_choice_they_made_themselves_is_not_overwritten_later():
    """If they deliberately pick JPG while on a JPEG folder, moving to a RAW
    folder must not yank them back to 16-bit."""
    s = _Stub()
    s._sync(_folder(["a.jpg"]))
    s._format_combo.setCurrentText("JPG")
    s._sync(_folder(["a.cr2"]))
    assert s._format_combo.currentText() == "JPG"


def test_a_deliberate_choice_after_being_moved_off_16_bit_sticks():
    """The nastiest ordering: they are moved off 16-bit by a JPEG folder, then
    deliberately pick JPG. The next RAW folder must NOT drag them back to
    16-bit -- that would override the choice they just made."""
    import star_trail_cleanr as S
    s = _Stub()
    # the real handler is what retires the held preference, so wire it up
    s._jpeg_quality = _FakeEnable()
    s._jpeg_quality_label = _FakeEnable()
    s._format_combo.currentTextChanged.connect(
        lambda t: S.MainWindow._on_format_changed(s, t))

    s._format_combo.setCurrentText("TIFF 16-bit")
    s._sync(_folder(["a.jpg"]))
    assert s._format_combo.currentText() == "TIFF 8-bit"
    s._format_combo.setCurrentText("JPG")               # their own decision
    s._sync(_folder(["a.cr2"]))
    assert s._format_combo.currentText() == "JPG", (
        "their deliberate choice was overridden by a preference the app was "
        "still holding from before")


class _FakeEnable:
    def setEnabled(self, v):
        pass


def test_someone_already_on_jpg_is_left_alone():
    s = _Stub()
    s._format_combo.setCurrentText("JPG")
    s._sync(_folder(["a.jpg"]))
    assert s._format_combo.currentText() == "JPG"


# ── the saved preference must survive our own switching ─────────────────────

def test_our_own_switch_does_not_overwrite_the_saved_setting():
    """The window saves the format on every change. Without the guard flag, being
    moved to 8-bit would persist as if the user had chosen it, and their 16-bit
    preference would be gone for good."""
    saved = []
    s = _Stub()
    s._format_combo.currentTextChanged.connect(
        lambda t: None if s._format_auto_change else saved.append(t))
    s._format_combo.setCurrentText("TIFF 16-bit")       # the user's own choice
    assert saved == ["TIFF 16-bit"]
    s._sync(_folder(["a.jpg"]))                          # ours, must not persist
    assert saved == ["TIFF 16-bit"], f"our own switch was saved: {saved}"


def test_the_guard_flag_is_cleared_even_if_something_throws():
    """A stuck flag would silently stop saving the user's choices from then on."""
    body = GUI[GUI.index("def _sync_format_options("):]
    body = body[:body.index("\n    def ", 10)]
    assert "finally:" in body, (
        "the auto-change guard is not in a finally block; one exception would "
        "leave it set and the format setting would stop being remembered")


# ── wiring ─────────────────────────────────────────────────────────────────

def test_it_runs_whenever_the_folder_changes():
    body = GUI[GUI.index("def _update_frame_count("):]
    body = body[:body.index("\n    def ", 10)]
    assert "_sync_format_options" in body, (
        "nothing re-checks the format options when the input folder changes")


def test_the_disabled_option_explains_itself():
    body = GUI[GUI.index("def _sync_format_options("):]
    body = body[:body.index("\n    def ", 10)]
    assert "setToolTip" in body, (
        "a greyed-out option with no reason given just looks broken")
