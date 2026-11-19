"""The Star Trail window lets you choose the format, but only downward.

Bruce, 2026-08-30: "If I create a new star trail. I should be able to pick the
format (go down in quality...not up). Gray out what's not avaible."

DOWN IS ALWAYS ALLOWED. Somebody stacking 16-bit frames who wants a JPEG to post,
or an 8-bit TIFF to keep the file manageable, should be able to say so. Before
this, the depth was whatever the frames happened to be and the request was simply
ignored -- asking for "TIFF 8-bit" from 16-bit frames still produced 16-bit,
because the format was inferred back out of the filename and an extension cannot
tell the two TIFFs apart.

UP IS NEVER ALLOWED, because it is not real. 8-bit frames written into a 16-bit
file give a file roughly thirty times larger holding exactly the same 256 levels
(measured on a real 5472x3648 frame, 2026-08-30). So that option is greyed out.

TIFF 8-BIT IS NEVER GREYED. It is not an upgrade over JPEG, it is the same depth
without the compression.

These tests build the REAL StarTrailPanel against REAL folders on disk. The panel
starts no background threads, so nothing has to be joined afterwards.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _frames(depth, ext, n=10):
    """A folder of real frames at a known depth."""
    import cv2
    import tifffile
    d = Path(tempfile.mkdtemp()) / "frames"
    d.mkdir(parents=True)
    for i in range(n):
        p = d / f"IMG_{2000 + i}{ext}"
        if depth == 16:
            tifffile.imwrite(str(p), np.full((20, 30, 3), 4000, np.uint16),
                             photometric="rgb")
        elif ext in (".tif", ".tiff"):
            cv2.imwrite(str(p), np.full((20, 30, 3), 60, np.uint8))
        else:
            cv2.imwrite(str(p), np.full((20, 30, 3), 60, np.uint8))
    return str(d)


def _panel(cleaned, original=None):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import star_trail_cleanr as S
    return S.StarTrailPanel(cleaned, "", original_folder=original)


def _enabled(cb, key):
    from PySide6.QtCore import Qt
    item = cb.model().item(cb.findData(key))
    return bool(item.flags() & Qt.ItemIsEnabled)


# ── what the frames can supply ─────────────────────────────────────────────

def test_eight_bit_frames_grey_out_16_bit():
    p = _panel(_frames(8, ".jpg"))
    assert not _enabled(p._fmt_cb, "tif16"), (
        "16-bit is offered for 8-bit frames; the file would be about thirty "
        "times larger holding the same 256 levels")


def test_an_eight_bit_tiff_folder_also_greys_out_16_bit():
    """The case a filename cannot answer: .tif says nothing about depth, so the
    header has to be read."""
    p = _panel(_frames(8, ".tif"))
    assert not _enabled(p._fmt_cb, "tif16")


def test_sixteen_bit_frames_offer_everything():
    p = _panel(_frames(16, ".tif"))
    for key in ("jpg", "tif8", "tif16"):
        assert _enabled(p._fmt_cb, key), f"{key} was taken away from 16-bit frames"


def test_eight_bit_tiff_and_jpg_are_never_greyed():
    """Going DOWN is the whole point, and 8-bit TIFF from JPEG frames is not an
    upgrade -- it is the same depth without the compression."""
    p = _panel(_frames(8, ".jpg"))
    assert _enabled(p._fmt_cb, "jpg")
    assert _enabled(p._fmt_cb, "tif8")


def test_unreadable_frames_take_no_choice_away():
    """Removing an option on a guess is worse than offering an oversized one."""
    d = Path(tempfile.mkdtemp())
    for i in range(10):
        (d / f"IMG_{i}.tif").write_bytes(b"not an image")
    p = _panel(str(d))
    assert _enabled(p._fmt_cb, "tif16")


def test_the_default_matches_the_frames_own_format():
    """Anyone who ignores this dropdown gets what this window always produced."""
    assert _panel(_frames(16, ".tif"))._fmt_cb.currentData() == "tif16"
    assert _panel(_frames(8, ".tif"))._fmt_cb.currentData() == "tif8"
    assert _panel(_frames(8, ".jpg"))._fmt_cb.currentData() == "jpg"


def test_switching_source_rechecks_what_is_available():
    """Cleaned and Original are commonly different: JPEG originals cleaned out to
    16-bit TIFF. Switching between them must change what is on offer."""
    cleaned = _frames(16, ".tif")
    original = _frames(8, ".jpg")
    p = _panel(cleaned, original=original)
    # Be explicit: the panel restores a remembered Source, so a saved "original"
    # on the machine running the tests would otherwise decide the starting point.
    p._src_cb.setCurrentIndex(p._src_cb.findData("cleaned"))
    assert _enabled(p._fmt_cb, "tif16"), "16-bit frames should allow 16-bit"
    p._src_cb.setCurrentIndex(p._src_cb.findData("original"))
    assert not _enabled(p._fmt_cb, "tif16"), (
        "still offering 16-bit after switching to 8-bit original frames")


def test_a_16_bit_choice_falls_back_to_tiff_not_jpg():
    """If their choice cannot be honoured by these frames, keep them on a TIFF
    rather than dropping them all the way to JPEG."""
    p = _panel(_frames(16, ".tif"), original=_frames(8, ".jpg"))
    p._src_cb.setCurrentIndex(p._src_cb.findData("cleaned"))
    p._fmt_touched = True
    p._fmt_cb.setCurrentIndex(p._fmt_cb.findData("tif16"))
    p._src_cb.setCurrentIndex(p._src_cb.findData("original"))
    assert p._fmt_cb.currentData() == "tif8", (
        f"landed on {p._fmt_cb.currentData()!r}")


# ── the choice has to reach the builder ────────────────────────────────────

def test_the_chosen_format_is_passed_through_not_inferred():
    """The old code read the format back out of the filename, and an extension
    cannot tell an 8-bit TIFF from a 16-bit one -- so 'TIFF 8-bit' was
    impossible to ask for."""
    body = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    body = body[body.index("def _start_build("):]
    body = body[:body.index("\n    def ", 10)]
    assert "_fmt_cb.currentData()" in body, (
        "the build no longer sends the format the user picked")
    assert 'endswith(".tif")' not in body, (
        "the format is being guessed from the filename again")


def test_the_format_is_part_of_the_filename():
    """A JPG build and a TIFF build of otherwise identical settings must not
    collide on one name."""
    body = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    body = body[body.index("def _build_out_path("):]
    body = body[:body.index("\n    def ", 10)]
    assert "16bit" in body and "8bit" in body


# ── going down actually produces a smaller-depth file ──────────────────────

def _stack16():
    a = np.full((20, 30, 3), 40000, np.uint16)
    a[5, 5] = 65535
    return a


def test_16_bit_frames_can_be_saved_as_an_8_bit_tiff():
    import tifffile
    import make_share_clip as msc
    d = Path(tempfile.mkdtemp())
    out = str(d / "t.tif")
    msc.make_star_trail(str(d), out_path=out, stack=_stack16(), out_format="tif8")
    with tifffile.TiffFile(out) as tf:
        assert tf.pages[0].bitspersample == 8, (
            "asked for an 8-bit TIFF and got something else")


def test_16_bit_frames_can_be_saved_as_a_jpeg():
    import cv2
    import make_share_clip as msc
    d = Path(tempfile.mkdtemp())
    out = str(d / "t.jpg")
    msc.make_star_trail(str(d), out_path=out, stack=_stack16(), out_format="jpg")
    back = cv2.imread(out)
    assert back is not None and back.dtype == np.uint8


def test_asking_for_16_bit_still_keeps_16_bit():
    """The down-conversion must not fire when nobody asked for it."""
    import tifffile
    import make_share_clip as msc
    d = Path(tempfile.mkdtemp())
    out = str(d / "t.tif")
    msc.make_star_trail(str(d), out_path=out, stack=_stack16(), out_format="tif16")
    with tifffile.TiffFile(out) as tf:
        assert tf.pages[0].bitspersample == 16
    assert int(tifffile.imread(out).max()) == 65535


def test_the_in_run_trail_still_keeps_its_depth():
    """THE REGRESSION THIS ALREADY CAUSED. make_star_trail's out_format defaults
    to 'jpg', and the end-of-run path passed an explicit .tif path without ever
    naming the format -- so the new down-conversion flattened every 16-bit trail
    until the format was threaded through finalize()."""
    import inspect
    from modules.share_stacker import ShareStacker
    sig = inspect.signature(ShareStacker.finalize)
    assert "out_format" in sig.parameters, (
        "finalize no longer carries the run's format, so make_star_trail falls "
        "back to 'jpg' and converts every 16-bit trail down to 8-bit")
