"""Frames are ordered by the SHUTTER time, or not at all.

WHAT BRUCE SAW (2026-08-30): "there is a hesitation on this video? like it went
forward, then went backwards?" His timelapse played 001, 002, 004, 003, 005 --
one swap, early on, exactly one visible stumble.

WHY. Frame order came from the photo's capture time, and that lookup FELL BACK to
the plain file date (EXIF DateTime) when there was no shutter time
(DateTimeOriginal). The plain date is not a capture time at all: an editor stamps
it when it EXPORTS the file. Cheryl's frames came out of Camera Raw, which
finished exporting 004 two seconds before 003, and cleaning to 16-bit TIFF drops
the EXIF block -- so the cleaned frames had nothing left but those export stamps.
The renderer sorted by them faithfully and produced the stumble.

THE FIX IS ONE CONDITION, not a rewrite. Bruce pushed back on my first, larger
proposal: "seems like a complicated fix for putting items in the right order
based on their file name." He was right. Only a genuine shutter time is trusted
now; without one, capture_time returns None, the all-or-nothing rule in
order_by_capture_time leaves the list alone, and filename order stands -- which is
right far more often than an export stamp is.

WHAT THE RULE STILL EXISTS FOR, and must keep doing: a camera rolling over
mid-shoot (IMG_9999 then IMG_0001, so the last frame sorts first) and frames
merged from two cards. Filename order cannot fix those. Those frames DO carry
real shutter times, so they are still put right.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATETIME_ORIGINAL = 0x9003     # the shutter
DATETIME = 0x0132              # the file date; an editor's export stamp


def _shot(folder, name, shutter=None, filedate=None):
    """One frame carrying whichever dates we say."""
    from PIL import Image
    im = Image.fromarray(np.full((40, 60, 3), 30, np.uint8))
    ex = im.getexif()
    if shutter:
        ex.get_ifd(0x8769)[DATETIME_ORIGINAL] = shutter
    if filedate:
        ex[DATETIME] = filedate
    im.save(os.path.join(folder, name), exif=ex)


def test_an_export_stamp_is_not_a_capture_time():
    """The heart of it. A file with only a plain date must report no capture
    time, so nothing downstream can be tempted to sort by it."""
    from modules.io_safe import capture_time
    d = tempfile.mkdtemp()
    _shot(d, "003.jpg", filedate="2025:05:01 20:47:22")
    assert capture_time(os.path.join(d, "003.jpg")) is None, (
        "the export stamp is being reported as a capture time again; it will "
        "reorder frames that were already in the right order")


def test_a_real_shutter_time_is_still_read():
    from modules.io_safe import capture_time
    d = tempfile.mkdtemp()
    _shot(d, "a.jpg", shutter="2026:06:16 23:58:00")
    got = capture_time(os.path.join(d, "a.jpg"))
    assert got is not None and got.year == 2026 and got.minute == 58, got


def test_cheryls_sequence_plays_in_order():
    """Reproduces the reported stumble: export stamps that disagree with the
    filenames, and no shutter time to appeal to."""
    import timelapse_maker as tm
    d = tempfile.mkdtemp()
    for name, stamp in (("003.jpg", "2025:05:01 20:47:22"),   # exported LAST
                        ("004.jpg", "2025:05:01 20:47:20"),   # exported first
                        ("001.jpg", "2025:05:01 20:47:16"),
                        ("002.jpg", "2025:05:01 20:47:18")):
        _shot(d, name, filedate=stamp)
    got = [os.path.basename(f) for f in tm.ordered_frames(d)]
    assert got == ["001.jpg", "002.jpg", "003.jpg", "004.jpg"], (
        f"the video would stumble again: {got}")


def test_a_camera_rollover_is_still_put_right():
    """What the capture-time rule exists FOR. Filename order puts 0001 first;
    only the shutter times can rescue this."""
    import timelapse_maker as tm
    d = tempfile.mkdtemp()
    seq = [("IMG_9998.jpg", "2026:06:16 23:58:00"),
           ("IMG_9999.jpg", "2026:06:16 23:59:00"),
           ("IMG_0001.jpg", "2026:06:17 00:00:00"),
           ("IMG_0002.jpg", "2026:06:17 00:01:00")]
    for name, when in seq:
        _shot(d, name, shutter=when)
    got = [os.path.basename(f) for f in tm.ordered_frames(d)]
    assert got == [n for n, _ in seq], (
        f"a rollover mid-shoot is no longer corrected: {got}")


def test_a_shutter_time_still_wins_over_a_disagreeing_export_stamp():
    """Both present: the shutter is the truth, the export stamp is noise."""
    import timelapse_maker as tm
    d = tempfile.mkdtemp()
    # export stamps are in the REVERSE of the shooting order
    _shot(d, "IMG_9999.jpg", shutter="2026:06:16 23:59:00", filedate="2025:05:01 20:47:29")
    _shot(d, "IMG_0001.jpg", shutter="2026:06:17 00:00:00", filedate="2025:05:01 20:47:16")
    got = [os.path.basename(f) for f in tm.ordered_frames(d)]
    assert got == ["IMG_9999.jpg", "IMG_0001.jpg"], got


def test_frames_with_no_dates_at_all_keep_filename_order():
    import timelapse_maker as tm
    d = tempfile.mkdtemp()
    for n in ("003.jpg", "001.jpg", "002.jpg"):
        _shot(d, n)
    got = [os.path.basename(f) for f in tm.ordered_frames(d)]
    assert got == ["001.jpg", "002.jpg", "003.jpg"], got


def test_the_fallback_is_gone_from_both_read_paths():
    """RAW reads its time from the embedded preview on a separate code path. The
    fallback existed in BOTH and both had to go, or RAW sequences would still be
    reordered by an export stamp."""
    src = (REPO / "modules/io_safe.py").read_text(encoding="utf-8")
    body = src[src.index("def capture_time("):]
    body = body[:body.index("\ndef ", 10)]
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    assert "0x0132" not in code, (
        "capture_time falls back to the plain file date again; an editor's "
        "export stamp will reorder a correct sequence")
    # Two read paths (RAW's embedded preview, and everything else), and each
    # looks in two places: the EXIF sub-block a camera writes, and the top level
    # where our own 16-bit TIFF writer has to put it. Four references in total.
    assert code.count("sub.get(0x9003) or ex.get(0x9003)") == 2, (
        "the shutter time must be looked for in BOTH places on BOTH read paths; "
        "dropping the top-level lookup makes our own 16-bit TIFFs unorderable, "
        "dropping a read path does the same for RAW")


# ── the other half: stop LOSING the shutter time on 16-bit TIFF ────────────

def test_the_16bit_writer_carries_the_shutter_time_not_the_file_date():
    """Cleaning to 16-bit TIFF used to replace the capture time with an export
    stamp, permanently. The writer copied tag 306 (the file's own date, which an
    editor stamps on export) and never looked at the shutter time, which lives in
    an EXIF sub-block this writer cannot carry across.

    Cheryl's cleaned frames therefore claimed to be taken two days after the
    shoot, seconds apart and out of order. Verified fixed on a real clean of her
    actual frames (2026-08-30): all five kept their true shutter times."""
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    i = src.index('_mk = _ex.get(271)')
    body = src[i - 2500:i]
    assert "0x9003" in body, (
        "the 16-bit TIFF writer is back to copying only the file date, so the "
        "real capture time is thrown away on every 16-bit clean")
    assert "_sub.get(0x9003) or _ex.get(306)" in body, (
        "the shutter time must be PREFERRED over the file date, not the reverse")


def test_a_shutter_time_written_at_the_top_level_is_read_back():
    """tifffile cannot build an EXIF sub-block, so the writer puts the shutter
    time at the top level of the TIFF. The reader has to look there or the write
    is pointless."""
    import tempfile
    import numpy as np
    import tifffile
    from modules.io_safe import capture_time
    d = tempfile.mkdtemp()
    p = os.path.join(d, "003.tif")
    tifffile.imwrite(p, np.zeros((20, 30, 3), np.uint16), photometric="rgb",
                     extratags=[(306, 's', 0, "2025:05:01 20:47:22", True),
                                (0x9003, 's', 0, "2025:04:29 23:38:38", True)])
    got = capture_time(p)
    assert got is not None and str(got) == "2025-04-29 23:38:38", (
        f"read {got!r}; the shutter time at the top level was not found, or the "
        f"export stamp won")
