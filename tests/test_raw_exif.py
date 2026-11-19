"""RAW files must report their camera, lens and settings like any other photo.

Reported by Kari Tuomi, 2026-08-27: "In the log these fields are empty, although
raw file has all exif info." His Star Log read Unknown five times over.

CAUSE: both places that read EXIF opened the photograph with PIL, and PIL cannot
open a RAW at all. It is not that the tags were missing -- every RAW carries a
small preview picture holding the camera's own tags, and this app ALREADY reads
the capture date that way in `io_safe.capture_time`. Nothing looked there for the
rest of them.

IT WAS NOT JUST HIS LOG. Measured across 128 real runs the same day:

    format   runs   no camera    no lens
    jpg        71   13 (18%)    22 (30%)
    raw        33   27 (81%)    32 (96%)

A third of all runs contributed nothing to the community gear stats, which is
most of the reason half our photographers show as "Unknown" there.

THE FIX IS ONE SHARED READER, `io_safe.exif_tags`, used by both the Star Log's
Camera Info block and the anonymous run summary. They were separate pieces of
code making the same mistake, so fixing one would have left the other wrong.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

GUI = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
IO = (REPO / "modules" / "io_safe.py").read_text(encoding="utf-8")


def test_the_shared_reader_handles_raw_through_the_preview():
    body = IO[IO.index("def exif_tags("):]
    body = body[:body.index("\ndef _sub_ifd")]
    assert "RAW_EXTS" in body, "the RAW branch is gone; RAW would return nothing again"
    assert "extract_thumb" in body, (
        "the reader no longer looks in the RAW's embedded preview, which is the "
        "only place a RAW's tags can be read from without decoding it")
    assert "get_ifd" in IO, "the Exif sub-block holds lens, ISO and exposure"


def test_both_readers_use_it_rather_than_opening_the_photograph():
    """Two separate places used to open the file with PIL. Either one reverting
    puts back half the bug -- the log or the stats, silently."""
    assert GUI.count("exif_tags") >= 2, (
        "the Star Log's Camera Info block and the anonymous run summary must "
        "BOTH go through the shared reader; one of them has reverted")
    gear = GUI[GUI.index("def _leader_gear_exif("):]
    gear = gear[:gear.index("\nclass ")]
    assert "Image.open(frames[0])" not in gear, (
        "the run summary is opening the photograph with PIL again, which returns "
        "nothing for a RAW")


def test_megapixels_come_from_the_header_reader():
    """PIL could not size a RAW either, so megapixels went missing with the rest.
    The header reader understands RAW."""
    gear = GUI[GUI.index("def _leader_gear_exif("):]
    gear = gear[:gear.index("\nclass ")]
    assert "image_size" in gear, (
        "megapixels are back to being measured by opening the photograph, which "
        "fails on RAW")


def test_a_missing_or_unreadable_file_returns_nothing_quietly():
    """This runs at the end of every clean. It must never raise, whatever it is
    handed."""
    from modules.io_safe import exif_tags
    assert exif_tags("/no/such/file.CR2") == {}
    assert exif_tags(__file__) == {}          # a text file, not a photo


def test_a_plain_image_still_reports_its_tags():
    """The ordinary path must be untouched: a JPEG with EXIF still reads."""
    import io as _io
    import tempfile
    from PIL import Image
    from modules.io_safe import exif_tags

    img = Image.new("RGB", (40, 30), (10, 10, 10))
    ex = Image.Exif()
    ex[0x010F] = "TestCam"          # Make
    ex[0x0110] = "Model One"        # Model
    p = Path(tempfile.mkdtemp()) / "shot.jpg"
    img.save(p, exif=ex)

    tags = exif_tags(str(p))
    assert tags.get("Make") == "TestCam"
    assert tags.get("Model") == "Model One"
