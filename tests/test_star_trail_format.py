"""The star trail is saved in the same format as the cleaned frames.

Asked for by Jon Bertsch, 2026-08-27: "maybe save the star trail output as a TIFF
rather than a jpg to avoid data loss." Bruce settled the rule: no new setting,
the star trail simply follows whatever the frames were cleaned as. A JPEG clean
gets a JPEG trail; a 16-bit TIFF clean gets a 16-bit TIFF trail.

WRITING A TIFF IS ONLY HALF OF IT. Every frame used to be read with
IMREAD_COLOR, which forces 8 bits, so saving a 16-bit file around that data would
have been a bigger file holding exactly the same information -- worse than the
JPEG, because it looks like the feature while delivering none of it. The reads
had to change too, and that is what these tests mostly guard.

The cost is not what anyone expected. Measured on a 44MP frame, keeping the full
depth is FASTER than converting down (44ms a frame against 69ms), because the
conversion costs more than stacking bigger numbers. The only real cost is memory,
133 MB to 266 MB for the running stack, and it falls only on people who chose
16-bit -- reading a JPEG at "full depth" still gives 8 bits.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

MSC = (REPO / "make_share_clip.py").read_text(encoding="utf-8")
GUI = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")


def test_the_star_trail_stack_reads_frames_at_their_real_depth():
    """The half that actually preserves anything. If this reverts to IMREAD_COLOR,
    a 16-bit TIFF trail becomes 8-bit data in a 16-bit container."""
    body = MSC[MSC.index("def _stack_fullres("):]
    body = body[:body.index("\ndef ", 10)]
    assert "IMREAD_UNCHANGED" in body, (
        "the full-resolution star trail stack is forcing 8-bit again; a 16-bit "
        "clean would produce a 16-bit FILE holding 8-bit DATA")


def test_the_video_stack_stays_eight_bit():
    """The opposite guarantee. An encoder cannot use more than 8 bits, so reading
    the video's frames at full depth would cost time and preserve nothing."""
    body = MSC[MSC.index("class IncrementalStack"):]
    body = body[:body.index("\nclass ", 10)] if "\nclass " in body[10:] else body
    assert "self.canvas is not None" in body, (
        "the incremental stacker no longer distinguishes the video canvas from "
        "the full-res trail, so one of them is reading at the wrong depth")


def test_the_extension_follows_the_cleaned_format():
    from make_share_clip import star_trail_ext
    assert star_trail_ext("jpg") == ".jpg"
    assert star_trail_ext("tif8") == ".tif"
    assert star_trail_ext("tif16") == ".tif"
    assert star_trail_ext(None) == ".jpg", "an unknown format must fall back to JPEG"
    assert star_trail_ext("") == ".jpg"


def test_sixteen_bit_data_is_written_through_tifffile():
    """PIL has no 16-bit RGB save mode -- the cleaned-frame writer already learned
    this the hard way in astro_clean_v5."""
    body = MSC[MSC.index("def write_star_trail("):]
    body = body[:body.index("\ndef ", 10)]
    assert "tifffile" in body and "np.uint16" in body, (
        "16-bit trails are no longer written through tifffile; PIL cannot save "
        "16-bit RGB and would either fail or quietly drop depth")


def test_a_16bit_trail_really_holds_16bit_values():
    """End to end on real files: the saved trail must contain values an 8-bit
    file could not represent."""
    import tempfile
    import cv2
    import tifffile
    from make_share_clip import write_star_trail

    d = Path(tempfile.mkdtemp())
    stack = np.full((30, 40, 3), 40000, np.uint16)
    stack[10, 10] = 51599                      # a value 8 bits cannot hold
    out = d / "trail.tif"
    assert write_star_trail(str(out), stack)

    back = tifffile.imread(str(out))
    assert back.dtype == np.uint16, "the depth was lost on the way to disk"
    assert int(back.max()) == 51599
    # written RGB, read RGB: the colours must not be swapped on the round trip
    assert np.array_equal(back, cv2.cvtColor(stack, cv2.COLOR_BGR2RGB))


def test_the_old_format_twin_is_removed_so_nothing_goes_stale():
    """Re-cleaning used to overwrite the one star trail. Once the extension
    follows the format, a JPEG trail and a TIFF trail have different names, so
    without this the older one would sit there for good, looking current."""
    import tempfile
    from make_share_clip import drop_stale_twin

    d = Path(tempfile.mkdtemp())
    (d / "STC_cleaned_star_trail.jpg").write_bytes(b"old")
    (d / "STC_cleaned_star_trail.tif").write_bytes(b"new")
    (d / "something_else.jpg").write_bytes(b"keep me")

    drop_stale_twin(str(d / "STC_cleaned_star_trail.tif"))
    assert not (d / "STC_cleaned_star_trail.jpg").exists(), "the stale twin survived"
    assert (d / "STC_cleaned_star_trail.tif").exists(), "it deleted the wrong file"
    assert (d / "something_else.jpg").exists(), (
        "it reached beyond the star trail's own name")


def test_nothing_hardcodes_the_jpg_extension_any_more():
    """Seven places used to name '.jpg' outright. A TIFF clean would have left
    the preview saying there was no star trail while the file sat in the folder."""
    import re
    code = "\n".join(l for l in GUI.splitlines() if not l.strip().startswith("#"))
    for banned in ('STC_cleaned_star_trail.jpg', 'STC_original_star_trail.jpg',
                   'STC_star_trail_*.jpg'):
        assert banned not in code, (
            f"'{banned}' is hardcoded again; a TIFF star trail would be invisible "
            f"to the preview and to the arrows beside it")


def test_the_window_asks_the_files_not_the_current_setting():
    """This window can be opened on a folder cleaned months ago. The frames on
    disk are the only thing that knows what format they are."""
    assert "def cleaned_star_ext(" in GUI, "the format detector is gone"
    body = GUI[GUI.index("def cleaned_star_ext("):]
    body = body[:body.index("\ndef ", 10)]
    assert "listdir" in body, (
        "it no longer looks at the actual files, so opening an old folder would "
        "name the trail after whatever the app happens to be set to now")
