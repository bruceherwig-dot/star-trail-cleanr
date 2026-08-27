"""A cleaned TIFF must not inherit tags describing how the SOURCE stored pixels.

Field report, 2026-08-23, Steven Labkoff: 600 cleaned 8-bit TIFFs opened as "a
screen full of static" in Photoshop and Lightroom, while the same files looked
perfect in Finder and stacked correctly in StarStaX. Re-running as 16-bit fixed
it.

CAUSE: his sources were Lightroom TIFF exports, which are compressed with a
horizontal PREDICTOR (tag 317). We copy the source EXIF onto our output, and 317
was not on the strip list. Our 8-bit TIFFs are written UNCOMPRESSED (since v2.26,
for Sequator), so the file claimed "every row is stored as differences from the
previous pixel" over data that had never been differenced. A reader that honours
the tag runs a cumulative sum along each row, and a photograph becomes noise.
Photoshop honours it; Finder and StarStaX ignore it when compression is NONE,
which is exactly why it looked fine in one place and destroyed in another.

16-bit was unaffected because that path writes via tifffile and embeds no EXIF.

THE GENERAL RULE, which is what this test defends: we re-encode the pixels, so
ANY tag describing the source's storage is a lie about our file. Keep what
describes the photograph (capture time, camera, exposure); drop everything about
encoding.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

# Tags that describe HOW pixels are stored. None may survive onto our output.
ENCODING_TAGS = {
    256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample",
    259: "Compression", 262: "PhotometricInterpretation", 273: "StripOffsets",
    277: "SamplesPerPixel", 278: "RowsPerStrip", 279: "StripByteCounts",
    284: "PlanarConfiguration", 317: "Predictor", 320: "ColorMap",
    322: "TileWidth", 323: "TileLength", 324: "TileOffsets",
    325: "TileByteCounts", 338: "ExtraSamples", 339: "SampleFormat",
}


def _strip_list_from_source():
    """The real set the engine strips, read out of the source so the test cannot
    drift away from the code it is defending."""
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    i = src.index("_TIFF_STRUCTURAL = {")
    block = src[i:src.index("}", i)]
    tags = set()
    for line in block.splitlines()[1:]:
        head = line.split("#")[0].strip().rstrip(",")
        if not head:
            continue
        tags.add(int(head, 16) if head.lower().startswith("0x") else int(head))
    return tags


def test_the_predictor_tag_is_stripped():
    """The exact tag that turned 600 photographs into static."""
    assert 317 in _strip_list_from_source(), (
        "Predictor (317) must be stripped. Left in, it tells Photoshop the rows "
        "are stored as differences when they are not, and every frame renders "
        "as noise.")


def test_every_encoding_tag_is_stripped():
    """Not just the one that bit us. Any of these inherited onto a file we
    encoded ourselves is a statement about someone else's file."""
    strip = _strip_list_from_source()
    missing = {c: n for c, n in ENCODING_TAGS.items() if c not in strip}
    assert not missing, (
        "these describe the SOURCE's pixel storage and would be inherited by "
        f"our re-encoded output: {missing}")


def test_a_lightroom_style_source_produces_a_clean_file(tmp_path=None):
    """End to end: an EXIF block carrying Predictor must not reach the output."""
    import tempfile
    import tifffile

    d = Path(tempfile.mkdtemp())
    img = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)

    ex = Image.Exif()
    ex[317] = 2                              # Predictor: horizontal differencing
    ex[339] = 1                              # SampleFormat
    ex[306] = "2019:06:12 22:14:03"          # DateTime, which we DO want kept
    ex[271] = "Canon"                        # Make, also kept
    source_exif = ex.tobytes()

    strip = _strip_list_from_source()
    cleaned = Image.Exif()
    cleaned.load(source_exif)
    for tag in strip:
        cleaned.pop(tag, None)

    out = d / "cleaned.tif"
    Image.fromarray(img).save(str(out), "TIFF", exif=cleaned.tobytes())

    with tifffile.TiffFile(str(out)) as tf:
        page = tf.pages[0]
        codes = {t.code for t in page.tags}
        assert 317 not in codes, "Predictor reached the output file"
        assert 339 not in codes, "SampleFormat reached the output file"
        # The photograph's own details must survive.
        assert 306 in codes, "capture time was lost"
        assert 271 in codes, "camera make was lost"
        # And the pixels must read back exactly.
        assert np.array_equal(page.asarray(), img), \
            "the written file does not read back as the image we wrote"
