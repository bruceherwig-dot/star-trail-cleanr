"""Insta360 .insp photos are accepted as frames.

An .insp file is a plain JPEG (JPEG marker at the start, every JPEG reader opens it)
with the camera's own data appended after the picture. Real files checked
2026-11-09: 11904x5952, PIL / OpenCV / the app's own readers all open them. Only the
list of accepted extensions left them out, so a folder of .insp files looked empty.
These tests use a synthetic JPEG saved under the .insp name, so they need no sample
file and run anywhere.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _insp(folder, name, size=(40, 60)):
    """A real JPEG written under an .insp name, with junk appended like the
    camera's trailer."""
    import cv2
    import numpy as np
    p = Path(folder) / name
    ok, buf = cv2.imencode(".jpg", np.full((size[0], size[1], 3), 90, np.uint8))
    assert ok
    p.write_bytes(buf.tobytes() + b"\x00" * 16 + b"INSTA360TRAILER")
    return p


def test_insp_is_an_accepted_frame_type():
    from modules.frame_list import is_image_name, IMAGE_EXTS, glob_patterns
    assert ".insp" in IMAGE_EXTS
    assert is_image_name("IMG_20261107_002650_00_743.insp")
    assert is_image_name("IMG_1.INSP")
    assert "*.insp" in glob_patterns() and "*.INSP" in glob_patterns()


def test_macos_companion_files_of_insp_are_still_ignored():
    from modules.frame_list import is_image_name
    assert not is_image_name("._IMG_1.insp")


def test_insp_folder_lists_in_order_and_reads():
    from modules.frame_list import ordered_frame_list, gather_frames
    from modules.io_safe import image_size, robust_imread
    d = tempfile.mkdtemp()
    for n in ("IMG_10.insp", "IMG_2.insp", "IMG_1.insp"):
        _insp(d, n)
    got = [Path(p).name for p in ordered_frame_list(d)]
    assert got == ["IMG_1.insp", "IMG_2.insp", "IMG_10.insp"], got
    assert [Path(p).name for p in gather_frames(d)] == got
    first = ordered_frame_list(d)[0]
    assert image_size(first) == (60, 40)             # (width, height)
    img = robust_imread(first)
    assert img is not None and img.shape[:2] == (40, 60)


def test_insp_and_jpg_of_the_same_frame_count_once():
    """Same rank as a JPG: a renamed copy beside the original is one frame."""
    from modules.frame_list import dedupe_frames
    d = tempfile.mkdtemp()
    a = _insp(d, "IMG_5.insp")
    b = Path(d) / "IMG_5.jpg"
    b.write_bytes(a.read_bytes())
    assert len(dedupe_frames([str(a), str(b)])) == 1


def test_insp_loses_to_tiff_and_raw_like_a_jpg_does():
    from modules.frame_list import dedupe_frames
    got = dedupe_frames(["/x/F1.insp", "/x/F1.tif"], prefer_raw=False)
    assert [Path(p).suffix for p in got] == [".tif"]


def test_an_all_insp_folder_counts_as_eight_bit_only():
    """16-bit output cannot recover detail a JPEG never held, so an all-.insp folder
    must grey out 16-bit exactly like an all-JPG one."""
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert '_EIGHT_BIT_ONLY_EXTS = {".jpg", ".jpeg", ".insp"}' in src
    assert 'if exts <= {".jpg", ".jpeg", ".insp"}:' in src


def test_the_screen_and_faq_name_the_new_type():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert "(.JPG, .TIF 8/16-bit, .INSP, and RAW)" in src
    assert "Insta360 .INSP, and RAW files" in src


def test_capture_time_is_read_from_an_insp_jpeg():
    """Frame order comes from capture time; it must not choke on the extension."""
    from modules.io_safe import capture_time
    d = tempfile.mkdtemp()
    p = _insp(d, "IMG_9.insp")
    capture_time(str(p))            # no EXIF in the synthetic file: None is fine, a crash is not


# ---- carrying the Insta360 data through cleaning (modules/insp_carry.py) -------------

def _fake_insp(folder, name="IMG_1.insp", n_blocks=3, trailer_len=300, version=3):
    """A JPEG plus the camera's data, built the way a real .insp is laid out: extra
    64 KB-style application blocks before the picture data, and a trailer whose last
    40 bytes are [its own length][version][signature]."""
    import struct
    import cv2
    import numpy as np
    from modules.insp_carry import SIGNATURE
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), 90, np.uint8))
    jpeg = buf.tobytes()
    # place the camera's blocks right after the start marker and any APP0
    at = 2
    while jpeg[at] == 0xFF and 0xE0 <= jpeg[at + 1] <= 0xEF:
        at += 2 + struct.unpack(">H", jpeg[at + 2:at + 4])[0]
    blocks = [b"\xff\xe2" + struct.pack(">H", 2 + 50) + bytes([i + 1]) * 50
              for i in range(n_blocks)]
    exif = b"\xff\xe1" + struct.pack(">H", 2 + 8) + b"Exif\x00\x00II"
    jpeg = jpeg[:at] + exif + b"".join(blocks) + jpeg[at:]
    body = bytes(range(256))[:trailer_len - 40] if trailer_len > 40 else b""
    body = (body * 2)[:trailer_len - 40]
    trailer = body + struct.pack("<I", trailer_len) + struct.pack("<I", version) + SIGNATURE
    p = Path(folder) / name
    p.write_bytes(jpeg + trailer)
    return p, blocks, trailer


def _plain_jpeg(folder, name="clean.jpg"):
    import cv2
    import numpy as np
    p = Path(folder) / name
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), 200, np.uint8))
    p.write_bytes(buf.tobytes())
    return p


def test_read_parts_finds_the_blocks_and_the_trailer():
    from modules.insp_carry import read_parts
    d = tempfile.mkdtemp()
    src, blocks, trailer = _fake_insp(d)
    parts = read_parts(str(src))
    assert parts is not None
    assert parts[0] == blocks, "the camera's blocks (and only those, in order) must come back"
    assert parts[1] == trailer


def test_carry_puts_blocks_and_trailer_onto_the_cleaned_picture():
    from modules.insp_carry import carry_insta360, read_parts
    from PIL import Image
    import cv2
    d = tempfile.mkdtemp()
    src, blocks, trailer = _fake_insp(d)
    new = _plain_jpeg(d)
    out = Path(d) / "clean.insp"
    assert carry_insta360(str(src), str(new), str(out)) is True
    assert not new.exists(), "the temporary plain .jpg must be removed"
    data = out.read_bytes()
    assert data.endswith(trailer), "the trailer must be copied byte for byte"
    assert read_parts(str(out)) == (blocks, trailer)
    assert Image.open(out).size == (60, 40)                 # still a valid JPEG
    assert cv2.imread(str(out)).shape[:2] == (40, 60)


def test_the_cleaned_pixels_are_the_new_ones_not_the_originals():
    """Only the data is carried; the picture is the cleaned one."""
    from modules.insp_carry import carry_insta360
    import cv2
    d = tempfile.mkdtemp()
    src, _b, _t = _fake_insp(d)                              # picture is mid-gray (90)
    new = _plain_jpeg(d)                                     # cleaned picture is lighter (200)
    out = Path(d) / "clean.insp"
    assert carry_insta360(str(src), str(new), str(out))
    assert cv2.imread(str(out)).mean() > 150


def test_a_plain_jpeg_is_not_treated_as_insp():
    from modules.insp_carry import read_parts, carry_insta360
    d = tempfile.mkdtemp()
    plain = _plain_jpeg(d, "photo.jpg")
    new = _plain_jpeg(d, "new.jpg")
    assert read_parts(str(plain)) is None
    assert carry_insta360(str(plain), str(new), str(Path(d) / "x.insp")) is False
    assert new.exists(), "on failure the plain .jpg must be left in place"
    assert not (Path(d) / "x.insp").exists()


def test_a_trailer_length_that_does_not_fit_is_refused():
    import struct
    from modules.insp_carry import read_parts, SIGNATURE
    d = tempfile.mkdtemp()
    src, _b, _t = _fake_insp(d)
    raw = bytearray(src.read_bytes())
    raw[-40:-36] = struct.pack("<I", len(raw) + 10)          # longer than the file
    src.write_bytes(bytes(raw))
    assert read_parts(str(src)) is None


def test_a_picture_that_does_not_end_where_the_trailer_starts_is_refused():
    import struct
    from modules.insp_carry import read_parts
    d = tempfile.mkdtemp()
    src, _b, _t = _fake_insp(d)
    raw = bytearray(src.read_bytes())
    raw[-40:-36] = struct.pack("<I", 120)                    # trailer "starts" mid-picture
    src.write_bytes(bytes(raw))
    assert read_parts(str(src)) is None


def test_only_insp_sources_are_exported_as_insp():
    """The engine carries the data only for an .insp source writing JPG, and passes
    the source to both write calls. TIFF output and every other source are untouched."""
    eng = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert 'str(src).lower().endswith(".insp")' in eng
    assert eng.count("dpi=dpi, src=fp)") == 2, "both _write_output calls must pass the source"
    jpg_at = eng.index('if args.output_format == "jpg":')
    carry_at = eng.index("carry_insta360(str(src)")
    tif_at = eng.index('elif args.output_format == "tif8":')
    assert jpg_at < carry_at < tif_at, "the carry must live inside the JPG branch only"
