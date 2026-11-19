"""The star trail actually gets written at the end of a run.

THE BUG THIS EXISTS FOR (2026-08-30). ShareStackThread.run() asked itself for
`self.output_format` to decide whether the trail should be a .jpg or a .tif.
Nothing ever set that attribute. So the star trail step raised AttributeError on
its very first line, at the end of EVERY run, and the Star Trail window came up
showing nothing newer than the previous session's files. The run itself was
perfect; only the last step died.

WHY IT WAS NOT CAUGHT. The change that introduced it was verified by calling the
stacking functions in make_share_clip directly and proving the output was
byte-for-byte identical. Those functions were never the problem. The WIRING to
them was, and nothing exercised it. So these tests drive the real thread's real
run() method end to end over real files on disk, and assert on the file that
comes out -- not on the pieces underneath.

run() is called directly rather than through start(). It is an ordinary method,
so this exercises exactly the code a run executes, with no event loop, no
threads left alive at exit, and no waiting.
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

GUI = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")


def _sequence(out_format):
    """A tiny but REAL pair of folders: originals as JPEG, cleaned frames in the
    format the run would have written. Enough frames to survive the 3-and-3
    test-shot skip."""
    import cv2
    import tifffile

    root = Path(tempfile.mkdtemp())
    orig = root / "src"
    clean = root / "src" / "cleaned"
    ws = clean / "STC Extras"
    for d in (orig, clean, ws):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        img = np.full((60, 80, 3), 20, np.uint8)
        img[30, 5 + i * 6] = 250                       # a star that moves
        cv2.imwrite(str(orig / f"IMG_{2000 + i}.jpg"), img)
        if out_format == "tif16":
            tifffile.imwrite(str(clean / f"IMG_{2000 + i}.tif"),
                             cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint16) * 257,
                             photometric="rgb")
        elif out_format == "tif8":
            cv2.imwrite(str(clean / f"IMG_{2000 + i}.tif"), img)
        else:
            cv2.imwrite(str(clean / f"IMG_{2000 + i}.jpg"), img)
    return str(orig), str(clean), str(ws)


def _run_thread(out_format, want_original=True):
    """Build the real ShareStackThread and run it to completion. Returns the
    workspace folder and whatever it reported producing."""
    import star_trail_cleanr as S

    orig, clean, ws = _sequence(out_format)
    t = S.ShareStackThread(orig, clean, ws, True, False,
                           want_original_star=want_original,
                           output_format=out_format)
    produced = {}
    failures = []
    t.done.connect(lambda d: produced.update(d))
    t.failed.connect(lambda m: failures.append(m))
    t.finish()                    # queue the "render now" command
    t.run()                       # the real method, called straight
    assert not failures, f"the star trail step failed: {failures}"
    return ws, produced


def test_a_jpeg_run_writes_a_jpeg_star_trail():
    ws, produced = _run_thread("jpg")
    assert os.path.isfile(os.path.join(ws, "STC_cleaned_star_trail.jpg")), (
        f"no star trail was written; the thread reported {produced}")


def test_a_16bit_run_writes_a_16bit_tiff_star_trail():
    """THE ONE THAT MATTERS. The in-run stacker read every cleaned frame with
    IMREAD_COLOR, which forces 8 bits, and shared that read with the video. So a
    16-bit run wrote a .tif holding 8-bit data -- the feature's name with none of
    its substance, and worse than a JPEG because it looks delivered.

    The file's own header is checked, not just the array we get back: an 8-bit
    TIFF written through the fallback writer would still read as a valid image."""
    import tifffile
    ws, produced = _run_thread("tif16")
    p = os.path.join(ws, "STC_cleaned_star_trail.tif")
    assert os.path.isfile(p), f"no star trail was written; reported {produced}"
    with tifffile.TiffFile(p) as tf:
        bits = tf.pages[0].bitspersample
    assert bits == 16, (
        f"the trail is a TIFF but only {bits} bits per channel -- the depth was "
        f"dropped somewhere between the cleaned frames and the file")


def test_the_trail_keeps_values_eight_bits_could_not_hold():
    """Depth in the header is not proof of depth in the picture. The stacked
    values themselves must include ones no 8-bit file could represent."""
    import tifffile
    ws, _ = _run_thread("tif16")
    back = tifffile.imread(os.path.join(ws, "STC_cleaned_star_trail.tif"))
    assert back.dtype == np.uint16
    assert int(back.max()) > 255, (
        f"every value fits in 8 bits (max {back.max()}); the trail was built "
        f"from frames that had already been flattened")


def test_a_transparency_channel_does_not_wreck_the_stack():
    """Reading at full depth also keeps an alpha channel that the old 8-bit read
    discarded. A Photoshop TIFF can carry one, and a 4-channel frame among
    3-channel ones would stack as nonsense."""
    import cv2
    from make_share_clip import _to_bgr
    rgba = np.dstack([np.full((6, 8, 3), 90, np.uint8),
                      np.full((6, 8), 255, np.uint8)])
    assert _to_bgr(rgba).shape == (6, 8, 3), "the alpha channel survived"
    assert _to_bgr(np.zeros((6, 8, 3), np.uint8)).shape == (6, 8, 3)
    assert _to_bgr(None) is None


def test_an_8bit_tiff_run_writes_a_tiff_star_trail():
    ws, _ = _run_thread("tif8")
    assert os.path.isfile(os.path.join(ws, "STC_cleaned_star_trail.tif"))


def test_the_uncleaned_trail_follows_the_same_format():
    """Bruce's decision when Jon asked for TIFF output: both trails, cleaned and
    uncleaned, follow the format the frames were cleaned as."""
    ws, _ = _run_thread("tif16")
    assert os.path.isfile(os.path.join(ws, "STC_original_star_trail.tif")), (
        "the uncleaned trail did not follow the cleaned format")


def test_the_trail_is_a_real_lighten_stack_not_a_blank():
    """A file existing is not proof it holds the picture. Every moving star must
    be present in the finished trail."""
    import cv2
    ws, _ = _run_thread("jpg")
    trail = cv2.imread(os.path.join(ws, "STC_cleaned_star_trail.jpg"))
    assert trail is not None
    # frames 3..6 survive the 3-and-3 skip, so those star positions must be lit
    lit = [int(trail[30, 5 + i * 6].max()) for i in range(3, 7)]
    assert all(v > 150 for v in lit), f"stars missing from the trail: {lit}"


def test_the_format_is_handed_in_not_guessed_at():
    """The exact shape of the bug: run() reading a value nobody supplied."""
    body = GUI[GUI.index("class ShareStackThread"):]
    body = body[:body.index("\nclass ", 10)]
    # comments describe the old bug by name, so judge the CODE only
    body = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    assert "self.output_format" not in body, (
        "ShareStackThread is reading self.output_format again -- nothing sets "
        "it, so the star trail step raises at the end of every run")
    assert "output_format=" in body or "output_format," in body, (
        "the thread no longer accepts the run's output format at all")


def test_the_run_and_the_trail_read_the_format_from_one_place():
    """Two copies of the dropdown-to-format mapping is how they drifted apart."""
    assert GUI.count('"TIFF 16-bit": "tif16"') == 1, (
        "the format mapping is duplicated again; the run and the star trail can "
        "now disagree about what format the frames were written in")
    assert "_current_output_format" in GUI
