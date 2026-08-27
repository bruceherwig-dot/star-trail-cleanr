"""Black-and-white source frames must survive the whole run.

Field crash, 2026-08-25. A user pointed Star Trail CleanR at a folder of
telescope sub-frames converted to TIFF (SeeStar, `Startrails_sub/Tif`) and every
batch died with:

    File "modules/repair.py", line 712, in repair_frame
      _wmax = frame[y0:y1, x0:x1].max(2)
    AxisError: axis 2 is out of bounds for array of dimension 2

CAUSE: those TIFFs hold ONE channel, not three. Converting a FITS sub without
debayering, or shooting a mono astro camera, gives a greyscale file, and
IMREAD_UNCHANGED hands it back as a flat (H,W) array. Every stage downstream is
written for (H,W,3): Star Bridge repair asks for the brightest of the three
colours at a pixel, and the 16-bit TIFF writer converts BGR to RGB. Both need a
third axis that isn't there. Greyscale input had never worked; the crashing line
arrived with the v2.72 sky collar and nobody had tried it until now.

THE FIX IS CENTRAL, not a guard per stage: the reader promotes a single-channel
photo to three channels once, next to the orientation fix, so the worker, the
detector and the tools all inherit it. The writer then hands the frames back
greyscale, because a promoted 16-bit sub written as RGB is three times the size
it arrived as, and a night's subs is hundreds of files.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.io_safe import robust_imread, is_single_channel  # noqa: E402
from modules.repair import repair_frame                       # noqa: E402


def _mono_tiff(dtype=np.uint16):
    """Write a throwaway single-channel TIFF and return its path."""
    import tifffile
    rng = np.random.default_rng(0)
    top = 3000 if dtype == np.uint16 else 200
    arr = (rng.random((120, 160)) * top).astype(dtype)
    p = Path(tempfile.mkdtemp()) / f"mono_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(str(p), arr)
    return p


def _repaired_mono():
    """Load a greyscale TIFF and run the exact repair call that crashed."""
    img = robust_imread(str(_mono_tiff()), cv2.IMREAD_UNCHANGED)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.line(mask, (30, 30), (120, 90), 255, 5)
    frames = [img.copy(), img.copy(), img.copy()]
    out = repair_frame(frames[1], mask, 1, frames,
                       neighbor_masks=[None, mask, None], polygon_segs=[mask])
    return img, out


def test_the_reader_promotes_a_greyscale_file_to_three_channels():
    img = robust_imread(str(_mono_tiff()), cv2.IMREAD_UNCHANGED)
    assert img is not None, "a plain greyscale TIFF must be readable"
    assert img.ndim == 3 and img.shape[2] == 3, (
        "a single-channel source reached the pipeline as a 2D array; every "
        "stage downstream assumes three channels and will crash on it")
    assert img.dtype == np.uint16, "the promotion must not change bit depth"


def test_grayscale_flag_callers_still_get_one_channel():
    """The foreground mask and the hot-pixel map are READ as single channel on
    purpose. Promoting those would break them."""
    img = robust_imread(str(_mono_tiff(np.uint8)), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.ndim == 2, (
        "IMREAD_GRAYSCALE must keep returning a flat mask")


def test_repair_survives_a_greyscale_frame():
    """The exact call shape that crashed in the field."""
    img, out = _repaired_mono()
    assert out.shape == img.shape and out.dtype == img.dtype


def test_the_channels_stay_identical_so_the_writer_can_collapse_them():
    """The writer only writes greyscale back when the three channels really are
    equal. If repair ever started tinting them, that check would quietly turn
    every mono run into a 3x-size colour file -- this says so out loud."""
    _, out = _repaired_mono()
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


def test_the_file_on_disk_can_still_be_recognised_as_greyscale():
    """After promotion nothing in memory remembers what the source was, so the
    writer asks the file. Colour sources must not be mistaken for greyscale."""
    import tifffile
    assert is_single_channel(str(_mono_tiff())) is True

    colour = Path(tempfile.mkdtemp()) / "colour.tif"
    tifffile.imwrite(str(colour),
                     np.zeros((40, 50, 3), np.uint16), photometric="rgb")
    assert is_single_channel(str(colour)) is False


def test_collapsing_back_returns_the_original_pixels():
    """The writer's collapse must be exact, not a colour-converted approximation."""
    src = (np.arange(120 * 160, dtype=np.uint16) % 3000).reshape(120, 160)
    three = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(three, cv2.COLOR_BGR2RGB)
    assert np.array_equal(rgb[:, :, 0], src)


def test_the_engine_still_has_both_greyscale_write_branches():
    """Guard the writer wiring itself: the greyscale branch must exist in both
    TIFF paths, or the fix silently reverts to 3x-size colour output."""
    engine = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert "_grey_plane" in engine, "the greyscale collapse helper is gone"
    assert "minisblack" in engine, (
        "the 16-bit TIFF writer no longer has a greyscale photometric; a mono "
        "source would be written as RGB")
    assert 'mode="L"' in engine, (
        "the 8-bit TIFF writer no longer has a greyscale branch")
