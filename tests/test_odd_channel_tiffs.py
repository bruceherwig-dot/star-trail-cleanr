"""Smoke tests for TIFFs with unusual channel counts (modules/io_safe.py).

A user's Photoshop TIFF carried seven channels per pixel (red, green, blue plus
spot/alpha extras). OpenCV refuses more than four and Pillow raises, so tifffile
was the only reader that accepted it, and it handed all seven straight through.
The app only knows how to strip a fourth channel, so seven reached the trail
detector and OpenCV threw when it tried to convert the colours. The Sentry alert
that surfaced it was only Pillow's own log line about a file it never had to read.

These lock the trim in place. Offline: the TIFFs are generated here.

UPDATED 2026-08-25. Two of these used to assert that a one-channel file came
back one-channel, and grey-plus-alpha came back as plain grey. That expectation
was itself the bug: a flat (H,W) array is a shape the pipeline cannot use, and a
user's folder of greyscale telescope subs crashed every batch in Star Bridge
repair (`AxisError: axis 2 is out of bounds`). The reader now promotes any
single-channel photo to three channels, so the expectations below say three.
The purpose of the tests is unchanged -- an odd channel count must come back as
something the rest of the app can actually work on. See tests/test_mono_input.py.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _write_tiff(path, channels, height=24, width=32):
    """Write a TIFF with an exact channel count. Returns the array written."""
    import tifffile
    if channels == 1:
        arr = (np.random.rand(height, width) * 255).astype("uint8")
    else:
        arr = (np.random.rand(height, width, channels) * 255).astype("uint8")
    # A real colour TIFF is tagged rgb; anything else is tagged greyscale with
    # extra samples, which is how Photoshop stores spot/alpha channels. Tagging
    # a 3-channel file greyscale would make readers hand back only the first
    # sample, which is a quirk of the fixture rather than of our code.
    photometric = "rgb" if channels in (3, 4) else "minisblack"
    extrasamples = None
    if channels == 4:
        extrasamples = "unassalpha"
    kwargs = {"photometric": photometric, "planarconfig": "contig"}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(path), arr, **kwargs)
    return arr


def test_seven_channel_tiff_is_trimmed_to_rgb(tmp_path=None):
    """THE regression net: the shape the rest of the app can actually use."""
    import tempfile
    from modules.io_safe import robust_imread_diag
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "seven.tif"
        _write_tiff(p, 7)
        img, diag = robust_imread_diag(str(p), _retry_delays=())
        assert img is not None, f"a seven-channel TIFF must still load: {diag}"
        assert img.ndim == 3 and img.shape[2] == 3, (
            f"expected three channels after the trim, got {img.shape}")


def test_trimmed_channels_survive_an_opencv_colour_conversion():
    """The actual downstream failure was OpenCV throwing on the colour
    conversion in the trail detector, so prove the trimmed frame survives it."""
    import tempfile
    import cv2
    from modules.io_safe import robust_imread_diag
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "many.tif"
        _write_tiff(p, 7)
        img, _ = robust_imread_diag(str(p), _retry_delays=())
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # raised before the trim
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # raised before the trim


def test_grey_plus_alpha_tiff_comes_back_usable():
    """Two channels is grey plus alpha; it hit the same untrimmed path. The
    alpha is dropped and the grey is promoted, so what arrives downstream is an
    ordinary three-channel photo."""
    import tempfile
    from modules.io_safe import robust_imread_diag
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "grey_alpha.tif"
        _write_tiff(p, 2)
        img, diag = robust_imread_diag(str(p), _retry_delays=())
        assert img is not None, f"a grey+alpha TIFF must still load: {diag}"
        assert img.ndim == 3 and img.shape[2] == 3, (
            f"expected a usable three-channel image, got {img.shape}")


def test_normal_tiffs_are_untouched():
    """The trim must not disturb the ordinary cases. One channel is promoted to
    three (see the note at the top of this file); three and four are unchanged."""
    import tempfile
    from modules.io_safe import robust_imread_diag
    with tempfile.TemporaryDirectory() as d:
        for channels, want_ndim, want_ch in ((1, 3, 3), (3, 3, 3), (4, 3, 4)):
            p = Path(d) / f"normal_{channels}.tif"
            _write_tiff(p, channels)
            img, diag = robust_imread_diag(str(p), _retry_delays=())
            assert img is not None, f"{channels}-channel TIFF failed to load: {diag}"
            assert img.ndim == want_ndim, f"{channels}ch -> {img.shape}"
            if want_ch is not None:
                assert img.shape[2] == want_ch, f"{channels}ch -> {img.shape}"


def test_handled_image_library_logging_is_not_reported_as_a_crash():
    """Pillow's refusals are expected (io_safe tries readers in turn), so they
    must not reach Sentry as events in either process."""
    gui = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    worker = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    for name, src in (("star_trail_cleanr.py", gui), ("astro_clean_v5.py", worker)):
        assert "ignore_logger" in src, f"{name} must silence handled PIL logging"
        assert '"PIL"' in src or "'PIL'" in src, (
            f"{name} must name the PIL logger it silences")
