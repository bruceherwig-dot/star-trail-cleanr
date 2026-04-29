"""16-bit TIFF input must be normalized to 8-bit BGR before reaching SAHI.

Regression test for the v1.81 → v1.91 bug a Windows tester (gmeye) hit when
loading 16-bit RGB TIFFs from a Nikon Z6ii. SAHI's internal PIL loader calls
Image.fromarray on whatever it receives, and PIL has no 16-bit RGB mode, so
a uint16 RGB array crashes with `Cannot handle this data type: (1, 1, 3), <u2`.

The previous version of this test relied on cv2.imwrite to create the test
fixture. That writes through the same OpenCV TIFF codec that decodes, so
round-trip downcasts to uint8 and the test passed even when real-world TIFFs
(written by Lightroom / tifffile / other professional tools) did NOT downcast
on Windows. This rewrite uses tifffile so the on-disk file looks like what
users actually export.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _write_realistic_16bit_rgb_tiff(path: Path):
    """Write a 16-bit RGB TIFF the way Lightroom / pro photo apps do.

    Uses tifffile (the library most scientific and pro-photo TIFF writers
    use under the hood) rather than cv2.imwrite, so on Windows the OpenCV
    decoder can fail to downcast — exactly the failure mode that hit users.
    """
    import tifffile
    arr = np.zeros((128, 192, 3), dtype=np.uint16)
    arr[..., 0] = 5000
    arr[..., 1] = 20000
    arr[..., 2] = 50000
    tifffile.imwrite(str(path), arr, photometric="rgb", compression="deflate")


class _FakeSlicedResult:
    object_prediction_list = []


def _patch_sahi_capture(captured: dict):
    """Replace SAHI calls with stubs that record the image they received.

    Returns an undo callable. Patches both the hybrid and non-hybrid paths
    so the test stays correct regardless of which one is active.
    """
    import sahi.predict as _sp
    import sahi.slicing as _ss

    real_gsp = _sp.get_sliced_prediction
    real_si = _ss.slice_image

    def fake_get_sliced_prediction(image, detection_model, **kwargs):
        captured["image"] = image
        return _FakeSlicedResult()

    class _FakeSliceResult:
        original_image_height = 1
        original_image_width = 1
        images = []
        starting_pixels = []
        def __len__(self): return 0

    def fake_slice_image(image, **kwargs):
        captured["image"] = image
        return _FakeSliceResult()

    _sp.get_sliced_prediction = fake_get_sliced_prediction
    _ss.slice_image = fake_slice_image

    def undo():
        _sp.get_sliced_prediction = real_gsp
        _ss.slice_image = real_si

    return undo


# ── Tests against realistic 16-bit RGB TIFF input ──────────────────────────

def test_detect_frame_handles_real_16bit_rgb_tiff_path():
    """Path-mode: detect_frame opens the file itself, normalizes to uint8 BGR,
    SAHI never sees uint16 — regardless of what cv2.imread returns on the
    host OS for this particular TIFF."""
    import modules.detect_trails as dt
    captured = {}
    undo = _patch_sahi_capture(captured)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "real_16bit_rgb.tif"
            _write_realistic_16bit_rgb_tiff(p)
            mask = dt.detect_frame(model=None, image=str(p),
                                   tile_size=640, overlap=0.2, dilate=0)
    finally:
        undo()

    arr = captured.get("image")
    assert arr is not None, "SAHI was never reached"
    assert isinstance(arr, np.ndarray), \
        f"SAHI got {type(arr).__name__}, expected numpy array"
    assert arr.dtype == np.uint8, \
        f"SAHI got {arr.dtype}, expected uint8 (the v1.91 crash)"
    assert arr.ndim == 3 and arr.shape[2] == 3, \
        f"SAHI got shape {arr.shape}, expected (H, W, 3)"


def test_detect_frame_handles_uint16_rgb_array_input():
    """Array-mode: the production worker passes its pre-loaded array. If a
    16-bit array arrives directly (legacy callers, future code), the safety
    net inside detect_frame must still convert it to uint8 BGR."""
    import modules.detect_trails as dt
    captured = {}
    undo = _patch_sahi_capture(captured)
    try:
        u16 = np.full((100, 100, 3), 30000, dtype=np.uint16)
        dt.detect_frame(model=None, image=u16,
                        tile_size=640, overlap=0.2, dilate=0)
    finally:
        undo()

    arr = captured["image"]
    assert arr.dtype == np.uint8, f"uint16 input not downcast, got {arr.dtype}"
    assert arr.shape == (100, 100, 3)


def test_detect_frame_handles_uint8_array_passthrough():
    """Array-mode happy path: pre-loaded uint8 BGR is passed through unchanged."""
    import modules.detect_trails as dt
    captured = {}
    undo = _patch_sahi_capture(captured)
    try:
        u8 = np.zeros((50, 50, 3), dtype=np.uint8)
        dt.detect_frame(model=None, image=u8,
                        tile_size=640, overlap=0.2, dilate=0)
    finally:
        undo()

    arr = captured["image"]
    assert arr.dtype == np.uint8
    assert arr.shape == (50, 50, 3)


def test_detect_frame_handles_grayscale_array():
    """Single-channel input is widened to BGR so SAHI sees 3 channels."""
    import modules.detect_trails as dt
    captured = {}
    undo = _patch_sahi_capture(captured)
    try:
        gray = np.zeros((50, 50), dtype=np.uint8)
        dt.detect_frame(model=None, image=gray,
                        tile_size=640, overlap=0.2, dilate=0)
    finally:
        undo()

    arr = captured["image"]
    assert arr.shape == (50, 50, 3), f"grayscale not widened, got {arr.shape}"


def test_detect_frame_handles_rgba_array():
    """4-channel input is reduced to BGR. SAHI's PIL conversion does not
    handle 4-channel BGR arrays cleanly."""
    import modules.detect_trails as dt
    captured = {}
    undo = _patch_sahi_capture(captured)
    try:
        rgba = np.zeros((50, 50, 4), dtype=np.uint8)
        dt.detect_frame(model=None, image=rgba,
                        tile_size=640, overlap=0.2, dilate=0)
    finally:
        undo()

    arr = captured["image"]
    assert arr.shape == (50, 50, 3), f"4-channel not reduced, got {arr.shape}"


def test_production_call_site_passes_array_not_path():
    """The worker (astro_clean_v5.py) must hand detect_frame the pre-loaded
    8-bit array, not the path. Re-loading inside detect_frame is wasteful
    AND the v1.91 crash source — the default cv2.imread does not reliably
    downcast 16-bit on every Windows + TIFF combination."""
    text = (REPO / "astro_clean_v5.py").read_text()
    assert "detect_frame(model, frames_8bit_all[i]" in text, (
        "astro_clean_v5.py no longer hands detect_frame the pre-loaded 8-bit "
        "array. It must NOT regress to passing a file path — that re-loads "
        "from disk and risks the v1.91 uint16-to-SAHI crash on Windows."
    )
    assert "detect_frame(model, str(fp)" not in text, (
        "astro_clean_v5.py is passing a file path to detect_frame again — "
        "that re-loads from disk and can leak uint16 to SAHI on Windows."
    )
