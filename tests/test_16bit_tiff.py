"""16-bit TIFF input must reach the trail detector as 8-bit BGR.

Regression test for the bug a Windows user hit on v1.8 with Lightroom-exported
16-bit TIFFs from a Nikon Z6ii: SAHI's internal PIL loader crashes on 16-bit
RGB with `Cannot handle this data type: (1, 1, 3), <u2`. Our fix is to load
the file with cv2.imread (which downcasts to 8-bit BGR) and pass the array
to SAHI instead of the file path.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _write_synthetic_16bit_rgb_tiff(path: Path):
    arr = np.zeros((64, 96, 3), dtype=np.uint16)
    arr[..., 0] = 5000
    arr[..., 1] = 20000
    arr[..., 2] = 50000
    cv2.imwrite(str(path), arr)


def test_cv2_imread_downcasts_16bit_rgb_tiff_to_uint8_bgr():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "synthetic_16bit_rgb.tif"
        _write_synthetic_16bit_rgb_tiff(p)
        img = cv2.imread(str(p))
        assert img is not None, "cv2.imread returned None on 16-bit RGB TIFF"
        assert img.dtype == np.uint8, f"expected uint8, got {img.dtype}"
        assert img.shape == (64, 96, 3), f"unexpected shape {img.shape}"


def test_detect_frame_passes_uint8_array_to_sahi_not_path():
    """detect_frame must hand SAHI the loaded uint8 array, not the file path.
    SAHI's PIL loader crashes on 16-bit RGB TIFFs; passing the array bypasses
    that loader entirely. We probe both code paths by patching SAHI."""
    import modules.detect_trails as dt

    captured = {}

    class _FakeResult:
        object_prediction_list = []

    def _fake_get_sliced_prediction(image, detection_model, **kwargs):
        captured["image"] = image
        return _FakeResult()

    import sahi.predict as _sp
    real_fn = _sp.get_sliced_prediction
    _sp.get_sliced_prediction = _fake_get_sliced_prediction
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "synthetic_16bit_rgb.tif"
            _write_synthetic_16bit_rgb_tiff(p)
            dt.detect_frame(model=None, img_path=str(p),
                            tile_size=640, overlap=0.2, dilate=0)
    finally:
        _sp.get_sliced_prediction = real_fn

    image_arg = captured.get("image")
    assert image_arg is not None, "SAHI was never called"
    assert isinstance(image_arg, np.ndarray), \
        f"SAHI got {type(image_arg).__name__}, expected numpy array"
    assert image_arg.dtype == np.uint8, \
        f"SAHI got dtype {image_arg.dtype}, expected uint8 (the bug)"
    assert image_arg.shape[2] == 3, \
        f"SAHI got shape {image_arg.shape}, expected 3-channel BGR"
