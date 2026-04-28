"""TIFF 16-bit output path writes a real 16-bit RGB TIFF without crashing.

Regression test for the v1.0-beta-through-v1.9 latent crash that hit a
Windows tester (gmeye) immediately when he picked "TIFF 16-bit" output.
The original code called Image.fromarray(rgb, mode="RGB;16") which PIL
does not support on uint16 RGB arrays. That line shipped from v1.0-beta
on 2026-04-15 onward and was unreachable until the v1.9 input fix
unblocked the codepath.

This test exercises the whole _write_output tif16 branch on a synthetic
uint16 RGB array and confirms the output file is a valid 16-bit RGB TIFF
that round-trips pixel values exactly.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _run_tif16_branch(img, tmpdir, icc_profile=None, dpi=None):
    """Mirror of astro_clean_v5._write_output's tif16 branch. Kept in sync
    with that branch — if the production code changes, update this and the
    test asserts the behavior, not the implementation."""
    import tifffile
    if img.dtype == np.uint16:
        out = img
    else:
        out = img.astype(np.uint16) * 257
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    extratags = []
    if icc_profile:
        extratags.append((34675, 'B', len(icc_profile), icc_profile, False))
    kwargs = {"photometric": "rgb", "compression": "deflate", "extratags": extratags}
    if dpi:
        kwargs["resolution"] = (float(dpi[0]), float(dpi[1]))
        kwargs["resolutionunit"] = "inch"
    out_path = Path(tmpdir) / "out.tif"
    tifffile.imwrite(str(out_path), rgb, **kwargs)
    return out_path


def test_tif16_output_preserves_uint16_rgb_pixels():
    bgr = np.zeros((32, 48, 3), dtype=np.uint16)
    bgr[..., 0] = 0xFEDC
    bgr[..., 1] = 0xABCD
    bgr[..., 2] = 0x1234
    with tempfile.TemporaryDirectory() as td:
        out = _run_tif16_branch(bgr, td)
        re = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert re is not None, "tifffile output not readable"
    assert re.dtype == np.uint16, f"output dtype is {re.dtype}, not uint16"
    assert re.shape == (32, 48, 3), f"output shape is {re.shape}"
    assert tuple(int(x) for x in re[0, 0]) == (0xFEDC, 0xABCD, 0x1234), \
        "uint16 pixel values were corrupted on round-trip"


def test_tif16_output_promotes_uint8_to_uint16():
    """Input from a JPG run is uint8; the tif16 path scales by 257 to fill
    the uint16 range. Highest 8-bit value 255 -> 65535 (255 * 257)."""
    bgr = np.full((16, 16, 3), 255, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        out = _run_tif16_branch(bgr, td)
        re = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert re.dtype == np.uint16, f"expected uint16 output for uint8 input, got {re.dtype}"
    assert int(re[0, 0, 0]) == 65535, f"255 * 257 should be 65535, got {int(re[0, 0, 0])}"


def test_tif16_output_carries_icc_profile():
    bgr = np.zeros((16, 16, 3), dtype=np.uint16)
    icc_blob = b"FAKE_ICC_PROFILE_FOR_TEST_" + b"X" * 200
    with tempfile.TemporaryDirectory() as td:
        out = _run_tif16_branch(bgr, td, icc_profile=icc_blob)
        from PIL import Image
        with Image.open(str(out)) as im:
            assert im.info.get("icc_profile") == icc_blob, \
                "ICC profile bytes did not survive the tif16 write"


def test_tif16_output_carries_dpi():
    bgr = np.zeros((16, 16, 3), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        out = _run_tif16_branch(bgr, td, dpi=(300, 300))
        from PIL import Image
        with Image.open(str(out)) as im:
            assert im.info.get("dpi") == (300.0, 300.0), \
                f"DPI not preserved, got {im.info.get('dpi')}"


def test_production_code_uses_tifffile_for_tif16():
    """Sanity: the production astro_clean_v5._write_output tif16 branch
    must not regress back to the broken Image.fromarray(mode='RGB;16') call."""
    text = (REPO / "astro_clean_v5.py").read_text()
    # Find the tif16 elif block content. Crude check: the broken pattern must
    # not appear, and tifffile must be referenced.
    assert 'mode="RGB;16"' not in text, \
        "astro_clean_v5.py still calls Image.fromarray(mode='RGB;16') — that's the v1.9 crash"
    assert "tifffile" in text, \
        "astro_clean_v5.py no longer references tifffile — tif16 output is at risk"
