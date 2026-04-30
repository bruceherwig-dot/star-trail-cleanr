"""Robust image read with tifffile fallback and per-file diagnosis.

Regression test for v1.99: a Windows tester (Warren) hit a TIFFReadRGBAStrip
crash from OpenCV's bundled libtiff on a TIFF that tifffile reads fine. The
worker exited mid-batch instead of recovering. This suite locks in:

  * cv2.imread happy path is unchanged (drop-in compatible).
  * tifffile fallback rescues TIFFs OpenCV can't decode.
  * When every reader fails, robust_imread_diag returns a useful diagnosis
    so the worker can show the user the actual reason.
  * The production worker imports the diag variant and the in-pipeline call
    sites still use the wrapper (no regression to bare cv2.imread).
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_robust_imread_passes_through_when_opencv_works():
    from modules.io_safe import robust_imread
    arr = np.zeros((50, 60, 3), dtype=np.uint8)
    arr[10, 20] = (10, 20, 30)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ok.png"
        cv2.imwrite(str(p), arr)
        img = robust_imread(p)
    assert img is not None, "cv2.imread happy path returned None"
    assert tuple(img[10, 20].tolist()) == (10, 20, 30)


def test_robust_imread_recovers_via_tifffile_when_cv2_fails():
    """Simulates the v1.98 Warren crash by patching cv2.imread to return
    None. tifffile fallback must produce a usable BGR array."""
    import tifffile
    from modules import io_safe
    from modules.io_safe import robust_imread

    arr = np.zeros((64, 96, 3), dtype=np.uint16)
    arr[..., 0] = 5000
    arr[..., 1] = 20000
    arr[..., 2] = 50000

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "real.tif"
        tifffile.imwrite(str(p), arr, photometric="rgb", compression="deflate")

        real_imread = io_safe.cv2.imread
        try:
            io_safe.cv2.imread = lambda *a, **kw: None
            img = robust_imread(p, cv2.IMREAD_UNCHANGED, _retry_delays=())
        finally:
            io_safe.cv2.imread = real_imread

    assert img is not None, "tifffile fallback did not recover the read"
    assert img.shape == (64, 96, 3), f"shape wrong: {img.shape}"
    assert img.dtype == np.uint16, f"dtype lost: {img.dtype}"
    assert int(img[0, 0, 0]) == 50000, "channel 0 should be blue (BGR layout)"
    assert int(img[0, 0, 2]) == 5000, "channel 2 should be red (BGR layout)"


def test_robust_imread_returns_none_when_every_reader_fails():
    from modules.io_safe import robust_imread
    img = robust_imread("/tmp/nonexistent_xyz_qzqz.tif", _retry_delays=())
    assert img is None


def test_pil_fallback_rescues_jpeg_when_cv2_fails():
    """v1.99.1 fix: cv2.imread on Windows fails on paths with non-ASCII
    characters (Slovak, Czech, German, French, etc.) BEFORE it tries to
    decode. We can't reproduce the Windows-specific path failure on macOS,
    but we can simulate the pattern by patching cv2.imread to return None
    on a JPEG. PIL must rescue it.

    This is the regression test for tester `magio` whose run died on
    `C:\\Users\\magio\\Desktop\\Štrba\\svetlá\\ZFC_6071.jpg`.
    """
    import cv2 as _cv2
    from modules import io_safe
    from modules.io_safe import robust_imread

    # Uniform-color image: JPEG compression is lossless on flat regions, so
    # the exact pixel value survives the round trip. The test cares about
    # "did PIL rescue the read," not about JPEG fidelity.
    arr = np.full((40, 60, 3), 100, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ZFC_6071.jpg"
        _cv2.imwrite(str(p), arr)

        real_imread = io_safe.cv2.imread
        try:
            io_safe.cv2.imread = lambda *a, **kw: None
            img = robust_imread(p, _cv2.IMREAD_UNCHANGED, _retry_delays=())
        finally:
            io_safe.cv2.imread = real_imread

    assert img is not None, (
        "PIL fallback failed to rescue a JPEG when cv2.imread returned None. "
        "This is the v1.99.1 fix for Windows non-ASCII paths."
    )
    assert img.shape == (40, 60, 3)
    assert img.dtype == np.uint8
    # Uniform-color JPEG — every pixel should be exactly 100 in every channel.
    assert int(img[10, 20, 0]) == 100
    assert int(img[10, 20, 1]) == 100
    assert int(img[10, 20, 2]) == 100


def test_pil_fallback_present_in_diag():
    """When all readers fail, the diagnosis must name Pillow as one of the
    attempts so support / Sentry data can show users WHICH readers we tried."""
    from modules.io_safe import robust_imread_diag
    img, diag = robust_imread_diag(
        "/tmp/definitely_does_not_exist_qzqz_v2.jpg", _retry_delays=()
    )
    assert img is None
    assert diag is not None
    assert "OpenCV" in diag, "diagnosis missing OpenCV"
    assert "Pillow" in diag, (
        "diagnosis is missing Pillow attempt — the v1.99.1 fallback isn't "
        "being exercised"
    )


def test_diag_reports_underlying_reason_on_failure():
    """The diag variant must return a non-empty diagnosis when reads fail,
    so the worker can show the user the actual cause instead of a vague
    'cannot read'. Captures the v1.99 contract: every failure surfaces a
    reason. """
    from modules.io_safe import robust_imread_diag
    img, diag = robust_imread_diag(
        "/tmp/definitely_does_not_exist_qzqz.tif", _retry_delays=()
    )
    assert img is None
    assert diag is not None and len(diag) > 0, (
        "diagnosis missing for failed read"
    )
    assert "OpenCV" in diag, "diagnosis should name the reader that was tried"


def test_diag_returns_none_diagnosis_on_success():
    from modules.io_safe import robust_imread_diag
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ok.png"
        cv2.imwrite(str(p), arr)
        img, diag = robust_imread_diag(p)
    assert img is not None
    assert diag is None, f"diagnosis should be None on success, got: {diag}"


def test_grayscale_flag_through_tifffile_fallback():
    """IMREAD_GRAYSCALE on a 3-channel TIFF goes through tifffile cleanly."""
    import tifffile
    from modules import io_safe
    from modules.io_safe import robust_imread

    arr = np.full((30, 40, 3), 100, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gray.tif"
        tifffile.imwrite(str(p), arr, photometric="rgb")

        real_imread = io_safe.cv2.imread
        try:
            io_safe.cv2.imread = lambda *a, **kw: None
            img = robust_imread(p, cv2.IMREAD_GRAYSCALE, _retry_delays=())
        finally:
            io_safe.cv2.imread = real_imread

    assert img is not None
    assert img.ndim == 2, f"expected 2D grayscale, got shape {img.shape}"
    assert img.shape == (30, 40)


def test_production_callsites_use_robust_imread():
    """Lock in the Warren v1.98 fix: the frame-load loop and 16-bit hot
    pixel reload must use the wrapper, not bare cv2.imread."""
    text = (REPO / "astro_clean_v5.py").read_text()
    assert "from modules.io_safe import robust_imread, robust_imread_diag" in text, (
        "astro_clean_v5.py no longer imports the io_safe helpers"
    )
    assert "robust_imread_diag(fp, cv2.IMREAD_UNCHANGED)" in text, (
        "frame-load loop no longer uses robust_imread_diag — without the "
        "diagnosis the worker can't tell the user why a read failed"
    )
    assert "orig = robust_imread(frame_files[i], cv2.IMREAD_UNCHANGED)" in text, (
        "16-bit hot-pixel reload regressed to bare cv2.imread"
    )

    dt_text = (REPO / "modules" / "detect_trails.py").read_text()
    assert "from .io_safe import robust_imread" in dt_text, (
        "modules/detect_trails.py no longer imports robust_imread"
    )
    assert "img = robust_imread(image, cv2.IMREAD_UNCHANGED)" in dt_text, (
        "detect_frame regressed to bare cv2.imread"
    )


def test_worker_prompts_gui_on_unreadable_file():
    """The worker must hand the bad-file decision to the GUI rather than
    silently skipping or sys.exit-ing. Lock in the sentinel + stdin protocol
    so the GUI can show the user a real modal."""
    text = (REPO / "astro_clean_v5.py").read_text()
    assert "STC_BAD_FILE_PROMPT:" in text, (
        "worker no longer emits the bad-file prompt sentinel — GUI can't "
        "show a modal without it"
    )
    assert "_prompt_gui_for_bad_file" in text, (
        "_prompt_gui_for_bad_file helper is missing from the worker"
    )
    assert "sys.stdin.readline()" in text, (
        "worker no longer reads stdin — it can't wait for the user's decision"
    )
    # Core pointers must still be rebound after skipping.
    assert "core_start -= skipped_before_core" in text, (
        "core_start adjustment for skipped frames is missing"
    )
    assert "core_end -= skipped_before_core + skipped_in_core" in text, (
        "core_end adjustment for skipped frames is missing"
    )


def test_worker_captures_unreadable_files_to_sentry():
    """The worker must fire a Sentry warning on each skipped file so we get
    diagnostic data even from users who never email us. Fingerprint must be
    set so a tester with many bad files doesn't flood the Sentry inbox with
    distinct issues."""
    text = (REPO / "astro_clean_v5.py").read_text()
    assert "_capture_unreadable_file_to_sentry" in text, (
        "Sentry capture helper is missing — we'd never see these failures"
    )
    assert "worker_unreadable_file" in text, (
        "Sentry fingerprint / event tag is missing — events will scatter "
        "instead of grouping into one issue"
    )
    assert 'level="warning"' in text, (
        "Sentry event must be 'warning', not 'error' — these are handled, "
        "not crashes, and the level affects email volume"
    )


def test_gui_wires_bad_file_dialog():
    """The GUI side of the dialog protocol: signal, slot, stdin response."""
    text = (REPO / "star_trail_cleanr.py").read_text()
    assert "bad_file_prompt = Signal(str, str)" in text, (
        "CleanerWorker is missing the bad_file_prompt signal"
    )
    assert "too_many_bad_files = Signal(int)" in text, (
        "CleanerWorker is missing the too_many_bad_files signal"
    )
    assert "STC_BAD_FILE_PROMPT:" in text, (
        "GUI no longer detects the worker's bad-file sentinel"
    )
    assert "self._proc.stdin.write(decision" in text, (
        "GUI no longer writes the user's decision back to the worker"
    )
    assert "stdin=subprocess.PIPE" in text, (
        "Popen no longer pipes stdin — worker can't receive the GUI's "
        "decision"
    )
    assert "def _on_bad_file_prompt" in text, (
        "MainWindow slot for bad-file modal is missing"
    )
    assert "def _on_too_many_bad_files" in text, (
        "MainWindow slot for the run-wide cap notice is missing"
    )
    assert "self.worker.bad_file_prompt.connect" in text, (
        "MainWindow no longer connects to the bad_file_prompt signal"
    )
    assert "self.worker.too_many_bad_files.connect" in text, (
        "MainWindow no longer connects to the too_many_bad_files signal"
    )
    # Run-wide cap: 1 strike — after the user-confirmed first skip, any
    # second failure auto-stops the run.
    assert "BAD_FILE_SKIP_CAP = 1" in text, (
        "run-wide skip cap is missing or has been changed without updating "
        "this test"
    )
