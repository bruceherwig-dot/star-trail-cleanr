"""A folder that mixes 8-bit and 16-bit frames must be evened out, not crashed.

Regression test for the NightscapeArt crash (v2.46, Windows): a run died on the
LAST batch with "this folder mixes 8-bit and 16-bit images ... move one set into
a different folder". The real cause was a folder where most frames had a 16-bit
TIFF/RAW twin we keep, but a few tail frames existed only as an 8-bit JPG -- so
the sequence was 16-bit for the bulk and 8-bit at the tail, and the per-batch
dtype guard fired on whichever batch straddled the change (the last one), after
every earlier batch had already been cleaned.

The fix: the GUI computes the sequence-wide majority bit depth up front and the
worker brings every frame in each batch to that one depth, so the run completes
and every frame is cleaned. These tests lock in:
  * the header-only depth probe (image_bitdepth) buckets formats correctly,
  * the worker's target-depth and conversion helpers behave,
  * the old hard-stop message is gone and the worker accepts --expected-bitdepth,
  * the GUI passes the chosen depth to the worker.

No model inference; runs in milliseconds.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.io_safe import image_bitdepth


def _load_worker():
    """Import astro_clean_v5 tolerating SystemExit, like test_frame_dedup does."""
    spec = importlib.util.spec_from_file_location("astro_clean_v5", REPO / "astro_clean_v5.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


# ── image_bitdepth: header-only depth bucket ───────────────────────────────

def test_bitdepth_jpeg_is_8():
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.jpg"
        Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8)).save(p, "JPEG")
        assert image_bitdepth(p) == 8, image_bitdepth(p)


def test_bitdepth_8bit_tiff_is_8():
    import tifffile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.tif"
        tifffile.imwrite(str(p), np.zeros((64, 96, 3), dtype=np.uint8), photometric="rgb")
        assert image_bitdepth(p) == 8, image_bitdepth(p)


def test_bitdepth_16bit_tiff_is_16():
    import tifffile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.tif"
        tifffile.imwrite(str(p), np.full((64, 96, 3), 40000, dtype=np.uint16),
                         photometric="rgb")
        assert image_bitdepth(p) == 16, image_bitdepth(p)


def test_bitdepth_raw_extension_is_16_without_decoding():
    # RAW is always debayered to 16-bit; the probe returns 16 by extension
    # without opening the file (so it stays cheap and needs no rawpy fixture).
    assert image_bitdepth("/does/not/exist/frame.nef") == 16
    assert image_bitdepth("/does/not/exist/frame.cr3") == 16


def test_bitdepth_unreadable_is_none():
    assert image_bitdepth("/does/not/exist/frame.jpg") is None


# ── worker: target-depth selection ─────────────────────────────────────────

def test_resolve_target_honors_expected_16():
    w = _load_worker()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]  # majority is 8-bit...
    # ...but the GUI's explicit choice wins.
    assert w._resolve_target_dtype(frames, 16) == np.uint16


def test_resolve_target_honors_expected_8():
    w = _load_worker()
    frames = [np.zeros((4, 4, 3), dtype=np.uint16)]
    assert w._resolve_target_dtype(frames, 8) == np.uint8


def test_resolve_target_falls_back_to_majority():
    w = _load_worker()
    u8 = np.zeros((4, 4, 3), dtype=np.uint8)
    u16 = np.zeros((4, 4, 3), dtype=np.uint16)
    assert w._resolve_target_dtype([u8, u8, u16], None) == np.uint8
    assert w._resolve_target_dtype([u16, u16, u8], None) == np.uint16


def test_resolve_target_tie_prefers_16():
    w = _load_worker()
    u8 = np.zeros((4, 4, 3), dtype=np.uint8)
    u16 = np.zeros((4, 4, 3), dtype=np.uint16)
    assert w._resolve_target_dtype([u8, u16], None) == np.uint16


# ── worker: depth conversion ───────────────────────────────────────────────

def test_match_bitdepth_same_dtype_is_identity():
    w = _load_worker()
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    assert w._match_bitdepth(a, np.uint8) is a  # identity drives the change-count


def test_match_bitdepth_8_to_16_fills_range():
    w = _load_worker()
    a = np.array([[0, 1, 255]], dtype=np.uint8)
    out = w._match_bitdepth(a, np.uint16)
    assert out.dtype == np.uint16
    # 255 must map to full-scale 65535 (the *257 fill the save path uses).
    assert out.tolist() == [[0, 257, 65535]], out.tolist()


def test_match_bitdepth_16_to_8_drops_low_byte():
    w = _load_worker()
    a = np.array([[0, 256, 65535]], dtype=np.uint16)
    out = w._match_bitdepth(a, np.uint8)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1, 255]], out.tolist()


def test_mixed_batch_normalizes_to_uniform_dtype():
    """The whole point: a mixed list collapses to one dtype with no crash."""
    w = _load_worker()
    frames = [np.zeros((4, 4, 3), dtype=np.uint16),
              np.zeros((4, 4, 3), dtype=np.uint8),
              np.zeros((4, 4, 3), dtype=np.uint16)]
    target = w._resolve_target_dtype(frames, 16)
    out = [w._match_bitdepth(f, target) for f in frames]
    assert len({f.dtype for f in out}) == 1, [f.dtype for f in out]
    assert out[0].dtype == np.uint16


# ── source guards: old crash gone, new wiring present ──────────────────────

def test_old_hard_stop_message_removed():
    text = (REPO / "astro_clean_v5.py").read_text()
    assert "mixes 8-bit and 16-bit images" not in text, (
        "The old hard-stop that aborted the run on a mixed-depth folder is back. "
        "Mixed depth must be evened out, not refused."
    )


def test_worker_accepts_expected_bitdepth_flag():
    text = (REPO / "astro_clean_v5.py").read_text()
    assert '"--expected-bitdepth"' in text, (
        "astro_clean_v5.py no longer defines --expected-bitdepth; the GUI relies "
        "on it to pin the sequence-wide target depth."
    )


def test_gui_passes_expected_bitdepth():
    text = (REPO / "star_trail_cleanr.py").read_text()
    assert "--expected-bitdepth" in text, (
        "The GUI no longer passes --expected-bitdepth to the worker, so batches "
        "would each guess their own target and could disagree across a run."
    )
