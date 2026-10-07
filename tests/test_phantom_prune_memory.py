"""Phantom pruning must not run a huge frame out of memory.

Sentry 2026-10-07 (v2.98, Windows 11): 11648 x 8736 frames (102 MP, a Fujifilm GFX100)
died with "Unable to allocate 776 MiB for an array with shape (8736, 11648) and data
type float64". The stage kept one full-frame mask per detection (about 100 MB each at
that size, dozens of detections), ran short, and a catch-all then fell through to the
SAHI library's own mask path, which needs a float64 array the size of the frame.

The fix: past a memory budget the stage builds masks one at a time, with identical
results, and a MemoryError is reported plainly instead of falling into the heavier path.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import detect_pipeline as dp
from modules.detect_pipeline import PipelineState, StageConfig, _SyntheticPred


class _Log:
    """A stand-in for StageLog that records what the stage counted."""
    def __init__(self):
        self.counts, self.events = {}, []

    def count(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def event(self, reason, **detail):
        self.events.append((reason, detail))


def _scene():
    """Dark empty sky, with: a thick real trail (bright under it), and a thin dotted
    phantom line over empty sky that the stage should remove."""
    h, w = 400, 700
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), 20, np.uint8) + rng.integers(0, 3, (h, w, 3), dtype=np.uint8)
    real = np.zeros((h, w), bool)
    cv2.line(img, (50, 60), (650, 120), (230, 230, 230), 7)      # a real, bright trail
    cv2.line(real.view(np.uint8), (50, 60), (650, 120), 1, 7)
    real = real.astype(bool)
    thick = np.zeros((h, w), np.uint8)
    cv2.line(thick, (50, 60), (650, 120), 1, 11)
    phantom = np.zeros((h, w), np.uint8)
    for x in range(60, 640, 6):                                  # dotted, over empty sky
        phantom[300:302, x:x + 3] = 1
    preds = [_SyntheticPred(thick.astype(bool), 0.9), _SyntheticPred(phantom.astype(bool), 0.5)]
    return img, preds


def _run(budget):
    img, preds = _scene()
    old = dp._PHANTOM_MASK_CACHE_BYTES
    dp._PHANTOM_MASK_CACHE_BYTES = budget
    try:
        state = PipelineState(image=img, raw_detections=list(preds))
        log = _Log()
        state = dp.stage_prune_phantoms(state, StageConfig(prune_phantoms=True), log)
    finally:
        dp._PHANTOM_MASK_CACHE_BYTES = old
    return state, log


def test_the_lean_path_gives_exactly_the_same_answer_as_the_normal_one():
    normal, nlog = _run(10 ** 12)          # budget far above the need: the cached path
    lean, llog = _run(1)                   # budget below any need: one mask at a time
    assert nlog.counts.get("detections_dropped", 0) + nlog.counts.get("detections_trimmed", 0) > 0, \
        "the scene must actually have something pruned, or this compares nothing"
    assert nlog.counts == llog.counts, (nlog.counts, llog.counts)
    assert len(normal.raw_detections) == len(lean.raw_detections)
    h, w = normal.image.shape[:2]
    for a, b in zip(normal.raw_detections, lean.raw_detections):
        ma, mb = dp._pred_to_mask(a, h, w), dp._pred_to_mask(b, h, w)
        assert np.array_equal(ma, mb), "a detection's mask differs between the two paths"


def test_the_lean_path_does_not_hold_the_masks():
    lean, _ = _run(1)
    assert all(m is None for m in lean.pred_masks), \
        "lean mode must not keep full-frame masks alive (that is the memory it exists to save)"
    normal, _ = _run(10 ** 12)
    assert any(m is not None for m in normal.pred_masks), \
        "an ordinary frame must keep the fast cached path"


def test_the_budget_separates_a_gfx100_from_a_busy_44mp_frame():
    """Numbers from the field: a busy 44 MP frame (53 detections) holds ~2.3 GB of masks
    and must stay on the fast path; a 102 MP frame with a few dozen detections must not."""
    b = dp._PHANTOM_MASK_CACHE_BYTES
    assert 53 * 44_000_000 < b, "a busy 44 MP frame would be slowed for no reason"
    assert 30 * 11648 * 8736 > b, "a GFX100 frame with 30 detections would still run out of memory"


def test_a_memory_error_is_reported_plainly_not_swallowed_into_the_heavier_path():
    from modules import trail_grouper as tg

    class _Mask:
        full_shape = (8736, 11648)
        segmentation = [[10, 10, 50, 10, 50, 50, 10, 50]]

        @property
        def bool_mask(self):
            raise AssertionError("the library's float64 path must never be reached")

    class _Pred:
        mask = _Mask()

    real_zeros = np.zeros

    def _short_of_memory(*a, **k):
        raise MemoryError("Unable to allocate 100 MiB")
    tg.np.zeros = _short_of_memory
    try:
        try:
            tg._pred_to_mask(_Pred(), 8736, 11648)
            raised = None
        except MemoryError as e:
            raised = str(e)
    finally:
        tg.np.zeros = real_zeros
    assert raised is not None, "a MemoryError must propagate, not be swallowed"
    assert "11648" in raised and "8736" in raised and "memory" in raised.lower(), raised
