"""Turning a detection into a mask must stay cheap, in both stages.

Field case, 2026-08-26. Kari Tuomi's PC took 66 seconds a frame where a Mac took
34. Profiling the two biggest processor-bound stages on his own frames found the
same thing at the top of both: building a full-frame mask for each AI detection.

Three separate wastes, all fixed together, all output-identical:

1. SAHI's `mask.bool_mask` returns FLOAT64 despite its name. Its decoder does
   `np.zeros([h, w])` with no dtype, fills the polygon, then computes
   `.astype(bool)` and DISCARDS the result without assigning it. On a 44MP frame
   that is a 354 MB array where 44 MB would do -- and it is a property, so it
   recomputes on every single access. We now draw the outline into uint8
   ourselves, copying SAHI's own rounding so the filled pixels are identical.

2. Both stages built the same masks from the same detections, one after the
   other. They are now built once and handed along in the pipeline state.

3. The trim loop swept the whole photograph three times per detection to weigh
   blobs a few hundred pixels across. It now works inside each detection's own
   bounding box. THIS WAS THE THIRD APPEARANCE of that pattern, and the second
   in this very file -- the previous fix was to the loop DIRECTLY ABOVE it.

Measured on four of his frames: phantom pruning 10.30s -> 4.25s, polygon fitting
4.97s -> 2.91s, whole detect 23% faster, every output mask hash and polygon set
unchanged.
"""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

GROUPER = (REPO / "modules" / "trail_grouper.py").read_text()
PIPELINE = (REPO / "modules" / "detect_pipeline.py").read_text()


def _strip_comments(src):
    """Drop comment bodies before searching for banned patterns, so the comment
    that EXPLAINS a ban cannot trip the test that enforces it. (That exact
    own-goal happened on 2026-08-25.)"""
    return "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())


def test_masks_are_decoded_straight_into_uint8():
    """If this reverts to the library property, every detection allocates a
    354 MB float64 array again."""
    body = GROUPER[GROUPER.index("def _pred_to_mask("):]
    body = body[:body.index("\ndef ", 10)]
    assert "fillPoly" in body, (
        "the direct outline decoder is gone; we are back to SAHI's float64 "
        "bool_mask property, which is 8x the memory it needs and recomputes "
        "on every access")
    assert "np.uint8" in body, "the mask must be built as uint8, not float"
    assert "segmentation" in body, "we should read the outline, not the raster"
    assert "bool_mask" in body, (
        "the fallback for predictions that carry no outline (our own synthetic "
        "ones) must stay, or they return nothing")


def test_the_two_stages_share_one_set_of_masks():
    assert "pred_masks" in PIPELINE, "the shared mask list is gone"
    fit = PIPELINE[PIPELINE.index("def stage_fit_polygons("):]
    fit = fit[:fit.index("\ndef ", 10)]
    assert "state.pred_masks" in fit, (
        "polygon fitting no longer reuses the masks phantom pruning built, so "
        "every detection is redrawn twice per frame")
    assert "_pred_to_mask(pred, h, w)" in fit, (
        "the fall-back path is gone; an uncached detection must still work")


def test_the_trim_loop_stays_inside_the_bounding_box():
    prune = PIPELINE[PIPELINE.index("def stage_prune_phantoms("):]
    prune = prune[:prune.index("\ndef ", 10)]
    code = _strip_comments(prune)
    assert "notkill[" in code, (
        "the trim loop no longer slices `notkill` to a box; it is back to "
        "combining across the whole frame once per detection")
    for banned in ("(m > 0).sum()", "trimmed.sum()"):
        assert banned not in code, (
            f"`{banned}` is a full-frame pass per detection -- the third "
            f"occurrence of the pattern in ARCHITECTURE.md's sharp edges list")


def test_a_synthetic_prediction_without_an_outline_still_works():
    """Phantom pruning replaces trimmed detections with its own objects, which
    carry a raster and no outline. Those must survive the fast path."""
    from modules.detect_pipeline import _PredMaskWrap
    from modules.trail_grouper import _pred_to_mask

    class _NoOutline:
        def __init__(self, bm):
            self.mask = _PredMaskWrap(bm)

    bm = np.zeros((40, 50), bool)
    bm[10:20, 10:30] = True
    out = _pred_to_mask(_NoOutline(bm), 40, 50)
    assert out is not None, "a prediction with only a raster must still return a mask"
    assert out.dtype == np.uint8 and out.shape == (40, 50)
    assert int((out > 0).sum()) == 200, "the fallback must preserve the pixels"


def test_a_missing_mask_still_returns_nothing():
    from modules.trail_grouper import _pred_to_mask

    class _NoMask:
        mask = None

    assert _pred_to_mask(_NoMask(), 10, 10) is None
