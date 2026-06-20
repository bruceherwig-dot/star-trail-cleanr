"""Run-time estimator: input-format classification + the calibrated cost model.

These lock the up-front ballpark math (the part that doesn't need a real run):
format detection from filename + bit depth, and that the per-megapixel costs
produce sane estimates with RAW/16-bit slower than JPG/8-bit. The live per-machine
tuning is only exercisable on a real multi-batch run, not here.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import star_trail_cleanr as stc


def test_format_key_classifies_each_input():
    assert stc._est_format_key(["IMG_0001.jpg"], 8) == "jpg"
    assert stc._est_format_key(["IMG_0001.JPEG"], 8) == "jpg"
    assert stc._est_format_key(["scene.tif"], 8) == "tiff8"
    assert stc._est_format_key(["scene.tif"], 16) == "tiff16"
    assert stc._est_format_key(["scene.tiff"], 16) == "tiff16"
    assert stc._est_format_key(["DSC_0001.NEF"], None) == "raw"
    assert stc._est_format_key(["IMG_1234.CR3"], None) == "raw"
    # empty list must not crash; falls back to the JPG baseline
    assert stc._est_format_key([], None) == "jpg"


def test_cost_model_ballpark_is_sane():
    # 100 frames at 24 MP JPG: frames * MP * K, padded 1.2x -> roughly 8-9 min,
    # matching Bruce's real 24MP JPG runs. (EST_PAD_FACTOR is 1.20 in the worker.)
    spf = stc._EST_SEC_PER_MP["jpg"] * 24.0
    est = 100 * spf * 1.20
    assert 7 * 60 < est < 11 * 60, f"24MP JPG 100-frame estimate off: {est:.0f}s"


def test_raw_and_16bit_cost_more_than_jpg_and_8bit():
    assert stc._EST_SEC_PER_MP["raw"] > stc._EST_SEC_PER_MP["jpg"]
    assert stc._EST_SEC_PER_MP["tiff16"] > stc._EST_SEC_PER_MP["tiff8"]


def test_default_cost_sits_within_the_known_formats():
    vals = list(stc._EST_SEC_PER_MP.values())
    assert min(vals) <= stc._EST_SEC_PER_MP_DEFAULT <= max(vals)
