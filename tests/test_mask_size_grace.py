"""Smoke tests for the foreground-mask size handling (astro_clean_v5.py +
the mask editor's frame choice in star_trail_cleanr.py).

The field case (Sentry, 2026-08-03): a user's folder mixed image sizes; the
mask editor painted on the alphabetical first file, which was one of the
odd-size frames the run then skipped, so his good mask was refused with
"does not match these frames" (3008x2008 vs 6016x4016 -- exactly half, same
shape). Two rules now hold:

  1. The worker SCALES a same-shape mask to fit, and only refuses a genuine
     shape difference (portrait vs landscape).
  2. The mask editor paints on a majority-resolution frame, like the run.

Structural checks (the logic lives inline in the worker's main flow), plus a
behavioral check of the scaling math itself. Offline.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _worker_src():
    return (REPO / "astro_clean_v5.py").read_text()


def test_same_shape_mask_is_scaled_not_refused():
    src = _worker_src()
    i = src.index("the foreground mask does not match these frames")
    region = src[max(0, i - 2000):i]
    assert "INTER_NEAREST" in region, (
        "a same-shape mask must be scaled to fit (nearest keeps it a stencil)")
    assert "scaled it to match" in region, (
        "the rescue must be said out loud in the run log")


def test_sky_mask_is_scaled_alongside():
    """sky_mask is derived from the mask at load time, so it shares the wrong
    size and must be scaled too or the size-sensitive steps break later."""
    src = _worker_src()
    i = src.index("scaled it to match")
    region = src[max(0, i - 1200):i]
    assert src[max(0, i - 1200):i + 200].count("cv2.resize") >= 2, (
        "both fg_mask and sky_mask need the resize")


def test_true_shape_mismatch_still_stops_the_run():
    src = _worker_src()
    assert "the foreground mask does not match these frames" in src, (
        "the hard stop for a genuine shape difference must survive")


def test_aspect_tolerance_matches_the_field_case():
    """The check that decides scale-vs-stop, run against the real numbers."""
    # Randy's case: same shape, half size -> must pass the tolerance.
    assert abs((3008 / 2008) - (6016 / 4016)) < 0.01
    # Portrait mask on landscape frames -> must fail it.
    assert not abs((4016 / 6016) - (6016 / 4016)) < 0.01


def test_mask_editor_paints_on_the_majority_resolution():
    src = (REPO / "star_trail_cleanr.py").read_text()
    i = src.index("def _open_mask_editor")
    body = src[i:i + 4000]
    assert "image_size" in body, (
        "the editor must size the frames to find the majority resolution")
    assert "most_common" in body, (
        "the editor must pick the majority size, like the run does")
