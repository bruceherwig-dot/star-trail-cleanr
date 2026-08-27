"""Per-component work must use the component's bounding box, not the whole frame.

This pattern has now cost real users twice:

  sky_dots._fill_specks (2026-08-09)   `out[lab == i] = fill` for each of ~2,000
      specks on a 30MP stack. Minutes of waiting.
  detect_pipeline.stage_prune_phantoms (2026-08-25)  `(lab == i) & (ph > 0)` for
      each of 183 components on a 44MP frame: ~16 BILLION element operations to
      inspect blobs a few hundred pixels across. It was 57% of the stage, and
      that stage was the single largest in detection -- 44% of it, more than the
      AI inference. Measured on a user's own frame: 2.62s -> 0.00s, output
      byte-identical.

connectedComponentsWithStats ALREADY returns each component's bounding box, and
the surrounding code is usually reading it for width/height anyway. Slicing to it
is free.

This test reads the source rather than timing anything: timing tests are flaky on
shared machines, and the shape of the code is what actually matters.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _function_body(text, name, code_only=True):
    """The named function's source. With code_only, comments are stripped:
    these tests search for a banned EXPRESSION, and the comment explaining why
    it is banned necessarily quotes it. Without this, documenting the bug fails
    the test that guards the fix."""
    i = text.index(f"def {name}")
    rest = text[i:]
    nxt = rest.find("\ndef ", 1)
    body = rest[:nxt] if nxt > 0 else rest
    if not code_only:
        return body
    return "\n".join(line.split("#")[0] for line in body.splitlines())


def test_phantom_pruning_slices_to_the_bounding_box():
    src = (REPO / "modules" / "detect_pipeline.py").read_text(encoding="utf-8")
    body = _function_body(src, "stage_prune_phantoms")
    assert "CC_STAT_LEFT" in body and "CC_STAT_TOP" in body, (
        "the per-component loop must slice to the component's bounding box; "
        "connectedComponentsWithStats already provides it")
    # The exact full-frame expression that was there before.
    assert not re.search(r"\(lab == i\)\s*&\s*\(ph > 0\)", body), (
        "full-frame per-component comparison is back: this was 57% of the "
        "largest stage in detection")


def test_the_speck_fill_still_slices_too():
    src = (REPO / "modules" / "sky_dots.py").read_text(encoding="utf-8")
    body = _function_body(src, "_fill_specks")
    assert "CC_STAT_LEFT" in body, \
        "the speck fill must keep working inside each blob's bounding box"
    assert not re.search(r"out\[lab == i\]", body), \
        "full-frame assignment per speck is back"


def test_the_cheap_test_comes_first():
    """A component too short to matter should be rejected before any pixel work.
    Ordering the tests cheapest-first is free and skips most components."""
    src = (REPO / "modules" / "detect_pipeline.py").read_text(encoding="utf-8")
    body = _function_body(src, "stage_prune_phantoms")
    i_extent = body.index("_PHANTOM_MIN_EXTENT")
    i_slice = body.index("CC_STAT_LEFT")
    assert i_extent < i_slice, (
        "the extent check must come before the pixel work, so short components "
        "cost nothing")
