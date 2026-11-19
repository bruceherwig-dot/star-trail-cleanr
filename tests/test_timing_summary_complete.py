"""The timing summary must show every stage that ran, not a hand-written subset.

Field case, 2026-08-25. Kari Tuomi reported runs taking 84 seconds a frame on an
RTX 3080 Ti. His Star Log's timing table showed detect at 50.6s/frame with only
21.6s of that itemised, so roughly 29 seconds a frame belonged to nothing. Two
rounds of emails went into theorising about RAW decoding (measured afterwards at
2.2s/frame, under 3% of the run) while the real answer sat inside detect,
unnamed.

CAUSE: the summary's rows are a hand-written list. `prune_phantoms` was added to
the pipeline and enabled in the real run, but no row was ever added for it, so
its time vanished from every report while still counting inside its parent --
which is exactly what makes a parent look mysteriously slow. Measured on one of
his own frames it is the LARGEST stage in detection: 6.55s of 15.0s, 44%, more
than the AI inference itself.

THE FIX IS STRUCTURAL, not "add the missing row": anything timed that has no row
is printed anyway, and the summary states any remaining gap between detect and
its parts. A future stage cannot go missing the same way.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

ENGINE = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")


def test_every_timed_stage_is_printed_even_without_a_row():
    i = ENGINE.find("_STEP_ORDER = [")
    assert i > 0, "the timing table is gone"
    body = ENGINE[i:i + 4000]
    assert "_listed" in body, (
        "the summary must track which keys it printed and print the rest; "
        "otherwise a stage with no row is silently dropped from every report")
    assert "if _key in _listed:" in body, \
        "the catch-all loop over unlisted timings is missing"


def test_the_summary_reports_any_unaccounted_detect_time():
    i = ENGINE.find("_STEP_ORDER = [")
    body = ENGINE[i:i + 4000]
    assert "unaccounted" in body, (
        "the summary must state the gap between detect and the sum of its "
        "stages. A silent gap is what hid prune_phantoms for months.")


def test_prune_phantoms_really_is_enabled_in_the_real_run():
    """If this ever goes False, the 44% finding above no longer applies and the
    performance notes need revisiting rather than being trusted."""
    i = ENGINE.find("new_cfg = dp.StageConfig(")
    assert i > 0, "the engine's pipeline config is gone"
    cfg = ENGINE[i:i + 800]
    assert re.search(r"prune_phantoms\s*=\s*True", cfg), (
        "prune_phantoms is no longer enabled in the real run; the recorded "
        "timing breakdown was measured with it ON")


def test_the_pipeline_stage_list_and_the_summary_cannot_drift_silently():
    """Every stage detect_frame can run should either have a row or be caught by
    the catch-all. This checks the catch-all exists for the ones without rows."""
    pipeline = (REPO / "modules" / "detect_pipeline.py").read_text(encoding="utf-8")
    stages = set(re.findall(r'with flog\.stage\("([a-z_]+)"\)', pipeline))
    assert stages, "could not find the pipeline's stages"
    rows = set(re.findall(r'\("dp_([a-z_]+)_s"', ENGINE))
    missing = stages - rows
    # Missing rows are ALLOWED now, precisely because the catch-all prints them.
    # What is not allowed is missing rows AND no catch-all.
    if missing:
        i = ENGINE.find("_STEP_ORDER = [")
        assert "_listed" in ENGINE[i:i + 4000], (
            f"these stages have no row and there is no catch-all to print "
            f"them: {sorted(missing)}")
