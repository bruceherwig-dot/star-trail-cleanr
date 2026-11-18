"""The step timer (modules/step_timer.py): what it records, how it adds up, and that the
stages promised in the timing table really record their steps.

Bruce's rule (2026-11-18): measurements stay in the program. A step that can take real time
records when it ran, what it did and how long it took, and the end-of-batch table shows how
much of each stage is still unexplained.
"""
import re
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import step_timer as st   # noqa: E402


def setup_function(_f=None):
    st.reset()


def test_a_step_records_calls_seconds_and_what_it_did():
    st.reset()
    with st.step("demo.work") as s:
        time.sleep(0.02)
        s.count("did_thing", 3)
    st.add("demo.work", 0.5)
    st.count("demo.work", "did_thing")
    snap = st.snapshot()
    calls, secs = snap["steps"]["demo.work"]
    assert calls == 2 and 0.5 + 0.015 < secs < 0.8
    assert snap["counts"]["demo.work"] == {"did_thing": 4}


def test_reset_clears_everything_and_snapshot_is_a_copy():
    st.reset()
    st.add("a.b", 1.0)
    snap = st.snapshot()
    st.reset()
    assert st.snapshot() == {"steps": {}, "counts": {}}
    assert snap["steps"]["a.b"] == [1, 1.0]


def test_unaccounted_counts_only_two_part_names_of_that_group():
    st.reset()
    st.add("fit.one", 3.0)
    st.add("fit.two", 2.0)
    st.add("fit.two.inner", 1.5)        # nested detail: not counted twice
    st.add("other.thing", 9.0)          # a different group
    gap, pct = st.unaccounted(10.0, st.snapshot(), "fit")
    assert abs(gap - 5.0) < 1e-9 and abs(pct - 50.0) < 1e-9
    assert st.unaccounted(10.0, st.snapshot(), "nothing_recorded") is None
    assert st.unaccounted(0.0, st.snapshot(), "fit") is None


def test_several_threads_can_record_at_once_without_losing_any():
    st.reset()

    def work():
        for _ in range(500):
            st.add("thr.x", 0.001)
            st.count("thr.x", "k")
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    snap = st.snapshot()
    assert snap["steps"]["thr.x"][0] == 4000
    assert snap["counts"]["thr.x"]["k"] == 4000


def _src(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def test_every_stage_the_table_checks_really_records_its_steps():
    """The engine prints an 'unaccounted' line for these stages. A stage that stopped
    recording would silently print nothing, so each must still have its steps in the code."""
    engine = _src("astro_clean_v5.py")
    groups = re.findall(r'\("(\w+)",\s*"(\w+_s)",\s*"[^"]+"\)', engine)
    names = {g for g, _k in groups}
    assert {"phantom", "fit", "hotpix"} <= names
    code = "\n".join(_src(f) for f in ("modules/detect_pipeline.py", "modules/hot_pixels.py",
                                       "astro_clean_v5.py"))
    for g in names:
        assert re.search(r'(_st|step_timer|_step_timer)\.(add|step|count)\("%s\.' % g, code), \
            f"no step records itself under '{g}.'"


def test_the_important_blind_spots_are_recorded_by_name():
    detect = _src("modules/detect_pipeline.py")
    for n in ("phantom.mask_build", "phantom.mask_merge", "phantom.sky_estimate",
              "phantom.thin_parts", "phantom.bridge_and_label", "phantom.judge_lines",
              "phantom.trim_detections", "fit.mask_build", "fit.edge_scan", "fit.peel_split",
              "fit.split_crossing", "fit.crossing_evidence", "fit.parallel_split", "fit.props",
              "fit.thin_fat", "fit.group", "fit.poly_fit", "fit.finish"):
        assert f'"{n}"' in detect, n
    repair = _src("modules/repair.py")
    assert "mask_checks_s" in repair and "repair.fill_method" in repair and "repair.grain_fill" in repair
    engine = _src("astro_clean_v5.py")
    for n in ("hotpix.map_merge_and_save", "hotpix.ground_check", "hotpix.repaint"):
        assert f'"{n}"' in engine, n
    assert '"hotpix.search"' in _src("modules/hot_pixels.py")


def test_the_run_log_and_the_app_report_carry_the_new_fields():
    engine = _src("astro_clean_v5.py")
    assert '"step_counts"' in engine and '"timing_gaps_percent"' in engine
    assert "BATCH_STEP_SECONDS:" in engine
    gui = _src("star_trail_cleanr.py")
    assert "BATCH_STEP_SECONDS:" in gui and '"stage_seconds"' in gui
    assert '"cpu_cores"' in gui and '"ram_gb"' in gui
