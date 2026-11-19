"""The run timing block for the Star Log (modules/run_timing_report.py)."""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import run_timing_report as rtr   # noqa: E402
from modules import step_guide                  # noqa: E402


def _batch(scale=1.0):
    """One batch's detail, shaped like the engine's BATCH_STEP_DETAIL line."""
    t = {"frame_read_s": [1, 2.7 * scale], "hot_pixel_s": [1, 1.3 * scale], "model_load_s": [1, 0.8 * scale],
         "star_size_s": [1, 1.8 * scale], "dp_tiled_inference_s": [7, 12.2 * scale],
         "dp_prune_phantoms_s": [7, 1.7 * scale], "dp_fit_polygons_s": [7, 10.4 * scale],
         "static_fp_s": [1, 2.1 * scale], "repair_s": [6, 13.5 * scale], "write_s": [6, 1.8 * scale],
         "st_fit.peel_split_s": [201, 3.1 * scale], "st_fit.group_s": [7, 1.9 * scale],
         "st_fit.split_crossing_s": [194, 1.4 * scale], "st_phantom.thin_parts_s": [7, 0.4 * scale],
         "st_hotpix.search_s": [1, 0.35 * scale], "st_write.encode_s": [6, 0.7 * scale]}
    c = {"fit.peel_split": {"blobs_tried": 201, "peeled_into_pieces": 28},
         "repair.fill_method": {"single_shift": 168, "next_only": 21},
         "fit.edge_scan": {"searched_tile_box": 201}}
    return {"batch_seconds": 60.0 * scale, "timing": t, "counts": c}


def test_batches_add_up():
    acc = rtr.new_accumulator()
    rtr.merge(acc, _batch())
    rtr.merge(acc, _batch(2.0))
    assert acc["batches"] == 2
    assert abs(acc["batch_seconds"] - 180.0) < 1e-9
    assert acc["timing"]["repair_s"][0] == 12 and abs(acc["timing"]["repair_s"][1] - 40.5) < 1e-9
    assert acc["counts"]["fit.peel_split"]["blobs_tried"] == 402


def test_the_block_says_where_time_went_in_plain_english():
    acc = rtr.new_accumulator()
    for _ in range(3):
        rtr.merge(acc, _batch())
    lines = rtr.render(acc, run_seconds=300, frames=18,
                       machine=rtr.machine_summary("mps", 10, 32.0))
    text = "\n".join(lines)
    assert lines[0].startswith("Run timing")
    assert "10 cores, 32 GB memory, Apple graphics chip" in text
    assert "Repairing frames" in text and "AI finding trails" in text
    assert step_guide.STAGES[4][2] in text, "each stage row carries its purpose sentence"
    assert "Cutting crossings into one rectangle per trail" in text
    assert "blobs examined" in text and "rectangles made" in text      # counts in plain words
    assert "single shift" in text                                       # how trails were filled in
    assert len(lines) <= 90


def test_percentages_use_the_batch_seconds_and_the_remainder_is_shown():
    acc = rtr.new_accumulator()
    rtr.merge(acc, _batch())
    text = "\n".join(rtr.render(acc))
    assert "Everything else" in text
    m = re.search(r"Repairing frames\s+\S+(?: \S+)?\s+(\d+)%", text)
    assert m and 20 <= int(m.group(1)) <= 24, text        # 13.5 of 60 s is about 22%


def test_nothing_recorded_gives_no_block():
    assert rtr.render(rtr.new_accumulator()) == []
    assert rtr.render(None) == []


def test_a_broken_batch_line_never_raises():
    acc = rtr.new_accumulator()
    rtr.merge(acc, {"timing": "garbage"})
    rtr.merge(acc, None)
    assert isinstance(acc, dict)


def test_no_file_names_or_paths_in_the_block():
    acc = rtr.new_accumulator()
    rtr.merge(acc, _batch())
    text = "\n".join(rtr.render(acc, run_seconds=60, frames=6, machine=rtr.machine_summary("cuda", 8, 16)))
    assert not re.search(r"[/\\][A-Za-z0-9_]", text), "no path-like text"
    assert not re.search(r"\.(jpg|tif|tiff|cr2|nef|png)\b", text, re.I)


def test_the_block_sits_above_the_cut_down_log_and_the_app_reads_the_line():
    gui = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    block = gui.index("The run timing block: where the WHOLE run's time went")
    scroll = gui.index("Star Log (what scrolled in the run window)")
    assert block < scroll
    assert 'startswith("BATCH_STEP_DETAIL:")' in gui
    eng = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert '"BATCH_STEP_DETAIL: "' in eng and '"step_guide"' in eng
