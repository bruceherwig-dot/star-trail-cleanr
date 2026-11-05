"""The run log must let a stranger read what happened, where, why, and with what result.

Every keep-or-toss event the detection stages write must carry a position, and every event name must
have a plain-English line in the run log's legend. This test reads the source of detect_pipeline.py
and fails when someone adds an event that leaves out the position or the legend entry, so the log
cannot quietly slide back to counts that say "something happened somewhere".
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.run_logger import LOG_LEGEND  # noqa: E402

SOURCE = (REPO / "modules" / "detect_pipeline.py").read_text(encoding="utf-8")

# Bookkeeping events that describe a stage, not a piece, so they have no single position.
NO_POSITION_NEEDED = {"load_failed", "phantoms_pruned", "substep_timing", "fit_timing", "stub",
                      "regroup", "skip", "ground_filter_skipped"}
# Events built from a dict at the call site (the position is inside the dict, checked by the log reader).
BUILT_FROM_DICT = {"phantom_removed", "bridge_gap_miss"}
# Names that are written through a helper from a list the large-square fix returns.
VIA_HELPER = {"fat_thinned", "fat_dropped_covered", "fat_dropped_redundant"}


def _calls():
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("event", "decision") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                yield node, first.value


def test_every_piece_event_carries_a_position():
    bad = []
    for node, name in _calls():
        if name in NO_POSITION_NEEDED or name in BUILT_FROM_DICT:
            continue
        kws = {k.arg for k in node.keywords}
        if node.func.attr == "decision":
            if len(node.args) < 5:
                bad.append("%s: decision() needs rule, action, bbox, why, result" % name)
            continue
        if "bbox" not in kws and not ({"cx", "cy"} <= kws):
            bad.append("%s (line %d): no bbox or cx/cy" % (name, node.lineno))
    assert not bad, "events without a position: " + "; ".join(bad)


def test_every_event_name_is_explained_in_the_legend():
    explained = set(LOG_LEGEND["detect_stage_event_reasons"])
    names = {n for _, n in _calls()} | VIA_HELPER | BUILT_FROM_DICT
    missing = sorted(n for n in names if n not in explained and n not in {"substep_timing", "fit_timing", "stub"})
    assert not missing, "event names with no legend line in modules/run_logger.py: %s" % missing


def test_the_legend_states_the_four_part_standard():
    text = LOG_LEGEND["decision_event_fields"]
    for word in ("reason", "action", "bbox", "why", "result"):
        assert word in text, word


# ---- the events actually written, not only the source text ----------------------------------------

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules import detect_pipeline as dp  # noqa: E402
from modules.trail_grouper import thin_fat_detections  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402


def test_a_drop_inside_a_crop_is_logged_at_its_full_frame_position():
    crop = np.zeros((200, 300), np.uint8)
    cv2.line(crop, (20, 100), (28, 100), 255, 3)             # a tiny star-sized dash
    cfg = dp.StageConfig(star_len_px=40.0, star_width_px=10.0, min_aspect=1.0)
    flog = dp.FrameLog(frame_name="t")
    with flog.stage("fit_polygons") as s:
        assert dp._props_with_log(crop, cfg, s, offset=(1000, 4000)) is None
    ev = [e for e in flog.stages[0]["events"] if e["reason"] == "drop"]
    assert len(ev) == 1, ev
    x0, y0, x1, y1 = ev[0]["bbox"]
    assert 4000 <= x0 <= x1 <= 4300 and 1000 <= y0 <= y1 <= 1200, ev[0]["bbox"]
    for part in ("action", "why", "result", "gate"):
        assert ev[0][part], part
    assert "cutoff" in ev[0]["why"]


def test_the_large_square_fix_says_which_piece_it_thinned_and_where():
    H, W = 2000, 3000
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (1000, 1000), (1700, 1400), 255, 28)
    cv2.line(m, (1000, 1400), (1700, 1000), 255, 28)
    ys, xs = np.nonzero(m)
    rp = max(regionprops(label(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])), key=lambda r: r.area)
    _, vecs = np.linalg.eigh(rp.inertia_tensor)
    d = {"centroid": np.array([rp.centroid[0] + ys.min(), rp.centroid[1] + xs.min()]), "u": vecs[:, 0],
         "minor": rp.axis_minor_length, "major": rp.axis_major_length, "area": rp.area,
         "coords": np.column_stack((ys, xs)), "conf": 0.9}
    events = []
    out, stats = thin_fat_detections([d], H, W, events=events)
    assert stats["thinned"] == 1 and len(events) == 1, (stats, events)
    e = events[0]
    assert e["rule"] == "fat_thinned" and e["action"] == "thinned"
    x0, y0, x1, y1 = e["bbox"]
    assert x0 <= 1050 and x1 >= 1650 and y0 <= 1050 and y1 >= 1350, e["bbox"]
    assert "band" in e["result"] and "fat" in e["why"]
