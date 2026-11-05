"""Smoke tests for the ground-edge filter (ground_mark in modules/trail_grouper.py and
stage_ground_filter in modules/detect_pipeline.py).

The foreground mask is rough on purpose, so the model sometimes marks the fringe of
trees or a roof edge just outside it as a trail. Such a mark has dark ground on one
side along its whole length and no bright line inside it. These tests paint that
scene synthetically and check the filter drops only that, never a trail in open sky
and never a bright streak lying along a horizon. No model, no image files.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import detect_pipeline as dp  # noqa: E402
from modules.trail_grouper import ground_mark  # noqa: E402

H, W, HORIZON = 1200, 1600, 600


def _scene(streak_at=None):
    """Bright sky above row HORIZON, dark ground below; optional bright thin line at a row."""
    rng = np.random.default_rng(1)
    g = np.full((H, W), 150.0, np.float32)
    g[HORIZON:] = 30.0
    g += rng.normal(0, 4, (H, W)).astype(np.float32)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if streak_at is not None:
        cv2.line(g, (300, streak_at), (1000, streak_at), 235, 3)
    return g


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ALONG_HORIZON = _rect(300, HORIZON - 20, 1000, HORIZON + 20)   # straddles the tree line
IN_SKY = _rect(300, 200, 1000, 240)
IN_GROUND = _rect(300, 900, 1000, 940)


def test_a_mark_along_the_horizon_with_no_line_is_dropped():
    mark, info = ground_mark(_scene(), ALONG_HORIZON)
    assert mark, info
    assert info["ridge"] is not None and info["ridge"] < 5, info


def test_a_bright_streak_along_the_horizon_is_kept():
    mark, info = ground_mark(_scene(streak_at=HORIZON), ALONG_HORIZON)
    assert not mark, "a real low trail was dropped: %s" % info
    assert info["ridge"] > 20, info


def test_trails_in_open_sky_or_open_ground_are_never_touched():
    plain = _scene()
    streaky = _scene(streak_at=220)
    assert not ground_mark(plain, IN_SKY)[0]
    assert not ground_mark(streaky, IN_SKY)[0]
    assert not ground_mark(plain, IN_GROUND)[0], "dark on BOTH sides is not a ground edge"


def test_a_short_dash_at_the_horizon_is_kept_because_the_line_test_cannot_run():
    """A dash under 40 px long cannot be tested for a bright line, so it must not be dropped as ground
    (v3.0 dropped 1,979 such plane dashes on the Sean run)."""
    short = _rect(500, HORIZON - 6, 530, HORIZON + 6)
    mark, info = ground_mark(_scene(), short)
    assert not mark, "an untestable short mark was dropped: %s" % info


def test_a_polygon_too_close_to_the_frame_edge_to_judge_is_kept():
    mark, _ = ground_mark(_scene(), _rect(300, 0, 1000, 30))
    assert not mark


def test_the_stage_removes_the_polygon_its_segment_and_its_pixels_only():
    gray = _scene()
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    polys = [ALONG_HORIZON, IN_SKY]
    segs = []
    for p in polys:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 1, 2)], 255)
        segs.append(m)
    final = np.maximum(segs[0], segs[1])
    state = dp.PipelineState(image=img, polygons=list(polys), polygon_segs=list(segs),
                             final_mask=final.copy())
    flog = dp.FrameLog("t")
    with flog.stage("ground_filter") as s:
        state = dp.stage_ground_filter(state, dp.StageConfig(ground_filter=True), s)
    assert state.polygons == [IN_SKY] and len(state.polygon_segs) == 1
    assert state.final_mask[HORIZON, 600] == 0, "the dropped polygon's pixels are still painted"
    assert state.final_mask[220, 600] == 255, "the kept polygon's pixels were removed"
    rec = flog.stages[0]
    assert rec["counts"]["ground_marks_dropped"] == 1
    ev = [e for e in rec["events"] if e["reason"] == "ground_mark_dropped"]
    assert len(ev) == 1 and ev[0]["bbox"][0] == 300, "the drop must be logged with its position"


def test_the_filter_is_off_unless_asked_for():
    assert dp.StageConfig().ground_filter is False
