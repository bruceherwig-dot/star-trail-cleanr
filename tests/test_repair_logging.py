"""Repair logging must actually record the per-trail fill method and cascade tier.

Locks the 2026-07-02 fix. The normal repair path (per-polygon) used to drop ALL
per-segment detail and log only timing, so a run log couldn't say which fill method
or which star-tracking tier fired for a trail -- defeating the purpose of the log
(you had to re-run to answer "did the black fill ever fire?"). This asserts the
polygon path threads the debug capture through, so every repaired trail records its
method + cascade, and that the log legend explains those fields in plain English.
"""
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _synthetic_frames():
    """Three frames with a few drifting 'stars' and a bright 'trail' in the middle
    frame, enough to exercise tracking + fill and produce a logged segment."""
    H = W = 200

    def frame(shift):
        f = np.full((H, W, 3), 40, np.uint8)
        for (y, x) in [(30, 30), (60, 120), (150, 80), (100, 160)]:
            f[y + shift:y + shift + 3, x:x + 3] = 220
        return f

    f0, f1, f2 = frame(0), frame(3), frame(6)
    mask = np.zeros((H, W), np.uint8)
    mask[95:105, 40:140] = 255          # a 100x10 trail (> MIN_AREA)
    f1 = f1.copy()
    f1[95:105, 40:140] = (200, 180, 160)
    return [f0, f1, f2], mask


def test_polygon_repair_logs_method_and_cascade():
    from modules.repair import repair_frame
    frames, mask = _synthetic_frames()
    dbg = {}
    repair_frame(frames[1], mask, 1, frames, neighbor_masks=[None, None, None],
                 polygon_segs=[mask], debug_out=dbg)
    comps = dbg.get("components")
    assert comps, "polygon-path repair must populate debug_out['components'], not only timing"
    segs = comps[0].get("segments")
    assert segs, "each repaired component must record its segments"
    s = segs[0]
    assert s.get("method"), "each segment must record HOW it was filled (method)"
    assert s.get("cascade"), "each segment must record the star-tracking cascade tier"


def test_edge_frame_protects_static_foreground():
    """First/last frame (one neighbor) must NOT slide an unmasked static object.

    Locks the 2026-07-02 fix: on an edge frame the two-neighbor still-vs-moving
    routing can't run, so a static trunk/rock used to get nicked by the single-
    neighbor slide (a light notch that survives the stack). The fix reaches to the
    second same-side neighbor and keeps static pixels unshifted.
    """
    from modules.repair import repair_frame
    H = W = 200

    def frame(shift):
        f = np.full((H, W, 3), 40, np.uint8)
        for (y, x) in [(30, 30), (60, 150), (150, 40)]:
            f[y + shift:y + shift + 3, x:x + 3] = 220
        f[:, 150:158] = 8                       # a STATIC dark bar (a "trunk")
        return f

    f0, f1, f2 = frame(0), frame(4), frame(8)
    mask = np.zeros((H, W), np.uint8)
    mask[95:105, 60:160] = 255                  # trail runs up to the dark bar
    f0 = f0.copy()
    f0[95:105, 60:160] = (200, 180, 160)
    dbg = {}
    out = repair_frame(f0, mask, 0, [f0, f1, f2], neighbor_masks=[None, None, None],
                       polygon_segs=[mask], debug_out=dbg)
    seg = dbg["components"][0]["segments"][0]
    assert seg.get("edge_still_px", 0) > 0, "edge-frame protection must fire on the first frame"
    # the static dark bar under the trail must stay dark, not be slid/lightened
    assert int(out[95:105, 150:158].max()) < 60, "static foreground was nicked on the edge frame"


def test_legend_documents_repair_fields():
    from modules.run_logger import LOG_LEGEND
    rk = LOG_LEGEND.get("key_repair_fields", {})
    for field in ("method", "cascade", "tracking_ok", "sky_filled_px"):
        assert field in rk, f"the log legend must explain the repair field '{field}' in plain English"
    # The method explanation must name the crayon and black fills so a reader knows them.
    assert "crayon" in rk["method"].lower() and "black" in rk["method"].lower()


def test_a_short_dash_polygon_under_500px_is_repaired():
    """The detector's polygons are repaired whatever their size. A 37 x 7 px plane dash (about 260 px) used to
    be skipped by the 500 px floor, so detected trails were left in the picture (Sean run, 2026-11-04)."""
    from modules.repair import repair_frame
    frames, _ = _synthetic_frames()
    dash = np.zeros((200, 200), np.uint8)
    dash[97:104, 60:97] = 255
    assert 0 < int(dash.sum() // 255) < 500
    f1 = frames[1].copy()
    f1[97:104, 60:97] = (240, 240, 240)             # the dash itself
    dbg = {}
    out = repair_frame(f1, dash, 1, [frames[0], f1, frames[2]], neighbor_masks=[None, None, None],
                       polygon_segs=[dash], debug_out=dbg)
    assert dbg["components"], "a short dash polygon must produce a repaired component"
    assert out[97:104, 60:97].max() < 200, "the dash must be gone after repair"


def test_the_connected_components_path_still_skips_tiny_noise():
    from modules.repair import repair_frame
    frames, _ = _synthetic_frames()
    speck = np.zeros((200, 200), np.uint8)
    speck[100:104, 100:110] = 255                    # 40 px, no polygon list given
    dbg = {}
    repair_frame(frames[1], speck, 1, frames, neighbor_masks=[None, None, None], debug_out=dbg)
    assert not dbg.get("components"), "tiny specks on the old path stay skipped"
