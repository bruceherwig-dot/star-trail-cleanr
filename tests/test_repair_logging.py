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


def _plane_moves_with_the_stars():
    """Three frames. The stars drift (-12, +20) px per frame, and a short plane dash drifts with them, so after
    the repair's slide each neighbour's dash lands exactly on this frame's dash. The dash is far smaller than
    the drift, so at the SAME position the neighbours show no dash there."""
    H = W = 260
    sx, sy = -12, 20
    stars = [(40, 150), (200, 60), (90, 40), (180, 200), (60, 90), (220, 150), (130, 120), (30, 40), (160, 30)]

    def frame(k):                                      # k = -1, 0, +1 relative to the middle frame
        f = np.full((H, W, 3), 40, np.uint8)
        for (y, x) in stars:
            yy, xx = y + k * sy, x + k * sx
            if 2 <= yy < H - 4 and 2 <= xx < W - 4:
                f[yy:yy + 3, xx:xx + 3] = 230
        # the plane dash: 12 x 6 px, centred at (110, 110) in the middle frame, moving with the stars
        cy, cx = 110 + k * sy, 110 + k * sx
        f[cy - 3:cy + 3, cx - 6:cx + 6] = (240, 240, 240)
        return f

    frames = [frame(-1), frame(0), frame(1)]
    masks = []
    for k in (-1, 0, 1):
        m = np.zeros((H, W), np.uint8)
        cy, cx = 110 + k * sy, 110 + k * sx
        m[cy - 5:cy + 5, cx - 8:cx + 8] = 255          # each frame's own detection, a little bigger than the dash
        masks.append(m)
    return frames, masks


def _dash_left_after_repair():
    from modules.repair import repair_frame
    frames, masks = _plane_moves_with_the_stars()
    dbg = {}
    out = repair_frame(frames[1], masks[1], 1, frames, neighbor_masks=masks,
                       polygon_segs=[masks[1]], debug_out=dbg)
    assert dbg["components"] and dbg["components"][0]["segments"][0].get("tracking_ok"), \
        "the test needs the star tracking to succeed"
    return int(out[105:115, 104:116].max())


def test_a_trail_that_moves_with_the_stars_is_not_pasted_back():
    assert _dash_left_after_repair() < 150


def test_without_sliding_the_neighbour_masks_the_dash_comes_back():
    """Documents the old behaviour so the test above is known to be sensitive to the fix."""
    import modules.repair as repair
    real = repair._shift_mask
    repair._shift_mask = lambda m, dx, dy: m > 0
    try:
        assert _dash_left_after_repair() >= 150
    finally:
        repair._shift_mask = real


def test_a_wrongly_outlined_static_object_is_still_pasted_back_unshifted():
    """A bush the model outlined sits at the SAME place in every frame. The old unshifted flag caught it and the
    repair pasted the real bush back. Sliding the flag away with the stars lost that; the shifted flag is added
    to the unshifted one, so the bush stays flagged and comes back intact."""
    from modules.repair import repair_frame
    H = W = 260
    sx, sy = -12, 20
    stars = [(40, 150), (200, 60), (90, 40), (180, 200), (60, 90), (220, 150), (130, 120), (30, 40), (160, 30)]

    def frame(k):
        f = np.full((H, W, 3), 40, np.uint8)
        for (y, x) in stars:
            yy, xx = y + k * sy, x + k * sx
            if 2 <= yy < H - 4 and 2 <= xx < W - 4:
                f[yy:yy + 3, xx:xx + 3] = 230
        f[100:120, 100:130] = (20, 60, 20)             # a dark bush: it does NOT move between frames
        return f

    frames = [frame(-1), frame(0), frame(1)]
    m = np.zeros((H, W), np.uint8)
    m[100:120, 100:130] = 255                           # the same wrong outline in every frame
    out = repair_frame(frames[1], m, 1, frames, neighbor_masks=[m, m, m], polygon_segs=[m], debug_out={})
    # the bush must come back as the bush (dark green), not as slid sky (40)
    assert abs(int(out[110, 115, 1]) - 60) <= 12, "the static bush was replaced by slid sky"
