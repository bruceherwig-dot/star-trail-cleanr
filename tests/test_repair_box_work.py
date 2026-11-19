"""Repair works inside each trail's box, and reports where all of its time went.

Measured 2026-11-16 on six 24 MP frames with 114 trail pieces: repair took 66
seconds, and 20 of them were ONE line, `np.where(sub_mask)`, scanning the whole
photograph once per trail segment to learn four numbers (the segment's box).
Splitting a trail into segments did the same kind of whole-frame work. And the
timing log itemised only about a third of repair, so nobody could see any of it.

These tests lock three things:
  * splitting a trail inside its box returns exactly the masks the whole-frame
    version returned (the old version is kept here as the reference);
  * repair_frame reports the new timers, including "compose_s", the part of
    each segment no named timer claims;
  * the engine prints repair's parts and states what is left over.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import repair as R                      # noqa: E402
from modules.repair import _split_component, repair_frame   # noqa: E402


def _old_split_component(comp_full):
    """The whole-frame version this replaced, word for word in its effect."""
    H, W = comp_full.shape
    seg_len_max = R.MAX_SEG_LENGTH * math.sqrt((H * W) / R._REF_FRAME_PX)
    _rr = np.flatnonzero(np.any(comp_full, axis=1))
    _cc = np.flatnonzero(np.any(comp_full, axis=0))
    if len(_rr) == 0:
        return []
    _r0, _c0 = int(_rr[0]), int(_cc[0])
    _sub = comp_full[_r0:int(_rr[-1]) + 1, _c0:int(_cc[-1]) + 1]
    ys, xs = np.where(_sub)
    pts = np.column_stack([xs + _c0, ys + _r0]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    trail_len = float(max(rect[1]))
    if trail_len <= seg_len_max:
        m = np.zeros(comp_full.shape, dtype=np.uint8)
        m[comp_full] = 255
        return [m]
    n_segs = math.ceil(trail_len / seg_len_max)
    box = cv2.boxPoints(rect)
    e01 = np.linalg.norm(box[1] - box[0])
    e12 = np.linalg.norm(box[2] - box[1])
    if e01 >= e12:
        a0, a1, b0, b1 = box[0], box[1], box[3], box[2]
    else:
        a0, a1, b0, b1 = box[1], box[2], box[0], box[3]
    result = []
    for si in range(n_segs):
        t0, t1 = si / n_segs, (si + 1) / n_segs
        corners = np.array([
            a0 + t0 * (a1 - a0), a0 + t1 * (a1 - a0),
            b0 + t1 * (b1 - b0), b0 + t0 * (b1 - b0),
        ], dtype=np.int32)
        seg = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(seg, [corners.reshape(-1, 1, 2)], 255)
        seg[~comp_full] = 0
        if seg.any():
            result.append(seg)
    return result if result else [np.uint8(comp_full) * 255]


def _trail(h, w, p0, p1, width):
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, p0, p1, 255, width)
    return m > 0


def _bbox(mask):
    rr = np.flatnonzero(mask.any(axis=1))
    cc = np.flatnonzero(mask.any(axis=0))
    return (int(cc[0]), int(rr[0]), int(cc[-1] - cc[0] + 1), int(rr[-1] - rr[0] + 1))


_CASES = [
    # (height, width, start, end, thickness): short, long, diagonal, edge-hugging
    (600, 900, (100, 100), (160, 130), 5),          # short: one segment
    (1200, 1800, (50, 60), (1700, 1100), 7),        # long diagonal: several segments
    (1200, 1800, (0, 5), (1799, 5), 9),             # along the top edge
    (1200, 1800, (1795, 0), (1795, 1199), 9),       # along the right edge
    (1500, 1500, (10, 1490), (1490, 10), 3),        # thin anti-diagonal, corner to corner
    (800, 2400, (0, 0), (2399, 799), 21),           # fat, touches two corners
]


def test_split_inside_the_box_matches_the_whole_frame_version():
    for h, w, p0, p1, width in _CASES:
        comp = _trail(h, w, p0, p1, width)
        old = _old_split_component(comp)
        for label, new in (("with the caller's box", _split_component(comp, bbox=_bbox(comp))),
                           ("working the box out itself", _split_component(comp))):
            assert len(new) == len(old), \
                f"{label}: {len(new)} segments, expected {len(old)} for trail {p0}->{p1}"
            for i, (a, b) in enumerate(zip(old, new)):
                assert a.shape == b.shape and a.dtype == b.dtype
                assert np.array_equal(a, b), \
                    f"{label}: segment {i} differs for trail {p0}->{p1} width {width}"


def test_split_matches_on_many_random_trails_including_ones_leaving_the_frame():
    """The first box-only version differed by a few pixels at a trail's ends,
    because OpenCV draws a polygon edge differently when the canvas cuts it.
    Random trails, many running off the frame, are what catch that."""
    rng = np.random.default_rng(3)
    compared = 0
    for _ in range(150):
        h = int(rng.integers(300, 1400))
        w = int(rng.integers(300, 2000))
        p0 = (int(rng.integers(-200, w + 200)), int(rng.integers(-200, h + 200)))
        p1 = (int(rng.integers(-200, w + 200)), int(rng.integers(-200, h + 200)))
        comp = _trail(h, w, p0, p1, int(rng.integers(1, 40)))
        if not comp.any():
            continue
        old = _old_split_component(comp)
        new = _split_component(comp, bbox=_bbox(comp))
        assert len(old) == len(new), f"segment count differs for {p0}->{p1} in {w}x{h}"
        for a, b in zip(old, new):
            assert np.array_equal(a, b), f"segment differs for {p0}->{p1} in {w}x{h}"
        compared += len(old)
    assert compared > 300, "the random trails stopped exercising the split"


def test_segments_together_cover_the_trail_exactly_once():
    comp = _trail(1200, 1800, (50, 60), (1700, 1100), 7)
    segs = _split_component(comp, bbox=_bbox(comp))
    assert len(segs) > 1, "this trail is long enough that it must be split"
    union = np.zeros(comp.shape, bool)
    for s in segs:
        union |= s > 0
    assert not (union & ~comp).any(), "a segment marked pixels outside the trail"


def _frames_with_trail():
    rng = np.random.default_rng(7)
    h, w = 700, 1000
    base = rng.integers(18, 30, (h, w, 3), dtype=np.uint8)
    frames = []
    for k in range(3):
        f = base.copy()
        for (y, x) in rng.integers(40, 660, (160, 2)):
            cv2.circle(f, (int(x) + 2 * k, int(y)), 1, (220, 220, 220), -1)
        frames.append(f)
    mask = np.zeros((h, w), np.uint8)
    cv2.line(mask, (120, 150), (880, 520), 255, 9)
    cv2.line(frames[1], (120, 150), (880, 520), (240, 240, 250), 3)
    return frames, mask


def test_repair_reports_every_part_of_its_time():
    frames, mask = _frames_with_trail()
    dbg = {}
    out = repair_frame(frames[1], mask, 1, frames, neighbor_masks=[None, mask, None],
                       polygon_segs=[mask], debug_out=dbg)
    assert out.shape == frames[1].shape
    timing = dbg.get("timing") or {}
    for key in ("box_s", "colour_s", "compose_s", "darken_s", "split_s", "paste_s"):
        assert key in timing, f"repair no longer reports '{key}'; its time would be invisible"
    assert all(v >= 0 for v in timing.values())


def test_repair_no_longer_scans_the_whole_frame_per_segment():
    src = (REPO / "modules" / "repair.py").read_text(encoding="utf-8")
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "np.where(sub_mask)" not in code, (
        "repair is scanning the whole photograph for each trail segment again; "
        "that one line was 31% of repair time. Look inside the component's box.")
    assert "seg[~comp_full]" not in code and "m[comp_full] = 255" not in code, (
        "_split_component is doing whole-frame work per trail again")


def test_engine_itemises_repair_and_states_the_leftover():
    engine = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert '_tacc("rp_" + _rk' in engine, \
        "the engine must pass repair's own timers into the timing summary"
    assert "(repair unaccounted)" in engine, (
        "the timing summary must state the gap between repair and its parts; "
        "a silent gap hid more than half of repair's time until 2026-11-16")
