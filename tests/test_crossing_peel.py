"""Smoke tests for the crossing peel (modules/crossing_peel.py).

Where trails cross, the detector outlines the tangle as ONE blob. The peel cuts it into one
clean rectangle per trail (K, L, V, T, three or more trails, parallel trails) and must leave
alone a gentle bend, a near-straight fork and a short ragged chunk. These tests draw those
shapes synthetically (no model, no image files) and check the number of pieces, that every
trail is covered by a piece that is mostly that one trail, and that the rectangles are thin.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.crossing_peel import peel_split  # noqa: E402

N = 2400            # canvas side
W = 45              # trail width
C = N // 2
FRAME_PX = 24_000_000


def _line(cx, cy, ang, back, fwd):
    a = np.radians(ang)
    dx, dy = np.cos(a), np.sin(a)
    return (cx - dx * back, cy - dy * back, cx + dx * fwd, cy + dy * fwd)


def _draw(lines):
    """Blob (uint8 0/255, cropped to its bounding box) and one 0/255 mask per line, same crop."""
    full = np.zeros((N, N), np.uint8)
    each = []
    for x1, y1, x2, y2 in lines:
        t = np.zeros((N, N), np.uint8)
        cv2.line(t, (int(x1), int(y1)), (int(x2), int(y2)), 255, W)
        each.append(t)
        full |= t
    ys, xs = np.where(full > 0)
    sl = (slice(ys.min() - 5, ys.max() + 6), slice(xs.min() - 5, xs.max() + 6))
    return full[sl].copy(), [t[sl].copy() for t in each]


def _line_is_covered(line_mask, others, pieces):
    """True when the pieces that are mostly this one line cover at least 80% of its core
    (the line minus the zone within one width of every other line, i.e. minus the junctions)."""
    near_other = np.zeros(line_mask.shape, bool)
    for o in others:
        near_other |= cv2.dilate(o, np.ones((W, W), np.uint8)) > 0
    core = (line_mask > 0) & ~near_other
    if core.sum() == 0:
        return True
    on_line = cv2.dilate(line_mask, np.ones((9, 9), np.uint8)) > 0
    got = np.zeros(line_mask.shape, bool)
    for p in pieces:
        pb = p > 0
        if pb.sum() and (pb & on_line).sum() / pb.sum() >= 0.7:   # a piece that is mostly this line
            got |= pb
    return (got & core).sum() / core.sum() >= 0.8


def _check_split(lines):
    blob, each = _draw(lines)
    pieces, info = peel_split(blob, frame_px=FRAME_PX)
    assert info["reason"] == "split", info
    assert len(pieces) == len(lines), "expected %d pieces, got %d" % (len(lines), len(pieces))
    for i, t in enumerate(each):
        assert _line_is_covered(t, [o for j, o in enumerate(each) if j != i], pieces), "line %d not covered" % i
    # rectangles stay thin: no piece wider than 1.6 trail widths
    for p in pieces:
        ys, xs = np.where(p > 0)
        short = min(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2))[1])
        assert short <= 1.6 * W, "piece is %.0f px wide for a %d px trail" % (short, W)


def test_x_crossing_two_pieces():
    _check_split([_line(C, C, 0, 600, 600), _line(C, C, 40, 600, 600)])


def test_low_angle_x_kept_by_crossing_evidence():
    _check_split([_line(C, C, 0, 600, 600), _line(C, C, 25, 600, 600)])


def test_k_shape_three_pieces():
    _check_split([_line(C, C, 90, 600, 600), _line(C, C, 40, 0, 600), _line(C, C, -40, 0, 600)])


def test_v_and_l_shapes():
    _check_split([_line(C - 300, C, 60, 0, 700), _line(C - 300, C, 120, 0, 700)])
    _check_split([_line(C - 300, C - 300, 0, 0, 600), _line(C - 300, C - 300, 90, 0, 600)])


def test_three_four_and_five_trails_through_one_point():
    for angles in ([0, 60, 120], [0, 45, 90, 135], [0, 36, 72, 108, 144]):
        _check_split([_line(C, C, a, 600, 600) for a in angles])


def test_parallel_trails_stay_separate_pieces():
    # a 2x2 grid and an X with a parallel trail: the older band rescue merged parallel trails into one band
    _check_split([_line(C, C - 200, 0, 600, 600), _line(C, C + 200, 0, 600, 600),
                  _line(C - 200, C, 90, 600, 600), _line(C + 200, C, 90, 600, 600)])
    _check_split([_line(C, C, 0, 600, 600), _line(C, C, 90, 600, 600), _line(C, C + 90, 0, 600, 600)])


def test_a_gentle_bend_stays_whole():
    # one trail that turns 25 degrees: a bend, not a crossing
    full = np.zeros((N, N), np.uint8)
    pts = np.array([[300, 900], [1100, 1100], [1100 + int(800 * np.cos(np.radians(11 + 25))),
                                                 1100 + int(800 * np.sin(np.radians(11 + 25)))]], np.int32)
    cv2.polylines(full, [pts.reshape(-1, 1, 2)], False, 255, W)
    ys, xs = np.where(full > 0)
    blob = full[ys.min() - 5:ys.max() + 6, xs.min() - 5:xs.max() + 6]
    pieces, info = peel_split(blob, frame_px=FRAME_PX)
    assert len(pieces) == 1 and pieces[0] is blob, info


def test_a_single_straight_trail_and_a_small_blob_stay_whole():
    blob, _ = _draw([_line(C, C, 17, 700, 700)])
    pieces, info = peel_split(blob, frame_px=FRAME_PX)
    assert len(pieces) == 1, info
    tiny = np.zeros((120, 120), np.uint8)
    tiny[40:80, 20:100] = 255
    pieces, info = peel_split(tiny, frame_px=FRAME_PX)
    assert len(pieces) == 1 and info["reason"] == "too_small", info


def test_a_short_ragged_chunk_beside_a_trail_does_not_count_as_a_branch():
    # the forest case (Spain DSC06712): a long trail with a short chunk at its foot pointing another way
    full = np.zeros((N, N), np.uint8)
    cv2.line(full, (900, 300), (1000, 1500), 255, W)           # the trail
    cv2.line(full, (1000, 1500), (1100, 1560), 255, W)         # a stub only about 2 trail widths long
    ys, xs = np.where(full > 0)
    blob = full[ys.min() - 5:ys.max() + 6, xs.min() - 5:xs.max() + 6]
    pieces, info = peel_split(blob, frame_px=FRAME_PX)
    assert len(pieces) == 1, info


def test_a_trail_broken_at_the_crossing_comes_back_as_one_trail():
    # the frame 123 pattern: the faint trail's outline is pinched at the crossing, so its two halves
    # are separate pieces; they must be joined back into ONE piece or the seam bridge fires on the end
    full = np.zeros((N, N), np.uint8)
    cv2.line(full, (300, 1300), (2100, 1100), 255, W)                      # the bright trail, whole
    cv2.line(full, (300, 800), (1050, 1020), 255, 20)                      # faint trail, left half
    cv2.line(full, (1350, 1100), (2100, 1320), 255, 20)                    # faint trail, right half
    ys, xs = np.where(full > 0)
    blob = full[ys.min() - 5:ys.max() + 6, xs.min() - 5:xs.max() + 6]
    pieces, info = peel_split(blob, frame_px=FRAME_PX)
    long_pieces = [p for p in pieces if _long(p) > 1200]
    assert info["reason"] == "split" and len(long_pieces) >= 1, info


def _long(p):
    ys, xs = np.where(p > 0)
    return max(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2))[1])


def test_peel_is_off_by_default_in_the_pipeline_config():
    from modules.detect_pipeline import StageConfig
    assert StageConfig().peel_crossings is False
    assert StageConfig(peel_crossings=True).peel_crossings is True
