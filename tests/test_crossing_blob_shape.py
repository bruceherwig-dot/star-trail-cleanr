"""The crossing steps measure a blob's tightest rotated rectangle from its outline, not
from every pixel (2026-11-21). The rectangle only depends on the blob's convex hull,
and the outline has the same hull, so the two must agree. These lock that on
synthetic tangles of the shapes the crossing steps exist for (X, L, T, V, three or
more lines, parallel pairs), on rotated copies, on awkward blobs (rings, two pieces,
an exact-threshold rectangle), and on the decisions the two callers make with the
numbers (the elongation cutoff and the fill cutoff). Offline.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import crossing_splitter as cs  # noqa: E402

N = 700
W = 20


def _line(c, ang, back, fwd):
    a = np.radians(ang)
    dx, dy = np.cos(a), np.sin(a)
    return (int(c[0] - dx * back), int(c[1] - dy * back), int(c[0] + dx * fwd), int(c[1] + dy * fwd))


def _draw(lines, w=W):
    m = np.zeros((N, N), np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(m, (x1, y1), (x2, y2), 255, w)
    ys, xs = np.where(m > 0)
    return m[ys.min() - 3:ys.max() + 4, xs.min() - 3:xs.max() + 4].copy()


def _shapes():
    c = (N // 2, N // 2)
    out = {}
    for ang in (90, 60, 40, 25, 12):
        out[f"X_{ang}"] = [_line(c, 0, 300, 300), _line(c, ang, 300, 300)]
    out["L"] = [_line((200, 400), 0, 0, 300), _line((200, 400), -90, 0, 300)]
    out["T"] = [_line(c, 0, 300, 300), _line((c[0], c[1]), 90, 0, 280)]
    out["V"] = [_line((350, 600), -60, 0, 400), _line((350, 600), -120, 0, 400)]
    out["star_3"] = [_line(c, a, 300, 300) for a in (0, 60, 120)]
    out["mesh_4"] = [_line(c, a, 300, 300) for a in (0, 45, 90, 135)]
    out["parallel_pair"] = [_line((350, 300), 0, 300, 300), _line((350, 360), 0, 300, 300)]
    out["fat_bend"] = [_line((200, 350), 0, 0, 250), _line((450, 350), 25, 0, 200)]
    return {k: _draw(v) for k, v in out.items()}


def _all_pixels_rect(mask):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    return cs._blob_rect_all_pixels(pts)


def _outline_rect(mask):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    return cs._blob_rect(mask, pts)


def _ratios(mask, rect):
    area = int((mask > 0).sum())
    major, minor = max(rect[1]), max(1e-6, min(rect[1]))
    return major / minor, area / max(1.0, major * minor)


def _check_same(mask, label):
    r1, r2 = _all_pixels_rect(mask), _outline_rect(mask)
    s1, s2 = sorted(r1[1]), sorted(r2[1])
    assert abs(s1[0] - s2[0]) < 1e-3 and abs(s1[1] - s2[1]) < 1e-3, f"{label}: sides differ {s1} vs {s2}"
    e1, f1 = _ratios(mask, r1)
    e2, f2 = _ratios(mask, r2)
    assert abs(e1 - e2) < 1e-5 and abs(f1 - f2) < 1e-5, f"{label}: ratios differ"
    assert (e1 > cs._ELONGATION_THRESH) == (e2 > cs._ELONGATION_THRESH), f"{label}: elongation decision flipped"
    assert (f1 > cs._EVIDENCE_MAX_FILL) == (f2 > cs._EVIDENCE_MAX_FILL), f"{label}: fill decision flipped"


def test_the_outline_gives_the_same_rectangle_on_every_kind_of_tangle():
    for name, m in _shapes().items():
        _check_same(m, name)


def test_the_outline_gives_the_same_rectangle_on_rotated_copies():
    for name, m in _shapes().items():
        for ang in (7, 19, 33, 58, 71):
            h, w = m.shape
            rot = cv2.warpAffine(m, cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), (w, h),
                                 flags=cv2.INTER_NEAREST)
            if rot.max() > 0:
                _check_same(rot, f"{name} rotated {ang}")


def test_awkward_blobs_agree_too():
    ring = np.zeros((200, 200), np.uint8)
    cv2.circle(ring, (100, 100), 80, 255, 14)                 # a hole in the middle
    two = np.zeros((200, 400), np.uint8)
    two[40:90, 20:120] = 255
    two[110:170, 240:380] = 255                               # two separate pieces
    for name, m in (("ring", ring), ("two pieces", two)):
        _check_same(m, name)


def test_a_rectangle_exactly_on_the_elongation_cutoff_lands_on_the_same_side():
    # The rectangle is fitted to pixel centres, so a block 91 by 21 pixels is 90 by 20
    # between centres: exactly 4.5 times as long as it is wide, the cutoff itself.
    m = np.zeros((60, 140), np.uint8)
    m[20:41, 20:111] = 255
    r1, r2 = _all_pixels_rect(m), _outline_rect(m)
    e1, _ = _ratios(m, r1)
    e2, _ = _ratios(m, r2)
    assert abs(e1 - cs._ELONGATION_THRESH) < 0.1, "this block sits on the cutoff"
    assert e1 == e2 and (e1 > cs._ELONGATION_THRESH) == (e2 > cs._ELONGATION_THRESH)


def test_the_callers_decide_the_same_with_the_outline_as_with_every_pixel():
    """Run the real split and crossing-evidence steps both ways: the outline version
    (as shipped) and every-pixel version (the definition) must give the same pieces."""
    shapes = _shapes()
    frame_px = 6000 * 4000
    shipped = cs._blob_rect
    try:
        for name, m in shapes.items():
            cs._blob_rect = shipped
            a_pieces = cs.split_crossing(m, frame_px=frame_px)
            a_evid = cs.has_crossing_evidence(m, frame_px=frame_px)
            cs._blob_rect = lambda mask, pts: cs._blob_rect_all_pixels(pts)
            b_pieces = cs.split_crossing(m, frame_px=frame_px)
            b_evid = cs.has_crossing_evidence(m, frame_px=frame_px)
            assert a_evid == b_evid, f"{name}: crossing-evidence answer differs"
            assert len(a_pieces) == len(b_pieces), f"{name}: piece count differs"
            for pa, pb in zip(a_pieces, b_pieces):
                assert np.array_equal(pa, pb), f"{name}: a piece differs"
    finally:
        cs._blob_rect = shipped


def test_the_outline_measurement_is_much_faster_on_a_big_blob():
    import time
    m = _shapes()["mesh_4"]
    big = cv2.resize(m, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(big > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    t = time.perf_counter()
    for _ in range(3):
        cs._blob_rect_all_pixels(pts)
    slow = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(3):
        cs._blob_rect(big, pts)
    fast = time.perf_counter() - t
    assert fast < slow, f"the outline version should be faster ({fast:.3f}s vs {slow:.3f}s)"
