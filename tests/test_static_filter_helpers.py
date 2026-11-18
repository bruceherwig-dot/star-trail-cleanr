"""The two static-filter helpers must give exactly the answers the whole-frame versions gave.

Measured 2026-11-18 (6-frame batch, trail-heavy set): dropping rejected outlines took 3.8 s in one
call and the bright-trail check 1.9 s over 11 calls, because both counted across the whole photograph
for every detection. The original whole-frame code is kept here as the definition.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.static_filter_helpers import is_bright_trail_boxed, drop_rejected_segments   # noqa: E402

RATIO = 2.5


def _old_is_bright(comp_pixels, img_bgr):
    """THE ORIGINAL (astro_clean_v5._is_bright_trail), whole-frame."""
    if img_bgr is None:
        return False, 0.0
    ys, xs = np.where(comp_pixels)
    if len(ys) == 0:
        return False, 0.0
    inside = img_bgr[ys, xs].astype(np.float32)
    inside_bright = float(np.percentile(np.max(inside, axis=1), 90))
    m = comp_pixels.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(m, kernel) > 0) & ~comp_pixels
    ry, rx = np.where(ring)
    if len(ry) == 0:
        return False, 0.0
    surrounding = img_bgr[ry, rx].astype(np.float32)
    surround_med = float(np.median(np.max(surrounding, axis=1)))
    if surround_med < 1:
        return False, 0.0
    ratio = round(inside_bright / surround_med, 2)
    return ratio >= RATIO, ratio


def _old_drop(segs, corners, rm):
    """THE ORIGINAL loop from astro_clean_v5, whole-frame."""
    kept_s, kept_c, n_drop = [], [], 0
    for sj, seg in enumerate(segs):
        seg_on = seg > 0
        seg_px = int(seg_on.sum())
        if seg_px and int((seg_on & rm).sum()) * 2 >= seg_px:
            n_drop += 1
            continue
        kept_s.append(seg)
        if sj < len(corners):
            kept_c.append(corners[sj])
    return kept_s, kept_c, n_drop


def test_bright_trail_check_matches_the_whole_frame_version():
    for seed in range(300):
        rng = np.random.default_rng(seed)
        H, W = int(rng.integers(40, 160)), int(rng.integers(40, 160))
        sky = rng.integers(5, 60, (H, W, 3)).astype(np.uint8)
        # a component somewhere, including touching the frame edges
        x1 = int(rng.integers(0, W - 3)); y1 = int(rng.integers(0, H - 3))
        x2 = min(W - 1, x1 + int(rng.integers(2, 40))); y2 = min(H - 1, y1 + int(rng.integers(2, 40)))
        crop = rng.random((y2 - y1 + 1, x2 - x1 + 1)) < rng.uniform(0.1, 0.9)
        if seed % 3 == 0:                       # a bright component
            sky[y1:y2 + 1, x1:x2 + 1][crop] = rng.integers(150, 255, (int(crop.sum()), 3))
        if seed % 17 == 0:
            sky[:] = 0                          # a black frame: the "no sky to compare" exit
        full = np.zeros((H, W), bool)
        full[y1:y2 + 1, x1:x2 + 1] = crop
        want = _old_is_bright(full, sky)
        got = is_bright_trail_boxed(crop, (x1, y1, x2, y2), sky, RATIO)
        assert got == want, (seed, got, want)


def test_empty_component_and_missing_image():
    crop = np.zeros((5, 5), bool)
    assert is_bright_trail_boxed(crop, (3, 3, 7, 7), np.zeros((20, 20, 3), np.uint8), RATIO) == (False, 0.0)
    assert is_bright_trail_boxed(crop, (3, 3, 7, 7), None, RATIO) == (False, 0.0)


def test_dropping_rejected_outlines_matches_the_whole_frame_loop():
    for seed in range(300):
        rng = np.random.default_rng(10_000 + seed)
        H, W = int(rng.integers(40, 200)), int(rng.integers(40, 200))
        segs, corners = [], []
        for _ in range(int(rng.integers(0, 9))):
            m = np.zeros((H, W), np.uint8)
            cs = [[0, 0], [1, 1]]
            if rng.random() < 0.9:
                x, y = int(rng.integers(0, W - 5)), int(rng.integers(0, H - 5))
                x2, y2 = min(W - 1, x + int(rng.integers(1, 60))), min(H - 1, y + int(rng.integers(1, 30)))
                cv2.rectangle(m, (x, y), (x2, y2), 255, -1)
                cs = [[x, y], [x2, y], [x2, y2], [x, y2]]   # outlines are drawn from their corners
            segs.append(m)
            corners.append(cs)
        if rng.random() < 0.3 and corners:
            corners = corners[:-1]               # one outline without a corner list
        rm = np.zeros((H, W), bool)
        for _ in range(int(rng.integers(0, 4))):
            x, y = int(rng.integers(0, W - 5)), int(rng.integers(0, H - 5))
            rm[y:y + int(rng.integers(1, 40)), x:x + int(rng.integers(1, 80))] = True
        a = _old_drop(segs, corners, rm)
        b = drop_rejected_segments(segs, corners, rm)
        assert a[2] == b[2] and len(a[0]) == len(b[0]) and a[1] == b[1], seed
        assert all(x is y for x, y in zip(a[0], b[0])), seed


def test_dropping_with_nothing_removed_keeps_everything_and_handles_other_dtypes():
    H, W = 30, 40
    segs = [np.full((H, W), 255, np.uint8), np.zeros((H, W), np.uint8)]
    ks, kc, n = drop_rejected_segments(segs, [1, 2], np.zeros((H, W), bool))
    assert n == 0 and len(ks) == 2 and kc == [1, 2]
    seg16 = [np.full((H, W), 7, np.uint16)]
    rm = np.ones((H, W), bool)
    assert drop_rejected_segments(seg16, [[0]], rm)[2] == _old_drop(seg16, [[0]], rm)[2] == 1


def _drawn_outlines(rng, h, w, n, trim_sky=False):
    """Outlines drawn from their corners the way the engine draws them (filled polygon),
    optionally trimmed like the sky trim (pixels only ever removed)."""
    segs, corners = [], []
    for _ in range(n):
        pts = [[int(rng.integers(-20, w + 20)), int(rng.integers(-20, h + 20))] for _ in range(int(rng.integers(3, 6)))]
        seg = np.zeros((h, w), np.uint8)
        cv2.fillPoly(seg, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], 255)
        if trim_sky:
            seg[: int(rng.integers(0, h // 2)), :] = 0
        segs.append(seg)
        corners.append(pts)
    return segs, corners


def test_boxed_drop_matches_the_whole_frame_loop_on_outlines_drawn_from_corners():
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        h, w = int(rng.integers(40, 160)), int(rng.integers(40, 160))
        segs, corners = _drawn_outlines(rng, h, w, int(rng.integers(0, 14)), trim_sky=bool(seed % 2))
        rm = np.zeros((h, w), bool)
        for _ in range(int(rng.integers(0, 4))):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            rm[y:y + int(rng.integers(1, 40)), x:x + int(rng.integers(1, 40))] = True
        # sometimes the removed area is exactly an outline's pixels (certain drop)
        if segs and seed % 3 == 0:
            rm |= segs[0] > 0
        stats = {}
        a = _old_drop(segs, corners, rm)
        b = drop_rejected_segments(segs, corners, rm, stats)
        assert a[2] == b[2], seed
        assert [id(x) for x in a[0]] == [id(x) for x in b[0]], seed
        assert a[1] == b[1], seed
        if rm.any():
            assert stats["boxed"] + stats["whole_frame"] == len(segs)


def test_outlines_without_corners_fall_back_to_the_whole_frame_and_are_counted():
    rng = np.random.default_rng(7)
    segs, corners = _drawn_outlines(rng, 80, 90, 6)
    rm = np.zeros((80, 90), bool)
    rm[10:50, 10:50] = True
    short = corners[:4]                       # two outlines have no corners
    stats = {}
    a = _old_drop(segs, short, rm)
    b = drop_rejected_segments(segs, short, rm, stats)
    assert a[2] == b[2] and a[1] == b[1] and [id(x) for x in a[0]] == [id(x) for x in b[0]]
    assert stats["whole_frame"] == 2 and stats["boxed"] == 4
    # unusable corner lists also fall back
    stats = {}
    b = drop_rejected_segments(segs, [[], None, "x", [[1, 2]], [], []], rm, stats)
    assert b[2] == _old_drop(segs, [[], None, "x", [[1, 2]], [], []], rm)[2]
