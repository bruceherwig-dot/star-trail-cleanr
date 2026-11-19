"""A star-sized dash that MOVED must be kept; a star must still be dropped.

Found 2026-11-04 on Silvana Della Camera, Tree and Trails, frames 1586-1596. A plane's white strobe dash is
the size of a star streak, so the star-size rule dropped it and it left a dotted line in the stack. Stars in
that set shift about 1 px between frames, the dash steps about 25 px. A star re-appears within 15 px in the
next frame (the star-size module's own definition); a dropped piece with no blob within 15 px in any
neighbour is not a star. These tests paint that from scratch: no model, no image files.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.mover_rescue import find_movers, polygon_for  # noqa: E402

H, W = 900, 1200


def _sky(seed):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(25, 2, (H, W, 3)), 0, 255).astype(np.uint8)


def _stars(img, shift, n=60):
    """The same star field in every frame, moved `shift` px."""
    rng = np.random.default_rng(7)
    for _ in range(n):
        x, y = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        cv2.line(img, (x + shift, y), (x + shift + 6, y + 1), (230, 230, 230), 2)


def _dash(img, cx, cy):
    cv2.line(img, (cx - 12, cy - 3), (cx + 12, cy + 3), (240, 240, 240), 3)


def _piece(cx, cy):
    ys, xs = np.mgrid[cy - 9:cy + 10, cx - 15:cx + 16]
    return {"coords": np.column_stack([ys.ravel(), xs.ravel()]),
            "bbox": [cx - 15, cy - 9, cx + 15, cy + 9]}


def _frames(dash_steps):
    """Three frames; a dash at (600, 450) in the middle one and `dash_steps` px away in the others; stars fixed."""
    prev, cur, nxt = _sky(1), _sky(2), _sky(3)
    for im, shift in ((prev, 0), (cur, 1), (nxt, 2)):
        _stars(im, shift)
    _dash(cur, 600, 450)
    _dash(prev, 600 - dash_steps, 450 - dash_steps // 4)
    _dash(nxt, 600 + dash_steps, 450 + dash_steps // 4)
    return prev, cur, nxt


def test_a_dash_that_moved_far_between_frames_is_rescued():
    prev, cur, nxt = _frames(dash_steps=25)
    movers, checked = find_movers([_piece(600, 450)], cur, [prev, nxt])
    assert checked == 1 and len(movers) == 1, movers
    assert all(d is None or d > 15 for d in movers[0]["partner_px"]), movers[0]["partner_px"]
    assert "moved" in movers[0]["why"]


def test_a_star_with_a_partner_in_the_neighbours_is_not_rescued():
    prev, cur, nxt = _frames(dash_steps=25)
    rng = np.random.default_rng(7)
    x, y = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))     # the first star of the field
    movers, _ = find_movers([_piece(x + 4, y + 1)], cur, [prev, nxt])
    assert movers == [], "a real star (it re-appears 1 px away) was kept as a mover"


def test_a_dash_that_barely_moves_looks_like_a_star_and_is_left_alone():
    """A dash moving under 15 px per frame has a partner within reach: the documented limit of the rule."""
    prev, cur, nxt = _frames(dash_steps=8)
    movers, _ = find_movers([_piece(600, 450)], cur, [prev, nxt])
    assert movers == []


def test_a_piece_over_the_foreground_is_never_rescued():
    prev, cur, nxt = _frames(dash_steps=25)
    sky = np.full((H, W), 255, np.uint8)
    sky[400:500, 550:650] = 0
    movers, _ = find_movers([_piece(600, 450)], cur, [prev, nxt], sky_mask=sky)
    assert movers == []


def test_with_no_neighbouring_frame_nothing_is_rescued():
    prev, cur, nxt = _frames(dash_steps=25)
    assert find_movers([_piece(600, 450)], cur, [None, None])[0] == []
    assert find_movers([_piece(600, 450)], cur, [])[0] == []


def test_a_piece_with_no_visible_blob_of_its_own_is_not_rescued():
    """A model speck on empty sky has nothing in this frame to be a dash."""
    prev, cur, nxt = _frames(dash_steps=25)
    movers, _ = find_movers([_piece(300, 150)], cur, [prev, nxt])
    assert movers == []


def test_the_rescued_piece_gets_a_box_around_its_pixels():
    box = polygon_for(_piece(600, 450))
    assert len(box) == 4
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    assert min(xs) <= 585 and max(xs) >= 615 and min(ys) <= 441 and max(ys) >= 459


def test_the_engine_runs_the_mover_rescue_after_the_edge_rescue_and_logs_it():
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert "Step 1e - Mover rescue" in src and "--no-mover-rescue" in src
    assert src.index("Step 1d - Edge rescue") < src.index("Step 1e - Mover rescue"), "the rescue must run after the edge rescue"
    for field in ("mover_rescued", "mover_checked", "star_dropped_all"):
        assert field in src, field


# ---- the partner radius follows the set's own star shift -----------------------------------------

from modules.mover_rescue import partner_radius  # noqa: E402


def test_the_partner_radius_is_a_few_times_the_measured_star_shift_within_limits():
    assert partner_radius(1.0) == 6.0              # stars barely move: look only a short way
    assert partner_radius(3.0) == 12.0
    assert partner_radius(8.0) == 15.0             # never wider than the star-size module's 15 px
    assert partner_radius(None) == 15.0 and partner_radius(0) == 15.0
    assert partner_radius(float("nan")) == 15.0


def test_a_dash_with_a_neighbour_blob_12_px_away_is_rescued_when_stars_barely_move():
    """Silvana 1592 and 1596: the neighbouring dash sat 12-14 px away. With stars shifting 1 px a partner
    can only be within about 6 px, so this is a mover; with no measured shift the old 15 px would miss it."""
    prev, cur, nxt = _frames(dash_steps=25)
    # move the neighbour dashes to 13 px away
    prev, cur, nxt = _sky(1), _sky(2), _sky(3)
    for im, shift in ((prev, 0), (cur, 1), (nxt, 2)):
        _stars(im, shift)
    _dash(cur, 600, 450)
    _dash(prev, 587, 446)
    _dash(nxt, 613, 454)
    p = _piece(600, 450)
    assert find_movers([p], cur, [prev, nxt], radius=partner_radius(1.0))[0], "a mover 13 px from its neighbours was missed"
    assert find_movers([p], cur, [prev, nxt])[0] == [], "setup: the wide 15 px radius should have blocked this one"


def test_the_measured_star_shift_is_reported_by_the_star_size_measurement():
    from modules.star_size import measure_star_size
    rng = np.random.default_rng(3)
    a, b = _sky(11), _sky(12)
    for _ in range(900):
        x, y = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        cv2.line(a, (x, y), (x + 5, y), (220, 220, 220), 2)
        cv2.line(b, (x + 2, y), (x + 7, y), (220, 220, 220), 2)
    r = measure_star_size([a, b])
    if r is not None:
        assert "shift" in r and 0.5 < r["shift"] < 5, r
