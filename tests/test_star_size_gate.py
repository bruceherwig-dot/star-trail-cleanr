"""Smoke tests for the star-size rule that replaced the fixed pixel-area floor.

A detection is kept if it is RED (aircraft light), LONGER than the longest star, or FATTER than a star; only a
small, thin, non-red piece is dropped. These tests build fake stars and trails (no model, no image files):
the star measurement finds the right size, the detector gate and the small-piece filter follow the three
rules, and a set with too few stars falls back to the old fixed-size rule.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import detect_pipeline as dp  # noqa: E402
from modules.detect_trails import filter_small_components  # noqa: E402
from modules.star_size import measure_star_size  # noqa: E402

H, W = 2000, 3000


def _sky(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(8, 14, (H, W, 3), dtype=np.uint8)


def _star_pair(n=1500, shift=3, length=5):
    """Two frames of the same fake star field; every star is a short horizontal streak, moved by `shift` px."""
    rng = np.random.default_rng(1)
    a, b = _sky(2), _sky(3)
    xs = rng.integers(60, W - 60, n)
    ys = rng.integers(60, H - 60, n)
    for x, y in zip(xs, ys):
        cv2.line(a, (int(x), int(y)), (int(x) + length, int(y)), (220, 220, 220), 2)
        cv2.line(b, (int(x) + shift, int(y)), (int(x) + shift + length, int(y)), (220, 220, 220), 2)
    return a, b


def _mask_with(draw):
    m = np.zeros((H, W), np.uint8)
    draw(m)
    return m


def _cfg(star_len=12.0, star_width=7.0):
    # min_aspect=1.0 switches the elongation gate off so these tests isolate the star-size rule.
    return dp.StageConfig(star_len_px=star_len, star_width_px=star_width, min_aspect=1.0)


def _props(mask, cfg, img=None, rgb=False):
    log = dp.StageLog(dp.FrameLog(frame_name="test"), "fit_polygons")
    return dp._props_with_log(mask, cfg, log, img=img, img_is_rgb=rgb)


def test_star_measurement_finds_the_short_star_streaks():
    a, b = _star_pair(length=5)
    r = measure_star_size([a, b])
    assert r is not None and r["stars"] >= 500
    assert 3 <= r["length"] <= 12, r
    assert r["width"] <= 6, r


def test_a_plane_dash_does_not_count_as_a_star():
    """A 60 px dash that is in a different place in the next frame must not raise the star length."""
    a, b = _star_pair(length=5)
    cv2.line(a, (500, 500), (560, 500), (255, 255, 255), 3)
    cv2.line(b, (900, 800), (960, 800), (255, 255, 255), 3)
    r = measure_star_size([a, b])
    assert r is not None and r["length"] < 25, r


def test_too_few_stars_means_no_measurement():
    a, b = _sky(0), _sky(1)
    assert measure_star_size([a, b]) is None


def test_short_thin_non_red_piece_is_dropped_as_star_sized():
    m = _mask_with(lambda x: cv2.line(x, (1000, 1000), (1005, 1000), 255, 3))
    assert _props(m, _cfg()) is None


def test_a_short_dash_longer_than_the_longest_star_is_kept():
    # The kind the old fixed area floor deleted: ~40 px long, ~5 px wide, well under 1000 px of area.
    m = _mask_with(lambda x: cv2.line(x, (1000, 1000), (1040, 1000), 255, 5))
    assert _props(m, _cfg()) is not None


def test_a_piece_fatter_than_a_star_is_kept():
    m = _mask_with(lambda x: cv2.ellipse(x, (1000, 1000), (14, 10), 0, 0, 360, 255, -1))
    # Long cutoff set high so only "fatter than a star" can save it; the aspect gate is switched off for this check.
    cfg = dp.StageConfig(star_len_px=40.0, star_width_px=7.0, min_aspect=1.0)
    assert _props(m, cfg) is not None


def test_a_short_red_piece_is_kept():
    m = _mask_with(lambda x: cv2.line(x, (1000, 1000), (1005, 1000), 255, 3))
    img = np.zeros((H, W, 3), np.uint8)
    img[995:1006, 995:1015] = (230, 40, 40)           # RGB: red
    assert _props(m, _cfg(), img=img, rgb=True) is not None


def test_without_star_sizes_the_old_area_floor_still_applies():
    short_dash = _mask_with(lambda x: cv2.line(x, (1000, 1000), (1025, 1000), 255, 3))
    assert _props(short_dash, dp.StageConfig()) is None            # under the fixed floor: dropped, as before
    big = _mask_with(lambda x: cv2.line(x, (500, 500), (900, 500), 255, 6))
    assert _props(big, dp.StageConfig()) is not None


def test_small_piece_filter_follows_the_three_rules():
    img = np.full((H, W, 3), 10, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (200, 200), (205, 200), 255, 3)       # star-sized, not red -> removed
    cv2.line(mask, (600, 600), (640, 600), 255, 5)       # longer than the longest star -> kept
    cv2.line(mask, (1000, 1000), (1005, 1000), 255, 3)   # star-sized but red (BGR) -> kept
    img[995:1006, 995:1015] = (40, 40, 230)
    dbg = {}
    out = filter_small_components(mask, img, 1000, debug_out=dbg, star_len=12.0, star_width=7.0)
    assert out[200, 203] == 0
    assert out[600, 620] > 0
    assert out[1000, 1003] > 0
    assert dbg["removed"] == 1 and dbg["kept_as_nav_light"] == 1


def test_small_piece_filter_without_star_sizes_is_unchanged():
    img = np.full((H, W, 3), 10, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (600, 600), (640, 600), 255, 5)       # 40 px dash, under the fixed 1000 px area floor
    out = filter_small_components(mask, img, 1000)
    assert out[600, 620] == 0


def test_round_short_piece_is_kept_when_the_star_rule_is_on_even_with_the_default_roundness_setting():
    # ~24 x 14 px oval (ratio ~1.7): the old roundness check (default min_aspect 2.0) would have thrown it out.
    m = _mask_with(lambda x: cv2.ellipse(x, (1000, 1000), (12, 7), 0, 0, 360, 255, -1))
    assert _props(m, dp.StageConfig(star_len_px=9.9, star_width_px=4.7)) is not None
    # With no star sizes (old rules) the roundness check still applies.
    assert _props(m, dp.StageConfig(min_aspect=2.0)) is None


def _static_fp_masks():
    from astro_clean_v5 import _suppress_static_fps  # noqa: F401
    sz = 300
    masks = [np.zeros((sz, sz), np.uint8) for _ in range(7)]
    for m in masks:                                   # a fixed object: identical band in all 7 frames
        cv2.rectangle(m, (20, 40), (140, 50), 255, -1)
    for i in (3, 4):                                  # a trail seen in only two consecutive frames
        cv2.rectangle(masks[i], (20, 200), (140, 210), 255, -1)
    return masks


def test_a_detection_in_the_same_place_in_only_one_other_frame_is_kept_with_the_new_match_rule():
    from astro_clean_v5 import _suppress_static_fps, STATIC_FP_MIN_MATCHES
    assert STATIC_FP_MIN_MATCHES >= 3
    masks = _static_fp_masks()
    _suppress_static_fps(masks, 3, 4, min_matches=STATIC_FP_MIN_MATCHES)
    assert masks[3][45, 80] == 0          # fixed object (matched 6 neighbours) is still removed
    assert masks[3][205, 80] > 0          # two-frame trail (matched 1 neighbour) survives


def test_the_old_match_rule_would_have_removed_the_two_frame_trail():
    from astro_clean_v5 import _suppress_static_fps
    masks = _static_fp_masks()
    _suppress_static_fps(masks, 3, 4, min_matches=1)
    assert masks[3][205, 80] == 0
