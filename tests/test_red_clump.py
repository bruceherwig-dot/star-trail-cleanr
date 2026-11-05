"""A small red light inside a big dark outline must count as red.

Found 2026-11-04 on Silvana Della Camera, Tree and Trails: one plane's red light moved across frames
1553-1567. The model outlined each dash (about 330 pixels) but only 18 of those pixels were clearly red,
so the old test (the middle value of the brightest fifth) read neutral, the star-size rule called the dash
a star and dropped it, and the small-piece filter's stricter colour ratio would have removed it anyway.
The shared red test now also counts warm pixels directly. The old test is kept, so nothing red stops being red.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.star_streak import _is_red  # noqa: E402
from modules.detect_trails import filter_small_components  # noqa: E402

H, W = 300, 400
SKY = (74, 76, 75)                     # B, G, R of a neutral night sky


def _scene(red_px=20, sky=SKY):
    """A neutral sky with a 28x15 px model outline and a few warm pixels inside it (BGR)."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = sky
    outline = np.zeros((H, W), np.uint8)
    cv2.ellipse(outline, (200, 150), (14, 8), 0, 0, 360, 255, -1)
    ys, xs = np.nonzero(outline)
    order = np.lexsort((xs, ys))
    for k in range(red_px):
        i = order[k * 3]
        img[ys[i], xs[i]] = (98, 122, 165)      # the dash's own colour, red minus blue 67
    return img, outline


def test_a_small_red_light_in_a_big_outline_counts_as_red():
    img, outline = _scene(red_px=20)
    assert _is_red(outline, img), "a clump of clearly warm pixels was not seen as red"


def test_the_old_middle_value_test_alone_would_have_missed_it():
    """Guards the test itself: with this few warm pixels the brightest-fifth middle reads neutral."""
    img, outline = _scene(red_px=20)
    px = img[outline > 0].astype(np.float32)
    bright = px.sum(axis=1)
    core = px[bright >= np.percentile(bright, 80)]
    assert float(np.median(core[:, 2]) - np.median(core[:, 0])) < 21, "setup: the old test must miss this"


def test_a_few_stray_warm_pixels_do_not_make_a_star_red():
    img, outline = _scene(red_px=4)
    assert not _is_red(outline, img)


def test_a_white_star_sized_dash_is_not_red():
    img, outline = _scene(red_px=0)
    img[outline > 0] = (200, 200, 200)
    assert not _is_red(outline, img)


def test_a_uniformly_orange_sky_does_not_make_everything_red():
    img, outline = _scene(red_px=0, sky=(90, 120, 170))
    assert not _is_red(outline, img)


def test_the_small_piece_filter_keeps_the_same_red_light():
    img, outline = _scene(red_px=20)
    mask = (outline > 0).astype(np.uint8) * 255
    dbg = {}
    out = filter_small_components(mask, img, debug_out=dbg, star_len=45.8, star_width=21.0)
    assert dbg["removed"] == 0 and dbg["kept_as_nav_light"] == 1, dbg
    assert out.sum() == mask.sum()


def test_the_small_piece_filter_still_removes_a_plain_star():
    img, outline = _scene(red_px=0)
    img[outline > 0] = (200, 200, 200)
    mask = (outline > 0).astype(np.uint8) * 255
    dbg = {}
    out = filter_small_components(mask, img, debug_out=dbg, star_len=45.8, star_width=21.0)
    assert dbg["removed"] == 1 and out.sum() == 0, dbg
