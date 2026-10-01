"""Speck removal must leave scenery-sized shapes alone.

Regression (2026-09-30, Sean Parker's aurora set): the soft dark edge of each cactus
was flagged as a ring of specks, the rings merged into one 298,255 px shape, and the
lift pass painted the whole shape with one sky colour taken near its centre (yellow
horizon glow), drawing a bright yellow outline around every cactus.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules import sky_dots


def test_a_speck_sized_shape_is_kept_and_a_huge_one_is_dropped():
    allmap = np.zeros((400, 600), np.uint8)
    allmap[50:55, 60:65] = 255            # 25 px: a real speck
    allmap[200:300, 100:500] = 255        # 40,000 px: scenery
    out = sky_dots._drop_oversize(allmap)
    assert out[52, 62] == 255, "a real speck must still be removed"
    assert out[250, 300] == 0, "a scenery-sized shape must be left alone"
    assert allmap[250, 300] == 255, "the caller's map must not be modified"


def test_a_ring_of_specks_around_scenery_is_not_painted_with_one_colour():
    """The Sean Parker case end to end: a yellow band low in the picture, red sky
    above, a dark post running up from one to the other with a thin flagged ring
    around it. The ring must come out exactly as shot, not painted yellow."""
    H, W = 300, 200
    big = np.zeros((H, W, 3), np.uint8)
    big[:200] = (60, 50, 140)             # red sky (BGR)
    big[200:] = (40, 230, 250)            # yellow glow
    big[40:260, 90:110] = (5, 5, 5)       # dark post
    ring = np.zeros((H, W), np.uint8)
    ring[36:264, 86:114] = 255
    ring[40:260, 90:110] = 0              # a ring about 4 px thick, 800+ px
    out, n, _ = sky_dots._fill_specks(big, sky_dots._drop_oversize(ring))
    assert n == 0
    assert np.array_equal(out, big), "scenery edges must be left exactly as shot"
