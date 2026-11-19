"""A crossing the model outlined as one merged shape must not lose a streak when it is cut into bands.

Found 2026-11-04 on Joshua Tree frame 3033: a long stem with two arms branching off the same way
(an "F"). The splitter makes one band per line DIRECTION, so the two parallel arms shared one band
and one arm (the brightest pair of streaks) was left in the picture, and the stem stopped short.

The rule these tests pin: after bands are cut, every streak the model outlined is still covered.
Nothing here is tuned to frame 3033; the shape is painted from scratch.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import trail_grouper as tg  # noqa: E402

H, W = 520, 480
WIDTH = 26          # model outlines are much wider than the streak inside them


def _thick(mask, p0, p1, width=WIDTH):
    cv2.line(mask, p0, p1, 1, width)


def _f_blob():
    """A long stem and two arms that run the same way, one above the other."""
    blob = np.zeros((H, W), np.uint8)
    stem = ((60, 10), (330, 470))
    arm_a = ((330, 250), (460, 215))
    arm_b = ((330, 400), (460, 365))
    for p0, p1 in (stem, arm_a, arm_b):
        _thick(blob, p0, p1)
    return blob, (stem, arm_a, arm_b)


def _covered(bands, p0, p1):
    cov = np.zeros((H, W), np.uint8)
    for b in bands:
        cov |= b
    line = np.zeros((H, W), np.uint8)
    cv2.line(line, p0, p1, 1, 1)
    return float(cov[line > 0].mean())


def test_every_streak_of_an_f_shaped_blob_is_covered():
    blob, streaks = _f_blob()
    bands = tg.bands_for_blob(blob)
    assert bands, "an F shape has crossing evidence and must be cut into bands"
    for p0, p1 in streaks:
        assert _covered(bands, p0, p1) > 0.9, "a streak the model outlined was left uncovered: %s" % (p0,)


def test_the_old_bands_alone_would_have_missed_one():
    """Guards the test itself: without the leftover step something in this shape is uncovered."""
    blob, streaks = _f_blob()
    old = tg._bands_from_blob(blob)
    assert old, "setup: the splitter must see this as a crossing"
    worst = min(_covered(old, p0, p1) for p0, p1 in streaks)
    assert worst < 0.9, "the F no longer trips the old splitter; make the shape harder"


def test_a_plain_x_gets_no_extra_bands():
    """Nothing to add when the bands already cover the blob."""
    blob = np.zeros((H, W), np.uint8)
    _thick(blob, (20, 20), (460, 500))
    _thick(blob, (20, 500), (460, 20))
    old = tg._bands_from_blob(blob)
    new = tg.bands_for_blob(blob)
    assert old and new and len(new) == len(old), (len(old or []), len(new or []))


def test_a_blob_with_no_bands_is_left_alone():
    blob = np.zeros((H, W), np.uint8)
    _thick(blob, (20, 260), (460, 260))
    assert tg.bands_for_blob(blob) is None
