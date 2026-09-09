"""The brightness veto must not overrule overwhelming static evidence.

Field case, 2026-09-05 (UofR Memorial Chapel): a floodlit building under a
light-polluted sky. The static-foreground suppressor found the rooflines correctly --
3,249 suppressions on that run -- but the brightness veto then RESCUED 1,065 of them,
at a median 94% overlap with their neighbours. The veto's premise, "brighter than its
surroundings means a real trail", holds against a dark sky and fails against a lit
building. Those rescued detections were then repaired, which is what put cloned pieces
of the building into the output.

The veto still has to work: a genuinely bright trail that MOVES must survive, which is
the case these tests pin down alongside the fix. Reference values from the code's own
notes: Sompting Church rooflines 78-97%, the Green Park real trail that earlier logic
wrongly suppressed matched at 2.4%.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import astro_clean_v5 as engine

H = W = 300


def _bar(x, y, w=90, h=10):
    m = np.zeros((H, W), np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


def _frames_with(masks, sky=30, bright=220):
    """Frames where each mask region is much brighter than the sky, so the
    brightness veto would fire on every one of them."""
    out = []
    for m in masks:
        f = np.full((H, W, 3), sky, np.uint8)
        f[m > 0] = bright
        out.append(f)
    return out


def test_a_static_bright_edge_is_suppressed_despite_being_bright():
    """The chapel case: same place every frame, bright every frame. Must be removed."""
    masks = [_bar(100, 150) for _ in range(5)]          # identical position = 100% overlap
    frames = _frames_with(masks)
    engine._suppress_static_fps(masks, 0, 5, frames_all=frames)
    assert not masks[2].any(), (
        "a bright detection sitting at the SAME position in every frame is a fixed "
        "object; the brightness veto must not rescue it")


def test_a_bright_trail_that_moves_is_kept():
    """The veto's real job. A bright trail marching across the frame must survive."""
    masks = [_bar(40 + 60 * i, 150) for i in range(5)]   # moves well clear each frame
    frames = _frames_with(masks)
    engine._suppress_static_fps(masks, 0, 5, frames_all=frames)
    assert masks[2].any(), "a moving bright trail must never be suppressed"


def test_a_modest_overlap_still_lets_brightness_win():
    """Between the ordinary 70% match bar and the certainty bar, the veto still rules --
    a real bright trail can overlap its own previous position by a little."""
    assert engine._VETO_OVERRIDE_IOU_PCT > engine._suppress_static_fps.__defaults__[0] * 100, \
        "the certainty bar must sit ABOVE the ordinary match threshold, leaving a middle ground"
    # A steady 10px drift on a 90px bar: consecutive frames overlap at (90-10)/(90+10) = 80%,
    # which matches at the 70% bar but sits under the 90% certainty bar. It has to DRIFT, not
    # alternate between two spots -- alternating makes every second frame identical, which is
    # 100% overlap and correctly reads as static.
    masks = [_bar(100 + 10 * i, 150) for i in range(5)]
    frames = _frames_with(masks)
    engine._suppress_static_fps(masks, 0, 5, frames_all=frames)
    assert masks[2].any(), (
        "at moderate overlap the brightness veto must still protect the detection")


def test_the_certainty_bar_needs_more_than_one_frame():
    """One neighbour agreeing is not enough to overrule brightness; a second is."""
    assert engine._VETO_OVERRIDE_MATCHES >= 2, (
        "a single high-overlap neighbour must not be able to cancel the veto -- two "
        "different aircraft on one flight path can align once")
