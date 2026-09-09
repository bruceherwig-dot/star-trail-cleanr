"""Bright static foreground survives repair, and the still routing is measurable.

Field case, 2026-09-05 (UofR Memorial Chapel, Bruce's own shoot): a FLOODLIT building.
Every foreground protection in the repair was built for foreground DARKER than the sky --
a Joshua-tree spike, a branch, a rock -- and gates on "darker than a fraction of the local
sky". Lit stonework is several times brighter, so it fell through every one of them, the
sky slide borrowed over it, and the result carried cloned pieces of the building.

These lock the mirror rule and, just as importantly, the two ways it must NOT fire: on
bright sky, and on a star. A star is at a pixel in one frame and gone by the next, so the
median across the window is plain sky -- which is why every gate reads the median rather
than this frame's pixel. Getting that wrong is what had the warm-pixel scrub deleting
stars (98% of what it removed was a correctly placed star) until it was disabled.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.repair import _brighten_fill, _BRIGHTEN_FG_FRAC


def _window(H, W, n, value):
    return np.full((n, H, W, 3), value, np.uint8)


def test_lit_building_under_a_trail_is_restored():
    """A lit wall erased to sky by the slide is pulled back by the median."""
    H = W = 40
    SKY, WALL = 40, 180                       # wall is 4.5x sky, well over the 1.6x gate
    patch_now = np.full((H, W, 3), SKY, np.uint8)     # slide erased the wall to sky
    wstack = _window(H, W, 7, SKY)
    wstack[:, :, 19:22] = WALL                        # the wall is there in EVERY frame
    dmed = np.median(wstack, axis=0).astype(np.uint8)

    comp_mask = np.zeros((H, W), bool)
    comp_mask[:, 15:26] = True                        # trail band over wall + sky

    out, px = _brighten_fill(patch_now, wstack, dmed, comp_mask, float(SKY))

    assert px > 0, "the lit wall under the trail should be restored"
    wall_out = out[:, 19:22].reshape(-1, 3).max(axis=1)
    assert np.median(wall_out) > 150, \
        f"wall should come back bright, got median {np.median(wall_out)}"
    sky_in_mask = out[:, 23:26].reshape(-1, 3).max(axis=1)
    assert np.median(sky_in_mask) <= SKY + 2, \
        "sky under the same mask must keep the slide, not be median-stamped"


def test_a_star_is_never_treated_as_bright_foreground():
    """THE failure this must not have. A star is bright in ONE frame of the window and has
    moved on by the next, so the median there is sky and the brightness gate never opens."""
    H = W = 30
    SKY = 40
    wstack = _window(H, W, 7, SKY)
    wstack[3, 14:17, 14:17] = 250            # a brilliant star, present in one frame only
    dmed = np.median(wstack, axis=0).astype(np.uint8)
    patch_now = np.full((H, W, 3), SKY, np.uint8)
    patch_now[14:17, 14:17] = 250            # the slide placed it correctly in the result

    comp_mask = np.zeros((H, W), bool); comp_mask[10:20, 10:20] = True
    out, px = _brighten_fill(patch_now, wstack, dmed, comp_mask, float(SKY))

    assert px == 0, "a one-frame star must never qualify as static bright foreground"
    assert np.array_equal(out, patch_now), "the star must be left exactly as the slide placed it"


def test_bright_sky_alone_does_not_qualify():
    """Light pollution or the Milky Way sits just above the local sky level. Only content
    several times brighter -- lit structure -- may be restored."""
    H = W = 30
    SKY = 40
    glow = int(SKY * (_BRIGHTEN_FG_FRAC - 0.3))       # bright, but under the gate
    wstack = _window(H, W, 7, glow)
    dmed = np.median(wstack, axis=0).astype(np.uint8)
    patch_now = np.full((H, W, 3), SKY, np.uint8)
    comp_mask = np.zeros((H, W), bool); comp_mask[10:20, 10:20] = True

    out, px = _brighten_fill(patch_now, wstack, dmed, comp_mask, float(SKY))
    assert px == 0, "bright sky under the gate must not be treated as foreground"


def test_a_moving_bright_thing_does_not_qualify():
    """Static is the second gate. Something bright that MOVES across the window fails it
    even though it is bright enough, because the frames do not agree with the median."""
    H = W = 30
    SKY = 40
    wstack = _window(H, W, 7, SKY)
    for i in range(7):                        # a bright blob marching across the window
        wstack[i, 12:18, 5 + 3 * i:8 + 3 * i] = 200
    dmed = np.median(wstack, axis=0).astype(np.uint8)
    patch_now = np.full((H, W, 3), SKY, np.uint8)
    comp_mask = np.ones((H, W), bool)

    out, px = _brighten_fill(patch_now, wstack, dmed, comp_mask, float(SKY))
    assert px == 0, "a bright object that moves must not be restored as foreground"


def test_the_two_rules_cannot_both_claim_a_pixel():
    """The darken rule takes pixels below 0.72x the local sky, this one above 1.60x, so the
    bands cannot overlap. A regression that moved either threshold across the other would
    let two rules write the same pixel."""
    from modules.repair import _DARKEN_FG_FRAC
    assert _DARKEN_FG_FRAC < _BRIGHTEN_FG_FRAC, \
        "the darken and brighten gates must stay on opposite sides of the local sky level"
    assert _DARKEN_FG_FRAC < 1.0 < _BRIGHTEN_FG_FRAC, \
        "one rule must be strictly below the sky level and the other strictly above"
