"""Darken-foreground restore: a trail crossing dark static foreground (a spike, branch,
rock) is rebuilt by a darken (min) blend across neighbor frames instead of being erased
by the sky slide. Unit-tests the _darken_fill helper directly, then checks repair_frame
reports foreground pixels saved end to end."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.repair import _darken_fill, repair_frame


def test_darken_fill_restores_dark_foreground():
    """A dark bar erased to sky in the repaired patch is pulled back to dark by the
    darken min, while sky pixels under the same mask are left untouched."""
    H = W = 40
    patch_now = np.full((H, W, 3), 60, np.uint8)      # slide erased everything to sky
    dmin = np.full((H, W, 3), 60, np.uint8)           # neighbor min = sky...
    dmin[:, 19:22] = 10                               # ...except the static dark bar

    comp_mask = np.zeros((H, W), bool)
    comp_mask[:, 15:26] = True                        # the trail band (covers bar + sky)
    collar = np.zeros((H, W), bool)
    collar[:, 5:14] = True                            # local sky ring (plenty of px)
    collar[:, 27:36] = True

    out, fg_px, sky = _darken_fill(patch_now, dmin, comp_mask, collar, 255, np.uint8)

    assert sky is not None and abs(sky - 60) < 1, f"local sky should read ~60, got {sky}"
    assert fg_px > 0, "the dark bar under the trail should be counted as foreground"
    # bar pixels restored to dark
    bar = out[:, 19:22].reshape(-1, 3).max(axis=1)
    assert np.median(bar) < 30, f"bar should be restored dark, got median {np.median(bar)}"
    # sky pixels under the mask untouched (kept at ~60, not darkened)
    sky_in_mask = out[:, 23:26].reshape(-1, 3).max(axis=1)
    assert np.median(sky_in_mask) > 50, f"sky under mask should stay ~60, got {np.median(sky_in_mask)}"


def test_darken_fill_noop_on_pure_sky():
    """A trail over open sky (no dark pixels) leaves the patch unchanged."""
    H = W = 30
    patch_now = np.full((H, W, 3), 70, np.uint8)
    dmin = np.full((H, W, 3), 68, np.uint8)           # all sky, nothing dark
    comp_mask = np.zeros((H, W), bool); comp_mask[:, 10:20] = True
    collar = np.zeros((H, W), bool); collar[:, 0:9] = True; collar[:, 21:30] = True
    out, fg_px, sky = _darken_fill(patch_now, dmin, comp_mask, collar, 255, np.uint8)
    assert fg_px == 0, "no foreground should be detected over open sky"
    assert np.array_equal(out, patch_now), "pure-sky patch must be left unchanged"


def _scene(n=5, size=80, sky=60, bar_cols=(38, 43), trail_rows=(25, 40),
           trail_cols=(20, 60), center=2):
    """N BGR frames: gray sky + a static dark bar, with a bright trail crossing the bar
    in the center frame only. Returns (frames, masks)."""
    rng = np.random.default_rng(7)
    frames, masks = [], []
    for i in range(n):
        f = np.full((size, size, 3), sky, np.uint8)
        f += rng.integers(0, 3, size=f.shape, dtype=np.uint8)
        f[:, bar_cols[0]:bar_cols[1]] = 10            # static dark bar, every frame
        m = np.zeros((size, size), np.uint8)
        if i == center:
            f[trail_rows[0]:trail_rows[1], trail_cols[0]:trail_cols[1]] = 220  # bright trail
            m[trail_rows[0]:trail_rows[1], trail_cols[0]:trail_cols[1]] = 255
        frames.append(f); masks.append(m)
    return frames, masks


def test_repair_frame_saves_foreground_end_to_end():
    """repair_frame removes the trail, keeps the dark bar dark, and reports the
    foreground pixels the darken step saved."""
    frames, masks = _scene()
    dbg = {}
    out = repair_frame(frames[2], masks[2], 2, frames, neighbor_masks=masks, debug_out=dbg)

    # the bar under the trail must still be dark (not filled with sky)
    bar_under_trail = out[30:35, 38:43].reshape(-1, 3).max(axis=1)
    assert np.median(bar_under_trail) < 35, \
        f"bar under trail should stay dark, got median {np.median(bar_under_trail)}"
    # the trail itself is gone (sky region under the trail is not bright)
    sky_under_trail = out[30:35, 22:30].reshape(-1, 3).max(axis=1)
    assert np.median(sky_under_trail) < 120, "trail should be removed from the sky"
    # and the darken step reported foreground saved (guards the routing stays wired)
    saved = sum(seg.get("fg_darken_px", 0)
                for comp in dbg.get("components", [])
                for seg in comp.get("segments", []))
    assert saved > 0, "darken step should report foreground pixels saved"
