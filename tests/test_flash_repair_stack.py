"""The repair half of the flash finder: flashes and blinking defects come out of the
finished stack, and NOTHING ELSE does.

Regressions that shaped it (Glenn Tetley's set, 2026-10-17): a solid star trail came out
of the repair broken into dashes, because false 'blinking defects' sat on the trail and
the repair replaced the pixels beside its white core with a dark baseline; and on an
overexposed ground the same baseline punched black squares into bright rock.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules import flash_repair as fr

H, W, N = 480, 640, 40


def _night(mutate=None, seed=3):
    rng = np.random.default_rng(seed)
    sx, sy, amp = rng.uniform(40, W - 90, 120), rng.uniform(40, H - 90, 120), rng.uniform(120, 230, 120)
    frames = []
    for k in range(N):
        canvas = np.zeros((H, W), np.float32)
        for x, y, a in zip(sx + 2 * k, sy + k, amp):
            xi, yi = int(round(x)), int(round(y))
            if 3 <= xi < W - 3 and 3 <= yi < H - 3:
                canvas[yi - 2:yi + 3, xi - 2:xi + 3] += a * np.array(
                    [[.05, .2, .3, .2, .05], [.2, .6, .8, .6, .2], [.3, .8, 1, .8, .3],
                     [.2, .6, .8, .6, .2], [.05, .2, .3, .2, .05]], np.float32)
        noise = rng.normal(18, 3, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(noise + canvas[:, :, None], 0, 255).astype(np.uint8))
    if mutate:
        mutate(frames)
    return frames


def _stack(frames):
    out = frames[0].copy()
    for f in frames[1:]:
        out = np.maximum(out, f)
    return out


def _run(frames):
    read = lambda i: frames[i]
    res = fr.find_flashes(N, read)
    defects = fr.find_blinking_defects(res.defect_candidates, N)
    stack = _stack(frames)
    out, stats = fr.repair_stack(stack, N, read, res.flags, defects)
    return stack, out, stats, res, defects


def test_a_one_frame_flash_comes_out_of_the_stack_and_nothing_else_changes():
    def mutate(frames):
        frames[12][300:302, 200:203] = (255, 40, 40)
    frames = _night(mutate)
    stack, out, stats, res, _d = _run(frames)
    clean = _stack(_night())                        # the same night with no flash at all
    assert np.abs(stack.astype(int) - clean.astype(int)).max() >= 100, "the flash is in the stack to begin with"
    win = (slice(294, 308), slice(194, 210))
    assert np.abs(out[win].astype(int) - clean[win].astype(int)).max() <= 12, \
        "after the repair the stack must match the night without the flash"
    changed = (out != stack).any(2)
    ys, xs = np.nonzero(changed)
    assert ys.min() >= 294 and ys.max() <= 308 and xs.min() >= 194 and xs.max() <= 210, \
        "pixels changed far from the flash"


def test_a_star_beside_a_flash_keeps_its_own_pixels():
    """Only the flash's own pixels are replaced. A star crossing the same window must keep
    its core: replacing the whole window notched star trails."""
    def mutate(frames):
        frames[12][300:302, 200:203] = (255, 40, 40)
        for k in range(N):                              # a bright neutral star passing 3 px away
            y, x = 304, 190 + 2 * k
            if x + 2 < W:
                frames[k][y - 1:y + 2, x - 1:x + 2] = (230, 230, 230)
    frames = _night(mutate)
    stack, out, *_ = _run(frames)
    star = np.zeros((H, W), bool)
    for k in range(N):
        y, x = 304, 190 + 2 * k
        if x + 2 < W:
            star[y - 1:y + 2, x - 1:x + 2] = True
    assert (out[star].max(1) >= stack[star].max(1) - 5).all(), "the star beside the flash lost pixels"


def test_a_repair_can_never_make_a_pixel_brighter():
    def mutate(frames):
        frames[12][300:302, 200:203] = (255, 40, 40)
        frames[25][100:102, 400:402] = (240, 240, 245)
    stack, out, *_ = _run(_night(mutate))
    assert (out.astype(int) - stack.astype(int)).max() <= 0, "a repair added light"


def test_a_blinking_coloured_defect_is_found_and_removed():
    bursts = (3, 4, 15, 16, 28, 29, 37)                      # separate bursts across the night

    def mutate(frames):
        for k in bursts:
            frames[k][200, 500] = (30, 30, 250)               # red (BGR), same pixel
    stack, out, stats, res, defects = _run(_night(mutate))
    assert any(abs(x - 500) <= 4 and abs(y - 200) <= 4 for x, y in defects), \
        "the blinking defect was not found: %r" % defects[:10]
    assert out[200, 500].max() < stack[200, 500].max() - 100 or out[200, 500, 2] < 150, \
        "the red defect must come out of the stack"


def test_a_star_fringe_that_passes_once_is_not_a_blinking_defect():
    def mutate(frames):
        for k in range(10, 18):                               # one contiguous stretch: a star going by
            frames[k][220 + k, 100 + 2 * k] = (250, 40, 40)
    _s, _o, _st, _res, defects = _run(_night(mutate))
    assert not any(100 <= x <= 140 and 230 <= y <= 240 for x, y in defects), \
        "a fringe that crosses once, in one stretch, is not a defect"


def test_a_solid_bright_trail_is_never_broken():
    """Glenn, 2026-10-17: 'the line on the left is nice and solid, the one on the right is
    terribly flawed'. A coloured fringe pixel beside a trail's white core, flashing on and
    off, must not cost the trail a single bright pixel."""
    def mutate(frames):
        for f in frames:
            cv2.line(f, (50, 400), (300, 150), (245, 245, 245), 1)           # a long solid trail in every frame
        for k in (2, 3, 14, 15, 27, 28, 36):
            frames[k][275, 175] = (255, 30, 30)                              # blue fringe beside the trail
    frames = _night(mutate)
    stack, out, *_ = _run(frames)
    trail = np.zeros((H, W), np.uint8)
    cv2.line(trail, (50, 400), (300, 150), 255, 1)
    on = trail > 0
    assert (out[on].max(1) >= stack[on].max(1) - 5).all(), \
        "the repair cost the trail bright pixels: %d of %d" % (
            int((out[on].max(1) < stack[on].max(1) - 5).sum()), int(on.sum()))


def test_a_speck_hidden_inside_a_bright_field_is_left_alone():
    """Nobody can see it, so touching it can only do harm (see the test above)."""
    def mutate(frames):
        for f in frames:
            f[150:250, 150:250] = (235, 235, 235)             # a bright field in every frame
        frames[10][200, 200] = (255, 40, 40)
    stack, out, stats, *_ = _run(_night(mutate))
    assert (out[150:250, 150:250] == stack[150:250, 150:250]).all(), \
        "a spot with nothing around it to stand out against must not be touched"


def test_nothing_to_repair_returns_the_stack_unchanged():
    frames = _night()
    stack = _stack(frames)
    out, stats = fr.repair_stack(stack, N, lambda i: frames[i], [], [])
    assert (out == stack).all() and stats["pixels_changed"] == 0
