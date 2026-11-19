"""The ground half of the flash finder, on synthetic sequences. Each case is a failure that
happened on a real photographer's set (Glenn Tetley's grass, Cheryl Hanscom Wilcox's
overexposed pink rock, Juniper's warm lit trees).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules import flash_repair as fr

H, W, N = 480, 640, 40
GROUND_TOP = 320


def _fg():
    m = np.zeros((H, W), bool)
    m[GROUND_TOP:] = True
    return m


def _frames(base=(18, 18, 18), noise=3, mutate=None, seed=4):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(N):
        f = np.empty((H, W, 3), np.float32)
        f[:] = base
        f += rng.normal(0, noise, (H, W, 3))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    if mutate:
        mutate(frames)
    return frames


def _stack(frames):
    out = frames[0].copy()
    for f in frames[1:]:
        out = np.maximum(out, f)
    return out


def _run(frames, trail_cut=25):
    read = lambda i: frames[i]
    g = fr.find_ground_specks(N, read, _fg())
    stack = _stack(frames)
    out, stats = fr.repair_ground(stack, N, read, g, trail_cut=trail_cut)
    return stack, out, stats, g


def test_a_two_frame_blue_speck_on_the_ground_is_found_and_removed():
    """It lasts two frames in a row, so it can never beat BOTH neighbouring frames: the
    ground is judged against each pixel's own long-run colour instead."""
    def mutate(frames):
        for k in (5, 6):
            frames[k][350:352, 400:404] = (250, 40, 40)
    stack, out, stats, g = _run(_frames(mutate=mutate))
    assert any(abs(x - 401) <= 6 and abs(y - 350) <= 6 for x, y in g.spots), g.spots
    assert stack[350, 401, 0] >= 240
    assert out[350:352, 400:404, 0].max() < 80, "the blue speck must come out of the stack"


def test_a_tall_red_and_white_streak_comes_out_completely():
    """Glenn's streak was 15 px tall with pale ends. 'Not long enough' left its tips."""
    def mutate(frames):
        for k in (7, 8):
            frames[k][340:355, 300] = (235, 235, 235)                 # pale column
            frames[k][345:350, 300] = (40, 40, 245)                   # strongly red middle
    stack, out, stats, g = _run(_frames(mutate=mutate))
    left = out[340:355, 298:303].max()
    assert left < 60, "part of the streak is still in the stack (brightest left: %d)" % left


def test_a_lighting_event_on_the_ground_is_skipped_not_repaired():
    """Cheryl's ground lit up in a few frames: frame 7 alone gave 580 false flags."""
    def mutate(frames):
        for y in range(GROUND_TOP + 10, H - 10, 20):
            for x in range(10, W - 10, 20):
                frames[20][y:y + 2, x:x + 2] = (40, 230, 250)         # hundreds of coloured spots at once
    stack, out, stats, g = _run(_frames(mutate=mutate))
    assert 20 in g.burst_frames, "a frame-wide event must be recognised"
    assert g.spots == []
    assert (out == stack).all(), "nothing may be painted because of a lighting event"


def test_a_small_speck_on_an_overexposed_ground_leaves_no_black_square():
    """The 'usual value' is the quiet frames' ground. Painting a whole window with it put
    black squares into bright rock."""
    cream = (238, 233, 224)

    def mutate(frames):
        for k in (3, 4, 5):
            frames[k][360, 200:202] = (40, 230, 250)                  # a tiny yellow dot
    stack, out, stats, g = _run(_frames(base=cream, noise=2, mutate=mutate))
    assert stats["pixels_changed"] <= 40, "far too many pixels changed: %d" % stats["pixels_changed"]
    win = out[345:376, 185:216]
    assert win.min() >= 200, "a dark square was painted into the bright ground (min %d)" % win.min()


def test_a_star_trail_crossing_the_ground_edge_is_not_cut():
    """A yellowish trail tip at the mask edge was cut once. A spot on a long bright line is
    a trail, not a speck."""
    def mutate(frames):
        for f in frames:
            cv2.line(f, (100, 280), (260, GROUND_TOP + 40), (245, 245, 245), 1)
        for k in (4, 5, 6):
            frames[k][GROUND_TOP + 28, 237] = (40, 220, 250)
    stack, out, stats, g = _run(_frames(mutate=mutate))
    trail = np.zeros((H, W), np.uint8)
    cv2.line(trail, (100, 280), (260, GROUND_TOP + 40), 255, 1)
    on = trail > 0
    assert (out[on].max(1) >= stack[on].max(1) - 5).all(), "the trail lost pixels"


def test_ground_that_is_coloured_scenery_does_nothing():
    """Juniper's warm lit trees gave ~100 coloured flags per frame. That is not specks."""
    rng = np.random.default_rng(9)

    def mutate(frames):
        for f in frames:
            for _ in range(60):
                y, x = int(rng.integers(GROUND_TOP + 5, H - 5)), int(rng.integers(5, W - 5))
                f[y, x] = (30, 40, 245)
    stack, out, stats, g = _run(_frames(mutate=mutate))
    assert g.bail, "it must say the ground is too busy to be specks"
    assert (out == stack).all()


def test_no_foreground_mask_means_no_ground_work():
    g = fr.find_ground_specks(N, lambda i: _frames()[0], None)
    assert g.bail and g.spots == []


def test_a_ground_repair_can_never_make_a_pixel_brighter():
    def mutate(frames):
        for k in (5, 6):
            frames[k][350:352, 400:404] = (250, 40, 40)
    stack, out, *_ = _run(_frames(mutate=mutate))
    assert (out.astype(int) - stack.astype(int)).max() <= 0
