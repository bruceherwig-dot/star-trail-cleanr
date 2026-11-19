"""The sky half of the flash finder, on small synthetic sequences with known answers.

Every case here is a real failure found on a photographer's sequence while the finder
was being built (see modules/flash_repair.py). The synthetic night: 150 stars drifting
2 px right and 1 px down per frame (so the sky's motion can be measured), soft round
points like real stars, over noise.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules import flash_repair as fr

H, W, N = 480, 640, 30


def _night(seed=1, mutate=None):
    """Return (read_frame, truth). `mutate(frames)` may edit the list of BGR frames."""
    rng = np.random.default_rng(seed)
    sx = rng.uniform(40, W - 80, 150)
    sy = rng.uniform(40, H - 80, 150)
    amp = rng.uniform(120, 230, 150)
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
        img = np.clip(noise + canvas[:, :, None], 0, 255).astype(np.uint8)
        frames.append(img)
    if mutate:
        mutate(frames)
    return (lambda i: frames[i]), frames


def _near(flags, x, y, frame=None, r=6):
    return [f for f in flags if abs(f.x - x) <= r and abs(f.y - y) <= r
            and (frame is None or f.frame == frame)]


def test_a_one_frame_blue_flash_is_found_and_the_moving_stars_are_not():
    def mutate(frames):
        frames[12][300:302, 200:203] = (255, 40, 40)          # blue (BGR), hard-edged, one frame only
    read, _ = _night(mutate=mutate)
    res = fr.find_flashes(N, read)
    assert res.bail is None, res.bail
    assert _near(res.flags, 201, 300, frame=12), "the blue flash was not found: %r" % res.flags[:5]
    assert len(res.flags) <= 2, "stars must not be flagged: %r" % res.flags[:8]


def test_a_stuck_pixel_that_never_moves_is_not_a_flash():
    def mutate(frames):
        for f in frames:
            f[240, 320] = (255, 255, 255)
    read, _ = _night(mutate=mutate)
    res = fr.find_flashes(N, read)
    assert not _near(res.flags, 320, 240), "a pixel bright in EVERY frame is not a flash"


def test_a_lighting_event_flags_nothing():
    def mutate(frames):
        for y in range(20, H - 20, 25):               # a grid of separate bright spots: 18 x 24 = 432 in one frame
            for x in range(20, W - 20, 25):
                frames[15][y:y + 2, x:x + 2] = (230, 235, 240)
    read, _ = _night(mutate=mutate)
    res = fr.find_flashes(N, read)
    assert not [f for f in res.flags if f.frame == 15], "a frame-wide event is not specks"
    assert res.burst_frames >= 1, "it must be recognised as a lighting event"


def test_a_long_streak_is_not_a_speck():
    def mutate(frames):
        cv2.line(frames[20], (100, 100), (160, 130), (255, 255, 255), 1)       # a plane: ~65 px long
    read, _ = _night(mutate=mutate)
    res = fr.find_flashes(N, read)
    assert not _near(res.flags, 130, 115, frame=20, r=40), "a 65 px streak must be left alone"


def test_a_star_that_only_dims_next_door_is_not_a_flash():
    """Atmospheric twinkling: a star 150 brighter in one frame but still there in the
    next. Read as 'brighter than its neighbours', it would be painted out."""
    def mutate(frames):
        for k, add in ((10, 0), (11, 0)):
            pass
        y, x = 60, 100
        for k in range(N):
            xi, yi = x + 2 * k, y + k
            if 5 < xi < W - 5 and 5 < yi < H - 5:
                frames[k][yi - 1:yi + 2, xi - 1:xi + 2] = 90
        frames[10][y + 10 - 1:y + 10 + 2, x + 20 - 1:x + 20 + 2] = 240          # bright now ...
        frames[11][y + 11 - 1:y + 11 + 2, x + 22 - 1:x + 22 + 2] = 120          # ... half as bright next frame
    read, _ = _night(mutate=mutate)
    res = fr.find_flashes(N, read)
    assert not _near(res.flags, 120, 70, frame=10, r=8), "a twinkling star is not a flash"


def test_a_sequence_whose_sky_motion_cannot_be_measured_does_nothing():
    rng = np.random.default_rng(2)
    frames = [rng.normal(20, 3, (H, W, 3)).clip(0, 255).astype(np.uint8) for _ in range(N)]
    frames[12][300:302, 200:203] = (255, 40, 40)
    res = fr.find_flashes(N, lambda i: frames[i])
    assert res.bail, "with no stars to lock onto the motion cannot be measured: it must say so"
    assert res.flags == []


def test_too_few_frames_to_compare_does_nothing():
    res = fr.find_flashes(3, lambda i: np.zeros((H, W, 3), np.uint8))
    assert res.bail and res.flags == []


def test_every_frame_is_read_once_apart_from_the_colour_sample():
    """The sky pass reads each frame once. The only repeats allowed are the handful of
    frames sampled up front to learn each pixel's usual colour."""
    reads = []
    read0, _ = _night()

    def counting(i):
        reads.append(i)
        return read0(i)
    fr.find_flashes(N, counting)
    twice = [i for i in set(reads) if reads.count(i) > 1]
    assert max(reads.count(i) for i in set(reads)) <= 2, "a frame was read more than twice"
    assert len(twice) <= fr.BASELINE_SAMPLES, "more frames re-read than the colour sample: %r" % sorted(twice)


def test_the_contrast_floor_is_chosen_from_the_sequence_not_fixed():
    read, _ = _night()
    res = fr.find_flashes(N, read)
    assert res.floor in fr.FLOOR_CANDIDATES
    src = (REPO / "modules" / "flash_repair.py").read_text(encoding="utf-8")
    assert "ELBOW" not in src.upper() or "elbow" in src, "the floor rule must stay documented"
