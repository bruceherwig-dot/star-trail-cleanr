"""The colour-first stuck-pixel search must give exactly the reference's map.

The map keeps a pixel only if it passes BOTH the colour test and the brightness
test. The colour test is cheap; the brightness test (a 9x9 median over three
colour planes) is about 85 percent of the search. The colour-first search runs the
colour test on every frame, then the median test only in small tiles around pixels
that passed it. Measured 2026-11-18 on three real 20-frame batches: identical maps,
1.8 to 5.7 times faster. These tests hold it to the reference pixel for pixel,
especially where tiles meet each other and the image border, because a median at a
tile edge is exactly where a missing margin would show.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import hot_pixels as HP                                   # noqa: E402
from modules.hot_pixels import (_build_hot_pixel_map_colour_first,    # noqa: E402
                                _build_hot_pixel_map_reference,
                                build_hot_pixel_map)


def _noisy_frames(seed, n=8, h=160, w=220, stuck=40):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    spots = [(int(rng.integers(0, h)), int(rng.integers(0, w)),
              rng.integers(0, 256, 3, dtype=np.uint8)) for _ in range(stuck)]
    frames = []
    for _ in range(n):
        f = np.clip(base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
        for (y, x, col) in spots:
            if rng.random() < 0.9:
                f[y, x] = col
        for _ in range(60):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            f[y, x] = int(rng.integers(120, 256))
        for _ in range(60):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            f[y, x] = rng.integers(0, 256, 3, dtype=np.uint8)
        frames.append(f)
    return frames


def _colour_first_always(frames, threshold=2.0, min_fraction=0.8, margin=60.0):
    """Run the colour-first search with its 'covers too much of the frame' fallback switched off,
    so the tile path is exercised even on these small test images."""
    saved = HP._CF_MAX_COVER
    HP._CF_MAX_COVER = 10.0
    try:
        return _build_hot_pixel_map_colour_first(frames, threshold, min_fraction, margin)
    finally:
        HP._CF_MAX_COVER = saved


def _same(frames, **kw):
    ref = _build_hot_pixel_map_reference(frames, **kw)
    got = build_hot_pixel_map(frames, **kw)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert np.array_equal(ref, got)
    return ref


def test_identical_on_random_batches_of_many_sizes_and_the_new_path_is_used():
    flagged = 0
    for seed, (h, w) in enumerate([(160, 220), (129, 197), (64, 64), (65, 130), (200, 63),
                                   (30, 40), (7, 9), (257, 131), (190, 190), (128, 128),
                                   (333, 100), (97, 301)]):
        frames = _noisy_frames(seed, n=5 + seed % 6, h=h, w=w, stuck=20)
        ref = _same(frames)
        fast = _colour_first_always(frames)
        assert fast is not None, f"the colour-first path was not used for {h}x{w}"
        assert np.array_equal(fast, ref), f"colour-first differs at {h}x{w}"
        flagged += int((ref > 0).sum())
    assert flagged > 40


def test_stuck_pixels_on_tile_edges_image_borders_and_next_to_bright_bands():
    """Stuck pixels exactly on the 64 pixel tile boundaries and the image corners, with bright
    bands running across the boundaries so each pixel's 9x9 window spans two tiles."""
    for h, w in ((130, 200), (129, 197), (64, 64), (66, 70)):
        rng = np.random.default_rng(h * 1000 + w)
        spots = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (63, 63), (63, 64), (64, 63),
                 (64, 64), (62, 65), (65, 62), (60, 60), (68, 68), (h // 2, 63), (h // 2, 64)]
        spots = [(y, x) for y, x in spots if y < h and x < w]
        frames = []
        for k in range(7):
            f = rng.integers(10, 30, (h, w, 3), dtype=np.uint8)
            f[:, 61:67] = 140                          # bright vertical band over the tile edge
            f[61:67, :] = 150                          # and a horizontal one
            for (y, x) in spots:
                f[y, x] = (6, 12, 255)                 # a strongly coloured stuck pixel
            frames.append(f)
        ref = _same(frames)
        fast = _colour_first_always(frames)
        assert fast is not None and np.array_equal(fast, ref)


def test_exact_edge_of_the_brightness_test_with_a_cheap_colour_candidate():
    """Red-only patches whose background is 30..59 (not a colour candidate), and a test pixel at
    exactly twice the background (not brighter than twice) and one step over. Only the test
    pixels pass the colour test, so the colour-first path decides every one of them."""
    h, w = 40, 16 * 30 + 30
    frames = []
    for _ in range(5):
        f = np.zeros((h, w, 3), np.uint8)
        for b in range(30, 60):
            x = 16 * (b - 30) + 12
            f[:, x - 6:x + 6, 2] = b
            f[10, x, 2] = 2 * b
            f[25, x, 2] = 2 * b + 1
        frames.append(f)
    ref = _same(frames)
    fast = _colour_first_always(frames)
    assert fast is not None and np.array_equal(fast, ref)
    assert (ref[25] > 0).any() and not (ref[10] > 0).any(), "the edge cases stopped being exercised"


def test_exact_edge_of_the_colour_test():
    h, w = 30, 400
    frames = []
    for _ in range(5):
        f = np.zeros((h, w, 3), np.uint8)
        x = 5
        for mx in (120, 180, 255):
            for a in range(0, 61, 3):
                for bsum in (2 * (mx - 60) - 1, 2 * (mx - 60), 2 * (mx - 60) + 1):
                    other = bsum - a
                    if 0 <= other <= mx and 0 <= a <= mx and x < w - 5:
                        f[15, x] = (a, other, mx)
                        x += 6
        frames.append(f)
    ref = _same(frames)
    fast = _colour_first_always(frames)
    assert fast is not None and np.array_equal(fast, ref)
    assert (ref[15] > 0).any() and (ref[15] == 0).any()


def test_a_colourful_frame_falls_back_and_still_matches():
    """Most of the frame passes the colour test, so the tiles would cover it all: the colour-first
    search steps aside (returns None) and the answer is the whole-number search's."""
    rng = np.random.default_rng(4)
    frames = []
    for _ in range(6):
        f = np.zeros((120, 150, 3), np.uint8)
        f[:, :, 2] = 200 + rng.integers(0, 20, (120, 150), dtype=np.uint8)   # all red
        f[5, 5] = (6, 12, 255)
        frames.append(f)
    assert _build_hot_pixel_map_colour_first(frames, 2.0, 0.8, 60.0) is None
    _same(frames)


def test_a_sparse_full_size_batch_takes_the_fast_path_at_the_normal_setting():
    """A frame big enough that a few stuck pixels cover well under 40 percent of it."""
    rng = np.random.default_rng(8)
    h, w = 400, 500
    base = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    spots = [(int(rng.integers(0, h)), int(rng.integers(0, w))) for _ in range(8)]
    frames = []
    for _ in range(8):
        f = np.clip(base.astype(np.int16) + rng.integers(-5, 6, base.shape), 0, 255).astype(np.uint8)
        for (y, x) in spots:
            f[y, x] = (8, 20, 250)
        frames.append(f)
    fast = _build_hot_pixel_map_colour_first(frames, 2.0, 0.8, 60.0)
    assert fast is not None, "the normal setting sent a sparse frame to the fallback"
    assert np.array_equal(fast, _build_hot_pixel_map_reference(frames))
    assert (fast > 0).sum() >= 6


def test_no_candidates_one_frame_and_two_frames():
    quiet = [np.full((50, 60, 3), 30, np.uint8) for _ in range(6)]
    out = _same(quiet)
    assert not out.any()
    for n in (1, 2, 3):
        _same(_noisy_frames(n, n=n, h=70, w=90, stuck=10))


def test_other_inputs_still_use_the_reference():
    frames = _noisy_frames(9, n=6)
    _same(frames, threshold=2.5)
    sixteen = [f.astype(np.uint16) * 257 for f in frames]
    assert not HP._whole_number_search_applies(sixteen, 2.0, 60.0), "16-bit frames must not take the fast paths"
    for threshold, excess in ((1.0, 60.0), (3.0, 30.0), (4.0, 100.0)):
        _same(frames, threshold=threshold, min_channel_excess=excess)


def test_it_works_without_threads():
    import concurrent.futures as cf
    frames = _noisy_frames(21, n=6, h=130, w=140)
    want = _build_hot_pixel_map_reference(frames)
    real = cf.ThreadPoolExecutor

    class _NoThreads:
        def __init__(self, *a, **k):
            raise RuntimeError("threads unavailable")
    cf.ThreadPoolExecutor = _NoThreads
    try:
        got = _colour_first_always(frames)
    finally:
        cf.ThreadPoolExecutor = real
    assert got is not None and np.array_equal(want, got)


def test_the_tile_settings_stay_safe():
    assert HP._CF_MARGIN >= 4, "the margin must cover half of the 9 pixel median window"
    assert 0.1 <= HP._CF_MAX_COVER <= 0.6
    src = (REPO / "modules" / "hot_pixels.py").read_text(encoding="utf-8")
    assert "os.cpu_count" not in src
