"""The faster stuck-pixel search must give exactly the reference's map.

Measured 2026-11-16: finding stuck pixels was 6 to 17 percent of every masked
run. The search compared every pixel with its local background and tested its
colour balance using decimal arithmetic on full-size copies of each frame, one
frame at a time. The same two questions can be answered in whole-number
arithmetic on the 8-bit data. About a third less time, less memory, same map.

"Same map" is the whole promise, because this map decides which ground pixels
get painted over and where the speck remover looks in the sky. So these tests
compare the two versions pixel for pixel, especially at the exact edges of both
tests, where a rounding difference would show.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import hot_pixels as HP                                   # noqa: E402
from modules.hot_pixels import (build_hot_pixel_map,                   # noqa: E402
                                _build_hot_pixel_map_reference,
                                _build_hot_pixel_map_whole_numbers,
                                _whole_number_search_applies)


def _noisy_frames(seed, n=8, h=160, w=220, stuck=40):
    """Frames with moving neutral stars, coloured stuck pixels in fixed places,
    coloured flecks that come and go, and plenty of noise."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    spots = [(int(rng.integers(5, h - 5)), int(rng.integers(5, w - 5)),
              rng.integers(0, 256, 3, dtype=np.uint8)) for _ in range(stuck)]
    frames = []
    for k in range(n):
        f = np.clip(base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
        for (y, x, col) in spots:
            if rng.random() < 0.9:                       # most frames, not all
                f[y, x] = col
        for _ in range(60):                              # moving neutral stars
            y, x = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            f[y, x] = int(rng.integers(120, 256))
        for _ in range(60):                              # one-frame coloured flecks
            y, x = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            f[y, x] = rng.integers(0, 256, 3, dtype=np.uint8)
        frames.append(f)
    return frames


def test_identical_to_the_reference_on_many_random_batches():
    flagged = 0
    for seed in range(12):
        frames = _noisy_frames(seed, n=5 + seed % 6)
        ref = _build_hot_pixel_map_reference(frames)
        new = build_hot_pixel_map(frames)
        assert new.dtype == ref.dtype and new.shape == ref.shape
        assert np.array_equal(ref, new), f"maps differ for random batch {seed}"
        flagged += int((ref > 0).sum())
    assert flagged > 50, "the random batches stopped containing stuck pixels to find"


def test_identical_at_the_exact_edge_of_the_brightness_test():
    """Background b, pixel exactly 2b (not brighter than twice) and 2b + 1 (just
    brighter), for every background level, including 0 and levels whose double
    is over 255. The colour is made strongly lopsided so only brightness decides."""
    h, w = 40, 16 * 140 + 30
    frames = []
    for _ in range(5):
        f = np.zeros((h, w, 3), np.uint8)
        for b in range(0, 140):
            x = 16 * b + 12                                # patches do not overlap
            f[:, x - 6:x + 6, 2] = b                       # a flat patch of background
            f[10, x, 2] = min(255, 2 * max(b, 1))           # exactly on the edge
            f[25, x, 2] = min(255, 2 * max(b, 1) + 1)       # one step over
        frames.append(f)
    ref = _build_hot_pixel_map_reference(frames, min_channel_excess=0.0)
    new = build_hot_pixel_map(frames, min_channel_excess=0.0)
    assert np.array_equal(ref, new)
    assert (ref > 0).any() and (ref[10] == 0).any(), "the edge cases stopped being exercised"


def test_identical_at_the_exact_edge_of_the_colour_test():
    """Pixels whose colour imbalance is exactly at the margin, half a step under
    and half a step over (the reference works in halves)."""
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
                        f[15, x] = (a, other, mx)           # B, G, R: R is the tall one
                        x += 6
        frames.append(f)
    ref = _build_hot_pixel_map_reference(frames)
    new = build_hot_pixel_map(frames)
    assert np.array_equal(ref, new)
    assert (ref[15] > 0).any() and (ref[15] == 0).any()


def test_other_whole_number_settings_also_match():
    frames = _noisy_frames(99, n=7)
    for threshold, excess in ((1.0, 60.0), (3.0, 30.0), (2.0, 0.5), (4.0, 100.0)):
        assert _whole_number_search_applies(frames, threshold, excess)
        ref = _build_hot_pixel_map_reference(frames, threshold=threshold, min_channel_excess=excess)
        new = build_hot_pixel_map(frames, threshold=threshold, min_channel_excess=excess)
        assert np.array_equal(ref, new), f"maps differ at threshold {threshold}, margin {excess}"


def test_unusual_inputs_use_the_reference():
    frames = _noisy_frames(5, n=5)
    assert not _whole_number_search_applies(frames, 2.5, 60.0), "a fractional multiplier is not exact"
    assert not _whole_number_search_applies(frames, 2.0, 60.3), "a margin with an odd fraction is not exact"
    sixteen = [f.astype(np.uint16) * 257 for f in frames]
    assert not _whole_number_search_applies(sixteen, 2.0, 60.0), "16-bit frames must use the reference"
    assert not _whole_number_search_applies([], 2.0, 60.0)
    # And the public function still answers for them, through the reference.
    out = build_hot_pixel_map(frames, threshold=2.5)
    assert np.array_equal(out, _build_hot_pixel_map_reference(frames, threshold=2.5))


def test_it_works_without_threads(monkeypatch=None):
    """A machine that cannot start threads must get the same map, just slower."""
    import concurrent.futures as cf
    frames = _noisy_frames(21, n=6)
    want = _build_hot_pixel_map_reference(frames)
    real = cf.ThreadPoolExecutor

    class _NoThreads:
        def __init__(self, *a, **k):
            raise RuntimeError("threads unavailable")

    cf.ThreadPoolExecutor = _NoThreads
    try:
        got = _build_hot_pixel_map_whole_numbers(frames, 2.0, 0.8, 60.0)
    finally:
        cf.ThreadPoolExecutor = real
    assert np.array_equal(want, got)


def test_frames_in_flight_stay_capped():
    assert 1 < HP._GROUP_FAST <= 8, (
        "the number of frames worked on at once must stay small and fixed; "
        "each one holds several full-size working copies")
    src = (REPO / "modules" / "hot_pixels.py").read_text(encoding="utf-8")
    assert "os.cpu_count" not in src, (
        "do not scale the stuck-pixel search with the core count; on a big "
        "camera that trades a speed problem for a memory one")
