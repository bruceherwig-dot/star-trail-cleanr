"""Reading frames ahead and judging frames at once must not change any answer.

2026-11-19: the hot-pixel step went from 95 s to 56 s on 200 real frames by reading
frames on several threads and judging several at once. These lock that the answers are
exactly the ones the one-at-a-time code gave. Offline, synthetic data.
"""
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_frames_come_back_in_order_and_are_read_once():
    from modules.read_ahead import read_ahead
    seen, lock = [], threading.Lock()

    def read(i):
        with lock:
            seen.append(i)
        return i * 10

    got = list(read_ahead(read, list(range(30)), threads=4))
    assert got == [(i, i * 10) for i in range(30)], "in the order asked, with the right result"
    assert sorted(seen) == list(range(30)), "every frame read exactly once"


def test_it_handles_empty_and_single_item_lists():
    from modules.read_ahead import read_ahead
    assert list(read_ahead(lambda i: i, [], threads=4)) == []
    assert list(read_ahead(lambda i: i + 1, [5], threads=4)) == [(5, 6)]


def test_a_failed_read_surfaces_where_the_one_at_a_time_code_would_have_raised():
    from modules.read_ahead import read_ahead

    def read(i):
        if i == 3:
            raise ValueError("bad frame")
        return i

    out = []
    try:
        for item, r in read_ahead(read, list(range(8)), threads=3):
            out.append(r)
    except ValueError:
        pass
    assert out == [0, 1, 2], "frames before the failure are delivered first"


def _frames(n=12, h=80, w=120):
    rng = np.random.default_rng(11)
    frames = []
    for i in range(n):
        img = (rng.random((h, w, 3)) * 14).astype(np.uint8)
        img[29:32, 39:42] = (60, 60, 230)           # a stuck red pixel with its bloom, every frame
        img[30, 40] = (30, 30, 255)
        img[49:52, 89:92] = (245, 245, 245)         # a stuck white pixel with its bloom, every frame
        img[50, 90] = (255, 255, 255)
        img[20, 10 + 7 * i] = (255, 255, 255)       # a star that moves
        frames.append(img)
    return frames


def test_judging_frames_at_once_gives_exactly_the_one_at_a_time_answer():
    import tools.hot_pixels_v2 as H
    frames = _frames()
    keep = H._workers_for
    try:
        H._workers_for = lambda h, w: 1
        ch1 = H.build_hot_pixel_map_chromatic(frames, 22, 11, min_fraction=0.6)
        wh1 = H.build_hot_pixel_map_white(frames, 18, 9, min_fraction=0.6)
        H._workers_for = lambda h, w: 4
        ch4 = H.build_hot_pixel_map_chromatic(frames, 22, 11, min_fraction=0.6)
        wh4 = H.build_hot_pixel_map_white(frames, 18, 9, min_fraction=0.6)
    finally:
        H._workers_for = keep
    assert np.array_equal(ch1, ch4) and np.array_equal(wh1, wh4)
    assert ch1[30, 40] == 255 and wh1[50, 90] == 255, "the stuck pixels are still found"
    assert ch1[20, 10] == 0 and wh1[20, 10] == 0, "the moving star is still not"


def test_worker_count_is_at_least_one_and_at_most_four():
    import tools.hot_pixels_v2 as H
    assert 1 <= H._workers_for(3648, 5472) <= 4
    assert H._workers_for(1, 1) >= 1
