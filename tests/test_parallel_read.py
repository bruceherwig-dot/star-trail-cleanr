"""Frames may be read a few at a time, but must come back exactly as before:
same order, same pixels, same bad-file handling, and OpenCV's log level restored.
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.io_safe import robust_imread_diag                        # noqa: E402
from modules.parallel_read import read_frames_in_order                # noqa: E402


def _make_folder(d):
    rng = np.random.default_rng(3)
    files = []
    for k in range(7):
        p = d / f"f{k}.jpg"
        cv2.imwrite(str(p), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
        files.append(p)
    p = d / "f7.tif"
    cv2.imwrite(str(p), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
    files.append(p)
    p = d / "f8.tif"
    cv2.imwrite(str(p), rng.integers(0, 65535, (60, 80, 3), dtype=np.uint16))
    files.append(p)
    p = d / "f9.png"
    cv2.imwrite(str(p), rng.integers(0, 255, (60, 80), dtype=np.uint8))   # black and white
    files.append(p)
    p = d / "f10.png"
    cv2.imwrite(str(p), rng.integers(0, 255, (60, 80, 4), dtype=np.uint8))  # has transparency
    files.append(p)
    return files


def _fast(p):
    return robust_imread_diag(p, cv2.IMREAD_UNCHANGED, _retry_delays=())


def test_same_frames_in_the_same_order_as_one_at_a_time():
    with tempfile.TemporaryDirectory() as t:
        files = _make_folder(Path(t))
        want = [(p, *_fast(p)) for p in files]
        for workers in (1, 2, 3, 4):
            got = list(read_frames_in_order(files, _fast, workers=workers))
            assert [g[0] for g in got] == files
            for (wp, wi, wd), (gp, gi, gd) in zip(want, got):
                assert wd == gd
                assert wi.dtype == gi.dtype and wi.shape == gi.shape
                assert np.array_equal(wi, gi), f"{wp.name} differs with {workers} workers"


def test_order_holds_when_later_frames_finish_first():
    paths = [Path(f"/x/{k}") for k in range(6)]

    def slow_early(p):
        time.sleep(0.2 - 0.03 * int(p.name))      # frame 0 slowest, frame 5 fastest
        return np.full((2, 2, 3), int(p.name), np.uint8), None
    got = list(read_frames_in_order(paths, slow_early, workers=3))
    assert [g[0] for g in got] == paths
    assert [int(g[1][0, 0, 0]) for g in got] == list(range(6))


def test_bad_file_comes_back_in_its_place_with_its_reason():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        files = _make_folder(d)[:5]
        bad = d / "broken.jpg"
        bad.write_bytes(b"not a picture")
        files.insert(2, bad)
        got = list(read_frames_in_order(files, _fast, workers=3))
        assert [g[0] for g in got] == files
        assert got[2][1] is None and got[2][2]
        assert all(g[1] is not None for i, g in enumerate(got) if i != 2)


def test_log_level_is_muted_once_and_restored_once_even_when_stopped_early():
    """Not every OpenCV build has the logging API, so count the calls instead."""
    import modules.io_safe as IO
    calls = []
    real = (IO._silence_cv2_logs, IO._restore_cv2_logs)
    IO._silence_cv2_logs = lambda: calls.append("mute") or "old-level"
    IO._restore_cv2_logs = lambda prev: calls.append(("restore", prev))
    try:
        paths = [Path(str(k)) for k in range(6)]
        ok = lambda p: (np.zeros((2, 2, 3), np.uint8), None)      # noqa: E731
        list(read_frames_in_order(paths, ok, workers=3))
        assert calls == ["mute", ("restore", "old-level")]
        del calls[:]
        gen = read_frames_in_order(paths, ok, workers=3)
        next(gen)
        gen.close()                                # e.g. the user answers Stop
        assert calls == ["mute", ("restore", "old-level")]
    finally:
        IO._silence_cv2_logs, IO._restore_cv2_logs = real


def test_a_reader_error_is_raised_at_that_frames_turn():
    paths = [Path(f"/x/{k}") for k in range(4)]

    def reader(p):
        if p.name == "2":
            raise RuntimeError("boom")
        return np.zeros((2, 2, 3), np.uint8), None
    seen = []
    try:
        for p, _, _ in read_frames_in_order(paths, reader, workers=3):
            seen.append(p.name)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the error was swallowed")
    assert seen == ["0", "1"]


def test_reads_really_overlap_and_stay_capped():
    live = [0, 0]
    lock = threading.Lock()

    def reader(p):
        with lock:
            live[0] += 1
            live[1] = max(live[1], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return np.zeros((2, 2, 3), np.uint8), None
    list(read_frames_in_order([Path(str(k)) for k in range(12)], reader, workers=3))
    assert 2 <= live[1] <= 3


def test_works_without_threads():
    import concurrent.futures as cf
    import modules.parallel_read as PR
    real = PR.ThreadPoolExecutor

    class _NoThreads:
        def __init__(self, *a, **k):
            raise RuntimeError("threads unavailable")
    PR.ThreadPoolExecutor = _NoThreads
    try:
        paths = [Path(str(k)) for k in range(4)]
        got = list(read_frames_in_order(
            paths, lambda p: (np.full((2, 2, 3), int(p.name), np.uint8), None), workers=3))
    finally:
        PR.ThreadPoolExecutor = real
    assert [int(g[1][0, 0, 0]) for g in got] == [0, 1, 2, 3]
    assert cf.ThreadPoolExecutor is not None


def test_engine_loads_frames_through_the_helper():
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert "read_frames_in_order(frame_files_all, _read_one)" in src
    assert "robust_imread_diag(fp, cv2.IMREAD_UNCHANGED)" not in src
