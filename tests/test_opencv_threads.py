"""OpenCV's threads must be turned back on after the AI library switches them off.

Importing ultralytics runs cv2.setNumThreads(0). Measured 2026-11-18 inside a real
20-frame batch, connected-components labelling then cost 120 ms a call (70 calls, 8.2 s
a batch) against 4.7 ms with threads on. The labels are identical either way.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.cpu_threads import restore_opencv_threads                  # noqa: E402


def _with_threads(n, fn):
    saved = cv2.getNumThreads()
    try:
        cv2.setNumThreads(n)
        return fn()
    finally:
        cv2.setNumThreads(saved)


def test_threads_come_back_after_being_switched_off():
    cores = cv2.getNumberOfCPUs()

    def run():
        got = restore_opencv_threads()
        return got, cv2.getNumThreads()
    for off in (0, 1):
        got, now = _with_threads(off, run)
        if cores > 1:
            assert got == now == cores, f"threads {off} -> {now}, expected {cores}"
        else:
            assert got == now


def test_it_leaves_an_already_threaded_opencv_alone():
    cores = cv2.getNumberOfCPUs()

    def run():
        cv2.setNumThreads(cores)
        return restore_opencv_threads()
    assert _with_threads(cores, run) == cv2.getNumThreads() or cores == 1


def test_component_labels_are_identical_with_threads_on_or_off():
    rng = np.random.default_rng(0)
    for seed in range(12):
        r = np.random.default_rng(seed)
        h, w = int(r.integers(40, 900)), int(r.integers(40, 900))
        m = (r.random((h, w)) < r.choice([0.002, 0.02, 0.2, 0.5])).astype(np.uint8)
        for _ in range(int(r.integers(0, 12))):
            cv2.line(m, (int(r.integers(0, w)), int(r.integers(0, h))),
                     (int(r.integers(0, w)), int(r.integers(0, h))), 1, int(r.integers(1, 9)))
        off = _with_threads(0, lambda: cv2.connectedComponentsWithStats(m))
        on = _with_threads(max(2, cv2.getNumberOfCPUs()), lambda: cv2.connectedComponentsWithStats(m))
        assert off[0] == on[0]
        assert np.array_equal(off[1], on[1]), f"labels differ in case {seed}"
        assert np.array_equal(off[2], on[2]), f"stats differ in case {seed}"
        assert np.allclose(off[3], on[3]), f"centroids differ in case {seed}"
    assert rng is not None


def test_engine_turns_threads_back_on_after_loading_the_model():
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    load = src.index("model, _device_used = load_model(")
    restore = src.index("restore_opencv_threads()")
    assert restore > load, "OpenCV threads must be restored AFTER the model load imports ultralytics"
