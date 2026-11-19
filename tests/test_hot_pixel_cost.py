"""Building the stuck-pixel map must stay parallel, and must stay bounded.

Found 2026-08-26 on a 100-frame run of a masked sequence. This step cost 18.3
seconds per 20-frame batch, second only to repair, and it had never been noticed
in any earlier measurement because it ONLY RUNS WHEN THE USER PAINTED A
FOREGROUND MASK. Every test sequence used that day had none, so it reported 0ms
every time. Running somebody else's photographs is what found it.

The cost is entirely the median filter that finds each pixel's local background:
318ms per 24MP channel, three channels per frame, twenty frames per batch, and
OpenCV does not thread that operation. Sixty filters ran one after another while
eleven cores idled.

They are independent, so they now run a few frames ahead of the counting. On real
masked frames: 7.7s to 3.2s for eight frames, every cleaned file byte-for-byte
identical.

THE GROUP SIZE IS CAPPED ON PURPOSE. Each frame in flight holds three filtered
copies, so four 24MP frames is about 288 MB while the step runs. Letting that
scale with a 20-frame batch on a 44MP camera would trade a speed problem for a
memory one -- which is exactly the mistake made earlier the same day when mask
sharing pushed peak memory from 8.5 GB to 14.9 GB.
"""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SRC = (REPO / "modules" / "hot_pixels.py").read_text(encoding="utf-8")


def test_the_median_filters_still_run_in_parallel():
    assert "ThreadPoolExecutor" in SRC, (
        "the median filters are back to running one at a time; that was 18.3 "
        "seconds per batch on a masked run")
    assert "_grouped_medians" in SRC, "the grouped-median helper is gone"


def test_the_group_size_is_capped_and_small():
    m = re.search(r"_GROUP\s*=\s*(\d+)", SRC)
    assert m, "the cap on frames in flight is gone"
    assert 1 < int(m.group(1)) <= 8, (
        f"_GROUP is {m.group(1)}: it must stay small and FIXED. Each frame in "
        "flight holds three filtered copies of itself, so scaling this with the "
        "batch size would balloon memory on a big camera")


def test_it_still_falls_back_when_threads_are_unavailable():
    assert "except Exception:" in SRC and "yield f, _medians(f)" in SRC, (
        "the sequential fall-back is gone; a machine that cannot start threads "
        "would fail instead of simply running slower")


def test_a_stuck_pixel_is_still_found_and_a_star_is_not():
    """The behaviour the speed work must not have changed: a pixel that is bright
    and oddly coloured in the same place every frame is a defect; a neutral
    bright dot is a star and must survive."""
    import cv2
    from modules.hot_pixels import build_hot_pixel_map

    rng = np.random.default_rng(3)
    frames = []
    for _ in range(6):
        f = (rng.random((120, 160, 3)) * 20).astype(np.uint8)
        cv2.circle(f, (40, 40), 2, (210, 210, 210), -1)   # a neutral star
        f[80, 100] = (255, 15, 15)                        # a stuck pixel, every frame
        frames.append(f)

    m = build_hot_pixel_map(frames)
    assert m[80, 100] > 0, "the stuck pixel was not found"
    assert m[40, 40] == 0, "a neutral star was flagged as a defect"
