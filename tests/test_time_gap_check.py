"""The load-time pause check (modules/frame_list.find_time_gaps).

A pause in the shoot (battery swap, card swap) breaks every star trail there. The
check runs when a folder loads, before any cleaning, and only ever warns: it must
never reorder or drop a frame, and it must stay silent when it cannot judge.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.frame_list import (find_time_gaps, format_duration, gap_notice,
                                GAP_MAX_REPORTED)

T0 = datetime(2026, 11, 19, 21, 0, 0)


def _seq(offsets):
    frames = [f"/x/IMG_{i:04d}.jpg" for i in range(len(offsets))]
    return frames, {f: T0 + timedelta(seconds=s) for f, s in zip(frames, offsets)}


def test_steady_shoot_has_no_gap():
    frames, times = _seq([i * 20 for i in range(50)])
    assert find_time_gaps(frames, times) == []


def test_a_long_pause_is_found():
    offs = [i * 20 for i in range(30)] + [29 * 20 + 840 + i * 20 for i in range(1, 20)]
    frames, times = _seq(offs)
    gaps = find_time_gaps(frames, times)
    assert len(gaps) == 1
    pos, secs, typical = gaps[0]
    assert pos == 29 and secs == 860 and typical == 20
    note = gap_notice(frames, gaps)[0]
    assert "14m 20s" in note and "IMG_0029" in note and "IMG_0030" in note


def test_small_jitter_never_triggers():
    # 20 s typical, one 50 s gap: over 3x typical but under nothing to worry about
    # only if below the 30 s floor; here 50 > 30 and > 60? no: 50 < 60, so silent.
    offs = [i * 20 for i in range(20)]
    offs = offs[:10] + [o + 30 for o in offs[10:]]
    frames, times = _seq(offs)
    assert find_time_gaps(frames, times) == []


def test_fast_shoot_floor_protects_against_noise():
    # 2 s typical: a 9 s gap is over 3x typical but below the 30 s floor.
    offs = [i * 2 for i in range(20)]
    offs = offs[:10] + [o + 7 for o in offs[10:]]
    frames, times = _seq(offs)
    assert find_time_gaps(frames, times) == []


def test_missing_time_means_silence():
    frames, times = _seq([i * 20 for i in range(10)] )
    times[frames[3]] = None
    assert find_time_gaps(frames, times) == []


def test_too_few_frames_means_silence():
    frames, times = _seq([0, 20, 4000])
    assert find_time_gaps(frames, times) == []


def test_same_second_frames_do_not_zero_the_typical_gap():
    # several frames share a second; the typical gap is taken from real gaps only
    offs = [0, 0, 20, 20, 40, 60, 80, 100, 120]
    frames, times = _seq(offs)
    assert find_time_gaps(frames, times) == []


def test_only_the_biggest_few_are_reported_in_shooting_order():
    offs, t = [], 0
    for i in range(60):
        offs.append(t)
        t += 20 if i % 10 else 600 + i      # a pause every ten frames
    frames, times = _seq(offs)
    gaps = find_time_gaps(frames, times)
    assert len(gaps) == GAP_MAX_REPORTED
    assert [g[0] for g in gaps] == sorted(g[0] for g in gaps)


def test_check_does_not_touch_the_frame_list():
    offs = [i * 20 for i in range(10)] + [10 * 20 + 900]
    frames, times = _seq(offs)
    before = list(frames)
    find_time_gaps(frames, times)
    assert frames == before


def test_duration_wording_has_no_raw_seconds():
    assert format_duration(45) == "45s"
    assert format_duration(215) == "3m 35s"
    assert format_duration(240) == "4m"
    assert format_duration(3840) == "1h 4m"


def test_gui_uses_the_gap_aware_loader():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert "ordered_frame_list_and_gaps(folder)" in src
    assert "gap_notice(frames, gaps)" in src
