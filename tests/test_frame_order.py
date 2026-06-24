"""Frames must be ordered NUMBER-AWARE, so a folder named 1.jpg ... 900.jpg
(GoPro and other un-padded numeric names) processes in true capture order.

A plain text sort puts "10" right after "1" and "2" after "100", which
scrambles the sequence -- the Star Bridge repair then borrows from the wrong
neighbor frames and the final stack is out of order. natural_key fixes it, and
zero-padded names (the common case) must be UNCHANGED.
"""
import os
import tempfile
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.frame_list import (natural_key, dedupe_frames,
                                order_by_capture_time, write_manifest, read_manifest)


def _dt(s):
    return datetime.strptime(s, "%Y:%m:%d %H:%M:%S")


def test_unpadded_numbers_order_numerically():
    names = [f"{n}.jpg" for n in (1, 2, 9, 10, 11, 100, 101, 900)]
    shuffled = ["100.jpg", "1.jpg", "11.jpg", "900.jpg", "2.jpg", "10.jpg", "101.jpg", "9.jpg"]
    assert sorted(shuffled, key=natural_key) == names


def test_plain_text_sort_would_be_wrong():
    # Guards the premise: a plain str sort really does scramble these, so this
    # test is meaningful (1, 10, 100, 2 ... is the bug we fixed).
    names = ["1.jpg", "2.jpg", "10.jpg", "100.jpg"]
    assert sorted(names) != sorted(names, key=natural_key)
    assert sorted(names, key=natural_key) == ["1.jpg", "2.jpg", "10.jpg", "100.jpg"]


def test_zero_padded_names_unchanged():
    # The common real-world case (IMG_0001, 143A8819, LRT_00651): number-aware
    # order must equal the existing plain-text order exactly, so nobody else is
    # affected by the fix.
    for prefix in ("IMG_", "143A8", "LRT_000", "DSC0"):
        padded = [f"{prefix}{n:04d}.jpg" for n in (1, 2, 9, 10, 99, 100)]
        scrambled = list(reversed(padded))
        assert sorted(scrambled, key=natural_key) == padded
        assert sorted(scrambled) == padded  # plain sort already correct here


def test_dedupe_frames_returns_numeric_order():
    # dedupe_frames is the shared function both the GUI and worker order through;
    # its output must be numeric so batch indices line up on both sides.
    paths = [f"/folder/{n}.jpg" for n in (3, 1, 20, 2, 10)]
    out = dedupe_frames(paths, prefer_raw=True)
    assert [p.split("/")[-1] for p in out] == ["1.jpg", "2.jpg", "3.jpg", "10.jpg", "20.jpg"]


def test_natural_key_mixed_names_no_crash():
    # Mixed naming in one folder must not raise (int-vs-str comparison guard).
    mixed = ["frame1.jpg", "1frame.jpg", "10.jpg", "IMG_2.jpg", "2.jpg"]
    sorted(mixed, key=natural_key)  # just must not throw


def test_capture_time_fixes_counter_rollover():
    # The camera wrapped: IMG_0001/0002 were shot AFTER IMG_9998/9999, so
    # filename order is wrong; capture time puts them back in true sequence.
    paths = ["/x/IMG_0001.jpg", "/x/IMG_0002.jpg", "/x/IMG_9998.jpg", "/x/IMG_9999.jpg"]
    times = {
        "/x/IMG_9998.jpg": _dt("2026:04:17 03:48:00"),
        "/x/IMG_9999.jpg": _dt("2026:04:17 03:48:31"),
        "/x/IMG_0001.jpg": _dt("2026:04:17 03:49:02"),
        "/x/IMG_0002.jpg": _dt("2026:04:17 03:49:33"),
    }
    assert order_by_capture_time(paths, times) == [
        "/x/IMG_9998.jpg", "/x/IMG_9999.jpg", "/x/IMG_0001.jpg", "/x/IMG_0002.jpg"]


def test_capture_time_all_or_nothing_fallback():
    # A single missing timestamp keeps the existing (filename) order -- no
    # risky mixing of two ordering schemes.
    paths = ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"]
    times = {"/x/a.jpg": _dt("2026:04:17 03:48:00"),
             "/x/b.jpg": None,
             "/x/c.jpg": _dt("2026:04:17 03:47:00")}
    assert order_by_capture_time(paths, times) == paths


def test_capture_time_same_second_tiebreak_is_natural():
    paths = ["/x/IMG_2.jpg", "/x/IMG_10.jpg", "/x/IMG_1.jpg"]
    t = _dt("2026:04:17 03:48:00")
    assert order_by_capture_time(paths, {p: t for p in paths}) == [
        "/x/IMG_1.jpg", "/x/IMG_2.jpg", "/x/IMG_10.jpg"]


def test_capture_time_empty():
    assert order_by_capture_time([], {}) == []


def test_frame_manifest_roundtrip():
    paths = ["/a/IMG_9998.jpg", "/a/IMG_9999.jpg", "/a/IMG_0001.jpg"]
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "m.txt")
        write_manifest(mp, paths)
        assert read_manifest(mp) == paths
