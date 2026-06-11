"""Frames must be ordered NUMBER-AWARE, so a folder named 1.jpg ... 900.jpg
(GoPro and other un-padded numeric names) processes in true capture order.

A plain text sort puts "10" right after "1" and "2" after "100", which
scrambles the sequence -- the Star Bridge repair then borrows from the wrong
neighbor frames and the final stack is out of order. natural_key fixes it, and
zero-padded names (the common case) must be UNCHANGED.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.frame_list import natural_key, dedupe_frames


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
