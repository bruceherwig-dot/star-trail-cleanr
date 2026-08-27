"""The original-source star trail stacks the SAME SHOTS as the cleaned one.

Bruce asked whether an original-frames star trail applies the same first/last
test-shot skip as the cleaned build. Same constant, same code -- but the skip
counts FILES, and an original folder can hold RAW+JPG twins of every shot plus
strays the run skipped, so "first 3 files" was not "first 3 shots" and comet
mode would fade each shot twice. _list_frames_matched (make_share_clip.py)
derives the shot list from the cleaned folder instead and matches each shot
back to one original file. These lock that behavior. Offline.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _msc():
    spec = importlib.util.spec_from_file_location("msc", REPO / "make_share_clip.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _folders(td, twins=True):
    orig = os.path.join(td, "orig"); os.makedirs(orig)
    cleaned = os.path.join(td, "cleaned"); os.makedirs(cleaned)
    for i in range(1, 13):
        open(os.path.join(orig, f"IMG_{i:04d}.jpg"), "w").write("x")
        if twins:
            open(os.path.join(orig, f"IMG_{i:04d}.CR2"), "w").write("x")
        open(os.path.join(cleaned, f"IMG_{i:04d}.jpg"), "w").write("x")
    open(os.path.join(orig, "stray_smallframe.jpg"), "w").write("x")
    return orig, cleaned


def test_matched_list_is_the_cleaned_shots_with_the_same_skip():
    msc = _msc()
    with tempfile.TemporaryDirectory() as td:
        orig, cleaned = _folders(td)
        names = msc._list_frames_matched(orig, cleaned)
        # 12 shots minus the 3+3 test-shot skip = 6, and the same 6 the
        # cleaned build uses.
        assert [os.path.splitext(n)[0] for n in names] == \
               [f"IMG_{i:04d}" for i in range(4, 10)]


def test_one_file_per_shot_even_with_twins_and_strays():
    msc = _msc()
    with tempfile.TemporaryDirectory() as td:
        orig, cleaned = _folders(td)
        names = msc._list_frames_matched(orig, cleaned)
        assert len(names) == len(set(os.path.splitext(n)[0] for n in names)), \
            "a shot must never appear twice (comet mode fades each entry)"
        assert all(n.lower().endswith(".jpg") for n in names), \
            "the non-RAW twin is the one to stack"
        assert not any("stray" in n for n in names)


def test_missing_originals_are_dropped_loudly_not_silently():
    msc = _msc()
    with tempfile.TemporaryDirectory() as td:
        orig, cleaned = _folders(td, twins=False)
        os.remove(os.path.join(orig, "IMG_0005.jpg"))
        names = msc._list_frames_matched(orig, cleaned)
        assert "IMG_0005.jpg" not in names and len(names) == 5


def test_panel_passes_match_cleaned_for_original_source():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert "--match-cleaned" in src, \
        "the Star Trail tab must request shot matching when stacking originals"
