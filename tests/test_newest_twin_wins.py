"""A cleaned folder holding last month's TIFFs next to today's JPEGs of the same shots.

Found 2026-11-04 on Joshua Tree 80 sec: the folder held 88 TIFFs from 2026-08-30 and 89 JPEGs from a new clean.
The Star Trail window's builds stacked the old TIFFs ("the TIFF wins"), so it showed a satellite the newest clean
had already removed, while the run's own end-of-run trail (which only reads this run's files) was correct.

The newer copy of a shot is the one the latest clean wrote, so it is the one that should be stacked. Copies
written within a couple of seconds of each other (one run writing both formats) keep the old TIFF-wins rule.
Nothing is deleted or modified: the older file is only left out of the stack.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from make_share_clip import _list_frames  # noqa: E402

N = 12


def _folder(tif_age_s, jpg_age_s):
    """N shots, each present as .tif and .jpg, with the given ages (seconds before now)."""
    import time
    d = Path(tempfile.mkdtemp())
    now = time.time()
    for i in range(N):
        for ext, age in ((".tif", tif_age_s), (".jpg", jpg_age_s)):
            p = d / f"IMG_{2946 + i}{ext}"
            p.write_bytes(b"x")
            os.utime(p, (now - age, now - age))
    return d


def test_a_newer_jpeg_beats_an_older_tiff_of_the_same_shot():
    d = _folder(tif_age_s=60 * 86400, jpg_age_s=10)
    names = _list_frames(str(d))
    assert names and all(n.endswith(".jpg") for n in names), names


def test_a_newer_tiff_still_beats_an_older_jpeg():
    d = _folder(tif_age_s=10, jpg_age_s=60 * 86400)
    names = _list_frames(str(d))
    assert names and all(n.endswith(".tif") for n in names), names


def test_copies_written_together_keep_the_tiff_wins_rule():
    d = _folder(tif_age_s=30, jpg_age_s=30)
    names = _list_frames(str(d))
    assert names and all(n.endswith(".tif") for n in names), names


def test_the_older_files_are_left_alone_on_disk():
    d = _folder(tif_age_s=60 * 86400, jpg_age_s=10)
    before = sorted(os.listdir(d))
    _list_frames(str(d))
    assert sorted(os.listdir(d)) == before
    assert len([n for n in before if n.endswith(".tif")]) == N
