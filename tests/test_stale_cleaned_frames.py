"""The end-of-run star trail shows THIS run's frames, whatever format they are.

THE INVARIANT. The cleaning worker overwrites the cleaned folder in place instead
of clearing it, so a folder can still hold output from an earlier session. The
trail built at the end of a run must show what that run produced -- somebody who
just watched 142 trails come out should not be handed last week's picture with
those trails still in it. Age decides: anything older than the moment the run
started is a leftover.

THE BUG THIS EXISTS FOR (2026-08-30). Deduplicating shots and checking their age
were done in the wrong order. The one-per-shot pick prefers a TIFF over a JPEG --
correct for "which copy is better" -- and only afterwards was each pick tested for
age. So a run writing JPEGs into a folder holding last week's TIFFs produced NO
STAR TRAIL AT ALL: every shot's TIFF won the pick, every TIFF was then rejected as
stale, and the fresh JPEGs were never in the running. Zero frames folded, no
stack, and the run ended reporting the trail as skipped.

WHY NOT JUST STACK THE OLD TIFFS AND SAVE A JPEG (Bruce's suggestion). Because
they are a different render of the same shots -- possibly a different model,
settings, or foreground mask -- and a frame limit makes it obvious: 88 shots
cleaned to TIFF last week, 20 frames run to JPEG today, and that rule stacks the
88 old ones and ignores everything you just made. The output is a JPEG either
way, so the extra depth is discarded at the moment of saving regardless. In the
Star Trail WINDOW, where there is no run to be faithful to, picking the best copy
of each shot is right -- and that is what happens there.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STALE_VALUE = 100      # brightness written by the imaginary earlier run
FRESH_VALUE = 200      # brightness written by the run under test


def _write(folder, i, ext, value):
    import cv2
    import tifffile
    p = os.path.join(folder, f"IMG_{2000 + i}{ext}")
    if ext == ".tif":
        tifffile.imwrite(p, np.full((40, 60, 3), value * 257, np.uint16),
                         photometric="rgb")
    else:
        cv2.imwrite(p, np.full((40, 60, 3), value, np.uint8))
    return p


def _run(stale_ext, fresh_ext, n_fresh=10, n_stale=10):
    """Set up a folder with an earlier run's output, start a stacker, then write
    this run's frames. Returns (frames folded, brightness of the trail)."""
    import cv2
    from modules.share_stacker import ShareStacker

    root = Path(tempfile.mkdtemp())
    orig = root / "src"
    clean = orig / "cleaned"
    clean.mkdir(parents=True)
    for i in range(10):
        cv2.imwrite(str(orig / f"IMG_{2000 + i}.jpg"),
                    np.full((40, 60, 3), 30, np.uint8))

    for i in range(n_stale):
        _write(str(clean), i, stale_ext, STALE_VALUE)
    old = time.time() - 86400                       # yesterday
    for f in os.listdir(clean):
        os.utime(clean / f, (old, old))

    s = ShareStacker(str(orig), str(clean), want_star=True, want_video=False)
    for i in range(n_fresh):
        _write(str(clean), i, fresh_ext, FRESH_VALUE)
    s.scan_cleaned()

    if s.after_full is None or s.after_full.result() is None:
        return 0, None
    res = s.after_full.result()
    scale = 257 if res.dtype == np.uint16 else 1
    return s.after_full.used, int(res.max()) // scale


def test_a_jpeg_run_into_a_folder_of_old_tiffs_still_builds_a_trail():
    """THE BROKEN CASE. Nothing was folded at all before the ordering was fixed."""
    used, value = _run(".tif", ".jpg")
    assert used > 0, "no frames were folded; the run would end with no star trail"
    assert value == FRESH_VALUE, (
        f"the trail was built from the {'earlier run' if value == STALE_VALUE else 'wrong'} "
        f"frames (brightness {value}, expected {FRESH_VALUE})")


def test_a_tiff_run_into_a_folder_of_old_jpegs_is_unchanged():
    """The direction that always worked -- the fix must not cost it."""
    used, value = _run(".jpg", ".tif")
    assert used > 0 and value == FRESH_VALUE


def test_the_same_format_twice_is_unchanged():
    used, value = _run(".jpg", ".jpg")
    assert used > 0 and value == FRESH_VALUE


def test_a_frame_limited_run_uses_its_own_few_frames():
    """The case that settles the argument: a short run into a folder holding a
    full earlier sequence must stack the few frames it just made, not the many
    old ones sitting beside them."""
    used, value = _run(".tif", ".jpg", n_fresh=6, n_stale=10)
    assert used > 0, "the limited run produced no trail"
    assert value == FRESH_VALUE, (
        "the trail came from the earlier full run instead of the frames this "
        "run actually cleaned")


def test_a_folder_of_nothing_but_leftovers_builds_nothing():
    """The invariant's other half. A run that has written nothing yet must not
    quietly present the previous run's trail as if it were new."""
    used, value = _run(".tif", ".jpg", n_fresh=0)
    assert used == 0 and value is None, (
        f"stacked {used} leftover frame(s) from an earlier run")


def test_the_age_test_runs_before_the_one_per_shot_pick():
    """The ordering IS the fix -- guard it by name, since reversing the two lines
    reintroduces the empty trail without failing anything else obvious."""
    body = (REPO / "make_share_clip.py").read_text(encoding="utf-8")
    body = body[body.index("def _list_frames("):]
    body = body[:body.index("\ndef ", 10)]
    assert "keep(os.path.join(folder, f))" in body, (
        "the caller's file test is no longer applied while listing, so it can "
        "only run after a shot's copy has already been chosen")
    src = (REPO / "modules/share_stacker.py").read_text(encoding="utf-8")
    assert "keep=self._written_by_this_run" in src, (
        "the end-of-run stacker no longer filters to this run's frames as it lists")


def test_the_window_still_picks_the_best_copy_of_each_shot():
    """The other place, deliberately different: no run to be faithful to, so the
    higher-quality copy of a shot wins."""
    import make_share_clip as msc
    d = Path(tempfile.mkdtemp())
    for i in range(10):
        _write(str(d), i, ".jpg", 100)
        _write(str(d), i, ".tif", 100)
    got = msc._list_frames(str(d))
    assert got and all(n.endswith(".tif") for n in got), (
        f"the window should stack the TIFF copy of each shot: {got}")
