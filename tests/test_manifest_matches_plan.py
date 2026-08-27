"""The frame list the engine is given must BE the list the batch plan counted.

Field case, 2026-08-11 (Sentry, two users, one Windows one Mac): every run died
with "Worker exited 1: ERROR: need >= 3 frames (got 0)". The GUI wrote its frame
manifest BEFORE applying the frame range, the frame limit and the resolution
filter, then planned batches on what survived those. So the engine was told
"frames 0 to 11" and handed a list that still contained every file the GUI had
decided to skip. It sliced that list, dropped the wrong-sized frames, and had
nothing left.

The second half of the same bug was silent: the list of matching frames was
rebuilt with sorted(), which threw away the capture-time order the manifest had
just been written in. Filename order and shooting order differ whenever a camera
rolls over (IMG_9999 -> IMG_0001) or two cards are merged, and then every batch
repairs frames using the wrong neighbours, with no error at all.

These tests read the GUI source and check the ORDER OF OPERATIONS, because the
scan lives inside a Qt worker thread that cannot be driven headlessly here.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SRC = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")


def _pos(needle):
    i = SRC.find(needle)
    assert i > 0, f"anchor vanished from star_trail_cleanr.py: {needle!r}"
    return i


def test_manifest_is_written_after_every_filter():
    """The manifest write must come after the range, the limit and the size
    filter -- otherwise it describes a different set of frames than the plan."""
    write = _pos("write_manifest(_mf.name, frames)")
    frame_range = _pos("frames = frames[self.frame_start : self.frame_end + 1]")
    limit = _pos("frames = frames[:total]")
    size_filter = _pos("frames = matching")

    assert write > frame_range, \
        "manifest is written before the frame range is applied"
    assert write > limit, \
        "manifest is written before the frame limit is applied"
    assert write > size_filter, \
        "manifest is written before odd-sized frames are dropped -- this is the " \
        "bug that produced 'need >= 3 frames (got 0)' for two users"


def test_manifest_is_written_before_the_batch_plan():
    """Belt and braces: the plan must be computed from the same list, after."""
    write = _pos("write_manifest(_mf.name, frames)")
    plan = _pos("starts = list(range(0, total, batch_size))")
    assert write < plan, "the batch plan must be built from the manifest's list"


def test_matching_frames_keep_their_capture_time_order():
    """Building the matching list must filter in place, never re-sort.

    sorted() here silently reverts to filename order and the engine then cleans
    in a different order than the plan assumed.
    """
    line = re.search(r"^\s*matching = .*$", SRC, re.M)
    assert line, "the matching-frames line vanished"
    text = line.group(0)
    assert "sorted(" not in text, (
        "matching frames are being re-sorted, which throws away capture-time "
        f"order: {text.strip()}")
    assert "for f in frames" in text, (
        "matching must be filtered from `frames` (already in capture order), "
        f"not rebuilt from a dict: {text.strip()}")


def _mixed_size_folder():
    """A folder like the two users had: mostly one size, some files another,
    with the odd ones sitting early enough to poison the first batch."""
    import numpy as np
    import cv2
    import tempfile

    d = Path(tempfile.mkdtemp())
    order = []
    for i in range(20):
        odd = i < 6                       # the first six are a different size
        h, w = (40, 60) if odd else (80, 120)
        p = d / f"F{i:03d}.jpg"
        cv2.imwrite(str(p), np.zeros((h, w, 3), np.uint8))
        order.append((str(p), not odd))
    return d, order


def test_a_batch_of_the_right_frames_survives_the_loader():
    """The real mechanism, end to end, with the real loader.

    Handed the list the plan was built from (dominant size only), the first
    batch comes back full. Handed the unfiltered list the GUI used to write --
    the same plan indices now land on the odd-sized files and the batch is
    emptied by the resolution filter. That second case is what both users hit.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "astro_clean_v5", REPO / "astro_clean_v5.py")
    worker = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(worker)
    except SystemExit:
        pass

    d, order = _mixed_size_folder()
    everything = [p for p, _ in order]
    matching = [p for p, ok in order if ok]      # what the plan counts: 14 frames

    # Correct: manifest == the planned list.
    sliced, cs, ce = worker.load_with_neighbors(
        d, 0, 12, expected_width=120, expected_height=80, ordered_files=matching)
    assert len(sliced[cs:ce]) == 12, (
        "a batch of 12 dominant-size frames must survive the loader, got "
        f"{len(sliced[cs:ce])}")

    # The bug: manifest still holds the skipped files, plan indices unchanged.
    sliced, cs, ce = worker.load_with_neighbors(
        d, 0, 12, expected_width=120, expected_height=80, ordered_files=everything)
    assert len(sliced[cs:ce]) < 12, (
        "this is the 2026-08-11 failure and the test is no longer reproducing it")


def test_the_engine_says_why_a_batch_came_up_empty():
    """The message a user sees must name the cause, not just the count."""
    worker = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    i = worker.find("if n < 3:")
    assert i > 0, "the frame-count guard vanished"
    body = worker[i:i + 1200]
    assert "last_dropped" in body, \
        "the guard must know how many frames the resolution filter removed"
    assert "different size" in body, \
        "when the size filter emptied the batch, the message must say so"
