"""Paging between star trails must not shrink the list you are paging through.

WHAT BRUCE SAW (2026-08-30): "I went to look at both photos then the left right
disappeared?" A run produced two trails -- the cleaned one and the matching
original-source one. The window opened on "STC_cleaned_star_trail.tif · 1 of 2",
he pressed an arrow, and on the next refresh the counter and both arrows vanished,
leaving him stranded on the original trail with no way back to the cleaned one.

THE CAUSE. The list of trails was assembled from a glob for on-demand builds plus
two names -- and the cleaned trail was not one of the names. It was found only
through the `baseline_path` argument, which the panel overwrites with whatever is
currently on screen as the user pages. So stepping onto the original trail left
the cleaned one listed by nothing at all: one file found, arrows hidden, stuck.

It survived a long time because the cleaned trail was almost always the one being
displayed, so the pointer happened to be right. Two run-made trails plus one arrow
press is the condition that breaks it, and that is what a normal run now produces.

TWO THINGS WERE WRONG AND BOTH ARE FIXED. Every run-made name is now listed
explicitly, and the moving on-screen pointer no longer doubles as the fixed
baseline. Either fix alone would have hidden the symptom; only the second stops it
returning by another route.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _workspace(names_with_ages):
    """A cleaned folder whose workspace holds the given trails, at set ages so
    the newest-first ordering is deterministic."""
    import cv2
    d = Path(tempfile.mkdtemp())
    cleaned = d / "cleaned"
    ws = cleaned / "STC Extras"
    ws.mkdir(parents=True)
    for name, age in names_with_ages:
        p = ws / name
        cv2.imwrite(str(p), np.full((40, 60, 3), 100, np.uint8))
        t = time.time() - age
        os.utime(p, (t, t))
    return str(cleaned), str(ws)


def _panel(cleaned, baseline):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import star_trail_cleanr as S
    return S.StarTrailPanel(cleaned, baseline)


def test_the_list_does_not_shrink_as_you_page():
    """The reported bug, end to end through the real panel."""
    cleaned, ws = _workspace([("STC_original_star_trail.tif", 20),
                              ("STC_cleaned_star_trail.tif", 10)])
    p = _panel(cleaned, os.path.join(ws, "STC_cleaned_star_trail.tif"))
    seen = []
    for _ in range(4):
        seen.append(len(p._builds))
        p._step_build(+1)
    assert set(seen) == {2}, f"the list shrank while paging: {seen}"


def test_the_arrows_stay_up():
    cleaned, ws = _workspace([("STC_original_star_trail.tif", 20),
                              ("STC_cleaned_star_trail.tif", 10)])
    p = _panel(cleaned, os.path.join(ws, "STC_cleaned_star_trail.tif"))
    p._step_build(+1)
    p._step_build(+1)
    assert not p._prev_btn.isHidden() and not p._next_btn.isHidden(), (
        "the arrows disappeared after paging, stranding the user on one image")


def test_you_can_get_back_to_the_cleaned_trail():
    """Being stuck on the original trail was the actual harm."""
    cleaned, ws = _workspace([("STC_original_star_trail.tif", 20),
                              ("STC_cleaned_star_trail.tif", 10)])
    p = _panel(cleaned, os.path.join(ws, "STC_cleaned_star_trail.tif"))
    p._step_build(+1)
    p._step_build(+1)
    p._step_build(-1)
    assert os.path.basename(p._star_path) == "STC_cleaned_star_trail.tif", (
        f"could not get back; landed on {os.path.basename(p._star_path)}")


def test_the_counter_keeps_counting():
    cleaned, ws = _workspace([("STC_original_star_trail.tif", 20),
                              ("STC_cleaned_star_trail.tif", 10)])
    p = _panel(cleaned, os.path.join(ws, "STC_cleaned_star_trail.tif"))
    assert "1 of 2" in p._name_lbl.text()
    p._step_build(+1)
    assert "2 of 2" in p._name_lbl.text()
    p._step_build(+1)
    assert "2 of 2" in p._name_lbl.text(), (
        f"the counter vanished on a repeat click: {p._name_lbl.text()!r}")


# ── the two underlying rules ───────────────────────────────────────────────

def test_both_run_made_trails_are_listed_by_name():
    """Neither matches the Build glob, so each must be asked for explicitly.
    Passing no baseline at all is the sharpest version of the test: the list must
    still find both."""
    import star_trail_cleanr as S
    cleaned, ws = _workspace([("STC_original_star_trail.tif", 20),
                              ("STC_cleaned_star_trail.tif", 10)])
    got = {os.path.basename(x) for x in S._star_trail_builds(None, cleaned)}
    assert got == {"STC_cleaned_star_trail.tif", "STC_original_star_trail.tif"}, got


def test_either_format_is_found():
    import star_trail_cleanr as S
    cleaned, ws = _workspace([("STC_original_star_trail.jpg", 20),
                              ("STC_cleaned_star_trail.tif", 10)])
    got = {os.path.basename(x) for x in S._star_trail_builds(None, cleaned)}
    assert got == {"STC_cleaned_star_trail.tif", "STC_original_star_trail.jpg"}, got


def test_the_screen_pointer_is_not_the_baseline():
    """The real defect: one pointer doing two jobs. Guard it by name, because the
    explicit listing alone would hide a re-introduction rather than fail on it."""
    gui = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    body = gui[gui.index("def _refresh_builds("):]
    body = body[:body.index("\n    def ", 10)]
    assert "_star_trail_builds(self._baseline_star" in body, (
        "the build list is being taken from the on-screen path again; it moves "
        "as the user pages, so the list shrinks under them")


def test_on_demand_builds_still_appear():
    """The glob's own job must survive the change."""
    import star_trail_cleanr as S
    cleaned, ws = _workspace([("STC_cleaned_star_trail.tif", 30),
                              ("STC_star_trail_plain_cleaned.jpg", 20),
                              ("STC_star_trail_comet-long_cleaned_16bit.tif", 10)])
    got = [os.path.basename(x) for x in S._star_trail_builds(None, cleaned)]
    assert len(got) == 3, got
    assert got[0] == "STC_star_trail_comet-long_cleaned_16bit.tif", (
        f"newest should sort first: {got}")
