"""Both creator tabs open at their defaults, every time.

WHY (Bruce, 2026-08-30). The two tabs behaved differently for no reason a user
could see. The Timelapse tab remembered every choice between sessions; the Star
Trail tab remembered only its Source and reset everything else. So setting 30
frames a second once stuck forever, while picking Comet Mode and a thickness you
liked had to be redone every single time.

He chose to make them match by having BOTH reset rather than both remember. A
window that always opens the same way is predictable, and a remembered setting
you have forgotten choosing is exactly the kind that quietly produces a file you
did not expect.

The saved values are left on disk rather than deleted -- they are simply never
read again -- so reversing this decision restores the user's old choices intact.

These tests plant non-default values for everything and assert the windows ignore
them, which is the only way to tell "opens at the default" apart from "happens to
have the default saved".
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# What each control must show when a window opens, whatever is on disk.
TIMELAPSE_DEFAULTS = {"source": "cleaned", "style": "plain", "size": "4k",
                      "fps": 24, "blend": 3, "format": "mp4"}


def _frames():
    """A folder pair with enough frames for both panels to build against."""
    import cv2
    d = Path(tempfile.mkdtemp())
    cleaned = d / "cleaned"
    cleaned.mkdir()
    for i in range(10):
        img = np.full((40, 60, 3), 30, np.uint8)
        cv2.imwrite(str(d / f"IMG_{i}.jpg"), img)
        cv2.imwrite(str(cleaned / f"IMG_{i}.jpg"), img)
    return str(cleaned), str(d)


def _with_planted_settings(fn):
    """Run fn() with every remembered key set to something that is NOT the
    default, then put the user's real settings back exactly as they were."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import star_trail_cleanr as S

    planted = {"timelapse_source": "original", "timelapse_style": "accumulate",
               "timelapse_size": "1080p", "timelapse_fps": 60,
               "timelapse_blend": 5, "timelapse_format": "mov",
               "startrail_source": "original"}
    saved = {k: S.SETTINGS.value(k) for k in planted}
    try:
        for k, v in planted.items():
            S.SETTINGS.setValue(k, v)
        return fn(S)
    finally:
        # Never leave test values in the real settings file.
        for k, v in saved.items():
            S.SETTINGS.remove(k) if v is None else S.SETTINGS.setValue(k, v)
        S.SETTINGS.sync()


def test_the_timelapse_tab_ignores_what_was_saved():
    cleaned, original = _frames()

    def check(S):
        t = S.TimelapsePanel(cleaned, original_folder=original)
        return {"source": t._source_cb.currentData(), "style": t._style_cb.currentData(),
                "size": t._size_cb.currentData(), "fps": t._fps_cb.currentData(),
                "blend": t._blend_cb.currentData(), "format": t._fmt_cb.currentData()}

    got = _with_planted_settings(check)
    assert got == TIMELAPSE_DEFAULTS, (
        f"the tab came up on remembered values instead of its defaults: {got}")


def test_the_star_trail_tab_ignores_what_was_saved():
    cleaned, original = _frames()

    def check(S):
        p = S.StarTrailPanel(cleaned, "", original_folder=original)
        return p._src_cb.currentData()

    assert _with_planted_settings(check) == "cleaned", (
        "Source came up on the remembered choice; it should open on Cleaned")


def test_the_star_trail_tabs_other_choices_are_at_their_defaults():
    """These were never remembered, and must stay that way now that Source has
    joined them -- the point is that the whole window is predictable."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import star_trail_cleanr as S
    cleaned, original = _frames()
    p = S.StarTrailPanel(cleaned, "", original_folder=original)
    assert p._mode_normal.isChecked(), "should open on Normal (Lighten), not Comet"
    assert p._size_cb.currentData() == 0, "trail thickness should open at Normal"
    assert not p._hotpix_chk.isChecked(), "speck removal should open unticked"
    assert not p._reverse_chk.isChecked(), "reverse order should open unticked"
    assert float(p._comet_len_cb.currentData()) == 1.0, "trail length should open at Long"


def test_nothing_writes_the_remembered_keys_any_more():
    """A stray setValue left anywhere would re-create the split behaviour for
    that one control, which is worse than either tab's old behaviour."""
    gui = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in gui.splitlines() if not l.strip().startswith("#"))
    for key in ("timelapse_size", "timelapse_fps", "timelapse_blend",
                "timelapse_style", "timelapse_format", "timelapse_source",
                "startrail_source"):
        assert f'setValue("{key}"' not in code, (
            f"{key} is being saved again; that tab will start opening on a "
            f"remembered value while the other one does not")
