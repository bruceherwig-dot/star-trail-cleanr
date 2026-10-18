"""The Summary / Star Trail / Timelapse window keeps its tabs along the TOP, and its red
close box sits just below the bar in the right-hand corner with a visible margin
(Bruce, 2026-10-18: he tried the tabs at the bottom, then put them back, and asked for
some margin on the right of the red box).

This reads the source instead of building a window. A first version built real windows,
and on the Apple-silicon Mac and Windows build machines the test run then aborted at
interpreter exit ("QObject: shared QObject was deleted directly", v2.99 build 1) even
though every test passed.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _creator_window_source():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    start = src.index("class CreatorWindow(QDialog):")
    end = src.index("class MaskEditorWindow(", start)
    return src[start:end]


def test_the_tab_bar_stays_at_the_top():
    body = _creator_window_source()
    assert "setTabPosition" not in body, "the tabs belong along the top of the window"


def test_the_red_x_sits_below_the_bar_with_a_margin():
    body = _creator_window_source()
    pos = body[body.index("def _position_close_x"):body.index("def resizeEvent")]
    m = re.search(r"\bm = (\d+)\b", pos)
    assert m and 10 <= int(m.group(1)) <= 20, "the red X needs a visible right margin"
    assert "self._tabs.tabBar().height()" in pos, "the red X must sit below the tab bar"


def test_the_window_is_wide_enough_to_keep_the_red_x_clear_of_the_preview():
    """The 420 px preview is centred, so it ends (width - 420) / 2 px from the right edge.
    The red X needs its width (32) plus its margin to fit in that gap at the narrowest
    window."""
    body = _creator_window_source()
    min_w = int(re.search(r"setMinimumWidth\((\d+)\)", body).group(1))
    pos = body[body.index("def _position_close_x"):body.index("def resizeEvent")]
    margin = int(re.search(r"\bm = (\d+)\b", pos).group(1))
    gap = (min_w - 420) // 2
    assert gap >= 32 + margin, "at the minimum width the red X would touch the preview (%d px gap)" % gap
