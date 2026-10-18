"""The Summary / Star Trail / Timelapse tabs sit along the TOP of the window, and the red
close box sits just below the bar in the right-hand corner with a visible margin
(Bruce, 2026-10-18: he tried the tabs at the bottom and put them back)."""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_tab_bar_is_at_the_top_and_the_red_x_sits_below_it_with_a_margin():
    _app()
    import star_trail_cleanr as S
    from PySide6.QtWidgets import QApplication, QTabWidget
    with tempfile.TemporaryDirectory() as d:
        w = S.CreatorWindow(d, "", summary_html="<p>line</p>")
        w.resize(560, 800)
        w.show()
        for _ in range(8):
            QApplication.instance().processEvents()
        assert w._tabs.tabPosition() == QTabWidget.North
        bar = w._tabs.tabBar()
        assert bar.y() <= 2, "the tab bar must run along the top of the window"
        assert w._close_x.y() >= bar.height(), "the red X must sit below the bar, not over the tab names"
        margin = w.width() - (w._close_x.x() + w._close_x.width())
        assert 10 <= margin <= 20, "in the right-hand corner, with a visible margin (%d px)" % margin
        w.close()


def test_the_red_x_never_touches_the_preview_even_at_the_narrowest_window():
    _app()
    import star_trail_cleanr as S
    from PySide6.QtWidgets import QApplication
    with tempfile.TemporaryDirectory() as d:
        w = S.CreatorWindow(d, "", summary_html="<p>line</p>")
        w.resize(200, 800)                       # asks for far less than the window allows
        w.show()
        for _ in range(8):
            QApplication.instance().processEvents()
        assert w.width() >= 520
        w.select_tab("star")
        for _ in range(4):
            QApplication.instance().processEvents()
        x_left = w._close_x.x()
        from PySide6.QtWidgets import QLabel
        pics = [c for c in w._star_panel.findChildren(QLabel) if c.isVisible() and c.width() >= 400]
        assert pics, "the preview box must be found"
        for c in pics:
            assert c.mapTo(w, c.rect().topRight()).x() < x_left, "the red X must clear the preview box"
        w.close()
