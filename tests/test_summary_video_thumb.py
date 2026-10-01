"""The Summary tab shows the before-and-after clip as a picture with a play symbol.

It must work when the clip is readable, fall back to a plain box when it is not,
and show nothing at all when there is no clip.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2
import numpy as np


def _make_clip(path, n=20, w=200, h=160):
    """A tiny clip whose frames change, so a frame picked from the middle is not
    the same as the first one."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), i * 12, np.uint8))
    vw.release()


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_poster_comes_from_the_middle_not_the_first_frame():
    _app()
    import star_trail_cleanr as S
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "c.mp4"
        _make_clip(clip)
        img = S._video_poster(str(clip), 0.4)
    assert img is not None and not img.isNull()
    assert (img.width(), img.height()) == (200, 160)
    # frame 8 of 20 is far brighter than frame 0 (which is black)
    assert img.pixelColor(100, 80).red() > 40


def test_an_unreadable_or_missing_clip_gives_no_poster_but_still_a_box():
    _app()
    import star_trail_cleanr as S
    with tempfile.TemporaryDirectory() as d:
        junk = Path(d) / "junk.mp4"
        junk.write_bytes(b"not a video")
        assert S._video_poster(str(junk)) is None
        assert S._video_poster(str(Path(d) / "missing.mp4")) is None
        t = S._VideoThumb(str(junk), width_px=160, max_h=128)
    assert (t.width(), t.height()) == (160, 128), "a plain 5:4 box, never zero-sized"


def test_the_thumbnail_is_capped_in_height_and_clicks():
    _app()
    import star_trail_cleanr as S
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "tall.mp4"
        _make_clip(clip, w=120, h=200)             # a portrait clip
        t = S._VideoThumb(str(clip), width_px=160, max_h=128)
        assert t.height() <= 128
        hits = []
        t.clicked.connect(lambda: hits.append(1))
        QTest.mouseClick(t, Qt.LeftButton, Qt.NoModifier, QPoint(t.width() // 2, t.height() // 2))
    assert hits == [1]


def test_the_summary_shows_the_thumbnail_only_when_the_clip_exists():
    _app()
    import star_trail_cleanr as S
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "c.mp4"
        _make_clip(clip)
        with_clip = S.SummaryPanel("<b>done</b>", None, str(clip))
        without = S.SummaryPanel("<b>done</b>", None, str(Path(d) / "nope.mp4"))
        assert len(with_clip.findChildren(S._VideoThumb)) == 1
        assert len(without.findChildren(S._VideoThumb)) == 0


def test_the_window_opens_tall_enough_for_its_tallest_tab():
    """Wrapped text can squash into fewer lines than it wants, so a window sized
    from minimums alone can open too short for the Summary tab and clip its text.
    CreatorWindow._fit_height holds the window to what the tabs really need."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QRect
    _app()
    import star_trail_cleanr as S

    class _Screen:                       # a tall screen, so the 90% cap is not in play
        def availableGeometry(self):
            return QRect(0, 0, 1728, 1600)
    real = QApplication.primaryScreen
    QApplication.primaryScreen = staticmethod(lambda: _Screen())
    try:
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "STC_share_video.mp4"
            _make_clip(clip)
            cleaned = Path(d) / "cleaned"
            (cleaned / "STC Extras").mkdir(parents=True)
            (cleaned / "STC Extras" / "STC_share_video.mp4").write_bytes(clip.read_bytes())
            w = S.CreatorWindow(str(cleaned), "", summary_html="<p>line of summary text</p>" * 18)
            w.resize(586, 500)
            w.show()
            for _ in range(5):
                QApplication.instance().processEvents()
            sp = w._summary_panel
            assert sp.height() >= sp.sizeHint().height(), (
                f"window left the Summary tab {sp.height()} px for a {sp.sizeHint().height()} px need")
            w.close()
    finally:
        QApplication.primaryScreen = real


_SAVED = ("Swept <b>7,969</b> trails<br>across <b>572</b> frames.<br><br>"
          "<span style='font-size:20px; font-weight:bold;'>TIME SAVED: ~66 hours</span><br><br>"
          "<span style='font-size:15px;'>Cleaned using your graphics.</span><br><br>"
          "<b>Ready for the fun part!</b><br>Create your star trail.<br><br>"
          "<span style='font-size:14px; color:#666;'>Thought it'd take <b>58m 03s</b>. "
          "Took <b>1h 14m 53s</b>. My apologies.</span>")


def _words(html):
    import re
    return sorted(re.sub(r"<[^>]+>|&nbsp;", " ", html).split())


def test_the_time_line_moves_above_ready_for_the_fun_part_and_nothing_else_changes():
    _app()
    import star_trail_cleanr as S
    out = S._tidy_summary_html(_SAVED)
    assert out.index("Thought it'd take") < out.index("Ready for the fun part!")
    assert out.index("TIME SAVED") < out.index("Thought it'd take")
    assert _words(out) == _words(_SAVED), "same words, only moved"
    assert "<br><br>" not in out, "full blank lines become the shorter gap"
    assert S._tidy_summary_html(out) == out, "tidying twice changes nothing"


def test_text_without_a_time_line_is_left_alone_apart_from_the_gaps():
    _app()
    import star_trail_cleanr as S
    plain = "Sky was clean.<br><br><b>Ready for the fun part!</b><br>Go."
    out = S._tidy_summary_html(plain)
    assert _words(out) == _words(plain)
    assert out.index("Sky was clean") < out.index("Ready")
    assert S._tidy_summary_html("") == ""


def test_the_window_remeasures_when_its_width_changes():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QRect
    import time
    _app()
    import star_trail_cleanr as S

    class _Screen:
        def availableGeometry(self):
            return QRect(0, 0, 1728, 1600)
    real = QApplication.primaryScreen
    QApplication.primaryScreen = staticmethod(lambda: _Screen())
    try:
        with tempfile.TemporaryDirectory() as d:
            w = S.CreatorWindow(d, "", summary_html=_SAVED * 4)
            w.show()

            def settle():
                for _ in range(12):
                    QApplication.instance().processEvents()
                    time.sleep(0.03)
            settle()
            for width in (480, 700, 520):
                w.resize(width, 400)
                settle()
                sp = w._summary_panel
                need = sp.layout().totalHeightForWidth(sp.width())
                assert sp.height() >= need, (
                    f"at {width} px wide the Summary tab got {sp.height()} px but needs {need}")
            w.close()
    finally:
        QApplication.primaryScreen = real


def test_the_picture_says_what_it_is_and_how_to_watch_it():
    """The words beside the before-and-after picture: what it is on the left, how to
    watch it and where it lives on the right. The folder link must still work."""
    _app()
    import star_trail_cleanr as S
    from PySide6.QtWidgets import QLabel
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "c.mp4"
        _make_clip(clip)
        sp = S.SummaryPanel("<b>done</b>", None, str(clip))
        texts = [lbl.text() for lbl in sp.findChildren(QLabel)]
    assert any("before and after clip" in t for t in texts), "the picture must say what it is"
    assert any("Click thumbnail to watch" in t and "open folder" in t for t in texts)
    assert sp._video_lbl is not None and "href='folder'" in sp._video_lbl.text()
