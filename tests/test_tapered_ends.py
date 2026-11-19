"""Tapered Ends star trail style: trails fade in at the start and out at the end.

The middle of the sequence stays at full brightness, and the size of the fade is
a share of the frames at each end, so it scales from 100 frames to 1000. These
lock the weights, the stack itself on synthetic frames, and the tab's defaults.
Offline.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _msc():
    spec = importlib.util.spec_from_file_location("msc_taper", REPO / "make_share_clip.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ramp_scales_with_the_sequence_and_has_a_floor():
    m = _msc()
    assert m.taper_ramp_frames(100, 0.2) == 20
    assert m.taper_ramp_frames(760, 0.2) == 152          # about the 150 of the hand method
    assert m.taper_ramp_frames(100, 0.1) == 10
    assert m.taper_ramp_frames(15, 0.1) == 3, "each end tapers at least 3 frames"
    assert m.taper_ramp_frames(4, 0.3) == 2, "the two ends may never overlap"


def test_weights_climb_to_full_and_stay_full_in_the_middle():
    m = _msc()
    n, ramp = 100, 20
    w = [m.taper_weight(i, n, ramp) for i in range(n)]
    assert w[0] < 0.01 and w[-1] < 0.01, "first and last frame are nearly invisible"
    assert all(abs(x - 1.0) < 1e-9 for x in w[ramp:n - ramp]), "the middle is untouched"
    assert all(w[i] < w[i + 1] for i in range(ramp)), "fade-in rises every frame"
    assert all(w[n - 1 - i] == w[i] for i in range(n)), "both ends are mirror images"


def test_stack_fades_the_trail_ends_but_not_the_middle():
    import cv2
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        names = []
        for i in range(30):
            img = np.zeros((20, 80, 3), np.uint8)
            img[10, 5 + 2 * i] = (200, 200, 200)        # a star moving 2 px per frame
            nm = f"IMG_{i:04d}.png"
            cv2.imwrite(os.path.join(td, nm), img)
            names.append(nm)
        out = m._taper_stack_fullres(td, names, 0.2)    # 6 frames fade at each end
    row = out[10, :, 0].astype(int)
    # 7.5 light stops down is about a tenth of the stored value (200 -> ~19)
    assert row[5] < 30, "the first position is mostly gone"
    assert row[5 + 2 * 29] < 30, "the last position is mostly gone"
    assert abs(row[5] - 200 * 2.0 ** (-7.5 / 2.2)) <= 2, "fade is in light stops, not raw values"
    assert row[5 + 2 * 15] == 200, "the middle of the trail is full brightness"
    assert row[5 + 2 * 3] < row[5 + 2 * 5] < 200, "the fade climbs toward full"


def test_the_tab_opens_on_normal_and_offers_the_twenty_percent_default():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import cv2
    import star_trail_cleanr as S
    d = Path(tempfile.mkdtemp())
    cleaned = d / "cleaned"
    cleaned.mkdir()
    for i in range(30):
        img = np.full((40, 60, 3), 30, np.uint8)
        cv2.imwrite(str(d / f"IMG_{i}.jpg"), img)
        cv2.imwrite(str(cleaned / f"IMG_{i}.jpg"), img)
    p = S.StarTrailPanel(str(cleaned), "", original_folder=str(d))
    assert p._mode_normal.isChecked(), "must still open on Normal"
    assert float(p._taper_cb.currentData()) == 0.2, "taper length opens at Medium (20%)"
    assert not p._taper_cb.isEnabled(), "taper length is grayed until Tapered Ends is picked"
    p._mode_taper.setChecked(True)
    assert p._taper_cb.isEnabled() and not p._comet_len_cb.isEnabled()
    assert "taper-20" in os.path.basename(p._build_out_path())


def test_style_options_sit_under_the_style_and_the_tab_never_changes_height():
    """The options of the picked style show in ONE row under the style radios, and
    picking a different style never changes how tall the tab needs to be. A tab
    that grew past 90% of a laptop screen squashed and drew its caption over the
    picture (2026-11-10)."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import cv2
    import star_trail_cleanr as S
    d = Path(tempfile.mkdtemp())
    cleaned = d / "cleaned"
    cleaned.mkdir()
    for i in range(30):
        img = np.full((40, 60, 3), 30, np.uint8)
        cv2.imwrite(str(d / f"IMG_{i}.jpg"), img)
        cv2.imwrite(str(cleaned / f"IMG_{i}.jpg"), img)
    p = S.StarTrailPanel(str(cleaned), "", original_folder=str(d))
    heights = {}
    for btn, page in ((p._mode_normal, 0), (p._mode_comet, 1), (p._mode_taper, 2)):
        btn.setChecked(True)
        assert p._style_opts.currentIndex() == page, "options row must follow the style"
        heights[page] = p.layout().totalHeightForWidth(560)
    assert len(set(heights.values())) == 1, f"height changed with the style: {heights}"
    assert heights[0] <= 760, f"tab needs {heights[0]} px; it must stay compact"
