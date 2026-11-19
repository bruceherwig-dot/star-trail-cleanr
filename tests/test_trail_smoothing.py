"""Trail Smoothing: every frame is softened before stacking so a dotted trail
becomes one continuous soft line.

Replaced Trail Thickness on the Star Trail tab (2026-11-19). These lock the
effect on synthetic frames, that the ground stays sharp, that 16-bit stays 16-bit,
that it works with Fading Ends and Comet, that Off changes nothing, and the tab's
new controls (Trail Smoothing row; comet lengths Short / Medium / Long, Long first).
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
    spec = importlib.util.spec_from_file_location("msc_smooth", REPO / "make_share_clip.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _star_frames(td, n=30, dtype=np.uint8, step=6, ground=False):
    """A star moving `step` px per frame along row 40, saved as PNGs. Optionally a
    bright static 'ground' block in the bottom rows, the same in every frame."""
    import cv2
    top = 255 if dtype == np.uint8 else 65535
    names = []
    for i in range(n):
        img = np.zeros((80, 220, 3), dtype)
        img[40, 10 + step * i] = (top, top, top)
        if ground:
            img[65:80, :] = (top // 3, top // 3, top // 3)
            img[70, 30 + 2 * i] = (top, top, top)      # a sharp detail in the ground
        nm = f"IMG_{i:04d}.png"
        cv2.imwrite(os.path.join(td, nm), img)
        names.append(nm)
    return names


def test_a_dotted_trail_becomes_one_continuous_line():
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        names = _star_frames(td)
        plain = m._stack_fullres(td, names, "t")
        soft = m._smooth_stack_fullres(td, names, 3)
    path = slice(10 + 6 * 3, 10 + 6 * 26)          # inside the trail, away from its ends
    assert plain[40, path, 0].min() == 0, "the plain trail has gaps between its dots"
    assert soft[40, path, 0].min() > 30, "the smoothed trail has no gaps"


def test_bigger_numbers_are_softer_and_wider():
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        names = _star_frames(td)
        s1 = m._smooth_stack_fullres(td, names, 1)
        s5 = m._smooth_stack_fullres(td, names, 5)
    col = 10 + 6 * 15
    wide = lambda a: int((a[:, col, 0] > 15).sum())
    assert wide(s5) > wide(s1), "radius 5 spreads over more rows than radius 1"


def test_the_input_frame_is_never_changed():
    m = _msc()
    im = np.zeros((40, 60, 3), np.uint8)
    im[20, 30] = (255, 255, 255)
    before = im.copy()
    m.smooth_frame(im, 3)
    assert (im == before).all()


def test_the_ground_stays_sharp_with_a_mask():
    m = _msc()
    mask = np.zeros((80, 220), np.uint8)
    mask[60:, :] = 255                              # white = ground
    with tempfile.TemporaryDirectory() as td:
        names = _star_frames(td, ground=True)
        plain = m._stack_fullres(td, names, "t")
        soft = m._smooth_stack_fullres(td, names, 3, fg=mask)
        unmasked = m._smooth_stack_fullres(td, names, 3)
    g = mask > 127
    assert (soft[g] == plain[g]).all(), "the ground is exactly the normal stack"
    assert not (unmasked[g] == plain[g]).all(), "without a mask the ground is smoothed too"
    assert soft[:60].max() > 0, "the sky is still smoothed"


def test_sixteen_bit_frames_give_a_sixteen_bit_trail():
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        names = _star_frames(td, dtype=np.uint16)
        soft = m._smooth_stack_fullres(td, names, 3)
    assert soft.dtype == np.uint16
    assert soft.max() > 255 * 100, "values use the 16-bit range, not 8-bit"


def test_it_works_with_fading_ends_and_with_comet():
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        names = _star_frames(td)
        flat = m._smooth_stack_fullres(td, names, 3)
        fade = m._smooth_stack_fullres(td, names, 3, taper=0.2)
        comet = m._smooth_stack_fullres(td, names, 3, comet_tail_frames=10)
    first, mid, last = 10 + 6 * 1, 10 + 6 * 15, 10 + 6 * 28
    assert fade[40, first, 0] < flat[40, first, 0], "Fading Ends dims the start"
    assert fade[40, last, 0] < flat[40, last, 0], "and the end"
    assert abs(int(fade[40, mid, 0]) - int(flat[40, mid, 0])) <= 1, "the middle is untouched"
    assert comet[40, first, 0] < comet[40, last, 0], "a comet tail fades toward the old end"


def test_smoothing_off_changes_nothing():
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        _star_frames(td, n=30)
        with tempfile.TemporaryDirectory() as od:      # outputs must not sit among the frames
            a = os.path.join(od, "a.tif"); b = os.path.join(od, "b.tif")
            m.make_star_trail(td, a, out_format="tif8")
            m.make_star_trail(td, b, out_format="tif8", smooth_px=0)
            assert open(a, "rb").read() == open(b, "rb").read()


def test_the_whole_command_runs_end_to_end():
    m = _msc()
    import cv2
    with tempfile.TemporaryDirectory() as td:
        _star_frames(td, n=30)
        with tempfile.TemporaryDirectory() as od:
            out = os.path.join(od, "smooth.tif")
            m.make_star_trail(td, out, out_format="tif8", smooth_px=3)
            im = cv2.imread(out)
    assert im is not None and im[40, 10 + 6 * 10:10 + 6 * 20, 0].min() > 20


def test_the_timers_record_where_the_time_goes():
    from modules import step_timer
    m = _msc()
    step_timer.reset()
    m.smooth_frame(np.zeros((40, 60, 3), np.uint8), 3)
    names = set(step_timer.snapshot()["steps"])
    assert {"smooth.blur", "smooth.gamma_up", "smooth.gamma_down", "smooth.boost"} <= names


def test_the_tab_has_the_smoothing_row_and_three_comet_lengths():
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
    assert [p._size_cb.itemText(i) for i in range(p._size_cb.count())] == \
        ["Off", "1", "2", "3", "4", "5"]
    assert p._size_cb.currentData() == 0, "smoothing opens Off"
    assert [p._len_cb.itemText(i) for i in range(p._len_cb.count())] == \
        ["Short", "Medium", "Long"], "one Length choice, no percentages"
    p._mode_comet.setChecked(True)
    assert p._comet_fraction() == 2.0, "Comet opens at Long (twice the sequence)"
    assert p._reverse_chk.isVisible() or not p.isVisible()
    p._len_cb.setCurrentIndex(0)
    assert p._comet_fraction() == 0.5
    p._mode_taper.setChecked(True)
    assert p._taper_fraction() == 0.2, "picking Fading Ends resets Length to Medium"
    assert not p._reverse_chk.isEnabled(), "Reverse order is for Comet only"
    p._len_cb.setCurrentIndex(2)
    assert p._taper_fraction() == 0.3
    p._mode_comet.setChecked(True)
    assert p._len_cb.currentData() == 2, "picking Comet resets Length to Long"
    p._size_cb.setCurrentIndex(p._size_cb.findData(3))
    assert "smooth3" in os.path.basename(p._build_out_path())
    p._mode_comet.setChecked(True)
    assert "comet-long" in os.path.basename(p._build_out_path())


def test_the_gui_passes_smooth_not_thicken():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert '"--smooth", str(smooth)' in src
    assert '"--thicken", str(thick)' not in src


def test_the_real_command_finds_the_ground_mask_in_the_workspace():
    """make_star_trail must pick up STC Extras/foreground_mask.png itself (a local
    import further down the function once made the name unusable here, silently,
    and the whole frame got smoothed)."""
    import cv2
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        _star_frames(td, n=30, ground=True)
        os.makedirs(os.path.join(td, "STC Extras"))
        mask = np.zeros((80, 220), np.uint8)
        mask[60:, :] = 255
        cv2.imwrite(os.path.join(td, "STC Extras", "foreground_mask.png"), mask)
        with tempfile.TemporaryDirectory() as od:
            plain = os.path.join(od, "p.tif"); soft = os.path.join(od, "s.tif")
            m.make_star_trail(td, plain, out_format="tif8")
            m.make_star_trail(td, soft, out_format="tif8", smooth_px=3)
            p, s = cv2.imread(plain), cv2.imread(soft)
    assert (p[60:] == s[60:]).all(), "the ground came out exactly as the normal stack"
