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


def _star_frames(td, n=30, dtype=np.uint8, step=6, ground=False, width=220):
    """A star moving `step` px per frame along row 40, saved as PNGs. Optionally a
    bright static 'ground' block in the bottom rows, the same in every frame."""
    import cv2
    top = 255 if dtype == np.uint8 else 65535
    names = []
    for i in range(n):
        img = np.zeros((80, width, 3), dtype)
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
        _star_frames(td, n=30, width=5472)       # the reference width: radius used as given
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
    assert {"smooth.bands"} <= names, "the smoothing is timed as one wall-clock step"
    # the stack loop records the wait for each frame to be read and the averaging
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        step_timer.reset()
        m._smooth_stack_fullres(td, _star_frames(td, n=6), 2)
    names = set(step_timer.snapshot()["steps"])
    assert {"smooth.read_wait", "smooth.average_stack"} <= names


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
        ["Off", "Light", "Medium", "Strong"], "words, not numbers"
    assert [p._size_cb.itemData(i) for i in range(p._size_cb.count())] == [0, 2, 3, 5]
    assert p._size_cb.currentData() == 0, "smoothing opens Off"
    assert p._smooth_note.text() == "Will take longer to render", "the slowness note beside the dropdown"
    assert p._smooth_note.parentWidget() is p._size_cb.parentWidget(), "on the same row as the dropdown"
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
    p._size_cb.setCurrentIndex(p._size_cb.findData(0))
    p._mode_taper.setChecked(True)
    assert p._size_cb.currentData() == 3, "picking Fading Ends sets Smoothing to Medium"
    assert p._taper_fraction() == 0.2, "and Length to Medium"
    p._size_cb.setCurrentIndex(0)
    p._mode_normal.setChecked(True)
    assert p._size_cb.currentData() == 0, "another style leaves Smoothing as it is"
    p._size_cb.setCurrentIndex(p._size_cb.findData(5))
    p._mode_comet.setChecked(True)
    assert p._size_cb.currentData() == 5, "Comet leaves Smoothing as it is"
    p._size_cb.setCurrentIndex(0)
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


def test_the_radius_scales_with_the_pictures_width():
    m = _msc()
    assert m.smooth_radius_for_width(3, 5472) == 3.0, "the reference width is unchanged"
    assert abs(m.smooth_radius_for_width(3, 8256) - 3 * 8256 / 5472) < 1e-9
    assert m.smooth_radius_for_width(3, 2736) == 1.5, "a narrower frame gets a smaller blur"
    assert m.smooth_radius_for_width(3, 5472, reference=None) == 3.0, "no reference: as given"


def test_a_fractional_radius_gives_a_proper_disc_and_whole_ones_are_unchanged():
    m = _msc()
    k = m._smooth_kernel(4.5)
    assert abs(float(k.sum()) - 1.0) < 1e-5 and k.shape[0] == 2 * 5 + 3
    k3 = m._smooth_kernel(3)
    assert k3.shape == (9, 9), "a whole radius is the same disc as before"


def test_a_wider_frame_gets_a_proportionally_wider_blur():
    m = _msc()
    def spread(width):
        with tempfile.TemporaryDirectory() as td:
            names = _star_frames(td, n=12, step=width // 40, width=width)
            out = m._smooth_stack_fullres(td, names, 3, reference_width=2000)
        col = 10 + (width // 40) * 5
        return int((out[:, col, 0] > 15).sum())
    # twice the width: the blur grows by the same factor, so the vertical spread grows
    assert spread(4000) > spread(2000)


def test_the_command_scales_by_default_and_reports_it():
    src = (REPO / "make_share_clip.py").read_text(encoding="utf-8")
    assert "reference_width=SMOOTH_REFERENCE_WIDTH" in src


def _reference_smooth(im, radius):
    """The plain, one-piece way: what the speed-up must still give."""
    m = _msc()
    top = 65535.0 if im.dtype == np.uint16 else 255.0
    x = im.astype(np.float32) * np.float32(1.0 / top)
    lin = np.power(x, np.float32(2.2))
    hl = np.clip((x - np.float32(0.70)) / np.float32(0.30), 0, 1)
    lin = lin * (np.float32(1.0) + np.float32(4.0) * hl)
    import cv2
    b = cv2.filter2D(lin, -1, m._smooth_kernel(radius), borderType=cv2.BORDER_REFLECT)
    return np.power(np.clip(b, 0, 1), np.float32(1 / 2.2)) * np.float32(top)


def _noisy(h, w, dtype, seed):
    rng = np.random.default_rng(seed)
    top = 65535 if dtype == np.uint16 else 255
    im = (rng.random((h, w, 3)) ** 6 * top).astype(dtype)      # mostly dark, a few bright
    im[h // 2, w // 2] = top
    return im


def test_the_speed_up_gives_the_same_picture_as_the_plain_way():
    m = _msc()
    for dtype in (np.uint8, np.uint16):
        im = _noisy(300, 500, dtype, 1)
        top = 65535.0 if dtype == np.uint16 else 255.0
        ref = _reference_smooth(im, 3)
        one = m.smooth_frame(im, 3, bands=1, workers=1)
        many = m.smooth_frame(im, 3, bands=8, workers=4)
        tol = top * 2e-5
        assert np.abs(one - ref).max() < tol, "lookup table matches the per-pixel maths"
        assert np.abs(many - one).max() < tol, "bands and threads match the one-piece result"


def test_bands_agree_at_the_seams_and_at_the_edges():
    m = _msc()
    im = np.zeros((200, 120, 3), np.uint8)
    for y in (0, 24, 25, 99, 100, 101, 199):               # bright rows on and around the seams
        im[y, :, :] = 255
    one = m.smooth_frame(im, 4.5, bands=1, workers=1)
    for nb in (2, 5, 8):
        many = m.smooth_frame(im, 4.5, bands=nb, workers=nb)
        assert np.abs(many - one).max() < 255 * 2e-5, f"{nb} bands differ from the whole frame"


def test_thread_count_leaves_cores_free_and_has_a_ceiling():
    m = _msc()
    assert 1 <= m.smooth_threads() <= m.SMOOTH_MAX_THREADS


def test_the_options_explain_what_happens_to_the_ground():
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
    for tip in (p._size_cb.toolTip(), p._mode_comet.toolTip(), p._mode_taper.toolTip()):
        assert "ground" in tip and "normal star trail" in tip
    assert "ground mask" in p._size_cb.toolTip(), "smoothing says it uses the mask where there is one"


# ---- memory safeguard (2026-11-20) ---------------------------------------------------
# A 45 MP 16-bit sequence peaked at about 6.8 GB. The stack now trims its extra copies
# and, when the computer cannot hold the working copies of a huge frame, is built in
# horizontal strips. The picture must be the same either way.

def _frames_for_strips(td, n=8, h=200, w=260, dtype=np.uint16):
    import cv2
    top = 65535 if dtype == np.uint16 else 255
    rng = np.random.default_rng(11)
    names = []
    for i in range(n):
        img = (rng.integers(0, top // 12, (h, w, 3))).astype(dtype)
        img[(17 * i) % (h - 3):(17 * i) % (h - 3) + 3, (31 * i) % (w - 5):(31 * i) % (w - 5) + 5] = top
        nm = f"IMG_{i:04d}.png"
        cv2.imwrite(os.path.join(td, nm), img)
        names.append(nm)
    return names


def test_the_memory_plan_uses_one_piece_when_there_is_room():
    m = _msc()
    gb = 2 ** 30
    assert m.smooth_memory_plan(4000, 6000, 1, False, available=64 * gb)[0] == 1
    assert m.smooth_memory_plan(6000, 7500, 2, True, available=64 * gb)[0] == 1
    assert m.smooth_memory_plan(6000, 7500, 2, True, available=None)[0] in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def test_the_memory_plan_cuts_a_huge_frame_into_strips_when_memory_is_short():
    m = _msc()
    gb = 2 ** 30
    one = m.smooth_memory_plan(6000, 7500, 2, True, available=8 * gb)      # 45 MP, 16-bit, 8 GB free
    assert 2 <= one[0] <= m.SMOOTH_MAX_STRIPS
    assert one[1] > 5 * gb, "the whole frame at once needs several GB"
    tight = m.smooth_memory_plan(6000, 7500, 2, True, available=4 * gb)[0]
    assert tight >= one[0], "less free memory never means fewer strips"
    assert m.smooth_memory_plan(6000, 7500, 2, True, available=1 * gb)[0] == m.SMOOTH_MAX_STRIPS
    assert m.smooth_memory_plan(60, 80, 2, True, available=1000)[0] == 1, "a tiny frame is never cut up"


def test_smoothing_only_some_rows_matches_those_rows_of_the_whole_frame():
    m = _msc()
    for dtype in (np.uint8, np.uint16):
        im = _noisy(300, 400, dtype, 5)
        top = 65535.0 if dtype == np.uint16 else 255.0
        whole = m.smooth_frame(im, 3, bands=1, workers=1)
        for y0, y1 in ((0, 90), (90, 210), (210, 300)):
            part = m.smooth_frame(im, 3, rows=(y0, y1))
            assert part.shape[0] == y1 - y0
            assert np.abs(part - whole[y0:y1]).max() < top * 2e-5, f"rows {y0}-{y1} differ"


def test_the_stack_built_in_strips_is_the_same_picture_as_the_whole_frame():
    import contextlib
    import io
    m = _msc()
    for dtype, kw, use_ground in ((np.uint8, {}, False), (np.uint16, {"taper": 0.25}, True),
                                  (np.uint8, {"comet_tail_frames": 3}, True)):
        with tempfile.TemporaryDirectory() as td:
            names = _frames_for_strips(td, dtype=dtype)
            fg = np.zeros((200, 260), np.uint8)
            fg[150:, :] = 255
            args = dict(fg=fg if use_ground else None, reference_width=260, **kw)
            m._available_bytes = lambda: None
            whole = m._smooth_stack_fullres(td, names, 3, **args)
            m._available_bytes = lambda: 4_000_000          # far too little for one piece
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                strips = m._smooth_stack_fullres(td, names, 3, **args)
        assert "strips" in out.getvalue(), "it should say it is working in strips"
        assert whole.dtype == strips.dtype and whole.shape == strips.shape
        assert np.array_equal(whole, strips), f"strips changed the picture ({dtype.__name__}, {kw})"


def test_progress_lines_keep_climbing_when_the_stack_is_built_in_strips():
    import contextlib
    import io
    import re
    m = _msc()
    with tempfile.TemporaryDirectory() as td:
        names = _frames_for_strips(td, n=6, dtype=np.uint8)
        m._available_bytes = lambda: 3_000_000
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m._smooth_stack_fullres(td, names, 3, reference_width=260)
    lines = [tuple(int(x) for x in mo.groups()) for mo in
             re.finditer(r"smooth star trail: (\d+)/(\d+)", out.getvalue())]
    assert lines, "no progress lines"
    totals = {t for _, t in lines}
    assert len(totals) == 1 and totals.pop() > 6, "the total counts every strip's pass"
    done = [d for d, _ in lines]
    assert done == sorted(done) and done[-1] == lines[-1][1], "the count only climbs and ends at the total"


def test_the_row_is_called_smooth_and_thicken_and_says_it_thickens():
    """The blur also makes trails wider, which "Trail Smoothing" did not say (Bruce,
    2026-11-21). The row reads Smooth & Thicken and the hover text says so."""
    from PySide6.QtWidgets import QApplication, QLabel
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
    labels = [lab.text() for lab in p.findChildren(QLabel)]
    assert "Smooth & Thicken" in labels, "the row label"
    assert "Trail Smoothing" not in labels, "the old row label is gone"
    assert "thickens" in p._size_cb.toolTip(), "the hover text says it thickens the trails"
    assert [p._size_cb.itemText(i) for i in range(p._size_cb.count())] == ["Off", "Light", "Medium", "Strong"]
    assert p._smooth_note.text() == "Will take longer to render", "the render-time note is unchanged"
