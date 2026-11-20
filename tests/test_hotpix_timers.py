"""The hot-pixel steps record where their time goes (permanent step timers).

2026-11-19: on 200 real frames the hot-pixel phase took 95 s, about twice the smoothing
stack, and nothing in the program said why. Each frame turned out to be read from disk
about 5 times by the flash finder alone. These timers and read counters stay in the code
so the Star Log always shows it. They only observe: the output must not change.
Offline, synthetic frames.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _frames(n=14, h=120, w=160):
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        img = (rng.random((h, w, 3)) * 12).astype(np.uint8)
        img[60, 10 + 8 * i] = (250, 250, 250)            # a star drifting across
        out.append(img)
    return out


def test_the_flash_finder_records_its_steps_and_reads():
    from modules import step_timer
    from modules.flash_repair import run_on_folder
    frames = _frames()
    stack = np.maximum.reduce(frames)
    step_timer.reset()
    run_on_folder("x", [f"f{i}.png" for i in range(len(frames))], stack, None,
                  lambda p: frames[int(Path(p).stem[1:])], say=lambda *_: None)
    snap = step_timer.snapshot()["steps"]
    assert "flashes.find" in snap and "ground.find" in snap
    assert snap["flashes.read"][0] >= len(frames), "every frame is read at least once"


def test_the_summary_line_is_plain_and_ignored_by_the_progress_bar(capsys=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location("msc_hp", REPO / "make_share_clip.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from modules import step_timer
    import io, contextlib
    step_timer.reset()
    step_timer.add("dots.scan", 4.0)
    step_timer.add("flashes.find", 6.0)
    step_timer.add("flashes.read", 2.0)
    step_timer.add("flashes.read", 2.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m._print_hotpix_times(10)
    text = buf.getvalue()
    assert "scanning for specks 4s" in text and "finding flashes 6s" in text
    assert "flashes 0.2 (4s)" in text, "two reads over ten frames, summed seconds shown"
    for line in text.splitlines():
        tail = line.rsplit(":", 1)[-1]
        assert "/" not in tail, "must not look like a progress line"
