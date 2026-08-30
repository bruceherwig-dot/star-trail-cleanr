"""The timelapse can be built two ways, and neither may disturb the other.

Suggested by Jon Bertsch, 2026-08-27: "While the star circle is being created you
could also make a timelapse video of each step as the frames are blended that
shows how each image adds to the circle." It was cheap to build because the app
already computes exactly those intermediate stacks during a run and throws each
one away when the next arrives.

  Moving Stars (traditional timelapse)  -- one photo per movie frame, style=plain
  Building Trails (star accumulation)   -- every photo so far, style=accumulate

WHY THE STYLES MUST NOT BE CONFUSED WITH SMOOTHING. "blended" keeps a ROLLING
window of the last N photos and forgets older ones, which is what makes it a
flicker smoother. "accumulate" never forgets, so the trails grow and the final
frame is the finished star trail. Smoothing is therefore meaningless in the
accumulation style and is greyed out rather than left sitting there doing nothing.

TESTING NOTE, learned the hard way here: THE VIDEO ENCODER IS NOT DETERMINISTIC.
The same code rendering the same photos twice produced files of 4.35 MB and
4.32 MB with different hashes. Comparing finished .mp4 files proves nothing about
a code change. These tests compare the FRAMES handed to the encoder instead.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

MAKER = (REPO / "timelapse_maker.py").read_text(encoding="utf-8")
GUI = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")


def _frames_for(style, n=6, size=(40, 30)):
    """Run the real render loop over synthetic photos, capturing what it would
    have encoded. No file is written and no encoder is involved."""
    import tempfile
    import cv2
    import timelapse_maker as tm

    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(4)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 10, np.uint8)
        # one moving dot, so accumulation has something visible to pile up
        img[size[1] // 2, 3 + i * 4] = 240
        cv2.imwrite(str(d / f"f{i:03d}.jpg"), img)

    got = []

    class _Cap:
        backend = "capture"
        def write(self, f): got.append(f.copy())
        def close(self): pass

    real = tm._open_writer
    tm._open_writer = lambda *a, **k: _Cap()
    try:
        tm.render(str(d), "/dev/null", size_key="1080p", fps=15, style=style)
    finally:
        tm._open_writer = real
    return got


def test_the_renderer_offers_all_three_styles():
    assert '"accumulate"' in MAKER, "the accumulation style is gone from the renderer"
    assert 'choices=["plain", "blended", "accumulate"]' in MAKER, (
        "the command line no longer accepts the accumulation style")


def test_accumulation_never_forgets_a_photo():
    """The defining property: brightness at any pixel can only rise. If a rolling
    window crept in, the trails would fade behind the moving star instead of
    staying drawn."""
    frames = _frames_for("accumulate")
    lit = [int((f > 100).sum()) for f in frames]
    assert lit == sorted(lit), f"the picture dimmed as it played: {lit}"
    assert lit[-1] > lit[0], "nothing accumulated at all"


def test_the_traditional_style_forgets_everything():
    """The opposite property, and the guarantee that adding accumulation did not
    quietly change the timelapse people already make."""
    frames = _frames_for("plain")
    lit = [int((f > 100).sum()) for f in frames]
    assert max(lit) - min(lit) <= 2, (
        f"a plain timelapse should show one photo per frame, not a pile-up: {lit}")


def test_the_last_accumulated_frame_is_the_finished_star_trail():
    frames = _frames_for("accumulate")
    combined = frames[0].copy()
    for f in frames[1:]:
        np.maximum(combined, f, out=combined)
    assert np.array_equal(combined, frames[-1]), (
        "the final frame must BE the star trail, not an approximation of it")


def test_the_window_offers_the_styles_by_their_plain_names():
    assert "Moving Stars (traditional timelapse)" in GUI
    assert "Building Trails (star accumulation)" in GUI


def test_smoothing_is_greyed_out_when_it_would_do_nothing():
    assert "_sync_style" in GUI, "the greying hook is gone"
    body = GUI[GUI.index("def _sync_style("):]
    body = body[:body.index("\n    def ", 10)]
    assert "_blend_cb.setEnabled" in body, (
        "Smoothing is live again in the accumulation style, where it changes "
        "nothing at all")


def test_the_two_styles_cannot_overwrite_each_other():
    assert '"accumulation" if _style == "accumulate" else "traditional"' in GUI, (
        "the style is no longer in the filename; rendering both kinds with the "
        "same size and frame rate would collide")
