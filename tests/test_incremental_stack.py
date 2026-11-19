"""IncrementalStack must produce a result bit-identical to the batch stackers.

Building the star-trail / before-after stack DURING a run (one frame at a time,
overlapped with cleaning) instead of in a second full pass afterward is only safe
if it equals the all-at-once result. Lighten-max is order-independent, so it should
-- these tests lock that in so a future change can't silently corrupt the output.
"""
import os
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_share_clip as msc


def _make_frames(tmp, n=6, w=64, h=48):
    """Write n synthetic frames with scattered bright dots (like stars), so the
    lighten-max stack is non-trivial (different bright pixels per frame)."""
    names = []
    rng = np.random.RandomState(0)
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        ys = rng.randint(0, h, 5)
        xs = rng.randint(0, w, 5)
        img[ys, xs] = rng.randint(100, 255, (5, 3))
        name = f"f{i:03d}.png"
        cv2.imwrite(os.path.join(tmp, name), img)
        names.append(name)
    return names


def test_fullres_incremental_equals_batch():
    """Full-resolution (star trail): feeding frames one by one == _stack_fullres."""
    with tempfile.TemporaryDirectory() as tmp:
        names = _make_frames(tmp)
        batch = msc._stack_fullres(tmp, names, "batch")
        inc = msc.IncrementalStack("inc")          # canvas=None -> full-res
        for n in names:
            inc.feed_path(os.path.join(tmp, n))
        assert inc.used == len(names), f"fed {inc.used}, expected {len(names)}"
        assert np.array_equal(inc.result(), batch), "full-res incremental != batch stack"


def test_canvas_incremental_equals_batch():
    """Video canvas: feeding frames one by one == _stack at the same canvas size."""
    with tempfile.TemporaryDirectory() as tmp:
        names = _make_frames(tmp)
        cw, ch = 80, 100
        batch = msc._stack(tmp, names, cw, ch, "batch")
        inc = msc.IncrementalStack("inc", canvas=(cw, ch))
        for n in names:
            inc.feed_path(os.path.join(tmp, n))
        assert np.array_equal(inc.result(), batch), "canvas incremental != batch stack"


def test_feed_image_matches_feed_path():
    """Folding in an already-decoded frame (the zero-extra-read path) matches reading it."""
    with tempfile.TemporaryDirectory() as tmp:
        names = _make_frames(tmp, n=4)
        by_path = msc.IncrementalStack("p")
        by_image = msc.IncrementalStack("i")
        for n in names:
            p = os.path.join(tmp, n)
            by_path.feed_path(p)
            by_image.feed_image(cv2.imread(p, cv2.IMREAD_COLOR), n)
        assert np.array_equal(by_path.result(), by_image.result())


def test_missing_frame_recorded_not_raised():
    """A missing frame is recorded for the loud warning, never raised (no-silent-drop)."""
    with tempfile.TemporaryDirectory() as tmp:
        names = _make_frames(tmp, n=3)
        inc = msc.IncrementalStack("inc")
        inc.feed_path(os.path.join(tmp, names[0]))
        inc.feed_path(os.path.join(tmp, "does_not_exist.png"))
        inc.feed_path(os.path.join(tmp, names[1]))
        assert inc.used == 2
        assert inc.missing == ["does_not_exist.png"]


def test_make_star_trail_prebuilt_stack_no_crash():
    """make_star_trail(stack=...) must save the prebuilt stack WITHOUT referencing the
    'names' list that only exists on the folder-stacking path. Regression: that print
    threw UnboundLocalError, which saved the image but then killed the run before the
    video step (the in-run stacker's whole point)."""
    with tempfile.TemporaryDirectory() as tmp:
        stack = np.full((24, 32, 3), 50, np.uint8)
        out = os.path.join(tmp, "star.jpg")
        msc.make_star_trail("ignored-when-stack-given", out_path=out, stack=stack)
        assert os.path.exists(out), "prebuilt-stack star trail not written"


if __name__ == "__main__":
    test_fullres_incremental_equals_batch()
    test_canvas_incremental_equals_batch()
    test_feed_image_matches_feed_path()
    test_missing_frame_recorded_not_raised()
    test_make_star_trail_prebuilt_stack_no_crash()
    print("all incremental-stack tests passed")
