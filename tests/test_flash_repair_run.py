"""The flash finder as the star trail step uses it: frames on disk in, a cleaned stack and
a printed summary out."""
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules import flash_repair as fr

H, W, N = 480, 640, 30


def _write_night(folder, flash=True):
    rng = np.random.default_rng(1)
    sx, sy = rng.uniform(40, W - 80, 150), rng.uniform(40, H - 80, 150)
    amp = rng.uniform(120, 230, 150)
    frames, names = [], []
    for k in range(N):
        canvas = np.zeros((H, W), np.float32)
        for x, y, a in zip(sx + 2 * k, sy + k, amp):
            xi, yi = int(round(x)), int(round(y))
            if 3 <= xi < W - 3 and 3 <= yi < H - 3:
                canvas[yi - 2:yi + 3, xi - 2:xi + 3] += a * np.array(
                    [[.05, .2, .3, .2, .05], [.2, .6, .8, .6, .2], [.3, .8, 1, .8, .3],
                     [.2, .6, .8, .6, .2], [.05, .2, .3, .2, .05]], np.float32)
        img = np.clip(rng.normal(18, 3, (H, W, 3)) + canvas[:, :, None], 0, 255).astype(np.uint8)
        if flash and k == 12:
            img[300:302, 200:203] = (255, 40, 40)
        frames.append(img)
        names.append("f%03d.png" % k)
        cv2.imwrite(str(Path(folder) / names[-1]), img)
    return names, np.max(frames, axis=0)


def test_the_flash_comes_out_of_the_stack_and_the_summary_is_printed():
    with tempfile.TemporaryDirectory() as d:
        names, stack = _write_night(d)
        said = []
        out = fr.run_on_folder(d, names, stack, None, lambda p: cv2.imread(p), say=said.append)
    assert out.shape == stack.shape and (out <= stack).all(), "the repair may only remove light"
    assert int(stack[300:302, 200:203, 0].min()) > 200 and int(out[300:302, 200:203, 0].max()) < 120, "the flash must come out"
    kinds = [fr.parse_marker(s) for s in said]
    assert any(k and k[0] == "line" and "sky" in k[1] for k in kinds), said
    counts = [k[1] for k in kinds if k and k[0] == "counts"]
    assert len(counts) == 1 and counts[0]["flash_spots"] >= 1 and counts[0]["ground_stopped"] is True


def test_a_reader_that_fails_leaves_the_stack_alone_and_says_so():
    with tempfile.TemporaryDirectory() as d:
        names, stack = _write_night(d, flash=False)
        said = []
        out = fr.run_on_folder(d, names, stack, None, lambda p: None, say=said.append)
    assert (out == stack).all()
    assert any(fr.MARKER_COUNTS in s for s in said)
