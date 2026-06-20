"""Quick-and-dirty star trail: full-res lighten-max stack of the cleaned frames.

Plants a unique bright block in each synthetic frame, then confirms make_star_trail:
  - writes a readable JPG at the source resolution,
  - is a pixelwise lighten-max (a block bright in any KEPT frame is bright in the output),
  - honors the first-3 / last-3 skip (blocks only in skipped frames never appear),
  - raises (no silent empty file) when the folder is missing or has no frames.
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from make_share_clip import make_star_trail

CANVAS = 300        # square canvas, comfortably larger than the planted blocks
BLOCK = 12          # block half-extent survives JPG compression as a bright region


def _pos(i):
    """A unique, well-separated (row, col) center for frame i (0..9)."""
    col = 30 + (i % 5) * 55     # 30, 85, 140, 195, 250
    row = 60 + (i // 5) * 150   # 60 for 0-4, 210 for 5-9
    return row, col


def _write_frames(folder, n=10):
    """N dark frames, each with one bright white block at its own _pos(i)."""
    for i in range(n):
        f = np.full((CANVAS, CANVAS, 3), 10, dtype=np.uint8)
        r, c = _pos(i)
        f[r - BLOCK:r + BLOCK, c - BLOCK:c + BLOCK] = 250
        cv2.imwrite(str(Path(folder) / f"frame_{i:02d}.jpg"), f,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def _region_mean(img, i):
    r, c = _pos(i)
    return float(img[r - BLOCK:r + BLOCK, c - BLOCK:c + BLOCK].mean())


def _run(cleaned_dir, out_path):
    """Call make_star_trail with its console output swallowed (keeps the suite quiet)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return make_star_trail(cleaned_dir, out_path)


def test_star_trail_is_full_res_lighten_max_of_kept_frames():
    with tempfile.TemporaryDirectory() as d:
        cleaned = Path(d) / "cleaned"
        cleaned.mkdir()
        _write_frames(cleaned, n=10)
        out = Path(d) / "cleaned_star_trail.jpg"

        ret = _run(str(cleaned), str(out))

        assert Path(ret) == out
        assert out.exists(), "no star trail written"
        img = cv2.imread(str(out))
        assert img is not None, "star trail JPG is unreadable"
        assert img.shape == (CANVAS, CANVAS, 3), f"wrong dimensions: {img.shape}"

        # _list_frames skips the first 3 and last 3, so of 10 frames only 3,4,5,6
        # are stacked. Their blocks must be bright; every other block must be dark.
        for i in (3, 4, 5, 6):
            assert _region_mean(img, i) > 150, f"kept frame {i} block missing from stack"
        for i in (0, 1, 2, 7, 8, 9):
            assert _region_mean(img, i) < 60, f"skipped frame {i} block leaked into stack"


def test_star_trail_missing_folder_raises():
    raised = False
    try:
        _run("/no/such/cleaned/folder/xyz", "/tmp/should_not_be_written.jpg")
    except SystemExit:
        raised = True
    assert raised, "missing cleaned folder should raise, not write a file"


def test_star_trail_empty_folder_raises():
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "cleaned"
        empty.mkdir()
        raised = False
        try:
            _run(str(empty), str(Path(d) / "out.jpg"))
        except SystemExit:
            raised = True
        assert raised, "empty cleaned folder should raise, not write a file"
