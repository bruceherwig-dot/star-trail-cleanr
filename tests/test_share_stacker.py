"""ShareStacker fed cleaned frames incrementally (as a run produces them) must end
up with the SAME stacks as stacking the cleaned folder all at once. This is what
makes building the star trail / video DURING the run safe instead of in a second pass.
"""
import os
import sys
import tempfile
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_share_clip as msc
from modules.share_stacker import ShareStacker


def _write_frames(folder, n, w=96, h=72, seed=0):
    rng = np.random.RandomState(seed)
    names = []
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        ys = rng.randint(0, h, 6)
        xs = rng.randint(0, w, 6)
        img[ys, xs] = rng.randint(100, 255, (6, 3))
        name = f"f{i:03d}.png"
        cv2.imwrite(os.path.join(folder, name), img)
        names.append(name)
    return names


def test_incremental_after_stack_matches_batch():
    """Star-trail (full-res) + video (canvas) after-stacks built by folding cleaned
    frames in chunks == the batch stack of the same kept frames."""
    with tempfile.TemporaryDirectory() as orig, tempfile.TemporaryDirectory() as clean:
        _write_frames(orig, 14, seed=1)
        # Create the stacker BEFORE the cleaned frames exist (as in a real run), so the
        # mtime guard recognizes the frames written next as THIS run's output.
        ss = ShareStacker(orig, clean, want_star=True, want_video=True)
        ss.build_before()
        all_clean = _write_frames(clean, 14, seed=2)   # cleaned written now -> this-run mtime
        # Simulate the run revealing cleaned frames a few at a time: hide them all,
        # then restore in chunks, scanning between (mimics batch boundaries).
        import shutil
        staged = tempfile.mkdtemp()
        for n in all_clean:
            shutil.move(os.path.join(clean, n), os.path.join(staged, n))
        for chunk_start in range(0, len(all_clean), 5):
            for n in all_clean[chunk_start:chunk_start + 5]:
                shutil.move(os.path.join(staged, n), os.path.join(clean, n))
            ss.scan_cleaned()
        ss.scan_cleaned()   # final scan
        shutil.rmtree(staged)

        kept = msc._list_frames(clean)
        batch_full = msc._stack_fullres(clean, kept, "batch")
        assert np.array_equal(ss.after_full.result(), batch_full), "after full-res != batch"

        cw, ch = ss._cw, ss._img_h
        batch_canvas = msc._stack(clean, kept, cw, ch, "batch")
        assert np.array_equal(ss.after_vid.result(), batch_canvas), "after canvas != batch"


def test_before_stack_matches_batch():
    """The before-stack (originals, video canvas) == the batch stack of the originals."""
    with tempfile.TemporaryDirectory() as orig, tempfile.TemporaryDirectory() as clean:
        names = _write_frames(orig, 12, seed=3)
        _write_frames(clean, 12, seed=4)
        ss = ShareStacker(orig, clean, want_star=False, want_video=True)
        ss.build_before()
        kept = msc._list_frames(orig)
        cw, ch = ss._cw, ss._img_h
        batch = msc._stack(orig, kept, cw, ch, "batch")
        assert np.array_equal(ss.before_vid.result(), batch), "before canvas != batch"


def test_nothing_enabled_does_no_work():
    """No box checked -> no stacks ever allocated (zero overhead)."""
    with tempfile.TemporaryDirectory() as orig, tempfile.TemporaryDirectory() as clean:
        _write_frames(orig, 8)
        _write_frames(clean, 8)
        ss = ShareStacker(orig, clean, want_star=False, want_video=False)
        ss.build_before()
        ss.scan_cleaned()
        assert ss.after_full is None and ss.after_vid is None and ss.before_vid is None


def test_stale_leftover_frames_skipped_until_rewritten():
    """The re-run case: cleaned/ is pre-populated with a PRIOR run's frames (old mtime).
    Those must NOT be folded; only once THIS run rewrites a frame (fresh mtime) does it
    count. Otherwise a re-run would stack yesterday's output instead of today's."""
    with tempfile.TemporaryDirectory() as orig, tempfile.TemporaryDirectory() as clean:
        _write_frames(orig, 10, seed=5)
        # "Yesterday's" cleaned frames, backdated a day.
        names = _write_frames(clean, 10, seed=6)
        old = time.time() - 86400
        for n in names:
            os.utime(os.path.join(clean, n), (old, old))
        ss = ShareStacker(orig, clean, want_star=True, want_video=False)   # starts NOW
        ss.scan_cleaned()
        assert ss.after_full.used == 0, "stale leftover frames must be skipped"
        # This run rewrites them (same content, fresh mtime) -> now they count.
        _write_frames(clean, 10, seed=6)
        ss.scan_cleaned()
        assert ss.after_full.used == len(msc._list_frames(clean)), "rewritten frames not folded"


if __name__ == "__main__":
    test_incremental_after_stack_matches_batch()
    test_before_stack_matches_batch()
    test_nothing_enabled_does_no_work()
    test_stale_leftover_frames_skipped_until_rewritten()
    print("all share-stacker tests passed")
