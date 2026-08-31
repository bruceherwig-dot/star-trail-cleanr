"""A cleaned folder holding BOTH .jpg and .tif copies of the same shots.

Found 2026-08-30, the same day the star trail started following the cleaned
frames' format. Bruce re-cleaned a folder as TIFF that had already been cleaned
as JPEG, so it held IMG_2946.jpg and IMG_2946.tif side by side -- the same shot,
twice, at two different bit depths.

THREE SEPARATE FAULTS, NONE OF WHICH RAISED AN ERROR:

  1. Every shot was stacked twice, so the shot count doubled and the "skip the
     first 3 and last 3 test shots" rule cut real photos instead.
  2. Lighten-max compares raw numbers, and the two depths do not share a scale.
     8-bit accumulator first: every 16-bit frame was TRUNCATED, 40000 landing as
     64, turning bright stars nearly black. 16-bit first: every 8-bit frame lost
     every pixel, because 255 cannot beat values that run to 65535, so those
     photos contributed nothing at all.
  3. The window named the trail from whichever file sorted first (".jpg"), while
     the stacker built from the TIFFs (16-bit). A 16-bit picture cannot be
     written into a JPEG, so the save failed and no star trail appeared.

All three are silent by nature, which is what makes them worth a test file.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _folder(names):
    """A throwaway folder holding real image files under the given names."""
    import cv2
    import tifffile
    d = Path(tempfile.mkdtemp())
    for n in names:
        p = d / n
        if p.suffix.lower() in (".tif", ".tiff"):
            tifffile.imwrite(str(p), np.full((8, 10, 3), 400, np.uint16),
                             photometric="rgb")
        else:
            cv2.imwrite(str(p), np.full((8, 10, 3), 40, np.uint8))
    return d


# ── 1. one file per shot ────────────────────────────────────────────────────

def test_a_shot_cleaned_twice_is_stacked_once():
    from make_share_clip import _list_frames
    names = []
    for i in range(2946, 2956):                 # 10 shots, each present twice
        names += [f"IMG_{i}.jpg", f"IMG_{i}.tif"]
    got = _list_frames(str(_folder(names)))
    stems = [os.path.splitext(n)[0] for n in got]
    assert len(stems) == len(set(stems)), (
        f"the same shot is being stacked more than once: {got}")


def test_the_tiff_copy_is_the_one_kept():
    """Same picture, more of it kept. It also has to match what cleaned_star_ext
    answers, or the trail cannot be saved (see the naming test below)."""
    from make_share_clip import _list_frames
    got = _list_frames(str(_folder(
        [f"IMG_{i}.{e}" for i in range(2946, 2956) for e in ("jpg", "tif")])))
    assert all(n.endswith(".tif") for n in got), f"kept the JPEG copies: {got}"


def test_the_test_shot_skip_counts_shots_not_files():
    """With twins counted separately, 3-and-3 cut six FILES, which is only three
    real shots at each end -- so real photos were dropped from the trail."""
    from make_share_clip import _list_frames, SKIP_FIRST, SKIP_LAST
    names = [f"IMG_{i}.{e}" for i in range(1, 21) for e in ("jpg", "tif")]
    got = _list_frames(str(_folder(names)))
    assert len(got) == 20 - SKIP_FIRST - SKIP_LAST, (
        f"expected 14 shots after the test-shot skip, got {len(got)}")


def test_a_raw_and_jpg_originals_folder_is_deduplicated_too():
    """The same fault has always been present on the ORIGINALS side, where
    RAW+JPG twins are routine. Non-RAW wins: same picture, far faster to read."""
    from make_share_clip import _list_frames
    d = _folder([f"IMG_{i}.jpg" for i in range(2946, 2956)])
    for i in range(2946, 2956):
        (d / f"IMG_{i}.cr2").write_bytes(b"not really a raw file")
    got = _list_frames(str(d))
    assert all(n.endswith(".jpg") for n in got), f"picked the RAW twins: {got}"
    assert len(got) == 10 - 3 - 3


def test_a_raw_only_shot_still_survives():
    """Preferring non-RAW must never mean DROPPING a shot that is RAW only."""
    from make_share_clip import _list_frames
    d = _folder([f"IMG_{i}.jpg" for i in range(2946, 2956)])
    for i in (2950, 2951):
        os.remove(d / f"IMG_{i}.jpg")
        (d / f"IMG_{i}.cr2").write_bytes(b"raw only")
    got = _list_frames(str(d))
    assert len(got) == 10 - 3 - 3, f"a RAW-only shot was dropped: {got}"


# ── 2. never mix two depths ─────────────────────────────────────────────────

def test_a_16bit_frame_is_not_truncated_by_an_8bit_stack():
    """The first fault, in isolation. Without the guard, 40000 became 64."""
    from make_share_clip import _match_depth
    acc = np.full((4, 4, 3), 30, np.uint8)
    im = np.full((4, 4, 3), 40000, np.uint16)
    im, acc = _match_depth(im, acc)
    np.maximum(acc, im, out=acc)
    assert acc.dtype == np.uint16, "the stack stayed 8-bit and lost the frame"
    assert int(acc.max()) == 40000, f"the bright frame was truncated to {acc.max()}"


def test_an_8bit_frame_can_still_win_a_pixel_in_a_16bit_stack():
    """The second fault. A bright 8-bit frame (255) must beat a dim 16-bit one,
    not lose automatically because its numbers are smaller."""
    from make_share_clip import _match_depth
    acc = np.full((4, 4, 3), 1000, np.uint16)       # a dim 16-bit frame
    im = np.full((4, 4, 3), 255, np.uint8)          # a WHITE 8-bit frame
    im, acc = _match_depth(im, acc)
    np.maximum(acc, im, out=acc)
    assert int(acc.max()) == 65535, (
        f"white came out as {acc.max()}; the 8-bit frame lost to a dimmer one")


def test_white_stays_white_when_a_frame_is_promoted():
    """257, not 256: 255 has to land exactly on 65535 or every promoted frame
    comes out fractionally grey."""
    from make_share_clip import _match_depth
    im, _ = _match_depth(np.full((2, 2, 3), 255, np.uint8),
                         np.zeros((2, 2, 3), np.uint16))
    assert int(im.max()) == 65535


def test_both_stackers_are_guarded_not_just_one():
    """One runs after a clean, the other DURING it. Bruce hit the live one."""
    body = (REPO / "make_share_clip.py").read_text(encoding="utf-8")
    fullres = body[body.index("def _stack_fullres("):]
    fullres = fullres[:fullres.index("\nclass ")]
    assert "_match_depth" in fullres, "the after-the-run stacker is unguarded"
    inc = body[body.index("class IncrementalStack"):]
    assert "_match_depth" in inc[:inc.index("def report(")], (
        "the in-run stacker is unguarded -- this is the one a live clean uses")


def test_a_mixed_folder_says_so_in_the_log():
    """Bruce's standing rule: nothing is dropped or altered silently."""
    # The message is built from several string literals across several source
    # lines, so strip the quoting and the line breaks before looking for it.
    src = (REPO / "make_share_clip.py").read_text(encoding="utf-8")
    body = " ".join(src.replace('f"', " ").replace('"', " ").split())
    assert body.count("a different bit depth than the rest") >= 2, (
        "a mixed-depth stack no longer reports itself in the Star Log "
        "(BOTH stackers must say so -- the one that runs after a clean and the "
        "one that runs during it)")


# ── 3. the name and the data must agree ─────────────────────────────────────

def test_one_tiff_anywhere_names_the_trail_a_tiff():
    """The stacker keeps the TIFF copy, so the stack is 16-bit. If the name says
    .jpg the save fails outright -- a JPEG cannot hold 16 bits."""
    sys.path.insert(0, str(REPO))
    from star_trail_cleanr import cleaned_star_ext
    d = _folder([f"IMG_{i}.{e}" for i in range(2946, 2950) for e in ("jpg", "tif")])
    assert cleaned_star_ext(str(d)) == ".tif", (
        "named from whichever file sorted first; 'IMG_2946.jpg' sorts ahead of "
        "'IMG_2946.tif', so a 16-bit stack would be handed a JPEG path")


def test_a_plain_jpeg_folder_is_unaffected():
    from star_trail_cleanr import cleaned_star_ext
    assert cleaned_star_ext(str(_folder(["a.jpg", "b.jpg"]))) == ".jpg"


def test_a_deep_stack_handed_a_jpeg_path_still_produces_a_trail():
    """Belt and braces. Unreachable once the naming agrees, but 'no star trail
    and no reason given' is the worst possible outcome, so it degrades instead."""
    import cv2
    from make_share_clip import write_star_trail
    d = Path(tempfile.mkdtemp())
    out = d / "trail.jpg"
    assert write_star_trail(str(out), np.full((8, 10, 3), 40000, np.uint16)), (
        "a 16-bit stack with a .jpg path produced no file at all")
    back = cv2.imread(str(out))
    assert back is not None and back.dtype == np.uint8
