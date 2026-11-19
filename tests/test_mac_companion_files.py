"""macOS companion files must never be mistaken for the user's photos.

WHAT HAPPENED (Jon B, 2026-08-30, shipped v2.92). He cleaned 399 frames to 16-bit
TIFF on a portable drive, then pressed the timelapse button and got:

    TypeError: cannot unpack non-iterable NoneType object

On a drive that cannot store extended attributes natively -- exFAT, FAT, most
network shares -- macOS keeps them in a hidden companion file named "._" plus the
original name. It carries the SAME extension, so "._JBA7985.tif" looks like a
photo to any scan that only checks the ending, and a leading dot sorts it FIRST.
His folder listed as 798 photos. The renderer asked the first one for its
dimensions, got None back, and unpacked it.

THE APP HELPED CREATE THEM: it stamps a Finder comment onto every cleaned frame,
which on such a drive has nowhere to live but a companion file. 399 frames, 399
companions.

Bruce never saw it because his drives are APFS, which stores the attribute
natively and creates nothing.

THREE DEFENCES, all tested here:
  1. no folder scan counts them -- one shared rule, so the cleaner, the star
     trail and the timelapse cannot drift apart again;
  2. the renderer survives a file it cannot read instead of dying on it;
  3. the comment is not written at all when the drive cannot hold it.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The first bytes of a real AppleDouble file: magic 00 05 16 07.
APPLEDOUBLE = b"\x00\x05\x16\x07" + b"\x00" * 60


def _code_of(filename, signature):
    """One function's CODE, with its docstring and comments stripped out.

    Tests that grep a function for a forbidden call must read what the computer
    runs, not what the prose says. Good comments name the mistake they exist to
    prevent -- so a docstring explaining "os.setxattr does not exist on macOS"
    reads, to a plain substring search, exactly like the bug it warns about. That
    caught me twice in one day (2026-08-30)."""
    import re
    src = (REPO / filename).read_text(encoding="utf-8")
    body = src[src.index(signature):]
    end = body.find("\n    def ", 10)
    body = body[:end] if end > 0 else body
    body = re.sub(r'""".*?"""', "", body, flags=re.S)
    return "\n".join(l for l in body.splitlines()
                     if not l.strip().startswith("#"))


def _folder_with_companions(n=8, ext=".jpg"):
    """A folder as it looks on a drive without native extended attributes: every
    real photo shadowed by a same-extension companion."""
    import cv2
    d = Path(tempfile.mkdtemp())
    for i in range(n):
        stem = f"_JBA{7985 + i}"
        img = np.full((120, 160, 3), 20, np.uint8)
        img[60, 10 + i * 15] = 240          # a star that moves
        cv2.imwrite(str(d / (stem + ext)), img)
        (d / ("._" + stem + ext)).write_bytes(APPLEDOUBLE)
    return str(d), n


# ── 1. nothing counts them as photos ───────────────────────────────────────

def test_the_shared_rule_rejects_them():
    from modules.frame_list import is_image_name
    assert not is_image_name("._JBA7985.tif"), "the companion file passed as a photo"
    assert not is_image_name("._JBA7985.jpg")
    assert is_image_name("_JBA7985.tif"), "a real photo was rejected"
    assert is_image_name("IMG_0001.JPG"), "upper-case extensions must still count"


def test_other_hidden_files_are_rejected_too():
    from modules.frame_list import is_image_name
    assert not is_image_name(".DS_Store")
    assert not is_image_name(".hidden.jpg")
    assert not is_image_name("Icon\r"), "the Finder's custom-icon file"


def test_non_images_are_still_rejected():
    from modules.frame_list import is_image_name
    assert not is_image_name("notes.txt")
    assert not is_image_name("run_log.jsonl")


def test_the_cleaning_pipeline_does_not_count_them():
    from modules.frame_list import gather_frames
    d, n = _folder_with_companions()
    got = gather_frames(d)
    assert len(got) == n, f"expected {n} photos, listed {len(got)}"


def test_the_timelapse_does_not_count_them():
    """Jon's exact failure point."""
    import timelapse_maker as tm
    d, n = _folder_with_companions()
    got = tm.ordered_frames(d)
    assert len(got) == n, f"expected {n} photos, listed {len(got)}"
    assert not os.path.basename(got[0]).startswith("._"), (
        f"a companion file sorted first and would be asked for its size: {got[0]}")


def test_the_star_trail_does_not_count_them():
    import make_share_clip as msc
    d, n = _folder_with_companions()
    got = msc._list_frames(d)
    assert all(not x.startswith("._") for x in got), got
    assert len(got) == n - 3 - 3, (
        f"the 3-and-3 test-shot skip counted companions as shots: {got}")


def test_every_lister_uses_the_one_rule():
    """They disagreed once and it cost a user a run. Guard by name."""
    for f in ("timelapse_maker.py", "make_share_clip.py", "modules/frame_list.py"):
        src = (REPO / f).read_text(encoding="utf-8")
        assert "is_image_name" in src, f"{f} still decides for itself what a photo is"


# ── 2. the renderer survives an unreadable file ────────────────────────────

def test_a_render_completes_with_companions_all_through_the_folder():
    import timelapse_maker as tm
    d, n = _folder_with_companions()
    out = os.path.join(d, "out.mp4")
    assert tm.render(d, out, size_key="1080p", fps=15) == 0
    assert os.path.isfile(out) and os.path.getsize(out) > 0


def test_an_unreadable_first_file_does_not_end_the_render():
    """Defence in depth: even if something unreadable slips past the listing, the
    render must move on rather than die on it."""
    import cv2
    import timelapse_maker as tm
    d = Path(tempfile.mkdtemp())
    (d / "000_broken.jpg").write_bytes(b"not an image at all")
    for i in range(6):
        cv2.imwrite(str(d / f"IMG_{100 + i}.jpg"), np.full((120, 160, 3), 30, np.uint8))
    out = str(d / "out.mp4")
    assert tm.render(str(d), out, size_key="1080p", fps=15) == 0, (
        "one unreadable file still ends the whole render")
    assert os.path.isfile(out)


def test_a_folder_of_nothing_readable_explains_itself():
    """Not a traceback. A person has to be able to act on it."""
    import timelapse_maker as tm
    d = Path(tempfile.mkdtemp())
    for i in range(4):
        (d / f"broken{i}.tif").write_bytes(b"not an image")
    assert tm.render(str(d), str(d / "x.mp4"), size_key="1080p", fps=15) == 2, (
        "expected a clean failure code, not a crash or a bogus success")


# ── 3. the comment is not written where it would litter ────────────────────

def test_the_probe_uses_the_tool_that_exists_on_macos():
    """os.setxattr is LINUX-ONLY. An earlier version of this check called it, so
    it raised on every Mac and would have switched the Finder comment off for
    everyone instead of only where it does harm."""
    code = _code_of("astro_clean_v5.py", "def _volume_keeps_xattrs(")
    assert "os.setxattr" not in code, (
        "os.setxattr does not exist on macOS; the probe would fail every time")
    assert "/usr/bin/xattr" in code


def test_the_probe_cleans_up_after_itself():
    """It writes a file into the user's output folder to find its answer. Leaving
    that behind would be its own version of the litter it exists to prevent."""
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    body = src[src.index("def _volume_keeps_xattrs("):]
    body = body[:body.index("\n    def ", 10)]
    assert "finally:" in body and "os.remove" in body, (
        "the probe file is not removed on every path")


def test_the_comment_is_gated_on_the_probe():
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    body = src[src.index("def _write_finder_comment("):]
    body = body[:body.index("\n    def ", 10)]
    assert "_volume_keeps_xattrs" in body, (
        "the Finder comment is written again without asking whether the drive "
        "can hold it, which is what created the companion files")
