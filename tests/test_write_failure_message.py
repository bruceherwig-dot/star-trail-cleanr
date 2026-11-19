"""A write failure must not blame permissions once frames have already landed.

Sentry 2026-09-08 (Sean, Windows, output on a D: drive): the run wrote five frames
and then failed on the sixth with "[Errno 22] Invalid argument". The message told
him the folder might be read-only, synced by OneDrive, or open in another app. All
three were impossible -- five files had just been written to that same folder --
so the one thing we told him was the one thing it could not be.

Verified against a real full volume on 2026-09-08: 6 frames written, then
"[Errno 28] No space left on device", and the message now reports the count and
the free space instead of guessing.

The write path is a closure inside main() and needs a genuinely failing disk to
exercise, so these lock the SHAPE of the decision: a counter that only advances on
success, both branches present, and the free space reported either way.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SRC = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")


def _write_output_body():
    i = SRC.find("def _write_output(stem: str")
    assert i > 0, "the _write_output wrapper vanished"
    j = SRC.find("def _grey_plane", i)
    assert j > i
    return SRC[i:j]


def test_the_counter_only_advances_on_a_successful_write():
    body = _write_output_body()
    assert "_write_output.written = 0" in body, \
        "the written-frames counter must be initialised"
    inner = body.find("_write_output_inner(")
    bump = body.find("_write_output.written += 1")
    fail = body.find("except (PermissionError, OSError)")
    assert 0 < inner < bump < fail, (
        "the counter must be incremented AFTER the write returns and BEFORE the "
        "failure branch -- counting an attempt would defeat the whole test")


def test_both_causes_are_distinguished():
    body = _write_output_body()
    assert re.search(r"if\s+_done\s*==\s*0", body), \
        "the message must branch on whether any frame was written"
    assert "read-only drive" in body, \
        "with nothing written, the permissions advice is still right"
    assert "not a permissions problem" in body, (
        "once frames have landed, the message must say so -- this is the "
        "sentence that would have saved Sean hunting the wrong thing")
    for cause in ("full", "disconnected", "security software"):
        assert cause in body, f"the partway message should name '{cause}' as a cause"


def test_the_free_space_is_reported():
    body = _write_output_body()
    assert "disk_usage" in body, \
        "free space is the number that separates a full drive from a lost one"
    assert "already written" in body, \
        "the count of written frames must reach the user and the crash report"


def test_the_permissions_advice_is_not_given_when_frames_landed():
    """Guard against a future edit collapsing the branches back into one."""
    body = _write_output_body()
    tail = body[body.find("if _done == 0"):]
    else_part = tail[tail.find("else:"):]
    assert "OneDrive" not in else_part, \
        "OneDrive cannot be the cause once frames have been written to the folder"


def test_an_unwritable_output_folder_fails_immediately_and_cleanly():
    """Ask the question in the first second, not after the slowest part of the job.

    Every later step assumed the output folder was writable: the run-log folder,
    cleaned_dir, masks_dir and each frame written were all unguarded, so an
    unwritable folder produced a raw Python traceback from pathlib.mkdir -- AFTER
    loading the frames and running detection. Found 2026-09-08 while reproducing
    Sean's write failure.

    This runs the real engine as a subprocess, which is what the app does.
    """
    import os
    import subprocess
    import tempfile

    src = Path(tempfile.mkdtemp())
    ro = Path(tempfile.mkdtemp())
    # A folder cannot be created underneath a plain FILE, on any operating system.
    # (chmod 0o500 was the first attempt: Windows ignores it, so on the Windows
    # build machine the folder stayed writable and the run went on to fail
    # somewhere else.)
    blocker = ro / "blocker"
    blocker.write_bytes(b"x")
    out_dir = blocker / "cleaned"
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO / "astro_clean_v5.py"), str(src),
             "-o", str(out_dir), "--model", "nonexistent.pt",
             "--start", "0", "--batch", "3"],
            capture_output=True, text=True, timeout=120)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 2, (
            f"an unwritable output folder must exit cleanly with code 2, "
            f"got {proc.returncode}:\n{out[-1500:]}")
        assert "Cannot write to the output folder" in out, \
            f"the user must get the plain-English message, got:\n{out[-1500:]}"
        assert "Traceback" not in out, \
            f"a raw traceback must never reach the user:\n{out[-1500:]}"
    finally:
        for d in (src, ro):
            try:
                import shutil
                shutil.rmtree(d)
            except OSError:
                pass
