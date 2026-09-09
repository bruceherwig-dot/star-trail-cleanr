"""The engine's explanation must survive the trip to the user.

Sentry 2026-09-08 (Sean): the report is titled

    Worker exited 2: ERROR: Cannot write cleaned frame to:

and stops at the colon. Nothing truncated it. The app kept only the last line
beginning with "ERROR:", and the engine puts the heading on that line and the
part that helps -- the folder, whether frames were already written, what that
rules out, the free space -- on the lines below it. All of it was thrown away
before the user or the crash report ever saw it, for every multi-line message
the engine can print.

The block below is the real output of a run against a volume that filled up,
captured 2026-09-08.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from star_trail_cleanr import worker_error_message

REAL_FULL_DISK_RUN = [
    "  cleaning 5/8: 1M3A9261.jpg - 1 trail",
    "FRAME_TRAIL_COUNT: 13",
    "  cleaning 6/8: 1M3A9262.jpg - 1 trail",
    "FRAME_TRAIL_COUNT: 14",
    "",
    "ERROR: Cannot write cleaned frame to:",
    "  /Volumes/STCFULL/cleaned",
    "",
    "6 frame(s) were written to this folder first, so it is not a permissions "
    "problem. The drive stopped accepting files partway through: it may be full, "
    "it may have disconnected, or security software may be blocking it. Check the "
    "free space, then try a folder on your main internal drive.",
    "",
    "(Detail: OSError: [Errno 28] No space left on device; 6 frame(s) already "
    "written; 0.0 GB free on that drive)",
]


def test_the_explanation_reaches_the_user():
    msg = worker_error_message("", REAL_FULL_DISK_RUN)
    assert msg.startswith("ERROR: Cannot write cleaned frame to:"), \
        "the heading must still lead the message"
    for needed, why in (
            ("/Volumes/STCFULL/cleaned", "the user must be told WHICH folder"),
            ("not a permissions problem", "the sentence that stops a wrong hunt"),
            ("disconnected", "the likely causes must survive"),
            ("0.0 GB free", "the number that settles it must survive"),
            ("already written", "the frame count must survive")):
        assert needed in msg, f"{why}: '{needed}' was dropped"


def test_the_old_behaviour_would_have_failed_this():
    """Guard the guard: the single-line rule really does lose everything."""
    old = [l for l in REAL_FULL_DISK_RUN if l.startswith("ERROR:")][-1]
    assert old == "ERROR: Cannot write cleaned frame to:"
    assert "0.0 GB free" not in old and "not a permissions problem" not in old, \
        "if this ever passes, the engine changed and this test is watching nothing"


def test_progress_lines_before_the_error_are_not_included():
    msg = worker_error_message("", REAL_FULL_DISK_RUN)
    assert "cleaning 6/8" not in msg and "FRAME_TRAIL_COUNT" not in msg, \
        "the message must start at the error, not replay the run"


def test_a_real_crash_still_reports_its_exception():
    """stderr wins when there is one: an unhandled crash ends with the exception,
    and that line is more useful than anything on stdout."""
    msg = worker_error_message(
        "Traceback (most recent call last):\n  File x\nMemoryError: out of memory",
        ["loading 1/20: a.jpg", "ERROR: something earlier"])
    assert msg == "MemoryError: out of memory"


def test_no_error_line_falls_back_to_the_last_thing_said():
    assert worker_error_message("", ["loading 1/20", "something odd"]) == "something odd"
    assert worker_error_message("", []) == "unknown error"


def test_the_sentry_title_stays_groupable():
    """Sentry groups issues by the message it is given. err_msg now contains the
    user's folder, their frame count and their free space, so sending all of it
    would give every occurrence its own group -- ten users hitting one bug would
    arrive as ten bugs. The title must be the first line only, with the full text
    carried as an extra."""
    gui = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    i = gui.find("sentry_sdk.capture_message(")
    assert i > 0, "the worker-failure Sentry call vanished"
    block = gui[max(0, i - 900):i + 300]
    assert 'set_extra("user_message"' in block, \
        "the full message must reach Sentry as an extra"
    assert "splitlines()[0]" in block, \
        "the Sentry title must be the first line only, or grouping breaks"
    call = gui[i:i + 220]
    assert "{err_msg}" not in call, (
        "the whole multi-line block must not be the Sentry title: it carries the "
        "user's path and numbers, and would split one bug into many issues")
