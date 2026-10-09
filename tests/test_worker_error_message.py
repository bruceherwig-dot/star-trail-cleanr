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


# ── console notices are not errors (Sentry 2026-10-07, Windows 11, 102 MP frames) ──

NOTICE_WIN = "Press Ctrl-Break to quit"
NOTICE_MAC = "Press Ctrl-C to quit"


def test_a_console_notice_alone_does_not_hide_the_engines_own_error():
    """The report was titled "Worker exited 1: Press Ctrl-Break to quit". The engine
    had printed its own ERROR block on stdout, but the error stream held the console
    notice, which won the 'stderr wins' rule."""
    engine = ["Batch 2/2", "ERROR: Not enough memory for this batch.",
              "  Each frame is 11648 x 8736.", "  Close other programs and try again."]
    for notice in (NOTICE_WIN, NOTICE_MAC, NOTICE_WIN + "\n"):
        msg = worker_error_message(notice, engine)
        assert msg.startswith("ERROR: Not enough memory for this batch."), (notice, msg)
        assert "11648 x 8736" in msg and "Close other programs" in msg


def test_a_console_notice_after_a_real_crash_does_not_hide_the_exception():
    """A real Python crash ends its traceback with the exception, and the notice
    can come after it. The exception is what the user and Sentry need."""
    tb = ("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n"
          "MemoryError: Unable to allocate 305 MiB for an array\n")
    for notice in (NOTICE_WIN, NOTICE_MAC):
        msg = worker_error_message(tb + notice + "\n", ["Detecting..."])
        assert msg == "MemoryError: Unable to allocate 305 MiB for an array", msg
    # and the notice may also come BEFORE the traceback
    msg = worker_error_message(NOTICE_WIN + "\n" + tb, [])
    assert msg == "MemoryError: Unable to allocate 305 MiB for an array", msg


def test_only_a_console_notice_and_nothing_else_still_says_something_honest():
    assert worker_error_message(NOTICE_WIN, []) == "unknown error"
    assert worker_error_message(NOTICE_WIN, ["loading 1/20", "something odd"]) == "something odd"
    # a console notice on stdout is not the "last thing said" either
    assert worker_error_message("", ["something odd", NOTICE_MAC]) == "something odd"


def test_a_real_line_that_merely_mentions_ctrl_c_is_kept():
    """Only a line that is JUST the console hint is dropped."""
    line = "RuntimeError: stopped: Press Ctrl-C to quit was ignored by the worker"
    assert worker_error_message(line, ["loading 1/20: a.jpg"]) == line


# ── harmless warnings must not hide the engine's own error (Sentry 2026-10-09) ──

# The REAL event from v1.992 on a Mac (Jims-MacBook-Pro, batch 12 of 48): the library's
# deprecation warning on the error stream, and the engine's own explanation of a drive that
# disconnected mid-run on stdout. The user and Sentry were shown the warning's source line.
REAL_MAC_WARNING = (
    "sahi/utils/torch_utils.py:80: UserWarning: 'has_mps' is deprecated, please use "
    "'torch.backends.mps.is_built()'\n"
    "  elif mps and getattr(torch, \"has_mps\", False) and torch.backends.mps.is_available():"
    "  # prefer MPS if available\n")
REAL_MAC_STDOUT = [
    "Step 1 complete - 20/20 frames have trails",
    "Step 2 - repairing frames (skipping first/last 0)",
    "ERROR: Cannot write cleaned frame to:",
    "/Volumes/Laptop Dropbox/Hoosiers Nest Astrophotography 20261008/191_PANA JPG/cleaned",
    "The output folder may be on a read-only drive, synced by OneDrive, or a file there may "
    "be open in another app. Pick a different output folder and try again.",
    "(Detail: OSError: [Errno 6] Device not configured: '/Volumes/Laptop Dropbox/Hoosiers "
    "Nest Astrophotography 20261008/191_PANA JPG/cleaned/PETE0221.jpg')",
]


def test_the_real_mac_event_shows_the_engines_error_not_the_warning_line():
    msg = worker_error_message(REAL_MAC_WARNING, REAL_MAC_STDOUT)
    assert msg.startswith("ERROR: Cannot write cleaned frame to:"), msg
    assert "Device not configured" in msg and "PETE0221.jpg" in msg, \
        "the detail that names the cause must reach the user"
    assert "has_mps" not in msg, "the library's warning line must not be the message"


def test_a_warning_never_outranks_the_engines_error_but_a_traceback_does():
    engine = ["ERROR: Not enough memory for this batch.", "  Close other programs."]
    warn = "UserWarning: something is deprecated\n  some_source_line_of_the_library()\n"
    assert worker_error_message(warn, engine).startswith("ERROR: Not enough memory")
    # a warning, then a real crash: the crash is what happened
    tb = warn + ("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n"
                 "ValueError: bad frame\n")
    assert worker_error_message(tb, engine) == "ValueError: bad frame"


def test_without_an_engine_error_the_last_real_error_line_is_still_reported():
    """A native crash prints its own text on the error stream and no ERROR block
    on stdout: that text is all there is, and it must still come through."""
    native = "UserWarning: x\n  src()\nforrtl: severe (157): Program Exception - access violation\n"
    assert worker_error_message(native, ["loading 1/20: a.jpg"]).startswith("forrtl: severe (157)")

