"""Windows 11 detection helper.

Regression test for v1.99: a Windows tester (Warren) reported that the
support email body said "Windows 10" even though he was on Windows 11.
Root cause: `platform.release()` returns "10" on both Windows 10 and 11
because Microsoft kept the kernel version at 10.0. The fix reads the
build number from `platform.version()` — build >= 22000 means Windows 11.
"""
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _import_windows_helper():
    """Pull `_windows_release_label` out of star_trail_cleanr.py without
    importing the whole GUI (which needs PySide6 + display)."""
    src_path = REPO / "star_trail_cleanr.py"
    src = src_path.read_text()
    start = src.index("def _windows_release_label():")
    end = src.index("\nclass CleanerWorker", start)
    snippet = src[start:end]
    namespace = {}
    exec(snippet, namespace)
    return namespace["_windows_release_label"]


def test_windows_11_detected_from_build_number():
    """Build 22000 is the first Windows 11 build."""
    helper = _import_windows_helper()
    with mock.patch("platform.version", return_value="10.0.22000"):
        assert helper() == "11"
    with mock.patch("platform.version", return_value="10.0.22631"):
        assert helper() == "11"
    with mock.patch("platform.version", return_value="10.0.26100"):
        assert helper() == "11"


def test_windows_10_falls_back_to_release():
    """Below build 22000 we trust `platform.release()`."""
    helper = _import_windows_helper()
    with mock.patch("platform.version", return_value="10.0.19045"), \
         mock.patch("platform.release", return_value="10"):
        assert helper() == "10"


def test_malformed_version_string_falls_back_safely():
    """If platform.version() returns something we can't parse, fall back
    to platform.release() rather than crashing."""
    helper = _import_windows_helper()
    with mock.patch("platform.version", return_value="who-knows"), \
         mock.patch("platform.release", return_value="10"):
        assert helper() == "10"


def test_callsites_use_helper_not_bare_release():
    """Both the support email body and the run summary must call the
    helper. If either regresses to platform.release(), Warren's bug
    is back."""
    src = (REPO / "star_trail_cleanr.py").read_text()
    # Two replacement spots: support email body + run summary.
    assert src.count("Windows {_windows_release_label()}") >= 1, (
        "support email body no longer uses the helper"
    )
    assert src.count("_windows_release_label()") >= 2, (
        "expected at least two callers of _windows_release_label() — "
        "the support email body and the run summary"
    )
