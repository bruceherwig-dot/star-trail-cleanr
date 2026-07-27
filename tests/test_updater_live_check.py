"""Smoke tests for the Windows updater live-check gate.

Why it exists: the existing updater gate proves the WinSparkle engine LOADS
inside the frozen build. It never proved a check SUCCEEDS. Three Windows testers
in a row hit "An error occurred in retrieving update information", each was told
it was their security software, and no Windows install has ever been observed
updating itself. This gate runs a real check against the live feed on a clean
cloud Windows machine, where nothing is blocking anything, so the answer cannot
keep being guessed at.

These tests only check the wiring is present, since the check itself needs
Windows and a network. Offline.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_app_has_the_live_check_mode():
    src = (REPO / "star_trail_cleanr.py").read_text()
    assert "STC_UPDATER_CHECK" in src, "the built app must expose the live-check mode"
    block = src[src.index("STC_UPDATER_CHECK"):][:2000]
    assert "check_for_updates_quiet" in block, "it must run a real check"
    for handler in ("set_quiet_handlers", "set_error_handler"):
        assert handler in block, f"it must listen for the {handler} outcome"


def test_exit_codes_distinguish_the_outcomes():
    """A Windows GUI program detaches from the console, so printed output never
    reaches the CI log. The exit code is the only channel that survives, and the
    interesting cases must not collapse into one number."""
    src = (REPO / "star_trail_cleanr.py").read_text()
    m = re.search(r'"found":\s*(\d+),\s*"up-to-date":\s*(\d+),\s*"error":\s*(\d+)'
                  r'\s*\}\s*\.get\(\s*_outcome\s*,\s*(\d+)\s*\)', src)
    assert m, "expected the outcome-to-exit-code mapping"
    found, uptodate, error, timeout = (int(g) for g in m.groups())
    assert found == 0 and uptodate == 0, "a successful fetch must exit 0"
    assert len({error, timeout}) == 2, "error and timeout must be tellable apart"
    assert 0 not in (error, timeout), "failures must not exit 0"


def test_ci_runs_the_live_check_on_windows():
    workflow = (REPO / ".github" / "workflows" / "build.yml").read_text()
    assert "STC_UPDATER_CHECK" in workflow, "CI must run the live check"
    assert "Updater live-check gate" in workflow, "the step needs a findable name"
    # It reports rather than blocks for now, on purpose: the condition predates
    # the gate. If someone makes it blocking, this reminder goes with it.
    step = workflow[workflow.index("Updater live-check gate"):][:1600]
    assert ("continue-on-error: true" in step) or ("exit 1" in step), (
        "the step must either report deliberately or block deliberately")


def test_the_live_check_uses_the_feed_users_actually_read():
    """Checking a different feed than shipped users read would prove nothing."""
    src = (REPO / "star_trail_cleanr.py").read_text()
    m = re.search(r'appcast_url="([^"]+)"', src)
    assert m, "could not find the Windows feed URL"
    assert m.group(1).startswith("https://"), m.group(1)
    assert "appcast-windows.xml" in m.group(1), m.group(1)
