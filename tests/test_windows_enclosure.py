"""The Windows update feed must advertise an installer, not an archive.

Field failure, every Windows user, from the day the updater shipped until
2026-08-21. WinSparkle does not unpack anything: it hands the downloaded file to
Windows and lets the file association decide (ShellExecuteEx with no arguments,
winsparkle/src/ui.cpp). Our feed advertised StarTrailCleanRSetup.zip, so clicking
"update" downloaded a zip, opened an Explorer window, and installed nothing. No
error, no crash, nothing in Sentry -- the user simply stayed on the old version.

Proven on a clean Windows runner: installed 2.84, ran exactly what the live feed
advertised, and the machine was still on 2.84 afterwards.

Five separate "Windows updater" fixes shipped between 2026-05-01 and 2026-08-21
without anyone noticing, because every one of them tested whether the app could
FIND an update, never what happened when it ran one.

The .zip still exists and is still what the website serves: it dodges the Edge
SmartScreen gate for people downloading by hand. It is not an update.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _windows_entry():
    src = (REPO / "scripts" / "publish_appcast.py").read_text(encoding="utf-8")
    i = src.index('"key": "windows"')
    block = src[i:i + 400]
    m = re.search(r'"release_filename":\s*"([^"]+)"', block)
    assert m, "could not find the Windows release_filename in publish_appcast.py"
    return m.group(1)


def test_windows_feed_points_at_an_installer():
    fn = _windows_entry()
    assert fn.lower().endswith((".exe", ".msi")), (
        f"the Windows update feed advertises {fn!r}. Windows cannot install that "
        "by executing it, so every one-click update silently does nothing.")


def test_the_publisher_refuses_to_publish_an_archive():
    """Belt and braces: even if someone edits the table, publishing must stop."""
    src = (REPO / "scripts" / "publish_appcast.py").read_text(encoding="utf-8")
    assert '.exe", ".msi"' in src or '(".exe", ".msi")' in src, \
        "the publish-time guard on the Windows enclosure type is gone"
    assert "cannot install by executing it" in src, \
        "the guard must explain WHY, or the next person will just delete it"


def test_the_installer_itself_still_ships_both_files():
    """The website needs the zip; the updater needs the exe. Both, every build."""
    wf = (REPO / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    assert "installer/StarTrailCleanRSetup.exe" in wf, \
        "the bare installer is no longer uploaded, so the update feed will 404"
    assert "StarTrailCleanR-Windows/StarTrailCleanRSetup.exe" in wf, \
        "the bare installer is not attached to the release"
    assert "StarTrailCleanR-Windows/StarTrailCleanRSetup.zip" in wf, \
        "the zip is what the website serves; it must keep shipping"

    mirror = (REPO / "scripts" / "mirror_upload.py").read_text(encoding="utf-8")
    assert "StarTrailCleanRSetup.exe" in mirror, \
        "the feed reads from the mirror, so the installer must be mirrored too"
