"""Verify the Inno Setup installer config is in place and wired into the build.

Born from the v1.1-beta lesson: shipping without thinking like a fresh Windows
machine. The raw zip forced testers through a 60,000-file Explorer extract that
took over a day. The installer solves that, but only if both the .iss script
and the workflow step exist together. These tests catch a half-shipped fix."""
from pathlib import Path

REPO = Path(__file__).parent.parent
ISS = REPO / "installer" / "StarTrailCleanR.iss"
WORKFLOW = REPO / ".github" / "workflows" / "build.yml"


def test_iss_file_exists():
    assert ISS.exists(), f"Inno Setup script missing: {ISS}"


def test_iss_references_pyinstaller_output():
    text = ISS.read_text()
    assert "dist\\StarTrailCleanR\\*" in text, \
        "Installer script does not reference dist\\StarTrailCleanR\\* as source"


def test_iss_uses_version_placeholder():
    text = ISS.read_text()
    assert "{#AppVersion}" in text, "Installer script does not use {#AppVersion} placeholder"


def test_iss_references_app_icon():
    text = ISS.read_text()
    assert "StarTrailCleanR.ico" in text, "Installer script does not reference the app icon"


def test_workflow_invokes_inno_setup():
    text = WORKFLOW.read_text().lower()
    assert "innosetup" in text or "iscc" in text, \
        "build.yml does not install or invoke Inno Setup"


def test_workflow_uploads_setup_exe():
    text = WORKFLOW.read_text()
    assert "StarTrailCleanRSetup" in text, \
        "build.yml does not reference the Setup.exe artifact"
