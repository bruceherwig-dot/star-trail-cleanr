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


def test_workflow_uploads_setup_zip():
    text = WORKFLOW.read_text()
    assert "StarTrailCleanRSetup.zip" in text, \
        "build.yml does not reference the Setup.zip artifact"


def test_workflow_wraps_installer_in_zip():
    # Edge SmartScreen quarantines unsigned .exe downloads behind a hidden
    # Keep dropdown. Wrapping the installer in a zip skips that gate.
    # Regressing this back to a bare .exe upload would break the user UX.
    text = WORKFLOW.read_text()
    assert "Compress-Archive" in text, \
        "build.yml is missing the Compress-Archive step that wraps the installer in a zip"
    assert "StarTrailCleanRSetup.zip" in text, \
        "build.yml does not produce StarTrailCleanRSetup.zip"


def test_iss_parameterizes_output_name():
    """The .iss supports /DOutputName= override so variant installers
    (e.g. StarTrailCleanRSetup-NVIDIA) can reuse the same script."""
    text = ISS.read_text()
    assert "#ifndef OutputName" in text or "#define OutputName" in text, \
        "Installer script does not define OutputName as an overridable symbol"
    assert "OutputBaseFilename={#OutputName}" in text, \
        "OutputBaseFilename is not wired to the OutputName symbol"


def test_workflow_has_mac_intel_build():
    """Intel Mac build job present, runs on Intel runner, produces its own zip."""
    text = WORKFLOW.read_text()
    assert "build-mac-intel:" in text, "build.yml is missing the Mac Intel build job"
    assert "macos-15-intel" in text or "macos-26-intel" in text, \
        "Mac Intel job does not select an Intel macOS runner"
    assert "StarTrailCleanR-Mac-Intel.zip" in text, \
        "Mac Intel zip filename missing from workflow"


def test_workflow_has_windows_nvidia_build():
    """Windows NVIDIA build job present, uses CUDA PyTorch index, produces its own zip."""
    text = WORKFLOW.read_text()
    assert "build-windows-nvidia:" in text, "build.yml is missing the Windows NVIDIA build job"
    assert "whl/cu121" in text or "whl/cu124" in text or "whl/cu126" in text, \
        "Windows NVIDIA job does not install a CUDA PyTorch variant from pytorch.org"
    assert "StarTrailCleanRSetup-NVIDIA.zip" in text, \
        "Windows NVIDIA installer zip filename missing from workflow"
    assert "OutputName=StarTrailCleanRSetup-NVIDIA" in text, \
        "Windows NVIDIA job does not override the installer output name via /DOutputName"


def test_release_job_includes_all_four_artifacts():
    """All four installer variants must be attached to the GitHub release."""
    text = WORKFLOW.read_text()
    for needed in (
        "StarTrailCleanR-Mac-AppleSilicon.zip",
        "StarTrailCleanR-Mac-Intel.zip",
        "StarTrailCleanRSetup.zip",
        "StarTrailCleanRSetup-NVIDIA.zip",
    ):
        assert needed in text, f"Release files list is missing {needed}"


def test_build_helper_strips_unused_cuda_libs():
    """NCCL (multi-GPU comms) and nvJPEG (GPU image I/O) are safe to drop because
    the pipeline is single-GPU and reads images via OpenCV on the CPU. Ensure the
    build cleanup keeps targeting them."""
    build_helper = REPO / "build_helper.py"
    text = build_helper.read_text()
    for token in ("nccl", "nvjpeg"):
        assert token in text.lower(), \
            f"build_helper.py does not reference {token} for CUDA library cleanup"
