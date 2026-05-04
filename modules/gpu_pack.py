"""
GPU override pack management for Windows NVIDIA users.

The GPU pack is CUDA-enabled PyTorch wheels extracted into a persistent
folder that app updates never touch.

Override dir: %LOCALAPPDATA%\StarTrailCleanR\gpu_override\
  torch/              <- extracted from torch CUDA wheel
  torchvision/        <- extracted from torchvision CUDA wheel
  torch_version.txt   <- written after a successful install
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

CUDA_SUFFIX = "cu128"
PYTHON_TAG = "cp311"

_APP_DIR = "StarTrailCleanR"
_OVERRIDE_DIR = "gpu_override"


def get_override_dir() -> Path:
    """Return %LOCALAPPDATA%\\StarTrailCleanR\\gpu_override\\ on Windows."""
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if not localappdata:
        localappdata = str(Path.home() / "AppData" / "Local")
    return Path(localappdata) / _APP_DIR / _OVERRIDE_DIR


def is_installed() -> bool:
    """True if the override folder exists and has a version tag file."""
    d = get_override_dir()
    return d.is_dir() and (d / "torch_version.txt").is_file()


def get_installed_version() -> Optional[str]:
    """Return the installed torch version string, or None."""
    ver_file = get_override_dir() / "torch_version.txt"
    if not ver_file.is_file():
        return None
    try:
        return ver_file.read_text(encoding="utf-8").strip().split("+")[0]
    except OSError:
        return None


def _read_bundled_file(filename: str) -> Optional[str]:
    if hasattr(sys, "_MEIPASS"):
        path = Path(sys._MEIPASS) / filename
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip().split("+")[0]
            except OSError:
                pass
    return None


def get_expected_torch_version() -> Optional[str]:
    return _read_bundled_file("stc_expected_torch_version.txt")


def get_expected_torchvision_version() -> Optional[str]:
    return _read_bundled_file("stc_expected_torchvision_version.txt")


def get_download_urls() -> Optional[Tuple[str, str, str, str]]:
    """
    Return (torch_url, torchvision_url, torch_ver, tv_ver) for the CUDA wheels
    that match this app build, or None if versions cannot be determined.
    """
    torch_ver = get_expected_torch_version()
    tv_ver = get_expected_torchvision_version()
    if not torch_ver or not tv_ver:
        return None

    base = f"https://download.pytorch.org/whl/{CUDA_SUFFIX}"
    torch_url = (
        f"{base}/torch-{torch_ver}%2B{CUDA_SUFFIX}"
        f"-{PYTHON_TAG}-{PYTHON_TAG}-win_amd64.whl"
    )
    tv_url = (
        f"{base}/torchvision-{tv_ver}%2B{CUDA_SUFFIX}"
        f"-{PYTHON_TAG}-{PYTHON_TAG}-win_amd64.whl"
    )
    return torch_url, tv_url, torch_ver, tv_ver


def write_version_tag(torch_ver: str) -> bool:
    """Write torch_version.txt after a successful install. Returns True on success."""
    ver_file = get_override_dir() / "torch_version.txt"
    try:
        ver_file.write_text(torch_ver, encoding="utf-8")
        return True
    except OSError:
        return False
