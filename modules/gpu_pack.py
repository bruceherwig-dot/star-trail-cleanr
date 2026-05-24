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
from typing import List, Optional, Tuple

CUDA_SUFFIX = "cu128"
PYTHON_TAG = "cp311"

# Mirrors tried in order when pytorch.org returns 403.
# Aliyun carries a complete copy of the pytorch wheel index including cu128 Windows wheels.
_MIRROR_BASES = [
    f"https://download.pytorch.org/whl/{CUDA_SUFFIX}",
    f"https://mirrors.aliyun.com/pytorch-wheels/{CUDA_SUFFIX}",
]

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
    """Return (torch_url, torchvision_url, torch_ver, tv_ver) from the primary mirror."""
    sets = get_all_download_url_sets()
    return sets[0] if sets else None


def get_all_download_url_sets() -> List[Tuple[str, str, str, str]]:
    """
    Return one (torch_url, torchvision_url, torch_ver, tv_ver) tuple per mirror,
    in priority order. The downloader tries each in sequence and stops on first success.
    Returns an empty list if the expected wheel versions cannot be determined.
    """
    torch_ver = get_expected_torch_version()
    tv_ver = get_expected_torchvision_version()
    if not torch_ver or not tv_ver:
        return []

    result = []
    for base in _MIRROR_BASES:
        torch_url = (
            f"{base}/torch-{torch_ver}%2B{CUDA_SUFFIX}"
            f"-{PYTHON_TAG}-{PYTHON_TAG}-win_amd64.whl"
        )
        tv_url = (
            f"{base}/torchvision-{tv_ver}%2B{CUDA_SUFFIX}"
            f"-{PYTHON_TAG}-{PYTHON_TAG}-win_amd64.whl"
        )
        result.append((torch_url, tv_url, torch_ver, tv_ver))
    return result


def write_version_tag(torch_ver: str) -> bool:
    """Write torch_version.txt after a successful install. Returns True on success."""
    ver_file = get_override_dir() / "torch_version.txt"
    try:
        ver_file.write_text(torch_ver, encoding="utf-8")
        return True
    except OSError:
        return False


def clear_gpu_files() -> tuple:
    """Remove all GPU pack files from the override directory.

    Uses an onerror handler + Windows shell fallback + 3-retry loop to handle
    read-only files and transient antivirus locks.

    Returns (success: bool, error_detail: str).
    """
    import shutil
    import stat
    import time
    import subprocess as _sp

    override_dir = get_override_dir()
    if not override_dir.exists():
        return True, ""

    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    def _try_remove(target: Path) -> bool:
        if not target.exists():
            return True
        for attempt in range(3):
            try:
                if target.is_dir():
                    shutil.rmtree(str(target), onerror=_onerror)
                else:
                    try:
                        os.chmod(str(target), stat.S_IWRITE)
                    except Exception:
                        pass
                    target.unlink(missing_ok=True)
                if not target.exists():
                    return True
            except Exception:
                pass
            try:
                if target.is_dir():
                    _sp.run(["cmd", "/c", "rmdir", "/s", "/q", str(target)],
                            capture_output=True, timeout=10)
                else:
                    _sp.run(["cmd", "/c", "del", "/f", "/q", str(target)],
                            capture_output=True, timeout=10)
                if not target.exists():
                    return True
            except Exception:
                pass
            if attempt < 2:
                time.sleep(0.5)
        return not target.exists()

    targets = ["torch", "torchvision", "torch_version.txt",
               "torch_pack.whl", "torchvision_pack.whl"]
    failed = [t for t in targets if not _try_remove(override_dir / t)]
    if failed:
        return False, f"Could not remove: {', '.join(failed)}\n\nFolder: {override_dir}"
    return True, ""


def chmod_extracted_files(override_dir: Path) -> None:
    """Set write permission on every .pyd and .dll in the override dir.

    Called after extraction so future cleanup is never blocked by read-only
    flags that zipfile preserves from inside the wheel archive.
    """
    import stat
    for subdir in ("torch", "torchvision"):
        target = override_dir / subdir
        if not target.is_dir():
            continue
        for root, _dirs, files in os.walk(str(target)):
            for fname in files:
                if fname.endswith((".pyd", ".dll")):
                    try:
                        os.chmod(os.path.join(root, fname),
                                 stat.S_IWRITE | stat.S_IREAD)
                    except Exception:
                        pass
