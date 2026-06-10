"""
GPU override pack management for Windows NVIDIA users.

Plain-English summary
---------------------
Star Trail CleanR ships with a CPU-only build of PyTorch (the math library that
runs the trail-detection AI model). That works everywhere but is slow. Windows
users who have an NVIDIA graphics card can install an optional "GPU pack" to make
detection run much faster on their GPU instead of the CPU.

The GPU pack is simply the CUDA-enabled (NVIDIA-accelerated) PyTorch and
torchvision wheels (pre-built Python packages) downloaded from the internet and
unzipped into a persistent folder on the user's machine. That folder lives
OUTSIDE the app bundle, so reinstalling or updating the app never deletes it.
At runtime the app adds that folder to its import path so the GPU build is loaded
in preference to the bundled CPU build.

What this file is responsible for
---------------------------------
This module is the bookkeeping/plumbing layer for that GPU pack. It does NOT do
the downloading, unzipping, or path-injection itself (other code does that). It
provides the helpers those steps rely on:
  - where the override folder lives (`get_override_dir`)
  - whether a pack is already installed and which version (`is_installed`,
    `get_installed_version`)
  - which exact wheel versions this app build expects, read from files baked into
    the frozen app bundle (`get_expected_torch_version`, etc.)
  - the precise download URLs to fetch those wheels, with mirror fallbacks
    (`get_download_urls`, `get_all_download_url_sets`)
  - recording a successful install (`write_version_tag`)
  - permission fix-ups and a robust delete for cleanup/uninstall
    (`chmod_extracted_files`, `clear_gpu_files`)

Override dir: %LOCALAPPDATA%\StarTrailCleanR\gpu_override\
  torch/              <- extracted from torch CUDA wheel
  torchvision/        <- extracted from torchvision CUDA wheel
  torch_version.txt   <- written after a successful install
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# CUDA build identifier used in PyTorch wheel filenames and URLs.
# "cu128" = the CUDA 12.8 build of PyTorch. This must match the CUDA version the
# bundled expected-version files were generated against, or the URLs won't resolve.
CUDA_SUFFIX = "cu128"
# Python ABI tag in the wheel filenames. "cp311" = CPython 3.11. The app must be
# frozen against Python 3.11 for these wheels to be importable.
PYTHON_TAG = "cp311"

# Download hosts tried in order. The downloader walks this list and stops on the
# first that succeeds.
# Why a second host: the official pytorch.org index sometimes returns HTTP 403
# (e.g. from certain regions/networks). Aliyun carries a complete copy of the
# pytorch wheel index, including the cu128 Windows wheels, and serves as fallback.
_MIRROR_BASES = [
    f"https://download.pytorch.org/whl/{CUDA_SUFFIX}",
    f"https://mirrors.aliyun.com/pytorch-wheels/{CUDA_SUFFIX}",
]

# Folder-name pieces used to build the persistent override path under LOCALAPPDATA.
_APP_DIR = "StarTrailCleanR"
_OVERRIDE_DIR = "gpu_override"


def get_override_dir() -> Path:
    """Return the path to the persistent GPU-pack folder on Windows.

    The folder is %LOCALAPPDATA%\\StarTrailCleanR\\gpu_override\\, a per-user
    location that survives app reinstalls and updates, which is why the GPU pack
    is stored there rather than inside the app bundle.

    If the LOCALAPPDATA environment variable is missing for some reason, falls
    back to the conventional AppData\\Local path under the user's home directory.

    Returns the folder as a pathlib.Path. (The folder itself may or may not exist
    yet; this just computes where it should be.)
    """
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if not localappdata:
        localappdata = str(Path.home() / "AppData" / "Local")
    return Path(localappdata) / _APP_DIR / _OVERRIDE_DIR


def is_installed() -> bool:
    """Return True if a GPU pack appears to be installed.

    "Installed" means the override folder exists AND contains the
    torch_version.txt tag file. The tag file is written only at the very end of a
    successful install (see `write_version_tag`), so its presence is the signal
    that the install completed rather than failing partway through.
    """
    d = get_override_dir()
    return d.is_dir() and (d / "torch_version.txt").is_file()


def get_installed_version() -> Optional[str]:
    """Return the torch version recorded for the currently installed GPU pack.

    Reads torch_version.txt from the override folder and returns the bare version
    number (e.g. "2.6.0"). The "+cu128" CUDA suffix, if present, is stripped off
    by splitting on "+" and keeping the part before it.

    Returns None if no pack is installed (file missing) or the file can't be read.
    """
    ver_file = get_override_dir() / "torch_version.txt"
    if not ver_file.is_file():
        return None
    try:
        # split("+")[0] drops the build metadata, e.g. "2.6.0+cu128" -> "2.6.0".
        return ver_file.read_text(encoding="utf-8").strip().split("+")[0]
    except OSError:
        return None


def _read_bundled_file(filename: str) -> Optional[str]:
    """Read a single-line text file that was baked into the frozen app bundle.

    Internal helper. PyInstaller-frozen apps unpack their bundled data files into
    a temporary folder whose path is exposed as `sys._MEIPASS`. This looks for
    `filename` there, reads it, strips whitespace, and drops any "+suffix" build
    metadata (same convention as `get_installed_version`).

    `filename` is the data file's name as bundled (e.g.
    "stc_expected_torch_version.txt").

    Returns the cleaned string, or None when:
      - not running frozen (no `sys._MEIPASS`, e.g. live Python source), or
      - the file isn't present, or
      - it can't be read.
    """
    if hasattr(sys, "_MEIPASS"):
        path = Path(sys._MEIPASS) / filename
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip().split("+")[0]
            except OSError:
                pass
    return None


def get_expected_torch_version() -> Optional[str]:
    """Return the torch version this app build was made for, or None.

    This is the version the GPU pack downloader should fetch so the CUDA torch
    matches the rest of the bundled app. It's read from a file baked into the
    bundle at build time. Returns None when running from live source (the file is
    only present in a frozen build).
    """
    return _read_bundled_file("stc_expected_torch_version.txt")


def get_expected_torchvision_version() -> Optional[str]:
    """Return the torchvision version this app build was made for, or None.

    The torchvision counterpart of `get_expected_torch_version`; the two must be a
    compatible pair. Read from a bundled build-time file; None when running from
    live source.
    """
    return _read_bundled_file("stc_expected_torchvision_version.txt")


def get_download_urls() -> Optional[Tuple[str, str, str, str]]:
    """Return the download info for the PRIMARY (first) mirror only.

    Convenience wrapper around `get_all_download_url_sets` that hands back just the
    top-priority entry: (torch_url, torchvision_url, torch_ver, tv_ver).

    Returns None if the expected wheel versions can't be determined (e.g. running
    from live source where the bundled version files are absent).
    """
    sets = get_all_download_url_sets()
    return sets[0] if sets else None


def get_all_download_url_sets() -> List[Tuple[str, str, str, str]]:
    """
    Build the full set of download URLs, one per mirror, in priority order.

    Each list entry is a tuple (torch_url, torchvision_url, torch_ver, tv_ver):
      - torch_url / torchvision_url: direct links to the CUDA Windows wheels on
        that mirror,
      - torch_ver / tv_ver: the bare version numbers (same for every entry; they
        come from the bundle, only the host differs between entries).

    The downloader is expected to try the entries in order and stop on the first
    that succeeds (this is what gives us the pytorch.org -> Aliyun fallback).

    Returns an empty list if the expected wheel versions cannot be determined,
    which means there is nothing to download (e.g. running from live source).
    """
    torch_ver = get_expected_torch_version()
    tv_ver = get_expected_torchvision_version()
    if not torch_ver or not tv_ver:
        return []

    result = []
    for base in _MIRROR_BASES:
        # Wheel filename convention, e.g.:
        #   torch-2.6.0+cu128-cp311-cp311-win_amd64.whl
        # The "+" between version and CUDA suffix must be URL-encoded as %2B, and
        # the cp311 ABI tag appears twice (Python tag and ABI tag).
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
    """Stamp the override folder as a completed install.

    Writes the installed torch version into torch_version.txt in the override
    folder. This file is the "install succeeded" marker that `is_installed` and
    `get_installed_version` look for, so it should be written LAST, only after the
    wheels have been extracted successfully.

    `torch_ver` is the version string to record (e.g. "2.6.0" or "2.6.0+cu128").

    Returns True if the file was written, False on any OS error (e.g. permissions).
    """
    ver_file = get_override_dir() / "torch_version.txt"
    try:
        ver_file.write_text(torch_ver, encoding="utf-8")
        return True
    except OSError:
        return False


def clear_gpu_files() -> tuple:
    """Delete the installed GPU pack (used to uninstall or re-install cleanly).

    Removes the known GPU-pack items from the override folder: the extracted
    torch/ and torchvision/ subfolders, the torch_version.txt tag, and any
    leftover downloaded .whl files. It deliberately removes the contents rather
    than the override folder itself.

    On Windows, GPU files are stubborn to delete: wheel archives mark .pyd/.dll
    files read-only, and antivirus/Windows can briefly lock DLLs that were just
    loaded. To beat that, deletion is hardened three ways:
      1. an `onerror` handler that clears the read-only flag and retries,
      2. a fallback to the native Windows shell commands (rmdir / del), and
      3. a 3-attempt loop with a short pause between tries for transient locks.

    Returns a tuple (success: bool, error_detail: str). On failure, error_detail
    lists which items could not be removed and the folder they're in; on success
    it is an empty string.
    """
    import shutil
    import stat
    import time
    import subprocess as _sp

    override_dir = get_override_dir()
    # Nothing to do if the folder was never created.
    if not override_dir.exists():
        return True, ""

    def _onerror(func, path, exc_info):
        # shutil.rmtree calls this when it can't remove something. The usual
        # cause here is a read-only file: clear the read-only bit and retry the
        # operation (func) on that same path. Swallow anything still failing so
        # the outer retry/shell-fallback logic gets its turn.
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    def _try_remove(target: Path) -> bool:
        # Try hard to delete one file or folder. Returns True once it's gone.
        if not target.exists():
            return True
        for attempt in range(3):
            # Attempt 1 of this pass: the normal Python delete path.
            try:
                if target.is_dir():
                    shutil.rmtree(str(target), onerror=_onerror)
                else:
                    # Clear read-only first so unlink() can proceed.
                    try:
                        os.chmod(str(target), stat.S_IWRITE)
                    except Exception:
                        pass
                    target.unlink(missing_ok=True)
                if not target.exists():
                    return True
            except Exception:
                pass
            # Attempt 2 of this pass: fall back to the native Windows shell, which
            # can sometimes remove files Python's APIs can't (force flags).
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
            # Still here: likely a transient lock (antivirus, recent DLL load).
            # Wait briefly and loop, but don't sleep after the final attempt.
            if attempt < 2:
                time.sleep(0.5)
        return not target.exists()

    # The complete set of items a GPU pack can leave behind. The two *_pack.whl
    # names are downloaded wheel files that may linger if a previous install was
    # interrupted before/after extraction.
    targets = ["torch", "torchvision", "torch_version.txt",
               "torch_pack.whl", "torchvision_pack.whl"]
    failed = [t for t in targets if not _try_remove(override_dir / t)]
    if failed:
        return False, f"Could not remove: {', '.join(failed)}\n\nFolder: {override_dir}"
    return True, ""


def chmod_extracted_files(override_dir: Path) -> None:
    """Make the extracted GPU binaries writable so a later delete won't be blocked.

    Run this right after unzipping the wheels. Python's zipfile preserves the
    read-only permission bits stored inside the wheel archive, which would later
    make `clear_gpu_files` fight to delete those files. Walking the torch/ and
    torchvision/ folders and clearing read-only on every .pyd (Windows compiled
    Python extension) and .dll (native library) up front avoids that fight.

    `override_dir` is the GPU-pack folder (as returned by `get_override_dir`).

    Returns nothing. Best-effort: any individual chmod failure is ignored so one
    locked file doesn't abort the whole pass.
    """
    import stat
    # Only torch/ and torchvision/ are extracted from wheels; nothing else here
    # carries the read-only binaries.
    for subdir in ("torch", "torchvision"):
        target = override_dir / subdir
        if not target.is_dir():
            continue
        for root, _dirs, files in os.walk(str(target)):
            for fname in files:
                # Only the compiled binaries (.pyd/.dll) get the read-only flag
                # from inside the wheel; plain .py files don't need touching.
                if fname.endswith((".pyd", ".dll")):
                    try:
                        os.chmod(os.path.join(root, fname),
                                 stat.S_IWRITE | stat.S_IREAD)
                    except Exception:
                        pass
