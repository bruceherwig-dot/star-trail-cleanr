"""Per-user data folder for Star Trail CleanR.

What this file is for
---------------------
Star Trail CleanR ships as a packaged desktop app. The app's program files get
wiped and replaced every time the user reinstalls or updates. Anything we want
to KEEP across those reinstalls (most importantly, the trail-detection model the
app downloads after install, plus a small note recording which model version is
on disk) has to live somewhere OUTSIDE the install bundle.

This module owns that "outside" location. It knows the right per-user data
folder for each operating system, makes sure the folder exists, and provides the
handful of paths and read/write helpers the rest of the app uses to store and
look up the downloaded model and its version label.

Where the folder lives (the OS-standard spot for app data on each platform):
- Mac:     ~/Library/Application Support/StarTrailCleanR/
- Windows: %APPDATA%/StarTrailCleanR/   (falls back to the home folder if APPDATA is unset)
- Linux:   ~/.config/StarTrailCleanR/

How it fits into the app
------------------------
The model-update / model-download code calls these helpers to decide where to
save a freshly downloaded model, to record its version, and to check whether an
up-to-date model is already present so it can skip re-downloading.

Design note: the functions here are deliberately failure-tolerant. They never
raise on a filesystem problem (e.g. a read-only home directory). Instead they
return a best-effort path, None, or False, so a folder/permissions hiccup can
never crash the app at startup.
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Name of the per-user folder created under each OS's app-data location.
APP_NAME = "StarTrailCleanR"
# File name of the downloaded trail-detection model stored in the user folder.
MODEL_FILENAME = "best.pt"
# File name of the small text note recording which model version is installed.
MODEL_VERSION_FILENAME = "model_version.txt"


def get_user_folder() -> Path:
    """Return the per-user data folder, creating it if it does not exist yet.

    Picks the correct OS-standard app-data location based on the platform the
    app is running on, then ensures our app's subfolder exists inside it.

    Returns:
        A Path to the StarTrailCleanR data folder (e.g.
        ~/Library/Application Support/StarTrailCleanR on Mac).

    Why it exists / when used:
        Every other helper here builds on this — it is the single source of
        truth for where per-user files live, so the model path and version note
        always agree on one folder.

    Never raises: if the folder can't be created (e.g. permissions, read-only
    home), we swallow the error and still return the intended path. Callers
    decide what to do if a later read/write then fails.
    """
    # Choose the conventional app-data base directory for the current OS.
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        # %APPDATA% is the standard Windows roaming app-data dir; if it's
        # somehow unset, fall back to the user's home folder.
        base = Path(os.environ.get("APPDATA") or str(Path.home()))
    else:
        # Linux / other Unix: XDG-style config directory.
        base = Path.home() / ".config"
    folder = base / APP_NAME
    try:
        # parents=True creates the base dir too if needed; exist_ok=True makes
        # repeated calls a no-op rather than an error.
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Deliberately ignored so this never crashes the app. We still return
        # the intended path below; a real problem surfaces later as a failed
        # read/write that the caller handles gracefully.
        pass
    return folder


def get_installed_model_path() -> Path:
    """Return the full path where a downloaded model file lives (or would live).

    Returns:
        The path to ``best.pt`` inside the per-user folder. Note this just builds
        the path; it does NOT check whether the file actually exists.

    Why it exists / when used:
        The model-download code uses this to know where to save a fetched model,
        and other code uses it to locate an already-downloaded model to load.
    """
    return get_user_folder() / MODEL_FILENAME


def get_installed_model_version() -> Optional[str]:
    """Return the version label of the model installed in the user folder.

    Returns:
        The trimmed version string (e.g. "v4") if a complete model install is
        present, otherwise None.

    Why it exists / when used:
        The model-update check compares this against the latest available version
        to decide whether a new download is needed. Returning None here means
        "treat it as nothing usable installed, (re)download."

    Guard against half-installed state: BOTH the model file AND the version note
    must exist before we trust the version. If only one is present (a partial
    download, or a stale note with no model), we report None rather than a
    version that can't be backed by a real model file.
    """
    model = get_installed_model_path()
    note = get_user_folder() / MODEL_VERSION_FILENAME
    # Require both pieces; either one missing means "not a real install".
    if not model.is_file() or not note.is_file():
        return None
    try:
        txt = note.read_text(encoding="utf-8").strip()
        # An empty/whitespace-only note counts as "no version recorded".
        return txt or None
    except OSError:
        # Unreadable note -> behave as if nothing is installed.
        return None


def save_installed_model_version(version: str) -> bool:
    """Record the version label of the model now present in the user folder.

    Args:
        version: The version string to store (e.g. "v4"). It is trimmed before
            writing; a None or empty value results in an empty note file.

    Returns:
        True if the note was written successfully, False if a filesystem error
        prevented it (the function never raises).

    Why it exists / when used:
        Called right after a model download succeeds so a later
        ``get_installed_model_version()`` can report what's on disk and the
        update check can skip redundant downloads.
    """
    note = get_user_folder() / MODEL_VERSION_FILENAME
    try:
        note.write_text((version or "").strip(), encoding="utf-8")
        return True
    except OSError:
        # Caller gets False instead of an exception so a failed write can't crash
        # the app; worst case the version simply isn't recorded.
        return False
