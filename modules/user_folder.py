"""Per-user data folder for Star Trail CleanR.

Lives outside the install bundle so downloaded models and preferences
survive app reinstalls.
- Mac: ~/Library/Application Support/StarTrailCleanR/
- Windows: %APPDATA%/StarTrailCleanR/
- Linux: ~/.config/StarTrailCleanR/
"""
import os
import sys
from pathlib import Path
from typing import Optional

APP_NAME = "StarTrailCleanR"
MODEL_FILENAME = "best.pt"
MODEL_VERSION_FILENAME = "model_version.txt"


def get_user_folder() -> Path:
    """Return the per-user data folder, creating it if missing. Never raises."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or str(Path.home()))
    else:
        base = Path.home() / ".config"
    folder = base / APP_NAME
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return folder


def get_installed_model_path() -> Path:
    """Full path where a downloaded model would live."""
    return get_user_folder() / MODEL_FILENAME


def get_installed_model_version() -> Optional[str]:
    """Return the version label of the model in the user folder, or None if nothing installed.

    Both the model file AND the version note must be present for this to return non-None.
    That way a partial download (or a version note without the file) doesn't mislead us.
    """
    model = get_installed_model_path()
    note = get_user_folder() / MODEL_VERSION_FILENAME
    if not model.is_file() or not note.is_file():
        return None
    try:
        txt = note.read_text(encoding="utf-8").strip()
        return txt or None
    except OSError:
        return None


def save_installed_model_version(version: str) -> bool:
    """Write the version label to the user folder's note file. Returns True on success."""
    note = get_user_folder() / MODEL_VERSION_FILENAME
    try:
        note.write_text((version or "").strip(), encoding="utf-8")
        return True
    except OSError:
        return False
