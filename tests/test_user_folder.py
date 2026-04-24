"""Smoke tests for modules/user_folder.py."""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_user_folder_imports():
    import modules.user_folder as m
    for name in ("get_user_folder", "get_installed_model_path",
                 "get_installed_model_version", "save_installed_model_version",
                 "APP_NAME", "MODEL_FILENAME", "MODEL_VERSION_FILENAME"):
        assert hasattr(m, name), f"missing {name}"


def test_get_user_folder_creates_and_returns_path():
    from modules.user_folder import get_user_folder, APP_NAME
    folder = get_user_folder()
    assert folder.exists()
    assert folder.is_dir()
    assert folder.name == APP_NAME


def test_get_installed_model_path_is_in_user_folder():
    from modules.user_folder import get_user_folder, get_installed_model_path, MODEL_FILENAME
    path = get_installed_model_path()
    assert path.name == MODEL_FILENAME
    assert path.parent == get_user_folder()


def test_get_installed_model_version_no_crash():
    """Never raises on missing or partial state. Returns None or a string."""
    from modules.user_folder import get_installed_model_version
    result = get_installed_model_version()
    assert result is None or isinstance(result, str)
