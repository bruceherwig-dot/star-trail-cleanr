"""Verify everything the frozen app needs at runtime actually exists in the repo.

Born from the v1.0-beta incident: the YOLO model path was hardcoded to Bruce's
Mac, so the app crashed on every other machine. These tests catch that class of
bug before it ships."""
from pathlib import Path

REPO = Path(__file__).parent.parent
ASSETS = REPO / "assets"


def test_yolo_model_exists():
    """The AI model must be in assets/ so the build can bundle it."""
    model = ASSETS / "best.pt"
    assert model.exists(), f"YOLO model missing: {model}"
    assert model.stat().st_size > 1_000_000, "best.pt looks too small to be a real model"


def test_about_photo_exists():
    """The About tab references bruce_silhouette.jpg."""
    photo = ASSETS / "bruce_silhouette.jpg"
    assert photo.exists(), f"About photo missing: {photo}"


def test_app_icon_exists():
    """The GUI header references icon_1024.png."""
    icon = ASSETS / "icon_1024.png"
    assert icon.exists(), f"App icon missing: {icon}"


def test_processing_script_exists():
    """The GUI launches astro_clean_v5.py as a subprocess."""
    script = REPO / "astro_clean_v5.py"
    assert script.exists(), f"Processing script missing: {script}"


def test_version_file_exists():
    """The GUI reads version.txt at startup."""
    vf = REPO / "version.txt"
    assert vf.exists(), f"version.txt missing: {vf}"
    assert vf.read_text().strip(), "version.txt is empty"


def test_build_helper_bundles_assets():
    """build_helper.py must include assets/ in the PyInstaller data list."""
    bh = REPO / "build_helper.py"
    assert bh.exists(), "build_helper.py missing"
    text = bh.read_text()
    assert "assets" in text, "build_helper.py does not reference assets/ folder"


def test_build_helper_bundles_model():
    """build_helper.py must explicitly bundle best.pt."""
    bh = REPO / "build_helper.py"
    text = bh.read_text()
    assert "best.pt" in text, "build_helper.py does not bundle best.pt"
