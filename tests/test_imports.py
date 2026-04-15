"""Every pipeline module must import without error."""
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _import(name):
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


def test_modules_hot_pixels_imports():
    m = _import("modules.hot_pixels")
    assert hasattr(m, "fix_hot_pixels")
    assert hasattr(m, "build_hot_pixel_map")


def test_modules_repair_imports():
    _import("modules.repair")


def test_modules_detect_trails_imports():
    _import("modules.detect_trails")


def test_astro_clean_v5_module_imports():
    spec = importlib.util.spec_from_file_location("astro_clean_v5", REPO / "astro_clean_v5.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
