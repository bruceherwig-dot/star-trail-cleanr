"""Smoke tests for modules/update_check.py."""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_update_check_imports():
    import modules.update_check as m
    assert hasattr(m, "check_for_update")
    assert hasattr(m, "parse_tag")
    assert hasattr(m, "parse_local")
    assert hasattr(m, "get_download_url")


def test_parse_tag_basic():
    from modules.update_check import parse_tag
    assert parse_tag("v1.4-beta") == 1.4
    assert parse_tag("v1.5-beta") == 1.5
    assert parse_tag("v2.0") == 2.0
    assert parse_tag("v10.0-beta") == 10.0


def test_parse_tag_failures():
    from modules.update_check import parse_tag
    assert parse_tag("") is None
    assert parse_tag(None) is None
    assert parse_tag("not-a-tag") is None
    assert parse_tag(123) is None


def test_parse_local_basic():
    from modules.update_check import parse_local
    assert parse_local("1.406") == 1.406
    assert parse_local("1.400") == 1.4
    assert parse_local("  1.5  ") == 1.5


def test_parse_local_failures():
    from modules.update_check import parse_local
    assert parse_local("") is None
    assert parse_local(None) is None
    assert parse_local("abc") is None


def test_version_comparison_logic():
    """User on 1.400 installer, remote v1.5-beta -> update available."""
    from modules.update_check import parse_tag, parse_local
    assert parse_local("1.400") < parse_tag("v1.5-beta")


def test_version_comparison_dev_ahead():
    """Bruce's dev Mac at 1.406, latest tag still v1.4-beta -> NO update banner."""
    from modules.update_check import parse_tag, parse_local
    assert parse_local("1.406") > parse_tag("v1.4-beta")


def test_version_comparison_equal():
    """User on 1.400, latest tag v1.4-beta -> equal, no banner."""
    from modules.update_check import parse_tag, parse_local
    assert parse_local("1.400") == parse_tag("v1.4-beta")


def test_download_url_per_platform():
    """URL is platform-specific and points at one of the four real assets."""
    from modules.update_check import (
        get_download_url, MAC_AS_ASSET, MAC_INTEL_ASSET, WIN_ASSET, LINUX_ASSET,
    )
    url = get_download_url()
    assert url.startswith("https://github.com/")
    assert "/releases/latest/download/" in url
    assert any(url.endswith(a) for a in (MAC_AS_ASSET, MAC_INTEL_ASSET, WIN_ASSET, LINUX_ASSET))


def _patched_detect(platform_value, machine_value):
    """Run _detect_asset with sys.platform and platform.machine monkey-patched."""
    import modules.update_check as uc
    orig_platform = uc.sys.platform
    orig_machine = uc.platform.machine
    try:
        uc.sys.platform = platform_value
        uc.platform.machine = lambda: machine_value
        return uc._detect_asset()
    finally:
        uc.sys.platform = orig_platform
        uc.platform.machine = orig_machine


def test_detect_asset_mac_apple_silicon():
    from modules.update_check import MAC_AS_ASSET
    assert _patched_detect("darwin", "arm64") == MAC_AS_ASSET


def test_detect_asset_mac_intel():
    from modules.update_check import MAC_INTEL_ASSET
    assert _patched_detect("darwin", "x86_64") == MAC_INTEL_ASSET


def test_detect_asset_windows():
    from modules.update_check import WIN_ASSET
    assert _patched_detect("win32", "AMD64") == WIN_ASSET


def test_detect_asset_linux():
    from modules.update_check import LINUX_ASSET
    assert _patched_detect("linux", "x86_64") == LINUX_ASSET
    assert _patched_detect("linux2", "x86_64") == LINUX_ASSET


def test_detect_asset_unknown_falls_back_safely():
    """Unknown platform should still return a usable asset, not None or empty."""
    from modules.update_check import WIN_ASSET, MAC_AS_ASSET
    # Unknown OS -> safe fallback
    assert _patched_detect("haiku", "x86_64") == WIN_ASSET
    # Mac with unknown chip -> Apple Silicon (more common in 2026)
    assert _patched_detect("darwin", "weird-future-chip") == MAC_AS_ASSET


def test_check_for_update_bad_local_returns_none():
    """A junk local version string returns None (no crash, no network call)."""
    from modules.update_check import check_for_update
    assert check_for_update("") is None
    assert check_for_update(None) is None
    assert check_for_update("not-a-number") is None
