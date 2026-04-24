"""Smoke tests for modules/model_update.py."""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_model_update_imports():
    import modules.model_update as m
    for name in ("check_for_model_update", "parse_model_tag",
                 "parse_release_body", "find_model_asset_url",
                 "local_model_version", "BUNDLED_MODEL_VERSION"):
        assert hasattr(m, name), f"missing {name}"


def test_parse_model_tag_basic():
    from modules.model_update import parse_model_tag
    assert parse_model_tag("model-v1") == 1.0
    assert parse_model_tag("model-v2") == 2.0
    assert parse_model_tag("model-v10") == 10.0
    assert parse_model_tag("model-v2.5") == 2.5


def test_parse_model_tag_failures():
    from modules.model_update import parse_model_tag
    assert parse_model_tag("") is None
    assert parse_model_tag(None) is None
    assert parse_model_tag("v1.0-beta") is None  # app tag, not model tag
    assert parse_model_tag("model-") is None
    assert parse_model_tag(42) is None


def test_parse_release_body_summary_and_credits():
    from modules.model_update import parse_release_body
    body = "Better dim trail detection.\n\nCredits: gkyle, Silvana, Cheryl"
    result = parse_release_body(body)
    assert result["summary"] == "Better dim trail detection."
    assert result["credits"] == "gkyle, Silvana, Cheryl"


def test_parse_release_body_summary_only():
    from modules.model_update import parse_release_body
    result = parse_release_body("Better dim trail detection.")
    assert result["summary"] == "Better dim trail detection."
    assert result["credits"] == ""


def test_parse_release_body_empty_and_none():
    from modules.model_update import parse_release_body
    for b in ("", None):
        r = parse_release_body(b)
        assert r["summary"] == ""
        assert r["credits"] == ""


def test_parse_release_body_credits_case_insensitive():
    from modules.model_update import parse_release_body
    r = parse_release_body("First line\ncredits: foo, bar")
    assert r["credits"] == "foo, bar"


def test_find_model_asset_url_picks_pt():
    from modules.model_update import find_model_asset_url
    assets = [
        {"name": "notes.txt", "browser_download_url": "http://example.com/notes.txt"},
        {"name": "best.pt", "browser_download_url": "http://example.com/best.pt"},
    ]
    assert find_model_asset_url(assets) == "http://example.com/best.pt"


def test_find_model_asset_url_none_when_no_pt():
    from modules.model_update import find_model_asset_url
    assert find_model_asset_url([]) is None
    assert find_model_asset_url(None) is None
    assert find_model_asset_url([{"name": "readme.md", "browser_download_url": "x"}]) is None


def test_local_model_version_returns_a_model_tag():
    """With or without a user-folder download, the answer is a valid model-v* string."""
    from modules.model_update import local_model_version, parse_model_tag
    v = local_model_version()
    assert isinstance(v, str)
    assert parse_model_tag(v) is not None


def test_bundled_version_is_valid_model_tag():
    from modules.model_update import BUNDLED_MODEL_VERSION, parse_model_tag
    assert parse_model_tag(BUNDLED_MODEL_VERSION) is not None
