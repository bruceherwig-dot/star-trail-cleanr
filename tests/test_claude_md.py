"""CLAUDE.md still contains the critical standing rules.

If future edits drop any of these markers, the test fails and flags the omission
before the change ships. The goal is not to lock CLAUDE.md's exact wording — it's
to make sure the rules future-Claude depends on are still present."""
from pathlib import Path

CLAUDE_MD = Path(__file__).parent.parent / "CLAUDE.md"


def _read():
    assert CLAUDE_MD.exists(), f"CLAUDE.md missing at {CLAUDE_MD}"
    return CLAUDE_MD.read_text()


def test_sacred_data_rule_present():
    text = _read().lower()
    assert "sacred" in text and "original source images" in text, "sacred-data rule removed from CLAUDE.md"
    assert "reviewed labelme" in text, "reviewed-labels sacred rule removed from CLAUDE.md"


def test_version_bump_rule_present():
    text = _read().lower()
    assert "version.txt" in text and ".001" in text, "version-bump rule removed from CLAUDE.md"


def test_release_checklist_present():
    text = _read().lower()
    assert "release checklist" in text, "release checklist section removed from CLAUDE.md"
    assert "changelog.md" in text, "CHANGELOG.md step missing from release checklist"


def test_trained_models_location_present():
    text = _read().lower()
    assert "yolo_runs" in text, "trained models location (~/Documents/yolo_runs/) not documented in CLAUDE.md"


def test_cvat_setup_documented():
    text = _read().lower()
    assert "cvat" in text, "CVAT setup section removed from CLAUDE.md"
    assert "cvat_credentials" in text, "CVAT credentials file path removed from CLAUDE.md"
