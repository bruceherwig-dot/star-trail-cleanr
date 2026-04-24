"""version.txt sanity checks."""
from pathlib import Path

REPO = Path(__file__).parent.parent
VERSION_FILE = REPO / "version.txt"


def test_version_file_exists():
    assert VERSION_FILE.exists(), f"version.txt missing at {VERSION_FILE}"


def test_version_parses_as_float():
    text = VERSION_FILE.read_text().strip()
    v = float(text)
    assert v >= 0


def test_version_in_sane_range():
    v = float(VERSION_FILE.read_text().strip())
    assert 0.001 <= v < 10.0, f"version.txt = {v} outside sane range (0.001 to 10.0)"


def test_version_matches_tag_format():
    """version.txt holds the shipped tag number (e.g., '1.7' for v1.7-beta).
    Must have exactly one decimal point, no trailing zeros padding."""
    text = VERSION_FILE.read_text().strip()
    assert "." in text, f"version.txt = '{text}' has no decimal point; expected single-decimal format like '1.7'"
    parts = text.split(".")
    assert len(parts) == 2, f"version.txt = '{text}' has multiple decimals; expected single-decimal format like '1.7'"
    assert parts[0].isdigit() and parts[1].isdigit(), f"version.txt = '{text}' has non-numeric parts"
