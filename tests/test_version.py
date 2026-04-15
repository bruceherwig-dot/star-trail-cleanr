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


def test_version_has_three_decimals():
    text = VERSION_FILE.read_text().strip()
    if "." not in text:
        raise AssertionError(f"version.txt = '{text}' has no decimal point; expected three-decimal format like '0.058'")
    decimals = text.split(".")[1]
    assert len(decimals) == 3, f"version.txt = '{text}' has {len(decimals)} decimals; expected exactly 3"
