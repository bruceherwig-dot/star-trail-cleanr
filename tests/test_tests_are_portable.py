"""The tests must pass on the machines that BUILD the app, not just on a Mac.

Found 2026-08-26, on the first practice run after the build jobs started running
this suite. Windows failed and every Mac passed, for a reason that had nothing to
do with the product:

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f

Reading a file without saying what text encoding it is uses the machine's default.
On macOS and Linux that default is UTF-8, so it works by luck. On Windows it is a
Western European codepage that cannot represent the characters our source files
contain (the multiplication sign in "8152x5432", dashes in comments), so thirty
one test files failed the moment they read a source file.

Our sources are UTF-8. Saying so costs nothing and makes the tests behave the
same everywhere. This guard stops the habit creeping back one new test at a time,
because the failure only ever shows up on a machine none of us runs day to day.

The product code is NOT affected: its only such read already names the encoding.
"""
import re
from pathlib import Path

TESTS = Path(__file__).parent
REPO = TESTS.parent


def _bare_reads(src):
    """Lines that read a file without naming an encoding. Binary reads are fine:
    bytes have no encoding to get wrong."""
    bad = []
    for n, line in enumerate(src.splitlines(), 1):
        if "encoding=" in line or '"rb"' in line or "'rb'" in line:
            continue
        if re.search(r"\.read_text\(\s*\)", line):
            bad.append((n, line.strip()))
        elif re.search(r"open\([^)]*\)\.read\(\)", line):
            bad.append((n, line.strip()))
    return bad


def test_no_test_reads_a_file_without_naming_the_encoding():
    offenders = []
    for p in sorted(TESTS.glob("*.py")):
        for n, line in _bare_reads(p.read_text(encoding="utf-8")):
            offenders.append(f"{p.name}:{n}  {line[:70]}")
    assert not offenders, (
        "these read a file without an encoding, which passes on a Mac and fails "
        "on the Windows build machine:\n  " + "\n  ".join(offenders))


def test_the_product_names_its_encodings_too():
    """Same rule where it matters more: a user's Windows machine, not just CI."""
    offenders = []
    for p in [REPO / "star_trail_cleanr.py", REPO / "astro_clean_v5.py"] + \
             sorted((REPO / "modules").glob("*.py")):
        for n, line in _bare_reads(p.read_text(encoding="utf-8")):
            offenders.append(f"{p.name}:{n}  {line[:70]}")
    assert not offenders, (
        "shipping code reads a file without an encoding; on a Windows user's "
        "machine this raises on any character outside their codepage:\n  "
        + "\n  ".join(offenders))


def test_the_build_actually_runs_this_suite():
    """The whole reason the Windows problem surfaced. If this step is ever
    removed, we are back to testing one program and shipping another."""
    wf = (REPO / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    assert wf.count("tests/run_all.py") >= 4, (
        "the four build jobs must each run the smoke suite on their own machine "
        "before packaging; that is what caught the Windows encoding failure")
