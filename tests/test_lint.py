"""Pyflakes lint gate on runtime code.

Catches the class of bug that shipped v1.97-beta broken: a `screen` reference
in `MainWindow.__init__` whose assignment got dropped during a refactor.
Pyflakes resolves names statically in well under a second; this gate runs on
every release tag so the same shape of crash cannot reach users again.

The gate is targeted, not strict. It fails only on crash-class findings
(undefined names, names that may be undefined from star imports). Unused
imports and unused locals are tolerated for now; tightening to full-strict
is a separate cleanup.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files whose every name reference must resolve. These are the files that
# run when a user launches the app or starts a cleaning batch. Anything
# unresolved here will crash a real user.
RUNTIME_FILES = [
    REPO_ROOT / "star_trail_cleanr.py",
    REPO_ROOT / "astro_clean_v5.py",
] + sorted((REPO_ROOT / "modules").glob("*.py"))

CRASH_CLASS_SUBSTRINGS = (
    "undefined name",
    "may be undefined",
)


def _run_pyflakes(files):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", *map(str, files)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise AssertionError(
            "pyflakes is not installed. Install it with "
            "`python3 -m pip install pyflakes` (or add it to the CI deps)."
        ) from e
    if "No module named pyflakes" in result.stderr:
        raise AssertionError(
            "pyflakes is not installed. Install it with "
            "`python3 -m pip install pyflakes` (or add it to the CI deps)."
        )
    return result.stdout.splitlines() + result.stderr.splitlines()


def test_runtime_files_have_no_undefined_names():
    findings = _run_pyflakes(RUNTIME_FILES)
    crash_class = [
        line for line in findings
        if any(token in line for token in CRASH_CLASS_SUBSTRINGS)
    ]
    assert not crash_class, (
        "pyflakes found crash-class name references in runtime code:\n"
        + "\n".join(crash_class)
    )


def test_runtime_files_exist():
    missing = [str(p) for p in RUNTIME_FILES if not p.exists()]
    assert not missing, f"runtime file list points at missing paths: {missing}"


if __name__ == "__main__":
    test_runtime_files_exist()
    test_runtime_files_have_no_undefined_names()
    print("Lint gate passed.")
