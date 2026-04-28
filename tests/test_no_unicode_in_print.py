"""Worker-side print() statements must contain only ASCII.

Regression test for the v1.8 mojibake bug: a Windows tester saw
'repairing 17/20: _DSC0023.tif â€" 0 trails' in the log because em-dashes
in our print() strings were emitted as UTF-8 bytes by the worker but
decoded as cp1252 by the GUI's subprocess pipe on Windows.

The GUI side now forces UTF-8 on the pipe, but defense-in-depth: don't
emit any non-ASCII characters from the worker's print() calls in the
first place. Comments and docstrings are exempt; they never reach stdout.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

# Files whose print() output ends up in the GUI log panel.
WORKER_FILES = [
    REPO / "astro_clean_v5.py",
    REPO / "modules" / "detect_trails.py",
    REPO / "modules" / "repair.py",
    REPO / "modules" / "hot_pixels.py",
    REPO / "modules" / "slope_match.py",
    REPO / "modules" / "clean_sky.py",
    REPO / "modules" / "align.py",
]


def _strings_in_call(node):
    """Yield every string literal value that's part of a call's args,
    walking into f-string parts so we catch them too."""
    for arg in node.args + [kw.value for kw in node.keywords]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield sub.value, sub.lineno


def _print_calls_with_non_ascii(path: Path):
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "print":
            for s, lineno in _strings_in_call(node):
                non_ascii = [c for c in s if ord(c) > 127]
                if non_ascii:
                    hits.append((path.name, lineno, non_ascii[:5], s[:60]))
    return hits


def test_worker_print_strings_are_ascii_only():
    all_hits = []
    for path in WORKER_FILES:
        all_hits.extend(_print_calls_with_non_ascii(path))
    if all_hits:
        msg_lines = ["Non-ASCII characters found in worker print() statements:"]
        for fname, lineno, chars, snippet in all_hits:
            char_repr = " ".join(f"U+{ord(c):04X}({c!r})" for c in chars)
            msg_lines.append(f"  {fname}:{lineno}  chars={char_repr}  text={snippet!r}")
        msg_lines.append("Replace with ASCII (em dash -> '-', arrow -> '->', "
                         "multiplication sign -> 'x') so Windows users do not "
                         "see mojibake in the GUI log panel.")
        raise AssertionError("\n".join(msg_lines))
