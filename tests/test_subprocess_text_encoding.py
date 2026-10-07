"""Helper programs' output must be read as UTF-8, never the computer's own encoding.

Regression (2026-10-07): a v2.97 user on a Chinese Windows (default text encoding
gbk) crashed with "'gbk' codec can't decode byte 0x87 ... illegal multibyte
sequence", raised in subprocess's reader thread. Two places ran a helper with
`subprocess.run(cmd, capture_output=True, text=True)`, which reads with the
computer's default encoding; anything it did not recognise raised and the helper's
real error message was lost.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _child(code):
    return [sys.executable, "-c", code]


def test_run_captured_reads_utf8_and_survives_bytes_that_are_not_text():
    from modules.io_safe import run_captured
    # a Chinese folder name in a status line, then bytes that are valid in no encoding
    code = ("import sys; w = sys.stdout.buffer.write; "
            "w('wrote C:\\\\Users\\\\\u674e\\\\\u661f\\\\star.jpg \\u2026 done\\n'.encode('utf-8')); "
            "w(b'bad byte: \\x87 \\xff end\\n'); "
            "sys.stderr.buffer.write(b'warning \\xa6\\x87 from the encoder\\n')")
    r = run_captured(_child(code))
    assert r.returncode == 0
    assert "\u674e" in r.stdout and "done" in r.stdout, r.stdout
    assert "bad byte" in r.stdout and "end" in r.stdout, "output after a bad byte must survive"
    assert "from the encoder" in r.stderr, "stderr is read the same way"


def test_run_captured_does_not_depend_on_the_computers_default_encoding():
    """Pretend the computer's default is gbk (a Chinese Windows). The plain
    text=True call crashes on UTF-8 output; run_captured must not."""
    import subprocess
    from modules.io_safe import run_captured
    code = ("import sys; sys.stdout.buffer.write(('x' * 60 + ' \\u674e\\u661f ' + 'y' * 10 + "
            "' \\u2026 done\\n').encode('utf-8'))")
    cmd = _child(code)
    # the OLD call, with the gbk default made explicit, fails the way the user's did
    crashed = False
    try:
        subprocess.run(cmd, capture_output=True, text=True, encoding="gbk")
    except UnicodeDecodeError:
        crashed = True
    assert crashed, "the test input must actually trip a gbk read, or it proves nothing"
    r = run_captured(cmd)
    assert r.returncode == 0 and "done" in r.stdout


def test_run_captured_keeps_the_callers_environment_and_encoding_choice():
    import os
    from modules.io_safe import run_captured
    env = dict(os.environ)
    env["STC_PROBE"] = "hello"
    env["PYTHONIOENCODING"] = "utf-8"
    r = run_captured(_child("import os; print(os.environ.get('STC_PROBE'), "
                            "os.environ.get('PYTHONIOENCODING'))"), env=env)
    assert r.stdout.split() == ["hello", "utf-8"], r.stdout
    # with no env passed, the helper is still asked to print UTF-8
    r = run_captured(_child("import os; print(os.environ.get('PYTHONIOENCODING'))"))
    assert r.stdout.strip() == "utf-8"


def _app_sources():
    skip_dirs = {"tests", "tools", "archive", "dataset_pipeline", "runs", "build", "dist"}
    files = [REPO / n for n in ("star_trail_cleanr.py", "make_share_clip.py",
                                "timelapse_maker.py", "astro_clean_v5.py")]
    files += sorted((REPO / "modules").glob("*.py"))
    return [f for f in files if f.exists() and not (set(f.relative_to(REPO).parts) & skip_dirs)]


def test_no_app_code_reads_a_helpers_output_with_the_default_encoding():
    """The guard: any subprocess call in the app that asks for TEXT output must also
    say which encoding, or it will crash on a Chinese (gbk) Windows the moment the
    helper prints something that encoding does not know. Uses the syntax tree, so a
    comment or docstring that names the mistake cannot trip it."""
    offenders = []
    for path in _app_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("run", "Popen", "check_output"):
                continue
            base = fn.value.id if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) else ""
            if base != "subprocess":
                continue
            kws = {k.arg: k.value for k in node.keywords if k.arg}
            wants_text = any(
                isinstance(kws.get(a), ast.Constant) and kws[a].value is True
                for a in ("text", "universal_newlines"))
            if wants_text and "encoding" not in kws:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        "these read a helper's output as text with the computer's default encoding "
        "(gbk on a Chinese Windows crashes): " + ", ".join(offenders)
        + " -- use modules.io_safe.run_captured, or pass encoding='utf-8', errors='replace'")


def test_both_share_helpers_use_the_safe_reader():
    app = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    i = app.find("run_captured(cmd)")
    assert i > 0, "the share-output job must read its helper through run_captured"
    stack = (REPO / "modules" / "share_stacker.py").read_text(encoding="utf-8")
    assert "run_captured(cmd)" in stack, "the video encode must read its helper through run_captured"
