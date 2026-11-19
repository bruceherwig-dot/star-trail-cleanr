"""The engine's error output must be drained while it runs.

2026-11-19: a Windows user with ~900 JPEGs froze at "Batch 2": OpenCV wrote one
warning per undecodable frame, the stderr pipe filled, the engine blocked on the
write and the GUI (reading only stdout) waited forever. This reproduces the
same shape with a fake engine and fails (by timeout) if the pipe is not drained.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from star_trail_cleanr import StderrDrain

# Writes far more than any pipe buffer to stderr, then a normal stdout line.
FAKE_ENGINE = (
    "import sys\n"
    "for i in range(20000):\n"
    "    sys.stderr.write('[ WARN] imread_ cannot decode frame %d\\n' % i)\n"
    "sys.stderr.write('REAL ERROR LAST LINE\\n'); sys.stderr.flush()\n"
    "print('loading 20/20', flush=True)\n"
)


def test_flood_of_warnings_does_not_hang_the_engine():
    proc = subprocess.Popen([sys.executable, "-c", FAKE_ENGINE],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    drain = StderrDrain(proc.stderr)
    lines = [l for l in proc.stdout]          # the GUI's own loop
    try:
        proc.wait(timeout=30)                 # a deadlock would never finish
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("engine blocked on a full stderr pipe")
    assert lines == ["loading 20/20\n"]
    text = drain.text()
    assert "REAL ERROR LAST LINE" in text     # the tail survives for crash reports
    assert "imread_ cannot decode frame 0" in text   # and so does the head
    assert "elided" in text
    assert len(text) < 200_000                # memory stays capped


def test_gui_uses_the_drain_not_a_late_read():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert "self._stderr_drain = StderrDrain(self._proc.stderr)" in src
    assert "self._proc.stderr.read()" not in src
