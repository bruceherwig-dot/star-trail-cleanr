"""Keep the machine awake during a cleaning run.

A long run must not be interrupted by system idle sleep. A laptop on battery will
otherwise sleep mid-run and freeze the job for as long as it stays asleep (a real
case: a 300-frame run showed 2h 15m wall-clock, of which ~1h 57m was the machine
asleep). This holds a SYSTEM-sleep assertion only -- the display may still dim and
sleep, which is fine and saves battery; the run just won't be dropped.

Best-effort and safe by construction: if the platform mechanism is missing or
fails, the run continues normally without protection. Nothing here raises into the
caller -- acquire()/release() swallow every error.

Platforms:
  macOS   -- `caffeinate -i -w <our pid>` subprocess. -i prevents idle system
             sleep; -w ties caffeinate's life to ours, so it can never be left
             holding the machine awake if we crash.
  Windows -- SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), cleared
             with ES_CONTINUOUS on release.
  Linux   -- `systemd-inhibit ... sleep infinity` subprocess (harmless no-op if
             systemd-inhibit is absent).
"""

import os
import sys
import subprocess

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    """Hold a system keep-awake assertion across a run. Idempotent and
    exception-safe: acquire()/release() can be called any number of times and in
    any order without raising."""

    def __init__(self):
        self._proc = None          # macOS / Linux: the helper subprocess
        self._win_active = False   # Windows: whether the exec state is set
        self._active = False

    def acquire(self):
        """Start keeping the system awake. No-op if already active or if the
        platform mechanism is unavailable."""
        if self._active:
            return
        try:
            if sys.platform == "darwin":
                self._proc = subprocess.Popen(
                    ["caffeinate", "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform.startswith("win"):
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
                self._win_active = True
            else:
                self._proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep",
                     "--why=Star Trail CleanR is cleaning", "--mode=block",
                     "sleep", "infinity"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._active = True
        except Exception:
            # Never break a run because keep-awake could not start.
            self._proc = None
            self._win_active = False
            self._active = False

    def release(self):
        """Stop keeping the system awake. Safe to call when not active."""
        try:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None
            if self._win_active:
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
                except Exception:
                    pass
                self._win_active = False
        finally:
            self._active = False

    @property
    def active(self):
        return self._active
