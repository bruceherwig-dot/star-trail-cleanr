"""
Sparkle updater integration for macOS.

Loads Sparkle.framework via PyObjC at runtime and exposes a single function
check_for_updates() to be wired to a "Check for Updates..." menu item or
button. Sparkle's native dialogs handle the entire UI — we don't draw any
update UI ourselves.

References used to write this module:
- fman blog post: https://fman.io/blog/codesigning-and-automatic-updates-for-pyqt-apps/
- Sparkle 2 documentation
- Architecture decisions in memory: project_v2x_auto_update_architecture.md

Two known gotchas (logged in the architecture doc, do not remove):
1. SPUStandardUpdaterController init is fussy about delegate arguments. The
   correct Sparkle 2 form is initWithStartingUpdater:updaterDelegate:userDriverDelegate:
   passing (True, None, None). Earlier APIs (SUUpdater) are deprecated.
2. Python's GC can free the Sparkle controller object mid-check, causing a
   crash. Keep a strong module-scope reference (_updater_controller below).

This module is a no-op on Windows, Linux, and dev (live-source) launches.
On Windows, the WinSparkle ctypes wrapper is in a sibling module.

DIAGNOSTIC LOGGING (added v2.02-beta):
Every step writes a timestamped line to ~/.star_trail_cleanr/sparkle_debug.log
because PyInstaller --windowed redirects stderr away from terminals on Mac,
making the silent-fail mode hard to debug. Read the log file when an
auto-update test doesn't fire to see which step actually failed.
"""
import os
import sys
import time
import traceback

_updater_controller = None  # module-scope strong reference, prevents GC

_LOG_PATH = os.path.expanduser("~/.star_trail_cleanr/sparkle_debug.log")


def _log(msg):
    """Append a timestamped line to the diagnostic log. Best-effort, never
    raises. Created here because PyInstaller --windowed swallows Python
    stderr on Mac, so traceback.print_exc() output is invisible."""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def _bundled_sparkle_framework_path():
    """Return absolute path to Sparkle.framework inside the .app, or None
    if we're not in a frozen Mac bundle."""
    if sys.platform != "darwin":
        _log("not-darwin: skip")
        return None
    has_meipass = hasattr(sys, "_MEIPASS")
    _log(f"sys.platform={sys.platform} hasattr(_MEIPASS)={has_meipass}")
    if has_meipass:
        _log(f"sys._MEIPASS={sys._MEIPASS}")
    _log(f"sys.executable={sys.executable}")
    if not has_meipass:
        _log("no _MEIPASS: returning None (live source mode)")
        return None
    candidate = os.path.abspath(
        os.path.join(os.path.dirname(sys.executable), "..", "Frameworks", "Sparkle.framework")
    )
    exists = os.path.exists(candidate)
    _log(f"framework candidate={candidate} exists={exists}")
    return candidate if exists else None


def init_sparkle():
    """Initialize Sparkle. Call once at app startup AFTER the QApplication
    has been created (Sparkle attaches to the active NSApplication, which
    PySide6 creates). Safe to call on non-Mac platforms (no-op)."""
    global _updater_controller

    _log("=" * 60)
    _log("init_sparkle: ENTERED")

    framework_path = _bundled_sparkle_framework_path()
    if framework_path is None:
        _log("init_sparkle: framework_path is None, returning silently")
        return

    try:
        _log("init_sparkle: importing objc")
        import objc
        _log(f"init_sparkle: objc imported from {objc.__file__}")
        ns = {}
        _log(f"init_sparkle: calling objc.loadBundle on {framework_path}")
        objc.loadBundle("Sparkle", ns, bundle_path=framework_path)
        _log(f"init_sparkle: loadBundle done, ns has {len(ns)} entries")
        SPUStandardUpdaterController = ns.get("SPUStandardUpdaterController")
        _log(f"init_sparkle: SPUStandardUpdaterController={SPUStandardUpdaterController!r}")
        if SPUStandardUpdaterController is None:
            _log("init_sparkle: controller class missing from loaded bundle, aborting")
            return
        _log("init_sparkle: calling alloc().initWithStartingUpdater...")
        _updater_controller = (
            SPUStandardUpdaterController.alloc()
            .initWithStartingUpdater_updaterDelegate_userDriverDelegate_(True, None, None)
        )
        _log(f"init_sparkle: controller={_updater_controller!r}")
        _log("init_sparkle: SUCCESS")
    except Exception:
        _log(f"init_sparkle: EXCEPTION\n{traceback.format_exc()}")
        traceback.print_exc()


def check_for_updates():
    """Trigger an update check. Sparkle handles the entire UI (native dialog,
    download progress, install confirmation, restart). No-op if init_sparkle
    didn't run successfully."""
    if _updater_controller is None:
        _log("check_for_updates: no controller, no-op")
        return
    _log("check_for_updates: invoking checkForUpdates_(None)")
    try:
        _updater_controller.checkForUpdates_(None)
        _log("check_for_updates: returned")
    except Exception:
        _log(f"check_for_updates: EXCEPTION\n{traceback.format_exc()}")
