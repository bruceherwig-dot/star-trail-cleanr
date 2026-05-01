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
"""
import os
import sys
import traceback

_updater_controller = None  # module-scope strong reference, prevents GC


def _bundled_sparkle_framework_path():
    """Return absolute path to Sparkle.framework inside the .app, or None
    if we're not in a frozen Mac bundle."""
    if sys.platform != "darwin":
        return None
    if not hasattr(sys, "_MEIPASS"):
        return None
    candidate = os.path.abspath(
        os.path.join(os.path.dirname(sys.executable), "..", "Frameworks", "Sparkle.framework")
    )
    return candidate if os.path.exists(candidate) else None


def init_sparkle():
    """Initialize Sparkle. Call once at app startup AFTER the QApplication
    has been created (Sparkle attaches to the active NSApplication, which
    PySide6 creates). Safe to call on non-Mac platforms (no-op)."""
    global _updater_controller

    framework_path = _bundled_sparkle_framework_path()
    if framework_path is None:
        return

    try:
        import objc
        ns = {}
        objc.loadBundle("Sparkle", ns, bundle_path=framework_path)
        SPUStandardUpdaterController = ns["SPUStandardUpdaterController"]
        _updater_controller = (
            SPUStandardUpdaterController.alloc()
            .initWithStartingUpdater_updaterDelegate_userDriverDelegate_(True, None, None)
        )
    except Exception:
        traceback.print_exc()


def check_for_updates():
    """Trigger an update check. Sparkle handles the entire UI (native dialog,
    download progress, install confirmation, restart). No-op if init_sparkle
    didn't run successfully."""
    if _updater_controller is None:
        return
    _updater_controller.checkForUpdates_(None)
