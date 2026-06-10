"""
Sparkle updater integration for macOS.

Loads Sparkle.framework via PyObjC at runtime. Sparkle's native dialogs
handle the entire UI — we don't draw any update UI ourselves.

HOW UPDATES REACH THE USER (plain English):
- check_for_updates_in_background() runs on EVERY app launch. It quietly asks
  the update server "is there a newer version?" If yes, Sparkle pops its own
  "A new version is available — Install" window and does the download +
  install-in-place + relaunch. If the user is already current, nothing shows.
  This is the seamless, one-click experience — no website, no manual download.
- check_for_updates() is the same one-click install path, but user-initiated
  (the Settings "Check for Updates" button and the in-app update banner). It
  may show a "you're up to date" message because the user explicitly asked.
- Manual website download is the LINUX-only fallback (no Sparkle on Linux).

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
_updater_delegate = None  # module-scope strong reference, prevents GC
_on_update_found_callback = None  # set by init_sparkle()

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


def init_sparkle(on_update_found=None):
    """Initialize Sparkle. Call once at app startup AFTER the QApplication
    has been created (Sparkle attaches to the active NSApplication, which
    PySide6 creates). Safe to call on non-Mac platforms (no-op).

    on_update_found: optional zero-arg callable invoked when Sparkle finds
    a valid newer version. Used to dismiss the startup splash so Sparkle's
    native popup doesn't fight for z-order with our splash window."""
    global _updater_controller, _updater_delegate, _on_update_found_callback
    _on_update_found_callback = on_update_found

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
        _log("init_sparkle: building delegate")
        from Foundation import NSObject

        class _SparkleDelegate(NSObject):
            def updater_didFindValidUpdate_(self, updater, item):
                _log("delegate: updater_didFindValidUpdate_ fired")
                cb = _on_update_found_callback
                if cb is not None:
                    try:
                        cb()
                    except Exception:
                        _log(f"delegate: callback raised\n{traceback.format_exc()}")

        _updater_delegate = _SparkleDelegate.alloc().init()
        _log(f"init_sparkle: delegate={_updater_delegate!r}")

        _log("init_sparkle: calling alloc().initWithStartingUpdater...")
        _updater_controller = (
            SPUStandardUpdaterController.alloc()
            .initWithStartingUpdater_updaterDelegate_userDriverDelegate_(True, _updater_delegate, None)
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


def check_for_updates_in_background():
    """Silently check for an update RIGHT NOW. Called on EVERY app launch.

    Plain English: when the user opens Star Trail CleanR, this asks the update
    server "is there a newer version?" If yes, Sparkle shows its own native
    "A new version is available — Install" window and handles the download,
    in-place install, and relaunch. If the user is already current, NOTHING
    appears — no popup, no "you're up to date" box. This is what makes a new
    release reach people the moment they open the app, instead of waiting on
    Sparkle's once-a-day timer.

    Verified against Sparkle's SPUUpdater.h (2.x):
    - The correct silent-when-current call is updater.checkForUpdatesInBackground.
    - It must run on the main thread, right after the updater is started and
      before the run loop spins. init_sparkle() starts the updater, and the GUI
      calls this immediately afterward, before the Qt event loop, so the timing
      is correct.
    - It only checks when automatic checks are enabled; build_helper.py sets
      SUEnableAutomaticChecks=true in the bundle's Info.plist, which satisfies
      that and also suppresses Sparkle's first-run permission prompt."""
    if _updater_controller is None:
        _log("check_for_updates_in_background: no controller, no-op")
        return
    _log("check_for_updates_in_background: invoking updater().checkForUpdatesInBackground()")
    try:
        _updater_controller.updater().checkForUpdatesInBackground()
        _log("check_for_updates_in_background: returned")
    except Exception:
        _log(f"check_for_updates_in_background: EXCEPTION\n{traceback.format_exc()}")
