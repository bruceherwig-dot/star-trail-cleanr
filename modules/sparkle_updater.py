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


def bring_app_to_front():
    """Activate the app so a Sparkle window that's just been shown lands IN FRONT
    of the Qt main window instead of behind it.

    Why this exists: Sparkle draws its own native 'A new version is available'
    window, but nothing was pulling the app forward when it appeared, so on a Qt
    app the main window stayed on top and the update prompt opened hidden behind
    it -- the user sees nothing and concludes the updater is dead (exactly what
    happened on 2026-06-19 with the 2.51 build). macOS-only, best-effort, never
    raises."""
    if sys.platform != "darwin":
        return
    try:
        import objc
        NSApplication = objc.lookUpClass("NSApplication")
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        _log("bring_app_to_front: activated app")
    except Exception:
        _log(f"bring_app_to_front: EXCEPTION\n{traceback.format_exc()}")


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
        # Get NSObject from the pyobjc bridge that is ALREADY imported and
        # bundled (the `objc` module loaded a few lines up). Do NOT import it
        # from the `Foundation` wrapper package -- that package is not bundled
        # by PyInstaller, so `from Foundation import NSObject` raised
        # ModuleNotFoundError inside every frozen Mac build, killing the
        # updater engine at startup. That single line is why no Mac user ever
        # received an automatic update (confirmed live on 2026-06-10: shipped
        # 2.49 logs this exact exception, then "no controller, no-op" forever).
        NSObject = objc.lookUpClass("NSObject")

        # An Objective-C delegate object that Sparkle calls back into when
        # events happen. Subclassing NSObject (not a plain Python class) is
        # required so PyObjC can hand it to Sparkle as a real Cocoa object.
        # The only callback we implement is "a valid newer version was found";
        # all other Sparkle events use its built-in default behavior.
        class _SparkleDelegate(NSObject):
            def updater_didFindValidUpdate_(self, updater, item):
                """Sparkle delegate callback: fires when Sparkle confirms a
                genuinely newer version is available (before its install popup
                appears). The method name with trailing underscores is PyObjC's
                spelling of the Objective-C selector updater:didFindValidUpdate:,
                so Sparkle invokes it automatically. `updater` is the Sparkle
                updater object and `item` describes the found release; we ignore
                both and simply run the optional on-update-found callback, whose
                job is to dismiss our startup splash so it doesn't sit on top of
                Sparkle's native window. Any error in that callback is logged and
                swallowed so it can't crash the update flow."""
                _log("delegate: updater_didFindValidUpdate_ fired")
                cb = _on_update_found_callback
                if cb is not None:
                    try:
                        cb()
                    except Exception:
                        _log(f"delegate: callback raised\n{traceback.format_exc()}")
                # Backstop: pull the app forward now, before the popup appears.
                bring_app_to_front()

            def standardUserDriverWillHandleShowingUpdate_forUpdate_state_(
                    self, handleShowingUpdate, update, state):
                """SPUStandardUserDriverDelegate hook: Sparkle calls this the
                instant before it shows its update window. That's the exact right
                moment to pull the app to the front so the window opens on top of
                the Qt main window, not behind it. Implemented as a no-arg-suffix
                PyObjC selector matching standardUserDriverWillHandleShowingUpdate:forUpdate:state:."""
                _log("delegate: standardUserDriverWillHandleShowingUpdate fired")
                bring_app_to_front()

            def standardUserDriverWillShowModalAlert(self):
                """Older SPUStandardUserDriverDelegate hook (modal alerts such as
                'you're up to date'); same front-bringing behavior. Harmless if a
                given Sparkle build never calls it."""
                _log("delegate: standardUserDriverWillShowModalAlert fired")
                bring_app_to_front()

        _updater_delegate = _SparkleDelegate.alloc().init()
        _log(f"init_sparkle: delegate={_updater_delegate!r}")

        # Build and START Sparkle's standard updater controller. Argument
        # order matters and is the Sparkle 2 form (see module gotcha #1):
        #   startingUpdater=True   -> start the updater immediately
        #   updaterDelegate        -> our delegate above (gets didFindValidUpdate)
        #   userDriverDelegate     -> the SAME delegate: it also implements the
        #     SPUStandardUserDriverDelegate "will show update" hook so we can pull
        #     the app to the front right as the popup appears (it otherwise opens
        #     behind the Qt main window). Still uses Sparkle's default UI.
        # The returned controller is stored in the module-scope strong
        # reference so Python's garbage collector can't free it mid-check
        # (gotcha #2).
        _log("init_sparkle: calling alloc().initWithStartingUpdater...")
        _updater_controller = (
            SPUStandardUpdaterController.alloc()
            .initWithStartingUpdater_updaterDelegate_userDriverDelegate_(True, _updater_delegate, _updater_delegate)
        )
        _log(f"init_sparkle: controller={_updater_controller!r}")
        _log("init_sparkle: SUCCESS")
    except Exception:
        _log(f"init_sparkle: EXCEPTION\n{traceback.format_exc()}")
        traceback.print_exc()


def updater_alive():
    """True when the Sparkle engine started successfully (controller exists).
    Used by the CI updater-alive gate (STC_UPDATER_SMOKE): the built app exits
    nonzero when the engine is dead, failing the build before it ships. Added
    after five versions shipped with a dead engine that nothing ever tested."""
    return _updater_controller is not None


def check_for_updates():
    """Trigger an update check. Sparkle handles the entire UI (native dialog,
    download progress, install confirmation, restart).

    Returns True when the check was handed to Sparkle, False when the updater
    engine never started (init_sparkle failed or never ran) or the call blew
    up. Callers MUST treat False as "show the user a visible fallback" -- a
    Check for Updates button that silently does nothing is exactly the failure
    a real Mac user hit on 2026-06-10 (engine dead on their machine, button
    mute, no way to know why)."""
    if _updater_controller is None:
        _log("check_for_updates: no controller, no-op")
        return False
    _log("check_for_updates: invoking checkForUpdates_(None)")
    try:
        _updater_controller.checkForUpdates_(None)
        _log("check_for_updates: returned")
        return True
    except Exception:
        _log(f"check_for_updates: EXCEPTION\n{traceback.format_exc()}")
        return False


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
