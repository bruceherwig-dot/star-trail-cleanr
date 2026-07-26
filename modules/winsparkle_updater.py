"""
WinSparkle updater integration for Windows.

Loads WinSparkle.dll via ctypes at runtime and exposes init/cleanup/check
functions. WinSparkle's native dialogs handle the entire UI; we don't draw
update UI ourselves.

References used to write this module:
- WinSparkle headers (winsparkle.h) at https://github.com/vslavik/winsparkle
- Architecture decisions in memory: project_v2x_auto_update_architecture.md

Why a hand-rolled ctypes shim and not the pywinsparkle PyPI package: that
package was last released 2019-04-08 (v1.6.0) and is abandoned. WinSparkle
itself is active (v0.9.2 was released 2025-10-13). The C API is small
enough that a 30-line shim is the cleaner long-term option.

This module is a no-op on Mac, Linux, and dev (live-source) launches.

HOW UPDATES REACH THE USER ON WINDOWS (plain English):
- check_for_updates_in_background() runs on EVERY app launch. It quietly asks
  the update server "is there a newer version?" If yes, WinSparkle pops its
  native "update available" window and does the download + install-in-place +
  relaunch. If the user is already current, nothing appears. Seamless, one
  click — no website, no manual download.
- check_for_updates() is the same install path but user-initiated (the
  Settings "Check for Updates" button and the in-app update banner).
- Manual website download is the LINUX-only fallback (no WinSparkle on Linux).
"""
import ctypes
import os
import sys

# Module-level handle to the loaded WinSparkle.dll (a ctypes CDLL object).
# Stays None until init_winsparkle() successfully loads the DLL, and is reset
# to None if loading fails. Every public function below treats `_dll is None`
# as "WinSparkle is not available here" and quietly does nothing — this is how
# the whole module becomes a harmless no-op on Mac, Linux, and dev launches.
_dll = None

# WinSparkle's error callback type: void (*)(void). Defined once at module scope.
_WIN_SPARKLE_ERROR_CB = ctypes.CFUNCTYPE(None)

# True only between a USER-initiated check (Settings button / banner) and its
# result. The silent per-launch background check clears it. The error callback
# below surfaces a fallback ONLY when the user themselves triggered the check,
# so a quietly failing background check never pops a dialog every launch.
_user_initiated = False

# _error_cb is kept at module scope so the ctypes callback object is not garbage
# collected while WinSparkle holds a pointer to it. _error_handler is the
# no-argument Python callable the GUI registers via set_error_handler().
_error_cb = None
_error_handler = None


def _on_winsparkle_error():
    """Called by WinSparkle, on ITS thread, when an update operation fails
    (e.g. the appcast can't be retrieved -- the "error retrieving update
    information" case). Surfaces the fallback only when the user just triggered
    the check and a handler is registered. Must never raise."""
    global _user_initiated
    try:
        if _user_initiated and _error_handler is not None:
            _error_handler()
    except Exception:
        pass
    _user_initiated = False


def set_error_handler(fn):
    """Register a no-argument callable invoked when a user-initiated WinSparkle
    update fails. It runs on WinSparkle's thread, so it must be thread-safe; the
    GUI passes a Qt signal's .emit, which marshals safely to the UI thread."""
    global _error_handler
    _error_handler = fn


# Quiet-check support (2026-07-25): a user-initiated check runs WITHOUT the
# engine's own windows (win_sparkle_check_update_without_ui) and reports its
# outcome through these callbacks. The engine's install window is only shown
# AFTER a confirmed find, so a machine whose Windows networking layer blocks
# the engine (a real tester's machine) sees OUR dialog instead of the engine's
# dead-end "Update Error!" box. Callback objects live at module scope so ctypes
# keeps them alive while WinSparkle holds their pointers.
_found_cb = None
_notfound_cb = None
_found_handler = None
_notfound_handler = None


def _on_winsparkle_found():
    """Engine thread: a quiet check confirmed a newer version exists."""
    global _user_initiated
    try:
        if _user_initiated and _found_handler is not None:
            _found_handler()
    except Exception:
        pass
    _user_initiated = False


def _on_winsparkle_notfound():
    """Engine thread: a quiet check completed; already on the newest version."""
    global _user_initiated
    try:
        if _user_initiated and _notfound_handler is not None:
            _notfound_handler()
    except Exception:
        pass
    _user_initiated = False


def set_quiet_handlers(found, notfound):
    """Register the outcome callables for quiet checks (thread-safe required;
    the GUI passes Qt signal .emit functions)."""
    global _found_handler, _notfound_handler
    _found_handler = found
    _notfound_handler = notfound


def _find_winsparkle_dll():
    """Return absolute path to WinSparkle.dll inside the frozen bundle, or
    None if we're not in a frozen Windows bundle. The build places the DLL
    at the top of the bundle (next to the .exe) so Windows' default DLL
    search finds it without PATH manipulation. We also check _MEIPASS as a
    fallback for compatibility with older PyInstaller layouts."""
    if sys.platform != "win32":
        return None
    # `sys._MEIPASS` only exists inside a PyInstaller-frozen bundle. Its absence
    # means we're running live Python source (a dev launch), where there is no
    # bundled DLL to find — so updates are intentionally disabled.
    if not hasattr(sys, "_MEIPASS"):
        return None
    candidates = [
        os.path.join(os.path.dirname(sys.executable), "WinSparkle.dll"),
        os.path.join(sys._MEIPASS, "WinSparkle.dll"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def init_winsparkle(appcast_url, app_name, app_version, company_name="Star Trail CleanR"):
    """Initialize WinSparkle. Call once early in app startup. Safe to call
    on non-Windows platforms (no-op).

    Arguments are passed as native Win32 wide strings (UTF-16). Once init
    is called, WinSparkle is allowed to start its background check loop.
    """
    global _dll

    dll_path = _find_winsparkle_dll()
    if dll_path is None:
        return

    try:
        _dll = ctypes.CDLL(dll_path)
        _dll.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
        _dll.win_sparkle_set_app_details.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
        ]
        _dll.win_sparkle_set_automatic_check_for_updates.argtypes = [ctypes.c_int]
        _dll.win_sparkle_init.argtypes = []
        _dll.win_sparkle_check_update_with_ui.argtypes = []
        _dll.win_sparkle_check_update_without_ui.argtypes = []
        _dll.win_sparkle_cleanup.argtypes = []

        # WinSparkle requires ALL configuration before win_sparkle_init().
        # (Verified against winsparkle.h: config calls must precede init.)
        # Turning automatic checks on enables WinSparkle's own periodic timer
        # as a backstop and persists the setting; the explicit per-launch
        # check below is what actually surfaces updates on open.
        _dll.win_sparkle_set_appcast_url(appcast_url)
        _dll.win_sparkle_set_app_details(company_name, app_name, app_version)
        _dll.win_sparkle_set_automatic_check_for_updates(1)
        # Register an error callback so a failed user-initiated update (e.g. the
        # appcast can't be fetched on a machine where a security suite or proxy
        # blocks WinSparkle's networking) can surface a helpful manual-download
        # fallback. Guarded: older WinSparkle DLLs may not export this symbol.
        global _error_cb
        try:
            _dll.win_sparkle_set_error_callback.argtypes = [_WIN_SPARKLE_ERROR_CB]
            _error_cb = _WIN_SPARKLE_ERROR_CB(_on_winsparkle_error)
            _dll.win_sparkle_set_error_callback(_error_cb)
        except Exception:
            pass
        # Quiet-check outcome callbacks (found / not found). Guarded the same
        # way: an older DLL without these exports just means quiet checks are
        # unavailable and check_for_updates_quiet() returns False.
        global _found_cb, _notfound_cb
        try:
            _dll.win_sparkle_set_did_find_update_callback.argtypes = [_WIN_SPARKLE_ERROR_CB]
            _found_cb = _WIN_SPARKLE_ERROR_CB(_on_winsparkle_found)
            _dll.win_sparkle_set_did_find_update_callback(_found_cb)
            _dll.win_sparkle_set_did_not_find_update_callback.argtypes = [_WIN_SPARKLE_ERROR_CB]
            _notfound_cb = _WIN_SPARKLE_ERROR_CB(_on_winsparkle_notfound)
            _dll.win_sparkle_set_did_not_find_update_callback(_notfound_cb)
        except Exception:
            _found_cb = _notfound_cb = None
        _dll.win_sparkle_init()
    except Exception:
        import traceback
        traceback.print_exc()
        _dll = None


def updater_alive():
    """True when the WinSparkle engine loaded successfully. Mirrors
    sparkle_updater.updater_alive(): used to suppress the orange banner when
    the native one-click updater owns notification, and by fallback logic to
    detect a dead engine."""
    return _dll is not None


def check_for_updates_quiet():
    """User-initiated check WITHOUT any engine windows. The outcome arrives via
    the handlers registered with set_quiet_handlers / set_error_handler:
    found -> caller shows the engine's install window (check_for_updates());
    not found -> caller shows its own up-to-date note;
    error -> caller shows its own explain-and-download dialog.

    Returns True when the check was dispatched, False when the engine never
    loaded or this DLL lacks the outcome callbacks -- callers treat False as
    "fall back to the old visible path" so the button never does nothing."""
    if _dll is None or _found_cb is None:
        return False
    try:
        global _user_initiated
        _user_initiated = True
        _dll.win_sparkle_check_update_without_ui()
        return True
    except Exception:
        return False


def check_for_updates():
    """Trigger a foreground update check (shows native UI). User-initiated:
    the Settings 'Check for Updates' button and the in-app update banner.

    Returns True when the check was handed to WinSparkle, False when the
    updater engine never loaded -- callers treat False as "show the user a
    visible fallback" so the button never silently does nothing."""
    if _dll is None:
        return False
    try:
        global _user_initiated
        _user_initiated = True   # so a failure surfaces the manual-download fallback
        _dll.win_sparkle_check_update_with_ui()
        return True
    except Exception:
        return False


def check_for_updates_in_background():
    """Silently check for an update RIGHT NOW. Called on EVERY app launch.

    Plain English: when the user opens Star Trail CleanR on Windows, this
    asks the update server "is there a newer version?" If yes, WinSparkle
    shows its native "update available" window and handles the download,
    in-place install, and relaunch. If the user is already current, nothing
    appears. This is what makes a new release reach people the moment they
    open the app, not on WinSparkle's periodic timer.

    Verified against winsparkle.h: win_sparkle_check_update_without_ui shows
    no progress UI and no "you're up to date" box — it surfaces the update
    window only when a newer version exists. Safe to call right after
    win_sparkle_init(); it returns immediately and runs on its own thread."""
    if _dll is None:
        return
    global _user_initiated
    _user_initiated = False   # a silent background failure must not pop a dialog
    _dll.win_sparkle_check_update_without_ui()


def cleanup():
    """Tell WinSparkle to stop its background thread cleanly. Call on app
    shutdown."""
    if _dll is None:
        return
    _dll.win_sparkle_cleanup()
