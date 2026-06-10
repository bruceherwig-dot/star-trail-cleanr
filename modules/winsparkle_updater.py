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

_dll = None


def _find_winsparkle_dll():
    """Return absolute path to WinSparkle.dll inside the frozen bundle, or
    None if we're not in a frozen Windows bundle. The build places the DLL
    at the top of the bundle (next to the .exe) so Windows' default DLL
    search finds it without PATH manipulation. We also check _MEIPASS as a
    fallback for compatibility with older PyInstaller layouts."""
    if sys.platform != "win32":
        return None
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
        _dll.win_sparkle_init()
    except Exception:
        import traceback
        traceback.print_exc()
        _dll = None


def check_for_updates():
    """Trigger a foreground update check (shows native UI). User-initiated:
    the Settings 'Check for Updates' button and the in-app update banner."""
    if _dll is None:
        return
    _dll.win_sparkle_check_update_with_ui()


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
    _dll.win_sparkle_check_update_without_ui()


def cleanup():
    """Tell WinSparkle to stop its background thread cleanly. Call on app
    shutdown."""
    if _dll is None:
        return
    _dll.win_sparkle_cleanup()
