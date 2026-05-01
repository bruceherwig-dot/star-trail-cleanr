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
"""
import ctypes
import os
import sys

_dll = None


def _find_winsparkle_dll():
    """Return absolute path to WinSparkle.dll inside the frozen bundle, or
    None if we're not in a frozen Windows bundle. The dll is placed next to
    the .exe by the build (see build_helper.py spec binaries entry)."""
    if sys.platform != "win32":
        return None
    if not hasattr(sys, "_MEIPASS"):
        return None
    candidate = os.path.join(sys._MEIPASS, "WinSparkle.dll")
    return candidate if os.path.exists(candidate) else None


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
        _dll.win_sparkle_init.argtypes = []
        _dll.win_sparkle_check_update_with_ui.argtypes = []
        _dll.win_sparkle_cleanup.argtypes = []

        _dll.win_sparkle_set_appcast_url(appcast_url)
        _dll.win_sparkle_set_app_details(company_name, app_name, app_version)
        _dll.win_sparkle_init()
    except Exception:
        import traceback
        traceback.print_exc()
        _dll = None


def check_for_updates():
    """Trigger a foreground update check (shows native UI). Background
    checks happen automatically once init has been called."""
    if _dll is None:
        return
    _dll.win_sparkle_check_update_with_ui()


def cleanup():
    """Tell WinSparkle to stop its background thread cleanly. Call on app
    shutdown."""
    if _dll is None:
        return
    _dll.win_sparkle_cleanup()
