"""Confirm the cause of the Windows update-check failure — round 3.

THE FINDING (2026-07-27, rounds 1-2 + reading winsparkle.h v0.9.2 line 182):
`win_sparkle_set_appcast_url` takes a NARROW `const char *`. Every other text
setter in the engine takes wide `const wchar_t *`. Our app declares the URL
wide too (modules/winsparkle_updater.py), so the engine reads the wide bytes
as a narrow string and stops at the first zero byte: the URL it has been
checking since 2026-05-01 is the single letter "h". That fails instantly,
before any networking — which is why the failure took ~1 second, hit every
feed, every engine version and every machine, while raw fetches of the real
URL all worked, and why the error dialog blamed the user's connection.

This round proves cause and fix together, on the same clean CI machine:
  - URL passed WIDE, as the app does today  -> expect ERROR  (the bug)
  - URL passed NARROW, per the header       -> expect WORKS  (the fix)
  - a C host doing it correctly (winsparkle_c_host.c, compiled in the
    workflow) -> expect WORKS (the way every other app drives this engine)

Each engine test runs in its own subprocess (the engine can only be
initialised once per process). Run by updater-diagnose.yml, manual dispatch
only. Windows-only; stdlib only; ships nowhere.
"""
import ctypes
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_DLL = os.path.join(REPO_ROOT, "vendored", "winsparkle", "WinSparkle.dll")

OUR_FEED = "https://api.startrailcleanr.com/appcast-windows.xml"
EXAMPLE_FEED = "https://winsparkle.org/example/appcast.xml"

# Mirror the real app's identity exactly so the engine behaves as shipped.
APP_NAME = "Star Trail CleanR"
APP_VERSION = "2.81"


def engine_check(dll_path, url, url_mode):
    """Run inside the subprocess: one real check. `url_mode` is 'wide' (the
    app's current, wrong declaration) or 'narrow' (per winsparkle.h).
    Prints the outcome; exits 0 found/up-to-date, 2 error, 3 timeout."""
    CB = ctypes.CFUNCTYPE(None)
    result = {"outcome": None}

    cb_found = CB(lambda: result.__setitem__("outcome", "found"))
    cb_none = CB(lambda: result.__setitem__("outcome", "up-to-date"))
    cb_error = CB(lambda: result.__setitem__("outcome", "error"))

    dll = ctypes.CDLL(dll_path)
    if url_mode == "narrow":
        # The header's actual signature: const char *.
        dll.win_sparkle_set_appcast_url.argtypes = [ctypes.c_char_p]
        dll.win_sparkle_set_appcast_url(url.encode("ascii"))
    else:
        # What the app has done since 2026-05-01.
        dll.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
        dll.win_sparkle_set_appcast_url(url)
    dll.win_sparkle_set_app_details.argtypes = [ctypes.c_wchar_p] * 3
    dll.win_sparkle_set_automatic_check_for_updates.argtypes = [ctypes.c_int]
    dll.win_sparkle_set_did_find_update_callback.argtypes = [CB]
    dll.win_sparkle_set_did_not_find_update_callback.argtypes = [CB]
    dll.win_sparkle_set_error_callback.argtypes = [CB]

    dll.win_sparkle_set_app_details(APP_NAME, APP_NAME, APP_VERSION)
    dll.win_sparkle_set_automatic_check_for_updates(0)
    dll.win_sparkle_set_did_find_update_callback(cb_found)
    dll.win_sparkle_set_did_not_find_update_callback(cb_none)
    dll.win_sparkle_set_error_callback(cb_error)
    dll.win_sparkle_init()
    dll.win_sparkle_check_update_without_ui()

    deadline = time.time() + 45
    while result["outcome"] is None and time.time() < deadline:
        time.sleep(0.2)
    outcome = result["outcome"] or "timed out"
    print(f"ENGINE {outcome}", flush=True)
    try:
        dll.win_sparkle_cleanup()
    except Exception:
        pass
    sys.exit({"found": 0, "up-to-date": 0, "error": 2}.get(outcome, 3))


def run_engine_test(label, dll_path, url, url_mode):
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--engine", dll_path, url, url_mode],
                       capture_output=True, text=True, timeout=120)
    verdict = {0: "WORKS", 2: "ERROR", 3: "TIMED OUT"}.get(p.returncode,
                                                           f"exit {p.returncode}")
    print(f"  {label:44} -> {verdict}", flush=True)
    return p.returncode


def main():
    if sys.platform != "win32":
        print("Windows-only diagnostic; nothing to do here.")
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "--engine":
        engine_check(sys.argv[2], sys.argv[3], sys.argv[4])  # exits

    print("=== URL passed WIDE (the app's current declaration) ===", flush=True)
    run_engine_test("wide vs example feed", OUR_DLL, EXAMPLE_FEED, "wide")
    run_engine_test("wide vs our feed", OUR_DLL, OUR_FEED, "wide")

    print("\n=== URL passed NARROW (per winsparkle.h line 182) ===", flush=True)
    run_engine_test("narrow vs example feed", OUR_DLL, EXAMPLE_FEED, "narrow")
    run_engine_test("narrow vs our feed", OUR_DLL, OUR_FEED, "narrow")

    print("\nExpected if the finding is right: both wide ERROR, both narrow "
          "WORKS. The C-host step in the workflow is the same confirmation "
          "from a compiled program.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
