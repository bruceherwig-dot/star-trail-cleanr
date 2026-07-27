"""Corner the Windows update-check failure on a clean CI machine — round 2.

Round 1 (2026-07-27) established: raw synchronous WinInet fetches every feed
fine (HTTP 200), but the engine errors against EVERY feed — including
WinSparkle's own reference feed — and our vendored DLL is byte-identical to the
official 0.9.2 x64 release. So the feeds are exonerated; the failure lives in
the engine version or in how we drive it.

Round 2 does two things:

  1. REPLICATE the engine's exact download recipe, translated line by line
     from WinSparkle 0.9.2 src/download.cpp: async WinInet session, HTTP/2
     option, gzip/deflate decoding, the same flags, the same status callback.
     The engine swallows the Windows error number behind "An error occurred in
     retrieving update information"; this prints it.

  2. RACE three official engine versions with the identical harness:
     our 0.9.2, the current 0.9.4, and 0.8.3 (the version most apps ran for
     years). If an older or newer engine works, the fix is swapping one file.

Each engine test runs in its own subprocess because WinSparkle can only be
initialised once per process. Console script: every line is visible in CI logs,
unlike the frozen GUI app. Run by .github/workflows/updater-diagnose.yml
(manual dispatch only). Windows-only; exits cleanly elsewhere. Stdlib only.
"""
import ctypes
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_DLL = os.path.join(REPO_ROOT, "vendored", "winsparkle", "WinSparkle.dll")

OUR_FEED = "https://api.startrailcleanr.com/appcast-windows.xml"
EXAMPLE_FEED = "https://winsparkle.org/example/appcast.xml"

# Mirror the real app's identity exactly so the engine behaves as shipped.
APP_NAME = "Star Trail CleanR"
APP_VERSION = "2.81"

# Official builds to race against ours. Same release archives everyone uses.
RACE_VERSIONS = ("0.9.4", "0.8.3")
RELEASE_URL = "https://github.com/vslavik/winsparkle/releases/download/v{v}/WinSparkle-{v}.zip"

# WinInet error numbers worth recognising on sight (subset of wininet.h).
WININET_ERRORS = {
    12002: "TIMEOUT", 12005: "INVALID_URL", 12007: "NAME_NOT_RESOLVED",
    12029: "CANNOT_CONNECT", 12030: "CONNECTION_ABORTED",
    12031: "CONNECTION_RESET", 12032: "FORCE_RETRY",
    12038: "SEC_CERT_CN_INVALID", 12044: "CLIENT_AUTH_CERT_NEEDED",
    12045: "SEC_CERT_INVALID_CA", 12057: "SEC_CERT_REV_FAILED",
    12152: "INVALID_SERVER_RESPONSE", 12157: "SECURITY_CHANNEL_ERROR",
    12175: "SECURITY_ERROR",
}


def errname(code):
    return f"{code} ({WININET_ERRORS.get(code, 'see wininet.h')})" if code else "0"


# ── 1. The engine's download recipe, replicated from 0.9.2 download.cpp ──────

def replica_fetch(url):
    """Perform the appcast fetch exactly the way WinSparkle 0.9.2 does, and
    print what every stage says. Returns nothing; output is the point."""
    wininet = ctypes.WinDLL("wininet", use_last_error=True)

    INTERNET_OPEN_TYPE_PRECONFIG = 0
    INTERNET_FLAG_ASYNC = 0x10000000
    INTERNET_FLAG_RELOAD = 0x80000000
    INTERNET_FLAG_NO_CACHE_WRITE = 0x04000000
    INTERNET_FLAG_PRAGMA_NOCACHE = 0x00000100
    INTERNET_FLAG_SECURE = 0x00800000
    INTERNET_OPTION_ENABLE_HTTP_PROTOCOL = 148
    HTTP_PROTOCOL_FLAG_HTTP2 = 0x2
    INTERNET_OPTION_HTTP_DECODING = 65
    HTTP_QUERY_STATUS_CODE = 19
    HTTP_QUERY_FLAG_NUMBER = 0x20000000
    STATUS_HANDLE_CREATED = 60
    STATUS_REQUEST_COMPLETE = 100

    wininet.InternetOpenA.restype = ctypes.c_void_p
    wininet.InternetOpenA.argtypes = [ctypes.c_char_p, ctypes.c_ulong,
                                      ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_ulong]
    wininet.InternetOpenUrlA.restype = ctypes.c_void_p
    wininet.InternetOpenUrlA.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_char_p, ctypes.c_ulong,
                                         ctypes.c_ulong, ctypes.c_void_p]
    wininet.InternetSetOptionW.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                           ctypes.c_void_p, ctypes.c_ulong]
    wininet.HttpQueryInfoA.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                       ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p]
    wininet.InternetReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_ulong, ctypes.c_void_p]
    wininet.InternetCloseHandle.argtypes = [ctypes.c_void_p]

    CB = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong)
    wininet.InternetSetStatusCallback.restype = ctypes.c_void_p
    wininet.InternetSetStatusCallback.argtypes = [ctypes.c_void_p, CB]

    class AsyncResult(ctypes.Structure):
        _fields_ = [("dwResult", ctypes.c_void_p), ("dwError", ctypes.c_ulong)]

    state = {"handle": None, "async_error": None}
    done = threading.Event()

    def status_cb(hint, cctx, status, info, infolen):
        if status == STATUS_HANDLE_CREATED and info:
            state["handle"] = AsyncResult.from_address(info).dwResult
        elif status == STATUS_REQUEST_COMPLETE:
            if info:
                state["async_error"] = AsyncResult.from_address(info).dwError
            done.set()

    cb = CB(status_cb)  # keep a reference for the call's duration

    # download.cpp MakeUserAgent(): "<app>/<version> WinSparkle/<ver> (Win64)"
    agent = f"{APP_NAME}/{APP_VERSION} WinSparkle/0.9.2 (Win64)".encode()
    inet = wininet.InternetOpenA(agent, INTERNET_OPEN_TYPE_PRECONFIG,
                                 None, None, INTERNET_FLAG_ASYNC)
    if not inet:
        print(f"    InternetOpen FAILED, error {errname(ctypes.get_last_error())}",
              flush=True)
        return
    try:
        opt = ctypes.c_ulong(HTTP_PROTOCOL_FLAG_HTTP2)
        ok2 = wininet.InternetSetOptionW(inet, INTERNET_OPTION_ENABLE_HTTP_PROTOCOL,
                                         ctypes.byref(opt), ctypes.sizeof(opt))
        dec = ctypes.c_ulong(1)
        okd = wininet.InternetSetOptionW(inet, INTERNET_OPTION_HTTP_DECODING,
                                         ctypes.byref(dec), ctypes.sizeof(dec))
        print(f"    session open; http2-option={'ok' if ok2 else 'refused'} "
              f"decoding-option={'ok' if okd else 'refused'}", flush=True)
        wininet.InternetSetStatusCallback(inet, cb)

        headers = b"Accept-Encoding: gzip, deflate\r\n"
        flags = (INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_RELOAD
                 | INTERNET_FLAG_PRAGMA_NOCACHE)
        if url.lower().startswith("https"):
            flags |= INTERNET_FLAG_SECURE
        h = wininet.InternetOpenUrlA(inet, url.encode(), headers, len(headers),
                                     flags, ctypes.c_void_p(1))
        if h:
            state["handle"] = h
            print("    open-url returned synchronously", flush=True)
        else:
            err = ctypes.get_last_error()
            if err != 997:  # ERROR_IO_PENDING
                print(f"    InternetOpenUrl FAILED immediately, error {errname(err)}",
                      flush=True)
                return
            print("    open-url pending (async, as the engine expects)...", flush=True)

        if not done.wait(30):
            print("    REQUEST NEVER COMPLETED within 30s (no callback)", flush=True)
            return
        print(f"    request complete; async error = {errname(state['async_error'])}",
              flush=True)
        if state["async_error"]:
            return

        conn = ctypes.c_void_p(state["handle"])
        status = ctypes.c_ulong(0)
        size = ctypes.c_ulong(ctypes.sizeof(status))
        if wininet.HttpQueryInfoA(conn, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER,
                                  ctypes.byref(status), ctypes.byref(size), None):
            print(f"    HTTP status {status.value}", flush=True)
        buf = ctypes.create_string_buffer(200)
        read = ctypes.c_ulong(0)
        if wininet.InternetReadFile(conn, buf, 200, ctypes.byref(read)):
            head = " ".join(buf.raw[:read.value].decode("utf-8", "replace").split())
            print(f"    first bytes: {head[:120]!r}", flush=True)
        wininet.InternetCloseHandle(conn)
    finally:
        wininet.InternetCloseHandle(inet)


# ── 2. Drive a WinSparkle.dll (subprocess mode: --engine <dll> <url>) ─────────

def engine_check(dll_path, url):
    """Run inside the subprocess: one real check with the given engine build.
    Prints the outcome; exits 0 found/up-to-date, 2 error, 3 timeout."""
    CB = ctypes.CFUNCTYPE(None)
    result = {"outcome": None}

    cb_found = CB(lambda: result.__setitem__("outcome", "found"))
    cb_none = CB(lambda: result.__setitem__("outcome", "up-to-date"))
    cb_error = CB(lambda: result.__setitem__("outcome", "error"))

    dll = ctypes.CDLL(dll_path)
    dll.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
    dll.win_sparkle_set_app_details.argtypes = [ctypes.c_wchar_p] * 3
    dll.win_sparkle_set_automatic_check_for_updates.argtypes = [ctypes.c_int]

    dll.win_sparkle_set_appcast_url(url)
    dll.win_sparkle_set_app_details(APP_NAME, APP_NAME, APP_VERSION)
    # 0: only the explicit check below runs, so the outcome is unambiguous.
    dll.win_sparkle_set_automatic_check_for_updates(0)
    # Older engines may lack some callbacks; register what exists.
    for name, cb in (("win_sparkle_set_did_find_update_callback", cb_found),
                     ("win_sparkle_set_did_not_find_update_callback", cb_none),
                     ("win_sparkle_set_error_callback", cb_error)):
        try:
            fn = getattr(dll, name)
            fn.argtypes = [CB]
            fn(cb)
        except AttributeError:
            print(f"ENGINE note: {os.path.basename(dll_path)} lacks {name}",
                  flush=True)
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


def run_engine_test(label, dll_path, url):
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--engine", dll_path, url],
                       capture_output=True, text=True, timeout=120)
    verdict = {0: "WORKS", 2: "ERROR", 3: "TIMED OUT"}.get(p.returncode,
                                                           f"exit {p.returncode}")
    print(f"  {label:46} -> {verdict}", flush=True)
    extra = (p.stdout or "").strip()
    if extra and "note" in extra:
        print(f"      {extra}", flush=True)
    return p.returncode


def fetch_official_dll(version, outdir):
    """Download an official WinSparkle release and return its x64 DLL path."""
    zpath = os.path.join(outdir, f"ws-{version}.zip")
    urllib.request.urlretrieve(RELEASE_URL.format(v=version), zpath)
    with zipfile.ZipFile(zpath) as z:
        member = f"WinSparkle-{version}/x64/Release/WinSparkle.dll"
        z.extract(member, outdir)
    return os.path.join(outdir, member.replace("/", os.sep))


# ── Orchestration ─────────────────────────────────────────────────────────────

def main():
    if sys.platform != "win32":
        print("Windows-only diagnostic; nothing to do here.")
        return 0
    if len(sys.argv) == 4 and sys.argv[1] == "--engine":
        engine_check(sys.argv[2], sys.argv[3])  # exits

    print("=== 1. The engine's exact download recipe, step by step ===", flush=True)
    for label, url in (("winsparkle example feed", EXAMPLE_FEED),
                       ("our feed", OUR_FEED)):
        print(f"  {label}: {url}", flush=True)
        try:
            replica_fetch(url)
        except Exception as e:
            print(f"    replica crashed: {type(e).__name__}: {e}", flush=True)

    print("\n=== 2. Engine race: same harness, three official builds ===", flush=True)
    outdir = os.path.join(os.getcwd(), "diag_dlls")
    os.makedirs(outdir, exist_ok=True)
    engines = [("ours 0.9.2 (vendored)", OUR_DLL)]
    for v in RACE_VERSIONS:
        try:
            engines.append((f"official {v}", fetch_official_dll(v, outdir)))
        except Exception as e:
            print(f"  could not fetch {v}: {e}", flush=True)
    for label, dll in engines:
        for feed_label, url in (("example", EXAMPLE_FEED), ("our feed", OUR_FEED)):
            run_engine_test(f"{label} vs {feed_label}", dll, url)

    print("\nDone. Section 1 names the failing network stage and error number; "
          "section 2 says whether swapping the engine version fixes it.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
