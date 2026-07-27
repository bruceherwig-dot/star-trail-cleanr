"""Corner the Windows update-check failure on a clean CI machine.

Why this exists: the in-app Windows update check fails with WinSparkle's
"An error occurred in retrieving update information" for every tester AND on a
clean cloud Windows machine (proved by the updater live-check gate, 2026-07-27).
The engine gives no detail, so this script triangulates where the failure lives:

  1. Fetch each feed with WinInet — the exact Windows networking WinSparkle
     uses (not a browser, not Python's urllib) — and print status + first bytes.
  2. Drive the vendored WinSparkle.dll directly against several feeds:
       - WinSparkle's own example feed (the reference that works for everyone)
       - our live feed on api.startrailcleanr.com
       - the legacy GitHub Pages feed
  3. Serve VARIANTS of our feed from localhost and bisect the content:
       - an exact copy of our live feed        (content vs transport)
       - + sparkle:os="windows" on <enclosure> (the attribute we never set)
       - version moved onto the enclosure      (attribute vs child-element form)
       - the example feed served locally       (control: proves localhost works)

Each engine test runs in its own subprocess because WinSparkle can only be
initialised once per process. This is a console script, so unlike the frozen
GUI app its output is fully visible in the CI log.

Run by .github/workflows/updater-diagnose.yml (manual dispatch only).
Windows-only; exits cleanly elsewhere. No third-party dependencies.
"""
import ctypes
import http.server
import os
import subprocess
import sys
import threading
import time
import urllib.request

DLL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "vendored", "winsparkle", "WinSparkle.dll")

OUR_FEED = "https://api.startrailcleanr.com/appcast-windows.xml"
GHPAGES_FEED = "https://bruceherwig-dot.github.io/star-trail-cleanr/appcast-windows.xml"
EXAMPLE_FEED = "https://winsparkle.org/example/appcast.xml"

# Mirror the real app's identity exactly so the engine behaves as shipped.
APP_NAME = "Star Trail CleanR"
APP_VERSION = "2.81"

LOCAL_PORT = 8765


# ── 1. Raw fetch through WinInet, the plumbing WinSparkle actually uses ───────

def raw_wininet_fetch(url):
    """Fetch a URL via WinInet and return (status_or_None, detail_string)."""
    wininet = ctypes.WinDLL("wininet", use_last_error=True)
    wininet.InternetOpenW.restype = ctypes.c_void_p
    wininet.InternetOpenW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong,
                                      ctypes.c_wchar_p, ctypes.c_wchar_p,
                                      ctypes.c_ulong]
    wininet.InternetOpenUrlW.restype = ctypes.c_void_p
    wininet.InternetOpenUrlW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                         ctypes.c_wchar_p, ctypes.c_ulong,
                                         ctypes.c_ulong, ctypes.c_void_p]
    wininet.HttpQueryInfoW.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                       ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p]
    wininet.InternetReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_ulong, ctypes.c_void_p]
    wininet.InternetCloseHandle.argtypes = [ctypes.c_void_p]

    INTERNET_OPEN_TYPE_PRECONFIG = 0
    INTERNET_FLAG_RELOAD = 0x80000000
    INTERNET_FLAG_NO_CACHE_WRITE = 0x04000000
    HTTP_QUERY_STATUS_CODE = 19
    HTTP_QUERY_FLAG_NUMBER = 0x20000000

    h = wininet.InternetOpenW(f"{APP_NAME}/{APP_VERSION}",
                              INTERNET_OPEN_TYPE_PRECONFIG, None, None, 0)
    if not h:
        return None, f"InternetOpen failed, Windows error {ctypes.get_last_error()}"
    try:
        hu = wininet.InternetOpenUrlW(
            h, url, None, 0,
            INTERNET_FLAG_RELOAD | INTERNET_FLAG_NO_CACHE_WRITE, None)
        if not hu:
            return None, f"InternetOpenUrl failed, Windows error {ctypes.get_last_error()}"
        try:
            status = ctypes.c_ulong(0)
            size = ctypes.c_ulong(ctypes.sizeof(status))
            ok = wininet.HttpQueryInfoW(
                hu, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER,
                ctypes.byref(status), ctypes.byref(size), None)
            buf = ctypes.create_string_buffer(400)
            read = ctypes.c_ulong(0)
            wininet.InternetReadFile(hu, buf, 400, ctypes.byref(read))
            head = buf.raw[:read.value].decode("utf-8", "replace")
            head = " ".join(head.split())[:160]
            return (status.value if ok else None), f"first bytes: {head!r}"
        finally:
            wininet.InternetCloseHandle(hu)
    finally:
        wininet.InternetCloseHandle(h)


# ── 2. Drive WinSparkle.dll itself (subprocess mode: --engine <url>) ──────────

def engine_check(url):
    """Run inside the subprocess: one real WinSparkle check against `url`.
    Prints the outcome and exits 0 found / 0 up-to-date / 2 error / 3 timeout."""
    CB = ctypes.CFUNCTYPE(None)
    result = {"outcome": None}

    cb_found = CB(lambda: result.__setitem__("outcome", "found"))
    cb_none = CB(lambda: result.__setitem__("outcome", "up-to-date"))
    cb_error = CB(lambda: result.__setitem__("outcome", "error"))

    dll = ctypes.CDLL(DLL_PATH)
    dll.win_sparkle_set_appcast_url.argtypes = [ctypes.c_wchar_p]
    dll.win_sparkle_set_app_details.argtypes = [ctypes.c_wchar_p] * 3
    dll.win_sparkle_set_automatic_check_for_updates.argtypes = [ctypes.c_int]
    dll.win_sparkle_set_did_find_update_callback.argtypes = [CB]
    dll.win_sparkle_set_did_not_find_update_callback.argtypes = [CB]
    dll.win_sparkle_set_error_callback.argtypes = [CB]

    dll.win_sparkle_set_appcast_url(url)
    dll.win_sparkle_set_app_details(APP_NAME, APP_NAME, APP_VERSION)
    # 0: only the explicit check below runs, so the outcome is unambiguous.
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
    print(f"ENGINE {outcome}: {url}", flush=True)
    try:
        dll.win_sparkle_cleanup()
    except Exception:
        pass
    sys.exit({"found": 0, "up-to-date": 0, "error": 2}.get(outcome, 3))


def run_engine_test(label, url):
    """Spawn the subprocess engine test and report its verdict."""
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--engine", url],
                       capture_output=True, text=True, timeout=120)
    verdict = {0: "WORKS", 2: "ERROR", 3: "TIMED OUT"}.get(p.returncode,
                                                           f"exit {p.returncode}")
    detail = (p.stdout or "").strip()
    print(f"  {label:34} -> {verdict}   ({detail})", flush=True)
    return p.returncode


# ── 3. Local feed variants for content bisection ──────────────────────────────

def make_variants(our_xml, example_xml, outdir):
    """Write the bisection variants; returns [(label, filename)]."""
    v = []

    def w(name, text):
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
            f.write(text)
        v.append(name)

    w("v1_ours_exact.xml", our_xml)
    w("v2_ours_os_attr.xml",
      our_xml.replace("<enclosure url=", '<enclosure sparkle:os="windows" url=', 1))
    # v3: version as enclosure ATTRIBUTES instead of child elements.
    import re
    m_ver = re.search(r"<sparkle:version>([^<]+)</sparkle:version>", our_xml)
    m_short = re.search(r"<sparkle:shortVersionString>([^<]+)</sparkle:shortVersionString>",
                        our_xml)
    v3 = our_xml
    if m_ver and m_short:
        v3 = re.sub(r"\s*<sparkle:version>[^<]+</sparkle:version>", "", v3)
        v3 = re.sub(r"\s*<sparkle:shortVersionString>[^<]+</sparkle:shortVersionString>",
                    "", v3)
        v3 = v3.replace(
            "<enclosure url=",
            f'<enclosure sparkle:version="{m_ver.group(1)}" '
            f'sparkle:shortVersionString="{m_short.group(1)}" url=', 1)
    w("v3_ours_version_as_attr.xml", v3)
    w("v4_example_control.xml", example_xml)
    return v


def serve_dir(path):
    """Serve `path` on localhost in a daemon thread; returns the server."""
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(
        *a, directory=path, **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", LOCAL_PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ── Orchestration ─────────────────────────────────────────────────────────────

def main():
    if sys.platform != "win32":
        print("Windows-only diagnostic; nothing to do here.")
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--engine":
        engine_check(sys.argv[2])  # exits

    print(f"WinSparkle.dll: {DLL_PATH}  exists={os.path.isfile(DLL_PATH)}", flush=True)

    print("\n=== 1. Raw fetches via WinInet (the networking WinSparkle uses) ===",
          flush=True)
    for label, url in (("our server", OUR_FEED),
                       ("github pages", GHPAGES_FEED),
                       ("winsparkle example", EXAMPLE_FEED)):
        status, detail = raw_wininet_fetch(url)
        print(f"  {label:20} -> HTTP {status}   {detail}", flush=True)

    print("\n=== 2. The engine itself, against remote feeds ===", flush=True)
    run_engine_test("example feed (reference)", EXAMPLE_FEED)
    run_engine_test("our feed (live server)", OUR_FEED)
    run_engine_test("our feed (github pages)", GHPAGES_FEED)

    print("\n=== 3. The engine against local variants of our feed ===", flush=True)
    our_xml = urllib.request.urlopen(OUR_FEED, timeout=30).read().decode("utf-8")
    example_xml = urllib.request.urlopen(EXAMPLE_FEED, timeout=30).read().decode("utf-8")
    outdir = os.path.join(os.getcwd(), "diag_feeds")
    os.makedirs(outdir, exist_ok=True)
    names = make_variants(our_xml, example_xml, outdir)
    srv = serve_dir(outdir)
    try:
        for name in names:
            run_engine_test(name, f"http://127.0.0.1:{LOCAL_PORT}/{name}")
    finally:
        srv.shutdown()

    print("\nDone. Read the three sections together: WinInet shows what the wire "
          "says, section 2 splits engine-vs-feed, section 3 names the line.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
