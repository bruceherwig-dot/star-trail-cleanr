"""The in-app update checks must never crash on a truncated network response.

Locks the 2026-07-02 fix. A real user (Sentry, Windows, star-trail-cleanr@2.44 but
the same code shipped in 2.67) crashed when GitHub cut off the releases list mid-
read: resp.read() raised http.client.IncompleteRead, which is a subclass of
HTTPException -- NOT of OSError or URLError -- so the update checks' except tuple did
not catch it and it escaped to the excepthook. Both check_for_model_update (model
update) and check_for_update (app-update banner) now catch http.client.HTTPException
so a cut-off read quietly becomes "no update" instead of a crash.

If either guard regresses, an intermittent truncated response takes the app down.
"""
import http.client
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


class _TruncatedResponse:
    """A stand-in for urlopen()'s response whose read() fails the way a cut-off
    chunked transfer does: raising http.client.IncompleteRead partway through."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *a, **k):
        raise http.client.IncompleteRead(b"partial", 46098)


def _with_truncated_urlopen(call):
    """Run `call` with urllib.request.urlopen swapped for one that returns a
    response that truncates on read, then restore the real urlopen."""
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: _TruncatedResponse()
    try:
        return call()
    finally:
        urllib.request.urlopen = real


def test_model_update_survives_truncated_response():
    from modules.model_update import check_for_model_update
    result = _with_truncated_urlopen(check_for_model_update)
    assert result is None, "a truncated releases response must yield None, not raise"


def test_app_update_survives_truncated_response():
    from modules.update_check import check_for_update
    # retries=1 so the test does not sleep on the retry backoff.
    result = _with_truncated_urlopen(
        lambda: check_for_update("2.67-beta", timeout_s=0.1, retries=1))
    assert result is None, "a truncated latest-release response must yield None, not raise"


def test_both_guards_name_httpexception():
    # Lock the fix in the source too, so a future refactor of the except tuple
    # cannot silently drop the IncompleteRead family and reopen the crash.
    for fname in ("update_check.py", "model_update.py"):
        src = (REPO / "modules" / fname).read_text(encoding="utf-8")
        assert "http.client.HTTPException" in src, \
            f"{fname} must catch http.client.HTTPException (covers IncompleteRead)"


import json as _json


class _JsonResponse:
    """A stand-in urlopen response whose read() returns a fixed JSON body."""
    def __init__(self, obj):
        self._body = _json.dumps(obj).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *a, **k):
        return self._body


def _route_urlopen(failsafe_obj):
    """urlopen replacement: any api.github.com request truncates (as if GitHub is
    blocked); the api.startrailcleanr.com failsafe returns failsafe_obj as JSON."""
    def _open(req, *a, **k):
        url = req if isinstance(req, str) else getattr(req, "full_url", "")
        if "api.github.com" in url:
            return _TruncatedResponse()
        return _JsonResponse(failsafe_obj)
    return _open


def _with_urlopen(fn, call):
    real = urllib.request.urlopen
    urllib.request.urlopen = fn
    try:
        return call()
    finally:
        urllib.request.urlopen = real


def test_app_update_uses_failsafe_when_github_blocked():
    from modules.update_check import check_for_update
    fb = {"app": {"tag": "9.99-beta", "downloads": {
        "mac-as": "https://api.startrailcleanr.com/dl/mac-as",
        "mac-intel": "https://api.startrailcleanr.com/dl/mac-intel",
        "windows": "https://api.startrailcleanr.com/dl/windows",
        "linux": "https://api.startrailcleanr.com/dl/linux"}}}
    result = _with_urlopen(
        _route_urlopen(fb),
        lambda: check_for_update("2.67-beta", timeout_s=0.1, retries=1))
    assert result is not None and result["tag"] == "9.99-beta", \
        "when GitHub is blocked, the failsafe must supply the app update"
    assert isinstance(result.get("download_url"), str) and result["download_url"]
    assert result.get("via_failsafe") is True, \
        "a failsafe-sourced update must be flagged so the GUI offers a manual download"


def test_model_update_uses_failsafe_when_github_blocked():
    from modules.model_update import check_for_model_update
    fb = {"model": {"tag": "model-v99",
                    "download_url": "https://api.startrailcleanr.com/dl/best.pt",
                    "summary": "newer model", "credits": ""}}
    result = _with_urlopen(_route_urlopen(fb), check_for_model_update)
    assert result is not None and result["tag"] == "model-v99", \
        "when GitHub is blocked, the failsafe must supply the model update"
    assert result["download_url"].endswith("best.pt")
