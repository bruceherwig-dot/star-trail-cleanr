"""
Anonymous usage reporting for Star Trail CleanR ("ET Phones Home").

At the end of a successful cleaning run, the app sends ONE small anonymous
report to the backend at api.startrailcleanr.com, so the website can show
community totals (trails cleaned, hours saved, number of users) and so we can
see what formats, cameras, and settings people actually use.

PRIVACY (must always hold): no images, no file names, no file paths, no email,
no names are ever sent. A random per-install ID counts users without
identifying anyone. The server turns the connection into a coarse country and
then discards the address; the app never sends an IP or any location.

GATING: the CALLER only invokes send() when the user has opted in (the same
toggle whose wording covers usage data). Sending is fire-and-forget on a
background thread and ANY failure is swallowed silently -- it must never block a
run or show the user an error.

DEV EXCLUSION: our own test runs are flagged dev=true (running from source, or a
marker file present on a test machine) so the published stats can exclude them
without changing the app.
"""
import json
import os
import ssl
import sys
import threading
import urllib.request
import uuid

ENDPOINT = "https://api.startrailcleanr.com/collect.php"
SCHEMA_VERSION = 2
_TIMEOUT_S = 8

_APP_DIR = os.path.join(os.path.expanduser("~"), ".star_trail_cleanr")
_INSTALL_ID_FILE = os.path.join(_APP_DIR, "install_id.txt")
_DEV_SECRET_FILE = os.path.join(_APP_DIR, "stc_collect_secret.txt")
_DEV_EXCLUDE_MARKER = os.path.join(_APP_DIR, ".dev_telemetry_exclude")

# Shared secret: baked at build time by CI (gitignored `_collect_config.py`).
# Absent when running from source, where we fall back to the local dev file.
try:
    from _collect_config import SECRET as _BAKED_SECRET
except ImportError:
    _BAKED_SECRET = ""


def _get_secret():
    """Return the shared secret (baked build value, else the local dev file),
    or '' if neither is present (in which case nothing is ever sent)."""
    if _BAKED_SECRET:
        return _BAKED_SECRET
    try:
        with open(_DEV_SECRET_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def get_install_id():
    """Return a stable random anonymous install ID, creating it once on first
    use. Counts unique installs; tied to nothing personal."""
    try:
        with open(_INSTALL_ID_FILE) as f:
            existing = f.read().strip()
            if existing:
                return existing
    except OSError:
        pass
    new_id = uuid.uuid4().hex
    try:
        os.makedirs(_APP_DIR, exist_ok=True)
        with open(_INSTALL_ID_FILE, "w") as f:
            f.write(new_id + "\n")
    except OSError:
        pass
    return new_id


def _is_dev():
    """True for our own test runs: running from source (not a frozen build), or
    a deliberate marker file on a test machine. Published stats exclude these."""
    if not getattr(sys, "frozen", False):
        return True
    return os.path.exists(_DEV_EXCLUDE_MARKER)


def build_payload(facts):
    """Wrap the caller's facts with the identity + dev fields that every report
    carries. Pure (no I/O beyond reading/creating the install ID); testable."""
    payload = dict(facts or {})
    payload["schema"] = SCHEMA_VERSION
    payload["install_id"] = get_install_id()
    payload["dev"] = _is_dev()
    return payload


def _ssl_context():
    """Verify TLS against certifi where available (a frozen app has no system CA
    bundle); mirrors modules/update_check.py. Falls back to the default context."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _post(secret, payload):
    """POST one report. Never raises; any failure is silent on purpose."""
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={"Content-Type": "application/json", "X-STC-Key": secret},
        )
        urllib.request.urlopen(req, timeout=_TIMEOUT_S, context=_ssl_context()).read()
    except Exception:
        pass


def send(facts):
    """Fire-and-forget: send one anonymous run report on a background thread.
    The caller must have already confirmed the user opted in. No-op (silently)
    when no secret is configured. Never raises."""
    secret = _get_secret()
    if not secret:
        return
    try:
        payload = build_payload(facts)
        threading.Thread(target=_post, args=(secret, payload), daemon=True).start()
    except Exception:
        pass
