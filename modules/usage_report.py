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
import time
import urllib.request
import uuid

ENDPOINT = "https://api.startrailcleanr.com/collect.php"
SCHEMA_VERSION = 3          # 3 adds previous_version / updated_via
_TIMEOUT_S = 8

_APP_DIR = os.path.join(os.path.expanduser("~"), ".star_trail_cleanr")
_INSTALL_ID_FILE = os.path.join(_APP_DIR, "install_id.txt")
_DEV_SECRET_FILE = os.path.join(_APP_DIR, "stc_collect_secret.txt")
_DEV_EXCLUDE_MARKER = os.path.join(_APP_DIR, ".dev_telemetry_exclude")
# Which version this install last reported, and whether the in-app updater was
# used to leave it. See _version_change() -- these two files exist to answer one
# question we have never been able to answer: does the one-click update work?
_LAST_VERSION_FILE = os.path.join(_APP_DIR, "last_version.txt")
_UPDATER_MARKER = os.path.join(_APP_DIR, "updater_engaged.txt")
_UPDATER_WINDOW_S = 7 * 24 * 3600

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


def note_updater_engaged():
    """Record that the IN-APP updater was just used. Called the moment the user
    sets an update going; harmless and silent if it can't be written.

    Without this we cannot tell an in-place update from someone downloading the
    installer again by hand: both look identical in the data (same install,
    new version). Telling them apart is the entire point -- as of 2026-08-18 no
    Windows machine had ever been OBSERVED updating in place, and we could not
    say whether that meant the updater was broken or simply that people
    re-download."""
    try:
        os.makedirs(_APP_DIR, exist_ok=True)
        with open(_UPDATER_MARKER, "w") as f:
            f.write(str(int(time.time())) + "\n")
    except Exception:
        # Deliberately every exception, not just OSError. This runs inside the
        # updater path, and a bad path raises ValueError rather than OSError --
        # a note about telemetry must never be able to stop an update.
        pass


def _version_change(current):
    """How this install got to the version it is running now.

    Returns (previous_version, updated_via) where updated_via is 'in_app' when
    the updater was engaged in the last week, else 'manual'. Both are None on a
    fresh install and on every ordinary run -- the fields only ever appear on
    the FIRST run after the version actually changed, so this is a handful of
    reports per release, not a new field on every run."""
    if not current:
        return None, None
    previous = None
    try:
        with open(_LAST_VERSION_FILE) as f:
            previous = f.read().strip() or None
    except Exception:
        pass

    if previous != current:
        try:
            os.makedirs(_APP_DIR, exist_ok=True)
            with open(_LAST_VERSION_FILE, "w") as f:
                f.write(str(current) + "\n")
        except Exception:
            pass

    if not previous or previous == current:
        return None, None       # fresh install, or nothing changed

    via = "manual"
    try:
        if os.path.exists(_UPDATER_MARKER):
            age = time.time() - os.path.getmtime(_UPDATER_MARKER)
            if age < _UPDATER_WINDOW_S:
                via = "in_app"
            os.remove(_UPDATER_MARKER)
    except Exception:
        pass
    return previous, via


def build_payload(facts):
    """Wrap the caller's facts with the identity + dev fields that every report
    carries, and -- on the first run after a version change only -- the version
    it came from and whether the in-app updater brought it there.

    Does small I/O in the user's own app folder (install ID, last version, the
    updater marker) and nothing else; still testable by pointing those paths
    somewhere temporary."""
    payload = dict(facts or {})
    payload["schema"] = SCHEMA_VERSION
    payload["install_id"] = get_install_id()
    payload["dev"] = _is_dev()
    previous, via = _version_change(payload.get("app_version"))
    if previous:
        payload["previous_version"] = previous
        payload["updated_via"] = via
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
