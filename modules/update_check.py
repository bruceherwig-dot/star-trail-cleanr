"""App-update check against GitHub's latest release.

What this file is for, in plain English:
  When Star Trail CleanR starts up, it wants to tell the user "a newer version
  is available" without getting in the way. This module is the small helper
  that answers one question: "Is there a release on GitHub newer than the copy
  the user is running?" If yes, it hands back the new version's tag and a
  download link the GUI can show in a banner. If no — or if anything at all
  goes wrong — it stays silent.

How it fits into the app:
  The GUI calls check_for_update() with the local version string (read from
  version.txt). That function contacts GitHub's public "latest release" API,
  parses the version numbers, compares them, and returns either an update
  dictionary or None. The GUI also uses get_download_url() to build a direct,
  always-current download link for the right operating system and CPU.

Why it never raises:
  Quiet on any failure (offline, DNS error, timeout, rate limit, parse error).
  An update check is a nice-to-have, never a reason to block or crash startup,
  so every failure path returns None and the caller simply skips the banner.
"""
import http.client
import json
import os
import platform
import re
import sys
import time
import traceback
import ssl
import urllib.error
import urllib.request
from typing import Optional


def _verified_ssl_context():
    """An SSL context that verifies against certifi's BUNDLED CA roots. The
    frozen app cannot rely on the system root store being reachable, which is
    what produced the silent CERTIFICATE_VERIFY_FAILED that hid the update banner
    (2026-06-20). The check now carries its own trusted roots. Falls back to the
    plain default context only if certifi is somehow unavailable."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

# Diagnostic log for the update-banner check. The check itself must never
# crash or block startup, but its failures must ALSO never be invisible:
# the orange banner silently not appearing in frozen Mac builds went unnoticed
# for an unknown number of versions because every error was swallowed with no
# trace (discovered 2026-06-10). One line per check attempt, plus the full
# traceback on failure. Same location as sparkle_debug.log so a user can be
# asked for both files at once.
_LOG_PATH = os.path.expanduser("~/.star_trail_cleanr/update_check.log")


def _log(msg: str):
    """Append a timestamped line to the diagnostic log. Best-effort, never
    raises (a logging failure must not break the check it documents)."""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass

# GitHub "owner/repo" slug for the public Star Trail CleanR repository. Used to
# build both the API URL (below) and the human-facing download URLs.
REPO = "bruceherwig-dot/star-trail-cleanr"
# GitHub REST endpoint that returns metadata for the single most recent
# NON-prerelease release. (This is why model-only "model-v*" releases must be
# published as prereleases — otherwise they would show up here and the app
# would offer a model file as if it were an app update.)
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
# Self-hosted failsafe on our own domain (website/latest.php). Consulted only when
# GitHub is unreachable (offline, firewall, or GitHub blocked at the country
# level) -- our server proxies + caches the GitHub release info, and the app can
# reach our domain even where it cannot reach GitHub. Layer 1 of the update
# failsafe design (project_update_failsafe_design).
FAILSAFE_URL = "https://api.startrailcleanr.com/latest.php"
# Default network timeout, in seconds, for the GitHub request. Kept short so a
# slow or dead network never noticeably delays the app.
TIMEOUT_S = 5

# Asset filenames published by .github/workflows/build.yml on every tag.
# Keep these in sync with the asset filenames attached in the workflow's
# Create Release step (the path: values of the upload-artifact steps), not the
# upload-artifact step name: fields.
MAC_AS_ASSET = "StarTrailCleanR-Mac-AppleSilicon.dmg"
MAC_INTEL_ASSET = "StarTrailCleanR-Mac-Intel.dmg"
# The .zip ON PURPOSE. This constant is the MANUAL download path -- the banner's
# "download it yourself" link -- where a human saves the file and opens it, and
# the zip wrapper is what keeps Edge SmartScreen from quarantining an unsigned
# .exe behind a near-hidden "Keep" option.
#
# The UPDATER is a different thing entirely and needs the bare installer: see
# scripts/publish_appcast.py, where the Windows appcast enclosure is the .exe,
# because WinSparkle executes what it downloads and a zip installs nothing.
# Both files ship every release. The full story is in AUTO_UPDATE.md
# ("WINDOWS SHIPS TWO FILES ON PURPOSE"). Do not "tidy" these into one.
WIN_ASSET = "StarTrailCleanRSetup.zip"
LINUX_ASSET = "StarTrailCleanR-Linux-x86_64.tar.gz"

# Backwards-compatible alias; older callers imported MAC_ASSET. Points at the
# Apple Silicon build (the most common Mac case) so anything still using it
# does not silently break for AS users.
MAC_ASSET = MAC_AS_ASSET


def _version_tuple(s) -> Optional[tuple]:
    """Parse 'v2.47-beta', '2.47', or '1.406' into a comparable tuple of ints
    like (2, 47). Splits on '.', reads the leading number of each component,
    and stops at the first non-numeric tail (e.g. the '-beta' suffix).
    Returns None if there is no leading numeric component.

    Component-wise integer comparison is used instead of float() so the banner
    agrees with Sparkle's own version comparator: 2.10 is NEWER than 2.9, and
    2.100 is NEWER than 2.99 — both of which a plain float comparison gets
    backwards (float('2.10') == 2.1 < 2.9). Our versions are major.build-counter
    (e.g. 2.46, 2.47, 2.48 ...), so the counter crossing 9->10 or 99->100 must
    not flip the ordering."""
    if not s or not isinstance(s, str):
        return None
    parts = []
    # Drop any leading 'v'/'V' (e.g. "v2.47"), then split on dots and read the
    # leading run of digits from each piece. Stop at the first piece with no
    # leading digit, which is how the '-beta' suffix (and anything after it)
    # gets discarded.
    for chunk in s.strip().lstrip("vV").split("."):
        m = re.match(r"\d+", chunk)
        if not m:
            break
        parts.append(int(m.group(0)))
    return tuple(parts) if parts else None


def parse_tag(tag) -> Optional[tuple]:
    """Convert a release tag like 'v2.47-beta' to a comparable version tuple
    like (2, 47). Returns None on parse failure."""
    return _version_tuple(tag)


def parse_local(version_str) -> Optional[tuple]:
    """Convert a local version.txt string like '2.47' to a comparable version
    tuple like (2, 47). Returns None on failure."""
    return _version_tuple(version_str)


def _detect_asset() -> str:
    """Pick the right release asset for this OS + CPU.

    Falls back to the Windows installer for any unrecognized combination
    rather than returning None — calling code expects a usable URL even
    in odd environments. Mac chip detection uses platform.machine() which
    returns 'arm64' for Apple Silicon and 'x86_64' for Intel; sys.platform
    alone cannot distinguish the two.
    """
    if sys.platform == "darwin":
        machine = (platform.machine() or "").lower()
        if machine in ("arm64", "aarch64"):
            return MAC_AS_ASSET
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return MAC_INTEL_ASSET
        # Unknown Mac chip — Apple Silicon is the more common case in 2026.
        return MAC_AS_ASSET
    if sys.platform.startswith("linux"):
        return LINUX_ASSET
    if sys.platform in ("win32", "cygwin"):
        return WIN_ASSET
    return WIN_ASSET


def get_download_url() -> str:
    """Stable GitHub URL for this OS + CPU. Auto-resolves to the latest release."""
    base = f"https://github.com/{REPO}/releases/latest/download"
    return f"{base}/{_detect_asset()}"


def _platform_key() -> str:
    """The latest.php download key that matches this OS + CPU, mirroring
    _detect_asset()'s platform logic (Apple Silicon vs Intel vs Windows vs Linux)."""
    if sys.platform == "darwin":
        machine = (platform.machine() or "").lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return "mac-intel"
        return "mac-as"
    if sys.platform.startswith("linux"):
        return "linux"
    return "windows"


def _failsafe_download_url(app: dict) -> Optional[str]:
    """The failsafe payload's download URL for this platform, or None if absent."""
    downloads = (app or {}).get("downloads") or {}
    url = downloads.get(_platform_key())
    return url if isinstance(url, str) and url else None


def _fetch_failsafe(timeout_s: float) -> Optional[dict]:
    """Fetch our self-hosted failsafe endpoint (website/latest.php), used only when
    GitHub is unreachable. Returns the parsed dict or None on ANY failure (a failure
    just means "show nothing"). The read is guarded exactly like the GitHub call, so
    a truncated response can never crash the banner check."""
    try:
        req = urllib.request.Request(
            FAILSAFE_URL, headers={"User-Agent": "StarTrailCleanR-Failsafe"})
        with urllib.request.urlopen(req, timeout=timeout_s,
                                    context=_verified_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, dict) else None
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
            json.JSONDecodeError, OSError, ValueError):
        return None


def check_for_update(local_version_str: str, timeout_s: float = TIMEOUT_S,
                     retries: int = 1) -> Optional[dict]:
    """Ask GitHub for the latest release and compare.

    Returns {'tag': str, 'download_url': str} when a newer release exists.
    Returns None when the user is current OR when any failure occurs.

    timeout_s lets the pre-window launch path use a tighter budget (~1.5s)
    so a slow network never visibly delays startup. retries lets the BACKGROUND
    banner check (which runs off the UI thread and never delays startup) try a
    few times before giving up: the check routinely takes ~3.5s, so a single slow
    moment was tipping it over the timeout and blanking the banner. One transient
    timeout must never mean "no update notice" for something this important.
    """
    local = parse_local(local_version_str)
    if local is None:
        _log(f"check: local version unparseable ({local_version_str!r}) -> no banner")
        return None
    attempts = max(1, retries)
    data = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                API_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "StarTrailCleanR-UpdateCheck",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout_s,
                                        context=_verified_ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
                json.JSONDecodeError, OSError, ValueError):
            # http.client.HTTPException covers IncompleteRead (a truncated chunked
            # response) and siblings, which are not OSError/URLError -- otherwise a
            # cut-off read escapes this guard and crashes (Sentry 2026-07-02).
            _log(f"check attempt {attempt}/{attempts} FAILED "
                 f"(local={local_version_str}):\n{traceback.format_exc()}")
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 5.0))   # brief backoff before retry
            # else: leave data as None and fall through to the failsafe below.
    # data holds the GitHub payload, or None if every attempt failed -- offline, or
    # GitHub blocked at the network/country level. In the None case, ask our own
    # endpoint, which the app can still reach when GitHub cannot.
    if data is not None:
        tag = data.get("tag_name")
        download_url = get_download_url()
        via_failsafe = False
    else:
        fb = _fetch_failsafe(timeout_s)
        app = (fb or {}).get("app") or {}
        tag = app.get("tag")
        # Prefer the failsafe's own download URL (Layer 2 will point this at our
        # mirror for blocked users); fall back to the GitHub URL otherwise.
        download_url = _failsafe_download_url(app) or get_download_url()
        via_failsafe = True   # GitHub was unreachable; this came from our server
        if tag:
            _log(f"check: GitHub unreachable, using failsafe -> latest={tag}")
    remote = parse_tag(tag)
    # Only surface an update when the remote version is STRICTLY greater than the
    # local one. Equal versions (user is current) and unparseable tags both fall
    # through to None, so no banner is shown.
    if remote is None or remote <= local:
        _log(f"check: local={local_version_str} latest={tag} -> current, no banner")
        return None
    _log(f"check: local={local_version_str} latest={tag} -> UPDATE, banner should show")
    # via_failsafe=True means GitHub was blocked/unreachable and this update was
    # confirmed via our own server -- the GUI then offers a manual download from
    # the website instead of the (unreachable) one-click updater.
    return {"tag": tag, "download_url": download_url, "via_failsafe": via_failsafe}
