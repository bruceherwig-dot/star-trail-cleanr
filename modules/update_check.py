"""App-update check against GitHub's latest release.

Quiet on any failure (offline, DNS error, timeout, rate limit, parse error).
Returns None so the caller just skips the banner.
"""
import json
import platform
import re
import sys
import urllib.error
import urllib.request
from typing import Optional

REPO = "bruceherwig-dot/star-trail-cleanr"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
TIMEOUT_S = 5

# Asset filenames published by .github/workflows/build.yml on every tag.
# Keep these in sync with the artifact-upload step names in that workflow.
MAC_AS_ASSET = "StarTrailCleanR-Mac-AppleSilicon.dmg"
MAC_INTEL_ASSET = "StarTrailCleanR-Mac-Intel.dmg"
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


def check_for_update(local_version_str: str, timeout_s: float = TIMEOUT_S) -> Optional[dict]:
    """Ask GitHub for the latest release and compare.

    Returns {'tag': str, 'download_url': str} when a newer release exists.
    Returns None when the user is current OR when any failure occurs.

    timeout_s lets the pre-window launch path use a tighter budget (~1.5s)
    so a slow network never visibly delays startup. Background banner
    callers keep the default 5s.
    """
    local = parse_local(local_version_str)
    if local is None:
        return None
    try:
        req = urllib.request.Request(
            API_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "StarTrailCleanR-UpdateCheck",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    tag = data.get("tag_name")
    remote = parse_tag(tag)
    if remote is None or remote <= local:
        return None
    return {"tag": tag, "download_url": get_download_url()}
