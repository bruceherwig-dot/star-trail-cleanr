"""App-update check against GitHub's latest release.

Quiet on any failure (offline, DNS error, timeout, rate limit, parse error).
Returns None so the caller just skips the banner.
"""
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Optional

REPO = "bruceherwig-dot/star-trail-cleanr"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
TIMEOUT_S = 5

MAC_ASSET = "StarTrailCleanR-Mac-AppleSilicon.zip"
WIN_ASSET = "StarTrailCleanRSetup.zip"


def parse_tag(tag) -> Optional[float]:
    """Convert a release tag like 'v1.4-beta' to 1.4. Returns None on parse failure."""
    if not tag or not isinstance(tag, str):
        return None
    m = re.match(r"^v(\d+(?:\.\d+)?)", tag)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_local(version_str) -> Optional[float]:
    """Convert a local version.txt string like '1.406' to 1.406. None on failure."""
    if not version_str or not isinstance(version_str, str):
        return None
    try:
        return float(version_str.strip())
    except (ValueError, AttributeError):
        return None


def get_download_url() -> str:
    """Stable GitHub URL for the running platform. Auto-resolves to the latest release."""
    base = f"https://github.com/{REPO}/releases/latest/download"
    asset = MAC_ASSET if sys.platform == "darwin" else WIN_ASSET
    return f"{base}/{asset}"


def check_for_update(local_version_str: str) -> Optional[dict]:
    """Ask GitHub for the latest release and compare.

    Returns {'tag': str, 'download_url': str} when a newer release exists.
    Returns None when the user is current OR when any failure occurs.
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
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    tag = data.get("tag_name")
    remote = parse_tag(tag)
    if remote is None or remote <= local:
        return None
    return {"tag": tag, "download_url": get_download_url()}
