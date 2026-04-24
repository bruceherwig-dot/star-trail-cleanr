"""Model-update check against the GitHub repo's releases.

Queries /releases, filters to tags starting with 'model-', returns the newest
one if it beats the currently installed (user folder) or bundled model.

Silent on any failure: offline, timeout, rate limit, parse error. Caller
interprets None as "show nothing."
"""
import json
import re
import urllib.error
import urllib.request
from typing import Optional

from modules.user_folder import get_installed_model_version

REPO = "bruceherwig-dot/star-trail-cleanr"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
TIMEOUT_S = 5

# Version label of the model shipped inside the app bundle. Bumped only when
# we publish a new app release that carries a newer bundled model. Downloaded
# models in the user folder always take precedence over this.
BUNDLED_MODEL_VERSION = "model-v2"

_TAG_RE = re.compile(r"^model-v(\d+(?:\.\d+)?)")


def parse_model_tag(tag) -> Optional[float]:
    """Convert 'model-v2' or 'model-v2.5' to a float. None on parse failure."""
    if not tag or not isinstance(tag, str):
        return None
    m = _TAG_RE.match(tag.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_release_body(body) -> dict:
    """Parse the free-text release body into a summary line and a credits line.

    Rules:
    - First non-empty line is the summary.
    - Any line starting with 'Credits:' (case-insensitive) becomes the credits line.
    """
    result = {"summary": "", "credits": ""}
    if not body or not isinstance(body, str):
        return result
    lines = [ln.rstrip() for ln in body.splitlines()]
    for ln in lines:
        if ln.strip():
            result["summary"] = ln.strip()
            break
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("credits:"):
            result["credits"] = s.split(":", 1)[1].strip()
            break
    return result


def find_model_asset_url(assets) -> Optional[str]:
    """Pick the browser_download_url of the first .pt asset in a release's asset list."""
    if not assets or not isinstance(assets, list):
        return None
    for a in assets:
        if not isinstance(a, dict):
            continue
        name = (a.get("name") or "").lower()
        if name.endswith(".pt"):
            url = a.get("browser_download_url")
            if url:
                return url
    return None


def local_model_version() -> str:
    """Return the currently-in-use model version: user folder first, else bundled."""
    installed = get_installed_model_version()
    return installed if installed else BUNDLED_MODEL_VERSION


def check_for_model_update() -> Optional[dict]:
    """Return {'tag', 'summary', 'credits', 'download_url'} when a newer model release exists.

    Returns None when the user is current, when no model-* releases exist, or
    when any network / parse / asset-discovery failure occurs.
    """
    local_tag = local_model_version()
    local_num = parse_model_tag(local_tag)
    if local_num is None:
        return None
    try:
        req = urllib.request.Request(
            RELEASES_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "StarTrailCleanR-ModelUpdateCheck",
            },
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(releases, list):
        return None
    best = None
    best_num = None
    for rel in releases:
        if not isinstance(rel, dict):
            continue
        num = parse_model_tag(rel.get("tag_name"))
        if num is None:
            continue
        if best_num is None or num > best_num:
            best = rel
            best_num = num
    if best is None or best_num is None or best_num <= local_num:
        return None
    download_url = find_model_asset_url(best.get("assets"))
    if not download_url:
        return None
    parsed = parse_release_body(best.get("body") or "")
    return {
        "tag": best.get("tag_name"),
        "summary": parsed["summary"],
        "credits": parsed["credits"],
        "download_url": download_url,
    }
