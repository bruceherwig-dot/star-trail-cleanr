"""Check GitHub for a newer trail-detection AI model than the one in use.

What this file is for
---------------------
Star Trail CleanR ships with an AI model (the "trail detector") baked into the
app. Separately, newer/better models can be published on the project's GitHub
"Releases" page under tags that look like `model-v4`, `model-v5`, etc. This file
is the small piece of code that, on demand, asks GitHub "is there a model newer
than the one I'm running?" and, if so, hands back the details the app needs to
offer the user a download.

How it fits into the app
------------------------
- The app calls `check_for_model_update()` (the one public entry point here).
- That function compares the locally-in-use model version against every
  `model-*` release on GitHub and returns the newest one that is strictly newer.
- The returned dictionary (tag, summary, credits, download_url) is what the GUI
  uses to show an "update available" message with a download link.
- If nothing is newer, or anything goes wrong, it returns None and the app
  simply shows nothing.

Two sources of "the model I'm running"
--------------------------------------
1. A model the user previously downloaded into their personal app folder
   (`get_installed_model_version()`), which always wins if present.
2. The model bundled inside the app at build time (`BUNDLED_MODEL_VERSION`).

Fail-silent by design
---------------------
Every failure mode here -- offline, request timeout, GitHub rate-limit,
malformed JSON -- is swallowed and turned into a None return. An update check is
a "nice to have," so it must never raise an error or interrupt the user. The
caller treats None as "show nothing."
"""
import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Optional

from modules.user_folder import get_installed_model_version


def _verified_ssl_context():
    """SSL context that verifies against certifi's BUNDLED CA roots, so the
    model-update check works in the frozen app where the system root store may be
    unreachable (the same CERTIFICATE_VERIFY_FAILED that hid the update banner).
    Falls back to the default context only if certifi is unavailable."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

# The GitHub repository that hosts the app and its model releases.
REPO = "bruceherwig-dot/star-trail-cleanr"
# GitHub API endpoint that lists releases. per_page=100 is requested so that a
# single request sees ALL releases (including prereleases, which is how
# model-only releases are published) rather than just the default first page --
# important because the newest model could be buried below newer app releases.
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=100"
# How long (seconds) to wait for GitHub before giving up. Kept short so a slow
# or unreachable network never stalls the app.
TIMEOUT_S = 5

# Version label of the model shipped inside the app bundle. Bumped only when
# we publish a new app release that carries a newer bundled model. Downloaded
# models in the user folder always take precedence over this.
BUNDLED_MODEL_VERSION = "model-v5"

# Matches a model tag and captures its numeric version, e.g. "model-v4" -> "4"
# or "model-v2.5" -> "2.5". Anchored at the start so it only accepts the exact
# "model-v<number>" prefix; anything after the number (suffixes, dates) is
# ignored.
_TAG_RE = re.compile(r"^model-v(\d+(?:\.\d+)?)")


def parse_model_tag(tag) -> Optional[float]:
    """Turn a model release tag into a comparable number.

    Input: a tag string such as 'model-v2' or 'model-v2.5' (anything else,
    including None or non-strings, is treated as "not a model tag").
    Returns: the numeric version as a float (2.0, 2.5, ...) so two versions can
    be compared with `>`; or None if the tag isn't a recognizable model tag.

    Why it exists: this is the single place that defines what a valid model tag
    looks like and how its version is read, so both the local model and every
    GitHub release are compared on the same footing.
    """
    if not tag or not isinstance(tag, str):
        return None
    m = _TAG_RE.match(tag.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        # Defensive: the regex already guarantees a numeric capture, but guard
        # against any odd input so this never raises.
        return None


def parse_release_body(body) -> dict:
    """Pull a short summary and an optional credits line out of a release's notes.

    Input: the free-text body of a GitHub release (the description the author
    typed). May be None or non-string, in which case empty values come back.
    Returns: a dict with two keys, 'summary' and 'credits', each a string
    (empty string if not found).

    Extraction rules:
    - 'summary' = the first non-empty line of the body.
    - 'credits' = the text after the colon on the first line that starts with
      'Credits:' (case-insensitive), e.g. "Credits: Jane Doe" -> "Jane Doe".

    Why it exists: the GUI shows these two snippets in the update prompt, so this
    keeps the (loose, human-written) release notes format in one place.
    """
    result = {"summary": "", "credits": ""}
    if not body or not isinstance(body, str):
        return result
    # Split into lines; trailing whitespace stripped so blank-but-spaced lines
    # are still detected as blank below.
    lines = [ln.rstrip() for ln in body.splitlines()]
    # Summary = first line that has any visible text.
    for ln in lines:
        if ln.strip():
            result["summary"] = ln.strip()
            break
    # Credits = first "Credits:"-prefixed line. split(":", 1) keeps only the
    # first colon as the separator so the credit text itself may contain colons.
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("credits:"):
            result["credits"] = s.split(":", 1)[1].strip()
            break
    return result


def find_model_asset_url(assets) -> Optional[str]:
    """Find the download link for the model file attached to a release.

    Input: a release's 'assets' list as returned by the GitHub API (each asset
    is a dict with a 'name' and a 'browser_download_url').
    Returns: the download URL of the first attachment whose filename ends in
    '.pt' (the PyTorch model weights format); or None if there is no such
    attachment.

    Why it exists: a model release attaches the actual weights file (best.pt),
    and this is what the app downloads. Returning None here causes the whole
    update check to abort, so a release with no .pt file is treated as "no
    update available."
    """
    if not assets or not isinstance(assets, list):
        return None
    for a in assets:
        if not isinstance(a, dict):
            continue
        # Case-insensitive match on the .pt extension; the model weights file is
        # the only asset we care about.
        name = (a.get("name") or "").lower()
        if name.endswith(".pt"):
            url = a.get("browser_download_url")
            if url:
                return url
    return None


def local_model_version() -> str:
    """Report which model version the app is actually running right now.

    Returns: the version tag of the model in use, as a string like 'model-v4'.

    Order of precedence (this matters):
    1. A model the user downloaded into their personal app folder, if one exists
       (`get_installed_model_version()`), because a user-downloaded model is
       always newer than or equal to what shipped.
    2. Otherwise the model bundled into the app at build time
       (`BUNDLED_MODEL_VERSION`).

    Why it exists: `check_for_model_update()` needs a single answer to "what do I
    have?" before it can decide whether GitHub has anything newer.
    """
    installed = get_installed_model_version()
    return installed if installed else BUNDLED_MODEL_VERSION


def check_for_model_update() -> Optional[dict]:
    """The one public entry point: is there a newer AI model to offer the user?

    This is what the app calls. It compares the model currently in use against
    every model release on GitHub and, if a strictly-newer one is found, returns
    the details needed to prompt the user to download it.

    Returns a dict with these keys when an update exists:
        'tag'          -- the release tag, e.g. 'model-v5'
        'summary'      -- first line of the release notes (what's new)
        'credits'      -- the "Credits:" line from the notes, if any
        'download_url' -- direct link to the .pt model file to download

    Returns None (meaning "show nothing") when:
        - the local model version can't be parsed,
        - the network request / JSON parse fails for any reason (offline,
          timeout, rate-limited, malformed response),
        - there are no recognizable model-* releases,
        - the newest release is not strictly newer than what's installed, or
        - the newest release has no .pt file attached to download.
    """
    # Step 1: figure out what we already have, as a comparable number. If we
    # can't even read our own version, bail rather than risk a bad comparison.
    local_tag = local_model_version()
    local_num = parse_model_tag(local_tag)
    if local_num is None:
        return None
    # Step 2: ask GitHub for the list of releases. The whole network section is
    # wrapped so ANY failure (offline, timeout, rate limit, bad JSON) becomes a
    # quiet None instead of an error the user would see.
    try:
        req = urllib.request.Request(
            RELEASES_URL,
            headers={
                "Accept": "application/vnd.github+json",
                # GitHub requires a User-Agent header or it rejects the request.
                "User-Agent": "StarTrailCleanR-ModelUpdateCheck",
            },
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S,
                                    context=_verified_ssl_context()) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(releases, list):
        return None
    # Step 3: scan all releases and keep the model release with the highest
    # version number. Non-model releases (app releases, etc.) parse to None and
    # are skipped.
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
    # Step 4: only proceed if we found a model release that beats ours. "<="
    # means same-or-older counts as "no update."
    if best is None or best_num is None or best_num <= local_num:
        return None
    # Step 5: a model release is only usable if it has the weights file attached.
    # No .pt file -> treat as no update.
    download_url = find_model_asset_url(best.get("assets"))
    if not download_url:
        return None
    # Step 6: pull the human-readable bits and hand the package back to the GUI.
    parsed = parse_release_body(best.get("body") or "")
    return {
        "tag": best.get("tag_name"),
        "summary": parsed["summary"],
        "credits": parsed["credits"],
        "download_url": download_url,
    }
