"""Publish the Star Trail CleanR auto-update feeds (appcasts).

WHAT SUCCESS LOOKS LIKE (check these, do not assume the job going green means
the release reached anyone):
  - All six feeds advertise the new version: three on GitHub Pages
    (appcast-windows.xml, appcast-mac-apple-silicon.xml, appcast-mac-intel.xml)
    and the same three on the mirror at api.startrailcleanr.com, which is what
    the app actually reads from v2.80 on.
  - The mirror feeds' enclosure URLs point at the MIRROR copies of the
    installers, and those files exist at the advertised byte length.
  - https://api.github.com/.../releases/latest resolves to the new tag. That is
    a SEPARATE channel: the orange in-app banner reads it, Sparkle does not.
  - The Windows enclosure is the .exe installer, never the .zip. WinSparkle
    executes what it downloads; a zip opens Explorer and installs nothing.

If any of that is wrong the release did NOT ship, however green the build looks.
This script hard-fails rather than publishing a feed that would leave users
stranded, so a failure here is the system working. Read the error and fix the
cause; never tag around it.

This is the SIMPLE, STURDY publisher that runs automatically inside GitHub
Actions on every release tag (see .github/workflows/build.yml, job
`publish-appcast`). It replaces the old delta-based local flow
(scripts/release_signer.py) for normal releases.

What it does, per platform (Apple Silicon, Intel, Windows):
  1. Sign the just-built installer with the Sparkle ed25519 key
     (`sign_update -p -f <key> <file>`), producing an EdDSA signature + byte
     length.
  2. Read the platform's current appcast XML and PREPEND one new <item> for
     this version, pointing at this version's own GitHub release download URL.
     It never rewrites the URLs of older items, so it cannot corrupt history
     the way a global find/replace can.
  3. The workflow then commits + pushes the three updated XMLs to gh-pages.

Why no deltas: Sparkle's generate_appcast delta step was failing on our bundle
("Diffing code signed extended attributes are not supported") and silently
shipping zero deltas anyway, while also mislabeling old-version URLs. Full
download per update is what already happens in practice; this path makes it
correct and reliable. Deltas can be reintroduced later as an optimization.

Two modes:
  publish:  python publish_appcast.py <tag> --artifacts-dir DIR --key-file F \
                                            --gh-pages-dir DIR
  verify:   python publish_appcast.py <tag> --verify
            (fetches all three LIVE feeds and exits non-zero unless every one
             advertises <tag> at the top — this is the hard gate that makes a
             half-published or unpublished release fail the build loudly.)

Idempotent: if a feed's top item is already this version, publish leaves it
untouched and reports success, so re-runs are safe.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# --- Fixed locations and identifiers used throughout this script ---
# REPO_ROOT is the project's top folder (two levels up from this script).
REPO_ROOT = Path(__file__).resolve().parent.parent
# The Sparkle command-line signing tool shipped inside the repo. Used to sign
# each installer so the app can verify the download is genuine before updating.
SIGN_UPDATE = REPO_ROOT / "vendored" / "sparkle-bin" / "sign_update"
# The GitHub repo that hosts both the release downloads and the gh-pages feeds.
GH_REPO = "bruceherwig-dot/star-trail-cleanr"
# Base URL where a tagged release's installer files live on GitHub.
RELEASE_BASE = f"https://github.com/{GH_REPO}/releases/download"
# Base URL where the live appcast feeds are served from (GitHub Pages).
# Still published every release: installs older than v2.80 read these.
PAGES_BASE = "https://bruceherwig-dot.github.io/star-trail-cleanr"
# Our own server: the feeds the app reads from v2.80 on (chosen 2026-07-24
# after a tester's machine blocked the updater's GitHub fetch while our site
# worked fine). Single-item feeds whose download links point at the mirror
# installer copies uploaded by scripts/mirror_upload.py each release.
MIRROR_BASE = "https://api.startrailcleanr.com"
MIRROR_DL = f"{MIRROR_BASE}/downloads"
MIRROR_DIR = "/home/dh_bmigjp/api.startrailcleanr.com"
MIRROR_HOST = "pdx1-shared-a4-09.dreamhost.com"
MIRROR_USER = "dh_bmigjp"
# XML namespace Sparkle uses for its custom tags (version, signature, etc.).
SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"

# Per-platform release/appcast wiring. Each entry maps one platform to its
# installer filename ("release_filename") and its feed filename ("appcast").
# `extra` holds the platform-specific Sparkle item fields observed in the
# existing feeds (minimum OS version, CPU requirement) so new items match what
# older items advertise. The leading spaces inside these strings are
# intentional: they pre-indent the raw XML lines to line up with the rest of the
# generated <item> block in build_item_xml.
PLATFORMS = [
    {
        "key": "mac-apple-silicon",
        "release_filename": "StarTrailCleanR-Mac-AppleSilicon.dmg",
        "appcast": "appcast-mac-apple-silicon.xml",
        "extra": (
            "            <sparkle:minimumSystemVersion>10.13</sparkle:minimumSystemVersion>\n"
            "            <sparkle:hardwareRequirements>arm64</sparkle:hardwareRequirements>\n"
        ),
    },
    {
        "key": "mac-intel",
        "release_filename": "StarTrailCleanR-Mac-Intel.dmg",
        "appcast": "appcast-mac-intel.xml",
        "extra": (
            "            <sparkle:minimumSystemVersion>10.13</sparkle:minimumSystemVersion>\n"
        ),
    },
    {
        # THE ENCLOSURE MUST BE THE .exe, NOT THE .zip. WinSparkle hands the
        # downloaded file to Windows and lets the file association decide what
        # to do with it (ShellExecuteEx, no arguments -- winsparkle/src/ui.cpp),
        # so a zip opens an Explorer window and installs NOTHING. That is what
        # made every Windows one-click update do nothing from the day the
        # updater shipped: proven on a clean Windows runner on 2026-08-21, which
        # installed 2.84, ran what this feed advertised, and was still on 2.84
        # afterwards. WinSparkle's own guide: "the enclosure is typically some
        # kind of installer: an MSI, Inno Setup installer, NSIS installer".
        # The .zip still ships and is still what the website serves -- it exists
        # to dodge Edge SmartScreen for people downloading by hand. Only the
        # UPDATER needs the bare installer. Do not "tidy" these back together.
        "key": "windows",
        "release_filename": "StarTrailCleanRSetup.exe",
        "appcast": "appcast-windows.xml",
        "extra": "",
    },
]


def parse_version(tag):
    """Split a release tag into the two version strings Sparkle needs.

    Input: a git tag like 'v2.47-beta'.
    Returns a pair: (numeric, short). For 'v2.47-beta' that's ('2.47',
    '2.47-beta'). The numeric form is the bare number Sparkle compares to decide
    "is this newer?"; the short form is the human-readable label shown in the
    update dialog. Exits the whole program if the tag is malformed, because every
    later step depends on a valid version.
    """
    # Capture the leading number (e.g. "2.47") right after the "v"; the optional
    # group allows a single-component tag like "v3" as well as "v3.1".
    m = re.match(r"v(\d+(?:\.\d+)?)", tag)
    if not m:
        sys.exit(f"Could not parse version from tag '{tag}'. Expected like v2.47-beta.")
    return m.group(1), tag.lstrip("v")


def find_installer(artifacts_dir, filename):
    """Find one platform's installer file inside the downloaded artifacts folder.

    Inputs: artifacts_dir is the folder GitHub Actions downloaded the build
    outputs into; filename is the installer to look for (e.g.
    'StarTrailCleanR-Mac-AppleSilicon.dmg'). Returns the full path to the first
    match. Searches every subfolder ('**', recursive) because GitHub's
    download-artifact step drops each artifact into its own nested folder, so the
    file is never at a fixed depth. Exits if the installer is missing, since
    there's nothing to sign or publish without it.
    """
    hits = glob.glob(os.path.join(artifacts_dir, "**", filename), recursive=True)
    if not hits:
        sys.exit(f"Installer '{filename}' not found anywhere under {artifacts_dir}")
    return hits[0]


def sign_file(path, key_file):
    """Cryptographically sign one installer with Sparkle's signing tool.

    Inputs: path is the installer to sign; key_file is the private ed25519 key.
    Runs the bundled `sign_update` tool, which prints the EdDSA signature to
    standard output. Returns a pair: (ed_signature_string, file_size_in_bytes).
    Both values go into the appcast <enclosure> so the user's copy of the app can
    confirm the download is authentic and complete before installing. Exits if
    the signing tool reports a failure.
    """
    # -p prints the signature to stdout instead of editing a file; -f points the
    # tool at the private key.
    result = subprocess.run(
        [str(SIGN_UPDATE), "-p", "-f", str(key_file), str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"sign_update failed for {path}:\n{result.stderr}")
    return result.stdout.strip(), os.path.getsize(path)


def top_item_version(xml_text):
    """Read the version number of the newest entry in an appcast feed.

    Input: the full text of an appcast XML document. Returns the numeric version
    string (e.g. '2.47') from the FIRST <item> in the feed, which by convention
    is the newest. Returns None if the XML won't parse, has no items, or the item
    has no version tag. Used both to skip work when a feed already advertises this
    release, and to confirm during verification that a feed is now live.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Malformed or partially-fetched XML — treat as "no known version".
        return None
    item = root.find(".//item")
    if item is None:
        return None
    return item.findtext(f"{{{SPARKLE_NS}}}version")


def build_item_xml(platform, tag, numeric, short, sig, length, pub_date):
    """Assemble the appcast <item> XML block describing this release.

    Inputs: platform is one entry from PLATFORMS (filenames + per-platform extra
    tags); tag is the git tag (e.g. 'v2.47-beta') used to build the download URL;
    numeric and short are the two version strings from parse_version; sig and
    length are the signature and byte size from sign_file; pub_date is the
    formatted publish timestamp. Returns the <item> block as a ready-to-insert
    text string. This is the single entry Sparkle reads to offer the update: its
    title, version, download URL, size, and signature. The download URL points at
    THIS version's own GitHub release, so older items' URLs are never disturbed.
    """
    url = f"{RELEASE_BASE}/{tag}/{platform['release_filename']}"
    return (
        "        <item>\n"
        f"            <title>Version {short}</title>\n"
        f"            <pubDate>{pub_date}</pubDate>\n"
        f"            <sparkle:version>{numeric}</sparkle:version>\n"
        f"            <sparkle:shortVersionString>{short}</sparkle:shortVersionString>\n"
        f"{platform['extra']}"
        f'            <enclosure url="{url}" length="{length}" '
        f'type="application/octet-stream" sparkle:edSignature="{sig}"/>\n'
        "        </item>\n"
    )


def insert_item(xml_text, item_xml):
    """Splice a new <item> block into the feed so it lands as the newest entry.

    Inputs: xml_text is the current feed; item_xml is the block from
    build_item_xml. Returns the full feed text with the new item inserted.

    The new item is placed right after the channel's <language> line, which sits
    above every <item>, so the fresh release becomes the first (newest) item. If
    there's no <language> tag, it falls back to inserting just before the first
    existing <item>. This text-splice approach (rather than re-serializing the
    XML) is deliberate: it leaves every older item byte-for-byte untouched, which
    is the whole safety guarantee of this publisher. Exits if the feed has
    neither a <language> tag nor any <item> to anchor against.
    """
    # Preferred anchor: insert immediately after the <language> line.
    m = re.search(r"<language>[^<]*</language>\s*\n", xml_text)
    if m:
        at = m.end()
    else:
        # Fallback anchor: insert just before the first <item>.
        m2 = re.search(r"[ \t]*<item>", xml_text)
        if not m2:
            sys.exit("Appcast has no <language> tag and no <item> to anchor insertion.")
        at = m2.start()
    return xml_text[:at] + item_xml + xml_text[at:]


def fetch_live(appcast, timeout=20):
    """Download the currently-published version of one feed from GitHub Pages.

    Input: appcast is the feed's filename (e.g. 'appcast-windows.xml'); timeout
    is the per-request network timeout in seconds. Returns the live feed's text.
    This reads what users' apps would actually see, NOT the local file. The
    '?cb=<timestamp>' on the URL plus the no-cache headers force a fresh copy past
    GitHub's CDN, which otherwise keeps serving a stale cached feed for a while
    after a push — important so verification doesn't pass or fail on old data.
    """
    # Unique query string each call busts the CDN cache so we always see the
    # newest published feed.
    return fetch_url(f"{PAGES_BASE}/{appcast}", timeout=timeout)


def fetch_url(url, timeout=20):
    """Download one URL as text with cache-busting, shared by both the GitHub
    Pages fetch (fetch_live) and the mirror-feed verification."""
    full = f"{url}?cb={int(time.time())}"
    req = urllib.request.Request(
        full, headers={"Cache-Control": "no-cache", "Pragma": "no-cache",
                       "User-Agent": "StarTrailCleanR-AppcastPublish"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def head_length(url, timeout=20):
    """Content length of one URL from a HEAD request (no download), used to
    confirm each mirror installer matches the size its feed item advertises."""
    req = urllib.request.Request(
        f"{url}?cb={int(time.time())}", method="HEAD",
        headers={"User-Agent": "StarTrailCleanR-AppcastPublish"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.headers.get("Content-Length", -1))


def do_publish(args):
    """Publish mode: sign each installer and prepend its <item> to every feed.

    Input: args is the parsed command line (needs tag, artifacts_dir, key_file,
    gh_pages_dir). For each of the three platforms it updates the appcast file in
    the local gh-pages working copy on disk; it does NOT commit or push (the
    GitHub Actions workflow does that afterward). This is the half of the script
    that actually changes feeds; do_verify is the read-only check.

    Per platform the steps are: skip if the feed already tops out at this version
    (idempotent, so re-runs are safe), otherwise find the installer, sign it,
    build the new item, and splice it in as the newest entry.
    """
    numeric, short = parse_version(args.tag)
    # Fail fast if the signing tool or private key is missing — nothing can be
    # signed without both.
    if not SIGN_UPDATE.exists():
        sys.exit(f"Missing signing tool: {SIGN_UPDATE}")
    if not Path(args.key_file).exists():
        sys.exit(f"Signing key not found: {args.key_file}")
    # RFC-822 style date Sparkle expects in <pubDate>, always in UTC (+0000).
    pub_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    gh = Path(args.gh_pages_dir)

    for platform in PLATFORMS:
        print(f"\n=== {platform['key']} ===")
        xml_path = gh / platform["appcast"]
        if not xml_path.exists():
            sys.exit(f"Appcast not found in gh-pages dir: {xml_path}")
        xml_text = xml_path.read_text()

        # Idempotency guard: if this version is already on top, do nothing so a
        # repeated run can't add a duplicate item.
        if top_item_version(xml_text) == numeric:
            print(f"  already advertises {numeric} at top — leaving untouched")
            continue

        installer = find_installer(args.artifacts_dir, platform["release_filename"])
        sig, length = sign_file(installer, args.key_file)
        size_mb = length / (1024 * 1024)
        print(f"  signed {platform['release_filename']} ({size_mb:.1f} MB)")
        item = build_item_xml(platform, args.tag, numeric, short, sig, length, pub_date)
        xml_path.write_text(insert_item(xml_text, item))
        print(f"  prepended Version {short} -> {platform['appcast']}")

    print("\nAppcast files updated. The workflow will commit + push them to gh-pages.")


def do_publish_mirror(args):
    """Mirror mode: publish single-item feeds on our own server.

    From v2.80 the app reads its update feed from api.startrailcleanr.com, so a
    machine that blocks GitHub (VPN, firewall, security software, or a country
    block) can still check for and download updates. For each platform this:

      1. Fetches the LIVE GitHub Pages feed (the publish-appcast job has already
         verified it tops out at this release).
      2. Takes only its newest <item> and repoints the download link at the
         mirror installer copy (same bytes, so the signature still verifies —
         the mirror holds only the CURRENT release under stable filenames,
         which is why the mirror feed carries one item, not history).
      3. Uploads the feed to the server, then HARD-FAILS unless the live mirror
         feed advertises this version AND each mirror installer's actual size
         matches the size the feed item advertises (catches a mirror that
         didn't refresh).

    Needs DREAMHOST_PASSWORD in the environment; a missing password is a
    FAILURE, not a skip — these feeds are what shipped apps read."""
    numeric, _ = parse_version(args.tag)
    password = os.environ.get("DREAMHOST_PASSWORD", "")
    if not password:
        sys.exit("DREAMHOST_PASSWORD not set — the mirror feeds are what the "
                 "app reads; refusing to skip.")
    import io
    import paramiko

    uploads = {}
    for platform in PLATFORMS:
        appcast = platform["appcast"]
        live = fetch_live(appcast)
        top = top_item_version(live)
        if top != numeric:
            sys.exit(f"{appcast}: GitHub Pages feed tops at {top!r}, expected "
                     f"{numeric!r} — publish-appcast must succeed first.")
        start = live.find("<item>")
        end = live.rfind("</item>") + len("</item>")
        if start == -1 or end <= start:
            sys.exit(f"{appcast}: no <item> found in live feed")
        first_end = live.index("</item>", start) + len("</item>")
        item = live[start:first_end]
        # THE GATE THAT WOULD HAVE CAUGHT THIS IN APRIL.
        # On Windows, WinSparkle executes whatever the enclosure points at and
        # lets the file association decide (ShellExecuteEx, no arguments). Point
        # it at an archive and Windows opens an Explorer window: nothing
        # installs, no error is raised, and the user is left on the old version
        # believing they updated. That is exactly what shipped from the day the
        # Windows updater went live until 2026-08-21, through five separate
        # "updater fixes", because nothing ever checked WHAT we were advertising.
        # Proven on a clean Windows runner: installed 2.84, ran what the feed
        # offered, still 2.84.
        if platform["key"] == "windows":
            fn = platform["release_filename"]
            if not fn.lower().endswith((".exe", ".msi")):
                sys.exit(
                    f"{appcast}: the Windows enclosure is {fn!r}, which Windows "
                    "cannot install by executing it. WinSparkle's guide: 'the "
                    "enclosure is typically some kind of installer: an MSI, Inno "
                    "Setup installer, NSIS installer, and so on.' Point the feed "
                    "at the installer. The .zip is for people downloading by "
                    "hand (it dodges SmartScreen); it is not an update.")
        gh_url = f"{RELEASE_BASE}/{args.tag}/{platform['release_filename']}"
        mirror_url = f"{MIRROR_DL}/{platform['release_filename']}"
        if gh_url not in item:
            sys.exit(f"{appcast}: newest item does not reference {gh_url}; "
                     "refusing to guess which link to repoint.")
        item = item.replace(gh_url, mirror_url)
        feed = live[:start] + item + live[end:]
        uploads[appcast] = (feed, item, mirror_url)

    t = paramiko.Transport((MIRROR_HOST, 22))
    t.connect(username=MIRROR_USER, password=password)
    sftp = paramiko.SFTPClient.from_transport(t)
    for appcast, (feed, _item, _url) in uploads.items():
        remote = f"{MIRROR_DIR}/{appcast}"
        sftp.putfo(io.BytesIO(feed.encode("utf-8")), remote)
        sftp.chmod(remote, 0o644)
        print(f"uploaded {appcast} -> {remote}")
    sftp.close()
    t.close()

    # Hard gate: the live mirror feeds must advertise this version, and each
    # mirror binary must match the byte length its feed item promises.
    failures = []
    for appcast, (_feed, item, mirror_url) in uploads.items():
        live = fetch_url(f"{MIRROR_BASE}/{appcast}")
        top = top_item_version(live)
        if top != numeric:
            failures.append(f"{appcast}: live mirror feed tops at {top!r}")
            continue
        m = re.search(r'length="(\d+)"', item)
        want = int(m.group(1)) if m else -1
        got = head_length(mirror_url)
        if got != want:
            failures.append(f"{appcast}: mirror installer {mirror_url} is "
                            f"{got} bytes, feed advertises {want} — mirror "
                            "out of sync with this release")
            continue
        print(f"  OK  {appcast} -> {numeric}, installer size matches ({got} bytes)")
    if failures:
        print("\nFAILED mirror publish:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"\nAll mirror feeds live at {numeric} with matching installers.")


def do_verify(args):
    """Verify mode: poll the three LIVE feeds until all advertise this version.

    Input: args needs tag, timeout (max seconds to wait), and interval (seconds
    between polls). Returns nothing on success; calls sys.exit(1) on failure.
    This is the hard gate run as a separate CI step after the push: it keeps
    re-checking the real published feeds until every one shows the new version at
    the top, or it gives up and fails the build. Without it, a publish that
    silently missed a feed would look green while users never get the update.
    """
    numeric, _ = parse_version(args.tag)
    deadline = time.time() + args.timeout
    # Start with all three feeds "pending"; each is removed once it goes live.
    pending = {p["appcast"] for p in PLATFORMS}
    print(f"Verifying all three live feeds advertise {numeric} (up to {args.timeout}s)...")
    last = {}  # remembers each feed's last-seen top version for the failure report
    while time.time() < deadline and pending:
        for appcast in list(pending):
            try:
                top = top_item_version(fetch_live(appcast))
            except Exception as e:                       # noqa: BLE001 - transient net/CDN
                # A network/CDN hiccup is expected while the push propagates;
                # record it and keep retrying until the deadline.
                last[appcast] = f"fetch error: {e}"
                continue
            last[appcast] = top
            if top == numeric:
                print(f"  OK  {appcast} -> {top}")
                pending.discard(appcast)
        # Wait before the next round only if some feeds still haven't caught up.
        if pending:
            time.sleep(args.interval)
    # Any feed still pending after the deadline means propagation never finished.
    if pending:
        print("\nFAILED: these feeds did not advertise the new version in time:")
        for appcast in sorted(pending):
            print(f"  {appcast}: top is {last.get(appcast)!r}, expected {numeric!r}")
        sys.exit(1)
    print(f"\nAll three feeds are live at {numeric}.")


def main():
    """Command-line entry point: parse arguments and run publish or verify.

    Reads the command line, then dispatches to do_verify when --verify is given,
    or to do_publish otherwise. In publish mode it first checks that the three
    required options (--artifacts-dir, --key-file, --gh-pages-dir) are all
    present and exits with a clear message if any are missing.
    """
    parser = argparse.ArgumentParser(description="Publish/verify Star Trail CleanR appcasts.")
    parser.add_argument("tag", help="Release tag, e.g. v2.47-beta")
    parser.add_argument("--verify", action="store_true",
                        help="Verify-only: poll live feeds, exit non-zero unless all advertise <tag>.")
    parser.add_argument("--publish-mirror", action="store_true",
                        help="Publish single-item feeds to our own server (the feeds the app "
                             "reads from v2.80 on); needs DREAMHOST_PASSWORD in the environment.")
    parser.add_argument("--artifacts-dir", help="Folder containing the built installers (searched recursively).")
    parser.add_argument("--key-file", help="Path to the Sparkle ed25519 private key file.")
    parser.add_argument("--gh-pages-dir", help="Checked-out gh-pages working copy to update in place.")
    parser.add_argument("--timeout", type=int, default=900, help="Verify: max seconds to wait for propagation.")
    parser.add_argument("--interval", type=int, default=20, help="Verify: seconds between polls.")
    args = parser.parse_args()

    if args.verify:
        do_verify(args)
    elif args.publish_mirror:
        do_publish_mirror(args)
    else:
        missing = [n for n in ("artifacts_dir", "key_file", "gh_pages_dir")
                   if getattr(args, n) is None]
        if missing:
            sys.exit("publish mode needs: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        do_publish(args)


if __name__ == "__main__":
    main()
