"""Publish the Star Trail CleanR auto-update feeds (appcasts).

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

REPO_ROOT = Path(__file__).resolve().parent.parent
SIGN_UPDATE = REPO_ROOT / "vendored" / "sparkle-bin" / "sign_update"
GH_REPO = "bruceherwig-dot/star-trail-cleanr"
RELEASE_BASE = f"https://github.com/{GH_REPO}/releases/download"
PAGES_BASE = "https://bruceherwig-dot.github.io/star-trail-cleanr"
SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"

# Per-platform release/appcast wiring. `extra` holds the platform-specific
# Sparkle item fields observed in the existing feeds (min OS, CPU requirement)
# so new items match what older items advertise.
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
        "key": "windows",
        "release_filename": "StarTrailCleanRSetup.zip",
        "appcast": "appcast-windows.xml",
        "extra": "",
    },
]


def parse_version(tag):
    """'v2.47-beta' -> ('2.47', '2.47-beta'). Exits on a malformed tag."""
    m = re.match(r"v(\d+(?:\.\d+)?)", tag)
    if not m:
        sys.exit(f"Could not parse version from tag '{tag}'. Expected like v2.47-beta.")
    return m.group(1), tag.lstrip("v")


def find_installer(artifacts_dir, filename):
    """Locate a release installer under artifacts_dir (searched recursively,
    because GitHub's download-artifact nests each artifact in its own folder)."""
    hits = glob.glob(os.path.join(artifacts_dir, "**", filename), recursive=True)
    if not hits:
        sys.exit(f"Installer '{filename}' not found anywhere under {artifacts_dir}")
    return hits[0]


def sign_file(path, key_file):
    """Sign one file, returning (ed_signature, byte_length)."""
    result = subprocess.run(
        [str(SIGN_UPDATE), "-p", "-f", str(key_file), str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"sign_update failed for {path}:\n{result.stderr}")
    return result.stdout.strip(), os.path.getsize(path)


def top_item_version(xml_text):
    """Return the <sparkle:version> of the first <item>, or None."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    item = root.find(".//item")
    if item is None:
        return None
    return item.findtext(f"{{{SPARKLE_NS}}}version")


def build_item_xml(platform, tag, numeric, short, sig, length, pub_date):
    """One <item> block for this release, in a format consistent across feeds."""
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
    """Insert a new <item> right after the channel's <language> line so it
    becomes the newest entry. Falls back to before the first <item> if the
    language tag is absent."""
    m = re.search(r"<language>[^<]*</language>\s*\n", xml_text)
    if m:
        at = m.end()
    else:
        m2 = re.search(r"[ \t]*<item>", xml_text)
        if not m2:
            sys.exit("Appcast has no <language> tag and no <item> to anchor insertion.")
        at = m2.start()
    return xml_text[:at] + item_xml + xml_text[at:]


def fetch_live(appcast, timeout=20):
    """Fetch a live appcast from GitHub Pages, defeating CDN caching."""
    url = f"{PAGES_BASE}/{appcast}?cb={int(time.time())}"
    req = urllib.request.Request(
        url, headers={"Cache-Control": "no-cache", "Pragma": "no-cache",
                      "User-Agent": "StarTrailCleanR-AppcastPublish"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def do_publish(args):
    numeric, short = parse_version(args.tag)
    if not SIGN_UPDATE.exists():
        sys.exit(f"Missing signing tool: {SIGN_UPDATE}")
    if not Path(args.key_file).exists():
        sys.exit(f"Signing key not found: {args.key_file}")
    pub_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    gh = Path(args.gh_pages_dir)

    for platform in PLATFORMS:
        print(f"\n=== {platform['key']} ===")
        xml_path = gh / platform["appcast"]
        if not xml_path.exists():
            sys.exit(f"Appcast not found in gh-pages dir: {xml_path}")
        xml_text = xml_path.read_text()

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


def do_verify(args):
    """Poll all three LIVE feeds until each advertises this version at the top,
    or fail loudly. This is the gate that turns a missed/partial publish into a
    red build instead of silent breakage."""
    numeric, _ = parse_version(args.tag)
    deadline = time.time() + args.timeout
    pending = {p["appcast"] for p in PLATFORMS}
    print(f"Verifying all three live feeds advertise {numeric} (up to {args.timeout}s)...")
    last = {}
    while time.time() < deadline and pending:
        for appcast in list(pending):
            try:
                top = top_item_version(fetch_live(appcast))
            except Exception as e:                       # noqa: BLE001 - transient net/CDN
                last[appcast] = f"fetch error: {e}"
                continue
            last[appcast] = top
            if top == numeric:
                print(f"  OK  {appcast} -> {top}")
                pending.discard(appcast)
        if pending:
            time.sleep(args.interval)
    if pending:
        print("\nFAILED: these feeds did not advertise the new version in time:")
        for appcast in sorted(pending):
            print(f"  {appcast}: top is {last.get(appcast)!r}, expected {numeric!r}")
        sys.exit(1)
    print(f"\nAll three feeds are live at {numeric}.")


def main():
    parser = argparse.ArgumentParser(description="Publish/verify Star Trail CleanR appcasts.")
    parser.add_argument("tag", help="Release tag, e.g. v2.47-beta")
    parser.add_argument("--verify", action="store_true",
                        help="Verify-only: poll live feeds, exit non-zero unless all advertise <tag>.")
    parser.add_argument("--artifacts-dir", help="Folder containing the built installers (searched recursively).")
    parser.add_argument("--key-file", help="Path to the Sparkle ed25519 private key file.")
    parser.add_argument("--gh-pages-dir", help="Checked-out gh-pages working copy to update in place.")
    parser.add_argument("--timeout", type=int, default=900, help="Verify: max seconds to wait for propagation.")
    parser.add_argument("--interval", type=int, default=20, help="Verify: seconds between polls.")
    args = parser.parse_args()

    if args.verify:
        do_verify(args)
    else:
        missing = [n for n in ("artifacts_dir", "key_file", "gh_pages_dir")
                   if getattr(args, n) is None]
        if missing:
            sys.exit("publish mode needs: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        do_publish(args)


if __name__ == "__main__":
    main()
