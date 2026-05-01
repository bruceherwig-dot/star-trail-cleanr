"""Sign a Star Trail CleanR release and publish appcast entries.

Usage: python scripts/release_signer.py vX.XX-beta

Steps:
  1. Download the three signed artifacts (Apple Silicon DMG, Intel DMG, Windows
     zip) from the GitHub release if they aren't already in /tmp.
  2. Sign each with vendored/sparkle-bin/sign_update. Sparkle reads the EdDSA
     private key from the macOS Keychain (you'll see Allow prompts).
  3. Edit the three appcast XMLs on the gh-pages branch via a temporary
     worktree, inserting a new <item> at the top of each <channel>.
  4. Commit and push gh-pages.

Linux tar.gz isn't signed because there's no Sparkle/WinSparkle equivalent on
Linux yet (todo #57: Linux keeps the amber banner).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIGN_UPDATE = REPO_ROOT / "vendored" / "sparkle-bin" / "sign_update"
KEY_FILE = Path.home() / ".star_trail_cleanr" / "sparkle_ed_private.key"
GH_REPO = "bruceherwig-dot/star-trail-cleanr"
RELEASE_BASE = f"https://github.com/{GH_REPO}/releases/download"

PLATFORMS = [
    {
        "filename": "StarTrailCleanR-Mac-AppleSilicon.dmg",
        "appcast": "appcast-mac-apple-silicon.xml",
    },
    {
        "filename": "StarTrailCleanR-Mac-Intel.dmg",
        "appcast": "appcast-mac-intel.xml",
    },
    {
        "filename": "StarTrailCleanRSetup.zip",
        "appcast": "appcast-windows.xml",
    },
]


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result


def parse_version(tag):
    m = re.match(r"v(\d+(?:\.\d+)?)", tag)
    if not m:
        sys.exit(f"Could not parse version from tag '{tag}'. Expected like v2.03-beta.")
    numeric = m.group(1)
    short = tag.lstrip("v")
    return numeric, short


def download_artifacts(tag, dest):
    dest.mkdir(parents=True, exist_ok=True)
    have_all = all((dest / p["filename"]).exists() for p in PLATFORMS)
    if have_all:
        print(f"All artifacts already in {dest}, skipping download.")
        return
    print(f"Downloading {tag} artifacts to {dest}...")
    run([
        "gh", "release", "download", tag,
        "--clobber", "--dir", str(dest),
        "--repo", GH_REPO,
    ])


def sign_one(path):
    if not path.exists():
        sys.exit(f"Missing artifact: {path}")
    result = subprocess.run(
        [str(SIGN_UPDATE), "-p", "-f", str(KEY_FILE), str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"sign_update failed for {path}:\n{result.stderr}")
    sig = result.stdout.strip()
    length = path.stat().st_size
    return sig, length


def build_item(tag, numeric, short, filename, sig, length, pub_date):
    url = f"{RELEASE_BASE}/{tag}/{filename}"
    return (
        f"    <item>\n"
        f"      <title>Version {short}</title>\n"
        f"      <pubDate>{pub_date}</pubDate>\n"
        f"      <sparkle:version>{numeric}</sparkle:version>\n"
        f"      <sparkle:shortVersionString>{short}</sparkle:shortVersionString>\n"
        f"      <enclosure\n"
        f'        url="{url}"\n'
        f'        sparkle:edSignature="{sig}"\n'
        f'        length="{length}"\n'
        f'        type="application/octet-stream" />\n'
        f"    </item>\n"
    )


def insert_item(xml_text, new_item, short):
    if f"<title>Version {short}</title>" in xml_text:
        sys.exit(f"Appcast already has Version {short} — refusing to duplicate.")
    marker = "<language>en</language>\n"
    idx = xml_text.find(marker)
    if idx == -1:
        sys.exit("Could not find <language>en</language> insertion point.")
    insert_at = idx + len(marker)
    return xml_text[:insert_at] + new_item + xml_text[insert_at:]


def update_gh_pages(tag, signed):
    work = Path("/tmp") / f"gh-pages-{tag}"
    if work.exists():
        run(["git", "worktree", "remove", "--force", str(work)])
    run(["git", "fetch", "origin", "gh-pages"])
    run(["git", "worktree", "add", str(work), "gh-pages"])
    try:
        for entry in signed:
            xml_path = work / entry["appcast"]
            text = xml_path.read_text()
            text = insert_item(text, entry["item_xml"], entry["short"])
            xml_path.write_text(text)
            print(f"  + Updated {entry['appcast']}")
        run(["git", "-C", str(work), "add", "-A"])
        run([
            "git", "-C", str(work), "commit", "-m",
            f"Publish {tag} to all three appcasts",
        ])
        run(["git", "-C", str(work), "push", "origin", "gh-pages"])
    finally:
        run(["git", "worktree", "remove", "--force", str(work)])


def wait_for_pages(tag, short):
    url = f"https://bruceherwig-dot.github.io/star-trail-cleanr/{PLATFORMS[0]['appcast']}"
    needle = f"Version {short}"
    print(f"Waiting for {url} to advertise {needle}...")
    deadline = time.time() + 180
    while time.time() < deadline:
        result = subprocess.run(
            ["curl", "-fsSL", url],
            capture_output=True, text=True,
        )
        if needle in result.stdout:
            print("Appcast is live.")
            return True
        time.sleep(5)
    print("Timed out waiting for GitHub Pages to propagate (try the URL manually).")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Release tag, e.g. v2.03-beta")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Local folder of already-downloaded artifacts (skips download).")
    parser.add_argument("--skip-push", action="store_true",
                        help="Edit gh-pages locally but don't push (dry run).")
    args = parser.parse_args()

    if not SIGN_UPDATE.exists():
        sys.exit(f"sign_update not found at {SIGN_UPDATE}")
    if not KEY_FILE.exists():
        sys.exit(
            f"Sparkle private key not found at {KEY_FILE}.\n"
            f"Export it from Keychain with: "
            f"/tmp/sparkle_v2_prep/bin/generate_keys -x {KEY_FILE}"
        )

    numeric, short = parse_version(args.tag)
    pub_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    if args.artifacts_dir:
        artifacts = Path(args.artifacts_dir)
    else:
        artifacts = Path("/tmp") / f"{args.tag}-release"
        download_artifacts(args.tag, artifacts)

    print("\nSigning artifacts...")
    signed = []
    for p in PLATFORMS:
        path = artifacts / p["filename"]
        sig, length = sign_one(path)
        item_xml = build_item(args.tag, numeric, short, p["filename"], sig, length, pub_date)
        signed.append({"appcast": p["appcast"], "item_xml": item_xml, "short": short})
        size_mb = length / (1024 * 1024)
        print(f"  {p['filename']}: signed ({size_mb:.1f} MB)")

    print("\nUpdating gh-pages...")
    if args.skip_push:
        print("--skip-push set: would write three XMLs and push. Stopping.")
        return
    update_gh_pages(args.tag, signed)

    wait_for_pages(args.tag, short)
    print(f"\nDone. {args.tag} is live on all three appcasts.")


if __name__ == "__main__":
    main()
