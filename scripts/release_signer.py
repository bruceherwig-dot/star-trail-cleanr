"""Sign a Star Trail CleanR release, generate delta updates, and publish appcasts.

Usage:
  python scripts/release_signer.py vX.XX-beta              # full release flow
  python scripts/release_signer.py vX.XX-beta --skip-push  # local dry run

Flow:
  1. Download the three signed artifacts (Apple Silicon DMG, Intel DMG, Windows
     zip) from the GitHub release.
  2. Park each artifact in ~/.star_trail_cleanr/release_archive/<platform>/
     with a version-suffixed filename so generate_appcast can compute deltas
     against past versions.
  3. For each platform, sync the latest appcast XML from gh-pages into the
     archive folder (so generate_appcast updates it in place).
  4. Run vendored generate_appcast on each platform folder. It signs the new
     archive, computes binary deltas against past versions, and rewrites the
     appcast XML with delta enclosure entries.
  5. Post-process the XML: rewrite URLs (placeholder prefix → per-version
     GitHub release URL) and restore "-beta" suffix on shortVersionString.
  6. Upload the delta files (*.delta) to the GitHub release as additional
     assets so the URLs in the appcast are valid.
  7. Push the rewritten appcast XMLs to gh-pages.

Why all this: every release before v2.05-beta forced users to redownload the
full ~600 MB bundle. With deltas, each user only fetches the bytes that
actually changed since their version (typically <50 MB).

Linux tar.gz isn't signed because there's no Sparkle/WinSparkle equivalent on
Linux yet (todo #57: Linux keeps the amber banner forever).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIGN_UPDATE = REPO_ROOT / "vendored" / "sparkle-bin" / "sign_update"
GENERATE_APPCAST = REPO_ROOT / "vendored" / "sparkle-bin" / "generate_appcast"
KEY_FILE = Path.home() / ".star_trail_cleanr" / "sparkle_ed_private.key"
ARCHIVE_ROOT = Path.home() / ".star_trail_cleanr" / "release_archive"
GH_REPO = "bruceherwig-dot/star-trail-cleanr"
RELEASE_BASE = f"https://github.com/{GH_REPO}/releases/download"
URL_PLACEHOLDER = f"{RELEASE_BASE}/PLACEHOLDER"

PLATFORMS = [
    {
        "key": "mac-apple-silicon",
        "release_filename": "StarTrailCleanR-Mac-AppleSilicon.dmg",
        "archive_stem": "StarTrailCleanR-Mac-AppleSilicon",
        "archive_ext": ".dmg",
        "appcast": "appcast-mac-apple-silicon.xml",
        "uses_deltas": True,
    },
    {
        "key": "mac-intel",
        "release_filename": "StarTrailCleanR-Mac-Intel.dmg",
        "archive_stem": "StarTrailCleanR-Mac-Intel",
        "archive_ext": ".dmg",
        "appcast": "appcast-mac-intel.xml",
        "uses_deltas": True,
    },
    {
        # Windows still ships the full bundle every release. Sparkle's
        # generate_appcast only computes deltas for Mac .app bundles, and
        # WinSparkle has no delta-update support. We fall back to the
        # manual "insert item at top of channel" approach for Windows.
        "key": "windows",
        "release_filename": "StarTrailCleanRSetup.zip",
        "archive_stem": "StarTrailCleanRSetup",
        "archive_ext": ".zip",
        "appcast": "appcast-windows.xml",
        "uses_deltas": False,
    },
]


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result


def parse_version(tag):
    """v2.05-beta -> ('2.05', '2.05-beta')"""
    m = re.match(r"v(\d+(?:\.\d+)?)", tag)
    if not m:
        sys.exit(f"Could not parse version from tag '{tag}'. Expected like v2.05-beta.")
    return m.group(1), tag.lstrip("v")


def download_artifacts(tag, dest):
    dest.mkdir(parents=True, exist_ok=True)
    have_all = all((dest / p["release_filename"]).exists() for p in PLATFORMS)
    if have_all:
        print(f"All artifacts already in {dest}, skipping download.")
        return
    print(f"Downloading {tag} artifacts to {dest}...")
    run([
        "gh", "release", "download", tag,
        "--clobber", "--dir", str(dest),
        "--repo", GH_REPO,
    ])


def park_artifact(platform, tag, src_artifact_dir, archive_dir):
    """Copy the new release's artifact into the platform's archive folder
    with a version-suffixed filename. generate_appcast disambiguates archives
    by filename, so each version needs a unique name in the same folder."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    src = src_artifact_dir / platform["release_filename"]
    dst_name = f"{platform['archive_stem']}-{tag}{platform['archive_ext']}"
    dst = archive_dir / dst_name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f"  archive already has {dst_name} (same size) — keeping")
        return dst
    shutil.copy2(src, dst)
    return dst


def sync_appcast_from_gh_pages(platform, archive_dir):
    """Pull the latest appcast XML from gh-pages into the archive folder.
    generate_appcast updates the file in place rather than starting fresh,
    which preserves channel decoration (title, link, description, language)."""
    url = f"https://bruceherwig-dot.github.io/star-trail-cleanr/{platform['appcast']}"
    target = archive_dir / platform["appcast"]
    print(f"  syncing appcast from {url}")
    result = subprocess.run(
        ["curl", "-fsSL", "-o", str(target), url],
        capture_output=True,
    )
    if result.returncode != 0:
        # If the appcast doesn't exist on gh-pages yet (first ever run),
        # let generate_appcast create one from scratch.
        print(f"  no remote appcast yet, generate_appcast will create one")
        return False
    return True


def list_existing_deltas(archive_dir):
    return set(p.name for p in archive_dir.glob("*.delta"))


def rename_deltas_with_platform_prefix(archive_dir, platform, new_delta_names):
    """generate_appcast names deltas after the bundle's CFBundleName, which is
    "StarTrailCleanR" for both Mac architectures. That collides on the GitHub
    release (same filename, different content). Rename the new deltas to embed
    the platform key so they upload cleanly. Returns a list of (old_name,
    new_path) so the caller can rewrite the appcast XML."""
    pairs = []
    for old_name in new_delta_names:
        new_name = f"{platform['archive_stem']}-{old_name.removeprefix('StarTrailCleanR')}"
        # Prefix doesn't drop "StarTrailCleanR" cleanly when the archive_stem
        # already starts with it, so guard: archive_stem already begins with
        # the brand, removing it from old_name avoids "StarTrailCleanR-Mac-AppleSilicon-StarTrailCleanR..." duplication.
        old = archive_dir / old_name
        new = archive_dir / new_name
        if old != new:
            old.rename(new)
        pairs.append((old_name, new_name, new))
    return pairs


def update_xml_delta_filenames(xml_text, renames):
    """Patch the appcast XML so each delta enclosure URL uses the new filename.
    `renames` is the list of (old_name, new_name, _path) tuples returned by
    rename_deltas_with_platform_prefix."""
    for old_name, new_name, _ in renames:
        # generate_appcast inlines the filename right after the URL prefix,
        # so a plain string replace is sufficient and unambiguous.
        xml_text = xml_text.replace(old_name, new_name)
    return xml_text


def run_generate_appcast(archive_dir):
    run([
        str(GENERATE_APPCAST),
        "--ed-key-file", str(KEY_FILE),
        "--download-url-prefix", URL_PLACEHOLDER + "/",
        "--maximum-versions", "5",
        "--maximum-deltas", "5",
        str(archive_dir),
    ])


def post_process_appcast_xml(xml_text, tag, short):
    """Rewrite generate_appcast's XML output for our publishing layout.

    1. URL fix-up: generate_appcast emits URLs like
         <prefix>/PLACEHOLDER/StarTrailCleanR-Mac-AppleSilicon-v2.05-beta.dmg
       but the actual GitHub release URL is
         <prefix>/v2.05-beta/StarTrailCleanR-Mac-AppleSilicon.dmg
       So we strip the version suffix from full-bundle filenames and replace
       PLACEHOLDER with the version-specific release path. Delta filenames
       stay as-is — they're uploaded under those names to the new release.

    2. Beta-suffix restore: build_helper.py sets CFBundleShortVersionString
       to the bare numeric (e.g. "2.05") because Sparkle's version comparison
       is happier with numeric strings. generate_appcast reads that into
       sparkle:shortVersionString. We rewrite the new release's entry so the
       displayed version reads "2.05-beta" instead of "2.05" — the brand
       wants the -beta tag visible to the user."""
    # 1a. Strip version suffix from full-bundle filenames.
    # Pattern: PLACEHOLDER/<stem>-vX.YY[-suffix].<ext> -> PLACEHOLDER/<stem>.<ext>
    def strip_full_bundle_suffix(m):
        prefix, stem, version_part, ext = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{prefix}/{stem}{ext}"

    pattern = re.compile(
        r"(" + re.escape(URL_PLACEHOLDER) + r")/([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*?)(-v\d+(?:\.\d+)?(?:-[A-Za-z0-9]+)?)(\.dmg|\.zip)"
    )
    xml_text = pattern.sub(strip_full_bundle_suffix, xml_text)

    # 1b. Replace remaining PLACEHOLDER with the new release tag's folder.
    xml_text = xml_text.replace(URL_PLACEHOLDER, f"{RELEASE_BASE}/{tag}")

    # 2. Restore -beta suffix on the new release's shortVersionString.
    # Matches sparkle:shortVersionString containing the bare numeric for THIS
    # release only — leave older entries alone.
    numeric, _ = parse_version(tag)
    bare = f"<sparkle:shortVersionString>{numeric}</sparkle:shortVersionString>"
    branded = f"<sparkle:shortVersionString>{short}</sparkle:shortVersionString>"
    xml_text = xml_text.replace(bare, branded, 1)

    return xml_text


def sign_one(path):
    """Sign one file with sign_update -p, return (signature, length)."""
    result = subprocess.run(
        [str(SIGN_UPDATE), "-p", "-f", str(KEY_FILE), str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"sign_update failed for {path}:\n{result.stderr}")
    return result.stdout.strip(), path.stat().st_size


def build_windows_item_xml(tag, numeric, short, sig, length, pub_date):
    url = f"{RELEASE_BASE}/{tag}/StarTrailCleanRSetup.zip"
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


def insert_windows_item(xml_text, new_item, short):
    if f"<title>Version {short}</title>" in xml_text:
        sys.exit(f"Appcast already has Version {short} — refusing to duplicate.")
    marker = "<language>en</language>\n"
    idx = xml_text.find(marker)
    if idx == -1:
        sys.exit("Could not find <language>en</language> insertion point.")
    insert_at = idx + len(marker)
    return xml_text[:insert_at] + new_item + xml_text[insert_at:]


def fetch_gh_pages_xml(appcast_name):
    """Pull the current gh-pages copy of an appcast XML as a string."""
    url = f"https://bruceherwig-dot.github.io/star-trail-cleanr/{appcast_name}"
    result = subprocess.run(
        ["curl", "-fsSL", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Could not fetch {url}: {result.stderr}")
    return result.stdout


def upload_delta_files_to_release(tag, delta_files):
    """Upload generated *.delta files as additional assets on the GitHub
    release for this version. The appcast URLs we wrote in post_process
    point here."""
    if not delta_files:
        print("  (no delta files to upload — first release with deltas)")
        return
    print(f"\nUploading {len(delta_files)} delta file(s) to release {tag}...")
    cmd = ["gh", "release", "upload", tag, "--clobber", "--repo", GH_REPO]
    cmd.extend(str(d) for d in delta_files)
    run(cmd)


def update_gh_pages(tag, appcast_xmls):
    work = Path("/tmp") / f"gh-pages-{tag}"
    if work.exists():
        run(["git", "worktree", "remove", "--force", str(work)])
    run(["git", "fetch", "origin", "gh-pages"])
    run(["git", "worktree", "add", str(work), "gh-pages"])
    try:
        for appcast_name, xml_text in appcast_xmls.items():
            (work / appcast_name).write_text(xml_text)
            print(f"  + Updated {appcast_name}")
        run(["git", "-C", str(work), "add", "-A"])
        run([
            "git", "-C", str(work), "commit", "-m",
            f"Publish {tag} with delta updates",
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
    parser.add_argument("tag", help="Release tag, e.g. v2.05-beta")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Local folder of already-downloaded artifacts.")
    parser.add_argument("--skip-push", action="store_true",
                        help="Generate everything locally but don't upload deltas or push gh-pages.")
    args = parser.parse_args()

    for required in (SIGN_UPDATE, GENERATE_APPCAST):
        if not required.exists():
            sys.exit(f"Missing tool: {required}")
    if not KEY_FILE.exists():
        sys.exit(f"Sparkle private key not found at {KEY_FILE}")

    numeric, short = parse_version(args.tag)

    # 1. Get the new release's artifacts.
    if args.artifacts_dir:
        artifacts = Path(args.artifacts_dir)
    else:
        artifacts = Path("/tmp") / f"{args.tag}-release"
        download_artifacts(args.tag, artifacts)

    appcast_xmls = {}     # appcast_name -> XML text ready to publish
    all_new_deltas = []   # list of Paths to delta files we should upload
    pub_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    for platform in PLATFORMS:
        print(f"\n=== {platform['key']} ===")
        archive_dir = ARCHIVE_ROOT / platform["key"]

        # 2. Park the new artifact alongside the old ones (Mac uses these
        #    for delta computation; Windows just keeps a record).
        park_artifact(platform, args.tag, artifacts, archive_dir)

        if platform["uses_deltas"]:
            # Mac: generate_appcast handles signing, deltas, and XML rewrite.
            sync_appcast_from_gh_pages(platform, archive_dir)
            deltas_before = list_existing_deltas(archive_dir)
            run_generate_appcast(archive_dir)
            deltas_after = list_existing_deltas(archive_dir)
            new_delta_names = sorted(deltas_after - deltas_before)
            renames = rename_deltas_with_platform_prefix(archive_dir, platform, new_delta_names)
            if renames:
                print(f"  generated {len(renames)} delta file(s):")
                for _, new_name, path in renames:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"    {new_name} ({size_mb:.1f} MB)")
            all_new_deltas.extend(path for _, _, path in renames)
            xml_path = archive_dir / platform["appcast"]
            raw_xml = xml_path.read_text()
            renamed_xml = update_xml_delta_filenames(raw_xml, renames)
            cooked_xml = post_process_appcast_xml(renamed_xml, args.tag, short)
        else:
            # Windows: sign manually, splice item into the existing appcast.
            artifact_path = artifacts / platform["release_filename"]
            sig, length = sign_one(artifact_path)
            size_mb = length / (1024 * 1024)
            print(f"  signed {platform['release_filename']} ({size_mb:.1f} MB)")
            current_xml = fetch_gh_pages_xml(platform["appcast"])
            new_item = build_windows_item_xml(args.tag, numeric, short, sig, length, pub_date)
            cooked_xml = insert_windows_item(current_xml, new_item, short)

        appcast_xmls[platform["appcast"]] = cooked_xml

    if args.skip_push:
        print("\n--skip-push set: skipping delta upload and gh-pages push.")
        print("Inspect the generated XMLs and delta files in:")
        for platform in PLATFORMS:
            print(f"  {ARCHIVE_ROOT / platform['key']}/")
        return

    # 6. Upload delta files to the new GitHub release.
    upload_delta_files_to_release(args.tag, all_new_deltas)

    # 7. Push the rewritten appcasts to gh-pages.
    print("\nUpdating gh-pages...")
    update_gh_pages(args.tag, appcast_xmls)

    wait_for_pages(args.tag, short)
    print(f"\nDone. {args.tag} is live with deltas across all three appcasts.")


if __name__ == "__main__":
    main()
