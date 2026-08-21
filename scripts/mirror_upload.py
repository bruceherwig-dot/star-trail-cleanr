"""Upload the built installers to the DreamHost failsafe mirror.

Layer 2 of the update failsafe (see project_update_failsafe_design / AUTO_UPDATE):
users who cannot reach github.com (GitHub blocked at the country level) need a
non-GitHub place to download the app. website/latest.php serves the mirror URL for
any installer that is present on our server, else the GitHub URL -- so this upload
is what "lights up" the mirror for each release.

Runs in CI (the mirror-installers job in build.yml) after the four build jobs.
Also runnable locally against a folder of installers.

Usage:  python3 scripts/mirror_upload.py <artifacts_dir>
    <artifacts_dir> is searched recursively for each expected installer filename.

Credentials: the SFTP password comes from the DREAMHOST_PASSWORD environment
variable (a GitHub Actions secret in CI). The host/user are not secret. Never
prints the password.

CRITICAL PATH since v2.80: the app's update engine reads its feed from
api.startrailcleanr.com, and that feed's download links point at these mirror
copies. A missing password or missing installer now FAILS the build (exit 1) --
a release whose mirror didn't refresh would advertise an update its own
download links can't serve. (Before v2.80 this was a best-effort no-op layer.)
"""
import glob
import os
import sys

HOST = "pdx1-shared-a4-09.dreamhost.com"
PORT = 22
USER = "dh_bmigjp"
DL_DIR = "/home/dh_bmigjp/api.startrailcleanr.com/downloads"
CACHE = "/home/dh_bmigjp/stc_data/latest_cache.json"

# Stable, version-less installer filenames -- must match website/latest.php's
# platform_files() and the app's asset constants in modules/update_check.py.
EXPECTED = [
    "StarTrailCleanR-Mac-AppleSilicon.dmg",
    "StarTrailCleanR-Mac-Intel.dmg",
    "StarTrailCleanRSetup.zip",
    # The bare installer as well as the zip. The Windows update feed points at
    # THIS file (see scripts/publish_appcast.py): WinSparkle executes whatever
    # it downloads, and a zip installs nothing. The feed reads from this mirror,
    # so if the .exe is not uploaded here the update is a 404 instead.
    "StarTrailCleanRSetup.exe",
    "StarTrailCleanR-Linux-x86_64.tar.gz",
]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: mirror_upload.py <artifacts_dir>")
        return 1
    artifacts_dir = sys.argv[1]

    password = os.environ.get("DREAMHOST_PASSWORD", "").strip()
    if not password:
        print("DREAMHOST_PASSWORD not set -- FAILING: the app's update feed "
              "points at this mirror, so a release without it is broken.")
        return 1

    # Locate each expected installer anywhere under the artifacts dir (the CI
    # download-artifact step nests each in its own subfolder).
    found = {}
    missing = []
    for name in EXPECTED:
        hits = glob.glob(os.path.join(artifacts_dir, "**", name), recursive=True)
        if hits:
            found[name] = hits[0]
        else:
            missing.append(name)
    if missing:
        print("FAILING: these installers were not found under "
              f"{artifacts_dir}: {', '.join(missing)} -- the update feed "
              "points at the mirror, so every installer must be present.")
        return 1

    import paramiko
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=password)
    sftp = paramiko.SFTPClient.from_transport(t)

    # Make sure the mirror folder exists.
    try:
        sftp.stat(DL_DIR)
    except IOError:
        sftp.mkdir(DL_DIR)
        sftp.chmod(DL_DIR, 0o755)

    uploaded = 0
    for name, local_path in found.items():
        remote = f"{DL_DIR}/{name}"
        sftp.put(local_path, remote)
        sftp.chmod(remote, 0o644)
        size = sftp.stat(remote).st_size
        print(f"uploaded {name} -> {remote} ({size} bytes)")
        uploaded += 1

    # Clear the endpoint's cache so latest.php recomputes with the new files now.
    try:
        sftp.remove(CACHE)
        print("cleared latest.php cache")
    except IOError:
        pass

    sftp.close()
    t.close()
    print(f"mirror upload complete: {uploaded}/{len(EXPECTED)} installers "
          f"({len(EXPECTED) - len(found)} missing this release)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
