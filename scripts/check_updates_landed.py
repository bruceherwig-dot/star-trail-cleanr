"""Has anyone actually UPDATED, or are they all fresh downloads?

One command that answers the question we could not answer for four months:
are people's installed copies moving up on their own, or is every new version
number just somebody downloading the app again?

    python3 scripts/check_updates_landed.py            # all platforms
    python3 scripts/check_updates_landed.py windows    # one platform

WHY IT EXISTS
The Windows one-click update never installed anything from the day it shipped
until 2026-08-21: the update feed advertised a .zip, and Windows opens an archive
rather than installing it. Nothing raised an error, so no crash report ever
showed it, and in the usage data a broken updater and a user re-downloading the
app look exactly the same. The only way to tell them apart was to follow each
install's anonymous id across versions -- which is what this does.

WHAT IT READS
The anonymous usage log on our own server (reports.jsonl). Every report carries a
random per-install id, the app version, and the platform. Nothing personal is in
it, and nothing is written here: this is read-only.

HOW TO READ THE OUTPUT
  UPDATED IN PLACE   the same install id was seen on an older version and later
                     on a newer one. The app on that machine moved up. This is
                     the signal. Note it does NOT prove they clicked update:
                     re-downloading the installer looks the same from here.
  fresh install      an id whose first ever report is the current version.
  updated_via        only present from v2.86 onward, on the FIRST run after a
                     version change. "in_app" means they clicked update in the
                     app; "manual" means they installed it themselves. THIS is
                     the airtight answer, and the earliest transition it can
                     describe is 2.86 -> 2.87, because the marker is written by
                     the version being left behind.

Needs paramiko and the DreamHost password at ~/.star_trail_cleanr/dreamhost_credentials.
"""
import os
import sys

HOST = "pdx1-shared-a4-09.dreamhost.com"
USER = "dh_bmigjp"
LOG = "/home/dh_bmigjp/stc_data/reports.jsonl"

# Runs ON the server so the log never leaves it.
REMOTE = r'''
python3 - <<'EOF'
import json, collections
recs = []
for line in open("%s"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    r = d.get("report", {})
    if r.get("dev"):
        continue
    recs.append((d.get("received", ""), r.get("install_id"), str(r.get("app_version")),
                 str(r.get("platform")), r.get("previous_version"), r.get("updated_via")))
recs.sort()

hist = collections.defaultdict(list)
plat, prov = {}, []
for ts, uid, ver, p, prev, via in recs:
    if not uid:
        continue
    plat[uid] = p
    if prev:
        prov.append((ts, p, prev, ver, via))
    if not hist[uid] or hist[uid][-1][1] != ver:
        hist[uid].append((ts, ver))

def key(v):
    try:
        return [int(x) for x in v.split(".")]
    except Exception:
        return [0]

want = %r
print("reports: %%d   installs: %%d   latest: %%s" %% (
    len(recs), len(hist), recs[-1][0][:16] if recs else "-"))

for platform in sorted(set(plat.values())):
    if want and platform != want:
        continue
    ids = [u for u, p in plat.items() if p == platform]
    moved = [u for u in ids if len(hist[u]) > 1 and key(hist[u][-1][1]) > key(hist[u][0][1])]
    print("\n%%s: %%d installs, %%d UPDATED IN PLACE" %% (platform, len(ids), len(moved)))
    for u in moved:
        print("   " + " -> ".join("%%s @ %%s" %% (v, t[:10]) for t, v in hist[u]))
    current = collections.Counter(hist[u][-1][1] for u in ids)
    print("   running now: " + ", ".join(
        "%%s x%%d" %% (v, n) for v, n in sorted(current.items(), key=lambda kv: key(kv[0]), reverse=True)))

print("\nreports stating HOW they got there (v2.86+ only): %%d" %% len(prov))
for ts, p, prev, ver, via in prov:
    print("   %%s  %%s  %%s -> %%s  via %%s" %% (ts[:16], p, prev, ver, via))
if not prov:
    print("   none yet. Expected until a machine already on 2.86+ moves to the next")
    print("   version: the marker is written by the version being left behind.")
EOF
'''


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        import paramiko
    except ImportError:
        print("needs paramiko:  pip install paramiko")
        return 1
    cred = os.path.expanduser("~/.star_trail_cleanr/dreamhost_credentials")
    try:
        pw = open(cred).read().strip()
    except OSError:
        print(f"no credentials at {cred}")
        return 1

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, 22, USER, pw, timeout=30)
    _, out, err = ssh.exec_command(REMOTE % (LOG, want))
    text = out.read().decode().strip() or err.read().decode().strip()
    ssh.close()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
