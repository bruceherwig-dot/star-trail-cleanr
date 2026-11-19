#!/usr/bin/env python3
"""Signature gate for the Mac app: fail the build if the app has no valid seal.

WHY THIS EXISTS. v3.04-beta (2026-11-12) shipped an Apple Silicon app with NO
signature seal. The signing step in build_helper.py failed on a leftover Qt
build file, printed a warning, and the build carried on green. Anyone who
downloaded that build from the website was told by macOS:

    "StarTrailCleanR is damaged and can't be opened. You should move it to
    the Trash."

with no way round it (two users wrote in within two days). Nothing in the
build or the tests looked at the signature, so nothing could see it.

WHAT IT CHECKS, on the app inside the finished disk image (the thing a user
actually downloads), or on a .app folder directly:

  1. Contents/_CodeSignature/CodeResources exists (the seal itself).
  2. `codesign -dvv` reports sealed resources and a bound Info.plist.
  3. `codesign --verify` passes, with ONE tolerated complaint: the vendored
     Sparkle.framework carries a LICENSE.txt in its root, which codesign
     reports as "unsealed contents present in the root directory of an
     embedded framework". Every release up to v3.03 shipped in that state and
     opened normally, so it is allowed here. Any other complaint fails.

Usage:
    python scripts/check_mac_signature.py dist/StarTrailCleanR.app
    python scripts/check_mac_signature.py StarTrailCleanR-Mac-AppleSilicon.dmg

Exit code 0 = sealed and acceptable. 1 = not acceptable (reason printed).
If this gate fails: fix the signing. Do not skip or weaken the gate.
"""
import os
import subprocess
import sys
import tempfile

APP_NAME = "StarTrailCleanR.app"

# The one complaint we tolerate (see the docstring). Both pieces must be
# present for the tolerance to apply.
_KNOWN_COMPLAINT = "unsealed contents present in the root directory of an embedded framework"
_KNOWN_SUBCOMPONENT = "Sparkle.framework"


def verify_is_acceptable(returncode, output):
    """Decide whether a `codesign --verify --verbose=2` result is acceptable.

    Returns (ok, reason). Pure function so the smoke tests can exercise it
    without a Mac or a built app.
    """
    if returncode == 0:
        return True, "signature verifies"
    # Drop codesign's per-file progress lines; what is left is the complaint.
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()
             and not ln.startswith("--prepared:")
             and not ln.startswith("--validated:")]
    complaint = [ln for ln in lines if not ln.startswith("In subcomponent:")]
    subcomponents = [ln for ln in lines if ln.startswith("In subcomponent:")]
    only_known = (
        len(complaint) == 1
        and _KNOWN_COMPLAINT in complaint[0]
        and len(subcomponents) == 1
        and subcomponents[0].rstrip("/").endswith(_KNOWN_SUBCOMPONENT)
    )
    if only_known:
        return True, "only the known Sparkle.framework LICENSE.txt complaint"
    return False, "codesign --verify failed: " + " | ".join(lines[-4:] or ["(no output)"])


def describe_is_sealed(output):
    """Decide whether `codesign -dvv` output shows a sealed bundle.

    The broken v3.04 app printed "Sealed Resources=none" and
    "Info.plist=not bound". A sealed app prints "Sealed Resources version=..."
    and "Info.plist entries=...". Returns (ok, reason).
    """
    text = output or ""
    if "Sealed Resources version=" not in text:
        return False, "the app's resources are not sealed"
    if "Info.plist entries=" not in text:
        return False, "the app's Info.plist is not bound to the signature"
    return True, "sealed"


def check_app(app_path):
    """Run all three checks on one .app folder. Returns (ok, reason)."""
    if not os.path.isdir(app_path):
        return False, f"app not found: {app_path}"
    seal = os.path.join(app_path, "Contents", "_CodeSignature", "CodeResources")
    if not os.path.isfile(seal):
        return False, "no signature seal (Contents/_CodeSignature/CodeResources is missing)"
    d = subprocess.run(["codesign", "-dvv", app_path], capture_output=True, text=True)
    ok, reason = describe_is_sealed(d.stdout + d.stderr)
    if not ok:
        return False, reason
    v = subprocess.run(["codesign", "--verify", "--verbose=2", app_path],
                       capture_output=True, text=True)
    return verify_is_acceptable(v.returncode, v.stdout + v.stderr)


def check_dmg(dmg_path):
    """Mount a disk image read-only (no Finder window), check the app inside,
    and unmount. Returns (ok, reason)."""
    if not os.path.isfile(dmg_path):
        return False, f"disk image not found: {dmg_path}"
    mount = tempfile.mkdtemp(prefix="stc_sigcheck_")
    att = subprocess.run(
        ["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", mount, dmg_path],
        capture_output=True, text=True)
    if att.returncode != 0:
        return False, f"could not open the disk image: {att.stderr.strip()}"
    try:
        return check_app(os.path.join(mount, APP_NAME))
    finally:
        subprocess.run(["hdiutil", "detach", mount], capture_output=True)


# ── Sealing (used by build_helper.py) ────────────────────────────────────────
# Leftovers that third-party packages ship inside their wheels and PyInstaller
# copies into the app: debug-symbol bundles and compiler object files. They are
# never loaded at runtime, they are not signed, and `codesign` refuses to seal
# an app that contains them ("code object is not signed at all"). Both kinds
# were in the v3.04 app:
#   PySide6/Qt/qml/Qt/labs/assetdownloader/objects-RelWithDebInfo/...cpp.o
#   objc/_objc.cpython-311-darwin.so.dSYM/...
# PyInstaller renames "." to "__dot__" in some folder names, so match both.
_MAX_SEAL_ATTEMPTS = 200


def is_build_leftover(name):
    """True for a file or folder NAME that is a debug/build leftover."""
    return (name.endswith(".dSYM") or name.endswith("__dot__dSYM")
            or name.startswith("objects-") or name.endswith(".o"))


def _inside_leftover(path, app_path):
    """True when any part of `path` below the app is a build leftover."""
    rel = os.path.relpath(path, app_path)
    return any(is_build_leftover(part) for part in rel.split(os.sep))


def remove_build_leftovers(app_path):
    """Delete debug-symbol bundles and compiler object files from the app.
    Returns the list of removed paths (relative to the app)."""
    import shutil
    removed = []
    for root, dirs, files in os.walk(app_path, topdown=True, followlinks=False):
        for d in list(dirs):
            if is_build_leftover(d):
                full = os.path.join(root, d)
                if os.path.islink(full):
                    os.unlink(full)
                else:
                    shutil.rmtree(full, ignore_errors=True)
                dirs.remove(d)
                removed.append(os.path.relpath(full, app_path))
        for f in files:
            if is_build_leftover(f):
                full = os.path.join(root, f)
                try:
                    os.unlink(full)
                    removed.append(os.path.relpath(full, app_path))
                except OSError:
                    pass
    return removed


def _unsigned_subcomponent(codesign_output):
    """Pull the path out of codesign's 'In subcomponent: <path>' line, or None."""
    for ln in (codesign_output or "").splitlines():
        if ln.startswith("In subcomponent:"):
            return ln.split(":", 1)[1].strip()
    return None


def seal_app(app_path):
    """Ad-hoc sign the app bundle so it carries a valid seal. Returns (ok, reason).

    1. Remove known build leftovers.
    2. Sign the bundle. If codesign stops on a nested file that is "not signed
       at all", deal with THAT file (delete it if it is a leftover, otherwise
       ad-hoc sign it) and try again, so one stray file in some future package
       version cannot silently leave the whole app unsealed again.
    3. Run the same checks the gate runs.
    """
    removed = remove_build_leftovers(app_path)
    if removed:
        # There can be hundreds (one debug bundle per library); show a few.
        print(f"  removed {len(removed)} build leftovers (debug symbols, object files), e.g.:")
        for rel in removed[:5]:
            print(f"    {rel}")
    handled = set()
    for _ in range(_MAX_SEAL_ATTEMPTS):
        cs = subprocess.run(["codesign", "--force", "--sign", "-", app_path],
                            capture_output=True, text=True)
        if cs.returncode == 0:
            break
        out = cs.stdout + cs.stderr
        sub = _unsigned_subcomponent(out)
        if "not signed at all" not in out or not sub or sub in handled \
                or not os.path.exists(sub):
            return False, "codesign failed: " + " | ".join(out.strip().splitlines()[-3:])
        handled.add(sub)
        rel = os.path.relpath(sub, app_path)
        if _inside_leftover(sub, app_path):
            os.unlink(sub)
            print(f"  removed build leftover: {rel}")
        else:
            one = subprocess.run(["codesign", "--force", "--sign", "-", sub],
                                 capture_output=True, text=True)
            if one.returncode != 0:
                return False, f"could not sign {rel}: {(one.stdout + one.stderr).strip()}"
            print(f"  signed nested file: {rel}")
    else:
        return False, f"still unsigned files after {_MAX_SEAL_ATTEMPTS} attempts"
    return check_app(app_path)


def main(argv):
    if sys.platform != "darwin":
        print("check_mac_signature: not macOS, nothing to check")
        return 0
    if len(argv) == 3 and argv[1] == "--seal":
        ok, reason = seal_app(argv[2])
        if ok:
            print(f"App sealed: {reason}")
            return 0
        print(f"SEALING FAILED for {argv[2]}: {reason}")
        print("A Mac app without a valid seal is reported as 'damaged' to everyone "
              "who downloads it. Fix the cause; do not ship this build.")
        return 1
    if len(argv) != 2:
        print(__doc__)
        return 1
    target = argv[1]
    ok, reason = check_dmg(target) if target.lower().endswith(".dmg") else check_app(target)
    if ok:
        print(f"Signature gate PASSED for {target}: {reason}")
        return 0
    print(f"Signature gate FAILED for {target}: {reason}")
    print("A Mac app without a valid seal is reported as 'damaged' to everyone who "
          "downloads it. Fix the signing step; do not skip this gate.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
