"""The Mac app must never ship without a signature seal again.

v3.04-beta (2026-11-12): the signing step failed on a leftover Qt build file,
printed a warning, and the Apple Silicon app shipped unsealed. macOS told every
new downloader the app "is damaged and can't be opened". These tests lock the
three things that prevent a repeat:

  * the verdict logic in scripts/check_mac_signature.py,
  * build_helper.py stops the build when sealing fails,
  * both Mac build jobs run the gate on the finished disk image.

They need no Mac and no built app, so they run on every build machine.
"""
import importlib.util
import io
import tokenize
from pathlib import Path

REPO = Path(__file__).parent.parent


def _load_gate():
    path = REPO / "scripts" / "check_mac_signature.py"
    spec = importlib.util.spec_from_file_location("check_mac_signature", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code_only(text):
    """Python source with comments and string literals removed, so a comment
    that NAMES the old mistake cannot satisfy or trip a search."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


# The exact text macOS printed for the broken v3.04 app.
_V304_VERIFY = ("StarTrailCleanR.app: code has no resources but signature "
                "indicates they must be present\n")
_V304_DESCRIBE = ("Identifier=StarTrailCleanR-5555494485ea7d883a0f499b3adb263dcc3a84ef\n"
                  "Signature=adhoc\nInfo.plist=not bound\nSealed Resources=none\n")

# What every release up to v3.03 printed (sealed; one known Sparkle complaint).
_V303_VERIFY = (
    "--prepared:/x/StarTrailCleanR.app/Contents/Frameworks/shapely/libgeos.dylib\n"
    "--validated:/x/StarTrailCleanR.app/Contents/Frameworks/shapely/libgeos.dylib\n"
    "dist/StarTrailCleanR.app: unsealed contents present in the root directory "
    "of an embedded framework\n"
    "In subcomponent: /x/dist/StarTrailCleanR.app/Contents/Frameworks/Sparkle.framework\n")
_V303_DESCRIBE = ("Identifier=StarTrailCleanR\nSignature=adhoc\n"
                  "Info.plist entries=14\nSealed Resources version=2 rules=13 files=28758\n")


def test_broken_v304_app_is_rejected():
    gate = _load_gate()
    ok, _ = gate.verify_is_acceptable(1, _V304_VERIFY)
    assert not ok, "the v3.04 'no resources' signature must fail the gate"
    ok, _ = gate.describe_is_sealed(_V304_DESCRIBE)
    assert not ok, "an app with 'Sealed Resources=none' must fail the gate"


def test_sealed_app_is_accepted():
    gate = _load_gate()
    assert gate.verify_is_acceptable(0, "valid on disk\n")[0]
    assert gate.describe_is_sealed(_V303_DESCRIBE)[0]


def test_only_the_known_sparkle_complaint_is_tolerated():
    gate = _load_gate()
    assert gate.verify_is_acceptable(1, _V303_VERIFY)[0], \
        "the long-standing Sparkle LICENSE.txt complaint must not fail the build"
    # Same complaint about a different framework: not tolerated.
    other = _V303_VERIFY.replace("Sparkle.framework", "QtCore.framework")
    assert not gate.verify_is_acceptable(1, other)[0]
    # A different complaint about Sparkle: not tolerated.
    changed = _V303_VERIFY.replace(
        "unsealed contents present in the root directory of an embedded framework",
        "a sealed resource is missing or invalid")
    assert not gate.verify_is_acceptable(1, changed)[0]
    # No output at all with a failing exit code: not tolerated.
    assert not gate.verify_is_acceptable(1, "")[0]


def test_build_leftovers_are_recognised():
    gate = _load_gate()
    for name in ("objects-RelWithDebInfo",
                 "QmlAssetDownloaderPrivateplugin_init.cpp.o",
                 "_objc.cpython-311-darwin.so.dSYM",
                 "_objc__dot__cpython-311-darwin__dot__so__dot__dSYM"):
        assert gate.is_build_leftover(name), name
    for name in ("libgeos.dylib", "QtCore", "_objc.cpython-311-darwin.so",
                 "best.pt", "Sparkle.framework", "objects.py"):
        assert not gate.is_build_leftover(name), name


def test_leftovers_are_removed_from_a_fake_app(tmp_path=None):
    import tempfile
    gate = _load_gate()
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp) / "StarTrailCleanR.app"
        junk = app / "Contents" / "Frameworks" / "PySide6" / "objects-RelWithDebInfo" / "x"
        junk.mkdir(parents=True)
        (junk / "x.cpp.o").write_bytes(b"\0")
        dsym = app / "Contents" / "Frameworks" / "objc" / "_objc__dot__so__dot__dSYM"
        dsym.mkdir(parents=True)
        (dsym / "dwarf").write_bytes(b"\0")
        keep = app / "Contents" / "Frameworks" / "objc" / "_objc.cpython-311-darwin.so"
        keep.write_bytes(b"\0")
        removed = gate.remove_build_leftovers(str(app))
        assert len(removed) == 2, removed
        assert not junk.exists() and not dsym.exists()
        assert keep.exists(), "a real library must never be removed"


def test_build_helper_stops_when_sealing_fails():
    code = _code_only((REPO / "build_helper.py").read_text(encoding="utf-8"))
    assert "seal . returncode != 0" in code, \
        "build_helper.py must test the sealing result"
    after = code.split("seal . returncode != 0", 1)[1][:200]
    assert "sys . exit ( 1 )" in after, \
        "build_helper.py must stop the build when the Mac app cannot be sealed"
    raw = (REPO / "build_helper.py").read_text(encoding="utf-8")
    assert "check_mac_signature.py" in raw and "'--seal'" in raw


def test_both_mac_jobs_run_the_gate_on_the_disk_image():
    lines = (REPO / ".github" / "workflows" / "build.yml").read_text(
        encoding="utf-8").splitlines()
    runs = [ln.strip() for ln in lines
            if ln.strip().startswith("run:") and "check_mac_signature.py" in ln]
    assert any("StarTrailCleanR-Mac-AppleSilicon.dmg" in r for r in runs), \
        "the Apple Silicon job must run the signature gate on its disk image"
    assert any("StarTrailCleanR-Mac-Intel.dmg" in r for r in runs), \
        "the Intel job must run the signature gate on its disk image"
