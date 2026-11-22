"""The Mac app says which macOS it needs (2026-11-21).

A user on macOS Catalina (10.15) reported that the app "never opens": it failed
silently. The build now writes LSMinimumSystemVersion into the app's Info.plist, so
macOS itself shows "You can't use this version of the application with this version
of macOS". These lock that the key is written (macOS 15, the oldest seen working; one number everywhere), that it happens BEFORE the app is
re-sealed (editing the plist afterwards would break the seal), and that the install
note and the website say the same thing. Offline.
"""
from pathlib import Path

REPO = Path(__file__).parent.parent


def test_the_build_writes_the_minimum_macos_into_the_plist_before_sealing():
    src = (REPO / "build_helper.py").read_text(encoding="utf-8")
    assert "MAC_MINIMUM_MACOS = '15.0'" in src, "the minimum is one named setting"
    assert "LSMinimumSystemVersion" in src
    i_key = src.index("Set :LSMinimumSystemVersion")
    i_seal = src.index("check_mac_signature.py'),\n         '--seal'")
    assert i_key < i_seal, "the plist is edited before the bundle is sealed"


def test_a_failure_to_set_it_stops_the_build():
    src = (REPO / "build_helper.py").read_text(encoding="utf-8")
    i = src.index("could not set the minimum macOS")
    assert "sys.exit(1)" in src[i:i + 200], "a build must not ship without the minimum"


def test_the_install_note_and_the_website_say_the_same_number_as_the_build():
    note = (REPO / "installers" / "open_me_first_mac.txt").read_text(encoding="utf-8")
    assert "needs macOS 15 (Sequoia) or newer" in note
    site = (REPO / "website" / "index.html").read_text(encoding="utf-8")
    assert site.count("macOS 15 or newer") == 2, "both Mac download buttons say it"
