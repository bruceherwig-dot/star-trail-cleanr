"""Inter font must be bundled and loadable.

The GUI bundles Inter (OFL-licensed) and forces every widget to render in it,
so widget widths are identical across Mac, Windows, and Linux. Without this,
each platform's system default font (San Francisco / Segoe UI / DejaVu Sans)
renders the same point size at different widths and clips fixed-width controls
on whichever OS we didn't develop on. The 55-px JPEG quality spinbox that
hid the number on Windows was an example.

This test ensures:
  1. The two font files exist on disk (so the build picks them up).
  2. The startup snippet that loads them and sets the app font is present.
  3. The HTML content uses Inter as the primary font-family.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_inter_files_present():
    fonts_dir = REPO / "assets" / "fonts"
    for fname in ("Inter-Regular.ttf", "Inter-Bold.ttf"):
        fpath = fonts_dir / fname
        assert fpath.exists(), f"missing bundled font: {fpath}"
        size = fpath.stat().st_size
        # Real Inter ttfs are ~400 KB each. Tripwire for accidental empty file.
        assert size > 50_000, f"{fname} suspiciously small ({size} bytes)"


def test_ofl_license_present():
    """Inter ships under SIL Open Font License. We must distribute the
    license alongside the binaries. Missing this is a license violation."""
    lic = REPO / "assets" / "fonts" / "Inter-LICENSE.txt"
    assert lic.exists(), "missing assets/fonts/Inter-LICENSE.txt"
    text = lic.read_text()
    assert "SIL Open Font License" in text, \
        "Inter-LICENSE.txt does not look like the OFL we shipped"


def test_startup_loads_inter_and_sets_app_font():
    src = (REPO / "star_trail_cleanr.py").read_text()
    assert "QFontDatabase.addApplicationFont" in src, \
        "GUI must register Inter via addApplicationFont at startup"
    assert "Inter-Regular.ttf" in src and "Inter-Bold.ttf" in src, \
        "GUI must reference both Inter weights at startup"
    assert "app.setFont(" in src, \
        "GUI must call app.setFont(...) so every widget uses Inter"
    assert 'QFontDatabase.families()' in src, \
        "GUI must verify Inter loaded before calling setFont (graceful fallback)"


def test_html_widgets_use_inter_first():
    """The FAQ and About panels render HTML in QTextBrowser. Their CSS must
    list Inter first so they match the rest of the app, with platform fonts
    as fallback if Inter ever fails to load."""
    src = (REPO / "star_trail_cleanr.py").read_text()
    assert "font-family: Inter," in src, \
        "HTML body styles must list Inter as the first font-family entry"
    assert "font-family: -apple-system, sans-serif" not in src, \
        "old Mac-only font stack still present — would render different widths on Windows"
