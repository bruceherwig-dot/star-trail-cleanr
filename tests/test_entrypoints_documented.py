"""The files someone opens first must say what they are.

`star_trail_cleanr.py` (8,900 lines, the whole app) and `astro_clean_v5.py` (the
cleaning engine) had NO module docstring at all until 2026-08-21. Opening either
told you nothing: not what it does, not that they are two halves of one system
that run as separate processes, not which one runs first.

That matters more here than in most projects, because the root directory also
holds three superseded engines with nearly identical names, so a reader who
lands in the wrong file has no way to tell.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

ENTRY_POINTS = ["star_trail_cleanr.py", "astro_clean_v5.py", "make_share_clip.py"]


def test_entry_points_have_a_module_docstring():
    for name in ENTRY_POINTS:
        doc = ast.get_docstring(ast.parse((REPO / name).read_text()))
        assert doc and len(doc) > 200, (
            f"{name} has no real module docstring. It is one of the first files "
            "anyone opens; it must say what it is and how it relates to the rest.")


def test_the_worker_contract_is_written_down():
    """The app runs the engine as a separate process and they can disagree about
    which frames to clean. That has already broken users; it must stay explained
    where someone editing either side will see it."""
    app = (REPO / "star_trail_cleanr.py").read_text()[:6000]
    eng = (REPO / "astro_clean_v5.py").read_text()[:6000]
    assert "--cleanr-worker" in app, "the app must explain how it spawns the engine"
    assert "manifest" in app.lower() and "manifest" in eng.lower(), \
        "both sides must document the frame-manifest contract between them"


def test_superseded_engines_say_so_when_present():
    """Three old engines sit beside the live one with near-identical names.

    They are GITIGNORED, so they exist only in a working copy that has them and
    never in a fresh clone or on CI -- hence the skip. Where they do exist they
    must announce themselves, because that is exactly where someone greps and
    reads the wrong code.
    """
    for name in ["astro_clean.py", "astro_clean_v2.py", "astro_clean_v3.py"]:
        f = REPO / name
        if not f.exists():
            continue
        doc = ast.get_docstring(ast.parse(f.read_text())) or ""
        assert "SUPERSEDED" in doc, (
            f"{name} must say it is not the live engine; astro_clean_v5.py is.")


def test_the_map_exists_and_is_linked():
    arch = REPO / "ARCHITECTURE.md"
    assert arch.exists(), "ARCHITECTURE.md is the map a newcomer needs"
    text = arch.read_text()
    for expected in ["--cleanr-worker", "astro_clean_v5.py", "archive/"]:
        assert expected in text, f"the map must cover {expected}"
    assert "ARCHITECTURE.md" in (REPO / "CLAUDE.md").read_text(), \
        "CLAUDE.md must point at the map, or nobody will find it"
