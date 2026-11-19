"""The release job that keeps `main` equal to the latest release must stay safe.

It may only move main FORWARD (fast-forward) to the tagged commit, and must refuse, loudly,
when main holds a commit the release lacks. Never a force push, never a file-by-file copy.
(Simulated against a scratch repository on 2026-11-18: normal, already-current, blocked and
after-link cases all behaved; see project_current_state in the notes.)
"""
from pathlib import Path

REPO = Path(__file__).parent.parent


def _job():
    text = (REPO / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    a = text.index("\n  sync-changelog:")
    b = text.index("\n  mirror-installers:", a)
    return text[a:b]


def test_the_job_moves_main_forward_and_checks_it_may():
    job = _job()
    assert "fetch-depth: 0" in job, "needs the full history to compare main with the tag"
    assert "git merge-base --is-ancestor origin/main" in job
    assert '${GITHUB_SHA}:refs/heads/main' in job
    assert "if: startsWith(github.ref, 'refs/tags/v')" in job
    assert "needs: [release]" in job


def test_the_job_never_forces_or_copies_files():
    job = _job()
    code = "\n".join(l for l in job.splitlines() if not l.lstrip().startswith("#"))
    for bad in ("--force", " -f ", "+refs/heads", "cp tagged/", "git reset"):
        assert bad not in code, bad
    assert "exit 1" in code, "must fail loudly when main cannot be moved forward"
