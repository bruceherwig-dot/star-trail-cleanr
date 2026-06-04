"""Smoke tests for JPG+TIFF de-duplication and batch-frame alignment.

Reproduces the Silvana "River Reflection" crash: a folder mixing single frames
with JPG+TIFF pairs used to inflate the frame count. The GUI planned batches on
files (not real frames), then the worker removed the twins downstream -- which
collapsed the final batch below the 3-frame minimum ("need >= 3 frames, got 2")
and double-cleaned seam frames. These tests lock in the fix: de-duplicate once,
up front, with one shared rule used by both the GUI and the worker.

No model inference; runs in milliseconds.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.frame_list import dedupe_jpg_tiff, gather_frames


def _load_worker():
    """Import astro_clean_v5 the same way test_imports does (tolerate SystemExit)."""
    spec = importlib.util.spec_from_file_location("astro_clean_v5", REPO / "astro_clean_v5.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


# ── dedupe_jpg_tiff rule ──────────────────────────────────────────────────

def test_dedupe_keeps_tiff_drops_jpg_twin():
    files = ["/x/F001.jpg", "/x/F001.tiff", "/x/F002.jpg", "/x/F003.tiff"]
    out = dedupe_jpg_tiff(files)
    assert out == ["/x/F001.tiff", "/x/F002.jpg", "/x/F003.tiff"], out


def test_dedupe_no_pairs_is_identity():
    files = ["/x/F003.jpg", "/x/F001.tiff", "/x/F002.tiff"]
    assert dedupe_jpg_tiff(files) == sorted(files)


def test_dedupe_count_matches_unique_stems():
    files = [f"/x/F{i:03d}.tiff" for i in range(10)]
    files += ["/x/F002.jpg", "/x/F005.jpg", "/x/F007.jpg"]
    assert len(dedupe_jpg_tiff(files)) == 10


def test_dedupe_preserves_input_type():
    # Paths in -> Paths out; strings in -> strings out.
    paths = [Path("/x/F001.jpg"), Path("/x/F001.tiff")]
    out = dedupe_jpg_tiff(paths)
    assert out == [Path("/x/F001.tiff")]
    assert isinstance(out[0], Path)


# ── batch planning + worker slicing stay aligned ──────────────────────────

def _plan(total, max_batch):
    """Mirror star_trail_cleanr's batch planning and per-batch size exactly
    (the last batch absorbs the remainder; a sub-3 tail is merged back)."""
    n_batches = (total + max_batch - 1) // max_batch
    batch_size = (total + n_batches - 1) // n_batches if n_batches else max_batch
    starts = list(range(0, total, batch_size))
    if len(starts) > 1 and (total - starts[-1]) < 3:
        starts.pop()
    plan = []
    for i, s in enumerate(starts):
        this_batch = (total - s) if i == len(starts) - 1 else min(batch_size, total - s)
        plan.append((s, this_batch))
    return plan


def _make_folder(n_unique, dup_indices):
    d = Path(tempfile.mkdtemp())
    for i in range(n_unique):
        (d / f"F{i:03d}.tiff").write_bytes(b"")
        if i in dup_indices:
            (d / f"F{i:03d}.jpg").write_bytes(b"")
    return d


def test_mixed_folder_full_coverage_no_double_clean_no_sub3():
    worker = _load_worker()
    scenarios = [
        (10, {2, 5, 7}, 4),
        (20, {0, 3, 9, 19}, 6),
        (7, {1, 6}, 5),
        (33, set(range(0, 33, 2)), 5),     # heavy, every-other frame duplicated
    ]
    for n_unique, dups, max_batch in scenarios:
        d = _make_folder(n_unique, dups)

        # GUI side: dedup the full list, THEN count and plan.
        uniq = dedupe_jpg_tiff(gather_frames(str(d)))
        total = len(uniq)
        assert total == n_unique, (total, n_unique)

        plan = _plan(total, max_batch)

        cleaned = []
        for (start, this_batch) in plan:
            # Worker side: the real loader, sliced by the planned indices.
            sliced, cs, ce = worker.load_with_neighbors(d, start, this_batch)
            core = [Path(x).stem for x in sliced[cs:ce]]
            ctx = (n_unique, dups, max_batch, start, this_batch, core)
            assert len(core) >= 3, ("sub-3 batch", ctx)
            cleaned += core

        expected = [f"F{i:03d}" for i in range(n_unique)]
        assert sorted(set(cleaned)) == expected, ("coverage", cleaned, expected)
        assert len(cleaned) == len(set(cleaned)), ("double-cleaned", cleaned)
