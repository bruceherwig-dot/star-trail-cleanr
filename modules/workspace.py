"""Where a run's artifacts live: the per-run "STC Extras" folder.

History
-------
This folder used to be ``<input>/cleanr_workspace/`` (sitting next to the user's
originals). As of 2026-06-19 it moves INSIDE the cleaned/output folder and is
renamed **"STC Extras"**, so a user dives INTO the cleaned folder to find the Star
Log, share video, Red Trail Map, etc., instead of climbing back out to the
originals. Existing datasets keep their old ``cleanr_workspace/`` and are still
found via the fallback in :func:`find_workspace`, so nothing breaks before (or
even without) a migration.

It holds: the foreground mask, hot-pixel map, the ``masks/`` subfolder (detection
masks + ``<stem>_polys.json``), run logs, the Star Log, the share video, and the
Red Trail Map.

Two helpers, two jobs:
- :func:`output_workspace` — the WRITE location (writers know the output folder).
- :func:`find_workspace` — the READ resolver (readers get a dataset folder and
  must locate the workspace whether it's new, default-new, or legacy).
"""
import glob
import os

WORKSPACE_NAME = "STC Extras"          # new: lives inside the cleaned/output folder
LEGACY_WORKSPACE = "cleanr_workspace"  # old: lived next to the originals
ARCHIVE_NAME = "Archive"               # older logs are tucked here to keep the top tidy

# Run logs that pile up (one timestamp per file). We keep only the NEWEST of each
# at the top level and move the rest into Archive/. Timestamps are zero-padded
# (YYYY-MM-DD_HH-MM-SS), so a plain name sort is chronological.
_LOG_PATTERNS = ("run_log_*.jsonl",
                 "star_log_*.txt",
                 "star_trail_cleanr_log_*.txt",   # legacy name; kept so old logs still tidy up
                 "run_summary_*.txt")


def output_workspace(output_folder, create=False):
    """Return the run-artifact folder for a given cleaned/output folder:
    ``<output_folder>/STC Extras``. Pass ``create=True`` to make it. This is the
    WRITE path, used by the app and engine which already know the output folder."""
    ws = os.path.join(output_folder, WORKSPACE_NAME)
    if create:
        os.makedirs(ws, exist_ok=True)
    return ws


def find_workspace(folder):
    """READ resolver: given a dataset folder, return its workspace dir, or None.

    Checks, in order:
      1. ``folder/STC Extras``            (folder IS the cleaned/output folder)
      2. ``folder/cleaned/STC Extras``    (folder is input, default output layout)
      3. ``folder/cleanr_workspace``      (legacy, next to the originals)

    Returns the first that exists, so old and migrated datasets both resolve.
    Note: a run sent to a NON-default custom output folder won't be found from the
    input folder by step 2; that's an accepted limit (the default cleaned folder
    is the normal case)."""
    if not folder:
        return None
    for candidate in (
        os.path.join(folder, WORKSPACE_NAME),
        os.path.join(folder, "cleaned", WORKSPACE_NAME),
        os.path.join(folder, LEGACY_WORKSPACE),
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def plan_log_archive(workspace_dir):
    """Return ``[(src, dest), ...]`` for the older run logs that should move into
    ``<workspace>/Archive/`` — every log file EXCEPT the newest of each type. The
    newest of each stays at the top level. Returns [] if nothing needs moving.

    Moves only, never deletes or overwrites: each log filename carries a unique
    timestamp, so an Archive name never collides."""
    if not workspace_dir or not os.path.isdir(workspace_dir):
        return []
    archive = os.path.join(workspace_dir, ARCHIVE_NAME)
    moves = []
    for pattern in _LOG_PATTERNS:
        files = sorted(glob.glob(os.path.join(workspace_dir, pattern)))
        for f in files[:-1]:                       # all but the newest (last after sort)
            moves.append((f, os.path.join(archive, os.path.basename(f))))
    return moves


def archive_old_logs(workspace_dir):
    """Tidy a workspace: move all but the newest of each run-log type into
    ``<workspace>/Archive/``. Returns the number of files moved. Safe to call after
    every run -- it never touches the newest logs, never overwrites, never deletes,
    and silently does nothing if there's nothing to move."""
    moves = plan_log_archive(workspace_dir)
    if not moves:
        return 0
    os.makedirs(os.path.join(workspace_dir, ARCHIVE_NAME), exist_ok=True)
    moved = 0
    for src, dest in moves:
        if os.path.exists(dest):                   # never overwrite (shouldn't happen)
            continue
        os.rename(src, dest)
        moved += 1
    return moved
