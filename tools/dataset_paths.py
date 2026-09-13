"""Single source of truth for CVAT-task -> image-folder mappings that the review
tools (trail_fixr.py, mask_checkr.py, tile_fixr.py) all read.

ADD A DATASET HERE, ONCE. Datasets whose originals live OUTSIDE the default
TRAILS_ROOT (/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images)
-- e.g. originals kept under Photos/Astrophotography -- go in TASK_ABS_FOLDERS,
keyed by the exact CVAT task name, valued with the absolute folder path.

All three tools import TASK_ABS_FOLDERS from here, so one line covers all of them.
"""

from pathlib import Path

TASK_ABS_FOLDERS = {
    "Bombay Beach 6D - streaky stars (streak-filter 150) - review":
        "/Volumes/T7 Shield/Photos/Astrophotography/_2026/26.6 Bombay Beach - Soly/Star Trail Canon 6D/ST Export Originals",
    # The folder has extra words, and the task says "80sec" where the folder says
    # "80 sec", so auto-match can't reach it. (The folder's old "Johsua" typo was
    # fixed on disk at some point; this alias and both pipeline configs all broke
    # the day that rename happened -- corrected 2026-07-29.)
    "Bruce Herwig - Joshua Tree 6.26 80sec":
        "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Bruce Herwig Joshua Tree 6.26 80 sec exposure Star Trail",
    # Tile-review tasks: tiles live in staging folders (TileFixR knows these; this lets the
    # other tools resolve them too).
    "Bridge misses (107 tiles) - review":
        "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/bridge_review_107",
    "Crossing augmentation review":
        "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/bridge_fix_tiles_2026_06/crossing_aug_review",
    # Bruce's own 2026-09-04 shoot, two cameras on one night. The frames stayed in
    # the photo library rather than being copied under "star trail images", so
    # nothing auto-matches. The "Before" subfolder is the source set -- its sibling
    # "cleaned" holds STC output, including the foreground mask under "STC Extras".
    # (CVAT tasks 68 and 69.)
    "Bruce Herwig - UofR Memorial Chapel - EOS R Left Side":
        "/Volumes/T7 Shield/Photos/Astrophotography/_2026/26.8 Star Trail UofR Memorial Chapel/EOS R Left Side/Before",
    "Bruce Herwig - UofR Memorial Chapel - Canon 6D Right Side":
        "/Volumes/T7 Shield/Photos/Astrophotography/_2026/26.8 Star Trail UofR Memorial Chapel/Canon 6D Right Side/Before",
    # Bruce's April 2026 shoot, rocks in the foreground, dark sky (CVAT task 70).
    # The camera counter ROLLS OVER inside this set: IMG_9799-9999 were shot first,
    # then IMG_0001-0245. Anything that walks these frames by filename gets the
    # order wrong at the seam, so the CVAT task was built in capture-time order.
    "Bruce Herwig - Joshua Tree 26.4 Star Trails":
        "/Volumes/T7 Shield/Photos/Astrophotography/_2026/26.4 Astro Adventure Joshua Tree Borrego Death Valley/Joshua Tree/Star Trails/JPGS",
}


def _norm(s):
    """Lowercase; treat _ and - as spaces; collapse whitespace. So 'borrego_springs_1',
    'Borrego Springs 1' and 'borrego-springs-1' all compare equal."""
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def longest_prefix_folder(task_name, trails_root):
    """Auto-resolve a task whose name STARTS WITH its folder name, e.g.
    'borrego_springs_1 - missed vertical trail (4929-4933)' -> folder 'borrego_springs_1'.

    Among the immediate child folders of trails_root, return the one whose normalized name
    equals the task name or is a whole-word prefix of it; LONGEST match wins (so a more
    specific folder beats a shorter one). Returns a Path, or None if nothing fits. This is a
    last-resort fallback -- the tools try exact/alias/version matches first."""
    trails_root = Path(trails_root)
    if not trails_root.exists():
        return None
    nt = _norm(task_name)
    best, best_len = None, 0
    for child in trails_root.iterdir():
        if not child.is_dir():
            continue
        nc = _norm(child.name)
        if nc and (nt == nc or nt.startswith(nc + " ")) and len(nc) > best_len:
            best, best_len = child, len(nc)
    return best


# ── Run-artifact workspace location (2026-06-19) ──────────────────────────────
# Was <input>/cleanr_workspace/; now lives inside the cleaned folder as "STC Extras".
# Existing datasets keep cleanr_workspace and still resolve via the fallback below.
WORKSPACE_NAME = "STC Extras"
LEGACY_WORKSPACE = "cleanr_workspace"


def resolve_workspace(folder):
    """Find a dataset's run-artifact folder, returning a Path. Checks the new spot
    inside the cleaned folder first, then the legacy spot next to the originals:
      folder/STC Extras  ->  folder/cleaned/STC Extras  ->  folder/cleanr_workspace
    Returns the first that exists; falls back to the legacy path (which may not
    exist) so callers' existing 'not found' checks still fire cleanly."""
    folder = Path(folder)
    for c in (folder / WORKSPACE_NAME, folder / "cleaned" / WORKSPACE_NAME,
              folder / LEGACY_WORKSPACE):
        if c.is_dir():
            return c
    return folder / LEGACY_WORKSPACE
