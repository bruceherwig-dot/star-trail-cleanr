"""
frame_list.py — single source of truth for listing the frames in a folder.

WHY THIS EXISTS
---------------
The GUI (star_trail_cleanr.py) and the worker (astro_clean_v5.py) each list the
photos in the input folder independently. The GUI counts them and splits them
into batches, handing the worker a starting frame number and a count. The worker
then re-lists the same folder and slices by those numbers. For the numbers to
mean the same thing on both sides, both MUST produce the identical frame list.

A folder can hold both a JPG and a TIFF of the same frame. We keep the TIFF and
drop the JPG twin. The bug this module fixes: that de-duplication used to happen
inside the worker AFTER the GUI had already counted files and planned batches,
so the count was inflated by the duplicate twins. Consequences were a final
batch that collapsed below the 3-frame minimum (a hard crash) and seam frames
that got cleaned twice. Removing duplicates here, once, before any counting or
slicing, on BOTH sides, keeps the two in lockstep.
"""

import os

# Extensions we treat as frames. Lower-case; callers compare case-insensitively.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_TIF_EXTS = {".tif", ".tiff"}

# Minimum frame size for trail detection. The detector tiles each frame into
# 640px windows; a frame whose shorter side is below this both detects poorly
# (trails are only a few pixels) and is below the tile size, which the tiler
# cannot form. Real camera frames are always far larger (thousands of px);
# only downsized web previews (e.g. 800x533) fall below this. The app blocks
# these up front with a clear message instead of producing junk or crashing.
MIN_FRAME_SHORT_SIDE = 1280


def frame_too_small(width, height):
    """True if a frame is too small for reliable trail detection: its shorter
    side is under MIN_FRAME_SHORT_SIDE pixels."""
    return min(int(width), int(height)) < MIN_FRAME_SHORT_SIDE


def _stem_and_ext(path):
    """Return (stem, lowercased extension) for a str or Path, type-agnostic."""
    base = os.path.basename(str(path))
    stem, ext = os.path.splitext(base)
    return stem, ext.lower()


def dedupe_jpg_tiff(paths):
    """Drop JPG/other twins when a TIFF of the same frame name exists.

    Groups inputs by file stem (name without extension). When a stem has more
    than one file, the TIFF wins; if no TIFF is present, the first in sorted
    order is kept. Returns a new list, sorted, containing the original path
    objects unchanged (strings stay strings, Paths stay Paths) so callers can
    keep using their existing types. Fully deterministic — same input always
    yields the same output, which is what lets the GUI and worker agree.
    """
    chosen = {}
    for fp in sorted(paths, key=lambda p: str(p)):
        stem, ext = _stem_and_ext(fp)
        if stem in chosen:
            _, prev_ext = _stem_and_ext(chosen[stem])
            if ext in _TIF_EXTS and prev_ext not in _TIF_EXTS:
                chosen[stem] = fp
        else:
            chosen[stem] = fp
    return sorted(chosen.values(), key=lambda p: str(p))


def gather_frames(folder):
    """List every image file in a folder, sorted, as a list of full path strings.

    Case-insensitive extension match. Does NOT de-duplicate — call
    dedupe_jpg_tiff on the result when a unique-frame list is needed.
    """
    out = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                out.append(full)
    return sorted(out)
