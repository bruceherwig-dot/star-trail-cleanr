"""
frame_list.py — single source of truth for listing the frames in a folder.

WHY THIS EXISTS
---------------
The GUI (star_trail_cleanr.py) and the worker (astro_clean_v5.py) each list the
photos in the input folder independently. The GUI counts them and splits them
into batches, handing the worker a starting frame number and a count. The worker
then re-lists the same folder and slices by those numbers. For the numbers to
mean the same thing on both sides, both MUST produce the identical frame list.

A folder can hold more than one file for the same frame: a JPG and a TIFF, or a
camera RAW alongside an in-camera JPG. We keep ONE per frame. The bug this module
fixes: that de-duplication used to happen inside the worker AFTER the GUI had
already counted files and planned batches, so the count was inflated by the
duplicate twins. Consequences were a final batch that collapsed below the
3-frame minimum (a hard crash) and seam frames that got cleaned twice. Removing
duplicates here, once, before any counting or slicing, on BOTH sides, keeps the
two in lockstep.

TWIN PREFERENCE (RAW vs JPG/TIFF)
---------------------------------
When a frame exists as both a RAW and a JPG/TIFF, the user picks which to process
(the GUI prompts once per folder, defaulting to RAW). That choice flows in as
`prefer_raw`. A frame that exists in ONLY one format is always kept regardless of
the preference, so nothing is ever dropped to zero.
"""

import os

# Extensions we treat as frames. Lower-case; callers compare case-insensitively.
# Camera RAW formats (libraw/rawpy coverage). A RAW is debayered to a 16-bit RGB
# array at load time in modules/io_safe.py; the detection/repair pipeline never
# sees a RAW file directly.
RAW_EXTS = {
    ".cr2", ".cr3", ".crw",          # Canon
    ".nef", ".nrw",                  # Nikon
    ".arw", ".srf", ".sr2",          # Sony
    ".raf",                          # Fujifilm
    ".dng",                          # Adobe / generic / DJI / Pixel
    ".orf",                          # Olympus / OM System
    ".rw2",                          # Panasonic
    ".pef",                          # Pentax
    ".srw",                          # Samsung
    ".raw", ".rwl",                  # Leica
    ".3fr", ".fff",                  # Hasselblad
    ".iiq",                          # Phase One
    ".mrw",                          # Minolta
    ".dcr", ".kdc",                  # Kodak
    ".mos",                          # Leaf
    ".erf",                          # Epson
    ".mef",                          # Mamiya
    ".gpr",                          # GoPro
}
_TIF_EXTS = {".tif", ".tiff"}
_JPG_EXTS = {".jpg", ".jpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"} | RAW_EXTS

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


def is_raw(path):
    """True if a path's extension is a camera RAW format."""
    return _stem_and_ext(path)[1] in RAW_EXTS


def _format_rank(ext, prefer_raw):
    """Higher rank wins when two files share a frame stem.

    prefer_raw=True  -> RAW beats TIFF beats JPG beats anything else.
    prefer_raw=False -> TIFF beats JPG beats other; RAW ranks LAST so it is
                        only kept when it is the sole file for that frame.
    Ties (e.g. two TIFFs) are broken by sorted order in the caller (first kept).
    """
    if prefer_raw:
        if ext in RAW_EXTS:
            return 3
        if ext in _TIF_EXTS:
            return 2
        if ext in _JPG_EXTS:
            return 1
        return 0
    else:
        if ext in _TIF_EXTS:
            return 3
        if ext in _JPG_EXTS:
            return 2
        if ext in RAW_EXTS:
            return 0
        return 1


def dedupe_frames(paths, prefer_raw=True):
    """Keep one file per frame stem, honoring the RAW-vs-JPG/TIFF preference.

    Groups inputs by file stem (name without extension). When a stem has more
    than one file, the highest-ranked format wins (see _format_rank). A frame
    present in only one format is always kept, whatever the preference, so a
    RAW-only frame survives even when the user chose JPG/TIFF. Returns a new
    list, sorted, with the original path objects unchanged (strings stay
    strings, Paths stay Paths). Fully deterministic, so the same input always
    yields the same output, which is what lets the GUI and worker agree.
    """
    chosen = {}
    for fp in sorted(paths, key=lambda p: str(p)):
        stem, ext = _stem_and_ext(fp)
        if stem in chosen:
            _, prev_ext = _stem_and_ext(chosen[stem])
            if _format_rank(ext, prefer_raw) > _format_rank(prev_ext, prefer_raw):
                chosen[stem] = fp
        else:
            chosen[stem] = fp
    return sorted(chosen.values(), key=lambda p: str(p))


def dedupe_jpg_tiff(paths):
    """Backwards-compatible alias. RAW wins, then TIFF, then JPG (the default
    preference). Kept so existing callers and tests keep working; new code
    should call dedupe_frames(paths, prefer_raw=...) directly."""
    return dedupe_frames(paths, prefer_raw=True)


def count_raw_twins(paths):
    """Count frame stems that exist as BOTH a RAW and a non-RAW (JPG/TIFF/PNG)
    file. This is what the GUI uses to decide whether to ask the user which
    format to process. Returns 0 when there are no such pairs."""
    by_stem = {}
    for fp in paths:
        stem, ext = _stem_and_ext(fp)
        if ext in IMAGE_EXTS:
            by_stem.setdefault(stem, set()).add(ext)
    n = 0
    for exts in by_stem.values():
        has_raw = any(e in RAW_EXTS for e in exts)
        has_other = any(e not in RAW_EXTS for e in exts)
        if has_raw and has_other:
            n += 1
    return n


def glob_patterns():
    """Return shell-glob patterns for every supported extension, both lower and
    upper case, for callers that scan a folder with glob.glob (case-sensitive on
    Linux). Keeps those callers in sync with IMAGE_EXTS automatically."""
    pats = []
    for e in sorted(IMAGE_EXTS):
        bare = e.lstrip(".")
        pats.append("*." + bare.lower())
        pats.append("*." + bare.upper())
    return pats


def gather_frames(folder):
    """List every image file in a folder, sorted, as a list of full path strings.

    Case-insensitive extension match. Does NOT de-duplicate — call
    dedupe_frames on the result when a unique-frame list is needed.
    """
    out = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                out.append(full)
    return sorted(out)
