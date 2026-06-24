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
import re

# Number-aware ("natural") ordering of frame paths. A plain text sort puts
# "10.jpg" right after "1.jpg" (and "2.jpg" all the way after "100.jpg"), which
# scrambles the capture order for folders whose filenames are NOT zero-padded --
# e.g. GoPro sets named 1.jpg ... 900.jpg. Scrambled order means the Star Bridge
# repair borrows from the wrong neighbor frames and the final stack is out of
# sequence. natural_key splits each path into text and number chunks and sorts
# the number chunks numerically, so 1, 2, ... 9, 10, 11 come out in true order.
# Zero-padded names (IMG_0001, 143A8819, LRT_00651) are UNAFFECTED -- their text
# order already equals their numeric order. re.split with a capturing group
# always alternates text/number chunks at fixed positions, so two paths compare
# text-to-text and int-to-int by position and never raise a type error.
_NUM_CHUNK_RE = re.compile(r"(\d+)")


def natural_key(path):
    """Sort key giving number-aware order for a file path (str or Path).
    Used for EVERY frame-list ordering so the GUI and worker stay in lockstep."""
    return [int(chunk) if chunk.isdigit() else chunk.lower()
            for chunk in _NUM_CHUNK_RE.split(str(path))]


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
# Every extension the app accepts as a frame: JPG, PNG, TIFF, plus all RAW
# formats above. Note PNG has no dedicated rank set of its own, so in twin
# de-duplication it is treated as a generic non-RAW "other" format.
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
    # Pre-sort the inputs so that, when two files of the SAME format share a
    # stem (e.g. two TIFFs), the first one in sorted order is the one kept --
    # higher-ranked formats still override later, but equal ranks never do
    # (the > comparison below is strict). This pre-sort is what makes the
    # tie-break deterministic and identical on the GUI and worker sides.
    chosen = {}
    for fp in sorted(paths, key=natural_key):
        stem, ext = _stem_and_ext(fp)
        if stem in chosen:
            _, prev_ext = _stem_and_ext(chosen[stem])
            # Replace the already-chosen file only if this one ranks STRICTLY
            # higher; equal-rank duplicates keep the earlier (sorted) one.
            if _format_rank(ext, prefer_raw) > _format_rank(prev_ext, prefer_raw):
                chosen[stem] = fp
        else:
            chosen[stem] = fp
    return sorted(chosen.values(), key=natural_key)


def dedupe_jpg_tiff(paths):
    """Backwards-compatible alias. RAW wins, then TIFF, then JPG (the default
    preference). Kept so existing callers and tests keep working; new code
    should call dedupe_frames(paths, prefer_raw=...) directly."""
    return dedupe_frames(paths, prefer_raw=True)


def count_raw_twins(paths):
    """Count frame stems that exist as BOTH a RAW and a non-RAW (JPG/TIFF/PNG)
    file. This is what the GUI uses to decide whether to ask the user which
    format to process. Returns 0 when there are no such pairs."""
    # Build a map of frame stem -> set of extensions present for that frame.
    # Only recognized image extensions are counted; any other file is ignored.
    by_stem = {}
    for fp in paths:
        stem, ext = _stem_and_ext(fp)
        if ext in IMAGE_EXTS:
            by_stem.setdefault(stem, set()).add(ext)
    # A "twin" is a stem that has at least one RAW extension AND at least one
    # non-RAW extension (JPG/TIFF/PNG). Two RAWs of the same stem, or two JPGs,
    # do not count -- only mixed RAW/non-RAW pairs trigger the format prompt.
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
    return sorted(out, key=natural_key)


def order_by_capture_time(paths, times):
    """Return `paths` reordered by true capture time when EVERY path has a
    timestamp in `times` (a dict path -> datetime or None); otherwise return
    `paths` unchanged (it is assumed already natural_key-sorted). This
    all-or-nothing rule per folder avoids mixing two ordering schemes. Ties
    (same-second frames) break by natural_key. Fixes filename orders that no
    longer match shooting order -- a camera file-number rollover (IMG_9999 ->
    IMG_0001 mid-shoot) or frames merged from two cards."""
    paths = list(paths)
    if not paths:
        return paths
    if any(times.get(p) is None for p in paths):
        return paths
    return sorted(paths, key=lambda p: (times[p], natural_key(p)))


def write_manifest(manifest_path, ordered_paths):
    """Write the canonical ordered frame list (one absolute path per line) so
    the worker can use the GUI's exact order instead of re-deriving it. This
    keeps the two sides in perfect lockstep and means the worker never re-reads
    every frame's timestamp on every batch."""
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in ordered_paths:
            f.write(str(p) + "\n")


def read_manifest(manifest_path):
    """Read a frame manifest written by write_manifest: the ordered list of
    path strings (blank lines ignored)."""
    with open(manifest_path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]
