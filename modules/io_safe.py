"""Robust image reader with tifffile fallback and transient-IO retry.

Some valid TIFFs (BigTIFF, certain LZW/predictor combos, some camera-export
variants) make OpenCV's bundled libtiff fail with `TIFFReadRGBAStrip` errors
even though `tifffile` reads them fine. USB and network drives also drop
reads occasionally on long jobs. This module wraps the imread call with a
fallback ladder so one bad read doesn't kill a whole batch, and exposes the
underlying reason every reader gave so callers can show it to the user.
"""
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .frame_list import RAW_EXTS


_TIFF_EXTS = {".tif", ".tiff"}


def _silence_cv2_logs():
    """Mute OpenCV's own console logging and return the previous level.

    When OpenCV's libtiff fails to read a quirky TIFF it prints noisy
    `TIFFReadRGBAStrip` warnings to the terminal even though we are about to
    rescue the file with tifffile or Pillow. Those warnings would alarm the
    user for no reason, so the reader silences OpenCV while it works the
    fallback ladder and restores the prior level afterward via
    `_restore_cv2_logs`. Returns the old log level (to hand back later), or
    None if OpenCV's logging API isn't available on this build.
    """
    try:
        from cv2.utils import logging as _cvlog
        prev = _cvlog.getLogLevel()
        _cvlog.setLogLevel(_cvlog.LOG_LEVEL_SILENT)
        return prev
    except Exception:
        return None


def _restore_cv2_logs(prev):
    """Put OpenCV's console log level back to what `_silence_cv2_logs` returned.

    `prev` is the value handed back by `_silence_cv2_logs`. If it's None
    (logging API unavailable, or nothing was changed) this is a no-op. Always
    called in a `finally` block so the level is restored even if a read raises.
    """
    if prev is None:
        return
    try:
        from cv2.utils import logging as _cvlog
        _cvlog.setLogLevel(prev)
    except Exception:
        pass


def _try_cv2(path: str, flags: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Attempt the fast path: read the file with OpenCV's `cv2.imread`.

    This is the first reader tried for non-RAW files because it covers the vast
    majority of inputs. `flags` is an OpenCV imread flag (e.g. IMREAD_UNCHANGED
    to keep native bit depth, IMREAD_COLOR / IMREAD_GRAYSCALE for an 8-bit
    result). Returns a `(image, error)` pair: on success `(array, None)`, on
    failure `(None, reason_string)` describing why so callers can build a
    multi-reader diagnosis. cv2 returns images in BGR layout.
    """
    try:
        # IMREAD_IGNORE_ORIENTATION: never let cv2 silently rotate by EXIF.
        # Orientation is applied once, centrally, in robust_imread_diag so every
        # backend (cv2/tifffile/PIL) and every flag behave identically.
        img = cv2.imread(path, flags | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            return None, "returned no image (unsupported format or unreadable bytes)"
        return img, None
    except Exception as e:
        return None, f"raised {type(e).__name__}: {e}"


def _match_flag_depth(arr: np.ndarray, flags: int) -> np.ndarray:
    """Honor OpenCV's 8-bit contract for IMREAD_COLOR / IMREAD_GRAYSCALE.

    cv2.imread returns 8-bit for those flags natively; the tifffile and PIL
    fallbacks (and rawpy) must do the same, or a 16-bit image they rescue would
    reach an 8-bit display (the mask painter, previews) as colour noise.
    IMREAD_UNCHANGED keeps the native depth, so the worker's full-quality path
    is unaffected.
    """
    if flags == cv2.IMREAD_UNCHANGED or arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    a = arr.astype(np.float64)
    mx = float(a.max()) if a.size else 0.0
    if mx > 255:
        a = a * (255.0 / mx)
    return np.clip(a, 0, 255).astype(np.uint8)


def _try_pil(path: str, flags: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read with Pillow. The whole reason this fallback exists: cv2.imread
    on Windows uses ANSI file APIs that cannot open paths containing
    non-ASCII characters (Slovak, Czech, German umlauts, French accents,
    Cyrillic, CJK, etc.). It fails BEFORE it even tries to decode the file,
    so the OpenCV "tried 3 times" message is misleading — the file is fine,
    we just can't open it. Pillow uses Python's normal file APIs which
    handle Unicode correctly on every platform. Returns BGR layout to
    match OpenCV's convention.

    EXIF orientation is NOT applied here; it is applied once, centrally, in
    robust_imread_diag, so every backend and every flag return identically
    (un-)oriented pixels.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im)
    except Exception as e:
        return None, f"raised {type(e).__name__}: {e}"
    if arr is None or arr.size == 0:
        return None, "returned empty image"
    arr = _match_flag_depth(arr, flags)

    if arr.ndim == 2:
        if flags == cv2.IMREAD_GRAYSCALE:
            return arr, None
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR), None

    if arr.ndim == 3:
        if arr.shape[2] == 3:
            if flags == cv2.IMREAD_GRAYSCALE:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), None
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), None
        if arr.shape[2] == 4:
            if flags == cv2.IMREAD_GRAYSCALE:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY), None
            if flags == cv2.IMREAD_COLOR:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), None
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA), None
    return arr, None


def _try_rawpy(path: str, flags: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Decode a camera RAW (CR2/CR3/NEF/ARW/RAF/DNG/...) with rawpy (libraw).

    Debayers to a 16-bit RGB array, then returns it in OpenCV's BGR convention
    so the rest of the pipeline treats it exactly like a 16-bit TIFF.

    Two settings matter for a star-trail SEQUENCE and are deliberately fixed:
      * no_auto_bright=True  disables per-frame auto exposure. Auto-brighten
        would scale each frame's levels independently, so frames would no longer
        line up brightness-wise and the lighten-max stack would band/flicker.
      * use_camera_wb=True   uses the white balance the camera recorded, the
        same for every frame, rather than re-estimating per frame.

    Bit depth follows OpenCV's flag semantics, exactly like the other readers:
      * IMREAD_UNCHANGED -> full 16-bit (what the worker uses to preserve depth)
      * IMREAD_COLOR / IMREAD_GRAYSCALE -> 8-bit (what callers like the mask
        painter expect; they build an 8-bit display and would show 16-bit data
        as colour noise).
    Orientation is applied from the file's own flag (rawpy's default), so no
    separate EXIF-rotate step is needed.
    """
    try:
        import rawpy
    except Exception as e:
        return None, (f"rawpy not available ({type(e).__name__}: {e}); "
                      "RAW decoding requires the rawpy package")
    bps = 16 if flags == cv2.IMREAD_UNCHANGED else 8
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=bps,
            )
    except Exception as e:
        return None, f"raised {type(e).__name__}: {e}"
    if rgb is None or rgb.size == 0:
        return None, "returned empty image"
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        if flags == cv2.IMREAD_GRAYSCALE:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None
    return rgb, None


def _try_tifffile(path: str, flags: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read a TIFF with tifffile (handles BigTIFF, unusual compressions, and
    camera-export variants OpenCV's libtiff chokes on). Returns the array in
    OpenCV's BGR layout convention, plus an error string on failure.
    """
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception as e:
        return None, f"raised {type(e).__name__}: {e}"
    if arr is None:
        return None, "returned no image"
    arr = _match_flag_depth(arr, flags)

    if arr.ndim == 2:
        return arr, None
    if arr.ndim != 3:
        return arr, None

    # Channel counts the rest of the app can't use.
    #
    # tifffile is the only reader here that will hand back an image with more
    # than four channels: OpenCV refuses anything outside 1-4, and Pillow raises
    # (logging "More samples per pixel than can be decoded" on its way out, which
    # is the only trace of this a crash report used to show). So tifffile quietly
    # succeeded and passed, say, a seven-channel Photoshop export straight into a
    # pipeline that only understands three or four, and OpenCV threw much further
    # downstream when it tried to convert the colours.
    #
    # Trim here, where the RGB-to-BGR convention is already handled, so every
    # caller (worker, previews, mask editor, share stacker) gets a normal photo:
    #   more than 4 -> keep red, green and blue; drop spot/alpha extras
    #   exactly 2   -> grey plus alpha; keep the grey
    if arr.shape[2] > 4:
        print(f"  Note: {Path(path).name} has {arr.shape[2]} colour channels. "
              f"Using its red, green and blue; any extra channels "
              f"(spot colours, saved selections, transparency) are ignored.",
              flush=True)
        arr = np.ascontiguousarray(arr[:, :, :3])
    elif arr.shape[2] == 2:
        arr = np.ascontiguousarray(arr[:, :, 0])
        return arr, None
    elif arr.shape[2] not in (3, 4):
        return arr, None

    if flags == cv2.IMREAD_GRAYSCALE:
        code = cv2.COLOR_RGBA2GRAY if arr.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(arr, code), None
    if flags == cv2.IMREAD_COLOR:
        code = cv2.COLOR_RGBA2BGR if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR
        return cv2.cvtColor(arr, code), None
    code = cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR
    return cv2.cvtColor(arr, code), None


def _imread_diag_inner(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_UNCHANGED,
    *,
    _retry_delays: Tuple[float, ...] = (1.0, 3.0),
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read with fallbacks. Returns (img_or_None, diagnosis_or_None).

    RAW files (extensions in RAW_EXTS) skip the ladder below entirely and go
    straight to rawpy, with the same transient-IO retry; cv2/tifffile/Pillow
    cannot decode a raw sensor file.

    Order of attempts for non-RAW files:
      1. cv2.imread (fast path, covers ~95% of files)
      2. tifffile.imread for .tif / .tiff (BigTIFF, unusual compressions,
         camera-export variants OpenCV can't decode)
      3. PIL.Image.open (Unicode-path safe — cv2 on Windows can't open
         files whose path contains non-ASCII characters; PIL can. Also
         handles formats and edge cases cv2 misses).
      4. After all three fail, sleep and retry. Default delays are 1s
         then 3s, so a brief external-drive hiccup or a USB drive waking
         from sleep gets up to ~4 seconds total to recover before we
         surface anything to the user. The retry success case is silent
         — the load loop keeps going with a small pause and the user
         never sees a modal.

    Tests pass `_retry_delays=()` to skip the waits entirely.

    On success: diagnosis is None.
    On failure: diagnosis is a multi-line string with what each reader said.
    """
    p = str(path)
    suffix = Path(p).suffix.lower()
    is_tiff = suffix in _TIFF_EXTS
    is_raw_file = suffix in RAW_EXTS

    prev = _silence_cv2_logs()
    try:
        attempts = []

        # RAW files: only rawpy can decode them. OpenCV/tifffile/Pillow all
        # fail on a raw sensor file, so skip that ladder and go straight to
        # rawpy, with the same transient-IO retry the other readers get.
        if is_raw_file:
            img, why = _try_rawpy(p, flags)
            if img is not None:
                return img, None
            attempts.append(("rawpy", why))
            for n, delay in enumerate(_retry_delays, start=1):
                if delay > 0:
                    time.sleep(delay)
                img, why = _try_rawpy(p, flags)
                if img is not None:
                    return img, None
                if attempts[-1][1] != why:
                    attempts.append((f"rawpy (retry {n})", why))
            diag = "\n".join(f"    {label}: {why}" for label, why in attempts)
            return None, diag

        img, why = _try_cv2(p, flags)
        if img is not None:
            return img, None
        attempts.append(("OpenCV", why))

        if is_tiff:
            img, why = _try_tifffile(p, flags)
            if img is not None:
                return img, None
            attempts.append(("tifffile", why))

        img, why = _try_pil(p, flags)
        if img is not None:
            return img, None
        attempts.append(("Pillow", why))

        for n, delay in enumerate(_retry_delays, start=1):
            if delay > 0:
                time.sleep(delay)

            img, why = _try_cv2(p, flags)
            if img is not None:
                return img, None
            if attempts[-1][1] != why:
                attempts.append((f"OpenCV (retry {n})", why))

            if is_tiff:
                img, why = _try_tifffile(p, flags)
                if img is not None:
                    return img, None
                if attempts[-1][1] != why:
                    attempts.append((f"tifffile (retry {n})", why))

            img, why = _try_pil(p, flags)
            if img is not None:
                return img, None
            if attempts[-1][1] != why:
                attempts.append((f"Pillow (retry {n})", why))

        diag = "\n".join(f"    {label}: {why}" for label, why in attempts)
        return None, diag
    finally:
        _restore_cv2_logs(prev)


# EXIF Orientation tag -> numpy transform that turns stored pixels upright.
# Verified to match PIL.ImageOps.exif_transpose. Applied once, centrally, so the
# whole app (worker, mask painter, previews) always works on upright pixels and
# the cleaned output gets a "normal" orientation tag (no double-rotation).
_ORIENT_OPS = {
    2: lambda a: np.fliplr(a),
    3: lambda a: np.rot90(a, 2),
    4: lambda a: np.flipud(a),
    5: lambda a: np.swapaxes(a, 0, 1),                 # transpose (main diagonal)
    6: lambda a: np.rot90(a, 3),                       # rotate 90 clockwise
    7: lambda a: np.rot90(np.swapaxes(a, 0, 1), 2),    # transverse (anti-diagonal)
    8: lambda a: np.rot90(a, 1),                       # rotate 90 counter-clockwise
}


def exif_orientation(path: Union[str, Path]) -> int:
    """Return the EXIF Orientation tag (1-8), or 1 if absent/unreadable."""
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            ex = im.getexif()
            return int(ex.get(0x0112, 1)) if ex else 1
    except Exception:
        return 1


def _apply_orientation(path: Union[str, Path], img: np.ndarray) -> np.ndarray:
    """Turn stored pixels upright per the file's EXIF Orientation, preserving
    bit depth (operates on the array). No-op for orientation 1 / no tag."""
    op = _ORIENT_OPS.get(exif_orientation(path))
    return np.ascontiguousarray(op(img)) if op is not None else img


def _promote_grey(img: np.ndarray, flags: int) -> np.ndarray:
    """Give a single-channel photo the three channels the rest of the app assumes.

    Black-and-white sources exist in the wild: telescope sub-frames converted from
    FITS without debayering (SeeStar and friends), mono astro cameras, and any
    greyscale TIFF export. IMREAD_UNCHANGED hands those back as a flat (H,W)
    array, and every stage downstream is written for (H,W,3): Star Bridge repair
    asks for the brightest of the three colours at a pixel, and the 16-bit TIFF
    writer converts BGR to RGB. Both crash on a two-dimensional array -- the
    field crash of 2026-08-25 (`AxisError: axis 2 is out of bounds`) on a folder
    of greyscale telescope subs.

    Promoting here, next to the central orientation fix, means one edit covers the
    worker, the detector and the tools instead of each growing its own guard.

    IMREAD_GRAYSCALE is left alone: callers asking for it (the foreground mask,
    the hot-pixel map) want a single channel and would break if handed three.
    """
    if flags == cv2.IMREAD_GRAYSCALE or img.ndim != 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def robust_imread_diag(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_UNCHANGED,
    *,
    _retry_delays: Tuple[float, ...] = (1.0, 3.0),
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read with fallbacks, then apply EXIF orientation and the single-channel
    promotion centrally, so every backend/flag returns upright pixels with the
    channel count the rest of the app expects. RAW files are oriented by rawpy
    already, so orientation is left untouched here."""
    img, diag = _imread_diag_inner(path, flags, _retry_delays=_retry_delays)
    if img is not None and Path(str(path)).suffix.lower() not in RAW_EXTS:
        img = _apply_orientation(path, img)
    if img is not None:
        img = _promote_grey(img, flags)
    return img, diag


def robust_imread(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_UNCHANGED,
    *,
    _retry_delays: Tuple[float, ...] = (1.0, 3.0),
) -> Optional[np.ndarray]:
    """Drop-in cv2.imread replacement. Returns the image or None.

    Use `robust_imread_diag` instead if you want to surface the underlying
    reason a read failed.
    """
    img, _ = robust_imread_diag(path, flags, _retry_delays=_retry_delays)
    return img


def image_size(path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """Return (width, height) for an image without a full decode where possible.

    Dimensions are UPRIGHT: a frame shot sideways (RAW rotate flag, or EXIF
    orientation 5-8) reports its rotated width/height, matching the shape the
    worker actually loads. This is what lets the pre-run scan tell portrait from
    landscape -- their stored sensor sizes are identical and only diverge once
    rotated upright.

    Handles RAW via rawpy (.sizes). For everything else it tries Pillow first;
    if Pillow fails on a TIFF it falls back to tifffile. Mirrors the reader's
    format coverage so the GUI's pre-flight scan never rejects a file the worker
    would actually be able to process. Returns None if the size can't be
    determined.
    """
    p = str(path)
    ext = Path(p).suffix.lower()

    if ext in RAW_EXTS:
        try:
            import rawpy
            with rawpy.imread(p) as raw:
                s = raw.sizes
                w, h = int(s.width), int(s.height)
                # rawpy/libraw flip 5 (90 deg CCW) and 6 (90 deg CW) turn the
                # frame a quarter turn, so the upright image swaps width/height.
                # The worker loads RAW upright (rawpy applies the flip), so report
                # upright dims here too -- otherwise a portrait and a landscape
                # shot read as the same size and the pre-run check can't tell them
                # apart; they only diverge once the frame is rotated upright.
                return (h, w) if s.flip in (5, 6) else (w, h)
        except Exception:
            return None

    try:
        from PIL import Image
        with Image.open(p) as im:
            w, h = int(im.size[0]), int(im.size[1])
            ex = im.getexif()
            orient = int(ex.get(0x0112, 1)) if ex else 1
        # EXIF orientation 5-8 rotate the frame a quarter turn; the worker turns
        # such frames upright on load (via exif_orientation), so report the
        # upright (swapped) dims to match.
        return (h, w) if orient in (5, 6, 7, 8) else (w, h)
    except Exception:
        pass

    if ext in _TIFF_EXTS:
        try:
            import tifffile
            with tifffile.TiffFile(p) as tf:
                page = tf.pages[0]
                w, h = int(page.imagewidth), int(page.imagelength)
            # Match the worker, which orients non-RAW frames via exif_orientation.
            return (h, w) if exif_orientation(p) in (5, 6, 7, 8) else (w, h)
        except Exception:
            return None
    return None


def is_single_channel(path: Union[str, Path]) -> bool:
    """True when the FILE on disk holds one channel (a black-and-white photo).

    The reader promotes such frames to three channels so the pipeline can work on
    them (see `_promote_grey`), which means nothing downstream can tell afterwards
    what the source actually was. The writer needs to know: a greyscale telescope
    sub written back as RGB is three times the size it arrived as, and hundreds of
    subs is real disk. Reads the header only, never the pixels.

    False for RAW (always debayered to colour), and False on any doubt -- writing
    colour is the safe direction to be wrong in.
    """
    p = str(path)
    if Path(p).suffix.lower() in RAW_EXTS:
        return False
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.mode in ("L", "I", "I;16", "I;16B", "I;16L", "1", "F", "LA")
    except Exception:
        pass
    if Path(p).suffix.lower() in _TIFF_EXTS:
        try:
            import tifffile
            with tifffile.TiffFile(p) as tf:
                return int(tf.pages[0].samplesperpixel) == 1
        except Exception:
            return False
    return False


def exif_tags(path: Union[str, Path]) -> dict:
    """Every readable EXIF tag for a photograph, by name, RAW files included.

    Returns a plain dict like {"Make": "Panasonic", "Model": "DC-S1RM2",
    "LensModel": "VILTROX AF 16mm F1.8 L", "FNumber": 2.0, ...}, merging the
    file's main tag block with its Exif sub-block so callers do not have to know
    which one a given tag lives in. Empty dict if nothing is readable. Never
    raises.

    WHY THIS EXISTS. PIL cannot open a RAW file at all, so anything that reached
    for EXIF by opening the photograph got nothing from a RAW: the Star Log's
    Camera Info block read "Unknown" for every field, and the anonymous run
    summary reported no camera on 81% of RAW runs and no lens on 96% of them
    (measured across 128 real runs, 2026-08-27). A third of all runs contributed
    nothing to the community gear stats, which is most of the reason half our
    photographers show as Unknown there.

    The tags were never missing. Every RAW carries a small preview picture that
    holds the camera's own tags, and this app already reads the capture date that
    way (see `capture_time`). This does the same for the rest of them.

    Reads one small embedded picture, once, not the RAW itself -- so it costs a
    fraction of a decode and is safe to call per run.
    """
    from PIL.ExifTags import TAGS
    p = str(path)
    ex = None
    try:
        if Path(p).suffix.lower() in RAW_EXTS:
            import io as _io
            import rawpy
            from PIL import Image
            with rawpy.imread(p) as raw:
                thumb = raw.extract_thumb()
            if getattr(thumb, "format", None) == rawpy.ThumbFormat.JPEG:
                with Image.open(_io.BytesIO(thumb.data)) as t:
                    ex = t.getexif()
                    sub = _sub_ifd(ex)
                    return _named(ex, sub, TAGS)
            return {}
        from PIL import Image
        with Image.open(p) as im:
            ex = im.getexif()
            if not ex:
                return {}
            # get_ifd() must be called while the file is still open.
            sub = _sub_ifd(ex)
            return _named(ex, sub, TAGS)
    except Exception:
        return {}


def _sub_ifd(ex):
    """The Exif sub-block (lens, exposure, ISO live here, not in the main one)."""
    try:
        return ex.get_ifd(0x8769) or {}
    except Exception:
        return {}


def _named(ex, sub, TAGS):
    """Merge the two tag blocks into one dict keyed by human tag name. The sub
    block wins on a clash: it holds the photograph-specific values."""
    out = {}
    for block in (ex, sub):
        for k, v in (block or {}).items():
            out[TAGS.get(k, k)] = v
    return out


def capture_time(path: Union[str, Path]):
    """Return the moment the SHUTTER FIRED (EXIF DateTimeOriginal) as a datetime,
    or None. RAW reads it from the embedded preview (PIL cannot open a RAW
    directly); everything else reads it straight. Never raises.

    ONLY THE SHUTTER TIME COUNTS, and it used to fall back to the plain file date
    (EXIF DateTime) when there was no shutter time. That fallback is not a
    capture time at all -- an editor stamps it when it EXPORTS the file -- and it
    silently reordered a real sequence (2026-08-30).

    Cheryl's frames came out of Camera Raw, which finished exporting 004 two
    seconds before 003. Her originals still carry the true shutter times, a
    minute apart and in order, but cleaning to 16-bit TIFF drops the EXIF block,
    so the cleaned frames had only the export stamps left. The timelapse
    faithfully sorted by those and played 001, 002, 004, 003 -- a visible
    hesitation early in the video, which is how Bruce found it.

    The callers (this is used ONLY for ordering, never for display) treat a None
    as "no reliable time" and keep the filename order, which is right far more
    often than an export stamp is. The one thing filename order cannot handle is
    a camera file-number rollover mid-shoot or two cards merged, and those frames
    do carry real shutter times, so they are still put right.
    """
    from datetime import datetime
    p = str(path)
    ext = Path(p).suffix.lower()
    raw_s = None
    try:
        if ext in RAW_EXTS:
            import io as _io
            import rawpy
            from PIL import Image
            with rawpy.imread(p) as raw:
                thumb = raw.extract_thumb()
            if getattr(thumb, "format", None) == rawpy.ThumbFormat.JPEG:
                ex = Image.open(_io.BytesIO(thumb.data)).getexif()
                try:
                    sub = ex.get_ifd(0x8769)
                except Exception:
                    sub = {}
                # DateTimeOriginal ONLY -- see above. Looked for in the EXIF
                # sub-block where a camera puts it, and at the top level,
                # where our own 16-bit TIFF writer has to put it (tifffile
                # cannot build a sub-block).
                raw_s = sub.get(0x9003) or ex.get(0x9003)
        else:
            from PIL import Image
            with Image.open(p) as im:
                ex = im.getexif()
                try:
                    sub = ex.get_ifd(0x8769)
                except Exception:
                    sub = {}
                # DateTimeOriginal ONLY -- see above. Looked for in the EXIF
                # sub-block where a camera puts it, and at the top level,
                # where our own 16-bit TIFF writer has to put it (tifffile
                # cannot build a sub-block).
                raw_s = sub.get(0x9003) or ex.get(0x9003)
    except Exception:
        return None
    if not raw_s:
        return None
    try:
        return datetime.strptime(str(raw_s).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


# Pillow modes that carry more than 8 bits per sample. Everything else
# ('L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', '1', ...) is 8-bit. 'I' (32-bit
# int) and 'F' (32-bit float) are bucketed as 16 because the pipeline only
# distinguishes two storage depths: uint8 vs uint16.
_PIL_16BIT_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def image_bitdepth(path: Union[str, Path]) -> Optional[int]:
    """Return the storage bit depth bucket for an image: 8 or 16 (or None if it
    can't be determined). Header-only where possible, so it stays cheap on a
    folder of full-resolution frames.

    The pipeline only cares about two buckets, because the worker loads frames
    with IMREAD_UNCHANGED and gets either a uint8 or a uint16 array:
      * RAW is always 16 (debayered to 16-bit by rawpy).
      * TIFF is read from the file's sample depth via tifffile -- Pillow has no
        16-bit RGB mode and mis-reports a 16-bit RGB TIFF as 8-bit, so it can't
        be trusted for the format that matters most here.
      * JPEG / PNG are decided by the Pillow mode (JPEG is always 8; a 16-bit
        PNG reports an I-family mode).
    Anything deeper than 8-bit reports 16. Mirrors the reader's format coverage
    so the GUI's pre-flight scan and the worker agree on a frame's depth.
    """
    p = str(path)
    ext = Path(p).suffix.lower()

    if ext in RAW_EXTS:
        return 16

    if ext in _TIFF_EXTS:
        try:
            import tifffile
            with tifffile.TiffFile(p) as tf:
                dt = tf.pages[0].dtype
            if dt is not None:
                return 16 if dt.itemsize >= 2 else 8
        except Exception:
            pass  # fall through to Pillow as a last resort

    try:
        from PIL import Image
        with Image.open(p) as im:
            mode = im.mode
        return 16 if mode in _PIL_16BIT_MODES else 8
    except Exception:
        return None


def robust_imwrite(path: Union[str, Path], image: np.ndarray,
                   params: Optional[list] = None) -> bool:
    """Drop-in cv2.imwrite replacement that handles non-ASCII paths.

    cv2.imwrite on Windows uses ANSI file APIs and fails to write files
    whose path contains non-ASCII characters (same root cause as the
    cv2.imread Unicode-path bug). Pillow uses Python's normal file APIs
    which handle Unicode correctly on every platform.

    Tries cv2 first (fast path), falls back to Pillow on failure.
    Accepts BGR / BGRA / grayscale numpy arrays — same convention as
    cv2.imwrite. Returns True on success, False on failure.

    `params` is the same optional list cv2.imwrite takes, e.g.
    [cv2.IMWRITE_JPEG_QUALITY, 95]. It is ALSO honoured on the Pillow
    fallback, which matters more than it looks: Pillow defaults JPEG to
    quality 75, so without this a picture saved on a non-ASCII path -- the
    exact case this function exists for -- would come out visibly worse than
    the same picture saved next door. Reported by a user whose folder had
    Scandinavian letters in it (2026-08-23); he could not export a star trail
    at all, and the naive fix would have quietly given him a worse one.
    """
    p = str(path)

    prev = _silence_cv2_logs()
    try:
        try:
            if cv2.imwrite(p, image, params) if params else cv2.imwrite(p, image):
                return True
        except Exception:
            pass

        try:
            from PIL import Image
            arr = image
            if arr.ndim == 2:
                # Grayscale (uint8 or uint16)
                if arr.dtype == np.uint16:
                    im = Image.fromarray(arr, mode="I;16")
                else:
                    im = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3:
                if arr.shape[2] == 3:
                    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(rgb)
                elif arr.shape[2] == 4:
                    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                    im = Image.fromarray(rgba)
                else:
                    return False
            else:
                return False
            # Carry the encoder settings across to Pillow. Only quality is
            # worth translating: it is the one that changes what the user sees.
            kw = {}
            if params:
                for i in range(0, len(params) - 1, 2):
                    if params[i] == cv2.IMWRITE_JPEG_QUALITY:
                        kw["quality"] = int(params[i + 1])
                    elif params[i] == cv2.IMWRITE_PNG_COMPRESSION:
                        kw["compress_level"] = int(params[i + 1])
            im.save(p, **kw)
            return True
        except Exception:
            return False
    finally:
        _restore_cv2_logs(prev)
