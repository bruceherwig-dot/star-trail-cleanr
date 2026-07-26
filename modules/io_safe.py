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


def robust_imread_diag(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_UNCHANGED,
    *,
    _retry_delays: Tuple[float, ...] = (1.0, 3.0),
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read with fallbacks, then apply EXIF orientation centrally so every
    backend/flag returns upright pixels. RAW files are oriented by rawpy already,
    so they are left untouched here."""
    img, diag = _imread_diag_inner(path, flags, _retry_delays=_retry_delays)
    if img is not None and Path(str(path)).suffix.lower() not in RAW_EXTS:
        img = _apply_orientation(path, img)
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


def capture_time(path: Union[str, Path]):
    """Return the photo's capture time (EXIF DateTimeOriginal, else DateTime) as
    a datetime, or None if unavailable. Used to order frames by true shooting
    order, so a camera file-number rollover or a two-card merge can't scramble
    the sequence. RAW reads the time from its embedded preview (PIL can't open a
    RAW directly); everything else reads it straight. Never raises.
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
                raw_s = sub.get(0x9003) or ex.get(0x0132)
        else:
            from PIL import Image
            with Image.open(p) as im:
                ex = im.getexif()
                try:
                    sub = ex.get_ifd(0x8769)
                except Exception:
                    sub = {}
                raw_s = sub.get(0x9003) or ex.get(0x0132)
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


def robust_imwrite(path: Union[str, Path], image: np.ndarray) -> bool:
    """Drop-in cv2.imwrite replacement that handles non-ASCII paths.

    cv2.imwrite on Windows uses ANSI file APIs and fails to write files
    whose path contains non-ASCII characters (same root cause as the
    cv2.imread Unicode-path bug). Pillow uses Python's normal file APIs
    which handle Unicode correctly on every platform.

    Tries cv2 first (fast path), falls back to Pillow on failure.
    Accepts BGR / BGRA / grayscale numpy arrays — same convention as
    cv2.imwrite. Returns True on success, False on failure.
    """
    p = str(path)

    prev = _silence_cv2_logs()
    try:
        try:
            if cv2.imwrite(p, image):
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
            im.save(p)
            return True
        except Exception:
            return False
    finally:
        _restore_cv2_logs(prev)
