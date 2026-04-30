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


_TIFF_EXTS = {".tif", ".tiff"}


def _silence_cv2_logs():
    try:
        from cv2.utils import logging as _cvlog
        prev = _cvlog.getLogLevel()
        _cvlog.setLogLevel(_cvlog.LOG_LEVEL_SILENT)
        return prev
    except Exception:
        return None


def _restore_cv2_logs(prev):
    if prev is None:
        return
    try:
        from cv2.utils import logging as _cvlog
        _cvlog.setLogLevel(prev)
    except Exception:
        pass


def _try_cv2(path: str, flags: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        img = cv2.imread(path, flags)
        if img is None:
            return None, "returned no image (unsupported format or unreadable bytes)"
        return img, None
    except Exception as e:
        return None, f"raised {type(e).__name__}: {e}"


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

    if arr.ndim == 2:
        return arr, None
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return arr, None

    if flags == cv2.IMREAD_GRAYSCALE:
        code = cv2.COLOR_RGBA2GRAY if arr.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(arr, code), None
    if flags == cv2.IMREAD_COLOR:
        code = cv2.COLOR_RGBA2BGR if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR
        return cv2.cvtColor(arr, code), None
    code = cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR
    return cv2.cvtColor(arr, code), None


def robust_imread_diag(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_UNCHANGED,
    *,
    _retry_delays: Tuple[float, ...] = (1.0, 3.0),
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read with fallbacks. Returns (img_or_None, diagnosis_or_None).

    Order of attempts:
      1. cv2.imread (fast path, covers ~95% of files)
      2. tifffile.imread for .tif / .tiff (BigTIFF, unusual compressions,
         camera-export variants OpenCV can't decode)
      3. After both fail, sleep and retry. Default delays are 1s then 3s,
         so a brief external-drive hiccup or a USB drive waking from sleep
         gets up to ~4 seconds total to recover before we surface anything
         to the user. The retry success case is silent — the load loop
         keeps going with a small pause and the user never sees a modal.

    Tests pass `_retry_delays=()` to skip the waits entirely.

    On success: diagnosis is None.
    On failure: diagnosis is a multi-line string with what each reader said.
    """
    p = str(path)
    is_tiff = Path(p).suffix.lower() in _TIFF_EXTS

    prev = _silence_cv2_logs()
    try:
        attempts = []

        img, why = _try_cv2(p, flags)
        if img is not None:
            return img, None
        attempts.append(("OpenCV", why))

        if is_tiff:
            img, why = _try_tifffile(p, flags)
            if img is not None:
                return img, None
            attempts.append(("tifffile", why))

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

        diag = "\n".join(f"    {label}: {why}" for label, why in attempts)
        return None, diag
    finally:
        _restore_cv2_logs(prev)


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
