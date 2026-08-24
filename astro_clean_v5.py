#!/usr/bin/env python3
"""The cleaning engine — finds the trails in a batch of frames and paints them out.

This is where the actual image work happens. It is a COMMAND-LINE PROGRAM, not a
library: the desktop app (star_trail_cleanr.py) runs it as a separate process,
once per batch of up to 20 frames, and reads its printed output to drive the
progress bar. Nothing here draws any interface.

WHAT IT DOES TO ONE BATCH
  1. Load the batch, plus one neighbour frame on each side (the repair needs the
     frames either side to borrow clean pixels from).
  2. Detect trails: a YOLO segmentation model looks at each frame in 640x640
     tiles, and the results are merged into one mask per frame. Anything below
     the horizon is excluded when the user has painted a foreground mask.
  3. Repair each trail with Star Bridge: take the same patch of sky from the
     frame before and the frame after, morph between them, and lay that over the
     trail. The stars underneath survive because they really were there in the
     neighbouring frames, a few pixels along.
  4. Write a cleaned copy of every frame in the batch. Originals are never
     touched.

WHY BATCHES OF 20
The repair assumes the sky has barely moved between a frame and its neighbours.
Over more than a few minutes the stars rotate far enough that borrowed pixels no
longer line up, so the app cuts the sequence into batches and runs this engine
once per batch.

THE CONTRACT WITH THE APP (both sides must agree, or users get nothing)
  - `--frame-manifest` is the authoritative, de-duplicated, capture-time-ordered
    list of frames. When it is given it is used VERBATIM; this program does not
    re-derive the order. `--start` and `--batch` index into that list.
  - `--expected-width/height/bitdepth` describe the frames the app decided to
    keep. Anything else in the folder is filtered out here.
  - Progress is reported by printing lines the app parses. Changing that output
    format changes the progress bar.
  - Exit non-zero on failure, with a message that says what a person should do
    about it -- this text can end up in front of a user.

RUN IT BY HAND (useful for debugging one batch):
    python3 astro_clean_v5.py <folder> -o <out> --model assets/best.pt \
        --start 0 --batch 20

See ARCHITECTURE.md for how this fits the rest, and modules/detect_trails.py and
modules/repair.py for the two halves of the actual work.
"""
import sys, os
# Apple Silicon: torchvision::nms is not implemented for the MPS (GPU)
# device in the PyTorch version we ship, so YOLO warmup crashes during
# inference for every Apple Silicon Mac user that lets the model run on
# MPS. PYTORCH_ENABLE_MPS_FALLBACK=1 tells PyTorch to silently use the
# CPU for ops that aren't implemented on MPS. Negligible perf hit on
# small ops like NMS, fixes the crash. Must be set BEFORE any torch
# import (including those pulled in by ultralytics / sahi).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The detection engine (ultralytics, pulled in by sahi) monkey-patches PIL's
# image opener so the FIRST time any image fails to open it tries to pip-INSTALL
# a missing add-on ("pi-heif") at RUNTIME. In our frozen app that install runs
# the app itself as the installer, which relaunches/hangs it -- a zombie that
# won't quit and blocks restart. It only ever fired on RAW, because only RAW
# makes PIL's open fail. Forbid the engine from auto-downloading anything: the
# failed open then raises cleanly (we catch it) instead of hanging, and any
# future "auto-install" surprise is closed off too. Must be set BEFORE
# ultralytics is imported -- its AUTOINSTALL flag is read at import time.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
# matplotlib (pulled in transitively by ultralytics) scans the OS font set when
# it loads and logs an ERROR for any installed Type1/AFM font with a
# non-standard header — e.g. old Adobe fonts on some Macs ("Value error parsing
# header in AFM"). We never draw with matplotlib, so that complaint is pure
# noise, but Sentry's default logging integration promotes any ERROR-level log
# into a reported event, producing false crash reports. Silencing the
# matplotlib logger to CRITICAL drops the font-scan chatter at the source (it
# never reaches stderr or Sentry). Must run before ultralytics imports.
import logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
"""
astro_clean_v5.py — trail detection and repair worker

ROLE IN THE APP
---------------
This script runs as a subprocess, not directly. Star Trail CleanR (the GUI in
star_trail_cleanr.py) spawns this file via --cleanr-worker for each 20-frame batch.
It reads raw frames from disk, finds every airplane and satellite trail in each frame,
removes them, writes cleaned replacements, and logs per-frame results to a JSONL file.
The GUI monitors stdout for progress updates while this worker runs.

THE TWO STEPS
-------------
Step 1 — DETECT (modules/detect_pipeline.py, which internally uses
         modules/trail_grouper.py for grouping/polygon fitting; a few helpers
         come from modules/detect_trails.py)
  The YOLO model (Trail DetectoR) was trained to recognize the pixel shape of a trail
  in a single astrophotography frame. Because our frames are large (often 6000x4000+)
  and the model works on 640x640 tiles, we use SAHI (Slicing Aided Hyper Inference)
  to divide each frame into overlapping tiles, run the model on each tile, then stitch
  all results back to full-frame coordinates. The grouper fuses tile-boundary fragments
  that belong to the same physical trail into one clean detection polygon.

  The foreground/sky mask is supplied to the worker as a file (--foreground-mask):
  white = foreground to exclude. The worker loads it and inverts it to a sky mask,
  which is then applied to each detection mask so that foreground false positives are
  discarded before they reach the repair step. (The mask itself is built upstream in
  the GUI, not in this file.)

  The static FP suppressor runs after detection. It looks at each detected region and
  checks whether the same pixel region is detected in neighboring frames too. If a
  "trail" shows up in the same location across many frames, it's a static object (a
  bright edge, a roofline, an illuminated antenna) — not a moving trail. Those
  detections are removed before repair.

Step 2 — REPAIR (modules/repair.py)
  For each detected trail, Star Bridge synthesizes what the frame would look like
  without it. It tracks bright star features from the frame before (N-1) to the frame
  after (N+1) using Lucas-Kanade sparse optical flow, measures how far the stars moved
  between those two neighbors, then shifts each neighbor by half that motion and
  averages them. The result is a synthetic version of frame N with the stars in the
  right position and the trail gone. Only the masked trail pixels are replaced — the
  rest of the frame is untouched.

  First/last frame fallback: only one neighbor is available, so that neighbor is
  used directly (no blending).

  Tracking failure fallback: if too few stars are found or their displacements are
  implausible, the trail pixels are painted with the local sky color plus matched
  grain (the "crayon" fill), sampled from the surrounding sky. Only when there is too
  little nearby sky to sample does it fall back to pure black. Either way the fill is
  invisible-or-near-invisible in a lighten-max composite because the real star pixel
  from another frame wins.

KEY ASSUMPTIONS
---------------
- Fixed tripod: all users shoot on non-tracking mounts. The foreground is
  pixel-perfect static in every frame. Stars move; everything else stays still.
- Trails span 2-20 consecutive frames for both airplanes and satellites.
- Source files are predominantly full-resolution TIFFs (not JPEGs). Performance
  decisions should treat TIFF as the primary case.
- Batch size is capped at 20 frames. Beyond that, accumulated star rotation
  between the first and last frame of a batch becomes large enough to degrade
  the Star Bridge repair quality.
- The GUI passes one extra frame before and after each batch (overlap frames)
  so repair can stitch trails that cross batch boundaries.
"""

import argparse
import os
import shutil
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List

from modules.detect_trails import (
    load_model, apply_sky_mask, filter_small_components
)
from modules import detect_pipeline as dp
from modules.trail_grouper import detection_props
from modules.repair import repair_frame
from modules.run_logger import RunLogger

# Star Bridge repair combine mode. "single_shift" is the SHIPPED default
# (v2.49-beta, 2026-06-10): borrow one colour-matched neighbor shifted into
# place -- no averaging, so faint stars keep full brightness instead of washing
# out. Confirmed better than the old "average" (blend both warped neighbors) by
# Bruce's side-by-side review across multiple real datasets. "average" remains
# available via the STC_REPAIR_COMBINE env var for comparison runs.
REPAIR_COMBINE = os.environ.get("STC_REPAIR_COMBINE", "single_shift")


def _raw_labeled_from_state(state, h, w):
    """Build the raw-SAHI labeled mask (pixel value = prediction index 1..N) from
    a new-pipeline state, matching the (final_mask, raw_labeled) contract that the
    legacy detect_frame_polygon returns. Used so MaskViewR's yellow raw layer and
    the run log keep working when the new pipeline is the detector."""
    raw = np.zeros((h, w), dtype=np.uint8)
    for ri, pred in enumerate(state.raw_detections or []):
        label_id = min(ri + 1, 255)
        try:
            bm = pred.mask.bool_mask
        except AttributeError:
            continue
        if not isinstance(bm, np.ndarray):
            continue
        bm_bool = bm.astype(bool) if bm.dtype != np.bool_ else bm
        if bm_bool.shape == (h, w):
            raw[bm_bool] = label_id
        else:
            rz = cv2.resize(bm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            raw[rz > 0] = label_id
    return raw
from modules.io_safe import robust_imread, robust_imread_diag, robust_imwrite
from modules.frame_list import dedupe_frames, natural_key, IMAGE_EXTS, RAW_EXTS
from modules.workspace import WORKSPACE_NAME   # the run-artifact folder (in the output folder)


def _source_metadata(path):
    """Read (exif_bytes, icc_profile, dpi) from a frame's ORIGINAL source file.

    JPEG/TIFF are read straight from the file. RAW files cannot be opened by PIL
    at all -- and merely asking PIL to try would trip the detection engine's
    patched opener into a runtime pi-heif auto-install that hangs the app (see
    the YOLO_AUTOINSTALL note at the top of this file). So for RAW we read the
    metadata from the JPEG preview the RAW file embeds, which carries the capture
    date and shoot settings (verified on Canon CR3 and Fuji RAF). A RAW whose
    preview is a bitmap, or any unreadable file, yields (None, None, None) so the
    frame still cleans with no date rather than failing. Never raises.
    """
    try:
        from PIL import Image as _PILImage
        ext = os.path.splitext(str(path))[1].lower()
        if ext in RAW_EXTS:
            import rawpy
            import io as _io
            with rawpy.imread(str(path)) as _r:
                _thumb = _r.extract_thumb()
            if not str(_thumb.format).endswith("JPEG"):
                return (None, None, None)
            _src = _io.BytesIO(_thumb.data)
        else:
            _src = str(path)
        with _PILImage.open(_src) as _im:
            try:
                _e = _im.getexif()
                _exif = _e.tobytes() if _e else None
            except Exception:
                _exif = _im.info.get("exif")
            return (_exif, _im.info.get("icc_profile"), _im.info.get("dpi"))
    except Exception:
        return (None, None, None)


# ── Black-box recorder ──────────────────────────────────────────────────────
# A crash breadcrumb the worker forces to disk after every frame at every stage.
# If the OS hard-kills the whole app (e.g. a macOS low-memory force-quit, which
# leaves no Python exception and no Sentry trace), the last line on disk still
# shows exactly which frame and stage we died on, and how much memory was left
# at that moment. The GUI rotates this file each run and folds an abnormal
# previous run's tail into the Star Log. Every write is wrapped so logging can
# never itself break a run.
_CRUMB_PATH = os.path.join(
    os.path.expanduser("~"), ".star_trail_cleanr", "last_run_progress.log")


def _crumb(stage, idx=None, total=None, name="", extra=""):
    """Append one flushed+fsync'd breadcrumb line. Never raises."""
    try:
        pos = f"{idx}/{total}" if idx is not None else ""
        mem = ""
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / (1024 ** 3)
            avail = psutil.virtual_memory().available / (1024 ** 3)
            mem = f"| proc {rss:.2f}G | free {avail:.2f}G "
        except Exception:
            pass
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {stage} {pos} | "
                f"{name} {mem}{extra}".rstrip() + "\n")
        os.makedirs(os.path.dirname(_CRUMB_PATH), exist_ok=True)
        with open(_CRUMB_PATH, "a") as _f:
            _f.write(line)
            _f.flush()
            os.fsync(_f.fileno())
    except Exception:
        pass


def _init_worker_sentry():
    """Initialize Sentry in the worker subprocess if the GUI passed a DSN.

    The GUI sets STC_SENTRY_DSN in the worker's environment ONLY when the
    user has opted into crash reporting AND a DSN was baked into the build.
    If the env var is missing or empty, this is a no-op — Sentry stays
    inactive and the worker has no reporting hookup. This preserves the
    opt-in privacy contract end-to-end.

    Worker-side Sentry catches unhandled exceptions inside the processing
    loop (detect, repair, file I/O). Crashes that die before this init runs
    are still reported by the GUI's stderr-capture safety net.
    """
    dsn = os.environ.get("STC_SENTRY_DSN", "")
    if not dsn:
        return
    try:
        import sentry_sdk
        version = "?"
        try:
            base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base, "version.txt")) as vf:
                version = vf.read().strip()
        except Exception:
            pass
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0,
            send_default_pii=False,
            release=f"star-trail-cleanr@{version}",
        )
        sentry_sdk.set_tag("component", "worker")
        # Pillow logs its refusals via logging.error, and Sentry's logging
        # integration turns those into crash reports. io_safe deliberately lets
        # readers fail in turn (OpenCV, then tifffile, then Pillow), so a file
        # only one of them can open is normal and must not page us. A file NO
        # reader can open still reports, via the worker's own error path with
        # the full per-reader diagnosis.
        from sentry_sdk.integrations.logging import ignore_logger
        for _lg in ("PIL", "PIL.TiffImagePlugin", "PIL.Image"):
            ignore_logger(_lg)
    except Exception:
        pass


_init_worker_sentry()


def _capture_unreadable_file_to_sentry(fp, diag):
    """Fire a Sentry warning event when a file gets skipped because no reader
    could decode it. Best-effort — silently no-ops if Sentry isn't initialized.

    Fingerprint groups every skip into one Sentry issue so a tester with many
    bad files doesn't flood the inbox; individual events still carry the
    per-file path, size, extension, and reader diagnoses for triage.
    """
    try:
        import sentry_sdk
        import platform as _plat
        size_bytes = -1
        mtime = None
        try:
            st = fp.stat()
            size_bytes = st.st_size
            mtime = st.st_mtime
        except Exception:
            pass
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("event_type", "worker_unreadable_file")
            scope.set_tag("file_extension", fp.suffix.lower() or "(none)")
            scope.set_tag("os", _plat.system())
            scope.set_tag("os_release", _plat.release())
            scope.set_extra("file_path", str(fp))
            scope.set_extra("file_name", fp.name)
            scope.set_extra("file_size_bytes", size_bytes)
            scope.set_extra("file_mtime", mtime)
            scope.set_extra("reader_diagnosis", diag or "(none)")
            scope.fingerprint = ["worker_unreadable_file"]
            sentry_sdk.capture_message(
                "Worker skipped unreadable file",
                level="warning",
            )
    except Exception:
        pass


def _prompt_gui_for_bad_file(fp, diag):
    """Ask the GUI what to do with an unreadable file. Blocks reading stdin
    until the GUI writes back a single-line response.

    Emits a `STC_BAD_FILE_PROMPT:` sentinel with a JSON payload (path, name,
    diagnosis) on stdout, then reads one line from stdin. Expected response
    is "CONTINUE" (skip this frame) or "STOP" (graceful run end). If stdin is
    closed or anything goes wrong, default to STOP — safer than guessing the
    user's intent.
    """
    import json
    payload = {
        "path": str(fp),
        "name": fp.name,
        "diagnosis": diag or "",
    }
    print(f"STC_BAD_FILE_PROMPT: {json.dumps(payload)}", flush=True)
    try:
        line = sys.stdin.readline()
    except Exception:
        return "STOP"
    if not line:
        return "STOP"
    response = line.strip().upper()
    return "CONTINUE" if response == "CONTINUE" else "STOP"


def _filter_by_resolution(files: List[Path],
                          expected_width: int = None,
                          expected_height: int = None) -> List[Path]:
    """Keep only files matching the expected (or dominant) resolution.
    Uses PIL header-only reads, no full image decode. Silent (no per-file output).

    JPG+TIFF duplicate removal is NO LONGER done here. It now happens once on
    the full folder list (via dedupe_jpg_tiff) before any index slicing, in both
    load_with_neighbors / load_frame_files and the GUI, so the GUI's batch plan
    and the worker's slicing stay aligned. Removing it per-slice was the cause of
    the misaligned batches and the sub-3 final-batch crash.
    """
    if len(files) <= 1:
        return files

    # Use the shared size helper, NOT PIL directly: PIL can't open a camera RAW
    # (CR2/NEF/ARW/...), so a PIL-only check returns None for every RAW frame and
    # drops the entire batch as "different resolution". image_size handles RAW
    # (rawpy), TIFF (tifffile fallback), and JPG/PNG (PIL). Returns (w, h)/None.
    from modules.io_safe import image_size as _hdr_size

    if expected_width and expected_height:
        target = (expected_width, expected_height)
    else:
        sample = files[:min(10, len(files))]
        sizes = [s for s in (_hdr_size(fp) for fp in sample) if s is not None]
        if not sizes:
            return files
        from collections import Counter
        target = Counter(sizes).most_common(1)[0][0]

    filtered = [fp for fp in files if _hdr_size(fp) == target]
    skipped = len(files) - len(filtered)
    if skipped:
        word = "frame" if skipped == 1 else "frames"
        print(f"  Skipped {skipped} {word} with different resolution")
    return filtered


def load_frame_files(frame_dir: Path, start: int, batch: int,
                     expected_width: int = None,
                     expected_height: int = None,
                     prefer_raw: bool = True) -> List[Path]:
    """Return the sorted list of image file paths for one batch (no neighbors).

    Lists every image in `frame_dir`, drops twin copies of the same frame (a
    JPG/TIFF/RAW pair of one shot) using the same RAW-vs-JPG/TIFF preference the
    GUI uses, THEN slices `[start : start+batch]`. Deduping before slicing is what
    keeps the worker's frame numbering aligned with the GUI's batch plan. Finally
    drops any frame whose resolution doesn't match the rest. `batch <= 0` means
    "take everything from start to the end". Inputs: `start`/`batch` are frame
    indices into the deduped, sorted list; `expected_width`/`expected_height`
    pin the target resolution (skips auto-detection); `prefer_raw` picks RAW over
    JPG/TIFF when both exist for the same shot.

    Note: this helper is the no-neighbor variant; the actual run uses
    load_with_neighbors so repair can see one frame on each side.
    """
    files = sorted((p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                   key=natural_key)
    # Drop twins (JPG/TIFF/RAW of the same frame) on the FULL list before
    # slicing, so frame indices match the GUI's (which dedups the same way,
    # with the same RAW-vs-JPG/TIFF preference, before planning batches).
    files = dedupe_frames(files, prefer_raw=prefer_raw)
    sliced = files[start:start + batch] if batch > 0 else files[start:]
    return _filter_by_resolution(sliced, expected_width, expected_height)


def load_with_neighbors(frame_dir: Path, start: int, batch: int,
                        expected_width: int = None,
                        expected_height: int = None,
                        prefer_raw: bool = True,
                        ordered_files=None):
    """Load batch frames plus one neighbor on each side for repair context.

    Returns (all_files, core_start, core_end) where all_files includes
    up to one extra frame before and after, and core_start/core_end
    mark the indices of the actual batch frames within all_files.

    When `ordered_files` is given (the GUI's per-run frame manifest), it is the
    canonical de-duped, capture-time-ordered list and is used verbatim, so the
    worker never re-derives order and the two sides cannot disagree.
    """
    if ordered_files is not None:
        all_sorted = [Path(p) for p in ordered_files]
    else:
        all_sorted = sorted((p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                            key=natural_key)
        # Drop twins on the FULL list before slicing/indexing, so frame numbers
        # match the GUI's batch plan (which dedups identically, same preference, up front).
        all_sorted = dedupe_frames(all_sorted, prefer_raw=prefer_raw)
    total = len(all_sorted)

    end = start + batch if batch > 0 else total
    end = min(end, total)

    # Extend by one frame on each side if available
    ext_start = max(0, start - 1)
    ext_end = min(total, end + 1)

    sliced = all_sorted[ext_start:ext_end]
    n_before_filter = len(sliced)
    sliced = _filter_by_resolution(sliced, expected_width, expected_height)
    # How many the resolution filter threw away, so the caller can say WHY a
    # batch came up empty instead of just reporting the count.
    load_with_neighbors.last_dropped = n_before_filter - len(sliced)
    load_with_neighbors.last_total = total

    core_start = start - ext_start
    core_end = core_start + (end - start)
    core_end = min(core_end, len(sliced))

    return sliced, core_start, core_end


def _resolve_target_dtype(frames, expected_bitdepth):
    """Pick the one dtype every frame in a batch should share.

    A folder can legitimately hold both 8-bit and 16-bit frames (e.g. most
    frames have a 16-bit TIFF/RAW twin we keep, but a few tail frames exist only
    as an 8-bit JPG). The hot-pixel, sky-mask and Star Bridge steps all assume a
    single dtype per batch, so the loader brings every frame to one target.

    Prefers the sequence-wide majority the GUI computed and passed as
    `expected_bitdepth` (8 or 16) so every batch normalizes to the same depth and
    the output stays consistent. When that isn't given (worker run standalone),
    falls back to this batch's own majority, with ties going to 16-bit so frames
    are never needlessly down-converted.
    """
    if expected_bitdepth == 16:
        return np.uint16
    if expected_bitdepth == 8:
        return np.uint8
    n16 = sum(1 for f in frames if f.dtype == np.uint16)
    n8 = sum(1 for f in frames if f.dtype == np.uint8)
    return np.uint16 if n16 >= n8 else np.uint8


def _match_bitdepth(img, target_dtype):
    """Return `img` converted to `target_dtype` (np.uint8 or np.uint16), or the
    same array unchanged when it already matches (the caller relies on identity
    to count how many frames were actually converted).

    8-bit -> 16-bit scales by 257 to fill the range (255 -> 65535); 16-bit ->
    8-bit drops the low byte. These are the exact conversions the save path
    already uses, so a normalized frame and a frame natively at that depth look
    identical downstream.
    """
    if img.dtype == target_dtype:
        return img
    if target_dtype == np.uint16:
        return img.astype(np.uint16) * 257
    return (img >> 8).astype(np.uint8)


_CENTROID_MOTION_PX = 20   # centroid offset above which a neighbor is a different object
_BRIGHT_TRAIL_RATIO  = 2.5  # 90th-pct inside brightness / median surrounding; above = real trail


def _is_bright_trail(comp_pixels, img_bgr):
    """Return (is_bright, ratio): True if mask pixels stand out from surroundings.

    Compares the 90th-percentile max-channel brightness of pixels inside the
    component mask against the median brightness of a surrounding ring. Any color
    (red, white, green) triggers the veto as long as the pixels are significantly
    brighter than the local sky background.
    """
    if img_bgr is None:
        return False, 0.0
    ys, xs = np.where(comp_pixels)
    if len(ys) == 0:
        return False, 0.0
    inside = img_bgr[ys, xs].astype(np.float32)
    inside_bright = float(np.percentile(np.max(inside, axis=1), 90))
    m = comp_pixels.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(m, kernel) > 0) & ~comp_pixels
    ry, rx = np.where(ring)
    if len(ry) == 0:
        return False, 0.0
    surrounding = img_bgr[ry, rx].astype(np.float32)
    surround_med = float(np.median(np.max(surrounding, axis=1)))
    if surround_med < 1:
        return False, 0.0
    ratio = round(inside_bright / surround_med, 2)
    return ratio >= _BRIGHT_TRAIL_RATIO, ratio


def _suppress_static_fps(masks_all, core_start, core_end,
                         iou_threshold=0.70, min_matches=1,
                         raw_masks_all=None, frames_all=None, debug_out=None,
                         timing_out=None, removed_out=None):
    """Remove detection components that are static false positives.

    PURPOSE: The AI sometimes detects fixed foreground objects (building edges,
    rooflines, fence posts) as airplane trails because they are long, thin, and
    high-contrast against the sky. These false positives appear at the SAME pixel
    position in every frame. Real trails MOVE between frames, so even a trail that
    spans many frames never sits at the same pixel position twice; a static
    foreground object sits at exactly the same position in every frame. Suppression
    keys on same-position recurrence (high IoU at a near-identical centroid), not on
    how many frames a detection spans.

    HOW IT WORKS: For each detected component in a core frame N, we check the 8
    surrounding frames {N-4, N-3, N-2, N-1, N+1, N+2, N+3, N+4}. This covers any
    pair of frames that could serve as N-2 and N+2 of some center point (max
    distance = 4). The component counts as 1 occurrence (itself); if it also
    matches any 1 neighbor at iou_threshold IoU, the total is 2 and it is
    suppressed. The intermediate frames between the two matching frames do not
    need to fire.

    SPATIAL OPTIMIZATION: Neighbor comparisons use only the component's bounding
    box region instead of the full frame.  All pixels of a component lie within
    their own bbox, so (comp_pixels & nb_hit) is equivalent to
    (comp_pixels[bbox] & nb_hit[bbox]).  For a 6000x4000 frame with a 400x12
    trail bbox, this reduces each comparison from 24 million pixel ops to ~4 800 --
    roughly a 5 000x reduction per check.  Neighbor boolean masks are also
    precomputed once before the outer loop so the (mask > 0) conversion is not
    repeated for every component.

    Each neighbor is checked against BOTH the fitted polygon mask AND the raw SAHI
    hit map (raw_masks_all). A static FP that passes elongation filters in some
    frames but not others still leaves raw SAHI evidence in the skipped frames.
    Using both sources closes the gap.

    WHY IoU: Intersection over Union measures whether two detections are the same
    object. For the same physical object in two frames the IoU is 70-98% because
    the regions are similar in size and position. For a large merged detection that
    merely contains a small unrelated patch from a neighboring frame, the union is
    huge and the IoU is tiny (2-3%) even when the small patch is mostly inside the
    large one. Forward/reverse overlap ratios cannot distinguish this case because
    they use each region's own area as the denominator rather than the union.
    Validated values: Sompting Church rooflines 78-97%, cx=1802 skip-frame FP 73%,
    Green Park real trail (false suppression with prior logic) 2.4%.

    WHY ±4 WINDOW: Any two frames that are the N-2 and N+2 of some center point
    are at most 4 apart. Checking ±4 ensures that any such pair can see each other
    and trigger mutual suppression, even when the frames between them are empty.
    Real trails cannot reach 70% IoU with a static foreground object because they
    are moving -- their pixel position changes every frame.

    TWO VETOES prevent suppression of real trails even when IoU fires:

    Veto 1 -- Bright/distinctive pixels: if the detection pixels are significantly
    brighter than the local sky background (ratio >= _BRIGHT_TRAIL_RATIO), the
    component is a real trail (nav light, strobe, or bright streak) and is kept.
    This catches red, white, green, and any other bright airplane lights. Requires
    frames_all to be passed; skipped otherwise.

    Veto 3 -- Frame edge: if the component's bbox touches any image edge within
    20px, suppression is skipped entirely. Trails go off the page; static
    foreground objects (rooflines, fence posts, building edges) do not. This
    is the same 20px threshold used by the edge rescue in filter_masks_with_props
    and applies the same reasoning: proximity to the frame boundary is strong
    evidence of a real trail, not a fixed object.

    Veto 2 -- Centroid motion: when a neighbor matches at IoU >= iou_threshold,
    the centroid of the neighbor's pixels inside the component bounding box is
    compared to the component centroid. If the offset exceeds _CENTROID_MOTION_PX,
    the neighbor is at a different position and is NOT counted as a static match.
    This prevents two different airplanes on the same flight path from mutually
    suppressing each other.
    """
    # Precompute per-component records for every frame so the inner compare loop
    # can do a fast bbox-overlap check before touching any pixels.  Each record
    # stores the component's bbox, area, cropped boolean pixels, and source tag.
    # This replaces the prior full-frame OR mask: instead of slicing a 6000x4000
    # array for every (query component, neighbor frame) pair, we iterate a small
    # list (~5-15 items) and skip with four integer comparisons when bboxes don't
    # overlap -- zero numpy work for the common case of a real trail with no
    # neighbor at the same location.
    _t0_pre = time.perf_counter()
    _nb_comps = {}  # frame_idx -> list of {x1,y1,x2,y2,area,crop,source}
    for _fi in range(len(masks_all)):
        _m = masks_all[_fi]
        _src = "polygon"
        _use = (_m > 0).astype(np.uint8) if (_m is not None and _m.max() > 0) else None
        if _use is None and raw_masks_all is not None:
            _r = raw_masks_all[_fi] if _fi < len(raw_masks_all) else None
            if _r is not None and _r.max() > 0:
                _use = (_r > 0).astype(np.uint8)
                _src = "raw_sahi"
        if _use is None:
            _nb_comps[_fi] = []
            continue
        _nc, _lbl, _stats, _ = cv2.connectedComponentsWithStats(_use)
        _comps = []
        for _ci in range(1, _nc):
            _cx1 = int(_stats[_ci, cv2.CC_STAT_LEFT])
            _cy1 = int(_stats[_ci, cv2.CC_STAT_TOP])
            _cw  = int(_stats[_ci, cv2.CC_STAT_WIDTH])
            _ch  = int(_stats[_ci, cv2.CC_STAT_HEIGHT])
            _cx2, _cy2 = _cx1 + _cw - 1, _cy1 + _ch - 1
            _ca  = int(_stats[_ci, cv2.CC_STAT_AREA])
            _crop = (_lbl[_cy1:_cy2 + 1, _cx1:_cx2 + 1] == _ci)
            _comps.append((_cx1, _cy1, _cx2, _cy2, _ca, _crop, _src))
        _nb_comps[_fi] = _comps
    if timing_out is not None:
        timing_out["sfp_precompute_s"] = time.perf_counter() - _t0_pre

    # Pass 1: identify every component to suppress using the ORIGINAL unmodified masks.
    # All overlap checks happen before any mask is zeroed, so later frames see the
    # same reference masks as earlier frames. The prior single-pass approach zeroed
    # masks in order, causing the last 2 frames in a batch to lose their reference
    # points (already-zeroed earlier frames), which prevented suppression there.
    suppress_maps  = {}  # frame_idx -> bool array of pixels to zero
    debug_by_frame = {}  # frame_idx -> [suppressed records] for log
    kept_by_veto   = {}  # frame_idx -> [veto records] for log

    _t0_cmp = time.perf_counter()
    for i in range(core_start, core_end):
        mask = masks_all[i]
        if mask.max() == 0:
            continue

        h_mask, w_mask = mask.shape
        to_suppress = np.zeros(mask.shape, dtype=bool)

        for (x1, y1, x2, y2, comp_area, comp_crop, _) in _nb_comps.get(i, []):
            if comp_area == 0:
                continue

            # Centroid from bbox-local crop -- no full-frame array needed.
            _ys_loc, _xs_loc = np.where(comp_crop)
            cx = int(_xs_loc.mean()) + x1
            cy = int(_ys_loc.mean()) + y1

            # Frame-edge veto: trails go off the page -- static foreground
            # objects do not. If the bbox touches any image edge within 20px,
            # this is a trail entering or exiting the frame, not a static FP.
            if y1 <= 19 or y2 >= h_mask - 20 or x1 <= 19 or x2 >= w_mask - 20:
                kept_by_veto.setdefault(i, []).append({
                    "area": comp_area,
                    "cx": cx, "cy": cy,
                    "bbox": [x1, y1, x2, y2],
                    "veto": "frame_edge",
                    "reason": (
                        f"Suppression skipped: bbox touches frame edge within 20px "
                        f"(x1={x1} y1={y1} x2={x2} y2={y2}, frame {w_mask}x{h_mask}). "
                        f"Trails go off the page -- static foreground objects do not."
                    ),
                })
                continue

            matched_neighbors = []
            for ni in [i - 4, i - 3, i - 2, i - 1, i + 1, i + 2, i + 3, i + 4]:
                if ni < 0 or ni >= len(masks_all):
                    continue
                # Accumulate the union of all overlapping neighbor components
                # into a query-bbox-sized boolean array.  A static FP detected
                # as multiple fragments in a neighbor frame still triggers
                # suppression because the fragments are OR'd together before
                # IoU is computed -- same result as the original full-frame OR.
                _union_in_bbox = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=bool)
                _first_src = None
                for (_nx1, _ny1, _nx2, _ny2, _na, _ncrop, _nsrc) in _nb_comps.get(ni, []):
                    # Fast bbox overlap test -- O(1), no numpy.
                    if _nx2 < x1 or _nx1 > x2 or _ny2 < y1 or _ny1 > y2:
                        continue
                    # Overlap region in global coordinates.
                    _ox1, _oy1 = max(x1, _nx1), max(y1, _ny1)
                    _ox2, _oy2 = min(x2, _nx2), min(y2, _ny2)
                    _nb_slice = _ncrop[_oy1 - _ny1:_oy2 - _ny1 + 1,
                                       _ox1 - _nx1:_ox2 - _nx1 + 1]
                    _union_in_bbox[_oy1 - y1:_oy2 - y1 + 1,
                                   _ox1 - x1:_ox2 - x1 + 1] |= _nb_slice
                    if _first_src is None:
                        _first_src = _nsrc
                _union_area = int(_union_in_bbox.sum())
                if _union_area == 0:
                    continue
                _intersection = int((comp_crop & _union_in_bbox).sum())
                if _intersection == 0:
                    continue
                _iou = _intersection / (comp_area + _union_area - _intersection)
                if _iou < iou_threshold:
                    continue
                # Centroid motion veto: union centroid in the query bbox must
                # be close to the query component centroid.
                _ncy_arr, _ncx_arr = np.where(_union_in_bbox)
                if len(_ncy_arr) > 0:
                    _nb_cx_f = float(_ncx_arr.mean()) + x1
                    _nb_cy_f = float(_ncy_arr.mean()) + y1
                    _centroid_dist = float(np.sqrt((cx - _nb_cx_f) ** 2
                                                   + (cy - _nb_cy_f) ** 2))
                    if _centroid_dist > _CENTROID_MOTION_PX:
                        continue
                matched_neighbors.append({
                    "frame_idx": ni,
                    "source": _first_src or "polygon",
                    "iou_pct": round(_iou * 100, 1),
                    "local_nb_area": _union_area,
                })

            if len(matched_neighbors) >= min_matches:
                # Bright-trail veto: if pixels inside the component are
                # significantly brighter than the surrounding sky, it is a
                # real trail (nav light, strobe, or bright streak) -- keep it
                # regardless of the neighbor match count.
                if frames_all is not None and i < len(frames_all):
                    _comp_pixels_full = np.zeros(mask.shape, dtype=bool)
                    _comp_pixels_full[y1:y2 + 1, x1:x2 + 1] = comp_crop
                    is_bright, bright_ratio = _is_bright_trail(_comp_pixels_full, frames_all[i])
                    if is_bright:
                        match_desc = "; ".join(
                            f"frame {m['frame_idx']} via {m['source']} (IoU {m['iou_pct']}%)"
                            for m in matched_neighbors
                        )
                        kept_by_veto.setdefault(i, []).append({
                            "area": comp_area,
                            "cx": cx, "cy": cy,
                            "bbox": [x1, y1, x2, y2],
                            "match_count": len(matched_neighbors),
                            "matched_neighbors": matched_neighbors,
                            "veto": "bright_trail",
                            "bright_ratio": bright_ratio,
                            "reason": (
                                f"Kept despite {len(matched_neighbors)} neighbor match(es) "
                                f"({match_desc}): detection pixels are {bright_ratio}x "
                                f"brighter than surroundings -- real trail "
                                f"(nav light, strobe, or bright streak)."
                            ),
                        })
                        continue  # do not suppress

                to_suppress[y1:y2 + 1, x1:x2 + 1] |= comp_crop
                match_desc = "; ".join(
                    f"frame {m['frame_idx']} via {m['source']} (IoU {m['iou_pct']}%)"
                    for m in matched_neighbors
                )
                reason = (
                    f"Matched {match_desc} at IoU >= {iou_threshold*100:.0f}%. "
                    f"Same pixel position in {len(matched_neighbors)} neighboring "
                    f"frame(s) within ±4 window -- consistent with static foreground "
                    f"object (roofline, fence post, building edge). "
                    f"cx={cx} cy={cy} area={comp_area}px."
                )
                debug_by_frame.setdefault(i, []).append({
                    "area": comp_area,
                    "cx": cx, "cy": cy,
                    "bbox": [x1, y1, x2, y2],
                    "match_count": len(matched_neighbors),
                    "matched_neighbors": matched_neighbors,
                    "reason": reason,
                })

        if to_suppress.any():
            suppress_maps[i] = to_suppress

    if timing_out is not None:
        timing_out["sfp_compare_s"] = time.perf_counter() - _t0_cmp

    # Pass 2: apply all suppressions now that overlap checks are complete.
    # removed_out (when given) receives {frame_index: boolean removed-pixel map}
    # so the caller can apply the SAME verdict to its per-polygon lists -- the
    # repair step and the _polys.json export read those lists, and without this
    # they kept cleaning/exporting detections the suppressor had rejected
    # (the GoPro plume: rejected on 21 straight frames, yet repaired and
    # uploaded to CVAT on all 21 -- todo #94).
    _t0_apply = time.perf_counter()
    suppressed_count = 0
    for i, to_suppress in suppress_maps.items():
        masks_all[i][to_suppress] = 0
        if removed_out is not None:
            removed_out[i] = to_suppress
        n_cc, _ = cv2.connectedComponents(to_suppress.astype(np.uint8))
        suppressed_count += max(0, n_cc - 1)
    if timing_out is not None:
        timing_out["sfp_apply_s"] = time.perf_counter() - _t0_apply

    if debug_out is not None:
        debug_out["suppressed_by_frame"] = debug_by_frame
        debug_out["kept_by_veto_by_frame"] = kept_by_veto

    return suppressed_count


def main():
    """Entry point for one batch. Parses command-line arguments, runs the full
    detect-then-repair pipeline on the requested frame range, and writes cleaned
    frames to the output folder.

    This is the whole job the worker subprocess does. The GUI launches one of
    these per 20-frame batch with the input/output folders, the model path, and
    the batch window (--start / --batch). There are no return values: progress and
    results are printed to stdout (the GUI parses lines like FRAME_TRAIL_COUNT and
    BATCH_TRAIL_COUNT), and the process exits non-zero on fatal errors.

    The flow, top to bottom:
      1. Parse args and set up output folders + (dev-only) the JSONL run logger.
      2. Define the nested output-writing / EXIF helpers (closures over args).
      3. Load the batch frames plus one neighbor on each side, reading EXIF/ICC,
         stripping alpha, applying orientation, evening out any mixed bit depth
         to one target, and bailing on mixed portrait/landscape shapes.
      4. Optional hot-pixel repair on the foreground (only when a mask is given).
      5. Step 1 — detect trails per frame via the SAHI pipeline, then run the
         second-scrub (1b), static-FP suppressor (1c), and edge rescue (1d).
      6. Step 2 — Star Bridge repair each detected trail and write each frame.
      7. Print timing + summary and close the logger.

    Many helpers below are defined INSIDE main() on purpose: they close over
    `args`, `output_dir`, `cleaned_dir`, and the per-run `_stamp` string, so they
    can't be module-level functions without threading all of that through.
    """
    parser = argparse.ArgumentParser(
        description="astro_clean_v5 — YOLO-based airplane trail removal")
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--frame-manifest", default=None,
                        help="Path to a file listing the exact ordered frame "
                             "paths (one per line) the GUI planned. When given, "
                             "the worker uses this order verbatim instead of "
                             "re-listing the folder, so the two stay in lockstep.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--device", default="auto",
                        help="Inference device: auto, cuda, mps, cpu "
                             "(default auto picks cuda > mps > cpu)")
    parser.add_argument("--min-area", type=int, default=1000,
                        help="Min trail component area in pixels (default 1000)")
    parser.add_argument("--foreground-mask", type=str, default=None,
                        help="Path to foreground mask (white=foreground to exclude)")
    parser.add_argument("--skip-boundary", type=int, default=0,
                        help="Skip first/last N frames from output (default 0)")
    parser.add_argument("--hot-pixel-map", type=str, default=None,
                        help="Path to hot pixel map file (load if exists, save if not)")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save detection masks to <output>/STC Extras/masks/")
    parser.add_argument("--save-detections", action="store_true",
                        help="Save ONLY the per-frame detection polygons "
                             "(<stem>_polys.json) to <output>/STC Extras/masks/, no "
                             "PNGs. Feeds the Red Trail Map.")
    parser.add_argument("--twin-prefer", choices=["raw", "nonraw"], default="raw",
                        help="When a frame exists as both a RAW and a JPG/TIFF, "
                             "which to process. Mirrors the GUI's one-time prompt; "
                             "default RAW.")
    parser.add_argument("--output-format", choices=["jpg", "tif8", "tif16"],
                        default="jpg",
                        help="Output file format (default jpg)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality 60-100 (default 95)")
    parser.add_argument("--expected-width", type=int, default=None,
                        help="Expected image width — when provided, skips per-batch resolution detection")
    parser.add_argument("--expected-height", type=int, default=None,
                        help="Expected image height")
    parser.add_argument("--expected-bitdepth", type=int, choices=[8, 16], default=None,
                        help="Target storage bit depth (8 or 16). Frames not at "
                             "this depth are converted so each batch is uniform. "
                             "The GUI passes the sequence-wide majority; when "
                             "omitted the worker uses the batch's own majority.")
    parser.add_argument("--second-scrub", action="store_true",
                        help="Run detection a second time on each frame rotated 180°, merging any new trails found")
    parser.add_argument("--streak-filter", action="store_true",
                        help="Dev: drop detections shorter than this set's measured star-streak "
                             "length (unless red) -- for long-exposure sets where stars streak")
    parser.add_argument("--run-log-ts", default=None,
                        help="Shared run-log timestamp from the app so every batch of one "
                             "run appends to ONE run_log_<ts>.jsonl instead of one file per "
                             "batch. Omitted on a standalone run -> the engine stamps its own.")
    args = parser.parse_args()

    _crumb("BATCHSTART", extra=f"start={args.start} count={args.batch}")

    # Dev flag: auto-enable mask saving when ~/.star_trail_cleanr/.dev_save_masks exists
    if not args.save_masks and (Path.home() / ".star_trail_cleanr" / ".dev_save_masks").exists():
        args.save_masks = True

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # ── Dev-only run logger ───────────────────────────────────────────────────
    # Written to {output_dir}/STC Extras/run_log_{timestamp}.jsonl.
    # Dev-only: sys.frozen is True in the frozen bundle, so users never get this file.
    _is_dev = not getattr(sys, "frozen", False)
    if _is_dev:
        _ws_dir = output_dir / WORKSPACE_NAME
        _ws_dir.mkdir(parents=True, exist_ok=True)
        # Use the app-supplied run timestamp so every batch of one run appends to
        # the same file; fall back to our own stamp for a standalone run.
        _log_ts = args.run_log_ts or time.strftime("%Y-%m-%d_%H-%M-%S")
        logger = RunLogger(str(_ws_dir / f"run_log_{_log_ts}.jsonl"))
    else:
        logger = None

    def _write_output(stem: str, img: np.ndarray, icc_profile=None, exif_bytes=None, dpi=None):
        """Save one cleaned frame, turning a write failure into a clear user message.

        Wraps _write_output_inner so that the common "can't write here" failures
        (read-only drive, OneDrive sync lock, file open in another app) print a
        plain-English instruction and exit cleanly with code 2 instead of dumping a
        raw traceback. `stem` is the output filename without extension; `img` is the
        BGR image array; `icc_profile`/`exif_bytes`/`dpi` are metadata to embed.
        """
        from PIL import Image
        try:
            return _write_output_inner(stem, img, icc_profile=icc_profile,
                                       exif_bytes=exif_bytes, dpi=dpi)
        except (PermissionError, OSError) as _err:
            print(
                f"\nERROR: Cannot write cleaned frame to:\n  {output_dir}\n\n"
                "The output folder may be on a read-only drive, synced by "
                "OneDrive, or a file there may be open in another app. "
                "Pick a different output folder and try again.\n\n"
                f"(Detail: {type(_err).__name__}: {_err})",
                flush=True,
            )
            sys.exit(2)

    def _write_output_inner(stem: str, img: np.ndarray, icc_profile=None, exif_bytes=None, dpi=None):
        """Encode and write one cleaned frame in the chosen output format.

        Branches on args.output_format:
          - "jpg": down-shift 16-bit to 8-bit if needed, convert BGR->RGB, save via
            PIL with no chroma subsampling and the (size-fitted) EXIF.
          - "tif8": same 8-bit path but written as TIFF; retries without EXIF if
            libtiff rejects the EXIF block.
          - "tif16": full 16-bit RGB written via tifffile, because PIL has no native
            16-bit RGB save mode. 8-bit inputs are scaled up by *257 to fill 16 bits.
        ICC profile and DPI are embedded when supplied. After writing, stamps the
        Software string into the macOS Finder comment. `img` is a BGR array;
        OpenCV/PIL color order conversions happen here.
        """
        from PIL import Image
        if args.output_format == "jpg":
            out = img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb, mode="RGB")
            save_kwargs = {"quality": int(args.jpeg_quality), "subsampling": 0}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            _jpeg_exif = _fit_exif_for_jpeg(exif_bytes)
            if _jpeg_exif:
                save_kwargs["exif"] = _jpeg_exif
            if dpi:
                save_kwargs["dpi"] = dpi
            out_path = str(cleaned_dir / (stem + ".jpg"))
            pil.save(out_path, "JPEG", **save_kwargs)
        elif args.output_format == "tif8":
            out = img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb, mode="RGB")
            save_kwargs = {}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            _tiff_exif = _fit_exif_for_tiff(exif_bytes)
            if _tiff_exif:
                save_kwargs["exif"] = _tiff_exif
            if dpi:
                save_kwargs["dpi"] = dpi
            out_path = str(cleaned_dir / (stem + ".tif"))
            try:
                pil.save(out_path, "TIFF", **save_kwargs)
            except RuntimeError:
                save_kwargs.pop("exif", None)
                pil.save(out_path, "TIFF", **save_kwargs)
        else:  # tif16
            # PIL has no first-class 16-bit RGB image mode, so its fromarray
            # raises KeyError on uint16 RGB arrays. Use tifffile (a scientific
            # TIFF library, pinned explicitly in build_helper.py) to write
            # the file. Lazy import keeps it out of the JPG / tif8 hot path.
            import tifffile
            if img.dtype == np.uint16:
                out = img
            else:
                out = img.astype(np.uint16) * 257
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            extratags = []
            if icc_profile:
                # TIFF tag 34675 = InterColorProfile (ICC). 'B' = byte array.
                extratags.append((34675, 'B', len(icc_profile), icc_profile, False))
            # tifffile (unlike the JPEG / tif8 PIL path) does NOT embed the EXIF
            # block, so the capture date was dropped for EVERY source on 16-bit
            # TIFF -- the exact format RAW shooters are told to use for full bit
            # depth. Carry the human-meaningful tags across as standard TIFF
            # tags. Wrapped so a malformed EXIF block can never break the write.
            try:
                if exif_bytes:
                    from PIL import Image as _PILImage
                    _ex = _PILImage.Exif(); _ex.load(exif_bytes)
                    _dt = _ex.get(306)      # DateTime  (capture time)
                    _mk = _ex.get(271)      # Make
                    _md = _ex.get(272)      # Model
                    if _dt:
                        extratags.append((306, 's', 0, str(_dt), True))
                    if _mk:
                        extratags.append((271, 's', 0, str(_mk), True))
                    if _md:
                        extratags.append((272, 's', 0, str(_md), True))
            except Exception:
                pass
            # Orientation normal: pixels are turned upright before write, so the
            # source's rotate tag must not ride along (would double-rotate).
            extratags.append((274, 'H', 1, 1, True))
            tiff_kwargs = {
                "photometric": "rgb",
                "software": _stamp,
                "extratags": extratags,
            }
            if dpi:
                # tifffile expects (xres, yres) floats and a unit string.
                tiff_kwargs["resolution"] = (float(dpi[0]), float(dpi[1]))
                tiff_kwargs["resolutionunit"] = "inch"
            out_path = str(cleaned_dir / (stem + ".tif"))
            tifffile.imwrite(out_path, rgb, **tiff_kwargs)
        _write_finder_comment(out_path)
    cleaned_dir = output_dir
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = (output_dir / WORKSPACE_NAME / "masks"
                 if (args.save_masks or args.save_detections) else None)
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)

    # ── Load foreground mask → invert to sky mask ─────────────────────────
    sky_mask = None
    fg_mask = None
    if args.foreground_mask:
        fg_mask = robust_imread(args.foreground_mask, cv2.IMREAD_GRAYSCALE)
        if fg_mask is None:
            print(f"  WARN: foreground mask could not be loaded - continuing without it: {args.foreground_mask}")
        else:
            sky_mask = cv2.bitwise_not(fg_mask)
            print(f"  Applying sky mask")

    # ── Load frames ───────────────────────────────────────────────────────
    _ordered = None
    if args.frame_manifest:
        from modules.frame_list import read_manifest
        _ordered = read_manifest(args.frame_manifest)
    frame_files_all, core_start, core_end = load_with_neighbors(
        input_dir, args.start, args.batch,
        args.expected_width, args.expected_height,
        prefer_raw=(args.twin_prefer == "raw"),
        ordered_files=_ordered)
    frame_files = frame_files_all[core_start:core_end]  # core batch files
    n = len(frame_files)
    n_all = len(frame_files_all)
    if n < 3:
        # Say WHAT HAPPENED, not just the count. Two users lost a day to this on
        # 2026-08-11 (Sentry): the message read "need >= 3 frames (got 0)", which
        # tells nobody that the app had handed itself a frame list that did not
        # match its own batch plan. If the resolution filter is what emptied the
        # batch, name that -- it is the difference between a mystery and a
        # sentence a person can act on.
        dropped = getattr(load_with_neighbors, "last_dropped", 0)
        if dropped:
            print(f"ERROR: none of the frames in this batch are "
                  f"{args.expected_width}x{args.expected_height} -- "
                  f"{dropped} were a different size and were skipped, leaving "
                  f"{n}. This batch cannot be cleaned.")
        else:
            print(f"ERROR: this batch has only {n} frame(s); cleaning needs at "
                  f"least 3 (each frame is repaired using its neighbours).")
        sys.exit(1)

    # Grab ICC profile + DPI from the first core frame so output inherits the color
    # profile (Adobe RGB, ProPhoto, etc.) instead of being tagged as raw sRGB. These
    # are constant across a sequence. EXIF (capture date/time, exposure, lens, GPS) is
    # read PER FRAME at write time from each frame's own file — never shared across the
    # batch — so every cleaned frame keeps its own original capture metadata.
    # _source_metadata reads RAW from its embedded preview (PIL can't open a RAW
    # directly, and asking it to would trip the engine's pi-heif auto-install
    # hang); JPEG/TIFF read from the file. ICC may be absent on a RAW preview,
    # leaving output tagged sRGB, which is correct for a RAW preview.
    _, icc_profile, dpi = _source_metadata(frame_files_all[core_start])

    # Build the Software-tag stamp that goes into every cleaned file's EXIF.
    # Format: "Star Trail CleanR v<app> / Trail Detector v<model> / www.startrailcleanr.com"
    def _resolve_app_version():
        """Read the app version string from version.txt for the EXIF Software stamp.

        Looks beside the executable in the frozen bundle (sys._MEIPASS) or beside
        this source file when running live. Returns "?" if the file is missing or
        unreadable so the stamp still builds.
        """
        try:
            base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base, "version.txt")) as vf:
                return vf.read().strip()
        except Exception:
            return "?"

    def _resolve_model_version():
        """Return the trained detector version (e.g. "v4") for the EXIF Software stamp.

        Asks model_update.local_model_version() for the installed model release tag
        (like "model-v4") and extracts the numeric part as "vN". Falls back to the
        raw tag, then "?", on any failure.
        """
        try:
            from modules.model_update import local_model_version
            import re
            tag = local_model_version()
            m = re.match(r"^model-v(\d+(?:\.\d+)?)", tag or "")
            return f"v{m.group(1)}" if m else (tag or "?")
        except Exception:
            return "?"

    _stamp = f"Star Trail CleanR v{_resolve_app_version()} / Trail Detector {_resolve_model_version()} / www.startrailcleanr.com"

    def _stamp_exif(source_bytes):
        """Return EXIF bytes with Software stamp added. Preserves all source EXIF unchanged."""
        try:
            from PIL import Image as _PILImage
            ex = _PILImage.Exif()
            if source_bytes:
                ex.load(source_bytes)
            ex[0x0131] = _stamp  # Software tag — shown in Lightroom, Photoshop, and EXIF viewers
            ex[0x0112] = 1       # Orientation = normal. Pixels are turned upright on read,
                                 # so the source's rotate tag must NOT ride along, or the
                                 # viewer would rotate already-upright pixels (double rotation).
            return ex.tobytes()
        except Exception:
            return source_bytes

    def _frame_exif_bytes(path):
        """Read a single frame's OWN EXIF bytes from its source file. For RAW the
        EXIF is read from the file's embedded preview (PIL cannot open a RAW
        directly); for JPEG/TIFF, from the file itself. Returns None when no EXIF
        is available. Called once per output frame so each cleaned file inherits
        ITS OWN capture date/time, exposure, and camera — never the batch
        leader's. Only the Software comment and orientation tag are added on top
        (see _stamp_exif)."""
        return _source_metadata(path)[0]

    def _write_finder_comment(out_path: str) -> None:
        """Write _stamp to macOS Finder Comments field via Finder AppleScript. No-op if not macOS."""
        if sys.platform != 'darwin':
            return
        try:
            import subprocess as _sp
            _safe = _stamp.replace('"', '')
            _sp.run(
                ['osascript', '-e',
                 f'tell application "Finder" to set comment of (POSIX file "{out_path}" as alias) to "{_safe}"'],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    def _fit_exif_for_jpeg(exif_bytes):
        """Ensure EXIF fits in JPEG's 65535-byte APP1 limit. Customer data takes priority.
        Falls back: full EXIF > MakerNote-stripped EXIF > no EXIF (never crashes).
        MakerNote (0x927C) is manufacturer binary data no standard tool can read.
        """
        _JPEG_EXIF_MAX = 65000
        if not exif_bytes or len(exif_bytes) <= _JPEG_EXIF_MAX:
            return exif_bytes
        try:
            from PIL import Image as _PILImage
            ex = _PILImage.Exif()
            ex.load(exif_bytes)
            ex.pop(0x927C, None)
            trimmed = ex.tobytes()
            if len(trimmed) <= _JPEG_EXIF_MAX:
                return trimmed
        except Exception:
            pass
        return None  # save without EXIF rather than crash; originals are untouched

    def _fit_exif_for_tiff(exif_bytes):
        """Strip tags that libtiff owns on the output file and MakerNote from TIFF EXIF.
        Structural tags (width, height, compression, etc.) describe the source file's
        geometry and conflict with libtiff's internal state on a new file. MakerNote
        contains proprietary binary data with internal offsets into the source file.
        Returns cleaned bytes; on any error returns the original so the save still runs.
        """
        if not exif_bytes:
            return exif_bytes
        # Tags libtiff sets itself when writing a new TIFF — passing them causes
        # RuntimeError: Error setting from dictionary
        _TIFF_STRUCTURAL = {
            256,  # ImageWidth
            257,  # ImageLength
            258,  # BitsPerSample
            259,  # Compression
            262,  # PhotometricInterpretation
            273,  # StripOffsets
            277,  # SamplesPerPixel
            278,  # RowsPerStrip
            279,  # StripByteCounts
            284,  # PlanarConfiguration
            322,  # TileWidth
            323,  # TileLength
            324,  # TileOffsets
            325,  # TileByteCounts
            0x927C,  # MakerNote — proprietary binary with source-file offsets
        }
        try:
            from PIL import Image as _PILImage
            ex = _PILImage.Exif()
            ex.load(exif_bytes)
            for tag in _TIFF_STRUCTURAL:
                ex.pop(tag, None)
            return ex.tobytes()
        except Exception:
            return exif_bytes

    print(f"Loading {n} frames...")
    frames_all = []
    files_kept = []
    skipped = []
    skipped_before_core = 0
    skipped_in_core = 0
    _tread = time.perf_counter()
    _raw_decode_s = 0.0   # cumulative RAW debayer time (rawpy); 0 for JPG/TIFF
    _raw_decode_n = 0
    for fi, fp in enumerate(frame_files_all):
        is_core = core_start <= fi < core_end
        is_before_core = fi < core_start
        _crumb("LOAD", fi + 1, len(frame_files_all), fp.name)
        _t_read1 = time.perf_counter()
        img, diag = robust_imread_diag(fp, cv2.IMREAD_UNCHANGED)
        if fp.suffix.lower() in RAW_EXTS and img is not None:
            # RAW frames are debayered with rawpy, which is far slower than
            # decoding a JPG/TIFF. Track it so the per-frame cost of RAW input
            # is visible in the run log instead of hidden in total load time.
            _raw_decode_s += time.perf_counter() - _t_read1
            _raw_decode_n += 1
        if img is None:
            # Best-effort developer telemetry — captured before we ask the GUI
            # so we still have data even if the user clicks Stop.
            _capture_unreadable_file_to_sentry(fp, diag)

            # Log the per-file detail to the Star Log scroll for support emails.
            print(
                "\n  Bad file:\n"
                f"    {fp}\n"
                "  Reason:\n"
                f"{diag}",
                flush=True,
            )

            decision = _prompt_gui_for_bad_file(fp, diag)
            if decision == "STOP":
                print(
                    "\n  Run stopped at user's request. Partial output (the "
                    "frames cleaned so far) is preserved in the output folder.",
                    flush=True,
                )
                sys.exit(0)

            # CONTINUE: skip this frame, keep loading the rest of the batch.
            if is_before_core:
                skipped_before_core += 1
            elif is_core:
                skipped_in_core += 1
            skipped.append((fp, diag))
            continue

        # EXIF orientation (sideways / portrait shots) is now applied centrally
        # in robust_imread for every format and flag, so `img` is already upright
        # here. The cleaned file's orientation tag is reset to normal at save time
        # (see _stamp_exif), so viewers never re-rotate — fixes the double-rotation
        # that turned portrait-shot output 90 degrees off.

        # Some TIFFs carry an embedded alpha channel. IMREAD_UNCHANGED preserves
        # it, producing (H,W,4) arrays that crash Star Bridge repair when mixed
        # with normal (H,W,3) neighbor frames. Strip the alpha here.
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            if alpha.min() < 255:
                print(
                    f"  Warning: {fp.name} has a transparency layer that was ignored."
                    f" If you masked this image intentionally, re-export it as a"
                    f" standard RGB TIFF and try again.",
                    flush=True,
                )
            else:
                print(f"  Note: stripped alpha channel from {fp.name} - RGB data is intact", flush=True)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if is_core:
            core_pos = fi - core_start + 1
            h, w = img.shape[:2]
            ch = img.shape[2] if img.ndim == 3 else 1
            print(f"  loading {core_pos}/{n}: {fp.name} ({w}x{h}, {ch}ch)", flush=True)

        frames_all.append(img)
        files_kept.append(fp)
    _read_s = time.perf_counter() - _tread

    # Rebind to kept-only lists with adjusted core pointers so downstream
    # indexing stays correct even when one or more files were skipped.
    frame_files_all = files_kept
    core_start -= skipped_before_core
    core_end -= skipped_before_core + skipped_in_core
    frame_files = frame_files_all[core_start:core_end]
    n = len(frame_files)
    n_all = len(frame_files_all)

    if n < 1:
        print(
            "\nERROR: every frame in this batch was unreadable, so there is\n"
            "nothing to clean. See the per-file reasons above."
        )
        sys.exit(1)

    frames = frames_all[core_start:core_end]
    h, w = frames[0].shape[:2]
    if skipped:
        print(
            f"\n  Note: {len(skipped)} file(s) skipped because they couldn't\n"
            f"  be read. The cleaned output will have gap(s) at those\n"
            f"  positions. Continuing with {n} frame(s).",
            flush=True,
        )
    print(f"  {n} frames loaded ({w}x{h})", flush=True)
    if _raw_decode_n:
        print(
            f"  RAW debayer: {_raw_decode_s:.1f}s for {_raw_decode_n} frame(s) "
            f"({_raw_decode_s / _raw_decode_n:.2f}s/frame)",
            flush=True,
        )
        if logger is not None:
            logger.log({
                "type": "raw_decode_timing",
                "frames": _raw_decode_n,
                "total_s": round(_raw_decode_s, 3),
                "avg_s": round(_raw_decode_s / _raw_decode_n, 3),
            })

    # Even out mixed bit depths instead of refusing the run. A folder can
    # legitimately hold both 8-bit and 16-bit frames -- most often when most
    # frames have a 16-bit TIFF/RAW twin we keep, but a few tail frames exist
    # only as an 8-bit JPG. Every frame (core + neighbors) is brought to one
    # target depth: the sequence-wide majority the GUI passes via
    # --expected-bitdepth, or this batch's own majority when run standalone.
    # This replaces an older hard stop that aborted the WHOLE run on the first
    # mixed batch -- which was usually the last one, after every earlier batch
    # had already been cleaned, leaving the user a half-finished job and a
    # message that wrongly blamed them for keeping jpg+tif copies (the one case
    # our up-front twin-merge already handles).
    _target_dtype = _resolve_target_dtype(frames_all, args.expected_bitdepth)
    _n_normalized = 0
    for _i in range(len(frames_all)):
        _conv = _match_bitdepth(frames_all[_i], _target_dtype)
        if _conv is not frames_all[_i]:
            frames_all[_i] = _conv
            _n_normalized += 1
    if _n_normalized:
        _depth_label = "16-bit" if _target_dtype == np.uint16 else "8-bit"
        print(f"  Evened out {_n_normalized} frame(s) to {_depth_label} so the "
              f"whole batch is one format.", flush=True)
    # Re-point the core view at the (possibly replaced) arrays.
    frames = frames_all[core_start:core_end]

    # The foreground mask, hot-pixel step and Star Bridge repair all assume every
    # frame in the batch shares one shape. A mixed portrait/landscape batch (or a
    # mask painted for the other orientation) would otherwise crash deep in the
    # hot-pixel step with an opaque OpenCV "sizes do not match" error. Stop early
    # with a plain message instead.
    shapes = {f.shape[:2] for f in frames_all}
    if len(shapes) > 1:
        print("\nERROR: this batch mixes portrait and landscape frames. "
              "Run the portrait frames and the landscape frames as separate "
              "batches -- each orientation needs its own foreground mask because "
              "the framing is different.")
        sys.exit(1)
    # An empty foreground mask (nothing painted) means "exclude nothing" -- treat
    # it as no mask. It then never blocks on a shape mismatch and never reaches
    # the size-sensitive hot-pixel / sky-mask steps.
    if fg_mask is not None and not np.any(fg_mask):
        print("  Note: foreground mask is empty (nothing painted); "
              "running without it.", flush=True)
        fg_mask = None
        sky_mask = None
    if fg_mask is not None and fg_mask.shape[:2] != (h, w):
        # Same shape, different size: the mask is right, just painted at another
        # resolution (a real user painted on a half-size frame the run then
        # skipped -- Sentry, 2026-08-03). A mask is a stencil, so scaling it to
        # fit loses nothing that matters. Only a genuine shape difference (e.g.
        # portrait mask on landscape frames) still stops the run.
        mh, mw = fg_mask.shape[:2]
        if mh > 0 and h > 0 and abs((mw / mh) - (w / h)) < 0.01:
            fg_mask = cv2.resize(fg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if sky_mask is not None and sky_mask.shape[:2] != (h, w):
                sky_mask = cv2.resize(sky_mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            print(f"  Note: the foreground mask was painted at {mw}x{mh}; "
                  f"scaled it to match these {w}x{h} frames.", flush=True)
        else:
            print("\nERROR: the foreground mask does not match these frames "
                  f"(mask is {mw}x{mh}, frames are {w}x{h}). "
                  "It was likely painted for a different orientation or image size. "
                  "Re-create the foreground mask for these frames, then try again.")
            sys.exit(1)

    # 16-bit handling
    is_16bit = frames_all[0].dtype == np.uint16
    if is_16bit:
        frames_8bit_all = [(f >> 8).astype(np.uint8) for f in frames_all]
        frames_8bit = frames_8bit_all[core_start:core_end]
    else:
        frames_8bit_all = frames_all
        frames_8bit = frames

    _thot = time.perf_counter()
    if fg_mask is not None:
        from modules.hot_pixels import build_hot_pixel_map

        # Build THIS batch's stuck-pixel map, then accumulate it into the shared
        # map file so detections build up across batches instead of freezing on
        # batch 1 (a stuck pixel that only shows up later still gets caught).
        hot_map = build_hot_pixel_map(frames_8bit)
        if args.hot_pixel_map and os.path.isfile(args.hot_pixel_map):
            prev = robust_imread(args.hot_pixel_map, cv2.IMREAD_GRAYSCALE)
            if prev is not None and prev.shape[:2] == hot_map.shape[:2]:
                hot_map = cv2.bitwise_or(hot_map, prev)
        if args.hot_pixel_map and int((hot_map > 0).sum()) > 0:
            robust_imwrite(args.hot_pixel_map, hot_map)

        # Foreground (ground) stuck pixels are repaired here, per frame, with the
        # plain inpaint (the textured ground hides it). SKY stuck pixels are NOT
        # touched here -- they are cleaned once, at the end, on the finished
        # star-trail stack (make_star_trail), and only when the user asked for a
        # star trail. The map saved above is the running union so that end step
        # has the complete set of stuck-pixel locations.
        if hot_map.max() > 0:
            dilated = cv2.dilate(hot_map,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            dilated = cv2.bitwise_and(dilated, fg_mask)

        if hot_map.max() > 0 and dilated.max() > 0:
            if is_16bit:
                for i in range(len(frames)):
                    orig = robust_imread(frame_files[i], cv2.IMREAD_UNCHANGED)
                    orig8 = (orig >> 8).astype(np.uint8)
                    rep8 = cv2.inpaint(orig8, dilated, 3, cv2.INPAINT_NS)
                    rep16 = rep8.astype(np.uint16) * 257
                    orig[dilated > 0] = rep16[dilated > 0]
                    frames[i] = orig
                frames_8bit = [(f >> 8).astype(np.uint8) for f in frames]
            else:
                for i in range(len(frames)):
                    frames[i] = cv2.inpaint(frames[i], dilated, 3, cv2.INPAINT_NS)
                frames_8bit = frames

            for j, idx in enumerate(range(core_start, core_end)):
                frames_all[idx] = frames[j]
                frames_8bit_all[idx] = frames_8bit[j]
    _hotpix_s = time.perf_counter() - _thot

    # Resolution scaling for min_area filter.
    # The min-area threshold (smallest detection kept) is calibrated for a
    # reference sensor (5472x3648, a common ~20MP body). On a larger sensor a real
    # trail covers proportionally more pixels, so scale the threshold up by the
    # frame's pixel count relative to that reference. max(args.min_area, ...) means
    # smaller sensors keep the unscaled floor rather than dropping below it.
    REF_PIXELS = 5472 * 3648
    sc_area = (w * h) / REF_PIXELS
    min_area_scaled = max(args.min_area, int(args.min_area * sc_area))

    t_total = time.time()

    # ── Step 1: Detect trails (YOLO) ────────────────────────────────────
    print("\nStep 1 - detecting trails", flush=True)
    print("  Loading AI trail detector...", flush=True)
    _tload = time.perf_counter()
    model, _device_used = load_model(str(args.model), args.confidence, args.device)
    _model_load_s = time.perf_counter() - _tload

    # SAY WHAT IS ACTUALLY HAPPENING -- when it works as well as when it does
    # not. Bruce, 2026-08-23: "we want to let the user know when something fails,
    # but better when it works as advertised. and if it fails what we need to do
    # to fix it." A user with a 3080 Ti spent a 9-hour run squinting at Task
    # Manager trying to work out whether his card was being used at all.
    #
    # This is printed by the WORKER, the process that actually loaded the model,
    # so it reports the truth rather than what the app expected. The machine
    # -readable marker on the next line is what the app records for the run
    # summary and the usage report. Only the first batch says it out loud; the
    # marker goes on every batch so a mid-run fallback cannot hide.
    print(f"STC_DEVICE:{_device_used}", flush=True)
    if args.start == 0:
        if _device_used == "cuda":
            _card = ""
            try:
                from modules.nvidia_detect import detect_nvidia
                _out, _detail = detect_nvidia()
                if _out == "yes" and _detail:
                    _card = " (" + str(_detail).split(" (driver")[0] + ")"
            except Exception:
                pass
            print(f"  Using your NVIDIA graphics card{_card}.", flush=True)
        elif _device_used == "mps":
            print("  Using your Mac's graphics processor.", flush=True)
        else:
            print("  Using the processor. See the note above if this machine "
                  "has a graphics card.", flush=True)

    masks_all = []
    raw_masks_all = []
    segs_all = []           # per-frame polygon segment lists for independent repair
    corners_all = []        # per-frame polygon corner coordinate lists for JSON export
    edge_candidates_all = []  # per-frame lists of edge-touching detections that failed elongation
    detect_infos = []  # buffered per-frame detect data; written after static FP pass
    running_trail_total = 0

    _timing = {}

    def _tacc(key, elapsed):
        """Accumulate timing for one named step into the _timing dict.

        Each entry is [call_count, total_seconds]; this adds one call and the given
        `elapsed` seconds. Used throughout the batch so the end-of-run timing table
        can show calls / total / average per pipeline step.
        """
        if key in _timing:
            _timing[key][0] += 1
            _timing[key][1] += elapsed
        else:
            _timing[key] = [1, elapsed]

    # Startup cost, measured not guessed: reading frames off the drive, building
    # the hot-pixel map, and loading the model (this batch's subprocess). The
    # first-inference warmup shows up as the first frame's tiled_inference.
    _tacc("frame_read_s", _read_s)
    _tacc("hot_pixel_s", _hotpix_s)
    _tacc("model_load_s", _model_load_s)

    # New modular pipeline is THE detector -- no fallback. Stages 3/4
    # (fallback_polys, link_gaps) are not built yet -> off. The static FP
    # suppressor runs as Step 1c below, so the pipeline's own one stays off.
    new_cfg = dp.StageConfig(
        tile_size=args.tile_size, overlap=args.overlap,
        fit_polygons=True, fallback_polys=False, link_gaps=False,
        seam_second_pass=True, suppress_fp=False,
        prune_phantoms=True,
        # Hard-negative phantom logging is dev-only (Bruce's source runs), never
        # in the shipped frozen bundle. sys.frozen is True only in the bundle.
        log_phantom_negatives=not getattr(sys, "frozen", False),
    )

    # Dev streaking-star filter: measure this set's star-streak ceiling ONCE (read off the
    # frames, never a formula), then drop detections shorter than it unless red. Off by default.
    streak_ceiling = None
    streak_dropped_total = 0
    streak_red_kept_total = 0
    if args.streak_filter:
        from modules.star_streak import MIN_TRAIL_PX
        streak_ceiling = MIN_TRAIL_PX
        print(f"  Streaking-star filter ON: dropping detections shorter than "
              f"{streak_ceiling:.0f}px unless red", flush=True)

    for i, fp in enumerate(frame_files_all):
        is_neighbor = i < core_start or i >= core_end
        dbg = {} if logger is not None else None

        edge_cands = []
        frame_segs = []
        frame_corners = []
        _crumb("DETECT", i + 1, len(frame_files_all), fp.name)
        _t0 = time.perf_counter()
        state = dp.detect_frame(model=model, image=frames_8bit_all[i],
                                foreground_mask=fg_mask, frame_name=fp.stem,
                                cfg=new_cfg)
        _tacc("new_pipeline_s", time.perf_counter() - _t0)
        for _sname, _ssec in state.stage_seconds.items():
            _tacc(f"dp_{_sname}_s", _ssec)
        if dbg is not None:
            # Record per-stage timing + what fired (counts/events) in this frame's
            # log record, so a real run answers "what fired / how long" directly.
            dbg["detect_stages"] = state.stage_log
        mask = state.final_mask
        frame_segs = list(state.polygon_segs)
        frame_corners = list(state.polygons)
        raw_labeled = _raw_labeled_from_state(state, *mask.shape[:2]) \
            if mask is not None else None
        edge_candidates_all.append(edge_cands)
        if mask is None:
            masks_all.append(np.zeros((h, w), dtype=np.uint8))
            raw_masks_all.append(np.zeros((h, w), dtype=np.uint8))
            segs_all.append([])
            corners_all.append([])
            if dbg is not None:
                dbg.update({"frame": fp.stem, "frame_idx": i,
                            "is_neighbor": is_neighbor,
                            "detect_error": "detect_frame returned None mask"})
            detect_infos.append(dbg)
            continue

        sky_px_removed = 0
        if sky_mask is not None:
            before_px = int((mask > 0).sum())
            _t0 = time.perf_counter()
            mask = apply_sky_mask(mask, sky_mask)
            _tacc("apply_sky_mask_s", time.perf_counter() - _t0)
            sky_px_removed = before_px - int((mask > 0).sum())
            # New pipeline doesn't sky-mask its predictions (it only skips fully
            # foreground tiles), so trim each repair segment to sky here and drop
            # any that vanish -- keeps repair off the foreground. Corners stay
            # index-aligned with segs for the _polys.json export.
            if frame_segs:
                kept_segs, kept_corners = [], []
                for _si, _seg in enumerate(frame_segs):
                    _sm = apply_sky_mask(_seg, sky_mask)
                    if (_sm > 0).any():
                        kept_segs.append(_sm)
                        if _si < len(frame_corners):
                            kept_corners.append(frame_corners[_si])
                frame_segs = kept_segs
                frame_corners = kept_corners

        # Dev streaking-star filter: drop segs shorter than the measured ceiling (unless red),
        # and clear their pixels from the mask so mask/segs/repair/export stay consistent.
        if args.streak_filter and streak_ceiling and frame_segs:
            from modules.star_streak import filter_segs
            _ks, _kc, _dropped, _nred = filter_segs(
                frame_segs, frame_corners, frames_8bit_all[i], streak_ceiling)
            if _dropped:
                _keep_u = np.zeros(mask.shape, dtype=bool)
                for _s in _ks:
                    _keep_u |= (_s > 0)
                for _s in _dropped:
                    mask[(_s > 0) & ~_keep_u] = 0
                streak_dropped_total += len(_dropped)
                streak_red_kept_total += _nred
            frame_segs, frame_corners = _ks, _kc

        small_dbg = {} if dbg is not None else None
        if min_area_scaled > 0 and mask.max() > 0:
            _t0 = time.perf_counter()
            mask = filter_small_components(mask, frames_8bit_all[i], min_area_scaled,
                                           debug_out=small_dbg)
            _tacc("filter_small_s", time.perf_counter() - _t0)

        masks_all.append(mask)
        raw_masks_all.append(raw_labeled if raw_labeled is not None
                             else np.zeros((h, w), dtype=np.uint8))
        segs_all.append(frame_segs)
        corners_all.append(frame_corners)
        if is_neighbor:
            print(f"  detecting neighbor: {fp.name}", flush=True)
        else:
            core_num = i - core_start + 1
            print(f"  detecting {core_num}/{n}: {fp.name}", flush=True)

        if dbg is not None:
            dbg.update({
                "frame":                  fp.stem,
                "frame_idx":              i,
                "is_neighbor":            is_neighbor,
                "sky_mask_pixels_removed": sky_px_removed,
                "small_filter":           small_dbg or {},
            })
        detect_infos.append(dbg)

    if args.second_scrub:
        # Optional second detection pass on each frame rotated 180 degrees. The
        # detector's tile grid and learned orientation biases differ once the
        # image is flipped, so a trail the first pass missed can surface this time.
        # The image AND the foreground mask are both rotated so they stay aligned;
        # the resulting mask is rotated back and OR'd (np.maximum) into the
        # first-pass mask, so the scrub can only ADD trails, never remove them.
        print("\nStep 1b - second scrub (180-degree rotation)", flush=True)
        try:
            for i, fp in enumerate(frame_files_all):
                rotated = np.rot90(frames_8bit_all[i], 2)
                _t0 = time.perf_counter()
                rot_fg = np.rot90(fg_mask, 2) if fg_mask is not None else None
                state2 = dp.detect_frame(model=model, image=rotated,
                                         foreground_mask=rot_fg,
                                         frame_name=fp.stem + "_rot180", cfg=new_cfg)
                _tacc("second_scrub_s", time.perf_counter() - _t0)
                mask2 = state2.final_mask
                if mask2 is None:
                    continue
                mask2 = np.rot90(mask2, 2)
                if sky_mask is not None:
                    mask2 = apply_sky_mask(mask2, sky_mask)
                if min_area_scaled > 0 and mask2.max() > 0:
                    mask2 = filter_small_components(mask2, frames_8bit_all[i], min_area_scaled)
                masks_all[i] = np.maximum(masks_all[i], mask2)
                is_neighbor = i < core_start or i >= core_end
                if not is_neighbor:
                    core_num = i - core_start + 1
                    if masks_all[i].max() > 0:
                        n_cc, _ = cv2.connectedComponents((masks_all[i] > 0).astype(np.uint8))
                        trail_count = max(0, n_cc - 1)
                    else:
                        trail_count = 0
                    print(f"  second scrub {core_num}/{n}: {fp.name} - {trail_count} trail{'s' if trail_count != 1 else ''}", flush=True)
        except Exception as e:
            print(f"  WARN: second scrub failed ({e}) - continuing with first-pass results only", flush=True)

    if args.streak_filter and streak_ceiling:
        print(f"  Streaking-star filter: dropped {streak_dropped_total} short detection(s) "
              f"below {streak_ceiling:.0f}px as star streaks "
              f"(kept {streak_red_kept_total} red as nav light{'s' if streak_red_kept_total != 1 else ''})",
              flush=True)

    print("\nStep 1c - removing static false positives...", flush=True)
    static_fp_dbg = {} if logger is not None else None
    _sfp_timing = {}
    _sfp_removed = {}
    _t0 = time.perf_counter()
    n_static = _suppress_static_fps(masks_all, core_start, core_end,
                                    raw_masks_all=raw_masks_all,
                                    frames_all=frames_all,
                                    debug_out=static_fp_dbg,
                                    timing_out=_sfp_timing,
                                    removed_out=_sfp_removed)
    _tacc("static_fp_s", time.perf_counter() - _t0)
    for _k, _v in _sfp_timing.items():
        _tacc(_k, _v)
    if n_static:
        print(f"  Suppressed {n_static} static detection(s) present in 2+ neighboring frames",
              flush=True)
    else:
        print("  No static false positives found", flush=True)

    # Apply the suppressor's verdict to the per-polygon lists too. Repair
    # iterates segs_all (one Star Bridge pass per polygon) and the _polys.json
    # export writes corners_all, so a polygon left in these lists gets cleaned
    # and uploaded to CVAT even though its pixels were just erased from the
    # mask (todo #94: 29 rejected outlines reached CVAT training review, and
    # repair kept "fixing" spots the suppressor rejected). A polygon is dropped
    # when at least half its pixels were suppressed -- the suppressor removes
    # whole components, so affected polygons lose essentially all their pixels;
    # untouched polygons lose none. segs and corners are index-aligned pairs
    # (same contract as the sky-mask trim above).
    _sfp_polys_removed = {}
    if _sfp_removed:
        for _fi, _rm in _sfp_removed.items():
            if _fi >= len(segs_all) or not segs_all[_fi]:
                continue
            _kept_s, _kept_c = [], []
            _n_drop = 0
            for _sj, _seg in enumerate(segs_all[_fi]):
                _seg_on = _seg > 0
                _seg_px = int(_seg_on.sum())
                if _seg_px and int((_seg_on & _rm).sum()) * 2 >= _seg_px:
                    _n_drop += 1
                    continue
                _kept_s.append(_seg)
                if _sj < len(corners_all[_fi]):
                    _kept_c.append(corners_all[_fi][_sj])
            if _n_drop:
                segs_all[_fi] = _kept_s
                corners_all[_fi] = _kept_c
                _sfp_polys_removed[_fi] = _n_drop
        _total_polys_dropped = sum(_sfp_polys_removed.values())
        if _total_polys_dropped:
            print(f"  Removed {_total_polys_dropped} rejected polygon(s) from the "
                  f"repair and export lists", flush=True)

    # Step 1d - Edge rescue: recover trails that were clipped at the frame boundary.
    #
    # The elongation filter rejects detections whose bounding box is too square
    # (aspect ratio < 2:1), because real trails are long and thin. But a trail
    # that exits the frame at any edge gets clipped to a short stub that fails
    # this test even though it is real. The grouper flags these as "edge
    # candidates" instead of discarding them entirely.
    #
    # Here we check each edge candidate against the post-suppression masks of
    # neighboring frames (within ±4). If 2 or more neighboring frames contain
    # a mask component whose major-axis direction matches the candidate within
    # 15 degrees, the candidate is reinstated: its pixels are merged into the
    # frame's mask so repair can fill it in.
    #
    # Running AFTER the static FP suppressor means rescued candidates are checked
    # against already-cleaned neighbor masks -- static objects that got suppressed
    # in neighbor frames won't accidentally rescue a matching edge stub.
    _EDGE_WINDOW      = 4    # check up to 4 frames before and after
    _EDGE_SLOPE_TOL   = 15.0 # degrees -- how closely slopes must match
    _EDGE_MIN_MATCHES = 1    # 1 neighbor suffices: static FPs were already removed
                             # by the suppressor above; raw fallback handles the
                             # case where every frame has a stub that failed elongation
    _t0 = time.perf_counter()
    n_rescued = 0
    if any(len(ec) > 0 for ec in edge_candidates_all):
        print("\nStep 1d - rescuing edge-clipped detections...", flush=True)
        try:
            for i, cands in enumerate(edge_candidates_all):
                if not cands:
                    continue
                for cand in cands:
                    match_count = 0
                    for offset in range(-_EDGE_WINDOW, _EDGE_WINDOW + 1):
                        if offset == 0:
                            continue
                        j = i + offset
                        if j < 0 or j >= len(masks_all):
                            continue
                        nb_mask = masks_all[j]
                        if nb_mask.max() == 0:
                            # Fitted mask is empty in this neighbor -- fall back to
                            # raw SAHI hits. If the trail stub also failed elongation
                            # in every neighboring frame (circular dependency), the
                            # raw hits still confirm the trail's slope and location.
                            raw_nb = raw_masks_all[j] if j < len(raw_masks_all) else None
                            if raw_nb is not None and raw_nb.max() > 0:
                                nb_mask = (raw_nb > 0).astype(np.uint8) * 255
                            else:
                                continue
                        # Split the neighbor mask into individual components and
                        # check each one for a slope match with the edge candidate.
                        n_labels, labeled = cv2.connectedComponents(
                            (nb_mask > 0).astype(np.uint8))
                        for lbl_id in range(1, n_labels):
                            comp = (labeled == lbl_id).astype(np.uint8) * 255
                            nb_props = detection_props(comp, min_aspect=0.0)
                            if nb_props is None:
                                continue
                            cos_sim = min(abs(float(np.dot(cand["u"], nb_props["u"]))), 1.0)
                            adiff = min(np.degrees(np.arccos(cos_sim)),
                                        180.0 - np.degrees(np.arccos(cos_sim)))
                            if adiff <= _EDGE_SLOPE_TOL:
                                match_count += 1
                                break  # one matching component per frame is enough
                    if match_count >= _EDGE_MIN_MATCHES:
                        masks_all[i] = np.maximum(masks_all[i], cand["mask"])
                        n_rescued += 1
        except Exception as e:
            # Edge rescue is a best-effort step. If it fails for any reason
            # (unexpected mask shape, bad coords, etc.) we log the error and
            # continue with whatever masks were already restored. Detection and
            # repair still run; only the clipped-trail recovery is skipped.
            print(f"  WARN: edge rescue failed and was skipped ({e})", flush=True)
        if n_rescued:
            print(f"  Restored {n_rescued} edge-clipped detection(s) confirmed by "
                  f"{_EDGE_MIN_MATCHES}+ neighboring frames", flush=True)
        else:
            print("  No edge-clipped detections qualified for rescue", flush=True)

    _tacc("edge_rescue_s", time.perf_counter() - _t0)

    # Write all detect records now that static FP suppression is complete.
    # final_trail_components is computed here (post-suppression) from masks_all.
    if logger is not None:
        _sfp_by_frame  = (static_fp_dbg or {}).get("suppressed_by_frame", {})
        _veto_by_frame = (static_fp_dbg or {}).get("kept_by_veto_by_frame", {})

        def _fp_in_sky(cx, cy):
            """Is the point (cx, cy) over open sky rather than foreground?

            Used to tag each suppressed false positive in the run log: a FP over sky
            is a genuine miss worth harvesting as a hard-negative training example,
            while a FP on the foreground (wall, tree, building) is expected and not
            training material. Returns True when the foreground mask is 0 at that
            pixel. With no mask we can't tell, so default to True (treat as sky).
            """
            if fg_mask is None or cx is None or cy is None:
                return True
            yy = min(fg_mask.shape[0] - 1, max(0, int(cy)))
            xx = min(fg_mask.shape[1] - 1, max(0, int(cx)))
            return bool(fg_mask[yy, xx] == 0)

        _harvest_sky = _harvest_fg = _harvest_miss = 0
        for _i, _dbg in enumerate(detect_infos):
            if _dbg is None:
                continue
            _m = masks_all[_i] if _i < len(masks_all) else None
            if _m is not None and _m.max() > 0:
                _ncc, _ = cv2.connectedComponents((_m > 0).astype(np.uint8))
                _dbg["final_trail_components"] = max(0, _ncc - 1)
            else:
                _dbg["final_trail_components"] = 0
            # Tag every suppressed false positive sky-vs-foreground at log time,
            # so future analysis never has to re-derive it from the mask.
            _sfp = _sfp_by_frame.get(_i, [])
            for _d in _sfp:
                _sky = _fp_in_sky(_d.get("cx"), _d.get("cy"))
                _d["in_sky"] = _sky
                _d["note"] = ("over open sky -- genuine false positive, hard-negative candidate"
                              if _sky else
                              "on foreground (wall/tree/building) -- expected, not training material")
                if _sky:
                    _harvest_sky += 1
                else:
                    _harvest_fg += 1
            _dbg["static_fp_suppressed"]    = _sfp
            _dbg["static_fp_kept_by_veto"]  = _veto_by_frame.get(_i, [])
            if _i in _sfp_polys_removed:
                # How many polygons the verdict pulled out of the repair/export
                # lists on this frame (todo #94) -- proof in the log that
                # rejected detections no longer reach repair, MaskViewR, or CVAT.
                _dbg["suppressed_polys_removed"] = _sfp_polys_removed[_i]
            for _st in (_dbg.get("detect_stages") or []):
                for _ev in (_st.get("events") or []):
                    if _ev.get("reason") == "bridge_gap_miss":
                        _harvest_miss += 1
            _dbg["type"] = "detect"
            logger.log(_dbg)

        logger.log({
            "type": "harvest",
            "_doc": "Training examples this batch produced. sky_false_positives = hard negatives; "
                    "missed_trails_bridged = hard positives; foreground false positives are skipped.",
            "sky_false_positives": _harvest_sky,
            "foreground_false_positives": _harvest_fg,
            "missed_trails_bridged": _harvest_miss,
        })

    masks_per_frame = masks_all[core_start:core_end]
    trail_frames = sum(1 for m in masks_per_frame if m.max() > 0)
    # Count individual trails (connected components) across all frames in this batch
    batch_trail_count = 0
    for m in masks_per_frame:
        if m.max() == 0:
            continue
        n_cc, _ = cv2.connectedComponents((m > 0).astype(np.uint8))
        batch_trail_count += max(0, n_cc - 1)  # subtract background

    print(f"  Step 1 complete - {trail_frames}/{n} frames have trails", flush=True)

    if masks_dir:
        import json as _json
        raw_masks_per_frame = raw_masks_all[core_start:core_end]
        corners_per_frame = corners_all[core_start:core_end]
        for fp, mask, raw_mask, frm_corners in zip(
                frame_files, masks_per_frame, raw_masks_per_frame, corners_per_frame):
            # PNG mask dumps are the dev (--save-masks) path only. The Red Trail
            # Map needs just the polygon JSON, so --save-detections skips the PNGs.
            if args.save_masks:
                robust_imwrite(masks_dir / (fp.stem + ".png"), mask)
                if raw_mask.max() > 0:
                    robust_imwrite(masks_dir / (fp.stem + "_raw.png"), raw_mask)
            if frm_corners:
                h_fr, w_fr = mask.shape
                polys_data = {
                    "frame": fp.stem,
                    "width": w_fr,
                    "height": h_fr,
                    "polygons": [{"id": idx, "corners": c}
                                 for idx, c in enumerate(frm_corners)],
                }
                (masks_dir / (fp.stem + "_polys.json")).write_text(
                    _json.dumps(polys_data))

    # ── Step 2: Repair ────────────────────────────────────────────────────
    sb = args.skip_boundary
    print(f"\nStep 2 - cleaning frames (skipping first/last {sb})", flush=True)

    # Build neighbor_masks aligned with frames_all so repair_frame can check
    # whether each neighbor frame has a trail at the component being repaired.
    # Boundary frames (before/after core) have no mask — leave as None.
    neighbor_masks = [None] * len(frames_all)
    for _j, _m in enumerate(masks_all):
        _abs = _j
        if 0 <= _abs < len(frames_all):
            neighbor_masks[_abs] = _m

    n_repaired = 0
    running_trail_total = 0
    total_trail = 0
    for i, (fp, img, mask) in enumerate(zip(frame_files, frames, masks_per_frame)):
        _crumb("CLEAN", i + 1, n, fp.name)
        trail_px = int((mask > 0).sum())
        total_trail += trail_px
        if trail_px > 0:
            n_cc, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
            trail_count = max(0, n_cc - 1)
        else:
            trail_count = 0
        trail_label = f"{trail_count} trail{'s' if trail_count != 1 else ''}"
        skip = (sb > 0) and (i < sb or i >= n - sb)

        if not skip:
            # Read THIS frame's own EXIF (capture date/time, exposure, lens, GPS) from its
            # source file and add only our Software comment + orientation tag. Per frame —
            # so cleaned frames never inherit a neighbor's capture time.
            frame_exif = _stamp_exif(_frame_exif_bytes(fp))
            if trail_px > 0:
                repair_dbg = {} if logger is not None else None
                _t0 = time.perf_counter()
                cleaned = repair_frame(img, mask, i + core_start,
                                       frames_all,
                                       neighbor_masks=neighbor_masks,
                                       polygon_segs=segs_all[i + core_start],
                                       combine=REPAIR_COMBINE,
                                       debug_out=repair_dbg)
                _rep_s = time.perf_counter() - _t0
                _tacc("repair_s", _rep_s)
                _tw = time.perf_counter()
                _write_output(fp.stem, cleaned, icc_profile=icc_profile, exif_bytes=frame_exif, dpi=dpi)
                _tacc("write_s", time.perf_counter() - _tw)
                n_repaired += 1
                if logger is not None:
                    repair_dbg["type"] = "repair"
                    repair_dbg["frame"] = fp.stem
                    repair_dbg["frame_idx"] = i + core_start
                    repair_dbg["trail_px"] = trail_px
                    repair_dbg["repair_sec"] = round(_rep_s, 3)
                    logger.log(repair_dbg)
            else:
                _tw = time.perf_counter()
                _write_output(fp.stem, img, icc_profile=icc_profile, exif_bytes=frame_exif, dpi=dpi)
                _tacc("write_s", time.perf_counter() - _tw)
                if logger is not None:
                    logger.log({"type": "repair", "frame": fp.stem,
                                "frame_idx": i + core_start,
                                "trail_px": 0, "components": []})

        print(f"  cleaning {i+1}/{n}: {fp.name} - {trail_label}", flush=True)
        running_trail_total += trail_count
        print(f"FRAME_TRAIL_COUNT: {running_trail_total}", flush=True)

    def _fmt_s(s):
        """Format a duration in seconds as a short human string for the timing table.

        Picks the most readable unit: "Nm SSs" for a minute or more, "N.Ns" for one
        second or more, and "Nms" for sub-second times.
        """
        m, sec = divmod(int(s), 60)
        if m:
            return f"{m}m {sec:02d}s"
        if s >= 1.0:
            return f"{s:.1f}s"
        return f"{s * 1000:.0f}ms"

    if _timing:
        _STEP_ORDER = [
            ("new_pipeline_s",          "detect (new pipeline)"),
            ("dp_tiled_inference_s",    "  tiled_inference (SAHI)"),
            ("dp_fit_polygons_s",       "  fit_polygons"),
            ("dp_seam_second_pass_s",   "  seam_second_pass"),
            ("dp_fallback_polys_s",     "  fallback_polys"),
            ("dp_link_gaps_s",          "  link_gaps"),
            ("dp_suppress_fp_s",        "  suppress_fp (pipeline)"),
            ("apply_sky_mask_s",  "apply_sky_mask"),
            ("filter_small_s",    "filter_small_comps"),
            ("second_scrub_s",    "second_scrub"),
            ("static_fp_s",          "static_fp_suppressor"),
            ("sfp_precompute_s",     "  sfp_precompute"),
            ("sfp_compare_s",        "  sfp_compare"),
            ("sfp_apply_s",          "  sfp_apply"),
            ("edge_rescue_s",        "edge_rescue"),
            ("repair_s",          "repair_frame"),
        ]
        print("\nTiming summary:")
        print(f"  {'Step':<26} {'Calls':>6}  {'Total':>8}  {'Avg':>8}")
        for _key, _label in _STEP_ORDER:
            if _key not in _timing:
                print(f"  {_label:<26} {'0':>6}  {'--':>8}  {'--':>8}")
                continue
            _cnt, _tot = _timing[_key]
            _avg = _tot / _cnt if _cnt else 0.0
            print(f"  {_label:<26} {_cnt:>6}  {_fmt_s(_tot):>8}  {_fmt_s(_avg):>8}")

    elapsed = time.time() - t_total
    mins, secs = divmod(int(elapsed), 60)
    per_frame = elapsed / n
    pf_m, pf_s = divmod(int(per_frame), 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    pf_str = f"{pf_m}m {pf_s}s" if pf_m else f"{pf_s}s"
    print(f"\nDone in {time_str}  ({pf_str}/frame)")
    print(f"  {n_repaired}/{n} frames repaired")
    print(f"  avg trail px/frame: {total_trail // n}")
    print(f"BATCH_TRAIL_COUNT: {batch_trail_count}", flush=True)
    print(f"BATCH_FRAME_COUNT: {n}", flush=True)
    _crumb("BATCHOK", extra=f"{n_repaired}/{n} cleaned")
    print(f"\nOutput: {output_dir}")

    if logger is not None:
        logger.log({
            "type":                 "summary",
            "input_dir":            str(input_dir),
            "output_dir":           str(output_dir),
            "batch_start":          args.start,
            "batch_size":           args.batch,
            "total_frames":         n,
            "trail_frames":         trail_frames,
            "total_trail_components": batch_trail_count,
            "frames_repaired":      n_repaired,
            "elapsed_sec":          round(elapsed, 1),
            "model":                str(args.model),
            "confidence":           args.confidence,
            "dilate":               args.dilate,
            "min_area":             args.min_area,
            "min_area_scaled":      min_area_scaled,
            "second_scrub":         args.second_scrub,
            # Per-step timing for the whole batch so the log shows where time went
            # (detection vs repair vs image write vs suppressor vs filters).
            "timing":               {k: {"calls": v[0], "total_s": round(v[1], 2),
                                         "avg_s": round(v[1] / v[0], 3) if v[0] else 0.0}
                                     for k, v in sorted(_timing.items())},
        })
        logger.close()


if __name__ == "__main__":
    main()
