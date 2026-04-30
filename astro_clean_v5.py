#!/usr/bin/env python3
import sys, os
# Apple Silicon: torchvision::nms is not implemented for the MPS (GPU)
# device in the PyTorch version we ship, so YOLO warmup crashes during
# inference for every Apple Silicon Mac user that lets the model run on
# MPS. PYTORCH_ENABLE_MPS_FALLBACK=1 tells PyTorch to silently use the
# CPU for ops that aren't implemented on MPS. Negligible perf hit on
# small ops like NMS, fixes the crash. Must be set BEFORE any torch
# import (including those pulled in by ultralytics / sahi).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
"""
astro_clean_v5.py — YOLO-based astrophotography airplane trail removal

ALGORITHM: Per-frame YOLO segmentation + temporal median repair

Pipeline:
  1. Detect trails per frame using YOLO/SAHI tiled inference.
     - Apply foreground/sky mask to suppress false positives.
     - Filter small components (preserve red nav lights).
  2. Repair: Star Bridge morph from neighbors, black fill fallback
     (black is transparent in lighten-max stacks).
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
    load_model, detect_frame, apply_sky_mask, filter_small_components
)
from modules.repair import repair_frame
from modules.io_safe import robust_imread, robust_imread_diag


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
    When a folder has both JPG and TIFF of the same frame, keep the TIFF.
    """
    if len(files) <= 1:
        return files

    # De-duplicate: if both foo.jpg and foo.tiff exist, keep the TIFF
    stems_seen = {}
    tif_exts = {'.tif', '.tiff'}
    for fp in files:
        stem = fp.stem
        ext = fp.suffix.lower()
        if stem in stems_seen:
            prev_ext = stems_seen[stem].suffix.lower()
            if ext in tif_exts and prev_ext not in tif_exts:
                stems_seen[stem] = fp
        else:
            stems_seen[stem] = fp
    deduped = sorted(stems_seen.values())
    n_dupes = len(files) - len(deduped)
    if n_dupes:
        print(f"  De-duplicated {n_dupes} file(s) (JPG+TIFF pairs -> kept TIFF)")
    files = deduped

    from PIL import Image as _PILImage

    def _hdr_size(fp):
        try:
            with _PILImage.open(str(fp)) as im:
                return im.size  # (w, h)
        except Exception:
            return None

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
                     expected_height: int = None) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)
    sliced = files[start:start + batch] if batch > 0 else files[start:]
    return _filter_by_resolution(sliced, expected_width, expected_height)


def load_with_neighbors(frame_dir: Path, start: int, batch: int,
                        expected_width: int = None,
                        expected_height: int = None):
    """Load batch frames plus one neighbor on each side for repair context.

    Returns (all_files, core_start, core_end) where all_files includes
    up to one extra frame before and after, and core_start/core_end
    mark the indices of the actual batch frames within all_files.
    """
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    all_sorted = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)
    total = len(all_sorted)

    end = start + batch if batch > 0 else total
    end = min(end, total)

    # Extend by one frame on each side if available
    ext_start = max(0, start - 1)
    ext_end = min(total, end + 1)

    sliced = all_sorted[ext_start:ext_end]
    sliced = _filter_by_resolution(sliced, expected_width, expected_height)

    core_start = start - ext_start
    core_end = core_start + (end - start)
    core_end = min(core_end, len(sliced))

    return sliced, core_start, core_end


def main():
    parser = argparse.ArgumentParser(
        description="astro_clean_v5 — YOLO-based airplane trail removal")
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch", type=int, default=20)
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
                        help="Save detection masks to output_dir/masks/")
    parser.add_argument("--output-format", choices=["jpg", "tif8", "tif16"],
                        default="jpg",
                        help="Output file format (default jpg)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality 60-100 (default 95)")
    parser.add_argument("--expected-width", type=int, default=None,
                        help="Expected image width — when provided, skips per-batch resolution detection")
    parser.add_argument("--expected-height", type=int, default=None,
                        help="Expected image height")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    def _write_output(stem: str, img: np.ndarray, icc_profile=None, exif_bytes=None, dpi=None):
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
        from PIL import Image
        if args.output_format == "jpg":
            out = img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb, mode="RGB")
            save_kwargs = {"quality": int(args.jpeg_quality), "subsampling": 0}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            if dpi:
                save_kwargs["dpi"] = dpi
            pil.save(str(cleaned_dir / (stem + ".jpg")), "JPEG", **save_kwargs)
        elif args.output_format == "tif8":
            out = img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb, mode="RGB")
            save_kwargs = {"compression": "tiff_deflate"}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            if dpi:
                save_kwargs["dpi"] = dpi
            pil.save(str(cleaned_dir / (stem + ".tif")), "TIFF", **save_kwargs)
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
            tiff_kwargs = {
                "photometric": "rgb",
                "compression": "deflate",
                "extratags": extratags,
            }
            if dpi:
                # tifffile expects (xres, yres) floats and a unit string.
                tiff_kwargs["resolution"] = (float(dpi[0]), float(dpi[1]))
                tiff_kwargs["resolutionunit"] = "inch"
            tifffile.imwrite(str(cleaned_dir / (stem + ".tif")), rgb, **tiff_kwargs)
    cleaned_dir = output_dir
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks" if args.save_masks else None
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)

    # ── Load foreground mask → invert to sky mask ─────────────────────────
    sky_mask = None
    fg_mask = None
    if args.foreground_mask:
        fg_mask = cv2.imread(args.foreground_mask, cv2.IMREAD_GRAYSCALE)
        if fg_mask is None:
            print(f"ERROR: cannot load foreground mask: {args.foreground_mask}")
            sys.exit(1)
        sky_mask = cv2.bitwise_not(fg_mask)
        print(f"  Applying sky mask")

    # ── Load frames ───────────────────────────────────────────────────────
    frame_files_all, core_start, core_end = load_with_neighbors(
        input_dir, args.start, args.batch,
        args.expected_width, args.expected_height)
    frame_files = frame_files_all[core_start:core_end]  # core batch files
    n = len(frame_files)
    n_all = len(frame_files_all)
    if n < 3:
        print(f"ERROR: need >= 3 frames (got {n})")
        sys.exit(1)

    # Grab ICC profile + EXIF from the first core frame so output inherits color
    # profile (Adobe RGB, ProPhoto, etc.) and camera metadata instead of being
    # tagged as raw sRGB.
    icc_profile = None
    exif_bytes = None
    dpi = None
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(str(frame_files_all[core_start])) as _meta_im:
            icc_profile = _meta_im.info.get("icc_profile")
            exif_bytes = _meta_im.info.get("exif")
            dpi = _meta_im.info.get("dpi")
    except Exception as _e:
        print(f"  WARN: could not read color profile ({_e})")

    # Build the Software-tag stamp that goes into every cleaned file's EXIF.
    # Format: "Star Trail CleanR v<app> / Trail Detector v<model> / www.startrailcleanr.com"
    def _resolve_app_version():
        try:
            base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base, "version.txt")) as vf:
                return vf.read().strip()
        except Exception:
            return "?"

    def _resolve_model_version():
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
        """Return EXIF bytes with our stamp in three places for max viewer
        compatibility. Preserves all other EXIF unchanged.
            0x010E ImageDescription — shown as "Description/Caption" in Preview, Photoshop, Lightroom
            0x0131 Software         — shown in EXIF viewers, Lightroom, Photoshop Origin panel
            0x9C9C XPComment        — shown in Windows Explorer "Comments" column (UTF-16LE encoded)
        """
        try:
            from PIL import Image as _PILImage
            ex = _PILImage.Exif()
            if source_bytes:
                ex.load(source_bytes)
            ex[0x010E] = _stamp  # ImageDescription (ASCII)
            ex[0x0131] = _stamp  # Software (ASCII)
            ex[0x9C9C] = _stamp.encode('utf-16le') + b'\x00\x00'  # XPComment (UTF-16LE, null-terminated)
            return ex.tobytes()
        except Exception:
            return source_bytes

    exif_bytes = _stamp_exif(exif_bytes)

    print(f"Loading {n} frames...")
    frames_all = []
    files_kept = []
    skipped = []
    skipped_before_core = 0
    skipped_in_core = 0
    for fi, fp in enumerate(frame_files_all):
        is_core = core_start <= fi < core_end
        is_before_core = fi < core_start
        if is_core:
            core_pos = fi - core_start + 1
            print(f"  loading {core_pos}/{n}: {fp.name}", flush=True)

        img, diag = robust_imread_diag(fp, cv2.IMREAD_UNCHANGED)
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

        # JPEGs may have EXIF rotation tags. IMREAD_UNCHANGED ignores them,
        # but SAHI applies them → mask/frame orientation mismatch.
        # Re-read with IMREAD_COLOR to get EXIF rotation, if it changes shape.
        if fp.suffix.lower() in {'.jpg', '.jpeg'}:
            img_exif = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img_exif is not None and img_exif.shape[:2] != img.shape[:2]:
                img = img_exif
        frames_all.append(img)
        files_kept.append(fp)

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

    dtypes = {str(f.dtype) for f in frames_all}
    if len(dtypes) > 1:
        print("\nERROR: this folder mixes 8-bit and 16-bit images "
              "(for example, both .jpg and .tif copies of the same photos). "
              "Move one set into a different folder so every frame is the "
              "same format, then try again.")
        sys.exit(1)

    # 16-bit handling
    is_16bit = frames_all[0].dtype == np.uint16
    if is_16bit:
        frames_8bit_all = [(f >> 8).astype(np.uint8) for f in frames_all]
        frames_8bit = frames_8bit_all[core_start:core_end]
    else:
        frames_8bit_all = frames_all
        frames_8bit = frames

    if fg_mask is not None:
        from modules.hot_pixels import build_hot_pixel_map

        hot_map = None
        if args.hot_pixel_map and os.path.isfile(args.hot_pixel_map):
            hot_map = cv2.imread(args.hot_pixel_map, cv2.IMREAD_GRAYSCALE)
        if hot_map is None:
            hot_map = build_hot_pixel_map(frames_8bit)
            n_defective = int((hot_map > 0).sum())
            if args.hot_pixel_map and n_defective > 0:
                cv2.imwrite(args.hot_pixel_map, hot_map)

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

    # Resolution scaling for min_area filter
    REF_PIXELS = 5472 * 3648
    sc_area = (w * h) / REF_PIXELS
    min_area_scaled = max(args.min_area, int(args.min_area * sc_area))

    t_total = time.time()

    # ── Step 1: Detect trails (YOLO) ────────────────────────────────────
    print("\nStep 1 - detecting trails", flush=True)
    print("  Loading AI trail detector...", flush=True)
    model = load_model(str(args.model), args.confidence, args.device)

    masks_all = []
    for i, fp in enumerate(frame_files_all):
        mask = detect_frame(model, frames_8bit_all[i], args.tile_size,
                            args.overlap, args.dilate)
        if mask is None:
            masks_all.append(np.zeros((h, w), dtype=np.uint8))
            continue

        if sky_mask is not None:
            mask = apply_sky_mask(mask, sky_mask)

        if min_area_scaled > 0 and mask.max() > 0:
            mask = filter_small_components(mask, frames_8bit_all[i], min_area_scaled)

        masks_all.append(mask)
        if mask.max() > 0:
            n_cc, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
            trail_count = max(0, n_cc - 1)
        else:
            trail_count = 0
        trail_label = f"{trail_count} trail{'s' if trail_count != 1 else ''}"
        is_neighbor = i < core_start or i >= core_end
        if is_neighbor:
            print(f"  detecting neighbor: {fp.name} - {trail_label}", flush=True)
        else:
            core_num = i - core_start + 1
            print(f"  detecting {core_num}/{n}: {fp.name} - {trail_label}", flush=True)

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
        for fp, mask in zip(frame_files, masks_per_frame):
            cv2.imwrite(str(masks_dir / (fp.stem + ".png")), mask)

    # ── Step 2: Repair ────────────────────────────────────────────────────
    sb = args.skip_boundary
    print(f"\nStep 2 - repairing frames (skipping first/last {sb})", flush=True)

    n_repaired = 0
    total_trail = 0
    for i, (fp, img, mask) in enumerate(zip(frame_files, frames, masks_per_frame)):
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
            if trail_px > 0:
                # Use i + core_start as index into the full (with-neighbors) arrays
                cleaned = repair_frame(img, mask, i + core_start,
                                       frames_all)
                _write_output(fp.stem, cleaned, icc_profile=icc_profile, exif_bytes=exif_bytes, dpi=dpi)
                n_repaired += 1
            else:
                _write_output(fp.stem, img, icc_profile=icc_profile, exif_bytes=exif_bytes, dpi=dpi)

        print(f"  repairing {i+1}/{n}: {fp.name} - {trail_label}", flush=True)

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
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
