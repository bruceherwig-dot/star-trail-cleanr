#!/usr/bin/env python3
import sys
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
        print(f"  De-duplicated {n_dupes} file(s) (JPG+TIFF pairs \u2192 kept TIFF)")
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

    def _write_output(stem: str, img: np.ndarray, icc_profile=None, exif_bytes=None):
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
            pil.save(str(cleaned_dir / (stem + ".tif")), "TIFF", **save_kwargs)
        else:  # tif16
            if img.dtype == np.uint16:
                out = img
            else:
                out = img.astype(np.uint16) * 257
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb, mode="RGB;16")
            save_kwargs = {"compression": "tiff_deflate"}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            pil.save(str(cleaned_dir / (stem + ".tif")), "TIFF", **save_kwargs)
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
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(str(frame_files_all[core_start])) as _meta_im:
            icc_profile = _meta_im.info.get("icc_profile")
            exif_bytes = _meta_im.info.get("exif")
    except Exception as _e:
        print(f"  WARN: could not read color profile ({_e})")

    print(f"Loading {n} frames...")
    frames_all = []
    for fi, fp in enumerate(frame_files_all):
        if core_start <= fi < core_end:
            core_pos = fi - core_start + 1
            print(f"  loading {core_pos}/{n}: {fp.name}", flush=True)
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"\nERROR: cannot read {fp.name}")
            sys.exit(1)
        # JPEGs may have EXIF rotation tags. IMREAD_UNCHANGED ignores them,
        # but SAHI applies them → mask/frame orientation mismatch.
        # Re-read with IMREAD_COLOR to get EXIF rotation, if it changes shape.
        if fp.suffix.lower() in {'.jpg', '.jpeg'}:
            img_exif = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img_exif is not None and img_exif.shape[:2] != img.shape[:2]:
                img = img_exif
        frames_all.append(img)
    frames = frames_all[core_start:core_end]  # core batch frames
    h, w = frames[0].shape[:2]
    print(f"  {n} frames loaded ({w}\u00d7{h})")

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
                    orig = cv2.imread(str(frame_files[i]), cv2.IMREAD_UNCHANGED)
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
    print("\nStep 1 \u2014 detecting trails", flush=True)
    print("  Loading AI trail detector...", flush=True)
    model = load_model(str(args.model), args.confidence, args.device)

    masks_all = []
    for i, fp in enumerate(frame_files_all):
        mask = detect_frame(model, str(fp), args.tile_size,
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
            print(f"  detecting neighbor: {fp.name} \u2014 {trail_label}", flush=True)
        else:
            core_num = i - core_start + 1
            print(f"  detecting {core_num}/{n}: {fp.name} \u2014 {trail_label}", flush=True)

    masks_per_frame = masks_all[core_start:core_end]
    trail_frames = sum(1 for m in masks_per_frame if m.max() > 0)
    # Count individual trails (connected components) across all frames in this batch
    batch_trail_count = 0
    for m in masks_per_frame:
        if m.max() == 0:
            continue
        n_cc, _ = cv2.connectedComponents((m > 0).astype(np.uint8))
        batch_trail_count += max(0, n_cc - 1)  # subtract background
    print(f"  Step 1 complete \u2014 {trail_frames}/{n} frames have trails", flush=True)

    if masks_dir:
        for fp, mask in zip(frame_files, masks_per_frame):
            cv2.imwrite(str(masks_dir / (fp.stem + ".png")), mask)

    # ── Step 2: Repair ────────────────────────────────────────────────────
    sb = args.skip_boundary
    print(f"\nStep 2 \u2014 repairing frames (skipping first/last {sb})", flush=True)

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
                _write_output(fp.stem, cleaned, icc_profile=icc_profile, exif_bytes=exif_bytes)
                n_repaired += 1
            else:
                _write_output(fp.stem, img, icc_profile=icc_profile, exif_bytes=exif_bytes)

        print(f"  repairing {i+1}/{n}: {fp.name} \u2014 {trail_label}", flush=True)

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
