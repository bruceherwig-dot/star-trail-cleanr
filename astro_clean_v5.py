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
    load_model, detect_frame_polygon, apply_sky_mask, filter_small_components
)
from modules.trail_grouper import detection_props
from modules.repair import repair_frame
from modules.run_logger import RunLogger
from modules.io_safe import robust_imread, robust_imread_diag, robust_imwrite


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
                         timing_out=None):
    """Remove detection components that are static false positives.

    PURPOSE: The AI sometimes detects fixed foreground objects (building edges,
    rooflines, fence posts) as airplane trails because they are long, thin, and
    high-contrast against the sky. These false positives appear at the SAME pixel
    position in every frame. Real airplane trails appear in at most 1-2 frames
    and move significantly between frames.

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
    # Precompute boolean "hit" mask for every frame (polygon mask OR raw SAHI,
    # whichever is available) so the (mask > 0) conversion is not repeated for
    # every component comparison.  Building this once covers all 8 neighbor
    # lookups per component without re-allocating arrays.
    _t0_pre = time.perf_counter()
    _nb_hits    = {}   # frame_idx -> np.ndarray bool, or None
    _nb_sources = {}   # frame_idx -> "polygon" | "raw_sahi"
    for _fi in range(len(masks_all)):
        _m   = masks_all[_fi]
        _hit = (_m > 0) if (_m is not None and _m.max() > 0) else None
        _src = "polygon"
        if _hit is None and raw_masks_all is not None:
            _r = raw_masks_all[_fi] if _fi < len(raw_masks_all) else None
            if _r is not None and _r.max() > 0:
                _hit = (_r > 0)
                _src = "raw_sahi"
        _nb_hits[_fi]    = _hit
        _nb_sources[_fi] = _src
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

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8))

        to_suppress = np.zeros(mask.shape, dtype=bool)

        for comp_id in range(1, n_labels):
            comp_pixels = (labels == comp_id)
            comp_area = int(comp_pixels.sum())
            if comp_area == 0:
                continue

            # Centroid and bounding box for log readability.
            ys, xs = np.where(comp_pixels)
            cx = int(xs.mean()); cy = int(ys.mean())
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            # Frame-edge veto: trails go off the page -- static foreground
            # objects do not. If the bbox touches any image edge within 20px,
            # this is a trail entering or exiting the frame, not a static FP.
            h_mask, w_mask = mask.shape
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
                # Use precomputed boolean mask for this neighbor frame.
                # Falls back automatically to raw SAHI via _nb_sources if no
                # polygon mask was available (already resolved at precompute time).
                nb_hit = _nb_hits.get(ni)
                if nb_hit is None:
                    continue
                source = _nb_sources[ni]
                # Spatial fast-path: slice to the component's bounding box before
                # any pixel operation.  All component pixels lie within [y1:y2+1,
                # x1:x2+1] by definition, so the bbox AND gives the same result as
                # a full-frame AND while touching ~5 000x fewer pixels on a typical
                # 6000x4000 frame with a 400x12 trail component.
                bbox_nb = nb_hit[y1:y2 + 1, x1:x2 + 1]
                if not bbox_nb.any():
                    continue  # no neighbor pixels in this region -- skip instantly
                bbox_comp = comp_pixels[y1:y2 + 1, x1:x2 + 1]
                intersection = int((bbox_comp & bbox_nb).sum())
                if intersection == 0:
                    continue
                local_nb_area = int(bbox_nb.sum())
                union = comp_area + local_nb_area - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    # Centroid motion veto: if the neighbor detection in this
                    # bounding box has its center far from the component center,
                    # it is a different object at a different position -- not a
                    # static repeat of the same object (e.g. a large trail from
                    # a different airplane passing through the same area).
                    nb_in_bbox = bbox_nb  # reuse already-sliced region
                    if nb_in_bbox.any():
                        ny, nx = np.where(nb_in_bbox)
                        nb_cx_f = float(nx.mean()) + x1
                        nb_cy_f = float(ny.mean()) + y1
                        centroid_dist = float(np.sqrt((cx - nb_cx_f) ** 2 + (cy - nb_cy_f) ** 2))
                        if centroid_dist > _CENTROID_MOTION_PX:
                            continue  # different object -- skip this neighbor
                    matched_neighbors.append({
                        "frame_idx": ni,
                        "source": source,
                        "iou_pct": round(iou * 100, 1),
                        "local_nb_area": local_nb_area,
                    })

            if len(matched_neighbors) >= min_matches:
                # Bright-trail veto: if pixels inside the component are
                # significantly brighter than the surrounding sky, it is a
                # real trail (nav light, strobe, or bright streak) -- keep it
                # regardless of the neighbor match count.
                if frames_all is not None and i < len(frames_all):
                    is_bright, bright_ratio = _is_bright_trail(comp_pixels, frames_all[i])
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

                to_suppress |= comp_pixels
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
    _t0_apply = time.perf_counter()
    suppressed_count = 0
    for i, to_suppress in suppress_maps.items():
        masks_all[i][to_suppress] = 0
        n_cc, _ = cv2.connectedComponents(to_suppress.astype(np.uint8))
        suppressed_count += max(0, n_cc - 1)
    if timing_out is not None:
        timing_out["sfp_apply_s"] = time.perf_counter() - _t0_apply

    if debug_out is not None:
        debug_out["suppressed_by_frame"] = debug_by_frame
        debug_out["kept_by_veto_by_frame"] = kept_by_veto

    return suppressed_count


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
                        help="Save detection masks to cleanr_workspace/masks/")
    parser.add_argument("--output-format", choices=["jpg", "tif8", "tif16"],
                        default="jpg",
                        help="Output file format (default jpg)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality 60-100 (default 95)")
    parser.add_argument("--expected-width", type=int, default=None,
                        help="Expected image width — when provided, skips per-batch resolution detection")
    parser.add_argument("--expected-height", type=int, default=None,
                        help="Expected image height")
    parser.add_argument("--second-scrub", action="store_true",
                        help="Run detection a second time on each frame rotated 180°, merging any new trails found")
    args = parser.parse_args()

    # Dev flag: auto-enable mask saving when ~/.star_trail_cleanr/.dev_save_masks exists
    if not args.save_masks and (Path.home() / ".star_trail_cleanr" / ".dev_save_masks").exists():
        args.save_masks = True

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # ── Dev-only run logger ───────────────────────────────────────────────────
    # Written to {input_dir}/cleanr_workspace/run_log_{timestamp}.jsonl.
    # Dev-only: sys.frozen is True in the frozen bundle, so users never get this file.
    _is_dev = not getattr(sys, "frozen", False)
    if _is_dev:
        _ws_dir = input_dir / "cleanr_workspace"
        _ws_dir.mkdir(parents=True, exist_ok=True)
        _log_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger = RunLogger(str(_ws_dir / f"run_log_{_log_ts}.jsonl"))
    else:
        logger = None

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
            tiff_kwargs = {
                "photometric": "rgb",
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
    masks_dir = input_dir / "cleanr_workspace" / "masks" if args.save_masks else None
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
            dpi = _meta_im.info.get("dpi")
            # info.get("exif") works for JPEGs but returns None for TIFFs.
            # getexif().tobytes() works for both formats.
            try:
                _exif_obj = _meta_im.getexif()
                exif_bytes = _exif_obj.tobytes() if _exif_obj else None
            except Exception:
                exif_bytes = _meta_im.info.get("exif")
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
        """Return EXIF bytes with Software stamp added. Preserves all source EXIF unchanged."""
        try:
            from PIL import Image as _PILImage
            ex = _PILImage.Exif()
            if source_bytes:
                ex.load(source_bytes)
            ex[0x0131] = _stamp  # Software tag — shown in Lightroom, Photoshop, and EXIF viewers
            return ex.tobytes()
        except Exception:
            return source_bytes

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
            img_exif = robust_imread(fp, cv2.IMREAD_COLOR)
            if img_exif is not None and img_exif.shape[:2] != img.shape[:2]:
                img = img_exif

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
            hot_map = robust_imread(args.hot_pixel_map, cv2.IMREAD_GRAYSCALE)
        if hot_map is None:
            hot_map = build_hot_pixel_map(frames_8bit)
            n_defective = int((hot_map > 0).sum())
            if args.hot_pixel_map and n_defective > 0:
                robust_imwrite(args.hot_pixel_map, hot_map)

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
    raw_masks_all = []
    edge_candidates_all = []  # per-frame lists of edge-touching detections that failed elongation
    detect_infos = []  # buffered per-frame detect data; written after static FP pass
    running_trail_total = 0

    _timing = {}

    def _tacc(key, elapsed):
        if key in _timing:
            _timing[key][0] += 1
            _timing[key][1] += elapsed
        else:
            _timing[key] = [1, elapsed]

    for i, fp in enumerate(frame_files_all):
        is_neighbor = i < core_start or i >= core_end
        dbg = {} if logger is not None else None

        edge_cands = []
        _ft = {}
        result = detect_frame_polygon(model, frames_8bit_all[i], args.tile_size,
                                      args.overlap, args.dilate,
                                      return_raw=True,
                                      debug_out=dbg,
                                      edge_candidates_out=edge_cands,
                                      sky_mask=sky_mask,
                                      timing_out=_ft)
        for _k, _v in _ft.items():
            _tacc(_k, _v)
        edge_candidates_all.append(edge_cands)
        mask, raw_labeled = result
        if mask is None:
            masks_all.append(np.zeros((h, w), dtype=np.uint8))
            raw_masks_all.append(np.zeros((h, w), dtype=np.uint8))
            if dbg is not None:
                dbg.update({"frame": fp.stem, "frame_idx": i,
                            "is_neighbor": is_neighbor,
                            "detect_error": "detect_frame_polygon returned None"})
            detect_infos.append(dbg)
            continue

        sky_px_removed = 0
        if sky_mask is not None:
            before_px = int((mask > 0).sum())
            _t0 = time.perf_counter()
            mask = apply_sky_mask(mask, sky_mask)
            _tacc("apply_sky_mask_s", time.perf_counter() - _t0)
            sky_px_removed = before_px - int((mask > 0).sum())

        small_dbg = {} if dbg is not None else None
        if min_area_scaled > 0 and mask.max() > 0:
            _t0 = time.perf_counter()
            mask = filter_small_components(mask, frames_8bit_all[i], min_area_scaled,
                                           debug_out=small_dbg)
            _tacc("filter_small_s", time.perf_counter() - _t0)

        masks_all.append(mask)
        raw_masks_all.append(raw_labeled if raw_labeled is not None
                             else np.zeros((h, w), dtype=np.uint8))
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
        print("\nStep 1b - second scrub (180-degree rotation)", flush=True)
        try:
            for i, fp in enumerate(frame_files_all):
                rotated = np.rot90(frames_8bit_all[i], 2)
                _t0 = time.perf_counter()
                mask2 = detect_frame_polygon(model, rotated, args.tile_size, args.overlap, args.dilate,
                                         sky_mask=np.rot90(sky_mask, 2) if sky_mask is not None else None)
                _tacc("second_scrub_s", time.perf_counter() - _t0)
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

    print("\nStep 1c - removing static false positives...", flush=True)
    static_fp_dbg = {} if logger is not None else None
    _sfp_timing = {}
    _t0 = time.perf_counter()
    n_static = _suppress_static_fps(masks_all, core_start, core_end,
                                    raw_masks_all=raw_masks_all,
                                    frames_all=frames_all,
                                    debug_out=static_fp_dbg,
                                    timing_out=_sfp_timing)
    _tacc("static_fp_s", time.perf_counter() - _t0)
    for _k, _v in _sfp_timing.items():
        _tacc(_k, _v)
    if n_static:
        print(f"  Suppressed {n_static} static detection(s) present in 2+ neighboring frames",
              flush=True)
    else:
        print("  No static false positives found", flush=True)

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
        for _i, _dbg in enumerate(detect_infos):
            if _dbg is None:
                continue
            _m = masks_all[_i] if _i < len(masks_all) else None
            if _m is not None and _m.max() > 0:
                _ncc, _ = cv2.connectedComponents((_m > 0).astype(np.uint8))
                _dbg["final_trail_components"] = max(0, _ncc - 1)
            else:
                _dbg["final_trail_components"] = 0
            _dbg["static_fp_suppressed"]    = _sfp_by_frame.get(_i, [])
            _dbg["static_fp_kept_by_veto"]  = _veto_by_frame.get(_i, [])
            _dbg["type"] = "detect"
            logger.log(_dbg)

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
        raw_masks_per_frame = raw_masks_all[core_start:core_end]
        for fp, mask, raw_mask in zip(frame_files, masks_per_frame, raw_masks_per_frame):
            robust_imwrite(masks_dir / (fp.stem + ".png"), mask)
            if raw_mask.max() > 0:
                robust_imwrite(masks_dir / (fp.stem + "_raw.png"), raw_mask)

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
                repair_dbg = {} if logger is not None else None
                _t0 = time.perf_counter()
                cleaned = repair_frame(img, mask, i + core_start,
                                       frames_all,
                                       neighbor_masks=neighbor_masks,
                                       debug_out=repair_dbg)
                _tacc("repair_s", time.perf_counter() - _t0)
                _write_output(fp.stem, cleaned, icc_profile=icc_profile, exif_bytes=exif_bytes, dpi=dpi)
                n_repaired += 1
                if logger is not None:
                    repair_dbg["type"] = "repair"
                    repair_dbg["frame"] = fp.stem
                    repair_dbg["frame_idx"] = i + core_start
                    repair_dbg["trail_px"] = trail_px
                    logger.log(repair_dbg)
            else:
                _write_output(fp.stem, img, icc_profile=icc_profile, exif_bytes=exif_bytes, dpi=dpi)
                if logger is not None:
                    logger.log({"type": "repair", "frame": fp.stem,
                                "frame_idx": i + core_start,
                                "trail_px": 0, "components": []})

        print(f"  cleaning {i+1}/{n}: {fp.name} - {trail_label}", flush=True)
        running_trail_total += trail_count
        print(f"FRAME_TRAIL_COUNT: {running_trail_total}", flush=True)

    def _fmt_s(s):
        m, sec = divmod(int(s), 60)
        if m:
            return f"{m}m {sec:02d}s"
        if s >= 1.0:
            return f"{s:.1f}s"
        return f"{s * 1000:.0f}ms"

    if _timing:
        _STEP_ORDER = [
            ("sahi_s",            "sahi_inference"),
            ("sky_mask_s",        "sky_mask_zero"),
            ("raw_labeled_s",     "raw_labeled_mask"),
            ("try_split_s",       "try_split"),
            ("elongation_s",      "elongation_filter"),
            ("edge_cand_s",       "edge_candidates"),
            ("group_s",           "group_detections"),
            ("bridge_s",          "gap_bridge"),
            ("bridge_rot90_s",     "gap_bridge_rot90"),
            ("poly_fit_s",        "poly_fit"),
            ("dilate_s",          "dilation"),
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
        })
        logger.close()


if __name__ == "__main__":
    main()
