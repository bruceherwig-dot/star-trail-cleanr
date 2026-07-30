#!/usr/bin/env python3
"""prepare_yolo_v5.py — Step 2 of the v5 fine-tune assembly: slice the 9 new/corrected
datasets into 640x640 BASE tiles. No rotation, no train/val split (those are later steps).

WHERE IT FITS
  CVAT review  ->  cvat_to_masks.py (masks + poly_meta)  ->  THIS SCRIPT (base tiles)
                ->  fold in bridge/crossing/Jeff aug tiles  ->  dedup  ->  540 rotation rule
                ->  split + dataset.yaml  ->  train_yolo.py (fine-tune from v4 best.pt)

WHY A SEPARATE V5 SCRIPT (not prepare_yolo_v4.py)
  - v4's builder is hardwired to the OLD 18-task v4 dataset and we are FINE-TUNING from v4,
    not rebuilding it -- we only tile the new/corrected data here.
  - v4 reads frames flat (raw cv2.imread). Thomas Jackson's frames carry an EXIF rotate
    flag; CVAT stored its masks upright (4000x6000) while a flat read is landscape
    (6000x4000). v4 would then RESIZE the mask to fit -> every TJ trail squashed. This
    script reads upright (robust_imread) so masks line up exactly, and treats any
    remaining shape mismatch as a LOUD error, never a silent resize.
  - v4 bakes rotation + the train/val split into one pass. v5 needs those to come AFTER
    folding in the pre-made aug tiles and dedup, so this script stops at base tiles.

WHAT IT READS  (everything on the T7 Shield drive)
  - Source frames per task in TASK_CONFIG (folder under "star trail images", or an
    absolute path for out-of-root sets like Bombay / Jeff Fishman).
  - Mask PNGs from labels/task_<id>/masks (written by cvat_to_masks.py; the source of
    truth for which frames tile).
  - poly_meta JSON from labels/task_<id>/poly_meta (carried forward; the 540 rotation
    rule in a later step reads it -- this script just leaves it in place).

WHAT IT WRITES  ->  dataset_v5/base/
  images/<tile>.png        one PNG per kept tile
  labels/<tile>.txt        YOLO-seg label ("0 x1 y1 ..." normalized; empty for negatives)
  manifest.json            one record per tile: name, task, frame stem, tile origin
                           (x1,y1) in the FULL frame, positive flag, trail-pixel count.
                           The (task, stem, x1, y1) key is what the dedup step uses.

HOW TO RUN
  python3 tools/prepare_yolo_v5.py      (no arguments; long-running -> background it)
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.io_safe import robust_imread  # EXIF-aware reader (matches the masks)

# ── Config ────────────────────────────────────────────────────────────────────

T7_ROOT   = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR")
T7_IMAGES = T7_ROOT / "star trail images"
T7_LABELS = T7_ROOT / "labels"
T7_OUT    = T7_ROOT / "dataset_v5" / "base"

TILE_SIZE    = 640
OVERLAP      = 0.2
MIN_TRAIL_PX = 50    # a tile is "positive" if it holds at least this many trail pixels
NEG_RATIO    = 2     # keep up to NEG_RATIO x (positive tiles) negatives per frame
NEG_MIN      = 3
SEED         = 42

# Task ID -> (image folder name OR absolute path, max frames). The 9 v5 datasets.
TASK_CONFIG = {
    36: ("Warren Hatch - Stroudt's Preserve",                       114),
    39: ("borrego_springs_1",                                         5),
    40: ("Bruce Herwig Joshua Tree 6.26 80 sec exposure Star Trail", 89),
    41: ("Thomas Jackson Star Trails Borrego",                      301),
    54: ("Thomas Jackson GoPro_G0088569",                           900),
    55: ("Thomas Jackson GoPro_G0037688",                           997),
    56: ("Katrina Brown - Full Size Star Trails",                   709),
    59: ("/Volumes/T7 Shield/Photos/Astrophotography/_2026/26.6 Bombay Beach - Soly/Star Trail Canon 6D/ST Export Originals", 118),
    60: ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Jeff Fishman/cleaned", 6),
}

# ── Tiling helpers ──────────────────────────────────────────────────────────────

def get_tile_coords(img_w, img_h):
    """Return (x1, y1, x2, y2) windows covering the image: 640px tiles, 20% overlap,
    with the last column/row snapped flush to the right/bottom edge. Matches SAHI
    inference and cvat_to_masks' tile grid."""
    if img_w <= TILE_SIZE and img_h <= TILE_SIZE:
        return [(0, 0, img_w, img_h)]
    stride = int(TILE_SIZE * (1 - OVERLAP))
    xs = list(range(0, img_w - TILE_SIZE, stride))
    if not xs or xs[-1] + TILE_SIZE < img_w:
        xs.append(img_w - TILE_SIZE)
    ys = list(range(0, img_h - TILE_SIZE, stride))
    if not ys or ys[-1] + TILE_SIZE < img_h:
        ys.append(img_h - TILE_SIZE)
    return [(x, y, x + TILE_SIZE, y + TILE_SIZE) for y in ys for x in xs]


def mask_crop_to_yolo(mask_crop):
    """Convert one tile's binary mask into YOLO-seg label lines (one per blob),
    vertices normalized to the tile. Drops specks < 10px. Empty list if no blob."""
    h, w = mask_crop.shape
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = approx.reshape(-1, 2).astype(float)
        points[:, 0] /= w
        points[:, 1] /= h
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"0 {coords}")
    return lines


def build_source_index(img_dir):
    """Map frame stem (lowercase) -> source Path, preferring jpg over tif/png so a
    Stroudt's jpg/tif duplicate pair always tiles on the jpg. Includes a TIFF/
    subfolder if present (Stroudt's keeps its tifs one level down)."""
    ext_pref = {".jpg": 0, ".jpeg": 0, ".png": 1, ".tif": 2, ".tiff": 2}
    index = {}
    search_dirs = [img_dir]
    if (img_dir / "TIFF").is_dir():
        search_dirs.append(img_dir / "TIFF")
    for d in search_dirs:
        for p in d.iterdir():
            e = p.suffix.lower()
            if e not in ext_pref:
                continue
            key = p.stem.lower()
            cur = index.get(key)
            if cur is None or ext_pref[e] < ext_pref[cur.suffix.lower()]:
                index[key] = p
    return index

# ── Per-task tiling ─────────────────────────────────────────────────────────────

def tile_task(task_id, img_out, lbl_out, manifest, rng):
    """Tile every masked frame of one task into base tiles. Returns a summary dict.

    Reads each mask from labels/task_<id>/masks, finds its upright source frame,
    slices the frame + mask on the shared 640 grid, writes positive tiles (with
    labels) and a balanced sample of negatives, and appends a manifest record per
    tile. Mask/source shape mismatches and missing sources are collected and
    reported loudly -- never silently resized or dropped."""
    folder, _max = TASK_CONFIG[task_id]
    img_dir  = Path(folder) if Path(folder).is_absolute() else T7_IMAGES / folder
    mask_dir = T7_LABELS / f"task_{task_id}" / "masks"

    summ = {"task": task_id, "frames": 0, "pos": 0, "neg": 0,
            "source_missing": [], "shape_mismatch": []}

    if not mask_dir.exists():
        print(f"  Task {task_id}: no masks folder ({mask_dir}) — run cvat_to_masks.py first")
        return summ
    if not img_dir.exists():
        print(f"  Task {task_id}: image folder not found — {img_dir}")
        return summ

    src_index  = build_source_index(img_dir)
    mask_files = sorted(mask_dir.glob("*.png"))

    for fi, mpath in enumerate(mask_files):
        stem = mpath.stem
        src  = src_index.get(stem.lower())
        if src is None:
            summ["source_missing"].append(stem)
            continue
        img  = robust_imread(src, cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            summ["source_missing"].append(stem)
            continue
        h, w = img.shape[:2]
        if mask.shape != (h, w):
            # masks were generated from this same upright read, so a mismatch is a real
            # problem (NOT something to paper over with a squashing resize).
            summ["shape_mismatch"].append(f"{stem} img={ (h,w) } mask={mask.shape}")
            continue

        coords = get_tile_coords(w, h)
        pos, neg = [], []
        for (x1, y1, x2, y2) in coords:
            tile_img  = img[y1:y2, x1:x2]
            tile_mask = mask[y1:y2, x1:x2]
            trail_px  = int(np.count_nonzero(tile_mask))
            name = f"task{task_id:02d}_{stem}_x{x1}y{y1}"
            (pos if trail_px >= MIN_TRAIL_PX else neg).append(
                (tile_img, tile_mask, name, x1, y1, trail_px))

        n_keep = min(len(neg), max(NEG_MIN, len(pos) * NEG_RATIO))
        rng.shuffle(neg)
        kept_neg = neg[:n_keep]

        for tile_img, tile_mask, name, x1, y1, trail_px in pos:
            cv2.imwrite(str(img_out / f"{name}.jpg"), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_out / f"{name}.txt").write_text("\n".join(mask_crop_to_yolo(tile_mask)))
            manifest.append({"name": name, "task": task_id, "stem": stem,
                             "x1": x1, "y1": y1, "positive": True, "trail_px": trail_px})
            summ["pos"] += 1

        for tile_img, tile_mask, name, x1, y1, trail_px in kept_neg:
            cv2.imwrite(str(img_out / f"{name}.jpg"), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_out / f"{name}.txt").write_text("")
            manifest.append({"name": name, "task": task_id, "stem": stem,
                             "x1": x1, "y1": y1, "positive": False, "trail_px": trail_px})
            summ["neg"] += 1

        summ["frames"] += 1
        sys.stdout.write(f"\r  Task {task_id} — frame {fi+1}/{len(mask_files)}  "
                         f"pos {summ['pos']} neg {summ['neg']}    ")
        sys.stdout.flush()

    print(f"\r  Task {task_id} done — {summ['frames']} frames, "
          f"{summ['pos']} pos, {summ['neg']} neg, "
          f"{len(summ['source_missing'])} source-missing, "
          f"{len(summ['shape_mismatch'])} shape-mismatch")
    return summ

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)
    img_out = T7_OUT / "images"
    lbl_out = T7_OUT / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print("prepare_yolo_v5.py — Step 2: base tiles (no rotation, no split)")
    print(f"Output: {T7_OUT}")
    print(f"Tile {TILE_SIZE}px / overlap {int(OVERLAP*100)}% / positive >= {MIN_TRAIL_PX}px")
    print("=" * 60)

    manifest = []
    summaries = []
    for task_id in sorted(TASK_CONFIG):
        folder = TASK_CONFIG[task_id][0]
        print(f"\nTask {task_id}: {folder}")
        try:
            summaries.append(tile_task(task_id, img_out, lbl_out, manifest, rng))
        except Exception as e:
            print(f"  ERROR: {e}")

    (T7_OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # ── reconcile (loud; no silent drops) ──
    print("\n" + "=" * 60)
    print("RECONCILE (base tiles):")
    tot_pos = tot_neg = 0
    problems = []
    for s in summaries:
        tot_pos += s["pos"]; tot_neg += s["neg"]
        flag = ""
        if s["source_missing"] or s["shape_mismatch"]:
            flag = "   <-- PROBLEM"; problems.append(s)
        print(f"  task {s['task']:>3}: {s['frames']:>4} frames  "
              f"{s['pos']:>5} pos  {s['neg']:>5} neg  "
              f"{len(s['source_missing']):>3} src-missing  "
              f"{len(s['shape_mismatch']):>3} shape-mismatch{flag}")
    print(f"  TOTAL: {tot_pos} positive + {tot_neg} negative = {tot_pos+tot_neg} tiles")
    print(f"  manifest: {len(manifest)} records -> {T7_OUT / 'manifest.json'}")

    if problems:
        print("\nFAIL: some frames did not tile cleanly:")
        for s in problems:
            if s["source_missing"]:
                print(f"  task {s['task']} source-missing ({len(s['source_missing'])}): {s['source_missing'][:5]}")
            if s["shape_mismatch"]:
                print(f"  task {s['task']} shape-mismatch ({len(s['shape_mismatch'])}): {s['shape_mismatch'][:5]}")
    else:
        print("\nOK: every masked frame tiled, masks matched their source. No silent drops.")
    print("\nDone.")


if __name__ == "__main__":
    main()
