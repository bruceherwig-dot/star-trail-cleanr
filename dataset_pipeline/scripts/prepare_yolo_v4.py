#!/usr/bin/env python3
"""prepare_yolo_v4.py — Build the rotation-augmented YOLO training dataset for Trail DetectoR v4.

WHAT THIS IS
  A maintained step in the training-data pipeline. It turns Bruce's reviewed CVAT
  annotations (already converted to per-frame mask PNGs + poly_meta sidecars by
  tools/cvat_to_masks.py) into a ready-to-train YOLO segmentation dataset: full-res
  frames sliced into 640x640 tiles, with extra rotated copies of the "hard" trail
  tiles so the detector sees trails at more orientations.

HOW TO RUN
  python3 tools/prepare_yolo_v4.py
  No arguments. All inputs and the output location are hard-coded in the Config block
  below (everything lives on the T7 Shield drive). When it finishes it prints the exact
  train command (tools/train_yolo.py ...) to run next.

WHERE IT FITS IN THE PIPELINE
  CVAT review  →  cvat_to_masks.py (masks + poly_meta)  →  THIS SCRIPT (tiles + aug)
                                                          →  train_yolo.py  →  best.pt

WHAT IT READS
  - Per-task source frames listed in TASK_CONFIG, from "star trail images/<folder>".
  - Per-task mask PNGs from "labels/task_<id>/masks" (one white-on-black mask per frame).
  - Per-task poly_meta JSON sidecars from "labels/task_<id>/poly_meta" (one per frame),
    which describe each annotated polygon (its angle, length, whether Bruce flagged it
    red, how many tiles it spans). These drive the rotation decision.
  - The gkyle external dataset, which ships as pre-cut 512px tiles with its own fixed
    train/val split and no poly_meta — so it gets no rotation augmentation.

ROTATION RULES (decided per frame, then applied to every positive tile in that frame)
  - Any polygon is_red OR spans tile_count >= 3  → add 90, 180, 270 degree copies.
  - Any polygon with long_axis > 1000px, by its angle:
      0-20 deg   → add 90 only
      20-70 deg  → add 90, 180, 270
      70-90 deg  → no extra rotations
  - Everything else → no extra rotations.

WHAT IT WRITES
  T7/dataset_tiled_v4/ in standard YOLO segmentation layout:
    images/train, images/val, labels/train, labels/val, plus dataset.yaml.

KNOWN GOTCHA (from session notes, 2026-06-08)
  This script reads masks from labels/task_<id>/masks. If cvat_to_masks.py only wrote a
  partial set of masks for a task, only those frames are picked up here — that was the
  root cause of the "Thomas Jackson 14-of-107" undercount. The mask folder is the source
  of truth for which frames make it into the dataset.
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

T7_ROOT   = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR")
T7_IMAGES = T7_ROOT / "star trail images"
T7_LABELS = T7_ROOT / "labels"
T7_OUT    = T7_ROOT / "dataset_tiled_v4"

TILE_SIZE     = 640
OVERLAP       = 0.2
MIN_TRAIL_PX  = 50
NEG_RATIO     = 2
NEG_MIN       = 3
VAL_FRACTION  = 0.2
SEED          = 42

# Task ID → (image folder name, max frames)
TASK_CONFIG = {
    1:  ("Bruce Herwig - Joshua Tree - Juniper and Monolith", 400),
    2:  ("Bruce Herwig - first star trail data",              135),
    5:  ("Silvana Della Camera - Tree and Trails",            251),
    8:  ("Bruce Herwig - Borrego - Gomphothere",              154),
    9:  ("Silvana Della Camera - Lighthouse",                 608),
    15: ("Greg Meyer Arizona Brightened",                     400),
    19: ("Pioneertown 6mm Fisheye Training",                  112),
    20: ("Thomas Jackson Star Trails Borrego",                107),
    21: ("Cheryl Hanscom Wilcox - Milky Way 101",             101),
    22: ("Silvana Della Camera - Boardwalk",                   64),
    26: ("Silvana Della Camera - River Reflection",            98),
    29: ("Sean Parker - Arizona Star Trails",                 206),
    31: ("Cheryl Hanscom Wilcox - Alabama Hills",             192),
    32: ("borrego_springs_1",                                 400),
    33: ("Shiu Wan - 2013_11_30 Green Park Star Trail",        99),
    34: ("Shiu Wan - 2023-03-14 Sompting Church",              52),
    35: ("Warren Hatch - Barnegat Light - Camera 2",          298),
    36: ("Warren Hatch - Stroudt's Preserve",                 114),
}

GKYLE_TRAIN_IMAGES = T7_ROOT / "external_datasets/gkyle_startrails/512-streaks/train/images"
GKYLE_VAL_IMAGES   = T7_ROOT / "external_datasets/gkyle_startrails/512-streaks/validation/images"
GKYLE_TRAIN_MASKS  = T7_LABELS / "gkyle_startrails/masks_train"
GKYLE_VAL_MASKS    = T7_LABELS / "gkyle_startrails/masks_val"

# ── Tiling ────────────────────────────────────────────────────────────────────

def get_tile_coords(img_w, img_h):
    """Return the list of (x1, y1, x2, y2) tile windows that cover an image.

    Slides a 640x640 window across the frame with 20% overlap (matching SAHI
    inference). If the frame is smaller than one tile it returns a single window
    covering the whole image. The last column/row is snapped flush to the right/
    bottom edge so the far edges are never left uncovered.
    """
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
    """Convert one tile's binary mask into YOLO segmentation label lines.

    Traces the outline of every white blob in the mask, drops tiny specks
    (area < 10px), simplifies each outline to a short polygon, and rewrites the
    vertices as fractions of the tile's width/height. Returns one text line per
    polygon in YOLO-seg format ("0 x1 y1 x2 y2 ..." where class 0 is "trail").
    Returns an empty list if the tile has no usable blobs.
    """
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

# ── Rotation ──────────────────────────────────────────────────────────────────

# cv2 rotation code → (label transform as lambda)
ROTATIONS = {
    90:  (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y: (y,       1.0 - x)),
    180: (cv2.ROTATE_180,                 lambda x, y: (1.0 - x, 1.0 - y)),
    270: (cv2.ROTATE_90_CLOCKWISE,        lambda x, y: (1.0 - y, x      )),
}


def rotation_set_for_frame(poly_meta):
    """Return set of rotation angles {90,180,270} to apply to positive tiles in this frame."""
    rots = set()
    for p in poly_meta:
        if p.get("is_red") or p.get("tile_count", 1) >= 3:
            return {90, 180, 270}  # can't do better, stop early
        la = p.get("long_axis", 0)
        if la > 1000:
            angle = p.get("angle_deg", 45)
            if angle < 20:
                rots.add(90)
            elif angle < 70:
                rots |= {90, 180, 270}
    return rots


def rotate_yolo_label(label_text, rot_deg):
    """Apply rotation transform to a YOLO polygon label string."""
    fn = ROTATIONS[rot_deg][1]
    out_lines = []
    for line in label_text.strip().splitlines():
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        new_coords = []
        for i in range(0, len(coords), 2):
            nx, ny = fn(coords[i], coords[i+1])
            new_coords += [nx, ny]
        out_lines.append(cls + " " + " ".join(f"{v:.6f}" for v in new_coords))
    return "\n".join(out_lines)

# ── Source collection ─────────────────────────────────────────────────────────

def collect_sources():
    """Return list of (img_path, mask_path, poly_meta_path_or_None, prefix, is_gkyle_fixed_split)."""
    sources = []
    img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    for task_id, (folder, max_frames) in sorted(TASK_CONFIG.items()):
        img_dir   = T7_IMAGES / folder
        mask_dir  = T7_LABELS / f"task_{task_id}" / "masks"
        meta_dir  = T7_LABELS / f"task_{task_id}" / "poly_meta"
        prefix    = f"task{task_id:02d}"

        if not img_dir.exists():
            print(f"  WARNING task {task_id}: image folder not found — {img_dir}")
            continue
        if not mask_dir.exists():
            print(f"  WARNING task {task_id}: masks not found — run cvat_to_masks.py first")
            continue

        mask_files = sorted(mask_dir.glob("*.png"))
        img_lookup = {}
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in img_exts:
                continue
            if p.stem not in img_lookup or p.suffix.lower() in {'.jpg', '.jpeg'}:
                img_lookup[p.stem] = p

        count = 0
        for m in mask_files:
            if m.stem not in img_lookup:
                continue
            meta_path = meta_dir / f"{m.stem}.json" if meta_dir.exists() else None
            sources.append((img_lookup[m.stem], m, meta_path, prefix, False))
            count += 1

        print(f"  Task {task_id} ({folder}): {count} pairs")

    # gkyle — fixed train/val split, no rotation augmentation
    for img_dir, mask_dir, tag in [
        (GKYLE_TRAIN_IMAGES, GKYLE_TRAIN_MASKS, "gkyle_train"),
        (GKYLE_VAL_IMAGES,   GKYLE_VAL_MASKS,   "gkyle_val"),
    ]:
        if not img_dir.exists() or not mask_dir.exists():
            print(f"  WARNING gkyle: {tag} folder missing")
            continue
        mask_files = sorted(mask_dir.glob("*.png"))
        img_lookup = {}
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in img_exts:
                img_lookup[p.stem] = p
        count = 0
        for m in mask_files:
            if m.stem not in img_lookup:
                continue
            sources.append((img_lookup[m.stem], m, None, tag, True))
            count += 1
        print(f"  gkyle {tag}: {count} pairs")

    return sources

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("prepare_yolo_v4.py — building v4 training dataset")
    print(f"Output: {T7_OUT}")
    print("=" * 60)
    print("Collecting source pairs...")
    all_sources = collect_sources()

    # Separate gkyle (fixed split) from the rest
    gkyle_train = [(ip, mp, mtp, pfx) for ip, mp, mtp, pfx, fixed in all_sources
                   if fixed and pfx == "gkyle_train"]
    gkyle_val   = [(ip, mp, mtp, pfx) for ip, mp, mtp, pfx, fixed in all_sources
                   if fixed and pfx == "gkyle_val"]
    regular     = [(ip, mp, mtp, pfx) for ip, mp, mtp, pfx, fixed in all_sources
                   if not fixed]

    print(f"\nRegular pairs: {len(regular)}   gkyle train: {len(gkyle_train)}   gkyle val: {len(gkyle_val)}")

    # Stratified train/val split on regular sources
    print("Splitting train/val...")
    trail_pairs, no_trail_pairs = [], []
    for entry in regular:
        mask = cv2.imread(str(entry[1]), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            trail_pairs.append(entry)
        else:
            no_trail_pairs.append(entry)

    random.shuffle(trail_pairs)
    random.shuffle(no_trail_pairs)
    nv_t = max(1, int(len(trail_pairs)    * VAL_FRACTION))
    nv_n = max(1, int(len(no_trail_pairs) * VAL_FRACTION))

    val_pairs   = trail_pairs[:nv_t]  + no_trail_pairs[:nv_n]
    train_pairs = trail_pairs[nv_t:]  + no_trail_pairs[nv_n:]
    random.shuffle(val_pairs)
    random.shuffle(train_pairs)

    splits = {
        "train": train_pairs + gkyle_train,
        "val":   val_pairs   + gkyle_val,
    }
    print(f"Train sources: {len(splits['train'])}   Val sources: {len(splits['val'])}")

    # Create output dirs
    for split in ("train", "val"):
        (T7_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (T7_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    grand = {"train": {"pos": 0, "neg": 0, "aug": 0},
             "val":   {"pos": 0, "neg": 0, "aug": 0}}

    for split, pairs in splits.items():
        img_out   = T7_OUT / "images" / split
        lbl_out   = T7_OUT / "labels" / split

        for i, (img_path, mask_path, meta_path, prefix) in enumerate(pairs):
            sys.stdout.write(
                f"\r  {split}: {i+1}/{len(pairs)}  "
                f"pos={grand[split]['pos']} neg={grand[split]['neg']} aug={grand[split]['aug']}    ")
            sys.stdout.flush()

            img  = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img_h, img_w = img.shape[:2]
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            # Load poly_meta and decide rotation set for this frame
            poly_meta = []
            if meta_path and meta_path.exists():
                try:
                    poly_meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
            rots = rotation_set_for_frame(poly_meta)

            coords = get_tile_coords(img_w, img_h)
            xs_sorted = sorted(set(c[0] for c in coords))
            ys_sorted = sorted(set(c[1] for c in coords))
            x_idx = {x: i for i, x in enumerate(xs_sorted)}
            y_idx = {y: i for i, y in enumerate(ys_sorted)}

            pos_tiles = []
            neg_tiles = []

            for (x1, y1, x2, y2) in coords:
                tile_img  = img[y1:y2, x1:x2]
                tile_mask = mask[y1:y2, x1:x2]
                row = y_idx[y1]; col = x_idx[x1]
                name = f"{prefix}_{img_path.stem}_r{row:02d}c{col:02d}"
                if int(np.sum(tile_mask > 0)) >= MIN_TRAIL_PX:
                    pos_tiles.append((tile_img, tile_mask, name))
                else:
                    neg_tiles.append((tile_img, tile_mask, name))

            n_keep = min(len(neg_tiles), max(NEG_MIN, len(pos_tiles) * NEG_RATIO))
            random.shuffle(neg_tiles)
            kept_neg = neg_tiles[:n_keep]

            # Write positive tiles (base + rotations)
            for tile_img, tile_mask, name in pos_tiles:
                poly_lines = mask_crop_to_yolo(tile_mask)
                label_text = "\n".join(poly_lines)

                cv2.imwrite(str(img_out / f"{name}.png"), tile_img)
                (lbl_out / f"{name}.txt").write_text(label_text)
                grand[split]["pos"] += 1

                for rot_deg in sorted(rots):
                    cv2_code = ROTATIONS[rot_deg][0]
                    rot_img  = cv2.rotate(tile_img, cv2_code)
                    rot_lbl  = rotate_yolo_label(label_text, rot_deg) if label_text.strip() else ""
                    aug_name = f"{name}_r{rot_deg}"
                    cv2.imwrite(str(img_out / f"{aug_name}.png"), rot_img)
                    (lbl_out / f"{aug_name}.txt").write_text(rot_lbl)
                    grand[split]["aug"] += 1

            # Write negative tiles (no rotations)
            for tile_img, tile_mask, name in kept_neg:
                cv2.imwrite(str(img_out / f"{name}.png"), tile_img)
                (lbl_out / f"{name}.txt").write_text("")
                grand[split]["neg"] += 1

        total = grand[split]["pos"] + grand[split]["neg"] + grand[split]["aug"]
        print(f"\r  {split}: {total} tiles  "
              f"(pos={grand[split]['pos']} neg={grand[split]['neg']} aug={grand[split]['aug']})              ")

    # dataset.yaml
    yaml = T7_OUT / "dataset.yaml"
    yaml.write_text(f"""# Trail DetectoR v4 — rotation-augmented tiled dataset
# Tile size: {TILE_SIZE}px  Overlap: {int(OVERLAP*100)}%  (matches SAHI inference)
path: {T7_OUT.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: trail
""")

    total_all = sum(grand[s]["pos"] + grand[s]["neg"] + grand[s]["aug"] for s in ("train","val"))
    print(f"\nDataset ready:  {T7_OUT}")
    print(f"Config:         {yaml}")
    print(f"Total tiles:    {total_all}")
    for split in ("train","val"):
        t = grand[split]
        print(f"  {split}: pos={t['pos']}  neg={t['neg']}  aug={t['aug']}  "
              f"total={t['pos']+t['neg']+t['aug']}")
    print(f"\nTrain with:  python3 tools/train_yolo.py --imgsz {TILE_SIZE} "
          f"--dataset \"{yaml}\"")


if __name__ == "__main__":
    main()
