#!/usr/bin/env python3
"""
prepare_yolo_dataset.py — Convert binary trail masks to YOLOv8-seg format,
using SAHI-style tiling to match inference distribution.

WHAT THIS IS
============
A maintained step in the training-data pipeline. It takes folders of
full-resolution star-trail source images paired with binary trail masks
(white = trail, black = sky), slices both into overlapping 640x640 tiles,
and writes out a ready-to-train YOLOv8 segmentation dataset: tile images,
matching YOLO polygon label files, a train/val split, and a dataset.yaml.

WHERE IT FITS
=============
The mask source for this script is the per-dataset "masks" folders produced
upstream from reviewed annotations (the *_to_masks / generate_training_labels
side of the pipeline). The dataset this script writes is then handed to the
YOLO trainer. The closing print line spells out the exact train command.

HOW TO RUN
==========
Edit the `sources` list and `output_dir` in the `if __name__ == "__main__"`
block at the bottom, then run:

    python3 prepare_yolo_dataset.py

Each `sources` entry is an (images_dir, masks_dir) pair, optionally extended
with a name prefix and a max image count: (images_dir, masks_dir, prefix,
max_count). Image and mask are matched by filename stem; masks are the
.png files in masks_dir. The script prints per-source pair counts, the
train/val split, tile totals, and the final dataset location.

WHY TILING MATTERS
==================
At inference, SAHI slices full-resolution images (e.g. 5472×3648) into
overlapping 640×640 patches. A 20px-wide trail stays 20px wide in each tile.

Without tiling, YOLO resizes the whole 5472×3648 image to 640×640 during
training — the same trail shrinks to ~2px wide and becomes nearly invisible.
The model learns nothing useful at that scale.

This script tiles full-res source images into the same 640×640 patches at
dataset-prep time, so training and inference see identical trail widths.

NEGATIVE TILE BALANCE
=====================
Most tiles from a sky image contain no trail. Including too many pure-sky
tiles trains the model to predict "nothing" for every patch. The neg_ratio
parameter controls how many sky-only tiles are kept per trail tile.

SMALL IMAGES (gkyle 512×512 crops)
===================================
Images that already fit within tile_size (e.g. 512×512 gkyle crops) are
passed through as single tiles — no tiling applied. YOLO letterboxes them
to 640×640 during training, exactly as SAHI would at a tile boundary.
The mismatch between 512px training crops and 640px inference tiles is
minor (~0.8× scale) compared to the original 5472→640 full-image resize.

Output structure:
    dataset_tiled/
        images/
            train/   ← 640×640 PNG tile crops
            val/
        labels/
            train/   ← YOLO polygon .txt files (coordinates within tile)
            val/
        dataset.yaml
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np


# --- Tiling parameters — must match SAHI inference settings in infer_trails.py ---
TILE_SIZE     = 640    # pixels — same as --tile-size in infer_trails.py
OVERLAP       = 0.2    # 20% overlap — same as --overlap in infer_trails.py

# --- Negative tile sampling ---
NEG_RATIO     = 2      # max sky-only tiles kept per trail tile per source image
NEG_MIN       = 3      # always keep at least this many sky tiles per source image
MIN_TRAIL_PX  = 50     # trail pixels required for a tile to be "positive"


def get_tile_coords(img_w, img_h, tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    Work out where to cut the image into tiles.

    Return a list of (x1, y1, x2, y2) tile regions that together cover the
    whole image, stepping across by `stride` (tile_size minus the overlap)
    so neighboring tiles share an `overlap` fraction of pixels. An extra
    column/row pinned to the right/bottom edge is appended if the last
    regular step would leave a strip uncovered, so the image edges are never
    missed (those edge tiles overlap their neighbor by more than the usual
    amount).

    Images smaller than tile_size in both dimensions are returned as a single
    full-image tile (the small-image passthrough). Every other tile is exactly
    tile_size x tile_size.
    """
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]

    stride = int(tile_size * (1 - overlap))

    xs = list(range(0, img_w - tile_size, stride))
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(img_w - tile_size)

    ys = list(range(0, img_h - tile_size, stride))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(img_h - tile_size)

    return [(x, y, x + tile_size, y + tile_size) for y in ys for x in xs]


def mask_crop_to_yolo(mask_crop):
    """
    Turn one tile's white-on-black trail mask into YOLO polygon label lines.

    Takes a binary mask crop (H x W uint8, 255 = trail, 0 = sky) and traces the
    outline of each separate trail blob with cv2.findContours. Tiny specks
    (contour area under 10px) are dropped as noise; each remaining outline is
    simplified with approxPolyDP and skipped if it collapses to fewer than 3
    points. The surviving polygon points are normalized to the [0,1] range
    relative to the tile's width and height (the format YOLO expects).

    Returns a list of "0 x1 y1 x2 y2 ..." lines (class id 0 = "trail", one line
    per blob). Returns an empty list when the tile holds no trail, which is what
    marks the written label file as a sky-only / negative tile.
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


def generate_tiles(img_path, mask_path, prefix,
                   tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    Load one source image plus its mask and cut both into matching tiles.

    Reads the image (color) and mask (grayscale). If either fails to load it
    prints a warning and returns []. If the mask's size doesn't match the image
    (e.g. a downscaled mask), it is nearest-neighbor resized to the image so the
    two line up exactly before cutting. It then asks get_tile_coords for the
    tile grid and crops the same rectangle out of both image and mask for every
    tile, naming each tile with a human-readable row/column index.

    Returns a list of (tile_img, tile_mask, tile_name) where:
      tile_img  — BGR numpy array (tile_size x tile_size, or smaller for the
                  small-image passthrough)
      tile_mask — grayscale numpy array, same size as tile_img
      tile_name — unique filename stem "{prefix}_{stem}_r{row:02d}c{col:02d}"

    Returns [] on load failure.
    """
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"\n  WARNING: could not read {img_path.name} or mask")
        return []

    img_h, img_w = img.shape[:2]

    # Guard: resize mask if it doesn't match image dimensions
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    coords = get_tile_coords(img_w, img_h, tile_size, overlap)
    stem   = img_path.stem

    # Build row/col index for human-readable tile names
    xs_sorted = sorted(set(c[0] for c in coords))
    ys_sorted = sorted(set(c[1] for c in coords))
    x_idx = {x: i for i, x in enumerate(xs_sorted)}
    y_idx = {y: i for i, y in enumerate(ys_sorted)}

    tiles = []
    for (x1, y1, x2, y2) in coords:
        tile_img  = img[y1:y2, x1:x2]
        tile_mask = mask[y1:y2, x1:x2]
        row  = y_idx[y1]
        col  = x_idx[x1]
        name = f"{prefix}_{stem}_r{row:02d}c{col:02d}"
        tiles.append((tile_img, tile_mask, name))

    return tiles


def prepare_dataset(sources, output_dir, val_fraction=0.2, seed=42,
                    tile_size=TILE_SIZE, overlap=OVERLAP,
                    neg_ratio=NEG_RATIO, neg_min=NEG_MIN):
    """
    Build the whole tiled YOLO segmentation dataset end to end. Main entry point.

    Steps, in order:
      1. Collect source image/mask pairs from every entry in `sources`, matching
         image to mask by filename stem (preferring .jpg/.jpeg when a stem has
         several image files), honoring an optional per-source max image count.
      2. Split the SOURCE images (not the tiles) into train and val, stratified
         so images that contain a trail and images that don't are each split by
         val_fraction. This keeps all tiles from one source image on the same
         side of the train/val line, avoiding leakage between near-identical
         overlapping tiles.
      3. For each source image: cut it into tiles, label each tile positive
         (>= MIN_TRAIL_PX trail pixels) or negative (sky-only), keep every
         positive tile, and keep only a sampled subset of the negatives
         (max(neg_min, positives * neg_ratio)) so the model isn't drowned in
         empty-sky tiles. Write each kept tile as a lossless PNG plus its YOLO
         polygon .txt label.
      4. Write dataset.yaml (paths, one class "trail") and print a summary plus
         the exact train command to run next.

    sources: list of (images_dir, masks_dir), optionally with a name prefix and
             a max image count: (images_dir, masks_dir, prefix, max_count)
    output_dir: destination directory (e.g. dataset_tiled/)
    val_fraction: fraction of source images held out for validation
    seed: random seed so the shuffle/split and negative sampling are repeatable
    tile_size, overlap: tiling geometry, passed through to the tiler
    neg_ratio: sky-only tiles kept per trail tile per source image
    neg_min: minimum sky tiles kept per source image regardless of trail count
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    # ── collect source image/mask pairs ──────────────────────────────────────
    pairs = []
    for entry in sources:
        images_dir    = Path(entry[0])
        masks_dir     = Path(entry[1])
        custom_prefix = entry[2] if len(entry) > 2 else None
        max_count     = entry[3] if len(entry) > 3 else None
        prefix        = custom_prefix if custom_prefix else images_dir.name

        mask_files = sorted(masks_dir.glob("*.png"))
        img_lookup = {}
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            if p.stem not in img_lookup or p.suffix.lower() in {'.jpg', '.jpeg'}:
                img_lookup[p.stem] = p

        matched = [(img_lookup[m.stem], m, prefix)
                   for m in mask_files if m.stem in img_lookup]
        matched = sorted(matched, key=lambda x: x[0].name)
        if max_count is not None:
            matched = matched[:max_count]
        print(f"  {prefix}: {len(matched)} pairs")
        pairs.extend(matched)

    print(f"Total source pairs: {len(pairs)}")

    # ── stratified train/val split on SOURCE images (not tiles) ──────────────
    print("Checking masks...", end=" ", flush=True)
    trail_pairs, no_trail_pairs = [], []
    for img_path, mask_path, prefix in pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            trail_pairs.append((img_path, mask_path, prefix))
        else:
            no_trail_pairs.append((img_path, mask_path, prefix))
    print(f"{len(trail_pairs)} with trail, {len(no_trail_pairs)} without")

    random.shuffle(trail_pairs)
    random.shuffle(no_trail_pairs)
    n_val_trail    = max(1, int(len(trail_pairs)    * val_fraction))
    n_val_no_trail = max(1, int(len(no_trail_pairs) * val_fraction))

    val_pairs   = trail_pairs[:n_val_trail]  + no_trail_pairs[:n_val_no_trail]
    train_pairs = trail_pairs[n_val_trail:]  + no_trail_pairs[n_val_no_trail:]
    random.shuffle(val_pairs)
    random.shuffle(train_pairs)
    print(f"Train source images: {len(train_pairs)}   "
          f"Val source images: {len(val_pairs)}")

    # ── create output directories ─────────────────────────────────────────────
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── tile, filter, and write ───────────────────────────────────────────────
    grand_total = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out   = output_dir / "images" / split
        label_out = output_dir / "labels" / split

        for i, (img_path, mask_path, prefix) in enumerate(split_pairs):
            sys.stdout.write(
                f"\r  {split}: source {i+1}/{len(split_pairs)} "
                f"(tiles: {grand_total[split]['pos']+grand_total[split]['neg']})    ")
            sys.stdout.flush()

            tiles = generate_tiles(img_path, mask_path, prefix, tile_size, overlap)
            if not tiles:
                continue

            # Separate positive (trail) and negative (sky-only) tiles
            pos_tiles, neg_tiles = [], []
            for tile_img, tile_mask, name in tiles:
                if int(np.sum(tile_mask > 0)) >= MIN_TRAIL_PX:
                    pos_tiles.append((tile_img, tile_mask, name))
                else:
                    neg_tiles.append((tile_img, tile_mask, name))

            # Sample negatives: max(neg_min, len(pos) * neg_ratio)
            n_keep = min(len(neg_tiles),
                         max(neg_min, len(pos_tiles) * neg_ratio))
            random.shuffle(neg_tiles)
            kept_tiles = pos_tiles + neg_tiles[:n_keep]

            for tile_img, tile_mask, name in kept_tiles:
                # Write tile image (PNG — lossless)
                cv2.imwrite(str(img_out / (name + ".png")), tile_img)

                # Convert mask crop → YOLO polygons (coords local to tile)
                poly_lines = mask_crop_to_yolo(tile_mask)
                (label_out / (name + ".txt")).write_text("\n".join(poly_lines))

                if poly_lines:
                    grand_total[split]["pos"] += 1
                else:
                    grand_total[split]["neg"] += 1

        total_tiles = grand_total[split]["pos"] + grand_total[split]["neg"]
        print(f"\r  {split}: {total_tiles} tiles  "
              f"({grand_total[split]['pos']} with trail, "
              f"{grand_total[split]['neg']} sky-only)              ")

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# YOLOv8-seg tiled dataset — airplane trail detection
# Tile size: {tile_size}px  Overlap: {int(overlap*100)}%  (matches SAHI inference defaults)
path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: trail
""")

    total = sum(grand_total[s]["pos"] + grand_total[s]["neg"]
                for s in ["train", "val"])
    print(f"\nDataset ready:  {output_dir}")
    print(f"Config:         {yaml_path}")
    print(f"Total tiles:    {total}  "
          f"({grand_total['train']['pos']+grand_total['val']['pos']} with trail, "
          f"{grand_total['train']['neg']+grand_total['val']['neg']} sky-only)")
    print(f"Tile size:      {tile_size}px  Overlap: {int(overlap*100)}%  "
          f"Neg ratio: {neg_ratio}:1 (min {neg_min})")
    print(f"\nTrain with:  python3 train_yolo.py --imgsz {tile_size} "
          f"--dataset \"{yaml_path}\"")


if __name__ == "__main__":
    # Run config: the concrete list of (images_dir, masks_dir[, prefix, max])
    # dataset folders on the T7 Shield drive that went into this build, plus the
    # output location. Edit these entries to add/remove datasets or change where
    # the tiled dataset is written. Comments below mark when sources were added.
    prepare_dataset(
        sources = [
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Silvana Della Camera - Lighthouse",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Silvana Della Camera - Lighthouse/masks"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/borrego_springs_1",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/borrego_springs_1/masks"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Cheryl Hanscom Wilcox - Milky Way 101",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Cheryl Hanscom Wilcox - Milky Way 101/masks"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Silvana Della Camera - Boardwalk",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Silvana Della Camera - Boardwalk/masks"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Cheryl Hanscom Wilcox - Alabama Hills",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Cheryl Hanscom Wilcox - Alabama Hills/masks"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Thomas Jackson Star Trails Borrego",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Thomas Jackson Star Trails Borrego/masks",
             "thomas_jackson", 450),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Greg Meyer Arizona",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Greg Meyer Arizona/masks",
             "greg_meyer", 479),
            # gkyle 512×512 crops — already at inference scale (small-image passthrough)
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/external_datasets/gkyle_startrails/512-streaks/train/images",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/gkyle_startrails/masks_train",
             "gkyle_train"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/external_datasets/gkyle_startrails/512-streaks/validation/images",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/gkyle_startrails/masks_val",
             "gkyle_val"),
            # v12 additions (2026-04-23): CVAT-reviewed + pioneertown
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Bruce Herwig - Joshua Tree - Juniper and Monolith",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Bruce Herwig - Joshua Tree - Juniper and Monolith/masks",
             "bruce_joshua_tree"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Bruce Herwig - first star trail data",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Bruce Herwig - first star trail data/masks",
             "bruce_first_star_trail"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Bruce Herwig - Borrego - Gomphothere",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Bruce Herwig - Borrego - Gomphothere/masks",
             "borrego_gomphothere"),
            ("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Pioneertown 6mm Fisheye Training",
             "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels/Pioneertown 6mm Fisheye Training/masks",
             "pioneertown_fisheye"),
        ],
        output_dir = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/dataset_tiled",
    )
