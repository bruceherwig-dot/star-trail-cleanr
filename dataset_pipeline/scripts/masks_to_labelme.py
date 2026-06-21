#!/usr/bin/env python3
"""
masks_to_labelme.py — Convert binary PNG masks to LabelMe JSON for review/editing.

What it is
----------
A maintained step in the CVAT / training-data pipeline. The pre-annotation
workflow is: run inference (infer_trails.py) to produce per-frame binary
trail masks, convert those masks to LabelMe polygon files with THIS script,
then push the LabelMe folder into CVAT (labelme_to_cvat.py) so Bruce reviews
with a head start instead of labeling from scratch. This script is the
mask-to-polygon bridge in that chain.

What it does
------------
For every image/mask pair (matched by filename stem) it:
  - turns each connected blob in the mask into a polygon (rotated bounding
    box, clipped to the image border) — see mask_to_shapes(),
  - writes one LabelMe JSON file per image with those polygons, and
  - symlinks the original image next to its JSON so the working folder opens
    directly in LabelMe.

Optionally (--original-masks), it also diffs each filtered mask against its
unfiltered original and adds a second set of polygons labeled "removed" for
the pixels that the filtering stage threw away, so a reviewer can see what
was dropped.

Output is a single working folder containing image symlinks + JSON files,
ready to open in LabelMe (or to feed onward into CVAT).

Usage:
    python3 masks_to_labelme.py \
        --images /Users/bruceherwig/Documents/frames/extra \
        --masks  /Users/bruceherwig/Documents/training_masks \
        --output /Users/bruceherwig/Documents/labelme_review \
        --trail-only
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely import clip_by_rect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.frame_list import natural_key  # noqa: E402


def mask_to_shapes(mask, label="trail"):
    """Convert binary mask to list of LabelMe polygon shapes.

    Uses minAreaRect for the rotated bounding box, then clips it against
    the image boundary via Shapely clip_by_rect so edge-exiting trails
    get clean polygons that follow the border instead of distorted corners.
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)

        # Clip the rotated rect against the image boundary
        rot_poly = Polygon(points.tolist())
        if not rot_poly.is_valid:
            rot_poly = rot_poly.buffer(0)
        clipped = clip_by_rect(rot_poly, 0, 0, w - 1, h - 1)

        if clipped.is_empty:
            continue
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda g: g.area)

        coords = list(clipped.exterior.coords[:-1])  # drop closing duplicate
        shapes.append({
            "label": label,
            "points": [[float(x), float(y)] for x, y in coords],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    return shapes


def make_labelme_json(img_path, mask, shapes):
    """Build the LabelMe JSON dict for one image.

    Wraps the already-computed polygon shapes in a LabelMe v5.0.1 file
    structure: it records the image filename (not the full path) and the
    image height/width taken from the mask, and stores no embedded image
    bytes (imageData is None, so LabelMe loads the pixels from the linked
    image on disk). Returns a plain dict ready to json.dumps to a .json file.
    """
    h, w = mask.shape[:2]
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }


def main():
    """Match images to masks, convert each mask to polygons, and write a LabelMe folder.

    Steps:
      1. Parse CLI args (image dir, mask dir, output dir, plus optional
         --original-masks, --trail-only, and --limit).
      2. Index images and masks by filename stem and keep only stems that
         appear in both (the matched pairs).
      3. With --trail-only, drop any pair whose mask is all-black (no trail
         pixels). With --limit N, keep only the first N pairs.
      4. For each pair: symlink the original image into the output folder,
         convert the mask to "trail" polygons, optionally add "removed"
         polygons by diffing against the original unfiltered mask, and write
         the per-image LabelMe JSON.
      5. Print progress and a final reminder of the `labelme <folder>` command
         to open the result.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="/Users/bruceherwig/Documents/frames/extra")
    parser.add_argument("--masks",  default="/Users/bruceherwig/Documents/training_masks")
    parser.add_argument("--output", default="/Users/bruceherwig/Documents/labelme_review")
    parser.add_argument("--original-masks", default=None,
                        help="Original (unfiltered) masks dir — removed regions become 'removed' polygons")
    parser.add_argument("--trail-only", action="store_true",
                        help="Only include frames where mask has trail pixels")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N frames (0 = no limit)")
    args = parser.parse_args()

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    img_lookup  = {p.stem: p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in exts}
    mask_lookup = {p.stem: p for p in Path(args.masks).iterdir()
                   if p.suffix.lower() == '.png'}

    pairs = [(img_lookup[s], mask_lookup[s])
             for s in sorted(mask_lookup, key=natural_key) if s in img_lookup]
    print(f"Found {len(pairs)} matched pairs")

    if args.trail_only:
        print("Filtering to trail frames...", end=" ", flush=True)
        pairs = [(ip, mp) for ip, mp in pairs
                 if cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE).max() > 0]
        print(f"{len(pairs)} with trail")

    if args.limit > 0:
        pairs = pairs[:args.limit]
        print(f"Limiting to {len(pairs)} frames")

    # Load original masks for "removed" polygons
    orig_lookup = {}
    if args.original_masks:
        orig_lookup = {p.stem: p for p in Path(args.original_masks).iterdir()
                       if p.suffix.lower() == '.png'}
        print(f"Original masks: {len(orig_lookup)} (will diff for 'removed' polygons)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    removed_total = 0
    print(f"Writing {len(pairs)} LabelMe JSON files → {out_dir}\n")
    for i, (img_path, mask_path) in enumerate(pairs):
        sys.stdout.write(f"\r  {i+1}/{len(pairs)}  {img_path.name}   ")
        sys.stdout.flush()

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Symlink image
        link = out_dir / img_path.name
        if not link.exists():
            link.symlink_to(img_path.resolve())

        # Trail shapes from filtered mask
        shapes = mask_to_shapes(mask)

        # Removed shapes from diff with original
        if img_path.stem in orig_lookup:
            orig = cv2.imread(str(orig_lookup[img_path.stem]), cv2.IMREAD_GRAYSCALE)
            if orig is not None:
                removed_px = ((orig > 0) & (mask == 0)).astype(np.uint8) * 255
                if removed_px.max() > 0:
                    removed_shapes = mask_to_shapes(removed_px, label="removed")
                    shapes.extend(removed_shapes)
                    removed_total += len(removed_shapes)

        # Write JSON
        data = make_labelme_json(img_path, mask, shapes)
        json_path = out_dir / (img_path.stem + ".json")
        json_path.write_text(json.dumps(data, indent=2))

    if removed_total:
        print(f"\n  {removed_total} 'removed' polygons included")
    print(f"\nDone. Open LabelMe on:\n  {out_dir}")
    print(f"\nRun: labelme {out_dir}")


if __name__ == "__main__":
    main()
