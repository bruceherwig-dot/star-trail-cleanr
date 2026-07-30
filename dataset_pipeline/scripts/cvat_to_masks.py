#!/usr/bin/env python3
"""cvat_to_masks.py — Export CVAT annotations to binary mask PNGs + polygon metadata.

What this is
------------
A maintained CVAT/training-data pipeline utility. It turns Bruce's reviewed trail
polygons (stored in the local CVAT server) into the binary mask PNGs that the YOLO
training pipeline consumes, plus a small per-frame metadata JSON used for filtering
and analysis. This is the "pull reviewed CVAT polygons -> training masks" step.

For each task (see TASK_CONFIG below):
  - Fetches reviewed polygon annotations from CVAT (first N frames per decided count)
  - Drops polygons < 60px long-axis unless they contain red pixels (nav light)
  - Renders surviving polygons to binary mask PNGs
  - Saves per-frame polygon metadata JSON (long_axis, angle_deg, is_red, tile_count)

How to run
----------
No arguments. Run it directly:  python3 tools/cvat_to_masks.py
It loops over every task in TASK_CONFIG, querying the local CVAT server at
http://localhost:8080 (auth: user bherwig2, password read from
~/.star_trail_cleanr/cvat_credentials). To change which tasks/frames are exported,
edit the TASK_CONFIG dictionary. Source images and all output live on the T7 Shield
external drive, so that drive must be mounted.

Output on T7:
  labels/task_<id>/masks/<stem>.png        (binary 0/255 mask, one per frame with polygons)
  labels/task_<id>/poly_meta/<stem>.json   (list of kept polygons + their measurements)
"""

import base64
import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.io_safe import robust_imread  # EXIF-aware reader (matches CVAT's orientation)

# ── Config ────────────────────────────────────────────────────────────────────

CVAT_URL  = "http://localhost:8080"
CVAT_USER = "bherwig2"
T7_LABELS = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/labels")
T7_IMAGES = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images")

TILE_SIZE = 640
OVERLAP   = 0.2
MIN_LONG_AXIS = 60   # drop polygons shorter than this (unless red)

# Task ID → (image folder name on T7, max frames to use)
# v5 retrain DATASETS (full frames). Folder is under "star trail images" unless it's an
# absolute path (out-of-root sets: Bombay, Jeff Fishman). Count = all reviewed frames.
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

# ── CVAT auth ─────────────────────────────────────────────────────────────────

def _password():
    """Read the CVAT password from the credentials file (~/.star_trail_cleanr/cvat_credentials)."""
    return (Path.home() / ".star_trail_cleanr" / "cvat_credentials").read_text().strip()

def _headers():
    """Build the HTTP Basic Auth header for CVAT requests from the username and stored password."""
    creds = base64.b64encode(f"{CVAT_USER}:{_password()}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}

def cvat_get(path):
    """GET the given CVAT API path and return the parsed JSON response."""
    req = urllib.request.Request(f"{CVAT_URL}{path}", headers=_headers())
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

# ── Geometry helpers ──────────────────────────────────────────────────────────

def long_axis(points):
    """Return the polygon's longest dimension in pixels.

    `points` is CVAT's flat [x0, y0, x1, y1, ...] vertex list. Measures the
    maximum distance between any two vertices, which for a thin trail polygon is
    effectively its length. Used by the < MIN_LONG_AXIS drop filter.
    """
    xs = points[0::2]; ys = points[1::2]
    pts = list(zip(xs, ys))
    max_d = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > max_d:
                max_d = d
    return max_d


def poly_angle(points):
    """Return long-axis angle in degrees (0-90) from horizontal."""
    xs = points[0::2]; ys = points[1::2]
    pts = np.array(list(zip(xs, ys)), dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    angle = abs(rect[2]) % 90
    return float(angle)


def tile_origins(w, h):
    """Return the (x-origins, y-origins) of the 640px tile grid for a w x h image.

    Mirrors the detection pipeline's tiling: 640px tiles stepping by TILE_SIZE *
    (1 - OVERLAP), with a final tile snapped flush to the right/bottom edge so the
    whole image is covered. tile_count() uses these origins to count how many tiles
    a polygon touches.
    """
    stride = int(TILE_SIZE * (1 - OVERLAP))
    xs = list(range(0, w - TILE_SIZE, stride))
    if not xs or xs[-1] + TILE_SIZE < w:
        xs.append(max(0, w - TILE_SIZE))
    ys = list(range(0, h - TILE_SIZE, stride))
    if not ys or ys[-1] + TILE_SIZE < h:
        ys.append(max(0, h - TILE_SIZE))
    return xs, ys


def tile_count(points, img_w, img_h):
    """Count how many 640px tiles the polygon has pixels in."""
    xs_p = points[0::2]; ys_p = points[1::2]
    pts = np.array(list(zip(xs_p, ys_p)), dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    tile_xs, tile_ys = tile_origins(img_w, img_h)
    count = 0
    for ty in tile_ys:
        for tx in tile_xs:
            if mask[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE].any():
                count += 1
    return count


def is_red(points, img):
    """Return True if the polygon region contains red nav-light pixels."""
    xs_p = points[0::2]; ys_p = points[1::2]
    pts = np.array(list(zip(xs_p, ys_p)), dtype=np.int32)
    pmask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(pmask, [pts], 255)
    pixels = img[pmask > 0]
    if len(pixels) == 0:
        return False
    b = float(pixels[:, 0].mean())
    g = float(pixels[:, 1].mean())
    r = float(pixels[:, 2].mean())
    return r > 80 and r > g * 1.5 and r > b * 1.5

# ── Per-frame processing ──────────────────────────────────────────────────────

def load_image(img_path):
    """Load an image as 8-bit BGR (EXIF orientation applied), or None if unreadable.

    Uses robust_imread so the pixel orientation matches what CVAT stored its polygons
    against: sideways-shot photos carry an EXIF rotate flag, and CVAT imports them
    upright. Reading them upright here too keeps polygons on-canvas instead of landing
    off the edge (the Thomas Jackson Borrego 4000x6000-vs-6000x4000 bug). A 4-channel
    (BGRA) image is reduced to BGR; is_red() and mask sizing rely on the result.
    """
    img = robust_imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def process_task(task_id):
    """Export one CVAT task's polygons to mask PNGs and metadata JSON.

    Looks up the task's image folder and frame cap from TASK_CONFIG, then:
      - finds the task's job and the first `max_frames` frame filenames
      - pulls all polygon annotations and groups them by frame index
      - for each frame, applies the drop filter (short polygons that aren't red),
        renders the survivors into one binary mask, and records per-polygon
        metadata (cvat_id, long_axis, angle_deg, is_red, tile_count)
      - writes masks/<stem>.png (only if any polygon survived) and
        poly_meta/<stem>.json under labels/task_<id>/ on T7

    Progress (frames processed, kept/dropped counts) is printed on one line.
    Returns nothing; missing image folder just prints a note and returns early.
    """
    folder_name, max_frames = TASK_CONFIG[task_id]
    img_dir = Path(folder_name) if Path(folder_name).is_absolute() else T7_IMAGES / folder_name

    if not img_dir.exists():
        print(f"  Task {task_id}: image folder not found — {img_dir}")
        return

    out_masks = T7_LABELS / f"task_{task_id}" / "masks"
    out_meta  = T7_LABELS / f"task_{task_id}" / "poly_meta"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # Get job
    jobs   = cvat_get(f"/api/jobs?task_id={task_id}")
    job_id = jobs["results"][0]["id"]

    # Get frame filenames
    meta_data    = cvat_get(f"/api/jobs/{job_id}/data/meta")
    all_frames   = meta_data.get("frames", [])
    frames_to_use = all_frames[:max_frames]

    # Get annotations
    ann    = cvat_get(f"/api/jobs/{job_id}/annotations")
    shapes = ann.get("shapes", [])

    # Group shapes by frame index (within the cap)
    by_frame = {}
    for s in shapes:
        if s.get("type") != "polygon":
            continue
        by_frame.setdefault(s["frame"], []).append(s)

    # Collapse to one entry per frame STEM, merging duplicate jpg/tif twins of the same
    # shot. Stroudt's keeps both a .jpg and a pixel-identical .tif in CVAT, and trails
    # were drawn on either copy; we want ONE training frame per shot carrying the union
    # of those trails, so the trainer never sees the same image twice (the duplication
    # that bloated v4). For every other dataset each stem is unique, so this is a no-op.
    by_stem = {}
    stem_order = []
    for frame_idx, frame_info in enumerate(frames_to_use):
        polys = by_frame.get(frame_idx, [])
        if not polys:
            continue
        stem = Path(frame_info["name"]).stem
        if stem not in by_stem:
            by_stem[stem] = []
            stem_order.append(stem)
        by_stem[stem].extend(polys)

    # Index source files by stem (case-insensitive), preferring jpg over tif/png so a
    # duplicate pair always renders on the jpg. Look in the folder and a TIFF/ subfolder
    # (Stroudt's keeps its tifs one level down).
    ext_pref = {".jpg": 0, ".jpeg": 0, ".png": 1, ".tif": 2, ".tiff": 2}
    src_index = {}
    search_dirs = [img_dir]
    if (img_dir / "TIFF").is_dir():
        search_dirs.append(img_dir / "TIFF")
    for d in search_dirs:
        for p in d.iterdir():
            e = p.suffix.lower()
            if e not in ext_pref:
                continue
            key = p.stem.lower()
            cur = src_index.get(key)
            if cur is None or ext_pref[e] < ext_pref[cur.suffix.lower()]:
                src_index[key] = p

    kept = dropped = 0
    frames_with_polys = 0
    masks_written = 0
    source_missing = []

    for si, stem in enumerate(stem_order):
        polys = by_stem[stem]
        frames_with_polys += 1

        src = src_index.get(stem.lower())
        img = load_image(src) if src else None
        if img is None:
            source_missing.append(stem)      # reviewed polys but no source image -> reported, never silent
            continue
        h, w = img.shape[:2]

        mask_out  = np.zeros((h, w), dtype=np.uint8)
        poly_meta = []

        for s in polys:
            pts   = s["points"]
            la    = long_axis(pts)
            red   = is_red(pts, img)

            # Drop filter
            if la < MIN_LONG_AXIS and not red:
                dropped += 1
                continue

            kept += 1
            angle  = poly_angle(pts)
            tc     = tile_count(pts, w, h)

            # Render to mask
            int_pts = np.array(
                [(int(x), int(y)) for x, y in zip(pts[0::2], pts[1::2])],
                dtype=np.int32,
            )
            cv2.fillPoly(mask_out, [int_pts], 255)

            poly_meta.append({
                "cvat_id":   s["id"],
                "long_axis": round(la, 1),
                "angle_deg": round(angle, 1),
                "is_red":    red,
                "tile_count": tc,
            })

        if mask_out.any():
            cv2.imwrite(str(out_masks / f"{stem}.png"), mask_out)
            masks_written += 1
        (out_meta / f"{stem}.json").write_text(json.dumps(poly_meta, indent=2))

        sys.stdout.write(
            f"\r  Task {task_id} — frame {si+1}/{len(stem_order)} "
            f"kept {kept} dropped {dropped}    "
        )
        sys.stdout.flush()

    print(f"\r  Task {task_id} done — {kept} kept, {dropped} dropped, "
          f"{masks_written} masks, {len(source_missing)} source-missing ({out_masks})")
    return {"task": task_id, "frames_with_polys": frames_with_polys,
            "masks_written": masks_written, "source_missing": source_missing}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Run the export for every task in TASK_CONFIG (in ascending task-id order).

    Prints a header with the output root and drop threshold, then calls
    process_task() per task, catching and printing any per-task error so one bad
    task doesn't abort the rest.
    """
    print("cvat_to_masks.py — building training masks from CVAT")
    print(f"Output root: {T7_LABELS}")
    print(f"Drop threshold: < {MIN_LONG_AXIS}px (unless red)")
    print("=" * 60)

    summaries = []
    for task_id in sorted(TASK_CONFIG):
        folder_name, max_frames = TASK_CONFIG[task_id]
        print(f"\nTask {task_id}: {folder_name} ({max_frames} frames)")
        try:
            summ = process_task(task_id)
            if summ:
                summaries.append(summ)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── count-check reconcile (catches the TJ-14-class silent shortfall) ──
    print("\n" + "=" * 60)
    print("RECONCILE (count-check):")
    problems = []
    for s in summaries:
        miss = len(s["source_missing"])
        print(f"  task {s['task']:>3}: {s['frames_with_polys']:>4} frames w/ polys  "
              f"{s['masks_written']:>4} masks  {miss:>3} source-missing"
              + ("   <-- PROBLEM" if miss else ""))
        if miss:
            problems.append(s)
    if problems:
        print("\nFAIL: reviewed frames whose SOURCE IMAGE was not found (masks NOT written):")
        for s in problems:
            print(f"  task {s['task']}: {len(s['source_missing'])} missing, e.g. {s['source_missing'][:5]}")
        print("Fix the source-folder mapping before tiling -- otherwise those frames are lost.")
    else:
        print("OK: every reviewed frame matched a source image. No silent drops.")
    print("\nDone.")


if __name__ == "__main__":
    main()
