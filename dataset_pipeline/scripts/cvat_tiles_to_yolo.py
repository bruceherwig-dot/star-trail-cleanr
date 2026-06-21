#!/usr/bin/env python3
"""cvat_tiles_to_yolo.py — Step 3 of the v5 fine-tune assembly: fold the pre-cut,
already-reviewed 640px tile piles from CVAT into the dataset as YOLO-seg tiles.

These piles are NOT full frames — they are curated hard-example tiles (bridge misses,
crossings, GoPro blind spots, Jeff Fishman blind trails) plus their rotation
augmentations. Each CVAT "frame" IS one 640 tile. So there is no tiling to do here;
this step just pulls the reviewed polygons + the tile image (BOTH from CVAT, the single
source of truth) and writes a YOLO label + the image.

WHERE IT FITS
  prepare_yolo_v5.py (base tiles)  ->  THIS SCRIPT (folded-in pile tiles)
                                    ->  dedup  ->  540 rotation rule  ->  split  -> train

PILES (decided with Bruce):
  42 Bridge misses (orig)     45 Bridge aug
  46 Crossings (orig)         47 Crossing aug  -> FIRST 300 tiles only (rest unreviewed)
  57 GoPro blind-spot aug     58 GoPro edge-trail recovery
  61 Jeff Fishman blind aug
  (GoPro is brand-new to training as of v5; v4 had none.)

orig vs aug TAG: written into each manifest record so the later 540 rotation step can
rotate originals but NOT re-rotate tiles that are already a rotation.

WHAT IT WRITES  ->  dataset_v5/aug/
  images/<tile>.png     downloaded from CVAT at original quality (exact uploaded tile)
  labels/<tile>.txt     YOLO-seg ("0 x y ...", normalized to the 640 tile; empty if the
                        reviewed tile has no trail -> a curated negative)
  manifest_aug.json     per-tile: name, task, kind (orig/aug), cvat_frame_name, n_polys

HOW TO RUN
  python3 tools/cvat_tiles_to_yolo.py     (no args; downloads ~2,800 tiles -> background it)
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────

CVAT = "http://localhost:8080"
USER = "bherwig2"
T7_OUT = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/dataset_v5/aug")

# task_id -> (kind, max_frames)   kind in {"orig","aug"}; max_frames None = all
PILES = {
    42: ("orig", None),   # Bridge misses (originals)
    45: ("aug",  None),   # Bridge augmentation
    46: ("orig", None),   # Crossings (originals)
    47: ("aug",  300),    # Crossing augmentation — first 300 only (reviewed)
    57: ("aug",  None),   # GoPro blind-spot augmentation
    58: ("aug",  None),   # GoPro edge-trail rotation recovery
    61: ("aug",  None),   # Jeff Fishman blind augmentation
}

NAMES = {42: "Bridge misses (orig)", 45: "Bridge aug", 46: "Crossings (orig)",
         47: "Crossing aug", 57: "GoPro blind aug", 58: "GoPro edge recovery",
         61: "Jeff blind aug"}

# ── CVAT ────────────────────────────────────────────────────────────────────────

def session():
    pw = (Path.home() / ".star_trail_cleanr" / "cvat_credentials").read_text().strip()
    s = requests.Session(); s.auth = (USER, pw)
    return s


def download_frame(s, job_id, number):
    """Return the original-quality tile image (BGR ndarray) for a CVAT frame, or None.
    Retries a couple times so a transient hiccup doesn't silently drop a tile."""
    for attempt in range(3):
        try:
            r = s.get(f"{CVAT}/api/jobs/{job_id}/data",
                      params={"type": "frame", "number": number, "quality": "original"},
                      timeout=120)
            if r.status_code == 200 and r.content:
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception:
            pass
        time.sleep(1.0 + attempt)
    return None


def poly_to_yolo_line(points, w, h):
    """CVAT flat [x0,y0,...] tile-pixel polygon -> YOLO-seg line normalized to the tile.
    Returns None if fewer than 3 vertices."""
    xs = points[0::2]; ys = points[1::2]
    if len(xs) < 3:
        return None
    # clip to [0,1] -- a few CVAT vertices sit a hair past the tile edge (e.g. 1.01),
    # which YOLO rejects as "non-normalized/out of bounds" and drops the whole tile.
    def c(v, d):
        return min(1.0, max(0.0, v / d))
    coords = " ".join(f"{c(x, w):.6f} {c(y, h):.6f}" for x, y in zip(xs, ys))
    return f"0 {coords}"

# ── Per-pile ──────────────────────────────────────────────────────────────────

def fold_pile(s, task_id, kind, max_frames, img_out, lbl_out, manifest):
    """Download + label every (capped) tile of one CVAT pile. Returns a summary dict."""
    job_id = s.get(f"{CVAT}/api/jobs?task_id={task_id}", timeout=60).json()["results"][0]["id"]
    meta = s.get(f"{CVAT}/api/jobs/{job_id}/data/meta", timeout=60).json()
    frames = meta["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]

    ann = s.get(f"{CVAT}/api/jobs/{job_id}/annotations", timeout=120).json()
    polys_by_frame = {}
    for sh in ann.get("shapes", []):
        if sh.get("type") == "polygon":
            polys_by_frame.setdefault(sh["frame"], []).append(sh["points"])

    summ = {"task": task_id, "tiles": 0, "with_trail": 0, "negatives": 0,
            "download_failed": [], "polys": 0}

    for fi, f in enumerate(frames):
        img = download_frame(s, job_id, fi)
        if img is None:
            summ["download_failed"].append(f["name"])
            continue
        h, w = img.shape[:2]

        lines = []
        for pts in polys_by_frame.get(fi, []):
            line = poly_to_yolo_line(pts, w, h)
            if line:
                lines.append(line)

        stem = Path(f["name"]).stem
        name = f"t{task_id:02d}__{stem}"
        cv2.imwrite(str(img_out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_out / f"{name}.txt").write_text("\n".join(lines))
        manifest.append({"name": name, "task": task_id, "kind": kind,
                         "cvat_frame_name": f["name"], "n_polys": len(lines)})

        summ["tiles"] += 1
        summ["polys"] += len(lines)
        if lines:
            summ["with_trail"] += 1
        else:
            summ["negatives"] += 1

        if (fi + 1) % 25 == 0 or fi + 1 == len(frames):
            sys.stdout.write(f"\r  task {task_id} ({NAMES[task_id]}) — "
                             f"{fi+1}/{len(frames)} tiles, {summ['polys']} polys    ")
            sys.stdout.flush()

    print(f"\r  task {task_id} done — {summ['tiles']} tiles "
          f"({summ['with_trail']} with trail, {summ['negatives']} negative), "
          f"{summ['polys']} polys, {len(summ['download_failed'])} download-failed")
    return summ

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    img_out = T7_OUT / "images"; lbl_out = T7_OUT / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    s = session()

    print("cvat_tiles_to_yolo.py — Step 3: fold in reviewed pile tiles")
    print(f"Output: {T7_OUT}")
    print("=" * 60)

    manifest = []
    summaries = []
    for task_id in sorted(PILES):
        kind, cap = PILES[task_id]
        cap_txt = f"first {cap}" if cap else "all"
        print(f"\nTask {task_id}: {NAMES[task_id]} [{kind}, {cap_txt}]")
        try:
            summaries.append(fold_pile(s, task_id, kind, cap, img_out, lbl_out, manifest))
        except Exception as e:
            print(f"  ERROR: {e}")

    (T7_OUT / "manifest_aug.json").write_text(json.dumps(manifest, indent=1))

    print("\n" + "=" * 60)
    print("RECONCILE (folded-in pile tiles):")
    tot_tiles = tot_polys = 0
    problems = []
    for s_ in summaries:
        tot_tiles += s_["tiles"]; tot_polys += s_["polys"]
        flag = "   <-- PROBLEM" if s_["download_failed"] else ""
        if s_["download_failed"]:
            problems.append(s_)
        print(f"  task {s_['task']:>3}: {s_['tiles']:>4} tiles  "
              f"{s_['with_trail']:>4} w/trail  {s_['negatives']:>4} neg  "
              f"{s_['polys']:>5} polys  {len(s_['download_failed']):>3} dl-failed{flag}")
    print(f"  TOTAL: {tot_tiles} tiles, {tot_polys} polys")
    print(f"  manifest: {len(manifest)} records -> {T7_OUT / 'manifest_aug.json'}")

    if problems:
        print("\nFAIL: some tiles failed to download (NOT silently skipped — listed):")
        for s_ in problems:
            print(f"  task {s_['task']} ({len(s_['download_failed'])}): {s_['download_failed'][:5]}")
    else:
        print("\nOK: every pile tile downloaded and labeled. No silent drops.")
    print("\nDone.")


if __name__ == "__main__":
    main()
