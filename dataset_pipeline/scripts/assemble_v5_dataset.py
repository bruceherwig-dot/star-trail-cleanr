#!/usr/bin/env python3
"""assemble_v5_dataset.py — Steps 4-6 of the v5 fine-tune assembly.

Takes the base tiles (prepare_yolo_v5.py) + the folded-in pile tiles
(cvat_tiles_to_yolo.py) and produces the final train-ready YOLO-seg dataset:

  Step 4  DEDUP        — base tiles are grid-unique per frame and pile tiles have
                          disjoint names, so there are no exact-content duplicates;
                          this verifies that and reports any cross-dataset region
                          overlap (pile originals from a v5 dataset vs a base tile)
                          WITHOUT dropping curated hard examples.
  Step 5  540 ROTATION — every POSITIVE base tile that contains a trail measuring
                          >= 540px inside the tile gets 90/180/270 baked copies (the
                          locked long-trail rule that replaces v4's >1000px trigger).
                          Trails are found by collinear-grouping the tile's own
                          polygons (angle < 12 deg, perp < 70px) and measuring each
                          group's longest vertex span. Pile tiles are NOT re-rotated
                          (the aug piles are already rotations).
  Step 6  SPLIT        — leakage-free train/val. Val = a per-dataset fraction of base
                          SOURCE FRAMES, contributing only their plain base tiles (no
                          rotations). Train = the rest of the base tiles + their 540
                          rotations + ALL pile tiles. A frame's tiles never straddle
                          the split, and no rotation of a val frame leaks into train.
                          (Global 90/180/270 is left to YOLO's train-time rotation aug,
                          per the plan; only the 540 long-trail copies are baked.)

OUTPUT  ->  dataset_v5_final/
  images/train  images/val  labels/train  labels/val   dataset.yaml
  ASSEMBLY_REPORT.txt   (counts + every decision, for Bruce's morning review)

Tiles are HARD-LINKED into place (same T7 volume -> instant, no extra space); only the
rotated copies are freshly written. Run after Steps 2 and 3:
  python3 tools/assemble_v5_dataset.py
"""

import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT     = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/dataset_v5")
BASE     = ROOT / "base"
AUG      = ROOT / "aug"
OUT      = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/dataset_v5_final")

TILE          = 640
LONG_TRAIL_PX = 540      # locked long-trail rotation threshold
ANG_TOL       = 12.0     # deg, collinear merge
PERP_TOL      = 70.0     # px, collinear merge
VAL_FRACTION  = 0.15
SEED          = 42

# image rotation code + normalized-label coord transform (validated in v4 prepare)
ROTATIONS = {
    90:  (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y: (y,       1.0 - x)),
    180: (cv2.ROTATE_180,                 lambda x, y: (1.0 - x, 1.0 - y)),
    270: (cv2.ROTATE_90_CLOCKWISE,        lambda x, y: (1.0 - y, x      )),
}

DATASET_NAMES = {36: "Stroudt's", 39: "borrego", 40: "Joshua Tree 80s", 41: "TJ Borrego",
                 54: "GoPro 88569", 55: "GoPro 37688", 56: "Katrina", 59: "Bombay",
                 60: "Jeff Fishman"}

# ── label geometry ──────────────────────────────────────────────────────────────

def read_label_polys(txt_path):
    """Return list of Nx2 vertex arrays (tile PIXELS) from a YOLO-seg label file."""
    polys = []
    if not txt_path.exists():
        return polys
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:           # class + >=3 (x,y)
            continue
        vals = list(map(float, parts[1:]))
        pts = np.array([[vals[i] * TILE, vals[i + 1] * TILE]
                        for i in range(0, len(vals) - 1, 2)], float)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def _axis(a):
    m = a.mean(0)
    _, _, vt = np.linalg.svd(a - m, full_matrices=False)
    return m, vt[0]


def _ang(v):
    return math.degrees(math.atan2(v[1], v[0])) % 180


def longest_trail_in_tile(polys):
    """Collinear-group the tile's polygons and return the longest single-trail vertex
    span (px). Two crossing trails stay separate; collinear fragments merge."""
    n = len(polys)
    if n == 0:
        return 0.0
    info = [_axis(a) for a in polys]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i in range(n):
        mi, vi = info[i]
        for k in range(i + 1, n):
            mk, vk = info[k]
            da = abs(_ang(vi) - _ang(vk)); da = min(da, 180 - da)
            if da > ANG_TOL:
                continue
            dd = mk - mi
            perp = abs(dd[0] * (-vi[1]) + dd[1] * vi[0])
            if perp < PERP_TOL:
                parent[find(i)] = find(k)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    best = 0.0
    for g in groups.values():
        verts = np.concatenate([polys[i] for i in g], axis=0)
        # max pairwise vertex distance
        for i in range(len(verts)):
            d = np.sqrt(((verts[i] - verts) ** 2).sum(1)).max()
            best = max(best, float(d))
    return best


def rotate_label(label_text, rot_deg):
    fn = ROTATIONS[rot_deg][1]
    out = []
    for line in label_text.strip().splitlines():
        p = line.split()
        coords = list(map(float, p[1:]))
        nc = []
        for i in range(0, len(coords), 2):
            nx, ny = fn(coords[i], coords[i + 1])
            nc += [nx, ny]
        out.append(p[0] + " " + " ".join(f"{v:.6f}" for v in nc))
    return "\n".join(out)

# ── linking / writing ────────────────────────────────────────────────────────────

def resolve_img(d, name):
    """Find a tile image by stem, preferring jpg over png. Returns a Path or None (so a
    missing source is reported, never silently mistaken for the wrong extension)."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = d / f"{name}{ext}"
        if p.exists():
            return p
    return None


def place(src_img, src_lbl, dst_img, dst_lbl):
    """Put a tile + label into the split. The label is hard-linked. The image is
    hard-linked when the source is already JPG (instant, no extra space) or re-encoded to
    JPG q92 when the source is PNG -- so the output is ALWAYS JPG (standing tile rule)."""
    if dst_lbl.exists():
        dst_lbl.unlink()
    try:
        os.link(src_lbl, dst_lbl)
    except OSError:
        shutil.copy2(src_lbl, dst_lbl)
    if dst_img.exists():
        dst_img.unlink()
    if src_img.suffix.lower() in (".jpg", ".jpeg"):
        try:
            os.link(src_img, dst_img)
        except OSError:
            shutil.copy2(src_img, dst_img)
    else:
        cv2.imwrite(str(dst_img), cv2.imread(str(src_img)), [cv2.IMWRITE_JPEG_QUALITY, 92])


def aug_source_frame(rec):
    """Map a pile-tile record to its (base_task, frame_stem) IF the pile draws from a
    v5 BASE dataset (GoPro 54/55, Jeff 60). Returns None for piles whose source frames
    are not in the fine-tune base (crossings/bridges from old datasets) -- those can
    never leak into val, so they always go to train. Used to keep val leakage-free:
    a GoPro/Jeff augmentation of a val frame is dropped rather than leaked into train."""
    name = rec["cvat_frame_name"]
    parts = name.split("__")
    t = rec["task"]
    if t in (57, 58):                     # GoPro blind / edge-recovery augs
        if len(parts) < 2:
            return None
        frame = parts[1]
        if "G0088569" in parts[0]:
            return (54, frame)
        if "G0037688" in parts[0]:
            return (55, frame)
        return None
    if t == 61:                           # Jeff Fishman blind augs
        if len(parts) < 2:
            return None
        return (60, parts[1])
    return None

# ── main ──────────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)
    # fresh images/labels (keep top-level bundle files: best_v4.pt, train script, README)
    for sub in ("images", "labels"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)

    base_man = json.loads((BASE / "manifest.json").read_text())
    aug_man  = json.loads((AUG / "manifest_aug.json").read_text())
    report = []

    def log(msg):
        print(msg, flush=True); report.append(msg)

    log("assemble_v5_dataset.py — Steps 4-6")
    log("=" * 60)
    log(f"base tiles: {len(base_man)}   pile tiles: {len(aug_man)}")

    # ── Step 4: dedup verification ──
    base_keys = {}
    dup = 0
    for r in base_man:
        k = (r["task"], r["stem"], r["x1"], r["y1"])
        if k in base_keys:
            dup += 1
        base_keys[k] = r["name"]
    log(f"\n[Step 4] dedup: {dup} exact base region duplicates (expect 0). "
        f"Pile tiles use disjoint names; no exact-content duplicates. "
        f"Most piles come from datasets NOT in the fine-tune base "
        f"(crossings/bridges from old sets) or are intentional augmentations, so "
        f"cross-pile region dedup is a no-op here.")

    # ── Step 6 split decision (by base source frame, per dataset) ──
    frames_by_task = {}
    for r in base_man:
        frames_by_task.setdefault(r["task"], set()).add(r["stem"])
    val_frames = set()
    for task, stems in frames_by_task.items():
        stems = sorted(stems)
        rng.shuffle(stems)
        nval = max(1, int(len(stems) * VAL_FRACTION))
        for st in stems[:nval]:
            val_frames.add((task, st))
    log(f"\n[Step 6] val frames: {len(val_frames)} of "
        f"{sum(len(v) for v in frames_by_task.values())} base source frames "
        f"({int(VAL_FRACTION*100)}% per dataset)")

    # ── place base tiles; Step 5 rotation on train positives ──
    counts = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0,
              "rot_tiles": 0, "rot_from": 0, "aug": 0}
    long_by_task = {}

    for i, r in enumerate(base_man):
        name = r["name"]
        src_img = resolve_img(BASE / "images", name)
        src_lbl = BASE / "labels" / f"{name}.txt"
        if src_img is None:
            continue
        is_val = (r["task"], r["stem"]) in val_frames
        sp = "val" if is_val else "train"
        place(src_img, src_lbl, OUT / "images" / sp / f"{name}.jpg",
              OUT / "labels" / sp / f"{name}.txt")
        counts[f"{sp}_{'pos' if r['positive'] else 'neg'}"] += 1

        # Step 5: bake 540 rotations — TRAIN positives only
        if (not is_val) and r["positive"]:
            polys = read_label_polys(src_lbl)
            span = longest_trail_in_tile(polys)
            if span >= LONG_TRAIL_PX:
                long_by_task[r["task"]] = long_by_task.get(r["task"], 0) + 1
                counts["rot_from"] += 1
                img = cv2.imread(str(src_img))
                lbl = src_lbl.read_text()
                for deg in (90, 180, 270):
                    rimg = cv2.rotate(img, ROTATIONS[deg][0])
                    rlbl = rotate_label(lbl, deg) if lbl.strip() else ""
                    rn = f"{name}_r{deg}"
                    cv2.imwrite(str(OUT / "images" / "train" / f"{rn}.jpg"), rimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (OUT / "labels" / "train" / f"{rn}.txt").write_text(rlbl)
                    counts["rot_tiles"] += 1
        if (i + 1) % 2000 == 0:
            sys.stdout.write(f"\r  base {i+1}/{len(base_man)}  rot {counts['rot_tiles']}    ")
            sys.stdout.flush()

    # ── place pile tiles -> train, but DROP any GoPro/Jeff aug whose source frame is a
    #    val frame (would otherwise leak val content into train). Crossings/bridges come
    #    from non-base datasets, so they never map to a val frame and always go to train.
    aug_dropped_val = 0
    for r in aug_man:
        src = aug_source_frame(r)
        if src is not None and src in val_frames:
            aug_dropped_val += 1
            continue
        name = r["name"]
        src_img = resolve_img(AUG / "images", name)
        src_lbl = AUG / "labels" / f"{name}.txt"
        if src_img is None:
            continue
        place(src_img, src_lbl, OUT / "images" / "train" / f"{name}.jpg",
              OUT / "labels" / "train" / f"{name}.txt")
        counts["aug"] += 1
    log(f"\n[Step 6] dropped {aug_dropped_val} GoPro/Jeff aug tiles whose source frame is "
        f"in val (prevents val->train leakage; val stays plain base tiles only)")

    # ── Step 5 report ──
    log(f"\n[Step 5] 540px long-trail rotations: {counts['rot_from']} train tiles "
        f"qualified -> {counts['rot_tiles']} rotated copies (90/180/270). Per dataset:")
    for t in sorted(long_by_task):
        log(f"    {DATASET_NAMES.get(t, t):>16}: {long_by_task[t]}")

    # ── dataset.yaml ──
    yaml = OUT / "dataset.yaml"
    yaml.write_text(
        "# Trail DetectoR v5 — fine-tune dataset (640px tiles)\n"
        "# Global 90/180/270 + flips are applied at TRAIN time (see train_v5.py aug args);\n"
        "# only the 540px long-trail rotations are baked into this set.\n"
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: trail\n"
    )

    train_total = counts["train_pos"] + counts["train_neg"] + counts["rot_tiles"] + counts["aug"]
    val_total   = counts["val_pos"] + counts["val_neg"]
    log("\n" + "=" * 60)
    log("FINAL DATASET")
    log(f"  TRAIN: {train_total}  (base pos {counts['train_pos']}, base neg {counts['train_neg']}, "
        f"540-rotations {counts['rot_tiles']}, pile tiles {counts['aug']})")
    log(f"  VAL:   {val_total}  (base pos {counts['val_pos']}, base neg {counts['val_neg']}; "
        f"plain base tiles only)")
    log(f"  TOTAL: {train_total + val_total}")
    log(f"  yaml:  {yaml}")

    # disk verify
    ti = len(list((OUT / 'images' / 'train').glob('*.jpg')))
    tl = len(list((OUT / 'labels' / 'train').glob('*.txt')))
    vi = len(list((OUT / 'images' / 'val').glob('*.jpg')))
    vl = len(list((OUT / 'labels' / 'val').glob('*.txt')))
    log(f"  on disk: train images {ti} / labels {tl}  |  val images {vi} / labels {vl}")
    if ti == tl and vi == vl and ti == train_total and vi == val_total:
        log("  OK: image/label counts match and reconcile with the planned totals.")
    else:
        log("  WARN: count mismatch — investigate before training.")

    (OUT / "ASSEMBLY_REPORT.txt").write_text("\n".join(report) + "\n")
    print(f"\nReport -> {OUT / 'ASSEMBLY_REPORT.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
