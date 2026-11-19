#!/usr/bin/env python3
"""
Known-problems test runner.

Loads every entry in tests/known_problems.jsonl, runs detect_frame_polygon,
and checks whether the result matches the expected outcome.

Pass/fail logic per problem type:
  parallel_merge + absent:
      FAIL if any component in the bbox has minor_axis > 60px AND elongation < 5.0
      (two-criteria fat-blob check; thin crossing trails are not flagged)

  bad_merge / bad_gap_bridge / x_crossing_not_split + absent:
      FAIL if any component whose bbox overlaps the problem bbox has
      bbox_IoU(component_bbox, problem_bbox) > 0.30
      (a merged polygon spans the whole problem bbox; fixed narrow trails do not)

  bad_merge / bad_gap_bridge + absent + centroid-type pass_condition:
      FAIL if any component centroid falls inside the problem bbox

  missed_trail / present:
      PASS if component centroid within 100px of (cx, cy) with area > min_area/2
      PASS (secondary) if target pixel (cx, cy) is directly inside a detected region
      with area > min_area/2 (handles edge trails and multi-trail merges where
      the polygon centroid is displaced from the annotated target point)

  needs_coordinates / bbox=null:
      SKIP

Usage:
  python3 tests/run_known_problems.py
  python3 tests/run_known_problems.py --id greg_meyer_143A8770_parallel_fat_blob
  python3 tests/run_known_problems.py --type parallel_merge
"""

import sys
import os
import json
import time
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from skimage.measure import label as sklabel, regionprops as skregionprops

JSONL = os.path.join(os.path.dirname(__file__), "known_problems.jsonl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "best.pt")


def _bbox_iou(comp_ys, comp_xs, px1, py1, px2, py2):
    """Bounding-box IoU: component's axis-aligned bbox vs. problem bbox."""
    if len(comp_xs) == 0:
        return 0.0
    cx1 = int(comp_xs.min()); cx2 = int(comp_xs.max())
    cy1 = int(comp_ys.min()); cy2 = int(comp_ys.max())
    comp_area = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    prob_area = (px2 - px1) * (py2 - py1)
    inter_x1 = max(cx1, px1); inter_x2 = min(cx2, px2)
    inter_y1 = max(cy1, py1); inter_y2 = min(cy2, py2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = comp_area + prob_area - inter
    return inter / union if union > 0 else 0.0


def _minor_elong(comp_mask):
    """Return (minor_axis, elongation) for the largest region in comp_mask."""
    from skimage.measure import label as sklabel2, regionprops as skregionprops2
    lbl = sklabel2(comp_mask)
    props = skregionprops2(lbl)
    if not props:
        return 0.0, 999.0
    p = max(props, key=lambda x: x.area)
    minor = p.axis_minor_length
    major = p.axis_major_length
    elong = major / max(minor, 1.0)
    return minor, elong


def _in_bbox(coords, px1, py1, px2, py2):
    return int(np.count_nonzero(
        (coords[:, 1] >= px1) & (coords[:, 1] <= px2) &
        (coords[:, 0] >= py1) & (coords[:, 0] <= py2)
    ))


def _check_static_fp_absent(model, image_path, neighbor_paths, px1, py1, px2, py2):
    """For static_fp absent entries: run suppressor on a mini-batch and check the target frame.

    Runs detect_frame_polygon on each neighbor frame, then the target frame.
    Assembles them into a mini-batch with the target as the last frame, calls
    _suppress_static_fps, and checks whether the FP in the target mask is gone.
    PASS if no component centroid falls inside the problem bbox after suppression.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from modules.detect_trails import detect_frame_polygon
    from astro_clean_v5 import _suppress_static_fps

    t0 = time.time()
    masks_all = []
    for np_ in neighbor_paths:
        m = detect_frame_polygon(model, np_, tile_size=640, overlap=0.2)
        masks_all.append(m if m is not None else np.zeros((1, 1), dtype=np.uint8))

    target_mask = detect_frame_polygon(model, image_path, tile_size=640, overlap=0.2)
    if target_mask is None:
        return "ERROR", "detect_frame_polygon returned None for target frame"
    masks_all.append(target_mask)

    core_start = len(masks_all) - 1
    core_end = len(masks_all)
    _suppress_static_fps(masks_all, core_start, core_end)
    elapsed = time.time() - t0

    post_mask = masks_all[core_start]
    labeled = sklabel(post_mask)
    regions = skregionprops(labeled)

    for r in regions:
        cy_c, cx_c = r.centroid
        if px1 <= cx_c <= px2 and py1 <= cy_c <= py2:
            return "FAIL", (f"FP centroid still present after suppression: "
                            f"cx={cx_c:.0f} cy={cy_c:.0f} area={r.area} ({elapsed:.1f}s)")
    return "PASS", f"FP suppressed (no centroid in bbox after suppressor) ({elapsed:.1f}s)"


def _check_parallel_merge_absent(model, image_path, px1, py1, px2, py2):
    """For parallel_merge absent entries: check per-group polygon widths.

    Crosses two crossing narrow polygons in the output mask into one X-shaped
    connected component that would look fat to regionprops. The right check is
    whether any fitted GROUP POLYGON (pre-mask) has width > 60px. That directly
    measures fat-blob-ness without being confused by crossing thin trails.
    """
    from modules.detect_trails import _sahi_predict, _load_as_rgb
    from modules.trail_grouper import filter_masks_with_props, group_detections, fit_polygon

    t0 = time.time()
    img_rgb, h, w = _load_as_rgb(image_path)
    preds = _sahi_predict(model, img_rgb, tile_size=640, overlap=0.2)
    _, det_list, _ = filter_masks_with_props(preds, h, w, img=img_rgb, debug_out={})
    groups = group_detections(det_list)
    elapsed = time.time() - t0

    worst_width = 0.0
    for grp in groups:
        n_in = sum(_in_bbox(det_list[k]["coords"], px1, py1, px2, py2) for k in grp)
        if n_in < 50:
            continue
        corners, width, _ = fit_polygon(grp, det_list)
        if width > worst_width:
            worst_width = width

    if worst_width > 60.0:
        return "FAIL", f"Fat polygon in bbox: width={worst_width:.0f}px ({elapsed:.1f}s)"
    return "PASS", f"Max polygon width={worst_width:.0f}px ({elapsed:.1f}s)"


def run_entry(entry, model):
    """Run one known-problem entry. Returns (result, detail_str)."""
    from modules.detect_trails import detect_frame_polygon, _load_as_rgb

    confidence = entry.get("confidence", "")
    if confidence == "needs_coordinates" or entry.get("bbox") is None:
        return "SKIP", "No coordinates"

    image_path = entry["image_path"]
    problem_type = entry["problem_type"]
    expected = entry["expected_outcome"]
    bbox = entry["bbox"]  # [x1, y1, x2, y2]
    pass_condition = entry.get("pass_condition", "")
    px1, py1, px2, py2 = bbox

    # ── parallel_merge absent: per-group polygon width check ─────────────────
    if problem_type == "parallel_merge" and expected == "absent":
        return _check_parallel_merge_absent(model, image_path, px1, py1, px2, py2)

    # ── static_fp absent: mini-batch suppressor check ────────────────────────
    if problem_type == "static_fp" and expected == "absent":
        neighbor_paths = entry.get("neighbor_frames", [])
        return _check_static_fp_absent(model, image_path, neighbor_paths, px1, py1, px2, py2)

    t0 = time.time()
    mask = detect_frame_polygon(model, image_path, tile_size=640, overlap=0.2)
    elapsed = time.time() - t0

    if mask is None:
        return "ERROR", f"detect_frame_polygon returned None ({elapsed:.1f}s)"

    h, w = mask.shape

    # ── Label all connected components ───────────────────────────────────────
    labeled = sklabel(mask)
    regions = skregionprops(labeled)

    # ── EXPECTED ABSENT ───────────────────────────────────────────────────────
    if expected == "absent":

        # Which components have any pixel in the problem bbox?
        comps_in_bbox = []
        for r in regions:
            ys, xs = np.where(labeled == r.label)
            in_box = (xs >= px1) & (xs <= px2) & (ys >= py1) & (ys <= py2)
            n = int(in_box.sum())
            if n > 50:
                comps_in_bbox.append((r.label, ys, xs, r))

        if not comps_in_bbox:
            return "PASS", f"No components in bbox ({elapsed:.1f}s)"

        # ── centroid-type pass_condition ──────────────────────────────────
        if "centroid falls within" in pass_condition.lower():
            worst_cx, worst_cy = None, None
            for lbl_id, ys, xs, r in comps_in_bbox:
                cy_c = float(ys.mean()); cx_c = float(xs.mean())
                if px1 <= cx_c <= px2 and py1 <= cy_c <= py2:
                    worst_cx, worst_cy = cx_c, cy_c
                    break
            if worst_cx is not None:
                return "FAIL", (f"Centroid in bbox: cx={worst_cx:.0f} cy={worst_cy:.0f} "
                                f"({elapsed:.1f}s)")
            return "PASS", f"No centroid inside bbox ({elapsed:.1f}s)"

        # ── Default: bounding-box IoU check ──────────────────────────────
        worst_iou = 0.0
        worst_area = 0
        for lbl_id, ys, xs, r in comps_in_bbox:
            biou = _bbox_iou(ys, xs, px1, py1, px2, py2)
            if biou > worst_iou:
                worst_iou = biou
                worst_area = r.area
        if worst_iou > 0.30:
            return "FAIL", (f"bbox_IoU={worst_iou:.3f} area={worst_area} "
                            f"({elapsed:.1f}s)")
        return "PASS", f"Max bbox_IoU={worst_iou:.3f} ({elapsed:.1f}s)"

    # ── EXPECTED PRESENT ──────────────────────────────────────────────────────
    if expected == "present":
        cx_target = entry.get("cx", 0)
        cy_target = entry.get("cy", 0)
        min_area = entry.get("area", 1000)

        best_dist = 9999.0
        best_area = 0
        for r in regions:
            cy_c, cx_c = r.centroid
            dist = math.sqrt((cx_c - cx_target) ** 2 + (cy_c - cy_target) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_area = r.area

        if best_dist <= 100.0 and best_area >= min_area * 0.5:
            return "PASS", (f"Found: dist={best_dist:.0f}px area={best_area} "
                            f"({elapsed:.1f}s)")

        # Secondary check: target pixel directly covered by a detected region.
        # Centroid-based distance fails when the polygon is larger than the
        # target trail (e.g. edge trails fitted to full extent, multi-trail merges).
        lbl_at_target = labeled[cy_target, cx_target] if (
            0 <= cy_target < mask.shape[0] and 0 <= cx_target < mask.shape[1]
        ) else 0
        if lbl_at_target > 0:
            for r in regions:
                if r.label == lbl_at_target and r.area >= min_area * 0.5:
                    return "PASS", (f"Found: target pixel covered area={r.area:.0f} "
                                    f"({elapsed:.1f}s)")

        return "FAIL", (f"Closest: dist={best_dist:.0f}px area={best_area} "
                        f"(need dist<=100 area>={min_area//2}) ({elapsed:.1f}s)")

    return "ERROR", f"Unknown expected_outcome: {expected}"


def main():
    parser = argparse.ArgumentParser(description="Run known-problem regression tests")
    parser.add_argument("--id", help="Run only this entry id")
    parser.add_argument("--type", help="Run only entries with this problem_type")
    args = parser.parse_args()

    entries = []
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if args.id:
        entries = [e for e in entries if e["id"] == args.id]
    if args.type:
        entries = [e for e in entries if e["problem_type"] == args.type]

    if not entries:
        print("No entries match filter.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}")
    from modules.detect_trails import load_model
    model, _dev = load_model(MODEL_PATH)

    print(f"\nRunning {len(entries)} entries...\n")

    col_w = [44, 22, 8, 60]
    hdr = (f"{'Entry':<{col_w[0]}}  {'Frame':<{col_w[1]}}  "
           f"{'Result':<{col_w[2]}}  {'Detail':<{col_w[3]}}")
    sep = "-" * (sum(col_w) + 8)
    print(hdr)
    print(sep)

    results = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    for entry in entries:
        eid = entry["id"]
        frame = entry["frame"]
        result, detail = run_entry(entry, model)
        results[result] += 1
        tag = f"[{result}]"
        print(f"{eid:<{col_w[0]}}  {frame:<{col_w[1]}}  {tag:<{col_w[2]}}  {detail}")

    print(sep)
    total = len(entries)
    print(f"  PASS: {results['PASS']}/{total}   FAIL: {results['FAIL']}   "
          f"SKIP: {results['SKIP']}   ERROR: {results['ERROR']}")
    if results["FAIL"] > 0 or results["ERROR"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
