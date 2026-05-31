"""
Detection pipeline (modular rebuild, started 2026-05-29).

Finds airplane and satellite trails in a single star-trail frame and returns a
polygon mask of every trail. This replaces the organically-grown
detect_trails.py + trail_grouper.py pair, which had become hard to reason about.

Design goals:
- Trust SAHI. The raw tiled detections are the signal. Filter as little as
  possible. The historical failure mode was over-filtering valid detections.
- Modular stages. Each stage is a separate function with a clear input/output
  contract and can be toggled on or off via StageConfig for isolated testing.
- Rich logging. Every gate and decision appends a structured record to the
  per-frame log so any miss can be diagnosed by reading the log, without a
  re-run.

Stage order:
  1. tiled_inference   SAHI tiled detection, skipping tiles fully covered by the
                       foreground mask.
  2. fit_polygons      Strip-based curved-trail polygon fit, one per detection
                       group.
  3. fallback_polys    Single-detection polygon for any group whose primary fit
                       covers under 70% of its detections.
  4. link_gaps         Join collinear polygon fragments separated by a small gap
                       (deterministic, tile-relative threshold).
  5. seam_second_pass  Targeted re-inference on the strip between two collinear
                       fragments when the gap is large and SAHI produced nothing
                       there (true seam gap, not a filter gap).
  6. suppress_fp       Drop static false positives by comparing against the same
                       region in neighbour frames.

The live app still calls detect_trails.detect_frame_polygon. This module is NOT
wired into the app until it passes the known_problems.jsonl regression suite.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import math

import cv2
import numpy as np
from skimage.measure import label as sklabel, regionprops as skregionprops

from .io_safe import robust_imread
from .crossing_splitter import split_crossing
from .trail_grouper import (
    group_detections, fit_polygon, fit_curved_group, _group_angle_spread,
    _try_split_parallel, _pred_to_mask,
    _CURVED_MIN_XSPAN, _CURVED_MIN_ANGLE_SPREAD, MIN_AREA, _REF_FRAME_PX,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class StageConfig:
    """Per-stage on/off switches plus core tunables. Toggle a stage off to
    isolate behaviour during testing."""
    tiled_inference: bool = True
    fit_polygons: bool = True
    fallback_polys: bool = True
    link_gaps: bool = True
    seam_second_pass: bool = True
    suppress_fp: bool = True

    # Core tunables (carried over from the old pipeline; revisit per stage).
    tile_size: int = 640
    overlap: float = 0.2
    fallback_coverage_threshold: float = 0.70
    skip_fully_masked_tiles: bool = True

    # Stage 2 filter gates.
    min_aspect: float = 2.0
    # The old "fat blob" gate dropped any wide detection whose fill fraction fell
    # below 62.5%. It was the prime suspect for the 143A8732 filter gap, so it is
    # OFF by default (trust SAHI). When off, detections it WOULD have dropped are
    # still logged so we can see its effect.
    fat_blob_gate: bool = False


# --------------------------------------------------------------------------- #
# Per-frame logging
# --------------------------------------------------------------------------- #

@dataclass
class FrameLog:
    """Accumulates one structured record per frame: every stage that ran, every
    gate that fired, timings, and what each stage added or removed. Serialised
    to one JSONL line."""
    frame_name: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def stage(self, name: str) -> "StageLog":
        """Open a logging context for one stage. Use as a `with` block."""
        return StageLog(self, name)

    def to_jsonl(self) -> str:
        """Return this frame's record as a single JSONL line."""
        record = asdict(self)
        record["total_seconds"] = round(time.time() - self.started_at, 3)
        return json.dumps(record)


class StageLog:
    """Context manager that records one stage's timing, events, and counts into
    the parent FrameLog."""

    def __init__(self, parent: FrameLog, name: str):
        self.parent = parent
        self.name = name
        self.events: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self._t0 = 0.0

    def __enter__(self) -> "StageLog":
        self._t0 = time.time()
        return self

    def event(self, reason: str, **detail: Any) -> None:
        """Record a single gate/decision event, e.g. a detection that passed or
        was dropped and why."""
        self.events.append({"reason": reason, **detail})

    def count(self, key: str, n: int = 1) -> None:
        """Increment a named counter for this stage (e.g. 'passed', 'dropped')."""
        self.counts[key] = self.counts.get(key, 0) + n

    def __exit__(self, *exc: Any) -> None:
        self.parent.stages.append({
            "stage": self.name,
            "seconds": round(time.time() - self._t0, 3),
            "counts": self.counts,
            "events": self.events,
        })


# --------------------------------------------------------------------------- #
# Stage skeletons
#
# Each stage takes the running pipeline state and its StageLog, and returns the
# updated state. None contains real detection logic yet -- they pass through and
# log that they were stubbed. We fill these in one at a time after the structure
# is agreed.
# --------------------------------------------------------------------------- #

@dataclass
class PipelineState:
    """The data flowing between stages. Each stage reads and updates this."""
    image: np.ndarray
    foreground_mask: Optional[np.ndarray] = None
    raw_detections: list[Any] = field(default_factory=list)   # SAHI predictions
    det_list: list[Any] = field(default_factory=list)         # filtered detection props
    groups: list[Any] = field(default_factory=list)           # grouped detection indices
    polygons: list[Any] = field(default_factory=list)         # fitted trail polygon corner sets
    polygon_segs: list[Any] = field(default_factory=list)     # per-polygon binary masks (1 per polygon, for repair + MaskViewR)
    final_mask: Optional[np.ndarray] = None                   # output mask
    stage_seconds: dict = field(default_factory=dict)         # {stage_name: seconds} for timing reporting


# --- Stage 1 helpers (ported verbatim from the proven detect_trails.py path so
#     detection output matches the shipped app exactly) ---------------------- #

class _PredMaskWrap:
    __slots__ = ("bool_mask",)
    def __init__(self, bm):
        self.bool_mask = bm


class _PredScoreWrap:
    __slots__ = ("value",)
    def __init__(self, v):
        self.value = v


class _SyntheticPred:
    """Duck-type replacement for a SAHI ObjectPrediction (same .mask.bool_mask
    and .score.value interface)."""
    __slots__ = ("mask", "score")
    def __init__(self, bool_mask, conf):
        self.mask = _PredMaskWrap(bool_mask)
        self.score = _PredScoreWrap(float(conf))


def _load_as_rgb(image):
    """Normalize a path or array (8/16-bit, gray/BGR/BGRA) to (rgb_uint8, h, w),
    or None on failure."""
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = robust_imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    return img, h, w


def _tile_starts(size, tile_size, stride):
    """Return tile start positions for one dimension, ensuring the final tile
    reaches the edge."""
    starts = []
    x = 0
    while x + tile_size <= size:
        starts.append(x)
        x += stride
    if not starts or starts[-1] + tile_size < size:
        starts.append(size - tile_size)
    return starts


def _sahi_predict(model, img, tile_size, overlap):
    """Run SAHI tiled inference and return the raw prediction list. NMS is
    disabled (match threshold 1.1) so tile-edge fragments survive."""
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(
        image=img, detection_model=model,
        slice_height=tile_size, slice_width=tile_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        perform_standard_pred=False,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=1.1,
        postprocess_class_agnostic=True,
        verbose=0,
    )
    return result.object_prediction_list


def _sahi_predict_skip(model, img_rgb, tile_size, overlap, fg_mask):
    """Tiled YOLO inference that skips 100%-foreground tiles. Returns
    (predictions, n_skipped). Predictions are _SyntheticPred objects."""
    h, w = img_rgb.shape[:2]
    stride = int(tile_size * (1 - overlap))
    xs = _tile_starts(w, tile_size, stride)
    ys = _tile_starts(h, tile_size, stride)

    fm = fg_mask if fg_mask.shape == (h, w) else cv2.resize(
        fg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    kept = []   # (tx, ty, crop_w, crop_h, bgr_crop)
    n_skipped = 0
    for ty in ys:
        for tx in xs:
            ty2 = min(h, ty + tile_size)
            tx2 = min(w, tx + tile_size)
            crop_h, crop_w = ty2 - ty, tx2 - tx
            if (fm[ty:ty2, tx:tx2] > 0).all():
                n_skipped += 1
                continue
            crop_bgr = np.ascontiguousarray(img_rgb[ty:ty2, tx:tx2, ::-1])
            if crop_h < tile_size or crop_w < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:crop_h, :crop_w] = crop_bgr
                crop_bgr = padded
            kept.append((tx, ty, crop_w, crop_h, crop_bgr))

    if not kept:
        return [], n_skipped

    yolo = model.model
    conf_thresh = getattr(model, "confidence_threshold", 0.25)
    results = yolo.predict(
        source=[t[4] for t in kept],
        conf=conf_thresh, verbose=False, imgsz=tile_size,
    )

    preds = []
    for (tx, ty, crop_w, crop_h, _), r in zip(kept, results):
        if r.masks is None:
            continue
        confs = r.boxes.conf.tolist() if r.boxes is not None else []
        for seg_idx, seg_xy in enumerate(r.masks.xy):
            if len(seg_xy) < 3:
                continue
            seg_conf = float(confs[seg_idx]) if seg_idx < len(confs) else 0.0
            local_u8 = np.zeros((tile_size, tile_size), dtype=np.uint8)
            cv2.fillPoly(local_u8, [np.array(seg_xy, dtype=np.int32)], 1)
            global_mask = np.zeros((h, w), dtype=bool)
            global_mask[ty:ty + crop_h, tx:tx + crop_w] = (
                local_u8[:crop_h, :crop_w].astype(bool))
            if global_mask.any():
                preds.append(_SyntheticPred(global_mask, seg_conf))

    return preds, n_skipped


def stage_tiled_inference(state: PipelineState, cfg: StageConfig,
                          log: StageLog, model: Any) -> PipelineState:
    """Stage 1: SAHI tiled inference, skipping fully foreground-masked tiles.
    Produces the raw, unfiltered SAHI predictions (the signal we trust)."""
    loaded = _load_as_rgb(state.image)
    if loaded is None:
        log.event("load_failed")
        return state
    img_rgb, h, w = loaded
    state.image = img_rgb

    if cfg.skip_fully_masked_tiles and state.foreground_mask is not None:
        preds, n_skipped = _sahi_predict_skip(
            model, img_rgb, cfg.tile_size, cfg.overlap, state.foreground_mask)
        log.count("tiles_skipped", n_skipped)
    else:
        preds = _sahi_predict(model, img_rgb, cfg.tile_size, cfg.overlap)

    state.raw_detections = preds
    log.count("raw_detections", len(preds))
    return state


def _props_with_log(mask: np.ndarray, cfg: StageConfig, log: StageLog,
                    skip_aspect: bool = False):
    """Region properties for one detection mask, or None if it fails a gate.
    Same gates as the shipped elongation filter, except the fat-blob gate is
    toggleable and every drop (and every fat-blob the gate would have killed)
    is logged with its reason.

    skip_aspect: when True, bypass the aspect-ratio (elongation) gate. Used for
    crossing-splitter output where the splitter already validated the pieces as
    real trail arms -- re-filtering by shape would kill short stubby tips."""
    area_scale = (mask.shape[0] * mask.shape[1]) / _REF_FRAME_PX
    min_area = MIN_AREA * area_scale
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    row_lo, row_hi = int(ys.min()), int(ys.max())
    col_lo, col_hi = int(xs.min()), int(xs.max())
    crop = mask[row_lo:row_hi + 1, col_lo:col_hi + 1]
    props = skregionprops(sklabel(crop))
    if not props:
        return None
    p = max(props, key=lambda x: x.area)
    minor = p.axis_minor_length
    if minor < 1 or p.area < min_area:
        log.count("dropped_area")
        log.event("drop", gate="area", area=float(round(p.area, 1)),
                  min_area=round(min_area, 1))
        return None
    if not skip_aspect and p.axis_major_length / minor < cfg.min_aspect:
        log.count("dropped_aspect")
        log.event("drop", gate="aspect",
                  aspect=round(p.axis_major_length / minor, 2))
        return None
    pixel_density = p.area / max(p.axis_major_length, 1)
    fill_frac = p.area / (minor * max(p.axis_major_length, 1))
    is_fat = minor > 50 * math.sqrt(area_scale) and minor > 1.6 * pixel_density
    if is_fat:
        if cfg.fat_blob_gate:
            log.count("dropped_fat_blob")
            log.event("drop", gate="fat_blob", minor=round(minor, 1),
                      fill_frac=round(fill_frac, 3))
            return None
        log.count("kept_would_fail_fat_blob")
        log.event("kept_would_fail_fat_blob", minor=round(minor, 1),
                  fill_frac=round(fill_frac, 3))
    _, vecs = np.linalg.eigh(p.inertia_tensor)
    return {
        "centroid": np.array([p.centroid[0] + row_lo, p.centroid[1] + col_lo]),
        "u":        vecs[:, 0],
        "minor":    minor,
        "major":    p.axis_major_length,
        "area":     p.area,
        "coords":   np.column_stack((ys, xs)),
    }


def _segs_from_polygons(polygons: list, h: int, w: int) -> list:
    """Rasterise each fitted polygon into its OWN binary mask. One uint8 mask per
    polygon (255=trail). The union of these equals the final mask, but kept as
    separate entries so Star Bridge repair morphs each trail/arm independently
    (repair_frame's polygon_segs path) and MaskViewR can outline each separately.
    Crossing-split arms are distinct polygons here, so they stay distinct."""
    segs = []
    for corners in polygons:
        seg = np.zeros((h, w), dtype=np.uint8)
        pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(seg, [pts], 255)
        segs.append(seg)
    return segs


def _fit_groups(det_list: list, groups: list, h: int, w: int):
    """Fit one polygon per group and rasterise them into a mask. Returns
    (final_mask, polygons). Curved fit is used when a group is long and bends;
    otherwise a single straight strip. Shared by stage 2 and the seam bridge so
    both produce identical polygons."""
    final = np.zeros((h, w), dtype=np.uint8)
    polygons = []
    for grp in groups:
        all_coords = np.vstack([det_list[i]["coords"] for i in grp])
        x_span = int(all_coords[:, 1].max() - all_coords[:, 1].min())
        if (x_span >= _CURVED_MIN_XSPAN
                and _group_angle_spread(grp, det_list) >= _CURVED_MIN_ANGLE_SPREAD):
            corner_sets = fit_curved_group(grp, det_list)
        else:
            corners, _, _ = fit_polygon(grp, det_list)
            corner_sets = [corners]
        for corners in corner_sets:
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(final, [pts], 255)
            polygons.append(corners)
    return final, polygons


def stage_fit_polygons(state: PipelineState, cfg: StageConfig,
                       log: StageLog) -> PipelineState:
    """Stage 2: filter raw SAHI detections, group fragments that belong to the
    same trail, and fit a polygon per group (strip-based fit for curved trails).
    No gap bridging here -- that is a later, deliberate stage."""
    preds = state.raw_detections
    h, w = state.image.shape[:2]
    final = np.zeros((h, w), dtype=np.uint8)
    if not preds:
        state.final_mask = final
        return state

    # Filter: crossing split -> parallel split -> gated props.
    # When the crossing splitter confirms a split (returns >1 piece), those
    # pieces bypass the aspect gate -- the splitter already validated them as
    # real trail arms. Short stubby tips would otherwise be killed.
    det_list = []
    t_premask = t_split = t_par = t_props = 0.0
    n_split_fired = 0
    for pred in preds:
        _t = time.perf_counter()
        m = _pred_to_mask(pred, h, w)
        t_premask += time.perf_counter() - _t
        if m is None:
            continue
        _t = time.perf_counter()
        crossing_pieces = split_crossing(m)
        t_split += time.perf_counter() - _t
        is_confirmed_crossing = len(crossing_pieces) > 1
        if is_confirmed_crossing:
            n_split_fired += 1
        for cm in crossing_pieces:
            _t = time.perf_counter()
            parallel_pieces = _try_split_parallel(cm)
            t_par += time.perf_counter() - _t
            for em in parallel_pieces:
                _t = time.perf_counter()
                props = _props_with_log(em, cfg, log,
                                        skip_aspect=is_confirmed_crossing)
                t_props += time.perf_counter() - _t
                if props is not None:
                    props["conf"] = float(
                        getattr(getattr(pred, "score", None), "value", 0.0) or 0.0)
                    det_list.append(props)
    log.count("detections_passed", len(det_list))
    log.count("crossings_split", n_split_fired)
    # Per-step timing so the log shows what fired and how long (no probes needed).
    log.event("substep_timing", pred_to_mask_s=round(t_premask, 3),
              split_crossing_s=round(t_split, 3),
              parallel_split_s=round(t_par, 3), props_s=round(t_props, 3))
    if not det_list:
        state.det_list = det_list
        state.final_mask = final
        return state

    # Group collinear, touching fragments into trails.
    _t = time.perf_counter()
    groups = group_detections(det_list)
    t_group = time.perf_counter() - _t
    log.count("groups", len(groups))

    # Fit one polygon per group (curved fit when the group is long and bends).
    _t = time.perf_counter()
    final, polygons = _fit_groups(det_list, groups, h, w)
    t_fit = time.perf_counter() - _t
    log.event("fit_timing", group_s=round(t_group, 3), poly_fit_s=round(t_fit, 3))

    log.count("polygons", len(polygons))
    log.count("mask_components",
              max(0, cv2.connectedComponents(
                  (final > 0).astype(np.uint8))[0] - 1))
    state.det_list = det_list
    state.groups = groups
    state.polygons = polygons
    state.polygon_segs = _segs_from_polygons(polygons, h, w)
    state.final_mask = final
    return state


def stage_fallback_polys(state: PipelineState, cfg: StageConfig,
                         log: StageLog) -> PipelineState:
    """Stage 3: single-detection polygon for any group under the coverage
    threshold."""
    log.event("stub", note="fallback_polys not yet implemented")
    return state  # TODO: implement


def stage_link_gaps(state: PipelineState, cfg: StageConfig,
                    log: StageLog) -> PipelineState:
    """Stage 4: join collinear polygon fragments separated by a small gap."""
    log.event("stub", note="link_gaps not yet implemented")
    return state  # TODO: implement


# --- Stage 5 helpers: seam-gap bridge by re-inference ----------------------- #
#
# The shipped pipeline bridged seam gaps GEOMETRICALLY: if two fragments passed
# six similarity gates it merged them with no proof a trail spans the gap. That
# is what over-joins two genuinely separate trails (143A8740). Here the six
# gates are only a cheap pre-filter to decide WHERE to look; the bridge is made
# only when a fresh inference tile centered in the gap produces a detection that
# actually connects the two fragments. The model, not geometry, decides.

_BRIDGE_MAX_GAP   = 550    # px: max tip-to-tip distance to attempt a bridge
_BRIDGE_MAX_ANGLE = 12.0   # degrees: PCA on fragment blobs is noisy
_BRIDGE_MAX_WIDTH = 3.0     # ratio: same as grouper width gate
_BRIDGE_TIP_ANGLE = 20.0   # degrees: tip-to-tip vector must align with trail dir
_BRIDGE_CLIP_TOL  = 3       # px: facing edge must be within this of the seam line


def _group_tips(grp, det_list):
    """Return (tip_min_rc, tip_max_rc, u_avg) for a group as row-col arrays."""
    all_dets = [det_list[i] for i in grp]
    all_coords = np.vstack([d["coords"] for d in all_dets])
    u_sum = np.zeros(2)
    for d in all_dets:
        u = d["u"] if u_sum.dot(d["u"]) >= 0 else -d["u"]
        u_sum += u * d["area"]
    u_avg = u_sum / np.linalg.norm(u_sum)
    centroid = all_coords.mean(axis=0)
    t_c = float(centroid @ u_avg)
    t = all_coords @ u_avg
    tip_min = centroid + (float(t.min()) - t_c) * u_avg
    tip_max = centroid + (float(t.max()) - t_c) * u_avg
    return tip_min, tip_max, u_avg


def _find_gap_bridge_tiles(groups, det_list, h, w, tile_size):
    """Return [(gi, gj, tile_x, tile_y), ...] for group pairs that show the
    seam-gap signature (similar angle/width, co-linear, tips within range, a
    tile seam line inside the gap). One tile per qualifying pair, centered on
    the gap. This only chooses WHERE to re-infer; it does not bridge."""
    extra = []
    seen = set()
    n = len(groups)
    _stride = int(tile_size * 0.8)

    def _tile_bounds_1d(size):
        bounds = set()
        k = 0
        while True:
            left = k * _stride
            if left > size - tile_size:
                break
            bounds.add(left)
            bounds.add(min(left + tile_size, size))
            k += 1
        bounds.add(size - tile_size)
        bounds.add(size)
        return bounds

    x_bounds = _tile_bounds_1d(w)
    y_bounds = _tile_bounds_1d(h)

    for gi in range(n):
        for gj in range(gi + 1, n):
            di_list = [det_list[k] for k in groups[gi]]
            dj_list = [det_list[k] for k in groups[gj]]

            u_i = di_list[0]["u"]
            u_j = dj_list[0]["u"]
            cos_sim = min(abs(float(np.dot(u_i, u_j))), 1.0)
            adiff = min(np.degrees(np.arccos(cos_sim)),
                        180.0 - np.degrees(np.arccos(cos_sim)))
            if adiff > _BRIDGE_MAX_ANGLE:
                continue

            minor_i = float(np.median([d["minor"] for d in di_list]))
            minor_j = float(np.median([d["minor"] for d in dj_list]))
            if max(minor_i, minor_j) / max(min(minor_i, minor_j), 1) > _BRIDGE_MAX_WIDTH:
                continue

            all_i = np.vstack([det_list[k]["coords"] for k in groups[gi]])
            all_j = np.vstack([det_list[k]["coords"] for k in groups[gj]])
            ci = all_i.mean(axis=0)
            cj = all_j.mean(axis=0)
            diff = cj - ci
            along = float(np.dot(diff, u_i))
            perp = float(np.sqrt(max(float(np.dot(diff, diff)) - along ** 2, 0.0)))
            if perp > 0.9 * max(minor_i, minor_j):
                continue

            tip_i_min, tip_i_max, _ = _group_tips(groups[gi], det_list)
            tip_j_min, tip_j_max, _ = _group_tips(groups[gj], det_list)
            combos = [
                (tip_i_min, tip_j_min), (tip_i_min, tip_j_max),
                (tip_i_max, tip_j_min), (tip_i_max, tip_j_max),
            ]
            best_dist, best_a, best_b = float("inf"), None, None
            for ta, tb in combos:
                d = float(np.linalg.norm(ta - tb))
                if d < best_dist:
                    best_dist, best_a, best_b = d, ta, tb
            if best_dist > _BRIDGE_MAX_GAP:
                continue

            min_c_i = float(all_i[:, 1].min()); max_c_i = float(all_i[:, 1].max())
            min_c_j = float(all_j[:, 1].min()); max_c_j = float(all_j[:, 1].max())
            min_r_i = float(all_i[:, 0].min()); max_r_i = float(all_i[:, 0].max())
            min_r_j = float(all_j[:, 0].min()); max_r_j = float(all_j[:, 0].max())
            x_lo = min(max_c_i, max_c_j); x_hi = max(min_c_i, min_c_j)
            y_lo = min(max_r_i, max_r_j); y_hi = max(min_r_i, min_r_j)
            x_seam = x_lo < x_hi and any(
                x_lo < b < x_hi
                and (abs(x_lo - b) <= _BRIDGE_CLIP_TOL or abs(x_hi - b) <= _BRIDGE_CLIP_TOL)
                for b in x_bounds)
            y_seam = y_lo < y_hi and any(
                y_lo < b < y_hi
                and (abs(y_lo - b) <= _BRIDGE_CLIP_TOL or abs(y_hi - b) <= _BRIDGE_CLIP_TOL)
                for b in y_bounds)
            if not (x_seam or y_seam):
                continue

            tip_vec = best_b - best_a
            tip_len = float(np.linalg.norm(tip_vec))
            if tip_len < 1.0:
                continue
            tip_unit = tip_vec / tip_len
            u_j_oriented = u_j if float(np.dot(u_i, u_j)) >= 0 else -u_j
            u_avg = u_i + u_j_oriented
            u_avg_norm = float(np.linalg.norm(u_avg))
            if u_avg_norm < 1e-6:
                continue
            u_avg = u_avg / u_avg_norm
            tip_cos = min(abs(float(np.dot(tip_unit, u_avg))), 1.0)
            if float(np.degrees(np.arccos(tip_cos))) > _BRIDGE_TIP_ANGLE:
                continue

            mid_rc = (best_a + best_b) / 2.0
            mid_y = int(round(float(mid_rc[0]))); mid_x = int(round(float(mid_rc[1])))
            new_tx = max(0, min(w - tile_size, mid_x - tile_size // 2))
            new_ty = max(0, min(h - tile_size, mid_y - tile_size // 2))
            pair_key = (gi, gj)
            if pair_key not in seen:
                seen.add(pair_key)
                extra.append((gi, gj, new_tx, new_ty))
    return extra


def _targeted_tile_dets(model, img_rgb, tile_x, tile_y, tile_size, h, w,
                        cfg, log, rot90=False):
    """Run one direct YOLO tile (optionally 90-degree rotated to catch
    orientation-sensitive misses) and return detection props in global coords.
    Each segment is kept whole (no crossing/parallel split) so a gap-spanning
    detection stays intact and can connect the two fragments."""
    ty1, tx1 = tile_y, tile_x
    ty2, tx2 = min(h, tile_y + tile_size), min(w, tile_x + tile_size)
    crop_h, crop_w = ty2 - ty1, tx2 - tx1
    crop = img_rgb[ty1:ty2, tx1:tx2, ::-1].copy()   # RGB -> BGR for yolo
    if rot90:
        crop = np.ascontiguousarray(np.rot90(crop, 1))
        pad_h, pad_w = crop_w, crop_h
    else:
        pad_h, pad_w = crop_h, crop_w
    if pad_h < tile_size or pad_w < tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:pad_h, :pad_w] = crop
        crop = padded

    yolo = model.model
    conf = getattr(model, "confidence_threshold", 0.25)
    results = yolo.predict(source=crop, conf=conf, verbose=False, imgsz=tile_size)

    new_dets = []
    for r in results:
        if r.masks is None:
            continue
        confs = r.boxes.conf.tolist() if r.boxes is not None else []
        for seg_idx, seg_xy in enumerate(r.masks.xy):
            if len(seg_xy) < 3:
                continue
            seg_conf = float(confs[seg_idx]) if seg_idx < len(confs) else 0.0
            local = np.zeros((tile_size, tile_size), dtype=np.uint8)
            cv2.fillPoly(local, [np.array(seg_xy, dtype=np.int32)], 255)
            if rot90:
                local = np.ascontiguousarray(np.rot90(local[:pad_h, :pad_w], 3))
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[ty1:ty2, tx1:tx2] = local[:crop_h, :crop_w]
            props = _props_with_log(global_mask, cfg, log)
            if props is not None:
                props["conf"] = seg_conf
                new_dets.append(props)
    return new_dets


def stage_seam_second_pass(state: PipelineState, cfg: StageConfig,
                           log: StageLog, model: Any) -> PipelineState:
    """Stage 5: bridge true seam gaps by re-inference. The six geometric gates
    pick candidate fragment pairs; a fresh tile is run in each gap; the new
    detections are added and the frame is re-grouped. Two fragments merge only
    if the model fires a detection that physically connects them across the gap
    (B10), and stay separate when it does not (143A8740)."""
    det_list = state.det_list
    groups = state.groups
    if model is None or not groups:
        return state
    h, w = state.image.shape[:2]
    img_rgb = state.image

    pairs = _find_gap_bridge_tiles(groups, det_list, h, w, cfg.tile_size)
    log.count("candidate_pairs", len(pairs))
    if not pairs:
        return state

    new_dets = []
    seen_tiles = set()
    for gi, gj, tx, ty in pairs:
        if (tx, ty) in seen_tiles:
            continue
        seen_tiles.add((tx, ty))
        new_dets.extend(_targeted_tile_dets(model, img_rgb, tx, ty,
                                            cfg.tile_size, h, w, cfg, log))
        new_dets.extend(_targeted_tile_dets(model, img_rgb, tx, ty,
                                            cfg.tile_size, h, w, cfg, log, rot90=True))
    log.count("tiles_run", len(seen_tiles))
    log.count("new_dets", len(new_dets))
    if not new_dets:
        return state

    groups_before = len(groups)
    det_list = list(det_list) + new_dets
    groups = group_detections(det_list)
    log.event("regroup", groups_before=groups_before, groups_after=len(groups))

    final, polygons = _fit_groups(det_list, groups, h, w)
    state.det_list = det_list
    state.groups = groups
    state.polygons = polygons
    state.polygon_segs = _segs_from_polygons(polygons, h, w)
    state.final_mask = final
    return state


# --- FP suppressor constants and helpers ---------------------------------- #

_SFP_PIXEL_DIFF_THRESH = 8.0   # mean abs pixel diff below this = "same content"
_SFP_MIN_MATCHES       = 1     # min neighbor matches to trigger suppression
_SFP_EDGE_PX           = 20    # frame edge veto zone (px)
_SFP_BRIGHT_RATIO      = 2.5   # 90th-pct inside / median surround; above = real trail


def _tile_coord(cx, cy, stride):
    """Convert pixel centroid to tile coordinate like 'B10'."""
    col = int(cx) // stride
    row = int(cy) // stride
    return f"{chr(ord('A') + row)}{col + 1}"


def _is_bright_trail(comp_mask_crop, img_crop):
    """Return (is_bright, ratio): True if detection pixels are significantly
    brighter than the local sky background. Catches airplane nav lights,
    strobes, and bright satellite streaks regardless of color."""
    ys, xs = np.where(comp_mask_crop)
    if len(ys) == 0:
        return False, 0.0
    inside = img_crop[ys, xs].astype(np.float32)
    inside_bright = float(np.percentile(np.max(inside, axis=1), 90))
    m_u8 = comp_mask_crop.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(m_u8, kernel) > 0) & ~comp_mask_crop
    ry, rx = np.where(ring)
    if len(ry) == 0:
        return False, 0.0
    surrounding = img_crop[ry, rx].astype(np.float32)
    surround_med = float(np.median(np.max(surrounding, axis=1)))
    if surround_med < 1:
        return False, 0.0
    ratio = round(inside_bright / surround_med, 2)
    return ratio >= _SFP_BRIGHT_RATIO, ratio


def stage_suppress_fp(state: PipelineState, cfg: StageConfig,
                      log: StageLog, neighbour_frames: Any) -> PipelineState:
    """Suppress static false positives by comparing pixel content in each
    detection region against the same region in neighbour frames.

    A static object (telescope mount, tree branch, building edge) has nearly
    identical pixel content in every frame. A real trail is bright in the
    current frame but absent in most neighbours. For each detection component,
    we compute the mean absolute pixel difference inside the component mask
    between the current frame and each neighbour. If the difference is below
    a threshold for enough neighbours, the component is static and is removed.

    Three vetoes protect real trails from false suppression:
      1. Frame edge: bbox within 20px of any edge -> keep (trails exit frame,
         static objects don't).
      2. Bright trail: detection pixels significantly brighter than surrounding
         sky -> keep (nav light, strobe, bright streak).
      3. Low match count: fewer than _SFP_MIN_MATCHES neighbours match -> keep.

    Logging includes pixel coordinates, tile coordinate (B10 style), match
    count, similarity scores, and plain-English reason. Only written when a
    log_path is provided (dev runs), not in shipped app."""
    if neighbour_frames is None or len(neighbour_frames) == 0:
        log.event("skip", note="no neighbour frames provided")
        return state

    mask = state.final_mask
    if mask is None or mask.max() == 0:
        return state

    image = state.image  # RGB uint8
    h, w = mask.shape

    # Load neighbour images as RGB arrays.
    nb_images = []
    for nb in neighbour_frames:
        if isinstance(nb, np.ndarray):
            nb_img = nb
            if nb_img.ndim == 3 and nb_img.shape[2] == 3:
                pass  # already HxWx3
            else:
                loaded = _load_as_rgb(nb)
                if loaded is None:
                    continue
                nb_img = loaded[0]
        else:
            loaded = _load_as_rgb(nb)
            if loaded is None:
                continue
            nb_img = loaded[0]
        if nb_img.shape[:2] != (h, w):
            continue
        nb_images.append(nb_img)

    if not nb_images:
        log.event("skip", note="no valid neighbour images loaded")
        return state

    log.count("neighbours_loaded", len(nb_images))

    # Connected components of final mask.
    mask_u8 = (mask > 0).astype(np.uint8)
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)

    to_suppress = np.zeros(mask.shape, dtype=bool)
    stride = int(cfg.tile_size * (1 - cfg.overlap))

    for cc_id in range(1, n_cc):
        x1 = int(cc_stats[cc_id, cv2.CC_STAT_LEFT])
        y1 = int(cc_stats[cc_id, cv2.CC_STAT_TOP])
        cw = int(cc_stats[cc_id, cv2.CC_STAT_WIDTH])
        ch = int(cc_stats[cc_id, cv2.CC_STAT_HEIGHT])
        x2, y2 = x1 + cw - 1, y1 + ch - 1
        area = int(cc_stats[cc_id, cv2.CC_STAT_AREA])
        comp_crop = (cc_labels[y1:y2 + 1, x1:x2 + 1] == cc_id)

        # Centroid from bbox-local crop.
        _loc_ys, _loc_xs = np.where(comp_crop)
        cx = int(_loc_xs.mean()) + x1
        cy = int(_loc_ys.mean()) + y1
        tile = _tile_coord(cx, cy, stride)

        # --- Veto 1: frame edge ------------------------------------------ #
        if (y1 <= _SFP_EDGE_PX - 1 or y2 >= h - _SFP_EDGE_PX
                or x1 <= _SFP_EDGE_PX - 1 or x2 >= w - _SFP_EDGE_PX):
            log.event("fp_veto_frame_edge", cx=cx, cy=cy, tile=tile,
                      bbox=[x1, y1, x2, y2], area=area,
                      detail=(f"bbox touches frame edge within {_SFP_EDGE_PX}px "
                              f"-- trails exit frame, static objects don't"))
            log.count("kept_frame_edge")
            continue

        # --- Compare pixel content against each neighbour ---------------- #
        current_patch = image[y1:y2 + 1, x1:x2 + 1].astype(np.float32)

        matched = []
        for ni, nb_img in enumerate(nb_images):
            nb_patch = nb_img[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
            diff = np.abs(current_patch - nb_patch)
            # Mean absolute difference inside the component mask only.
            mean_diff = float(diff[comp_crop].mean())
            if mean_diff < _SFP_PIXEL_DIFF_THRESH:
                matched.append({"neighbor_idx": ni,
                                "mean_diff": round(mean_diff, 2)})

        if len(matched) < _SFP_MIN_MATCHES:
            log.count("kept_no_match")
            continue

        # --- Veto 2: bright trail ---------------------------------------- #
        img_crop = image[y1:y2 + 1, x1:x2 + 1]
        is_bright, bright_ratio = _is_bright_trail(comp_crop, img_crop)
        if is_bright:
            match_str = ", ".join(
                f"nb{m['neighbor_idx']} diff={m['mean_diff']}"
                for m in matched)
            log.event("fp_veto_bright_trail", cx=cx, cy=cy, tile=tile,
                      bbox=[x1, y1, x2, y2], area=area,
                      bright_ratio=bright_ratio,
                      match_count=len(matched), matched=matched,
                      detail=(f"Kept: {bright_ratio}x brighter than surroundings "
                              f"despite {len(matched)} neighbor match(es) "
                              f"({match_str}). Real trail."))
            log.count("kept_bright_trail")
            continue

        # --- Suppress this component ------------------------------------- #
        to_suppress[y1:y2 + 1, x1:x2 + 1] |= comp_crop
        match_str = ", ".join(
            f"nb{m['neighbor_idx']} diff={m['mean_diff']}"
            for m in matched)
        log.event("fp_suppressed", cx=cx, cy=cy, tile=tile,
                  bbox=[x1, y1, x2, y2], area=area,
                  match_count=len(matched), matched=matched,
                  detail=(f"Static FP at tile {tile}: content nearly identical "
                          f"in {len(matched)} neighbour(s) ({match_str}). "
                          f"cx={cx} cy={cy} area={area}px."))
        log.count("suppressed")

    # Apply all suppressions.
    if to_suppress.any():
        state.final_mask[to_suppress] = 0
        n_killed = max(0,
                       cv2.connectedComponents(to_suppress.astype(np.uint8))[0] - 1)
        log.count("components_suppressed", n_killed)

    return state


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def detect_frame(model: Any,
                 image: np.ndarray,
                 foreground_mask: Optional[np.ndarray] = None,
                 neighbour_frames: Any = None,
                 frame_name: str = "frame",
                 cfg: Optional[StageConfig] = None,
                 log_path: Optional[str] = None) -> PipelineState:
    """Run the full detection pipeline on one frame and return the final state
    (state.final_mask is the trail mask). Each enabled stage runs in order and
    appends its record to the per-frame log. If log_path is given, the JSONL
    record is appended there."""
    cfg = cfg or StageConfig()
    flog = FrameLog(frame_name=frame_name)
    state = PipelineState(image=image, foreground_mask=foreground_mask)

    if cfg.tiled_inference:
        with flog.stage("tiled_inference") as s:
            state = stage_tiled_inference(state, cfg, s, model)
    if cfg.fit_polygons:
        with flog.stage("fit_polygons") as s:
            state = stage_fit_polygons(state, cfg, s)
    if cfg.fallback_polys:
        with flog.stage("fallback_polys") as s:
            state = stage_fallback_polys(state, cfg, s)
    if cfg.link_gaps:
        with flog.stage("link_gaps") as s:
            state = stage_link_gaps(state, cfg, s)
    if cfg.seam_second_pass:
        with flog.stage("seam_second_pass") as s:
            state = stage_seam_second_pass(state, cfg, s, model)
    if cfg.suppress_fp:
        with flog.stage("suppress_fp") as s:
            state = stage_suppress_fp(state, cfg, s, neighbour_frames)

    # Expose per-stage timing so callers (e.g. astro_clean_v5) can report where
    # detection time goes without re-parsing the JSONL.
    state.stage_seconds = {st["stage"]: st.get("seconds", 0.0)
                           for st in flog.stages}

    if log_path:
        with open(log_path, "a") as f:
            f.write(flog.to_jsonl() + "\n")

    return state
