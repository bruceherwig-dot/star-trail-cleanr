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
  1b. prune_phantoms   Remove thin, dotted phantom detections that sit on empty
                       sky. OFF by default in StageConfig, but the live worker
                       turns it on. Runs between tiled_inference and fit_polygons.
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

The live app (astro_clean_v5.py) now uses this module as its detector, calling
detect_frame() per frame. The older detect_trails.detect_frame_polygon path is
retained only for the known_problems.jsonl regression tests and is no longer on
the shipping path.
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
from .crossing_splitter import split_crossing, has_crossing_evidence
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
    # Stage 1b: remove thin, dotted, "nothing under it" phantom detections the
    # model emits over empty sky. OFF by default so test suites see legacy
    # behaviour; the STC worker turns it on.
    prune_phantoms: bool = False
    # Dev-only: log each removed phantom's location to the run log for
    # hard-negative training-data mining. OFF for shipped users (it only adds
    # noise to their logs that never reaches us); the STC worker turns it on
    # only when running from source (Bruce's machine), not in the frozen bundle.
    log_phantom_negatives: bool = False

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
    # One full-frame mask per raw detection, in the same order, built ONCE and
    # handed between stages. Turning a detection into a mask is the single most
    # expensive thing either of the two big stages does (the AI stores an
    # outline, so asking for the mask redraws it across the whole frame), and
    # phantom pruning and polygon fitting were each building the same 53 masks
    # from the same detections, one after the other. An entry may be None,
    # meaning "not cached, work it out" -- a consumer must fall back, never
    # assume. Empty when no stage has populated it.
    pred_masks: list[Any] = field(default_factory=list)
    det_list: list[Any] = field(default_factory=list)         # filtered detection props
    groups: list[Any] = field(default_factory=list)           # grouped detection indices
    polygons: list[Any] = field(default_factory=list)         # fitted trail polygon corner sets
    polygon_segs: list[Any] = field(default_factory=list)     # per-polygon binary masks (1 per polygon, for repair + MaskViewR)
    final_mask: Optional[np.ndarray] = None                   # output mask
    stage_seconds: dict = field(default_factory=dict)         # {stage_name: seconds} for timing reporting
    stage_log: list = field(default_factory=list)             # per-stage records (seconds + counts + events) for the run log


# --- Stage 1 helpers (ported verbatim from the proven detect_trails.py path so
#     detection output matches the shipped app exactly) ---------------------- #

class _PredMaskWrap:
    """Tiny holder that exposes a boolean mask as `.bool_mask`, matching the
    attribute a real SAHI prediction's `.mask` has. Lets our home-grown
    detections be read by the same code paths that read SAHI output."""
    __slots__ = ("bool_mask",)
    def __init__(self, bm):
        self.bool_mask = bm


class _PredScoreWrap:
    """Tiny holder that exposes a confidence number as `.value`, matching the
    attribute a real SAHI prediction's `.score` has."""
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
        # Floor at 0: when a whole dimension is smaller than the tile, size -
        # tile_size goes negative, which corrupts the crop bookkeeping and
        # crashes the mask paste. A sub-tile image becomes one padded tile.
        # No-op for any dimension >= tile_size.
        starts.append(max(0, size - tile_size))
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
    """Tiled YOLO inference that skips tiles fully covered by the foreground
    (sky) mask and blacks out the foreground inside any partially-covered tile
    before the model sees it.

    Inputs:
      model     -- the SAHI detection-model wrapper (its `.model` is the raw
                   ultralytics YOLO).
      img_rgb   -- the full frame as RGB uint8.
      tile_size -- side length of each square inference tile (px).
      overlap   -- fractional tile overlap (e.g. 0.2 = 20%); sets the stride.
      fg_mask   -- foreground/sky mask; non-zero marks ground/foreground to
                   ignore. Resized to the frame if it does not already match.

    Returns (predictions, n_skipped):
      predictions -- list of _SyntheticPred objects (one per kept mask segment),
                     each carrying a full-frame boolean mask and a confidence.
      n_skipped   -- count of tiles skipped because they were 100% foreground.

    Why it exists: this is the per-frame detection front door used by Stage 1
    when a foreground mask is present. Skipping all-foreground tiles saves
    inference time, and zeroing the foreground in mixed tiles stops YOLO from
    detecting "trails" on the ground (buildings, trees, telescope mounts).
    Ported to match the shipped detect_trails path exactly."""
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
            # ::-1 on the channel axis flips RGB -> BGR, the order ultralytics
            # YOLO expects; ascontiguousarray makes the reversed view a real
            # buffer the model can read.
            crop_bgr = np.ascontiguousarray(img_rgb[ty:ty2, tx:tx2, ::-1])
            # Black out the foreground BEFORE the model sees it -- the whole
            # point of the mask. Tile-skip only handles 100%-foreground tiles;
            # a partial-sky tile would otherwise let YOLO detect on the ground.
            fg_crop = fm[ty:ty2, tx:tx2]
            if fg_crop.shape == crop_bgr.shape[:2]:
                crop_bgr[fg_crop > 0] = 0
            if crop_h < tile_size or crop_w < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:crop_h, :crop_w] = crop_bgr
                crop_bgr = padded
            kept.append((tx, ty, crop_w, crop_h, crop_bgr))

    if not kept:
        return [], n_skipped

    # One batched YOLO call over every kept crop (the 5th tuple element is the
    # padded BGR tile). Order of `results` matches the order of `kept`.
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
        # Each detected segment is a polygon in TILE-local coords. Rasterise it
        # into a tile-sized mask, then paste it into a full-frame mask at this
        # tile's (tx, ty) origin so every detection lives in one shared space.
        for seg_idx, seg_xy in enumerate(r.masks.xy):
            if len(seg_xy) < 3:   # need >=3 points to form a polygon
                continue
            seg_conf = float(confs[seg_idx]) if seg_idx < len(confs) else 0.0
            local_u8 = np.zeros((tile_size, tile_size), dtype=np.uint8)
            cv2.fillPoly(local_u8, [np.array(seg_xy, dtype=np.int32)], 1)
            global_mask = np.zeros((h, w), dtype=bool)
            # Clamp the paste to the rows/cols actually available in the frame,
            # so a tile that runs past the edge can never trigger a broadcast
            # crash. No-op for in-bounds tiles (gh==crop_h, gw==crop_w).
            gh = min(crop_h, h - ty)
            gw = min(crop_w, w - tx)
            global_mask[ty:ty + gh, tx:tx + gw] = (
                local_u8[:gh, :gw].astype(bool))
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
                    skip_aspect: bool = False, frame_px=None):
    """Region properties for one detection mask, or None if it fails a gate.
    Same gates as the shipped elongation filter, except the fat-blob gate is
    toggleable and every drop (and every fat-blob the gate would have killed)
    is logged with its reason.

    skip_aspect: when True, bypass the aspect-ratio (elongation) gate. Used for
    crossing-splitter output where the splitter already validated the pieces as
    real trail arms -- re-filtering by shape would kill short stubby tips.
    frame_px: total pixels of the FULL frame for area normalisation; pass when
    `mask` is a crop so thresholds match full-frame. Defaults to mask size."""
    area_scale = (frame_px if frame_px else (mask.shape[0] * mask.shape[1])) / _REF_FRAME_PX
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
    # "Fat blob" test: a detection is suspiciously fat when it is BOTH wide in
    # absolute terms (minor axis > ~50px, scaled to the frame size) AND wider
    # than ~1.6x its own pixel density (i.e. it does not thin out like a streak).
    # pixel_density ~ area-per-unit-length; fill_frac ~ how solidly it fills its
    # bounding ellipse. Real thin trails fail both conditions.
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


# De-dup thresholds for redundant same-trail fragments (see _polys_and_segs_deduped).
_SEG_MERGE_CONTAINMENT = 0.90   # a polygon >=90% inside a bigger one is redundant
_SEG_MERGE_ANGLE       = 20.0   # deg: only treat as the same trail if angles agree


def _poly_angle(corners):
    """Orientation (degrees, 0-180) of a fitted polygon, from its longer edge."""
    c = np.asarray(corners, dtype=float)
    # Take the first two edges of the quad and keep the LONGER one -- that edge
    # runs along the trail, so its angle is the trail's orientation. mod 180
    # because a trail and its 180-degree flip are the same line.
    e01 = c[0] - c[1]
    e12 = c[1] - c[2]
    le = e12 if (e12[0] ** 2 + e12[1] ** 2) >= (e01[0] ** 2 + e01[1] ** 2) else e01
    return float(np.degrees(np.arctan2(le[1], le[0]))) % 180.0


def _angle_diff(a, b):
    """Smallest difference between two 0-180 orientations (handles the wrap)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _polys_and_segs_deduped(polygons: list, h: int, w: int):
    """Rasterise each fitted polygon into its OWN binary mask, then drop away
    redundant same-trail fragments so repair runs once per real trail AND the
    viewer shows one outline per trail.

    Returns (kept_polygons, kept_segs): index-aligned lists, one uint8 mask
    (255=trail) per kept polygon. Star Bridge repair morphs each segment
    independently (repair_frame's polygon_segs path), and MaskViewR outlines each
    kept polygon, so what you see equals what gets repaired.

    De-dup: a polygon that sits >=90% INSIDE a BIGGER polygon of similar angle is
    a redundant piece of the same trail (the detector produced an extra fragment
    on a trail another polygon already covers). It is NOT a crossing arm -- those
    differ in angle and only overlap at the junction, never 90% contained -- so
    crossings and genuinely separate trails are left untouched. Each redundant
    piece has its pixels folded into the bigger polygon's segment, then the
    redundant polygon AND its segment are dropped. ZERO coverage loss: the union
    of the kept segments still equals the union of all original polygons, so the
    final mask is unchanged; only the wasted repair pass and the duplicate
    outline go away. A bbox-overlap gate keeps the pairwise check cheap.
    """
    segs = []
    for corners in polygons:
        seg = np.zeros((h, w), dtype=np.uint8)
        pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(seg, [pts], 255)
        segs.append(seg)

    n = len(segs)
    if n < 2:
        return polygons, segs

    bmasks = [s > 0 for s in segs]
    areas = [int(b.sum()) for b in bmasks]
    angles = [_poly_angle(p) for p in polygons]
    # Per-polygon bbox (cheap axis reductions) to skip non-overlapping pairs.
    # len(bboxes) is the index of the polygon currently being processed (one
    # bbox is appended per iteration); an empty mask gets a None bbox.
    bboxes = []
    for b in bmasks:
        if areas[len(bboxes)] == 0:
            bboxes.append(None); continue
        rr = np.where(b.any(axis=1))[0]
        cc = np.where(b.any(axis=0))[0]
        bboxes.append((int(rr[0]), int(rr[-1]), int(cc[0]), int(cc[-1])))

    # Union-find: parent[x] points toward the representative of x's group.
    # _find walks to the representative, compressing the path as it goes so
    # later lookups are fast. Polygons that turn out to be the same trail get
    # unioned into one group below.
    parent = list(range(n))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # For every ordered pair (i, j) where j is the BIGGER polygon and the two
    # share an angle, measure how much of i sits inside j. If >=90% of i is
    # contained in j, i is a redundant fragment of j's trail -> union them.
    for i in range(n):
        if bboxes[i] is None:
            continue
        ri1, ri2, ci1, ci2 = bboxes[i]
        for j in range(n):
            if i == j or bboxes[j] is None or areas[j] <= areas[i]:
                continue
            if _angle_diff(angles[i], angles[j]) > _SEG_MERGE_ANGLE:
                continue
            # Intersect the two bounding boxes first; only count overlapping
            # pixels inside that shared rectangle (cheap), never the whole frame.
            rj1, rj2, cj1, cj2 = bboxes[j]
            r1 = max(ri1, rj1); r2 = min(ri2, rj2)
            c1 = max(ci1, cj1); c2 = min(ci2, cj2)
            if r1 > r2 or c1 > c2:
                continue  # bboxes do not overlap -> not contained
            inter = int((bmasks[i][r1:r2 + 1, c1:c2 + 1]
                         & bmasks[j][r1:r2 + 1, c1:c2 + 1]).sum())
            if inter / areas[i] >= _SEG_MERGE_CONTAINMENT:
                parent[_find(i)] = _find(j)
                break

    # Collect the final groups: members sharing a representative are one trail.
    comps = {}
    for i in range(n):
        comps.setdefault(_find(i), []).append(i)
    keep_flags = [True] * n
    for members in comps.values():
        if len(members) == 1:
            continue
        # Keep the largest polygon of the group; fold the rest into its mask.
        keep = max(members, key=lambda k: areas[k])
        for k in members:
            if k != keep:
                segs[keep] = np.maximum(segs[keep], segs[k])  # fold pixels in (coverage preserved)
                keep_flags[k] = False                          # drop the redundant fragment + its outline
    kept_polygons = [polygons[i] for i in range(n) if keep_flags[i]]
    kept_segs = [segs[i] for i in range(n) if keep_flags[i]]
    return kept_polygons, kept_segs


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


# --------------------------------------------------------------------------- #
# Stage 1b: phantom pruning
# --------------------------------------------------------------------------- #
# A "phantom" is a thin, dotted, usually perpendicular detection the model emits
# over EMPTY sky (a hair-thin spur off a trail tip, or a sparse dashed line in
# blank sky). Discriminator proven on real frames: it is a THIN part of the raw
# mask (survives subtracting the morphological opening, so it is not a solid
# trail body) AND has NO real light under it in the source (brightness barely
# above the local sky). Real trails and real crossings are solid + bright and
# sit on actual light, so they are never touched.
_PHANTOM_SIG = 12          # source must beat local sky by this to be a real streak
_PHANTOM_OPEN_R = 6        # parts thinner than ~2*R px are "thin" (spurs / lines)
_PHANTOM_MIN_INK = 12      # min phantom pixels in a bridged line
_PHANTOM_MIN_EXTENT = 40   # min bounding extent of a bridged line
_PHANTOM_KEEP_FRAC = 0.20  # drop a detection if < this fraction survives trimming


def _phantom_local_sky(gray: np.ndarray) -> np.ndarray:
    """Estimate the local sky level (the thin streak vanishes under a heavy
    median), at quarter resolution for speed, upsampled back."""
    sh, sw = max(1, gray.shape[0] // 4), max(1, gray.shape[1] // 4)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 31)
    return cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def stage_prune_phantoms(state: PipelineState, cfg: StageConfig,
                         log: StageLog) -> PipelineState:
    """Stage 1b: remove thin, dotted phantom detections that sit on empty sky.

    WHAT IT IS FOR: the AI sometimes traces a faint dotted line across empty sky
    where there is no trail at all. Left alone, repair would paint over that
    stretch of sky for no reason. This stage finds those and removes them before
    anything downstream sees them.

    IF YOU ARE HERE FOR SPEED, READ THIS FIRST. This used to be the largest
    stage in detection, bigger than the AI inference itself (44% of detect on a
    44MP frame against 42% for inference), and it was invisible for months
    because the run summary had no row for it. Two rounds of work in August 2026
    took it to roughly a quarter of that: 10.30s to 4.25s over four 44MP frames.
    Both wins came from the same place, and neither changed a single output
    pixel:
      * per-component work now slices to the component's own bounding box,
        twice -- the label loop and the trim loop below it
      * detection masks are built ONCE per frame and shared with polygon
        fitting through state.pred_masks, instead of each stage building its own
    Whatever you try next, MEASURE BEFORE AND AFTER AND KEEP THE OUTPUT
    BYTE-IDENTICAL. This stage decides which detections are dropped, so a subtle
    change silently alters what gets cleaned, and nobody would see it until a
    user's sky came back wrong.

    Builds the union of the raw SAHI masks, keeps only the THIN parts (union
    minus its morphological opening) that have NO real light under them in the
    source, bridges those dots into lines, and removes only the elongated lines.
    The removed pixels are subtracted from every raw detection; a detection that
    loses almost all of its pixels is dropped entirely. Runs after tiled
    inference and before polygon fitting, so phantoms never reach the output or
    repair. Solid/bright real trails and real crossings are untouched."""
    preds = state.raw_detections
    if not preds:
        return state
    h, w = state.image.shape[:2]
    gray = state.image.max(2).astype(np.uint8) if state.image.ndim == 3 else state.image.astype(np.uint8)

    pred_masks = [_pred_to_mask(p, h, w) for p in preds]
    union = np.zeros((h, w), np.uint8)
    for m in pred_masks:
        if m is not None:
            union[m > 0] = 1
    if union.sum() == 0:
        state.pred_masks = pred_masks     # built already; the next stage needs them
        return state

    sky = _phantom_local_sky(gray)
    real = (gray.astype(np.int16) - sky.astype(np.int16)) > _PHANTOM_SIG
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * _PHANTOM_OPEN_R + 1, 2 * _PHANTOM_OPEN_R + 1))
    core = cv2.morphologyEx(union, cv2.MORPH_OPEN, se)
    thin = (union > 0) & (core == 0)
    ph = (thin & (~real)).astype(np.uint8)
    # Only look at sky: never test or prune inside the foreground (sky-mask)
    # region, which also skips any tile that is 100% foreground.
    if state.foreground_mask is not None:
        fm = state.foreground_mask
        if fm.shape != (h, w):
            fm = cv2.resize(fm, (w, h), interpolation=cv2.INTER_NEAREST)
        ph[fm > 0] = 0
    if ph.sum() == 0:
        state.pred_masks = pred_masks     # built already; the next stage needs them
        return state

    bridged = cv2.dilate(ph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bridged)
    kill = np.zeros((h, w), bool)
    n_lines = 0
    phantom_records = []   # one per removed phantom line -> hard-negative training data
    for i in range(1, n):
        extent = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        # CHEAPEST TEST FIRST. A component that is too short is dropped without
        # touching a single pixel; only the survivors get examined.
        if extent < _PHANTOM_MIN_EXTENT:
            continue
        # WORK INSIDE THE COMPONENT'S OWN BOX, not across the whole frame.
        # This used to be `comp = (lab == i) & (ph > 0)`: two full-frame
        # comparisons per component. On a 44MP frame with 183 components that is
        # roughly 16 BILLION element operations to inspect blobs a few hundred
        # pixels across, and it was 57% of this stage -- itself the largest
        # stage in detection. The bounding box comes free from
        # connectedComponentsWithStats and the line above already reads it.
        # Same arithmetic, same result, a fraction of the memory traffic.
        x0, y0 = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        sub = (lab[y0:y0 + bh, x0:x0 + bw] == i) & (ph[y0:y0 + bh, x0:x0 + bw] > 0)
        ink = int(sub.sum())
        if ink < _PHANTOM_MIN_INK:
            continue
        kill[y0:y0 + bh, x0:x0 + bw] |= sub
        n_lines += 1
        if cfg.log_phantom_negatives:   # dev-only hard-negative mining
            cys, cxs = np.where(sub)
            cys = cys + y0
            cxs = cxs + x0
            phantom_records.append({
                "cx": int(cxs.mean()), "cy": int(cys.mean()),
                "bbox": [int(cxs.min()), int(cys.min()), int(cxs.max()), int(cys.max())],
                "area": ink,
                "note": "thin FP over empty sky; nothing to see here (hard negative)",
            })
    if not kill.any():
        state.pred_masks = pred_masks     # built already; the next stage needs them
        return state

    new_preds = []
    new_masks = []
    n_dropped = n_trimmed = 0
    notkill = ~kill
    for pred, m in zip(preds, pred_masks):
        if m is None:
            new_preds.append(pred)
            new_masks.append(None)
            continue
        # WORK INSIDE THE DETECTION'S OWN BOX. This loop used to sweep the WHOLE
        # photograph three times per detection: (m > 0).sum(), (m > 0) & notkill,
        # and trimmed.sum(). With 53 detections on a 44MP frame that is roughly
        # 7 billion element operations to weigh blobs a few hundred pixels
        # across, and it measured as a third of this stage. Every lit pixel is
        # inside the box by definition, so the counts are identical.
        #
        # THIS IS THE THIRD TIME THIS PATTERN HAS COST USERS -- sky_dots on
        # 2026-08-09, the component loop directly above on 2026-08-25, and this
        # one, which I walked straight past while fixing that one. See the sharp
        # edges list in ARCHITECTURE.md.
        rows = np.any(m, axis=1)
        if not rows.any():
            new_preds.append(pred)
            new_masks.append(m)
            continue
        cols = np.any(m, axis=0)
        rr = np.where(rows)[0]
        cc = np.where(cols)[0]
        r0, r1 = int(rr[0]), int(rr[-1]) + 1
        c0, c1 = int(cc[0]), int(cc[-1]) + 1
        sub = m[r0:r1, c0:c1] > 0
        orig = int(sub.sum())
        sub_kept = sub & notkill[r0:r1, c0:c1]
        kept = int(sub_kept.sum())
        if orig > 0 and kept < orig * _PHANTOM_KEEP_FRAC:
            n_dropped += 1
            continue
        if kept < orig:
            try:
                conf = float(pred.score.value)
            except Exception:
                conf = 0.5
            # The synthetic prediction still needs a FULL-frame mask, but only
            # the trimmed ones pay for that allocation now, not all 53.
            trimmed = np.zeros(m.shape, dtype=bool)
            trimmed[r0:r1, c0:c1] = sub_kept
            new_preds.append(_SyntheticPred(trimmed, conf))
            # Deliberately NOT cached: let the next stage derive this one the
            # normal way, so the cache can never disagree with what the pipeline
            # would have produced on its own.
            new_masks.append(None)
            n_trimmed += 1
        else:
            new_preds.append(pred)
            new_masks.append(m)

    state.raw_detections = new_preds
    state.pred_masks = new_masks
    log.count("phantom_lines", n_lines)
    log.count("detections_dropped", n_dropped)
    log.count("detections_trimmed", n_trimmed)
    if n_dropped or n_trimmed:
        log.event("phantoms_pruned", lines=n_lines, dropped=n_dropped, trimmed=n_trimmed)
    for rec in phantom_records:   # per-phantom location for hard-negative mining
        log.event("phantom_removed", **rec)
    return state


def stage_fit_polygons(state: PipelineState, cfg: StageConfig,
                       log: StageLog) -> PipelineState:
    """Stage 2: filter raw SAHI detections, group fragments that belong to the
    same trail, and fit a polygon per group (strip-based fit for curved trails).
    No gap bridging here -- that is a later, deliberate stage.

    WHAT COMES OUT: state.polygons (the corner sets a human would draw around
    each trail), state.polygon_segs (one mask per polygon, which repair walks),
    and state.final_mask (every trail pixel in one image). Everything after this
    point works from those, not from the AI's raw output.

    MASKS ARE BORROWED, NOT BUILT, where phantom pruning already made them (see
    PipelineState.pred_masks). This stage is the last reader, so it releases each
    mask as it takes it and clears the list at the end: holding all 53 of them to
    the end of the stage cost about 2.3 GB on a 44MP frame for no benefit. A
    cache entry of None, or a list whose length does not match the detections,
    means "build it here" -- the shortcut can never produce a different answer
    from doing the work."""
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
    frame_px = h * w
    t_premask = t_scan = t_split = t_par = t_props = 0.0
    n_split_fired = 0
    n_kept_whole = 0
    rescued_blobs = []   # (crop_mask, row0, col0) of kept-whole crossing tangles
    # Reuse the masks phantom pruning already built for these same detections
    # rather than redrawing all 53 of them from their outlines a second time.
    # The lengths must match exactly or the pairing is meaningless, and a None
    # entry means "not cached" -- both fall back to building it here, so this is
    # only ever a shortcut, never a different answer.
    cached_masks = (state.pred_masks
                    if len(state.pred_masks) == len(preds) else [])
    for _i, pred in enumerate(preds):
        _t = time.perf_counter()
        m = cached_masks[_i] if cached_masks else None
        if cached_masks:
            # HAND THE MASK OVER, DO NOT KEEP A SECOND COPY OF THE LIST'S HOLD
            # ON IT. Sharing masks between the stages saves real time, but 53
            # detections on a 44MP frame is about 2.3 GB, and keeping all of it
            # alive until the stage ends is trading memory we may not have for
            # speed we already banked. Measured 2026-08-26: peak went UP after
            # the sharing change. `m` keeps this one alive for the iteration and
            # it is released when the next iteration rebinds m, so only one mask
            # is held at a time instead of all of them.
            cached_masks[_i] = None
        if m is None:
            m = _pred_to_mask(pred, h, w)
        t_premask += time.perf_counter() - _t
        if m is None:
            continue
        # Scan ONCE to find this detection's bounding box, then crop. The split
        # and props steps each re-scan, but now on the small crop instead of the
        # whole frame. frame_px=h*w keeps their area thresholds normalised to the
        # full frame, so the result is identical -- just without the redundant
        # full-frame scans. coords/centroid come back crop-local and are offset.
        # Find the bbox with axis reductions (np.any), not np.where: ~30x cheaper
        # (1.6ms vs 49ms on a 24MP mask) and the exact same box, so the crop and
        # all downstream results are byte-identical.
        _t = time.perf_counter()
        rows = np.any(m, axis=1)
        if not rows.any():
            t_scan += time.perf_counter() - _t
            continue
        cols = np.any(m, axis=0)
        rr = np.where(rows)[0]
        cc = np.where(cols)[0]
        r0, r1 = int(rr[0]), int(rr[-1])
        c0, c1 = int(cc[0]), int(cc[-1])
        t_scan += time.perf_counter() - _t
        mc = m[r0:r1 + 1, c0:c1 + 1]
        off = np.array([r0, c0])
        _t = time.perf_counter()
        crossing_pieces = split_crossing(mc, frame_px=frame_px)
        t_split += time.perf_counter() - _t
        is_confirmed_crossing = len(crossing_pieces) > 1
        if is_confirmed_crossing:
            n_split_fired += 1
        else:
            # Crossing rescue: multi-touch tangles (3+ trails through one blob)
            # defeat the spine+tips split, so split_crossing returns the blob
            # whole -- and a squat unsplit tangle then dies at the aspect gate,
            # deleting every real trail the model found inside it (143A8819:
            # all four trails masked at 0.81 confidence, all dropped). If the
            # blob still shows the splitter's own crossing evidence (2+ line
            # directions at a real angle, thin-trail fill that a flood can't
            # have), keep the model's mask EXACTLY as-is: the blob's own pixels
            # are painted into the final mask after polygon fitting (a tangle
            # has no faithful simple polygon -- a fitted quad both over-covers
            # and trims arms). The repair cleans a whole tangle fine -- it
            # borrows sky by star tracking and needs no per-trail separation.
            _t = time.perf_counter()
            if has_crossing_evidence(mc, frame_px=frame_px):
                n_kept_whole += 1
                log.count("crossings_kept_whole")
                log.event("crossing_kept_whole",
                          area=int((mc > 0).sum()),
                          bbox=[int(c0), int(r0), int(c1), int(r1)])
                rescued_blobs.append(((mc > 0).astype(np.uint8), r0, c0))
                t_split += time.perf_counter() - _t
                continue
            t_split += time.perf_counter() - _t
        for cm in crossing_pieces:
            _t = time.perf_counter()
            parallel_pieces = _try_split_parallel(cm, frame_px=frame_px)
            t_par += time.perf_counter() - _t
            for em in parallel_pieces:
                _t = time.perf_counter()
                props = _props_with_log(em, cfg, log,
                                        skip_aspect=is_confirmed_crossing,
                                        frame_px=frame_px)
                t_props += time.perf_counter() - _t
                if props is not None:
                    props["coords"] = props["coords"] + off
                    props["centroid"] = props["centroid"] + off
                    props["conf"] = float(
                        getattr(getattr(pred, "score", None), "value", 0.0) or 0.0)
                    det_list.append(props)
    log.count("detections_passed", len(det_list))
    log.count("crossings_split", n_split_fired)
    # Per-step timing so the log shows what fired and how long (no probes needed).
    log.event("substep_timing", pred_to_mask_s=round(t_premask, 3),
              bbox_scan_s=round(t_scan, 3),
              split_crossing_s=round(t_split, 3),
              parallel_split_s=round(t_par, 3), props_s=round(t_props, 3))
    if not det_list and not rescued_blobs:
        state.det_list = det_list
        state.final_mask = final
        return state

    # Group collinear, touching fragments into trails.
    _t = time.perf_counter()
    groups = group_detections(det_list) if det_list else []
    t_group = time.perf_counter() - _t
    log.count("groups", len(groups))

    # Fit one polygon per group (curved fit when the group is long and bends).
    _t = time.perf_counter()
    final, polygons = _fit_groups(det_list, groups, h, w)
    t_fit = time.perf_counter() - _t
    log.event("fit_timing", group_s=round(t_group, 3), poly_fit_s=round(t_fit, 3))

    # Rescued crossing tangles enter the final mask as their EXACT model
    # pixels, with the blob's (lightly simplified) contour standing in as the
    # polygon so repair and the viewers treat it like any other region. This
    # keeps every arm the model masked (the fitted-quad path trimmed the 8819
    # vertical stub) without the quad's fat over-coverage.
    for _mb, _rb, _cb in rescued_blobs:
        final[_rb:_rb + _mb.shape[0], _cb:_cb + _mb.shape[1]][_mb > 0] = 255
        _cnts, _ = cv2.findContours(_mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for _cn in _cnts:
            if cv2.contourArea(_cn) < 50:
                continue
            _cn = cv2.approxPolyDP(_cn, 2.0, True).reshape(-1, 2)
            if len(_cn) < 3:
                continue
            polygons.append((_cn + np.array([_cb, _rb])).tolist())

    log.count("polygons", len(polygons))
    log.count("mask_components",
              max(0, cv2.connectedComponents(
                  (final > 0).astype(np.uint8))[0] - 1))
    state.det_list = det_list
    state.groups = groups
    # The shared masks have served their purpose: this is the last stage that
    # reads them, so let the memory go rather than carrying it through the rest
    # of the pipeline and back out to the caller.
    state.pred_masks = []
    state.polygons, state.polygon_segs = _polys_and_segs_deduped(polygons, h, w)
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
    """Find the two end points (tips) of a grouped trail and its overall
    direction.

    Inputs:
      grp      -- list of indices into det_list belonging to one trail group.
      det_list -- the full detection list; each entry has pixel `coords`, a
                  unit direction `u`, and an `area`.

    Returns (tip_min_rc, tip_max_rc, u_avg) as (row, col) arrays:
      tip_min / tip_max -- the two ends of the trail along its main axis.
      u_avg             -- area-weighted average unit direction of the group.

    Why it exists: Stage 5's seam bridge needs each fragment's two ends so it
    can measure tip-to-tip gaps between fragments. Used by _find_gap_bridge_tiles."""
    all_dets = [det_list[i] for i in grp]
    all_coords = np.vstack([d["coords"] for d in all_dets])
    # Average the per-detection directions, weighted by area, but flip any that
    # point the opposite way first (u and -u describe the same line) so they
    # reinforce instead of cancelling.
    u_sum = np.zeros(2)
    for d in all_dets:
        u = d["u"] if u_sum.dot(d["u"]) >= 0 else -d["u"]
        u_sum += u * d["area"]
    u_avg = u_sum / np.linalg.norm(u_sum)
    # Project every pixel onto the trail axis; the min and max projections mark
    # the two tips. t_c centres the projection on the centroid so the tips are
    # returned as actual row-col points back on the trail line.
    centroid = all_coords.mean(axis=0)
    t_c = float(centroid @ u_avg)
    t = all_coords @ u_avg
    tip_min = centroid + (float(t.min()) - t_c) * u_avg
    tip_max = centroid + (float(t.max()) - t_c) * u_avg
    return tip_min, tip_max, u_avg


def _find_gap_bridge_tiles(groups, det_list, h, w, tile_size):
    """Find pairs of trail fragments that are really one trail broken by a gap.

    Returns two lists:
      tiles      -- [(gi, gj, tile_x, tile_y), ...]: one re-inference tile per
                    qualifying pair, centered on the gap. This only chooses
                    WHERE to re-infer; it does not bridge.
      gap_misses -- [{cx, cy, tile, gap_px, frag_a_bbox, frag_b_bbox}, ...]: one
                    record per qualifying pair describing WHERE the detector
                    missed. A pair only qualifies after the geometric criteria
                    (matching angle, matching width, co-linear tips, a tile seam
                    inside the gap, tip-to-tip vector aligned with the trail)
                    prove the two fragments are a single real trail. So every
                    record is a spot where a real trail exists but the model
                    produced no detection in between -- a genuine model miss.
                    These are captured as TRAINING FEEDBACK: the highest-value
                    examples to add to the next training round so the model
                    learns to fire where it currently does not.
    """
    extra = []
    gap_misses = []
    seen = set()
    n = len(groups)
    _stride = int(tile_size * 0.8)

    def _tile_bounds_1d(size):
        """Collect every tile START and END coordinate along one dimension.
        These are the lines where the tile grid has a SEAM -- the spots where
        the model is most likely to have missed a trail. A candidate gap only
        bridges if a seam falls inside it (see x_seam / y_seam below)."""
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

            # Gate 1 (angle): the two fragments must point the same way. abs()
            # treats u and -u as equal; the min() picks the smaller of the angle
            # and its 180-complement so flipped directions still read as aligned.
            u_i = di_list[0]["u"]
            u_j = dj_list[0]["u"]
            cos_sim = min(abs(float(np.dot(u_i, u_j))), 1.0)
            adiff = min(np.degrees(np.arccos(cos_sim)),
                        180.0 - np.degrees(np.arccos(cos_sim)))
            if adiff > _BRIDGE_MAX_ANGLE:
                continue

            # Gate 2 (width): the two fragments must be of similar thickness. A
            # real trail keeps a consistent width; a 3x mismatch means different
            # objects. Median minor-axis is the per-fragment thickness.
            minor_i = float(np.median([d["minor"] for d in di_list]))
            minor_j = float(np.median([d["minor"] for d in dj_list]))
            if max(minor_i, minor_j) / max(min(minor_i, minor_j), 1) > _BRIDGE_MAX_WIDTH:
                continue

            all_i = np.vstack([det_list[k]["coords"] for k in groups[gi]])
            all_j = np.vstack([det_list[k]["coords"] for k in groups[gj]])

            # Facing tips: the two ends that sit nearest each other across the gap.
            tip_i_min, tip_i_max, _ = _group_tips(groups[gi], det_list)
            tip_j_min, tip_j_max, _ = _group_tips(groups[gj], det_list)
            combos = [
                (tip_i_min, tip_j_min), (tip_i_min, tip_j_max),
                (tip_i_max, tip_j_min), (tip_i_max, tip_j_max),
            ]
            # Gate 3 (gap length): of the four tip-pair combinations, take the
            # closest pair (best_a, best_b). If even that closest gap is wider
            # than _BRIDGE_MAX_GAP, the fragments are too far apart to bridge.
            best_dist, best_a, best_b = float("inf"), None, None
            for ta, tb in combos:
                d = float(np.linalg.norm(ta - tb))
                if d < best_dist:
                    best_dist, best_a, best_b = d, ta, tb
            if best_dist > _BRIDGE_MAX_GAP:
                continue

            # Lateral (perpendicular) offset measured ACROSS THE GAP at the facing
            # tips, not between the group centroids. A centroid-based offset blows
            # up when one fragment is far longer than the other: the long
            # fragment's centroid sits far down-trail, so a tiny angle difference
            # turns into a large perpendicular distance and wrongly rejects a
            # straight trail. The facing-tip offset stays local to the gap.
            diff = best_b - best_a
            along = float(np.dot(diff, u_i))
            perp = float(np.sqrt(max(float(np.dot(diff, diff)) - along ** 2, 0.0)))
            if perp > 0.9 * max(minor_i, minor_j):
                continue

            # Gate 4 (seam present): only bridge a gap that straddles a tile-grid
            # seam, because that is where the model's blind spot is. Build the
            # empty span between the two fragments' bounding boxes (x_lo..x_hi
            # horizontally, y_lo..y_hi vertically) and require a grid seam to fall
            # inside that span, within _BRIDGE_CLIP_TOL px of a fragment edge.
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

            # Gate 5 (tip vector aligned): the line connecting the two facing
            # tips must point along the trail's direction, not sideways. This
            # rejects two parallel-but-offset trails (their tips line up across,
            # not along). u_avg is the combined trail direction (u_j flipped to
            # agree with u_i first); the tip-to-tip vector must be within
            # _BRIDGE_TIP_ANGLE degrees of it.
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

            # Place one re-inference tile centred on the gap midpoint, clamped so
            # the tile stays fully inside the frame.
            mid_rc = (best_a + best_b) / 2.0
            mid_y = int(round(float(mid_rc[0]))); mid_x = int(round(float(mid_rc[1])))
            new_tx = max(0, min(w - tile_size, mid_x - tile_size // 2))
            new_ty = max(0, min(h - tile_size, mid_y - tile_size // 2))
            pair_key = (gi, gj)
            if pair_key not in seen:
                seen.add(pair_key)
                extra.append((gi, gj, new_tx, new_ty))
                # Record this confirmed gap as a model MISS for training feedback.
                # All criteria above have passed, so we KNOW these two fragments
                # are one real trail; the detector simply fired nothing in the
                # stretch between them. cx/cy is the gap centre, and the two
                # fragment boxes pinpoint exactly where the model should have
                # detected a trail but did not.
                gap_misses.append({
                    "cx": mid_x, "cy": mid_y,
                    "tile": _tile_coord(mid_x, mid_y, _stride),
                    "gap_px": round(float(best_dist), 1),
                    "frag_a_bbox": [int(all_i[:, 1].min()), int(all_i[:, 0].min()),
                                    int(all_i[:, 1].max()), int(all_i[:, 0].max())],
                    "frag_b_bbox": [int(all_j[:, 1].min()), int(all_j[:, 0].min()),
                                    int(all_j[:, 1].max()), int(all_j[:, 0].max())],
                })
    return extra, gap_misses


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
        # Rotate the crop a quarter turn so a trail the model misses at its true
        # orientation gets a second chance at the rotated orientation. Width and
        # height swap after the rotation, hence pad_h/pad_w are swapped too.
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
                # Undo the quarter turn (rot90 by 3 = -1) so the mask lines up
                # with the original, unrotated frame before pasting it back.
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

    pairs, gap_misses = _find_gap_bridge_tiles(groups, det_list, h, w, cfg.tile_size)
    log.count("candidate_pairs", len(pairs))
    # Training-feedback log. Each confirmed gap is a place the detector should
    # have fired but did not. Record it in the run log (one "bridge_gap_miss"
    # event per gap) REGARDLESS of whether the follow-up re-inference below
    # manages to fill the gap -- the miss is real either way, and these are the
    # highest-value examples to add to the next training round. This mirrors how
    # the FP suppressor logs every false positive it removes; here we log every
    # real trail the model failed to detect. Pull these from the run log when
    # assembling the next training batch.
    for _m in gap_misses:
        log.event("bridge_gap_miss", **_m)
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

    # Re-inference enrichment: when the model DID fire in a gap (e.g. B10), fold
    # those real pixels in and regroup, which physically connects the fragments
    # they span. When the model fired nothing (e.g. IMG_3540, model blind), this
    # adds nothing and the fragments stay split for now.
    if new_dets:
        groups_before = len(groups)
        det_list = list(det_list) + new_dets
        groups = group_detections(det_list)
        log.event("regroup", groups_before=groups_before, groups_after=len(groups))

    # Bridge on the criteria. Any candidate pair still unmerged after the step
    # above is a gap the 6 gates ALREADY confirmed is one real trail, the model
    # just couldn't see across it. The 6 gates ARE the decision (that is the
    # whole point of the bridge: it exists for detection failures), so merge the
    # two groups directly, like the original detect_trails bridge did. fit_groups
    # then spans the gap. Non-trail pairs (the telescope mount) never get here:
    # the 6 gates reject them, so they are not candidates. Re-find candidates on
    # the (possibly regrouped) set so indices are current.
    pairs2, _ = _find_gap_bridge_tiles(groups, det_list, h, w, cfg.tile_size)
    if pairs2:
        parent = list(range(len(groups)))
        def _bf(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        bridged = 0
        for gi, gj, _tx, _ty in pairs2:
            if _bf(gi) != _bf(gj):
                parent[_bf(gi)] = _bf(gj)
                bridged += 1
        comp = {}
        for idx, grp in enumerate(groups):
            comp.setdefault(_bf(idx), []).extend(grp)
        groups = list(comp.values())
        log.count("bridge_merges_on_criteria", bridged)

    final, polygons = _fit_groups(det_list, groups, h, w)
    state.det_list = det_list
    state.groups = groups
    state.polygons, state.polygon_segs = _polys_and_segs_deduped(polygons, h, w)
    state.final_mask = final
    return state


# --- FP suppressor constants and helpers ---------------------------------- #

_SFP_PIXEL_DIFF_THRESH = 8.0   # mean abs pixel diff below this = "same content"
_SFP_MIN_MATCHES       = 1     # min neighbor matches to trigger suppression
_SFP_EDGE_PX           = 20    # frame edge veto zone (px)
_SFP_BRIGHT_RATIO      = 2.5   # 90th-pct inside / median surround; above = real trail


def _tile_coord(cx, cy, stride):
    """Convert a pixel position (cx, cy) to a human-readable tile label like
    'B10' for the logs. Row becomes a letter (A, B, C, ...), column becomes a
    1-based number, both derived by dividing the pixel position by the tile
    stride. Used so log entries name the same grid cell Bruce sees in the
    MaskViewR / Mask CheckR overlays."""
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
    if cfg.prune_phantoms:
        with flog.stage("prune_phantoms") as s:
            state = stage_prune_phantoms(state, cfg, s)
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
    # Full per-stage records (seconds + counts + events) so callers can record
    # what fired and how long in their own run log without a separate log file.
    state.stage_log = flog.stages

    if log_path:
        with open(log_path, "a") as f:
            f.write(flog.to_jsonl() + "\n")

    return state
