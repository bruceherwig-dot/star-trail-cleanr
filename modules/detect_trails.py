"""
Trail detection — YOLO segmentation model with SAHI tiled inference.

WHY TILED INFERENCE
-------------------
The YOLO model processes images at 640x640 pixels. Astrophotography frames are
typically 6000x4000 or larger. If we simply downscaled the full frame to 640x640,
a trail that spans 3000 pixels in the original would become a 2-pixel smear — too
thin for the model to detect reliably. Instead, SAHI (Slicing Aided Hyper Inference)
divides each frame into overlapping 640x640 tiles, runs the model on every tile, then
stitches all the per-tile detections back to full-frame pixel coordinates. Overlapping
tiles (rather than non-overlapping) ensure that a trail near a tile edge is seen in
full by at least one tile and not missed at the seam.

WHAT COMES BACK FROM THE MODEL
-------------------------------
Each tile inference produces a set of predictions. Each prediction contains:
- A segmentation mask: a boolean array (640x640) marking which pixels the model
  thinks are part of a trail.
- A confidence score: how certain the model is (0.0 to 1.0). We threshold at 0.25.
- A bounding box for the detection.

The tile mask is scaled back to its position in the full frame and combined with
all other tile results to produce one full-frame binary mask per detection.

GROUPING AND POLYGON FITTING (modules/trail_grouper.py)
-------------------------------------------------------
A single physical trail often crosses several tile boundaries. SAHI produces one
detection fragment per tile, so a trail spanning three tiles comes back as three
separate mask blobs. The grouper (trail_grouper.py) fuses fragments that belong to
the same trail using geometric tests: angle alignment, width consistency, perpendicular
offset, and tip-to-tip distance. Once grouped, fit_polygon() wraps the fused pixel
mask in a clean 4-corner rectangle, which is the format written to the CVAT annotation
tool and used for repair masking.

SKY MASK
--------
apply_sky_mask() takes the combined detection mask and zeroes out any pixel that falls
in the "foreground" zone — the part of the frame that is landscape, buildings, or other
static terrestrial objects. The foreground mask is computed once per batch by finding
pixels that are bright across many frames (foreground is always lit; sky varies).
Removing foreground pixels from detections prevents bright edges, illuminated structures,
and other static bright objects from passing through to the repair step.

PUBLIC ENTRY POINT
------------------
detect_frame_polygon(frame, model, sky_mask, ...) — call this once per frame.
Returns a list of per-trail dicts, each with:
  - 'mask': full-frame binary uint8 array (255 = trail pixel)
  - 'polygon': 4-corner rectangle in pixel coordinates
  - 'conf': YOLO confidence score for this detection
  - timing sub-fields for the JSONL run log
"""
import cv2
import math
import numpy as np
import os
import time
from typing import Optional

from .io_safe import robust_imread
from .trail_grouper import (filter_masks, filter_masks_with_props, group_detections,
                            fit_polygon, fit_curved_group, _group_angle_spread,
                            _CURVED_MIN_XSPAN, _CURVED_MIN_ANGLE_SPREAD,
                            try_split, detection_props,
                            MIN_AREA, _REF_FRAME_PX)


# ── Device selection ──────────────────────────────────────────────────────────
# Picks the fastest compute device at startup. cuda > mps > cpu.
# Falls back to cpu and sets STC_CUDA_UNSUPPORTED env var on cuda errors.

def best_device() -> str:
    """Return the best available inference device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                _t = torch.zeros(1, device='cuda')
                _ = _t + _t
                torch.cuda.synchronize()
                del _t
                return "cuda"
            except Exception:
                os.environ['STC_CUDA_UNSUPPORTED'] = '1'
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ── Model loading ─────────────────────────────────────────────────────────────
# Loads the YOLO segmentation model via SAHI's AutoDetectionModel wrapper.
# Tries the requested device first; falls back to cpu on failure.

def load_model(model_path: str, confidence: float = 0.25,
               device: Optional[str] = None):
    """Load YOLOv8-seg model via SAHI AutoDetectionModel.

    device=None (or "auto") picks the best available: cuda > mps > cpu.
    Pass an explicit string to override.
    """
    if not device or device == "auto":
        device = best_device()
    from sahi import AutoDetectionModel
    try:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=confidence,
            device=device,
        )
    except Exception:
        if device != "cpu":
            os.environ['STC_CUDA_UNSUPPORTED'] = '1'
            model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(model_path),
                confidence_threshold=confidence,
                device="cpu",
            )
        else:
            raise
    return model


# ── Mask construction helpers ─────────────────────────────────────────────────
# _build_combined_mask: unions all SAHI prediction masks into one binary frame mask.
# _load_as_rgb: normalizes any input (path or array, 8/16-bit, gray/BGRA) to RGB uint8.
# _build_raw_labeled_mask: labels each prediction by index (1..N) before filtering,
#   used by MaskViewR to show the unfiltered SAHI output in yellow.

def _build_combined_mask(predictions, h, w):
    """Union every SAHI prediction mask into one full-frame binary mask.

    predictions: list of SAHI predictions (each has .mask.bool_mask).
    h, w:        target full-frame height and width.

    Returns an H x W uint8 mask where 255 = "some prediction marked this pixel as
    trail" and 0 = background. Predictions are combined with a pixel-wise maximum
    (logical OR), so overlapping detections simply merge. A prediction whose mask
    is already full-frame is used as-is; one at a different size is nearest-resized
    up to full frame first. Skips predictions with no mask.
    """
    out = np.zeros((h, w), dtype=np.uint8)
    for pred in predictions:
        if pred.mask is None:
            continue
        seg = pred.mask.bool_mask
        if seg is None:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        if seg.shape == (h, w):
            m[seg.astype(bool)] = 255
        else:
            # Mask is not full-frame (e.g. tile-local); scale it up to fit.
            m = cv2.resize(seg.astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST)
        if m.any():
            out = np.maximum(out, m)
    return out


def _load_as_rgb(image):
    """Normalize any input image to 8-bit RGB and report its size.

    image: either a file path (read from disk) or an already-loaded numpy array.

    Returns (rgb_uint8, h, w), or None if the file could not be read. Handles all
    the messy input variations so the rest of the pipeline only ever sees clean
    3-channel 8-bit RGB: 16-bit is down-shifted to 8-bit, float/other dtypes are
    clipped to 0-255, grayscale and BGRA are converted to RGB, and plain BGR is
    swapped to RGB. (Astrophotography sources are commonly 16-bit TIFFs, so the
    16-bit path matters in practice.)
    """
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


# ── SAHI tiled inference ──────────────────────────────────────────────────────
# Cuts the frame into overlapping tiles (tile_size x tile_size, overlap ratio),
# runs the YOLO model on each tile, and merges results with IOS NMS.
# postprocess_match_threshold=1.1 disables NMS suppression so tile-edge detections
# are not dropped. Returns the raw prediction list before any filtering.
# Log field: detect record sahi_raw_count = len(predictions) from this call.

def _sahi_predict(model, img, tile_size, overlap):
    """Run SAHI tiled inference and return raw prediction list."""
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


# ── Foreground-tile skip helpers ──────────────────────────────────────────────
# When a foreground mask is available, tiles that are 100% foreground contain
# no sky and therefore no trails. Skipping them saves one YOLO forward pass
# (~45ms on MPS) per skipped tile. On high-foreground datasets this cuts the
# fixed inference cost by 30–60%.
#
# _sahi_predict_skip replaces _sahi_predict when fg_mask is provided. It runs
# the same tile grid but batches only the non-foreground tiles through YOLO
# directly (bypassing SAHI's per-tile overhead). Returns (predictions, n_skipped).
# Predictions are _SyntheticPred objects with the same .mask.bool_mask and
# .score.value interface as SAHI ObjectPredictions.

class _PredMaskWrap:
    """Holds one boolean mask under the attribute name SAHI uses (.bool_mask).

    Exists so a _SyntheticPred can expose pred.mask.bool_mask exactly like a real
    SAHI prediction, letting the rest of the pipeline read it without caring
    whether it came from SAHI or from our fast skip-path inference.
    """
    __slots__ = ("bool_mask",)
    def __init__(self, bm):
        self.bool_mask = bm

class _PredScoreWrap:
    """Holds one confidence score under the attribute name SAHI uses (.value).

    Mirror of _PredMaskWrap for the score side, so pred.score.value works the
    same on our synthetic predictions as on real SAHI ones.
    """
    __slots__ = ("value",)
    def __init__(self, v):
        self.value = v

class _SyntheticPred:
    """A stand-in that looks exactly like a SAHI ObjectPrediction to our code.

    The fast foreground-skip path (_sahi_predict_skip) runs YOLO directly instead
    of through SAHI, so it gets raw YOLO masks, not SAHI prediction objects. We
    wrap each one in a _SyntheticPred so it exposes the same .mask.bool_mask and
    .score.value attributes the rest of the pipeline expects. That way the two
    inference paths are interchangeable downstream and nothing else needs to know
    which one produced a given detection.
    """
    __slots__ = ("mask", "score")
    def __init__(self, bool_mask, conf):
        self.mask  = _PredMaskWrap(bool_mask)
        self.score = _PredScoreWrap(float(conf))


def _tile_starts(size, tile_size, stride):
    """Compute the left/top start coordinates of every tile along one axis.

    size:      length of the image along this axis (width for x, height for y).
    tile_size: tile length (e.g. 640).
    stride:    step between consecutive tile starts (tile_size minus the overlap).

    Returns a list of start positions. Tiles are stepped by `stride` so they
    overlap, matching how SAHI lays out its grid. The final guard appends one
    last tile flush against the far edge (start = size - tile_size) whenever the
    stepped tiles don't already reach the edge, so no strip of pixels at the
    bottom/right is ever left uncovered.
    """
    starts = []
    x = 0
    while x + tile_size <= size:
        starts.append(x)
        x += stride
    # If the stepped tiles stop short of the far edge (or there are none at all
    # because the image is smaller than one tile), add a final edge-aligned tile.
    if not starts or starts[-1] + tile_size < size:
        starts.append(size - tile_size)
    return starts


def _sahi_predict_skip(model, img_rgb, tile_size, overlap, fg_mask):
    """Tiled YOLO inference that skips 100%-foreground tiles.

    img_rgb: H x W x 3 RGB numpy array (same format as _sahi_predict).
    fg_mask: H x W uint8, >0 = foreground pixel.
    Returns (predictions, n_skipped) where predictions is a list of
    _SyntheticPred objects compatible with the rest of the pipeline.
    """
    h, w = img_rgb.shape[:2]
    stride = int(tile_size * (1 - overlap))
    xs = _tile_starts(w, tile_size, stride)
    ys = _tile_starts(h, tile_size, stride)

    # Resize fg_mask to frame dims if needed (should always match but be safe)
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
            # YOLO expects BGR; our image is RGB
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


# ── Public: detect_frame (combined pixel mask, no polygon fitting) ────────────
# Used by the legacy detect_frame path. Not the active pipeline in STC v5.
# The active pipeline uses detect_frame_polygon below.

def detect_frame(model, image, tile_size: int = 640,
                 overlap: float = 0.2, dilate: int = 1) -> Optional[np.ndarray]:
    """Run SAHI tiled inference on one frame.

    Returns binary uint8 mask (255=trail, 0=sky) at original resolution,
    or None if the image cannot be read.
    """
    loaded = _load_as_rgb(image)
    if loaded is None:
        return None
    img, h, w = loaded

    passing = filter_masks(_sahi_predict(model, img, tile_size, overlap), h, w, img)
    mask = np.zeros((h, w), dtype=np.uint8)
    for m in passing:
        mask = np.maximum(mask, m)

    if dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    return mask


def _build_raw_labeled_mask(predictions, h, w):
    """Paint each raw SAHI prediction onto one mask with a distinct id (1..N).

    predictions: list of SAHI predictions, in order.
    h, w:        full-frame size.

    Returns an H x W uint8 mask where each pixel holds the 1-based index of the
    prediction covering it (0 = background). Unlike _build_combined_mask (which
    flattens everything to 255), this keeps detections individually identifiable
    so MaskViewR can show the unfiltered SAHI output as separate numbered shapes.
    Indices above 255 cannot fit in uint8 and are dropped (rare in practice).
    """
    out = np.zeros((h, w), dtype=np.uint8)
    for idx, pred in enumerate(predictions, start=1):
        if pred.mask is None or pred.mask.bool_mask is None:
            continue
        seg = pred.mask.bool_mask
        if seg.shape == (h, w):
            mask_bool = seg.astype(bool)
        else:
            m = cv2.resize(seg.astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST)
            mask_bool = m > 0
        if mask_bool.any() and idx <= 255:
            out[mask_bool] = idx
    return out


# ── Targeted gap-bridge pass ──────────────────────────────────────────────────
# After the initial grouper, scan every pair of groups. If two groups look like
# the same trail (similar angle, similar width, co-linear) but their nearest
# tips are farther apart than the grouper's tip-to-tip gate allows, run one
# extra inference tile centered in the gap between them.
#
# This targets the specific failure mode at tile seam corners where the model's
# confidence drops between two otherwise-connected detections, without firing on
# every trail tip (which the tile-edge-margin approach did, causing 17+ extra
# tiles per frame and unacceptable slowdown).
#
# Uses direct YOLO forward pass (not SAHI get_prediction) to keep per-tile
# overhead at ~5ms instead of ~300ms.

_BRIDGE_MAX_GAP        = 550   # px: max tip-to-tip distance to attempt a bridge
_BRIDGE_MAX_ANGLE      = 12.0  # degrees: loosened from 7.0; PCA on fragment blobs is noisy
_BRIDGE_MAX_WIDTH      = 3.0   # ratio: same as grouper gate 2
_BRIDGE_TIP_ANGLE      = 20.0  # degrees: tip-to-tip vector must align with average trail direction
_BRIDGE_CLIP_TOL       = 3     # px: facing edge must be within this many px of the seam boundary


def _group_tips(grp, det_list):
    """Find the two endpoints ("tips") of a grouped trail and its direction.

    grp:      list of indices into det_list naming the detections in this group.
    det_list: full list of detection-property dicts (each has "coords", a unit
              direction "u", and an "area").

    Returns (tip_min_rc, tip_max_rc, u_avg):
      - tip_min_rc / tip_max_rc: the two far ends of the trail as (row, col)
        points, found by projecting every pixel onto the trail's axis and taking
        the lowest and highest projections.
      - u_avg: the group's average direction as a (row, col) unit vector.

    Used by the gap-bridge pass to decide whether two groups are two ends of one
    physical trail. Each detection's direction is flipped to point the same way
    before averaging (a line's direction is ambiguous by 180 degrees), and the
    average is area-weighted so larger fragments dominate the direction estimate.
    """
    all_dets = [det_list[i] for i in grp]
    all_coords = np.vstack([d["coords"] for d in all_dets])
    # Sum direction vectors, flipping any that point the opposite way so they
    # reinforce instead of cancel; weight by area so big fragments count more.
    u_sum = np.zeros(2)
    for d in all_dets:
        u = d["u"] if u_sum.dot(d["u"]) >= 0 else -d["u"]
        u_sum += u * d["area"]
    u_avg = u_sum / np.linalg.norm(u_sum)
    centroid = all_coords.mean(axis=0)
    # Project all pixels onto the trail axis; the min/max projections are the tips.
    t_c = float(centroid @ u_avg)
    t = all_coords @ u_avg
    tip_min = centroid + (float(t.min()) - t_c) * u_avg
    tip_max = centroid + (float(t.max()) - t_c) * u_avg
    return tip_min, tip_max, u_avg


def _split_disconnected_groups(groups, det_list):
    """Split any group whose OR-mask has 2+ disconnected components.

    Catches grouper false merges: two separate trail fragments that passed the
    grouper's gates but are spatially disconnected. Each disconnected island
    becomes its own group for independent polygon fitting.
    A 1px dilation bridges hairline gaps between adjacent tile detections
    without connecting genuinely separate clusters.
    """
    result = []
    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for grp in groups:
        if len(grp) < 2:
            result.append(grp)
            continue
        all_c = np.vstack([det_list[k]["coords"] for k in grp])
        r0 = int(all_c[:, 0].min())
        r1 = int(all_c[:, 0].max())
        c0 = int(all_c[:, 1].min())
        c1 = int(all_c[:, 1].max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
        for k in grp:
            coords = det_list[k]["coords"]
            local[coords[:, 0] - r0, coords[:, 1] - c0] = 1
        n_labels, label_map = cv2.connectedComponents(cv2.dilate(local, _kernel))
        if n_labels <= 2:
            result.append(grp)
            continue
        sub = {}
        for k in grp:
            coords = det_list[k]["coords"]
            r_c = max(0, min(r1 - r0, int(round(float(coords[:, 0].mean()))) - r0))
            c_c = max(0, min(c1 - c0, int(round(float(coords[:, 1].mean()))) - c0))
            lbl = int(label_map[r_c, c_c])
            if lbl == 0:
                flat = label_map[coords[:, 0] - r0, coords[:, 1] - c0].flatten()
                counts = np.bincount(flat)
                lbl = int(counts[1:].argmax()) + 1 if len(counts) > 1 else 1
            sub.setdefault(lbl, []).append(k)
        result.extend(sub.values())
    return result


def _find_gap_bridge_tiles(groups, det_list, h, w, tile_size):
    """Return list of (gi, gj, tile_x, tile_y) for groups that need gap validation.

    Checks every pair of groups for the seam-gap signature: similar angle,
    similar width, co-linear centroids, tips within _BRIDGE_MAX_GAP px.
    For qualifying pairs, one tile is placed at the midpoint of the nearest tips.
    Each group pair appears at most once (deduped by (gi, gj)).
    """
    extra = []
    seen = set()
    n = len(groups)

    # Pre-compute all SAHI tile boundary positions (x and y) for this frame.
    # Stride = tile_size * (1 - overlap) = tile_size * 0.8 for overlap=0.2.
    # Boundaries are: left edges at stride multiples, right edges = left + tile_size,
    # plus the last tile's left edge (frame_size - tile_size) which may not fall on
    # a stride multiple, and its right edge (frame_size).
    _stride = int(tile_size * 0.8)

    def _tile_bounds_1d(size):
        """Return the set of every tile-edge coordinate (left and right) along
        one axis, including the flush-to-edge last tile. These are the lines
        where SAHI's NMS can clip a trail, so Gate 4 below only fires a bridge
        when the gap straddles one of them."""
        bounds = set()
        k = 0
        while True:
            left = k * _stride
            if left > size - tile_size:
                break
            bounds.add(left)
            bounds.add(min(left + tile_size, size))
            k += 1
        last_left = size - tile_size
        bounds.add(last_left)
        bounds.add(size)
        return bounds

    x_bounds = _tile_bounds_1d(w)
    y_bounds = _tile_bounds_1d(h)

    for gi in range(n):
        for gj in range(gi + 1, n):
            di_list = [det_list[k] for k in groups[gi]]
            dj_list = [det_list[k] for k in groups[gj]]

            # Gate 1: angle
            u_i = di_list[0]["u"]
            u_j = dj_list[0]["u"]
            cos_sim = min(abs(float(np.dot(u_i, u_j))), 1.0)
            adiff = min(np.degrees(np.arccos(cos_sim)),
                        180.0 - np.degrees(np.arccos(cos_sim)))
            if adiff > _BRIDGE_MAX_ANGLE:
                continue

            # Gate 2: width ratio
            minor_i = float(np.median([d["minor"] for d in di_list]))
            minor_j = float(np.median([d["minor"] for d in dj_list]))
            if max(minor_i, minor_j) / max(min(minor_i, minor_j), 1) > _BRIDGE_MAX_WIDTH:
                continue

            # Gate 3: perpendicular distance between group centroids
            all_i = np.vstack([det_list[k]["coords"] for k in groups[gi]])
            all_j = np.vstack([det_list[k]["coords"] for k in groups[gj]])
            ci = all_i.mean(axis=0)
            cj = all_j.mean(axis=0)
            diff = cj - ci
            along = float(np.dot(diff, u_i))
            perp = float(np.sqrt(max(float(np.dot(diff, diff)) - along ** 2, 0.0)))
            if perp > 0.9 * max(minor_i, minor_j):
                continue

            # Get tips for both groups
            tip_i_min, tip_i_max, u_avg_i = _group_tips(groups[gi], det_list)
            tip_j_min, tip_j_max, u_avg_j = _group_tips(groups[gj], det_list)

            # Find the two nearest tips (one from each group)
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

            # Gate 4: seam-in-gap — SAHI NMS signature.
            # A tile boundary line must fall strictly inside the spatial gap
            # between the two groups' facing bbox extremes. This fires only
            # when NMS actually clipped a trail at a seam, leaving a gap that
            # straddles a grid line. No tolerance: the boundary must be in
            # the gap, not merely near an edge.
            min_c_i = float(all_i[:, 1].min())
            max_c_i = float(all_i[:, 1].max())
            min_c_j = float(all_j[:, 1].min())
            max_c_j = float(all_j[:, 1].max())
            min_r_i = float(all_i[:, 0].min())
            max_r_i = float(all_i[:, 0].max())
            min_r_j = float(all_j[:, 0].min())
            max_r_j = float(all_j[:, 0].max())

            x_lo = min(max_c_i, max_c_j)
            x_hi = max(min_c_i, min_c_j)
            y_lo = min(max_r_i, max_r_j)
            y_hi = max(min_r_i, min_r_j)

            x_seam = x_lo < x_hi and any(
                x_lo < b < x_hi
                and (abs(x_lo - b) <= _BRIDGE_CLIP_TOL or abs(x_hi - b) <= _BRIDGE_CLIP_TOL)
                for b in x_bounds
            )
            y_seam = y_lo < y_hi and any(
                y_lo < b < y_hi
                and (abs(y_lo - b) <= _BRIDGE_CLIP_TOL or abs(y_hi - b) <= _BRIDGE_CLIP_TOL)
                for b in y_bounds
            )

            if not (x_seam or y_seam):
                continue

            # Gate 5: tip-direction alignment
            # The tip-to-tip vector must align with the average trail direction.
            # Compensates for the loosened angle gate: proves the gap closes
            # along the trail axis, not at some oblique angle.
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
            tip_adiff = float(np.degrees(np.arccos(tip_cos)))
            if tip_adiff > _BRIDGE_TIP_ANGLE:
                continue

            # Midpoint of the two nearest tips → center of the bridge tile
            mid_rc = (best_a + best_b) / 2.0
            mid_y = int(round(float(mid_rc[0])))
            mid_x = int(round(float(mid_rc[1])))
            new_tx = max(0, min(w - tile_size, mid_x - tile_size // 2))
            new_ty = max(0, min(h - tile_size, mid_y - tile_size // 2))
            pair_key = (gi, gj)
            if pair_key not in seen:
                seen.add(pair_key)
                extra.append((gi, gj, new_tx, new_ty))

    return extra


def _run_targeted_tile(model, img_rgb, tile_x, tile_y, tile_size, h, w):
    """Run single-tile YOLO inference; return detection props in global coords.

    Uses the underlying YOLO model directly (bypassing SAHI get_prediction
    overhead) so each extra tile costs ~5ms instead of ~300ms.
    Converts masks back to global image coordinates and filters through
    try_split + detection_props.
    """
    import torch
    ty1, tx1 = tile_y, tile_x
    ty2, tx2 = min(h, tile_y + tile_size), min(w, tile_x + tile_size)
    crop_h, crop_w = ty2 - ty1, tx2 - tx1
    crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    if crop_h < tile_size or crop_w < tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:crop_h, :crop_w] = crop
        crop = padded

    # model.model is the SAHI wrapper; model.model.model is the ultralytics YOLO
    yolo = model.model
    conf = getattr(model, "confidence_threshold", 0.30)
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
            local_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            pts = np.array(seg_xy, dtype=np.int32)
            cv2.fillPoly(local_mask, [pts], 255)
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[ty1:ty2, tx1:tx2] = local_mask[:crop_h, :crop_w]
            for cm in try_split(global_mask):
                props = detection_props(cm)
                if props is not None:
                    props["conf"] = seg_conf
                    new_dets.append(props)
    return new_dets


def _run_targeted_tile_rot90(model, img_rgb, tile_x, tile_y, tile_size, h, w):
    """Same as _run_targeted_tile but rotates the crop 90 degrees before inference.

    Catches trails that fall dead-center in a tile and are missed by the normal
    pass because the model is sensitive to orientation.  Results are rotated back
    before returning so coords are in global image space.
    """
    import torch
    ty1, tx1 = tile_y, tile_x
    ty2, tx2 = min(h, tile_y + tile_size), min(w, tile_x + tile_size)
    crop_h, crop_w = ty2 - ty1, tx2 - tx1
    crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    # rot90(k=1) turns (crop_h, crop_w) → (crop_w, crop_h)
    crop_rot = np.ascontiguousarray(np.rot90(crop, 1))
    if crop_w < tile_size or crop_h < tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:crop_w, :crop_h] = crop_rot
        crop_rot = padded

    yolo = model.model
    conf = getattr(model, "confidence_threshold", 0.30)
    results = yolo.predict(source=crop_rot, conf=conf, verbose=False, imgsz=tile_size)

    new_dets = []
    for r in results:
        if r.masks is None:
            continue
        confs = r.boxes.conf.tolist() if r.boxes is not None else []
        for seg_idx, seg_xy in enumerate(r.masks.xy):
            if len(seg_xy) < 3:
                continue
            seg_conf = float(confs[seg_idx]) if seg_idx < len(confs) else 0.0
            local_mask_rot = np.zeros((tile_size, tile_size), dtype=np.uint8)
            pts = np.array(seg_xy, dtype=np.int32)
            cv2.fillPoly(local_mask_rot, [pts], 255)
            # Trim padding rows/cols, then rotate back: (crop_w, crop_h) → (crop_h, crop_w)
            trimmed = local_mask_rot[:crop_w, :crop_h]
            restored = np.ascontiguousarray(np.rot90(trimmed, 3))
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[ty1:ty2, tx1:tx2] = restored
            for cm in try_split(global_mask):
                props = detection_props(cm)
                if props is not None:
                    props["conf"] = seg_conf
                    new_dets.append(props)
    return new_dets


# ── Public: detect_frame_polygon (fitted polygon mask) ────────────────────────
# Active detection entry point for STC v5. Pipeline:
def _poly_angle(fill):
    """Measure which way a filled shape points, as an angle in degrees [0, 180).

    fill: an H x W mask, nonzero where the shape is.

    Computes the shape's long ("principal") axis via the eigenvectors of its
    pixel covariance matrix (the standard PCA-on-a-blob trick) and returns that
    axis's angle. Used by _clip_overlapping_polygons to tell a genuine crossing
    (two trails at clearly different angles) from a same-trail seam overlap (two
    fragments at nearly the same angle). Returns 0.0 for shapes too small to fit.
    """
    ys, xs = np.where(fill > 0)
    if len(xs) < 4:
        return 0.0
    # Second moments of the pixel cloud about its centroid → 2x2 covariance.
    dr = ys - ys.mean(); dc = xs - xs.mean()
    Irr = float((dr * dr).mean()); Icc = float((dc * dc).mean()); Irc = float((dr * dc).mean())
    _, evecs = np.linalg.eigh(np.array([[Irr, Irc], [Irc, Icc]]))
    u = evecs[:, 1]  # eigenvector of the largest eigenvalue = the long axis
    return float(np.degrees(np.arctan2(u[1], u[0])) % 180)


def _clip_overlapping_polygons(poly_fills, poly_lengths, min_fragment):
    """Clip shorter polygon fills against longer ones at crossings.

    For each overlapping pair the polygon with the longer major-axis span keeps
    its fill intact; the shorter one has the longer one's area subtracted from
    it. If the subtraction splits the shorter polygon into two stubs (top and
    bottom at a crossing), both stubs are kept as separate repair patches
    provided each is >= min_fragment pixels. This produces 3 non-overlapping
    repair regions instead of one oversized merged blob.

    Overlap gate: skip if overlap < 3% of the smaller polygon's area. This
    catches genuine crossings while ignoring polygons that barely graze at tile
    seams. The ratio is resolution-independent.

    Angle gate: skip if the two polygons run at nearly the same angle (< 25
    degree difference). Same-trail seam overlaps at tile boundaries have nearly
    identical angles; genuine crossing trails differ by 25+ degrees.

    Gap creation: the winner fill is dilated 1px before subtraction, so each
    surviving stub ends 2px away from the winner's actual boundary. Combined
    with the 1px gap from the dilation, the stubs and winner form separate
    connected components in the final mask.

    poly_fills  : list of H×W uint8 masks, one per group
    poly_lengths: list of float diagonal span values, one per group
    min_fragment: keep a clipped stub only if its area >= this
    """
    _k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = [f.copy() for f in poly_fills]
    n = len(result)

    # Precompute per-fill bbox and area once rather than inside the O(N²) loop.
    bboxes = []
    areas = []
    for f in result:
        ys, xs = np.where(f > 0)
        if len(ys) == 0:
            bboxes.append(None)
            areas.append(0)
        else:
            bboxes.append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))
            areas.append(int(len(ys)))

    for i in range(n):
        if bboxes[i] is None:
            continue
        ri1, ri2, ci1, ci2 = bboxes[i]
        for j in range(i + 1, n):
            if bboxes[j] is None:
                continue
            rj1, rj2, cj1, cj2 = bboxes[j]

            # Bbox intersection check -- skips the vast majority of pairs cheaply.
            ir1 = max(ri1, rj1); ir2 = min(ri2, rj2)
            ic1 = max(ci1, cj1); ic2 = min(ci2, cj2)
            if ir1 > ir2 or ic1 > ic2:
                continue

            # All pixel work cropped to the intersection region.
            fi_crop = result[i][ir1:ir2 + 1, ic1:ic2 + 1]
            fj_crop = result[j][ir1:ir2 + 1, ic1:ic2 + 1]
            ov = int(np.count_nonzero((fi_crop > 0) & (fj_crop > 0)))
            if ov < 0.03 * min(areas[i], areas[j]):
                continue

            # Angle check on per-fill bbox crops (PCA angle is translation-invariant).
            ang_i = _poly_angle(result[i][ri1:ri2 + 1, ci1:ci2 + 1])
            ang_j = _poly_angle(result[j][rj1:rj2 + 1, cj1:cj2 + 1])
            adiff = abs(ang_i - ang_j)
            if adiff > 90:
                adiff = 180 - adiff
            if adiff < 25:
                continue

            winner, loser = (i, j) if poly_lengths[i] >= poly_lengths[j] else (j, i)

            # Clip within loser's bbox (+ 1px padding for dilation boundary).
            rl1, rl2, cl1, cl2 = bboxes[loser]
            h_f, w_f = result[loser].shape
            rl1p = max(0, rl1 - 1); rl2p = min(h_f - 1, rl2 + 1)
            cl1p = max(0, cl1 - 1); cl2p = min(w_f - 1, cl2 + 1)
            loser_crop  = result[loser ][rl1p:rl2p + 1, cl1p:cl2p + 1]
            winner_crop = result[winner][rl1p:rl2p + 1, cl1p:cl2p + 1]
            winner_exp  = cv2.dilate((winner_crop > 0).astype(np.uint8), _k)
            clipped = (loser_crop > 0) & ~winner_exp.astype(bool)
            nc, lbl = cv2.connectedComponents(clipped.astype(np.uint8))
            rebuilt = np.zeros_like(loser_crop)
            for ci in range(1, nc):
                if int((lbl == ci).sum()) >= min_fragment:
                    rebuilt[lbl == ci] = 255
            result[loser][rl1p:rl2p + 1, cl1p:cl2p + 1] = rebuilt

            # Refresh loser bbox/area after modification.
            ys, xs = np.where(result[loser] > 0)
            if len(ys) == 0:
                bboxes[loser] = None
                areas[loser]  = 0
            else:
                bboxes[loser] = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
                areas[loser]  = int(len(ys))

    return result


def _fill_tips(fill_mask):
    """Measure a fitted polygon's two endpoints, direction, and width.

    fill_mask: an H x W mask, nonzero where the fitted polygon is.

    Returns (tip_min_rc, tip_max_rc, u, width_px), or None if the polygon is too
    small to fit (fewer than 4 pixels):
      - tip_min_rc / tip_max_rc: the polygon's two far ends as (row, col) points.
      - u: the long-axis direction as a (row, col) unit vector.
      - width_px: how wide the polygon is across its short axis.

    Same PCA-on-a-blob approach as _group_tips, but it works from the rendered
    polygon fill instead of raw detection coordinates. _link_polygon_gaps uses it
    so its geometry is computed from the final polygons and is therefore stable
    run-to-run, unlike the raw-group version which varies with MPS noise.
    """
    ys, xs = np.where(fill_mask > 0)
    if len(xs) < 4:
        return None
    coords = np.stack([ys, xs], axis=1).astype(float)
    centroid = coords.mean(axis=0)
    # Covariance of the pixel cloud → eigh gives the long and short axes.
    dr = coords[:, 0] - centroid[0]
    dc = coords[:, 1] - centroid[1]
    Irr = float((dr * dr).mean())
    Icc = float((dc * dc).mean())
    Irc = float((dr * dc).mean())
    _, evecs = np.linalg.eigh(np.array([[Irr, Irc], [Irc, Icc]]))
    u = evecs[:, 1]  # major axis in (row, col) space
    # Project pixels onto the long axis: min/max projections mark the two tips.
    t = coords @ u
    t_c = float(centroid @ u)
    tip_min = centroid + (float(t.min()) - t_c) * u
    tip_max = centroid + (float(t.max()) - t_c) * u
    # Project onto the perpendicular axis to measure the polygon's width.
    perp = np.array([-u[1], u[0]])
    p = coords @ perp
    width_px = float(p.max() - p.min())
    return tip_min, tip_max, u, width_px


def _link_polygon_gaps(final, poly_fills, polygon_segs_out, polygon_corners_out,
                       h, w, tile_size):
    """Fill gaps between collinear polygon pairs that the group-level bridge missed.

    Operates on fitted polygon fills rather than raw detection groups, so the
    geometry is deterministic regardless of MPS run-to-run variation. Scans
    every pair of fills for: similar trail angle, a gap in the NMS-seam range,
    and tip-to-tip vector aligned with the trail direction. When all three pass,
    a connecting rectangle is filled between the two nearest tips, sized to the
    narrower of the two polygons.

    The minimum gap is a fraction of tile_size so it scales with the inference
    grid rather than frame resolution. The maximum gap reuses _BRIDGE_MAX_GAP
    because NMS seam gaps are bounded by tile geometry, not frame size.
    """
    min_gap = 0.05 * tile_size
    n = len(poly_fills)
    # Precompute tips once per fill (O(N)) rather than per pair (O(N²)).
    all_tips = [_fill_tips(poly_fills[k]) for k in range(n)]
    for i in range(n):
        ti = all_tips[i]
        if ti is None:
            continue
        tip_i_min, tip_i_max, u_i, w_i = ti
        for j in range(i + 1, n):
            tj = all_tips[j]
            if tj is None:
                continue
            tip_j_min, tip_j_max, u_j, w_j = tj

            # Gate 1: angle
            cos_sim = min(abs(float(np.dot(u_i, u_j))), 1.0)
            adiff = float(np.degrees(np.arccos(cos_sim)))
            if adiff > _BRIDGE_MAX_ANGLE:
                continue

            # Nearest tips
            best_dist, best_a, best_b = float("inf"), None, None
            for ta, tb in [(tip_i_min, tip_j_min), (tip_i_min, tip_j_max),
                           (tip_i_max, tip_j_min), (tip_i_max, tip_j_max)]:
                d = float(np.linalg.norm(ta - tb))
                if d < best_dist:
                    best_dist, best_a, best_b = d, ta, tb

            # Gate 2: gap range (bounded by tile geometry, not frame size)
            if best_dist < min_gap or best_dist > _BRIDGE_MAX_GAP:
                continue

            # Gate 3: tip-direction alignment
            tip_vec = best_b - best_a
            tip_unit = tip_vec / float(np.linalg.norm(tip_vec))
            u_j_oriented = u_j if float(np.dot(u_i, u_j)) >= 0 else -u_j
            u_avg = u_i + u_j_oriented
            u_avg_norm = float(np.linalg.norm(u_avg))
            if u_avg_norm < 1e-6:
                continue
            u_avg /= u_avg_norm
            tip_cos = min(abs(float(np.dot(tip_unit, u_avg))), 1.0)
            if float(np.degrees(np.arccos(tip_cos))) > _BRIDGE_TIP_ANGLE:
                continue

            # Connecting rectangle between nearest tips, width = narrower polygon
            gap_perp = np.array([-tip_unit[1], tip_unit[0]])
            half_w = min(w_i, w_j) / 2.0
            corners_rc = [
                best_a + half_w * gap_perp,
                best_a - half_w * gap_perp,
                best_b - half_w * gap_perp,
                best_b + half_w * gap_perp,
            ]
            # fillPoly and polygon_corners_out both use (x, y) = (col, row)
            corners_xy = [[int(round(c[1])), int(round(c[0]))] for c in corners_rc]
            pts = np.array(corners_xy, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(final, [pts], 255)
            if polygon_segs_out is not None:
                seg = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(seg, [pts], 255)
                polygon_segs_out.append(seg)
            if polygon_corners_out is not None:
                polygon_corners_out.append(corners_xy)


#   SAHI tiled inference → crossing splitter → elongation filter
#   → union-find grouper → targeted tip-clip pass → polygon fitting → dilation.
# sky_mask and small-component filter are applied by the caller (astro_clean_v5.py)
# after this returns, so final_trail_components in the log reflects those too.
# Log fields populated via debug_out: sahi_raw_count, grouper (per-stage counts),
# group_count, polygon_count, dilate_px.

def detect_frame_polygon(model, image, tile_size: int = 640,
                         overlap: float = 0.2, dilate: int = 1,
                         return_raw: bool = False, debug_out=None,
                         edge_candidates_out=None, sky_mask=None,
                         timing_out=None, fg_mask=None,
                         polygon_segs_out=None, polygon_corners_out=None):
    """Like detect_frame, but returns a tight fitted-polygon mask.

    Runs the same SAHI inference, then:
      crossing splitter → elongation filter → group collinear tile detections
      → fit one tight rectangle per trail group → fill as binary mask.

    Same output format as detect_frame. Uses identical code to polymakr so
    the repair mask and the CVAT annotation polygon are the same shape.

    sky_mask (np.ndarray, optional): 255=sky (keep), 0=foreground (zero out).
    Applied to raw SAHI predictions before any other processing so the
    elongation filter, grouper, and gap-bridge never see foreground pixels.

    If return_raw=True, returns (final_mask, raw_labeled_mask) where
    raw_labeled_mask has pixel value = SAHI prediction index (1..N), 0=background.
    raw_labeled reflects post-sky-mask predictions.

    edge_candidates_out (list, optional): if provided, any detections that
    failed the elongation filter but whose bbox touches a frame edge (within
    5px) are appended here as {"mask", "u", "bbox"} dicts. The pipeline
    rescue pass reinstates them if 2+ neighboring frames confirm the slope.

    debug_out (dict, optional): populated with detection stage counts for the
    run log. Pass an empty dict; it is filled before this function returns.
    """
    loaded = _load_as_rgb(image)
    if loaded is None:
        if return_raw:
            return None, None
        return None
    img, h, w = loaded

    if timing_out is not None:
        timing_out.update({"sahi_s": 0.0, "sky_mask_s": 0.0, "raw_labeled_s": 0.0,
                           "try_split_s": 0.0, "elongation_s": 0.0, "edge_cand_s": 0.0,
                           "group_s": 0.0, "bridge_s": 0.0, "bridge_rot90_s": 0.0,
                           "poly_fit_s": 0.0, "dilate_s": 0.0, "gap_pairs_count": 0,
                           "gap_bridge_merges": 0})

    # ── SAHI tiled inference ──────────────────────────────────────────────────
    _t0 = time.perf_counter()
    _tiles_skipped = 0
    if fg_mask is not None:
        predictions, _tiles_skipped = _sahi_predict_skip(model, img, tile_size, overlap, fg_mask)
    else:
        predictions = _sahi_predict(model, img, tile_size, overlap)
    if timing_out is not None:
        timing_out["sahi_s"] = time.perf_counter() - _t0

    # Zero foreground pixels from every SAHI prediction before any other step.
    # Grouper, gap-bridge, and polygon fitter never see foreground pixels.
    _t0 = time.perf_counter()
    if sky_mask is not None:
        _sm_bool = sky_mask > 0
        for _pred in predictions:
            if _pred.mask is None or _pred.mask.bool_mask is None:
                continue
            _bm = _pred.mask.bool_mask
            if _bm.shape == (h, w):
                _bm[~_sm_bool] = False
            else:
                _sm_r = cv2.resize(sky_mask, (_bm.shape[1], _bm.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0
                _bm[~_sm_r] = False
    if timing_out is not None:
        timing_out["sky_mask_s"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    raw_labeled = _build_raw_labeled_mask(predictions, h, w) if return_raw else None
    if timing_out is not None:
        timing_out["raw_labeled_s"] = time.perf_counter() - _t0

    # ── Crossing splitter + elongation filter ─────────────────────────────────
    # Returns normal passing masks, their props for the grouper, and any
    # edge-touching components that failed elongation (rescue candidates).
    grouper_dbg = {} if debug_out is not None else None
    _, det_list, edge_cands = filter_masks_with_props(predictions, h, w, img=img,
                                                      debug_out=grouper_dbg,
                                                      timing_out=timing_out)
    if edge_candidates_out is not None:
        edge_candidates_out.extend(edge_cands)

    if debug_out is not None:
        debug_out["sahi_raw_count"] = len(predictions)
        debug_out["sahi_tiles_skipped"] = _tiles_skipped
        debug_out["grouper"] = grouper_dbg

    # ── Union-find grouper + targeted tip-clip pass + polygon fitting ─────────
    if not det_list:
        final = np.zeros((h, w), dtype=np.uint8)
        if debug_out is not None:
            debug_out.update({"group_count": 0, "polygon_count": 0,
                              "gap_bridge_tiles": 0, "gap_bridge_new_dets": 0})
    else:
        _t0 = time.perf_counter()
        groups = group_detections(det_list)
        if timing_out is not None:
            timing_out["group_s"] = time.perf_counter() - _t0

        groups = _split_disconnected_groups(groups, det_list)

        # Check every pair of groups for the seam-gap signature. If two groups
        # look like the same trail but their tips are farther apart than the
        # grouper's gate allows, run one extra tile in the gap between them.
        _t0 = time.perf_counter()
        gap_pairs = _find_gap_bridge_tiles(groups, det_list, h, w, tile_size)
        bridge_tiles = len(gap_pairs)
        if timing_out is not None:
            timing_out["gap_pairs_count"] = bridge_tiles
        bridge_new_dets = 0
        bridge_merges = 0
        bridge_rot90_merges = 0
        if gap_pairs:
            # Union-find: merge every group pair the gap-bridge pass approved into
            # a single combined group. gp[x] is x's parent; _gf walks to the root.
            gp = list(range(len(groups)))

            def _gf(x):
                # Find the root of x, flattening the chain as we go (path halving).
                while gp[x] != x:
                    gp[x] = gp[gp[x]]
                    x = gp[x]
                return x

            for gi, gj, tx, ty in gap_pairs:
                if _gf(gi) == _gf(gj):
                    continue
                gp[_gf(gi)] = _gf(gj)
                bridge_merges += 1

            # Collect each set of detection indices under its root into one group.
            merged = {}
            for gi, grp in enumerate(groups):
                merged.setdefault(_gf(gi), []).extend(grp)
            groups = list(merged.values())

        if timing_out is not None:
            timing_out["bridge_s"] = time.perf_counter() - _t0
            timing_out["gap_bridge_merges"] = bridge_merges
        if debug_out is not None:
            debug_out["gap_bridge_tiles"] = bridge_tiles
            debug_out["gap_bridge_new_dets"] = bridge_new_dets
            debug_out["gap_bridge_merges"] = bridge_merges
            debug_out["gap_bridge_rot90_merges"] = bridge_rot90_merges

        _t0 = time.perf_counter()
        final = np.zeros((h, w), dtype=np.uint8)
        poly_count = 0
        poly_fills = []
        poly_lengths = []
        poly_corner_sets = []
        for grp in groups:
            all_coords = np.vstack([det_list[i]["coords"] for i in grp])
            x_span = int(all_coords[:,1].max() - all_coords[:,1].min())
            # A trail that is both wide enough and bends enough is treated as
            # curved and fitted with a chain of short rectangles; everything else
            # gets a single straight rectangle. Thresholds come from trail_grouper.
            if (x_span >= _CURVED_MIN_XSPAN
                    and _group_angle_spread(grp, det_list) >= _CURVED_MIN_ANGLE_SPREAD):
                corner_sets = fit_curved_group(grp, det_list)
            else:
                corners, _, _ = fit_polygon(grp, det_list)
                corner_sets = [corners]
            grp_fill = np.zeros((h, w), dtype=np.uint8)
            for corners in corner_sets:
                pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(grp_fill, [pts], 255)
                if polygon_segs_out is not None:
                    seg = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(seg, [pts], 255)
                    polygon_segs_out.append(seg)
                if polygon_corners_out is not None:
                    polygon_corners_out.append([[int(c[0]), int(c[1])] for c in corners])
            row_span = float(all_coords[:, 0].max() - all_coords[:, 0].min())
            col_span = float(all_coords[:, 1].max() - all_coords[:, 1].min())
            poly_fills.append(grp_fill)
            poly_lengths.append(math.sqrt(row_span ** 2 + col_span ** 2))
            poly_corner_sets.append(corner_sets)
            poly_count += len(corner_sets)
        # Scale the minimum-fragment size by frame area relative to a reference
        # frame, so the "keep a clipped stub?" threshold means the same physical
        # size on a small frame as on a 24MP one.
        area_scale = (h * w) / _REF_FRAME_PX
        poly_fills = _clip_overlapping_polygons(
            poly_fills, poly_lengths,
            min_fragment=int((MIN_AREA // 2) * area_scale),
        )
        for grp_fill in poly_fills:
            final |= grp_fill

        # Fallback: filtered detections where less than 70% of their pixel
        # coords are covered by the polygon fills get their own single-detection
        # rotated rectangle added to the mask.
        for i, det in enumerate(det_list):
            c = det["coords"]
            covered = int(np.count_nonzero(final[c[:, 0], c[:, 1]]))
            if covered / len(c) < 0.70:
                corners, _, _ = fit_polygon([i], det_list)
                pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(final, [pts], 255)
                if polygon_segs_out is not None:
                    seg = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(seg, [pts], 255)
                    polygon_segs_out.append(seg)
                if polygon_corners_out is not None:
                    polygon_corners_out.append([[int(corner[0]), int(corner[1])] for corner in corners])

        _link_polygon_gaps(final, poly_fills, polygon_segs_out, polygon_corners_out,
                           h, w, tile_size)

        if timing_out is not None:
            timing_out["poly_fit_s"] = time.perf_counter() - _t0
        if debug_out is not None:
            poly_confs = [round(max((det_list[i].get("conf", 0.0) for i in grp), default=0.0), 3)
                          for grp in groups]
            debug_out.update({"group_count": len(groups),
                              "polygon_count": poly_count,
                              "polygon_confidences": poly_confs})

    # ── Dilation ──────────────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    if dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        final = cv2.dilate(final, kernel)
    if timing_out is not None:
        timing_out["dilate_s"] = time.perf_counter() - _t0

    if debug_out is not None:
        debug_out["dilate_px"] = dilate

    if return_raw:
        return final, raw_labeled
    return final


# ── Post-detection filters (called by astro_clean_v5.py after detection) ─────
# apply_sky_mask: zeros pixels that fall in the foreground region.
#   sky_mask = bitwise_not(foreground_mask): 255=sky (keep), 0=foreground (zero).
#   Log field: detect record sky_mask_pixels_removed = pixels zeroed by this step.
# filter_small_components: removes components below min_area unless they are red
#   (airplane navigation light). Red check uses top-90th-percentile R channel.
#   Log fields: detect record small_filter.removed, kept_as_nav_light, removed_areas.

def apply_sky_mask(mask: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """Zero out mask pixels outside the sky region.

    sky_mask: 255=sky (keep), 0=foreground (zero out).
    """
    if sky_mask.shape != mask.shape:
        sky_mask = cv2.resize(sky_mask, (mask.shape[1], mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(mask, sky_mask)


def filter_small_components(mask: np.ndarray, img: np.ndarray,
                            min_area: int = 1000,
                            debug_out=None) -> np.ndarray:
    """Remove connected components smaller than min_area, unless red (nav light).

    img is BGR (as loaded by cv2). Channel 2 = R.
    debug_out (dict, optional): filled with removed count, kept_as_nav_light count,
    and removed_areas list (one int per removed component).
    """
    if img.shape[:2] != mask.shape[:2]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    if debug_out is not None:
        debug_out.update({"removed": 0, "kept_as_nav_light": 0, "removed_areas": []})
    out = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            continue
        component_mask = (labels == i)
        pixels = img[component_mask]
        r_vals = pixels[:, 2].astype(float)
        top_mask = r_vals >= np.percentile(r_vals, 90)
        top_r = float(r_vals[top_mask].mean())
        top_g = float(pixels[top_mask, 1].mean())
        top_b = float(pixels[top_mask, 0].mean())
        is_red = top_r > 80 and top_r > top_g * 1.4 and top_r > top_b * 1.4
        if not is_red:
            out[component_mask] = 0
            if debug_out is not None:
                debug_out["removed"] += 1
                debug_out["removed_areas"].append(int(area))
        elif debug_out is not None:
            debug_out["kept_as_nav_light"] += 1
    return out
