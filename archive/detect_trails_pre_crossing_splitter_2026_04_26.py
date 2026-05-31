"""YOLO/SAHI trail detection + mask post-processing."""
import cv2
import numpy as np
from typing import Optional

from . import slope_match

# Default ON. Toggle to False to disable post-NMS slope-match merging if a
# tester reports a regression. See modules/slope_match.py for the algorithm.
SLOPE_MATCH_ENABLED = True

# Hybrid axis-extension recovers trail content that SAHI's NMS step truncates
# at tile boundaries. Runs both NMS and NMM postprocess on a single SAHI
# inference pass, then uses the NMM result as evidence to extend the NMS mask
# along the trail's principal axis. Same per-frame inference cost as NMS-only
# (postprocess merges are cheap CPU operations).
HYBRID_AXIS_EXTEND_ENABLED = True

# Shape sanity + brightness-trim filter: applied after detection to drop
# bloated rectangular blobs that aren't really trails. See
# runs/experiments/2026_04_26_shape_sanity/ for derivation.
SHAPE_SANITY_ENABLED = True

_SANITY_AREA_MIN = 1500          # below this, no sanity check (pass automatically)
_SANITY_ASPECT_MIN = 5.0         # aspect ratio >= this passes
_SANITY_BRIGHT_FRAC_MIN = 0.03   # bright_frac >= this passes
_SANITY_BRIGHT_THRESHOLD = 60    # gray > this counts as a "bright" pixel
_TRIM_BBOX_EXPAND = 1.5          # 50% bbox expand to gather local sky baseline
_TRIM_SIGMA = 2.0                # keep pixels brighter than baseline + sigma * stdev
_TRIM_DILATE = 3                 # halo dilation after trim (px)
_TRIM_MIN_KEEP = 30              # if trim leaves <N px, drop the component
_TRIM_MIN_BLOB = 50              # drop post-trim isolated blobs smaller than this (stars)


def best_device() -> str:
    """Return the best available inference device: cuda > mps > cpu.

    Checks only what's actually usable RIGHT NOW on this machine. The CUDA
    check requires a CUDA-enabled PyTorch AND a working NVIDIA driver, so
    a user on Windows CPU-only, Intel Mac, or Apple Silicon with MPS each
    falls to the right branch.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model(model_path: str, confidence: float = 0.25,
               device: Optional[str] = None):
    """Load YOLOv8-seg model via SAHI AutoDetectionModel.

    device=None (or "auto") picks the best available: cuda > mps > cpu.
    Pass an explicit string to override.
    """
    if not device or device == "auto":
        device = best_device()
    from sahi import AutoDetectionModel
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=confidence,
        device=device,
    )
    return model


def _build_combined_mask(predictions, h, w, apply_slope_match):
    """Convert SAHI predictions into one full-frame uint8 mask."""
    per_prediction_masks = []
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
            m = cv2.resize(seg.astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST)
        if m.any():
            per_prediction_masks.append(m)

    if apply_slope_match and len(per_prediction_masks) >= 2:
        per_prediction_masks = slope_match.merge(per_prediction_masks)

    out = np.zeros((h, w), dtype=np.uint8)
    for m in per_prediction_masks:
        out = np.maximum(out, m)
    return out


def _any_truncation_at_tile_boundary(mask: np.ndarray, tile_size: int,
                                     overlap: float, edge_tol: int = 8) -> bool:
    """True if any connected component's bbox sits within edge_tol pixels of
    an inner tile boundary. Used as a cheap gate to skip the second SAHI pass
    on frames that have no truncation candidates — most frames don't.
    """
    if not mask.any():
        return False
    h, w = mask.shape
    stride = int(tile_size * (1 - overlap))

    def origins(extent):
        xs = list(range(0, extent - tile_size, stride))
        if not xs or xs[-1] + tile_size < extent:
            xs.append(extent - tile_size)
        return xs

    tile_x = origins(w)
    tile_y = origins(h)
    inner_x = sorted(set(tile_x[1:] + [x + tile_size for x in tile_x if x + tile_size < w]))
    inner_y = sorted(set(tile_y[1:] + [y + tile_size for y in tile_y if y + tile_size < h]))
    if not inner_x and not inner_y:
        return False

    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n_lab):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        x2 = x + int(stats[i, cv2.CC_STAT_WIDTH])
        y2 = y + int(stats[i, cv2.CC_STAT_HEIGHT])
        for b in inner_x:
            if abs(x - b) <= edge_tol or abs(x2 - b) <= edge_tol:
                return True
        for b in inner_y:
            if abs(y - b) <= edge_tol or abs(y2 - b) <= edge_tol:
                return True
    return False


_HYBRID_EDGE_TOL = 8
_HYBRID_SIZE_RATIO_CAP = 3.0
_HYBRID_WIDTH_FACTOR = 0.9


def _inner_tile_seams(extent: int, tile_size: int, overlap: float) -> list:
    """Pixel coordinates of inner tile boundaries along one axis."""
    stride = max(1, int(tile_size * (1 - overlap)))
    starts = list(range(0, max(0, extent - tile_size), stride))
    if not starts or starts[-1] + tile_size < extent:
        starts.append(max(0, extent - tile_size))
    inner = set()
    for s in starts[1:]:
        inner.add(s)
    for s in starts:
        e = s + tile_size
        if 0 < e < extent:
            inner.add(e)
    return sorted(inner)


def _hybrid_axis_extend(nms_mask: np.ndarray,
                        nmm_mask: np.ndarray,
                        tile_size: int,
                        overlap: float) -> np.ndarray:
    """Recover trail content NMS truncated at tile boundaries.

    For each connected blob present in the NMM mask but not the NMS mask
    that touches an NMS component, treat the NMM blob as evidence that the
    trail extends in the NMS component's principal-axis direction. Synthesize
    a thin extension along the NMS centerline using the NMS component's
    average perpendicular width — never the NMM blob's actual shape, since
    NMM tends to bloat perpendicular to the trail.

    Three guards (hybrid v6):
      1. Tile-edge guard: only extend if the NMS component touches an inner
         tile seam (evidence of true truncation).
      2. Size-ratio guard: skip when the NMM-only blob is more than
         _HYBRID_SIZE_RATIO_CAP times the NMS component's area.
      3. Width tightening: half-width × _HYBRID_WIDTH_FACTOR (0.9), drawn at
         a fixed 1-pixel stroke instead of a 60% padded line.

    See runs/active/big_box_2026_04_25/ for the v3..v6 experiment artifacts.
    """
    nmm_only = (nmm_mask > 0) & (nms_mask == 0)
    out = nms_mask.copy()
    if not nmm_only.any():
        return out

    h, w = nms_mask.shape[:2]
    inner_x = _inner_tile_seams(w, tile_size, overlap)
    inner_y = _inner_tile_seams(h, tile_size, overlap)

    def near_inner(c, bounds):
        return any(abs(c - b) <= _HYBRID_EDGE_TOL for b in bounds)

    n_nms, nms_labels, nms_stats, _ = cv2.connectedComponentsWithStats(nms_mask)
    n_only, only_labels, _, _ = cv2.connectedComponentsWithStats(nmm_only.astype(np.uint8))
    if n_nms < 2 or n_only < 2:
        return out

    # Per-NMS-component truncation: which sides touch an inner tile seam?
    nms_truncated = {}
    for i in range(1, n_nms):
        x = nms_stats[i, cv2.CC_STAT_LEFT]
        y = nms_stats[i, cv2.CC_STAT_TOP]
        x2 = x + nms_stats[i, cv2.CC_STAT_WIDTH]
        y2 = y + nms_stats[i, cv2.CC_STAT_HEIGHT]
        nms_truncated[i] = (near_inner(x, inner_x) or near_inner(x2, inner_x) or
                            near_inner(y, inner_y) or near_inner(y2, inner_y))

    dilate_kernel = np.ones((5, 5), np.uint8)
    for j in range(1, n_only):
        only_blob = (only_labels == j)
        only_area = int(only_blob.sum())
        if only_area < 30:
            continue
        only_dilated = cv2.dilate(only_blob.astype(np.uint8), dilate_kernel)
        touched = [t for t in np.unique(nms_labels[only_dilated > 0]) if t != 0]
        if not touched:
            continue
        best = max(touched, key=lambda t:
                   int(((nms_labels == t) & (only_dilated > 0)).sum()))
        # Tile-edge guard: skip if the NMS component is not actually truncated
        if not nms_truncated.get(best, False):
            continue
        # Size-ratio guard: skip if the NMM-only blob is much larger than NMS
        nms_area = int(nms_stats[best, cv2.CC_STAT_AREA])
        if nms_area == 0 or only_area > _HYBRID_SIZE_RATIO_CAP * nms_area:
            continue
        nms_blob = (nms_labels == best)
        ys_n, xs_n = np.where(nms_blob)
        if len(xs_n) < 5:
            continue
        pts_n = np.column_stack([xs_n, ys_n]).astype(np.float32)
        mean_n = pts_n.mean(axis=0)
        cov = np.cov(pts_n.T)
        try:
            _, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        axis = eigvecs[:, -1]
        perp = eigvecs[:, 0]
        proj_n_axis = (pts_n - mean_n) @ axis
        proj_n_perp = (pts_n - mean_n) @ perp
        n_min, n_max = float(proj_n_axis.min()), float(proj_n_axis.max())
        # Tighter perpendicular width: avg width × 0.9
        half_w = max(2.0, 2.0 * float(np.abs(proj_n_perp).mean()) * _HYBRID_WIDTH_FACTOR)
        ys_o, xs_o = np.where(only_blob)
        pts_o = np.column_stack([xs_o, ys_o]).astype(np.float32)
        proj_o = (pts_o - mean_n) @ axis
        ext_max = max(n_max, float(proj_o.max()))
        ext_min = min(n_min, float(proj_o.min()))
        for s in np.arange(ext_min, ext_max + 0.5, 0.5):
            if n_min <= s <= n_max:
                continue
            cx, cy = mean_n + s * axis
            p1 = (int(round(cx - half_w * perp[0])),
                  int(round(cy - half_w * perp[1])))
            p2 = (int(round(cx + half_w * perp[0])),
                  int(round(cy + half_w * perp[1])))
            cv2.line(out, p1, p2, 255, thickness=1)
    return out


def _sliced_inference_dual_postprocess(model, img_path: str,
                                       tile_size: int, overlap: float):
    """Run SAHI sliced inference once, then apply BOTH NMS and NMM postprocess
    on the same raw per-tile prediction list. Returns (nms_predictions,
    nmm_predictions). This avoids the 2x inference cost of calling
    get_sliced_prediction twice — the expensive step is the per-tile model
    forward pass, postprocess merges are cheap CPU operations.
    """
    import copy
    from sahi.slicing import slice_image
    from sahi.predict import get_prediction, POSTPROCESS_NAME_TO_CLASS

    slice_result = slice_image(
        image=img_path,
        slice_height=tile_size, slice_width=tile_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        auto_slice_resolution=True,
    )
    full_shape = [slice_result.original_image_height,
                  slice_result.original_image_width]

    raw_predictions = []
    for i in range(len(slice_result)):
        result = get_prediction(
            image=slice_result.images[i],
            detection_model=model,
            shift_amount=slice_result.starting_pixels[i],
            full_shape=full_shape,
        )
        for op in result.object_prediction_list:
            if op:
                raw_predictions.append(op.get_shifted_object_prediction())

    # Standard prediction on full image (matches get_sliced_prediction default)
    if len(slice_result) > 1:
        std = get_prediction(
            image=img_path, detection_model=model,
            shift_amount=[0, 0], full_shape=full_shape, postprocess=None,
        )
        raw_predictions.extend(std.object_prediction_list)

    if len(raw_predictions) <= 1:
        return raw_predictions, list(raw_predictions)

    nms_pp = POSTPROCESS_NAME_TO_CLASS["NMS"](
        match_threshold=0.5, match_metric="IOS", class_agnostic=False)
    nmm_pp = POSTPROCESS_NAME_TO_CLASS["NMM"](
        match_threshold=0.5, match_metric="IOS", class_agnostic=False)

    # Postprocess may mutate the list it receives — give each a fresh copy
    nms_predictions = nms_pp(copy.copy(raw_predictions))
    nmm_predictions = nmm_pp(copy.copy(raw_predictions))
    return nms_predictions, nmm_predictions


def detect_frame(model, img_path: str, tile_size: int = 640,
                 overlap: float = 0.2, dilate: int = 1) -> Optional[np.ndarray]:
    """Run SAHI tiled inference on one frame.

    Returns binary uint8 mask (255=trail, 0=sky) at original resolution,
    or None if image cannot be read.

    When HYBRID_AXIS_EXTEND_ENABLED is True, applies BOTH NMS and NMM
    postprocess on the same single SAHI inference pass and extends the NMS
    mask along trail axes where NMS truncated content at tile boundaries.
    Same per-frame inference cost as plain NMS (postprocess is cheap).
    Recovers ~95% of right-edge truncation cases (verified 2026-04-26).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if HYBRID_AXIS_EXTEND_ENABLED:
        nms_preds, nmm_preds = _sliced_inference_dual_postprocess(
            model, img_path, tile_size, overlap)
        mask = _build_combined_mask(nms_preds, h, w,
                                    apply_slope_match=SLOPE_MATCH_ENABLED)
        nmm_mask = _build_combined_mask(nmm_preds, h, w,
                                        apply_slope_match=False)
        mask = _hybrid_axis_extend(mask, nmm_mask, tile_size, overlap)
    else:
        from sahi.predict import get_sliced_prediction
        nms_result = get_sliced_prediction(
            image=str(img_path), detection_model=model,
            slice_height=tile_size, slice_width=tile_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type="NMS", verbose=0,
        )
        mask = _build_combined_mask(nms_result.object_prediction_list, h, w,
                                    apply_slope_match=SLOPE_MATCH_ENABLED)

    if dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    if SHAPE_SANITY_ENABLED:
        mask = _shape_sanity_filter(mask, img)

    return mask


def _shape_stats(comp_mask: np.ndarray, gray: np.ndarray) -> Optional[dict]:
    ys, xs = np.where(comp_mask)
    if len(xs) < 5:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    try:
        cov = np.cov(pts.T)
        eigvals, _ = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    aspect = np.sqrt(eigvals[-1]) / max(0.5, np.sqrt(eigvals[0]))
    area = int(comp_mask.sum())
    bright = int((gray[comp_mask] > _SANITY_BRIGHT_THRESHOLD).sum())
    return {"area": area, "aspect": float(aspect),
            "bright_frac": bright / max(1, area)}


def _passes_sanity(s: dict) -> bool:
    if s["area"] < _SANITY_AREA_MIN:
        return True
    if s["aspect"] >= _SANITY_ASPECT_MIN:
        return True
    if s["bright_frac"] >= _SANITY_BRIGHT_FRAC_MIN:
        return True
    return False


def _brightness_trim(comp_mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Trim a failed component to pixels above a local sky baseline."""
    H, W = comp_mask.shape
    ys, xs = np.where(comp_mask)
    bx0, bx1 = int(xs.min()), int(xs.max())
    by0, by1 = int(ys.min()), int(ys.max())
    bw, bh = bx1 - bx0, by1 - by0
    expand_x = int(bw * (_TRIM_BBOX_EXPAND - 1) / 2)
    expand_y = int(bh * (_TRIM_BBOX_EXPAND - 1) / 2)
    rx0 = max(0, bx0 - expand_x); rx1 = min(W, bx1 + expand_x + 1)
    ry0 = max(0, by0 - expand_y); ry1 = min(H, by1 + expand_y + 1)
    region = gray[ry0:ry1, rx0:rx1]
    region_mask = comp_mask[ry0:ry1, rx0:rx1]
    sky_pixels = region[~region_mask]
    if len(sky_pixels) < 100:
        return comp_mask
    baseline = float(np.median(sky_pixels))
    mad = float(np.median(np.abs(sky_pixels - baseline)))
    threshold = baseline + _TRIM_SIGMA * 1.4826 * mad
    trimmed = comp_mask & (gray > threshold)

    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(trimmed.astype(np.uint8))
    cleaned = np.zeros_like(trimmed)
    for k in range(1, n_lab):
        if stats[k, cv2.CC_STAT_AREA] >= _TRIM_MIN_BLOB:
            cleaned |= (lab == k)
    if _TRIM_DILATE > 0 and cleaned.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_TRIM_DILATE * 2 + 1, _TRIM_DILATE * 2 + 1))
        dilated = cv2.dilate(cleaned.astype(np.uint8), kernel).astype(bool)
        cleaned = dilated & comp_mask
    return cleaned


def _shape_sanity_filter(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Drop or trim mask components that don't have a trail-like shape.

    Each connected component is shape-checked. Small components pass
    automatically. Larger components must look elongated (aspect >= 5) or
    have visible bright pixels (>= 3% of area) to pass. Failures are
    trimmed to their pixels above a local sky baseline; if the trim leaves
    too few pixels, the component is dropped entirely.
    """
    if not (mask > 0).any():
        return mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_lab, labels, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out = np.zeros_like(mask)
    for i in range(1, n_lab):
        comp = (labels == i)
        s = _shape_stats(comp, gray)
        if s is None:
            continue
        if _passes_sanity(s):
            out[comp] = 255
            continue
        trimmed = _brightness_trim(comp, gray)
        if int(trimmed.sum()) < _TRIM_MIN_KEEP:
            continue
        out[trimmed] = 255
    return out


def apply_sky_mask(mask: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """Zero out mask pixels outside the sky region.

    sky_mask: 255=sky (keep), 0=foreground (zero out).
    """
    if sky_mask.shape != mask.shape:
        sky_mask = cv2.resize(sky_mask, (mask.shape[1], mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(mask, sky_mask)


def filter_small_components(mask: np.ndarray, img: np.ndarray,
                            min_area: int = 1000) -> np.ndarray:
    """Remove connected components smaller than min_area, unless red (nav light)."""
    # SAHI reads images with EXIF rotation applied; the main pipeline may
    # load with IMREAD_UNCHANGED (no rotation).  Resize img to match mask.
    if img.shape[:2] != mask.shape[:2]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    out = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            continue
        component_mask = (labels == i)
        pixels = img[component_mask]
        mean_b = float(pixels[:, 0].mean())
        mean_g = float(pixels[:, 1].mean())
        mean_r = float(pixels[:, 2].mean())
        is_red = mean_r > 80 and mean_r > mean_g * 1.5 and mean_r > mean_b * 1.5
        if not is_red:
            out[component_mask] = 0
    return out
