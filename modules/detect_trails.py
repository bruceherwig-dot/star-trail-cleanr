"""YOLO/SAHI trail detection."""
import cv2
import numpy as np
import os
from typing import Optional

from .io_safe import robust_imread
from .trail_grouper import filter_masks, filter_masks_with_props, group_detections, fit_polygon


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


def _build_combined_mask(predictions, h, w):
    """Convert SAHI predictions into one full-frame uint8 mask."""
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
            m = cv2.resize(seg.astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST)
        if m.any():
            out = np.maximum(out, m)
    return out


def _load_as_rgb(image):
    """Load image from path or array and return (rgb_uint8, h, w), or None on failure."""
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
    """Label each SAHI prediction by index (1..N) before any filtering."""
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


def detect_frame_polygon(model, image, tile_size: int = 640,
                         overlap: float = 0.2, dilate: int = 1,
                         return_raw: bool = False):
    """Like detect_frame, but returns a tight fitted-polygon mask.

    Runs the same SAHI inference, then:
      crossing splitter → elongation filter → group collinear tile detections
      → fit one tight rectangle per trail group → fill as binary mask.

    Same output format as detect_frame. Uses identical code to polymakr so
    the repair mask and the CVAT annotation polygon are the same shape.

    If return_raw=True, returns (final_mask, raw_labeled_mask) where
    raw_labeled_mask has pixel value = SAHI prediction index (1..N), 0=background.
    """
    loaded = _load_as_rgb(image)
    if loaded is None:
        if return_raw:
            return None, None
        return None
    img, h, w = loaded

    predictions = _sahi_predict(model, img, tile_size, overlap)
    raw_labeled = _build_raw_labeled_mask(predictions, h, w) if return_raw else None

    _, det_list = filter_masks_with_props(predictions, h, w, img=img)

    if not det_list:
        final = np.zeros((h, w), dtype=np.uint8)
    else:
        groups = group_detections(det_list)
        final = np.zeros((h, w), dtype=np.uint8)
        for grp in groups:
            corners, _, _ = fit_polygon(grp, det_list)
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(final, [pts], 255)

    if dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        final = cv2.dilate(final, kernel)

    if return_raw:
        return final, raw_labeled
    return final


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
        r_vals = pixels[:, 2].astype(float)
        top_mask = r_vals >= np.percentile(r_vals, 90)
        top_r = float(r_vals[top_mask].mean())
        top_g = float(pixels[top_mask, 1].mean())
        top_b = float(pixels[top_mask, 0].mean())
        is_red = top_r > 80 and top_r > top_g * 1.4 and top_r > top_b * 1.4
        if not is_red:
            out[component_mask] = 0
    return out
