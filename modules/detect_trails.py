"""YOLO/SAHI trail detection."""
import cv2
import numpy as np
from typing import Optional

from .io_safe import robust_imread


def best_device() -> str:
    """Return the best available inference device: cuda > mps > cpu."""
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


def detect_frame(model, image, tile_size: int = 640,
                 overlap: float = 0.2, dilate: int = 1) -> Optional[np.ndarray]:
    """Run SAHI tiled inference on one frame.

    `image` may be a numpy array (preferred) or a file path. Whatever arrives
    is normalized to 8-bit 3-channel BGR before handing to SAHI.

    Returns binary uint8 mask (255=trail, 0=sky) at original resolution,
    or None if the image cannot be read.
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
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]

    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(
        image=img, detection_model=model,
        slice_height=tile_size, slice_width=tile_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        postprocess_type="NMS", verbose=0,
    )
    mask = _build_combined_mask(result.object_prediction_list, h, w)

    if dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    return mask


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
        mean_b = float(pixels[:, 0].mean())
        mean_g = float(pixels[:, 1].mean())
        mean_r = float(pixels[:, 2].mean())
        is_red = mean_r > 80 and mean_r > mean_g * 1.5 and mean_r > mean_b * 1.5
        if not is_red:
            out[component_mask] = 0
    return out
