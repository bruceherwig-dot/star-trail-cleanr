"""YOLO/SAHI trail detection + mask post-processing."""
import cv2
import numpy as np
from typing import Optional


def load_model(model_path: str, confidence: float = 0.25,
               device: str = "mps"):
    """Load YOLOv8-seg model via SAHI AutoDetectionModel.

    Call once; pass returned object to detect_frame().
    """
    from sahi import AutoDetectionModel
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=confidence,
        device=device,
    )
    return model


def detect_frame(model, img_path: str, tile_size: int = 640,
                 overlap: float = 0.2, dilate: int = 1) -> Optional[np.ndarray]:
    """Run SAHI tiled inference on one frame.

    Returns binary uint8 mask (255=trail, 0=sky) at original resolution,
    or None if image cannot be read.
    """
    from sahi.predict import get_sliced_prediction

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    result = get_sliced_prediction(
        image=str(img_path),
        detection_model=model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMM",
        verbose=0,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    for pred in result.object_prediction_list:
        if pred.mask is None:
            # No polygon — skip. Bbox-fill would mask huge sky regions.
            continue
        seg = pred.mask.bool_mask
        if seg is None:
            continue

        if seg.shape == (h, w):
            mask[seg.astype(bool)] = 255
        else:
            seg_resized = cv2.resize(
                seg.astype(np.uint8) * 255, (w, h),
                interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, seg_resized)

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
