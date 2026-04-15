"""Phase-correlation frame alignment."""
import cv2
import numpy as np
from typing import List, Tuple


def _to_gray32(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _warp(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align_batch(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """Phase-correlate all frames to the middle frame. Stars -> fixed pixels.

    Returns (aligned_frames, star_trail_angle_deg).
    """
    ref_idx = len(frames) // 2
    ref_g = _to_gray32(frames[ref_idx])
    aligned = []
    shifts = []
    for i, f in enumerate(frames):
        if i == ref_idx:
            aligned.append(f)
            shifts.append((0.0, 0.0))
        else:
            shift, _ = cv2.phaseCorrelate(_to_gray32(f), ref_g)
            dx, dy = float(shift[0]), float(shift[1])
            aligned.append(_warp(f, dx, dy))
            shifts.append((dx, dy))
        print(f"    aligning {i+1}/{len(frames)}", flush=True)

    nz = [(dx, dy) for dx, dy in shifts if abs(dx) > 0.5 or abs(dy) > 0.5]
    if nz:
        angles = [float(np.degrees(np.arctan2(-dy, -dx))) % 180.0
                  for dx, dy in nz]
        star_angle = float(np.median(angles))
    else:
        star_angle = 0.0
    return aligned, star_angle
