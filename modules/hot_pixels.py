"""Hot/dead pixel cosmetic correction via temporal consistency.

Hot pixels are stuck at the same sensor position in every frame.
Stars move between frames due to Earth's rotation.  By comparing
the same (x, y) across multiple unaligned frames, we can distinguish
hot pixels (always bright) from stars (bright in only some frames).

Hot pixels can be stuck in a single channel (R, G, or B), so each
channel is checked independently.

Dead/cold pixels are the inverse: stuck at 0 regardless of scene content.
"""
import cv2
import numpy as np
from typing import List

MIN_BRIGHTNESS = 5


def build_hot_pixel_map(frames: List[np.ndarray], threshold: float = 2.0,
                        min_fraction: float = 0.8,
                        min_channel_excess: float = 60.0) -> np.ndarray:
    """Identify hot pixels by temporal consistency across the full frame.

    A pixel is flagged when, in at least ``min_fraction`` of the frames:
      * it is brighter than its local 9x9 median by ``threshold`` x, and
      * its brightest channel exceeds the mean of the other two channels
        by at least ``min_channel_excess`` (absolute 0-255 units). Real
        stars and scene highlights are roughly neutral; a Bayer defect
        dumps one channel well above the others, even if the demosaic
        bleeds some light into the neighbors.
    """
    h, w = frames[0].shape[:2]
    n = len(frames)
    min_hits = int(n * min_fraction)

    hit_b = np.zeros((h, w), dtype=np.uint16)
    hit_g = np.zeros((h, w), dtype=np.uint16)
    hit_r = np.zeros((h, w), dtype=np.uint16)
    excess_hits = np.zeros((h, w), dtype=np.uint16)

    for frame in frames:
        for ch, hit in enumerate([hit_b, hit_g, hit_r]):
            plane = frame[:, :, ch]
            med = cv2.medianBlur(plane, 9)
            med_safe = med.astype(np.float32)
            med_safe[med_safe < 1] = 1
            ratio = plane.astype(np.float32) / med_safe
            hit += ((ratio > threshold) & (plane > MIN_BRIGHTNESS)).astype(np.uint16)

        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
        chan_max = np.maximum(np.maximum(r, g), b)
        chan_min = np.minimum(np.minimum(r, g), b)
        chan_mid = r + g + b - chan_max - chan_min
        other_mean = (chan_min + chan_mid) / 2.0
        excess_hits += (chan_max - other_mean >= min_channel_excess).astype(np.uint16)

    ratio_hot = ((hit_b >= min_hits) | (hit_g >= min_hits) |
                 (hit_r >= min_hits))
    excess_hot = (excess_hits >= min_hits)
    return (ratio_hot & excess_hot).astype(np.uint8) * 255


def fix_hot_pixels(frames: List[np.ndarray], threshold: float = 2.0,
                   min_fraction: float = 0.8) -> List[np.ndarray]:
    """Detect and repair hot pixels across a batch of frames."""
    mask = build_hot_pixel_map(frames, threshold, min_fraction)
    n_defective = int((mask > 0).sum())

    if n_defective == 0:
        print("  No hot pixels detected")
        return frames

    print(f"  {n_defective} hot pixels detected")
    # Bayer demosaic spreads a single defect into a ~5px bloom and color
    # bleeds further into neighbors. Dilate generously, then use
    # Navier-Stokes inpainting so the fill comes from true uncontaminated
    # surround instead of a median that still sees halo.
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return [cv2.inpaint(f, dilated, 3, cv2.INPAINT_NS) for f in frames]
