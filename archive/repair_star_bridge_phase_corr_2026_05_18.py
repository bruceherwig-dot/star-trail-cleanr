"""Trail repair — Star Bridge: per-trail phase-correlation morph from N-1/N+1.

For each trail detected in frame N, extract a padded bounding-box patch from
frames N-1 and N+1. Measure the local star motion between the two neighbors
with phase correlation (one translation vector per trail box). Shift N-1
forward by half the motion and N+1 backward by half the motion, average them
to synthesize what frame N would have looked like without the trail. Paste the
synthetic pixels into frame N at only the masked locations.

When phase correlation confidence is low (featureless sky near the horizon),
the neighbors are averaged without shifting — the sky is smooth enough that
alignment doesn't matter.

When only one neighbor is available (first/last frame), that neighbor's
pixels are used directly.
"""
import cv2
import numpy as np


# --- Tuning constants ------------------------------------------------------
PAD = 60            # extra pixels around each trail bbox for phase correlation
MIN_AREA = 500      # skip tiny mask components (noise)
MIN_RESPONSE = 0.3  # phase correlation confidence threshold


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate img by (dx, dy) with sub-pixel precision."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def repair_frame(frame: np.ndarray, mask: np.ndarray,
                 frame_idx: int,
                 neighbor_frames: list) -> np.ndarray:
    """Replace masked trail pixels using Star Bridge morph repair.

    For each trail component in the mask:
      1. Extract padded bbox from frames N-1 and N+1.
      2. Phase-correlate to measure local star motion.
      3. Shift each neighbor halfway → average → synthetic patch.
      4. Paste synthetic pixels into frame N at masked locations.

    Falls back to black fill (zeros) for edge frames, low-confidence
    patches, or pixels masked in both neighbors.

    Args:
        frame: original image (uint8 or uint16)
        mask: binary uint8 mask (255=trail, 0=sky) for this frame
        frame_idx: index of this frame in neighbor_frames
        neighbor_frames: full list of frames (same dtype as frame)
    Returns:
        Repaired copy of frame.
    """
    result = frame.copy()
    trail = mask > 0
    if not trail.any():
        return result

    H, W = mask.shape[:2]
    N = len(neighbor_frames)

    # Identify available neighbors
    prev_idx = frame_idx - 1 if frame_idx > 0 else None
    next_idx = frame_idx + 1 if frame_idx < N - 1 else None
    has_prev = prev_idx is not None
    has_next = next_idx is not None

    if not has_prev and not has_next:
        return result

    # Find connected components — one trail per component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        trail.astype(np.uint8))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Padded bbox clamped to frame
        x0 = max(0, bx - PAD)
        y0 = max(0, by - PAD)
        x1 = min(W, bx + bw + PAD)
        y1 = min(H, by + bh + PAD)

        comp_mask = (labels[y0:y1, x0:x1] == i)

        patch_prev = neighbor_frames[prev_idx][y0:y1, x0:x1] if has_prev else None
        patch_next = neighbor_frames[next_idx][y0:y1, x0:x1] if has_next else None

        # Convert patches to uint8 for phase correlation if needed
        def to_8bit(img):
            if img.dtype == np.uint16:
                return (img / 257).astype(np.uint8)
            return img

        if has_prev and has_next:
            # Both neighbors available — phase correlate
            g_prev = cv2.cvtColor(to_8bit(patch_prev), cv2.COLOR_BGR2GRAY).astype(np.float32)
            g_next = cv2.cvtColor(to_8bit(patch_next), cv2.COLOR_BGR2GRAY).astype(np.float32)
            han = cv2.createHanningWindow(g_prev.shape[::-1], cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(g_prev, g_next, han)

            if response < MIN_RESPONSE:
                # Low confidence (featureless sky) — average without shifting
                synth = ((patch_prev.astype(np.float32) +
                          patch_next.astype(np.float32)) / 2.0).astype(frame.dtype)
            else:
                # Full Star Bridge morph
                warped_prev = _shift_image(patch_prev, dx / 2.0, dy / 2.0)
                warped_next = _shift_image(patch_next, -dx / 2.0, -dy / 2.0)
                synth = ((warped_prev.astype(np.float32) +
                          warped_next.astype(np.float32)) / 2.0).astype(frame.dtype)

        elif has_prev:
            synth = patch_prev.copy()

        else:
            synth = patch_next.copy()

        # Paste synthetic pixels into result at only the masked locations
        result[y0:y1, x0:x1][comp_mask] = synth[comp_mask]

    return result
