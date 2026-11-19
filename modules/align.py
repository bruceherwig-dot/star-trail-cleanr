"""Phase-correlation frame alignment.

Plain-English summary
----------------------
In a star-trail sequence the camera sits on a fixed tripod, so the *ground*
(buildings, trees, horizon) stays on exactly the same pixels in every frame,
but the *stars* drift a little from one frame to the next as the sky rotates.
This module's job is to slide each frame sideways/up-and-down by a whole-frame
amount so that the stars line up across the batch -- in other words, it picks a
shift that makes the sky "hold still."

It does this with OpenCV's phase correlation, a fast Fourier-based trick that
estimates how far one image has translated relative to another. Each frame is
nudged to match a single reference frame (the middle one). As a side product it
also reports the dominant direction the stars were moving (the star-trail
angle), measured from the per-frame shifts.

How it fits into the app
-------------------------
Note: the shipped v5 pipeline (detect -> Star Bridge repair -> lighten-max
stack) runs WITHOUT an alignment step -- it was removed for speed with no
quality loss (see CLAUDE.md). This module provides a translation-only alignment
that can be used by experiments and tools that need the stars registered to
fixed pixels (for example, align-and-average style background work). It is not
a rotation-aware aligner: it only corrects whole-frame x/y translation, which
is an approximation of true sky rotation and works best for small shifts.
"""
import cv2
import numpy as np
from typing import List, Tuple


def _to_gray32(img: np.ndarray) -> np.ndarray:
    """Convert a color (BGR) image to a single-channel float32 grayscale image.

    Phase correlation works on one grayscale channel and needs floating-point
    pixel values, so every frame is funneled through here before being compared
    or used as the reference.

    Input:  ``img`` -- a BGR color image (OpenCV's default channel order).
    Returns: the same image as a 2D float32 grayscale array.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _warp(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Slide an image by ``dx`` pixels horizontally and ``dy`` pixels vertically.

    This applies a pure translation (no rotation, no scaling) using an affine
    warp. ``dx``/``dy`` may be fractional, in which case the pixels are
    interpolated. It is used to push each frame onto the reference frame once
    the needed shift has been measured.

    Inputs:
        ``img`` -- the image to move.
        ``dx``  -- horizontal shift in pixels (positive moves content right).
        ``dy``  -- vertical shift in pixels (positive moves content down).
    Returns: a new image of the same size, shifted by (dx, dy).

    Edges left empty by the shift are filled by replicating the nearest border
    pixel (BORDER_REPLICATE) rather than by black, so the moved frame has no
    hard black margin.
    """
    h, w = img.shape[:2]
    # 2x3 affine matrix encoding a pure translation: x' = x + dx, y' = y + dy.
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align_batch(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """Shift every frame so the stars land on the same pixels across the batch.

    What it does
    ------------
    Picks the middle frame as the reference, then for every other frame measures
    how far it has translated relative to that reference (via phase correlation)
    and slides it back into register. The middle frame is kept untouched. The
    result is a batch where the sky "holds still" from frame to frame, as far as
    a whole-frame translation can manage.

    As a byproduct it estimates the dominant star-trail direction: each measured
    shift is also a little arrow showing which way that frame's stars had moved,
    and the typical (median) direction of those arrows is the star-trail angle.

    Input:
        ``frames`` -- the batch of color (BGR) frames, in order, as numpy arrays.
                       Assumed all the same size.

    Returns:
        A tuple ``(aligned_frames, star_trail_angle_deg)`` where
        ``aligned_frames`` is a new list (same length/order as the input) with
        each frame shifted onto the reference, and ``star_trail_angle_deg`` is
        the estimated star motion direction in degrees, in the range [0, 180).

    Note: this only corrects translation, not rotation, so it is an
    approximation of true sky rotation and is most accurate for small shifts.
    """
    # Use the middle frame as the anchor everything else is aligned to. It is
    # roughly central in time, which keeps the shifts (and any rotation error)
    # smaller on average than aligning to the first or last frame.
    ref_idx = len(frames) // 2
    ref_g = _to_gray32(frames[ref_idx])
    aligned = []
    shifts = []
    for i, f in enumerate(frames):
        if i == ref_idx:
            # The reference frame defines the target, so it is left as-is with a
            # zero shift -- never warped.
            aligned.append(f)
            shifts.append((0.0, 0.0))
        else:
            # phaseCorrelate returns the (dx, dy) that moves this frame onto the
            # reference; the second return value (response peak) is unused.
            shift, _ = cv2.phaseCorrelate(_to_gray32(f), ref_g)
            dx, dy = float(shift[0]), float(shift[1])
            aligned.append(_warp(f, dx, dy))
            shifts.append((dx, dy))
        # Per-frame progress line; flush so it streams live in the run log.
        print(f"    aligning {i+1}/{len(frames)}", flush=True)

    # Keep only shifts big enough to be real motion (> 0.5 px on either axis);
    # near-zero shifts carry no reliable direction and would just add noise.
    nz = [(dx, dy) for dx, dy in shifts if abs(dx) > 0.5 or abs(dy) > 0.5]
    if nz:
        # Each shift (dx, dy) moved a frame TOWARD the reference, so the stars
        # actually drifted in the opposite direction -- hence the negated
        # (-dy, -dx) in atan2. Wrapping with % 180 treats a line and its
        # reverse as the same orientation (a trail has no head/tail here), and
        # the median is robust to a few bad/outlier shifts.
        angles = [float(np.degrees(np.arctan2(-dy, -dx))) % 180.0
                  for dx, dy in nz]
        star_angle = float(np.median(angles))
    else:
        # No frame moved meaningfully (e.g. a very short or near-still batch);
        # report 0 rather than an angle derived from noise.
        star_angle = 0.0
    return aligned, star_angle
