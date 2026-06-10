"""Build a single "clean sky" background image from a stack of frames.

Plain English:
    Given many photos of the same scene taken one after another, a moving object
    (an airplane or satellite trail) only sits over any given spot for a frame or
    two, while the real sky and stars are present in (nearly) every frame. If you
    look at one pixel position across all the frames and pick the *middle* value
    (the median), the brief bright streak from a passing trail gets thrown out as
    an outlier, and what remains is the steady, trail-free sky for that pixel.
    Doing this for every pixel produces one composite image of the scene with the
    transient trails removed -- a "clean sky" reference.

How it fits into the app:
    This is a standalone helper that produces a trail-free background reference
    image. Per the project's v5 pipeline notes, the dedicated clean-sky background
    step was removed from the main run (the repair step now borrows directly from
    neighbor frames instead), so this module is not part of the current two-step
    Detect -> Repair flow. It remains a small, self-contained utility for building
    a temporal-percentile background when one is needed (inferred from its single
    public function and the absence of pipeline-specific glue here).

Input/output convention:
    Frames are expected to be already aligned (same scene registered to the same
    pixel grid) so that a given pixel position refers to the same point in the sky
    across every frame. Images are 3-channel uint8 in BGR order (OpenCV's default),
    and the returned image keeps that same shape and type.
"""
import numpy as np
from typing import List


def build_clean_sky(aligned: List[np.ndarray], percentile: int = 50) -> np.ndarray:
    """Combine many aligned frames into one trail-free background image.

    For each pixel and each color channel independently, this looks at that
    pixel's value across all the input frames and takes the requested temporal
    percentile (the default, 50, is the median). Transient bright trails appear
    in only a few frames, so they fall outside the median and are excluded; the
    steady sky and stars survive. The result is a single composite image with
    airplane/satellite trails removed.

    Args:
        aligned: List of frames, all the same height/width/channels, already
            registered so a pixel position means the same scene point in every
            frame. Each frame is a uint8 BGR array (3 channels).
        percentile: Which temporal percentile to take per pixel, 0-100.
            50 (median) is the airplane-free default. Lower values bias toward
            the darker (cleaner) sky; higher values bias toward brighter pixels.

    Returns:
        A single uint8 BGR image with the same shape as the input frames,
        representing the trail-free background.
    """
    n = len(aligned)
    # Progress line for the console; printed without a newline so "done" appends
    # to the same line once the stacking finishes.
    print(f"    rgb stack ({n} frames)... ", end="", flush=True)
    # Allocate the output up front using the first frame as a shape/type template.
    clean = np.empty(aligned[0].shape, dtype=np.uint8)
    # Process one color channel at a time. Stacking all frames for all 3 channels
    # at once would hold 3x the pixel data in memory; doing it per-channel (and
    # freeing the stack each pass) keeps peak memory to roughly one channel's
    # worth of frames -- relevant for the large full-resolution TIFFs this app
    # typically handles.
    for ch in range(3):
        # Shape: (n_frames, height, width) -- one channel from every frame.
        stack = np.stack([f[:, :, ch] for f in aligned], axis=0)
        # axis=0 collapses the frame dimension, leaving a per-pixel percentile.
        clean[:, :, ch] = np.percentile(stack, percentile, axis=0).astype(np.uint8)
        del stack  # release this channel's stack before building the next one
    print("done")
    return clean
