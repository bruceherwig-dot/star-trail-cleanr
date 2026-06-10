"""Chromatic bloom hot-pixel detector (H8 rev2, data-derived).

Detects stuck-photodiode defects — hot (stuck high) OR dead (stuck low) —
by the radial signature of their demosaic bloom. Per-frame, per-pixel, no
spatial-median rule, no batching dependency.

Rule (per channel c in {R, G, B}, per frame):

    d        = c - mean(other two)                    [signed bias map]
    d_out    = mean of d in annulus r=5..8            [local baseline]
    d_in     = mean of d in annulus r=1..2            [inner bloom]

    flag if  |d - d_out|  >= center_threshold  AND
             |d_in - d_out| >= inner_threshold  AND
             (d - d_out) * (d_in - d_out) > 0         [same direction]

Then OR across the three channels, accumulate per frame, and keep pixels
that fire in at least `min_fraction` of frames.

Thresholds derived from 100-frame measurements at 10 ground-truth
hot-pixel coordinates across two Canon datasets (JT, MW101) on 2026-04-14.
See runs/experiments/2026_04_14_chromatic_purity/README.md.
"""
import cv2
import numpy as np


def build_hot_pixel_map_chromatic(
    frames,
    center_threshold=40.0,
    inner_threshold=20.0,
    r_outer_lo=5, r_outer_hi=8,
    min_fraction=0.8,
):
    """Build a persistent hot/dead-pixel mask using the chromatic-bloom signature.

    Plain English: a stuck photosite shows up after demosaic as a small color
    smear where one channel (R, G, or B) is much higher or lower than the other
    two. This function looks for that color bias at a single bright (or dark)
    center pixel, compared against a ring of surrounding sky, and keeps only the
    spots that repeat in nearly every frame.

    For each frame it tries three "which channel is dominant" hypotheses
    (R-vs-rest, G-vs-rest, B-vs-rest). For each it builds a signed color-bias map,
    measures the bias in a far ring (r=5..8, the local baseline) and at the very
    center (the inner bloom), and flags a pixel only when:
      - the center bias stands out from the baseline by at least center_threshold,
      - the inner-bloom bias stands out by at least inner_threshold, and
      - both lean the same direction (both brighter, or both darker).
    Flags from the three channel hypotheses are OR'd together per frame.

    Across all frames it counts how often each pixel fired and keeps the ones
    that fired in at least min_fraction of the frames (the persistence gate that
    separates a fixed sensor defect from a star drifting through). Real stars move
    with Earth rotation so they never land on the same pixel often enough.

    Args:
        frames: list of HxWx3 BGR images (numpy arrays). Empty list returns None.
        center_threshold: minimum color-bias gap (center vs baseline ring) to flag.
        inner_threshold: minimum color-bias gap (inner bloom vs baseline ring).
        r_outer_lo, r_outer_hi: inner/outer radii of the baseline annulus.
        min_fraction: fraction of frames a pixel must fire in to be kept.

    Returns:
        A HxW uint8 mask (255 = persistent defect, 0 = clean), or None if no frames.
    """
    # Inner ring is fixed as the 8 immediate neighbors (3x3 minus center).
    # That's the true r=1..sqrt(2) annulus, no corner pollution, which matters
    # for weak defects like JT jt4 (H8 rev3, 2026-04-14).
    if not frames:
        return None
    h, w = frames[0].shape[:2]

    in_outer_side = 3
    in_inner_side = 1
    out_outer_side = 2 * r_outer_hi + 1
    out_inner_side = 2 * (r_outer_lo - 1) + 1

    in_outer_area = float(in_outer_side * in_outer_side)
    in_inner_area = float(in_inner_side * in_inner_side)
    in_ring_area = in_outer_area - in_inner_area

    out_outer_area = float(out_outer_side * out_outer_side)
    out_inner_area = float(out_inner_side * out_inner_side)
    out_ring_area = out_outer_area - out_inner_area

    def annulus_mean(src, outer_side, outer_area, inner_side, inner_area, ring_area):
        """Mean of src over a square ring (annulus), computed for every pixel at once.

        Takes the average over a big square box and subtracts out the average over
        a smaller centered square box, leaving just the ring between them. Uses fast
        box filters so the result is one value per pixel (the mean of that pixel's
        surrounding ring). When inner_side <= 1 the "inner box" is the single center
        pixel, so the ring excludes only the pixel itself.
        """
        outer = cv2.boxFilter(
            src, -1, (outer_side, outer_side),
            normalize=True, borderType=cv2.BORDER_REPLICATE
        )
        if inner_side <= 1:
            return (outer * outer_area - src * inner_area) / ring_area
        inner = cv2.boxFilter(
            src, -1, (inner_side, inner_side),
            normalize=True, borderType=cv2.BORDER_REPLICATE
        )
        return (outer * outer_area - inner * inner_area) / ring_area

    hit_count = np.zeros((h, w), np.uint16)

    # (dom, o1, o2) in BGR channel-index order
    hypotheses = [(2, 1, 0), (1, 2, 0), (0, 2, 1)]

    for frame in frames:
        f = frame.astype(np.float32)
        frame_hit = np.zeros((h, w), bool)

        for dom_i, o1_i, o2_i in hypotheses:
            d = f[:, :, dom_i] - 0.5 * (f[:, :, o1_i] + f[:, :, o2_i])

            d_out = annulus_mean(
                d, out_outer_side, out_outer_area,
                out_inner_side, out_inner_area, out_ring_area,
            )
            d_in = annulus_mean(
                d, in_outer_side, in_outer_area,
                in_inner_side, in_inner_area, in_ring_area,
            )

            delta_c = d - d_out
            delta_i = d_in - d_out

            cond = (
                (np.abs(delta_c) >= center_threshold) &
                (np.abs(delta_i) >= inner_threshold) &
                (delta_c * delta_i > 0)
            )
            frame_hit |= cond

        hit_count += frame_hit.astype(np.uint16)

    n = len(frames)
    thresh = int(np.ceil(min_fraction * n))
    persistent = hit_count >= thresh
    return (persistent.astype(np.uint8) * 255)


def build_hot_pixel_map_white(
    frames,
    center_threshold=30.0,
    inner_threshold=15.0,
    r_outer_lo=5, r_outer_hi=8,
    min_fraction=0.8,
):
    """Build a persistent hot/dead-pixel mask for neutral (colorless) defects.

    Plain English: the companion to build_hot_pixel_map_chromatic, for defects
    where all three color photosites are stuck together. Those show up as a plain
    bright (or dark) spot with no color tint, so the color-bias test would miss
    them. Same ring geometry and same "stand out from the surrounding sky, in both
    the center and the inner bloom, in the same direction, in nearly every frame"
    logic, but the test quantity is overall brightness mean(R,G,B) instead of one
    channel minus the other two.

    Stars also look like bright radial spikes in any single frame, but Earth
    rotation drags them to a new pixel between frames, so they fail the
    min_fraction persistence gate at any fixed (x, y); a real sensor defect stays
    put and survives it.

    Args:
        frames: list of HxWx3 BGR images (numpy arrays). Empty list returns None.
        center_threshold: minimum brightness gap (center vs baseline ring) to flag.
        inner_threshold: minimum brightness gap (inner bloom vs baseline ring).
        r_outer_lo, r_outer_hi: inner/outer radii of the baseline annulus.
        min_fraction: fraction of frames a pixel must fire in to be kept.

    Returns:
        A HxW uint8 mask (255 = persistent defect, 0 = clean), or None if no frames.
    """
    # White/silicon-level defect detector. All three photosites stuck,
    # so the defect shows as a neutral luminance spike with no chromatic
    # bias. Same ring geometry as the chromatic detector, but the test
    # quantity is mean(R,G,B) instead of one channel minus the other two.
    # Stars pass the radial-spike test in a single frame, but Earth
    # rotation moves them between samples, so they fail the persistence
    # gate at any fixed (x,y).
    if not frames:
        return None
    h, w = frames[0].shape[:2]

    in_outer_side = 3
    in_inner_side = 1
    out_outer_side = 2 * r_outer_hi + 1
    out_inner_side = 2 * (r_outer_lo - 1) + 1

    in_outer_area = float(in_outer_side * in_outer_side)
    in_inner_area = float(in_inner_side * in_inner_side)
    in_ring_area = in_outer_area - in_inner_area

    out_outer_area = float(out_outer_side * out_outer_side)
    out_inner_area = float(out_inner_side * out_inner_side)
    out_ring_area = out_outer_area - out_inner_area

    def annulus_mean(src, outer_side, outer_area, inner_side, inner_area, ring_area):
        """Mean of src over a square ring (annulus), computed for every pixel at once.

        Same helper as in build_hot_pixel_map_chromatic: averages a big box and
        subtracts a smaller centered box to isolate the ring between them, using
        fast box filters. When inner_side <= 1 the inner box is just the center
        pixel, so the ring excludes only that pixel.
        """
        outer = cv2.boxFilter(
            src, -1, (outer_side, outer_side),
            normalize=True, borderType=cv2.BORDER_REPLICATE
        )
        if inner_side <= 1:
            return (outer * outer_area - src * inner_area) / ring_area
        inner = cv2.boxFilter(
            src, -1, (inner_side, inner_side),
            normalize=True, borderType=cv2.BORDER_REPLICATE
        )
        return (outer * outer_area - inner * inner_area) / ring_area

    hit_count = np.zeros((h, w), np.uint16)

    for frame in frames:
        f = frame.astype(np.float32)
        luma = (f[:, :, 0] + f[:, :, 1] + f[:, :, 2]) / 3.0

        d_out = annulus_mean(
            luma, out_outer_side, out_outer_area,
            out_inner_side, out_inner_area, out_ring_area,
        )
        d_in = annulus_mean(
            luma, in_outer_side, in_outer_area,
            in_inner_side, in_inner_area, in_ring_area,
        )

        delta_c = luma - d_out
        delta_i = d_in - d_out

        cond = (
            (np.abs(delta_c) >= center_threshold) &
            (np.abs(delta_i) >= inner_threshold) &
            (delta_c * delta_i > 0)
        )
        hit_count += cond.astype(np.uint16)

    n = len(frames)
    thresh = int(np.ceil(min_fraction * n))
    persistent = hit_count >= thresh
    return (persistent.astype(np.uint8) * 255)
