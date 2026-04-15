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
