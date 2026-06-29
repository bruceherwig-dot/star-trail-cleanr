"""Hot/dead pixel cosmetic correction via temporal consistency.

WHAT THIS FILE IS (plain English)
---------------------------------
A camera sensor sometimes has individual photosites that are permanently
"stuck on" (hot pixels) or "stuck off" (dead/cold pixels) regardless of
what light actually hit them. In a star-trail photo they show up as tiny
fixed dots that look almost like stars, but they sit at the exact same spot
on the sensor in every single shot. This module finds those stuck dots and
quietly paints them out so they don't get mistaken for stars or left behind
as blemishes in the final stacked image.

HOW IT TELLS A HOT PIXEL FROM A REAL STAR
-----------------------------------------
The trick is that the camera is on a fixed tripod and the sky rotates, so a
real star drifts to a different pixel position from frame to frame, while a
hot pixel never moves. So if we look at the same (x, y) location across many
frames:
  * A hot pixel is bright in (almost) every frame.
  * A real star is bright in only a few frames, then moves on.

Two more clues are used to avoid false alarms:
  * Hot pixels are often stuck in just one color channel (Red, Green, or
    Blue), because the sensor's color filter is per-photosite. So each
    channel is checked on its own.
  * A real star or scene highlight is roughly white/neutral (all three
    channels similar), whereas a single-channel sensor defect dumps one
    channel far above the other two. That color imbalance is a strong
    "this is a defect, not a star" signal.

WHERE THIS FITS IN THE APP
--------------------------
This is a cosmetic cleanup step in the Star Trail CleanR pipeline. It runs
over a batch of frames, builds a single map of which sensor positions are
hot, then repairs those spots in every frame by inpainting (filling them
from the surrounding clean pixels). It does not touch trail detection or
Star Bridge repair; it just removes stuck-pixel blemishes.

Dead/cold pixels are conceptually the inverse (stuck at 0 instead of stuck
bright). NOTE: the code in this file as written detects and repairs HOT
pixels; there is no separate dead/cold-pixel routine here despite the
mention above.
"""
import cv2
import numpy as np
from typing import List

# Pixels dimmer than this (0-255 scale) are ignored when looking for hot
# pixels. Very dark pixels can show a large brightness ratio over their
# local median purely from noise, so this floor keeps faint background
# noise from being flagged as a defect.
MIN_BRIGHTNESS = 5


def build_hot_pixel_map(frames: List[np.ndarray], threshold: float = 2.0,
                        min_fraction: float = 0.8,
                        min_channel_excess: float = 60.0) -> np.ndarray:
    """Identify hot pixels by temporal consistency across the full frame.

    Builds a single black-and-white map (same width/height as one frame)
    marking every sensor position that behaves like a stuck-bright pixel
    across the whole batch.

    A position is flagged when BOTH of these hold across the batch:
      * in at least ``min_fraction`` of the frames it is brighter than its
        local 9x9 median by ``threshold`` x in at least one color channel, and
      * in at least ``min_fraction`` of the frames its brightest channel
        exceeds the mean of the other two channels by at least
        ``min_channel_excess`` (absolute 0-255 units). Real stars and scene
        highlights are roughly neutral; a Bayer defect dumps one channel
        well above the others, even if the demosaic bleeds some light into
        the neighbors.
    The two counts are tallied independently across the batch, so they do
    NOT have to occur in the same frames: a position bright in one subset of
    frames and color-imbalanced in a different subset can still be flagged.

    Parameters
    ----------
    frames
        The batch of frames to inspect, each a BGR image (OpenCV channel
        order) with the same dimensions. They do NOT need to be aligned;
        the whole method relies on stars moving while defects stay put.
    threshold
        How many times brighter than its local 9x9 median a pixel must be
        to count as "bright" in a given frame. Default 2.0 means twice the
        local background.
    min_fraction
        Fraction of frames (0-1) in which a position must qualify before it
        is considered a permanent defect. Default 0.8 means a pixel must
        misbehave in at least 80% of the frames.
    min_channel_excess
        Minimum color imbalance (0-255 units) between the brightest channel
        and the average of the other two for a pixel to count as defect-like
        in a frame. This is the "it's one-color, so it's a sensor defect,
        not a neutral star" gate.

    Returns
    -------
    np.ndarray
        A uint8 mask the size of one frame: 255 where a hot pixel was found,
        0 everywhere else. Suitable to pass straight to OpenCV inpainting.
    """
    h, w = frames[0].shape[:2]
    n = len(frames)
    # A position must qualify in this many frames (out of n) to be flagged.
    min_hits = int(n * min_fraction)

    # Per-position counters, one per color channel, tallying how many frames
    # a position looked "too bright vs its local median" in that channel.
    hit_b = np.zeros((h, w), dtype=np.uint16)
    hit_g = np.zeros((h, w), dtype=np.uint16)
    hit_r = np.zeros((h, w), dtype=np.uint16)
    # Separate counter for the color-imbalance test (one channel far above
    # the other two), tallied across all frames regardless of channel.
    excess_hits = np.zeros((h, w), dtype=np.uint16)

    for frame in frames:
        # --- Brightness test, done independently per color channel ---
        for ch, hit in enumerate([hit_b, hit_g, hit_r]):
            plane = frame[:, :, ch]
            # Local background for each pixel: median of its 9x9 neighborhood.
            # Median (not mean) so a lone bright defect doesn't inflate its
            # own background and hide itself.
            med = cv2.medianBlur(plane, 9)
            med_safe = med.astype(np.float32)
            # Clamp the background floor to 1 to avoid divide-by-zero (and
            # absurd ratios) where the local median is 0.
            med_safe[med_safe < 1] = 1
            ratio = plane.astype(np.float32) / med_safe
            # Count this frame's "bright" positions: well above local median
            # AND above the dark-noise floor.
            hit += ((ratio > threshold) & (plane > MIN_BRIGHTNESS)).astype(np.uint16)

        # --- Color-imbalance test, computed once per frame across channels ---
        # Note OpenCV's BGR order: index 0 is Blue, 1 Green, 2 Red.
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
        chan_max = np.maximum(np.maximum(r, g), b)
        chan_min = np.minimum(np.minimum(r, g), b)
        # Middle channel found by subtracting the max and min from the sum,
        # avoiding a separate sort.
        chan_mid = r + g + b - chan_max - chan_min
        other_mean = (chan_min + chan_mid) / 2.0
        # Flag positions where the brightest channel towers over the average
        # of the other two by at least min_channel_excess.
        excess_hits += (chan_max - other_mean >= min_channel_excess).astype(np.uint16)

    # A position passes the brightness test if it was bright often enough in
    # ANY single channel (defects are frequently stuck in just one channel).
    ratio_hot = ((hit_b >= min_hits) | (hit_g >= min_hits) |
                 (hit_r >= min_hits))
    excess_hot = (excess_hits >= min_hits)
    # Final hot-pixel mask requires BOTH the brightness pattern and the
    # color-imbalance pattern, which keeps neutral stars from being flagged.
    return (ratio_hot & excess_hot).astype(np.uint8) * 255


def fix_hot_pixels(frames: List[np.ndarray], threshold: float = 2.0,
                   min_fraction: float = 0.8) -> List[np.ndarray]:
    """Detect and repair hot pixels across a batch of frames.

    Convenience wrapper that first finds the shared hot-pixel map for the
    batch, then paints those spots out of every frame.

    NOTE: the shipped pipeline (astro_clean_v5.py) does NOT call this
    function. It calls ``build_hot_pixel_map`` directly and runs its own
    inpainting, which additionally AND-restricts the dilated hot map against
    the foreground/sky mask (so only sky pixels are repaired) and has a
    separate 16-bit path. This wrapper is used only by the test suite and
    standalone tools/ scripts.

    Parameters
    ----------
    frames
        The batch of BGR frames to clean. Same images are returned cleaned.
    threshold
        Brightness-over-local-median multiplier passed through to
        ``build_hot_pixel_map`` (default 2.0).
    min_fraction
        Fraction of frames a position must misbehave in to be treated as a
        permanent defect, passed through to ``build_hot_pixel_map``
        (default 0.8).

    Returns
    -------
    List[np.ndarray]
        The cleaned frames. If no hot pixels were found, the original frame
        list is returned unchanged (no copies made). Otherwise a new list of
        inpainted frames is returned.
    """
    mask = build_hot_pixel_map(frames, threshold, min_fraction)
    n_defective = int((mask > 0).sum())

    # Nothing stuck-bright found: skip the (relatively costly) inpaint and
    # hand back the exact same frames untouched.
    if n_defective == 0:
        print("  No hot pixels detected")
        return frames

    print(f"  {n_defective} hot pixels detected")
    # Bayer demosaic spreads a single defect into a ~5px bloom and color
    # bleeds further into neighbors. Dilate generously, then use
    # Navier-Stokes inpainting so the fill comes from true uncontaminated
    # surround instead of a median that still sees halo.
    # 13x13 ellipse: grows the mask outward to cover the bloom/color-bleed
    # halo, not just the exact stuck pixel. The "3" is inpaint's radius (how
    # far around each masked pixel it samples for the fill); INPAINT_NS is
    # the Navier-Stokes algorithm.
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return [cv2.inpaint(f, dilated, 3, cv2.INPAINT_NS) for f in frames]


def content_aware_fill(img, fill_mask, pad=40):
    """Replace the white areas of ``fill_mask`` in ``img`` with a content-aware
    fill that CONTINUES surrounding structure instead of smearing it.

    This is for SKY stuck pixels. A plain inpaint diffuses inward from the edge,
    which smears the thin star trails into a grey blob; a content-aware
    (exemplar) fill reconstructs the local pattern, so a trail running past the
    defect stays continuous and a dark gap stays dark.

    Uses OpenCV-contrib's ``cv2.xphoto`` inpaint (Frequency-Selective
    Reconstruction) when present, working on a small crop around each defect so
    it stays fast. If the contrib build is ever missing it falls back to a plain
    Navier-Stokes inpaint, so a missing dependency degrades gracefully and can
    never crash a run.

    Parameters
    ----------
    img
        BGR 8-bit image. Returned cleaned; only masked pixels change.
    fill_mask
        8-bit mask, non-zero where pixels should be replaced.
    pad
        Pixels of surrounding context given to the fill around each defect.
    """
    if fill_mask is None or not np.any(fill_mask):
        return img
    out = img.copy()
    has_xphoto = hasattr(cv2, "xphoto")
    h, w = img.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats((fill_mask > 0).astype(np.uint8), 8)
    for k in range(1, n):
        x, y, bw, bh, _ = stats[k]
        x0, y0 = max(x-pad, 0), max(y-pad, 0)
        x1, y1 = min(x+bw+pad, w), min(y+bh+pad, h)
        crop = out[y0:y1, x0:x1]
        m = fill_mask[y0:y1, x0:x1]
        try:
            if has_xphoto:
                # xphoto convention: non-zero = KNOWN (keep), zero = inpaint.
                dst = crop.copy()
                cv2.xphoto.inpaint(crop, cv2.bitwise_not(m), dst, cv2.xphoto.INPAINT_FSR_BEST)
                crop[m > 0] = dst[m > 0]
            else:
                rep = cv2.inpaint(crop, m, 3, cv2.INPAINT_NS)
                crop[m > 0] = rep[m > 0]
        except Exception:
            rep = cv2.inpaint(crop, m, 3, cv2.INPAINT_NS)
            crop[m > 0] = rep[m > 0]
    return out
