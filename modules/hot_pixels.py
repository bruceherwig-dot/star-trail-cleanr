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
import time

import cv2
import numpy as np
from typing import List

from . import step_timer as _step_timer

# Pixels dimmer than this (0-255 scale) are ignored when looking for hot
# pixels. Very dark pixels can show a large brightness ratio over their
# local median purely from noise, so this floor keeps faint background
# noise from being flagged as a defect.
MIN_BRIGHTNESS = 5


# How many frames the whole-number search works on at once. Same number, and
# same reason for keeping it small and FIXED, as _GROUP in the reference version
# below: every frame in flight holds several full-size working copies, so
# letting this grow with the batch or the core count would trade a speed problem
# for a memory one on a big camera. Measured 2026-11-16 on 20 frames of 22 MP:
# 4 at a time peaks about 1.2 GB (today's version: about 1.65 GB); 6 at a time
# is a little faster but peaks about 1.74 GB.
_GROUP_FAST = 4


def _whole_number_search_applies(frames, threshold, min_channel_excess) -> bool:
    """True when the whole-number search gives exactly the reference's answer.

    That needs ordinary 8-bit colour frames, a brightness multiplier that is a
    whole number (the default is 2), and a colour margin whose double is a whole
    number (the default is 60). Anything else uses the reference version.
    """
    try:
        if not frames:
            return False
        for f in frames:
            if f.dtype != np.uint8 or f.ndim != 3 or f.shape[2] != 3:
                return False
        t = float(threshold)
        e2 = 2.0 * float(min_channel_excess)
        return (t.is_integer() and 1.0 <= t <= 255.0
                and e2.is_integer() and 0.0 <= e2 <= 765.0)
    except Exception:
        return False


def _build_hot_pixel_map_whole_numbers(frames, threshold, min_fraction,
                                       min_channel_excess) -> np.ndarray:
    """The same map as the reference version, with the same two tests, done in
    whole-number arithmetic on 8-bit data instead of decimal arithmetic on
    full-size copies. About a third less time and less memory (measured
    2026-11-16); the map is identical, pixel for pixel.

    Why each step gives the same yes or no as the reference:

      * Brightness. The reference asks "pixel / background > threshold", with
        a background below 1 treated as 1. For whole numbers that is the same
        question as "pixel > threshold x max(background, 1)". The product is
        capped at 255, which changes nothing: an 8-bit pixel can never be
        greater than 255, so wherever the true product is 255 or more the
        answer is "no" either way.
      * Colour. The reference asks "max - (min + mid) / 2 >= margin". Doubling
        both sides: "2 x max - (min + mid) >= 2 x margin", and since
        min + mid = sum - max, that is "3 x max - sum >= 2 x margin", all in
        whole numbers. 3 x max is never smaller than the sum, so nothing goes
        negative.
    """
    h, w = frames[0].shape[:2]
    n = len(frames)
    min_hits = int(n * min_fraction)
    t = int(threshold)
    two_e = int(round(2.0 * float(min_channel_excess)))

    def _one(frame):
        """One frame's four yes/no planes: bright in B, G, R, and oddly coloured."""
        hits = []
        for ch in range(3):
            plane = np.ascontiguousarray(frame[:, :, ch])
            med = cv2.medianBlur(plane, 9)
            limit = cv2.multiply(cv2.max(med, 1), t)      # caps at 255
            hits.append((plane > limit) & (plane > MIN_BRIGHTNESS))
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        chan_max = np.maximum(np.maximum(r, g), b)
        total = b.astype(np.uint16)
        total += g
        total += r
        excess = chan_max.astype(np.uint16)
        excess *= 3
        excess -= total                                   # 3 x max - sum, never negative
        return hits, excess >= two_e

    hit_b = np.zeros((h, w), dtype=np.uint16)
    hit_g = np.zeros((h, w), dtype=np.uint16)
    hit_r = np.zeros((h, w), dtype=np.uint16)
    excess_hits = np.zeros((h, w), dtype=np.uint16)

    def _tally(result):
        hits, excess = result
        hit_b[...] += hits[0]
        hit_g[...] += hits[1]
        hit_r[...] += hits[2]
        excess_hits[...] += excess

    pool = None
    try:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=_GROUP_FAST)
    except Exception:
        pool = None                     # no threads: do the same work in order
    if pool is None:
        for f in frames:
            _tally(_one(f))
    else:
        with pool:
            for i in range(0, n, _GROUP_FAST):
                for result in pool.map(_one, frames[i:i + _GROUP_FAST]):
                    _tally(result)

    ratio_hot = ((hit_b >= min_hits) | (hit_g >= min_hits) |
                 (hit_r >= min_hits))
    excess_hot = (excess_hits >= min_hits)
    return (ratio_hot & excess_hot).astype(np.uint8) * 255


# The colour-first search works in square tiles (see _build_hot_pixel_map_colour_first).
# 4 pixels of margin is half of the 9-pixel median window, so the median at every pixel
# inside a tile is exactly what the whole-frame filter gives.
_CF_TILE = 64
_CF_MARGIN = 4
# When the colour test leaves candidates in tiles covering more than this share of the
# frame, the tile bookkeeping costs more than it saves (a colourful foreground can do
# this) and the whole-number search is used instead.
_CF_MAX_COVER = 0.40


def _build_hot_pixel_map_colour_first(frames, threshold, min_fraction,
                                      min_channel_excess):
    """The same map as _build_hot_pixel_map_whole_numbers, or None when it would not pay.

    The map keeps a pixel only if it passes BOTH tests (bright against its local
    median in enough frames, and oddly coloured in enough frames). The colour test
    costs about 50 ms a frame; the brightness test needs a 9x9 median over three
    colour planes, about 85 percent of the whole search. So the colour test is run
    on every frame first, and the median test only inside the small tiles that hold
    a pixel that passed it. Nowhere else can the final map be set, so the map is the
    same pixel for pixel (tests/test_hot_pixel_colour_first.py). Measured 2026-11-18
    on three real 20-frame batches: 1.8 to 5.7 times faster, identical maps.

    Returns None, doing nothing further, when the candidate tiles cover more than
    _CF_MAX_COVER of the frame; the caller then runs the whole-number search.
    """
    h, w = frames[0].shape[:2]
    n = len(frames)
    min_hits = int(n * min_fraction)
    t = int(threshold)
    two_e = int(round(2.0 * float(min_channel_excess)))

    def _colour(frame):
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        chan_max = np.maximum(np.maximum(r, g), b)
        total = b.astype(np.uint16)
        total += g
        total += r
        excess = chan_max.astype(np.uint16)
        excess *= 3
        excess -= total                                   # 3 x max - sum, never negative
        return excess >= two_e

    excess_hits = np.zeros((h, w), dtype=np.uint16)
    pool = None
    try:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=_GROUP_FAST)
    except Exception:
        pool = None
    if pool is None:
        for f in frames:
            excess_hits += _colour(f)
    else:
        with pool:
            for res in pool.map(_colour, frames):
                excess_hits += res
    cand = excess_hits >= min_hits
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    ntx = (w + _CF_TILE - 1) // _CF_TILE
    tile_ids = np.unique((ys // _CF_TILE).astype(np.int64) * ntx + (xs // _CF_TILE))
    # Covered area counted tile by tile, so edge tiles and frames smaller than a tile are
    # not over-counted.
    covered = sum((min(h, (int(i // ntx) + 1) * _CF_TILE) - int(i // ntx) * _CF_TILE)
                  * (min(w, (int(i % ntx) + 1) * _CF_TILE) - int(i % ntx) * _CF_TILE)
                  for i in tile_ids)
    if covered > _CF_MAX_COVER * h * w:
        return None
    tiles = [(int(i // ntx), int(i % ntx)) for i in tile_ids]

    hit_b = np.zeros((h, w), dtype=np.uint16)
    hit_g = np.zeros((h, w), dtype=np.uint16)
    hit_r = np.zeros((h, w), dtype=np.uint16)
    counters = (hit_b, hit_g, hit_r)

    def _tile(tile):
        """One tile, all frames. Tiles do not overlap, so workers never touch the
        same counters."""
        ty, tx = tile
        y0, y1 = ty * _CF_TILE, min(h, ty * _CF_TILE + _CF_TILE)
        x0, x1 = tx * _CF_TILE, min(w, tx * _CF_TILE + _CF_TILE)
        cy0, cy1 = max(0, y0 - _CF_MARGIN), min(h, y1 + _CF_MARGIN)
        cx0, cx1 = max(0, x0 - _CF_MARGIN), min(w, x1 + _CF_MARGIN)
        for frame in frames:
            for ch in range(3):
                plane = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1, ch])
                med = cv2.medianBlur(plane, 9)
                limit = cv2.multiply(cv2.max(med, 1), t)      # caps at 255
                hit = (plane > limit) & (plane > MIN_BRIGHTNESS)
                counters[ch][y0:y1, x0:x1] += hit[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]

    pool = None
    try:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=_GROUP_FAST)
    except Exception:
        pool = None
    if pool is None:
        for tile in tiles:
            _tile(tile)
    else:
        with pool:
            list(pool.map(_tile, tiles))

    ratio_hot = (hit_b >= min_hits) | (hit_g >= min_hits) | (hit_r >= min_hits)
    return (ratio_hot & cand).astype(np.uint8) * 255


def build_hot_pixel_map(frames: List[np.ndarray], threshold: float = 2.0,
                        min_fraction: float = 0.8,
                        min_channel_excess: float = 60.0) -> np.ndarray:
    """Identify hot pixels by temporal consistency across the full frame.

    This is the function the app calls. It returns exactly what
    ``_build_hot_pixel_map_reference`` returns (see that function for the two
    tests and what the parameters mean). For ordinary 8-bit colour frames with
    the usual settings it gets there through the whole-number search, which is
    faster and lighter; for anything else it runs the reference itself.
    tests/test_hot_pixel_whole_numbers.py holds the two to the same answer; where the
    colour test leaves few candidates it runs the median test only near them
    (tests/test_hot_pixel_colour_first.py).
    """
    _t0 = time.perf_counter()
    if _whole_number_search_applies(frames, threshold, min_channel_excess):
        fast = _build_hot_pixel_map_colour_first(
            frames, threshold, min_fraction, min_channel_excess)
        if fast is not None:
            _record_search("colour_first_tiles", fast, len(frames), _t0)
            return fast
        out = _build_hot_pixel_map_whole_numbers(
            frames, threshold, min_fraction, min_channel_excess)
        _record_search("whole_numbers_fallback", out, len(frames), _t0)
        return out
    out = _build_hot_pixel_map_reference(
        frames, threshold, min_fraction, min_channel_excess)
    _record_search("reference_unusual_input", out, len(frames), _t0)
    return out


def _record_search(path, found_map, n_frames, t0):
    """Note which stuck-pixel search ran, over how many frames, how many pixels it marked,
    and how long it took (see modules/step_timer.py). Observes only."""
    try:
        _step_timer.add("hotpix.search", time.perf_counter() - t0)
        _step_timer.count("hotpix.search", path)
        _step_timer.count("hotpix.search", "frames_searched", int(n_frames))
        _step_timer.count("hotpix.search", "pixels_marked", int(np.count_nonzero(found_map)))
    except Exception:
        pass


def _build_hot_pixel_map_reference(frames: List[np.ndarray], threshold: float = 2.0,
                                   min_fraction: float = 0.8,
                                   min_channel_excess: float = 60.0) -> np.ndarray:
    """Identify hot pixels by temporal consistency across the full frame.

    THE REFERENCE VERSION: the definition of the two tests, in plain decimal
    arithmetic. ``build_hot_pixel_map`` uses it directly for unusual inputs and
    the test suite uses it to check the faster whole-number search.

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

    # THE MEDIAN FILTERS ARE THE WHOLE COST OF THIS FUNCTION, and they are
    # independent of each other, so they are computed a few frames ahead while
    # the counting below stays exactly as it was.
    #
    # Measured 2026-08-26 on a 100-frame masked run: this step was 18.3 seconds
    # per 20-frame batch, second only to repair, and it had never been noticed
    # because it only runs when the user painted a foreground mask -- the test
    # sequences used all day had none, so it read 0ms every time.
    #
    # One 24MP median filter takes 318ms and OpenCV does NOT thread it, so the
    # work sat on one core while eleven idled: 60 filters per batch, one after
    # another. Six at a time is 5.3x faster with byte-identical output (verified
    # by comparing every filtered plane).
    #
    # The group is CAPPED rather than scaled to the batch, because each frame in
    # flight holds three filtered copies: four 24MP frames is about 288 MB while
    # the step runs, and letting that scale with a 20-frame batch on a 44MP
    # camera would trade a memory problem for a speed one. (Doing exactly that
    # by accident, earlier the same day, is why this note exists.)
    _GROUP = 4

    def _medians(frame):
        """The three per-channel median backgrounds for one frame."""
        return [cv2.medianBlur(frame[:, :, ch], 9) for ch in range(3)]

    def _grouped_medians(seq):
        """Yield each frame's medians, computing a few frames ahead in parallel.
        Falls back to plain sequential work if threads are unavailable."""
        try:
            from concurrent.futures import ThreadPoolExecutor
        except Exception:
            for f in seq:
                yield f, _medians(f)
            return
        with ThreadPoolExecutor(max_workers=_GROUP) as ex:
            for i in range(0, len(seq), _GROUP):
                chunk = seq[i:i + _GROUP]
                for f, meds in zip(chunk, ex.map(_medians, chunk)):
                    yield f, meds

    for frame, _meds in _grouped_medians(frames):
        # --- Brightness test, done independently per color channel ---
        for ch, hit in enumerate([hit_b, hit_g, hit_r]):
            plane = frame[:, :, ch]
            # Local background for each pixel: median of its 9x9 neighborhood.
            # Median (not mean) so a lone bright defect doesn't inflate its
            # own background and hide itself. Computed just above, in parallel.
            med = _meds[ch]
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


# The ground is the same pixels in every frame, so every speck of its texture
# and colour noise looks "stuck" to the temporal test. A real sensor defect is
# spread evenly over the whole frame, so the ground should hold about the same
# number of flags per area as the sky. When it holds far more, the flags are
# scenery. Found 2026-10-19 on King of Wings (ISO 10000): 16,707 ground flags,
# about 13x the sky's density, each painted over with a smooth blob in every
# frame, which the lighten-max stack turned into dark dots.
GROUND_MIN_FLAGS = 500        # a handful of real ground defects never trips it
GROUND_DENSITY_RATIO = 4.0    # ground flags per area vs sky flags per area
SKY_DENSITY_FLOOR = 50.0      # flags per megapixel; stops a spotless sky making any ground look noisy


def ground_looks_like_scenery(hot_map: np.ndarray, fg_mask: np.ndarray):
    """Decide whether the ground's stuck-pixel flags are really just scenery.

    ``hot_map`` is the stuck-pixel map (255 = flagged); ``fg_mask`` is the
    foreground mask (255 = ground, 0 = sky). Returns
    ``(is_scenery, ground_flags, ground_per_mpx, sky_per_mpx)``.
    """
    flagged = hot_map > 0
    ground = fg_mask > 0
    ground_px = int(ground.sum())
    sky_px = int(ground.size - ground_px)
    ground_flags = int((flagged & ground).sum())
    sky_flags = int((flagged & ~ground).sum())
    if ground_px == 0 or ground_flags < GROUND_MIN_FLAGS:
        return False, ground_flags, 0.0, 0.0
    ground_density = ground_flags / (ground_px / 1e6)
    sky_density = sky_flags / (sky_px / 1e6) if sky_px else 0.0
    limit = GROUND_DENSITY_RATIO * max(sky_density, SKY_DENSITY_FLOOR)
    return ground_density > limit, ground_flags, ground_density, sky_density


def repaint_ground_stuck_pixels_16bit(frame16: np.ndarray, dilated: np.ndarray) -> np.ndarray:
    """Return a copy of a 16-bit frame with the ground stuck-pixel spots painted over.

    The spots are filled from the surrounding pixels, worked out on an 8-bit
    copy and scaled back up (x257). The frame passed in is not changed.

    Until 2026-11-17 the engine read each frame off the disk again for this step,
    which for a RAW file meant decoding it a second time (about 0.9 s a frame on a
    24 MP RAW). The frame is already in memory, so it is used directly. That also
    fixes a mixed batch: a frame that was 8-bit on disk and was brought up to
    16-bit by the loader used to be read back as 8-bit here, and its spots came
    out black.
    """
    out = frame16.copy()
    on8 = (out >> 8).astype(np.uint8)
    filled8 = cv2.inpaint(on8, dilated, 3, cv2.INPAINT_NS)
    filled16 = filled8.astype(np.uint16) * 257
    out[dilated > 0] = filled16[dilated > 0]
    return out


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


def content_aware_fill(img, fill_mask, pad=40, progress=None):
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
    progress
        Optional callable(done, total) reported after each defect cluster is
        filled, so a caller driving a progress bar can show this step moving
        (the BEST-quality reconstruction costs real time on big clusters).
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
        if progress is not None:
            try:
                progress(k, n - 1)
            except Exception:
                pass   # a broken progress callback must never break the fill
    return out
