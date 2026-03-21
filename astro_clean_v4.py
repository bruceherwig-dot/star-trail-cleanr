#!/usr/bin/env python3
import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
"""
astro_clean_v4.py — Astrophotography airplane trail removal

ALGORITHM: Cross-frame motion-vector clustering

The key insight: an airplane trail appears as a series of DASHES (one per nav-light
flash) spread across a single frame. Between consecutive frames the whole airplane
moves by a fixed vector (dx, dy). So every dash in frame N has a corresponding dash
in frame N+1 shifted by exactly (dx, dy).

Stars (after phase-correlation alignment) have dx=dy=0. Scintillating colored stars
produce random isolated centroids with no consistent inter-frame motion vector.

Detection:
  1. Align all frames via phase correlation (stars → fixed pixels).
  2. Detect all components in every frame at a low threshold (default 20).
     — Low threshold catches faint trails; motion criterion filters stars.
  3. For every consecutive frame pair (N, N+1) compute all centroid-to-centroid
     displacement vectors within [min_move, max_move] pixels.
  4. Bin those vectors into a 2-D histogram (dx, dy). Airplane dashes all vote
     for the same bin; random coincidences scatter uniformly.
  5. Peaks in the histogram = confirmed airplane motion vectors.
  6. For each peak: collect every contributing centroid pair. Each centroid in
     the pair is part of the airplane trail — mark it for repair.
  7. Repair: replace marked pixels (+ band) with the temporal-median clean sky.

Usage:
    python3 astro_clean_v4.py /path/to/frames -o /path/to/output
    python3 astro_clean_v4.py /path/to/frames -o /path/to/output --start 30 --batch 20
    python3 astro_clean_v4.py /path/to/frames -o /path/to/output --thresh 15 --min-votes 4
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESH      = 20    # color/motion advantage diff vs adjacent frame
                            # Lower than v3 (25) — motion criterion filters stars
DEFAULT_BATCH       = 40
DEFAULT_SKIP_BOUND  = 2    # don't output first/last N frames (weak chain, warp artifacts)
DEFAULT_PERCENTILE  = 50    # temporal percentile for clean-sky repair
DEFAULT_MIN_PIXELS  = 5     # min component area to include as candidate centroid
DEFAULT_MIN_MOVE    = 15    # min inter-frame centroid distance (px) — filters stars at 0
DEFAULT_MAX_MOVE    = 600   # max inter-frame centroid distance — max plausible airplane speed
DEFAULT_BIN_SIZE    = 8     # (dx,dy) histogram bin size in pixels
DEFAULT_MIN_VOTES   = 2     # min histogram-bin votes; trail in N frames → N-1 votes (2 = 3-frame trail)
DEFAULT_MIN_CH_VOTES = 1   # min votes per channel (color AND gray); 1 = fires in 2 consecutive frames
DEFAULT_LOOK_AHEAD   = 5   # frames to look forward/backward for chain verification
DEFAULT_BAND_WIDTH  = 20    # half-width of repair band around each detected centroid (px)
DEFAULT_EDGE_MARGIN = 10    # skip centroids within N px of border (warpAffine artifacts ~5-15px shift)
DEFAULT_CLUSTER_RAD  = 80    # connectivity radius: two trail centroids in same cluster if within this distance
                             # ~40px dash spacing within one trail → all dashes connect; ~200px between parallel trails → separate
DEFAULT_MIN_CLUSTER  = 4    # minimum cluster SIZE (total dots in connected group) to keep
DEFAULT_CELL_SIZE    = 300   # grid cell size for temporal persistence filter (px at 5472px ref)
                             # larger cells ensure a jittering persistent light lands in the same cell across frames
DEFAULT_MAX_PREV     = 0.5   # max fraction of frames a region may have trail hits before it's rejected
DEFAULT_MIN_ELONG    = 4.0   # min PCA elongation for a cluster to be a trail (not a blob)
DEFAULT_MAX_GAP_FILL = 0    # gap fill disabled — gaps from below-threshold frames can't be filled
                             # set > 0 to experiment, but beware false positives from spurious peaks

# ── Hough pass (Step 6b) — continuous trail detection ──────────────────────
DEFAULT_MEDIAN_BATCH    = 0    # wider frame window for clean-sky median (0 = same as --batch)
                               # Use 40–60 to fix contamination when a trail spans >50% of batch frames
DEFAULT_HOUGH_RESIDUAL  = 10   # brightness above temporal median to count as candidate pixel
DEFAULT_HOUGH_VOTES     = 25   # HoughLinesP threshold (collinear pixels needed)
DEFAULT_HOUGH_MIN_LEN   = 200  # min line length (px at 5472px reference)
DEFAULT_HOUGH_MAX_GAP   = 40   # max gap in a line segment (px)
DEFAULT_HOUGH_ANGLE_TOL = 15.0 # degrees — cross-frame angle match tolerance
DEFAULT_HOUGH_RHO_TOL   = 200  # px at reference — cross-frame rho match tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Frame I/O and alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_files(frame_dir: Path, start: int, batch: int) -> List[Path]:
    exts  = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)
    return files[start:start + batch] if batch > 0 else files[start:]


def _to_gray32(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _warp(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align_batch(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """Phase-correlate all frames to the middle frame. Stars → fixed pixels.

    Returns (aligned_frames, star_trail_angle_deg).
    star_trail_angle_deg is the median direction of inter-frame star motion
    (opposite of the alignment shifts), used to reject star-trail Hough lines.
    """
    ref_idx = len(frames) // 2
    ref_g   = _to_gray32(frames[ref_idx])
    aligned = []
    shifts  = []
    for i, f in enumerate(frames):
        if i == ref_idx:
            aligned.append(f)
            shifts.append((0.0, 0.0))
        else:
            shift, _ = cv2.phaseCorrelate(_to_gray32(f), ref_g)
            dx, dy = float(shift[0]), float(shift[1])
            aligned.append(_warp(f, dx, dy))
            shifts.append((dx, dy))
        sys.stdout.write(f"\r    aligning {i+1}/{len(frames)}  ")
        sys.stdout.flush()
    print()

    # Star trail direction = opposite of alignment shift direction.
    # Use non-zero shifts only; median over all frames for robustness.
    nz = [(dx, dy) for dx, dy in shifts if abs(dx) > 0.5 or abs(dy) > 0.5]
    if nz:
        angles = [float(np.degrees(np.arctan2(-dy, -dx))) % 180.0
                  for dx, dy in nz]
        star_angle = float(np.median(angles))
    else:
        star_angle = 0.0
    print(f"    star trail angle: {star_angle:.1f}°")
    return aligned, star_angle


def build_clean_sky(aligned: List[np.ndarray], percentile: int) -> np.ndarray:
    """Per-pixel temporal percentile — airplane-free background for repair."""
    n = len(aligned)
    print(f"    rgb stack ({n} frames)... ", end="", flush=True)
    clean = np.empty(aligned[0].shape, dtype=np.uint8)
    for ch in range(3):
        stack = np.stack([f[:, :, ch] for f in aligned], axis=0)
        clean[:, :, ch] = np.percentile(stack, percentile, axis=0).astype(np.uint8)
        del stack
    print("done")
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame detection: find candidate dash centroids
# ─────────────────────────────────────────────────────────────────────────────

def compute_bg_norm(clean_rgb: np.ndarray,
                    max_boost: float = 3.0,
                    epsilon: float = 2.0) -> np.ndarray:
    """
    Compute per-row brightness boost factors from the clean-sky background.

    Rows brighter than the reference (horizon glow) get a boost > 1 so that
    faint airplane dashes there are amplified to the same apparent level as
    dark-sky dashes. Dark rows get factor = 1 (no change).

    Uses per-row 75th-percentile brightness so that a row dominated by dark
    pixels but with a bright sky band still registers the glow correctly.
    Works across different sky conditions: dark nights, moon glow, city glow.

    max_boost caps the amplification to avoid runaway noise.
    """
    bg = cv2.cvtColor(clean_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 75th-percentile per row captures sky brightness better than mean
    # (avoids being dominated by the many near-zero dark-sky pixels)
    row_bright = np.percentile(bg, 75, axis=1).astype(np.float32)  # (h,)

    # Smooth to a clean gradient (ignore individual bright stars)
    ksize = min((len(row_bright) // 10) | 1, 201)   # ~10% of frame height, odd
    sigma = ksize / 3.0
    row_smooth = cv2.GaussianBlur(
        row_bright.reshape(-1, 1), (1, ksize), sigma).flatten()

    # Reference = the darkest-sky part of the image (20th percentile of rows)
    # Minimum 5 counts to avoid divide-by-near-zero on pitch-black frames.
    ref = max(float(np.percentile(row_smooth, 20)), 5.0)

    # Boost rows that are brighter than reference; leave darker rows at 1.0
    factors = np.maximum(1.0,
                         (row_smooth + epsilon) / (ref + epsilon))
    factors = np.clip(factors, 1.0, max_boost).astype(np.float32)
    return factors   # shape (h,)


def _color_adv(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    red   = np.clip(r - (g + b) // 2, 0, 255).astype(np.uint8)
    green = np.clip(g - (r + b) // 2, 0, 255).astype(np.uint8)
    return red, green


def detect_candidates(frame: np.ndarray,
                      prev_al: Optional[np.ndarray],
                      next_al: Optional[np.ndarray],
                      threshold: int,
                      min_pixels: int,
                      edge_margin: int,
                      bg_norm: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Return (color_cents, gray_cents) — centroids from each channel separately.

    color_cents: components brighter than threshold in color advantage (red or green)
                 vs BOTH aligned neighbors. Catches colored nav lights.
    gray_cents:  components brighter than threshold in gray brightness
                 vs BOTH aligned neighbors. Catches white anti-collision strobes.

    Keeping channels separate lets us require BOTH to vote for the same motion
    vector, using the fact that colored nav lights and white strobes travel
    the same path on the same airplane.
    """
    h, w = frame.shape[:2]
    rc, gc = _color_adv(frame)
    neighbors = [nb for nb in [prev_al, next_al] if nb is not None]

    # Color channel: min across neighbors
    color_diffs = []
    for nb in neighbors:
        nr, ng = _color_adv(nb)
        rd = np.clip(rc.astype(np.int16) - nr.astype(np.int16), 0, 255).astype(np.uint8)
        gd = np.clip(gc.astype(np.int16) - ng.astype(np.int16), 0, 255).astype(np.uint8)
        color_diffs.append(np.maximum(rd, gd))
    color_hit = color_diffs[0].copy()
    for d in color_diffs[1:]:
        np.minimum(color_hit, d, out=color_hit)

    # Gray channel: min across neighbors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gray_diffs = []
    for nb in neighbors:
        gnb = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY).astype(np.int16)
        gray_diffs.append(np.clip(gray - gnb, 0, 255).astype(np.uint8))
    gray_hit = gray_diffs[0].copy()
    for d in gray_diffs[1:]:
        np.minimum(gray_hit, d, out=gray_hit)

    # Apply per-row brightness boost (horizon normalisation)
    if bg_norm is not None:
        norm_2d = bg_norm[:, np.newaxis]          # (h, 1) → broadcasts to (h, w)
        color_hit = np.clip(color_hit.astype(np.float32) * norm_2d, 0, 255).astype(np.uint8)
        gray_hit  = np.clip(gray_hit.astype(np.float32)  * norm_2d, 0, 255).astype(np.uint8)

    def _extract(hit_map: np.ndarray) -> List[Tuple[int, int]]:
        mask = (hit_map > threshold).astype(np.uint8) * 255
        _, _, stats, craws = cv2.connectedComponentsWithStats(mask)
        out = []
        for lbl in range(1, len(stats)):
            if stats[lbl, cv2.CC_STAT_AREA] < min_pixels:
                continue
            cx = int(round(craws[lbl][0]))
            cy = int(round(craws[lbl][1]))
            if edge_margin <= cx < w - edge_margin and edge_margin <= cy < h - edge_margin:
                out.append((cx, cy))
        return out

    return _extract(color_hit), _extract(gray_hit)


# ─────────────────────────────────────────────────────────────────────────────
# Motion-vector clustering
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_pairs(
        cents_a: List[Tuple[int, int]],
        cents_b: List[Tuple[int, int]],
        min_move: int,
        max_move: int,
) -> List[Tuple[int, int, int, int]]:
    """
    All (cx_a, cy_a, dx, dy) where centroid b = (cx_a+dx, cy_a+dy)
    and min_move <= sqrt(dx²+dy²) <= max_move.
    """
    if not cents_a or not cents_b:
        return []
    a = np.array(cents_a, dtype=np.int32)   # (Na, 2)
    b = np.array(cents_b, dtype=np.int32)   # (Nb, 2)

    pairs = []
    # For reasonable centroid counts this nested loop is fast enough.
    # For very large counts we could use a KD-tree, but 50–300 centroids/frame
    # → 50×300 = 15 000 comparisons per pair — well within budget.
    for cx, cy in a:
        dx = b[:, 0] - cx
        dy = b[:, 1] - cy
        dist = np.sqrt(dx.astype(np.float32)**2 + dy.astype(np.float32)**2)
        ok = (dist >= min_move) & (dist <= max_move)
        for j in np.where(ok)[0]:
            pairs.append((cx, cy, int(dx[j]), int(dy[j])))
    return pairs


def cluster_motion_vectors(
        color_pairs: List[Tuple[int, int, int, int, int]],  # (frame_idx, cx, cy, dx, dy)
        gray_pairs:  List[Tuple[int, int, int, int, int]],
        bin_size: int,
        min_votes: int,
        min_ch_votes: int,
        max_move: int,
) -> List[Tuple[int, int, int]]:
    """
    Bin (dx, dy) vectors from color and gray channels separately, then apply NMS.

    A peak is confirmed only when:
      total_votes >= min_votes  AND
      color_votes >= min_ch_votes  AND  gray_votes >= min_ch_votes

    Requiring both channels prevents pure colored-star scintillation
    (color only) or pure brightness noise (gray only) from creating false peaks.
    An airplane trail has both colored nav lights AND a white anti-collision
    strobe, so real peaks always accumulate votes in both channels.
    """
    if not color_pairs and not gray_pairs:
        return []

    offset = max_move
    size   = (2 * max_move) // bin_size + 2
    hist_total = np.zeros((size, size), dtype=np.int32)
    hist_color = np.zeros((size, size), dtype=np.int32)
    hist_gray  = np.zeros((size, size), dtype=np.int32)

    for pairs, hc in ((color_pairs, hist_color), (gray_pairs, hist_gray)):
        for _, _, _, dx, dy in pairs:
            ix = (dx + offset) // bin_size
            iy = (dy + offset) // bin_size
            if 0 <= ix < size and 0 <= iy < size:
                hc[iy, ix]         += 1
                hist_total[iy, ix] += 1

    nms_r     = 2
    local_max = cv2.dilate(hist_total.astype(np.float32),
                           np.ones((2*nms_r+1, 2*nms_r+1), np.uint8))
    peak_mask = (
        (hist_total == local_max.astype(np.int32)) &
        (hist_total >= min_votes) &
        (hist_color >= min_ch_votes) &
        (hist_gray  >= min_ch_votes)
    )

    peaks = []
    ys, xs = np.where(peak_mask)
    for y, x in zip(ys, xs):
        dx_c = x * bin_size - offset + bin_size // 2
        dy_c = y * bin_size - offset + bin_size // 2
        peaks.append((dx_c, dy_c, int(hist_total[y, x])))

    peaks.sort(key=lambda p: -p[2])
    return peaks


# ─────────────────────────────────────────────────────────────────────────────
# Repair and debug
# ─────────────────────────────────────────────────────────────────────────────

def filter_small_clusters(
        cents: List[Tuple[int, int]],
        cluster_radius: int,
        min_cluster_size: int = 5,
        min_elongation: float = 3.0,
) -> List[Tuple[int, int]]:
    """
    Find connected components of trail centroids (two centroids are connected
    if they are within cluster_radius of each other), then discard any component
    that is either:
      • smaller than min_cluster_size dots, OR
      • not elongated enough (PCA elongation < min_elongation)

    Real airplane trails:  10–80+ dots in a LINE → high elongation → kept.
    Blob false-positives: scattered dots in a cloud → elongation ≈ 1–2 → rejected.
    """
    if len(cents) == 0:
        return cents
    if len(cents) < min_cluster_size:
        return []

    n = len(cents)
    a = np.array(cents, dtype=np.float32)
    r2 = float(cluster_radius) ** 2

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            ddx = a[i, 0] - a[j, 0]
            ddy = a[i, 1] - a[j, 1]
            if ddx * ddx + ddy * ddy <= r2:
                union(i, j)

    # Group indices by root
    from collections import defaultdict
    comp: dict = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)

    # Determine which components pass both size AND elongation filters
    keep_roots: set = set()
    for root, members in comp.items():
        if len(members) < min_cluster_size:
            continue
        # PCA elongation check: ratio of principal eigenvalues
        pts = a[members]
        if len(pts) < 3:
            # 2 points always form a perfect line — keep if size passes
            keep_roots.add(root)
            continue
        cov = np.cov(pts[:, 0], pts[:, 1])
        eigs = np.linalg.eigvalsh(cov)   # sorted ascending
        e_max, e_min = float(eigs[1]), float(eigs[0])
        elongation = np.sqrt(e_max / (e_min + 1e-6))
        span = 2.0 * np.sqrt(max(e_max, 0.0))   # ≈ 2σ along principal axis
        # debug: uncomment to see per-cluster info
        # print(f"    cluster sz={len(members)} elong={elongation:.1f} span={span:.0f}px {'KEEP' if elongation>=min_elongation else 'DROP'}")
        if elongation >= min_elongation:
            keep_roots.add(root)

    return [cents[i] for i in range(n) if find(i) in keep_roots]


def _has_nearby(arr: np.ndarray, ex: float, ey: float, tol_sq: float) -> bool:
    """True if any point in arr is within sqrt(tol_sq) of (ex, ey)."""
    if len(arr) == 0:
        return False
    d2 = (arr[:, 0] - ex) ** 2 + (arr[:, 1] - ey) ** 2
    return bool(d2.min() <= tol_sq)


def chain_verified(
        i: int, cx: int, cy: int,
        dx: int, dy: int,
        n: int,
        cands_arr: List[np.ndarray],
        tol_sq: float,
        look_ahead: int,
        boundary_margin: int = 2,
) -> bool:
    """
    Check whether the centroid (cx, cy) in frame i can be tracked for at
    least 2 hits in the same direction (forward or backward), OR is confirmed
    by at least 1 hit on EACH side.

    Gap-tolerant: finds the FIRST hit anywhere within look_ahead steps, then
    looks for a SECOND hit up to 2 steps after that.  This tolerates one missed
    strobe flash (a one-frame gap in the trail) without breaking the chain.

    Rules — keep if ANY of:
      (a) 2 forward hits  (first within look_ahead, second ≤2 frames later)
      (b) 2 backward hits (same)
      (c) 1 forward hit AND 1 backward hit (centroid confirmed from both sides)
      (d) [boundary only] 1 hit from EITHER side — for frames within
          boundary_margin of the batch start/end, where the trail may only
          span 2 frames and cannot accumulate hits on both sides.

    Why this filters false positives:
      A coincidental match in frame N with one random star in N+1 passes the
      k=1 forward check but has no star at (x+2dx, y+2dy) or (x+3dx, y+3dy),
      and no star at (x-dx, y-dy).  It fails all three rules.
      Boundary relaxation (rule d) is still protected by the histogram peak
      requirement (min_votes) and the cluster filter (elongated ≥4 centroids).
    """
    def fwd(k: int) -> bool:
        j = i + k
        return j < n and _has_nearby(cands_arr[j], cx + k * dx, cy + k * dy, tol_sq)

    def bwd(k: int) -> bool:
        j = i - k
        return j >= 0 and _has_nearby(cands_arr[j], cx - k * dx, cy - k * dy, tol_sq)

    # Find first forward hit
    first_fwd = next((k for k in range(1, look_ahead + 1) if fwd(k)), None)
    # Find first backward hit
    first_bwd = next((k for k in range(1, look_ahead + 1) if bwd(k)), None)

    # (c) confirmed from both sides
    if first_fwd is not None and first_bwd is not None:
        return True

    # (a) two forward hits (second hit within 2 steps of first)
    if first_fwd is not None:
        for k2 in range(first_fwd + 1, min(first_fwd + 3, look_ahead + 2)):
            if fwd(k2):
                return True

    # (b) two backward hits
    if first_bwd is not None:
        for k2 in range(first_bwd + 1, min(first_bwd + 3, look_ahead + 2)):
            if bwd(k2):
                return True

    # (d) boundary relaxation: near start/end of batch, accept 1 hit either side
    at_boundary = (i < boundary_margin) or (i >= n - boundary_margin)
    if at_boundary and (first_fwd is not None or first_bwd is not None):
        return True

    return False


def filter_persistent_regions(
        trail_cents_per_frame: List[set],
        h: int, w: int,
        n_total_frames: int,
        cell_size: int,
        max_prevalence: float,
) -> List[set]:
    """
    Remove centroids in grid regions that have trail hits in > max_prevalence
    fraction of ALL frames in the batch.

    Persistent ground lights appear in the same region every frame (prevalence ≈ 1.0).
    Real airplane trails traverse the image — any one region sees them for only a few
    frames (prevalence ≈ 5–25%), well below the threshold.
    """
    grid_h = max(1, (h + cell_size - 1) // cell_size)
    grid_w = max(1, (w + cell_size - 1) // cell_size)
    cell_count = np.zeros((grid_h, grid_w), dtype=np.int32)

    for cents in trail_cents_per_frame:
        visited: set = set()
        for cx, cy in cents:
            gc = (min(cy // cell_size, grid_h - 1),
                  min(cx // cell_size, grid_w - 1))
            if gc not in visited:
                cell_count[gc] += 1
                visited.add(gc)

    max_hits = max_prevalence * n_total_frames
    result = []
    for cents in trail_cents_per_frame:
        kept = set()
        for cx, cy in cents:
            gy = min(cy // cell_size, grid_h - 1)
            gx = min(cx // cell_size, grid_w - 1)
            if cell_count[gy, gx] <= max_hits:
                kept.add((cx, cy))
        result.append(kept)
    return result


def fill_trail_gaps(
        trail_cents_per_frame: List[set],
        peaks: List[Tuple[int, int, int]],
        n: int, h: int, w: int,
        tol: int = 50,
        max_gap: int = 3,
        edge_margin: int = 10,
) -> int:
    """
    Project confirmed trajectories into gap frames.

    For each frame i with no trail centroids, look back up to max_gap frames
    for a source centroid and forward up to max_gap frames for a verification
    centroid, both consistent with the same confirmed peak (dx, dy).

    If source at frame s has centroid (cx, cy), and verification at frame v
    has a centroid near (cx + (i-s+v-i)*dx, cy + ...) = (cx + (v-s)*dx, ...),
    then the airplane was at (cx + (i-s)*dx, cy + (i-s)*dy) in frame i.

    Multiple passes: each pass fills gaps of size 1 by bridging from the
    nearest known frame on each side.  Running max_gap passes fills gaps
    up to max_gap frames wide (each pass extends one frame inward per side).
    """
    tol_sq = float(tol * tol)
    total_added = 0

    for _pass in range(max_gap):
        # Rebuild arrays each pass so newly filled frames count as sources
        cands = [
            np.array(list(s), dtype=np.float32) if s
            else np.empty((0, 2), dtype=np.float32)
            for s in trail_cents_per_frame
        ]
        added_this_pass = 0

        for i in range(n):
            if len(trail_cents_per_frame[i]) > 0:
                continue  # already has centroids

            synthetic: set = set()
            for dx_p, dy_p, _ in peaks:
                # Look back for a source frame, forward for a verification frame
                for back in range(1, max_gap + 1):
                    si = i - back
                    if si < 0 or len(cands[si]) == 0:
                        continue
                    for fwd in range(1, max_gap + 1):
                        vi = i + fwd
                        if vi >= n or len(cands[vi]) == 0:
                            continue
                        # For each centroid in the source frame, project to i
                        # and check the verification frame
                        for cx, cy in cands[si]:
                            px = int(round(cx + back * dx_p))
                            py = int(round(cy + back * dy_p))
                            if not (edge_margin < px < w - edge_margin and
                                    edge_margin < py < h - edge_margin):
                                continue
                            # Expected position at verification frame
                            vx = cx + (back + fwd) * dx_p
                            vy = cy + (back + fwd) * dy_p
                            if _has_nearby(cands[vi], vx, vy, tol_sq):
                                synthetic.add((px, py))

            trail_cents_per_frame[i].update(synthetic)
            added_this_pass += len(synthetic)

        total_added += added_this_pass
        if added_this_pass == 0:
            break  # no more gaps to fill

    return total_added


def snap_to_brightness_peak(
        cents: List[Tuple[int, int]],
        frame_gray: np.ndarray,
        snap_rad: int,
        min_residual: int = 10,
) -> List[Tuple[int, int]]:
    """Shift each centroid to the nearest bright residual pixel within snap_rad.

    Uses the residual image (|aligned - clean|) so stars cancel out and only
    airplane trail pixels are bright.  Picks the closest bright pixel to the
    original centroid (not the globally brightest) to avoid snapping to an
    adjacent brighter trail.  If no pixel exceeds min_residual, keeps original.
    """
    if snap_rad <= 0 or not cents:
        return cents
    h, w = frame_gray.shape
    snapped = []
    for cx, cy in cents:
        x0 = max(0, cx - snap_rad)
        x1 = min(w, cx + snap_rad + 1)
        y0 = max(0, cy - snap_rad)
        y1 = min(h, cy + snap_rad + 1)
        patch = frame_gray[y0:y1, x0:x1]
        bright_yx = np.argwhere(patch > min_residual)
        if len(bright_yx) == 0:
            snapped.append((cx, cy))  # no bright pixel — keep original
            continue
        # Pick the bright pixel closest to the original centroid
        cy_patch = cy - y0
        cx_patch = cx - x0
        dists = np.sqrt((bright_yx[:, 0] - cy_patch) ** 2 +
                        (bright_yx[:, 1] - cx_patch) ** 2)
        nearest = bright_yx[np.argmin(dists)]
        snapped.append((x0 + int(nearest[1]), y0 + int(nearest[0])))
    return snapped


def build_repair_mask(h: int, w: int,
                      centroids: List[Tuple[int, int]],
                      band_px: int,
                      max_connect_gap: int = 300,
                      trail_directions: Optional[List[Tuple[float, float]]] = None,
                      direction_tol_deg: float = 30.0) -> np.ndarray:
    """
    Draw filled circles at each centroid AND connect nearby centroids with
    thick lines so the full trail between strobe flashes is covered.

    Direction-aware clustering: two centroids are only connected if the vector
    between them is within direction_tol_deg of a known trail direction.  This
    prevents centroids from two crossing trails from being merged into one blob
    at their intersection — the cross-trail direction matches neither trail.

    trail_directions: list of (dx, dy) unit-ish vectors for known trajectories.
                      If None, falls back to direction-agnostic clustering.
    """
    from collections import defaultdict
    mask = np.zeros((h, w), dtype=np.uint8)
    if not centroids:
        return mask

    # Circles at every centroid
    for cx, cy in centroids:
        cv2.circle(mask, (cx, cy), band_px, 255, -1)

    if len(centroids) < 2:
        return mask

    pts = np.array(centroids, dtype=np.float32)
    n = len(pts)

    # Pre-compute known trail angles (mod 180 — direction, not orientation)
    known_angles: List[float] = []
    if trail_directions:
        for tdx, tdy in trail_directions:
            known_angles.append(float(np.degrees(np.arctan2(tdy, tdx))) % 180.0)

    def _direction_ok(ddx: float, ddy: float) -> bool:
        """True if (ddx, ddy) is within direction_tol_deg of any known trail."""
        if not known_angles:
            return True
        conn_angle = float(np.degrees(np.arctan2(ddy, ddx))) % 180.0
        for ka in known_angles:
            diff = abs(conn_angle - ka) % 180.0
            if min(diff, 180.0 - diff) < direction_tol_deg:
                return True
        return False

    # Union-Find: connect centroids within max_connect_gap AND same trail direction
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    gap2 = float(max_connect_gap) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ddx = pts[j, 0] - pts[i, 0]
            ddy = pts[j, 1] - pts[i, 1]
            if ddx * ddx + ddy * ddy <= gap2 and _direction_ok(ddx, ddy):
                union(i, j)

    clusters: dict = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    # For each cluster: sort along principal axis, draw lines between consecutive pts
    thickness = max(1, band_px * 2)
    for members in clusters.values():
        if len(members) < 2:
            continue
        cpts = pts[members]
        mean = cpts.mean(axis=0)
        centered = cpts - mean
        cov = np.cov(centered[:, 0], centered[:, 1])
        _, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, 1]
        projections = centered @ principal
        order = np.argsort(projections)
        sorted_pts = cpts[order]
        for k in range(len(sorted_pts) - 1):
            p1 = (int(sorted_pts[k,     0]), int(sorted_pts[k,     1]))
            p2 = (int(sorted_pts[k + 1, 0]), int(sorted_pts[k + 1, 1]))
            cv2.line(mask, p1, p2, 255, thickness)

    return mask


def repair_frame(frame: np.ndarray,
                 mask: np.ndarray,
                 clean_rgb: np.ndarray) -> np.ndarray:
    result = frame.copy()
    result[mask > 0] = clean_rgb[mask > 0]
    return result


def draw_debug(frame: np.ndarray,
               repair_mask: np.ndarray,
               trail_cents: List[Tuple[int, int]],
               label: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    sc  = max(1.0, w / 5472.0)
    lw  = max(2, int(3 * sc))
    rad = max(4, int(6 * sc))

    # Cyan fill where repair mask is set
    if repair_mask is not None and repair_mask.any():
        overlay = out.copy()
        overlay[repair_mask > 0] = [255, 255, 0]
        out = cv2.addWeighted(out, 0.5, overlay, 0.5, 0)

    # Green dot at each trail centroid
    for cx, cy in trail_cents:
        cv2.circle(out, (cx, cy), rad, (0, 255, 0), lw)

    if label:
        cv2.putText(out, label, (10, int(40 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, (0, 255, 255), lw)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hough pass — continuous trail detection (Step 6b)
# ─────────────────────────────────────────────────────────────────────────────

def detect_continuous_trails(
        frames: List[np.ndarray],
        clean_rgb: np.ndarray,
        n: int, h: int, w: int, sc: float,
        residual_thresh: int,
        hough_votes: int,
        min_line_len: int,
        max_line_gap: int,
        angle_tol_deg: float,
        rho_tol: float,
        edge_margin: int,
        band_px: int,
        star_angle_deg: float = -1.0,
        star_angle_tol: float = 5.0,
) -> List[set]:
    """
    Detect continuous (non-flashing) airplane trails via per-frame background
    subtraction + Hough, cross-validated across adjacent frames.

    For each frame: subtract temporal median → threshold → HoughLinesP.
    A line is kept only if a line with matching angle (±angle_tol) and similar
    perpendicular distance (±rho_tol) appears in frame N-1 or N+1.
    Points are sampled along each validated line and returned as repair centroids.
    """
    min_len_px = max(50, int(min_line_len * sc))
    rho_tol_px = rho_tol * sc
    edge_px    = max(5, int(edge_margin * sc))

    clean_gray = cv2.cvtColor(clean_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Step A: per-frame residual → Hough lines stored as (angle_deg, rho, x1,y1,x2,y2)
    all_frame_lines: List[List] = []
    for i in range(n):
        frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual   = np.clip(frame_gray - clean_gray, 0, 255).astype(np.uint8)
        _, thresh  = cv2.threshold(residual, residual_thresh, 255, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180,
                                threshold=hough_votes,
                                minLineLength=min_len_px,
                                maxLineGap=max_line_gap)
        frame_lines = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                if mx < edge_px or mx > w - edge_px or my < edge_px or my > h - edge_px:
                    continue
                dx, dy = float(x2 - x1), float(y2 - y1)
                length = np.hypot(dx, dy)
                if length < 1:
                    continue
                # Canonicalize direction so rho is consistent regardless of
                # which endpoint HoughLinesP chose as x1,y1.  When the segment
                # is returned reversed (dx<0), the normal vector flips sign and
                # rho changes by ~2× the line's distance from origin — easily
                # exceeding rho_tol and causing valid cross-frame matches to fail.
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                angle_deg = float(np.degrees(np.arctan2(dy, dx))) % 180.0

                # Reject lines parallel to star trail direction — these are
                # star trail segments in the residual, not airplane trails.
                if star_angle_deg >= 0:
                    adiff = abs(angle_deg - star_angle_deg)
                    adiff = min(adiff, 180.0 - adiff)
                    if adiff < star_angle_tol:
                        continue

                nx, ny    = -dy / length, dx / length   # unit normal
                rho       = float(x1) * nx + float(y1) * ny
                frame_lines.append((angle_deg, rho, x1, y1, x2, y2))
        all_frame_lines.append(frame_lines)

    # Step B: cross-validate each line against adjacent frames
    continuous_cents: List[set] = [set() for _ in range(n)]
    hough_directions: set = set()   # unique validated line angles (deg, mod 180)
    n_validated = 0

    for i in range(n):
        for angle, rho, x1, y1, x2, y2 in all_frame_lines[i]:
            validated = False
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                for angle2, rho2, *_ in all_frame_lines[j]:
                    adiff = abs(angle - angle2)
                    adiff = min(adiff, 180.0 - adiff)   # wrap at 180°
                    if adiff < angle_tol_deg and abs(rho - rho2) < rho_tol_px:
                        validated = True
                        break
                if validated:
                    break

            if validated:
                seg_len = np.hypot(x2 - x1, y2 - y1)
                n_pts   = max(2, int(seg_len / max(1, band_px)))
                for k in range(n_pts + 1):
                    t  = k / n_pts
                    cx = int(x1 + t * (x2 - x1))
                    cy = int(y1 + t * (y2 - y1))
                    if edge_px < cx < w - edge_px and edge_px < cy < h - edge_px:
                        continuous_cents[i].add((cx, cy))
                hough_directions.add(angle)
                n_validated += 1

    n_raw   = sum(len(l) for l in all_frame_lines)
    n_added = sum(len(s) for s in continuous_cents)
    print(f"  Hough pass: {n_raw} raw lines → {n_validated} validated → {n_added} repair centroids")
    # Convert Hough angles (degrees, mod 180) to (dx, dy) unit vectors
    hough_dirs_xy = [(float(np.cos(np.radians(a))), float(np.sin(np.radians(a))))
                     for a in hough_directions]
    return continuous_cents, hough_dirs_xy


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="astro_clean_v4 — motion-vector clustering airplane trail removal")
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--start",        type=int, default=0)
    parser.add_argument("--batch",        type=int, default=DEFAULT_BATCH)
    parser.add_argument("--thresh",       type=int, default=DEFAULT_THRESH,
                        help=f"Color/motion advantage threshold (default {DEFAULT_THRESH})")
    parser.add_argument("--pct",          type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--min-px",       type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--min-move",     type=int, default=DEFAULT_MIN_MOVE,
                        help=f"Min inter-frame centroid displacement px (default {DEFAULT_MIN_MOVE})")
    parser.add_argument("--max-move",     type=int, default=DEFAULT_MAX_MOVE,
                        help=f"Max inter-frame centroid displacement px (default {DEFAULT_MAX_MOVE})")
    parser.add_argument("--bin-size",     type=int, default=DEFAULT_BIN_SIZE,
                        help=f"Motion-vector histogram bin size px (default {DEFAULT_BIN_SIZE})")
    parser.add_argument("--min-votes",    type=int, default=DEFAULT_MIN_VOTES,
                        help=f"Min histogram votes to confirm airplane vector (default {DEFAULT_MIN_VOTES})")
    parser.add_argument("--min-ch-votes", type=int, default=DEFAULT_MIN_CH_VOTES,
                        help=f"Min votes per channel (color AND gray) to confirm peak (default {DEFAULT_MIN_CH_VOTES}, 0=disable)")
    parser.add_argument("--band-width",   type=int, default=DEFAULT_BAND_WIDTH,
                        help=f"Repair band radius around each dash centroid px (default {DEFAULT_BAND_WIDTH})")
    parser.add_argument("--edge-margin",  type=int, default=DEFAULT_EDGE_MARGIN)
    parser.add_argument("--cluster-rad",  type=int, default=DEFAULT_CLUSTER_RAD,
                        help=f"Connectivity radius px for cluster filter (default {DEFAULT_CLUSTER_RAD})")
    parser.add_argument("--min-cluster",  type=int, default=DEFAULT_MIN_CLUSTER,
                        help=f"Min total dots in connected cluster to keep (default {DEFAULT_MIN_CLUSTER})")
    parser.add_argument("--min-elongation", type=float, default=DEFAULT_MIN_ELONG,
                        help=f"Min PCA elongation for a cluster to be a trail (default {DEFAULT_MIN_ELONG})")
    parser.add_argument("--no-cluster-filter", action="store_true",
                        help="Disable spatial density filter (keep all matched centroids)")
    parser.add_argument("--cell-size",    type=int,   default=DEFAULT_CELL_SIZE,
                        help=f"Grid cell size px for persistence filter (default {DEFAULT_CELL_SIZE})")
    parser.add_argument("--max-prev",     type=float, default=DEFAULT_MAX_PREV,
                        help=f"Max frame prevalence fraction before region rejected (default {DEFAULT_MAX_PREV})")
    parser.add_argument("--no-persist-filter", action="store_true",
                        help="Disable temporal persistence filter")
    parser.add_argument("--max-gap-fill",  type=int, default=DEFAULT_MAX_GAP_FILL,
                        help=f"Max gap (frames) to fill by projecting confirmed trajectory (0=off, default {DEFAULT_MAX_GAP_FILL})")
    parser.add_argument("--look-ahead",   type=int, default=DEFAULT_LOOK_AHEAD,
                        help=f"Frames to look forward/backward for chain verification (default {DEFAULT_LOOK_AHEAD})")
    parser.add_argument("--skip-boundary", type=int, default=DEFAULT_SKIP_BOUND,
                        help=f"Skip first/last N frames from output (still used for voting; default {DEFAULT_SKIP_BOUND})")
    parser.add_argument("--no-debug",     action="store_true")
    # Hough pass (Step 6b) — continuous trail detection
    parser.add_argument("--hough-residual",  type=int,   default=DEFAULT_HOUGH_RESIDUAL,
                        help=f"Brightness above median to trigger Hough pixel (default {DEFAULT_HOUGH_RESIDUAL})")
    parser.add_argument("--hough-votes",     type=int,   default=DEFAULT_HOUGH_VOTES,
                        help=f"HoughLinesP collinear-pixel threshold (default {DEFAULT_HOUGH_VOTES})")
    parser.add_argument("--hough-min-len",   type=int,   default=DEFAULT_HOUGH_MIN_LEN,
                        help=f"Min Hough line length px at ref width (default {DEFAULT_HOUGH_MIN_LEN})")
    parser.add_argument("--hough-max-gap",   type=int,   default=DEFAULT_HOUGH_MAX_GAP,
                        help=f"Max gap in Hough line px (default {DEFAULT_HOUGH_MAX_GAP})")
    parser.add_argument("--hough-angle-tol", type=float, default=DEFAULT_HOUGH_ANGLE_TOL,
                        help=f"Cross-frame angle tolerance degrees (default {DEFAULT_HOUGH_ANGLE_TOL})")
    parser.add_argument("--hough-rho-tol",   type=int,   default=DEFAULT_HOUGH_RHO_TOL,
                        help=f"Cross-frame rho tolerance px at ref width (default {DEFAULT_HOUGH_RHO_TOL})")
    parser.add_argument("--no-hough",        action="store_true",
                        help="Disable Hough continuous-trail pass")
    parser.add_argument("--star-angle-tol",  type=float, default=5.0,
                        help="Reject Hough lines within this many degrees of star trail direction (default 20)")
    parser.add_argument("--snap-rad",        type=int, default=30,
                        help="Snap repair centroid to local brightness peak within this radius px "
                             "using residual image (aligned-clean) so stars cancel out; 0=off, default 15)")
    parser.add_argument("--median-batch",    type=int, default=DEFAULT_MEDIAN_BATCH,
                        help=f"Wider frame window for clean-sky median (0=same as --batch, e.g. 40 fixes "
                             f"contamination when a trail spans >50%% of --batch frames; default {DEFAULT_MEDIAN_BATCH})")
    args = parser.parse_args()

    input_dir   = Path(args.input_dir)
    output_dir  = Path(args.output_dir)
    debug_dir   = output_dir / "debug" if not args.no_debug else None
    cleaned_dir = output_dir / "cleaned_photos"
    for d in [d for d in [debug_dir, cleaned_dir] if d]:
        d.mkdir(parents=True, exist_ok=True)

    frame_files = load_frame_files(input_dir, args.start, args.batch)
    if len(frame_files) < 3:
        print(f"Need ≥3 frames (got {len(frame_files)})"); sys.exit(1)

    sc       = None   # set after first frame loaded
    band_px  = None

    print(f"astro_clean_v4 (motion-vector): {len(frame_files)} frames  "
          f"thresh={args.thresh}  min_move={args.min_move}  max_move={args.max_move}  "
          f"bin={args.bin_size}px  min_votes={args.min_votes}  band={args.band_width}px")

    # ── Load ───────────────────────────────────────────────────────────────────
    print(f"\nLoading {len(frame_files)} frames...", end=" ", flush=True)
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"\nERROR: cannot read {fp.name}"); sys.exit(1)
        frames.append(img)
    print("done")
    h, w = frames[0].shape[:2]
    sc      = w / 5472.0
    band_px = max(10, int(args.band_width * sc))
    print(f"Image: {w}×{h}  scale={sc:.3f}  band={band_px}px")
    t_total = time.time()

    # ── Step 1: Align ──────────────────────────────────────────────────────────
    print("\nStep 1 — aligning frames")
    aligned, star_angle_deg = align_batch(frames)

    # ── Step 2: Clean sky (for repair) ────────────────────────────────────────
    print("\nStep 2 — building clean-sky background")
    median_batch = args.median_batch if args.median_batch > 0 else args.batch
    if median_batch > args.batch:
        # Load a wider set of frames centred on the detection batch for the median.
        # The batch covers frames [start, start+batch). Centre the median window on
        # the same midpoint so we get equal context on both sides.
        batch_mid  = args.start + args.batch // 2
        med_half   = median_batch // 2
        med_start  = max(0, batch_mid - med_half)
        med_files  = load_frame_files(input_dir, med_start, median_batch)
        print(f"  median window: {len(med_files)} frames starting {med_files[0].name} "
              f"(detection batch={args.batch})")
        med_frames = []
        for fp in med_files:
            img = cv2.imread(str(fp))
            if img is None:
                print(f"\nWARN: cannot read {fp.name} — skipping")
                continue
            med_frames.append(img)
        med_aligned, _ = align_batch(med_frames)
        clean_rgb = build_clean_sky(med_aligned, args.pct)
        del med_frames, med_aligned
    else:
        clean_rgb = build_clean_sky(aligned, args.pct)

    # ── Step 3: Per-frame candidate detection (split by channel) ──────────────
    print("\nStep 3 — detecting candidate dash centroids (color + gray channels)")
    n = len(aligned)

    # Background gradient normalisation: boost diff signal in bright rows
    bg_norm = compute_bg_norm(clean_rgb)
    bright_rows = int((bg_norm > 1.05).sum())
    _bg_gray = cv2.cvtColor(clean_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    _row75 = np.percentile(_bg_gray, 75, axis=1)
    _ref = max(float(np.percentile(_row75, 20)), 5.0)
    print(f"  bg gradient: {bright_rows} rows boosted "
          f"(max boost {bg_norm.max():.2f}×, ref={_ref:.1f})")

    color_cents_all: List[List[Tuple[int, int]]] = []
    gray_cents_all:  List[List[Tuple[int, int]]] = []
    for i, fp in enumerate(frame_files):
        prev1 = aligned[i - 1] if i > 0   else None
        next1 = aligned[i + 1] if i < n-1 else None
        cc, gc = detect_candidates(aligned[i], prev1, next1,
                                   args.thresh, args.min_px, args.edge_margin,
                                   bg_norm)
        color_cents_all.append(cc)
        gray_cents_all.append(gc)
        sys.stdout.write(f"\r  {fp.name}  {len(cc):4d} color  {len(gc):4d} gray  ")
        sys.stdout.flush()
    print()

    # ── Step 4: Cross-frame motion-vector pairs (per channel) ─────────────────
    print("\nStep 4 — computing cross-frame motion vectors")
    min_move = max(5,   int(args.min_move * sc))
    max_move = max(100, int(args.max_move * sc))
    bin_size = max(2,   int(args.bin_size * sc))

    all_color_pairs: List[Tuple[int, int, int, int, int]] = []
    all_gray_pairs:  List[Tuple[int, int, int, int, int]] = []
    for i in range(n - 1):
        cp = compute_motion_pairs(color_cents_all[i], color_cents_all[i + 1], min_move, max_move)
        gp = compute_motion_pairs(gray_cents_all[i],  gray_cents_all[i + 1],  min_move, max_move)
        for cx, cy, dx, dy in cp:
            all_color_pairs.append((i, cx, cy, dx, dy))
        for cx, cy, dx, dy in gp:
            all_gray_pairs.append((i, cx, cy, dx, dy))
        sys.stdout.write(f"\r  pair {i}↔{i+1}: {len(cp):5d} color  {len(gp):5d} gray  ")
        sys.stdout.flush()
    print(f"\n  total: {len(all_color_pairs)} color pairs + {len(all_gray_pairs)} gray pairs")

    # ── Step 5: Cluster motion vectors ────────────────────────────────────────
    print("\nStep 5 — clustering motion vectors")
    # min_votes is an absolute threshold: "minimum number of frame pairs in which
    # a trail must appear." Airplane trail votes = (trail_duration_frames - 1),
    # which is independent of total batch size. Do NOT scale linearly with n_pairs —
    # that would kill detection of 15-frame trails in 100-frame batches.
    n_pairs = n - 1
    min_votes_scaled = args.min_votes
    min_ch_votes_scaled = args.min_ch_votes
    print(f"  min_votes={min_votes_scaled}  min_ch_votes={min_ch_votes_scaled}  "
          f"({n_pairs} pairs)")
    peaks = cluster_motion_vectors(
        all_color_pairs, all_gray_pairs,
        bin_size, min_votes_scaled, min_ch_votes_scaled, max_move)

    # Print vote distribution to help calibrate min_votes
    if peaks:
        all_votes_vals = [v for _, _, v in peaks]
        for threshold_show in [10, 15, 20, 25, 30]:
            n_above = sum(1 for v in all_votes_vals if v >= threshold_show)
            print(f"  peaks with votes ≥ {threshold_show:3d}: {n_above}")

    print(f"  {len(peaks)} peaks at min_votes={args.min_votes}")
    peak_dirs = [(float(dx), float(dy)) for dx, dy, _ in peaks]
    print(f"  top 10:")
    for i, (dx, dy, votes) in enumerate(peaks[:10]):
        print(f"    [{i}] dx={dx:+5d} dy={dy:+5d}  votes={votes}")

    # ── Step 6: Collect trail centroids with chain verification ───────────────
    print("\nStep 6 — collecting trail centroids (chain verification)")

    # Build per-frame candidate arrays for fast lookahead lookup.
    # Use ALL detected candidates (color + gray) so partial detections
    # in either channel can still provide a chain confirmation.
    cands_arr: List[np.ndarray] = []
    for i in range(n):
        cands = list(set(color_cents_all[i]) | set(gray_cents_all[i]))
        cands_arr.append(np.array(cands, dtype=np.float32) if cands
                         else np.zeros((0, 2), dtype=np.float32))

    # Tolerance for chain lookahead: larger than bin_size to tolerate
    # slight strobe-timing variation between flashes.
    chain_tol    = max(bin_size * 2, int(50 * sc))
    chain_tol_sq = float(chain_tol) ** 2

    trail_cents_per_frame: List[set] = [set() for _ in range(n)]
    all_pairs = all_color_pairs + all_gray_pairs
    n_chain_kept = 0
    n_chain_dropped = 0

    edge_margin_px = max(5, int(args.edge_margin * sc))
    # Store per-peak verified centroids so gap fill can run after cluster filter
    all_peak_vf: List[Tuple[int, int, List[set]]] = []

    for dx_peak, dy_peak, _ in peaks:
        half = bin_size
        # Collect all centroids claimed by this peak
        peak_claimed: List[set] = [set() for _ in range(n)]
        for frame_idx, cx, cy, dx, dy in all_pairs:
            if abs(dx - dx_peak) <= half and abs(dy - dy_peak) <= half:
                peak_claimed[frame_idx].add((cx, cy))
                if frame_idx + 1 < n:
                    peak_claimed[frame_idx + 1].add((cx + dx, cy + dy))

        # Chain-verify each claimed centroid
        peak_verified: List[set] = [set() for _ in range(n)]
        for i in range(n):
            for (cx, cy) in peak_claimed[i]:
                if chain_verified(i, cx, cy, dx_peak, dy_peak,
                                  n, cands_arr, chain_tol_sq, args.look_ahead):
                    peak_verified[i].add((cx, cy))
                    n_chain_kept += 1
                else:
                    n_chain_dropped += 1

        # Store for post-cluster gap fill
        if args.max_gap_fill > 0:
            all_peak_vf.append((dx_peak, dy_peak, [set(s) for s in peak_verified]))

        # Merge into shared pool
        for i in range(n):
            trail_cents_per_frame[i].update(peak_verified[i])

    print(f"  chain verification (look_ahead={args.look_ahead}, tol={chain_tol}px): "
          f"{n_chain_kept + n_chain_dropped} candidates → "
          f"{n_chain_kept} kept, {n_chain_dropped} dropped")

    # ── Spatial density filter ─────────────────────────────────────────────────
    # Real airplane dashes are accompanied by other dashes from the same trail.
    # Isolated single dots are coincidental false-positive matches.
    # Keep a centroid only if it has >=min_cluster other trail centroids within
    # cluster_rad pixels in the same frame.
    cluster_rad_px = max(30, int(args.cluster_rad * sc))
    if not args.no_cluster_filter:
        total_before = sum(len(s) for s in trail_cents_per_frame)
        for i in range(n):
            tc = list(trail_cents_per_frame[i])
            filtered = filter_small_clusters(tc, cluster_rad_px, args.min_cluster,
                                              args.min_elongation)
            trail_cents_per_frame[i] = set(filtered)
        total_after = sum(len(s) for s in trail_cents_per_frame)
        print(f"  cluster filter (connect_rad={cluster_rad_px}px, min_size={args.min_cluster}, min_elong={args.min_elongation}): "
              f"{total_before} → {total_after} centroids "
              f"({total_before - total_after} small-cluster dots removed)")
    else:
        total_after = sum(len(s) for s in trail_cents_per_frame)
        print(f"  cluster filter disabled — {total_after} centroids")

    # ── Temporal persistence filter ────────────────────────────────────────────
    # Reject centroids in image regions that have trail hits in > max_prev of frames.
    # Persistent ground lights: prevalence ≈ 100%.  Real airplane trails: ≈ 5–25%.
    cell_size_px = max(20, int(args.cell_size * sc))
    if not args.no_persist_filter:
        total_before = sum(len(s) for s in trail_cents_per_frame)
        trail_cents_per_frame = filter_persistent_regions(
            trail_cents_per_frame, h, w, n,
            cell_size_px, args.max_prev)
        total_after = sum(len(s) for s in trail_cents_per_frame)
        print(f"  persistence filter (cell={cell_size_px}px, max_prev={args.max_prev:.0%}): "
              f"{total_before} → {total_after} centroids "
              f"({total_before - total_after} persistent-region dots removed)")

    # ── Per-peak gap fill (post cluster+persistence filter) ───────────────────
    # The cluster filter may remove sparse middle-frame detections from a real
    # trail (e.g. only 2-3 dashes detected when strobe phase is weak).
    # For each confirmed peak, find frames where its verified centroids were
    # entirely removed by the cluster filter, then project from the nearest
    # surviving frames on both sides and add synthetic repair centroids directly
    # (bypassing cluster filter — the two-sided trajectory verification is
    # sufficient evidence).
    if args.max_gap_fill > 0 and all_peak_vf:
        n_gap_filled = 0
        # Minimum number of frames a peak must have surviving centroids across
        # to be eligible for gap fill.  Real multi-frame trails span many frames;
        # spurious low-vote peaks hit only 1-3 frames.
        min_survived_span = max(3, n // 5)   # at least 20% of batch, min 3

        for dx_p, dy_p, pv_frames in all_peak_vf:
            # Find which frames still have this peak's centroids after filtering.
            # Require >= 2 survived centroids per frame so that a single centroid
            # shared with another peak's cluster doesn't count as a "survived" frame.
            survived = []
            for i in range(n):
                s = pv_frames[i] & trail_cents_per_frame[i]
                survived.append(s if len(s) >= 2 else set())

            # Skip peaks with too few qualifying frames — not a real multi-frame trail
            if sum(1 for s in survived if s) < min_survived_span:
                continue
            sv_arrs = [
                np.array(list(s), dtype=np.float32) if s
                else np.empty((0, 2), dtype=np.float32)
                for s in survived
            ]

            for _gpass in range(args.max_gap_fill):
                added = 0
                for i in range(n):
                    # Only try frames where peak had verified hits but all were
                    # removed by cluster filter (genuinely sparse detection)
                    if len(survived[i]) > 0:
                        continue
                    if len(pv_frames[i]) == 0:
                        continue  # peak had no candidates here at all

                    synthetic: set = set()
                    for back in range(1, args.max_gap_fill + 1):
                        si = i - back
                        if si < 0 or len(sv_arrs[si]) == 0:
                            continue
                        for fwd in range(1, args.max_gap_fill + 1):
                            vi = i + fwd
                            if vi >= n or len(sv_arrs[vi]) == 0:
                                continue
                            for cx, cy in sv_arrs[si]:
                                px = int(round(float(cx) + back * dx_p))
                                py = int(round(float(cy) + back * dy_p))
                                if not (edge_margin_px < px < w - edge_margin_px and
                                        edge_margin_px < py < h - edge_margin_px):
                                    continue
                                vx = float(cx) + (back + fwd) * dx_p
                                vy = float(cy) + (back + fwd) * dy_p
                                if _has_nearby(sv_arrs[vi], vx, vy, chain_tol_sq):
                                    synthetic.add((px, py))

                    if synthetic:
                        trail_cents_per_frame[i].update(synthetic)
                        survived[i].update(synthetic)
                        sv_arrs[i] = np.array(list(survived[i]), dtype=np.float32)
                        added += len(synthetic)
                        n_gap_filled += len(synthetic)

                if added == 0:
                    break

        total_after = sum(len(s) for s in trail_cents_per_frame)
        print(f"  gap fill (per-peak, {args.max_gap_fill} passes): "
              f"+{n_gap_filled} synthetic centroids  →  {total_after} total")

    # ── Step 6b: Hough pass — continuous trail detection ──────────────────────
    hough_dirs: List[Tuple[float, float]] = []
    if not args.no_hough:
        print("\nStep 6b — detecting continuous trails (Hough on background residual)")
        cont_cents, hough_dirs = detect_continuous_trails(
            frames, clean_rgb, n, h, w, sc,
            residual_thresh = args.hough_residual,
            hough_votes     = args.hough_votes,
            min_line_len    = args.hough_min_len,
            max_line_gap    = args.hough_max_gap,
            angle_tol_deg   = args.hough_angle_tol,
            rho_tol         = args.hough_rho_tol,
            edge_margin     = args.edge_margin,
            band_px         = band_px,
            star_angle_deg  = star_angle_deg,
            star_angle_tol  = args.star_angle_tol,
        )
        for i in range(n):
            trail_cents_per_frame[i].update(cont_cents[i])

    # ── Final isolation filter ─────────────────────────────────────────────────
    # Remove small isolated clusters — real trails produce chains of 4+ adjacent
    # circles; stray dots and dot-pairs are false positives.
    # Use 2*band_px as connectivity radius and no elongation requirement here
    # (elongation was already enforced on the motion-vector clusters).
    total_before_iso = sum(len(s) for s in trail_cents_per_frame)
    for i in range(n):
        tc = list(trail_cents_per_frame[i])
        filtered = filter_small_clusters(tc, band_px * 2, min_cluster_size=4,
                                         min_elongation=2.0)
        trail_cents_per_frame[i] = set(filtered)
    n_isolated = total_before_iso - sum(len(s) for s in trail_cents_per_frame)
    if n_isolated:
        total_after = sum(len(s) for s in trail_cents_per_frame)
        print(f"  isolation filter: removed {n_isolated} small clusters → {total_after} centroids")

    # ── Second persistence filter (post-Hough) ─────────────────────────────────
    # The Hough pass runs after the first persistence filter, so persistent
    # features (Pleiades, ground lights) can slip through. Re-run it now.
    if not args.no_persist_filter:
        total_before = sum(len(s) for s in trail_cents_per_frame)
        trail_cents_per_frame = filter_persistent_regions(
            trail_cents_per_frame, h, w, n, cell_size_px, args.max_prev)
        total_after = sum(len(s) for s in trail_cents_per_frame)
        removed = total_before - total_after
        if removed:
            print(f"  post-Hough persistence filter: removed {removed} persistent-region dots → {total_after} centroids")

    # Combine motion-vector peak directions + Hough directions for line-connect
    all_trail_dirs: List[Tuple[float, float]] = peak_dirs + hough_dirs

    # ── Step 7: Repair ────────────────────────────────────────────────────────
    sb = args.skip_boundary
    snap_rad_px = max(0, int(args.snap_rad * sc))
    print(f"\nStep 7 — repairing frames (skipping first/last {sb} from output)"
          + (f"  snap_rad={snap_rad_px}px" if snap_rad_px > 0 else "  snap disabled"))
    n_repaired  = 0
    total_trail = 0

    for i, (fp, img) in enumerate(zip(frame_files, frames)):
        tc = list(trail_cents_per_frame[i])
        if snap_rad_px > 0:
            aligned_gray_snap = cv2.cvtColor(aligned[i], cv2.COLOR_BGR2GRAY)
            clean_gray_snap   = cv2.cvtColor(clean_rgb, cv2.COLOR_BGR2GRAY)
            residual_snap = np.abs(
                aligned_gray_snap.astype(np.int16) - clean_gray_snap.astype(np.int16)
            ).clip(0, 255).astype(np.uint8)
            tc = snap_to_brightness_peak(tc, residual_snap, snap_rad_px)
        mask     = build_repair_mask(h, w, tc, band_px,
                                     trail_directions=all_trail_dirs)
        trail_px = int((mask > 0).sum())
        total_trail += trail_px

        skip = (sb > 0) and (i < sb or i >= n - sb)

        if not skip:
            cleaned = repair_frame(img, mask, clean_rgb)
            cv2.imwrite(str(cleaned_dir / fp.name), cleaned)
            if trail_px > 0:
                n_repaired += 1

        if debug_dir:
            label = f"{fp.name}{'  [boundary-skipped]' if skip else ''}"
            dbg = draw_debug(img, mask, tc, label)
            cv2.imwrite(str(debug_dir / fp.name), dbg)

        sys.stdout.write(
            f"\r  {fp.name}{'*skip*' if skip else '      '}  "
            f"dashes={len(tc):4d}  trail_px={trail_px:7d}  ")
        sys.stdout.flush()

    elapsed = time.time() - t_total
    print(f"\n\nDone in {elapsed:.0f}s  ({elapsed/len(frames):.1f}s/frame)")
    print(f"  {n_repaired}/{len(frames)} frames repaired")
    print(f"  avg trail px/frame: {total_trail // len(frames)}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
