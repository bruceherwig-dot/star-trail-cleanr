# Astrophotography Airplane & Satellite Trail Removal — v4

## Release checklist (ALWAYS do all steps before tagging)
1. Update `CHANGELOG.md` with the new version and a plain-English summary of changes
2. Commit `CHANGELOG.md` together with the code changes (same commit or immediately before tag)
3. Tag the release (`git tag vX.XX-beta`)
4. Push commits and tag (`git push && git push origin vX.XX-beta`)
5. Wait for GitHub Actions build to complete, then post download links to user

## Active script
`astro_clean_v4.py` — Motion-vector histogram + Hough residual two-pass detection.

## Usage
```bash
# Single batch (20 frames recommended max due to star rotation)
python3 astro_clean_v4.py /path/to/frames -o /path/to/output --start 0 --batch 20

# Full dataset run (overlapping batches, step=16, covers all frames)
python3 run_full_batch.py  # or use the inline script below
```

## Full dataset batch script
```python
import subprocess, glob
frames = sorted(glob.glob("/path/to/frames/IMG_*.jpg"))
n = len(frames)
starts = list(range(0, n - 20 + 1, 16))
if starts[-1] + 20 < n:
    starts.append(n - 20)
for s in starts:
    subprocess.run(["python3", "astro_clean_v4.py", "/path/to/frames",
                    "-o", "/path/to/output", "--start", str(s), "--batch", "20"])
```

## Frame data
- Originals: `/Users/bruceherwig/Documents/frames/extra/` (1559 JPEGs, IMG_4030–IMG_5588)
- Final cleaned output: `/Users/bruceherwig/Documents/frames_v4_cleaned_full/` (1555 frames, 6.5GB)
- Batch size: 20 frames, step 16 → 98 batches for 1559 frames → 66 minutes total

## v4 Algorithm

**Step 1**: Phase-correlate all frames to middle frame (cv2.phaseCorrelate).
- Stars become fixed; airplanes move.
- Max batch size: 20 frames (~60s of shooting). Star rotation over 5+ minutes causes false motion vector clusters.

**Step 2**: Per-pixel temporal percentile (50th) across aligned frames → `clean_rgb` repair background.

**Step 3**: Per-frame candidate detection (color + gray channels).
- Color: `min(color_adv_diff_vs_prev, color_adv_diff_vs_next) > thresh`
- Gray: `min(gray_diff_vs_prev, gray_diff_vs_next) > motion_thresh`
- Connected components → centroids.

**Step 4**: Motion vector histogram.
- For each consecutive frame pair: compute all centroid-to-centroid vectors within [min_move, max_move].
- 2D histogram of (dx, dy) vectors → peaks = real airplane trajectories.
- `min_votes` (default=2): minimum histogram votes to confirm a trajectory. NOT scaled by batch size.

**Step 5**: Chain verification + gap fill.
- Verify each peak: centroids must form a spatial chain across frames.
- Rules: (a) 2 forward hits, (b) 2 backward hits, (c) 1 fwd + 1 bwd, (d) [boundary only] 1 hit either side.
- Boundary relaxation: frames within 2 of batch start/end only need 1 hit — handles trails entering/exiting.
- Gap fill: project trajectory into frames where detection failed (strobes skip beats).

**Step 6**: Cluster filter + persistence filter.
- Cluster filter: `min_size=4, min_elong=4.0` — removes isolated dots.
- Persistence filter: regions active in >50% of batch frames are persistent (lens artifacts) → rejected.

**Step 6b**: Hough pass on background residual (NEW in v4).
- Per frame: `frame_gray - clean_gray` → threshold → HoughLinesP.
- Cross-validate: line must appear in frame N-1 or N+1 (same angle ±15°, same rho ±200px).
- Purpose: catches continuous (non-flashing) airplane nav lights invisible to motion-vector pass.
- Key params: `hough_residual=10, hough_votes=25, hough_min_len=200px, hough_max_gap=40`.
- Star angle filter: reject Hough lines within ±5° of computed star trail direction.
- Isolation filter after: `filter_small_clusters(min_size=4)` removes stray dots.
- Post-Hough persistence filter: catches Pleiades-style star clusters that pass cross-frame validation.

**Step 7**: Repair detected trail pixels with temporal median (`clean_rgb`).
- Snap centroids to nearest bright pixel in residual (`|aligned - clean|`) within snap_rad before repair.
- Residual snap: stars cancel out → snap finds actual trail pixel, not nearby star.

## Current defaults (v4 best known state)
```
thresh          = 20     # color channel detection threshold
motion_thresh   = 30     # gray channel detection threshold
min_votes       = 2      # min histogram votes (2 = 3-frame trail minimum)
min_ch_votes    = 1      # min votes per channel (1 = appears in both channels at least once)
band_px         = 20     # repair band half-width (px at 5472px ref)
edge_margin     = 200    # ignore detections within 200px of edges
min_px          = 5      # min component pixels to keep
connect_rad     = 80     # cluster connection radius (px)
max_prev        = 50     # persistence filter threshold (% of frames)

# Hough pass (Step 6b)
hough_residual  = 10     # brightness above temporal median to threshold
hough_votes     = 25     # HoughLinesP vote threshold
hough_min_len   = 200    # min line length (px at 5472px ref)
hough_max_gap   = 40     # max gap in HoughLinesP segment
hough_angle_tol = 15.0   # degrees cross-frame angle tolerance
hough_rho_tol   = 200    # px cross-frame rho tolerance
star_angle_tol  = 5.0    # reject Hough lines within ±5° of star trail direction

# Repair (Step 7)
snap_rad        = 30     # snap centroid to nearest bright residual pixel within ±30px
                         # uses |aligned-clean| so stars cancel out; 0=off
median_batch    = 40     # wider temporal window for clean-sky median (recommended for
                         # trails spanning >50% of batch; 0 = same as --batch)
```

## Validation (wide 4-batch test, frames 4088–4155)
- **Orange trail 4107–4120**: fully detected and cleaned ✓
- **Second airplane 4137–4141**: fully detected ✓
- **Snap alignment**: mean centroid offset 4–10px (measured vs residual bright pixels); 0 centroids >40px off ✓
- **False positives**: occasional 4-dash blobs in quiet frames (min_votes=2 more permissive); repair footprint ~2000px² each — negligible in composite ✓
- **No star trail FPs** confirmed in debug frames ✓
- **4121-4122 miss**: third trail at batch boundary — handled by step=16 overlap in full run

## Previous full-run validation (v4 prior parameters)
- **Overall**: ~90% of airplane trails removed cleanly
- **Zero false positives** in main sky area confirmed
- **Correctly handled**: continuous orange trail 4114–4120
- **Correctly handled**: very bright double-track trail at 4930–4932 (96% reduction)
- **Known miss**: partial trail segments in dotted pattern (dashes alternately missed)
- **Moon rising** (frames ~4928+): correctly NOT touched (persistent non-trail source)

## Known limitations (v4)
1. **Continuous trails near moon/bright background**: The Hough pass works, but very bright backgrounds contaminate the temporal median → repair seam visible in cleaned frames. Residual pixels still show in lighten composite.
2. **Faint trails**: Below brightness threshold — out of scope by design. Only trails bright enough to visibly affect the composite need removal.
3. **Repair quality for very bright trails**: band_px may not cover full trail width → seam artifacts.
4. **Camera independence**: `hough_residual=15` is absolute. Should be relative to sensor noise floor (noise_sigma * multiplier) for camera-agnostic operation.

## v5 Ideas (not yet implemented)
1. **Adaptive Hough residual threshold**: `max(10, noise_sigma * 2.5)` per-frame instead of fixed 15.
2. **Wider repair band for bright trails**: scale band_px with local trail brightness.
3. **Repair quality**: use inpainting (cv2.inpaint) instead of flat median fill to avoid seam artifacts.
4. **Motion vector clustering (from v3 NEXT section)**: detect at low threshold (15–20), cluster vectors with DBSCAN → handles flashing AND continuous trails uniformly.
5. **Partial trail detection**: improve gap-fill to recover alternating missed dashes in partial trails.
6. **Camera-agnostic thresholds**: all fixed pixel thresholds → adaptive sigma-based.
7. **Snap centroids to local brightness peak** (IMPLEMENTED in v4): `--snap-rad 30` (default on). For each centroid, searches ±30px window in aligned-frame gray channel and shifts repair center to the local maximum. Centers repair on actual white fuselage trail, not the colored nav light offset.
8. **Wider median window (`--median-batch`)** (IMPLEMENTED in v4 as experiment): when a trail spans >50% of the 20-frame detection batch, the temporal median at that pixel IS the trail → repair fails. Using a 40–80 frame window for the median while keeping motion vector detection on 20 frames resolves contamination. Confirmed with frames 4030–4047 (18-frame trail). Need to evaluate whether this should be on by default in the full run (cost: ~2× Step 2 time).
9. **Hough cross-frame validation too strict for long trails** (FIXED in v4): Root cause was rho sign flip — when HoughLinesP returns a segment with reversed direction (dx<0), the normal vector negates and rho changes sign, making |rho_N - rho_{N+1}| jump by ~2× the line's distance from origin. Fix: canonicalize (dx,dy) direction before computing angle/rho so the comparison is consistent across frames.
10. **Distinguish star clusters from trail remnants in composite analysis**: persistent ~1500px anomalies across 50–100 frames in same region = star cluster (e.g. Pleiades), not missed trail. Algorithm correctly ignores these (persistence filter). Post-run analysis scripts must account for this to avoid false alarm reports.
11. **Variable repair band width based on detected trail size**: fixed `band_px=20` leaves seam artifacts on bright/wide trails (e.g. low-flying aircraft). Use perpendicular component width at detection time to set band per-trail: `band_px = clamp(perp_width // 2 + 10, min=15, max=60)`. Perpendicular width = component bounding box measured orthogonal to motion vector direction. Floor prevents over-erasing faint trails; ceiling prevents blowing out large sky regions. Also partially compensates for slightly curved trails (banking aircraft) where a narrow fixed band misses edges.
13. **Visual AI trail detection for curved trails (planned)**: Wide angle and fisheye lenses produce naturally curved trails due to lens distortion — a straight flight path becomes a gentle arc (wide angle) or a dramatically curved path (fisheye), especially away from the frame center. The current motion vector histogram assumes a fixed (dx, dy) per frame pair and cannot handle trails that curve across the frame. Fisheye is the extreme case: star trails near the edge curve sharply (see example image), and an airplane trail could follow almost any arc depending on where it crosses the distortion field. Additionally, phase correlation (Step 1) assumes translation-only alignment — this breaks down on fisheye frames where stars near the center shift differently than stars near the edge, degrading detection before curved trails are even considered. Planned approach: train a pixel-level segmentation model (outputs a trail/no-trail mask per pixel) on astrophotography frames including wide angle and fisheye examples. Model learns "this is a trail" from examples — no need to label straight vs curved separately, just label trail pixels vs sky. The before/after pairs from the existing 1559-frame v4 cleaned dataset are natural training data: input frame = uncleaned, label = difference mask between before and after. Model runs as an additional detection pass alongside the existing algorithm (not a replacement), feeding pixel masks into the same repair step. Model must be small enough to bundle with PyInstaller. Framework TBD (PyTorch preferred for bundling). Training data: existing sequence plus fisheye and wide angle examples.
12. **16-bit TIF support**: Strategy: "detect in 8-bit, repair in 16-bit". Load with `cv2.IMREAD_UNCHANGED`, detect bit depth from `img.dtype`. If 16-bit: create 8-bit working copies (`img >> 8`) for all detection/alignment — no changes needed to detection code since it already uses int16 intermediates. Keep originals in 16-bit for `build_clean_sky` (temporal median → uint16 output) and `repair_frame` (pixel replacement at native depth). `cv2.imwrite` already handles uint16 TIF correctly. Thresholds stay the same (they operate on 8-bit copies). Scope: ~5 targeted change sites — 2 imread calls, 1 depth-detection block, `build_clean_sky` dtype param, `repair_frame` dtype passthrough. Est. 40–60 lines.
  - **Confirmed broken (2026-03-21)**: `cv2.imread` without `IMREAD_UNCHANGED` on 16-bit TIF returns uint16 in OpenCV 4.x. All detection thresholds (thresh=20, motion_thresh=30, hough_residual=10) are calibrated for 8-bit (0–255). In uint16 space these are noise-level — sky shot noise and star scintillation easily exceed 20 counts → everything flagged → stars removed along with trails. **16-bit TIFs are unsupported until v5 #12 is implemented. Users must export 8-bit for now.**
14. **Resolution-aware parameter scaling (HIGH PRIORITY)**: Current `sc = w / 5472.0` only corrects width-based linear parameters. Fails completely for cameras with different aspect ratios or much higher megapixel counts. Confirmed broken on 5504×8256 (45MP portrait) camera: `sc ≈ 1.006` (width barely changed) but total pixels are 2.27× more → `min_px=5` passes 37× more noise components → 1878 gray detections per batch (vs normal ~30–80) → 5+ hour run instead of ~25 min.
  - **Fix**: use area-based and linear-distance-based scale factors separately:
    ```python
    ref_w, ref_h = 5472, 3648          # 20MP iPhone reference
    sc_linear = sqrt(w * h / (ref_w * ref_h))   # ~1.51 for 45MP
    sc_area   = (w * h) / (ref_w * ref_h)       # ~2.27 for 45MP
    min_px      = int(5   * sc_area)    # area-scaled: ~11px at 45MP
    min_move    = int(15  * sc_linear)  # linear: ~23px
    max_move    = int(600 * sc_linear)  # linear: ~906px
    connect_rad = int(80  * sc_linear)  # linear: ~121px
    edge_margin = int(200 * sc_linear)  # linear: ~302px
    snap_rad    = int(30  * sc_linear)  # linear: ~45px
    band_px     = int(20  * sc_linear)  # linear: ~30px
    ```
  - All these currently use `w/5472` (width-only) or no scaling at all (`min_px`, `edge_margin`, `snap_rad` are unscaled). The `ref_h=3648` should be confirmed from the original dataset (IMG_4030 dimensions).

## Why 20-frame batches (not larger)
- Star rotation over 60 seconds: <2px error → negligible for gap=1 detection.
- Star rotation over 5 minutes (100 frames): 5–10px at corners → creates false motion vector clusters.
- Phase correlation only corrects TRANSLATION, not rotation.
- Larger batches produce catastrophic false positives. 20 frames is the practical limit.

## v1 / v2 / v3 reference
- v1: `astro_clean.py` — stacked temporal diff + Hough. No phase alignment.
- v2: `astro_clean_v2.py` — three-frame comparison + connected component PCA. Phase-aligned.
- v3: `astro_clean_v3.py` — per-frame Hough on centroid images. Zero false positives but many misses.
- v4: `astro_clean_v4.py` — motion vector histogram + Hough residual. **Current best.**
