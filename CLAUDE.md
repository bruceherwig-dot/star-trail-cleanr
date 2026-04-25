# Astrophotography Airplane & Satellite Trail Removal — v5

## Standing instructions (apply to every response)
- **NEVER claim what our code does or doesn't do without reading the code first.** Bruce is not a programmer. He believes every technical claim I make and acts on it (defers features, reshapes plans). Before saying "our pipeline uses X," "we strip Y," "we can't do Z because of library W" — grep or Read the actual file. If I catch myself hedging with "I think" or "given that we use" about our own code, STOP and check. Past examples where I got this wrong: told Bruce EXIF couldn't be preserved when the code already had PIL pass-through, told him the tile-merge step was the polygon step, told him the detection grid matched the training grid when it didn't. Full rule and trip-wire phrases in `feedback_never_guess_about_our_code.md`.
- **Before giving any recommendation or advice:** verify the reasoning applies to the actual system in use. Check relevant source files or context before advising. Flag uncertainty explicitly with "I think" or "I'm not sure" — do not present unverified reasoning as fact.
- **Do NOT touch `version.txt` during development.** The per-edit .001 bump rule was retired 2026-04-24. `version.txt` now matches the shipped GitHub tag exactly (e.g., `1.7` for `v1.7-beta`) and is updated ONLY at release time, once, to match the new tag. See `feedback_version_scheme.md` for the ship procedure.

## Sacred data — NEVER touch
- **Original source images** — never delete, overwrite, or modify. Convert/copy alongside, never replace.
- **Reviewed LabelMe labels** — never regenerate or overwrite manually reviewed JSON files. Back up first.

## Current pipeline (v5) — 2 steps
1. **Detect** — per-frame YOLO/SAHI tiled inference with sky mask and component filtering
2. **Repair** — Star Bridge morph from neighbor frames N-1/N+1, black fill fallback
   - Black fill (zeros) is transparent in lighten-max stacks — real stars from other frames always win
   - No alignment step, no clean-sky background step (removed — 2x speedup, no quality loss)

## Key technical facts
- Star motion between frames is NOT a simple rotation — only true with pole-centered framing + no lens distortion. Off-axis framing and wide-angle lenses create non-uniform motion fields.
- Hot pixels are multi-pixel clusters (~5px wide after demosaic), not single-pixel defects. Detected via color imbalance filter (single-channel Bayer CFA defect → ratio 5-100x).
- GUI (`star_trail_cleanr.py`) chunks frames into 20-frame batches and runs `astro_clean_v5.py` as a subprocess per batch.

## Release checklist (ALWAYS do all steps before tagging)
0. **Run the smoke tests — must be all-green: `python3 tests/run_all.py`**
1. Update `CHANGELOG.md` with the new version and a plain-English summary of changes
2. Commit `CHANGELOG.md` together with the code changes (same commit or immediately before tag)
3. Tag the release (`git tag vX.XX-beta`)
4. Push commits and tag (`git push && git push origin vX.XX-beta`)
5. Watch the GitHub Actions build with `bash scripts/watch_ci.sh` — when it reports "Build succeeded", post the download links to the user. (This is the standard watcher; never invent a new one each release.)

## Smoke test suite (`tests/`)
Regression safety net for Claude's edits — Bruce does not run these himself.
- Run with: `python3 tests/run_all.py` (takes <1 second, no external deps)
- Suite covers: `version.txt` sanity, module-import health, hot-pixel detect+repair on synthetic frames, and CLAUDE.md structural invariants (sacred-data rule, version-bump rule, release checklist, trained-models location, CVAT setup).
- **Run these after any non-trivial edit to `modules/*.py`, `astro_clean_v5.py`, `star_trail_cleanr.py`, or CLAUDE.md itself.** If a test fails, fix the root cause — don't weaken the test.
- These are smoke tests, not quality tests. Visual ML correctness (trail detection quality, repair aesthetics) is still judged by Bruce's eye in Photoshop — the suite only catches structural regressions.

## Active script
`astro_clean_v5.py` — YOLO-based detection + Star Bridge morph repair.

## Key paths
- GUI: `star_trail_cleanr.py` (PySide6 native desktop)
- Detection: `modules/detect_trails.py`
- Repair: `modules/repair.py`
- Hot pixels: `modules/hot_pixels.py`
- Build: `build_helper.py` (PyInstaller)
- Tools: `tools/` folder (inference, training, LabelMe utilities)

## Trained YOLO models
All trained models live on the **local Mac** at `/Users/bruceherwig/Documents/yolo_runs/` — NOT on T7 Shield. Default ultralytics output path.
- Current best: `trail_detector_v11s_tiled/weights/best.pt` — mAP50 box 0.891, mAP50 mask 0.880 (2026-04-07)
- If you think a model is "missing" because it's not in `/Volumes/T7 Shield/AI Projects/Star Trail CleanR/models/`, check `~/Documents/yolo_runs/` first before panicking.

## CVAT setup (annotation review)
- Local Docker instance: `http://localhost:8080`, username `bherwig2`
- Password: stored at `~/.star_trail_cleanr/cvat_credentials` (chmod 600, outside repo). **Never hardcode the password in scripts.**
- Project in CVAT: "Star Trail CleanR" — single label `trail`, each dataset batch = separate Task
- Upload script: `tools/labelme_to_cvat.py` — reads password from credentials file at import time. To run for a new batch, edit only two lines: `IMAGE_AND_JSON_FOLDER` and `CVAT_TASK_ID`.
- Standard CVAT task settings: Image quality 95 (default 70 is too low for thin trail review), subset blank, labels auto-inherited from project.
- Pre-annotation workflow: run `tools/infer_trails.py` with current best model → `tools/masks_to_labelme.py` to convert masks → `tools/labelme_to_cvat.py` to push into CVAT as pre-annotations. Bruce reviews with a head start instead of labeling from scratch.

## Batch constraints
- Max 20 frames per batch (star rotation >5min causes false positives)
- GUI loads ±1 neighbor frame per batch for Star Bridge repair across batch edges

## Version history
- v1: stacked temporal diff + Hough (no alignment)
- v2: three-frame comparison + PCA (phase-aligned)
- v3: per-frame Hough on centroids (zero FPs, many misses)
- v4: motion vector histogram + Hough residual
- v5: YOLO AI detection + Star Bridge morph repair (**current**)
