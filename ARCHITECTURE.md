# How this codebase fits together (START HERE)

Written for someone — person or AI — opening this repository for the first time.
It answers: what runs when a user clicks Clean, which files matter, and which
files are dead weight that will mislead you.

For what the app is *for*, read `README.md`. For how updating works, read
`AUTO_UPDATE.md`. For the working rules of this project, read `CLAUDE.md`.

---

## The one thing to understand first

**The app runs the cleaning engine as a separate program, not as an import.**

`star_trail_cleanr.py` is the window. `astro_clean_v5.py` is the engine. To run
the engine, the app re-runs *its own executable* with a `--cleanr-worker` flag,
and the re-invoked process becomes the engine instead of opening a window.

That looks strange until you know why: in a frozen app there is no Python
interpreter lying around to call. `sys.executable` **is** the app. So "run a
script" has to mean "run myself, in worker mode".

Everything follows from that:

- The two sides share no memory. Instructions go out as command-line arguments;
  progress comes back as lines of text on stdout that the app parses.
- The engine cannot crash the window, and a long run cannot freeze it.
- **The two sides can disagree, and that is this codebase's most dangerous bug
  class.** The app decides which frames to clean, in what order, and writes them
  to a manifest; the engine reads that manifest verbatim. When those two lists
  were built at different moments, every run died with `need >= 3 frames (got 0)`
  and two users lost a day (fixed 2026-08-21, locked by
  `tests/test_manifest_matches_plan.py`).

---

## What happens during one run

1. **Scan the folder** (`star_trail_cleanr.py`, `modules/frame_list.py`)
   List the frames, drop duplicate RAW/JPG/TIFF twins of the same shot, order by
   true capture time rather than filename, measure every file, keep the dominant
   size and report anything skipped.
2. **Plan the batches and write the manifest**
   Up to 20 frames per batch. The manifest is written *after* every filter has
   been applied, so it is exactly the list the plan counts.
3. **Clean each batch** (`astro_clean_v5.py`, one process per batch)
   - *Detect* (`modules/detect_trails.py`, `modules/detect_pipeline.py`): a YOLO
     segmentation model reads each frame in 640x640 tiles; results merge into one
     mask per frame; a painted foreground mask keeps the landscape out of it.
   - *Repair* (`modules/repair.py`): "Star Bridge" takes the same patch of sky
     from the frames either side, morphs between them, and lays that over the
     trail. The stars survive because they genuinely were there a few pixels
     along, one frame earlier and later.
   - Write cleaned copies. **Originals are never modified.**
4. **Build the extras** (`make_share_clip.py`, `modules/share_stacker.py`)
   The star trail stack, the before/after video, the timelapse.
5. **Report** (`modules/usage_report.py`) — one small anonymous record, only if
   the user opted in.

---

## What a run produces (expected outcome)

If a run worked, this is what exists afterwards. **The original photos are never
modified** — everything is written alongside them.

```
<your photo folder>/            your originals, untouched
└── cleaned/                    one cleaned copy per frame, same filenames
    └── STC Extras/             everything else the run made
        ├── STC_cleaned_star_trail.jpg    star trail stacked from the CLEANED frames
        ├── STC_original_star_trail.jpg   the same stack from the ORIGINALS, to compare
        ├── STC_share_video.mp4           short before/after video
        ├── STC_star_trail_*.jpg          trails you built on the Star Trail tab; the
        │                                 filename records the settings used
        ├── STC_timelapse_*.mp4           timelapses you built
        ├── foreground_mask.png           the mask you painted, reused next run
        ├── hot_pixel_map.png             where this camera's stuck pixels are, found
        │                                 during the clean, 20 frames at a time
        ├── masks/                        what the detector found, per frame:
        │                                 <frame>.png, <frame>_raw.png, <frame>_polys.json
        ├── star_log_<date>.txt           THE file to read when something looked wrong
        ├── run_log_<date>.jsonl          machine-readable record of every step
        └── Archive/                      older logs, tucked out of the way
```

Older runs put this folder next to the originals as `cleanr_workspace/`. Both
names are still found (`modules/workspace.py`), so old sequences keep working.

**A good run:** every frame has a cleaned counterpart, the star trail opens with
no streaks across it, and the Star Log is short.

**A run worth investigating:** frames are missing from `cleaned/`, the Star Log
lists skipped files, or the star trail still shows a streak — which means the
detector missed it, and the masks folder will show what it did find at that frame.

**A failed run** stops with a message saying what happened. The message is meant to
be actionable on its own; if it is not, that is a bug in the message, not just in
the code.

---

## The files that matter

### Entry points
| file | what it is |
|---|---|
| `star_trail_cleanr.py` | The desktop app: every window, tab, and control, plus run orchestration. Large because it is the whole interface. Does no image work. |
| `astro_clean_v5.py` | The cleaning engine. Command-line program, one batch per invocation. **This is the live engine.** |
| `make_share_clip.py` | Star trail stacks and the before/after video. Also run as a worker process. |
| `timelapse_maker.py` | Timelapse rendering. |
| `mask_painter.py` | The foreground-mask painting window. |
| `build_helper.py` | PyInstaller packaging. Touch with care: dropping a `--collect-all` line here breaks the frozen app for every new install while working perfectly from source. |

### `modules/` — the real work
Detection: `detect_trails.py`, `detect_pipeline.py`, `crossing_splitter.py`,
`trail_grouper.py`, `slope_match.py`, `star_streak.py`
Repair and cleanup: `repair.py`, `hot_pixels.py`, `sky_dots.py`, `clean_sky.py`,
`align.py`
Plumbing: `io_safe.py` (every image read goes through it), `frame_list.py`
(single source of truth for what counts as a frame), `workspace.py`,
`run_logger.py`, `user_folder.py`, `keep_awake.py`
Hardware: `nvidia_detect.py`, `gpu_pack.py`
Updates and telemetry: `update_check.py`, `sparkle_updater.py`,
`winsparkle_updater.py`, `model_update.py`, `usage_report.py`

### `scripts/` — release machinery, not shipped in the app
`publish_appcast.py` (writes and verifies the update feeds),
`mirror_upload.py` (copies installers to our own server),
`release_signer.py` (manual fallback), `check_gpu_wheels.py`,
`smoke_built_bundle.py`, `diagnose_winsparkle.py`, `watch_ci.sh`

### `tests/` — run with `python3 tests/run_all.py`, takes about a second
Structural safety net, not a quality bar. Visual correctness is judged by eye.
Many of these tests exist because a specific bug reached real users; each says so
at the top, so a failure explains itself without archaeology.

### `tools/` and `dataset_pipeline/`
Development only: annotation review, dataset assembly, model training. Never
shipped. `tools/` holds the review GUIs (Trail ScreenR, TileFixR, TrailFixR,
Mask CheckR, Poly InspectR).

---

## Dead weight — present on the developer's machine, NOT in this repository

These are **gitignored**. They do not exist in a fresh clone and never reach CI,
so if you are reading the repository you will not meet them. They do exist in the
working copy where this app is developed, which is where the confusion happens —
a grep there returns hits in code that has not run since April.

| path | status |
|---|---|
| `astro_clean.py`, `astro_clean_v2.py`, `astro_clean_v3.py` | Superseded engine generations. The live engine is `astro_clean_v5.py`. |
| `archive/` | Snapshots taken just before big rewrites. |
| `v5_star_bridge_backup_2026_04_09/` | A whole-tree snapshot from April 2026 — every file in it has a live twin one directory up. |

Each of those carries a header or README saying so, in the working copy. If you
are grepping locally and land in one, it is almost certainly not the code that
runs.

---

## Where the sharp edges are

- **The GUI/engine contract.** Anything that changes which frames are cleaned, or
  their order, has to change on both sides at once. See the manifest note above.
- **Frozen versus source.** The app behaves differently when packaged: no console,
  no system Python, bundled data files. `build_helper.py` and the smoke tests in
  `tests/test_runtime_imports_bundled.py` exist for this.
- **Updating.** Two separate channels (a banner that reads GitHub releases, and
  the Sparkle/WinSparkle feeds) that are easy to confuse. `AUTO_UPDATE.md` is the
  source of truth and says so at the top.
- **Windows ships two installer files on purpose** (`.exe` for the updater, `.zip`
  for the website). They look like a duplicate; merging them silently breaks every
  Windows update. `AUTO_UPDATE.md` has the section.
- **Thresholds on pixel brightness must be relative to the local sky**, never a
  fixed number. A fixed bar is one camera on one night: the same code then behaves
  completely differently on a dark sky and a bright one.
- **Per-component work must slice to the component's bounding box.** Writing
  `mask == i` over a full frame inside a loop over components looks harmless and
  is catastrophic: on a 44MP frame with 183 components that is roughly 16 billion
  element operations to inspect blobs a few hundred pixels across.
  `connectedComponentsWithStats` already returns the box, and the surrounding
  code is usually reading it for width and height anyway. This has cost real
  users THREE times: `sky_dots._fill_specks` (minutes of waiting on a 30MP
  stack, 2026-08-09), `detect_pipeline.stage_prune_phantoms`'s component loop
  (57% of the largest stage in detection, 2.62s to 0.00s with byte-identical
  output, 2026-08-25), and the trim loop DIRECTLY BELOW that one, walked past
  while fixing it and found a day later (2026-08-26). When you fix one of these,
  read the whole function, not the loop you came for. Guarded by
  `tests/test_no_fullframe_per_component.py` and `tests/test_detect_mask_cost.py`.
- **Ask the AI's detection for its OUTLINE, never for its raster.** SAHI's
  `prediction.mask.bool_mask` is a property, so it re-renders on every access,
  and despite the name it returns float64: `np.zeros([h, w])` with no dtype,
  filled, then an `.astype(bool)` whose result is discarded without being
  assigned. On a 44MP frame that is 354 MB where 44 MB would do, once per
  detection, in every stage that asks. `trail_grouper._pred_to_mask` now fills
  the outline into uint8 itself, copying SAHI's own rounding so the pixels are
  identical, and the stages share one set of masks through
  `PipelineState.pred_masks` instead of each building its own. Together with the
  trim-loop fix that took phantom pruning and polygon fitting from 15.27s to
  7.16s over four 44MP frames, output byte-identical.
- **Anything timed must appear in the timing summary.** The summary's rows used
  to be a hand-written list, so a stage added later had no row and its time
  vanished from every report while still counting inside its parent. That hid
  `prune_phantoms`, the largest stage in detection, through two rounds of
  support emails about a slow machine. The summary now prints anything timed
  that has no row, and states any remaining gap.
- **Every stage assumes a three-channel photo, so the reader guarantees one.**
  Black-and-white sources are real: telescope sub-frames converted from FITS
  without debayering, and mono astro cameras. Those files hold a single channel,
  and the pipeline is written throughout for (height, width, 3) -- repair asks
  for the brightest of the three colours at a pixel, the 16-bit TIFF writer
  converts BGR to RGB. A user's folder of greyscale subs crashed every batch
  (`AxisError: axis 2 is out of bounds`, 2026-08-25). The promotion happens once
  in `modules/io_safe.py` (`_promote_grey`), beside the central orientation fix,
  so the worker, the detector and the tools all inherit it; do not add per-stage
  guards instead. Two consequences worth knowing: the writer asks the FILE on
  disk what it was (`is_single_channel`) so a greyscale source is handed back
  greyscale rather than at triple the size, and stuck-pixel detection is a
  genuine no-op on such frames because it tells a defect from a star by their
  colours. Guarded by `tests/test_mono_input.py`.
- **Sacred data.** Original source images are never modified, and manually
  reviewed annotation files are never regenerated.
