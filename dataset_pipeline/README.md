# Dataset Pipeline — how to rebuild the Trail DetectoR training sets

This folder is a frozen, off-machine backup of the complete recipe that builds the YOLO
training datasets (`dataset_tiled_v4`, `dataset_v5_final`), plus a full backup of the CVAT
review annotations. It exists so the datasets can be regenerated from source even if the
local Mac or the big derived tile folders are lost.

## Why this exists

The 75 GB `dataset_tiled_v4` and the other tiled datasets are **derived** artifacts — tiles
cut from the reviewed CVAT polygons. As long as this recipe, the CVAT annotations, and the
source frames survive, the tiles can be rebuilt. The recipe scripts normally live in `tools/`
and `runs/` in the working tree, but those are gitignored (dev tooling is kept out of the
shipped app repo), so this folder is the tracked snapshot that rides to GitHub.

## Source of truth — must be preserved (none of it is the tiles)

- **CVAT** (local Docker, `http://localhost:8080`) — the reviewed polygons. Also exported
  here under `cvat_annotations_2026_06_20/`.
- **`labels/`** on the T7 — the PNG masks rendered from CVAT.
- **`star trail images/`** on the T7 — the original source frames (sacred).

## Determinism

Every build script uses `SEED = 42` (a hard constant), with `VAL_FRACTION = 0.15`,
`TILE = 640`, `OVERLAP = 0.2`. The long-trail rotations are exact 90/180/270; the
augmentation tilts are exact 15/30/45/60/75. Same seed plus same inputs equals
byte-identical output — the train/val split and every rotated copy come out the same.

## Rebuild order (v5)

1. `cvat_to_masks.py` — pull reviewed polygons from CVAT, render PNG masks into `labels/`.
2. `prepare_yolo_v5.py` — cut source frames + masks into 640px base tiles (seed 42).
3. `cvat_tiles_to_yolo.py` — fold in the reviewed CVAT pile tiles (bridges, crossings,
   GoPro, Jeff Fishman) from tasks 42/45/46/47/57/58/61, including their rotation augs.
4. `assemble_v5_dataset.py` — dedup, bake the 540px long-trail 90/180/270 copies, do the
   leakage-free 15% val split (seed 42).
5. Verify against `ASSEMBLY_REPORT.txt`: it should land back on **40,014 tiles**.

## Rebuild (v4)

`prepare_yolo_v4.py` (seed 42). It reads today's corrected masks, so it produces a *cleaner*
v4 than the original — the first one had the over-thick masks and duplicate JPG/TIF frames
that were deliberately fixed for v5.

## CVAT annotation backup — `cvat_annotations_2026_06_20/`

Full annotation export of all 34 current CVAT tasks. This replaces the stale on-disk backup
from June 5, which only covered tasks 01-36 and none of the augmentation tasks. `tasks_manifest.json`
maps each task id to its name. The augmentation tile **images** are preserved separately inside
`dataset_v5_final` (the 1,832 aug tiles), so masks plus images are both recoverable.

## Scripts — `scripts/`

Reference copies of the build and augmentation scripts. To actually run them, use the working
copies in `tools/` and `runs/`; the copies here may have broken relative imports and are a
record of the recipe, not a run location.
