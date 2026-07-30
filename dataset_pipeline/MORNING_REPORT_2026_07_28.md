# Morning report — overnight v6 data prep (2026-07-28)

Ran while you slept. No training happened; originals and your reviewed
CVAT work were read-only throughout. Phases 3 and 4 took three attempts:
my first two runs launched the scripts in a way that could not find the
project's shared code. Fixed and verified before the third run.

## What ran
- Phase 1 OK: Ajay fisheye pre-annotation, 300 mask files, 2m.
  Every frame came back with trail detections — busy fisheye sky.
- Phase 2 OK: LabelMe pre-annotations ready for CVAT upload.

- Phase 3 OK: fresh CVAT export -> masks for 26 task folders, 6m 32s
- Phase 4 OK: tiled base rebuilt, 30454 tiles, 1.3G, 4m

## Waiting on you
- Create the CVAT task for the Ajay fisheye set (standard settings,
  image quality 95) and say the word; the pre-annotations in
  labels/Ajay Talwar - India -fisheye/labelme_json upload with the
  usual two-line edit to labelme_to_cvat.py. Then review. Fisheye
  trails curve, so expect rougher pre-annotations than usual.
- Name the remaining new sets; the full v6 assembly waits for them.
- Two decisions still open from the plan: the gkyle tile cap, and
  building the Stroudt's augmentation (action item 1).

Full log: dataset_pipeline/logs/overnight_prep_2026_07_28.log

## One number explained
The plan mentions 34 CVAT tasks; this export wrote 26 task folders. The
other 8 are the tile-based tasks (GoPro augs 57/58, crossing aug 47, gkyle's
pre-cut tiles, and kin) — they enter through the tile converter at assembly
time, not through this mask export. Nothing is missing.

## Correction (found 07-28 while answering a question)
"Fresh CVAT export for 26 task folders" overstates it: the export script's
task list covers only the 9 v5-era tasks, so only those were refreshed last
night. The other 17 folders (the v4-era sets, Gomphothere included) are
April-era copies that were already on the drive. Before v6 assembly the
export list must be extended to the FULL roster so every task gets a truly
fresh pull.
