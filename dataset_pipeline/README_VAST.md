# Trail DetectoR v5 — fine-tune bundle (for vast.ai)

This folder is self-contained. Upload the whole thing to a rented GPU and run one command.

## What's inside
- `images/{train,val}` + `labels/{train,val}` — the YOLOv8-seg dataset (640px tiles).
- `dataset.yaml` — auto-regenerated with the correct path by the train script.
- `best_v4.pt` — the shipped v4 model. We FINE-TUNE from this (not from scratch).
- `train_v5_finetune.py` — the training script (CUDA).
- `ASSEMBLY_REPORT.txt` — exact counts + every decision made while building the set.
- `MORNING_REPORT.md` — **read this first**: the open "train-well" questions for Bruce.

## Dataset at a glance
- TOTAL 43,590 tiles → TRAIN 37,834, VAL 5,756. Tiles are JPG q92 (~2.3 GB total).
- Train make-up: base tiles + 2,760 baked 540px long-trail rotations + 2,486 curated pile
  tiles (bridge misses, crossings, GoPro blind-spots, Jeff Fishman blind trails) + 3,043
  REHEARSAL tiles from the ~14 v4 datasets that v5 doesn't retrain (anti-forgetting).
- Source balance: GoPro is the biggest single source and is brand-new to training in v5
  (v4 had none); the corrected/old datasets + rehearsal keep the old domains represented.
- Val (5,756) = plain v5 base tiles + a 533-tile rehearsal slice from the old domains, so
  the val score is also a live forgetting alarm. No rotation/aug of a val frame is in train.

## Run it on vast.ai
1. **Rent an instance.** A single 24GB GPU is plenty: RTX 4090 / A5000 / A6000. Pick a
   PyTorch template (CUDA 12). ~30GB disk is ample (2.3GB dataset + env + outputs); do NOT
   set cache=disk (it would write a large .npy cache).
2. **Upload the one zip** (`dataset_v5_final.zip`, ~2.3GB, sits next to this folder on T7).
   From the Mac, fill in the host/port vast.ai shows (resumable):
   ```
   rsync -avP -e "ssh -p <PORT>" "dataset_v5_final.zip" root@<HOST>:/workspace/
   ```
   (Or use vast.ai's web/Jupyter upload for the single file.)
3. **On the instance — unzip and train:**
   ```
   cd /workspace && unzip -q dataset_v5_final.zip && cd dataset_v5_final
   pip install ultralytics
   python3 train_v5_finetune.py            # 100 epochs, auto batch, fine-tune from best_v4.pt
   ```
4. **Get the model back** (best weights land in `runs/trail_detector_v14_finetune/weights/best.pt`):
   ```
   rsync -avP -e "ssh -p <PORT>" root@<HOST>:/workspace/dataset_v5_final/runs/ "v5_runs/"
   ```
5. **Ship:** drop `best.pt` into `assets/best.pt`, and update `_DEV_FALLBACK_MODEL` in
   `star_trail_cleanr.py` in the same commit (CLAUDE.md release rule).

Rough cost/time: yolov8s-seg, ~100 epochs on a 4090 ≈ 6-12 hours, a few dollars to ~$10.
Early-stopping (patience 25) usually finishes sooner.

## Config (reconciled against v4's proven recipe + deep research, June 2026)
- BAKED into the dataset: the 540px long-trail 90/180/270 rotations.
- fliplr = 0.5 (ON); flipud = 0.0 (OFF — upside-down flips put ground-in-sky images the
  model never meets). mosaic = 0 (OFF — keeps training at the single-tile inference scale).
  degrees = 0 (free rotation off; rotations are baked).
- lr0 = 0.001 (deliberate 10x-below-scratch fine-tune rate; research-confirmed reasonable,
  not a magic number). No layer freezing — full fine-tune (research: freezing does NOT
  prevent forgetting). Train-from-checkpoint, NOT resume.
- Anti-forgetting: the 3,576 rehearsal tiles + low lr0 are the confirmed levers.
- Override any of these via flags: `--lr0 --epochs --batch --degrees --mosaic`.

BEFORE SHIPPING: compare v5 vs v4 on a few representative OLD sequences (bridge-fire /
flood / crossing counts). Keep v4 as fallback until v5 clearly wins. The val score now
includes old-domain rehearsal tiles, so watch it for forgetting during training too.
