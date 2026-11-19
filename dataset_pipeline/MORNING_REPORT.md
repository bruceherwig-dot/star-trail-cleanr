# v5 fine-tune — morning report & flagged questions

The dataset is built, verified, and packaged ready to upload (see README_VAST.md).
Below are the decisions worth your call before launching. Each has my recommendation, so
you can confirm or override quickly. Nothing here blocks training — the defaults are sane.

## Status (done overnight)
- Step 1 export fixed two real bugs the count-check caught: Thomas Jackson sideways-photo
  orientation (70 → 262 correct masks) and Stroudt's jpg/tif duplicate merge (39 → 54).
- Steps 2-3 tiled the 9 datasets (34,768 base tiles) + folded in 7 reviewed pile sets
  (2,804 tiles), all from CVAT, 0 silent drops.
- Steps 4-6 baked the 540px long-trail rotations, made a leakage-free 15% val split, and
  reconciled to 40,014 tiles on disk.

## FLAGGED QUESTIONS

### 1. Joshua Tree 80s — RESOLVED (real trails)
JT-80s = 80-SECOND exposures. Bruce confirmed the long trails are expected: real
airplane/sat trails captured over 80s span much more of the frame, so 4-9 long trails per
frame is legitimate, not mislabeled streaks. The polygons are genuinely long (only 4 of
684 under 60px). No change — JT-80s stays in the training set as built.
(Visual was /tmp/jt80s_trail_check.jpg.)

### 2. GoPro is 36% of the training set — intended weight?
GoPro is brand-new in v5 (v4 saw none), and it's now the single biggest source (36% of
tiles and of positives), counting base + blind-spot + edge augs. That's deliberate (we're
fixing GoPro blindness), but it could tilt the model toward GoPro's low-res/fisheye look.
- **My recommendation:** keep it full-strength this round — learning GoPro is the point,
  and the other sources still total ~64%.
- **Override option:** cap GoPro (e.g. drop the 1,032 blind-spot augs, or downsample base)
  if you'd rather protect DSLR performance. Say the word and I'll rebuild.

### 3. Fine-tune depth — GoPro is a new domain
We fine-tune from v4 at a gentle lr0=0.001. For a brand-new domain (GoPro) a gentle rate
can under-learn it.
- **My recommendation:** run as-is first (lr0=0.001, 100 epochs, early-stop patience 25);
  if GoPro recall looks weak in the val plots, re-run with `--lr0 0.005`.
- **Alternative:** train from the COCO yolov8s-seg base instead of v4 (learns GoPro
  harder, but relearns everything v4 already knew). I don't recommend it.

### 4. Model size — stay yolov8s or go bigger?
Fine-tuning from v4 keeps the small (s) architecture. The richer v5 set could support
yolov8m for more accuracy — but that means training from the COCO m base, not from v4.
- **My recommendation:** stay with s/fine-tune for a fast, low-risk first v5. Consider m
  as a follow-up if s plateaus.

### 5. Validation = plain base tiles only
Val holds no crossings/blind/aug examples (those all went to train, to teach them). So val
mAP measures general performance, not specifically the hard cases. That matches the plan
(real success = re-running the app on representative sequences, not val mAP).
- **Confirm** that's fine, or I can hold out a few hard examples for val instead.

### 6. Train-time augmentation
Flips on (flipud/fliplr 0.5); free rotation and mosaic OFF (rotation adds black corners,
mosaic hurts thin trails; the 540 rotations are baked instead). Easy to flip via flags.

## Recommended launch (if you're happy with the above)
`python3 train_v5_finetune.py` as-is, watch the val plots for GoPro recall, adjust lr0
only if needed.
