#!/usr/bin/env python3
"""train_v5_finetune.py — Fine-tune Trail DetectoR v5 from the v4 YOLOv8-seg model, on a
CUDA GPU (e.g. a rented vast.ai instance).

This is the cloud-GPU counterpart of tools/train_yolo.py (which targets the Mac's MPS
device) and is the YOLO pipeline's trainer -- NOT the standalone U-Net in train_v5.py.
It fine-tunes FROM the shipped v4 weights instead of training from scratch: the locked
v5 plan teaches the new + corrected material (GoPro, crossings, blind spots, corrected
Thomas Jackson / Stroudt's) on top of v4.

Portable: it rewrites dataset.yaml's `path:` to wherever the dataset actually lives at
run time, so the bundle works dropped anywhere on the instance. By default it assumes it
is sitting INSIDE the dataset folder (the bundle layout), next to best_v4.pt.

USAGE (on the vast.ai instance, inside the uploaded bundle)
  pip install ultralytics
  python3 train_v5_finetune.py                    # uses this folder + ./best_v4.pt
  python3 train_v5_finetune.py --epochs 120 --batch 32

KEY CHOICES (all overridable; explained in README_VAST.md)
  - base model: best_v4.pt (fine-tune, not scratch)
  - lr0 = 0.001  (10x below the 0.01 from-scratch default; refines v4 rather than
                  washing it out). Pass --lr0 0.01 to revert to the scratch default.
  - fliplr=0.5 (ON), flipud=0.0 (OFF)  fliplr is a safe mirror; flipud is OFF because an
                  upside-down flip puts ground-in-the-sky images the model never meets in
                  reality (matches the proven v4 recipe). Orientation variety otherwise comes
                  from the baked 540 rotations.
  - degrees=0    (the 540px long-trail 90/180/270 copies are already BAKED into the set;
                  free rotation would put black corners on tiles. Bump --degrees to add it.)
  - mosaic=0.0   (mosaic stitches 4 images; tends to hurt thin sparse trails. --mosaic 1.0
                  to experiment.)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_yaml(data_dir: Path) -> Path:
    """Write dataset.yaml inside data_dir with an absolute, machine-correct path."""
    yaml = data_dir / "dataset.yaml"
    yaml.write_text(
        "# Trail DetectoR v5 fine-tune dataset (auto-pathed by train_v5_finetune.py)\n"
        f"path: {data_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: trail\n"
    )
    return yaml


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default=str(here),
                    help="Dataset folder with images/ and labels/ (default: this folder).")
    ap.add_argument("--model",  default=str(here / "best_v4.pt"),
                    help="Base weights to fine-tune FROM.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--batch",  type=int, default=-1, help="-1 = auto-fit to GPU memory")
    ap.add_argument("--device", default="0", help="CUDA device id, or 'cpu'")
    ap.add_argument("--name",   default="trail_detector_v14_finetune")
    ap.add_argument("--lr0",    type=float, default=0.001)
    ap.add_argument("--optimizer", default="SGD",
                    help="MUST be explicit: optimizer='auto' silently IGNORES lr0 and picks "
                         "its own (MuSGD@0.01 on 8.4), discarding the gentle fine-tune rate.")
    ap.add_argument("--degrees", type=float, default=0.0)
    ap.add_argument("--mosaic",  type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    data_dir = Path(args.data)
    yaml = build_yaml(data_dir)
    print(f"Dataset:                    {data_dir}")
    print(f"Base model (fine-tune from): {args.model}")

    model = YOLO(args.model)
    results = model.train(
        data     = str(yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        name     = args.name,
        project  = str(here / "runs"),
        optimizer = args.optimizer,   # explicit so lr0 is honored (NOT 'auto')
        lr0      = args.lr0,
        degrees  = args.degrees,
        mosaic   = args.mosaic,
        fliplr   = 0.5,
        flipud   = 0.0,
        patience = args.patience,
        workers  = 8,
        save     = True,
        plots    = True,
    )

    best = here / "runs" / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print("Download best.pt to the Mac, drop it in assets/best.pt to ship, and update")
    print("_DEV_FALLBACK_MODEL in star_trail_cleanr.py in the same commit.")
    try:
        print(f"mAP50(M): {results.results_dict.get('metrics/mAP50(M)', 'see above')}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
