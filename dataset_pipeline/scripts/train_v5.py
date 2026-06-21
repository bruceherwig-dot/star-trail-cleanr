#!/usr/bin/env python3
"""
train_v5.py — Train a lightweight U-Net to detect airplane/satellite trails.

WHAT THIS IS
    A standalone training script for a small from-scratch U-Net that does
    per-pixel trail-vs-sky segmentation. You feed it a folder of original
    frames plus a matching folder of binary trail masks, and it trains a
    ~1.5M-parameter network and saves the best checkpoint to a .pt file.

    The model takes an RGB image and outputs a single-channel per-pixel
    "trail probability" map. At inference time you threshold that map at 0.5
    to turn it into a binary trail mask.

HOW IT FITS THE PROJECT (important context)
    The Star Trail CleanR app does NOT use this U-Net. The shipped detector
    (Trail DetectoR) is a YOLOv8 segmentation model trained with the
    ultralytics tooling (tools/train_*.py / prepare_yolo_*.py) and run via
    SAHI tiled inference (modules/detect_trails.py). This file is a separate,
    self-contained "v5"-era U-Net experiment that does the same job a
    different way. It has no imports from the rest of the codebase and is run
    by hand from the command line. Treat it as an alternative/legacy training
    approach, not part of the live pipeline.

INPUTS (command-line flags, see main())
    --images   Directory of original frames (JPG/PNG/TIFF), before cleaning.
    --masks    Directory of binary PNG masks (255 = trail, 0 = sky). These are
               the per-frame trail masks produced elsewhere in the project
               (e.g. a detection run's "--save-masks" output). Images and masks
               are paired by matching filename stem.
    --output   Where to save the trained model (default: trail_detector.pt).

OUTPUT
    A single .pt checkpoint (a dict holding the model weights plus the epoch,
    validation loss, F1, and the training image size). Only the
    best-validation-loss checkpoint seen so far is kept; it is overwritten
    each time a new best appears.

Usage:
    python3 train_v5.py \
        --images /Users/bruceherwig/Documents/frames/extra \
        --masks  /Users/bruceherwig/Documents/training_masks \
        --output trail_detector.pt \
        --epochs 30 --img-size 1024
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Model ────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    The basic building block of the U-Net: two 3x3 convolutions back to back,
    each followed by batch normalization and a ReLU. Takes a feature map with
    `in_ch` channels and produces one with `out_ch` channels at the same spatial
    size (padding keeps width/height unchanged). Used everywhere in both the
    encoder and decoder.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Run the input feature map through the two-conv block and return it."""
        return self.block(x)


class TrailUNet(nn.Module):
    """
    Lightweight U-Net for trail/sky pixel segmentation.
    Channels: 3 → 16 → 32 → 64 → 128 → 64 → 32 → 16 → 1
    ~1.5M parameters, ~6MB model file.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3,   16)
        self.enc2 = ConvBlock(16,  32)
        self.enc3 = ConvBlock(32,  64)
        self.enc4 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up4   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4  = ConvBlock(256, 128)
        self.up3   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3  = ConvBlock(128, 64)
        self.up2   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2  = ConvBlock(64, 32)
        self.up1   = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1  = ConvBlock(32, 16)

        # Output — single channel probability map
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        """
        Run one image batch through the network and return raw per-pixel logits
        (NOT yet passed through a sigmoid). The encoder downsamples the image
        through four stages plus a bottleneck, then the decoder upsamples back
        to full resolution, concatenating each encoder stage's features (the
        skip connections) so fine detail is preserved. Output has one channel
        at the same width/height as the input.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)   # raw logits


# ─── Loss ─────────────────────────────────────────────────────────────────────

def dice_loss(pred_logits, targets, smooth=1.0):
    """
    Dice loss: measures how much the predicted trail region overlaps the true
    trail region, scored as 1 minus the Dice coefficient (so 0 = perfect
    overlap, 1 = no overlap). Good for thin, sparse targets like trails where
    most pixels are sky, because it rewards getting the small foreground right
    instead of being swamped by the easy background. `smooth` avoids divide-by-
    zero when a frame has no trail pixels.
    """
    pred = torch.sigmoid(pred_logits)
    pred_f   = pred.view(-1)
    target_f = targets.view(-1)
    intersection = (pred_f * target_f).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth)


def combined_loss(pred_logits, targets):
    """
    The actual training loss: the sum of pixel-wise binary cross-entropy and
    Dice loss. BCE pushes each pixel's prediction toward correct, while Dice
    keeps the model focused on overlapping the (small) trail regions. Both are
    used together because either alone struggles on heavily sky-dominated frames.
    """
    bce  = F.binary_cross_entropy_with_logits(pred_logits, targets)
    dice = dice_loss(pred_logits, targets)
    return bce + dice


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TrailDataset(Dataset):
    """
    PyTorch dataset that serves up (image, mask) tensor pairs for training.
    Each item is loaded from disk on demand, resized so its longest side equals
    `img_size`, padded with zeros to a square, optionally augmented (flips and
    brightness/contrast jitter), and returned as float tensors. The mask is
    binarized (any pixel brighter than 127 counts as trail).
    """
    def __init__(self, pairs, img_size, augment=False):
        """
        pairs    : list of (image_path, mask_path)
        img_size : resize longest side to this (maintains aspect ratio padding)
        augment  : apply random augmentation
        """
        self.pairs    = pairs
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        """Number of image/mask pairs in this dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Load, resize, pad, and (if enabled) augment the pair at position `idx`,
        returning two tensors: the RGB image as channels-first floats scaled to
        0..1, and the binary trail mask as a single-channel float (1 = trail,
        0 = sky).
        """
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(str(img_path))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize to target size (keep aspect ratio, pad with zeros)
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        img  = np.pad(img,  ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)),          mode='constant')

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img  = img[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            # Random vertical flip
            if random.random() > 0.5:
                img  = img[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
            # Brightness / contrast jitter
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)   # contrast
                beta  = random.randint(-20, 20)     # brightness
                img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # To tensor
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_t, mask_t


# ─── Training ─────────────────────────────────────────────────────────────────

def find_pairs(images_dir, masks_dir):
    """
    Match up images with their masks by filename stem. Scans `images_dir` for
    image files (jpg/jpeg/png/tif/tiff) and `masks_dir` for PNG masks, then
    returns a sorted list of (image_path, mask_path) tuples for every mask whose
    stem has a matching image. Images without a mask, or masks without an image,
    are silently skipped.
    """
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    img_files = {p.stem: p for p in Path(images_dir).iterdir()
                 if p.suffix.lower() in exts}
    mask_files = {p.stem: p for p in Path(masks_dir).iterdir()
                  if p.suffix.lower() == '.png'}

    paired = []
    for stem, mask_path in sorted(mask_files.items()):
        if stem in img_files:
            paired.append((img_files[stem], mask_path))

    return paired


def train(args):
    """
    Run the full training loop end to end.

    Picks the best available device (Apple MPS, then CUDA, then CPU), finds the
    image/mask pairs, prints how many frames actually contain a trail, and makes
    a reproducible 80/20 train/validation split (fixed random seed). Builds the
    data loaders and the U-Net, then trains for `args.epochs` epochs using Adam
    with a cosine-annealing learning-rate schedule. After each epoch it reports
    train loss, validation loss, and pixel-level precision / recall / F1, and
    whenever validation loss hits a new low it saves that checkpoint to
    `args.output` (overwriting the previous best). Exits early with a message if
    no image/mask pairs are found.
    """
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # Find image/mask pairs
    pairs = find_pairs(args.images, args.masks)
    if not pairs:
        sys.exit(f"No matching image/mask pairs found.\n"
                 f"  images: {args.images}\n  masks:  {args.masks}")

    print(f"Found {len(pairs)} image/mask pairs")

    # Count trail vs no-trail for info
    trail_count = 0
    for _, m in pairs:
        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            trail_count += 1
    print(f"  {trail_count} with trail  ({trail_count/len(pairs)*100:.1f}%)")
    print(f"  {len(pairs)-trail_count} without trail  ({(len(pairs)-trail_count)/len(pairs)*100:.1f}%)")
    print()

    # Train/val split (80/20, by sorted filename so split is reproducible)
    random.seed(42)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    val_n  = max(1, int(len(shuffled) * 0.2))
    val_pairs   = shuffled[:val_n]
    train_pairs = shuffled[val_n:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")
    print(f"Image size: {args.img_size}×{args.img_size}")
    print()

    train_ds = TrailDataset(train_pairs, args.img_size, augment=True)
    val_ds   = TrailDataset(val_pairs,   args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = TrailUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (~{n_params*4/1e6:.1f}MB)")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    output_path   = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss  = 0.0
        tp = fp = fn = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += combined_loss(logits, masks).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                tp += (preds * masks).sum().item()
                fp += (preds * (1 - masks)).sum().item()
                fn += ((1 - preds) * masks).sum().item()
        val_loss /= len(val_loader)
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
              f"({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'val_loss':   val_loss,
                'f1':         f1,
                'img_size':   args.img_size,
            }, str(output_path))
            print(f"           ↑ saved best model  (val_loss={val_loss:.4f}  F1={f1:.3f})")

    print(f"\nDone. Best model: {output_path}  (val_loss={best_val_loss:.4f})")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    """
    Command-line entry point. Parses the --images / --masks / --output / --epochs
    / --img-size / --batch-size / --lr flags and hands them to train().
    """
    parser = argparse.ArgumentParser(
        description="Train U-Net trail detector for Star Trail CleanR v5.")
    parser.add_argument("--images",     required=True,
                        help="Directory of original frames (JPG)")
    parser.add_argument("--masks",      required=True,
                        help="Directory of binary PNG masks (from --save-masks)")
    parser.add_argument("--output",     default="trail_detector.pt",
                        help="Output model file (default: trail_detector.pt)")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--img-size",   type=int, default=512,
                        help="Resize images to this square size for training (default: 512)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
