#!/usr/bin/env python3
"""Smoke test the PyInstaller-bundled app by running the worker on one
synthetic frame and verifying it exits cleanly with an output file.

Catches the kind of bug that doesn't reproduce in dev but breaks the
shipped binary: bundled torchvision missing MPS ops, missing PyInstaller
hidden imports, library version mismatches, accidentally-stripped
modules, etc.

Designed to run on macOS, Windows, and Linux from CI immediately after
PyInstaller has built the app to dist/. If this script exits non-zero,
the build job fails and the broken artifact never reaches users.

Manual usage on a built bundle is also fine:
    python3 scripts/smoke_built_bundle.py
"""
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TIMEOUT_SECS = 240


def _bundled_paths():
    """Return (executable, worker_script, model_path) for the current platform."""
    sysname = platform.system()
    if sysname == "Darwin":
        app = REPO / "dist" / "StarTrailCleanR.app"
        return (
            app / "Contents" / "MacOS" / "StarTrailCleanR",
            app / "Contents" / "Frameworks" / "astro_clean_v5.py",
            app / "Contents" / "Frameworks" / "best.pt",
        )
    if sysname == "Windows":
        bundle = REPO / "dist" / "StarTrailCleanR"
        return (
            bundle / "StarTrailCleanR.exe",
            bundle / "_internal" / "astro_clean_v5.py",
            bundle / "_internal" / "best.pt",
        )
    # Linux
    bundle = REPO / "dist" / "StarTrailCleanR"
    return (
        bundle / "StarTrailCleanR",
        bundle / "_internal" / "astro_clean_v5.py",
        bundle / "_internal" / "best.pt",
    )


def _make_synthetic_frame(path):
    """Save one 640x640 BGR JPG with a few stars + a fake trail. The YOLO
    model doesn't have to detect it correctly; we just need the worker to
    load the model, run inference, and write an output frame without
    crashing."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for x, y in [(100, 200), (300, 100), (500, 400), (450, 250), (180, 480)]:
        cv2.circle(img, (x, y), 2, (220, 230, 255), -1)
    cv2.line(img, (50, 320), (590, 320), (240, 240, 240), 2)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    exe, worker, model = _bundled_paths()
    print(f"Bundle exec:  {exe}", flush=True)
    print(f"Worker:       {worker}", flush=True)
    print(f"Model:        {model}", flush=True)

    for p in (exe, worker, model):
        if not p.exists():
            print(f"FAIL: missing bundled file: {p}", file=sys.stderr)
            sys.exit(1)

    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        out_dir = Path(td) / "out"
        in_dir.mkdir()
        frame = in_dir / "smoke_frame.jpg"
        _make_synthetic_frame(frame)

        cmd = [
            str(exe), "--cleanr-worker", str(worker),
            str(in_dir), "-o", str(out_dir),
            "--model", str(model),
            "--start", "0", "--batch", "1",
            "--output-format", "jpg",
        ]
        print(f"Running: {' '.join(cmd)}", flush=True)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired as e:
            print(f"FAIL: worker did not finish within {TIMEOUT_SECS}s", file=sys.stderr)
            if e.stdout:
                print("---- worker stdout (partial) ----", file=sys.stderr)
                print(e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", "replace"),
                      file=sys.stderr)
            if e.stderr:
                print("---- worker stderr (partial) ----", file=sys.stderr)
                print(e.stderr if isinstance(e.stderr, str) else e.stderr.decode("utf-8", "replace"),
                      file=sys.stderr)
            sys.exit(1)

        print("---- worker stdout ----", flush=True)
        print(proc.stdout, flush=True)
        print("---- worker stderr ----", flush=True)
        print(proc.stderr, flush=True)

        if proc.returncode != 0:
            print(f"FAIL: worker exited {proc.returncode}", file=sys.stderr)
            sys.exit(1)

        outputs = list(out_dir.glob("*.jpg"))
        if not outputs:
            print(f"FAIL: no output frames produced in {out_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"PASS: bundled worker produced {len(outputs)} output frame(s)", flush=True)


if __name__ == "__main__":
    main()
