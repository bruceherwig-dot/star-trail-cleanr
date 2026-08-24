"""A folder name with non-ASCII letters must not stop the export.

Field report, 2026-08-23, Kari Tuomi (Finland): *"I failed to create star trail,
because I had Scandinavian letters in path. 'A with dots above'. They did not
seem to be issue in other than in final export stage."*

Exactly right, and the cause was ours. `cv2.imwrite` on Windows uses the ANSI
file APIs and cannot write a path containing characters outside the local code
page. `modules/io_safe.robust_imwrite` was written months earlier FOR THIS, and
falls back to Pillow -- but three call sites never used it: the star trail
export, the red trail map, and the two before/after stills. Cleaning worked all
the way through and then the keepsake he actually wanted failed at the last step.

THE TRAP IN THE OBVIOUS FIX: Pillow defaults JPEG to quality 75 while we write
95. Swapping in robust_imwrite without carrying the encoder settings across
would have quietly handed the affected users a WORSE picture -- a silent
downgrade visible only to the people we were fixing it for.

These tests force the Pillow fallback (which is what a Windows machine does on
such a path) so the behaviour is checked on every platform, not only Windows.
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import io_safe  # noqa: E402

SCANDINAVIAN = "Perseidejä takapihalta"      # his actual folder name


def _img():
    return (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)


def _force_pillow_fallback(monkey_target):
    """Make cv2.imwrite fail, the way it does on Windows for these paths."""
    original = io_safe.cv2.imwrite

    def failing(*a, **k):
        return False

    io_safe.cv2.imwrite = failing
    return original


def test_export_survives_a_non_ascii_folder_via_the_fallback():
    original = _force_pillow_fallback(io_safe)
    try:
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, SCANDINAVIAN)
            os.makedirs(folder)
            out = os.path.join(folder, "STC_cleaned_star_trail.jpg")
            ok = io_safe.robust_imwrite(out, _img(), [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok, "the fallback failed to write to a folder containing 'ä'"
            assert os.path.exists(out) and os.path.getsize(out) > 0
    finally:
        io_safe.cv2.imwrite = original


def test_the_fallback_keeps_the_quality_we_asked_for():
    """Pillow defaults to 75. Silently shipping 75 to these users would be a
    quality regression that only they would ever see."""
    from PIL import Image

    original = _force_pillow_fallback(io_safe)
    try:
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, SCANDINAVIAN)
            os.makedirs(folder)
            img = _img()
            ours = os.path.join(folder, "ours.jpg")
            io_safe.robust_imwrite(ours, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            ref95 = os.path.join(folder, "ref95.jpg")
            ref75 = os.path.join(folder, "ref75.jpg")
            rgb = Image.fromarray(img[:, :, ::-1])
            rgb.save(ref95, quality=95)
            rgb.save(ref75, quality=75)

            size = os.path.getsize(ours)
            assert abs(size - os.path.getsize(ref95)) < os.path.getsize(ref95) * 0.15, (
                f"written at the wrong quality: {size} bytes, "
                f"q95 is {os.path.getsize(ref95)}, q75 is {os.path.getsize(ref75)}")
    finally:
        io_safe.cv2.imwrite = original


def test_no_shipping_code_writes_images_with_bare_cv2():
    """The reason this bug survived: a safe writer existed and three call sites
    ignored it. Nothing catches that except a check like this one."""
    offenders = []
    targets = ["make_share_clip.py", "astro_clean_v5.py", "star_trail_cleanr.py",
               "timelapse_maker.py", "mask_painter.py"]
    targets += [str(p.relative_to(REPO)) for p in (REPO / "modules").glob("*.py")]
    for rel in targets:
        f = REPO / rel
        if not f.exists() or f.name == "io_safe.py":
            continue          # io_safe IS the wrapper; it calls cv2 on purpose
        for n, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#")[0]
            if "cv2.imwrite" in code and "robust_imwrite" not in code:
                offenders.append(f"{rel}:{n}")
    assert not offenders, (
        "these write images with bare cv2.imwrite, which fails on non-ASCII "
        "paths on Windows; use modules.io_safe.robust_imwrite instead: "
        + ", ".join(offenders))
