"""Smoke tests for modules/slope_match.py.

Synthetic mask scenarios — no model inference involved, runs in milliseconds.
Catches future regressions where a refactor breaks the merge logic.
"""
import math

import cv2
import numpy as np

from modules import slope_match


def _draw_thin_diagonal_mask(shape, p_lo, p_hi, thickness=10):
    """Draw a thin line mask from p_lo to p_hi with given thickness."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, (int(p_lo[0]), int(p_lo[1])),
             (int(p_hi[0]), int(p_hi[1])),
             255, thickness=thickness, lineType=cv2.LINE_AA)
    return (mask > 0).astype(np.uint8) * 255


def _draw_filled_blob(shape, center, size):
    """Draw a near-square filled blob (ratio ~1:1.3) for the aspect-filter test."""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy = center
    cv2.rectangle(mask,
                  (int(cx - size // 2), int(cy - int(size * 0.65))),
                  (int(cx + size // 2), int(cy + int(size * 0.65))),
                  255, thickness=-1)
    return mask


def test_two_overlapping_thin_trails_merge_to_one():
    """Two co-linear thin trails that share pixels in the middle MUST merge."""
    H, W = 1000, 1000
    # one trail from (200, 200) to (550, 550)
    a = _draw_thin_diagonal_mask((H, W), (200, 200), (550, 550), thickness=10)
    # second trail from (500, 500) to (800, 800) — overlaps trail a in the middle
    b = _draw_thin_diagonal_mask((H, W), (500, 500), (800, 800), thickness=10)
    out = slope_match.merge([a, b])
    assert len(out) == 1, (
        f"expected 1 merged mask, got {len(out)} (should have fused two co-linear "
        "thin trails that share pixels)"
    )


def test_two_separate_trails_stay_separate():
    """Two thin trails far apart with no shared pixels MUST stay separate."""
    H, W = 1000, 1000
    a = _draw_thin_diagonal_mask((H, W), (100, 100), (300, 300), thickness=10)
    # parallel slope but offset and far away — zero mask intersection
    b = _draw_thin_diagonal_mask((H, W), (700, 100), (900, 300), thickness=10)
    out = slope_match.merge([a, b])
    assert len(out) == 2, (
        f"expected 2 masks (no overlap, far apart), got {len(out)} — "
        "the contiguous-trail filter should have prevented this merge"
    )


def test_fat_blob_does_not_merge_with_thin_trail():
    """A fat blob (low aspect ratio) is not eligible for merging even if a
    thin co-linear neighbor exists."""
    H, W = 1000, 1000
    # near-square blob at (500, 500), 200x130 — aspect about 1.5
    blob = _draw_filled_blob((H, W), (500, 500), 200)
    # thin trail nearby that shares pixels with the blob
    trail = _draw_thin_diagonal_mask((H, W), (600, 500), (900, 500), thickness=10)
    out = slope_match.merge([blob, trail])
    assert len(out) == 2, (
        f"expected 2 masks (blob's aspect should fail the elongated filter), "
        f"got {len(out)}"
    )


def test_empty_input_returns_empty():
    """Empty input list returns empty list, no exceptions."""
    out = slope_match.merge([])
    assert out == [], f"expected empty list, got {out}"


def test_single_input_returns_single():
    """Single-mask input is returned untouched."""
    H, W = 500, 500
    a = _draw_thin_diagonal_mask((H, W), (100, 100), (400, 400), thickness=10)
    out = slope_match.merge([a])
    assert len(out) == 1
    assert np.array_equal(out[0], a)
