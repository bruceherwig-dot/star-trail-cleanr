"""Temporal percentile clean-sky background."""
import numpy as np
from typing import List


def build_clean_sky(aligned: List[np.ndarray], percentile: int = 50) -> np.ndarray:
    """Per-pixel temporal percentile -- airplane-free background for repair.

    Returns uint8 BGR image.
    """
    n = len(aligned)
    print(f"    rgb stack ({n} frames)... ", end="", flush=True)
    clean = np.empty(aligned[0].shape, dtype=np.uint8)
    for ch in range(3):
        stack = np.stack([f[:, :, ch] for f in aligned], axis=0)
        clean[:, :, ch] = np.percentile(stack, percentile, axis=0).astype(np.uint8)
        del stack
    print("done")
    return clean
