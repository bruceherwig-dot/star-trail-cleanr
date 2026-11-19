"""The star finder used by the repair track step must give exactly the old numbers.

Measured 2026-11-17: the track step took 14.6 s on 8 frames and 11 s of it was
the star finder comparing the WHOLE patch against each star's label, once per
star. It now looks only inside each star's own box. The old whole-patch version
is kept here as the definition: the two must agree to the last digit, because
the shift vote that follows decides where borrowed sky is placed.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import repair as R                                        # noqa: E402
from modules.repair import _find_star_streaks                          # noqa: E402


def _old_finder(g):
    sky = np.median(g)
    thr = max(sky + 22, np.percentile(g, 92))
    b = cv2.morphologyEx((g > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for k in range(1, n):
        if st[k, 4] < 6 or st[k, 4] > 4000:
            continue
        ys, xs = np.where(lab == k)
        w = g[ys, xs].astype(np.float64)
        out.append((float((xs * w).sum() / w.sum()),
                    float((ys * w).sum() / w.sum()), float(w.sum())))
    if len(out) > R._MAX_TRACK_STARS:
        out.sort(key=lambda s: s[2], reverse=True)
        out = out[:R._MAX_TRACK_STARS]
    return out


def _sky(seed, h, w, stars, noise=6):
    """A dark noisy patch with short bright smears (some touching the edge)."""
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(40, noise, (h, w)), 0, 255).astype(np.uint8)
    for _ in range(stars):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        length = int(rng.integers(2, 14))
        ang = float(rng.uniform(0, np.pi))
        x2, y2 = int(x + length * np.cos(ang)), int(y + length * np.sin(ang))
        cv2.line(g, (x, y), (x2, y2), int(rng.integers(120, 256)), int(rng.integers(1, 4)))
    return g


def test_identical_on_many_random_sky_patches():
    found = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        h, w = int(rng.integers(30, 700)), int(rng.integers(30, 700))
        g = _sky(seed, h, w, stars=int(rng.integers(0, 120)))
        old, new = _old_finder(g), _find_star_streaks(g)
        assert old == new, f"patch {seed} differs"
        found += len(old)
    assert found > 500, "the random patches stopped containing stars"


def test_more_than_the_cap_keeps_the_same_brightest_stars():
    g = _sky(3, 600, 600, stars=400)
    old = _old_finder(g)
    assert len(old) == R._MAX_TRACK_STARS
    assert _find_star_streaks(g) == old


def test_stars_on_every_edge_and_corner():
    g = np.full((80, 90), 40, np.uint8)
    for (y, x) in ((0, 0), (0, 89), (79, 0), (79, 89), (0, 40), (79, 40), (40, 0), (40, 89)):
        cv2.circle(g, (x, y), 3, 250, -1)
    assert len(_old_finder(g)) >= 4
    assert _find_star_streaks(g) == _old_finder(g)


def test_empty_flat_and_noise_only_patches():
    for g in (np.zeros((50, 50), np.uint8), np.full((50, 50), 128, np.uint8),
              np.random.default_rng(1).integers(0, 255, (60, 70), dtype=np.uint8)):
        assert _find_star_streaks(g) == _old_finder(g)


def test_the_track_step_uses_the_box_limited_finder():
    src = (REPO / "modules" / "repair.py").read_text(encoding="utf-8")
    assert "_find_star_streaks(gp), _find_star_streaks(gn)" in src
    assert "ys, xs = np.where(lab == k)" not in src, "the whole-patch lookup came back"
