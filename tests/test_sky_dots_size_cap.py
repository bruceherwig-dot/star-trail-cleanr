"""Speck removal must leave scenery-sized shapes alone.

Regression (2026-09-30, Sean Parker's aurora set): the soft dark edge of each cactus
was flagged as a ring of specks, the rings merged into one 298,255 px shape, and the
lift pass painted the whole shape with one sky colour taken near its centre (yellow
horizon glow), drawing a bright yellow outline around every cactus.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import cv2

from modules import sky_dots


def test_a_speck_sized_shape_is_kept_and_a_huge_one_is_dropped():
    allmap = np.zeros((400, 600), np.uint8)
    allmap[50:55, 60:65] = 255            # 25 px: a real speck
    allmap[200:300, 100:500] = 255        # 40,000 px: scenery
    out = sky_dots._drop_oversize(allmap)
    assert out[52, 62] == 255, "a real speck must still be removed"
    assert out[250, 300] == 0, "a scenery-sized shape must be left alone"
    assert allmap[250, 300] == 255, "the caller's map must not be modified"


def test_a_ring_of_specks_around_scenery_is_not_painted_with_one_colour():
    """The Sean Parker case end to end: a yellow band low in the picture, red sky
    above, a dark post running up from one to the other with a thin flagged ring
    around it. The ring must come out exactly as shot, not painted yellow."""
    H, W = 300, 200
    big = np.zeros((H, W, 3), np.uint8)
    big[:200] = (60, 50, 140)             # red sky (BGR)
    big[200:] = (40, 230, 250)            # yellow glow
    big[40:260, 90:110] = (5, 5, 5)       # dark post
    ring = np.zeros((H, W), np.uint8)
    ring[36:264, 86:114] = 255
    ring[40:260, 90:110] = 0              # a ring about 4 px thick, 800+ px
    out, n, _ = sky_dots._fill_specks(big, sky_dots._drop_oversize(ring))
    assert n == 0
    assert np.array_equal(out, big), "scenery edges must be left exactly as shot"


# ── a star near the pole is not a stuck pixel ────────────────────────────────

def test_a_pole_star_sized_shape_is_not_a_stuck_pixel_but_a_real_cluster_is():
    """Real defect clusters are ~5 px across (25 px at most); a bright pole star is
    a saturated blob dozens of pixels across. The raw-shape limit separates them."""
    allmap = np.zeros((300, 400), np.uint8)
    allmap[50:55, 60:65] = 255            # a 5x5 defect cluster (25 px): kept
    cv2.circle(allmap, (250, 150), 6, 255, -1)    # a ~113 px blob: a star
    out = sky_dots._drop_oversize(allmap, sky_dots.MAX_DEFECT_AREA, "test")
    assert out[52, 62] == 255, "a real defect cluster must still be removed"
    assert out[150, 250] == 0, "a pole-star-sized blob must be left alone"


def test_polaris_survives_the_speck_step_and_a_real_defect_does_not():
    """End to end: frames with a saturated star that stays on the same pixels all
    night (it sits at the pole) and a stuck colour cluster elsewhere, both in the
    run's saved map. The cluster must be painted out; the star must come through
    untouched."""
    import tempfile
    H, W = 240, 320
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        names = []
        for i in range(24):
            f = np.full((H, W, 3), 18, np.uint8)
            f += rng.integers(0, 3, size=f.shape, dtype=np.uint8)
            cv2.circle(f, (100 + i // 12, 80), 5, (255, 255, 255), -1)     # the pole star
            f[170:173, 220:223] = (240, 20, 240)                            # a stuck cluster
            f[20 + i * 8, 200 + i * 3] = 200                                # a moving star
            n = f"f{i:02d}.png"
            cv2.imwrite(str(Path(d) / n), f)
            names.append(n)
        big = np.max([cv2.imread(str(Path(d) / n)) for n in names], axis=0)
        run_map = np.zeros((H, W), np.uint8)
        cv2.circle(run_map, (100, 80), 6, 255, -1)          # marked as "stuck": wrongly
        run_map[170:173, 220:223] = 255                     # marked as stuck: rightly
        # a mask with no foreground in it: the all-sky case, and it keeps the
        # unmasked-landscape guard (built for 22 MP frames) out of a tiny test frame
        # The older 300 px limit (judged after the 1px growth) is switched off here so
        # that ONLY the new raw-shape limit decides: in Sean Parker's real set the old
        # limit did not catch Polaris (its map pieces were 99 and 170 px grown).
        old_cap = sky_dots.MAX_SPECK_AREA
        sky_dots.MAX_SPECK_AREA = 10 ** 9
        try:
            out = sky_dots.remove_specks(d, names, big, np.zeros((H, W), np.uint8),
                                         lambda p: cv2.imread(p), run_map=run_map)
        finally:
            sky_dots.MAX_SPECK_AREA = old_cap
    assert np.array_equal(out[70:91, 90:111], big[70:91, 90:111]), \
        "the pole star was painted over"
    assert int(out[171, 221].max()) < 120, "the real stuck cluster was not removed"


def test_a_clump_of_small_pieces_that_will_merge_is_left_alone_whole():
    """Many tiny marks packed together (a horizon of small lights, a cactus edge
    flagged as a ring of bits) are each under the per-piece limit but grow into one
    big clump. The clump is scenery: it must be removed whole, not painted bit by
    bit. A lone small cluster elsewhere must survive."""
    allmap = np.zeros((300, 600), np.uint8)
    for x in range(40, 400, 4):                    # a dense row of 2x2 marks, 4 px apart
        allmap[150:152, x:x + 2] = 255             # each 4 px; grown, they all join
    allmap[40:44, 500:504] = 255                   # a lone 4x4 defect, far away
    out = sky_dots._drop_oversize(allmap, sky_dots.MAX_SPECK_AREA, "test", grown=True)
    assert out[150:152, 40:400].sum() == 0, "the clump must be left alone whole"
    assert out[40:44, 500:504].all(), "a lone small defect must still be removed"


# ── a speck on a silhouette's edge must not become a pale blotch ─────────────

def _edge_scene():
    """Dark scenery on the left (value 8, columns 0-32), sky on the right (value
    100), a bright stuck-pixel cluster right on the boundary. The ring around the
    cluster is mostly sky (so the sky floor reads 100) while the patch itself
    straddles the edge, its two leftmost columns lying on the dark object -- the
    case that painted pale blotches onto trees and cacti."""
    big = np.full((60, 80, 3), 100, np.uint8)
    big[:, :33] = 8
    big[28:32, 32:37] = 255
    mask = np.zeros((60, 80), np.uint8)
    mask[27:33, 31:38] = 255                      # the cluster, grown by a pixel
    return big, mask


def test_a_speck_on_a_dark_edge_keeps_its_dark_side_dark_and_its_sky_side_at_sky():
    big, mask = _edge_scene()
    out, n, _ = sky_dots._fill_specks(big, mask)
    patch = mask > 0
    dark_side = patch & (np.arange(80)[None, :] < 33)
    sky_side = patch & (np.arange(80)[None, :] >= 36)     # clear of the edge's ramp
    assert out[dark_side].max() < 40, "the dark side of the patch was lifted to sky colour"
    assert out[sky_side].min() >= 90, "the sky side of the patch fell below the sky"
    # and the edge keeps a smooth ramp from dark to sky, not a jump
    row = out[30, 31:37, 0].astype(int)
    assert (np.diff(row) >= 0).all() and row[0] < 40 and row[-1] >= 90, row.tolist()


def test_an_ordinary_open_sky_speck_is_still_lifted_to_the_sky():
    """The scenery rule must not weaken the no-holes guarantee in open sky."""
    big = np.full((60, 80, 3), 100, np.uint8)
    big[30, 40] = 255
    mask = np.zeros((60, 80), np.uint8)
    mask[28:33, 38:43] = 255
    out, n, _ = sky_dots._fill_specks(big, mask)
    assert out[mask > 0].min() >= 95, "a patch in open sky came out darker than the sky"


def test_the_edge_test_fails_when_the_scenery_rule_is_off():
    """Prove the edge test above is a real test: with the rule off, the dark side IS
    lifted to the sky colour."""
    big, mask = _edge_scene()
    old = sky_dots._FG_FRAC
    sky_dots._FG_FRAC = 0.0
    try:
        out, _, _ = sky_dots._fill_specks(big, mask)
    finally:
        sky_dots._FG_FRAC = old
    dark_side = (mask > 0) & (np.arange(80)[None, :] < 33)
    assert out[dark_side].max() >= 40, "expected the old behaviour with the rule off"


def test_a_spot_that_is_not_brighter_than_its_surroundings_is_left_alone():
    """A flagged spot no brighter than the picture around it has nothing to remove.
    Real example: dark tree-edge pixels flagged as specks, then filled with a paler
    blend -- a gray blotch on a dark object. A genuinely bright speck is kept."""
    big = np.full((80, 120, 3), 100, np.uint8)         # sky
    big[30:34, 20:24] = 30                             # a dark scenery edge, flagged
    big[30:34, 80:84] = 255                            # a real bright speck, flagged
    mask = np.zeros((80, 120), np.uint8)
    mask[29:35, 19:25] = 255
    mask[29:35, 79:85] = 255
    out = sky_dots._drop_not_brighter(big, mask)
    assert out[31, 21] == 0, "the dark spot must be left alone"
    assert out[31, 81] == 255, "the bright speck must still be removed"
    assert mask[31, 21] == 255, "the caller's map must not be modified"
