"""The speck cleanup must not erase starlight (modules/sky_dots.py).

Field case, 2026-08-03: Bruce's Perseid star trail came back with bites taken
out of the trails. Measured on his real frames, the checkbox removed 38,878
pixels of which 45 were genuine fixed defects -- 0.12% -- while missing 54 of
the 99 real defects present. Cause: the small-blob pass judged the FINISHED
STACK alone, deciding "isolated dot?" by looking only 4-6 pixels out. His
sequences have a gap between exposures, so each star draws a DOTTED line, and
every bead of every trail read as a lone dot.

The fix: a blob may only be erased if the sampled frames agree it is stuck to
the sensor. A stuck pixel lands on the same pixel in every frame; a star passes
through any given pixel once. Nothing in the stack alone can tell them apart.

These use synthetic frames -- no photos needed, runs in milliseconds.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

BEADS = [(40, 40), (40, 52), (40, 64), (40, 76), (40, 88)]   # a dotted trail
HOT = (150, 150)                                             # a stuck pixel


def _scene():
    """A stack holding a dotted trail plus one stuck pixel, and sample frames
    where the stuck pixel is in EVERY frame and each bead in only one."""
    big = np.zeros((200, 200, 3), np.uint8)
    for (y, x) in BEADS:
        big[y - 1:y + 2, x - 1:x + 2] = 220
    big[HOT[0] - 1:HOT[0] + 2, HOT[1] - 1:HOT[1] + 2] = 200
    frames = []
    for i in range(20):
        f = np.zeros((200, 200, 3), np.uint8)
        f[HOT[0] - 1:HOT[0] + 2, HOT[1] - 1:HOT[1] + 2] = 200   # always there
        if i < len(BEADS):                                       # each bead once
            y, x = BEADS[i]
            f[y - 1:y + 2, x - 1:x + 2] = 220
        frames.append(f)
    return big, frames


def test_dotted_trail_beads_are_not_erased():
    """THE regression net: a bead of a dotted trail is starlight, not a defect."""
    from modules.sky_dots import _detect_map
    m = _detect_map(*_scene())
    for (y, x) in BEADS:
        assert m[y, x] == 0, f"bead at {(y, x)} was flagged -- that is a star"


def test_a_real_stuck_pixel_is_still_erased():
    from modules.sky_dots import _detect_map
    m = _detect_map(*_scene())
    assert m[HOT[0], HOT[1]] > 0, "a pixel bright in every frame must be removed"


def test_persistence_counts_frames_not_appearances():
    from modules.sky_dots import _persistence_count
    _, frames = _scene()
    assert _persistence_count(frames, HOT[1], HOT[0]) == len(frames)
    y, x = BEADS[0]
    assert _persistence_count(frames, x, y) <= 2


def test_a_removed_speck_never_leaves_a_hole():
    """Field case, 2026-08-09: the removals came back as BLACK DOTS.

    A speck on or beside a trail used to be erased from every frame and the
    sequence re-stacked, so that pixel ended up holding one frame's sky value
    while its neighbours held the brightest of hundreds of frames -- a hole
    darker than the sky. Measured on Bruce's Perseid trail: 171 of 683 such spots
    sat more than 8 levels below their own surroundings. The fill now runs on the
    finished stack and is floored at the surrounding sky, so this cannot recur.
    """
    from modules.sky_dots import _fill_specks

    big = np.full((120, 120, 3), 30, np.uint8)          # sky, well above black
    big[58:62, :] = 150                                 # a bright trail across it
    big[59, 60] = 255                                   # a speck sitting on it
    mask = np.zeros((120, 120), np.uint8)
    mask[57:62, 58:63] = 255                            # 5px patch, as in a real run

    out, n, _ = _fill_specks(big, mask)
    assert n == 1, "one speck, one fill"
    assert out[mask > 0].max(axis=1).min() >= 30, \
        "no filled pixel may end up darker than the sky around it"
    # The patch spans the trail's whole width, so the trail cannot come back at
    # full strength -- nothing is left to copy from across the gap. It must still
    # read as trail rather than sky. On Bruce's real stack the fill holds about
    # 78% of the trail's brightness, and only 2% of specks sit on a lit trail at
    # all, so a shortfall here is a dip in a few dozen places, not a notch.
    assert out[59, 60].max() >= 70, \
        f"the trail was notched: filled with {out[59, 60].max()}, trail is 150"


def test_a_speck_in_open_sky_is_filled_with_sky():
    """The other half: nothing to fill along, so match the sky -- not black, and
    not the brightness of a trail passing by on one side only."""
    from modules.sky_dots import _fill_specks

    big = np.full((120, 120, 3), 30, np.uint8)
    big[:, 20:24] = 200                                 # a trail well off to one side
    big[60, 60] = 255                                   # a lone speck in open sky
    mask = np.zeros((120, 120), np.uint8)
    mask[58:63, 58:63] = 255

    out, _, _ = _fill_specks(big, mask)
    assert 25 <= out[60, 60].max() <= 40, \
        f"open sky should read as sky, got {out[60, 60].max()}"


def test_a_lone_bright_dot_is_not_erased_without_proof():
    """The rule that ate Bruce's stars twice, in one test.

    A dot can be small, sharply peaked, oddly coloured and the brightest thing
    for 14 pixels around, and still be a star -- his stars move only 0.2 to 5.8 px
    between frames, so a bead's companions are inside the radius any isolation
    test looks past. Measured against ground truth, judging isolation at 6px
    erased 38,878 pixels for 45 real defects; at 14px, 1,693 dots for 22. Only
    persistence decides now. Nothing here is persistent, so nothing may go.
    """
    from modules.sky_dots import _detect_map

    big = np.zeros((200, 200, 3), np.uint8)
    big[99:102, 99:102] = 240                      # a lone, bright, isolated dot
    frames = []
    for i in range(20):
        f = np.zeros((200, 200, 3), np.uint8)
        if i == 7:                                  # present in ONE frame only
            f[99:102, 99:102] = 240
        frames.append(f)

    assert _detect_map(big, frames)[100, 100] == 0, \
        "a one-off dot was erased on looks alone -- that is how stars get eaten"


def test_the_runs_own_hot_pixel_map_is_honoured():
    """The clean marks stuck pixels batch by batch and saves them. Whatever it
    found must be removed, even where a 40-frame sample of the night disagrees --
    a defect that only wakes up late still appears in its own batch's map."""
    from modules.sky_dots import _detect_map

    big, frames = _scene()
    run_map = np.zeros((200, 200), np.uint8)
    run_map[170, 30] = 255                          # found during the run, nowhere else

    assert _detect_map(big, frames, run_map=run_map)[170, 30] > 0, \
        "the run's own map was ignored"


def test_a_one_off_fleck_is_removed_but_a_moving_star_is_not():
    """The fleck rule, both directions.

    The sky turns, so a real star's light sits at a known spot one frame earlier
    and another one frame later. Follow it there: a star is nearly as bright at
    those spots, a cosmic ray leaves empty sky. Measured on Bruce's Perseid
    sequence this erased every one of 12 spots known to appear in a single frame,
    and 3 of 800 control points on real trails.
    """
    from modules.sky_dots import _flecks

    pts = [(10, 10), (20, 20), (30, 30)]
    peak = np.array([200, 200, 30])      # bright fleck / bright star / too dim
    evidence = np.array([15, 180, 5])    # empty sky / the star arrived / unknown
    ids = _flecks(pts, peak, evidence)

    assert 1 in ids, "the sky's rotation does not account for this dot -- a fleck"
    assert 2 not in ids, "the star was there in both neighbouring frames"
    assert 3 not in ids, \
        "too dim to judge; guessing wrong here puts a mark on a real star"


def test_a_sequence_whose_motion_cannot_be_read_loses_no_stars():
    """If the sky's motion can't be measured, the fleck test is pointed at the
    wrong pixels and would condemn stars. Removing nothing beats that."""
    from modules import sky_dots

    src = (REPO / "modules" / "sky_dots.py").read_text(encoding="utf-8")
    i = src.index("def _sky_motion")
    body = src[i:i + 3000]
    assert "return None" in body, "an unreadable motion fit must bail, not guess"
    assert "spread > 1.5" in body, \
        "three pairs across the night must agree before the fit is believed"
    assert sky_dots._FLECK_RATIO <= 0.5, \
        "a looser ratio starts erasing faint stars (measured: 5 of 400 at 0.5)"


def test_sky_glow_is_not_mistaken_for_a_stuck_pixel():
    """Field case, 2026-08-10: 8,720 patches on a bright sky, on false evidence.

    "Present in this frame" used to mean "brighter than 40". On Bruce's dark EOS R
    sky 0.2% of plain sky clears 40; on his brighter 6D sky 23.8% does, and this
    reads the brightest of a 3x3 patch, so nearly every candidate came back
    persistent on sky glow alone. The test now asks whether the pixel stands out
    from the sky AROUND it, which is what build_hot_pixel_map has always done.
    """
    from modules.sky_dots import _persistence_count

    # A bright sky, no defect: every pixel is well above any fixed bar.
    bright = [np.full((60, 60, 3), 70, np.uint8) for _ in range(20)]
    assert _persistence_count(bright, 30, 30) == 0, \
        "sky glow was counted as a stuck pixel"

    # The same bright sky with a genuine defect standing above it.
    for f in bright:
        f[29:32, 29:32] = 140
    assert _persistence_count(bright, 30, 30) == len(bright), \
        "a defect that stands above a bright sky must still be caught"


def test_the_blob_pass_actually_consults_the_frames():
    """Guard the wiring: the check is worthless if the pass stops calling it."""
    src = (REPO / "modules" / "sky_dots.py").read_text(encoding="utf-8")
    i = src.index("def _detect_map")
    body = src[i:i + 6000]
    assert "_persistence_count" in body, \
        "the small-blob pass must still consult the frames"
    assert "run_map" in body, \
        "the run's own stuck-pixel map must still be folded in"
