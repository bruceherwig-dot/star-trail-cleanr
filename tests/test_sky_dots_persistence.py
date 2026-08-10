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


def test_the_blob_pass_actually_consults_the_frames():
    """Guard the wiring: the check is worthless if the pass stops calling it."""
    src = (REPO / "modules" / "sky_dots.py").read_text()
    i = src.index("def _detect_map")
    body = src[i:i + 3000]
    assert "_persistence_count" in body, \
        "the small-blob pass must require persistence before erasing"
