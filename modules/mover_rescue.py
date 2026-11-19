"""Keep a moving dash that the star-size rule would have thrown away.

WHAT: the star-size rule drops a small, thin, non-red piece as "a star". A plane's white strobe dash is
the same size as a star streak, so it gets dropped too, and it leaves a dotted line in the finished stack
(Silvana Della Camera, Tree and Trails, frames 1586-1596: 11 dashes in a row, each 29-44 px long against a
star cutoff of 45). Size cannot tell them apart, and neither can direction (the dashes pointed within 8
degrees of the stars around them). Motion can: between frames the stars there shift about 1 px, the dash
steps about 25 px.

HOW: this module uses the star-size module's own definition of a star, a blob that shows up AGAIN within
_PARTNER_PX (15 px, or a few times the set's measured star shift when that is smaller) in the next frame. A dropped
piece whose nearest blob in EVERY available neighbouring frame is farther than that is not a star: it moved. It is handed back to be added to the cleaning mask.

WHEN NOT: a piece is never rescued when it is over the foreground, when no blob of its own can be found in
its frame (a model speck, not a visible dash), when no neighbouring frame is available, or when ANY
neighbour has a blob within 15 px (a real star, or a dash moving less than 15 px per frame, which looks the
same as a star by this test; that limit is deliberate, a dense star field then keeps its old behaviour).

HOW WE KNOW IT WORKS: the run log's detect record carries "mover_rescued", one entry per piece with its
position, the distance to the nearest blob in each neighbour, and what became of it, plus "mover_checked",
the number of dropped pieces examined. A normal set rescues none or a handful; a run that rescues hundreds
means the rule is catching stars.

WHAT GUARDS IT: tests/test_mover_rescue.py (a moving dash is rescued, a star with a partner is not, a piece
over the foreground is not, no neighbour means no rescue).
"""
import numpy as np
import cv2

from modules.star_size import _star_blobs, _gray, _PARTNER_PX

WINDOW_PX = 150      # half-size of the crop around a piece when looking for blobs in a frame


def _blob_centres(gray, cx, cy):
    """Centres [x, y] of the star-like blobs in a window around (cx, cy), in full-frame pixels."""
    h, w = gray.shape[:2]
    x0, y0 = max(0, int(cx) - WINDOW_PX), max(0, int(cy) - WINDOW_PX)
    x1, y1 = min(w, int(cx) + WINDOW_PX), min(h, int(cy) + WINDOW_PX)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return np.zeros((0, 2))
    cents, _, _ = _star_blobs(gray[y0:y1, x0:x1], None)
    if len(cents) == 0:
        return np.zeros((0, 2))
    return np.asarray(cents, dtype=np.float64) + np.array([x0, y0], dtype=np.float64)


def _nearest(cents, cx, cy):
    """Distance in px from (cx, cy) to the nearest centre, or infinity if there are none."""
    if len(cents) == 0:
        return float("inf")
    return float(np.hypot(cents[:, 0] - cx, cents[:, 1] - cy).min())


def partner_radius(star_shift):
    """How far to look for a star's partner: a few times the set's measured star shift, never below 6 px
    (blob centres are good to about 2 px) and never above the star-size module's 15 px. With no measured
    shift, the full 15 px."""
    if not star_shift or not np.isfinite(star_shift):
        return float(_PARTNER_PX)
    return float(min(_PARTNER_PX, max(6.0, 3.0 * float(star_shift) + 3.0)))


def find_movers(pieces, frame, neighbours, sky_mask=None, radius=None):
    """Which of a frame's star-size-dropped pieces moved, so they are not stars.

    pieces: dicts with "coords" (N x 2 array of row, col) and "bbox" [x0, y0, x1, y1], from the detector.
    frame: this frame's image (BGR or grey). neighbours: the previous and next frames' images (None entries
    are skipped). sky_mask: 255 = sky, 0 = foreground, or None.
    radius: how far a star's partner may sit (see partner_radius); None means the full 15 px.
    Returns (movers, checked): movers is a list of dicts (piece, cx, cy, partner_px, why); checked is how many
    pieces were examined.
    """
    radius = float(_PARTNER_PX if radius is None else radius)
    if not pieces:
        return [], 0
    gray = _gray(frame)
    ngrays = [_gray(n) for n in neighbours if n is not None]
    movers = []
    checked = 0
    for p in pieces:
        checked += 1
        co = p["coords"]
        cx, cy = float(co[:, 1].mean()), float(co[:, 0].mean())
        if sky_mask is not None:
            yy = min(sky_mask.shape[0] - 1, max(0, int(cy)))
            xx = min(sky_mask.shape[1] - 1, max(0, int(cx)))
            if sky_mask[yy, xx] == 0:
                continue                                  # over the foreground
        if not ngrays:
            continue                                      # nothing to compare against
        if _nearest(_blob_centres(gray, cx, cy), cx, cy) > _PARTNER_PX:   # its own blob: always the wide radius
            continue                                      # no visible blob of its own: a model speck
        dists = [_nearest(_blob_centres(g, cx, cy), cx, cy) for g in ngrays]
        if all(d > radius for d in dists):
            far = ["%.0f px" % d if np.isfinite(d) else "none in view" for d in dists]
            movers.append({
                "piece": p, "cx": cx, "cy": cy, "partner_px": [None if not np.isfinite(d) else round(d, 1) for d in dists],
                "why": "star-sized and not red, but no blob within %.0f px in %s neighbouring frame(s) "
                       "(nearest: %s); a star in this set re-appears within %.0f px, so this moved"
                       % (radius, len(ngrays), ", ".join(far), radius)})
    return movers, checked


def polygon_for(piece, pad=2):
    """A rotated box (list of four [x, y] corners) around a piece's pixels, padded a little."""
    co = piece["coords"]
    pts = np.column_stack([co[:, 1], co[:, 0]]).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    box = cv2.boxPoints(((cx, cy), (w + 2 * pad, h + 2 * pad), ang))
    return box.round().astype(int).tolist()
