"""
Crossing peel -- cut a tangle of crossing trails into one clean rectangle per trail.

WHERE THIS FITS IN THE APP
--------------------------
Star Trail CleanR's detector often outlines two or more crossing trails as ONE connected
blob. The older splitter (modules/crossing_splitter.py) picks one "spine" line and treats
everything else as arms, which works for an X or a T but not for a K, a V, three or more
trails, or two parallel trails (they got merged back into one band). This module handles
those by working trail by trail:

  1. Skeleton the blob and measure the single-trail width (median of the skeleton's
     distance transform, doubled -- the median ignores junction bulges).
  2. Find the strongest straight line (most skeleton points on it), re-fit it through
     the points it actually found, and give it every blob pixel that lies along its own
     length and within about one trail width of it. Remove that line's skeleton points
     and repeat (at most 12 lines).
  3. Join line pieces that are one trail broken in two (where a crossing pinches the
     model's outline). Without this the broken end becomes a lonely fragment and the
     seam bridge in detect_pipeline re-runs the model there and adds fat pieces.
  4. Leftover blob pixels no line claimed: thin fringe is merged back into the piece it
     touches; a real leftover chunk stays as its own piece.
  5. GUARD: only split when it really is a crossing. At least two arms (each at least
     MIN_ARM_WIDTHS trail widths long) must point GUARD_DEG or more apart, OR the older
     splitter's own crossing evidence says "crossing" AND the lines run through each other (this
     keeps low-angle X crossings; two lines meeting only at their ends are a bend and stay whole).
     A gentle bend, a near-straight fork, or a ragged edge chunk (foreground clutter)
     stays one piece.
  6. Each line piece becomes a clean RECTANGLE: the line's axis, its length, and the
     median cross-section width of the model's outline plus RECT_MARGIN.

Measured (2026-11-05, see memory project_crossing_bench.md): synthetic crossings 50 of
50 lines recovered (the older splitter plus band rescue: 41 of 50); 12 Lake Mead frames
scored against the reviewed CVAT trails: coverage 0.839 against 0.767, no missed trail.
On the Spain set there was no measurable difference.

Public API
----------
peel_split(mask, frame_px=None) -> (pieces, info)
    pieces is [mask] unchanged when no split is warranted, or a list of uint8 0/255
    masks (same shape as `mask`) when it is a crossing. info is a dict that says why.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .crossing_splitter import has_crossing_evidence, _SPLIT_AREA_MIN, _REF_FRAME_PX

# -- Constants ----------------------------------------------------------------
MIN_LINE_WIDTHS = 1.5      # a line must be at least this many trail widths long to count
BAND_MULT = 0.62           # claim blob pixels within this x trail width of the line centre
MAX_LINES = 12             # most lines peeled from one blob
GAP_BREAK_WIDTHS = 4.0     # a gap bigger than this many trail widths ends a line's run
MIN_ARM_WIDTHS = 4.0       # an arm shorter than this does not count toward the guard's angle test
GUARD_DEG = 35.0           # arms must point at least this far apart (unless crossing evidence)
SLIVER_WIDTHS = 0.5        # leftover thinner than this x trail width is fringe: merged into its neighbour
LEFTOVER_MIN_WIDTHS2 = 1.5 # leftover chunk smaller than this many width^2 is outline slack: dropped
MERGE_ANGLE_DEG = 12.0     # two line pieces this close in direction may be one trail broken by a pinch
MERGE_GAP_WIDTHS = 6.0     # ...if the gap between them along the line is at most this many trail widths
MERGE_SIDE_WIDTHS = 0.8    # ...and one lies on the other's centre line within this many trail widths
RECT_MARGIN = 0.10         # width margin added to the model's own median width
INTERIOR_MARGIN_WIDTHS = 3.0 # lines meeting less than this far from a piece's ends are meeting end to end (a bend), not crossing
MIN_TRAIL_WIDTH_PX = 3.0   # a blob whose measured trail width is below this is not worth peeling


# -- Geometry helpers ----------------------------------------------------------

def _trail_width(blob):
    """Single-trail width of a 0/1 blob, plus its skeleton. Median of 2 x distance at the skeleton."""
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    sk = skeletonize(blob > 0)
    v = dt[sk]
    return (2.0 * float(np.median(v)) if len(v) else 0.0), sk


def _best_line(pts, w):
    """Strongest straight line through skeleton points pts (n,2 as x,y).

    Tries every whole degree; for each, bins the points by perpendicular position and
    keeps the angle and position with the most points (a bin plus its two neighbours).
    Returns (theta_deg, rho, member_mask) where members lie within about 1.6 bins."""
    best_score, best_th, best_rho = 0.0, 0, 0.0
    tol = max(1.5, 0.30 * w)
    for th in range(0, 180):
        a = np.radians(th)
        perp = -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)
        lo = perp.min()
        nb = int((perp.max() - lo) / tol) + 2
        h = np.bincount(((perp - lo) / tol).astype(int), minlength=nb).astype(float)
        hs = h + np.roll(h, 1) + np.roll(h, -1)
        k = int(hs.argmax())
        if hs[k] > best_score:
            best_score, best_th, best_rho = hs[k], th, lo + k * tol + tol / 2
    a = np.radians(best_th)
    perp = -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)
    return best_th, best_rho, np.abs(perp - best_rho) <= 1.6 * tol


def _apart(a, b):
    """Smallest angle between two orientations, in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _axis(piece):
    """Principal axis of a piece: (points as x,y float32, unit direction, a point on the line)."""
    ys, xs = np.nonzero(piece)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = [float(v[0]) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)]
    return pts, np.array([vx, vy]), np.array([x0, y0])


def _axis_angle(piece):
    """Orientation of a piece's principal axis in degrees, [0, 180)."""
    _, u, _ = _axis(piece)
    return float(np.degrees(np.arctan2(u[1], u[0])) % 180.0)


def _long_side(piece):
    """Length of the long side of the piece's tightest rotated box."""
    ys, xs = np.nonzero(piece)
    return float(max(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2))[1]))


def _has_interior_crossing(arm_pieces, w):
    """True when at least two arms cross THROUGH each other: the point where their lines meet lies inside
    at least one of the two pieces, INTERIOR_MARGIN_WIDTHS trail widths or more from both of its ends (a bend's first line still pokes about two widths past the corner). A bend has the two
    lines meeting only at their ends (one trail turning), so it returns False; a low-angle X returns True."""
    ax = [_axis(p) for p in arm_pieces]
    for i in range(len(ax)):
        for j in range(i + 1, len(ax)):
            pi, ui, ci = ax[i]
            pj, uj, cj = ax[j]
            if _apart(np.degrees(np.arctan2(ui[1], ui[0])) % 180, np.degrees(np.arctan2(uj[1], uj[0])) % 180) < 15.0:
                continue
            m = np.array([[ui[0], -uj[0]], [ui[1], -uj[1]]])
            if abs(np.linalg.det(m)) < 1e-6:
                continue
            t, _ = np.linalg.solve(m, cj - ci)
            x = ci + t * ui
            for pts, u, c in ((pi, ui, ci), (pj, uj, cj)):
                a = (pts - c) @ u
                s_ = float((x - c) @ u)
                if s_ - a.min() >= INTERIOR_MARGIN_WIDTHS * w and a.max() - s_ >= INTERIOR_MARGIN_WIDTHS * w:
                    return True
    return False


# -- Steps ---------------------------------------------------------------------

def _peel_lines(blob, w, sk):
    """Steps 2: peel straight lines off the blob. Returns a list of uint8 0/255 pieces, one per line."""
    ys, xs = np.nonzero(sk)
    pts = np.column_stack([xs, ys]).astype(float)
    alive = np.ones(len(pts), bool)
    by, bx = np.nonzero(blob)
    bp = np.column_stack([bx, by]).astype(float)
    pieces = []
    for _ in range(MAX_LINES):
        if alive.sum() < MIN_LINE_WIDTHS * w * 0.5:
            break
        p = pts[alive]
        th, rho, mem = _best_line(p, w)
        a = np.radians(th)
        ux, uy = np.cos(a), np.sin(a)
        along = p[:, 0] * ux + p[:, 1] * uy
        idx = np.nonzero(mem)[0]
        if len(idx) == 0:
            break
        order = idx[np.argsort(along[idx])]
        al = along[order]
        # one line = the longest unbroken run of member points (a gap over GAP_BREAK_WIDTHS ends a run)
        cuts = np.nonzero(np.diff(al) > GAP_BREAK_WIDTHS * w)[0]
        runs = np.split(np.arange(len(order)), cuts + 1)
        run = max(runs, key=lambda r: al[r[-1]] - al[r[0]])
        lo, hi = al[run[0]], al[run[-1]]
        # re-fit through the points actually found: the 1-degree search leaves up to ~20 px of sideways
        # error at the far end of a long line, which would leave thin edge slivers
        rp = p[order[run]]
        if len(rp) >= 5:
            vx, vy, x0, y0 = [float(v[0]) for v in cv2.fitLine(rp.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)]
            th = np.degrees(np.arctan2(vy, vx)) % 180
            a = np.radians(th)
            ux, uy = np.cos(a), np.sin(a)
            rho = -x0 * np.sin(a) + y0 * np.cos(a)
            al2 = rp[:, 0] * ux + rp[:, 1] * uy
            lo, hi = float(al2.min()), float(al2.max())
        if hi - lo < MIN_LINE_WIDTHS * w:
            # not a real line: retire these points so we do not loop on them
            ai = np.nonzero(alive)[0]
            alive[ai[order]] = False
            continue
        bal = bp[:, 0] * ux + bp[:, 1] * uy
        bpr = -bp[:, 0] * np.sin(a) + bp[:, 1] * np.cos(a)
        sel = (np.abs(bpr - rho) <= BAND_MULT * w) & (bal >= lo - 0.5 * w) & (bal <= hi + 0.5 * w)
        piece = np.zeros(blob.shape, np.uint8)
        piece[by[sel], bx[sel]] = 255
        pieces.append(piece)
        # retire every skeleton point in this band so a crossing line keeps its own points elsewhere
        pal = pts[:, 0] * ux + pts[:, 1] * uy
        ppr = -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)
        alive &= ~((np.abs(ppr - rho) <= 0.9 * w) & (pal >= lo - 0.5 * w) & (pal <= hi + 0.5 * w))
    return pieces


def _merge_collinear(pieces, w):
    """Step 3: join line pieces that are one trail broken in two. Returns a new list."""
    pieces = list(pieces)
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                pi, ui, ci = _axis(pieces[i])
                pj, uj, cj = _axis(pieces[j])
                if _apart(np.degrees(np.arctan2(ui[1], ui[0])) % 180,
                          np.degrees(np.arctan2(uj[1], uj[0])) % 180) > MERGE_ANGLE_DEG:
                    continue
                longer_is_i = len(pi) >= len(pj)
                u, c = (ui, ci) if longer_is_i else (uj, cj)
                nrm = np.array([-u[1], u[0]])
                short, long_ = (pj, pi) if longer_is_i else (pi, pj)
                if np.median(np.abs((short - c) @ nrm)) > MERGE_SIDE_WIDTHS * w:
                    continue
                a_s, a_l = (short - c) @ u, (long_ - c) @ u
                gap = max(a_s.min() - a_l.max(), a_l.min() - a_s.max(), 0.0)
                if gap > MERGE_GAP_WIDTHS * w:
                    continue
                pieces[i] = ((pieces[i] > 0) | (pieces[j] > 0)).astype(np.uint8) * 255
                del pieces[j]
                changed = True
                break
            if changed:
                break
    return pieces


def _add_leftovers(blob, lines, w):
    """Step 4: handle blob pixels no line claimed. Returns (lines_with_fringe_merged, leftover_pieces)."""
    cov = np.zeros(blob.shape, np.uint8)
    for q in lines:
        cov |= (q > 0).astype(np.uint8)
    left = (blob > 0) & (cov == 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), connectivity=8)
    extra = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < LEFTOVER_MIN_WIDTHS2 * w * w:
            continue
        comp = (lab == i)
        yy, xx = np.nonzero(comp)
        short_side = min(cv2.minAreaRect(np.column_stack([xx, yy]).astype(np.float32).reshape(-1, 1, 2))[1])
        if short_side < SLIVER_WIDTHS * w:
            ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            touch = [int((ring & (q > 0)).sum()) for q in lines]
            k = int(np.argmax(touch))
            if touch[k] > 0:
                lines[k][comp] = 255
                continue
        q = np.zeros(blob.shape, np.uint8)
        q[comp] = 255
        extra.append(q)
    return lines, extra


def _to_rectangle(piece):
    """Step 6: replace a line piece by the rectangle that fits it: straight axis, its length, and the
    MEDIAN cross-section width of the model's outline along that length plus RECT_MARGIN (median so
    junction bumps and flares do not widen the whole trail)."""
    pts, u, c = _axis(piece)
    nrm = np.array([-u[1], u[0]])
    a = (pts - c) @ u
    b = (pts - c) @ nrm
    lo, hi = float(a.min()), float(a.max())
    nb = max(3, int((hi - lo) / 15))
    edges = np.linspace(lo, hi, nb + 1)
    widths, mids = [], []
    for k in range(nb):
        sel = (a >= edges[k]) & (a <= edges[k + 1])
        if sel.sum() >= 3:
            widths.append(float(b[sel].max() - b[sel].min()) + 1.0)
            mids.append(float((b[sel].max() + b[sel].min()) / 2))
    if not widths:
        return piece
    wd = float(np.median(widths)) * (1.0 + RECT_MARGIN)
    mid = float(np.median(mids))
    corners = [c + u * al + nrm * (mid + sg * wd / 2) for al, sg in ((lo, -1), (hi, -1), (hi, 1), (lo, 1))]
    out = np.zeros(piece.shape, np.uint8)
    cv2.fillPoly(out, [np.round(np.array(corners)).astype(np.int32).reshape(-1, 1, 2)], 255)
    return out


# -- Public entry --------------------------------------------------------------

def peel_split(mask, frame_px=None):
    """Split a crossing blob into one rectangle per trail, or return it unchanged.

    mask: 2-D array, nonzero = blob (a crop around one detection is typical).
    frame_px: total pixels of the FULL frame, so the size floor matches the older splitter's;
    defaults to the mask's own size.

    Returns (pieces, info). pieces == [mask] when nothing is split. info["reason"] says why:
    "too_small", "no_lines" (fewer than two lines found), "not_a_crossing" (the guard kept it
    whole: gentle bend, near-straight fork, short ragged chunk) or "split"."""
    h, w_img = mask.shape[:2]
    area = int((mask > 0).sum())
    area_scale = (frame_px if frame_px else h * w_img) / _REF_FRAME_PX
    info = {"reason": "too_small", "lines": 0, "pieces": 1, "trail_width": 0.0, "arm_angle_apart": 0.0}
    if area < _SPLIT_AREA_MIN * area_scale:
        return [mask], info
    blob = (mask > 0).astype(np.uint8)
    w, sk = _trail_width(blob)
    info["trail_width"] = round(w, 1)
    if w < MIN_TRAIL_WIDTH_PX:
        info["reason"] = "no_lines"
        return [mask], info
    lines = _peel_lines(blob, w, sk)
    if len(lines) < 2:
        info["reason"] = "no_lines"
        return [mask], info
    lines = _merge_collinear(lines, w)
    info["lines"] = len(lines)
    if len(lines) < 2:
        info["reason"] = "no_lines"
        return [mask], info
    # the guard: judged on the LINE pieces only (leftover chunks point every which way)
    arm_pieces = [p for p in lines if (p > 0).sum() >= 5 and _long_side(p) >= MIN_ARM_WIDTHS * w]
    arms = [_axis_angle(p) for p in arm_pieces]
    apart = max((_apart(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]), default=0.0)
    info["arm_angle_apart"] = round(float(apart), 1)
    # below GUARD_DEG the older splitter's crossing evidence may still say crossing (a low-angle X), but only
    # when the lines run THROUGH each other: two lines meeting end to end at a shallow angle are a bend
    if apart < GUARD_DEG and not (has_crossing_evidence(mask, frame_px=frame_px)
                                  and _has_interior_crossing(arm_pieces, w)):
        info["reason"] = "not_a_crossing"
        return [mask], info
    lines, extra = _add_leftovers(blob, lines, w)
    pieces = [_to_rectangle(p) for p in lines] + extra
    info["reason"] = "split"
    info["pieces"] = len(pieces)
    return pieces, info
