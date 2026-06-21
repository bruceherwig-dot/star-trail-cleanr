"""Handful of bridge augmentation samples for Bruce to check before the full batch.

Uses the SAME recipe as aug_all_bridge.py (rotate full source frame at tilts,
re-cut 640 windows at offsets, carry the trail mask as a polygon) but only on a
few trails, and renders a contact sheet with the polygon drawn so Bruce can see
the trail rotates correctly and the label follows. Excludes the 3 review-removed
tiles. Writes images + a contact sheet; commits nothing to training.
"""
import os, json, glob, math
import sys
sys.path.insert(0, "/Users/bruceherwig/Claude_Code_Projects")
import cv2, numpy as np
from modules.io_safe import robust_imread
from tools.masks_to_labelme import mask_to_shapes

ROOT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images"
OUT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/bridge_fix_tiles_2026_06/aug_handful"
os.makedirs(OUT, exist_ok=True)
SZ = 640; HALF = SZ // 2
TILTS = [0, 15, 30, 45, 60, 75]
OFFS = [(-200, -200), (0, -200), (200, -200), (-200, 0), (0, 0),
        (200, 0), (-200, 200), (0, 200), (200, 200)]
EXTS = ('.jpg', '.jpeg', '.JPG', '.tif', '.tiff', '.TIF', '.png')
N_TRAILS = 4          # how many distinct trails to sample
REMOVED = {("Bruce Herwig - Borrego - Gomphothere", "1M3A3785"),
           ("Bruce Herwig - Borrego - Gomphothere", "1M3A3842"),
           ("Bruce Herwig - Borrego - Gomphothere", "1M3A3915")}

# gather bridge trails (same as aug_all_bridge)
recs = {}
for lp in glob.glob(os.path.join(ROOT, "*", "cleanr_workspace", "run_log_*.jsonl")):
    ds = lp.split("/star trail images/")[1].split("/")[0]
    for line in open(lp, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "detect":
            continue
        fr = d.get("frame")
        for st in (d.get("detect_stages") or []):
            if isinstance(st, dict) and st.get("stage") == "seam_second_pass":
                for ev in (st.get("events", []) or []):
                    if ev.get("reason") == "bridge_gap_miss":
                        k = (ds, fr, ev.get("cx"), ev.get("cy"))
                        if k not in recs:
                            recs[k] = {"ds": ds, "frame": fr, "cx": ev["cx"], "cy": ev["cy"]}
recs = [r for r in recs.values() if (r["ds"], r["frame"]) not in REMOVED]


def find_src(ds, fr):
    for e in EXTS:
        p = os.path.join(ROOT, ds, fr + e)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(ROOT, ds, "*", fr + ".*"))
    return g[0] if g else None


def comp_at(mask, cx, cy):
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    H, W = mask.shape
    cid = lab[min(cy, H - 1), min(cx, W - 1)]
    if cid == 0:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        i = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        cid = lab[ys[i], xs[i]]
    return (lab == cid).astype(np.uint8) * 255


def variants_for(rec):
    ds, fr, cx, cy = rec["ds"], rec["frame"], rec["cx"], rec["cy"]
    src = find_src(ds, fr)
    if not src:
        return []
    img = robust_imread(src, cv2.IMREAD_COLOR)
    fm = cv2.imread(os.path.join(ROOT, ds, "cleanr_workspace", "masks", fr + ".png"), 0)
    if img is None or fm is None or fm.shape[:2] != img.shape[:2]:
        return []
    comp = comp_at(fm, cx, cy)
    if comp is None or comp.max() == 0:
        return []
    H, W = img.shape[:2]
    ys0, xs0 = np.where(comp > 0)
    ext = max(xs0.max() - xs0.min(), ys0.max() - ys0.min())
    is_long = ext > SZ - 60
    ones = np.ones((H, W), np.uint8)
    out = []
    for th in TILTS:
        if th:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), th, 1.0)
            ri = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
            rc = cv2.warpAffine(comp, M, (W, H), flags=cv2.INTER_NEAREST)
            valid = cv2.warpAffine(ones, M, (W, H), flags=cv2.INTER_NEAREST)
        else:
            ri, rc, valid = img, comp, ones
        ys, xs = np.where(rc > 0)
        if len(xs) == 0:
            continue
        ccx, ccy = int(xs.mean()), int(ys.mean())
        if is_long:
            pts = np.column_stack([xs, ys]).astype(np.float32); m = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - m); axis = vt[0]; t = (pts - m) @ axis
            lo, hi = np.quantile(t, 0.02), np.quantile(t, 0.98)
            nseg = int(np.clip((hi - lo) // 256, 1, 8))
            positions = [tuple(map(int, m + axis * (lo + (hi - lo) * i / max(nseg, 1)))) for i in range(nseg + 1)]
        else:
            positions = [(ccx - dx, ccy - dy) for (dx, dy) in OFFS]
        for (tcx, tcy) in positions:
            x = int(tcx - HALF); y = int(tcy - HALF)
            if x < 0 or y < 0 or x + SZ > W or y + SZ > H:
                continue
            if not valid[y:y + SZ, x:x + SZ].all():
                continue
            tm = rc[y:y + SZ, x:x + SZ]
            if tm.max() == 0:
                continue
            shapes = mask_to_shapes(tm)
            if not shapes:
                continue
            ti = ri[y:y + SZ, x:x + SZ].copy()
            out.append((ti, shapes, f"{fr} tilt{th}"))
    return out, is_long


# pick a handful: try to get a mix until we have N_TRAILS that produced variants
picked = []
for rec in recs:
    res = variants_for(rec)
    if not res:
        continue
    vs, is_long = res
    if not vs:
        continue
    picked.append((rec, vs, is_long))
    print(f"  trail {rec['ds']}/{rec['frame']}: {len(vs)} variants ({'LONG' if is_long else 'short'})", flush=True)
    if len(picked) >= N_TRAILS:
        break

# save full variants + build contact sheet (sample up to 8 variants per trail)
CELL = 300; COLS = 8; PAD = 4; LBLH = 18
cells = []
for rec, vs, is_long in picked:
    step = max(1, len(vs) // 8)
    sample = vs[::step][:8]
    for ti, shapes, lbl in sample:
        disp = ti.copy()
        for s in shapes:
            cv2.polylines(disp, [np.array(s["points"], np.int32)], True, (0, 255, 0), 2)
        cells.append((disp, lbl))
    # also write the FULL set to disk for inspection
    od = os.path.join(OUT, f"{rec['ds']}__{rec['frame']}")
    os.makedirs(od, exist_ok=True)
    for i, (ti, shapes, lbl) in enumerate(vs):
        cv2.imwrite(os.path.join(od, f"v{i:02d}.jpg"), ti, [cv2.IMWRITE_JPEG_QUALITY, 90])

rows = math.ceil(len(cells) / COLS)
Wc = COLS * CELL + (COLS + 1) * PAD
Hc = rows * (CELL + LBLH) + (rows + 1) * PAD + 34
canvas = np.full((Hc, Wc, 3), 28, np.uint8)
cv2.putText(canvas, f"BRIDGE AUGMENTATION HANDFUL - {len(picked)} trails, green=carried label",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
for i, (im, lbl) in enumerate(cells):
    im = cv2.resize(im, (CELL, CELL))
    rr, cc = divmod(i, COLS)
    x = PAD + cc * (CELL + PAD); y = 34 + PAD + rr * (CELL + LBLH + PAD)
    canvas[y:y + CELL, x:x + CELL] = im
    cv2.putText(canvas, lbl[:34], (x + 2, y + CELL + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
sheet = os.path.join(OUT, "HANDFUL_contact_sheet.jpg")
cv2.imwrite(sheet, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("\nwrote", sheet)
print("full variants per trail in subfolders under", OUT)
