"""Handful of bridge augmentation samples driven by Bruce's REVIEWED CVAT task-42
polygons (not the old auto-detection masks). For a few reviewed tiles: map the
tile-local polygon back into full-frame coords (origin = cx-320, cy-320, verified
exact), rotate the full source frame + polygon at tilts, re-cut 640 windows at
offsets, and render a contact sheet with the carried polygon drawn. Excludes the
3 deleted frames automatically (they're CVAT deleted_frames). Commits nothing.
"""
import os, json, glob, math, csv
import sys
sys.path.insert(0, "/Users/bruceherwig/Claude_Code_Projects")
import cv2, numpy as np, requests
import shapely.geometry as sgeom
from modules.io_safe import robust_imread
from tools.masks_to_labelme import mask_to_shapes

ROOT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR"
IMGROOT = os.path.join(ROOT, "star trail images")
OUT = os.path.join(ROOT, "bridge_fix_tiles_2026_06", "aug_handful")
os.makedirs(OUT, exist_ok=True)
SZ = 640; HALF = SZ // 2
TILTS = [0, 15, 30, 45, 60, 75]
OFFS = [(-200, -200), (0, -200), (200, -200), (-200, 0), (0, 0),
        (200, 0), (-200, 200), (0, 200), (200, 200)]
EXTS = ('.jpg', '.jpeg', '.JPG', '.tif', '.tiff', '.TIF', '.png')
N_TRAILS = 4
JOB = 34

pw = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
s = requests.Session(); s.auth = ("bherwig2", pw)

# manifest: tile_image -> dataset, frame, cx, cy
man = {r["tile_image"]: r for r in csv.DictReader(
    open(os.path.join(ROOT, "bridge_fix_tiles_2026_06", "manifest.csv")))}

# CVAT: active frames + their reviewed polygons (tile-local coords)
meta = s.get(f"http://localhost:8080/api/jobs/{JOB}/data/meta").json()
deleted = set(meta.get("deleted_frames", []))
names = {i: f["name"] for i, f in enumerate(meta["frames"])}
ann = s.get(f"http://localhost:8080/api/jobs/{JOB}/annotations").json()
polys_by_frame = {}
for sh in ann.get("shapes", []):
    if sh.get("type") != "polygon":
        continue
    fi = sh["frame"]
    if fi in deleted:
        continue
    pts = sh["points"]
    arr = np.array([[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)], np.float32)
    polys_by_frame.setdefault(fi, []).append(arr)
print(f"active tiles with polys: {len(polys_by_frame)}", flush=True)


def find_src(ds, fr):
    for e in EXTS:
        p = os.path.join(IMGROOT, ds, fr + e)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(IMGROOT, ds, "*", fr + ".*"))
    return g[0] if g else None


def build_variants(ds, fr, ox, oy, tile_polys):
    src = find_src(ds, fr)
    if not src:
        return None
    img = robust_imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    # Work on a crop around the trail (tile center = cx,cy) instead of the whole
    # frame -- same local result, ~15x faster than rotating a 6000x4000 image.
    cxf, cyf = ox + HALF, oy + HALF
    CW = min(1500, W); CH = min(1500, H)
    cx0 = int(np.clip(cxf - CW // 2, 0, W - CW))
    cy0 = int(np.clip(cyf - CH // 2, 0, H - CH))
    crop = img[cy0:cy0 + CH, cx0:cx0 + CW]
    # Carry YOUR exact CVAT polygons (vertices) -- never rasterize + re-trace.
    polys_crop = [(p + np.array([ox - cx0, oy - cy0])).astype(np.float32)
                  for p in tile_polys if len(p) >= 3]
    if not polys_crop:
        return None
    allp = np.concatenate(polys_crop, axis=0)
    ext = max(allp[:, 0].ptp(), allp[:, 1].ptp())
    is_long = ext > SZ - 60
    ones = np.ones((CH, CW), np.uint8)
    out = []
    n_win = 0
    for th in TILTS:
        if th:
            M = cv2.getRotationMatrix2D((CW / 2, CH / 2), th, 1.0)
            ri = cv2.warpAffine(crop, M, (CW, CH), flags=cv2.INTER_LINEAR)
            valid = cv2.warpAffine(ones, M, (CW, CH), flags=cv2.INTER_NEAREST)
            R = M[:, :2]; tvec = M[:, 2]
            polys_rot = [pc @ R.T + tvec for pc in polys_crop]
        else:
            ri, valid = crop, ones
            polys_rot = polys_crop
        # center a 640 window on the rotated polygons (same framing as the tile)
        cpts = np.concatenate(polys_rot, axis=0)
        x = int(np.clip(cpts[:, 0].mean() - HALF, 0, CW - SZ))
        y = int(np.clip(cpts[:, 1].mean() - HALF, 0, CH - SZ))
        if not valid[y:y + SZ, x:x + SZ].all():
            continue
        win = sgeom.box(x, y, x + SZ, y + SZ)
        shapes = []
        for pr in polys_rot:
            poly = sgeom.Polygon(pr)
            if not poly.is_valid:
                poly = poly.buffer(0)
            inter = poly.intersection(win)        # only edge-clip to the tile; shape unchanged inside
            if inter.is_empty or inter.area < 4:
                continue
            geoms = list(inter.geoms) if inter.geom_type == "MultiPolygon" else [inter]
            for g in geoms:
                loc = [[round(px - x, 1), round(py - y, 1)] for px, py in g.exterior.coords[:-1]]
                if len(loc) >= 3:
                    shapes.append({"label": "trail", "points": loc, "shape_type": "polygon",
                                   "group_id": None, "flags": {}})
        if not shapes:
            continue
        out.append((ri[y:y + SZ, x:x + SZ].copy(), shapes, f"{fr} t{th}"))
        n_win += 1
    print(f"      build: polys={len(polys_crop)} is_long={is_long} windows={n_win}", flush=True)
    return out, is_long


picked = []
for fi, tile_polys in polys_by_frame.items():
    tn = names[fi]
    r = man.get(tn)
    if not r:
        continue
    ds, fr = r["dataset"], r["frame"]
    ox, oy = int(r["cx"]) - HALF, int(r["cy"]) - HALF
    print(f"  trying {tn} ({ds}/{fr})", flush=True)
    res = build_variants(ds, fr, ox, oy, tile_polys)
    if not res or not res[0]:
        continue
    vs, is_long = res
    picked.append((tn, ds, fr, vs, is_long))
    print(f"  {tn}: {len(vs)} variants ({'LONG' if is_long else 'short'}), {len(tile_polys)} poly(s)", flush=True)
    if len(picked) >= N_TRAILS:
        break

CELL = 300; COLS = 8; PAD = 4; LBLH = 18
cells = []
for tn, ds, fr, vs, is_long in picked:
    step = max(1, len(vs) // 8)
    for ti, shapes, lbl in vs[::step][:8]:
        disp = ti.copy()
        for sh in shapes:
            cv2.polylines(disp, [np.array(sh["points"], np.int32)], True, (0, 255, 0), 2)
        cells.append((disp, lbl))

rows = math.ceil(len(cells) / COLS)
Wc = COLS * CELL + (COLS + 1) * PAD
Hc = rows * (CELL + LBLH) + (rows + 1) * PAD + 34
canvas = np.full((Hc, Wc, 3), 28, np.uint8)
cv2.putText(canvas, f"BRIDGE AUG HANDFUL (from your CVAT-42 polys) - {len(picked)} tiles, green=carried label",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
for i, (im, lbl) in enumerate(cells):
    im = cv2.resize(im, (CELL, CELL))
    rr, cc = divmod(i, COLS)
    x = PAD + cc * (CELL + PAD); y = 34 + PAD + rr * (CELL + LBLH + PAD)
    canvas[y:y + CELL, x:x + CELL] = im
    cv2.putText(canvas, lbl[:34], (x + 2, y + CELL + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
sheet = os.path.join(OUT, "HANDFUL_from_cvat.jpg")
cv2.imwrite(sheet, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("\nwrote", sheet, flush=True)
