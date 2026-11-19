"""Augment the Jeff Fishman blind-bottom-trail polys (CVAT task 60) for the retrain.

Pulls Bruce's hand-drawn polygons from task 60 (full-frame coords). For each trail, rotates
the SOURCE (cleaned) frame through tilts 0/15/30/45/60/75 and cuts 640 tiles at a 3x3 grid of
off-center positions, carrying the EXACT polygon vertices by rotation + shapely edge-clip --
never rasterized/re-traced. The trails sit at the frame bottom, so the frame is mirror-padded
(BORDER_REFLECT_101) before rotation -> no black corners, no tiles silently dropped. Loud
reconcile at the end (expected vs made vs skipped). Saves tile JPG + LabelMe JSON.

Output: <ROOT>/jeff_fishman_blind_aug/   (then a second step pushes to a new CVAT task)
"""
import os, json, sys
sys.path.insert(0, "/Users/bruceherwig/Claude_Code_Projects")
import cv2, numpy as np, requests
import shapely.geometry as sgeom
from modules.io_safe import robust_imread

CVAT = "http://localhost:8080"; TASK = 60
CLEANED = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images/Jeff Fishman/cleaned"
OUT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR/jeff_fishman_blind_aug"
os.makedirs(OUT, exist_ok=True)
SZ = 640; HALF = SZ // 2
TILTS = [0, 15, 30, 45, 60, 75]
OFFS = [-160, 0, 160]            # 3x3 grid of tile-center offsets -> 9 positions
PAD = 900                        # mirror-pad margin so bottom trails never hit black

pw = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
s = requests.Session(); s.auth = ("bherwig2", pw)

# ---- pull task 60 polys (full-frame coords), per frame ----
meta = s.get(f"{CVAT}/api/tasks/{TASK}/data/meta", timeout=60).json()
stems = {i: os.path.splitext(f["name"])[0] for i, f in enumerate(meta["frames"])}
ann = s.get(f"{CVAT}/api/tasks/{TASK}/annotations", timeout=60).json()
polys_by_frame = {}
for sh in ann.get("shapes", []):
    if sh.get("type") != "polygon":
        continue
    p = sh["points"]
    arr = np.array([[p[i], p[i + 1]] for i in range(0, len(p) - 1, 2)], np.float32)
    if len(arr) >= 3:
        polys_by_frame.setdefault(sh["frame"], []).append(arr)
n_trails = sum(len(v) for v in polys_by_frame.values())
print(f"task {TASK}: {n_trails} polys on {len(polys_by_frame)} frames", flush=True)
expected = n_trails * len(TILTS) * len(OFFS) ** 2
print(f"expected tiles (before edge/clip drops): {expected}", flush=True)

made = 0; skipped_blank = 0; skipped_black = 0
for fi, frame_polys in polys_by_frame.items():
    stem = stems[fi]
    src = os.path.join(CLEANED, stem + ".jpg")
    img = robust_imread(src, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  WARN: could not read {src}", flush=True); continue
    # mirror-pad so a bottom trail's crop/rotation never falls on black
    img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    H, W = img.shape[:2]
    for ti, poly in enumerate(frame_polys):
        poly_p = poly + PAD                       # shift into padded coords
        cxf, cyf = poly_p[:, 0].mean(), poly_p[:, 1].mean()
        CW = CH = 1500
        cx0 = int(np.clip(cxf - CW / 2, 0, W - CW)); cy0 = int(np.clip(cyf - CH / 2, 0, H - CH))
        crop = img[cy0:cy0 + CH, cx0:cx0 + CW]
        poly_c = poly_p - np.array([cx0, cy0], np.float32)
        for th in TILTS:
            M = cv2.getRotationMatrix2D((CW / 2, CH / 2), th, 1.0)
            # reflect the rotated-out corners (no black) so every off-center tile is usable
            ri = cv2.warpAffine(crop, M, (CW, CH), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
            R = M[:, :2]; tvec = M[:, 2]
            pr = poly_c @ R.T + tvec
            pcx, pcy = pr[:, 0].mean(), pr[:, 1].mean()
            for ox in OFFS:
                for oy in OFFS:
                    x = int(np.clip(pcx + ox - HALF, 0, CW - SZ))
                    y = int(np.clip(pcy + oy - HALF, 0, CH - SZ))
                    tile = ri[y:y + SZ, x:x + SZ]
                    win = sgeom.box(x, y, x + SZ, y + SZ)
                    g = sgeom.Polygon(pr)
                    if not g.is_valid:
                        g = g.buffer(0)
                    inter = g.intersection(win)
                    if inter.is_empty or inter.area < 16:
                        skipped_blank += 1; continue
                    geoms = list(inter.geoms) if inter.geom_type == "MultiPolygon" else [inter]
                    shapes = []
                    for gg in geoms:
                        loc = [[round(px - x, 1), round(py - y, 1)] for px, py in gg.exterior.coords[:-1]]
                        if len(loc) >= 3:
                            shapes.append({"label": "trail", "points": loc, "group_id": None,
                                           "shape_type": "polygon", "flags": {}})
                    if not shapes:
                        skipped_blank += 1; continue
                    nm = f"jeff_fishman__{stem}__tr{ti}_t{th}_x{ox}_y{oy}"
                    cv2.imwrite(os.path.join(OUT, nm + ".jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                               "imagePath": nm + ".jpg", "imageData": None,
                               "imageHeight": SZ, "imageWidth": SZ},
                              open(os.path.join(OUT, nm + ".json"), "w"))
                    made += 1
    print(f"  {stem}: running total {made} tiles", flush=True)

print(f"\nRECONCILE: expected<= {expected} | made {made} | "
      f"skipped(trail clipped out) {skipped_blank} | skipped(edge-black) {skipped_black}", flush=True)
print(f"made + clipped + black = {made + skipped_blank + skipped_black} (should equal expected)", flush=True)
print("OUT:", OUT, flush=True)
