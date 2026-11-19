"""Full bridge augmentation = Bruce's review headstart.

Drives off the REVIEWED CVAT task-42 polygons (job 34). For every active bridge
tile, rotates the tile through tilts 15/30/45/60/75 (skips 0 = the unchanged
original) and carries YOUR EXACT CVAT polygons rotated with it -- each trail kept
separate, vertices transformed, never rasterized/re-traced. Saves each variant as
a 640 tile JPG + a LabelMe JSON so it can be pushed to CVAT for review.

Output: bridge_fix_tiles_2026_06/aug_bridge_review/  (flat: <tile>_t<deg>.jpg/.json)
"""
import os, json, glob, csv, collections
import sys
sys.path.insert(0, "/Users/bruceherwig/Claude_Code_Projects")
import cv2, numpy as np, requests
import shapely.geometry as sgeom
from modules.io_safe import robust_imread

ROOT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR"
IMGROOT = os.path.join(ROOT, "star trail images")
OUT = os.path.join(ROOT, "bridge_fix_tiles_2026_06", "aug_bridge_review")
os.makedirs(OUT, exist_ok=True)
SZ = 640; HALF = SZ // 2
TILTS = [15, 30, 45, 60, 75]          # 0 skipped: that's the untouched original
EXTS = ('.jpg', '.jpeg', '.JPG', '.tif', '.tiff', '.TIF', '.png')
JOB = 34

pw = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
s = requests.Session(); s.auth = ("bherwig2", pw)
man = {r["tile_image"]: r for r in csv.DictReader(
    open(os.path.join(ROOT, "bridge_fix_tiles_2026_06", "manifest.csv")))}
meta = s.get(f"http://localhost:8080/api/jobs/{JOB}/data/meta").json()
deleted = set(meta.get("deleted_frames", []))
names = {i: f["name"] for i, f in enumerate(meta["frames"])}
ann = s.get(f"http://localhost:8080/api/jobs/{JOB}/annotations").json()
polys_by_frame = {}
for sh in ann.get("shapes", []):
    if sh.get("type") != "polygon" or sh["frame"] in deleted:
        continue
    p = sh["points"]
    arr = np.array([[p[i], p[i + 1]] for i in range(0, len(p) - 1, 2)], np.float32)
    polys_by_frame.setdefault(sh["frame"], []).append(arr)
print(f"active tiles with polys: {len(polys_by_frame)}", flush=True)


def find_src(ds, fr):
    for e in EXTS:
        p = os.path.join(IMGROOT, ds, fr + e)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(IMGROOT, ds, "*", fr + ".*"))
    return g[0] if g else None


frame_cache = collections.OrderedDict()


def load(path):
    if path in frame_cache:
        return frame_cache[path]
    img = robust_imread(path, cv2.IMREAD_COLOR)
    frame_cache[path] = img
    if len(frame_cache) > 4:
        frame_cache.popitem(last=False)
    return img


def save_labelme(name, shapes):
    json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
               "imagePath": name + ".jpg", "imageData": None,
               "imageHeight": SZ, "imageWidth": SZ},
              open(os.path.join(OUT, name + ".json"), "w"))


made = 0
skipped = 0
for fi, tile_polys in polys_by_frame.items():
    tn = names[fi]
    r = man.get(tn)
    if not r:
        skipped += 1
        continue
    ds, fr = r["dataset"], r["frame"]
    ox, oy = int(r["cx"]) - HALF, int(r["cy"]) - HALF
    src = find_src(ds, fr)
    if not src:
        skipped += 1
        continue
    img = load(src)
    if img is None:
        skipped += 1
        continue
    H, W = img.shape[:2]
    cxf, cyf = ox + HALF, oy + HALF
    CW = min(1500, W); CH = min(1500, H)
    cx0 = int(np.clip(cxf - CW // 2, 0, W - CW))
    cy0 = int(np.clip(cyf - CH // 2, 0, H - CH))
    crop = img[cy0:cy0 + CH, cx0:cx0 + CW]
    polys_crop = [(p + np.array([ox - cx0, oy - cy0])).astype(np.float32)
                  for p in tile_polys if len(p) >= 3]
    if not polys_crop:
        skipped += 1
        continue
    ones = np.ones((CH, CW), np.uint8)
    stem = os.path.splitext(tn)[0]
    for th in TILTS:
        M = cv2.getRotationMatrix2D((CW / 2, CH / 2), th, 1.0)
        ri = cv2.warpAffine(crop, M, (CW, CH), flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(ones, M, (CW, CH), flags=cv2.INTER_NEAREST)
        R = M[:, :2]; tvec = M[:, 2]
        polys_rot = [pc @ R.T + tvec for pc in polys_crop]
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
            inter = poly.intersection(win)
            if inter.is_empty or inter.area < 4:
                continue
            geoms = list(inter.geoms) if inter.geom_type == "MultiPolygon" else [inter]
            for g in geoms:
                loc = [[round(px - x, 1), round(py - y, 1)] for px, py in g.exterior.coords[:-1]]
                if len(loc) >= 3:
                    shapes.append({"label": "trail", "points": loc, "group_id": None,
                                   "shape_type": "polygon", "flags": {}})
        if not shapes:
            continue
        name = f"{stem}_t{th}"
        cv2.imwrite(os.path.join(OUT, name + ".jpg"),
                    ri[y:y + SZ, x:x + SZ], [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_labelme(name, shapes)
        made += 1
    if made % 100 < len(TILTS):
        print(f"  ...{made} variants written", flush=True)

print(f"\nDONE: {made} augmented tiles, {skipped} tiles skipped. OUT: {OUT}", flush=True)
