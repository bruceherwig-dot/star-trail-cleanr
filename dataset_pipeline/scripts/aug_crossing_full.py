"""Crossing augmentation = Bruce's review headstart for the rotated crossings.

Pulls the CURRENT reviewed polygons from CVAT task 46 (so any deletions Bruce made
in TileFixR are reflected). For each crossing tile, goes back to the SOURCE frame
(so rotation has real surrounding pixels, no black corners), rotates through tilts
15/30/45/60/75, and carries the EXACT polygons by vertex rotation + shapely
edge-clip -- each trail separate, never rasterized/re-traced. Saves tile JPG +
LabelMe JSON. A second step pushes to a new CVAT task.

Output: bridge_fix_tiles_2026_06/crossing_aug_review/
"""
import os, json, glob, csv, collections
import sys
sys.path.insert(0, "/Users/bruceherwig/Claude_Code_Projects")
import cv2, numpy as np, requests
import shapely.geometry as sgeom
from modules.io_safe import robust_imread

ROOT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR"
IMGROOT = os.path.join(ROOT, "star trail images")
OUT = os.path.join(ROOT, "bridge_fix_tiles_2026_06", "crossing_aug_review")
os.makedirs(OUT, exist_ok=True)
SZ = 640; HALF = SZ // 2
TILTS = [15, 30, 45, 60, 75]
EXTS = ('.jpg', '.jpeg', '.JPG', '.tif', '.tiff', '.TIF', '.png')
TASK = 46
ALIASES = {
    "Thomas Jackson - Borrego": "Thomas Jackson Star Trails Borrego",
    "Greg Meyer - Arizona": "Greg Meyer Arizona",
    "Bruce Herwig - Pioneertown Fisheye": "Pioneertown 6mm Fisheye Training",
    "Bruce Herwig - Borrego Springs 1": "borrego_springs_1",
    "My First Star Trail": "Bruce Herwig - first star trail data",
}

pw = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
s = requests.Session(); s.auth = ("bherwig2", pw)

# safe_ds -> real dataset name (forward-compute from the crossings CSV's datasets)
def safe(ds):
    return ds.replace(" ", "_").replace("/", "-")[:40]
real_by_safe = {}
for r in csv.DictReader(open("/Users/bruceherwig/Claude_Code_Projects/runs/cvat_crossings.csv")):
    if "gkyle" not in r["dataset"].lower():
        real_by_safe[safe(r["dataset"])] = r["dataset"]

# CURRENT task-46 polygons (post Bruce's review), tile-local, skip deleted frames
job = s.get(f"http://localhost:8080/api/jobs?task_id={TASK}").json()["results"][0]["id"]
meta = s.get(f"http://localhost:8080/api/jobs/{job}/data/meta").json()
deleted = set(meta.get("deleted_frames", []))
names = {i: f["name"] for i, f in enumerate(meta["frames"])}
ann = s.get(f"http://localhost:8080/api/jobs/{job}/annotations").json()
polys_by_frame = {}
for sh in ann.get("shapes", []):
    if sh.get("type") != "polygon" or sh["frame"] in deleted:
        continue
    p = sh["points"]
    arr = np.array([[p[i], p[i + 1]] for i in range(0, len(p) - 1, 2)], np.float32)
    if len(arr) >= 3:
        polys_by_frame.setdefault(sh["frame"], []).append(arr)
print(f"task {TASK} active tiles with polys: {len(polys_by_frame)}", flush=True)


SAFE_KEYS = sorted(real_by_safe.keys(), key=len, reverse=True)  # longest-first prefix match


def parse_name(tile_name):
    """name = <safe_ds>__<stem>__<cx>_<cy>.jpg ; stems may start with '_', so match
    the known safe_ds PREFIX first, then split the remainder once from the right."""
    base = tile_name[:-4]
    for key in SAFE_KEYS:
        if base.startswith(key + "__"):
            rest = base[len(key) + 2:]
            if "__" not in rest:
                return None
            stem, cc = rest.rsplit("__", 1)
            try:
                cx, cy = (int(x) for x in cc.split("_"))
            except Exception:
                return None
            return real_by_safe[key], stem, cx, cy
    return None


_folder_cache = {}
def resolve_folder(ds):
    if ds in _folder_cache:
        return _folder_cache[ds]
    found = None
    for c in (ALIASES.get(ds, ds), ds, ds.split(" - v")[0].rstrip()):
        p = os.path.join(IMGROOT, c)
        if os.path.isdir(p):
            found = p; break
    if found is None and os.path.isdir(IMGROOT):
        base = ds.split(" - v")[0].rstrip().lower()
        for ch in os.listdir(IMGROOT):
            if os.path.isdir(os.path.join(IMGROOT, ch)) and base in ch.lower():
                found = os.path.join(IMGROOT, ch); break
    _folder_cache[ds] = found
    return found


_file_index = {}
def find_src(ds, fr):
    folder = resolve_folder(ds)
    if folder is None:
        return None
    if folder not in _file_index:
        idx = {}
        for e in EXTS:
            for p in glob.glob(os.path.join(folder, f"*{e}")):
                idx[os.path.splitext(os.path.basename(p))[0]] = p
        _file_index[folder] = idx
    return _file_index[folder].get(fr)


frame_cache = collections.OrderedDict()
def load(path):
    if path in frame_cache:
        return frame_cache[path]
    img = robust_imread(path, cv2.IMREAD_COLOR)
    frame_cache[path] = img
    if len(frame_cache) > 4:
        frame_cache.popitem(last=False)
    return img


made = 0; skipped = 0
for fi, tile_polys in polys_by_frame.items():
    tn = names[fi]
    parsed = parse_name(tn)
    if parsed is None:
        skipped += 1; continue
    ds, stem, cx, cy = parsed
    safe_ds = safe(ds)
    src = find_src(ds, stem)
    if not src:
        skipped += 1; continue
    img = load(src)
    if img is None:
        skipped += 1; continue
    H, W = img.shape[:2]
    # the SAME origin cut_crossing_tiles.py used (clamped tile top-left in source)
    x0 = int(np.clip(cx - HALF, 0, max(0, W - SZ)))
    y0 = int(np.clip(cy - HALF, 0, max(0, H - SZ)))
    # larger crop for clean rotation corners, centered on the tile center
    cxf, cyf = x0 + HALF, y0 + HALF
    CW = min(1500, W); CH = min(1500, H)
    cx0 = int(np.clip(cxf - CW // 2, 0, W - CW))
    cy0 = int(np.clip(cyf - CH // 2, 0, H - CH))
    crop = img[cy0:cy0 + CH, cx0:cx0 + CW]
    # tile-local poly -> source (+x0,y0) -> crop-local (-cx0,-cy0)
    polys_crop = [(p + np.array([x0 - cx0, y0 - cy0])).astype(np.float32)
                  for p in tile_polys if len(p) >= 3]
    if not polys_crop:
        skipped += 1; continue
    ones = np.ones((CH, CW), np.uint8)
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
        if len(shapes) < 1:
            continue
        nm = f"{safe_ds}__{stem}__{cx}_{cy}_t{th}"
        cv2.imwrite(os.path.join(OUT, nm + ".jpg"), ri[y:y + SZ, x:x + SZ], [cv2.IMWRITE_JPEG_QUALITY, 92])
        json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes, "imagePath": nm + ".jpg",
                   "imageData": None, "imageHeight": SZ, "imageWidth": SZ},
                  open(os.path.join(OUT, nm + ".json"), "w"))
        made += 1
    if made % 200 < len(TILTS):
        print(f"  ...{made} variants", flush=True)

print(f"\nDONE: {made} crossing-aug tiles, {skipped} tiles skipped. OUT: {OUT}", flush=True)
