"""Edge-trail rotation recovery (2026-06-12).

Task 57 ("GoPro blind-spot misses - augmentation review") shorted 45 of its 207
source trails: 37 got ONLY the upright t0 tile and 8 got a partial rotation set.
Those are exactly the trails that sit near a frame edge -- the extractor refused
any rotated tile whose 640px window would contain rotated-in black (off-frame)
pixels. Edge trails are a known model blind spot, so they need MORE variations,
not fewer.

This script re-derives the same flagged-trail spots from CVAT tasks 54 + 55
(identical rules and grouping as extract_gopro_misses_2026_06_11.py, so spot
names match), asks task 57 which (spot, tilt) tiles already exist, and recovers
ONLY the missing ones with a two-layer strategy:

  1. SLIDE: search nearby window positions that still hold the trail but dodge
     the black corner entirely. Real pixels only, no invention.
  2. MIRROR-FILL fallback: pad the working crop with mirrored real sky before
     rotating, so there is no black to dodge. Every reviewed polygon's mirror
     reflections are carried into the labels too -- a reflected trail in the
     padding must be labeled or it would train as a false negative.

Bruce's EXACT polygon vertices are carried (rotated + shapely-clipped, never
rasterized/re-traced -- the standing rule). Output goes to a NEW folder and a
NEW CVAT review task; task 57 (mid-review) is not touched.
"""
import os, sys, json, glob, collections, tempfile
sys.path.insert(0, "/Users/bruceherwig/Claude_Code_Projects")
import cv2, numpy as np, requests
import shapely.geometry as sgeom
from modules.io_safe import robust_imread
import tools.cvat_create_task as cct
import tools.labelme_to_cvat as l2c

ROOT = "/Volumes/T7 Shield/AI Projects/Star Trail CleanR"
IMGROOT = os.path.join(ROOT, "star trail images")
OUT = os.path.join(ROOT, "bridge_fix_tiles_2026_06", "gopro_edge_rotation_recovery")
os.makedirs(OUT, exist_ok=True)

SZ = 640; HALF = SZ // 2
TILTS = [0, 15, 30, 45, 60, 75]
FIRST_BRUCE_ID = 68564
MOVED_PX = 50.0
GROUP_RADIUS = 320.0
PAD = 400                      # mirror border; covers a 1500-crop's rotation wedge
SLIDE_MAX = HALF - 100         # furthest the window may slide off trail-center
SLIDE_STEP = 32
DONE_TASK = 57                 # the task whose existing tiles define "missing"
TASK_NAME = "GoPro edge-trail rotation recovery - review"

TASKS = {
    54: ("Thomas Jackson GoPro_G0088569", (65400, 66168)),
    55: ("Thomas Jackson GoPro_G0037688", (66169, 66979)),
}

URL = "http://localhost:8080"
PW = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
S = requests.Session(); S.auth = ("bherwig2", PW)


def centroid(arr):
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def safe(ds):
    return ds.replace(" ", "_").replace("/", "-")[:40]


frame_cache = collections.OrderedDict()
def load(path):
    if path in frame_cache:
        return frame_cache[path]
    img = robust_imread(path, cv2.IMREAD_COLOR)
    frame_cache[path] = img
    if len(frame_cache) > 4:
        frame_cache.popitem(last=False)
    return img


# ── what already exists in task 57 (authoritative: its frame names) ────────────
meta57 = S.get(f"{URL}/api/tasks/{DONE_TASK}/data/meta").json()
have = {os.path.splitext(f["name"])[0] for f in meta57["frames"]}
print(f"task {DONE_TASK} already holds {len(have)} tiles", flush=True)


def reflections(arr, W, H):
    """The 9 mirror copies (identity + 4 edges + 4 corners) of a polygon under
    REFLECT_101 padding of a W x H crop. Returned in crop coordinates; copies
    that fall outside the padded canvas are clipped later like any polygon."""
    out = []
    for fx in (0, 1, 2):                       # 0 identity, 1 left, 2 right
        for fy in (0, 1, 2):                   # 0 identity, 1 top, 2 bottom
            a = arr.copy()
            if fx == 1: a[:, 0] = -a[:, 0]
            if fx == 2: a[:, 0] = 2 * (W - 1) - a[:, 0]
            if fy == 1: a[:, 1] = -a[:, 1]
            if fy == 2: a[:, 1] = 2 * (H - 1) - a[:, 1]
            out.append(a)
    return out


def window_shapes(all_rot, x, y):
    """Clip every rotated polygon to the 640 window at (x, y); labelme shapes."""
    win = sgeom.box(x, y, x + SZ, y + SZ)
    shapes = []
    for pr in all_rot:
        poly = sgeom.Polygon(pr)
        if not poly.is_valid:
            poly = poly.buffer(0)
        inter = poly.intersection(win)
        if inter.is_empty or inter.area < 4:
            continue
        geoms = (list(inter.geoms) if inter.geom_type == "MultiPolygon" else [inter])
        for g in geoms:
            if g.geom_type != "Polygon":
                continue
            loc = [[round(px - x, 1), round(py - y, 1)]
                   for px, py in g.exterior.coords[:-1]]
            if len(loc) >= 3:
                shapes.append({"label": "trail", "points": loc, "group_id": None,
                               "shape_type": "polygon", "flags": {}})
    return shapes


def find_valid_slide(valid, fpts, cx, cy, W, H):
    """Find the fully-valid 640 window nearest the ideal center (cx, cy) that
    still holds the trail. Uses an integral image so each candidate is O(1).
    Returns (x, y) or None."""
    ii = cv2.integral(valid)            # valid is uint8 ones/zeros

    def all_valid(x, y):
        s = ii[y + SZ, x + SZ] - ii[y, x + SZ] - ii[y + SZ, x] + ii[y, x]
        return s == SZ * SZ

    ideal_x = int(np.clip(cx - HALF, 0, W - SZ))
    ideal_y = int(np.clip(cy - HALF, 0, H - SZ))
    cands = []
    for dx in range(-SLIDE_MAX, SLIDE_MAX + 1, SLIDE_STEP):
        for dy in range(-SLIDE_MAX, SLIDE_MAX + 1, SLIDE_STEP):
            x = int(np.clip(ideal_x + dx, 0, W - SZ))
            y = int(np.clip(ideal_y + dy, 0, H - SZ))
            cands.append((dx * dx + dy * dy, x, y))
    seen = set()
    for _, x, y in sorted(cands):
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if not all_valid(x, y):
            continue
        inside = ((fpts[:, 0] >= x) & (fpts[:, 0] < x + SZ)
                  & (fpts[:, 1] >= y) & (fpts[:, 1] < y + SZ)).mean()
        cx_ok = x + 64 <= cx <= x + SZ - 64 and y + 64 <= cy <= y + SZ - 64
        if inside >= 0.6 and cx_ok:
            return x, y
    return None


made_slide = made_mirror = still_missing = 0
spots_touched = set()
for tid, (ds, (lo, hi)) in TASKS.items():
    safe_ds = safe(ds)
    meta = S.get(f"{URL}/api/tasks/{tid}/data/meta").json()
    stem_by_idx = {i: os.path.splitext(f["name"])[0] for i, f in enumerate(meta["frames"])}
    ann = S.get(f"{URL}/api/tasks/{tid}/annotations").json()

    all_by_stem = collections.defaultdict(list)
    for sh in ann["shapes"]:
        if sh.get("type") != "polygon":
            continue
        p = sh["points"]
        arr = np.array([[p[i], p[i + 1]] for i in range(0, len(p) - 1, 2)], np.float32)
        if len(arr) >= 3:
            all_by_stem[stem_by_idx[sh["frame"]]].append((sh, arr))

    flagged_by_stem = collections.defaultdict(list)
    for stem, pairs in all_by_stem.items():
        pj = os.path.join(IMGROOT, ds, "cleanr_workspace", "masks", f"{stem}_polys.json")
        uploaded = []
        if os.path.exists(pj):
            for p in json.load(open(pj))["polygons"]:
                uploaded.append(np.array(p["corners"], np.float32))
        for sh, arr in pairs:
            is_new = sh["id"] >= FIRST_BRUCE_ID or sh.get("source") == "manual"
            is_moved = False
            if not is_new and lo <= sh["id"] <= hi and uploaded:
                cx, cy = centroid(arr)
                d = min(((cx - centroid(u)[0]) ** 2 + (cy - centroid(u)[1]) ** 2) ** 0.5
                        for u in uploaded)
                is_moved = d >= MOVED_PX
            if is_new or is_moved:
                flagged_by_stem[stem].append(arr)

    for stem, flagged in sorted(flagged_by_stem.items(),
                                key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        src = os.path.join(IMGROOT, ds, stem + ".jpg")
        if not os.path.exists(src):
            continue

        cents = [centroid(a) for a in flagged]
        parent = list(range(len(flagged)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i in range(len(flagged)):
            for j in range(i + 1, len(flagged)):
                if ((cents[i][0] - cents[j][0]) ** 2
                        + (cents[i][1] - cents[j][1]) ** 2) ** 0.5 <= GROUP_RADIUS:
                    parent[find(i)] = find(j)
        groups = collections.defaultdict(list)
        for i in range(len(flagged)):
            groups[find(i)].append(i)

        frame_polys = [arr for _, arr in all_by_stem[stem]]
        for gids in groups.values():
            gpts = np.concatenate([flagged[i] for i in gids], axis=0)
            gcx, gcy = float(gpts[:, 0].mean()), float(gpts[:, 1].mean())
            base = f"{safe_ds}__{stem}__{int(gcx)}_{int(gcy)}"
            missing = [th for th in TILTS if f"{base}_t{th}" not in have]
            if not missing:
                continue

            img = load(src)
            if img is None:
                continue
            H, W = img.shape[:2]
            CW = min(1500, W); CH = min(1500, H)
            cx0 = int(np.clip(gcx - CW // 2, 0, W - CW))
            cy0 = int(np.clip(gcy - CH // 2, 0, H - CH))
            crop = img[cy0:cy0 + CH, cx0:cx0 + CW]
            off = np.array([cx0, cy0], np.float32)
            flag_crop = [flagged[i] - off for i in gids]
            all_crop = [a - off for a in frame_polys]
            ones = np.ones((CH, CW), np.uint8)

            # mirror-padded twin, built once per spot
            pcrop = cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
            PH, PW2 = pcrop.shape[:2]
            pones = np.ones((PH, PW2), np.uint8)
            poff = np.array([PAD, PAD], np.float32)
            # labels in padded coords: all 9 mirror copies of every polygon
            all_pad = [r + poff for a in all_crop for r in reflections(a, CW, CH)]
            flag_pad = [fc + poff for fc in flag_crop]

            for th in missing:
                tile = None; how = None
                # layer 1: slide within the plain rotated crop
                M = cv2.getRotationMatrix2D((CW / 2, CH / 2), th, 1.0)
                ri = cv2.warpAffine(crop, M, (CW, CH), flags=cv2.INTER_LINEAR) if th else crop
                valid = (cv2.warpAffine(ones, M, (CW, CH), flags=cv2.INTER_NEAREST)
                         if th else ones)
                R = M[:, :2]; tvec = M[:, 2]
                flag_rot = [fc @ R.T + tvec for fc in flag_crop] if th else flag_crop
                all_rot = [ac @ R.T + tvec for ac in all_crop] if th else all_crop
                fpts = np.concatenate(flag_rot, axis=0)
                fcx, fcy = float(fpts[:, 0].mean()), float(fpts[:, 1].mean())
                pos = find_valid_slide(valid, fpts, fcx, fcy, CW, CH)
                if pos is not None:
                    x, y = pos
                    shapes = window_shapes(all_rot, x, y)
                    if shapes:
                        tile = ri[y:y + SZ, x:x + SZ]; how = "slide"

                # layer 2: mirror-filled rotation
                if tile is None:
                    Mp = cv2.getRotationMatrix2D((PW2 / 2, PH / 2), th, 1.0)
                    rip = (cv2.warpAffine(pcrop, Mp, (PW2, PH), flags=cv2.INTER_LINEAR)
                           if th else pcrop)
                    validp = (cv2.warpAffine(pones, Mp, (PW2, PH), flags=cv2.INTER_NEAREST)
                              if th else pones)
                    Rp = Mp[:, :2]; tvecp = Mp[:, 2]
                    flag_rp = [fp @ Rp.T + tvecp for fp in flag_pad] if th else flag_pad
                    all_rp = [ap @ Rp.T + tvecp for ap in all_pad] if th else all_pad
                    fpts_p = np.concatenate(flag_rp, axis=0)
                    fcxp, fcyp = float(fpts_p[:, 0].mean()), float(fpts_p[:, 1].mean())
                    x = int(np.clip(fcxp - HALF, 0, PW2 - SZ))
                    y = int(np.clip(fcyp - HALF, 0, PH - SZ))
                    if validp[y:y + SZ, x:x + SZ].all():
                        shapes = window_shapes(all_rp, x, y)
                        if shapes:
                            tile = rip[y:y + SZ, x:x + SZ]; how = "mirror"

                if tile is None:
                    still_missing += 1
                    continue
                nm = f"{base}_t{th}"
                cv2.imwrite(os.path.join(OUT, nm + ".jpg"), tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                           "imagePath": nm + ".jpg", "imageData": None,
                           "imageHeight": SZ, "imageWidth": SZ},
                          open(os.path.join(OUT, nm + ".json"), "w"))
                spots_touched.add(base)
                if how == "slide":
                    made_slide += 1
                else:
                    made_mirror += 1

print(f"\nRECOVERY DONE: {made_slide} by slide + {made_mirror} by mirror-fill "
      f"across {len(spots_touched)} trails; {still_missing} unrecoverable.",
      flush=True)

n_imgs = len(glob.glob(os.path.join(OUT, "*.jpg")))
if n_imgs == 0:
    print("nothing recovered; no task created")
    sys.exit(0)

# ── push to ONE new CVAT review task (same path that built task 57) ────────────
s2 = requests.Session()
s2.headers.update({"Authorization": "Token " + s2.post(
    URL + "/api/auth/login",
    json={"username": "bherwig2", "password": PW}).json()["key"]})
body = {"name": TASK_NAME, "project_id": 1, "subset": "", "segment_size": n_imgs}
tid = s2.post(URL + "/api/tasks", json=body).json()["id"]
print("created task", tid, flush=True)
files = cct.gather_images(OUT)
cct.upload_images(s2, tid, files)
cct.wait_for_ready(s2, tid)
coco = l2c.convert_labelme_to_coco(OUT)
zp = os.path.join(tempfile.gettempdir(), f"edge_recovery_{tid}.zip")
l2c.create_coco_zip(coco, zp)
l2c.upload_to_cvat(zp, tid, URL, "bherwig2", PW)
job = S.get(f"{URL}/api/jobs?task_id={tid}").json()["results"][0]["id"]
ann2 = S.get(f"{URL}/api/jobs/{job}/annotations").json()
print(f"VERIFY: task {tid} has {n_imgs} tiles and {len(ann2.get('shapes', []))} polygons")
print(f">>> {URL}/tasks/{tid}")
print("RECOVERYTASKID", tid)
