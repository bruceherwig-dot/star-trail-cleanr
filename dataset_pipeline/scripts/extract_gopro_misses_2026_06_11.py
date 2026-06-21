"""GoPro blind-spot training extraction (2026-06-11, Bruce asleep, results for morning).

Pulls Bruce's REVIEWED polygons from CVAT tasks 54 + 55 and extracts the
training-flagged trails per the baseline rule (cvat_baseline_meta_2026-06-11.json):
  NEW   = shape id >= 68564 or source == "manual"  -> a trail STC was blind to
  MOVED = original-upload shape whose centroid sits >= 50 px from the closest
          as-uploaded outline in that frame's <stem>_polys.json -> STC misplaced it

For each flagged trail (nearby flagged trails on one frame merge into one spot):
go back to the SOURCE frame, cut a 1500px working crop for clean rotation
corners, and emit a 640 tile at tilts 0/15/30/45/60/75. Bruce's EXACT polygon
vertices are carried (rotated + shapely-clipped at the tile edge, never
rasterized/re-traced -- the standing rule). EVERY reviewed polygon visible in a
window is labeled, not just the flagged one, so no unlabeled trail becomes a
false negative.

Then creates ONE new CVAT review task with all tiles + labels, using the same
helpers that built the bridge/crossing review tasks.
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
OUT = os.path.join(ROOT, "bridge_fix_tiles_2026_06", "gopro_misses_aug_review")
os.makedirs(OUT, exist_ok=True)

SZ = 640; HALF = SZ // 2
TILTS = [0, 15, 30, 45, 60, 75]   # 0 included: the straight tile isn't in any task yet
FIRST_BRUCE_ID = 68564
MOVED_PX = 50.0
GROUP_RADIUS = 320.0              # flagged centroids closer than this share one tile
PAD = 400                          # mirror border for edge trails (covers the rotation wedge)
SLIDE_MAX = HALF - 100             # furthest a window may slide off trail-center
SLIDE_STEP = 32
TASK_NAME = "GoPro blind-spot misses - augmentation review"

# EDGE-TRAIL HANDLING + NO SILENT DROPS (added 2026-06-12 after task 57 silently
# shorted 37 edge trails to a single upright tile each — 210 of 1242 tiles never
# made, never reported; Bruce caught it by eye in CVAT review).
# Layer 1: if the centered window touches rotated-in black, SLIDE the window to a
#          nearby fully-valid spot that still holds the trail (real pixels only).
# Layer 2: if no clean spot exists, MIRROR-FILL: pad the crop with reflected real
#          sky before rotating, and carry every polygon's mirror reflections into
#          the labels so a reflected trail never trains as background.
# Anything STILL not produced is recorded by name + reason, printed in a loud
# end-of-run block, and written to _SKIPPED_MANIFEST.txt next to the tiles.
# Standing rule: a skipped work item is never a bare `continue`.

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


def reflections(arr, W, H):
    """The 9 mirror copies (identity + 4 edges + 4 corners) of a polygon under
    REFLECT_101 padding of a W x H crop, in crop coordinates. Off-canvas copies
    are harmless; the window clip drops them like any other polygon."""
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
    still holds the trail (>=60% of its points, centroid well inside). Integral
    image makes each candidate O(1). Returns (x, y) or None."""
    ii = cv2.integral(valid)

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


made = 0
dropped = []   # (name, reason) for EVERY work item not produced — never silent
n_new_total = n_moved_total = 0
for tid, (ds, (lo, hi)) in TASKS.items():
    safe_ds = safe(ds)
    meta = S.get(f"{URL}/api/tasks/{tid}/data/meta").json()
    stem_by_idx = {i: os.path.splitext(f["name"])[0] for i, f in enumerate(meta["frames"])}
    ann = S.get(f"{URL}/api/tasks/{tid}/annotations").json()

    # every reviewed polygon per frame stem (full truth for window labeling)
    all_by_stem = collections.defaultdict(list)
    for sh in ann["shapes"]:
        if sh.get("type") != "polygon":
            continue
        p = sh["points"]
        arr = np.array([[p[i], p[i + 1]] for i in range(0, len(p) - 1, 2)], np.float32)
        if len(arr) >= 3:
            all_by_stem[stem_by_idx[sh["frame"]]].append((sh, arr))

    # flagged shapes: NEW + MOVED
    flagged_by_stem = collections.defaultdict(list)
    n_new = n_moved = 0
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
                n_new += int(is_new); n_moved += int(is_moved)
    n_new_total += n_new; n_moved_total += n_moved
    print(f"task {tid}: flagged {n_new} new + {n_moved} moved on "
          f"{len(flagged_by_stem)} frames", flush=True)

    for stem, flagged in sorted(flagged_by_stem.items(),
                                key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        src = os.path.join(IMGROOT, ds, stem + ".jpg")
        if not os.path.exists(src):
            dropped.append((f"{safe_ds}__{stem}", "source frame not found on T7"))
            continue
        img = load(src)
        if img is None:
            dropped.append((f"{safe_ds}__{stem}", "source frame unreadable"))
            continue
        H, W = img.shape[:2]

        # union-find: flagged trails on this frame whose centroids sit within
        # GROUP_RADIUS share one tile (avoids near-duplicate windows)
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
            CW = min(1500, W); CH = min(1500, H)
            cx0 = int(np.clip(gcx - CW // 2, 0, W - CW))
            cy0 = int(np.clip(gcy - CH // 2, 0, H - CH))
            crop = img[cy0:cy0 + CH, cx0:cx0 + CW]
            flag_crop = [flagged[i] - np.array([cx0, cy0], np.float32) for i in gids]
            all_crop = [a - np.array([cx0, cy0], np.float32) for a in frame_polys]
            ones = np.ones((CH, CW), np.uint8)
            # mirror-padded twin for the layer-2 fallback, built once per spot
            pcrop = cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
            PH, PW2 = pcrop.shape[:2]
            pones = np.ones((PH, PW2), np.uint8)
            poff = np.array([PAD, PAD], np.float32)
            all_pad = [r + poff for a in all_crop for r in reflections(a, CW, CH)]
            flag_pad = [fc + poff for fc in flag_crop]

            for th in TILTS:
                nm = f"{safe_ds}__{stem}__{int(gcx)}_{int(gcy)}_t{th}"
                tile = None
                M = cv2.getRotationMatrix2D((CW / 2, CH / 2), th, 1.0)
                ri = cv2.warpAffine(crop, M, (CW, CH), flags=cv2.INTER_LINEAR) if th else crop
                valid = (cv2.warpAffine(ones, M, (CW, CH), flags=cv2.INTER_NEAREST)
                         if th else ones)
                R = M[:, :2]; tvec = M[:, 2]
                flag_rot = [fc @ R.T + tvec for fc in flag_crop] if th else flag_crop
                all_rot = [ac @ R.T + tvec for ac in all_crop] if th else all_crop
                fpts = np.concatenate(flag_rot, axis=0)
                fcx, fcy = float(fpts[:, 0].mean()), float(fpts[:, 1].mean())

                # layer 0: the classic centered window, valid as-is
                x = int(np.clip(fcx - HALF, 0, CW - SZ))
                y = int(np.clip(fcy - HALF, 0, CH - SZ))
                if valid[y:y + SZ, x:x + SZ].all():
                    shapes = window_shapes(all_rot, x, y)
                    if shapes:
                        tile = ri[y:y + SZ, x:x + SZ]
                # layer 1: slide to dodge the rotated-in black corner
                if tile is None:
                    pos = find_valid_slide(valid, fpts, fcx, fcy, CW, CH)
                    if pos is not None:
                        x, y = pos
                        shapes = window_shapes(all_rot, x, y)
                        if shapes:
                            tile = ri[y:y + SZ, x:x + SZ]
                # layer 2: mirror-filled rotation (reflected labels carried)
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
                    xp = int(np.clip(fpts_p[:, 0].mean() - HALF, 0, PW2 - SZ))
                    yp = int(np.clip(fpts_p[:, 1].mean() - HALF, 0, PH - SZ))
                    if validp[yp:yp + SZ, xp:xp + SZ].all():
                        shapes = window_shapes(all_rp, xp, yp)
                        if shapes:
                            tile = rip[yp:yp + SZ, xp:xp + SZ]

                if tile is None:
                    dropped.append((nm, "no valid window: centered, slid, and "
                                        "mirror-filled all failed"))
                    continue
                cv2.imwrite(os.path.join(OUT, nm + ".jpg"), tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                           "imagePath": nm + ".jpg", "imageData": None,
                           "imageHeight": SZ, "imageWidth": SZ},
                          open(os.path.join(OUT, nm + ".json"), "w"))
                made += 1
        if made and made % 300 < len(TILTS):
            print(f"  ...{made} tiles", flush=True)

print(f"\nEXTRACTION DONE: {made} tiles from {n_new_total} new + {n_moved_total} "
      f"moved trails. OUT: {OUT}", flush=True)
if dropped:
    print("\n" + "!" * 70, flush=True)
    print(f"WARNING: {len(dropped)} work item(s) were NOT produced. These are "
          "missing from the training set until fixed:", flush=True)
    for nm, why in dropped:
        print(f"  - {nm}: {why}", flush=True)
    manifest = os.path.join(OUT, "_SKIPPED_MANIFEST.txt")
    with open(manifest, "w") as mf:
        mf.write("\n".join(f"{nm}\t{why}" for nm, why in dropped) + "\n")
    print(f"Manifest written: {manifest}", flush=True)
    print("!" * 70, flush=True)
else:
    print("Nothing skipped: every intended tile was produced.", flush=True)

# ── Push to a new CVAT review task (same path that built the bridge/crossing tasks)
s2 = requests.Session()
s2.headers.update({"Authorization": "Token " + s2.post(
    URL + "/api/auth/login",
    json={"username": "bherwig2", "password": PW}).json()["key"]})
n_imgs = len(glob.glob(os.path.join(OUT, "*.jpg")))
body = {"name": TASK_NAME, "project_id": 1, "subset": "", "segment_size": n_imgs}
tid = s2.post(URL + "/api/tasks", json=body).json()["id"]
print("created task", tid, flush=True)
files = cct.gather_images(OUT)
cct.upload_images(s2, tid, files)
cct.wait_for_ready(s2, tid)
coco = l2c.convert_labelme_to_coco(OUT)
zp = os.path.join(tempfile.gettempdir(), f"gopro_misses_{tid}.zip")
l2c.create_coco_zip(coco, zp)
l2c.upload_to_cvat(zp, tid, URL, "bherwig2", PW)
# verify
job = S.get(f"{URL}/api/jobs?task_id={tid}").json()["results"][0]["id"]
ann2 = S.get(f"{URL}/api/jobs/{job}/annotations").json()
print(f"VERIFY: task {tid} has {n_imgs} tiles and {len(ann2.get('shapes', []))} polygons")
print(f">>> {URL}/tasks/{tid}")
print("GOPROMISSTASKID", tid)
