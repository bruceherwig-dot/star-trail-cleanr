"""Count MERGED trails that are long as measured inside a tile, on the bridge's
SIMULATED grid (stride = tile*0.8 = 512, flush-last) -- the same grid
_find_gap_bridge_tiles uses. Merges per-frame fragments into trails first (the
fragments are how long trails are stored), then for each trail measures its
longest extent within any single tile. Reports a threshold sweep so we can see
540 vs 640. gkyle excluded. Read-only.
"""
import json, glob, os, collections, requests
import numpy as np
import shapely.geometry as sgeom
from shapely.ops import unary_union

BK = "/Users/bruceherwig/Claude_Code_Projects/runs/cvat_backup_2026_06_05"
SZ = 640
STRIDE = int(SZ * 0.8)            # 512, matches _find_gap_bridge_tiles
THRESHOLDS = [500, 540, 600, 640, 700]
ANG_TOL = 12.0                    # deg, collinear merge
PERP_TOL = 70.0                   # px, collinear merge

pw = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
s = requests.Session(); s.auth = ("bherwig2", pw)
name_wh = {}
url = "http://localhost:8080/api/tasks?page_size=100"
while url:
    j = s.get(url).json()
    for t in j["results"]:
        try:
            m = s.get(f"http://localhost:8080/api/tasks/{t['id']}/data/meta").json()
            if m.get("frames"):
                name_wh[t["name"]] = (m["frames"][0]["width"], m["frames"][0]["height"])
        except Exception:
            pass
    url = j.get("next")


def pp(p):
    if isinstance(p, str):
        p = json.loads(p)
    return np.array([[p[i], p[i + 1]] for i in range(0, len(p) - 1, 2)], float)


def axis(a):
    m = a.mean(0)
    _, _, vt = np.linalg.svd(a - m, full_matrices=False)
    return m, vt[0]


def ang(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 180


def origins(L):
    xs = list(range(0, max(1, L - SZ) + 1, STRIDE))
    if L > SZ and xs[-1] != L - SZ:
        xs.append(L - SZ)
    return xs if L > SZ else [0]


def max_extent(geom):
    pts = []
    gg = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for g in gg:
        if g.geom_type == "Polygon":
            pts.extend(list(g.exterior.coords))
    if len(pts) < 2:
        return 0.0
    a = np.array(pts)
    d = 0.0
    for i in range(len(a)):
        d = max(d, float(np.sqrt(((a[i] - a) ** 2).sum(1)).max()))
    return d


totals = collections.Counter()
print(f"{'dataset':40} " + " ".join(f">={t}" for t in THRESHOLDS))
for bp in sorted(glob.glob(os.path.join(BK, "task_*_annotations.json"))):
    d = json.load(open(bp))
    ds = d.get("name", "?")
    if "gkyle" in ds.lower():
        continue
    wh = name_wh.get(ds)
    if not wh:
        continue
    W, H = wh
    xs0, ys0 = origins(W), origins(H)
    byf = collections.defaultdict(list)
    for sh in d["annotations"]["shapes"]:
        if sh.get("type") != "polygon":
            continue
        a = pp(sh["points"])
        if len(a) >= 3:
            byf[sh["frame"]].append(a)
    per = collections.Counter()
    for fr, polys in byf.items():
        n = len(polys)
        info = [axis(a) for a in polys]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(n):
            mi, vi = info[i]
            for k in range(i + 1, n):
                mk, vk = info[k]
                da = abs(ang(vi) - ang(vk)); da = min(da, 180 - da)
                if da > ANG_TOL:
                    continue
                dd = mk - mi
                perp = abs(dd[0] * (-vi[1]) + dd[1] * vi[0])
                if perp < PERP_TOL:
                    parent[find(i)] = find(k)
        groups = collections.defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        for g in groups.values():
            shapes = []
            for i in g:
                poly = sgeom.Polygon(polys[i])
                shapes.append(poly.buffer(0) if not poly.is_valid else poly)
            trail = unary_union(shapes)
            bx0, by0, bx1, by1 = trail.bounds
            best = 0.0
            for tx in xs0:
                if tx > bx1 or tx + SZ < bx0:
                    continue
                for ty in ys0:
                    if ty > by1 or ty + SZ < by0:
                        continue
                    inter = trail.intersection(sgeom.box(tx, ty, tx + SZ, ty + SZ))
                    if not inter.is_empty:
                        best = max(best, max_extent(inter))
            for t in THRESHOLDS:
                if best >= t:
                    per[t] += 1
    for t in THRESHOLDS:
        totals[t] += per[t]
    if per[THRESHOLDS[-1]] or per[THRESHOLDS[0]]:
        print(f"{ds[:40]:40} " + " ".join(f"{per[t]:>4}" for t in THRESHOLDS))
print("\nTOTAL " + " ".join(f">={t}: {totals[t]}" for t in THRESHOLDS))
