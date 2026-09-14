"""Upload a GoPro/STC dataset to CVAT using STC's OWN polygons (_polys.json),
in true numeric frame order. Single source of truth: STC detects and fits the
polygons; this only reformats them to CVAT shapes and uploads. No mask-to-contour
re-derivation.

HONEST UPLOAD (added 2026-06-11): the app's _polys.json is written BEFORE the
static false-positive suppressor runs (Step 1c in astro_clean_v5.py), so it can
contain outlines the app itself rejected (e.g. a static plume above a ridge,
rejected on 21 straight frames, all 21 uploaded to task 54 anyway). This tool
now reads the run logs sitting next to the masks dir, collects every rejected
spot (static_fp_suppressed records), and SKIPS any outline that sits on one.
Every skip is printed (frame + location); nothing is dropped silently. If no
run logs are found the tool refuses to run rather than upload unfiltered.
Once the app applies the verdict to its own polygon list (todo #94), this
filter simply finds nothing to skip.

Pipeline per dataset:
  1. create a task under project 1 ("Star Trail CleanR", label 'trail' id 1)
  2. upload the source images in natural-numeric order (sorting_method=predefined)
  3. read each <stem>_polys.json from the masks dir, drop suppressor-rejected
     outlines, map frame name -> CVAT frame index, build polygon shapes,
     upload them as annotations.

Usage:
  python3 cvat_upload_from_polys.py --src <frames_dir> --masks <masks_dir>
        --name "<task name>" [--frames 2,10,100,50,9]   # subset for a test
        [--order capture] [--fg-mask <foreground_mask.png>] [--dry-run]

ADDED 2026-09-13 (after CVAT tasks 68-70 were built by a side path that should
have been this tool):

  --order capture   Upload frames in SHOOTING order (EXIF capture time) instead
                    of filename order. Needed whenever a camera counter rolls
                    over: the Joshua Tree 26.4 set runs IMG_9799-9999 and then
                    IMG_0001-0245, and a filename sort puts the last 245 frames
                    first so the stars jump at the seam in TileFixR. The default
                    stays "name" so earlier behaviour is unchanged, but the tool
                    now WARNS whenever the two orders disagree.

  --fg-mask PATH    Remove every polygon inside the foreground mask (white =
                    foreground, black = open sky). A polygon crossing the edge
                    keeps only its sky part, as its REAL clipped outline --
                    never a re-fitted rectangle, which beside a diagonal
                    roofline lands straight back on the building (that mistake
                    left 298 polygons on structure in task 68). Bruce's rule:
                    "remove all CVAT polygons in mask area".

  --dry-run         Do everything except talk to CVAT: order the frames, apply
                    both filters, and report the counts. Check before creating
                    a task.

Run logs: the app now moves older logs into "STC Extras/Archive". Those are
searched too -- the chapel set's own detection run lives there, and missing it
would have let suppressor-rejected outlines through without a word.
"""
import argparse, json, os, re, sys, time, glob
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CVAT = "http://localhost:8080"
USER = "bherwig2"
PW = open(os.path.expanduser("~/.star_trail_cleanr/cvat_credentials")).read().strip()
PROJECT_ID = 1
TRAIL_LABEL_ID = 1
S = requests.Session()
S.auth = (USER, PW)

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def natkey(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(p))]


def list_frames(src, only=None, order="name"):
    fs = [f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in EXTS]
    if only:
        keep = set(only)
        fs = [f for f in fs if os.path.splitext(f)[0] in keep]
    by_name = sorted(fs, key=natkey)

    from modules.io_safe import capture_time
    times = {f: capture_time(os.path.join(src, f)) for f in fs}
    timed = sorted((f for f in fs if times[f] is not None), key=lambda f: times[f])
    untimed = sorted((f for f in fs if times[f] is None), key=natkey)
    by_capture = timed + untimed

    if by_capture != by_name:
        seam = next(i for i, (a, b) in enumerate(zip(by_name, by_capture)) if a != b)
        print(f"WARNING: filename order and shooting order DISAGREE from position "
              f"{seam} (name says {by_name[seam]}, capture time says "
              f"{by_capture[seam]}). A rolled-over camera counter does this. "
              f"{'Using SHOOTING order.' if order == 'capture' else 'Using FILENAME order -- pass --order capture to fix.'}")
    if untimed and order == "capture":
        print(f"  {len(untimed)} frame(s) have no capture time; placed last in filename order")
    return by_capture if order == "capture" else by_name


def create_task(name):
    r = S.post(f"{CVAT}/api/tasks", json={"name": name, "project_id": PROJECT_ID})
    r.raise_for_status()
    return r.json()["id"]


def upload_images(task_id, src, frames, batch_size=20):
    # CVAT batched upload protocol: Upload-Start (with the data params), then
    # one Upload-Multiple request per batch of images, then Upload-Finish.
    # sorting_method=predefined keeps CVAT frame order = the order we send the
    # files (global client_files index), i.e. true numeric order.
    base = f"{CVAT}/api/tasks/{task_id}/data"
    # params sent as multipart fields (CVAT's /data endpoint rejects plain
    # form-urlencoded with 415; (None, value) tuples force multipart).
    params = [("image_quality", (None, "90")), ("sorting_method", (None, "predefined"))]
    r = S.post(base, files=params, headers={"Upload-Start": "true"})
    r.raise_for_status()
    n = len(frames)
    for i in range(0, n, batch_size):
        chunk = frames[i:i + batch_size]
        files, fhs = [], []
        for j, fn in enumerate(chunk):
            fh = open(os.path.join(src, fn), "rb")
            fhs.append(fh)
            files.append((f"client_files[{i + j}]", (fn, fh, "image/jpeg")))
        # CVAT requires image_quality on EVERY Upload-Multiple request too.
        files += params
        r = S.post(base, files=files, headers={"Upload-Multiple": "true"})
        for fh in fhs:
            fh.close()
        r.raise_for_status()
        print(f"  uploaded {min(i + batch_size, n)}/{n} images", flush=True)
    r = S.post(base, files=params, headers={"Upload-Finish": "true"})
    if not r.ok:
        print("FINISH error", r.status_code, r.text[:600])
    r.raise_for_status()


def wait_ready(task_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = S.get(f"{CVAT}/api/tasks/{task_id}/status")
        st = r.json() if r.ok else {}
        state = st.get("state")
        if state == "Finished":
            return True
        if state == "Failed":
            raise RuntimeError(f"task data processing failed: {st.get('message')}")
        time.sleep(2)
    raise TimeoutError("task data not ready in time")


def frame_index_map(task_id):
    meta = S.get(f"{CVAT}/api/tasks/{task_id}/data/meta").json()
    return {os.path.splitext(f["name"])[0]: i for i, f in enumerate(meta["frames"])}, meta


def job_id_for(task_id):
    jobs = S.get(f"{CVAT}/api/jobs?task_id={task_id}").json()["results"]
    return jobs[0]["id"]


def load_suppressed_boxes(masks_dir):
    """Read the run logs next to the masks dir and return, per frame stem, the
    bounding boxes of every detection the static FP suppressor rejected.

    The logs live in the cleanr_workspace folder (the masks dir's parent) as
    run_log_*.jsonl, one per batch. A spot rejected in ANY run counts: the
    suppressor keys on same-position recurrence, which a real trail can never
    show, so an old rejection at the same spot is still valid evidence.
    """
    workspace = os.path.dirname(os.path.abspath(masks_dir))
    # Archive/ too: the app tucks older logs away there, including the run that
    # produced the masks being uploaded.
    log_files = sorted(glob.glob(os.path.join(workspace, "run_log_*.jsonl"))
                       + glob.glob(os.path.join(workspace, "Archive", "run_log_*.jsonl")))
    if not log_files:
        sys.exit(f"ERROR: no run_log_*.jsonl found in {workspace} -- cannot "
                 "filter suppressor-rejected outlines. Refusing to upload "
                 "unfiltered. (Run the dataset through STC with mask saving "
                 "on, or point --masks at the right cleanr_workspace/masks.)")
    boxes = {}
    for lf in log_files:
        for line in open(lf):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "detect":
                continue
            for s in e.get("static_fp_suppressed", []):
                bb = s.get("bbox")
                if bb and len(bb) == 4:
                    boxes.setdefault(str(e.get("frame")), set()).add(tuple(bb))
    print(f"suppressor verdicts loaded: {sum(len(v) for v in boxes.values())} "
          f"rejected spot(s) on {len(boxes)} frame(s), from {len(log_files)} run logs")
    return boxes


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


FG_MIN_AREA = 400   # a sky remainder smaller than this (px) is a scrap, not a trail


def load_sky(fg_mask_path):
    """Open-sky map from a foreground mask (white = foreground, black = sky)."""
    import cv2
    fg = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
    if fg is None:
        sys.exit(f"ERROR: cannot read foreground mask {fg_mask_path}")
    print(f"foreground mask: {fg.shape[1]}x{fg.shape[0]}, "
          f"{100 * (fg > 127).mean():.1f}% foreground")
    return (fg == 0).astype("uint8")


def clip_to_sky(corners, sky):
    """Return the polygon's open-sky part as its real outline, or None.

    None when it lies wholly (or all but a scrap) inside the foreground. The
    clipped shape is the actual contour of what survives, simplified but never
    replaced by a bounding box."""
    import cv2
    import numpy as np
    m = np.zeros(sky.shape, np.uint8)
    cv2.fillPoly(m, [np.array(corners, float).reshape(-1, 2).astype(np.int32)], 1)
    before = int(m.sum())
    m = m * sky
    if int(m.sum()) < FG_MIN_AREA:
        return None, before
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < FG_MIN_AREA:
        return None, before
    poly = cv2.approxPolyDP(big, 0.01 * cv2.arcLength(big, True), True).reshape(-1, 2)
    if len(poly) < 3:
        return None, before
    return [[float(x), float(y)] for x, y in poly], int(m.sum()) < before * 0.98


def build_shapes(masks_dir, idx_map, suppressed_boxes, sky=None):
    shapes = []
    n_skipped = 0
    n_fg_deleted = n_fg_trimmed = 0
    for stem, frame_idx in idx_map.items():
        pj = os.path.join(masks_dir, f"{stem}_polys.json")
        if not os.path.exists(pj):
            continue
        data = json.load(open(pj))
        rejected = suppressed_boxes.get(stem, ())
        for poly in data.get("polygons", []):
            corners = poly.get("corners", [])
            if len(corners) < 3:
                continue
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            pbox = (min(xs), min(ys), max(xs), max(ys))
            hit = next((rb for rb in rejected if _bbox_iou(pbox, rb) > 0.3), None)
            if hit is not None:
                n_skipped += 1
                print(f"  SKIP frame {stem}: outline at x[{pbox[0]}-{pbox[2]}] "
                      f"y[{pbox[1]}-{pbox[3]}] -- FP detector rejected this spot")
                continue
            if sky is not None:
                clipped, trimmed = clip_to_sky(corners, sky)
                if clipped is None:
                    n_fg_deleted += 1
                    continue
                if trimmed:
                    n_fg_trimmed += 1
                corners = clipped
            pts = [float(v) for xy in corners for v in xy]
            shapes.append({
                "type": "polygon", "occluded": False, "outside": False,
                "z_order": 0, "rotation": 0.0, "points": pts, "frame": frame_idx,
                "label_id": TRAIL_LABEL_ID, "group": 0, "source": "auto",
                "attributes": [],
            })
    print(f"skipped {n_skipped} suppressor-rejected outline(s) total")
    if sky is not None:
        print(f"foreground mask: deleted {n_fg_deleted} outline(s) inside it, "
              f"trimmed {n_fg_trimmed} at its edge")
    return shapes


def upload_annotations(job_id, shapes):
    body = {"version": 0, "tags": [], "shapes": shapes, "tracks": []}
    r = S.put(f"{CVAT}/api/jobs/{job_id}/annotations?action=create", json=body)
    if not r.ok:
        print("annotation upload error:", r.status_code, r.text[:500])
        r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--order", choices=("name", "capture"), default="name",
                    help="frame order in the task: filename (default) or shooting time")
    ap.add_argument("--fg-mask", default=None,
                    help="foreground_mask.png: remove every polygon inside it")
    ap.add_argument("--dry-run", action="store_true",
                    help="order and filter only; create nothing in CVAT")
    args = ap.parse_args()
    only = [x for x in args.frames.split(",") if x] or None
    frames = list_frames(args.src, only, args.order)
    print(f"frames to upload ({len(frames)}): {[os.path.splitext(f)[0] for f in frames][:30]}")
    suppressed_boxes = load_suppressed_boxes(args.masks)
    sky = load_sky(args.fg_mask) if args.fg_mask else None
    if args.dry_run:
        idx_map = {os.path.splitext(f)[0]: i for i, f in enumerate(frames)}
        shapes = build_shapes(args.masks, idx_map, suppressed_boxes, sky)
        print(f"DRY RUN: would create '{args.name}' with {len(frames)} frames "
              f"and {len(shapes)} polygons on {len({s['frame'] for s in shapes})} "
              f"of them. Nothing sent to CVAT.")
        return
    tid = create_task(args.name)
    print(f"created task {tid}")
    upload_images(tid, args.src, frames)
    print("images posted, waiting for processing...")
    wait_ready(tid)
    idx_map, meta = frame_index_map(tid)
    order = [meta["frames"][i]["name"] for i in range(len(meta["frames"]))]
    print(f"CVAT frame order: {[os.path.splitext(n)[0] for n in order]}")
    jid = job_id_for(tid)
    shapes = build_shapes(args.masks, idx_map, suppressed_boxes, sky)
    print(f"uploading {len(shapes)} polygon shapes to job {jid}")
    upload_annotations(jid, shapes)
    # verify
    ann = S.get(f"{CVAT}/api/jobs/{jid}/annotations").json()
    print(f"VERIFY: task {tid} now has {len(ann.get('shapes', []))} shapes")
    print(f"TASK_ID={tid}")


if __name__ == "__main__":
    main()
