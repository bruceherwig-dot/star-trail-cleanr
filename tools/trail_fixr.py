#!/usr/bin/env python3
"""
TrailFixR — Visual polygon reviewer AND editor for YOLO trail detections.

WHAT IT IS
    A standalone PySide6 desktop app (one of the maintained developer tools in
    tools/). It is Bruce's polygon-quality loop for the CVAT "Star Trail
    CleanR" project: it pulls the reviewed trail polygons for one CVAT task,
    crops a padded window around each polygon, and shows them one at a time so
    he can confirm, fix, add, or delete annotations. Every change is batched
    and pushed straight back to CVAT — so the polygons here ARE the live CVAT
    annotations, not a local copy.

HOW TO RUN IT
    python3 tools/trail_fixr.py
    On launch it lists every CVAT task, lets you pick one plus a frame range
    (TaskPickerDialog), shows a splash while it loads/crops (SplashWindow),
    then opens the main editor (TrailFixR). Needs the local CVAT Docker
    instance up at http://localhost:8080 and the password file at
    ~/.star_trail_cleanr/cvat_credentials. Single-instance locked.

WHERE IT FITS
    Part of Bruce's 3-tool annotation-quality loop (Trail ScreenR -> TileFixR ->
    Mask CheckR). TrailFixR is the per-polygon reviewer/editor: it works one
    detection at a time, with brightness/contrast, zoom/pan, vertex dragging,
    box-extend handles, an auto-tighten snap, add-polygon mode, and a
    mark-for-delete flow. It can also consume "Claude flags" (a JSON list of
    server_ids that an automated pass flagged as likely false positives) so the
    flagged ones can be reviewed and accepted in bulk.

WHAT THE USER DOES HERE
    - Walk polygons with arrow keys / Tab / the scrubber / a jump box.
    - Brighten/contrast-boost the crop to see faint trails (display only).
    - Click a polygon to select it; drag its vertices or the diamond
      extend-handles to reshape it; press T to auto-snap it to the bright
      trail core; press A to draw a brand-new polygon.
    - Press Space/D/Del to mark a polygon for deletion (false positive).
    - Send all pending adds + edits + deletes to CVAT in one batch.
    - Copy a reference string for any polygon to paste back to Claude.
    - "Add To WeirdR" tags odd cases into a shared weirdr_list.json for later.

PERSISTENCE / SAFETY
    Per-task state lives under ~/.star_trail_cleanr/trail_fixr/task_<id>/
    (cursor position, delete marks, in-progress edits/adds, and any Claude
    flags). A separate per-task crop cache under ~/.star_trail_cleanr/cvat_cache/
    speeds relaunches by reusing crops whose polygon points haven't changed.
    Edits and marks auto-save on every change so a crash or relaunch never
    loses pending work.

This file only reads from / writes to CVAT and its own state/cache folders. It
does NOT touch the YOLO model, the detection pipeline, or any source images
(it only reads source images to make crops).

Usage:
    python3 tools/trail_fixr.py
"""

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import requests
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QButtonGroup, QMessageBox, QProgressBar,
    QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame,
    QLineEdit, QToolButton, QDialog, QComboBox, QSpinBox, QDialogButtonBox
)
from PySide6.QtGui import QPixmap, QImage, QKeySequence, QShortcut, QFont, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QRectF, QEvent, Signal, QLockFile, QDir, QTimer, QThread


class ClickableLabel(QLabel):
    """A QLabel that also fires a doubleClicked signal. Used for the
    "Brightness" / "Contrast" text labels so double-clicking the word resets
    that slider to 1.0x."""
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event):
        """Emit doubleClicked, then let the normal QLabel handling run."""
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

# --- Config ---
CVAT_URL = "http://localhost:8080"
CVAT_USER = "bherwig2"
TRAILS_ROOT = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images")

# Maps CVAT task names that don't match their folder names on T7.
TASK_FOLDER_ALIASES = {
    "My First Star Trail": "Bruce Herwig - first star trail data",
    "Thomas Jackson - Borrego": "Thomas Jackson Star Trails Borrego",
}

GKYLE_STAGING = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/external_datasets/gkyle_startrails/cvat_staging")

# These four globals are overwritten by the task picker at launch.
CVAT_TASK_ID = 15
FRAME_START = 0
FRAME_END = 963
IMG_DIR = TRAILS_ROOT / "Greg Meyer Arizona Brightened"
TASK_NAME = "Greg Meyer Arizona Brightened - v8 slope-match"
WEIRDR_PATH = Path(__file__).parent.parent / "weirdr_list.json"

MIN_CROP_PAD = 50
CROP_PAD_RATIO = 0.20  # 20% of the longer polygon dimension

# Per-task state directory — one subfolder per CVAT task, nothing shared.
STATE_DIR = Path.home() / ".star_trail_cleanr" / "trail_fixr"
LAST_PICK_PATH = Path.home() / ".star_trail_cleanr" / "trail_fixr_last_task.json"

# These are recomputed by _refresh_paths() every time the picker sets a new task.
STATE_PATH = STATE_DIR / f"task_{CVAT_TASK_ID}" / "state.json"
MARKS_PATH = STATE_DIR / f"task_{CVAT_TASK_ID}" / "marks.json"
CLAUDE_FLAGS_PATH = STATE_DIR / f"task_{CVAT_TASK_ID}" / "flags.json"

# --- Theme colors (light defaults, updated by _apply_theme) ---
MUTED_TEXT = "#666"
HINT_TEXT = "#888"
PANEL_BG = "#f5f5f5"
IMAGE_BG = "#1a1a2e"
INFO_BG = "#f0f0f0"
INFO_TEXT = "#000"

# --- Edit-overlay colors (BGR for cv2) ---
SELECTED_COLOR = (255, 255, 255)   # white outline on selected polygon
DELETE_COLOR = (60, 60, 220)       # red overlay for marked-for-delete
EDITED_COLOR = (0, 255, 255)       # yellow outline for edited polygons
ADDED_COLOR = (50, 255, 50)        # lime outline for newly added polygons
CURRENT_COLOR = (0, 255, 0)        # default green for current (entry) polygon


def _apply_theme():
    """Detect the OS light/dark setting and switch the panel/text/background
    color constants to a dark palette when the system is in dark mode. Light
    is the default; this only overrides on dark."""
    global MUTED_TEXT, HINT_TEXT, PANEL_BG, IMAGE_BG, INFO_BG, INFO_TEXT
    try:
        scheme = QApplication.styleHints().colorScheme()
        is_dark = (scheme == Qt.ColorScheme.Dark)
    except Exception:
        is_dark = False
    if is_dark:
        MUTED_TEXT = "#aaaaaa"
        HINT_TEXT = "#9aa4b0"
        PANEL_BG = "#2d3138"
        IMAGE_BG = "#0d0d1a"
        INFO_BG = "#2d3138"
        INFO_TEXT = "#e6e6e6"


def _refresh_paths():
    """Recompute per-task state paths after CVAT_TASK_ID changes."""
    global STATE_PATH, MARKS_PATH, CLAUDE_FLAGS_PATH
    task_dir = STATE_DIR / f"task_{CVAT_TASK_ID}"
    STATE_PATH = task_dir / "state.json"
    MARKS_PATH = task_dir / "marks.json"
    CLAUDE_FLAGS_PATH = task_dir / "flags.json"


def resolve_image_dir(task_name):
    """Find the local image folder for a CVAT task by name.
    Tries alias lookup, then exact match, then strips ' - v...' version suffix, then prefix match."""
    if "gkyle" in task_name.lower() and GKYLE_STAGING.exists():
        return GKYLE_STAGING
    if task_name in TASK_FOLDER_ALIASES:
        p = TRAILS_ROOT / TASK_FOLDER_ALIASES[task_name]
        if p.exists():
            return p
    candidates = [task_name]
    if " - v" in task_name:
        candidates.append(task_name.split(" - v")[0].rstrip())
    for c in candidates:
        p = TRAILS_ROOT / c
        if p.exists():
            return p
    base = candidates[-1]
    if TRAILS_ROOT.exists():
        for child in TRAILS_ROOT.iterdir():
            if child.is_dir() and base.lower() in child.name.lower():
                return child
    return None


def fetch_cvat_tasks(auth):
    """List all CVAT tasks, name-sorted. Returns list of {id, name, size}."""
    out = []
    url = f"{CVAT_URL}/api/tasks"
    while url:
        try:
            r = requests.get(url, auth=auth).json()
        except Exception as e:
            print(f"fetch_cvat_tasks error: {e}")
            break
        for t in r.get("results", []):
            out.append({"id": t["id"], "name": t["name"], "size": t.get("size", 0)})
        url = r.get("next")
    out.sort(key=lambda t: t["name"].lower())
    return out


def load_last_pick():
    """Return the last-used {task_id, first_frame, last_frame} so the picker
    can reopen on the same task and range. Falls back to a hardcoded default
    if the file is missing or unreadable."""
    if LAST_PICK_PATH.exists():
        try:
            data = json.loads(LAST_PICK_PATH.read_text())
            return data
        except Exception:
            pass
    return {"task_id": 15, "first_frame": 0, "last_frame": 962}


def save_last_pick(task_id, first_frame, last_frame):
    """Remember the chosen task + frame range to disk so the next launch
    defaults to it."""
    LAST_PICK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_PICK_PATH.write_text(json.dumps({
        "task_id": task_id,
        "first_frame": int(first_frame),
        "last_frame": int(last_frame),
    }, indent=2))


def read_cvat_password():
    """Read the CVAT password from the credentials file (kept outside the repo
    at ~/.star_trail_cleanr/cvat_credentials, never hardcoded)."""
    return (Path.home() / ".star_trail_cleanr" / "cvat_credentials").read_text().strip()


def get_cvat_data():
    """Get native annotations and frame names from CVAT."""
    password = read_cvat_password()
    auth = (CVAT_USER, password)

    meta = requests.get(f"{CVAT_URL}/api/tasks/{CVAT_TASK_ID}/data/meta", auth=auth).json()
    frame_names = [f["name"] for f in meta["frames"]]

    jobs = requests.get(f"{CVAT_URL}/api/jobs", params={"task_id": CVAT_TASK_ID}, auth=auth).json()
    job_id = jobs["results"][0]["id"]

    ann = requests.get(f"{CVAT_URL}/api/jobs/{job_id}/annotations", auth=auth).json()

    shapes = [s for s in ann["shapes"] if FRAME_START <= s["frame"] < FRAME_END]

    all_sorted = sorted(ann["shapes"], key=lambda s: (s["frame"], s["id"]))
    server_id_to_seq = {s["id"]: i + 1 for i, s in enumerate(all_sorted)}

    return shapes, frame_names, server_id_to_seq, job_id


def _cache_dir():
    """Per-task cache directory. Each (task, frame range) gets its own folder
    so different tasks don't collide."""
    base = Path.home() / ".star_trail_cleanr" / "cvat_cache"
    return base / f"task_{CVAT_TASK_ID}" / f"frames_{FRAME_START}_{FRAME_END - 1}"


def _cache_index_path():
    """Path to the cache index JSON (one per task/frame-range) that records,
    per polygon, the point hash and crop offsets used to validate cache hits."""
    return _cache_dir() / "index.json"


def _cache_crop_path(server_id):
    """Path to the cached crop JPG for one polygon, named by its CVAT
    server_id."""
    return _cache_dir() / "crops" / f"{int(server_id)}.jpg"


def _points_hash(pts):
    """Cheap stable hash of a polygon's point list. Used to detect when a
    polygon's shape has changed since the cache was written."""
    arr = np.asarray(pts, dtype=np.float32).round(2).tobytes()
    import hashlib
    return hashlib.md5(arr).hexdigest()


def _load_cache_index():
    """Returns the cached {server_id: meta} dict, or {} if no cache exists
    or if the cache is unreadable."""
    try:
        path = _cache_index_path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {int(k): v for k, v in data.get("entries", {}).items()}
    except Exception as exc:
        print(f"  cache: failed to read index ({exc}), ignoring cache")
        return {}


def _save_cache(entries_meta):
    """Write the cache index. entries_meta is a dict of
    {server_id: {hash, crop_x, crop_y, frame_idx, fname, img_w, img_h}}."""
    try:
        path = _cache_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "crops").mkdir(parents=True, exist_ok=True)
        # Write atomically via temp + rename
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "task_id": CVAT_TASK_ID,
            "frame_start": FRAME_START,
            "frame_end": FRAME_END,
            "entries": {str(k): v for k, v in entries_meta.items()},
        }, indent=1))
        tmp.replace(path)
    except Exception as exc:
        print(f"  cache: failed to write index ({exc})")


def load_and_analyze(progress_cb=None):
    """Load polygons from CVAT, crop with proportional padding.

    progress_cb(text: str, current: int, total: int) is called periodically
    so callers (the splash screen) can keep the UI responsive.

    Cache aware: keeps a per-task cache of crops + polygon-point hashes on
    disk under ~/.star_trail_cleanr/cvat_cache/. On relaunch, polygons whose
    points haven't changed since the cache was written reuse the cached crop
    instead of re-reading the source image - so a no-change relaunch on a
    4,000-polygon task drops from ~50 sec to ~5 sec.
    """
    if progress_cb:
        progress_cb("Connecting to CVAT and fetching annotations...", 0, 0)
    print("Fetching fresh annotations from CVAT...", flush=True)
    shapes, frame_names, server_id_to_seq, job_id = get_cvat_data()
    print(f"Got {len(shapes)} shapes in frames {FRAME_START}-{FRAME_END - 1}")

    cache_index = _load_cache_index()
    cache_dir = _cache_dir()
    crop_dir = cache_dir / "crops"
    if cache_index:
        print(f"  cache: index has {len(cache_index)} polygons "
              f"at {cache_dir}")

    if progress_cb:
        progress_cb(f"Got {len(shapes)} polygons. Diffing against cache...",
                    0, len(shapes))

    entries = []
    img_cache = {}
    new_cache_index = {}
    n_cache_hits = 0
    n_cache_misses = 0

    for si, shape in enumerate(sorted(shapes, key=lambda s: (s["frame"], s["id"]))):
        frame_idx = shape["frame"]
        if frame_idx >= len(frame_names):
            continue

        fname = frame_names[frame_idx]
        img_path = IMG_DIR / fname
        if not img_path.exists():
            img_path = IMG_DIR / "JPGs" / fname
        if not img_path.exists():
            img_path = IMG_DIR / "jpg" / fname

        sys.stdout.write(f"\r  Processing {si + 1}/{len(shapes)}  {fname}   ")
        sys.stdout.flush()
        # Update splash every 25 polygons so it pumps the event loop without
        # tanking throughput.
        if progress_cb and (si % 25 == 0):
            progress_cb(
                f"Processing {si + 1:,} of {len(shapes):,} polygons "
                f"(reused {n_cache_hits:,} from cache)...",
                si + 1, len(shapes))

        raw_pts = shape["points"]
        pts = np.array(list(zip(raw_pts[0::2], raw_pts[1::2])), dtype=np.float32)
        if len(pts) < 3:
            continue

        sid = shape["id"]
        h_pts = _points_hash(pts)
        cached = cache_index.get(sid)
        crop_raw = None

        # Try cache hit: same polygon, same points, same source frame
        if (cached is not None
                and cached.get("hash") == h_pts
                and cached.get("frame_idx") == frame_idx
                and cached.get("fname") == fname):
            crop_path = _cache_crop_path(sid)
            if crop_path.exists():
                cached_crop = cv2.imread(str(crop_path))
                if cached_crop is not None:
                    crop_raw = cached_crop
                    x_min = int(cached["crop_x"])
                    y_min = int(cached["crop_y"])
                    w = int(cached["img_w"])
                    h = int(cached["img_h"])
                    n_cache_hits += 1

        if crop_raw is None:
            # Cache miss - read source image and crop
            n_cache_misses += 1
            if frame_idx not in img_cache:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_cache[frame_idx] = img
                if len(img_cache) > 5:
                    oldest = min(img_cache.keys())
                    del img_cache[oldest]
            img = img_cache[frame_idx]
            h, w = img.shape[:2]

            bbox_w = pts[:, 0].max() - pts[:, 0].min()
            bbox_h = pts[:, 1].max() - pts[:, 1].min()
            pad = max(MIN_CROP_PAD, int(CROP_PAD_RATIO * max(bbox_w, bbox_h)))

            x_min = max(0, int(pts[:, 0].min()) - pad)
            y_min = max(0, int(pts[:, 1].min()) - pad)
            x_max = min(w, int(pts[:, 0].max()) + pad)
            y_max = min(h, int(pts[:, 1].max()) + pad)

            crop_raw = img[y_min:y_max, x_min:x_max].copy()

            # Persist the crop to the cache directory at JPG quality 95
            try:
                crop_path = _cache_crop_path(sid)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_path), crop_raw,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception:
                pass

        # crop_raw exists either way; derive x_max/y_max from its dimensions
        # so the cache-hit branch matches the cache-miss branch.
        x_max = x_min + crop_raw.shape[1]
        y_max = y_min + crop_raw.shape[0]

        # Record the cache entry for this polygon
        new_cache_index[sid] = {
            "hash": h_pts,
            "frame_idx": int(frame_idx),
            "fname": fname,
            "crop_x": int(x_min),
            "crop_y": int(y_min),
            "img_w": int(w),
            "img_h": int(h),
        }

        shifted_pts = pts.copy()
        shifted_pts[:, 0] -= x_min
        shifted_pts[:, 1] -= y_min

        seq_num = server_id_to_seq.get(shape["id"], 0)
        stem = Path(fname).stem

        entries.append({
            "frame": stem,
            "frame_idx": frame_idx,
            "server_id": shape["id"],
            "cvat_seq": seq_num,
            "label": f"{stem} #{shape['id']}",
            "crop_raw": crop_raw,
            "crop_x": x_min,
            "crop_y": y_min,
            "img_path": str(img_path),
            "img_w": w,
            "img_h": h,
            "poly_pts": shifted_pts.astype(np.int32),
            "flag_reason": "",
            "flagged": False,
            "delete": False,
        })

    # --- Load Claude flags if they exist ---
    if CLAUDE_FLAGS_PATH.exists():
        cf = json.loads(CLAUDE_FLAGS_PATH.read_text())
        # Flags are keyed by server_id (stable CVAT database ID). Fall back to cvat_seq
        # for legacy flag files written before the identifier fix.
        claude_flagged = {}
        for item in cf.get("flagged_for_deletion", []):
            key = item.get("server_id")
            if key is None:
                key = item.get("cvat_seq") or item.get("index")
            if key is not None:
                claude_flagged[key] = item.get("reason", "")
        if claude_flagged:
            for e in entries:
                if e["server_id"] in claude_flagged:
                    e["flagged"] = True
                    e["flag_reason"] = claude_flagged[e["server_id"]]
                elif e["cvat_seq"] in claude_flagged:
                    e["flagged"] = True
                    e["flag_reason"] = claude_flagged[e["cvat_seq"]]
            flagged_count = sum(1 for e in entries if e["flagged"])
            print(f"\n  Loaded {flagged_count} Claude flags")

    # --- Restore Bruce's previous marks (delete toggles) ---
    if MARKS_PATH.exists():
        try:
            saved = json.loads(MARKS_PATH.read_text())
            marked_sids = set(saved.get("marked_server_ids", []))
            if marked_sids:
                for e in entries:
                    if e["server_id"] in marked_sids:
                        e["delete"] = True
                restored = sum(1 for e in entries if e["delete"])
                print(f"  Restored {restored} previously marked polygons from {MARKS_PATH.name}")
        except Exception as exc:
            print(f"  WARNING: could not read marks file: {exc}")

    # Persist the updated cache (even if all-hit, this also drops stale sids).
    _save_cache(new_cache_index)
    # Remove any cached crop files for polygons that are no longer in CVAT.
    stale = set(cache_index.keys()) - set(new_cache_index.keys())
    if stale:
        for old_sid in stale:
            try:
                _cache_crop_path(old_sid).unlink(missing_ok=True)
            except Exception:
                pass
        print(f"  cache: dropped {len(stale)} stale crops")
    print(f"  cache: {n_cache_hits:,} hits, {n_cache_misses:,} re-cropped")

    print(f"\n\nReady: {len(entries)} polygons")
    return entries, job_id


# --- Zoomable image viewer ---

class ZoomableImageView(QGraphicsView):
    """QGraphicsView with pinch-to-zoom, scroll-to-zoom, click + drag signals.

    Emits at image-pixel coords:
      pressed_at_image_xy(int, int)        - left button on press
      moved_to_image_xy(int, int)          - mouse move while button held
      released_at_image_xy(int, int)       - mouse release
      clicked_at_image_xy(float, float)    - release within ~10px of press (no drag)
      double_clicked_at_image_xy(int, int) - left-button double click
      right_clicked_at_image_xy(int, int)  - right-button click
      hovered_at_image_xy(int, int)        - mouse move with no button held
      zoom_changed                         - emitted whenever the view's scale changes
    """
    clicked_at_image_xy = Signal(float, float)
    pressed_at_image_xy = Signal(int, int)
    moved_to_image_xy = Signal(int, int)
    released_at_image_xy = Signal(int, int)
    double_clicked_at_image_xy = Signal(int, int)
    right_clicked_at_image_xy = Signal(int, int)
    hovered_at_image_xy = Signal(int, int)
    zoom_changed = Signal()

    def __init__(self, parent=None):
        """Set up the scene + pixmap item, enable hand-drag panning, pinch
        gesture, and mouse tracking (so hover events fire even with no button
        held)."""
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # Default to ScrollHandDrag for pan; parent flips to NoDrag during a
        # vertex / extend-handle grab to avoid fighting our gesture handlers.
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(400)
        self._zoom_level = 0
        self._press_pos = None
        self._press_button = None
        self._is_dragging = False
        self.grabGesture(Qt.PinchGesture)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def current_scale(self):
        """Current zoom factor (1.0 = fit). Used to size hit radii and overlay
        line/handle thickness in screen pixels regardless of zoom."""
        return float(self.transform().m11()) or 1.0

    def _to_image_xy(self, viewport_pos):
        """Convert a viewport (mouse) point to image-pixel coords. Returns None
        if the point falls outside the displayed image."""
        pix = self._pixmap_item.pixmap()
        if pix is None or pix.isNull():
            return None
        scene_pos = self.mapToScene(viewport_pos)
        item_pos = self._pixmap_item.mapFromScene(scene_pos)
        x, y = item_pos.x(), item_pos.y()
        if x < 0 or y < 0 or x >= pix.width() or y >= pix.height():
            return None
        return (x, y)

    def mousePressEvent(self, event):
        """Left press inside the image starts a possible click/drag and emits
        pressed_at_image_xy; right press emits right_clicked_at_image_xy.
        Anything else falls through to the default (pan) handler."""
        pos = event.position().toPoint()
        xy = self._to_image_xy(pos)
        if event.button() == Qt.LeftButton and xy is not None:
            self._press_pos = pos
            self._press_button = Qt.LeftButton
            self._is_dragging = False
            self.pressed_at_image_xy.emit(int(xy[0]), int(xy[1]))
            event.accept()
            return
        if event.button() == Qt.RightButton and xy is not None:
            self.right_clicked_at_image_xy.emit(int(xy[0]), int(xy[1]))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """While the left button is held: emit moved_to_image_xy, mark a drag
        once the cursor travels >=10px, and hand-pan the view if the parent
        hasn't claimed the gesture (NoDrag mode). With no button held: emit
        hovered_at_image_xy so the parent can light up handles/cursors."""
        if self._press_button == Qt.LeftButton and self._press_pos is not None:
            pos = event.position().toPoint()
            xy = self._to_image_xy(pos)
            if xy is not None:
                dx = pos.x() - self._press_pos.x()
                dy = pos.y() - self._press_pos.y()
                if abs(dx) >= 10 or abs(dy) >= 10:
                    self._is_dragging = True
                self.moved_to_image_xy.emit(int(xy[0]), int(xy[1]))
            # Manual pan when the parent hasn't claimed the gesture
            if self.dragMode() == QGraphicsView.ScrollHandDrag:
                dx = pos.x() - self._press_pos.x()
                dy = pos.y() - self._press_pos.y()
                if dx or dy:
                    hbar = self.horizontalScrollBar()
                    vbar = self.verticalScrollBar()
                    hbar.setValue(hbar.value() - dx)
                    vbar.setValue(vbar.value() - dy)
                    self._press_pos = pos
            return
        # No button: hover - emit so parent can update cursor over handles
        xy = self._to_image_xy(event.position().toPoint())
        if xy is not None:
            self.hovered_at_image_xy.emit(int(xy[0]), int(xy[1]))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """On left release: if the cursor never moved past the drag threshold,
        emit clicked_at_image_xy (a true click); always emit
        released_at_image_xy so the parent can finish a drag."""
        if event.button() == Qt.LeftButton and self._press_pos is not None:
            release_pos = event.position().toPoint()
            xy = self._to_image_xy(release_pos)
            if xy is not None:
                if not self._is_dragging:
                    dx = release_pos.x() - self._press_pos.x()
                    dy = release_pos.y() - self._press_pos.y()
                    if abs(dx) < 10 and abs(dy) < 10:
                        self.clicked_at_image_xy.emit(float(xy[0]), float(xy[1]))
                self.released_at_image_xy.emit(int(xy[0]), int(xy[1]))
            self._press_pos = None
            self._press_button = None
            self._is_dragging = False
        super().mouseReleaseEvent(event)

    def set_pixmap(self, pixmap):
        """Replace pixmap AND reset view to fit (use on entry change)."""
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._zoom_level = 0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def update_pixmap(self, pixmap):
        """Replace pixmap WITHOUT resetting zoom or transform - used during
        live drag so the user's zoom/pan stays put."""
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def event(self, event):
        """Route trackpad gesture events to the pinch handler; everything else
        gets the default handling."""
        if event.type() == QEvent.Gesture:
            return self._handle_gesture(event)
        return super().event(event)

    def _handle_gesture(self, event):
        """Apply a pinch gesture as a zoom (scale by the pinch factor) and
        announce zoom_changed so the overlay redraws at the new scale."""
        pinch = event.gesture(Qt.PinchGesture)
        if pinch is None:
            return False
        factor = pinch.scaleFactor()
        if factor != 1.0:
            self._zoom_level += 1 if factor > 1.0 else -1
            self.scale(factor, factor)
            self.zoom_changed.emit()
        return True

    def wheelEvent(self, event):
        """Scroll up zooms in, scroll down zooms out, and zoom_changed fires so
        the overlay redraws."""
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.25
            self._zoom_level += 1
        else:
            factor = 0.8
            self._zoom_level -= 1
        self.scale(factor, factor)
        self.zoom_changed.emit()

    def mouseDoubleClickEvent(self, event):
        """Emit double_clicked_at_image_xy (the parent uses it to close an
        in-progress add-polygon), then reset the view to fit-the-window zoom."""
        if event.button() == Qt.LeftButton:
            xy = self._to_image_xy(event.position().toPoint())
            if xy is not None:
                self.double_clicked_at_image_xy.emit(int(xy[0]), int(xy[1]))
        self._zoom_level = 0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_changed.emit()

    def resizeEvent(self, event):
        """Re-fit the image to the window on resize, but only while at the
        default (un-zoomed) level so a manual zoom isn't undone."""
        super().resizeEvent(event)
        if self._zoom_level == 0 and self._pixmap_item.pixmap() and not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


# --- Pill-style filter button ---

PILL_STYLE = """
    QPushButton {{
        background: {bg}; color: {fg}; font-size: 12px; font-weight: bold;
        border: 1px solid {border}; border-radius: 14px;
        padding: 6px 16px; min-width: 80px;
    }}
    QPushButton:checked {{
        background: {checked_bg}; color: white; border-color: {checked_border};
    }}
    QPushButton:hover:!checked {{
        background: {hover_bg};
    }}
"""


def make_pill(text, color_key="blue"):
    """Create a checkable pill-style toggle button."""
    colors = {
        "blue":   {"bg": "#e8f0fe", "fg": "#1a5276", "border": "#b0c4de",
                    "checked_bg": "#1a6fc4", "checked_border": "#145da0", "hover_bg": "#d0e0f0"},
        "orange": {"bg": "#fef3e8", "fg": "#7e5109", "border": "#e0c8a0",
                    "checked_bg": "#e67e22", "checked_border": "#cf6d17", "hover_bg": "#fde8d0"},
        "red":    {"bg": "#fde8e8", "fg": "#922b21", "border": "#e0b0b0",
                    "checked_bg": "#c0392b", "checked_border": "#a93226", "hover_bg": "#f8d0d0"},
        "green":  {"bg": "#e8fee8", "fg": "#1e7a1e", "border": "#b0deb0",
                    "checked_bg": "#2a7a2a", "checked_border": "#1e5f1e", "hover_bg": "#d0f0d0"},
    }
    c = colors.get(color_key, colors["blue"])
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet(PILL_STYLE.format(**c))
    return btn


# --- Main GUI ---

class TrailFixR(QMainWindow):
    """The main editor window.

    Takes the loaded `entries` (one per polygon, each carrying a cropped image
    + the polygon in crop-local coords) and a CVAT `job_id`, and presents them
    one at a time for review and editing.

    Two coordinate systems are in play:
      - Each entry's crop is shown in CROP-LOCAL pixels (origin at the crop's
        top-left), with crop_x/crop_y telling you where that crop sits in the
        full source frame.
      - The editable polygons live in `polygons_by_id`, the source of truth,
        stored in FULL-FRAME pixels. When drawing, full-frame points are
        shifted by -crop_x/-crop_y to land in the crop; when editing, crop
        clicks are shifted by +crop_x/+crop_y back to full-frame.

    A polygon's `status` is one of "original" (unchanged), "edited" (vertices
    moved), or "added" (drawn this session, negative temp id until pushed).
    Marked-for-delete polygons are tracked separately in `marked_ids`. The Send
    button pushes adds + edits + deletes to CVAT in one batch via
    CvatSendWorker.
    """
    # ── Hit-test radii (image-pixel sizes; scale by inverse zoom) ───────────
    HANDLE_RADIUS = 18      # vertex hit radius in image pixels at scale=1
    HANDLE_DRAW_R = 4       # vertex visual radius
    EXTEND_HIT_R = 24       # extend handle hit radius
    UNDO_MAX = 50

    def __init__(self, entries, job_id):
        """Build the whole window: the central polygon registry, all edit/drag
        state, the saved cursor position + filter, the full layout (banner,
        toolbar, image view, info strip, nav bar, action bar), keyboard
        shortcuts, and the restore of any previously saved marks/edits."""
        super().__init__()
        self.entries = entries
        self.job_id = job_id

        # ── Source-of-truth polygon registry (FULL-FRAME coords) ───────────
        # status: "original" / "edited" / "added"
        self.polygons_by_id = {}
        self._frame_to_sids = {}  # {frame_idx: set(server_id)}
        self._build_polygons_by_id()

        # ── Edit state ─────────────────────────────────────────────────────
        self.selected_id = None
        self._drag_polygon_id = None
        self._drag_vertex_idx = None
        self._drag_full_idx = None
        self._drag_active = False
        self._drag_undo_snapshot = None
        # Extend-handle drag state
        self._extend_active = False
        self._extend_polygon_id = None
        self._extend_indices = None
        self._extend_axis = None
        self._extend_last_full = None
        self._extend_opposite_indices = None
        # Add-polygon-mode state
        self.add_mode = False
        self.add_pts_local = []     # crop-local coords of in-progress polygon
        self.add_hover_xy = None
        self.next_temp_id = -1
        # Hover affordance state
        self.hovered_handle = None
        # Undo/redo stacks (latest at end; cap at UNDO_MAX)
        self.undo_stack = []
        self.redo_stack = []
        # Marks (delete set) - mirrors entries[i]["delete"] for selection-based ops
        self.marked_ids = {e["server_id"] for e in entries if e.get("delete")}
        self._img_cache = {}
        self._last_displayed_idx = None
        self._weirdr_anim_timer = None
        self._weirdr_anim_step = 0

        saved_state = self._load_saved_state()
        self.filter_mode = saved_state.get("filter_mode", "all")
        self.current_idx = self._restore_position()
        indices = self.filtered_indices()
        if indices and self.current_idx not in indices:
            self.current_idx = indices[0]
        self.brightness = 1.0
        self.contrast = 1.0

        self.setWindowTitle("TrailFixR")
        self.setMinimumSize(1100, 900)

        # --- Build the full layout ---
        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_banner())
        root.addWidget(self._build_toolbar())

        # Image viewer (the star of the show)
        self.image_view = ZoomableImageView()
        self.image_view.setStyleSheet(f"background: {IMAGE_BG}; border: none;")
        self.image_view.clicked_at_image_xy.connect(self._on_image_clicked)
        self.image_view.pressed_at_image_xy.connect(self._on_image_pressed)
        self.image_view.moved_to_image_xy.connect(self._on_image_moved)
        self.image_view.released_at_image_xy.connect(self._on_image_released)
        self.image_view.double_clicked_at_image_xy.connect(self._on_image_double_clicked)
        self.image_view.right_clicked_at_image_xy.connect(self._on_image_right_clicked)
        self.image_view.hovered_at_image_xy.connect(self._on_image_hovered)
        self.image_view.zoom_changed.connect(self.refresh_view)
        root.addWidget(self.image_view, stretch=1)

        self.image_view.installEventFilter(self)

        root.addWidget(self._build_info_strip())
        root.addWidget(self._build_nav_bar())
        root.addWidget(self._build_action_bar())

        # Keyboard shortcuts
        shortcuts = [
            (QKeySequence(Qt.Key_Left),       self.go_prev),
            (QKeySequence(Qt.Key_Right),      self.go_next),
            (QKeySequence(Qt.Key_Space),      self._toggle_delete),
            (QKeySequence(Qt.Key_D),          self._toggle_delete),
            (QKeySequence(Qt.Key_Delete),     self._toggle_delete),
            (QKeySequence(Qt.Key_Backspace),  self._toggle_delete),
            (QKeySequence(Qt.Key_C),          self._copy_reference),
            (QKeySequence(Qt.Key_A),          self._toggle_add_mode),
            (QKeySequence(Qt.Key_T),          self._auto_tighten),
            (QKeySequence(Qt.Key_Escape),     self._on_escape),
            (QKeySequence(Qt.Key_Return),     self._close_add_polygon),
            (QKeySequence(Qt.Key_Enter),      self._close_add_polygon),
            (QKeySequence(Qt.Key_Tab),        self._tab_within_frame),
            (QKeySequence("Shift+Tab"),       self._shift_tab_within_frame),
            (QKeySequence.Undo,               self._undo),
            (QKeySequence.Redo,               self._redo),
        ]
        for seq, fn in shortcuts:
            sc = QShortcut(seq, self, fn)
            sc.setContext(Qt.ApplicationShortcut)

        # Make all labels selectable
        for lbl in self.findChildren(QLabel):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Default selection: the current entry's polygon, so a fresh user can
        # immediately drag handles without first clicking the trail.
        if self.entries and 0 <= self.current_idx < len(self.entries):
            self.selected_id = self.entries[self.current_idx]["server_id"]

        # Restore any saved edits / adds / undo from disk
        self._load_edits()

        self.max_visited = 0
        self.refresh_view()
        self._refresh_send_button()
        self._refresh_undo_button()

    # ── polygons_by_id (full-frame source of truth) ────────────────────────

    def _build_polygons_by_id(self):
        """Walk all entries and build the central polygon registry. Each entry
        stores its main polygon in CROP-LOCAL coords + crop_x/crop_y offsets;
        we add those back to recover full-frame coords."""
        for e in self.entries:
            sid = e["server_id"]
            if sid not in self.polygons_by_id:
                ff = e["poly_pts"].astype(np.float32).copy()
                ff[:, 0] += e["crop_x"]
                ff[:, 1] += e["crop_y"]
                self.polygons_by_id[sid] = {
                    "server_id": sid,
                    "frame_idx": e["frame_idx"],
                    "points": ff,
                    "original_points": ff.copy(),
                    "status": "original",
                    "label": "trail",
                }
                self._frame_to_sids.setdefault(e["frame_idx"], set()).add(sid)

    # ── Banner ──────────────────────────────────────────────────────────────

    def _build_banner(self):
        """Top dark banner: title, the polygon-count + dataset subtitle, the
        live "Send N changes to CVAT" button, a Relaunch button, and a close
        button. The Send button lives here so it's always visible while
        editing."""
        banner = QWidget()
        banner.setFixedHeight(64)
        banner.setStyleSheet("background-color: #0a1e3f;")
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        # Title
        title = QLabel("TrailFixR")
        title.setStyleSheet(
            "color: white; font-size: 22px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(title)

        # Subtitle: polygon count + dataset
        n = len(self.entries)
        sub = QLabel(f"{n} polygons  |  {TASK_NAME} (frames {FRAME_START}-{FRAME_END - 1})")
        sub.setStyleSheet(
            "color: #a8c0e0; font-size: 12px; background: transparent;"
        )
        layout.addWidget(sub)

        layout.addStretch()

        # Send-to-CVAT button (live counter) - lives on banner so it's
        # always visible while editing.
        self.send_btn = QPushButton("Send 0 changes to CVAT")
        self.send_btn.setFixedHeight(34)
        self.send_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 12px; "
            "font-weight: bold; border-radius: 17px; border: none; padding: 0 16px; }"
            "QPushButton:hover { background-color: #a93226; }"
            "QPushButton:disabled { background-color: #6a6a6a; color: #ddd; }"
        )
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._send_to_cvat)
        layout.addWidget(self.send_btn)

        # Relaunch button
        restart_btn = QPushButton("Relaunch")
        restart_btn.setFixedHeight(30)
        restart_btn.setStyleSheet(
            "QPushButton { background-color: #d0e4f5; color: #1a3a5c; font-size: 12px; "
            "font-weight: bold; border-radius: 15px; border: 1px solid #a0c4e0; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #b8d4ec; }"
        )
        restart_btn.setToolTip("Close and relaunch TrailFixR")
        restart_btn.clicked.connect(self._relaunch)
        layout.addWidget(restart_btn)

        # Close button
        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #d93025; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #b8271b; }"
        )
        close_btn.setToolTip("Quit TrailFixR")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        return banner

    # ── Toolbar (filters + brightness) ──────────────────────────────────────

    def _build_toolbar(self):
        """Two-row control strip under the banner. Row 1: the Show filter pills
        (All / Flagged / Not Flagged / Marked / Not Marked) plus the edit-mode
        buttons (Add, Auto-Tighten, Undo, Redo). Row 2 left: Brightness and
        Contrast sliders (display-only). Row 2 right: the scrubber, "N of M"
        label, and a jump-to-position box."""
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {PANEL_BG};")
        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        # Row 1: Filter pills
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        n_flagged = sum(1 for e in self.entries if e["flagged"])

        self.filter_buttons = {}
        self.filter_group = QButtonGroup(self)
        self.filter_group.setExclusive(True)

        show_label = QLabel("Show:")
        show_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;")
        row1.addWidget(show_label)

        filters = [
            ("all",         f"All ({len(self.entries)})",         "blue"),
            ("flagged",     f"Flagged ({n_flagged})",             "orange"),
            ("not_flagged", f"Not Flagged ({len(self.entries) - n_flagged})", "green"),
            ("delete",      "Marked (0)",                         "red"),
            ("not_marked",  f"Not Marked ({len(self.entries)})",  "green"),
        ]
        for mode, text, color in filters:
            pill = make_pill(text, color)
            if mode == self.filter_mode:
                pill.setChecked(True)
            self.filter_group.addButton(pill)
            self.filter_buttons[mode] = pill
            pill.clicked.connect(lambda checked, m=mode: self.set_filter(m))
            row1.addWidget(pill)

        row1.addStretch()

        # Edit-mode controls (right side of row 1)
        edit_btn_style = (
            "QPushButton { background: #e8f0fe; color: #1a5276; font-size: 12px; "
            "font-weight: bold; border: 1px solid #b0c4de; border-radius: 14px; "
            "padding: 6px 14px; min-width: 70px; }"
            "QPushButton:hover { background: #d0e0f0; }"
            "QPushButton:checked { background: #1a6fc4; color: white; "
            "border-color: #145da0; }"
            "QPushButton:disabled { color: #999; background: #f0f0f0; }"
        )

        self.add_btn = QPushButton("+ Add (A)")
        self.add_btn.setCheckable(True)
        self.add_btn.setStyleSheet(edit_btn_style)
        self.add_btn.setToolTip(
            "Toggle Add Polygon mode (A). Click 4 corners to make a polygon, "
            "or click the first vertex / press Enter to close earlier."
        )
        self.add_btn.clicked.connect(self._toggle_add_mode)
        row1.addWidget(self.add_btn)

        self.tighten_btn = QPushButton("Auto-Tighten (T)")
        self.tighten_btn.setStyleSheet(edit_btn_style)
        self.tighten_btn.setToolTip(
            "Snap the selected polygon to the bright trail core inside it. "
            "Press T as a shortcut."
        )
        self.tighten_btn.clicked.connect(self._auto_tighten)
        row1.addWidget(self.tighten_btn)


        self.undo_btn = QPushButton("Undo (0)")
        self.undo_btn.setStyleSheet(edit_btn_style)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Undo the last edit (Cmd+Z). Up to 50 steps.")
        self.undo_btn.clicked.connect(self._undo)
        row1.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo (0)")
        self.redo_btn.setStyleSheet(edit_btn_style)
        self.redo_btn.setEnabled(False)
        self.redo_btn.setToolTip("Redo the last undone edit (Cmd+Shift+Z).")
        self.redo_btn.clicked.connect(self._redo)
        row1.addWidget(self.redo_btn)

        layout.addLayout(row1)

        # Row 2: Brightness + Contrast on left half | Scrubber + frame info on right half
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        # ── LEFT HALF: Brightness + Contrast ──
        left_half = QHBoxLayout()
        left_half.setSpacing(8)

        bright_icon = ClickableLabel("Brightness")
        bright_icon.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;")
        bright_icon.setCursor(Qt.PointingHandCursor)
        bright_icon.setToolTip("Double-click to reset to 1.0x")
        bright_icon.doubleClicked.connect(lambda: self.bright_slider.setValue(10))
        left_half.addWidget(bright_icon)

        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setMinimum(10)
        self.bright_slider.setMaximum(40)
        self.bright_slider.setValue(10)
        self.bright_slider.setFixedWidth(180)
        self.bright_slider.setTickPosition(QSlider.TicksBelow)
        self.bright_slider.setTickInterval(10)
        self.bright_slider.valueChanged.connect(self.on_brightness_changed)
        left_half.addWidget(self.bright_slider)

        self.bright_value_label = QLabel("1.0x")
        self.bright_value_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        self.bright_value_label.setFixedWidth(36)
        left_half.addWidget(self.bright_value_label)

        left_half.addSpacing(16)

        contrast_label = ClickableLabel("Contrast")
        contrast_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;")
        contrast_label.setCursor(Qt.PointingHandCursor)
        contrast_label.setToolTip("Double-click to reset to 1.0x")
        contrast_label.doubleClicked.connect(lambda: self.contrast_slider.setValue(10))
        left_half.addWidget(contrast_label)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(40)
        self.contrast_slider.setValue(10)
        self.contrast_slider.setFixedWidth(180)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        left_half.addWidget(self.contrast_slider)

        self.contrast_value_label = QLabel("1.0x")
        self.contrast_value_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        self.contrast_value_label.setFixedWidth(36)
        left_half.addWidget(self.contrast_value_label)

        left_half.addStretch()  # pushes brightness/contrast to the left within their half

        row2.addLayout(left_half, stretch=1)

        # Separator between halves
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #ccc;")
        row2.addWidget(sep)

        # ── RIGHT HALF: Scrubber + frame info ──
        right_half = QHBoxLayout()
        right_half.setSpacing(8)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setMinimum(1)
        self.scrubber.setMaximum(max(1, len(self.entries)))
        self.scrubber.setValue(1)
        self.scrubber.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px; background: #b0c4de; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1a6fc4; border: 2px solid #145da0;
                width: 16px; height: 16px; margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1580e0;
            }
            QSlider::sub-page:horizontal {
                background: #1a6fc4; border-radius: 3px;
            }
        """)
        self.scrubber.valueChanged.connect(self._on_scrubber_changed)
        right_half.addWidget(self.scrubber, stretch=1)

        self.scrubber_label = QLabel("1 of 902")
        self.scrubber_label.setStyleSheet(
            f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;"
        )
        self.scrubber_label.setFixedWidth(80)
        right_half.addWidget(self.scrubber_label)

        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(56)
        self.jump_input.setFixedHeight(26)
        self.jump_input.setText("1")
        self.jump_input.setAlignment(Qt.AlignCenter)
        self.jump_input.setStyleSheet(
            "QLineEdit { font-size: 13px; font-weight: bold; border: 1px solid #b0c4de; "
            "border-radius: 4px; padding: 2px; background: white; }"
            "QLineEdit:focus { border-color: #1a6fc4; }"
        )
        self.jump_input.returnPressed.connect(self._on_jump_entered)
        right_half.addWidget(self.jump_input)

        # Magnifying glass icon (drawn via QPainter, no emoji)
        mag_pix = QPixmap(20, 20)
        mag_pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(mag_pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(MUTED_TEXT), 2.0)
        p.setPen(pen)
        p.drawEllipse(3, 3, 10, 10)
        p.drawLine(12, 12, 17, 17)
        p.end()
        search_icon = QLabel()
        search_icon.setPixmap(mag_pix)
        search_icon.setFixedSize(20, 20)
        search_icon.setStyleSheet("background: transparent;")
        right_half.addWidget(search_icon)

        row2.addLayout(right_half, stretch=1)

        layout.addLayout(row2)

        return toolbar

    # ── Info strip (below image) ────────────────────────────────────────────

    def _build_info_strip(self):
        """The thin strip directly under the image. Top line: frame number,
        filename, polygon id, position-in-filter, and any [FLAGGED] /
        [MARKED FALSE POSITIVE] tags. Bottom line: the flag reason, if any."""
        strip = QWidget()
        strip.setFixedHeight(52)
        strip.setStyleSheet(f"background: {INFO_BG};")
        layout = QVBoxLayout(strip)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(0)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {INFO_TEXT}; background: transparent;"
        )
        layout.addWidget(self.info_label)

        self.reason_label = QLabel()
        self.reason_label.setAlignment(Qt.AlignCenter)
        self.reason_label.setStyleSheet(
            "font-size: 12px; color: #c0392b; background: transparent;"
        )
        layout.addWidget(self.reason_label)

        return strip

    # ── Reference helpers ──────────────────────────────────────────────────

    def _build_reference_string(self, entry):
        """Format the human-readable reference string for the current polygon."""
        return (
            f"TrailFixR | "
            f"frame {entry['frame_idx']} ({entry['frame']}) | "
            f"polygon #{entry['cvat_seq']} (cvat id {entry['server_id']}) | "
            f"job {self.job_id}\n"
            f"{entry['img_path']}"
        )

    def _copy_reference(self):
        """Copy the current polygon's reference string to the clipboard (so it
        can be pasted back to Claude) and briefly flash "Copied" on the button.
        Bound to the C key."""
        if not self.entries:
            return
        entry = self.entries[self.current_idx]
        ref = self._build_reference_string(entry)
        QApplication.clipboard().setText(ref)
        original = self.copy_btn.text()
        self.copy_btn.setText("Copied")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1200, lambda: self.copy_btn.setText(original))

    # ── Navigation bar (prev / delete / next) ──────────────────────────────

    def _build_nav_bar(self):
        """The main navigation row: a keyboard-shortcut hint on the left, then
        Prev / Copy reference / Pull from CVAT / Open in CVAT / Next buttons on
        the right. "Pull from CVAT" re-fetches the current frame's polygons so
        edits made in the CVAT web UI show up without a full relaunch."""
        bar = QWidget()
        bar.setStyleSheet(f"background: {PANEL_BG};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)


        # Shortcut hint (center)
        hint_left = QLabel(
            "← / → = prev/next   |   Tab / Shift-Tab = next/prev in same frame   |   "
            "Space, D, Del = delete   |   A = add   |   T = tighten   |"
            "Cmd+Z = undo")
        hint_left.setStyleSheet(f"color: {HINT_TEXT}; font-size: 11px; background: transparent;")
        hint_left.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_left, stretch=1)

        # Prev + CVAT + Next together (right side)
        nav_style = (
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 15px; "
            "font-weight: bold; border-radius: 6px; border: none; padding: 0 24px; }"
            "QPushButton:hover { background-color: #1580e0; }"
            "QPushButton:disabled { background-color: #999; }"
        )

        self.prev_btn = QPushButton("Prev")
        self.prev_btn.setFixedHeight(44)
        self.prev_btn.setStyleSheet(nav_style)
        self.prev_btn.clicked.connect(self.go_prev)
        layout.addWidget(self.prev_btn)

        # Copy reference (between Prev and CVAT)
        self.copy_btn = QPushButton("Copy reference")
        self.copy_btn.setFixedHeight(44)
        self.copy_btn.setStyleSheet(
            "QPushButton { background-color: #d0e4f5; color: #1a3a5c; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; border: 1px solid #a0c4e0; "
            "padding: 0 18px; }"
            "QPushButton:hover { background-color: #b8d4ec; }"
        )
        self.copy_btn.setToolTip(
            "Copy this polygon's reference to the clipboard so you can paste it "
            "back to Claude. Press C as a shortcut."
        )
        self.copy_btn.clicked.connect(self._copy_reference)
        layout.addWidget(self.copy_btn)

        # Pull from CVAT - re-fetch this single polygon's coords from CVAT
        # so changes made in the CVAT web UI show up here without a full
        # relaunch.
        self.pull_btn = QPushButton("Pull from CVAT")
        self.pull_btn.setFixedHeight(44)
        self.pull_btn.setStyleSheet(
            "QPushButton { background-color: #d39e00; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; border: none; padding: 0 18px; }"
            "QPushButton:hover { background-color: #b8860b; }"
            "QPushButton:disabled { background-color: #888; }"
        )
        self.pull_btn.setToolTip(
            "Pull this polygon's latest coordinates from CVAT. Use after "
            "editing the trail in the CVAT web UI to refresh just this one "
            "polygon without a full reload.")
        self.pull_btn.clicked.connect(self._pull_from_cvat)
        layout.addWidget(self.pull_btn)

        # Open in CVAT (between Prev and Next, different color)
        self.cvat_btn = QPushButton("Open in CVAT")
        self.cvat_btn.setFixedHeight(44)
        self.cvat_btn.setStyleSheet(
            "QPushButton { background-color: #2a7a2a; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; border: none; padding: 0 18px; }"
            "QPushButton:hover { background-color: #339933; }"
        )
        self.cvat_btn.clicked.connect(self.open_in_cvat)
        layout.addWidget(self.cvat_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedHeight(44)
        self.next_btn.setStyleSheet(nav_style)
        self.next_btn.clicked.connect(self.go_next)
        layout.addWidget(self.next_btn)

        return bar

    # ── Action bar (secondary actions at bottom) ────────────────────────────

    def _build_action_bar(self):
        """Bottom dark bar with the secondary batch actions: "Accept all N
        flags" (mark every Claude-flagged polygon for deletion), a second
        "Send changes to CVAT" button, and "Add To WeirdR" (tag the current
        polygon into the shared weirdr_list.json)."""
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet("background: #0a1e3f;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)

        # Accept All Flagged (left)
        n_flagged = sum(1 for e in self.entries if e["flagged"])
        self.accept_btn = QPushButton(f"Accept all {n_flagged} flags")
        self.accept_btn.setFixedHeight(32)
        self.accept_btn.setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white; font-size: 12px; "
            "font-weight: bold; border-radius: 16px; border: none; padding: 0 16px; }"
            "QPushButton:hover { background-color: #d35400; }"
        )
        self.accept_btn.clicked.connect(self.accept_all_flagged)
        layout.addWidget(self.accept_btn)

        # Send all changes to CVAT (left). Same handler as the banner button;
        # batches add + edit + delete into one push.
        self.save_btn = QPushButton("Send changes to CVAT")
        self.save_btn.setFixedHeight(32)
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 12px; "
            "font-weight: bold; border-radius: 16px; border: none; padding: 0 16px; }"
            "QPushButton:hover { background-color: #a93226; }"
        )
        self.save_btn.setToolTip(
            "Send all pending adds, edits, and deletes to CVAT in one batch.")
        self.save_btn.clicked.connect(self.save_deletions)
        layout.addWidget(self.save_btn)

        self.weirdr_btn = QPushButton("Add To WeirdR")
        self.weirdr_btn.setFixedHeight(32)
        self.weirdr_btn.setStyleSheet(
            "QPushButton { background-color: #ede0f5; color: #4a1a6a; font-size: 12px; "
            "font-weight: bold; border-radius: 16px; border: none; padding: 0 16px; }"
            "QPushButton:hover { background-color: #ddd0f0; }"
        )
        self.weirdr_btn.clicked.connect(self._add_to_weirdr)
        layout.addWidget(self.weirdr_btn)

        layout.addStretch()

        return bar

    def _add_to_weirdr(self):
        """Append the current polygon (filename, dataset, server_id, job_id,
        date) to the shared weirdr_list.json at the repo root, for collecting
        odd/interesting cases. Skips duplicates and animates the button. The
        list is keyed by a "<frame> #<server_id>" tag."""
        if not self.entries or self.current_idx >= len(self.entries):
            return
        entry = self.entries[self.current_idx]
        tag = f"{entry['frame']} #{entry['server_id']}"
        try:
            weirdr = json.loads(WEIRDR_PATH.read_text()) if WEIRDR_PATH.exists() else []
        except Exception:
            weirdr = []
        already = any(e.get("tag") == tag for e in weirdr)
        if not already:
            weirdr.append({
                "source": "trail_fixr",
                "tag": tag,
                "filename": Path(entry["img_path"]).name,
                "dataset": TASK_NAME,
                "server_id": entry["server_id"],
                "job_id": self.job_id,
                "reason": "",
                "added": time.strftime("%Y-%m-%d"),
            })
            WEIRDR_PATH.write_text(json.dumps(weirdr, indent=2))
        if already:
            self.weirdr_btn.setText("Already listed")
            QTimer.singleShot(1500, lambda: self.weirdr_btn.setText("Add To WeirdR"))
        else:
            self.weirdr_btn.setEnabled(False)
            self._weirdr_anim_step = 0
            self._weirdr_anim_timer = QTimer(self)
            self._weirdr_anim_timer.setInterval(200)
            self._weirdr_anim_timer.timeout.connect(self._tick_weirdr_animation)
            self._weirdr_anim_timer.start()
            self._tick_weirdr_animation()

    def _tick_weirdr_animation(self):
        """Step the "Adding... / Added!" button animation one frame, then stop
        and re-enable the button when it reaches the end."""
        labels = ["Adding.", "Adding..", "Adding...", "Added!"]
        step = self._weirdr_anim_step
        self.weirdr_btn.setText(labels[min(step, len(labels) - 1)])
        self._weirdr_anim_step += 1
        if step >= len(labels) - 1:
            self._weirdr_anim_timer.stop()
            self._weirdr_anim_timer = None
            QTimer.singleShot(800, lambda: self.weirdr_btn.setEnabled(True))

    # ── Relaunch ────────────────────────────────────────────────────────────

    def _relaunch(self):
        """Spawn a fresh copy of TrailFixR and quit this one (brings you back to
        the task picker)."""
        subprocess.Popen([sys.executable, os.path.abspath(__file__)])
        self.close()
        QApplication.quit()

    # ── Position persistence ────────────────────────────────────────────────

    def _restore_position(self):
        """Read saved state and return the starting index.

        If the saved polygon still exists in CVAT, jump to it. If it was
        deleted, jump to the entry immediately before it (saved_index - 1).
        Falls back to 0 if nothing usable is available.
        """
        if not self.entries:
            return 0
        state = self._load_saved_state()
        if not state:
            return 0
        saved_sid = state.get("server_id")
        saved_idx = state.get("index", 0)
        if saved_sid is not None:
            for i, e in enumerate(self.entries):
                if e["server_id"] == saved_sid:
                    return i
        fallback = saved_idx - 1
        if fallback < 0:
            fallback = 0
        if fallback >= len(self.entries):
            fallback = len(self.entries) - 1
        return fallback

    def _load_saved_state(self):
        """Read the per-task state.json (saved cursor server_id, index, and
        filter mode), or {} if it's missing or unreadable."""
        try:
            if STATE_PATH.exists():
                return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
        return {}

    def _save_position(self):
        """Write the current cursor (polygon server_id, index, filter mode) to
        the per-task state.json so the next launch reopens here."""
        if not self.entries or self.current_idx >= len(self.entries):
            return
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps({
                "server_id": self.entries[self.current_idx]["server_id"],
                "index": self.current_idx,
                "filter_mode": self.filter_mode,
            }, indent=2))
        except Exception:
            pass

    def closeEvent(self, event):
        """On window close, persist the cursor position and any pending edits
        before quitting."""
        self._save_position()
        self._save_edits()
        super().closeEvent(event)

    # ── Edit primitives (lifted from TileFixR) ─────────────────────────────

    def _hit_r(self, base):
        """Scale a hit radius by inverse zoom so it stays roughly constant
        in screen pixels. Floor at 6 image pixels."""
        try:
            scale = max(0.05, self.image_view.current_scale())
        except Exception:
            scale = 1.0
        return max(6.0, base / scale)

    def _compute_extend_handles(self, ff_pts):
        """PCA-based handle compute. Returns list of dicts with kind
        ('length' on short sides, 'width' on long sides), center_full,
        vertex_indices to move together, and unit-axis vector."""
        if ff_pts is None or len(ff_pts) < 3:
            return []
        pts = ff_pts.astype(np.float32)
        mu = pts.mean(axis=0)
        centered = pts - mu
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return []
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        principal = evecs[:, 0]
        secondary = evecs[:, 1]
        n1 = float(np.hypot(principal[0], principal[1]))
        n2 = float(np.hypot(secondary[0], secondary[1]))
        if n1 < 1e-6 or n2 < 1e-6:
            return []
        principal = principal / n1
        secondary = secondary / n2

        def _make(axis_vec, kind):
            proj = centered @ axis_vec
            neg_idx = [i for i, p in enumerate(proj) if p < 0]
            pos_idx = [i for i, p in enumerate(proj) if p >= 0]
            out = []
            for indices in (neg_idx, pos_idx):
                if not indices:
                    continue
                grp = pts[indices]
                center = grp.mean(axis=0)
                out.append({
                    "kind": kind,
                    "center_full": (float(center[0]), float(center[1])),
                    "vertex_indices": list(indices),
                    "axis": axis_vec.copy(),
                })
            return out

        return _make(principal, "length") + _make(secondary, "width")

    def _visible_polys_for_entry(self, entry):
        """Return list of (server_id, crop_local_pts as int32) for every
        polygon whose bbox overlaps this entry's crop window. Includes the
        entry's main polygon. Ordered with the main polygon first."""
        out = []
        cx = entry["crop_x"]
        cy = entry["crop_y"]
        cw = entry["crop_raw"].shape[1]
        ch = entry["crop_raw"].shape[0]
        main_sid = entry["server_id"]
        # Look up every polygon in the same frame
        for sid in self._frame_to_sids.get(entry["frame_idx"], ()):
            poly = self.polygons_by_id.get(sid)
            if poly is None or poly["points"].shape[0] < 3:
                continue
            ff = poly["points"]
            x_min, y_min = ff[:, 0].min(), ff[:, 1].min()
            x_max, y_max = ff[:, 0].max(), ff[:, 1].max()
            # bbox overlap test against the crop rect
            if x_max < cx or x_min > cx + cw or y_max < cy or y_min > cy + ch:
                continue
            local = ff.copy()
            local[:, 0] -= cx
            local[:, 1] -= cy
            local_int = local.astype(np.int32)
            out.append((sid, local_int))
        # Make sure the main polygon is first in the list (for stable z-order)
        out.sort(key=lambda x: 0 if x[0] == main_sid else 1)
        return out

    def _nearest_extend_handle(self, ix, iy):
        """Nearest extend handle for the SELECTED polygon. Returns
        (server_id, handle, distance_pixels) or (None, None, None)."""
        if self.selected_id is None or not self.entries:
            return (None, None, None)
        poly = self.polygons_by_id.get(self.selected_id)
        if poly is None:
            return (None, None, None)
        entry = self.entries[self.current_idx]
        best = None
        best_dist2 = float("inf")
        for h in self._compute_extend_handles(poly["points"]):
            cx_full, cy_full = h["center_full"]
            cx = cx_full - entry["crop_x"]
            cy = cy_full - entry["crop_y"]
            d2 = (cx - ix) ** 2 + (cy - iy) ** 2
            if d2 < best_dist2:
                best = h
                best_dist2 = d2
        if best is None:
            return (None, None, None)
        return (self.selected_id, best, best_dist2 ** 0.5)

    def _extend_handle_under(self, ix, iy):
        """Return (server_id, handle) if an extend-handle is within hit range of
        (ix, iy), else (None, None)."""
        sid, h, dist = self._nearest_extend_handle(ix, iy)
        if sid is None or dist > self._hit_r(self.EXTEND_HIT_R):
            return (None, None)
        return (sid, h)

    def _nearest_vertex(self, ix, iy):
        """Nearest polygon vertex among ALL visible polygons in this entry's
        crop. Returns (server_id, vertex_idx_in_full_frame, distance)."""
        if not self.entries:
            return (None, None, None)
        entry = self.entries[self.current_idx]
        best = None
        best_dist2 = float("inf")
        for sid, local_pts in self._visible_polys_for_entry(entry):
            for vi, (vx, vy) in enumerate(local_pts):
                d2 = (vx - ix) ** 2 + (vy - iy) ** 2
                if d2 < best_dist2:
                    # vi here is the index into the LOCAL polygon (which is
                    # the same as the index into the full-frame polygon since
                    # we don't reorder). Save it directly.
                    best = (sid, vi)
                    best_dist2 = d2
        if best is None:
            return (None, None, None)
        sid, vi = best
        return (sid, vi, best_dist2 ** 0.5)

    def _vertex_under_any(self, ix, iy):
        """Return (server_id, vertex_index) if a polygon vertex is within hit
        range of (ix, iy), else (None, None)."""
        sid, vi, dist = self._nearest_vertex(ix, iy)
        if sid is None or dist > self._hit_r(self.HANDLE_RADIUS):
            return (None, None)
        return (sid, vi)

    # ── Drag state machine ─────────────────────────────────────────────────

    def _on_image_pressed(self, ix, iy):
        """Decide what a left-press starts. In add-mode it does nothing (the
        click handler appends a vertex). Otherwise it finds the nearest extend-
        handle and the nearest vertex and, closest-wins, begins either an
        extend-handle drag (the grabbed side's two vertices move together;
        width handles also push the opposite side out symmetrically) or a
        single-vertex drag. If neither is in range, it leaves the view in pan
        mode. A vertex grab also auto-selects that polygon. A snapshot of the
        polygon is stashed for undo."""
        if self.add_mode:
            self.image_view.setDragMode(QGraphicsView.NoDrag)
            return
        # Closest-handle-wins: prefer extend if both in range and closer
        ext_sid, ext_h, ext_dist = self._nearest_extend_handle(ix, iy)
        n_sid, n_vi, n_dist = self._nearest_vertex(ix, iy)
        ext_hit_r = self._hit_r(self.EXTEND_HIT_R)
        vert_hit_r = self._hit_r(self.HANDLE_RADIUS)
        ext_in = ext_sid is not None and ext_dist <= ext_hit_r
        vert_in = n_sid is not None and n_dist <= vert_hit_r
        prefer_extend = ext_in and (not vert_in or ext_dist <= n_dist)

        if prefer_extend:
            poly = self.polygons_by_id.get(ext_sid)
            if poly is not None:
                self.image_view.setDragMode(QGraphicsView.NoDrag)
                self.image_view.viewport().setCursor(Qt.ClosedHandCursor)
                entry = self.entries[self.current_idx]
                self._extend_active = True
                self._extend_polygon_id = ext_sid
                self._extend_indices = ext_h["vertex_indices"]
                self._extend_axis = ext_h["axis"]
                self._extend_last_full = (
                    float(ix + entry["crop_x"]),
                    float(iy + entry["crop_y"]),
                )
                # Mirror opposite side only for width handles (blue), not length (green)
                grabbed = set(ext_h["vertex_indices"])
                self._extend_opposite_indices = None
                if ext_h["kind"] == "width":
                    for h in self._compute_extend_handles(poly["points"]):
                        if h["kind"] == "width" and set(h["vertex_indices"]) != grabbed:
                            self._extend_opposite_indices = h["vertex_indices"]
                            break
                self._drag_undo_snapshot = (
                    poly["points"].copy(), poly["status"])
                self.refresh_view()
                return
        # else fall through to vertex grab
        if n_sid is None or n_dist > vert_hit_r:
            self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
            return
        # Auto-select on press: select polygon containing the nearest vertex
        # AND start dragging that vertex in the same gesture.
        if n_sid != self.selected_id:
            self.selected_id = n_sid
        self.image_view.setDragMode(QGraphicsView.NoDrag)
        self.image_view.viewport().setCursor(Qt.ClosedHandCursor)
        self._drag_polygon_id = n_sid
        self._drag_vertex_idx = n_vi
        self._drag_active = True
        self._drag_full_idx = None
        poly = self.polygons_by_id.get(n_sid)
        if poly is not None:
            self._drag_undo_snapshot = (poly["points"].copy(), poly["status"])
        else:
            self._drag_undo_snapshot = None
        self.refresh_view()

    def _on_image_moved(self, ix, iy):
        """Live-update the polygon as the cursor moves during a drag. For an
        extend-handle drag, the grabbed side's vertices follow the cursor (and
        a width handle's opposite side mirrors); for a vertex drag, the single
        nearest vertex follows. Either way the polygon flips to "edited" and the
        view redraws. Does nothing if no drag is active."""
        # Extend-handle drag: the grabbed side's 2 vertices follow the cursor
        # freely in 2D. Opposite side stays anchored. So grabbing the right
        # tip and dragging up moves both right-side vertices up together
        # (top-right and bottom-right); the left tip's two vertices don't
        # move.
        if self._extend_active:
            entry = self.entries[self.current_idx]
            poly = self.polygons_by_id.get(self._extend_polygon_id)
            if poly is None:
                return
            cur_full = (float(ix + entry["crop_x"]),
                        float(iy + entry["crop_y"]))
            dx = cur_full[0] - self._extend_last_full[0]
            dy = cur_full[1] - self._extend_last_full[1]
            for vi in self._extend_indices:
                poly["points"][vi][0] += dx
                poly["points"][vi][1] += dy
            if self._extend_opposite_indices:
                for vi in self._extend_opposite_indices:
                    poly["points"][vi][0] -= dx
                    poly["points"][vi][1] -= dy
            if poly["status"] == "original":
                poly["status"] = "edited"
            self._extend_last_full = cur_full
            self.refresh_view()
            return
        if not self._drag_active:
            return
        entry = self.entries[self.current_idx]
        poly = self.polygons_by_id.get(self._drag_polygon_id)
        if poly is None:
            return
        fx = ix + entry["crop_x"]
        fy = iy + entry["crop_y"]
        if not hasattr(self, "_drag_full_idx") or self._drag_full_idx is None:
            ff_pts = poly["points"]
            distances = (ff_pts[:, 0] - fx) ** 2 + (ff_pts[:, 1] - fy) ** 2
            self._drag_full_idx = int(np.argmin(distances))
        poly["points"][self._drag_full_idx] = [fx, fy]
        if poly["status"] == "original":
            poly["status"] = "edited"
        self.refresh_view()

    def _on_image_released(self, ix, iy):
        """Finish a drag on left release: push the pre-drag snapshot onto the
        undo stack, clear all drag/extend state, restore pan mode and the
        cursor, then auto-save the edit and refresh the Send counter."""
        # Extend-handle drag finishing
        if self._extend_active:
            sid = self._extend_polygon_id
            if self._drag_undo_snapshot is not None:
                pts_before, status_before = self._drag_undo_snapshot
                self._push_undo(("vertex", sid, pts_before, status_before))
            self._drag_undo_snapshot = None
            self._extend_active = False
            self._extend_polygon_id = None
            self._extend_indices = None
            self._extend_axis = None
            self._extend_last_full = None
            self._extend_opposite_indices = None
            if not self.add_mode:
                self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
                self.image_view.viewport().unsetCursor()
            self._save_edits()
            self._refresh_send_button()
            self.refresh_view()
            return
        was_dragging = self._drag_active
        if (was_dragging and self._is_drag_committed()
                and self._drag_undo_snapshot is not None):
            sid_for_undo = self._drag_polygon_id
            pts_before, status_before = self._drag_undo_snapshot
            self._push_undo(("vertex", sid_for_undo, pts_before, status_before))
        self._drag_undo_snapshot = None
        if self._drag_active:
            self._drag_active = False
            self._drag_polygon_id = None
            self._drag_vertex_idx = None
            self._drag_full_idx = None
            self._save_edits()
            self._refresh_send_button()
            self.refresh_view()
        if not self.add_mode:
            self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.image_view.viewport().unsetCursor()

    def _is_drag_committed(self):
        """True if a vertex drag actually moved a vertex (so it's worth pushing
        an undo entry); False for a press that never moved one."""
        return getattr(self, "_drag_full_idx", None) is not None

    def _on_image_double_clicked(self, ix, iy):
        """In add-mode, a double-click closes the in-progress polygon."""
        if self.add_mode:
            self._close_add_polygon()

    def _on_image_right_clicked(self, ix, iy):
        """In add-mode, a right-click undoes (pops) the last placed vertex."""
        if self.add_mode and self.add_pts_local:
            self.add_pts_local.pop()
            self.refresh_view()

    def _on_image_hovered(self, ix, iy):
        """Update cursor and handle-highlight as the mouse moves with no button
        held: show the rubber-band line in add-mode; otherwise highlight the
        nearest in-range extend-handle (resize cursor) or vertex (move cursor),
        closest-wins, and redraw only when the hovered handle changes."""
        if self.add_mode:
            if self.add_pts_local:
                self.add_hover_xy = (ix, iy)
                self.refresh_view()
            return
        if self._drag_active or self._extend_active:
            return
        viewport = self.image_view.viewport()
        ext_sid, ext_h, ext_dist = self._nearest_extend_handle(ix, iy)
        n_sid, n_vi, n_dist = self._nearest_vertex(ix, iy)
        ext_hit_r = self._hit_r(self.EXTEND_HIT_R)
        vert_hit_r = self._hit_r(self.HANDLE_RADIUS)
        ext_in = ext_sid is not None and ext_dist <= ext_hit_r
        vert_in = n_sid is not None and n_dist <= vert_hit_r
        prefer_extend = ext_in and (not vert_in or ext_dist <= n_dist)

        new_hover = None
        if prefer_extend:
            handles = self._compute_extend_handles(
                self.polygons_by_id[ext_sid]["points"])
            try:
                end_idx = handles.index(ext_h)
            except ValueError:
                end_idx = 0
            new_hover = ("extend", ext_sid, ext_h.get("kind", "length"), end_idx)
            viewport.setCursor(Qt.SizeHorCursor if ext_h.get("kind") == "length"
                                else Qt.SizeVerCursor)
        elif vert_in:
            new_hover = ("vertex", n_sid, n_vi)
            viewport.setCursor(Qt.SizeAllCursor)
        else:
            viewport.unsetCursor()

        if new_hover != self.hovered_handle:
            self.hovered_handle = new_hover
            self.refresh_view()

    def leaveEvent(self, event):
        """Clear any hover highlight when the mouse leaves the window."""
        if self.hovered_handle is not None:
            self.hovered_handle = None
            self.refresh_view()
        super().leaveEvent(event)

    # ── Add-polygon mode ───────────────────────────────────────────────────

    def _toggle_add_mode(self):
        """Turn add-polygon mode on or off (the A key / Add button). Entering
        sets a crosshair cursor and clears the selection; leaving discards any
        half-drawn polygon. Keeps the Add button's checked state in sync."""
        self.add_mode = not self.add_mode
        if hasattr(self, "add_btn"):
            self.add_btn.blockSignals(True)
            self.add_btn.setChecked(self.add_mode)
            self.add_btn.blockSignals(False)
        if self.add_mode:
            self.image_view.viewport().setCursor(Qt.CrossCursor)
            self.selected_id = None
        else:
            self.image_view.viewport().unsetCursor()
            self.add_pts_local = []
        self.add_hover_xy = None
        self.refresh_view()

    def _close_add_polygon(self):
        """Finish the in-progress add-mode polygon: require >=3 vertices,
        convert the crop-local points to full-frame, register it as a new
        "added" polygon with a negative temp id, push an undo entry, select it,
        leave add-mode, and auto-save. The negative id is replaced by a real
        CVAT id once the add is pushed on the next Send."""
        if not self.add_mode or not self.entries:
            return
        if len(self.add_pts_local) < 3:
            QMessageBox.information(self, "Need 3+ vertices",
                "A polygon needs at least 3 vertices. Click more, "
                "or press Esc to cancel.")
            return
        entry = self.entries[self.current_idx]
        ff_pts = np.array([(x + entry["crop_x"], y + entry["crop_y"])
                            for x, y in self.add_pts_local], dtype=np.float32)
        temp_id = self.next_temp_id
        self.next_temp_id -= 1
        self.polygons_by_id[temp_id] = {
            "server_id": temp_id,
            "frame_idx": entry["frame_idx"],
            "points": ff_pts,
            "original_points": ff_pts.copy(),
            "status": "added",
            "label": "trail",
        }
        self._frame_to_sids.setdefault(entry["frame_idx"], set()).add(temp_id)
        self._push_undo(("add", temp_id))
        self.add_pts_local = []
        self.selected_id = temp_id
        self._toggle_add_mode()  # turn off and refresh
        self._save_edits()
        self._refresh_send_button()

    def _on_escape(self):
        """Handle the Escape key: first press clears a half-drawn add-mode
        polygon; if add-mode is on with nothing drawn, it leaves add-mode;
        otherwise it just clears the current selection."""
        if self.add_mode and self.add_pts_local:
            self.add_pts_local = []
            self.refresh_view()
            return
        if self.add_mode:
            self._toggle_add_mode()
            return
        # Clear selection
        self.selected_id = None
        self.refresh_view()

    # ── Undo / redo ────────────────────────────────────────────────────────

    def _push_undo(self, op):
        """Record one reversible operation on the undo stack (capped at
        UNDO_MAX), clear the redo stack since a new action invalidates it, and
        refresh the Undo/Redo button labels."""
        self.undo_stack.append(op)
        if len(self.undo_stack) > self.UNDO_MAX:
            self.undo_stack = self.undo_stack[-self.UNDO_MAX:]
        # New action invalidates redo stack
        self.redo_stack.clear()
        self._refresh_undo_button()

    def _refresh_undo_button(self):
        """Update the Undo/Redo button labels with their stack depths and
        enable/disable each based on whether its stack has anything in it."""
        if hasattr(self, "undo_btn"):
            n = len(self.undo_stack)
            self.undo_btn.setText(f"Undo ({n})")
            self.undo_btn.setEnabled(n > 0)
        if hasattr(self, "redo_btn"):
            n = len(self.redo_stack)
            self.redo_btn.setText(f"Redo ({n})")
            self.redo_btn.setEnabled(n > 0)

    def _do_inverse_op(self, op, dest_stack):
        """Apply the inverse of op to state; push the inverse-op onto
        dest_stack so it can be undone again."""
        kind = op[0]
        if kind == "vertex":
            sid, pts_before, status_before = op[1], op[2], op[3]
            poly = self.polygons_by_id.get(sid)
            if poly is not None:
                pts_now = poly["points"].copy()
                status_now = poly["status"]
                poly["points"] = pts_before
                poly["status"] = status_before
                dest_stack.append(("vertex", sid, pts_now, status_now))
        elif kind == "add":
            temp_id = op[1]
            poly = self.polygons_by_id.pop(temp_id, None)
            for sids in self._frame_to_sids.values():
                sids.discard(temp_id)
            if self.selected_id == temp_id:
                self.selected_id = None
            if poly is not None:
                dest_stack.append(("re_add", temp_id, poly))
        elif kind == "re_add":
            # Reinstate a previously undone add
            temp_id, poly = op[1], op[2]
            self.polygons_by_id[temp_id] = poly
            self._frame_to_sids.setdefault(poly["frame_idx"], set()).add(temp_id)
            self.selected_id = temp_id
            dest_stack.append(("add", temp_id))
        elif kind == "mark":
            sid = op[1]
            self.marked_ids.discard(sid)
            self._sync_entry_delete_flag(sid, False)
            dest_stack.append(("unmark", sid))
        elif kind == "unmark":
            sid = op[1]
            self.marked_ids.add(sid)
            self._sync_entry_delete_flag(sid, True)
            dest_stack.append(("mark", sid))

    def _undo(self):
        """Pop the last operation off the undo stack, apply its inverse (which
        pushes a redo entry), then refresh buttons, the Send counter, filter
        counts, saved edits, and the view."""
        if not self.undo_stack:
            return
        op = self.undo_stack.pop()
        self._do_inverse_op(op, self.redo_stack)
        self._refresh_undo_button()
        self._refresh_send_button()
        self._save_edits()
        self.update_filter_counts()
        self.refresh_view()

    def _redo(self):
        """Pop the last undone operation off the redo stack, re-apply it (which
        pushes it back onto the undo stack), then refresh buttons, the Send
        counter, filter counts, saved edits, and the view."""
        if not self.redo_stack:
            return
        op = self.redo_stack.pop()
        self._do_inverse_op(op, self.undo_stack)
        self._refresh_undo_button()
        self._refresh_send_button()
        self._save_edits()
        self.update_filter_counts()
        self.refresh_view()

    def _sync_entry_delete_flag(self, sid, value):
        """Keep entries[i]['delete'] in sync with marked_ids for all entries
        whose main polygon is sid."""
        for e in self.entries:
            if e["server_id"] == sid:
                e["delete"] = bool(value)

    # ── Display helpers ─────────────────────────────────────────────────────

    def build_display_crop(self, entry):
        """Render the fully-decorated crop image for one entry: apply the
        brightness/contrast adjustment, then draw, in layered passes, the
        near-image-edge orange markers, every visible polygon in its base
        color, the red fill on marked-for-delete polygons, the yellow/lime
        edited/added outlines, the selected polygon's white outline + vertex
        handles, its diamond extend-handles, and any in-progress add-mode
        polygon. All overlay sizes are scaled by inverse zoom so they stay a
        constant thickness on screen. Returns the BGR image to display."""
        crop = entry["crop_raw"].copy()
        if self.brightness != 1.0 or self.contrast != 1.0:
            img = crop.astype(np.float32)
            if self.brightness != 1.0:
                img = img * self.brightness
            if self.contrast != 1.0:
                img = (img - 128.0) * self.contrast + 128.0
            crop = np.clip(img, 0, 255).astype(np.uint8)

        try:
            scale = max(0.05, self.image_view.current_scale())
        except Exception:
            scale = 1.0
        zinv = 1.0 / scale

        def s(n):
            return max(1, int(round(n * zinv)))

        # Edge indicators: bright orange line on any side within 50px of the full image boundary.
        # Drawn first so all polygon handles render on top.
        ch, cw = crop.shape[:2]
        cx0, cy0 = entry["crop_x"], entry["crop_y"]
        iw, ih = entry["img_w"], entry["img_h"]
        EDGE_T = 10
        EDGE_TOL = 50
        EDGE_COLOR = (0, 165, 255)  # bright orange (BGR)
        if cx0 <= EDGE_TOL:
            cv2.line(crop, (0, 0), (0, ch - 1), EDGE_COLOR, EDGE_T)
        if cy0 <= EDGE_TOL:
            cv2.line(crop, (0, 0), (cw - 1, 0), EDGE_COLOR, EDGE_T)
        if cx0 + cw >= iw - EDGE_TOL:
            cv2.line(crop, (cw - 1, 0), (cw - 1, ch - 1), EDGE_COLOR, EDGE_T)
        if cy0 + ch >= ih - EDGE_TOL:
            cv2.line(crop, (0, ch - 1), (cw - 1, ch - 1), EDGE_COLOR, EDGE_T)

        main_sid = entry["server_id"]
        visible = self._visible_polys_for_entry(entry)

        # Pass 1: every polygon in its base color
        for sid, local_pts in visible:
            poly = self.polygons_by_id.get(sid)
            is_main = (sid == main_sid)
            is_marked = sid in self.marked_ids
            if is_marked:
                color = DELETE_COLOR
                thick = s(2)
            elif is_main:
                color = CURRENT_COLOR
                thick = s(2)
            else:
                color = (255, 180, 100)
                thick = s(1)
            cv2.polylines(crop, [local_pts], True, color, thick)

        # Pass 2: faint red fill on marked polygons
        for sid, local_pts in visible:
            if sid in self.marked_ids:
                overlay = crop.copy()
                cv2.fillPoly(overlay, [local_pts], DELETE_COLOR)
                crop = cv2.addWeighted(overlay, 0.20, crop, 0.80, 0)

        # Pass 3: edited / added status outlines (on top of base color)
        for sid, local_pts in visible:
            poly = self.polygons_by_id.get(sid)
            if poly is None:
                continue
            if poly["status"] == "edited":
                cv2.polylines(crop, [local_pts], True, EDITED_COLOR, s(1))
            elif poly["status"] == "added":
                cv2.polylines(crop, [local_pts], True, ADDED_COLOR, s(1))

        # Pass 4: selected polygon - thick white outline + vertex handles
        hovered_vert = (self.hovered_handle if self.hovered_handle and
                          self.hovered_handle[0] == "vertex" else None)
        for sid, local_pts in visible:
            if sid != self.selected_id:
                continue
            cv2.polylines(crop, [local_pts], True, SELECTED_COLOR, s(2))
            for vi, (vx, vy) in enumerate(local_pts):
                is_hovered = (hovered_vert is not None
                               and hovered_vert[1] == sid
                               and hovered_vert[2] == vi)
                bump = 2 if is_hovered else 0
                cv2.circle(crop, (int(vx), int(vy)),
                            s(self.HANDLE_DRAW_R + 1 + bump),
                            (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(crop, (int(vx), int(vy)),
                            s(self.HANDLE_DRAW_R + bump),
                            (0, 0, 0), -1, cv2.LINE_AA)

        # Pass 5: extend-handle diamonds for selected polygon
        if self.selected_id is not None:
            poly_meta = self.polygons_by_id.get(self.selected_id)
            if poly_meta is not None:
                cx0 = entry["crop_x"]
                cy0 = entry["crop_y"]
                hovered_ext = (self.hovered_handle
                                if self.hovered_handle and
                                   self.hovered_handle[0] == "extend"
                                else None)
                handles_list = self._compute_extend_handles(poly_meta["points"])
                for end_idx, h in enumerate(handles_list):
                    cx = int(round(h["center_full"][0] - cx0))
                    cy = int(round(h["center_full"][1] - cy0))
                    if not (0 <= cx < crop.shape[1] and 0 <= cy < crop.shape[0]):
                        continue
                    is_hovered = (hovered_ext is not None
                                   and hovered_ext[1] == self.selected_id
                                   and hovered_ext[2] == h.get("kind")
                                   and hovered_ext[3] == end_idx)
                    bump = 2 if is_hovered else 0
                    size = s(self.HANDLE_DRAW_R + 3 + bump)
                    outer = np.array([
                        [cx, cy - size], [cx + size, cy],
                        [cx, cy + size], [cx - size, cy],
                    ], dtype=np.int32)
                    cv2.fillPoly(crop, [outer], (255, 255, 255))
                    inner_size = max(2, size - s(2))
                    inner = np.array([
                        [cx, cy - inner_size], [cx + inner_size, cy],
                        [cx, cy + inner_size], [cx - inner_size, cy],
                    ], dtype=np.int32)
                    if h.get("kind") == "width":
                        fill = (220, 110, 30)
                    else:
                        fill = (0, 200, 0)
                    cv2.fillPoly(crop, [inner], fill)

        # Pass 6: add-mode in-progress polygon (lime + dots)
        if self.add_mode and self.add_pts_local:
            pts = np.array(self.add_pts_local, dtype=np.int32)
            line_color = (50, 255, 50)
            for i in range(len(pts) - 1):
                cv2.line(crop, tuple(pts[i]), tuple(pts[i + 1]),
                          line_color, s(1), cv2.LINE_AA)
            if self.add_hover_xy is not None:
                last = (int(pts[-1][0]), int(pts[-1][1]))
                hx, hy = self.add_hover_xy
                snap_close = False
                if len(pts) >= 3:
                    fx, fy = int(pts[0][0]), int(pts[0][1])
                    if (fx - hx) ** 2 + (fy - hy) ** 2 <= 14 ** 2:
                        snap_close = True
                cv2.line(crop, last, (hx, hy), line_color, s(1), cv2.LINE_AA)
                if snap_close:
                    cv2.line(crop, (hx, hy), (fx, fy),
                              (0, 200, 255), s(2), cv2.LINE_AA)
            for vi, (vx, vy) in enumerate(pts):
                if vi == 0 and len(pts) >= 3:
                    cv2.circle(crop, (int(vx), int(vy)), s(6), line_color,
                                -1, cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)), s(7), (0, 0, 0),
                                s(1), cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)), s(9), (255, 255, 255),
                                s(1), cv2.LINE_AA)
                else:
                    cv2.circle(crop, (int(vx), int(vy)), s(3), line_color,
                                -1, cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)), s(4), (0, 0, 0),
                                s(1), cv2.LINE_AA)

        return crop


    def crop_to_pixmap(self, cv_img):
        """Convert a BGR OpenCV image into a Qt QPixmap for display."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def on_brightness_changed(self, value):
        """Update the display brightness multiplier from the slider (1-tick =
        0.1x), update its readout, and live-redraw the current crop. Display
        only — it never changes the source pixels."""
        self.brightness = value / 10.0
        self.bright_value_label.setText(f"{self.brightness:.1f}x")
        if self.entries:
            entry = self.entries[self.current_idx]
            crop = self.build_display_crop(entry)
            pix = self.crop_to_pixmap(crop)
            self.image_view.update_pixmap(pix)

    def on_contrast_changed(self, value):
        """Update the display contrast multiplier from the slider (1-tick =
        0.1x), update its readout, and live-redraw the current crop. Display
        only — it never changes the source pixels."""
        self.contrast = value / 10.0
        self.contrast_value_label.setText(f"{self.contrast:.1f}x")
        if self.entries:
            entry = self.entries[self.current_idx]
            crop = self.build_display_crop(entry)
            pix = self.crop_to_pixmap(crop)
            self.image_view.update_pixmap(pix)

    def open_in_cvat(self):
        """Open the CVAT web UI in a browser tab pointed straight at the current
        polygon (its job, frame, and server id), with a cache-busting timestamp
        so CVAT re-loads it fresh."""
        if not self.entries:
            return
        entry = self.entries[self.current_idx]
        frame_idx = entry["frame_idx"]
        sid = entry["server_id"]
        ts = int(time.time())
        url = (
            f"{CVAT_URL}/tasks/{CVAT_TASK_ID}/jobs/{self.job_id}"
            f"?frame={frame_idx}&type=shape&serverID={sid}&_t={ts}"
        )
        webbrowser.open_new_tab(url)

    def _pull_from_cvat(self):
        """Re-fetch ALL polygons on the current frame from CVAT by frame index.
        Updates existing, adds new, removes deleted. Used after send or manual pull."""
        if not self.entries:
            return

        entry = self.entries[self.current_idx]
        frame_idx = entry["frame_idx"]

        # Warn once if any locally-tracked polygon on this frame has unsaved edits
        local_frame_sids = [
            sid for sid in self._frame_to_sids.get(frame_idx, set())
            if sid >= 0
        ]
        edited_sids = [
            sid for sid in local_frame_sids
            if self.polygons_by_id.get(sid, {}).get("status") == "edited"
        ]
        if edited_sids:
            reply = QMessageBox.warning(self, "Pull from CVAT",
                f"{len(edited_sids)} polygon(s) on this frame have local edits "
                "that haven't been pushed yet.\n\nPulling will overwrite them "
                "with the CVAT versions.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.Cancel)
            if reply != QMessageBox.Yes:
                return

        self.pull_btn.setEnabled(False)
        self.pull_btn.setText("Pulling...")
        QApplication.processEvents()

        try:
            password = read_cvat_password()
            ann_resp = requests.get(
                f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                auth=(CVAT_USER, password)).json()
        except Exception as exc:
            self.pull_btn.setEnabled(True)
            self.pull_btn.setText("Pull from CVAT")
            QMessageBox.critical(self, "CVAT error",
                f"Failed to pull from CVAT:\n{exc}")
            return

        # Filter to only shapes on the current frame
        cvat_frame_shapes = [s for s in ann_resp.get("shapes", [])
                             if s["frame"] == frame_idx]
        cvat_frame_ids = {s["id"] for s in cvat_frame_shapes}

        # Remove locally-tracked polygons on this frame that CVAT no longer has
        for sid in list(local_frame_sids):
            if sid not in cvat_frame_ids:
                self.polygons_by_id.pop(sid, None)
                for sids in self._frame_to_sids.values():
                    sids.discard(sid)
                self.marked_ids.discard(sid)
                if self.selected_id == sid:
                    self.selected_id = None
                try:
                    _cache_crop_path(sid).unlink(missing_ok=True)
                except Exception:
                    pass

        # Read source image once — needed to re-crop all polygons with correct margins
        ref_entry = next(
            (e for e in self.entries if e["frame_idx"] == frame_idx), None
        )
        img_path = ref_entry["img_path"] if ref_entry else None
        img = cv2.imread(img_path) if img_path else None
        img_h, img_w = (img.shape[:2] if img is not None else (0, 0))

        all_shapes_sorted = sorted(ann_resp.get("shapes", []),
                                   key=lambda s: (s["frame"], s["id"]))
        server_id_to_seq = {s["id"]: i + 1
                            for i, s in enumerate(all_shapes_sorted)}
        cache_index = _load_cache_index()

        for shape in cvat_frame_shapes:
            sid = shape["id"]
            raw_pts = shape["points"]
            new_pts = np.array(list(zip(raw_pts[0::2], raw_pts[1::2])),
                               dtype=np.float32)
            if new_pts.shape[0] < 3:
                continue

            poly = self.polygons_by_id.get(sid)
            is_new = poly is None

            if is_new:
                # Register new polygon
                self.polygons_by_id[sid] = {
                    "server_id": sid,
                    "frame_idx": frame_idx,
                    "points": new_pts,
                    "original_points": new_pts.copy(),
                    "status": "original",
                }
                self._frame_to_sids.setdefault(frame_idx, set()).add(sid)
                poly = self.polygons_by_id[sid]
            else:
                poly["points"] = new_pts
                poly["original_points"] = new_pts.copy()
                poly["status"] = "original"

            # Re-crop with correct margins for both existing and new polygons
            if img is None or img_path is None:
                if not is_new:
                    # Fall back to in-memory shift for existing if image unavailable
                    for e in self.entries:
                        if e["server_id"] != sid:
                            continue
                        shifted = new_pts.copy()
                        shifted[:, 0] -= e["crop_x"]
                        shifted[:, 1] -= e["crop_y"]
                        e["poly_pts"] = shifted.astype(np.int32)
                        break
                continue

            # New polygon case falls through to here too
            if is_new and img_path is None:
                continue
            # Re-crop with correct margins
            bbox_w = new_pts[:, 0].max() - new_pts[:, 0].min()
            bbox_h = new_pts[:, 1].max() - new_pts[:, 1].min()
            pad = max(MIN_CROP_PAD, int(CROP_PAD_RATIO * max(bbox_w, bbox_h)))
            x_min = max(0, int(new_pts[:, 0].min()) - pad)
            y_min = max(0, int(new_pts[:, 1].min()) - pad)
            x_max = min(img_w, int(new_pts[:, 0].max()) + pad)
            y_max = min(img_h, int(new_pts[:, 1].max()) + pad)
            crop_raw = img[y_min:y_max, x_min:x_max].copy()
            shifted = new_pts.copy()
            shifted[:, 0] -= x_min
            shifted[:, 1] -= y_min

            if is_new:
                stem = Path(img_path).stem
                self.entries.append({
                    "frame": stem,
                    "frame_idx": frame_idx,
                    "server_id": sid,
                    "cvat_seq": server_id_to_seq.get(sid, 0),
                    "label": f"{stem} #{sid}",
                    "crop_raw": crop_raw,
                    "crop_x": x_min,
                    "crop_y": y_min,
                    "img_path": img_path,
                    "img_w": img_w,
                    "img_h": img_h,
                    "poly_pts": shifted.astype(np.int32),
                    "flag_reason": "",
                    "flagged": False,
                    "delete": False,
                })
            else:
                for e in self.entries:
                    if e["server_id"] != sid:
                        continue
                    e["crop_raw"] = crop_raw
                    e["crop_x"] = x_min
                    e["crop_y"] = y_min
                    e["img_w"] = img_w
                    e["img_h"] = img_h
                    e["poly_pts"] = shifted.astype(np.int32)
                    break

            try:
                crop_path = _cache_crop_path(sid)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_path), crop_raw,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cache_index[sid] = {
                    "hash": _points_hash(new_pts),
                    "frame_idx": int(frame_idx),
                    "fname": Path(img_path).name,
                    "crop_x": int(x_min),
                    "crop_y": int(y_min),
                    "img_w": int(img_w),
                    "img_h": int(img_h),
                }
            except Exception:
                pass

        _save_cache(cache_index)

        self._last_displayed_idx = None
        self.pull_btn.setEnabled(True)
        self.pull_btn.setText("Pull from CVAT")
        self._save_edits()
        self._refresh_send_button()
        self.update_filter_counts()
        self.refresh_view()

    # ── Filter logic ────────────────────────────────────────────────────────

    def filtered_indices(self):
        """Return the list of entry indices that match the active Show filter
        (all / flagged / not-flagged / marked-for-delete / not-marked). This is
        the ordering the scrubber, Prev/Next, and jump box all walk."""
        if self.filter_mode == "all":
            return list(range(len(self.entries)))
        elif self.filter_mode == "flagged":
            return [i for i, e in enumerate(self.entries) if e["flagged"]]
        elif self.filter_mode == "not_flagged":
            return [i for i, e in enumerate(self.entries) if not e["flagged"]]
        elif self.filter_mode == "delete":
            return [i for i, e in enumerate(self.entries) if e["delete"]]
        elif self.filter_mode == "not_marked" or self.filter_mode == "keep":
            return [i for i, e in enumerate(self.entries) if not e["delete"]]
        return []

    def set_filter(self, mode):
        """Switch the active Show filter, snap the cursor to the first matching
        entry if the current one is filtered out, redraw, and save the
        position."""
        self.filter_mode = mode
        indices = self.filtered_indices()
        if indices and self.current_idx not in indices:
            self.current_idx = indices[0]
        self.refresh_view()
        self._save_position()

    def _on_scrubber_changed(self, value):
        """Jump to the Nth entry within the current filter when the scrubber
        slider moves (1-based)."""
        indices = self.filtered_indices()
        if not indices:
            return
        idx = max(0, min(value - 1, len(indices) - 1))
        self.current_idx = indices[idx]
        self.refresh_view()

    def _on_jump_entered(self):
        """Jump to the position typed into the jump box (1-based within the
        current filter, clamped to range), then drop keyboard focus. Ignores
        blank or non-numeric input."""
        text = self.jump_input.text().strip()
        if not text:
            return
        try:
            target = int(text)
        except ValueError:
            return
        indices = self.filtered_indices()
        if not indices:
            return
        # Treat as position in current filter (1-based)
        pos = max(1, min(target, len(indices)))
        self.current_idx = indices[pos - 1]
        self.jump_input.clearFocus()
        self.refresh_view()

    # ── View refresh ────────────────────────────────────────────────────────

    def refresh_view(self):
        """Redraw everything for the current cursor position: the decorated
        crop image, the scrubber + "N of M" label + jump box, the info strip
        (frame, filename, id, FLAGGED / MARKED tags, flag reason), the red
        window border when the main polygon is marked, the Prev/Next enabled
        states, and the filter counts. When the cursor moved to a new entry it
        sets the raw pixmap first so the view re-fits before overlays are drawn.
        Shows an empty "No polygons in this view" state if the filter is
        empty."""
        indices = self.filtered_indices()
        if not indices:
            self.image_view.set_pixmap(QPixmap())
            self.scrubber_label.setText("0 of 0")
            self.jump_input.setText("")
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(0)
            self.scrubber.blockSignals(False)
            self.info_label.setText("No polygons in this view.")
            self.reason_label.setText("")
            self.image_view.setStyleSheet(f"background: {IMAGE_BG}; border: none;")
            self.update_filter_counts()
            return

        if self.current_idx not in indices:
            self.current_idx = indices[0]

        pos_in_filter = indices.index(self.current_idx) + 1
        total_in_filter = len(indices)

        self.scrubber.blockSignals(True)
        self.scrubber.setMaximum(max(1, total_in_filter))
        self.scrubber.setValue(pos_in_filter)
        self.scrubber.blockSignals(False)
        self.scrubber_label.setText(f"{pos_in_filter} of {total_in_filter}")
        self.jump_input.setText(str(pos_in_filter))

        entry = self.entries[self.current_idx]

        is_new_entry = (self._last_displayed_idx != self.current_idx)
        if is_new_entry:
            # Set the raw crop first so fitInView runs and the transform is
            # correct before build_display_crop() reads current_scale().
            self.image_view.set_pixmap(self.crop_to_pixmap(entry["crop_raw"]))
            self._last_displayed_idx = self.current_idx

        crop = self.build_display_crop(entry)
        pix = self.crop_to_pixmap(crop)
        self.image_view.update_pixmap(pix)

        # Build info line
        tags = []
        if entry["flagged"]:
            tags.append("[FLAGGED]")
        if entry["delete"]:
            tags.append("[MARKED FALSE POSITIVE]")
        tag_str = "    ".join(tags)
        if tag_str:
            tag_str = "    " + tag_str
        self.info_label.setText(
            f"frame {entry['frame_idx']}  |  "
            f"{entry['frame']}  #{entry['server_id']}    "
            f"({pos_in_filter} of {total_in_filter}){tag_str}"
        )

        if entry["flag_reason"]:
            self.reason_label.setText(f"Reason: {entry['flag_reason']}")
        else:
            self.reason_label.setText("")

        main_marked = entry["server_id"] in self.marked_ids
        if main_marked:
            self.image_view.setStyleSheet(f"background: {IMAGE_BG}; border: 10px solid red;")
        else:
            self.image_view.setStyleSheet(f"background: {IMAGE_BG}; border: none;")

        self.prev_btn.setEnabled(pos_in_filter > 1)
        self.next_btn.setEnabled(pos_in_filter < total_in_filter)

        self.update_filter_counts()

    def eventFilter(self, obj, event):
        """Installed on the image view; currently a pass-through to the default
        handling (kept as a hook for intercepting view events)."""
        return super().eventFilter(obj, event)

    def update_filter_counts(self):
        """Recompute the live counts shown on each Show filter pill (All,
        Flagged, Not Flagged, Marked, Not Marked) so they reflect the current
        flag/delete state."""
        n_del = sum(1 for e in self.entries if e["delete"])
        n_keep = len(self.entries) - n_del
        n_flagged = sum(1 for e in self.entries if e["flagged"])
        n_not_flagged = len(self.entries) - n_flagged
        self.filter_buttons["all"].setText(f"All ({len(self.entries)})")
        self.filter_buttons["flagged"].setText(f"Flagged ({n_flagged})")
        self.filter_buttons["not_flagged"].setText(f"Not Flagged ({n_not_flagged})")
        self.filter_buttons["delete"].setText(f"Marked ({n_del})")
        self.filter_buttons["not_marked"].setText(f"Not Marked ({n_keep})")

    # ── Navigation ──────────────────────────────────────────────────────────

    def go_prev(self):
        """Step to the previous polygon within the current filter, selecting its
        main polygon. Plays a soft "end" sound if already at the first one."""
        indices = self.filtered_indices()
        if not indices:
            return
        pos = indices.index(self.current_idx) if self.current_idx in indices else 0
        if pos > 0:
            self.current_idx = indices[pos - 1]
            self.selected_id = self.entries[self.current_idx]["server_id"]
            self.refresh_view()
        else:
            _play_end_sound()

    def go_next(self):
        """Step to the next polygon within the current filter, selecting its
        main polygon. Plays a soft "end" sound if already at the last one."""
        indices = self.filtered_indices()
        if not indices:
            return
        pos = indices.index(self.current_idx) if self.current_idx in indices else 0
        if pos < len(indices) - 1:
            self.current_idx = indices[pos + 1]
            self.selected_id = self.entries[self.current_idx]["server_id"]
            self.refresh_view()
        else:
            _play_end_sound()

    def _toggle_delete(self):
        """Toggle the marked-for-delete state on the SELECTED polygon
        (falls back to the current entry's main polygon if nothing is
        explicitly selected). Special case: an "added" polygon (created in
        this session, never pushed to CVAT) is removed immediately on
        delete, since there's nothing to delete on the server."""
        if not self.entries:
            return
        sid = self.selected_id
        if sid is None:
            sid = self.entries[self.current_idx]["server_id"]

        poly = self.polygons_by_id.get(sid)
        # Locally-added polygon being deleted: remove from registry entirely.
        # Cmd+Z restores via the "re_add" undo op.
        if poly is not None and poly["status"] == "added":
            popped = self.polygons_by_id.pop(sid, None)
            for sids in self._frame_to_sids.values():
                sids.discard(sid)
            if popped is not None:
                self._push_undo(("re_add", sid, popped))
            if self.selected_id == sid:
                self.selected_id = None
            self._save_edits()
            self._refresh_send_button()
            self.update_filter_counts()
            self.refresh_view()
            return

        # Existing polygon (in CVAT): toggle the mark and queue for delete
        # on the next Send.
        was_marked = sid in self.marked_ids
        if was_marked:
            self.marked_ids.discard(sid)
            self._sync_entry_delete_flag(sid, False)
            self._push_undo(("unmark", sid))
        else:
            self.marked_ids.add(sid)
            self._sync_entry_delete_flag(sid, True)
            self._push_undo(("mark", sid))
        self._save_marks()
        self._save_edits()
        self.update_filter_counts()
        self._refresh_send_button()
        self.refresh_view()

    def _on_image_clicked(self, ix, iy):
        """Click selects the smallest polygon under the click. To toggle the
        delete flag, click to select then press Space / Del / Backspace
        (or the Delete button)."""
        if not self.entries:
            return
        if self.add_mode:
            # Add-mode click is handled separately — append a vertex (or close
            # the polygon if near the first vertex). Mirrors TileFixR.
            SNAP_R = 14
            if len(self.add_pts_local) >= 3:
                fx, fy = self.add_pts_local[0]
                if (fx - ix) ** 2 + (fy - iy) ** 2 <= SNAP_R ** 2:
                    self._close_add_polygon()
                    return
            self.add_pts_local.append((int(ix), int(iy)))
            if len(self.add_pts_local) >= 4:
                self._close_add_polygon()
                return
            self.refresh_view()
            return
        entry = self.entries[self.current_idx]
        pt = (float(ix), float(iy))
        candidates = []  # (area, server_id)
        for sid, local_pts in self._visible_polys_for_entry(entry):
            if cv2.pointPolygonTest(local_pts, pt, False) >= 0:
                candidates.append((cv2.contourArea(local_pts), sid))
        if not candidates:
            self.selected_id = None
            self.refresh_view()
            return
        candidates.sort(key=lambda x: x[0])
        self.selected_id = candidates[0][1]
        self.refresh_view()

    def _save_marks(self):
        """Persist the set of marked-for-delete polygon server ids to the
        per-task marks.json so they survive a relaunch."""
        marked_sids = sorted(self.marked_ids)
        try:
            MARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
            MARKS_PATH.write_text(json.dumps({
                "frame_start": FRAME_START,
                "frame_end": FRAME_END,
                "count": len(marked_sids),
                "marked_server_ids": marked_sids,
            }, indent=2))
        except Exception:
            pass

    # ── Edit persistence (auto-saved on every change, restored on reload) ──

    @property
    def _edits_path(self):
        """Path to the per-task edits.json (pending edits + adds + temp-id
        cursor), kept alongside the other per-task state files."""
        return STATE_PATH.parent / "edits.json"

    def _save_edits(self):
        """Persist all polygon edits, adds, and the undo/redo cursor to disk.
        Auto-saved on every change so a crash or relaunch never loses work."""
        try:
            edits = []
            adds = []
            for sid, poly in self.polygons_by_id.items():
                if poly["status"] == "edited":
                    edits.append({
                        "server_id": int(sid),
                        "frame_idx": int(poly["frame_idx"]),
                        "points": [[float(x), float(y)] for x, y in poly["points"]],
                    })
                elif poly["status"] == "added":
                    adds.append({
                        "temp_id": int(sid),
                        "frame_idx": int(poly["frame_idx"]),
                        "points": [[float(x), float(y)] for x, y in poly["points"]],
                    })
            data = {
                "edits": edits,
                "adds": adds,
                "next_temp_id": int(self.next_temp_id),
            }
            self._edits_path.parent.mkdir(parents=True, exist_ok=True)
            self._edits_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_edits(self):
        """Restore previously saved edits and adds from disk. Called once at
        startup. Edits and adds restore as 'edited' / 'added' so they remain
        visible in the queue and get pushed on the next Send to CVAT."""
        try:
            path = self._edits_path
            if not path.exists():
                return
            data = json.loads(path.read_text())
            for edit in data.get("edits", []):
                sid = edit["server_id"]
                poly = self.polygons_by_id.get(sid)
                if poly is None:
                    continue
                pts = np.array(edit["points"], dtype=np.float32)
                if pts.shape[0] >= 3:
                    poly["points"] = pts
                    poly["status"] = "edited"
            for add in data.get("adds", []):
                tid = add["temp_id"]
                pts = np.array(add["points"], dtype=np.float32)
                if pts.shape[0] >= 3:
                    self.polygons_by_id[tid] = {
                        "server_id": tid,
                        "frame_idx": int(add["frame_idx"]),
                        "points": pts,
                        "original_points": pts.copy(),
                        "status": "added",
                        "label": "trail",
                    }
                    self._frame_to_sids.setdefault(
                        int(add["frame_idx"]), set()).add(tid)
            self.next_temp_id = int(data.get("next_temp_id", -1))
        except Exception as exc:
            print(f"  WARNING: could not restore edits: {exc}")

    # ── Send button counter / QoL helpers ──────────────────────────────────

    def _refresh_send_button(self):
        """Recount pending deletes + adds + edits and update both Send buttons'
        label ("Send N changes to CVAT") and enabled state."""
        n_del = len(self.marked_ids)
        n_add = sum(1 for p in self.polygons_by_id.values()
                     if p["status"] == "added")
        n_edit = sum(1 for p in self.polygons_by_id.values()
                      if p["status"] == "edited")
        total = n_del + n_add + n_edit
        if hasattr(self, "send_btn"):
            self.send_btn.setText(
                f"Send {total} change{'' if total == 1 else 's'} to CVAT")
            self.send_btn.setEnabled(total > 0)
        if hasattr(self, "save_btn"):
            self.save_btn.setEnabled(total > 0)

    # QoL 1: T-key Auto-Tighten - snap selected polygon to bright trail core
    def _auto_tighten(self):
        """Snap the selected polygon tight around the bright trail inside it
        (the T key). Reads the source image, measures a sky-brightness baseline
        from pixels outside the polygon, keeps the pixels inside that are well
        above that baseline (all blobs above a small size floor, so dashed
        satellite trails contribute every dash), finds the trail's axis with
        Hough line detection (so round stars cast no votes), and rebuilds the
        polygon as a tight oriented box around that axis with a small margin.
        Falls back to an SVD axis fit if Hough finds no line. Pushes an undo
        entry, marks the polygon edited, and auto-saves. Shows a warning dialog
        and does nothing if nothing is selected or there aren't enough
        sky/bright pixels to work with."""
        if not self.entries or self.selected_id is None:
            QMessageBox.information(self, "Nothing selected",
                "Click a polygon first, then press T to tighten it.")
            return
        poly = self.polygons_by_id.get(self.selected_id)
        if poly is None or poly["points"].shape[0] < 3:
            return
        entry = self.entries[self.current_idx]
        # Read the source image once - cache by frame_idx for re-use.
        img = self._img_cache.get(entry["img_path"])
        if img is None:
            img = cv2.imread(entry["img_path"])
            if img is None:
                QMessageBox.warning(self, "Auto-tighten failed",
                    "Could not read the source image for this polygon.")
                return
            self._img_cache[entry["img_path"]] = img
        ih, iw = img.shape[:2]
        ff = poly["points"]
        x0 = max(0, int(ff[:, 0].min()) - 6)
        y0 = max(0, int(ff[:, 1].min()) - 6)
        x1 = min(iw, int(ff[:, 0].max()) + 6)
        y1 = min(ih, int(ff[:, 1].max()) + 6)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        region = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Build the polygon mask in region-local coords
        local_pts = ff.copy()
        local_pts[:, 0] -= x0
        local_pts[:, 1] -= y0
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [local_pts.astype(np.int32)], 255)
        sky_pixels = gray[mask == 0]
        if len(sky_pixels) < 100:
            QMessageBox.warning(self, "Auto-tighten failed",
                "Not enough sky pixels around this polygon to compute a "
                "brightness baseline.")
            return
        baseline = float(np.median(sky_pixels))
        mad = float(np.median(np.abs(sky_pixels - baseline)))
        threshold = baseline + 4.0 * 1.4826 * mad
        bright = (gray > threshold) & (mask > 0)
        if int(bright.sum()) < 20:
            QMessageBox.warning(self, "Auto-tighten failed",
                "Couldn't find enough bright pixels inside this polygon. "
                "Try increasing brightness or adjust the polygon manually.")
            return
        # Keep all blobs above the noise floor so dashed satellite trails
        # contribute all their dashes to the PCA, not just the biggest one.
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright.astype(np.uint8))
        if n_lab <= 1:
            return
        MIN_BLOB = 5
        keep = np.zeros(bright.shape, dtype=bool)
        for i in range(1, n_lab):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB:
                keep |= (labels == i)
        ys, xs = np.where(keep)
        if len(xs) < 4:
            return
        MARGIN_ALONG = 10
        MARGIN_PERP  = 10
        pts_arr = np.column_stack([xs, ys]).astype(np.float32)
        # Use Hough line detection to find the trail axis. Hough votes are cast
        # by the line itself; round stars generate no consistent line votes and
        # are ignored. Fall back to SVD if Hough finds nothing.
        keep_img = keep.astype(np.uint8) * 255
        hough_lines = cv2.HoughLinesP(keep_img, 1, np.pi / 180,
                                      threshold=15, minLineLength=8,
                                      maxLineGap=40)
        if hough_lines is not None and len(hough_lines) > 0:
            angles = []
            mids = []
            for seg in hough_lines:
                x1, y1, x2, y2 = seg[0]
                angles.append(np.arctan2(float(y2 - y1), float(x2 - x1)))
                mids.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            med_angle = float(np.median(angles))
            axis = np.array([np.cos(med_angle), np.sin(med_angle)],
                            dtype=np.float32)
            perp = np.array([-np.sin(med_angle), np.cos(med_angle)],
                            dtype=np.float32)
            # Center on the trail line itself, not the pixel cloud centroid,
            # so stars above/below don't shift the box off the trail.
            center = np.array(mids, dtype=np.float32).mean(axis=0)
            proj_along = (pts_arr - center) @ axis
            proj_perp  = (pts_arr - center) @ perp
            # Perp extent: 90th-percentile absolute offset captures the trail
            # width while ignoring outlier stars.
            perp_half = float(np.percentile(np.abs(proj_perp), 90)) + MARGIN_PERP
            corners = np.array([
                center + (proj_along.min() - MARGIN_ALONG) * axis - perp_half * perp,
                center + (proj_along.min() - MARGIN_ALONG) * axis + perp_half * perp,
                center + (proj_along.max() + MARGIN_ALONG) * axis + perp_half * perp,
                center + (proj_along.max() + MARGIN_ALONG) * axis - perp_half * perp,
            ])
        else:
            center = pts_arr.mean(axis=0)
            _, _, vt = np.linalg.svd(pts_arr - center, full_matrices=False)
            axis = vt[0]
            perp = vt[1]
            proj_along = (pts_arr - center) @ axis
            proj_perp  = (pts_arr - center) @ perp
            corners = np.array([
                center + (proj_along.min() - MARGIN_ALONG) * axis + (proj_perp.min() - MARGIN_PERP) * perp,
                center + (proj_along.min() - MARGIN_ALONG) * axis + (proj_perp.max() + MARGIN_PERP) * perp,
                center + (proj_along.max() + MARGIN_ALONG) * axis + (proj_perp.max() + MARGIN_PERP) * perp,
                center + (proj_along.max() + MARGIN_ALONG) * axis + (proj_perp.min() - MARGIN_PERP) * perp,
            ])
        corners[:, 0] += x0
        corners[:, 1] += y0
        new_pts = corners.astype(np.float32)
        # Push undo, swap in the new polygon
        self._push_undo(("vertex", self.selected_id,
                          poly["points"].copy(), poly["status"]))
        poly["points"] = new_pts
        if poly["status"] == "original":
            poly["status"] = "edited"
        self._save_edits()
        self._refresh_send_button()
        self.refresh_view()

    # QoL 3: Tab navigation within the same frame
    def _tab_within_frame(self):
        """Tab: move to the next polygon that lives on the SAME source frame."""
        self._step_within_frame(+1)

    def _shift_tab_within_frame(self):
        """Shift-Tab: move to the previous polygon on the SAME source frame."""
        self._step_within_frame(-1)

    def _step_within_frame(self, direction):
        """Step one polygon forward or backward but only among entries on the
        current entry's frame (filter-aware), selecting the new one. Stays put
        if there's no neighbor in that direction within the frame."""
        if not self.entries:
            return
        cur = self.entries[self.current_idx]
        frame_idx = cur["frame_idx"]
        # Indices of entries in this frame, in their natural (filter-aware)
        # order. Walk the filter; pick neighbors in the same frame.
        indices = self.filtered_indices()
        same_frame = [i for i in indices if self.entries[i]["frame_idx"] == frame_idx]
        if not same_frame:
            return
        try:
            pos = same_frame.index(self.current_idx)
        except ValueError:
            pos = 0
        nxt = pos + direction
        if 0 <= nxt < len(same_frame):
            self.current_idx = same_frame[nxt]
            self.selected_id = self.entries[self.current_idx]["server_id"]
            self.refresh_view()

        self.refresh_view()

    # ── Batch actions ───────────────────────────────────────────────────────

    def accept_all_flagged(self):
        """Mark every Claude-flagged polygon for deletion in one go (after a
        confirm dialog). Individual ones can still be unmarked before sending."""
        flagged = [e for e in self.entries if e["flagged"]]
        if not flagged:
            QMessageBox.information(self, "Nothing flagged", "No polygons are flagged.")
            return

        msg = (f"Mark all {len(flagged)} Claude-flagged polygons for deletion?\n\n"
               "You can still unmark individual ones before saving.")
        reply = QMessageBox.question(self, "Accept all flagged", msg,
                                     QMessageBox.Yes | QMessageBox.Cancel)
        if reply != QMessageBox.Yes:
            return

        for e in flagged:
            e["delete"] = True
        self._save_marks()
        self.refresh_view()

    def save_deletions(self):
        """Legacy entry point - now sends all changes (add / edit / delete) to
        CVAT in one batched call."""
        self._send_to_cvat()

    def _send_to_cvat(self):
        """Push every pending change to CVAT in one batch. Snapshots the adds,
        edits, and marked-for-delete ids, hands them to a background
        CvatSendWorker thread (so the UI stays responsive), and runs the
        "Sending..." button animation until the worker reports back to
        _on_send_finished. Does nothing if there are no pending changes."""
        n_del = len(self.marked_ids)
        adds = [p for p in self.polygons_by_id.values() if p["status"] == "added"]
        edits = [p for p in self.polygons_by_id.values() if p["status"] == "edited"]
        if n_del + len(adds) + len(edits) == 0:
            return

        # Snapshot data for the worker — safe to pass across threads
        adds_data = [{"points": p["points"].copy(), "frame_idx": p["frame_idx"]}
                     for p in adds]
        edits_data = [{"server_id": p["server_id"], "points": p["points"].copy()}
                      for p in edits]
        marked_snapshot = set(self.marked_ids)

        password = read_cvat_password()
        self._send_worker = CvatSendWorker(
            self.job_id, CVAT_TASK_ID,
            (CVAT_USER, password),
            adds_data, edits_data, marked_snapshot,
        )
        self._send_worker.finished.connect(
            lambda ok, err: self._on_send_finished(ok, err, marked_snapshot))
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Sending")
        self._send_anim_frame = 0
        self._send_anim_timer = QTimer(self)
        self._send_anim_timer.timeout.connect(self._tick_send_animation)
        self._send_anim_timer.start(400)
        self._send_worker.start()

    def _tick_send_animation(self):
        """Advance the "Sending..." ellipsis animation on the Send button by one
        frame while a push is in flight."""
        dots = ("Sending", "Sending.", "Sending..", "Sending...")
        self._send_anim_frame = (self._send_anim_frame + 1) % len(dots)
        self.send_btn.setText(dots[self._send_anim_frame])

    def _on_send_finished(self, success, error_msg, deleted_sids):
        """Handle the background send result. On failure, show the CVAT error
        and leave the pending state intact to retry. On success, CVAT is now
        caught up: clear all marks, drop the deleted polygons (and their cached
        crops), reset every added/edited polygon back to "original", clear
        undo/redo, save state, refresh the UI, and re-pull the current frame so
        adds come back with their real CVAT ids."""
        if hasattr(self, "_send_anim_timer"):
            self._send_anim_timer.stop()
        self.send_btn.setEnabled(True)
        if not success:
            self._refresh_send_button()
            QMessageBox.critical(self, "CVAT error",
                f"CVAT update failed:\n{error_msg}")
            return

        # Reset local pending state - CVAT is now caught up.
        self.marked_ids.clear()
        for e in self.entries:
            e["delete"] = False
        for sid in deleted_sids:
            self.polygons_by_id.pop(sid, None)
            for frame_sids in self._frame_to_sids.values():
                frame_sids.discard(sid)
            if self.selected_id == sid:
                self.selected_id = None
            try:
                _cache_crop_path(sid).unlink(missing_ok=True)
            except Exception:
                pass
        for poly in list(self.polygons_by_id.values()):
            if poly["status"] in ("added", "edited"):
                poly["status"] = "original"
                poly["original_points"] = poly["points"].copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._save_marks()
        self._save_edits()
        self._refresh_undo_button()
        self._refresh_send_button()
        self.update_filter_counts()
        self.refresh_view()
        self._pull_from_cvat()


class TaskPickerDialog(QDialog):
    """Launch dialog: choose a CVAT task and frame range before loading."""

    def __init__(self, tasks, last_task_id, last_first, last_last):
        """Build the picker: a task dropdown (pre-selected to last time's task),
        first/last frame spin boxes (pre-filled to last time's range), a live
        "loading N frames" label, and a resolved-image-folder line that turns
        red and disables Load when no local folder can be found for the task."""
        super().__init__()
        self.tasks = tasks
        self.setWindowTitle("TrailFixR — Pick CVAT task")
        self.setMinimumWidth(580)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("TrailFixR")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        sub = QLabel("Pick a CVAT task and frame range to review.")
        sub.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(sub)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("CVAT task:"))
        self.task_combo = QComboBox()
        for t in tasks:
            label = f"{t['id']:>3}  —  {t['name']}  ({t['size']} frames)"
            self.task_combo.addItem(label, t["id"])
        for i in range(self.task_combo.count()):
            if self.task_combo.itemData(i) == last_task_id:
                self.task_combo.setCurrentIndex(i)
                break
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        row1.addWidget(self.task_combo, stretch=1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("First frame:"))
        self.first_spin = QSpinBox()
        self.first_spin.setMinimum(1)
        self.first_spin.setMaximum(99999)
        self.first_spin.setValue(max(1, int(last_first)))
        self.first_spin.valueChanged.connect(self._on_range_changed)
        row2.addWidget(self.first_spin)
        row2.addSpacing(20)
        row2.addWidget(QLabel("Last frame:"))
        self.last_spin = QSpinBox()
        self.last_spin.setMinimum(1)
        self.last_spin.setMaximum(99999)
        self.last_spin.setValue(max(1, int(last_last)))
        self.last_spin.valueChanged.connect(self._on_range_changed)
        row2.addWidget(self.last_spin)
        row2.addStretch()
        layout.addLayout(row2)

        self.range_label = QLabel()
        self.range_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.range_label)

        self.folder_label = QLabel()
        self.folder_label.setStyleSheet("color: #666; font-size: 11px; font-family: monospace;")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btns.button(QDialogButtonBox.Ok).setText("Load")
        self.btns.button(QDialogButtonBox.Cancel).setText("Quit")
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        layout.addWidget(self.btns)

        self._on_task_changed(self.task_combo.currentIndex())

    def _on_task_changed(self, idx):
        """When a different task is picked: clamp the frame spin-box ranges to
        that task's frame count, refresh the range label, and resolve the
        task's local image folder (showing it in green, or a red NOT FOUND
        message that disables the Load button)."""
        task = self.tasks[idx] if 0 <= idx < len(self.tasks) else None
        if task is None:
            return
        if task["size"] > 0:
            max_frames = task["size"]
            self.first_spin.setMaximum(max_frames)
            self.last_spin.setMaximum(max_frames)
            if self.first_spin.value() > max_frames:
                self.first_spin.setValue(1)
            if self.last_spin.value() > max_frames:
                self.last_spin.setValue(max_frames)
        self._on_range_changed()
        img_dir = resolve_image_dir(task["name"])
        if img_dir is None:
            self.folder_label.setText(
                f"image folder NOT FOUND for '{task['name']}'\n"
                f"  expected under: {TRAILS_ROOT}")
            self.folder_label.setStyleSheet(
                "color: #c0392b; font-size: 11px; font-family: monospace;")
            self.btns.button(QDialogButtonBox.Ok).setEnabled(False)
        else:
            self.folder_label.setText(f"image folder: {img_dir}")
            self.folder_label.setStyleSheet(
                "color: #2a7a2a; font-size: 11px; font-family: monospace;")
            self.btns.button(QDialogButtonBox.Ok).setEnabled(True)

    def selected_task(self):
        """Return the task dict the user picked, or None."""
        idx = self.task_combo.currentIndex()
        if 0 <= idx < len(self.tasks):
            return self.tasks[idx]
        return None

    def selected_first_frame(self):
        """Return the chosen first frame (1-based, as shown in the spin box)."""
        return int(self.first_spin.value())

    def selected_last_frame(self):
        """Return the chosen last frame (1-based, as shown in the spin box)."""
        return int(self.last_spin.value())

    def _on_range_changed(self):
        """Keep the last-frame value from dropping below the first-frame value
        and update the "loading N frames" label as either spin box changes."""
        if self.last_spin.value() < self.first_spin.value():
            self.last_spin.blockSignals(True)
            self.last_spin.setValue(self.first_spin.value())
            self.last_spin.blockSignals(False)
        n = self.last_spin.value() - self.first_spin.value() + 1
        self.range_label.setText(
            f"  loading {n} frame{'s' if n != 1 else ''} "
            f"({self.first_spin.value()}-{self.last_spin.value()})")


class CvatSendWorker(QThread):
    """Background thread that pushes a batch of annotation changes to CVAT so
    the UI doesn't freeze during the round-trips. Does adds, then edits, then
    deletes, and emits finished(success, error_message) when done."""
    finished = Signal(bool, str)   # success, error_message

    def __init__(self, job_id, task_id, auth, adds, edits, marked_ids):
        """Stash everything the thread needs to run without touching the GUI:
        the job/task ids, CVAT auth, the new polygons to add, the edited
        polygons to update, and the server ids to delete."""
        super().__init__()
        self.job_id = job_id
        self.task_id = task_id
        self.auth = auth
        self.adds = adds          # list of {points, frame_idx}
        self.edits = edits        # list of {server_id, points}
        self.marked_ids = set(marked_ids)

    def run(self):
        """Do the CVAT round-trips on the worker thread: look up the "trail"
        label id, then create the added polygons, update each edited polygon's
        points, and delete the marked shapes — re-fetching the annotation
        version before each PATCH (CVAT requires the current version). Emits
        finished(True, "") on success or finished(False, message) on any
        error."""
        try:
            auth = self.auth

            # Fetch label_id
            task_resp = requests.get(
                f"{CVAT_URL}/api/tasks/{self.task_id}", auth=auth).json()
            label_id = None
            labels_field = task_resp.get("labels")
            label_list = []
            if isinstance(labels_field, dict) and labels_field.get("url"):
                lab_resp = requests.get(labels_field["url"], auth=auth).json()
                label_list = lab_resp.get("results", lab_resp) if isinstance(lab_resp, dict) else lab_resp
            elif isinstance(labels_field, list):
                label_list = labels_field
            for lab in label_list or []:
                if isinstance(lab, dict) and lab.get("name") == "trail":
                    label_id = lab.get("id")
                    break
            if label_id is None and label_list:
                first = label_list[0] if isinstance(label_list[0], dict) else None
                if first:
                    label_id = first.get("id")

            # ADD
            if self.adds:
                shapes_to_add = [{
                    "type": "polygon",
                    "points": [float(v) for pt in poly["points"] for v in pt],
                    "frame": int(poly["frame_idx"]),
                    "label_id": label_id,
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "attributes": [],
                    "group": 0,
                    "source": "manual",
                } for poly in self.adds]
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                    auth=auth).json()
                r = requests.patch(
                    f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                    params={"action": "create"},
                    json={"version": ann_resp.get("version", 0),
                          "tags": [], "shapes": shapes_to_add, "tracks": []},
                    auth=auth)
                r.raise_for_status()

            # EDIT
            for poly in self.edits:
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                    auth=auth).json()
                shape = next((s for s in ann_resp["shapes"]
                              if s["id"] == poly["server_id"]), None)
                if shape is None:
                    continue
                shape = dict(shape)
                shape["points"] = [float(v) for pt in poly["points"] for v in pt]
                r = requests.patch(
                    f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                    params={"action": "update"},
                    json={"version": ann_resp.get("version", 0),
                          "tags": [], "shapes": [shape], "tracks": []},
                    auth=auth)
                r.raise_for_status()

            # DELETE
            if self.marked_ids:
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                    auth=auth).json()
                shapes_to_delete = [s for s in ann_resp["shapes"]
                                    if s["id"] in self.marked_ids]
                if shapes_to_delete:
                    r = requests.patch(
                        f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                        params={"action": "delete"},
                        json={"version": ann_resp.get("version", 0),
                              "tags": [], "shapes": shapes_to_delete, "tracks": []},
                        auth=auth)
                    r.raise_for_status()

            self.finished.emit(True, "")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class SplashWindow(QWidget):
    """Frameless always-on-top splash with a status line and progress bar.
    Shown immediately on launch so the user gets feedback while CVAT is being
    fetched and 4,000+ polygons are cropped."""

    def __init__(self):
        """Build the frameless dark splash: title, subtitle, a status line, an
        (initially indeterminate) progress bar, and a Cancel button that quits
        the app."""
        super().__init__()
        # Qt.SplashScreen on macOS is fragile - the OS treats SplashScreen-flagged
        # widgets as transient and can dismiss them on focus events, producing
        # the "splash flashed and disappeared" symptom. Use a regular Tool
        # window with frameless + stay-on-top instead; behaves predictably.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(420, 160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        title = QLabel("TrailFixR")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Loading polygons from CVAT")
        sub.setStyleSheet("font-size: 12px; color: #a8c0e0;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("font-size: 12px; color: #e6e6e6;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate until total is known
        self.progress.setFixedHeight(14)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { border: 1px solid #1a3a5c; border-radius: 7px; "
            "background: #0a1e3f; }"
            "QProgressBar::chunk { background: #1a6fc4; border-radius: 6px; }"
        )
        layout.addWidget(self.progress)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(28)
        cancel_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #a8c0e0; font-size: 12px; "
            "border: 1px solid #1a3a5c; border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { color: white; border-color: #4a7cbf; }"
        )
        cancel_btn.clicked.connect(lambda: QApplication.instance().quit())
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setStyleSheet("background-color: #0a1e3f; border-radius: 12px;")

    def update_progress(self, text, current=0, total=0):
        """Update the splash status line and progress bar (a known total shows a
        real percentage; total=0 shows the indeterminate scrolling bar), then
        pump the event loop so it actually repaints. This is the progress_cb
        passed into load_and_analyze."""
        self.status_label.setText(text)
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        else:
            # indeterminate scrolling bar
            self.progress.setRange(0, 0)
        QApplication.processEvents()


def _play_end_sound():
    """Play a soft system Tink sound — used to signal you've hit the first or
    last polygon and can't navigate further that way."""
    subprocess.Popen(["afplay", "-v", "0.1",
                      "/System/Library/Sounds/Tink.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    """App entry point. Takes the single-instance lock, fetches the CVAT task
    list (erroring out if Docker/CVAT is unreachable), shows the task picker,
    resolves the chosen task's image folder, sets the module-level task/frame
    globals, then shows the splash while load_and_analyze crops every polygon
    and finally opens the TrailFixR editor window."""
    global CVAT_TASK_ID, FRAME_START, FRAME_END, IMG_DIR, TASK_NAME

    print("TrailFixR 2")
    print("=" * 60, flush=True)

    app = QApplication(sys.argv)
    _apply_theme()

    # Single-instance lock.
    lock_dir = Path.home() / ".star_trail_cleanr"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(lock_dir / "trailscreenr.lock"))
    lock.setStaleLockTime(10_000)
    if not lock.tryLock(1500):
        QMessageBox.warning(None, "TrailFixR already running",
                            "Another TrailFixR is already running.\n\n"
                            "Switch to that window or close it first.\n"
                            "(If this looks wrong, the previous run may have "
                            "left a stale lock file - quit any running "
                            "TrailFixR and try again in 10 seconds.)")
        return

    # Fetch task list from CVAT. If this fails, Docker/CVAT isn't running.
    auth = (CVAT_USER, read_cvat_password())
    print("Fetching CVAT task list...", flush=True)
    tasks = fetch_cvat_tasks(auth)
    if not tasks:
        QMessageBox.critical(None, "Cannot reach CVAT",
            f"Could not connect to CVAT at {CVAT_URL}.\n\n"
            "Make sure Docker Desktop is running and try again.")
        return

    # Task picker dialog.
    last = load_last_pick()
    picker = TaskPickerDialog(
        tasks,
        last["task_id"],
        last.get("first_frame", 1),
        last.get("last_frame", 9999),
    )
    if picker.exec() != QDialog.Accepted:
        print("Cancelled.")
        return
    chosen = picker.selected_task()
    first_f = picker.selected_first_frame()
    last_f = picker.selected_last_frame()
    if chosen is None:
        return

    img_dir = resolve_image_dir(chosen["name"])
    if img_dir is None:
        QMessageBox.critical(None, "Image folder not found",
            f"Couldn't find an image folder for '{chosen['name']}' under "
            f"{TRAILS_ROOT}.\n\nPick a different task or fix the folder mapping.")
        return

    CVAT_TASK_ID = chosen["id"]
    FRAME_START = first_f - 1
    FRAME_END = last_f
    IMG_DIR = img_dir
    TASK_NAME = chosen["name"]
    _refresh_paths()
    save_last_pick(CVAT_TASK_ID, first_f, last_f)
    print(f"  task {CVAT_TASK_ID}: {TASK_NAME}", flush=True)
    print(f"  frames {FRAME_START}-{FRAME_END - 1} from {IMG_DIR}", flush=True)

    # Splash while polygons load (~5s cached, ~60-90s cold).
    splash = SplashWindow()
    splash.show()
    splash.update_progress("Loading polygons from CVAT...")

    entries, job_id = load_and_analyze(progress_cb=splash.update_progress)
    print(f"\nLoaded {len(entries)} polygons.")

    if not entries:
        splash.close()
        QMessageBox.information(None, "No polygons",
            "No polygons found for this task / frame range.")
        return

    splash.update_progress("Building the editor window...")
    window = TrailFixR(entries, job_id)
    splash.close()
    window.show()
    window.activateWindow()
    window.setFocus()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
