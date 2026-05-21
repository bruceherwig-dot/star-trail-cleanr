#!/usr/bin/env python3
"""
TileFixR — Per-tile CVAT polygon editor.

Sister tool of TileScreenR. Same per-tile view (walks 640x640 windows in
each frame), but pulls polygons from CVAT instead of from inference masks
and lets you click an individual polygon to mark it for deletion. Then
push deletions back to CVAT in one batch.

Phase 1 (this build): mark-for-delete + push.
Phase 2 (later):      drag-to-move vertices, push edits via PATCH.

Hardcoded for now: CVAT task 15 (Greg Meyer Arizona Brightened), first 20
frames. Future: task pulldown + frame-count picker.
"""
import json
import os
import random
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import requests
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.geometry import MultiPolygon, GeometryCollection
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QButtonGroup, QMessageBox, QFrame,
    QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLineEdit, QDialog, QComboBox, QSpinBox, QDialogButtonBox, QProgressBar
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut, QPainter, QPen, QColor, QTransform
)
from PySide6.QtCore import Qt, QRectF, QEvent, Signal, QLockFile, QTimer, QThread


# ── Config ──────────────────────────────────────────────────────────────────
CVAT_URL = "http://localhost:8080"
CVAT_USER = "bherwig2"
TRAILS_ROOT = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images")
GKYLE_STAGING = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/external_datasets/gkyle_startrails/cvat_staging")
TILE_SIZE = 640
OVERLAP = 0.2

# These globals are set by the launch picker (or fallback defaults below).
CVAT_TASK_ID = 16
FRAME_START = 0
FRAME_END = 9999            # exclusive; overridden by picker or SUSPECT_FRAMES
IMG_DIR = GKYLE_STAGING
TASK_NAME = ""
WEIRDR_PATH = Path(__file__).parent.parent / "weirdr_list.json"

# When non-empty, only these CVAT frame indices are loaded and tiled.
SUSPECT_FRAMES = []

STATE_DIR = Path.home() / ".star_trail_cleanr"
STATE_PATH = STATE_DIR / f"tile_fixr_state_task{CVAT_TASK_ID}.json"
MARKS_PATH = STATE_DIR / f"tile_fixr_marks_task{CVAT_TASK_ID}.json"
LAST_PICK_PATH = STATE_DIR / "tile_fixr_last_task.json"


def _refresh_paths():
    """Recompute STATE_PATH and MARKS_PATH after CVAT_TASK_ID changes."""
    global STATE_PATH, MARKS_PATH
    STATE_PATH = STATE_DIR / f"tile_fixr_state_task{CVAT_TASK_ID}.json"
    MARKS_PATH = STATE_DIR / f"tile_fixr_marks_task{CVAT_TASK_ID}.json"


TASK_FOLDER_ALIASES = {
    "My First Star Trail": "Bruce Herwig - first star trail data",
    "Thomas Jackson - Borrego": "Thomas Jackson Star Trails Borrego",
    "Greg Meyer - Arizona": "Greg Meyer Arizona",
    "Bruce Herwig - Pioneertown Fisheye": "Pioneertown 6mm Fisheye Training",
    "Bruce Herwig - Borrego Springs 1": "borrego_springs_1",
}


def resolve_image_dir(task_name):
    """Best-effort: find the local image folder for a CVAT task.
    Tries exact match, then strips ' - v...' suffix, then any prefix match.
    Also checks GKYLE_STAGING for gkyle tasks."""
    # Special case: gkyle tiles live outside TRAILS_ROOT
    if "gkyle" in task_name.lower() and GKYLE_STAGING.exists():
        return GKYLE_STAGING
    if task_name in TASK_FOLDER_ALIASES:
        p = TRAILS_ROOT / TASK_FOLDER_ALIASES[task_name]
        if p.exists():
            return p
    candidates = [task_name]
    # Strip " - v..." suffix (e.g. " - v8 slope-match", " - v3 inference")
    if " - v" in task_name:
        candidates.append(task_name.split(" - v")[0].rstrip())
    for c in candidates:
        p = TRAILS_ROOT / c
        if p.exists():
            return p
    # Last try: any folder that contains the (truncated) task name
    base = candidates[-1]
    if TRAILS_ROOT.exists():
        for child in TRAILS_ROOT.iterdir():
            if child.is_dir() and base.lower() in child.name.lower():
                return child
    return None


POLY_COLORS = [
    (0, 0, 255),      # red
    (255, 0, 255),    # magenta
    (0, 255, 0),      # green
    (0, 165, 255),    # orange
    (255, 255, 0),    # yellow
    (255, 105, 180),  # hot pink
    (147, 20, 255),   # deep pink
    (0, 215, 255),    # gold
    (255, 191, 0),    # deep sky blue
    (50, 205, 50),    # lime green
    (255, 100, 100),  # light red
    (100, 255, 100),  # light green
]
DELETE_COLOR = (0, 0, 255)        # bright red for marked-for-delete overlay
SELECTED_COLOR = (255, 255, 255)  # white thick outline for currently selected

# Theme defaults (light), updated by _apply_theme()
MUTED_TEXT = "#666"
HINT_TEXT = "#888"
PANEL_BG = "#f5f5f5"
IMAGE_BG = "#1a1a2e"
INFO_BG = "#f0f0f0"
INFO_TEXT = "#000"


def _apply_theme():
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


def read_cvat_password():
    return (Path.home() / ".star_trail_cleanr" / "cvat_credentials").read_text().strip()


# ── CVAT loading ────────────────────────────────────────────────────────────

def load_cvat_polygons(progress_cb=None):
    """Pull polygons from CVAT for FRAME_START..FRAME_END.

    Returns dict: frame_idx -> list of {server_id, points (Nx2 float), label}
    Plus: list of frame names (one per index), and the job_id for deep links.
    """
    if progress_cb:
        progress_cb("Connecting to CVAT and fetching annotations...", 0, 0)
    password = read_cvat_password()
    auth = (CVAT_USER, password)

    print(f"Connecting to CVAT task {CVAT_TASK_ID}...", flush=True)
    jobs_resp = requests.get(f"{CVAT_URL}/api/jobs",
                              params={"task_id": CVAT_TASK_ID}, auth=auth).json()
    job = jobs_resp["results"][0]
    job_id = job["id"]

    meta_resp = requests.get(f"{CVAT_URL}/api/jobs/{job_id}/data/meta",
                              auth=auth).json()
    frame_names = [f["name"] for f in meta_resp["frames"]]

    ann_resp = requests.get(f"{CVAT_URL}/api/jobs/{job_id}/annotations",
                              auth=auth).json()

    suspect_set = set(SUSPECT_FRAMES) if SUSPECT_FRAMES else None

    by_frame = {}
    for shape in ann_resp.get("shapes", []):
        if shape.get("type") not in ("polygon", "mask"):
            continue
        frame_idx = shape["frame"]
        if suspect_set is not None:
            if frame_idx not in suspect_set:
                continue
        elif frame_idx < FRAME_START or frame_idx >= FRAME_END:
            continue
        pts_flat = shape.get("points", [])
        if len(pts_flat) < 6:
            continue
        pts = np.array(pts_flat, dtype=np.float32).reshape(-1, 2)
        by_frame.setdefault(frame_idx, []).append({
            "server_id": shape["id"],
            "points": pts,
            "label": shape.get("label_id"),
            "frame_idx": frame_idx,
        })

    n = sum(len(v) for v in by_frame.values())
    if progress_cb:
        progress_cb(f"Got {n} polygons. Building tiles...", 0, 0)
    if SUSPECT_FRAMES:
        print(f"Loaded {n} polygons across {len(by_frame)} suspect frames "
              f"(task {CVAT_TASK_ID}, {len(SUSPECT_FRAMES)} frame list)", flush=True)
    else:
        print(f"Loaded {n} polygons across "
              f"{len(by_frame)} frames (task {CVAT_TASK_ID}, "
              f"frames {FRAME_START}-{FRAME_END - 1})", flush=True)
    return by_frame, frame_names, job_id


# ── Tile entry build ────────────────────────────────────────────────────────

def clip_polygon_to_tile(ff_pts: np.ndarray, tx: int, ty: int,
                          tx2: int, ty2: int):
    """Geometric clip of a full-frame polygon to a tile rectangle.

    Returns a list of int32 (N,2) arrays in tile-local coords (one per
    output ring; usually one). Preserves original polygon vertices and
    only adds new vertices where the polygon crosses the tile boundary.

    Empty list if the polygon doesn't intersect the tile at all.
    """
    if ff_pts.shape[0] < 3:
        return []
    try:
        poly = ShapelyPolygon(ff_pts.tolist())
        if not poly.is_valid:
            poly = poly.buffer(0)  # repair self-intersections
    except Exception:
        return []
    tile_box = shapely_box(tx, ty, tx2, ty2)
    inter = poly.intersection(tile_box)
    if inter.is_empty:
        return []
    rings = []
    geoms = []
    if isinstance(inter, ShapelyPolygon):
        geoms = [inter]
    elif isinstance(inter, (MultiPolygon, GeometryCollection)):
        geoms = [g for g in inter.geoms if isinstance(g, ShapelyPolygon)]
    for g in geoms:
        coords = list(g.exterior.coords)
        # Shapely closes the ring (last == first); drop the duplicate
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        local = np.array([(int(round(x - tx)), int(round(y - ty)))
                           for x, y in coords], dtype=np.int32)
        rings.append(local)
    return rings


def _tile_origins(extent: int, tile: int, stride: int) -> list:
    o = list(range(0, max(extent - tile, 0) + 1, stride))
    last = max(extent - tile, 0)
    if not o or o[-1] != last:
        o.append(last)
    return sorted(set(o))


def build_tile_entries_from_masks(masks_dir, frame_names, progress_cb=None):
    """Like build_tile_entries but loads mask PNGs instead of CVAT polygons.

    Reads IMG_DIR for tile images (should already point to cleaned folder).
    For each tile, finds contours in the mask crop and stores them as
    read-only polygon entries with server_id=-1.
    """
    img_files = sorted(IMG_DIR.glob("*.jpg")) or sorted(IMG_DIR.glob("*.JPG"))
    if not img_files:
        sub = IMG_DIR / "JPGs"
        img_files = sorted(sub.glob("*.jpg")) or sorted(sub.glob("*.JPG"))
    if not img_files:
        raise SystemExit(f"No cleaned frames in {IMG_DIR}")

    img_by_name = {p.name: p for p in img_files}
    stride = int(TILE_SIZE * (1 - OVERLAP))
    entries = []
    frame_list = SUSPECT_FRAMES if SUSPECT_FRAMES else list(range(FRAME_START, FRAME_END))

    for pos, frame_idx in enumerate(frame_list):
        if frame_idx >= len(frame_names):
            continue
        img_path = img_by_name.get(frame_names[frame_idx])
        if img_path is None:
            continue
        stem = img_path.stem

        sys.stdout.write(f"\r  Frame {pos + 1}/{len(frame_list)}  {img_path.name}   ")
        sys.stdout.flush()
        if progress_cb and (pos % 5 == 0):
            if progress_cb(
                    f"Building tiles: frame {pos + 1} of {len(frame_list)}...",
                    pos + 1, len(frame_list)):
                return [], []

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        mask_path = masks_dir / (stem + ".png")
        frame_mask = None
        if mask_path.exists():
            frame_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        xs_o = _tile_origins(W, TILE_SIZE, stride)
        ys_o = _tile_origins(H, TILE_SIZE, stride)
        n_cols = len(xs_o)
        n_rows = len(ys_o)

        for row_idx, ty in enumerate(ys_o):
            for col_idx, tx in enumerate(xs_o):
                tx2 = min(tx + TILE_SIZE, W)
                ty2 = min(ty + TILE_SIZE, H)
                crop = img[ty:ty2, tx:tx2].copy()
                if crop.shape[:2] != (TILE_SIZE, TILE_SIZE):
                    pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    pad[:crop.shape[0], :crop.shape[1]] = crop
                    crop = pad

                tile_polys = []
                if frame_mask is not None:
                    tile_mask = frame_mask[ty:ty2, tx:tx2]
                    if tile_mask.max() > 0:
                        contours, _ = cv2.findContours(
                            tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if contour.shape[0] < 3:
                                continue
                            pts = contour.reshape(-1, 2).astype(np.int32)
                            tile_polys.append({
                                "server_id": -1,
                                "color_index": 0,
                                "tile_local_pts": pts,
                                "frame_idx": frame_idx,
                            })

                tile_id = f"{stem}_t{tx:04d}_{ty:04d}"
                entries.append({
                    "frame": stem,
                    "frame_idx": frame_idx,
                    "img_filename": img_path.name,
                    "tile_x": tx,
                    "tile_y": ty,
                    "tile_col": col_idx,
                    "tile_row": row_idx,
                    "n_cols": n_cols,
                    "n_rows": n_rows,
                    "label": f"{stem} tile ({tx},{ty})",
                    "crop_raw": crop,
                    "polys": tile_polys,
                    "tile_id": tile_id,
                })
    print()
    return entries, []


def build_tile_entries(by_frame, frame_names, progress_cb=None):
    """Walk every 640x640 tile in each frame in [FRAME_START, FRAME_END).
    For each tile, list the polygons that intersect it (with tile-local
    coords). Also assign a stable color per polygon by its server_id."""
    img_files = sorted(IMG_DIR.glob("*.jpg")) or sorted(IMG_DIR.glob("*.JPG"))
    if not img_files:
        sub = IMG_DIR / "JPGs"
        img_files = sorted(sub.glob("*.jpg")) or sorted(sub.glob("*.JPG"))
    if not img_files:
        raise SystemExit(f"No source frames in {IMG_DIR}")

    # Name-based lookup so deleted or missing frames don't shift positions.
    img_by_name = {p.name: p for p in img_files}

    stride = int(TILE_SIZE * (1 - OVERLAP))
    entries = []

    # Stable color per polygon — by index across all polygons sorted by
    # (frame_idx, server_id).
    all_polys = []
    for fi in sorted(by_frame.keys()):
        for poly in sorted(by_frame[fi], key=lambda p: p["server_id"]):
            all_polys.append(poly)
            poly["color_index"] = len(all_polys) - 1

    frame_list = SUSPECT_FRAMES if SUSPECT_FRAMES else list(range(FRAME_START, FRAME_END))
    for pos, frame_idx in enumerate(frame_list):
        if frame_idx >= len(frame_names):
            continue
        img_path = img_by_name.get(frame_names[frame_idx])
        if img_path is None:
            continue
        stem = img_path.stem
        sys.stdout.write(f"\r  Frame {pos + 1}/{len(frame_list)}  {img_path.name}   ")
        sys.stdout.flush()
        if progress_cb and (pos % 5 == 0):
            if progress_cb(
                    f"Building tiles: frame {pos + 1} of {len(frame_list)}...",
                    pos + 1, len(frame_list)):
                return [], []

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        polys_for_frame = by_frame.get(frame_idx, [])

        xs_o = _tile_origins(W, TILE_SIZE, stride)
        ys_o = _tile_origins(H, TILE_SIZE, stride)
        n_cols = len(xs_o)
        n_rows = len(ys_o)
        for row_idx, ty in enumerate(ys_o):
            for col_idx, tx in enumerate(xs_o):
                tx2 = min(tx + TILE_SIZE, W)
                ty2 = min(ty + TILE_SIZE, H)
                crop = img[ty:ty2, tx:tx2].copy()
                if crop.shape[:2] != (TILE_SIZE, TILE_SIZE):
                    pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    pad[:crop.shape[0], :crop.shape[1]] = crop
                    crop = pad

                # Clip each polygon to this tile (full coords)
                tile_polys = []
                for poly in polys_for_frame:
                    pts = poly["points"]
                    if pts.shape[0] < 3:
                        continue
                    px0, py0 = pts[:, 0].min(), pts[:, 1].min()
                    px1, py1 = pts[:, 0].max(), pts[:, 1].max()
                    if px1 <= tx or px0 >= tx2 or py1 <= ty or py0 >= ty2:
                        continue
                    # Geometric clip — preserves the original CVAT vertex
                    # positions, only adds intersections with the tile edge.
                    rings = clip_polygon_to_tile(pts, tx, ty, tx2, ty2)
                    for c_pts in rings:
                        tile_polys.append({
                            "server_id": poly["server_id"],
                            "color_index": poly["color_index"],
                            "tile_local_pts": c_pts,
                            "frame_idx": frame_idx,
                        })

                tile_id = f"{stem}_t{tx:04d}_{ty:04d}"
                entries.append({
                    "frame": stem,
                    "frame_idx": frame_idx,
                    "img_filename": img_path.name,
                    "tile_x": tx,
                    "tile_y": ty,
                    "tile_col": col_idx,
                    "tile_row": row_idx,
                    "n_cols": n_cols,
                    "n_rows": n_rows,
                    "label": f"{stem} tile ({tx},{ty})",
                    "crop_raw": crop,
                    "polys": tile_polys,
                    "tile_id": tile_id,
                })
    print()
    return entries, all_polys


# ── Zoomable image view with click signal ───────────────────────────────────

class ZoomableImageView(QGraphicsView):
    """QGraphicsView with pinch-to-zoom, scroll-to-zoom, click + drag signals.

    Emits at image-pixel coords:
      pressed_at_image_xy(int, int)        — left button on press
      moved_to_image_xy(int, int)          — mouse move while button held
      released_at_image_xy(int, int)       — mouse release
      clicked_at_image_xy(int, int)        — release within ~4px of press (no drag)
      double_clicked_at_image_xy(int, int) — left-button double click
      right_clicked_at_image_xy(int, int)  — right-button click

    Drag-to-pan is intentionally suppressed so the parent can interpret presses
    as polygon-edit gestures. Use the scroll bars or the wheel for navigation.
    """
    clicked_at_image_xy = Signal(int, int)
    pressed_at_image_xy = Signal(int, int)
    moved_to_image_xy = Signal(int, int)
    released_at_image_xy = Signal(int, int)
    double_clicked_at_image_xy = Signal(int, int)
    right_clicked_at_image_xy = Signal(int, int)
    hovered_at_image_xy = Signal(int, int)
    zoom_changed = Signal()    # emitted whenever the view's scale changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # Default to ScrollHandDrag so the user can left-drag to pan when
        # zoomed in. Parent will temporarily flip this to NoDrag during a
        # polygon-vertex grab.
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(400)
        self._zoom_level = 0
        self.draw_mode = False   # when True, clicks outside image bounds are allowed
        self._rb_line  = None   # QGraphicsLineItem for the add-mode rubber-band
        self._press_pos = None
        self._press_button = None
        self._is_dragging = False
        self.grabGesture(Qt.PinchGesture)
        # Mouse-tracking for hover-cursor handling.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    _DRAW_PAD = 24

    def set_pixmap(self, pixmap):
        """Replace the pixmap AND reset view to fit (use on entry change)."""
        self._pixmap_item.setPixmap(pixmap)
        pad = self._DRAW_PAD
        self._scene.setSceneRect(QRectF(pixmap.rect()).adjusted(-pad, -pad, pad, pad))
        self._zoom_level = 0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def update_pixmap(self, pixmap):
        """Replace the pixmap WITHOUT resetting zoom or transform.
        Use this for incremental redraws during a vertex drag so the user's
        zoom/pan stays put."""
        self._pixmap_item.setPixmap(pixmap)

    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self._handle_gesture(event)
        return super().event(event)

    def _handle_gesture(self, event):
        pinch = event.gesture(Qt.PinchGesture)
        if pinch is None:
            return False
        factor = pinch.scaleFactor()
        if factor != 1.0:
            self._zoom_level += 1 if factor > 1.0 else -1
            self.scale(factor, factor)
        return True

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.10 if delta > 0 else 0.91
        self._zoom_level += 1 if delta > 0 else -1
        self.scale(factor, factor)
        self.zoom_changed.emit()

    def current_scale(self):
        """Current view-to-scene scale factor (>1 = zoomed in)."""
        return float(self.transform().m11()) or 1.0

    def _to_scene_xy(self, qpoint):
        scene_pt = self.mapToScene(qpoint)
        return int(scene_pt.x()), int(scene_pt.y())

    def _to_image_xy(self, qpoint):
        scene_pt = self.mapToScene(qpoint)
        ix = int(scene_pt.x())
        iy = int(scene_pt.y())
        if not (0 <= ix < self._pixmap_item.pixmap().width()
                and 0 <= iy < self._pixmap_item.pixmap().height()):
            return None
        return ix, iy

    def set_rubber_band(self, x1, y1, x2, y2):
        pen = QPen(QColor(50, 255, 50), 1, Qt.SolidLine)
        pen.setCosmetic(True)
        if self._rb_line is None:
            self._rb_line = self._scene.addLine(x1, y1, x2, y2, pen)
        else:
            self._rb_line.setLine(x1, y1, x2, y2)
            self._rb_line.setPen(pen)

    def clear_rubber_band(self):
        if self._rb_line is not None:
            self._scene.removeItem(self._rb_line)
            self._rb_line = None

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        xy = self._to_image_xy(pos)
        if xy is None and self.draw_mode:
            xy = self._to_scene_xy(pos)
        if event.button() == Qt.LeftButton and xy is not None:
            self._press_pos = pos
            self._press_button = Qt.LeftButton
            self._is_dragging = False
            self.pressed_at_image_xy.emit(xy[0], xy[1])
            # event.accept() keeps Qt tracking the mouse for subsequent
            # move/release events without invoking QGraphicsView's drag-mode
            # machinery (which can fight our manual pan / vertex grab logic).
            event.accept()
            return
        if event.button() == Qt.RightButton and xy is not None:
            self.right_clicked_at_image_xy.emit(xy[0], xy[1])
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_button == Qt.LeftButton and self._press_pos is not None:
            pos = event.position().toPoint()
            xy = self._to_image_xy(pos)
            if xy is None and self.draw_mode:
                xy = self._to_scene_xy(pos)
            if xy is not None:
                dx = pos.x() - self._press_pos.x()
                dy = pos.y() - self._press_pos.y()
                if abs(dx) >= 10 or abs(dy) >= 10:
                    self._is_dragging = True
                self.moved_to_image_xy.emit(xy[0], xy[1])
            # Manual pan: when the parent hasn't claimed this gesture (drag mode
            # is still ScrollHandDrag), scroll the view by the cursor delta.
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
        # No button: hover — emit so parent can update cursor over handles
        xy = self._to_image_xy(event.position().toPoint())
        if xy is None and self.draw_mode:
            xy = self._to_scene_xy(event.position().toPoint())
        if xy is not None:
            self.hovered_at_image_xy.emit(xy[0], xy[1])
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._press_pos is not None:
            release_pos = event.position().toPoint()
            xy = self._to_image_xy(release_pos)
            if xy is None and self.draw_mode:
                xy = self._to_scene_xy(release_pos)
            if xy is not None:
                if not self._is_dragging:
                    dx = release_pos.x() - self._press_pos.x()
                    dy = release_pos.y() - self._press_pos.y()
                    # 10px slop instead of 4 so Mac trackpad taps still
                    # register as clicks. A soft tap drifts 5-10 px between
                    # press and release; the old threshold ignored those as
                    # tiny drags. Real drags exceed 10 px easily.
                    if abs(dx) < 10 and abs(dy) < 10:
                        self.clicked_at_image_xy.emit(xy[0], xy[1])
                self.released_at_image_xy.emit(xy[0], xy[1])
            self._press_pos = None
            self._press_button = None
            self._is_dragging = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            xy = self._to_image_xy(event.position().toPoint())
            if xy is not None:
                self.double_clicked_at_image_xy.emit(xy[0], xy[1])
        self._zoom_level = 0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom_level == 0 and self._pixmap_item.pixmap() and not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


# ── Task picker dialog ──────────────────────────────────────────────────────

def fetch_cvat_tasks(auth):
    """List all CVAT tasks. Returns list of {id, name, size}, name-sorted."""
    out = []
    url = f"{CVAT_URL}/api/tasks"
    while url:
        try:
            r = requests.get(url, auth=auth).json()
        except Exception as e:
            print(f"fetch_cvat_tasks error: {e}")
            break
        for t in r.get("results", []):
            out.append({
                "id": t["id"],
                "name": t["name"],
                "size": t.get("size", 0),
            })
        url = r.get("next")
    out.sort(key=lambda t: t["name"].lower())
    return out


def load_last_pick():
    if LAST_PICK_PATH.exists():
        try:
            data = json.loads(LAST_PICK_PATH.read_text())
            # Back-compat: older state used 'n_frames' (count from 0)
            if "first_frame" not in data and "n_frames" in data:
                data["first_frame"] = 0
                data["last_frame"] = max(0, int(data["n_frames"]) - 1)
            return data
        except Exception:
            pass
    return {"task_id": 16, "first_frame": 0, "last_frame": 19}


def save_last_pick(task_id, first_frame, last_frame, source_mode="cvat"):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PICK_PATH.write_text(json.dumps({
        "task_id": task_id,
        "first_frame": int(first_frame),
        "last_frame": int(last_frame),
        "source_mode": source_mode,
    }, indent=2))


class TaskPickerDialog(QDialog):
    """Modal dialog shown at launch — pick a CVAT task + how many frames to load."""

    def __init__(self, tasks, last_task_id, last_first, last_last,
                 auth=None, last_source_mode="cvat"):
        super().__init__()
        self.tasks        = tasks
        self.auth         = auth
        self.last_task_id = last_task_id
        self.setWindowTitle("TileFixR — Pick CVAT task")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("TileFixR")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        sub = QLabel("Pick a CVAT task and a frame range to load.")
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
        self.folder_label.setStyleSheet("color: #666; font-size: 11px; "
                                         "font-family: monospace;")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        # Source mode radio buttons
        from PySide6.QtWidgets import QButtonGroup, QRadioButton
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Image source:"))
        self._rb_cvat    = QRadioButton("Source + CVAT polygons")
        self._rb_cleaned = QRadioButton("Cleaned + STC masks  (view only)")
        self._rb_cvat.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_cvat)
        mode_group.addButton(self._rb_cleaned)
        mode_row.addWidget(self._rb_cvat)
        mode_row.addWidget(self._rb_cleaned)
        mode_row.addStretch()
        layout.addLayout(mode_row)
        if last_source_mode == "cleaned":
            self._rb_cleaned.setChecked(True)

        # Buttons
        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btns.button(QDialogButtonBox.Ok).setText("Load")
        self.btns.button(QDialogButtonBox.Cancel).setText("Quit")
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        layout.addWidget(self.btns)

        self._on_task_changed(self.task_combo.currentIndex())

    def _on_task_changed(self, idx):
        task = self.tasks[idx] if 0 <= idx < len(self.tasks) else None
        if task is None:
            return
        # Fetch deleted_frames to get accurate count, then update the combo label.
        effective_size = task["size"]
        if self.auth and task["size"] > 0:
            try:
                meta = requests.get(
                    f"{CVAT_URL}/api/tasks/{task['id']}/data/meta",
                    auth=self.auth).json()
                deleted = meta.get("deleted_frames", [])
                effective_size = task["size"] - len(deleted)
                new_label = (f"{task['id']:>3}  —  {task['name']}  "
                             f"({effective_size} frames)")
                self.task_combo.setItemText(idx, new_label)
            except Exception:
                pass
        # Reset to 1/max when switching to a different task; restore saved values
        # when reopening on the same task.
        if effective_size > 0:
            max_frames = effective_size
            self.first_spin.setMaximum(max_frames)
            self.last_spin.setMaximum(max_frames)
            if task["id"] != self.last_task_id:
                self.first_spin.setValue(1)
                self.last_spin.setValue(max_frames)
            else:
                if self.first_spin.value() > max_frames:
                    self.first_spin.setValue(1)
                if self.last_spin.value() > max_frames:
                    self.last_spin.setValue(max_frames)
        self._on_range_changed()
        # Resolve image dir; warn if missing
        img_dir = resolve_image_dir(task["name"])
        if img_dir is None:
            self.folder_label.setText(
                f"⚠ image folder NOT FOUND for '{task['name']}'.\n"
                f"  expected under: {TRAILS_ROOT}\n"
                f"  pick a different task or fix the folder mapping.")
            self.folder_label.setStyleSheet(
                "color: #c0392b; font-size: 11px; font-family: monospace;")
            self.btns.button(QDialogButtonBox.Ok).setEnabled(False)
            self._rb_cleaned.setEnabled(False)
        else:
            masks_dir = img_dir / "cleaned" / "masks"
            has_masks = masks_dir.is_dir()
            self._rb_cleaned.setEnabled(has_masks)
            if not has_masks and self._rb_cleaned.isChecked():
                self._rb_cvat.setChecked(True)
            self.folder_label.setText(
                f"image folder: {img_dir}"
                + (f"\ncleaned/masks: {masks_dir}" if has_masks
                   else "\ncleaned/masks: not found"))
            self.folder_label.setStyleSheet(
                "color: #2a7a2a; font-size: 11px; font-family: monospace;")
            self.btns.button(QDialogButtonBox.Ok).setEnabled(True)

    def selected_task(self):
        idx = self.task_combo.currentIndex()
        if 0 <= idx < len(self.tasks):
            return self.tasks[idx]
        return None

    def selected_mode(self):
        return "cleaned" if self._rb_cleaned.isChecked() else "cvat"

    def selected_first_frame(self):
        return int(self.first_spin.value())

    def selected_last_frame(self):
        return int(self.last_spin.value())

    def _on_range_changed(self):
        if (self.last_spin.value() < self.first_spin.value()
                and not self.last_spin.hasFocus()):
            self.last_spin.blockSignals(True)
            self.last_spin.setValue(self.first_spin.value())
            self.last_spin.blockSignals(False)
        n = max(1, self.last_spin.value() - self.first_spin.value() + 1)
        self.range_label.setText(
            f"  → loading {n} frame{'s' if n != 1 else ''} "
            f"({self.first_spin.value()}–{self.last_spin.value()})")


# ── CVAT background send worker ─────────────────────────────────────────────

class CvatSendWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, task_id, job_id, marked_ids, adds, edits, auth):
        super().__init__()
        self.task_id = task_id
        self.job_id = job_id
        self.marked_ids = set(marked_ids)
        self.adds = adds
        self.edits = edits
        self.auth = auth

    def run(self):
        try:
            added_ok = edited_ok = deleted_ok = 0
            errors = []
            auth = self.auth
            job_id = self.job_id

            label_id = None
            task_resp = requests.get(
                f"{CVAT_URL}/api/tasks/{self.task_id}", auth=auth).json()
            labels_field = task_resp.get("labels")
            label_list = []
            if isinstance(labels_field, dict) and labels_field.get("url"):
                lab_resp = requests.get(labels_field["url"], auth=auth).json()
                label_list = (lab_resp.get("results", lab_resp)
                              if isinstance(lab_resp, dict) else lab_resp)
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

            if self.adds:
                shapes_to_add = []
                for poly in self.adds:
                    shapes_to_add.append({
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
                    })
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{job_id}/annotations", auth=auth).json()
                payload = {"version": ann_resp.get("version", 0),
                           "tags": [], "shapes": shapes_to_add, "tracks": []}
                r = requests.patch(
                    f"{CVAT_URL}/api/jobs/{job_id}/annotations",
                    params={"action": "create"}, json=payload, auth=auth)
                r.raise_for_status()
                added_ok = len(shapes_to_add)

            for poly in self.edits:
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{job_id}/annotations", auth=auth).json()
                shape = None
                for s in ann_resp["shapes"]:
                    if s["id"] == poly["server_id"]:
                        shape = s
                        break
                if shape is None:
                    errors.append(f"edit id {poly['server_id']} not found in CVAT")
                    continue
                shape = dict(shape)
                shape["points"] = [float(v) for pt in poly["points"] for v in pt]
                payload = {"version": ann_resp.get("version", 0),
                           "tags": [], "shapes": [shape], "tracks": []}
                r = requests.patch(
                    f"{CVAT_URL}/api/jobs/{job_id}/annotations",
                    params={"action": "update"}, json=payload, auth=auth)
                if r.status_code >= 400:
                    errors.append(f"edit id {poly['server_id']}: {r.text[:200]}")
                else:
                    edited_ok += 1

            if self.marked_ids:
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{job_id}/annotations", auth=auth).json()
                shapes_to_delete = [s for s in ann_resp["shapes"]
                                    if s["id"] in self.marked_ids]
                if shapes_to_delete:
                    payload = {"version": ann_resp.get("version", 0),
                               "tags": [], "shapes": shapes_to_delete, "tracks": []}
                    r = requests.patch(
                        f"{CVAT_URL}/api/jobs/{job_id}/annotations",
                        params={"action": "delete"}, json=payload, auth=auth)
                    r.raise_for_status()
                    deleted_ok = len(shapes_to_delete)

            self.finished.emit({
                "added_ok": added_ok,
                "edited_ok": edited_ok,
                "deleted_ok": deleted_ok,
                "errors": errors,
            })
        except Exception as exc:
            self.error.emit(str(exc))


def _play_end_sound():
    subprocess.Popen(["afplay", "-v", "0.1",
                      "/System/Library/Sounds/Tink.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Main window ─────────────────────────────────────────────────────────────

class TileFixR(QMainWindow):
    def __init__(self, entries, all_polys, frame_names, job_id, view_only=False):
        super().__init__()
        self.view_only = view_only
        self.entries = entries
        self.all_polys = all_polys
        # Central polygon registry (full-frame coords). Source of truth for edits.
        # status: "original" / "edited" / "added"
        self.polygons_by_id = {}
        for p in all_polys:
            self.polygons_by_id[p["server_id"]] = {
                "server_id": p["server_id"],
                "frame_idx": p["frame_idx"],
                "color_index": p["color_index"],
                "points": p["points"].copy().astype(np.float32),
                "original_points": p["points"].copy().astype(np.float32),
                "label": p.get("label"),
                "status": "original",
            }
        self.poly_by_id = self.polygons_by_id  # alias for back-compat

        # Reconciliation pass: re-clip every polygon using the same code path
        # as Pull from CVAT. Heals any discrepancy from the startup load path.
        _entries_by_frame: dict = {}
        for _e in self.entries:
            _entries_by_frame.setdefault(_e["frame_idx"], []).append(_e)
        for _sid, _poly in self.polygons_by_id.items():
            _ff = _poly["points"]
            if _ff.shape[0] < 3:
                continue
            _fi = _poly["frame_idx"]
            _ci = _poly["color_index"]
            _px0, _py0 = _ff[:, 0].min(), _ff[:, 1].min()
            _px1, _py1 = _ff[:, 0].max(), _ff[:, 1].max()
            for _e in _entries_by_frame.get(_fi, []):
                _tx, _ty = _e["tile_x"], _e["tile_y"]
                _tx2, _ty2 = _tx + TILE_SIZE, _ty + TILE_SIZE
                _e["polys"] = [p for p in _e["polys"] if p["server_id"] != _sid]
                if _px1 <= _tx or _px0 >= _tx2 or _py1 <= _ty or _py0 >= _ty2:
                    continue
                for _c in clip_polygon_to_tile(_ff, _tx, _ty, _tx2, _ty2):
                    _e["polys"].append({
                        "server_id": _sid,
                        "color_index": _ci,
                        "tile_local_pts": _c,
                        "frame_idx": _fi,
                    })

        self.frame_names = frame_names
        self.job_id = job_id
        self.marked_ids = set()
        self.selected_id = None
        # Vertex-drag state
        self._drag_polygon_id = None
        self._drag_vertex_idx = None
        self._drag_active = False
        self._drag_pending = False   # grabbed vertex but not yet past deadzone
        self._drag_press_ix = None
        self._drag_press_iy = None
        self.DRAG_DEADZONE_PX = 5    # image pixels before drag activates
        # Add-polygon-mode state
        self.add_mode = False
        self.add_pts_local = []   # in tile-local coords of current tile
        self.add_hover_xy = None  # current cursor pos for rubber-band preview
        self.next_temp_id = -1
        # Undo stack — list of (op_type, payload) tuples, max 50
        self.undo_stack = []
        self.UNDO_MAX = 50
        # Snapshot of polygon points at drag start (committed to stack on release)
        self._drag_undo_snapshot = None
        # Hover affordance state — tracks which handle is currently under the
        # cursor so build_display_crop can draw it larger / thicker.
        # Format: ("vertex", server_id, vertex_idx) or
        #         ("extend", server_id, kind, end_idx) or None.
        self.hovered_handle = None
        # Extend-handle drag state
        self._extend_active = False
        self._extend_polygon_id = None
        self._extend_indices = None       # list of vertex indices to move together
        self._extend_kind = None          # "length" or "width"
        self._extend_axis = None          # numpy array (dx, dy) unit vector
        self._extend_last_full = None     # last cursor pos in full-frame coords
        self._send_worker = None
        self._send_anim_timer = None
        self._send_anim_step = 0
        self._maskcheckr_anim_timer = None
        self._maskcheckr_anim_step = 0
        self._weirdr_anim_timer = None
        self._weirdr_anim_step = 0
        self._load_marks()
        saved_state = self._load_saved_state()
        self.filter_mode = saved_state.get("filter_mode", "all")
        if self.filter_mode not in {"all", "has_polygons", "no_polygons", "has_marked"}:
            self.filter_mode = "all"
        saved_tile_filter = saved_state.get("tile_filter")
        self.tile_filter = tuple(saved_tile_filter) if saved_tile_filter else None
        self.flagged_frame_indices = set()
        self.show_flagged = False
        self._scan_worker = None
        self._load_flags()
        self.current_idx = self._restore_position()
        idx = self.filtered_indices()
        if idx and self.current_idx not in idx:
            self.current_idx = idx[0]
        self.brightness = 1.0
        self.contrast = 1.0
        self._last_displayed_idx = None

        self.setWindowTitle("TileFixR")
        self.setMinimumSize(1100, 800)
        self.resize(1280, 900)

        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_banner())
        root.addWidget(self._build_toolbar())

        if saved_state.get("brightness"):
            self.bright_slider.setValue(saved_state["brightness"])
        if saved_state.get("contrast"):
            self.contrast_slider.setValue(saved_state["contrast"])
        if self.tile_filter is not None:
            row_letter = chr(ord('A') + self.tile_filter[0])
            self.tile_input.setText(f"{row_letter}{self.tile_filter[1] + 1}")
            QTimer.singleShot(0, self._on_tile_filter_apply)

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
        root.addWidget(self._build_image_area(), stretch=1)

        root.addWidget(self._build_info_strip())
        root.addWidget(self._build_action_bar())
        root.addWidget(self._build_nav_bar())
        root.addWidget(self._build_debug_bar())

        if self.view_only:
            self.add_btn.setEnabled(False)
            self.send_btn.setEnabled(False)
            self.pull_btn.setEnabled(False)
            self.setWindowTitle("TileFixR  —  View Only")

        def _guarded(fn):
            def wrapper():
                if isinstance(QApplication.focusWidget(), QLineEdit):
                    return
                fn()
            return wrapper

        def _nav_guarded(fn):
            def wrapper():
                focused = QApplication.focusWidget()
                if isinstance(focused, QLineEdit) and focused is not self.tile_input:
                    return
                fn()
            return wrapper

        for key, fn, guard in [
            (Qt.Key_Left,     self.go_prev,                _nav_guarded),
            (Qt.Key_Right,    self.go_next,                _nav_guarded),
            (Qt.Key_Delete,   self._toggle_selected_mark,  _guarded),
            (Qt.Key_Backspace, self._toggle_selected_mark, _guarded),
            (Qt.Key_Space,    self._toggle_selected_mark,  _guarded),
            (Qt.Key_Escape,   self._on_escape,             _guarded),
            (Qt.Key_A,        self._toggle_add_mode,       _guarded),
            (Qt.Key_Return,   self._close_add_polygon,     _guarded),
            (Qt.Key_Enter,    self._close_add_polygon,     _guarded),
            (Qt.Key_C,        self._copy_reference,        _guarded),
        ]:
            sc = QShortcut(QKeySequence(key), self, guard(fn))
            sc.setContext(Qt.ApplicationShortcut)
        # Undo: Cmd+Z on Mac, Ctrl+Z elsewhere — handled by QKeySequence.Undo
        undo_sc = QShortcut(QKeySequence.Undo, self, self._undo)
        undo_sc.setContext(Qt.ApplicationShortcut)

        # Shift+Arrow — jump to adjacent tile
        for key_str, dr, dc in [
            ("Shift+Up",    -1,  0),
            ("Shift+Down",   1,  0),
            ("Shift+Left",   0, -1),
            ("Shift+Right",  0,  1),
        ]:
            _dr, _dc = dr, dc
            sc = QShortcut(QKeySequence(key_str), self,
                           lambda _r=_dr, _c=_dc: self._go_adjacent(_r, _c))
            sc.setContext(Qt.ApplicationShortcut)

        for lbl in self.findChildren(QLabel):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.refresh_view()
        QTimer.singleShot(0, self._restore_zoom)

    # ── Banner ──────────────────────────────────────────────────────────────

    def _build_banner(self):
        banner = QWidget()
        banner.setFixedHeight(48)
        banner.setStyleSheet("background-color: #0a1e3f;")
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        title = QLabel("TileFixR")
        title.setStyleSheet(
            "color: white; font-size: 18px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(title)

        frame_desc = (f"{len(SUSPECT_FRAMES)} suspect frames" if SUSPECT_FRAMES
                      else f"frames {FRAME_START + 1}-{FRAME_END}")
        if self.entries:
            max_row = max(e["tile_row"] for e in self.entries)
            max_col = max(e["tile_col"] for e in self.entries)
            tile_range = f"A1–{chr(ord('A') + max_row)}{max_col + 1}"
        else:
            tile_range = ""
        sub = QLabel(f"task {CVAT_TASK_ID}  |  {IMG_DIR.name}  |  {frame_desc}  |  "
                      f"tiles {tile_range}  |  "
                      f"{len(self.all_polys)} polygons in {len(self.entries)} tiles")
        sub.setStyleSheet(
            "color: #a8c0e0; font-size: 14px; background: transparent;"
        )
        layout.addWidget(sub)

        layout.addStretch()

        # Send-to-CVAT button (upper right, count updates dynamically)
        self.send_btn = QPushButton("Send 0 changes to CVAT")
        self.send_btn.setFixedHeight(28)
        self.send_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 12px; "
            "font-weight: bold; border-radius: 17px; border: none; padding: 0 16px; }"
            "QPushButton:hover { background-color: #a93226; }"
            "QPushButton:disabled { background-color: #6a6a6a; color: #ddd; }"
        )
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._send_to_cvat)
        layout.addWidget(self.send_btn)

        relaunch_btn = QPushButton("Relaunch")
        relaunch_btn.setFixedHeight(26)
        relaunch_btn.setStyleSheet(
            "QPushButton { background-color: #d0e4f5; color: #1a3a5c; font-size: 12px; "
            "font-weight: bold; border-radius: 15px; border: 1px solid #a0c4e0; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #b8d4ec; }"
        )
        relaunch_btn.setToolTip("Quit and relaunch — picks up code edits "
                                "and a fresh CVAT pull")
        relaunch_btn.clicked.connect(self._relaunch)
        layout.addWidget(relaunch_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #d93025; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #b8271b; }"
        )
        close_btn.setToolTip("Quit TileFixR")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        return banner

    # ── Toolbar (filter + brightness + scrubber) ────────────────────────────

    def _build_toolbar(self):
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {PANEL_BG};")
        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(16, 5, 16, 5)
        layout.setSpacing(5)

        # Row 1 — filter pills
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        show_label = QLabel("Show:")
        show_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;")
        row1.addWidget(show_label)

        self.filter_buttons = {}
        self.filter_group = QButtonGroup(self)
        self.filter_group.setExclusive(True)

        n_with_polys = sum(1 for e in self.entries if e["polys"])
        n_no_polys = len(self.entries) - n_with_polys
        filters = [
            ("all",           f"All ({len(self.entries)})",            "blue"),
            ("has_polygons",  f"Has polygons ({n_with_polys})",        "green"),
            ("no_polygons",   f"No polygons ({n_no_polys})",           "amber"),
            ("has_marked",    "Has marked (0)",                        "red"),
        ]
        from_make_pill = make_pill
        for mode, text, color in filters:
            pill = from_make_pill(text, color)
            if mode == self.filter_mode:
                pill.setChecked(True)
            self.filter_group.addButton(pill)
            self.filter_buttons[mode] = pill
            pill.clicked.connect(lambda checked, m=mode: self.set_filter(m))
            row1.addWidget(pill)

        self.flagged_pill = make_pill("View Flagged (0)", "purple")
        self.flagged_pill.setEnabled(False)
        self.flagged_pill.setChecked(self.show_flagged)
        self.flagged_pill.clicked.connect(self._on_view_flagged_toggled)
        row1.addWidget(self.flagged_pill)
        row1.addStretch()

        tile_lbl = QLabel("Tile:")
        tile_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;")
        row1.addWidget(tile_lbl)
        self.tile_input = QLineEdit()
        self.tile_input.setFixedWidth(56)
        self.tile_input.setFixedHeight(26)
        self.tile_input.setPlaceholderText("E5")
        self.tile_input.setAlignment(Qt.AlignCenter)
        self.tile_input.returnPressed.connect(self._on_tile_filter_apply)
        self.tile_input.installEventFilter(self)
        row1.addWidget(self.tile_input)
        go_btn = QPushButton("Go")
        go_btn.setFixedHeight(26)
        go_btn.setFixedWidth(36)
        go_btn.clicked.connect(self._on_tile_filter_apply)
        row1.addWidget(go_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(26)
        clear_btn.setFixedWidth(44)
        clear_btn.clicked.connect(self._on_tile_filter_clear)
        row1.addWidget(clear_btn)

        self.scan_btn = QPushButton("Scan Tile")
        self.scan_btn.setFixedHeight(26)
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self._on_scan_tile)
        row1.addWidget(self.scan_btn)

        layout.addLayout(row1)

        # Row 2 — brightness/contrast on left, scrubber on right
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        left_half = QHBoxLayout()
        left_half.setSpacing(6)
        bright_lbl = QLabel("Brightness")
        bright_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        left_half.addWidget(bright_lbl)
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setMinimum(10)
        self.bright_slider.setMaximum(40)
        self.bright_slider.setValue(10)
        self.bright_slider.setFixedWidth(160)
        self.bright_slider.valueChanged.connect(self._on_brightness)
        left_half.addWidget(self.bright_slider)
        self.bright_value_label = QLabel("1.0x")
        self.bright_value_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        self.bright_value_label.setFixedWidth(36)
        left_half.addWidget(self.bright_value_label)

        left_half.addSpacing(16)
        contrast_lbl = QLabel("Contrast")
        contrast_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        left_half.addWidget(contrast_lbl)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(40)
        self.contrast_slider.setValue(10)
        self.contrast_slider.setFixedWidth(160)
        self.contrast_slider.valueChanged.connect(self._on_contrast)
        left_half.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel("1.0x")
        self.contrast_value_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        self.contrast_value_label.setFixedWidth(36)
        left_half.addWidget(self.contrast_value_label)
        left_half.addStretch()
        row2.addLayout(left_half, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #ccc;")
        row2.addWidget(sep)

        right_half = QHBoxLayout()
        right_half.setSpacing(8)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setMinimum(1)
        self.scrubber.setMaximum(max(1, len(self.entries)))
        self.scrubber.setValue(1)
        self.scrubber.valueChanged.connect(self._on_scrubber_changed)
        right_half.addWidget(self.scrubber, stretch=1)

        self.scrubber_label = QLabel("1 of 0")
        self.scrubber_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; font-weight: bold;")
        self.scrubber_label.setFixedWidth(80)
        right_half.addWidget(self.scrubber_label)

        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(56)
        self.jump_input.setFixedHeight(26)
        self.jump_input.setText("1")
        self.jump_input.setAlignment(Qt.AlignCenter)
        self.jump_input.returnPressed.connect(self._on_jump_entered)
        right_half.addWidget(self.jump_input)

        row2.addLayout(right_half, stretch=1)
        layout.addLayout(row2)
        return toolbar

    # ── Info strip ──────────────────────────────────────────────────────────

    def _build_info_strip(self):
        strip = QWidget()
        strip.setFixedHeight(28)
        strip.setStyleSheet(f"background: {INFO_BG};")
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)

        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {INFO_TEXT}; "
            "background: transparent;"
        )
        layout.addWidget(self.info_label)

        self.detail_label = QLabel()
        self.detail_label.setStyleSheet(
            f"font-size: 12px; color: {INFO_TEXT}; background: transparent;"
        )
        layout.addWidget(self.detail_label)

        layout.addStretch()

        self.selection_label = QLabel()
        self.selection_label.setStyleSheet(
            f"font-size: 12px; color: #c0392b; background: transparent;"
        )
        layout.addWidget(self.selection_label)
        return strip

    def _build_debug_bar(self):
        bar = QWidget()
        bar.setFixedHeight(18)
        bar.setStyleSheet(f"background: {INFO_BG};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)
        self.debug_label = QLabel("debug: idle")
        self.debug_label.setStyleSheet(
            "font-size: 10px; color: #888; background: transparent; "
            "font-family: monospace;"
        )
        layout.addWidget(self.debug_label)
        layout.addStretch()
        # Aliases so existing handlers keep working
        self.debug_press_label = self.debug_label
        self.debug_click_label = self.debug_label
        self.debug_move_label = self.debug_label
        return bar

    # ── Image area with adjacent-tile arrows ────────────────────────────────

    def _build_image_area(self):
        """Wrap the tile image view with four adjacent-tile navigation arrows."""
        arrow_style = (
            "QPushButton { background-color: #1a3a5c; color: white; font-size: 20px; "
            "border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a6fc4; }"
            "QPushButton:disabled { background-color: #111; color: #333; }"
        )

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # Top row — up arrow centered
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 2, 0, 2)
        self.adj_up = QPushButton("▲")
        self.adj_up.setFixedSize(192, 28)
        self.adj_up.setStyleSheet(arrow_style)
        self.adj_up.setToolTip("Jump to tile above  (Shift+Up)")
        self.adj_up.clicked.connect(lambda: self._go_adjacent(-1, 0))
        top_row.addStretch()
        top_row.addWidget(self.adj_up)
        top_row.addStretch()
        vlay.addLayout(top_row)

        # Middle row — left arrow | image view | right arrow
        mid_row = QHBoxLayout()
        mid_row.setContentsMargins(2, 0, 2, 0)
        mid_row.setSpacing(2)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        self.adj_left = QPushButton("◀")
        self.adj_left.setFixedSize(36, 192)
        self.adj_left.setStyleSheet(arrow_style)
        self.adj_left.setToolTip("Jump to tile left  (Shift+Left)")
        self.adj_left.clicked.connect(lambda: self._go_adjacent(0, -1))
        left_col.addStretch()
        left_col.addWidget(self.adj_left)
        left_col.addStretch()
        mid_row.addLayout(left_col)

        mid_row.addWidget(self.image_view, stretch=1)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        self.adj_right = QPushButton("▶")
        self.adj_right.setFixedSize(36, 192)
        self.adj_right.setStyleSheet(arrow_style)
        self.adj_right.setToolTip("Jump to tile right  (Shift+Right)")
        self.adj_right.clicked.connect(lambda: self._go_adjacent(0, 1))
        right_col.addStretch()
        right_col.addWidget(self.adj_right)
        right_col.addStretch()
        mid_row.addLayout(right_col)

        vlay.addLayout(mid_row, stretch=1)

        # Bottom row — down arrow centered
        bot_row = QHBoxLayout()
        bot_row.setContentsMargins(0, 2, 0, 2)
        self.adj_down = QPushButton("▼")
        self.adj_down.setFixedSize(192, 28)
        self.adj_down.setStyleSheet(arrow_style)
        self.adj_down.setToolTip("Jump to tile below  (Shift+Down)")
        self.adj_down.clicked.connect(lambda: self._go_adjacent(1, 0))
        bot_row.addStretch()
        bot_row.addWidget(self.adj_down)
        bot_row.addStretch()
        vlay.addLayout(bot_row)

        return container

    # ── Action bar (Mark, Send) ─────────────────────────────────────────────

    def _build_action_bar(self):
        bar = QWidget()
        bar.setStyleSheet(f"background: {PANEL_BG};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)

        self.mark_btn = QPushButton("Mark for delete  (Del)")
        self.mark_btn.setCheckable(True)
        self.mark_btn.setFixedHeight(36)
        self.mark_btn.setMinimumWidth(180)
        self.mark_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; font-weight: bold;
                border: 3px solid #999; border-radius: 8px;
                background: #f0f0f0; color: #333;
                padding: 8px 14px;
            }
            QPushButton:checked {
                background: #c0392b; color: white; border-color: #922b21;
            }
            QPushButton:hover:!checked {
                border-color: #c0392b; color: #c0392b;
            }
            QPushButton:disabled {
                background: #e0e0e0; color: #aaa; border-color: #ccc;
            }
        """)
        self.mark_btn.setEnabled(False)
        self.mark_btn.clicked.connect(self._toggle_selected_mark)
        layout.addWidget(self.mark_btn)

        self.add_btn = QPushButton("Add polygon  (A)")
        self.add_btn.setCheckable(True)
        self.add_btn.setFixedHeight(36)
        self.add_btn.setMinimumWidth(180)
        self.add_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; font-weight: bold;
                border: 3px solid #999; border-radius: 8px;
                background: #f0f0f0; color: #333;
                padding: 8px 14px;
            }
            QPushButton:checked {
                background: #2a7a2a; color: white; border-color: #1e5f1e;
            }
            QPushButton:hover:!checked {
                border-color: #2a7a2a; color: #1e5f1e;
            }
        """)
        self.add_btn.setToolTip(
            "Click to add vertices on the current tile. "
            "Double-click or press Enter to close the polygon. "
            "ESC cancels.")
        self.add_btn.clicked.connect(self._toggle_add_mode)
        layout.addWidget(self.add_btn)

        self.pending_label = QLabel("Pending: 0 add / 0 edit / 0 delete")
        self.pending_label.setStyleSheet(
            f"color: {INFO_TEXT}; font-size: 12px; font-weight: bold; padding: 0 12px;")
        layout.addWidget(self.pending_label)

        self.copy_btn = QPushButton("Copy reference")
        self.copy_btn.setFixedHeight(36)
        self.copy_btn.setStyleSheet(
            "QPushButton { background-color: #d0e4f5; color: #1a3a5c; font-size: 13px; "
            "font-weight: bold; border-radius: 8px; border: 1px solid #a0c4e0; "
            "padding: 8px 18px; }"
            "QPushButton:hover { background-color: #b8d4ec; }"
        )
        self.copy_btn.setToolTip(
            "Copy this tile's reference to the clipboard so you can paste it "
            "back to Claude. Press C as a shortcut.")
        self.copy_btn.clicked.connect(self._copy_reference)
        layout.addWidget(self.copy_btn)

        self.undo_btn = QPushButton("Undo (0)")
        self.undo_btn.setFixedHeight(36)
        self.undo_btn.setStyleSheet(
            "QPushButton { background-color: #e8eef5; color: #1a3a5c; font-size: 13px; "
            "font-weight: bold; border-radius: 8px; border: 1px solid #a0c4e0; "
            "padding: 8px 18px; }"
            "QPushButton:hover { background-color: #d0e4f5; }"
            "QPushButton:disabled { background-color: #f0f0f0; color: #aaa; border-color: #ccc; }"
        )
        self.undo_btn.setToolTip("Undo last edit (Cmd+Z / Ctrl+Z). "
                                  "Captures vertex moves, mark toggles, and added polygons.")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo)
        layout.addWidget(self.undo_btn)

        self.mask_checkr_btn = QPushButton("Mask CheckR")
        self.mask_checkr_btn.setFixedHeight(36)
        self.mask_checkr_btn.setStyleSheet(
            "QPushButton { background-color: #eaeaea; color: #444; font-size: 13px; "
            "font-weight: bold; border-radius: 8px; border: 1px solid #aaa; "
            "padding: 8px 18px; }"
            "QPushButton:hover { background-color: #dadada; }"
        )
        self.mask_checkr_btn.clicked.connect(self._launch_mask_checkr)
        layout.addWidget(self.mask_checkr_btn)

        self.weirdr_btn = QPushButton("Add To WeirdR")
        self.weirdr_btn.setFixedHeight(36)
        self.weirdr_btn.setStyleSheet(
            "QPushButton { background-color: #ede0f5; color: #4a1a6a; font-size: 13px; "
            "font-weight: bold; border-radius: 8px; border: 1px solid #c0a0e0; "
            "padding: 8px 18px; }"
            "QPushButton:hover { background-color: #ddd0f0; }"
        )
        self.weirdr_btn.clicked.connect(self._add_to_weirdr)
        layout.addWidget(self.weirdr_btn)

        layout.addStretch()
        # Send button lives in the banner (upper right) — see _build_banner().
        return bar

    def _add_to_weirdr(self):
        if not self.entries or self.current_idx >= len(self.entries):
            return
        entry = self.entries[self.current_idx]
        tag = entry["tile_id"]
        try:
            weirdr = json.loads(WEIRDR_PATH.read_text()) if WEIRDR_PATH.exists() else []
        except Exception:
            weirdr = []
        already = any(e.get("tag") == tag for e in weirdr)
        if not already:
            weirdr.append({
                "source": "tile_fixr",
                "tag": tag,
                "filename": entry["img_filename"],
                "dataset": TASK_NAME,
                "tile_x": entry["tile_x"],
                "tile_y": entry["tile_y"],
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
        labels = ["Adding.", "Adding..", "Adding...", "Added!"]
        step = self._weirdr_anim_step
        self.weirdr_btn.setText(labels[min(step, len(labels) - 1)])
        self._weirdr_anim_step += 1
        if step >= len(labels) - 1:
            self._weirdr_anim_timer.stop()
            self._weirdr_anim_timer = None
            QTimer.singleShot(800, lambda: self.weirdr_btn.setEnabled(True))

    # ── Nav bar ─────────────────────────────────────────────────────────────

    def _build_nav_bar(self):
        bar = QWidget()
        bar.setStyleSheet(f"background: {PANEL_BG};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 4, 16, 6)
        layout.setSpacing(12)

        nav_style = (
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 15px; "
            "font-weight: bold; border-radius: 6px; border: none; padding: 10px 28px; }"
            "QPushButton:hover { background-color: #1580e0; }"
            "QPushButton:disabled { background-color: #999; }"
        )

        self.prev_btn = QPushButton("Prev")
        self.prev_btn.setStyleSheet(nav_style)
        self.prev_btn.clicked.connect(self.go_prev)
        layout.addWidget(self.prev_btn)

        hint = QLabel("←/→ navigate    click polygon = select    C = copy ref    Esc = clear")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {HINT_TEXT}; font-size: 11px; background: transparent;")
        layout.addWidget(hint, stretch=1)

        self.pull_btn = QPushButton("Pull from CVAT")
        self.pull_btn.setStyleSheet(
            "QPushButton { background-color: #d39e00; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; border: none; padding: 10px 18px; }"
            "QPushButton:hover { background-color: #b8860b; }"
            "QPushButton:disabled { background-color: #888; }"
        )
        self.pull_btn.setToolTip("Re-fetch all polygons on the current frame from CVAT.")
        self.pull_btn.clicked.connect(self._pull_from_cvat)
        layout.addWidget(self.pull_btn)

        self.cvat_btn = QPushButton("Open in CVAT")
        self.cvat_btn.setStyleSheet(
            "QPushButton { background-color: #2a7a2a; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; border: none; padding: 10px 22px; }"
            "QPushButton:hover { background-color: #339933; }"
        )
        self.cvat_btn.clicked.connect(self._open_in_cvat)
        layout.addWidget(self.cvat_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet(nav_style)
        self.next_btn.clicked.connect(self.go_next)
        layout.addWidget(self.next_btn)
        return bar

    # ── Filter / navigation ─────────────────────────────────────────────────

    def _parse_tile_ref(self, text):
        text = text.strip().upper()
        if not text:
            return None
        i = 0
        while i < len(text) and text[i].isalpha():
            i += 1
        if i == 0 or i == len(text):
            return None
        try:
            col_1based = int(text[i:])
        except ValueError:
            return None
        letters = text[:i]
        if len(letters) != 1:
            return None
        row_idx = ord(letters[0]) - ord('A')
        col_idx = col_1based - 1
        if row_idx < 0 or col_idx < 0:
            return None
        return (row_idx, col_idx)

    def _on_tile_filter_apply(self):
        parsed = self._parse_tile_ref(self.tile_input.text())
        if parsed is not None:
            self.tile_filter = parsed
            idx = self.filtered_indices()
            if not idx:
                max_row = max(e["tile_row"] for e in self.entries)
                max_col = max(e["tile_col"] for e in self.entries)
                valid_max = chr(ord('A') + max_row) + str(max_col + 1)
                ref = self.tile_input.text().strip().upper()
                QMessageBox.warning(self, "Tile out of range",
                    f"Tile {ref} is out of range.\n\nValid range is A1 to {valid_max}.")
                self.tile_filter = None
                self.tile_input.clear()
                self.refresh_view()
                return
            if self.current_idx not in idx:
                self.current_idx = idx[0]
        else:
            self.tile_filter = None
        self._load_flags()
        self._update_scan_controls()
        self._save_state()
        self.refresh_view()

    def _on_tile_filter_clear(self):
        self.tile_filter = None
        self.tile_input.clear()
        self.show_flagged = False
        self.flagged_pill.setChecked(False)
        self._load_flags()
        self._update_scan_controls()
        self._save_state()
        self.refresh_view()

    def _flags_path(self):
        if self.tile_filter is None:
            return None
        row_letter = chr(ord('A') + self.tile_filter[0])
        col_num = self.tile_filter[1] + 1
        return IMG_DIR / "cleanr_workspace" / f"screener_flags_{row_letter}{col_num}.json"

    def _load_flags(self):
        self.flagged_frame_indices = set()
        path = self._flags_path()
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                self.flagged_frame_indices = set(data.get("flagged_frames", []))
            except Exception:
                pass

    def _update_scan_controls(self):
        tile_set = self.tile_filter is not None
        flags_exist = bool(self.flagged_frame_indices)
        n = len(self.flagged_frame_indices)
        self.scan_btn.setEnabled(tile_set and (self._scan_worker is None or not self._scan_worker.isRunning()))
        self.flagged_pill.setEnabled(tile_set and flags_exist)
        self.flagged_pill.setText(f"View Flagged ({n})")
        if not flags_exist:
            self.show_flagged = False
            self.flagged_pill.setChecked(False)

    def _on_view_flagged_toggled(self, checked):
        self.show_flagged = checked
        idx = self.filtered_indices()
        if idx and self.current_idx not in idx:
            self.current_idx = idx[0]
        self.refresh_view()

    def _on_scan_tile(self):
        if self.tile_filter is None:
            return
        tile_entries = [e for e in self.entries
                        if e["tile_row"] == self.tile_filter[0]
                        and e["tile_col"] == self.tile_filter[1]]
        if not tile_entries:
            return
        tx = tile_entries[0]["tile_x"]
        ty = tile_entries[0]["tile_y"]

        self._scan_worker = ScanWorker(IMG_DIR, tx, ty)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning... 0")
        self._scan_worker.start()

    def _on_scan_progress(self, current, total):
        self.scan_btn.setText(f"Scanning... {current}/{total}")

    def _on_scan_finished(self, flagged_indices):
        path = self._flags_path()
        if path:
            row_letter = chr(ord('A') + self.tile_filter[0])
            col_num = self.tile_filter[1] + 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "tile": f"{row_letter}{col_num}",
                "tile_row": self.tile_filter[0],
                "tile_col": self.tile_filter[1],
                "flagged_frames": flagged_indices,
            }, indent=2))
        self.scan_btn.setText("Scan Tile")
        self._load_flags()
        self._update_scan_controls()
        self.refresh_view()

    def _on_scan_failed(self, msg):
        self.scan_btn.setText("Scan Tile")
        self._update_scan_controls()
        QMessageBox.warning(self, "Scan failed", msg)

    def filtered_indices(self):
        if self.filter_mode == "all":
            base = list(range(len(self.entries)))
        elif self.filter_mode == "has_polygons":
            base = [i for i, e in enumerate(self.entries) if e["polys"]]
        elif self.filter_mode == "no_polygons":
            base = [i for i, e in enumerate(self.entries) if not e["polys"]]
        elif self.filter_mode == "has_marked":
            base = [i for i, e in enumerate(self.entries)
                    if any(p["server_id"] in self.marked_ids for p in e["polys"])]
        else:
            base = list(range(len(self.entries)))
        if self.tile_filter is not None:
            tr, tc = self.tile_filter
            base = [i for i in base
                    if self.entries[i]["tile_row"] == tr and self.entries[i]["tile_col"] == tc]
        if self.show_flagged and self.flagged_frame_indices:
            base = [i for i in base
                    if self.entries[i]["frame_idx"] in self.flagged_frame_indices]
        return base

    def set_filter(self, mode):
        self.filter_mode = mode
        idx = self.filtered_indices()
        if idx and self.current_idx not in idx:
            self.current_idx = idx[0]
        self._save_state()
        self.refresh_view()

    def go_prev(self):
        idx = self.filtered_indices()
        if not idx:
            return
        try:
            pos = idx.index(self.current_idx)
        except ValueError:
            pos = 0
        if pos > 0:
            self.current_idx = idx[pos - 1]
            self.selected_id = None
            self._save_state()
            self.refresh_view()
        else:
            _play_end_sound()

    def go_next(self):
        idx = self.filtered_indices()
        if not idx:
            return
        try:
            pos = idx.index(self.current_idx)
        except ValueError:
            pos = -1
        if pos < len(idx) - 1:
            self.current_idx = idx[pos + 1]
            self.selected_id = None
            self._save_state()
            self.refresh_view()
        else:
            _play_end_sound()

    def _go_adjacent(self, dr, dc):
        """Jump to the neighboring tile (dr/dc = row/col delta) at the same frame."""
        if not self.entries:
            return
        if self.tile_filter is not None:
            tr, tc = self.tile_filter
        else:
            e = self.entries[self.current_idx]
            tr, tc = e["tile_row"], e["tile_col"]
        new_tr, new_tc = tr + dr, tc + dc
        if not any(e["tile_row"] == new_tr and e["tile_col"] == new_tc
                   for e in self.entries):
            return
        current_frame_idx = self.entries[self.current_idx]["frame_idx"]
        self.tile_filter = (new_tr, new_tc)
        self.tile_input.setText(chr(ord('A') + new_tr) + str(new_tc + 1))
        self._load_flags()
        self._update_scan_controls()
        idx = self.filtered_indices()
        if not idx:
            return
        same_frame = [i for i in idx
                      if self.entries[i]["frame_idx"] == current_frame_idx]
        self.current_idx = same_frame[0] if same_frame else idx[0]
        self.selected_id = None
        self._save_state()
        self.refresh_view()

    def _refresh_adjacent_arrows(self):
        """Enable/disable the four adjacent-tile arrows based on what exists."""
        if not hasattr(self, 'adj_up'):
            return
        if not self.entries:
            for btn in (self.adj_up, self.adj_down, self.adj_left, self.adj_right):
                btn.setEnabled(False)
            return
        if self.tile_filter is not None:
            tr, tc = self.tile_filter
        else:
            e = self.entries[self.current_idx]
            tr, tc = e["tile_row"], e["tile_col"]
        def has_tile(r, c):
            return any(e["tile_row"] == r and e["tile_col"] == c
                       for e in self.entries)
        self.adj_up.setEnabled(has_tile(tr - 1, tc))
        self.adj_down.setEnabled(has_tile(tr + 1, tc))
        self.adj_left.setEnabled(has_tile(tr, tc - 1))
        self.adj_right.setEnabled(has_tile(tr, tc + 1))

    def eventFilter(self, obj, event):
        if obj is self.tile_input and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Left:
                self.go_prev()
                return True
            if event.key() == Qt.Key_Right:
                self.go_next()
                return True
        return super().eventFilter(obj, event)

    def _on_scrubber_changed(self, value):
        idx = self.filtered_indices()
        if not idx:
            return
        i = max(0, min(value - 1, len(idx) - 1))
        self.current_idx = idx[i]
        self.selected_id = None
        self._save_state()
        self.refresh_view()

    def _on_jump_entered(self):
        text = self.jump_input.text().strip()
        try:
            target = int(text)
        except ValueError:
            return
        idx = self.filtered_indices()
        if not idx:
            return
        pos = max(1, min(target, len(idx)))
        self.current_idx = idx[pos - 1]
        self.selected_id = None
        self.jump_input.clearFocus()
        self._save_state()
        self.refresh_view()

    # ── Display ─────────────────────────────────────────────────────────────

    def _on_brightness(self, value):
        self.brightness = value / 10.0
        self.bright_value_label.setText(f"{self.brightness:.1f}x")
        self.refresh_view()

    def _on_contrast(self, value):
        self.contrast = value / 10.0
        self.contrast_value_label.setText(f"{self.contrast:.1f}x")
        self.refresh_view()

    def build_display_crop(self, entry):
        crop = entry["crop_raw"].copy()
        if self.brightness != 1.0 or self.contrast != 1.0:
            img = crop.astype(np.float32)
            if self.brightness != 1.0:
                img = img * self.brightness
            if self.contrast != 1.0:
                img = (img - 128.0) * self.contrast + 128.0
            crop = np.clip(img, 0, 255).astype(np.uint8)
        # Inverse zoom factor — multiply image-pixel sizes by this so they
        # display at constant SCREEN size regardless of zoom level.
        try:
            scale = max(0.05, self.image_view.current_scale())
        except Exception:
            scale = 1.0
        zinv = 1.0 / scale

        def s(n):
            """Scale an image-pixel size by inverse zoom, floor at 1."""
            return max(1, int(round(n * zinv)))

        def s_f(n):
            """Same as s() but allow float radii (for circle).
            Floor at 1.0 so cv2 still draws."""
            return max(1.0, n * zinv)
        # Edge indicators: orange line on any side where no neighboring tile exists
        ch, cw = crop.shape[:2]
        EDGE_T = 10
        EDGE_COLOR = (0, 165, 255)  # bright orange (BGR)
        if self.tile_filter is not None and self.entries:
            tr, tc = self.tile_filter
            def _has_tile(r, c):
                return any(e["tile_row"] == r and e["tile_col"] == c
                           for e in self.entries)
            if not _has_tile(tr, tc - 1):  # nothing to the left
                cv2.line(crop, (0, 0), (0, ch - 1), EDGE_COLOR, EDGE_T)
            if not _has_tile(tr, tc + 1):  # nothing to the right
                cv2.line(crop, (cw - 1, 0), (cw - 1, ch - 1), EDGE_COLOR, EDGE_T)
            if not _has_tile(tr - 1, tc):  # nothing above
                cv2.line(crop, (0, 0), (cw - 1, 0), EDGE_COLOR, EDGE_T)
            if not _has_tile(tr + 1, tc):  # nothing below
                cv2.line(crop, (0, ch - 1), (cw - 1, ch - 1), EDGE_COLOR, EDGE_T)

        # Draw each polygon in its assigned color, plus tiny vertex hints
        for p in entry["polys"]:
            color = POLY_COLORS[p["color_index"] % len(POLY_COLORS)]
            cv2.polylines(crop, [p["tile_local_pts"]], True, color, s(2))
            for vx, vy in p["tile_local_pts"]:
                cv2.circle(crop, (int(vx), int(vy)), s(2), color, -1, cv2.LINE_AA)
        # Marked-for-delete overlay (thinnest red outline + faint red fill)
        for p in entry["polys"]:
            if p["server_id"] in self.marked_ids:
                cv2.polylines(crop, [p["tile_local_pts"]], True, DELETE_COLOR, s(2))
                overlay = crop.copy()
                cv2.fillPoly(overlay, [p["tile_local_pts"]], DELETE_COLOR)
                crop = cv2.addWeighted(overlay, 0.20, crop, 0.80, 0)
        # Edited / added overlay (thinnest yellow for edited, green for added)
        for p in entry["polys"]:
            poly_meta = self.polygons_by_id.get(p["server_id"])
            if poly_meta is None:
                continue
            if poly_meta["status"] == "edited":
                cv2.polylines(crop, [p["tile_local_pts"]], True, (0, 255, 255), s(2))
            elif poly_meta["status"] == "added":
                cv2.polylines(crop, [p["tile_local_pts"]], True, (0, 255, 0), s(2))
        # Selected polygon — thicker white outline + small handles
        hovered_vert = (self.hovered_handle if self.hovered_handle and
                          self.hovered_handle[0] == "vertex" else None)
        for p in entry["polys"]:
            if p["server_id"] == self.selected_id:
                cv2.polylines(crop, [p["tile_local_pts"]], True, SELECTED_COLOR, s(3))
                for vi, (vx, vy) in enumerate(p["tile_local_pts"]):
                    is_hovered = (hovered_vert is not None
                                   and hovered_vert[1] == p["server_id"]
                                   and hovered_vert[2] == vi)
                    bump = 2 if is_hovered else 0
                    cv2.circle(crop, (int(vx), int(vy)),
                                s(self.HANDLE_DRAW_R + 1 + bump),
                                (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)),
                                s(self.HANDLE_DRAW_R + bump),
                                (0, 0, 0), -1, cv2.LINE_AA)
                # Console diagnostic: vertex count for selected polygon in tile
                print(f"[selected] tile_id={entry['tile_id']} "
                      f"poly server_id={p['server_id']} "
                      f"vertices={len(p['tile_local_pts'])}", flush=True)
        # Diamond handles for the selected polygon:
        #   green = length (short-side, drag along trail length)
        #   blue  = width  (long-side,  drag along trail width)
        if self.selected_id is not None:
            poly_meta = self.polygons_by_id.get(self.selected_id)
            if poly_meta is not None:
                tx = entry["tile_x"]; ty = entry["tile_y"]
                hovered_ext = (self.hovered_handle
                                if self.hovered_handle and
                                   self.hovered_handle[0] == "extend"
                                else None)
                handles_list = self._compute_extend_handles(poly_meta["points"])
                for end_idx, h in enumerate(handles_list):
                    cx = int(round(h["center_full"][0] - tx))
                    cy = int(round(h["center_full"][1] - ty))
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
        # Add-mode in-progress polygon — thin lime line + dots.
        # The FIRST vertex is drawn larger as a snap-close target.
        if self.add_mode and self.add_pts_local:
            pts = np.array(self.add_pts_local, dtype=np.int32)
            line_color = (50, 255, 50)
            for i in range(len(pts) - 1):
                cv2.line(crop, tuple(pts[i]), tuple(pts[i + 1]),
                          line_color, s(2), cv2.LINE_AA)
            # Snap-close highlight: when cursor is near first vertex, show cyan line
            rb = self.image_view._rb_line
            if rb is not None and len(pts) >= 3:
                line = rb.line()
                hx, hy = int(line.x2()), int(line.y2())
                fx, fy = int(pts[0][0]), int(pts[0][1])
                if (fx - hx) ** 2 + (fy - hy) ** 2 <= 14 ** 2:
                    cv2.line(crop, (hx, hy), (fx, fy),
                              (0, 200, 255), s(2), cv2.LINE_AA)
            for vi, (vx, vy) in enumerate(pts):
                if vi == 0 and len(pts) >= 3:
                    cv2.circle(crop, (int(vx), int(vy)), s(6), (50, 255, 50),
                                -1, cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)), s(7), (0, 0, 0),
                                s(1), cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)), s(9), (255, 255, 255),
                                s(1), cv2.LINE_AA)
                else:
                    cv2.circle(crop, (int(vx), int(vy)), s(3), (50, 255, 50),
                                -1, cv2.LINE_AA)
                    cv2.circle(crop, (int(vx), int(vy)), s(4), (0, 0, 0),
                                s(1), cv2.LINE_AA)
        return crop

    def crop_to_pixmap(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def refresh_view(self):
        idx = self.filtered_indices()
        if not idx:
            self.image_view.set_pixmap(QPixmap())
            self.scrubber_label.setText("0 of 0")
            self.jump_input.setText("")
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(0)
            self.scrubber.blockSignals(False)
            self.info_label.setText("No tiles in this view.")
            self.detail_label.setText("")
            self.selection_label.setText("")
            self.mark_btn.setEnabled(False)
            self.mark_btn.setChecked(False)
            self._refresh_send_button()
            self._refresh_filter_counts()
            return

        if self.current_idx not in idx:
            self.current_idx = idx[0]

        pos = idx.index(self.current_idx) + 1
        total = len(idx)
        self.scrubber.blockSignals(True)
        self.scrubber.setMaximum(max(1, total))
        self.scrubber.setValue(pos)
        self.scrubber.blockSignals(False)
        self.scrubber_label.setText(f"{pos} of {total}")
        self.jump_input.setText(str(pos))

        entry = self.entries[self.current_idx]
        crop = self.build_display_crop(entry)
        pix = self.crop_to_pixmap(crop)
        if self._last_displayed_idx is None:
            self.image_view.set_pixmap(pix)  # first load — fit to view
        else:
            self.image_view.update_pixmap(pix)  # navigation — preserve zoom/pan
        self._last_displayed_idx = self.current_idx

        n_polys = len(entry['polys'])
        # Use the absolute CVAT frame_idx (matches the clipboard reference,
        # the filename's serial number, and TrailScreenR's "frame N" label).
        # Slice range stays in the window subtitle bar.
        tile_ref = chr(ord('A') + entry['tile_row']) + str(entry['tile_col'] + 1)
        self.info_label.setText(
            f"{entry['frame']}  ({n_polys} poly{'s' if n_polys != 1 else ''})  •  "
            f"frame {entry['frame_idx']}  •  "
            f"tile {tile_ref}  "
            f"({entry['tile_col'] + 1}/{entry['n_cols']} × "
            f"{entry['tile_row'] + 1}/{entry['n_rows']})  •  "
            f"({pos} of {total})"
        )
        self.detail_label.setText("")

        if self.selected_id is None:
            self.selection_label.setText("• click polygon to select")
            self.mark_btn.setEnabled(False)
            self.mark_btn.setChecked(False)
        else:
            marked = self.selected_id in self.marked_ids
            self.selection_label.setText(
                f"• selected id {self.selected_id} "
                f"{'(MARKED)' if marked else ''}"
            )
            self.mark_btn.setEnabled(True)
            self.mark_btn.blockSignals(True)
            self.mark_btn.setChecked(marked)
            self.mark_btn.blockSignals(False)

        self.prev_btn.setEnabled(pos > 1)
        self.next_btn.setEnabled(pos < total)

        self._refresh_send_button()
        self._refresh_filter_counts()
        self._refresh_adjacent_arrows()

    def _refresh_send_button(self):
        n_del = len(self.marked_ids)
        n_add = sum(1 for p in self.polygons_by_id.values()
                     if p["status"] == "added")
        n_edit = sum(1 for p in self.polygons_by_id.values()
                      if p["status"] == "edited")
        total = n_del + n_add + n_edit
        self.send_btn.setText(
            f"Send {total} change{'' if total == 1 else 's'} to CVAT")
        self.send_btn.setEnabled(total > 0)
        if hasattr(self, "pending_label"):
            self.pending_label.setText(
                f"Pending: {n_add} add / {n_edit} edit / {n_del} delete")

    def _refresh_filter_counts(self):
        n_with_polys = sum(1 for e in self.entries if e["polys"])
        n_no_polys = len(self.entries) - n_with_polys
        n_marked_tiles = sum(1 for e in self.entries
                             if any(p["server_id"] in self.marked_ids for p in e["polys"]))
        self.filter_buttons["all"].setText(f"All ({len(self.entries)})")
        self.filter_buttons["has_polygons"].setText(f"Has polygons ({n_with_polys})")
        if "no_polygons" in self.filter_buttons:
            self.filter_buttons["no_polygons"].setText(f"No polygons ({n_no_polys})")
        self.filter_buttons["has_marked"].setText(f"Has marked ({n_marked_tiles})")

    # ── Click / mark / unmark ───────────────────────────────────────────────

    def _on_image_clicked(self, ix, iy):
        if not self.entries:
            return
        entry = self.entries[self.current_idx]
        # Add mode: each click adds a vertex to the in-progress polygon.
        # If the click is within snap-radius of the FIRST vertex (and we have
        # at least 3 vertices placed), treat it as "close the polygon."
        if self.add_mode:
            SNAP_R = 14
            if len(self.add_pts_local) >= 4:
                fx, fy = self.add_pts_local[0]
                if (fx - ix) ** 2 + (fy - iy) ** 2 <= SNAP_R ** 2:
                    self._close_add_polygon()
                    return
            self.add_pts_local.append((ix, iy))
            # Auto-close after the 4th vertex (typical trail-rectangle case)
            if len(self.add_pts_local) >= 4:
                self._close_add_polygon()
                return
            self.refresh_view()
            return
        # Selection mode: pick smallest polygon under the click
        candidates = []
        for p in entry["polys"]:
            res = cv2.pointPolygonTest(p["tile_local_pts"], (float(ix), float(iy)), False)
            if res >= 0:
                area = cv2.contourArea(p["tile_local_pts"])
                candidates.append((area, p))
        if not candidates:
            self.selected_id = None
            self.debug_click_label.setText(
                f"click: ({ix},{iy}) — no polygon under click")
        else:
            candidates.sort(key=lambda x: x[0])
            chosen = candidates[0][1]
            self.selected_id = chosen["server_id"]
            self.debug_click_label.setText(
                f"click: selected poly {chosen['server_id']} "
                f"({len(chosen['tile_local_pts'])} vertices in this tile)")
        self.refresh_view()

    # ── Vertex-drag handlers ────────────────────────────────────────────────

    HANDLE_RADIUS = 18      # vertex hit radius (in IMAGE pixels at scale=1)
    HANDLE_DRAW_R = 4       # vertex visual radius (image pixels at scale=1)
    EXTEND_HIT_R = 24       # extend handle hit radius (image pixels at scale=1)

    def _hit_r(self, base):
        """Scale a hit radius by inverse zoom so it stays constant in SCREEN
        pixels regardless of zoom level. Floor 6 image-pixels so we don't
        end up with an impossibly tiny hit zone at extreme zoom."""
        try:
            scale = max(0.05, self.image_view.current_scale())
        except Exception:
            scale = 1.0
        return max(6.0, base / scale)

    # ── Extend handles (green diamonds at each end of selected polygon) ───

    def _compute_extend_handles(self, ff_pts):
        """Returns a list of handle dicts:
            {"kind": "length" | "width",
             "center_full": (x, y),
             "vertex_indices": [int, ...],
             "axis": np.array([dx, dy])}
        Length handles are at the short sides (perpendicular to principal axis,
        moves along principal). Width handles are at the long sides
        (perpendicular to secondary, moves along secondary).
        Empty list if PCA can't be computed."""
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
        # Normalise both axes to unit length
        n1 = float(np.hypot(principal[0], principal[1]))
        n2 = float(np.hypot(secondary[0], secondary[1]))
        if n1 < 1e-6 or n2 < 1e-6:
            return []
        principal = principal / n1
        secondary = secondary / n2

        def _make_handles(axis_vec, kind):
            proj = centered @ axis_vec
            neg_idx = [i for i, p in enumerate(proj) if p < 0]
            pos_idx = [i for i, p in enumerate(proj) if p >= 0]
            handles = []
            for indices in (neg_idx, pos_idx):
                if not indices:
                    continue
                group = pts[indices]
                center = group.mean(axis=0)
                handles.append({
                    "kind": kind,
                    "center_full": (float(center[0]), float(center[1])),
                    "vertex_indices": list(indices),
                    "axis": axis_vec.copy(),
                })
            return handles

        return _make_handles(principal, "length") + _make_handles(secondary, "width")

    def _nearest_extend_handle(self, ix, iy):
        """Find the nearest extend handle (any distance, not capped by hit
        radius). Returns (server_id, handle_dict, distance_pixels) or
        (None, None, None) if no extend handles available."""
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
            cx = cx_full - entry["tile_x"]
            cy = cy_full - entry["tile_y"]
            d2 = (cx - ix) ** 2 + (cy - iy) ** 2
            if d2 < best_dist2:
                best = h
                best_dist2 = d2
        if best is None:
            return (None, None, None)
        return (self.selected_id, best, best_dist2 ** 0.5)

    def _extend_handle_under(self, ix, iy):
        """Hit-test extend handles for the SELECTED polygon (within
        EXTEND_HIT_R). Returns (server_id, handle_dict) or (None, None)."""
        sid, h, dist = self._nearest_extend_handle(ix, iy)
        if sid is None or dist > self._hit_r(self.EXTEND_HIT_R):
            return (None, None)
        return (sid, h)

    def _vertex_under_any(self, ix, iy):
        """Return (server_id, vertex_idx) for the nearest polygon vertex
        within zoom-scaled HANDLE_RADIUS of (ix, iy). (None, None) if no hit."""
        sid, vi, dist = self._nearest_vertex(ix, iy)
        if sid is None or dist > self._hit_r(self.HANDLE_RADIUS):
            return (None, None)
        return (sid, vi)

    def _nearest_vertex(self, ix, iy):
        """Return (server_id, vertex_idx, distance_pixels) of the absolute
        nearest polygon vertex to (ix, iy). Returns (None, None, None) if no
        polygons in this tile. Used for both hit-test (caller checks distance)
        and diagnostics."""
        if not self.entries:
            return (None, None, None)
        entry = self.entries[self.current_idx]
        best = None
        best_dist2 = float("inf")
        for p in entry["polys"]:
            for vi, (vx, vy) in enumerate(p["tile_local_pts"]):
                d2 = (vx - ix) ** 2 + (vy - iy) ** 2
                if d2 < best_dist2:
                    best = (p["server_id"], vi)
                    best_dist2 = d2
        if best is None:
            return (None, None, None)
        # Apply HANDLE_RADIUS hit cap externally; here we just return nearest.
        sid, vi = best
        return (sid, vi, best_dist2 ** 0.5)

    def _on_image_pressed(self, ix, iy):
        if self.view_only:
            return
        if self.add_mode:
            self.image_view.setDragMode(QGraphicsView.NoDrag)
            self.debug_press_label.setText(
                f"press: ({ix},{iy}) in add mode")
            return
        # Closest-handle-wins: compute both distances, pick whichever is
        # closer AND within its respective hit radius. Tie → extend wins.
        ext_sid, ext_h, ext_dist = self._nearest_extend_handle(ix, iy)
        n_sid, n_vi, n_dist = self._nearest_vertex(ix, iy)
        ext_hit_r = self._hit_r(self.EXTEND_HIT_R)
        vert_hit_r = self._hit_r(self.HANDLE_RADIUS)
        ext_in_range = ext_sid is not None and ext_dist <= ext_hit_r
        vert_in_range = n_sid is not None and n_dist <= vert_hit_r
        prefer_extend = ext_in_range and (not vert_in_range or ext_dist <= n_dist)
        prefer_vertex = vert_in_range and not prefer_extend

        if prefer_extend:
            poly = self.polygons_by_id.get(ext_sid)
            if poly is not None:
                self.image_view.setDragMode(QGraphicsView.NoDrag)
                self.image_view.viewport().setCursor(Qt.ClosedHandCursor)
                entry = self.entries[self.current_idx]
                self._extend_active = True
                self._extend_polygon_id = ext_sid
                self._extend_indices = ext_h["vertex_indices"]
                self._extend_kind = ext_h.get("kind", "length")
                self._extend_axis = ext_h["axis"]
                self._extend_last_full = (
                    float(ix + entry["tile_x"]),
                    float(iy + entry["tile_y"]),
                )
                self._drag_undo_snapshot = (
                    poly["points"].copy(), poly["status"])
                self.debug_press_label.setText(
                    f"press: ({ix},{iy}) GRABBED extend handle "
                    f"({ext_dist:.0f}px from extend, "
                    f"{n_dist:.0f}px from nearest vertex)")
                self.refresh_view()
                return
        # else fall through to vertex grab
        if n_sid is None:
            self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.debug_press_label.setText(
                f"press: ({ix},{iy}) — no polygons in this tile, pan mode")
            return
        if n_dist > vert_hit_r:
            self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.debug_press_label.setText(
                f"press: ({ix},{iy}) MISS — nearest vertex was {n_dist:.0f}px "
                f"away (hit radius={vert_hit_r:.0f}px @ zoom)")
            return
        # Auto-select on press: if the nearest vertex belongs to an unselected
        # polygon, select it AND grab the vertex in the same motion. The old
        # two-step (select first, grab on second click) made users double-click,
        # which the canvas's mouseDoubleClickEvent caught as "reset zoom" and
        # the tile snapped back to fit. Bruce: "I only need to click once."
        if n_sid != self.selected_id:
            self.selected_id = n_sid
        self.image_view.setDragMode(QGraphicsView.NoDrag)
        self.selected_id = n_sid
        self._drag_polygon_id = n_sid
        self._drag_vertex_idx = n_vi
        self._drag_pending = True
        self._drag_active = False
        self._drag_press_ix = ix
        self._drag_press_iy = iy
        self._drag_full_idx = None
        # Snapshot polygon shape now so undo can restore it on release
        poly = self.polygons_by_id.get(n_sid)
        if poly is not None:
            self._drag_undo_snapshot = (
                poly["points"].copy(),
                poly["status"],
            )
        else:
            self._drag_undo_snapshot = None
        self.debug_press_label.setText(
            f"press: ({ix},{iy}) GRABBED vertex {n_vi} of poly {n_sid} "
            f"({n_dist:.1f}px away)")
        self.refresh_view()

    def _on_image_moved(self, ix, iy):
        # Extend-handle drag: the grabbed side's vertices follow the cursor
        # freely in 2D. Opposite side stays anchored. So grabbing the right
        # tip and dragging up moves the two right-side vertices up together;
        # the left tip's vertices don't move.
        if self._extend_active:
            entry = self.entries[self.current_idx]
            poly = self.polygons_by_id.get(self._extend_polygon_id)
            if poly is None:
                self.debug_move_label.setText(
                    f"move: extend — poly {self._extend_polygon_id} GONE")
                return
            cur_full = (
                float(ix + entry["tile_x"]),
                float(iy + entry["tile_y"]),
            )
            dx = cur_full[0] - self._extend_last_full[0]
            dy = cur_full[1] - self._extend_last_full[1]
            for vi in self._extend_indices:
                poly["points"][vi][0] += dx
                poly["points"][vi][1] += dy
            if self._extend_kind == "width":
                grabbed_set = set(self._extend_indices)
                for opp in self._compute_extend_handles(poly["points"]):
                    if opp["kind"] == "width" and set(opp["vertex_indices"]) != grabbed_set:
                        for vi in opp["vertex_indices"]:
                            poly["points"][vi][0] -= dx
                            poly["points"][vi][1] -= dy
                        break
            if poly["status"] == "original":
                poly["status"] = "edited"
            self._extend_last_full = cur_full
            self._reclip_polygon_in_tiles(self._extend_polygon_id)
            self.refresh_view()
            self.debug_move_label.setText(
                f"move: extend free 2D by ({dx:+.1f},{dy:+.1f})px "
                f"({len(self._extend_indices)} verts moved)")
            return
        if self._drag_pending:
            dist = ((ix - self._drag_press_ix) ** 2 + (iy - self._drag_press_iy) ** 2) ** 0.5
            if dist < self.DRAG_DEADZONE_PX:
                return
            self._drag_active = True
            self._drag_pending = False
            self.image_view.viewport().setCursor(Qt.ClosedHandCursor)
        if not self._drag_active:
            self.debug_move_label.setText(
                f"move: ({ix},{iy}) — drag_active=False, ignored")
            return
        # Update the polygon's full-frame point at drag_vertex_idx
        entry = self.entries[self.current_idx]
        poly = self.polygons_by_id.get(self._drag_polygon_id)
        if poly is None:
            self.debug_move_label.setText(
                f"move: poly id {self._drag_polygon_id} GONE")
            return
        # Convert tile-local (ix, iy) → full-frame
        fx = ix + entry["tile_x"]
        fy = iy + entry["tile_y"]
        self.debug_move_label.setText(
            f"move: drag tile ({ix},{iy}) → full ({fx},{fy})  "
            f"full_idx={getattr(self, '_drag_full_idx', None)}")
        # We track which vertex of the *clipped tile* polygon we're dragging,
        # but mutations happen on the full-frame points. We need to map the
        # tile-local vertex index back to the full-frame vertex index. Easiest
        # robust approach: at drag start, snapshot the full-frame index closest
        # to the press position.
        if not hasattr(self, "_drag_full_idx") or self._drag_full_idx is None:
            # Find closest full-frame vertex to the press point
            ff_pts = poly["points"]
            distances = (ff_pts[:, 0] - fx) ** 2 + (ff_pts[:, 1] - fy) ** 2
            self._drag_full_idx = int(np.argmin(distances))
        poly["points"][self._drag_full_idx] = [fx, fy]
        if poly["status"] == "original":
            poly["status"] = "edited"
        self._reclip_polygon_in_tiles(self._drag_polygon_id)
        self.refresh_view()

    def _on_image_released(self, ix, iy):
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
            self._extend_kind = None
            self._extend_axis = None
            self._extend_last_full = None
            if not self.add_mode:
                self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
                self.image_view.viewport().unsetCursor()
            self.debug_move_label.setText(
                f"move: extend release at ({ix},{iy})")
            self.refresh_view()
            return
        was_dragging = self._drag_active
        self._drag_pending = False
        self._drag_press_ix = None
        self._drag_press_iy = None
        # If a vertex drag actually moved the polygon, push the snapshot to
        # the undo stack now that the gesture is complete.
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
            self.refresh_view()
        # Restore default pan behavior + cursor unless still in add mode
        if not self.add_mode:
            self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.image_view.viewport().unsetCursor()
        self.debug_move_label.setText(
            f"move: release at ({ix},{iy})  was_dragging={was_dragging}")

    def _is_drag_committed(self):
        """True if _drag_full_idx was actually set (i.e., the user moved at
        least one mouse-move event during the drag)."""
        return getattr(self, "_drag_full_idx", None) is not None

    def _on_image_double_clicked(self, ix, iy):
        if self.add_mode:
            self._close_add_polygon()

    def _on_image_right_clicked(self, ix, iy):
        # In add mode: cancel the in-progress polygon
        if self.add_mode and self.add_pts_local:
            self.add_pts_local.pop()  # pop last vertex
            self.refresh_view()

    def _on_image_hovered(self, ix, iy):
        # In add mode, draw a scene rubber-band from the last vertex to the cursor.
        # Using a QGraphicsLineItem means the line extends into the gray area beyond
        # the tile boundary, not just within the 640px cv2 image.
        if self.add_mode:
            if self.add_pts_local:
                lx, ly = self.add_pts_local[-1]
                self.image_view.set_rubber_band(lx, ly, ix, iy)
                self.refresh_view()
            return
        if self._drag_active or self._extend_active:
            return
        viewport = self.image_view.viewport()

        # Determine what's hovered using the SAME closest-wins rule as press.
        ext_sid, ext_h, ext_dist = self._nearest_extend_handle(ix, iy)
        n_sid, n_vi, n_dist = self._nearest_vertex(ix, iy)
        ext_hit_r = self._hit_r(self.EXTEND_HIT_R)
        vert_hit_r = self._hit_r(self.HANDLE_RADIUS)
        ext_in = ext_sid is not None and ext_dist <= ext_hit_r
        vert_in = n_sid is not None and n_dist <= vert_hit_r
        prefer_extend = ext_in and (not vert_in or ext_dist <= n_dist)

        new_hover = None
        if prefer_extend:
            # Build a stable end-index tag from the handle's center
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
        # Cursor left the window — clear hover so handles don't stay highlighted
        if self.hovered_handle is not None:
            self.hovered_handle = None
            self.refresh_view()
        super().leaveEvent(event)

    # ── Add mode ────────────────────────────────────────────────────────────

    def _toggle_add_mode(self):
        if self.view_only:
            return
        self.add_mode = not self.add_mode
        self.image_view.draw_mode = self.add_mode
        self.add_btn.blockSignals(True)
        self.add_btn.setChecked(self.add_mode)
        self.add_btn.blockSignals(False)
        if self.add_mode:
            self.image_view.viewport().setCursor(Qt.CrossCursor)
            self.selected_id = None
        else:
            self.image_view.viewport().unsetCursor()
            self.add_pts_local = []
            self.image_view.clear_rubber_band()
        self.add_hover_xy = None
        self.refresh_view()

    def _close_add_polygon(self):
        if not self.add_mode or not self.entries:
            return
        if len(self.add_pts_local) < 3:
            QMessageBox.information(self, "Need 3+ vertices",
                "A polygon needs at least 3 vertices. Click more, "
                "or press ESC to cancel.")
            return
        entry = self.entries[self.current_idx]
        # Convert tile-local pts → full-frame
        ff_pts = np.array([(x + entry["tile_x"], y + entry["tile_y"])
                            for x, y in self.add_pts_local], dtype=np.float32)
        temp_id = self.next_temp_id
        self.next_temp_id -= 1
        # Stable color per polygon — append index after all existing polys
        next_color = len(self.polygons_by_id)
        self.polygons_by_id[temp_id] = {
            "server_id": temp_id,
            "frame_idx": entry["frame_idx"],
            "color_index": next_color,
            "points": ff_pts,
            "original_points": ff_pts.copy(),
            "label": None,
            "status": "added",
        }
        # Reclip into all affected tiles for this frame
        self._reclip_polygon_in_tiles(temp_id)
        # Push undo entry — undoing the add removes the polygon
        self._push_undo(("add", temp_id))
        # Exit add mode
        self.add_pts_local = []
        self.image_view.clear_rubber_band()
        self.selected_id = temp_id
        self._toggle_add_mode()  # switches off add_mode + restores cursor

    def _on_escape(self):
        if self.add_mode and self.add_pts_local:
            self.add_pts_local = []
            self.refresh_view()
            return
        if self.add_mode:
            self._toggle_add_mode()
            return
        self._clear_selection()

    # ── Reclip a polygon into all relevant tile entries ─────────────────────

    def _reclip_polygon_in_tiles(self, server_id):
        poly = self.polygons_by_id.get(server_id)
        if poly is None:
            return
        ff_pts = poly["points"]
        if ff_pts.shape[0] < 3:
            return
        frame_idx = poly["frame_idx"]
        color_index = poly["color_index"]
        px0, py0 = ff_pts[:, 0].min(), ff_pts[:, 1].min()
        px1, py1 = ff_pts[:, 0].max(), ff_pts[:, 1].max()

        for entry in self.entries:
            if entry["frame_idx"] != frame_idx:
                continue
            tx, ty = entry["tile_x"], entry["tile_y"]
            tx2 = tx + TILE_SIZE
            ty2 = ty + TILE_SIZE
            # Strip any existing copies of this polygon from this tile
            entry["polys"] = [p for p in entry["polys"] if p["server_id"] != server_id]
            if px1 <= tx or px0 >= tx2 or py1 <= ty or py0 >= ty2:
                continue
            # Geometric clip — keeps the original CVAT vertices intact.
            rings = clip_polygon_to_tile(ff_pts, tx, ty, tx2, ty2)
            for c_pts in rings:
                entry["polys"].append({
                    "server_id": server_id,
                    "color_index": color_index,
                    "tile_local_pts": c_pts,
                    "frame_idx": frame_idx,
                })

    # ── Undo ────────────────────────────────────────────────────────────────

    def _push_undo(self, op):
        """op is a tuple: (kind, ...payload). Pushed to a bounded stack."""
        self.undo_stack.append(op)
        if len(self.undo_stack) > self.UNDO_MAX:
            self.undo_stack = self.undo_stack[-self.UNDO_MAX:]
        self._refresh_undo_button()

    def _refresh_undo_button(self):
        n = len(self.undo_stack)
        self.undo_btn.setText(f"Undo ({n})")
        self.undo_btn.setEnabled(n > 0)

    def _undo(self):
        if not self.undo_stack:
            return
        op = self.undo_stack.pop()
        kind = op[0]
        if kind == "vertex":
            sid, pts_before, status_before = op[1], op[2], op[3]
            poly = self.polygons_by_id.get(sid)
            if poly is not None:
                poly["points"] = pts_before
                poly["status"] = status_before
                self._reclip_polygon_in_tiles(sid)
        elif kind == "mark":
            sid = op[1]
            self.marked_ids.discard(sid)
            self._save_marks()
        elif kind == "unmark":
            sid = op[1]
            self.marked_ids.add(sid)
            self._save_marks()
        elif kind == "add":
            temp_id = op[1]
            self.polygons_by_id.pop(temp_id, None)
            for entry in self.entries:
                entry["polys"] = [p for p in entry["polys"]
                                   if p["server_id"] != temp_id]
            if self.selected_id == temp_id:
                self.selected_id = None
        self._refresh_undo_button()
        self._refresh_send_button()
        self._refresh_filter_counts()
        self.refresh_view()
        self.debug_label.setText(f"undo: {kind} (stack now {len(self.undo_stack)})")

    def _toggle_selected_mark(self):
        if self.view_only:
            return
        if self.selected_id is None:
            return
        sid = self.selected_id
        if sid in self.marked_ids:
            self.marked_ids.discard(sid)
            # Undo of "unmark" is to re-mark
            self._push_undo(("unmark", sid))
        else:
            self.marked_ids.add(sid)
            # Undo of "mark" is to unmark
            self._push_undo(("mark", sid))
        self._save_marks()
        self.refresh_view()

    def _clear_selection(self):
        self.selected_id = None
        self.refresh_view()

    def _launch_mask_checkr(self):
        last_pick = Path.home() / ".star_trail_cleanr" / "mask_checkr_last.json"
        last_pick.parent.mkdir(parents=True, exist_ok=True)
        try:
            state = json.loads(last_pick.read_text())
        except Exception:
            state = {}
        state["task_id"] = CVAT_TASK_ID
        last_pick.write_text(json.dumps(state))
        script = Path(__file__).parent / "mask_checkr.py"
        subprocess.Popen([sys.executable, str(script)])
        self._maskcheckr_anim_step = 0
        self._maskcheckr_anim_timer = QTimer(self)
        self._maskcheckr_anim_timer.setInterval(400)
        self._maskcheckr_anim_timer.timeout.connect(self._tick_maskcheckr_animation)
        self._maskcheckr_anim_timer.start()
        self._tick_maskcheckr_animation()

    def _tick_maskcheckr_animation(self):
        labels = ["Opening", "Opening.", "Opening..", "Opening..."]
        self.mask_checkr_btn.setText(labels[self._maskcheckr_anim_step % len(labels)])
        self._maskcheckr_anim_step += 1
        if self._maskcheckr_anim_step >= len(labels):
            self._maskcheckr_anim_timer.stop()
            self._maskcheckr_anim_timer = None
            self.mask_checkr_btn.setText("Mask CheckR")

    # ── Send to CVAT ────────────────────────────────────────────────────────

    def _send_to_cvat(self):
        if self._send_worker is not None and self._send_worker.isRunning():
            return
        adds = [p for p in self.polygons_by_id.values() if p["status"] == "added"]
        edits = [p for p in self.polygons_by_id.values() if p["status"] == "edited"]
        total = len(self.marked_ids) + len(adds) + len(edits)
        if total == 0:
            return

        adds_copy = [
            {"server_id": p["server_id"], "frame_idx": p["frame_idx"],
             "points": p["points"].copy()}
            for p in adds
        ]
        edits_copy = [
            {"server_id": p["server_id"], "frame_idx": p["frame_idx"],
             "points": p["points"].copy()}
            for p in edits
        ]

        deleted_frames = {p["frame_idx"] for p in self.polygons_by_id.values()
                          if p["server_id"] in self.marked_ids}
        self._pending_send_frames = {p["frame_idx"] for p in adds + edits} | deleted_frames

        password = read_cvat_password()
        auth = (CVAT_USER, password)

        self._send_worker = CvatSendWorker(
            CVAT_TASK_ID, self.job_id,
            set(self.marked_ids), adds_copy, edits_copy, auth,
        )
        self._send_worker.finished.connect(self._on_send_finished)
        self._send_worker.error.connect(self._on_send_error)

        self.send_btn.setEnabled(False)
        self._send_anim_step = 0
        self._send_anim_timer = QTimer(self)
        self._send_anim_timer.setInterval(400)
        self._send_anim_timer.timeout.connect(self._tick_send_animation)
        self._send_anim_timer.start()
        self._tick_send_animation()

        self._send_worker.start()

    def _tick_send_animation(self):
        labels = ["Sending", "Sending.", "Sending..", "Sending..."]
        self.send_btn.setText(labels[self._send_anim_step % len(labels)])
        self._send_anim_step += 1

    def _stop_send_animation(self):
        if self._send_anim_timer is not None:
            self._send_anim_timer.stop()
            self._send_anim_timer = None

    def _on_send_finished(self, result):
        self._stop_send_animation()
        errors = result.get("errors", [])
        if errors:
            msg = "CVAT send had errors:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                msg += f"\n(+ {len(errors) - 5} more)"
            QMessageBox.critical(self, "CVAT error", msg)
        self.marked_ids.clear()
        for poly in list(self.polygons_by_id.values()):
            if poly["status"] in ("added", "edited"):
                poly["status"] = "original"
                poly["original_points"] = poly["points"].copy()
        self.undo_stack.clear()
        self._save_marks()
        self._refresh_undo_button()
        auth = (CVAT_USER, read_cvat_password())
        frames_to_refresh = getattr(self, "_pending_send_frames", set())
        if self.entries:
            frames_to_refresh.add(self.entries[self.current_idx]["frame_idx"])
        for frame_idx in frames_to_refresh:
            self._refresh_frame_from_cvat(frame_idx, auth)
        self._pending_send_frames = set()
        self._refresh_send_button()
        self.refresh_view()
        self._send_worker = None

    def _on_send_error(self, msg):
        self._stop_send_animation()
        QMessageBox.critical(self, "CVAT error", f"CVAT update failed:\n{msg}")
        self._refresh_send_button()
        self.refresh_view()
        self._send_worker = None

    def _pull_from_cvat(self):
        if not self.entries:
            return
        self.pull_btn.setEnabled(False)
        self.pull_btn.setText("Pulling...")
        QApplication.processEvents()
        try:
            auth = (CVAT_USER, read_cvat_password())
            ann_resp = requests.get(
                f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                auth=auth).json()
            current_frame_idx = self.entries[self.current_idx]["frame_idx"]
            frame_indices = sorted({e["frame_idx"] for e in self.entries
                                    if abs(e["frame_idx"] - current_frame_idx) <= 10})
            for frame_idx in frame_indices:
                self._refresh_frame_from_cvat(frame_idx, auth, ann_resp=ann_resp)
        except Exception as exc:
            QMessageBox.critical(self, "CVAT error", f"Pull failed:\n{exc}")
        finally:
            self.pull_btn.setEnabled(True)
            self.pull_btn.setText("Pull from CVAT")
        self.refresh_view()

    def _refresh_frame_from_cvat(self, frame_idx, auth, ann_resp=None):
        try:
            if ann_resp is None:
                ann_resp = requests.get(
                    f"{CVAT_URL}/api/jobs/{self.job_id}/annotations",
                    auth=auth).json()
        except Exception:
            return
        fresh_shapes = [s for s in ann_resp.get("shapes", [])
                        if s.get("frame") == frame_idx]
        old_ids = {sid for sid, p in self.polygons_by_id.items()
                   if p["frame_idx"] == frame_idx}
        for entry in self.entries:
            if entry["frame_idx"] == frame_idx:
                entry["polys"] = [p for p in entry["polys"]
                                  if p["server_id"] not in old_ids]
        for sid in old_ids:
            self.polygons_by_id.pop(sid, None)
        if self.selected_id in old_ids:
            self.selected_id = None
        for shape in fresh_shapes:
            raw = shape.get("points", [])
            if len(raw) < 6:
                continue
            pts = np.array([[raw[i], raw[i + 1]]
                            for i in range(0, len(raw) - 1, 2)],
                           dtype=np.float32)
            sid = shape["id"]
            color_index = sid % len(POLY_COLORS)
            self.polygons_by_id[sid] = {
                "server_id": sid,
                "frame_idx": frame_idx,
                "color_index": color_index,
                "points": pts,
                "original_points": pts.copy(),
                "label": shape.get("label_id"),
                "status": "original",
            }
            self._reclip_polygon_in_tiles(sid)

    # ── Copy reference ──────────────────────────────────────────────────────

    def _copy_reference(self):
        if not self.entries:
            return
        entry = self.entries[self.current_idx]
        tile_ref = chr(ord('A') + entry['tile_row']) + str(entry['tile_col'] + 1)
        ref = (
            f"TileFixR | task {CVAT_TASK_ID} | {IMG_DIR.name} | "
            f"frame {entry['frame_idx']} ({entry['img_filename']}) | "
            f"tile {tile_ref} "
            f"(col {entry['tile_col'] + 1}/{entry['n_cols']} "
            f"row {entry['tile_row'] + 1}/{entry['n_rows']}) | "
            f"origin ({entry['tile_x']}, {entry['tile_y']})"
        )
        QApplication.clipboard().setText(ref)
        original = self.copy_btn.text()
        self.copy_btn.setText("Copied")
        QTimer.singleShot(1200, lambda: self.copy_btn.setText(original))

    # ── CVAT deep link ──────────────────────────────────────────────────────

    def _open_in_cvat(self):
        if not self.entries:
            return
        entry = self.entries[self.current_idx]
        ts = int(time.time())
        url = f"{CVAT_URL}/tasks/{CVAT_TASK_ID}/jobs/{self.job_id}?frame={entry['frame_idx']}&_t={ts}"
        webbrowser.open_new_tab(url)

    # ── Persistence ─────────────────────────────────────────────────────────

    def _restore_position(self):
        if not self.entries:
            return 0
        state = self._load_saved_state()
        saved = state.get("tile_id")
        if saved:
            for i, e in enumerate(self.entries):
                if e["tile_id"] == saved:
                    return i
        idx = state.get("index", 0)
        if 0 <= idx < len(self.entries):
            return idx
        return 0

    def _load_saved_state(self):
        try:
            if STATE_PATH.exists():
                return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
        return {}

    def _save_state(self):
        if not self.entries or self.current_idx >= len(self.entries):
            return
        try:
            t = self.image_view.transform()
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps({
                "tile_id":        self.entries[self.current_idx]["tile_id"],
                "index":          self.current_idx,
                "filter_mode":    self.filter_mode,
                "tile_filter":    list(self.tile_filter) if self.tile_filter else None,
                "zoom_transform": [t.m11(), t.m12(), t.m13(),
                                   t.m21(), t.m22(), t.m23(),
                                   t.m31(), t.m32(), t.m33()],
                "zoom_level":     self.image_view._zoom_level,
                "scroll_x":       self.image_view.horizontalScrollBar().value(),
                "scroll_y":       self.image_view.verticalScrollBar().value(),
                "brightness":     self.bright_slider.value(),
                "contrast":       self.contrast_slider.value(),
            }, indent=2))
        except Exception:
            pass

    def _restore_zoom(self):
        state = self._load_saved_state()
        t_vals = state.get("zoom_transform")
        if t_vals and len(t_vals) == 9 and state.get("zoom_level", 0) != 0:
            self.image_view.setTransform(
                QTransform(t_vals[0], t_vals[1], t_vals[2],
                           t_vals[3], t_vals[4], t_vals[5],
                           t_vals[6], t_vals[7], t_vals[8]))
            self.image_view._zoom_level = state["zoom_level"]
            sx = state.get("scroll_x", 0)
            sy = state.get("scroll_y", 0)
            self.image_view.horizontalScrollBar().setValue(sx)
            self.image_view.verticalScrollBar().setValue(sy)

    def _load_marks(self):
        try:
            if MARKS_PATH.exists():
                data = json.loads(MARKS_PATH.read_text())
                self.marked_ids = set(int(x) for x in data.get("marked_server_ids", []))
        except Exception:
            self.marked_ids = set()

    def _save_marks(self):
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            MARKS_PATH.write_text(json.dumps({
                "task_id": CVAT_TASK_ID,
                "marked_server_ids": sorted(self.marked_ids),
            }, indent=2))
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_state()
        super().closeEvent(event)

    def _relaunch(self):
        self._save_state()
        subprocess.Popen([sys.executable, os.path.abspath(__file__)])
        self.close()
        QApplication.quit()


# ── Pill-style filter button ────────────────────────────────────────────────

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


class ScanWorker(QThread):
    progress = Signal(int, int)   # current, total
    finished = Signal(list)       # list of flagged frame indices
    failed   = Signal(str)

    def __init__(self, img_dir, tile_x, tile_y, tile_size=640):
        super().__init__()
        self.img_dir   = Path(img_dir)
        self.tile_x    = tile_x
        self.tile_y    = tile_y
        self.tile_size = tile_size

    def run(self):
        try:
            from PIL import Image
            from scipy import ndimage

            img_dir = self.img_dir
            ox, oy, ts = self.tile_x, self.tile_y, self.tile_size

            frames = sorted([f for f in img_dir.iterdir()
                             if f.suffix.lower() == ".jpg" and f.is_file()])
            if not frames:
                self.failed.emit("No JPG frames found")
                return

            mask_path = img_dir / "cleanr_workspace" / "foreground_mask.png"
            if not mask_path.exists():
                self.failed.emit("No foreground_mask.png in cleanr_workspace")
                return

            mask_full = np.array(Image.open(mask_path).convert("L"))
            mh, mw = mask_full.shape
            tile_mask_raw = mask_full[oy:min(oy+ts, mh), ox:min(ox+ts, mw)]
            tile_mask = np.zeros((ts, ts), dtype=np.uint8)
            tile_mask[:tile_mask_raw.shape[0], :tile_mask_raw.shape[1]] = tile_mask_raw
            is_sky = tile_mask < 128
            is_fg  = ~is_sky

            THRESH_SKY, THRESH_FG, MIN_PX, SAMPLE_N = 18, 28, 8, 75

            def crop_tile(fpath):
                arr = np.array(Image.open(fpath).crop((ox, oy, ox+ts, oy+ts))).astype(np.float32)
                if arr.shape[:2] != (ts, ts):
                    pad = np.zeros((ts, ts, 3), dtype=np.float32)
                    pad[:arr.shape[0], :arr.shape[1]] = arr
                    arr = pad
                return arr

            sample = random.sample(frames, min(SAMPLE_N, len(frames)))
            median = np.median([crop_tile(f) for f in sample], axis=0).astype(np.float32)

            cache = {}

            def get_binary(idx):
                if idx not in cache:
                    diff = (crop_tile(frames[idx]) - median).mean(axis=2)
                    cache[idx] = ((diff > THRESH_SKY) & is_sky) | ((diff > THRESH_FG) & is_fg)
                    for k in list(cache):
                        if k < idx - 2:
                            del cache[k]
                return cache[idx]

            flagged = []
            total = len(frames)
            for i in range(total):
                self.progress.emit(i + 1, total)
                binary = get_binary(i)
                from scipy import ndimage as _ndi
                labeled, n = _ndi.label(binary)
                for comp in range(1, n + 1):
                    if (labeled == comp).sum() < MIN_PX:
                        continue
                    ys, xs = np.where(labeled == comp)
                    in_neighbor = any(
                        0 <= ni < total and get_binary(ni)[ys, xs].mean() > 0.4
                        for ni in (i - 1, i + 1)
                    )
                    if not in_neighbor:
                        flagged.append(i)
                        break

            self.finished.emit(flagged)

        except Exception as e:
            self.failed.emit(str(e))


def make_pill(text, color_key="blue"):
    colors = {
        "blue":   {"bg": "#e8f0fe", "fg": "#1a5276", "border": "#b0c4de",
                    "checked_bg": "#1a6fc4", "checked_border": "#145da0", "hover_bg": "#d0e0f0"},
        "green":  {"bg": "#e8fee8", "fg": "#1e7a1e", "border": "#b0deb0",
                    "checked_bg": "#2a7a2a", "checked_border": "#1e5f1e", "hover_bg": "#d0f0d0"},
        "red":    {"bg": "#fde8e8", "fg": "#922b21", "border": "#e0b0b0",
                    "checked_bg": "#c0392b", "checked_border": "#a93226", "hover_bg": "#f8d0d0"},
        "amber":  {"bg": "#fff7d6", "fg": "#7a5a00", "border": "#e0cc80",
                    "checked_bg": "#d39e00", "checked_border": "#a87c00", "hover_bg": "#ffeeb0"},
        "purple": {"bg": "#f3e8fe", "fg": "#5b2d8e", "border": "#c9a0e0",
                    "checked_bg": "#7b3fc4", "checked_border": "#5b2d8e", "hover_bg": "#e8d0f8"},
    }
    c = colors.get(color_key, colors["blue"])
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet(PILL_STYLE.format(**c))
    return btn


class SplashWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.cancelled = False
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(420, 185)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(8)

        title = QLabel("TileFixR")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Loading from CVAT")
        sub.setStyleSheet("font-size: 12px; color: #a8c0e0;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("font-size: 12px; color: #e6e6e6;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
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
        cancel_btn.clicked.connect(lambda: setattr(self, "cancelled", True))
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setStyleSheet("background-color: #0a1e3f; border-radius: 12px;")

    def update_progress(self, text, current=0, total=0):
        self.status_label.setText(text)
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        else:
            self.progress.setRange(0, 0)
        QApplication.processEvents()
        return self.cancelled


def main():
    global CVAT_TASK_ID, FRAME_START, FRAME_END, IMG_DIR, TASK_NAME
    print("TileFixR")
    print("=" * 60, flush=True)

    app = QApplication(sys.argv)
    lock_dir = Path.home() / ".star_trail_cleanr"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(lock_dir / "tile_fixr.lock"))
    lock.setStaleLockTime(10_000)
    if not lock.tryLock(3000):
        QMessageBox.warning(None, "TileFixR already running",
            "Another TileFixR is already running. Switch to that window.")
        return

    _apply_theme()

    # ─ Task picker ─
    auth = (CVAT_USER, read_cvat_password())
    print("Fetching CVAT tasks...", flush=True)
    tasks = fetch_cvat_tasks(auth)
    if not tasks:
        QMessageBox.critical(None, "No CVAT tasks",
            "Could not fetch tasks from CVAT. Is CVAT running on "
            f"{CVAT_URL}?")
        return
    last = load_last_pick()
    picker = TaskPickerDialog(
        tasks, last["task_id"],
        last.get("first_frame", 1),
        last.get("last_frame", 9999),
        auth=auth,
        last_source_mode=last.get("source_mode", "cvat"),
    )
    if picker.exec() != QDialog.Accepted:
        print("Cancelled.")
        return
    chosen = picker.selected_task()
    first_f = picker.selected_first_frame()
    last_f = picker.selected_last_frame()
    mode = picker.selected_mode()
    if chosen is None:
        return

    # Resolve image dir, set globals for the rest of the session
    img_dir = resolve_image_dir(chosen["name"])
    if img_dir is None:
        QMessageBox.critical(None, "Image folder not found",
            f"Couldn't find an image folder for '{chosen['name']}' under "
            f"{TRAILS_ROOT}. Pick a different task or fix the folder mapping.")
        return
    CVAT_TASK_ID = chosen["id"]
    TASK_NAME = chosen["name"]
    FRAME_START = first_f - 1
    FRAME_END = last_f
    IMG_DIR = img_dir / "cleaned" if mode == "cleaned" else img_dir
    _refresh_paths()
    save_last_pick(CVAT_TASK_ID, first_f, last_f, source_mode=mode)
    print(f"  selected task {CVAT_TASK_ID}: {chosen['name']}", flush=True)
    if SUSPECT_FRAMES:
        print(f"  suspect mode: {len(SUSPECT_FRAMES)} specific frames from {IMG_DIR}", flush=True)
    else:
        print(f"  loading frames {FRAME_START}-{FRAME_END - 1} "
              f"({FRAME_END - FRAME_START} total) from {IMG_DIR}", flush=True)

    splash = SplashWindow()
    splash.show()
    splash.update_progress("Connecting to CVAT...")

    by_frame, frame_names, job_id = load_cvat_polygons(progress_cb=splash.update_progress)
    if mode == "cleaned":
        entries, all_polys = build_tile_entries_from_masks(
            img_dir / "cleaned" / "masks", frame_names,
            progress_cb=splash.update_progress)
        view_only = True
    else:
        entries, all_polys = build_tile_entries(by_frame, frame_names,
            progress_cb=splash.update_progress)
        view_only = False

    if splash.cancelled:
        splash.close()
        print("Cancelled.")
        return

    print(f"\nLoaded {len(entries)} tiles, {len(all_polys)} polygons.")

    if not entries:
        splash.close()
        QMessageBox.critical(None, "No tiles",
            "No tiles found — check that the task has frames in the local "
            f"folder ({IMG_DIR}).")
        return

    splash.update_progress("Building the editor window...")
    window = TileFixR(entries, all_polys, frame_names, job_id, view_only=view_only)
    splash.close()
    window.show()
    window.activateWindow()
    window.setFocus()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
