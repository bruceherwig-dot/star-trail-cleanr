#!/usr/bin/env python3
"""mask_viewr.py — full-frame Star Trail CleanR detection mask viewer (YOLO MaskViewR).

What it is
----------
A maintained developer review app. It walks through a dataset frame by frame
and overlays, on each source image, every trail region Star Trail CleanR
detected on that frame. You see, in one place, what the detector found and how
the pipeline turned those detections into the masks it actually repaired.

What it shows on each frame
---------------------------
- Colored, numbered outlines, one per repaired trail. When per-polygon JSON is
  present, each fitted polygon (including separately-split crossing arms) gets
  its own numbered outline; otherwise it falls back to outlining each connected
  blob in the saved mask PNG.
- Thin yellow outlines for the raw SAHI/YOLO detections (the "_raw" mask),
  drawn underneath so you can compare raw hits against the cleaned-up mask.
- An optional tile grid matching the SAHI 640x640 slicing, with A/B/C row
  letters down the left and 1/2/3 column numbers across the top, so a region
  can be named by tile (e.g. "B3").
- A clear "No detection" label on frames with nothing detected.

How it reads a frame
--------------------
For each source image `IMG.jpg` it looks in `<image folder>/cleanr_workspace/masks/`
for `IMG.png` (the final repaired mask), `IMG_raw.png` (raw YOLO/SAHI hits), and
`IMG_polys.json` (per-polygon fit data). Missing files degrade gracefully.

Interaction
-----------
- Arrow keys / Prev / Next / slider / type-a-frame to navigate.
- Scroll or pinch to zoom, drag to pan, double-click to fit.
- Click a trail or raw detection to select it and read its tile, centroid,
  bounding box, and pixel area.
- "Copy Frame" puts a one-line reference (frame, dataset, selected id, tile,
  coords, image path, mask path, mask write time) on the clipboard.
- "Add To WeirdR" appends the current frame to `weirdr_list.json` (a running
  list of odd frames worth revisiting).
- "Tiles" toggles the grid; "Relaunch" restarts the app; "O" reopens the
  folder picker; "Q"/Escape quits.

How to run
----------
`python3 tools/mask_viewr.py` — a setup dialog asks for the source images
folder (the masks folder is found automatically beside it). The last folder and
frame are remembered in `~/.star_trail_cleanr/mask_viewr_config.json`.

Where it fits
-------------
One of Bruce's annotation/detection review tools. Unlike Mask CheckR (which
pulls reviewed polygons from CVAT), MaskViewR reads the masks Star Trail CleanR
itself wrote to disk, so it answers "what did the live pipeline actually
detect and repair on this frame?"
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QEvent, QPointF, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget,
)

# Where the app remembers its last folder/frame, the flagged-frames list it
# appends to, the image types it will open, and the on-screen display cap.
STATE_DIR    = Path.home() / ".star_trail_cleanr"          # hidden per-user state folder
CONFIG_FILE  = STATE_DIR / "mask_viewr_config.json"        # remembers last folder + frame
WEIRDR_PATH  = Path(__file__).parent.parent / "weirdr_list.json"  # running "odd frame" list
IMAGE_EXTS   = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}  # source image types it will load
MAX_W, MAX_H = 1800, 1080                                  # largest on-screen render before downscaling

# One color per detected trail (BGR, same palette as TileFixR)
TRAIL_COLORS = [
    (0,   0,   255),   # red
    (255, 0,   255),   # magenta
    (0,   200, 50),    # green
    (0,   165, 255),   # orange
    (255, 255, 0),     # yellow
    (180, 105, 255),   # hot pink
    (255, 128, 0),     # blue
    (0,   215, 255),   # gold
    (50,  205, 50),    # lime
    (0,   128, 255),   # sky blue
]


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    """Return the saved settings dict (last folder, last frame), or {} if none/unreadable."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def save_config(data):
    """Write the settings dict back to disk, creating the state folder if needed."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


# ── Zoomable image view (adapted from TileFixR) ───────────────────────────────

class ZoomableImageView(QGraphicsView):
    """The image canvas: shows the rendered frame and handles zoom/pan/click.

    Scroll wheel or pinch zooms, drag pans, double-click fits the image. A plain
    click (mouse barely moved) fires the `clicked` signal with the scene
    coordinate, which the main window uses to select a trail. It also paints the
    tile-grid row/column labels pinned to the viewport edges so they stay put
    while you zoom and pan. Adapted from the matching view in TileFixR.
    """
    clicked = Signal(QPointF)

    def __init__(self, parent=None):
        # Build a single-image graphics scene set up for hand-drag panning and
        # pinch-zoom, on a dark background. _tile_bboxes holds the grid cells (in
        # scene coords) used to draw the edge labels; _mouse_press_pos tracks
        # where a press began so we can tell a click from a drag.
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setStyleSheet("background: #0d0d1a; border: none;")
        self.grabGesture(Qt.PinchGesture)
        self._tile_bboxes = []  # display-coord bboxes for persistent grid labels
        self._mouse_press_pos = None

    def mousePressEvent(self, event):
        """Remember where a left-button press started (to distinguish click from drag)."""
        if event.button() == Qt.LeftButton:
            self._mouse_press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """On left release, if the mouse barely moved (<5px) treat it as a click and emit it."""
        if event.button() == Qt.LeftButton and self._mouse_press_pos is not None:
            delta = event.pos() - self._mouse_press_pos
            if (delta.x() ** 2 + delta.y() ** 2) ** 0.5 < 5:
                self.clicked.emit(self.mapToScene(event.pos()))
        self._mouse_press_pos = None
        super().mouseReleaseEvent(event)

    def set_tile_bboxes(self, bboxes):
        """Store tile bboxes in display (scene) coords for overlay label drawing."""
        self._tile_bboxes = bboxes
        self.viewport().update()

    def _fit_with_strips(self):
        """Zoom-to-fit the image while reserving a 40px strip on the top and left edges.

        The reserved strip is where the tile row/column labels live, so this keeps
        the whole image visible without the labels covering its edges.
        """
        BOX_SIZE = 40
        sr = self._scene.sceneRect()
        if not sr.isValid() or sr.isEmpty():
            return
        vp = self.viewport()
        vw, vh = vp.width(), vp.height()
        avail_w = max(1.0, vw - BOX_SIZE)
        avail_h = max(1.0, vh - BOX_SIZE)
        s = min(avail_w / sr.width(), avail_h / sr.height())
        iw  = sr.width()  * s
        ih  = sr.height() * s
        ox  = BOX_SIZE + (avail_w - iw) / 2.0
        oy  = BOX_SIZE + (avail_h - ih) / 2.0
        margin = BOX_SIZE / s
        self._scene.setSceneRect(QRectF(
            sr.left() - margin, sr.top() - margin,
            sr.width() + margin, sr.height() + margin,
        ))
        self.resetTransform()
        self.scale(s, s)
        cx = sr.left() + (vw / 2.0 - ox) / s
        cy = sr.top()  + (vh / 2.0 - oy) / s
        self.centerOn(cx, cy)

    def drawForeground(self, painter, rect):
        """Draw tile row letters and column numbers pinned to the viewport edges.

        Runs on every repaint. It maps each stored tile box into current viewport
        pixels, paints a white label strip down the left edge (row letters A, B,
        C...) and across the top edge (column numbers 1, 2, 3...) with blue
        borders, so any region can be named by its tile while the image is zoomed
        or panned. Does nothing if there are no tile boxes or none are on screen.
        """
        super().drawForeground(painter, rect)
        if not self._tile_bboxes:
            return

        painter.save()
        painter.resetTransform()

        vp       = self.viewport().rect()
        BLUE     = QColor(50, 110, 200)
        BOX_SIZE = 40

        font = QFont("Arial", 26, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()

        row_spans = {}
        col_spans = {}

        for (sx1, sy1, sx2, sy2, row, col) in self._tile_bboxes:
            vp_tl = self.mapFromScene(QPointF(sx1, sy1))
            vp_br = self.mapFromScene(QPointF(sx2, sy2))
            vy1, vy2 = vp_tl.y(), vp_br.y()
            vx1, vx2 = vp_tl.x(), vp_br.x()
            if vy2 > 0 and vy1 < vp.height() and row not in row_spans:
                row_spans[row] = (max(int(vy1), 0), int(vy2))
            if vx2 > 0 and vx1 < vp.width() and col not in col_spans:
                col_spans[col] = (max(int(vx1), 0), int(vx2))

        if not row_spans or not col_spans:
            painter.restore()
            return

        sorted_rows = sorted(row_spans.items())
        sorted_cols = sorted(col_spans.items())
        row_starts  = [a for _, (a, b) in sorted_rows]
        col_starts  = [a for _, (a, b) in sorted_cols]

        # === ROW LABEL STRIP (left edge) ===
        # Pass 1: full-tile fills (vy1 to vy2); overlap zones painted white by both neighbours
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        for i, (_, (vy1, vy2)) in enumerate(sorted_rows):
            painter.drawRect(0, vy1, BOX_SIZE, min(vy2, vp.height()) - vy1)

        # Pass 2: borders at both vy1 and vy2 of every row
        row_ends  = [vy2 for _, (vy1, vy2) in sorted_rows]
        all_row_y = sorted(set(row_starts + row_ends))
        painter.setPen(QPen(BLUE, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(0,        all_row_y[0], 0,        all_row_y[-1])
        painter.drawLine(BOX_SIZE, all_row_y[0], BOX_SIZE, all_row_y[-1])
        for vy in all_row_y:
            painter.drawLine(0, vy, BOX_SIZE, vy)

        painter.setPen(BLUE)
        for i, (row_idx, (vy1, vy2)) in enumerate(sorted_rows):
            prev_vy2 = sorted_rows[i - 1][1][1] if i > 0 else vy1
            next_vy1 = sorted_rows[i + 1][1][0] if i + 1 < len(sorted_rows) else vy2
            ex_top   = max(max(vy1, prev_vy2), 0)
            ex_bot   = min(min(vy2, next_vy1), vp.height())
            center_y = (ex_top + ex_bot) // 2
            letter   = chr(ord('A') + row_idx)
            tw       = fm.horizontalAdvance(letter)
            tx       = (BOX_SIZE - tw) // 2
            ty       = center_y + fm.ascent() // 2
            painter.drawText(tx, ty, letter)

        # === COLUMN LABEL STRIP (top edge) ===
        # Pass 1: full-tile fills (vx1 to vx2); overlap zones painted white by both neighbours
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        for i, (_, (vx1, vx2)) in enumerate(sorted_cols):
            painter.drawRect(vx1, 0, min(vx2, vp.width()) - vx1, BOX_SIZE)

        # Pass 2: borders at both vx1 and vx2 of every column
        col_ends  = [vx2 for _, (vx1, vx2) in sorted_cols]
        all_col_x = sorted(set(col_starts + col_ends))
        painter.setPen(QPen(BLUE, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(all_col_x[0], 0,        all_col_x[-1], 0)
        painter.drawLine(all_col_x[0], BOX_SIZE, all_col_x[-1], BOX_SIZE)
        for vx in all_col_x:
            painter.drawLine(vx, 0, vx, BOX_SIZE)

        painter.setPen(BLUE)
        for i, (col_idx, (vx1, vx2)) in enumerate(sorted_cols):
            prev_vx2  = sorted_cols[i - 1][1][1] if i > 0 else vx1
            next_vx1  = sorted_cols[i + 1][1][0] if i + 1 < len(sorted_cols) else vx2
            ex_left   = max(max(vx1, prev_vx2), 0)
            ex_right  = min(min(vx2, next_vx1), vp.width())
            center_x  = (ex_left + ex_right) // 2
            number    = str(col_idx + 1)
            tw        = fm.horizontalAdvance(number)
            tx        = center_x - tw // 2
            ty        = (BOX_SIZE + fm.ascent()) // 2
            painter.drawText(tx, ty, number)

        painter.restore()

    def set_pixmap(self, pixmap, keep_zoom=False):
        """Show a new rendered frame.

        With keep_zoom=False (a fresh dataset/first frame) it fits the image to
        the window. With keep_zoom=True (stepping to the next frame) it preserves
        the current zoom level and center so you can scan the same spot across
        many frames without re-zooming each time.
        """
        if keep_zoom:
            old_transform = self.transform()
            center_in_scene = self.mapToScene(self.viewport().rect().center())
            self._item.setPixmap(pixmap)
            BOX_SIZE = 40
            s = old_transform.m11()
            pr = QRectF(pixmap.rect())
            margin = BOX_SIZE / s if s > 0 else 0
            self._scene.setSceneRect(QRectF(
                pr.left() - margin, pr.top() - margin,
                pr.width() + margin, pr.height() + margin,
            ))
            self.setTransform(old_transform)
            self.centerOn(center_in_scene)
        else:
            self._item.setPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self._fit_with_strips()

    def event(self, ev):
        """Handle trackpad pinch gestures as zoom; pass everything else through."""
        if ev.type() == QEvent.Gesture:
            pinch = ev.gesture(Qt.PinchGesture)
            if pinch and pinch.scaleFactor() != 1.0:
                self.scale(pinch.scaleFactor(), pinch.scaleFactor())
            return True
        return super().event(ev)

    def wheelEvent(self, ev):
        """Zoom in a little on scroll-up, out a little on scroll-down."""
        factor = 1.12 if ev.angleDelta().y() > 0 else 0.89
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        """Double-click resets zoom and re-fits the whole frame to the window."""
        self._scene.setSceneRect(QRectF(self._item.pixmap().rect()))
        self._fit_with_strips()

    def keyPressEvent(self, event):
        """Forward Left/Right arrows to the main window (frame nav); handle the rest normally."""
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            p = self.parent()
            if p:
                p.keyPressEvent(event)
        else:
            super().keyPressEvent(event)


# ── Setup dialog ──────────────────────────────────────────────────────────────

class SetupDialog(QDialog):
    """The opening folder picker.

    Asks for the source-images folder and confirms that Star Trail CleanR has
    been run on it (by checking for the `cleanr_workspace/masks` subfolder
    beside the images). Pre-fills the last-used folder from saved config. On
    "Open" it either accepts (handing back the image and masks folders) or warns
    that the masks folder is missing.
    """
    def __init__(self, parent=None, img_dir=None, mask_dir=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO MaskViewR — Setup")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setFixedWidth(560)
        self.setStyleSheet("background-color: #0a1e3f; border-radius: 12px;")

        self._img_dir  = Path(img_dir)  if img_dir  and Path(img_dir).is_dir()  else None
        self._mask_dir = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(10)

        title = QLabel("YOLO MaskViewR")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        sub = QLabel("Full-frame STC detection mask viewer")
        sub.setStyleSheet("font-size: 15px; color: #a8c0e0;")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)

        desc = QLabel(
            "Navigate frame by frame through any dataset.\n"
            "Each detected trail shows as a colored numbered outline.\n"
            "Frames with no detections are labelled clearly.\n"
            "Scroll or pinch to zoom · drag to pan."
        )
        desc.setStyleSheet("font-size: 15px; color: #c0d0e8; padding: 6px 0;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        lay.addWidget(desc)

        lay.addSpacing(6)

        lay.addWidget(self._section_label("Source images folder"))
        self._img_lbl, img_btn = self._make_row()
        self._img_lbl.setText(str(self._img_dir) if self._img_dir else "No folder selected")
        self._img_lbl.setStyleSheet(self._lbl_style(self._img_dir is not None))
        img_btn.clicked.connect(self._pick)
        row1 = QHBoxLayout()
        row1.addWidget(self._img_lbl, stretch=1)
        row1.addWidget(img_btn)
        lay.addLayout(row1)

        lay.addSpacing(12)

        self._open_btn = QPushButton("Open")
        self._open_btn.setFixedHeight(44)
        self._open_btn.setStyleSheet(
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 16px; "
            "font-weight: bold; border-radius: 8px; border: none; padding: 0 36px; }"
            "QPushButton:hover { background-color: #1580e0; }"
            "QPushButton:disabled { background-color: #1a3a5c; color: #4a6a8a; }"
        )
        self._open_btn.clicked.connect(self._try_accept)
        self._open_btn.setEnabled(self._img_dir is not None)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.adjustSize()

    def _section_label(self, text):
        """Build a small bold section header label (e.g. "Source images folder")."""
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #a8c0e0;")
        return lbl

    def _lbl_style(self, filled):
        """Return the path-label stylesheet: brighter text when a folder is chosen, dim when not."""
        color = "#e6e6e6" if filled else "#4a6a8a"
        return (
            f"font-size: 14px; color: {color}; background: #0d1e3a; "
            "border: 1px solid #1a3a5c; border-radius: 4px; padding: 4px 8px;"
        )

    def _make_row(self):
        """Build one path row: a label showing the folder plus a "Choose..." button. Returns both."""
        lbl = QLabel()
        lbl.setWordWrap(False)
        btn = QPushButton("Choose…")
        btn.setFixedHeight(32)
        btn.setFixedWidth(90)
        btn.setStyleSheet(
            "QPushButton { background-color: #1a3a5c; color: #a8c0e0; font-size: 14px; "
            "border: 1px solid #2a5a8c; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a5080; color: white; }"
        )
        return lbl, btn

    def _pick(self):
        """Open a native folder chooser for the source images and record the pick."""
        start = str(self._img_dir or Path.home())
        path  = QFileDialog.getExistingDirectory(self, "Select source images folder", start)
        if not path:
            return
        self._img_dir = Path(path)
        self._img_lbl.setText(str(self._img_dir))
        self._img_lbl.setStyleSheet(self._lbl_style(True))
        self._open_btn.setEnabled(True)

    def _try_accept(self):
        """Confirm the masks folder exists beside the images, then accept; else warn.

        Star Trail CleanR writes masks to `<images>/cleanr_workspace/masks`. If
        that folder is there, store it and close successfully. If not, show a
        "run STC first" message offering Try Again or Quit.
        """
        if not (self._img_dir and self._img_dir.is_dir()):
            return
        candidate = self._img_dir / "cleanr_workspace" / "masks"
        if candidate.is_dir():
            self._mask_dir = candidate
            self.accept()
            return
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Masks Not Found")
        msg.setText(
            "Can't find masks folder.\n\nMake sure you've run STC."
        )
        msg.setStyleSheet(
            "QMessageBox { background-color: #0a1e3f; color: white; font-size: 15px; }"
            "QLabel { color: white; font-size: 15px; }"
            "QPushButton { background-color: #1a3a5c; color: #a8c0e0; font-size: 14px; "
            "border: 1px solid #2a5a8c; border-radius: 4px; padding: 6px 18px; }"
            "QPushButton:hover { background-color: #1a5080; color: white; }"
        )
        try_btn = msg.addButton("Try Again", QMessageBox.AcceptRole)
        quit_btn = msg.addButton("Quit", QMessageBox.RejectRole)
        msg.exec()
        if msg.clickedButton() == quit_btn:
            QApplication.quit()

    def result_dirs(self):
        """Return the chosen (images folder, masks folder) pair after the dialog closes."""
        return self._img_dir, self._mask_dir


# ── Image rendering ───────────────────────────────────────────────────────────

def compute_tile_bboxes(img_h, img_w, tile_size=640, overlap=0.2):
    """Recreate the 640x640 overlapping tile grid SAHI uses to slice the frame.

    Mirrors the detector's slicing: 640px tiles stepping by 80% of that (20%
    overlap), with the last tile in each row/column pulled back to sit flush
    against the image edge. Used to draw the tile grid and to name a region by
    its tile (e.g. "B3"). Returns a list of
    (xmin, ymin, xmax, ymax, row_index, col_index) tuples in full-resolution
    image coordinates.
    """
    step = int(tile_size * (1 - overlap))
    bboxes = []
    y_min = 0
    row = 0
    while y_min < img_h:
        y_max = y_min + tile_size
        if y_max > img_h:
            ty2 = img_h
            ty1 = max(0, img_h - tile_size)
        else:
            ty1, ty2 = y_min, y_max
        x_min = 0
        col = 0
        while x_min < img_w:
            x_max = x_min + tile_size
            if x_max > img_w:
                tx2 = img_w
                tx1 = max(0, img_w - tile_size)
                bboxes.append((tx1, ty1, tx2, ty2, row, col))
                break
            bboxes.append((x_min, ty1, x_max, ty2, row, col))
            x_min += step
            col += 1
        if y_max > img_h:
            break
        y_min += step
        row += 1
    return bboxes


def load_image(path: Path) -> Optional[np.ndarray]:
    """Load any supported source image as an 8-bit BGR array, or None if unreadable.

    Handles the formats this tool opens: 16-bit images are scaled down to 8-bit,
    grayscale is expanded to color, and 4-channel (with alpha) is reduced to BGR,
    so the rest of the renderer always gets a plain 3-channel image.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def render_frame(img: np.ndarray, mask: Optional[np.ndarray],
                 raw_mask: Optional[np.ndarray] = None,
                 show_tiles: bool = False,
                 orig_h: int = 0, orig_w: int = 0,
                 selected_mask: Optional[np.ndarray] = None,
                 selected_color: tuple = (255, 255, 255),
                 poly_data: Optional[dict] = None) -> np.ndarray:
    """Paint all the overlays onto a copy of the frame and return the result.

    This is the heart of the display. In layered order it draws:
      1. the optional tile grid (gray boxes, when show_tiles is on);
      2. the raw SAHI/YOLO detections as thin yellow outlines (from raw_mask),
         underneath so they show through;
      3. the repaired trails as colored, numbered outlines — one per fitted
         polygon when poly_data is present (so split crossing arms read as
         separate trails), otherwise one per connected blob in `mask`;
      4. a big centered "No detection" label when nothing was found;
      5. a translucent highlight + thick outline on the currently-selected trail
         (selected_mask), tinted white for a final mask or yellow for a raw hit.

    Line widths and font sizes are scaled so outlines land at roughly 1-2px on
    screen regardless of how far the image was downscaled. orig_h/orig_w are the
    full-resolution dimensions, needed to place the tile grid and polygon points
    correctly on the (typically half-size) display image.
    """
    h, w = img.shape[:2]
    disp = img.copy()

    display_scale = min(MAX_W / w, MAX_H / h, 1.0)
    lw_grid  = max(1, round(2.0 / display_scale))
    lw_trail = max(1, round(1.0 / display_scale))
    lfs      = max(0.5, 0.45 / display_scale)

    # Tile grid (bottom layer) with row letters (left) and column numbers (top)
    if show_tiles:
        row_spans = {}
        col_spans = {}

        # tile bboxes are in full-res coords; scale to display image coords
        oh, ow = orig_h or h, orig_w or w
        tsx, tsy = w / ow, h / oh

        for (tx1, ty1, tx2, ty2, row, col) in compute_tile_bboxes(oh, ow):
            sx1 = int(tx1 * tsx); sy1 = int(ty1 * tsy)
            sx2 = int(tx2 * tsx); sy2 = int(ty2 * tsy)
            cv2.rectangle(disp, (sx1, sy1), (sx2 - 1, sy2 - 1), (80, 80, 80), lw_grid)
            if row not in row_spans:
                row_spans[row] = (sy1, sy2)
            if col not in col_spans:
                col_spans[col] = (sx1, sx2)


    # Yellow layer: full SAHI detection outlines (1px yellow contour per
    # raw prediction). Drawn BEFORE the red mask fill so it shows through.
    if raw_mask is not None and raw_mask.max() > 0:
        for det_idx in np.unique(raw_mask):
            if det_idx == 0:
                continue
            component = (raw_mask == det_idx).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, cnts, -1, (0, 255, 255), lw_trail)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = lfs
    thick = max(1, lw_trail)

    if poly_data and poly_data.get("polygons"):
        # Per-polygon path: one numbered outline per fitted polygon. Crossing-split
        # arms are separate polygons, so they show separately even though they
        # touch in the binary mask (which connected-components would merge into one).
        ow = orig_w or w
        oh = orig_h or h
        sx, sy = w / ow, h / oh
        for i, poly in enumerate(poly_data["polygons"]):
            corners = poly.get("corners", [])
            if len(corners) < 2:
                continue
            color = TRAIL_COLORS[i % len(TRAIL_COLORS)]
            pts = np.array([[int(round(x * sx)), int(round(y * sy))]
                            for x, y in corners], dtype=np.int32)
            cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, color, lw_trail * 2)

            bx, by = int(pts[:, 0].min()), int(pts[:, 1].min())
            bw = int(pts[:, 0].max() - bx)
            lbl = str(poly.get("id", i) + 1)   # 1-based, matches "mask #N"
            (tw, th), _ = cv2.getTextSize(lbl, font, fs * 1.1, thick)
            lx = max(0, bx + bw // 2 - tw // 2)
            ly = max(th + 2, by - 4)
            cv2.putText(disp, lbl, (lx + 1, ly + 1),
                        font, fs * 1.1, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(disp, lbl, (lx, ly),
                        font, fs * 1.1, color, thick, cv2.LINE_AA)
    elif mask is not None and mask.max() > 0:
        # Fallback: no per-polygon JSON, outline each connected component.
        n_labels, label_map = cv2.connectedComponents(
            (mask > 0).astype(np.uint8)
        )

        for label_id in range(1, n_labels):
            color = TRAIL_COLORS[(label_id - 1) % len(TRAIL_COLORS)]
            component = (label_map == label_id).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, cnts, -1, color, lw_trail * 2)

            bx, by, bw, _ = cv2.boundingRect(component)
            lbl = str(label_id)
            (tw, th), _ = cv2.getTextSize(lbl, font, fs * 1.1, thick)
            lx = max(0, bx + bw // 2 - tw // 2)
            ly = max(th + 2, by - 4)
            cv2.putText(disp, lbl, (lx + 1, ly + 1),
                        font, fs * 1.1, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(disp, lbl, (lx, ly),
                        font, fs * 1.1, color, thick, cv2.LINE_AA)
    elif raw_mask is None or raw_mask.max() == 0:
        label = "No detection"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = max(w / 1400, 0.7)
        thick = max(2, int(fs * 2.2))
        (tw, th), _ = cv2.getTextSize(label, font, fs * 1.2, thick)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.putText(disp, label, (cx + 2, cy + 2), font, fs * 1.2,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(disp, label, (cx, cy), font, fs * 1.2,
                    (80, 80, 255), thick, cv2.LINE_AA)

    if selected_mask is not None and selected_mask.max() > 0:
        sel = (selected_mask > 0)
        overlay = disp.copy()
        overlay[sel] = selected_color
        cv2.addWeighted(overlay, 0.25, disp, 0.75, 0, disp)
        sel_cnts, _ = cv2.findContours(
            sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(disp, sel_cnts, -1, selected_color, max(2, lw_trail * 4))

    return disp


def np_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert a rendered BGR image array into a Qt pixmap for display."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


# ── Main window ───────────────────────────────────────────────────────────────

class MaskViewR(QMainWindow):
    """The application window: nav bar, image canvas, and all the per-frame state.

    Owns the list of frames, the current index, the loaded image/mask/raw/polygon
    data for the frame on screen, and the click-to-select state. It wires the
    toolbar buttons and slider to navigation, loads each frame's masks from
    `cleanr_workspace/masks`, hands everything to `render_frame`, and handles
    clicks to select and report individual trails.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO MaskViewR")

        self.frames      = []
        self.mask_dir    = None
        self.img_dir     = None
        self.idx         = 0
        self._show_tiles = True

        self._view     = ZoomableImageView()

        # Per-frame state (populated in _show_current)
        self._current_fp              = None
        self._current_mask_path       = None
        self._current_orig_h          = 0
        self._current_orig_w          = 0
        self._current_img_half        = None
        self._current_mask_half       = None
        self._current_raw_mask_half   = None
        self._current_mask_full       = None
        self._current_mask_info       = ""
        self._current_poly_data       = None

        # Click-to-select state
        self._selected_mask_id        = None
        self._selected_component_mask = None
        self._selected_info           = {}
        self._selected_type           = None   # "mask" or "raw"
        self._current_raw_mask_full   = None

        self._view.clicked.connect(self._on_image_click)

        nav            = self._build_nav_bar()

        root = QWidget()
        root.setStyleSheet("background: #0d0d1a;")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(nav)
        lay.addWidget(self._view, stretch=1)
        self.setCentralWidget(root)
        self.resize(MAX_W, MAX_H + 90)

    def _build_nav_bar(self):
        """Build the two-row top bar: title/status/close banner, plus the button-and-slider toolbar.

        The toolbar holds Prev/Next, Copy Frame, the Tiles toggle, Add To WeirdR,
        a keyboard-hint line, and the frame scrubber slider with a type-a-frame
        box. Returns the assembled widget.
        """
        bar = QWidget()
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Row 1: banner (navy, matches TileFixR) ────────────────────────────
        banner = QWidget()
        banner.setFixedHeight(48)
        banner.setStyleSheet("background-color: #0a1e3f;")
        blay = QHBoxLayout(banner)
        blay.setContentsMargins(16, 0, 12, 0)
        blay.setSpacing(12)

        title = QLabel("YOLO MaskViewR")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent;")
        title.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard)
        blay.addWidget(title)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #a8c0e0; font-size: 13px; background: transparent;")
        self._status_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard)
        blay.addWidget(self._status_lbl, stretch=1)

        relaunch_btn = QPushButton("Relaunch")
        relaunch_btn.setFixedHeight(26)
        relaunch_btn.setStyleSheet(
            "QPushButton { background-color: #d0e4f5; color: #1a3a5c; font-size: 12px; "
            "font-weight: bold; border-radius: 15px; border: 1px solid #a0c4e0; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #b8d4ec; }"
        )
        relaunch_btn.clicked.connect(self._relaunch)
        blay.addWidget(relaunch_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #d93025; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #b8271b; }"
        )
        close_btn.clicked.connect(self.close)
        blay.addWidget(close_btn)

        outer.addWidget(banner)

        # ── Row 2: toolbar (gray, matches TileFixR) ───────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(42)
        toolbar.setStyleSheet("background: #2d3138;")
        tlay = QHBoxLayout(toolbar)
        tlay.setContentsMargins(12, 4, 12, 4)
        tlay.setSpacing(0)

        primary = (
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 12px; "
            "font-weight: bold; border-radius: 4px; border: none; padding: 4px 14px; }"
            "QPushButton:hover { background-color: #1580e0; }"
            "QPushButton:disabled { background-color: #3a4050; color: #666; }"
        )
        tool_btn = (
            "QPushButton { background-color: #404550; color: #e0e8f0; font-size: 12px; "
            "font-weight: bold; border-radius: 4px; border: 1px solid #606570; "
            "padding: 4px 12px; }"
            "QPushButton:hover { background-color: #505660; color: white; }"
        )

        # Left half — navigation + action buttons
        left = QHBoxLayout()
        left.setSpacing(8)

        self._prev_btn = QPushButton("◀  Prev")
        self._prev_btn.setStyleSheet(primary)
        self._prev_btn.clicked.connect(self.go_prev)
        left.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setStyleSheet(primary)
        self._next_btn.clicked.connect(self.go_next)
        left.addWidget(self._next_btn)

        self._copy_btn = QPushButton("Copy Frame")
        self._copy_btn.setStyleSheet(tool_btn)
        self._copy_btn.clicked.connect(self._copy_frame)
        left.addWidget(self._copy_btn)

        self._tiles_btn = QPushButton("Tiles: On")
        self._tiles_btn.setStyleSheet(tool_btn)
        self._tiles_btn.clicked.connect(self._toggle_tiles)
        left.addWidget(self._tiles_btn)

        self._weirdr_btn = QPushButton("Add To WeirdR")
        self._weirdr_btn.setStyleSheet(tool_btn)
        self._weirdr_btn.clicked.connect(self._add_to_weirdr)
        left.addWidget(self._weirdr_btn)

        hint = QLabel("← →  nav   scroll/pinch = zoom   dbl-click = fit   O = folders   Q = quit")
        hint.setStyleSheet("color: #666; font-size: 11px; padding-left: 10px;")
        left.addWidget(hint)

        left.addStretch()
        tlay.addLayout(left, stretch=1)

        # Right half — frame scrubber (at most half the toolbar width)
        right = QHBoxLayout()
        right.setSpacing(8)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(1)
        self._slider.setMaximum(1)
        self._slider.setValue(1)
        self._slider.valueChanged.connect(self._on_slider_changed)
        right.addWidget(self._slider, stretch=1)

        self._slider_lbl = QLabel("1 of 0")
        self._slider_lbl.setStyleSheet("color: #aaaaaa; font-size: 12px; font-weight: bold;")
        self._slider_lbl.setFixedWidth(72)
        right.addWidget(self._slider_lbl)

        self._frame_input = QLineEdit()
        self._frame_input.setFixedWidth(52)
        self._frame_input.setFixedHeight(26)
        self._frame_input.setText("1")
        self._frame_input.setAlignment(Qt.AlignCenter)
        self._frame_input.setStyleSheet(
            "QLineEdit { background: #404550; color: #e0e8f0; font-size: 13px; "
            "border: 1px solid #606570; border-radius: 4px; }"
        )
        self._frame_input.returnPressed.connect(self._on_frame_input_entered)
        self._frame_input.installEventFilter(self)
        right.addWidget(self._frame_input)

        tlay.addLayout(right, stretch=1)

        outer.addWidget(toolbar)

        return bar

    # ── Setup flow ────────────────────────────────────────────────────────────

    def run_setup(self):
        """Show the folder picker and load the chosen dataset (or close if cancelled at startup).

        Also reachable mid-session via the "O" key to switch datasets.
        """
        cfg = load_config()
        dlg = SetupDialog(
            self,
            img_dir=cfg.get("image_dir"),
        )
        if dlg.exec() != QDialog.Accepted:
            if not self.frames:
                self.close()
            return
        img_dir, mask_dir = dlg.result_dirs()
        self._load_dataset(img_dir, mask_dir, cfg.get("frame_index", 0))

    def _load_dataset(self, img_dir: Path, mask_dir, start_idx: int):
        """Take a folder pair, build the sorted frame list, jump to start_idx, and show it.

        Records the chosen folders and frame in config so the next launch reopens
        where you left off. Does nothing useful (just a status note) if the folder
        holds no supported images.
        """
        frames = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not frames:
            self._status_lbl.setText(f"No images found in {img_dir.name}")
            return
        self.frames   = frames
        self.img_dir  = img_dir
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.idx      = min(start_idx, len(frames) - 1)
        self.setWindowTitle(f"YOLO MaskViewR — {img_dir.name}")
        self._slider.setMaximum(len(frames))
        cfg = load_config()
        cfg.update(image_dir=str(img_dir),
                   mask_dir=str(mask_dir) if mask_dir else "",
                   frame_index=self.idx)
        save_config(cfg)
        self._show_current()

    # ── Display ───────────────────────────────────────────────────────────────

    def _show_current(self, keep_zoom=False):
        """Load the current frame and all its mask data, render it, and update the bar.

        For the frame at self.idx it loads the source image and, from the masks
        folder, the final mask PNG, the raw YOLO/SAHI mask, and the per-polygon
        JSON (any of which may be absent). It builds a half-resolution image for
        fast display, stashes everything in the `_current_*` fields for click
        handling, counts the detected trails for the status line, then re-renders
        and syncs the slider, frame box, and Prev/Next buttons. keep_zoom carries
        the current zoom across a frame step.
        """
        if not self.frames:
            return
        fp = self.frames[self.idx]
        img = load_image(fp)
        if img is None:
            self._status_lbl.setText(f"Could not load {fp.name}")
            return

        orig_h, orig_w = img.shape[:2]
        half_w, half_h = orig_w // 2, orig_h // 2
        img_half = cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_AREA)

        mask_full, mask_half, raw_mask_half, mask_info = None, None, None, ""
        raw_mask_full = None
        mask_path = None
        poly_data = None
        if self.mask_dir is None:
            mask_info = "no masks folder set"
        else:
            mp     = self.mask_dir / (fp.stem + ".png")
            raw_mp = self.mask_dir / (fp.stem + "_raw.png")
            if mp.exists():
                mask_path = mp
                mask_full = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mask_full is None or mask_full.max() == 0:
                    mask_full = None
                    mask_info = "no detection"
                else:
                    n_cc, _ = cv2.connectedComponents((mask_full > 0).astype(np.uint8))
                    n_trails = n_cc - 1
                    mask_info = f"{n_trails} trail{'s' if n_trails != 1 else ''} detected"
                    mask_half = cv2.resize(mask_full, (half_w, half_h), interpolation=cv2.INTER_NEAREST)
                if raw_mp.exists():
                    raw_mask_full_loaded = cv2.imread(str(raw_mp), cv2.IMREAD_GRAYSCALE)
                    if raw_mask_full_loaded is not None and raw_mask_full_loaded.max() > 0:
                        n_raw = int(np.max(raw_mask_full_loaded))
                        mask_info += f"  ({n_raw} raw YOLO hit{'s' if n_raw != 1 else ''})"
                        raw_mask_full = raw_mask_full_loaded
                        raw_mask_half = cv2.resize(raw_mask_full_loaded, (half_w, half_h),
                                                   interpolation=cv2.INTER_NEAREST)
                pjson = self.mask_dir / (fp.stem + "_polys.json")
                if pjson.exists():
                    try:
                        poly_data = json.loads(pjson.read_text())
                    except (ValueError, OSError):
                        poly_data = None
            else:
                mask_info = "mask not yet generated"

        self._current_fp             = fp
        self._current_mask_path      = mask_path
        self._current_orig_h         = orig_h
        self._current_orig_w         = orig_w
        self._current_img_half       = img_half
        self._current_mask_half      = mask_half
        self._current_raw_mask_half  = raw_mask_half
        self._current_mask_full      = mask_full
        self._current_raw_mask_full  = raw_mask_full
        self._current_mask_info      = mask_info
        self._current_poly_data      = poly_data
        self._selected_mask_id       = None
        self._selected_component_mask = None
        self._selected_info          = {}
        self._selected_type          = None

        self._rerender(keep_zoom=keep_zoom)

        n = len(self.frames)
        self._status_lbl.setText(f"{fp.name}   {self.idx + 1} / {n}   —   {mask_info}")
        self._prev_btn.setEnabled(self.idx > 0)
        self._next_btn.setEnabled(self.idx < n - 1)
        self._weirdr_btn.setText("Add To WeirdR")
        self._weirdr_btn.setEnabled(True)

        self._slider.blockSignals(True)
        self._slider.setValue(self.idx + 1)
        self._slider.blockSignals(False)
        self._slider_lbl.setText(f"{self.idx + 1} of {n}")
        self._frame_input.setText(str(self.idx + 1))

        cfg = load_config()
        cfg["frame_index"] = self.idx
        save_config(cfg)

    def _rerender(self, keep_zoom=False):
        """Repaint the on-screen image from the already-loaded frame state.

        Cheaper than _show_current: it reuses the cached half-size image and
        masks (no disk reload) and just re-runs render_frame. Called whenever
        only the overlay changes — toggling tiles, selecting/clearing a trail —
        and also pushes the current tile boxes to the view for its edge labels.
        """
        if self._current_img_half is None:
            return
        orig_h  = self._current_orig_h
        orig_w  = self._current_orig_w
        half_w  = orig_w // 2
        half_h  = orig_h // 2

        sel_half  = None
        sel_color = (255, 255, 255)  # white for colored mask, yellow for raw
        if self._selected_component_mask is not None:
            sel_half = cv2.resize(
                self._selected_component_mask.astype(np.uint8),
                (half_w, half_h), interpolation=cv2.INTER_NEAREST,
            )
            if self._selected_type == "raw":
                sel_color = (0, 255, 255)

        disp = render_frame(
            self._current_img_half,
            self._current_mask_half,
            self._current_raw_mask_half,
            show_tiles=self._show_tiles,
            orig_h=orig_h,
            orig_w=orig_w,
            selected_mask=sel_half,
            selected_color=sel_color,
            poly_data=self._current_poly_data,
        )
        self._view.set_pixmap(np_to_pixmap(disp), keep_zoom=keep_zoom)

        if self._show_tiles:
            bboxes = [(int(tx1 * half_w / orig_w), int(ty1 * half_h / orig_h),
                       int(tx2 * half_w / orig_w), int(ty2 * half_h / orig_h), row, col)
                      for tx1, ty1, tx2, ty2, row, col in compute_tile_bboxes(orig_h, orig_w)]
            self._view.set_tile_bboxes(bboxes)
        else:
            self._view.set_tile_bboxes([])

    def _on_image_click(self, scene_pos):
        """Select the trail under a click and report its details in the status line.

        Maps the click from the half-size display back to full-resolution pixels,
        then checks first the final mask (selects the connected blob hit) and, if
        that misses, the raw YOLO/SAHI mask (selects the raw detection hit). For
        the selection it records the tile, centroid, bounding box, and pixel area
        (used by Copy Frame), highlights it, and writes a one-line summary.
        Clicking empty background clears any current selection.
        """
        if self._current_mask_full is None and self._current_raw_mask_full is None:
            return
        orig_h = self._current_orig_h
        orig_w = self._current_orig_w
        half_w = orig_w // 2
        half_h = orig_h // 2

        fx = int(round(scene_pos.x() * orig_w / half_w))
        fy = int(round(scene_pos.y() * orig_h / half_h))
        fx = max(0, min(orig_w - 1, fx))
        fy = max(0, min(orig_h - 1, fy))

        # Try colored mask first
        clicked_id = 0
        if self._current_mask_full is not None:
            _, label_map, stats, centroids = cv2.connectedComponentsWithStats(
                (self._current_mask_full > 0).astype(np.uint8)
            )
            clicked_id = int(label_map[fy, fx])

        if clicked_id > 0:
            s    = stats[clicked_id]
            x1   = int(s[cv2.CC_STAT_LEFT])
            y1   = int(s[cv2.CC_STAT_TOP])
            x2   = x1 + int(s[cv2.CC_STAT_WIDTH])
            y2   = y1 + int(s[cv2.CC_STAT_HEIGHT])
            area = int(s[cv2.CC_STAT_AREA])
            cx   = int(round(centroids[clicked_id][0]))
            cy   = int(round(centroids[clicked_id][1]))
            tile_label = self._tile_for(cx, cy, orig_h, orig_w)
            self._selected_mask_id        = clicked_id
            self._selected_type           = "mask"
            self._selected_component_mask = (label_map == clicked_id).astype(np.uint8)
            self._selected_info = {
                "type": "mask", "mask_id": clicked_id,
                "cx": cx, "cy": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": area, "tile": tile_label,
            }
            self._rerender(keep_zoom=True)
            n, fp = len(self.frames), self._current_fp
            self._status_lbl.setText(
                f"{fp.name}   {self.idx + 1} / {n}   —   {self._current_mask_info}"
                f"   —   mask #{clicked_id} selected (tile {tile_label})"
            )
            return

        # Fall through: try raw SAHI detection
        if self._current_raw_mask_full is not None:
            raw_id = int(self._current_raw_mask_full[fy, fx])
            if raw_id > 0:
                ys, xs = np.where(self._current_raw_mask_full == raw_id)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                cx, cy  = int(round(xs.mean())), int(round(ys.mean()))
                area    = len(xs)
                tile_label = self._tile_for(cx, cy, orig_h, orig_w)
                self._selected_mask_id        = raw_id
                self._selected_type           = "raw"
                self._selected_component_mask = (self._current_raw_mask_full == raw_id).astype(np.uint8)
                self._selected_info = {
                    "type": "raw", "raw_id": raw_id,
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area": area, "tile": tile_label,
                }
                self._rerender(keep_zoom=True)
                n, fp = len(self.frames), self._current_fp
                self._status_lbl.setText(
                    f"{fp.name}   {self.idx + 1} / {n}   —   {self._current_mask_info}"
                    f"   —   raw detection #{raw_id} selected (tile {tile_label})"
                )
                return

        # Background click — clear selection
        self._selected_mask_id        = None
        self._selected_component_mask = None
        self._selected_info           = {}
        self._selected_type           = None
        self._rerender(keep_zoom=True)
        n, fp = len(self.frames), self._current_fp
        self._status_lbl.setText(
            f"{fp.name}   {self.idx + 1} / {n}   —   {self._current_mask_info}"
        )

    def _tile_for(self, cx, cy, orig_h, orig_w):
        """Return the tile name (e.g. "B3") containing point (cx, cy), or "?" if none."""
        for (tx1, ty1, tx2, ty2, row, col) in compute_tile_bboxes(orig_h, orig_w):
            if tx1 <= cx < tx2 and ty1 <= cy < ty2:
                return chr(ord('A') + row) + str(col + 1)
        return "?"

    # ── Navigation ────────────────────────────────────────────────────────────

    def go_prev(self):
        """Step to the previous frame (keeping zoom), if not already at the first."""
        if self.idx > 0:
            self.idx -= 1
            self._show_current(keep_zoom=True)

    def go_next(self):
        """Step to the next frame (keeping zoom), if not already at the last."""
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self._show_current(keep_zoom=True)

    def eventFilter(self, obj, event):
        """Make Left/Right arrows navigate frames even while the type-a-frame box has focus."""
        if obj is self._frame_input and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key.Key_Left:
                self.go_prev()
                return True
            if event.key() == Qt.Key.Key_Right:
                self.go_next()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Window-level keys: Right/Left navigate, O reopens folders, Q/Escape quits."""
        key = event.key()
        if key == Qt.Key.Key_Right:
            self.go_next()
        elif key == Qt.Key.Key_Left:
            self.go_prev()
        elif key == Qt.Key.Key_O:
            self.run_setup()
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    # ── Copy Frame ────────────────────────────────────────────────────────────

    def _copy_frame(self):
        """Copy a one-line reference for the current frame (or selected trail) to the clipboard.

        With a trail selected it builds the full reference — frame number and
        name, dataset, the selected mask/raw id, tile, centroid, bounding box,
        pixel area, the image path, the mask path, and when the mask file was
        written (so you can tell whether it predates the latest pipeline run).
        With nothing selected it copies a short frame-only line. Flashes the
        button to "Copied!" briefly.
        """
        if not self.frames:
            return
        fp      = self.frames[self.idx]
        dataset = self.img_dir.name if self.img_dir else "unknown"

        if self._selected_info:
            info = self._selected_info
            mask_path_str = str(self._current_mask_path) if self._current_mask_path else "no mask"
            if info.get("type") == "raw":
                id_field = f"raw detection #{info['raw_id']}"
            else:
                id_field = f"mask #{info['mask_id']}"
            # Stamp the mask file's modification time so you can tell at a glance
            # whether the mask predates or postdates the most recent pipeline run.
            mask_ts = ""
            if self._current_mask_path and self._current_mask_path.exists():
                mtime = os.path.getmtime(self._current_mask_path)
                import time as _time
                mask_ts = f" | mask written {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(mtime))}"
            text = (
                f"YOLO MaskViewR"
                f" | frame {self.idx + 1} ({fp.stem})"
                f" | dataset {dataset}"
                f" | {id_field}"
                f" | tile {info['tile']}"
                f" | cx={info['cx']} cy={info['cy']}"
                f" | bbox x1={info['x1']} y1={info['y1']} x2={info['x2']} y2={info['y2']}"
                f" | area={info['area']}px"
                f" | image {fp}"
                f" | mask {mask_path_str}"
                f"{mask_ts}"
            )
        else:
            text = f"YOLO MaskViewR | frame {self.idx + 1} ({fp.stem}) | {dataset}"

        QApplication.clipboard().setText(text)
        self._copy_btn.setText("Copied!")
        QTimer.singleShot(1200, lambda: self._copy_btn.setText("Copy Frame"))

    # ── Tile grid toggle ───────────────────────────────────────────────────────

    def _toggle_tiles(self):
        """Turn the tile-grid overlay on or off, update the button label, and repaint."""
        self._show_tiles = not self._show_tiles
        self._tiles_btn.setText("Tiles: On" if self._show_tiles else "Tiles: Off")
        self._rerender(keep_zoom=True)

    # ── Scrubber ──────────────────────────────────────────────────────────────

    def _on_slider_changed(self, value):
        """Jump to the frame the scrubber slider was dragged to (keeping zoom)."""
        if not self.frames:
            return
        new_idx = value - 1
        if new_idx != self.idx:
            self.idx = new_idx
            self._show_current(keep_zoom=True)

    def _on_frame_input_entered(self):
        """Jump to the frame number typed into the box, clamped to the valid range."""
        if not self.frames:
            return
        try:
            val = int(self._frame_input.text())
            val = max(1, min(len(self.frames), val))
        except ValueError:
            val = self.idx + 1
        self.idx = val - 1
        self._show_current(keep_zoom=True)

    # ── Relaunch ──────────────────────────────────────────────────────────────

    def _relaunch(self):
        """Save the current frame and restart the app as a fresh process (useful after edits)."""
        cfg = load_config()
        cfg["frame_index"] = self.idx
        save_config(cfg)
        subprocess.Popen([sys.executable, os.path.abspath(__file__)])
        self.close()
        QApplication.quit()

    # ── WeirdR ────────────────────────────────────────────────────────────────

    def _add_to_weirdr(self):
        """Append the current frame to the shared `weirdr_list.json` of odd frames to revisit.

        Writes the frame's tag, filename, and dataset (skipping duplicates). If
        it was already listed, says so; otherwise plays a short "Adding..."
        animation on the button.
        """
        if not self.frames:
            return
        fp  = self.frames[self.idx]
        tag = fp.stem
        try:
            weirdr = json.loads(WEIRDR_PATH.read_text()) if WEIRDR_PATH.exists() else []
        except Exception:
            weirdr = []
        already = any(e.get("tag") == tag for e in weirdr)
        if not already:
            weirdr.append({
                "source":  "mask_viewr",
                "tag":     tag,
                "filename": fp.name,
                "dataset": self.img_dir.name if self.img_dir else "",
                "reason":  "",
                "added":   time.strftime("%Y-%m-%d"),
            })
            WEIRDR_PATH.write_text(json.dumps(weirdr, indent=2))

        if already:
            self._weirdr_btn.setText("Already listed")
            QTimer.singleShot(1500, lambda: self._weirdr_btn.setText("Add To WeirdR"))
        else:
            self._weirdr_btn.setEnabled(False)
            self._weirdr_anim_step = 0
            self._weirdr_anim_timer = QTimer(self)
            self._weirdr_anim_timer.setInterval(200)
            self._weirdr_anim_timer.timeout.connect(self._tick_weirdr_anim)
            self._weirdr_anim_timer.start()
            self._tick_weirdr_anim()

    def _tick_weirdr_anim(self):
        """Advance the "Adding." -> "Added!" button animation one step per timer tick."""
        labels = ["Adding.", "Adding..", "Adding...", "Added!"]
        step = self._weirdr_anim_step
        self._weirdr_btn.setText(labels[min(step, len(labels) - 1)])
        self._weirdr_anim_step += 1
        if step >= len(labels) - 1:
            self._weirdr_anim_timer.stop()
            QTimer.singleShot(800, lambda: self._weirdr_btn.setEnabled(True))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Start the Qt app, open the window, and trigger the setup dialog shortly after launch."""
    app = QApplication(sys.argv)
    win = MaskViewR()
    win.show()
    QTimer.singleShot(50, win.run_setup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
