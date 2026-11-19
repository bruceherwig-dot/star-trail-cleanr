"""
mask_painter.py — Foreground mask painting widget for Star Trail CleanR.

WHAT THIS FILE IS
-----------------
This is the on-screen tool where the user hand-paints a "foreground mask" before
trail cleaning runs. The user looks at one of their photos and roughly brushes a
green overlay over everything that is NOT sky — the ground, rocks, buildings, the
horizon line. That green area tells the rest of the app: "don't bother looking
for airplane or satellite trails down here, and never try to 'repair' this part."

WHY IT EXISTS
-------------
The trail-detection AI only makes sense in the sky. If it ran over the foreground
it could mistake a roofline or a tree branch for a trail and damage the landscape.
A single hand-painted mask is reused for every frame in the sequence because all
the user's shots are on a fixed tripod, so the foreground sits in exactly the same
pixels in every photo.

HOW IT FITS INTO THE APP
------------------------
This widget is a screen inside the main PySide6 GUI (`star_trail_cleanr.py`). The
GUI shows it after the user picks a folder of photos. When the user clicks
"Save Mask", this widget hands back a black-and-white image (255 = foreground to
skip, 0 = sky to process) via the `mask_done` signal; the GUI then feeds that mask
into the detection/repair pipeline.

WHAT'S IN HERE (two classes + one helper)
-----------------------------------------
- `numpy_to_qimage`     — converts an OpenCV image (numpy array) into a Qt image.
- `MaskGraphicsView`    — the zoomable/pannable canvas that handles all the actual
                          painting, erasing, brush sizing, undo/redo, and the mask
                          data itself.
- `MaskPainterWidget`   — the whole screen: instruction banner, toolbar (brush
                          mode, sliders, undo, help, Save), the canvas above, and a
                          status bar of shortcuts.

CONTROLS (also shown to the user in the status bar / Help dialog)
-----------------------------------------------------------------
Left-click = paint, Right-click = toggle paint/erase, Scroll = brush size,
Shift+click = straight line, Space+drag = pan, Ctrl/Cmd+scroll or pinch = zoom.

IMPORTANT: the mask is stored as a numpy uint8 array where 255 means "foreground
to skip" and 0 means "sky to process". The green overlay the user sees is just a
visualization of that array.
"""

import math
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QMessageBox, QSizePolicy, QPinchGesture,
)
from PySide6.QtCore import Qt, Signal, QSettings, QRectF, QPointF, QEvent
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QCursor, QPen, QBrush,
    QFont, QKeySequence, QShortcut,
)


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert an OpenCV-style image array into a Qt image for on-screen display.

    OpenCV loads color photos in BGR channel order (blue, green, red), but Qt
    expects RGB, so a 3-channel array is reversed channel-wise before wrapping.
    Single-channel arrays (e.g. a grayscale mask) are wrapped as 8-bit grayscale.

    Input:
        arr  numpy uint8 image. If 3-D it is treated as BGR color; if 2-D it is
             treated as grayscale.
    Returns:
        A QImage referencing the pixel data. NOTE: for the grayscale path the
        QImage shares the array's memory (no copy), so the caller must keep the
        array alive while the QImage is in use; the color path copies (`.copy()`).
        `3 * w` and `w` are the bytes-per-row (stride) Qt needs.
    """
    h, w = arr.shape[:2]
    if arr.ndim == 3:
        rgb = arr[:, :, ::-1].copy()  # BGR → RGB (reverse the channel axis)
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QImage(arr.data, w, h, w, QImage.Format_Grayscale8)


class MaskGraphicsView(QGraphicsView):
    """The interactive canvas where the user actually paints the mask.

    This is a Qt graphics view stacked into three layers:
      Z=0  the photo (the frame the user is tracing against)
      Z=1  the green mask overlay (a translucent picture of the mask array)
      Z=10 the brush-circle cursor that follows the pointer
    It owns the mask itself as a numpy array (`_mask_np`) and rebuilds the green
    overlay whenever the mask changes. It also handles all direct interaction:
    left-drag to paint, right-click to flip paint/erase, scroll to size the brush,
    space+drag to pan, Ctrl/Cmd+scroll or pinch to zoom, and undo/redo.

    The parent `MaskPainterWidget` wraps this in a toolbar and reads the finished
    mask back out via `get_mask()`.
    """

    # Qt signals this view emits so the surrounding toolbar can stay in sync.
    brush_changed = Signal(int)   # emitted when brush radius changes (new radius px)
    mode_changed = Signal(bool)   # emitted when paint/erase toggles (True = erase)

    def __init__(self, scene, parent=None):
        """Set up the view's look, interaction state, and the mask buffers.

        `scene` is the QGraphicsScene this view draws into (created by the parent
        widget). All the `_` attributes here are just the starting state: no image
        loaded yet, brush at 150px, paint mode (not erase), empty undo/redo, and a
        default 50% overlay opacity. The actual mask array is not allocated until
        an image is loaded (see `load_image`)."""
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor("#808080")))
        self.setMouseTracking(True)

        self._painting = False
        self._erase_mode = False  # toggle: right-click on/off
        self._panning = False
        self._space_held = False
        self._last_pan_pos = None
        self._last_paint_pos = None  # for interpolation
        self._last_click_pos = None  # for Shift+click straight line
        self._last_right_click_time = 0  # guard against trackpad scroll on right-click
        self._brush_radius = 150
        self._min_brush = 5
        self._max_brush = 500

        # Mask data: numpy uint8, same size as image. 255 = foreground (skip).
        self._mask_np = None
        self._mask_overlay_item = None
        self._overlay_opacity = 0.5
        self._brightness = 1.0

        # Photo items
        self._photo_item = None
        self._img_w = 0
        self._img_h = 0

        # Undo
        self._undo_stack = []
        self._redo_stack = []
        self._max_undo = 50

        # Brush cursor indicator (scene item)
        self._cursor_circle = None

        # Pinch gesture for trackpad zoom
        self.grabGesture(Qt.PinchGesture)

        # Floating zoom overlay (built in _build_zoom_overlay)
        self._zoom_overlay = None
        self._zoom_label = None
        self._build_zoom_overlay()

    # ── Zoom overlay ─────────────────────────────────────────────────────────

    def _build_zoom_overlay(self):
        """Build the small floating zoom control that sits in the corner of the
        canvas: a minus button, a percentage readout, a plus button, and a "Fit"
        button. It is a child widget drawn on top of the view (not part of the
        scrollable scene), so it stays pinned in place while the photo pans and
        zooms underneath it. Called once from __init__."""
        self._zoom_overlay = QWidget(self)
        self._zoom_overlay.setStyleSheet(
            "QWidget { background-color: rgba(30,30,30,210); border-radius: 14px; }"
            "QPushButton { color: #ddd; background: transparent; border: none; "
            "font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { color: white; }"
            "QLabel { color: #ddd; font-size: 12px; background: transparent; }"
        )
        hl = QHBoxLayout(self._zoom_overlay)
        hl.setContentsMargins(10, 3, 10, 3)
        hl.setSpacing(4)

        out_btn = QPushButton("\u2212")
        out_btn.setFixedSize(24, 24)
        out_btn.setCursor(Qt.PointingHandCursor)
        out_btn.clicked.connect(lambda: self._zoom_by(1 / 1.25))
        hl.addWidget(out_btn)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(46)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        hl.addWidget(self._zoom_label)

        in_btn = QPushButton("+")
        in_btn.setFixedSize(24, 24)
        in_btn.setCursor(Qt.PointingHandCursor)
        in_btn.clicked.connect(lambda: self._zoom_by(1.25))
        hl.addWidget(in_btn)

        fit_btn = QPushButton("Fit")
        fit_btn.setFixedHeight(24)
        fit_btn.setMinimumWidth(34)
        fit_btn.setStyleSheet(
            "QPushButton { color: #ddd; background: transparent; border: none; "
            "font-size: 12px; font-weight: bold; padding: 0 6px; }"
            "QPushButton:hover { color: white; }"
        )
        fit_btn.setCursor(Qt.PointingHandCursor)
        fit_btn.clicked.connect(self._zoom_to_fit)
        hl.addWidget(fit_btn)

        self._zoom_overlay.adjustSize()
        self._zoom_overlay.raise_()

    def _position_zoom_overlay(self):
        """Re-pin the floating zoom control to the bottom-right corner of the
        viewport. Called on resize so it tracks the corner as the window changes
        size."""
        if self._zoom_overlay is None:
            return
        margin = 14
        vp = self.viewport()
        x = vp.width() - self._zoom_overlay.width() - margin
        y = vp.height() - self._zoom_overlay.height() - margin
        self._zoom_overlay.move(x, y)
        self._zoom_overlay.raise_()

    def _zoom_by(self, factor):
        """Zoom in or out by a multiplier (>1 zooms in, <1 zooms out), keeping the
        center of the view fixed. The +/- buttons, keyboard +/-, and Ctrl/Cmd+
        scroll all route through here. The anchor is temporarily forced to the
        view center (so button zooms feel centered) and then restored to whatever
        it was (normally anchored under the mouse)."""
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(factor, factor)
        self.setTransformationAnchor(old_anchor)
        self._update_zoom_label()

    def _zoom_to_fit(self):
        """Scale the view so the whole photo fits in the window. Backs the "Fit"
        button and the 0 key."""
        if self._photo_item:
            self.fitInView(self._photo_item, Qt.KeepAspectRatio)
            self._update_zoom_label()

    def _update_zoom_label(self):
        """Refresh the percentage readout in the floating zoom control. `m11()` is
        the horizontal scale factor of the current view transform, so 1.0 = 100%."""
        if self._zoom_label is None:
            return
        pct = int(round(self.transform().m11() * 100))
        self._zoom_label.setText(f"{pct}%")

    # ── Public API ───────────────────────────────────────────────────────────

    def load_image(self, img_np: np.ndarray):
        """Show a photo as the background and start a fresh, empty mask for it.

        `img_np` is a BGR color image (OpenCV order). This records the image size,
        keeps an untouched copy for re-rendering at different brightness, allocates
        a new all-zero mask the same size, and zooms to fit. Use this when opening
        a new photo; use `set_background_image` instead to swap the photo while
        keeping an existing mask. The `pad = 500` enlarges the scrollable scene
        beyond the photo edges so the user can pan/zoom past the borders."""
        self._img_h, self._img_w = img_np.shape[:2]
        self._original_img = img_np.copy()
        self._update_photo_display()

        # Initialize blank mask (all 0 = nothing masked yet)
        self._mask_np = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        self._refresh_overlay()
        pad = 500
        self.setSceneRect(self._photo_item.boundingRect().adjusted(-pad, -pad, pad, pad))
        self.fitInView(self._photo_item, Qt.KeepAspectRatio)
        self._update_zoom_label()

    def set_background_image(self, img_np: np.ndarray):
        """Swap the background photo only — keep the painted mask and the
        current zoom/pan. Used when stepping through frames to find a clearer
        sky to trace against. Star-trail frames share dimensions, so the mask
        carries over unchanged; if a frame somehow differs in size, the mask is
        resized to match so the painting still applies."""
        new_h, new_w = img_np.shape[:2]
        self._original_img = img_np.copy()
        if (new_h, new_w) != (self._img_h, self._img_w):
            if self._mask_np is not None:
                import cv2
                self._mask_np = cv2.resize(self._mask_np, (new_w, new_h),
                                           interpolation=cv2.INTER_NEAREST)
            self._img_h, self._img_w = new_h, new_w
            self._update_photo_display()
            self._refresh_overlay()
            pad = 500
            self.setSceneRect(
                self._photo_item.boundingRect().adjusted(-pad, -pad, pad, pad))
        else:
            self._update_photo_display()

    def load_mask(self, mask_np: np.ndarray):
        """Load a previously painted mask so the user can keep editing it.

        `mask_np` is a grayscale array where 255 = foreground (skip) and 0 = sky.
        If it does not match the current photo's size it is nearest-neighbor
        resized to fit (nearest-neighbor keeps the mask strictly black-or-white —
        no gray edges that would smear the foreground boundary). Used when
        reopening the mask editor on a folder that already has a saved mask."""
        if mask_np.shape[:2] != (self._img_h, self._img_w):
            import cv2
            mask_np = cv2.resize(mask_np, (self._img_w, self._img_h),
                                 interpolation=cv2.INTER_NEAREST)
        self._mask_np = mask_np.copy()
        self._refresh_overlay()

    def get_mask(self) -> np.ndarray:
        """Hand back a copy of the finished mask (uint8: 255 = foreground/skip,
        0 = sky/process). A copy is returned so the caller can't accidentally edit
        the live mask. This is what the pipeline ultimately consumes."""
        return self._mask_np.copy()

    def has_mask(self) -> bool:
        """True if the user has painted anything at all (any non-zero pixel). Used
        to warn the user before saving an empty mask."""
        return self._mask_np is not None and self._mask_np.any()

    def clear_mask(self):
        """Erase the entire mask back to all-sky. Snapshots first so a stray "Clear
        All" can be undone."""
        self._push_undo()
        self._mask_np[:] = 0
        self._refresh_overlay()

    def set_overlay_opacity(self, value: float):
        """Set how see-through the green overlay is. Clamped to 0.1–0.9 so the
        overlay is never fully invisible nor fully hides the photo underneath.
        Driven by the "Overlay" slider in the toolbar."""
        self._overlay_opacity = max(0.1, min(0.9, value))
        self._refresh_overlay()

    def set_brightness(self, value: float):
        """Brighten or darken the displayed photo (this is a viewing aid only — it
        does NOT change the saved photo or the mask). Clamped to 0.5–3.0. Helps
        the user see a dark horizon line. Driven by the "Brightness" slider."""
        self._brightness = max(0.5, min(3.0, value))
        self._update_photo_display()

    def set_brush_radius(self, radius: int):
        """Set the brush size in pixels, clamped to the allowed 5–500 range, then
        tell the toolbar (via `brush_changed`) so the label updates. Used by the
        scroll wheel and the [ / ] keys."""
        self._brush_radius = max(self._min_brush, min(self._max_brush, radius))
        self.brush_changed.emit(self._brush_radius)
        self._update_cursor()

    @property
    def brush_radius(self):
        """Current brush radius in pixels (read-only accessor)."""
        return self._brush_radius

    def undo(self):
        """Step back one paint action. Pushes the current mask onto the redo stack,
        then restores the most recent snapshot saved by `_push_undo`. No-op if
        there is nothing to undo."""
        if self._undo_stack:
            self._redo_stack.append(self._mask_np.copy())
            self._mask_np = self._undo_stack.pop()
            self._refresh_overlay()

    def redo(self):
        """Re-apply an action that was just undone. Mirror image of `undo`: pushes
        the current mask back onto the undo stack and restores the top of the redo
        stack. No-op if there is nothing to redo."""
        if self._redo_stack:
            self._undo_stack.append(self._mask_np.copy())
            self._mask_np = self._redo_stack.pop()
            self._refresh_overlay()

    # ── Display helpers ──────────────────────────────────────────────────────

    def _update_photo_display(self):
        """Redraw the background photo, applying the current brightness setting.

        Brightness is a gamma-style curve built as a 256-entry lookup table: each
        possible pixel value 0–255 is remapped, then `lut[img]` recolors the whole
        image in one vectorized step (fast). At brightness 1.0 the table is skipped
        entirely and the original pixels are shown untouched. Called whenever the
        photo or the brightness changes. Creates the photo scene item on first call
        (Z=0, the bottom layer) and reuses it afterward."""
        if not hasattr(self, '_original_img'):
            return
        img = self._original_img
        if self._brightness != 1.0:
            # Gamma lookup table: value^(1/brightness). >1 brightens, <1 darkens.
            lut = np.clip(255.0 * (np.arange(256) / 255.0) ** (1.0 / self._brightness),
                          0, 255).astype(np.uint8)
            img = lut[img]
        qimg = numpy_to_qimage(img)
        pixmap = QPixmap.fromImage(qimg)
        if self._photo_item is None:
            self._photo_item = self.scene().addPixmap(pixmap)
            self._photo_item.setZValue(0)  # bottom layer
        else:
            self._photo_item.setPixmap(pixmap)

    def _refresh_overlay(self):
        """Repaint the translucent green layer so it matches the mask array.

        Builds a four-channel (color + transparency) picture the same size as the
        photo: where the mask is set, the pixel is opaque-ish green; everywhere
        else it is fully transparent so the photo shows through. Call this after
        ANY change to `_mask_np` (paint, erase, clear, undo, opacity change) — it
        is the one place the on-screen green is regenerated. Sits at Z=1, above the
        photo and below the brush cursor.

        The `masked = _mask_np > 127` threshold treats the mask as black-or-white
        (a resized mask could contain in-between gray values; this picks the
        white side). `.copy()` is required because the QImage otherwise just
        references the temporary `argb` buffer, which would be freed."""
        if self._mask_np is None:
            return
        h, w = self._mask_np.shape
        alpha = int(255 * self._overlay_opacity)

        # Format_ARGB32 is stored as B,G,R,A in memory on little-endian machines.
        # Index 1 = Green, index 3 = Alpha; leaving R and B at 0 gives pure green.
        argb = np.zeros((h, w, 4), dtype=np.uint8)
        masked = self._mask_np > 127
        argb[masked, 1] = 255      # G channel → green where painted
        argb[masked, 3] = alpha    # A channel → opacity where painted
        overlay = QImage(argb.data, w, h, 4 * w, QImage.Format_ARGB32).copy()

        pixmap = QPixmap.fromImage(overlay)
        if self._mask_overlay_item is None:
            self._mask_overlay_item = self.scene().addPixmap(pixmap)
            self._mask_overlay_item.setZValue(1)  # above photo, below cursor
        else:
            self._mask_overlay_item.setPixmap(pixmap)

    def _update_cursor(self):
        """Reset the pointer to a crosshair and drop the old brush-circle outline.

        Called when the brush size changes; the circle is then rebuilt at the live
        size by the next `_move_cursor_circle`. The crosshair is the pointer shape
        the user sees over the canvas."""
        if self._cursor_circle is not None:
            self.scene().removeItem(self._cursor_circle)
            self._cursor_circle = None
        self.setCursor(Qt.CrossCursor)

    def _move_cursor_circle(self, scene_pos):
        """Draw the brush-size outline (the ring that shows how big the brush is)
        centered on the pointer at `scene_pos` (a point in photo coordinates).

        Removes the previous ring and draws a fresh one. The ring is RED in erase
        mode and WHITE in paint mode so the user can tell at a glance which mode
        they are in. `pen_width` is divided by the current zoom (`m11()`) so the
        outline stays a thin, constant on-screen thickness no matter how far the
        user has zoomed in. Drawn at Z=10, on top of everything."""
        r = self._brush_radius
        if self._cursor_circle is not None:
            self.scene().removeItem(self._cursor_circle)
        scale = self.transform().m11()
        pen_width = max(1.0, 1.5 / scale) if scale > 0 else 1.5
        # Red circle in erase mode, white in paint mode
        color = QColor(255, 80, 80, 200) if self._erase_mode else QColor(255, 255, 255, 180)
        pen = QPen(color, pen_width)
        self._cursor_circle = self.scene().addEllipse(
            scene_pos.x() - r, scene_pos.y() - r, 2 * r, 2 * r, pen)
        self._cursor_circle.setZValue(10)  # top layer, above the green overlay

    # ── Undo helpers ─────────────────────────────────────────────────────────

    def _push_undo(self):
        """Snapshot the current mask onto the undo stack BEFORE a new edit begins.

        Called at the start of each paint stroke and before Clear All. Keeps at
        most `_max_undo` (50) snapshots, discarding the oldest when full. Starting a
        new edit also clears the redo stack, since the redo history no longer
        applies once the user paints something new."""
        self._undo_stack.append(self._mask_np.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    # ── Painting ─────────────────────────────────────────────────────────────

    def _paint_at(self, scene_pos, erase=False):
        """Stamp one filled brush circle into the mask array at `scene_pos`.

        `scene_pos` is a point in photo (pixel) coordinates; the brush radius is
        the current `_brush_radius`. Painting sets the covered pixels to 255 (mark
        as foreground); erasing sets them to 0. The window is first clamped to the
        image edges (so a brush hanging off the edge doesn't go out of bounds), and
        a circular boolean mask (`xx² + yy² <= r²`) restricts the write to a true
        circle rather than the bounding square. Does NOT refresh the overlay — the
        caller (`_paint_line`) does that once per stroke for speed."""
        cx = int(scene_pos.x())
        cy = int(scene_pos.y())
        r = self._brush_radius

        # Clamp the brush's bounding box to the image so slicing stays in bounds.
        y0 = max(0, cy - r)
        y1 = min(self._img_h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(self._img_w, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return  # brush is entirely off the image — nothing to paint

        # Boolean circle relative to the (possibly clipped) box, centered on cx,cy.
        yy, xx = np.ogrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        circle = (xx * xx + yy * yy) <= r * r

        if erase:
            self._mask_np[y0:y1, x0:x1][circle] = 0
        else:
            self._mask_np[y0:y1, x0:x1][circle] = 255

    def _paint_line(self, from_pos, to_pos, erase=False):
        """Paint a continuous streak between two points by stamping circles along
        the line. Without this, fast mouse movement would leave gaps between
        individual circle stamps. Steps roughly every third of a brush radius (so
        consecutive stamps overlap and the streak looks solid), then refreshes the
        green overlay once at the end. Used both for click-drag strokes and for the
        Shift+click straight-line feature."""
        x0, y0 = from_pos.x(), from_pos.y()
        x1, y1 = to_pos.x(), to_pos.y()
        dist = math.hypot(x1 - x0, y1 - y0)
        # Step size ~ a third of the brush radius keeps consecutive stamps
        # overlapping so the streak has no gaps.
        step = max(1, self._brush_radius // 3)
        n_steps = max(1, int(dist / step))
        for i in range(n_steps + 1):
            t = i / n_steps if n_steps > 0 else 0  # 0..1 along the segment
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            self._paint_at(QPointF(px, py), erase=erase)
        self._refresh_overlay()

    # ── Mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Handle a mouse-button press: start panning, toggle mode, or begin a
        paint stroke.

        Priority order matters:
          1. If Space is held, this press starts a PAN (grab the canvas) and
             nothing is painted.
          2. A right-click TOGGLES paint/erase mode (and stamps the time, so the
             trackpad two-finger-tap that accompanies a right-click doesn't also
             resize the brush — see `wheelEvent`).
          3. A left-click begins painting: snapshot for undo, then either draw a
             straight line from the previous click (if Shift is down) or stamp a
             single circle. The two `_last_*_pos` values remember where we are so
             drags interpolate and Shift+click chains straight segments."""
        if self._space_held:
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.RightButton:
            # Toggle erase mode on/off. Record the time so wheelEvent can ignore
            # the stray scroll a Mac trackpad emits alongside a right-click.
            import time as _time
            self._last_right_click_time = _time.time()
            self._erase_mode = not self._erase_mode
            self.mode_changed.emit(self._erase_mode)
            return

        if event.button() == Qt.LeftButton:
            self._push_undo()
            self._painting = True

            # Shift+click: draw straight line from last click position
            if (event.modifiers() & Qt.ShiftModifier) and self._last_click_pos is not None:
                self._paint_line(self._last_click_pos, scene_pos, erase=self._erase_mode)
            else:
                self._paint_at(scene_pos, erase=self._erase_mode)
                self._refresh_overlay()

            self._last_paint_pos = scene_pos   # for drag interpolation
            self._last_click_pos = scene_pos   # anchor for the next Shift+click

    def _pos_in_image(self, scene_pos):
        """True if a scene point falls inside the photo's bounds. (Helper; not
        currently called elsewhere in this file.)"""
        if self._photo_item is None:
            return False
        return self._photo_item.boundingRect().contains(scene_pos)

    def _hide_cursor_circle(self):
        """Remove the brush-size ring from the scene. (Helper; not currently called
        elsewhere in this file.)"""
        if self._cursor_circle is not None:
            self.scene().removeItem(self._cursor_circle)
            self._cursor_circle = None

    def mouseMoveEvent(self, event):
        """Handle pointer motion: pan while dragging, keep painting while the left
        button is down, and keep the brush ring under the cursor otherwise.

        While panning, the scrollbars are nudged by the drag delta to move the
        view. While painting, `_paint_line` connects the previous point to the
        current one so a fast drag still produces a solid streak."""
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._panning:
            pass
        else:
            self._move_cursor_circle(scene_pos)
            self.setCursor(Qt.CrossCursor)

        if self._panning and self._last_pan_pos is not None:
            delta = event.position().toPoint() - self._last_pan_pos
            self._last_pan_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return

        if self._painting and self._last_paint_pos is not None:
            self._paint_line(self._last_paint_pos, scene_pos, erase=self._erase_mode)
            self._last_paint_pos = scene_pos

    def mouseReleaseEvent(self, event):
        """End a paint stroke and/or a pan when the button is released, resetting
        the drag state and restoring the crosshair cursor."""
        if event.button() == Qt.LeftButton:
            self._painting = False
            self._last_paint_pos = None
        if self._panning:
            self._panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        """Scroll wheel / trackpad: zoom with Ctrl/Cmd held, otherwise resize the
        brush.

        Two subtleties:
          - A Mac trackpad fires a phantom scroll right after a right-click
            (two-finger tap). The 300ms guard ignores scrolls that closely follow
            a right-click so the brush doesn't jump when the user only meant to
            toggle mode.
          - Trackpads emit many tiny scroll deltas (a mouse wheel emits one big
            120-unit 'click'). To make the brush grow smoothly, fractional changes
            are accumulated in `_brush_scroll_accum` and only applied once a whole
            pixel of change has built up. `step_per_click` scales the speed with
            the current brush size (6–25px) so big brushes resize faster."""
        # Ignore scroll events within 300ms of right-click (Mac trackpad two-finger tap)
        import time as _time
        if _time.time() - self._last_right_click_time < 0.3:
            event.accept()
            return

        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._zoom_by(factor)
        else:
            # Accumulate fractional trackpad deltas so brush grows smoothly.
            # One full wheel click == angleDelta 120. Trackpad fires many small
            # deltas — we accumulate and only apply when >= 1 unit of change.
            step_per_click = max(6, min(25, self._brush_radius // 8))
            self._brush_scroll_accum = getattr(self, '_brush_scroll_accum', 0.0)
            self._brush_scroll_accum += (event.angleDelta().y() / 120.0) * step_per_click
            delta = int(self._brush_scroll_accum)
            if delta == 0:
                event.accept()
                return
            self._brush_scroll_accum -= delta  # keep the leftover fraction
            self.set_brush_radius(self._brush_radius + delta)
            scene_pos = self.mapToScene(event.position().toPoint())
            self._move_cursor_circle(scene_pos)

    def keyPressEvent(self, event):
        """Keyboard shortcuts on the canvas.

        Space (held) = pan mode; E = toggle paint/erase; [ and ] = shrink/grow the
        brush by 10px; 0 (or Ctrl/Cmd+0) = fit to window; + / - = zoom. `isAutoRepeat`
        checks ignore the repeated key events the OS sends while a key is held, so
        Space and E only fire once per physical press. Anything else falls through
        to the default handler."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_E and not event.isAutoRepeat():
            self._erase_mode = not self._erase_mode
            self.mode_changed.emit(self._erase_mode)
        elif event.key() == Qt.Key_BracketLeft:
            self.set_brush_radius(self._brush_radius - 10)
        elif event.key() == Qt.Key_BracketRight:
            self.set_brush_radius(self._brush_radius + 10)
        elif event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            self._zoom_to_fit()
        elif event.key() == Qt.Key_0 and not event.modifiers():
            self._zoom_to_fit()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_by(1.25)
        elif event.key() == Qt.Key_Minus:
            self._zoom_by(1 / 1.25)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Releasing Space ends pan mode and restores the crosshair cursor."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.setCursor(Qt.CrossCursor)
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        """On window resize, re-fit the photo to the new size and re-pin the
        floating zoom control to the corner."""
        super().resizeEvent(event)
        if self._photo_item:
            self.fitInView(self._photo_item, Qt.KeepAspectRatio)
            self._update_zoom_label()
        self._position_zoom_overlay()

    def event(self, event):
        """Intercept trackpad pinch gestures for zoom before the default handling.

        Qt delivers pinch as a generic Gesture event; when the pinch's scale factor
        changes we zoom by that factor. All other events pass through to the base
        class unchanged."""
        if event.type() == QEvent.Gesture:
            pinch = event.gesture(Qt.PinchGesture)
            if pinch is not None:
                if pinch.changeFlags() & QPinchGesture.ScaleFactorChanged:
                    f = pinch.scaleFactor()
                    if f and f > 0:
                        self._zoom_by(f)
                event.accept()
                return True
        return super().event(event)


class MaskPainterWidget(QWidget):
    """The complete mask-painting screen the user sees, wrapping the canvas.

    Layout top to bottom: a green instruction banner (with the "Skyline hard to
    see?" frame-stepping arrows on its right), a dark toolbar (Back, Paint/Erase
    toggle, brush readout, overlay + brightness sliders, Undo/Redo, Clear All,
    Help, and the blue "Save Mask" button), the `MaskGraphicsView` canvas, and a
    one-line status bar of shortcuts.

    This widget owns no mask data of its own — the canvas (`self._view`) does. The
    widget's job is the surrounding chrome and routing button/slider events into
    the canvas. It signals the host GUI when the user finishes."""

    # Signals the host GUI (star_trail_cleanr.py) listens to.
    mask_done = Signal(np.ndarray)   # final mask, emitted when user clicks Save Mask
    mask_skipped = Signal()          # user chose to process the whole frame (no mask)
    go_back = Signal()               # user clicked Back to leave this screen

    def __init__(self, parent=None):
        """Build the screen. `_frame_paths`/`_frame_idx` track the folder's photos
        and which one is currently shown as the background to trace against (the
        banner arrows step through them)."""
        super().__init__(parent)
        self._frame_paths = []   # all photos in the folder, in order
        self._frame_idx = 0      # which one is currently shown as the background
        self._build_ui()
        self._setup_shortcuts()

    def _build_ui(self):
        """Construct and lay out every widget on the screen (banner, toolbar,
        canvas, status bar) and wire each control to its handler. Long but
        mechanical; the inline section comments mark each region. Called once from
        __init__."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Instruction banner ───────────────────────────────────────────────
        self._banner = QWidget()
        self._banner.setFixedHeight(100)
        self._banner.setStyleSheet("background-color: #2a3a2a;")
        banner_layout = QHBoxLayout(self._banner)
        banner_layout.setContentsMargins(16, 8, 16, 8)

        banner_text = QLabel(
            "<b style='font-size: 20px;'>Roughly paint over the ground, rocks, and buildings. Stay BELOW the skyline.</b><br>"
            "No need to mask trees. Trails are visible through branches and the AI will still detect them there.<br>"
            "You're just marking areas where you know trails won't appear, so the AI doesn't try to 'fix' the ground."
        )
        banner_text.setStyleSheet("color: #a0d0a0; font-size: 16px;")
        banner_layout.addWidget(banner_text)

        # ── Background-photo selector (right side of the green banner) ─────────
        # Lets the user swap which photo they trace the skyline against — the
        # first file isn't always the clearest. Switching changes only the
        # background; the painted mask is shared by every frame.
        banner_layout.addStretch()

        self._nav_widget = QWidget()
        nav_layout = QHBoxLayout(self._nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)

        nav_lbl = QLabel("Skyline hard to see?")
        nav_lbl.setStyleSheet("color: #a0d0a0; font-size: 14px;")
        nav_layout.addWidget(nav_lbl)

        self._prev_frame_btn = QPushButton("‹")
        self._next_frame_btn = QPushButton("›")
        for _b in (self._prev_frame_btn, self._next_frame_btn):
            _b.setFixedSize(32, 32)
            _b.setCursor(Qt.PointingHandCursor)
            _b.setStyleSheet(
                "QPushButton { color: #d0f0d0; background: #1e2e1e; "
                "border: 1px solid #3a5a3a; border-radius: 4px; "
                "font-size: 18px; font-weight: bold; }"
                "QPushButton:hover { color: white; background: #2e4e2e; "
                "border-color: #5a8a5a; }"
                "QPushButton:disabled { color: #4a5a4a; background: #243424; "
                "border-color: #2a3a2a; }")
        self._prev_frame_btn.clicked.connect(self._prev_frame)
        self._next_frame_btn.clicked.connect(self._next_frame)
        nav_layout.addWidget(self._prev_frame_btn)
        nav_layout.addWidget(self._next_frame_btn)

        self._nav_widget.hide()  # shown once a multi-photo folder is loaded
        banner_layout.addWidget(self._nav_widget)

        layout.addWidget(self._banner)

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(48)
        toolbar.setStyleSheet("background-color: #1e1e1e;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 4, 12, 4)
        tb_layout.setSpacing(16)

        # Back button
        back_btn = QPushButton("\u2190 Back")
        back_btn.setFixedHeight(30)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet(
            "QPushButton { color: #ddd; background: #333; border: 1px solid #555; "
            "border-radius: 4px; padding: 4px 12px; font-size: 13px; }"
            "QPushButton:hover { color: white; background: #555; border-color: #777; }")
        back_btn.clicked.connect(self.go_back.emit)
        tb_layout.addWidget(back_btn)

        self._add_separator(tb_layout)

        # Paint / Erase toggle button pair
        mask_lbl = QLabel("Mask:")
        mask_lbl.setStyleSheet("color: #ccc; font-size: 12px;")
        tb_layout.addWidget(mask_lbl)

        self._paint_btn = QPushButton("Paint")
        self._paint_btn.setFixedHeight(30)
        self._paint_btn.setFixedWidth(70)
        self._paint_btn.clicked.connect(lambda: self._set_mode(False))
        tb_layout.addWidget(self._paint_btn)

        self._erase_btn = QPushButton("Erase")
        self._erase_btn.setFixedHeight(30)
        self._erase_btn.setFixedWidth(70)
        self._erase_btn.clicked.connect(lambda: self._set_mode(True))
        tb_layout.addWidget(self._erase_btn)

        self._update_mode_btns()

        self._add_separator(tb_layout)

        # Brush size label
        self._brush_label = QLabel("Brush: 150px")
        self._brush_label.setStyleSheet("color: #ccc; font-size: 12px;")
        tb_layout.addWidget(self._brush_label)

        self._add_separator(tb_layout)

        # Overlay opacity slider
        lbl_ov = QLabel("Overlay")
        lbl_ov.setStyleSheet("color: #ccc; font-size: 12px;")
        tb_layout.addWidget(lbl_ov)
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setFixedWidth(100)
        self._opacity_slider.setRange(10, 90)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setStyleSheet("QSlider { max-height: 20px; }")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        tb_layout.addWidget(self._opacity_slider)

        self._add_separator(tb_layout)

        # Brightness slider
        lbl_br = QLabel("Brightness")
        lbl_br.setStyleSheet("color: #ccc; font-size: 12px;")
        tb_layout.addWidget(lbl_br)
        self._brightness_slider = QSlider(Qt.Horizontal)
        self._brightness_slider.setFixedWidth(100)
        self._brightness_slider.setRange(50, 300)
        self._brightness_slider.setValue(100)
        self._brightness_slider.setStyleSheet("QSlider { max-height: 20px; }")
        self._brightness_slider.valueChanged.connect(self._on_brightness_changed)
        tb_layout.addWidget(self._brightness_slider)

        self._add_separator(tb_layout)

        # Undo / Redo
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setStyleSheet(
            "QPushButton { color: #ccc; background: transparent; border: none; font-size: 12px; }"
            "QPushButton:hover { color: white; }")
        self._undo_btn.clicked.connect(self._on_undo)
        tb_layout.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("Redo")
        self._redo_btn.setStyleSheet(
            "QPushButton { color: #ccc; background: transparent; border: none; font-size: 12px; }"
            "QPushButton:hover { color: white; }")
        self._redo_btn.clicked.connect(self._on_redo)
        tb_layout.addWidget(self._redo_btn)

        # Clear All
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(
            "QPushButton { color: #cc4444; background: transparent; border: none; font-size: 12px; }"
            "QPushButton:hover { color: #ff6666; }")
        clear_btn.clicked.connect(self._on_clear)
        tb_layout.addWidget(clear_btn)

        self._add_separator(tb_layout)

        # Help button
        help_btn = QPushButton("?")
        help_btn.setFixedSize(28, 28)
        help_btn.setStyleSheet(
            "QPushButton { color: #ccc; background: #333; border: 1px solid #555; "
            "border-radius: 14px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { color: white; background: #555; }")
        help_btn.clicked.connect(self._show_help)
        tb_layout.addWidget(help_btn)

        tb_layout.addStretch()

        # Done button
        done_btn = QPushButton("Save Mask")
        done_btn.setFixedHeight(36)
        done_btn.setFixedWidth(200)
        done_btn.setStyleSheet(
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #1580e0; }")
        done_btn.clicked.connect(self._on_done)
        tb_layout.addWidget(done_btn)

        layout.addWidget(toolbar)

        # ── Canvas ───────────────────────────────────────────────────────────
        self._scene = QGraphicsScene()
        self._view = MaskGraphicsView(self._scene, self)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view.brush_changed.connect(self._update_brush_label)
        self._view.mode_changed.connect(lambda _: self._update_mode_btns())
        layout.addWidget(self._view)

        # ── Status bar ───────────────────────────────────────────────────────
        status_bar = QWidget()
        status_bar.setFixedHeight(28)
        status_bar.setStyleSheet("background-color: #1e1e1e;")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(12, 2, 12, 2)
        status_text = QLabel(
            "Click+drag: paint or erase  \u00b7  Scroll: brush size  \u00b7  "
            "Shift+click: straight line  \u00b7  Space+drag: pan  \u00b7  "
            "+ / \u2212 or pinch: zoom  \u00b7  0: fit  \u00b7  Cmd+Z: undo")
        status_text.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        sb_layout.addWidget(status_text)
        layout.addWidget(status_bar)

    def _add_separator(self, layout):
        """Add a thin vertical divider line into a toolbar layout, used to group
        related controls visually."""
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet("background-color: #444;")
        layout.addWidget(sep)

    def _setup_shortcuts(self):
        """Register the window-level undo/redo shortcuts. On macOS Qt maps Ctrl to
        the Cmd key automatically, so these are Cmd+Z / Cmd+Shift+Z on a Mac."""
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    # ── Public API ───────────────────────────────────────────────────────────

    def load_image(self, img_path: str):
        """Open an image file from disk into the canvas as a fresh background.

        Reads the file with `robust_imread` (the app's safe loader that handles
        odd paths/formats) in color; silently does nothing if it fails to load.
        Resets to an empty mask via the canvas's `load_image`."""
        import cv2
        from modules.io_safe import robust_imread
        img = robust_imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return
        self._view.load_image(img)
        self._update_brush_label(self._view.brush_radius)
        # Show banner each time a new image is loaded
        self._banner.show()

    def load_image_array(self, img_np):
        """Same as `load_image` but takes an already-decoded BGR numpy array
        instead of a file path. Used when the caller already has the pixels in
        memory."""
        self._view.load_image(img_np)
        self._update_brush_label(self._view.brush_radius)
        self._banner.show()

    def load_frames(self, paths, index=0):
        """Give the editor the full list of photos in the folder and show the
        one at `index` as the background to trace against. The arrows in the
        banner step through the rest."""
        self._frame_paths = list(paths)
        if self._frame_paths:
            self._frame_idx = max(0, min(index, len(self._frame_paths) - 1))
            self.load_image(self._frame_paths[self._frame_idx])
        else:
            self._frame_idx = 0
        self._update_frame_nav()

    def _prev_frame(self):
        """Banner left-arrow: show the previous photo as the background (mask kept).
        Stops at the first photo."""
        if self._frame_idx > 0:
            self._frame_idx -= 1
            self._show_background(self._frame_idx)

    def _next_frame(self):
        """Banner right-arrow: show the next photo as the background (mask kept).
        Stops at the last photo."""
        if self._frame_idx < len(self._frame_paths) - 1:
            self._frame_idx += 1
            self._show_background(self._frame_idx)

    def _show_background(self, idx):
        """Swap the background to photo `idx`, keeping the painted mask."""
        import cv2
        from modules.io_safe import robust_imread
        img = robust_imread(self._frame_paths[idx], cv2.IMREAD_COLOR)
        if img is not None:
            self._view.set_background_image(img)
        self._update_frame_nav()

    def _update_frame_nav(self):
        """Show the arrows only for multi-photo folders; grey out the ends."""
        n = len(self._frame_paths)
        if n <= 1:
            self._nav_widget.hide()
            return
        self._nav_widget.show()
        self._prev_frame_btn.setEnabled(self._frame_idx > 0)
        self._next_frame_btn.setEnabled(self._frame_idx < n - 1)

    def load_existing_mask(self, mask_path: str):
        """Load a saved mask PNG from disk so the user can continue editing it.
        Reads grayscale via the safe loader; ignores load failures. Call after the
        image is loaded so sizes line up."""
        import cv2
        from modules.io_safe import robust_imread
        mask = robust_imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            self._view.load_mask(mask)

    # ── Slots ────────────────────────────────────────────────────────────────
    # Small handlers connected to the toolbar controls; each just forwards to the
    # canvas. Slider values come in as integers (Qt sliders are integer-only) and
    # are divided by 100 to get the fractional opacity / brightness the canvas wants.

    def _on_opacity_changed(self, value):
        """Overlay slider moved: pass the 0–100 value to the canvas as 0.0–1.0."""
        self._view.set_overlay_opacity(value / 100.0)

    def _on_brightness_changed(self, value):
        """Brightness slider moved: 50–300 becomes 0.5–3.0 for the canvas."""
        self._view.set_brightness(value / 100.0)

    def _on_undo(self):
        """Undo button / shortcut: undo the last paint action on the canvas."""
        self._view.undo()

    def _on_redo(self):
        """Redo button / shortcut: redo the last undone action."""
        self._view.redo()

    def _set_mode(self, erase):
        """Switch the canvas between paint and erase (called by the toolbar's
        Paint/Erase buttons) and refresh which button looks active.

        NOTE: this reaches into the canvas's `_erase_mode` directly rather than
        going through a setter, so it does NOT emit `mode_changed`; the button
        styling is updated here instead."""
        self._view._erase_mode = erase
        self._update_mode_btns()

    def _update_mode_btns(self):
        """Restyle the Paint and Erase buttons so the active one is highlighted
        (green for paint, red for erase) and the other looks dimmed. Called after
        any mode change, from a button click or a right-click/E-key toggle on the
        canvas."""
        active_green = ("QPushButton { background-color: #2a7a2a; color: white; font-size: 12px; "
                        "font-weight: bold; border-radius: 4px 0 0 4px; border: none; }")
        active_red = ("QPushButton { background-color: #aa3333; color: white; font-size: 12px; "
                      "font-weight: bold; border-radius: 0 4px 4px 0; border: none; }")
        inactive = ("QPushButton { background-color: #444; color: #999; font-size: 12px; "
                    "font-weight: bold; border: none; ")
        inactive_left = inactive + "border-radius: 4px 0 0 4px; }"
        inactive_right = inactive + "border-radius: 0 4px 4px 0; }"

        erase = self._view._erase_mode if hasattr(self, '_view') else False
        if erase:
            self._paint_btn.setStyleSheet(inactive_left)
            self._erase_btn.setStyleSheet(active_red)
        else:
            self._paint_btn.setStyleSheet(active_green)
            self._erase_btn.setStyleSheet(inactive_right)

    def _show_help(self):
        """Pop up the shortcuts cheat-sheet dialog (the toolbar "?" button). The
        modifier label shows "Cmd" on macOS and "Ctrl" elsewhere so the listed
        shortcuts match the user's actual keyboard."""
        import sys as _sys
        mod = "Cmd" if _sys.platform == "darwin" else "Ctrl"
        QMessageBox.information(self, "Mask Editor Shortcuts",
            "Mouse:\n"
            "  Left-click + drag \u2014 paint or erase\n"
            "  Right-click \u2014 toggle Paint / Erase mode\n"
            "  Scroll wheel \u2014 change brush size\n"
            "  Shift + click \u2014 draw straight line from last point\n"
            "  Space + drag \u2014 pan the image\n"
            f"  {mod} + scroll \u2014 zoom in/out\n"
            "  Pinch (trackpad) \u2014 zoom in/out\n\n"
            "Keyboard:\n"
            "  E \u2014 toggle Paint / Erase mode\n"
            "  [ / ] \u2014 decrease / increase brush size\n"
            "  + / \u2212 \u2014 zoom in / out\n"
            "  0 \u2014 fit image to window\n"
            f"  {mod}+Z \u2014 undo\n"
            f"  {mod}+Shift+Z \u2014 redo\n"
            f"  {mod}+0 \u2014 fit image to window")

    def _on_clear(self):
        """Clear All button: wipe the whole mask (undoable)."""
        self._view.clear_mask()

    def _on_done(self):
        """Save Mask button: finish the screen and hand the result to the host GUI.

        If nothing was painted, warns the user that the whole frame will be
        processed and asks to confirm; on confirm it emits `mask_skipped`. If a
        mask exists it emits `mask_done` carrying a copy of the mask array. These
        signals are how the host moves on to the detection/repair pipeline."""
        if not self._view.has_mask():
            reply = QMessageBox.question(
                self, "No mask painted",
                "No foreground was masked. The AI will process the entire frame.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
            self.mask_skipped.emit()
        else:
            self.mask_done.emit(self._view.get_mask())

    def _update_brush_label(self, radius=None):
        """Refresh the toolbar's "Brush: Npx" readout. Connected to the canvas's
        `brush_changed` signal; if no radius is passed it reads the current one
        from the canvas."""
        if radius is None:
            radius = self._view.brush_radius
        self._brush_label.setText(f"Brush: {radius}px")
