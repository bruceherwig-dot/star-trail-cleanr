"""
mask_painter.py — Foreground mask painting widget for Star Trail CleanR.

Green overlay mask: user paints over ground/buildings/rocks so the AI skips those areas.
Left-click = paint, Right-click = erase, Scroll = brush size, Space+drag = pan.
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
    """Convert a BGR numpy array (uint8) to QImage RGB888."""
    h, w = arr.shape[:2]
    if arr.ndim == 3:
        rgb = arr[:, :, ::-1].copy()  # BGR → RGB
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QImage(arr.data, w, h, w, QImage.Format_Grayscale8)


class MaskGraphicsView(QGraphicsView):
    """Custom QGraphicsView with painting, panning, and zooming."""

    brush_changed = Signal(int)   # emitted when brush radius changes
    mode_changed = Signal(bool)   # emitted when erase mode toggles (True=erase)

    def __init__(self, scene, parent=None):
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
        if self._zoom_overlay is None:
            return
        margin = 14
        vp = self.viewport()
        x = vp.width() - self._zoom_overlay.width() - margin
        y = vp.height() - self._zoom_overlay.height() - margin
        self._zoom_overlay.move(x, y)
        self._zoom_overlay.raise_()

    def _zoom_by(self, factor):
        self.scale(factor, factor)
        self._update_zoom_label()

    def _zoom_to_fit(self):
        if self._photo_item:
            self.fitInView(self._photo_item, Qt.KeepAspectRatio)
            self._update_zoom_label()

    def _update_zoom_label(self):
        if self._zoom_label is None:
            return
        pct = int(round(self.transform().m11() * 100))
        self._zoom_label.setText(f"{pct}%")

    # ── Public API ───────────────────────────────────────────────────────────

    def load_image(self, img_np: np.ndarray):
        """Load a BGR numpy image as the background."""
        self._img_h, self._img_w = img_np.shape[:2]
        self._original_img = img_np.copy()
        self._update_photo_display()

        # Initialize blank mask
        self._mask_np = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        self._refresh_overlay()
        self.fitInView(self._photo_item, Qt.KeepAspectRatio)
        self._update_zoom_label()

    def load_mask(self, mask_np: np.ndarray):
        """Load an existing mask (255=foreground)."""
        if mask_np.shape[:2] != (self._img_h, self._img_w):
            import cv2
            mask_np = cv2.resize(mask_np, (self._img_w, self._img_h),
                                 interpolation=cv2.INTER_NEAREST)
        self._mask_np = mask_np.copy()
        self._refresh_overlay()

    def get_mask(self) -> np.ndarray:
        """Return the current mask as numpy uint8 (255=foreground)."""
        return self._mask_np.copy()

    def has_mask(self) -> bool:
        """True if any pixels are painted."""
        return self._mask_np is not None and self._mask_np.any()

    def clear_mask(self):
        """Clear the entire mask."""
        self._push_undo()
        self._mask_np[:] = 0
        self._refresh_overlay()

    def set_overlay_opacity(self, value: float):
        self._overlay_opacity = max(0.1, min(0.9, value))
        self._refresh_overlay()

    def set_brightness(self, value: float):
        self._brightness = max(0.5, min(3.0, value))
        self._update_photo_display()

    def set_brush_radius(self, radius: int):
        self._brush_radius = max(self._min_brush, min(self._max_brush, radius))
        self.brush_changed.emit(self._brush_radius)
        self._update_cursor()

    @property
    def brush_radius(self):
        return self._brush_radius

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(self._mask_np.copy())
            self._mask_np = self._undo_stack.pop()
            self._refresh_overlay()

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(self._mask_np.copy())
            self._mask_np = self._redo_stack.pop()
            self._refresh_overlay()

    # ── Display helpers ──────────────────────────────────────────────────────

    def _update_photo_display(self):
        """Recompute the displayed photo with brightness adjustment."""
        if not hasattr(self, '_original_img'):
            return
        img = self._original_img
        if self._brightness != 1.0:
            lut = np.clip(255.0 * (np.arange(256) / 255.0) ** (1.0 / self._brightness),
                          0, 255).astype(np.uint8)
            img = lut[img]
        qimg = numpy_to_qimage(img)
        pixmap = QPixmap.fromImage(qimg)
        if self._photo_item is None:
            self._photo_item = self.scene().addPixmap(pixmap)
            self._photo_item.setZValue(0)
        else:
            self._photo_item.setPixmap(pixmap)

    def _refresh_overlay(self):
        """Rebuild the green overlay from the mask numpy array."""
        if self._mask_np is None:
            return
        h, w = self._mask_np.shape
        alpha = int(255 * self._overlay_opacity)

        # Build ARGB overlay — QImage Format_ARGB32 is BGRA in memory on little-endian
        argb = np.zeros((h, w, 4), dtype=np.uint8)
        masked = self._mask_np > 127
        argb[masked, 1] = 255      # G channel
        argb[masked, 3] = alpha    # A channel
        overlay = QImage(argb.data, w, h, 4 * w, QImage.Format_ARGB32).copy()

        pixmap = QPixmap.fromImage(overlay)
        if self._mask_overlay_item is None:
            self._mask_overlay_item = self.scene().addPixmap(pixmap)
            self._mask_overlay_item.setZValue(1)
        else:
            self._mask_overlay_item.setPixmap(pixmap)

    def _update_cursor(self):
        """Update the brush circle cursor."""
        if self._cursor_circle is not None:
            self.scene().removeItem(self._cursor_circle)
            self._cursor_circle = None
        self.setCursor(Qt.CrossCursor)

    def _move_cursor_circle(self, scene_pos):
        """Position the brush circle indicator at the given scene coordinate."""
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
        self._cursor_circle.setZValue(10)
        # Prevent cursor circle from extending the scene bounding rect
        from PySide6.QtWidgets import QGraphicsItem
        self._cursor_circle.setFlag(QGraphicsItem.ItemHasNoContents, False)
        # Lock scene rect to the image bounds so cursor doesn't shift the view
        if self._photo_item:
            self.setSceneRect(self._photo_item.boundingRect())

    # ── Undo helpers ─────────────────────────────────────────────────────────

    def _push_undo(self):
        self._undo_stack.append(self._mask_np.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    # ── Painting ─────────────────────────────────────────────────────────────

    def _paint_at(self, scene_pos, erase=False):
        """Paint or erase a circle on the mask at the given scene position."""
        cx = int(scene_pos.x())
        cy = int(scene_pos.y())
        r = self._brush_radius

        y0 = max(0, cy - r)
        y1 = min(self._img_h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(self._img_w, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return

        yy, xx = np.ogrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        circle = (xx * xx + yy * yy) <= r * r

        if erase:
            self._mask_np[y0:y1, x0:x1][circle] = 0
        else:
            self._mask_np[y0:y1, x0:x1][circle] = 255

    def _paint_line(self, from_pos, to_pos, erase=False):
        """Interpolate between two points and paint along the line."""
        x0, y0 = from_pos.x(), from_pos.y()
        x1, y1 = to_pos.x(), to_pos.y()
        dist = math.hypot(x1 - x0, y1 - y0)
        # Step size = half brush radius for smooth coverage
        step = max(1, self._brush_radius // 3)
        n_steps = max(1, int(dist / step))
        for i in range(n_steps + 1):
            t = i / n_steps if n_steps > 0 else 0
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            self._paint_at(QPointF(px, py), erase=erase)
        self._refresh_overlay()

    # ── Mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._space_held:
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.RightButton:
            # Toggle erase mode on/off
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

            self._last_paint_pos = scene_pos
            self._last_click_pos = scene_pos

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self._move_cursor_circle(scene_pos)

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
        if event.button() == Qt.LeftButton:
            self._painting = False
            self._last_paint_pos = None
        if self._panning:
            self._panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        # Ignore scroll events within 300ms of right-click (Mac trackpad two-finger tap)
        import time as _time
        if _time.time() - self._last_right_click_time < 0.3:
            event.accept()
            return

        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            self._update_zoom_label()
        else:
            # Scale step with brush size, capped at 15 for smooth feel
            step = min(15, max(5, self._brush_radius // 10))
            delta = step if event.angleDelta().y() > 0 else -step
            self.set_brush_radius(self._brush_radius + delta)
            scene_pos = self.mapToScene(event.position().toPoint())
            self._move_cursor_circle(scene_pos)

    def keyPressEvent(self, event):
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
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.setCursor(Qt.CrossCursor)
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._photo_item:
            self.fitInView(self._photo_item, Qt.KeepAspectRatio)
            self._update_zoom_label()
        self._position_zoom_overlay()

    def event(self, event):
        if event.type() == QEvent.Gesture:
            pinch = event.gesture(Qt.PinchGesture)
            if pinch is not None:
                if pinch.changeFlags() & QPinchGesture.ScaleFactorChanged:
                    f = pinch.scaleFactor()
                    if f and f > 0:
                        self.scale(f, f)
                        self._update_zoom_label()
                event.accept()
                return True
        return super().event(event)


class MaskPainterWidget(QWidget):
    """Full mask painting screen — toolbar + canvas + status bar."""

    mask_done = Signal(np.ndarray)   # emitted with the final mask when user clicks Done
    mask_skipped = Signal()          # emitted when user skips masking
    go_back = Signal()               # emitted when user clicks Back

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._setup_shortcuts()

    def _build_ui(self):
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
            "No need to mask trees \u2014 trails are visible through branches and the AI will still detect them there.<br>"
            "You're just marking areas where you know trails won't appear."
        )
        banner_text.setStyleSheet("color: #a0d0a0; font-size: 16px;")
        banner_layout.addWidget(banner_text)

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
        back_btn.setStyleSheet(
            "QPushButton { color: #ccc; background: transparent; border: none; font-size: 13px; }"
            "QPushButton:hover { color: white; }")
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
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet("background-color: #444;")
        layout.addWidget(sep)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    # ── Public API ───────────────────────────────────────────────────────────

    def load_image(self, img_path: str):
        """Load an image file into the canvas."""
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return
        self._view.load_image(img)
        self._update_brush_label(self._view.brush_radius)
        # Show banner each time a new image is loaded
        self._banner.show()

    def load_image_array(self, img_np):
        """Load a BGR numpy array directly."""
        self._view.load_image(img_np)
        self._update_brush_label(self._view.brush_radius)
        self._banner.show()

    def load_existing_mask(self, mask_path: str):
        """Load a previously saved mask."""
        import cv2
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            self._view.load_mask(mask)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_opacity_changed(self, value):
        self._view.set_overlay_opacity(value / 100.0)

    def _on_brightness_changed(self, value):
        self._view.set_brightness(value / 100.0)

    def _on_undo(self):
        self._view.undo()

    def _on_redo(self):
        self._view.redo()

    def _set_mode(self, erase):
        self._view._erase_mode = erase
        self._update_mode_btns()

    def _update_mode_btns(self):
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
        self._view.clear_mask()

    def _on_done(self):
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
        if radius is None:
            radius = self._view.brush_radius
        self._brush_label.setText(f"Brush: {radius}px")
