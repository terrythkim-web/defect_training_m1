"""Image canvas (PRD F2.3): wheel zoom under the cursor, Space/middle-button pan, fit, pixel
coordinates and display-only brightness/contrast. Drawing tools are added in PROMPTS step 6."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QKeySequence, QMouseEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.imaging import apply_brightness_contrast

MIN_ZOOM, MAX_ZOOM = 0.02, 64.0
_PIXELATED_FROM = 2.0  # show crisp pixels from this zoom level on


def native_shortcut(keys: str) -> str:
    """Shortcut as the OS writes it: "Ctrl+0" on Linux/Windows, "⌘0" on macOS (Qt maps Ctrl to
    Command there for real shortcuts; text written by hand would still say Ctrl)."""
    return QKeySequence(keys).toString(QKeySequence.SequenceFormat.NativeText)


def numpy_to_qimage(image8: np.ndarray) -> QImage:
    """Deep-copied QImage from a uint8 gray (H, W) or BGR (H, W, 3) array."""
    image8 = np.ascontiguousarray(image8)
    h, w = image8.shape[:2]
    if image8.ndim == 2:
        fmt = QImage.Format.Format_Grayscale8
    elif image8.shape[2] == 3:
        fmt = QImage.Format.Format_BGR888
    else:
        raise ValueError(f"unsupported display image shape {image8.shape}")
    return QImage(image8.data, w, h, image8.strides[0], fmt).copy()


class ImageCanvas(QGraphicsView):
    cursor_moved = Signal(object, object)  # image pixel x, y (ints) or None, None outside
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)  # parented, or Python garbage-collects it
        self.setScene(self._scene)
        self.setObjectName("canvas")
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Panning is by Space/middle-drag; hidden scrollbars still track the view position.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self.scene().addItem(self._pixmap_item)
        self._placeholder = self.scene().addText("")
        self._placeholder.setDefaultTextColor(Qt.GlobalColor.white)

        self._base: np.ndarray | None = None  # display image before brightness/contrast
        self.brightness = 0
        self.contrast = 0
        self._fit_mode = True  # keep fitting on resize until the user zooms
        self._space_down = False
        self._middle_pan_from: QPointF | None = None

    # ---- image -------------------------------------------------------------------
    @property
    def has_image(self) -> bool:
        return self._base is not None

    @property
    def image_size(self) -> tuple[int, int] | None:
        return None if self._base is None else (self._base.shape[1], self._base.shape[0])

    def set_placeholder(self, text: str) -> None:
        self._placeholder.setPlainText(text)

    def set_image(self, image8: np.ndarray | None) -> None:
        """Show a uint8 display image (gray or BGR), or clear the canvas with None."""
        self._base = image8
        if image8 is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._placeholder.setVisible(True)
            self.scene().setSceneRect(self._placeholder.boundingRect())
            return
        self._placeholder.setVisible(False)
        self._render()
        w, h = image8.shape[1], image8.shape[0]
        # Margin so the image can be panned partly off-screen.
        self.scene().setSceneRect(QRectF(-w, -h, 3 * w, 3 * h))
        self.fit()

    def set_adjustment(self, brightness: int, contrast: int) -> None:
        """Display-only brightness/contrast in [-100, 100]; never touches labels or files."""
        self.brightness, self.contrast = brightness, contrast
        if self._base is not None:
            self._render()

    def _render(self) -> None:
        shown = apply_brightness_contrast(self._base, self.brightness, self.contrast)
        self._pixmap_item.setPixmap(QPixmap.fromImage(numpy_to_qimage(shown)))
        self._update_pixel_mode()

    # ---- zoom --------------------------------------------------------------------
    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def fit(self) -> None:
        if self._base is None:
            return
        self._fit_mode = True
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._after_zoom()

    def zoom_to(self, factor: float) -> None:
        factor = min(max(factor, MIN_ZOOM), MAX_ZOOM)
        self._fit_mode = False
        ratio = factor / self.zoom
        self.scale(ratio, ratio)  # scale() honours the under-mouse anchor
        self._after_zoom()

    def zoom_by(self, step: float) -> None:
        self.zoom_to(self.zoom * step)

    def _after_zoom(self) -> None:
        self._update_pixel_mode()
        self.zoom_changed.emit(self.zoom)

    def _update_pixel_mode(self) -> None:
        mode = (
            Qt.TransformationMode.FastTransformation
            if self.zoom >= _PIXELATED_FROM
            else Qt.TransformationMode.SmoothTransformation
        )
        self._pixmap_item.setTransformationMode(mode)

    # ---- coordinates ---------------------------------------------------------------
    def image_pos(self, view_pos) -> tuple[float, float] | None:
        """Continuous image coordinates under a viewport position, or None outside the image."""
        if self._base is None:
            return None
        p = self._pixmap_item.mapFromScene(self.mapToScene(view_pos))
        h, w = self._base.shape[:2]
        if 0 <= p.x() < w and 0 <= p.y() < h:
            return p.x(), p.y()
        return None

    # ---- events ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._base is None:
            return
        steps = event.angleDelta().y() / 120
        if steps:
            self.zoom_by(1.25 ** steps)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan_from = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._middle_pan_from is not None:
            delta = event.position() - self._middle_pan_from
            self._middle_pan_from = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
        pos = self.image_pos(event.position().toPoint())
        if pos is None:
            self.cursor_moved.emit(None, None)
        else:
            self.cursor_moved.emit(math.floor(pos[0]), math.floor(pos[1]))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_from is not None:
            self._middle_pan_from = None
            self.viewport().unsetCursor()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.cursor_moved.emit(None, None)
        super().leaveEvent(event)


class CanvasPanel(QWidget):
    """Canvas with a small view bar: fit, 100 %, zoom readout, brightness/contrast."""

    def __init__(self, canvas: ImageCanvas | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas or ImageCanvas()
        self.canvas.setParent(self)

        self.btn_fit = QPushButton(self.tr("맞춤"))
        self.btn_fit.setToolTip(self.tr("화면에 맞춤 ({0})").format(native_shortcut("Ctrl+0")))
        self.btn_fit.clicked.connect(self.canvas.fit)
        self.btn_100 = QPushButton("100%")
        self.btn_100.setToolTip(self.tr("실제 크기 ({0})").format(native_shortcut("Ctrl+1")))
        self.btn_100.clicked.connect(lambda: self.canvas.zoom_to(1.0))
        self.zoom_label = QLabel("")
        self.zoom_label.setMinimumWidth(56)
        self.canvas.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))

        self.brightness = _slider()
        self.contrast = _slider()
        self.brightness.valueChanged.connect(self._apply_adjustment)
        self.contrast.valueChanged.connect(self._apply_adjustment)
        self.btn_reset = QPushButton(self.tr("초기화"))
        self.btn_reset.setToolTip(self.tr("밝기·대비 초기화 (표시만 바뀌고 라벨·파일에는 영향 없음)"))
        self.btn_reset.clicked.connect(self.reset_adjustment)

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        for w in (self.btn_fit, self.btn_100, self.zoom_label):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(QLabel(self.tr("밝기")))
        bar.addWidget(self.brightness)
        bar.addWidget(QLabel(self.tr("대비")))
        bar.addWidget(self.contrast)
        bar.addWidget(self.btn_reset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)

    def _apply_adjustment(self) -> None:
        self.canvas.set_adjustment(self.brightness.value(), self.contrast.value())

    def reset_adjustment(self) -> None:
        self.brightness.setValue(0)
        self.contrast.setValue(0)


def _slider() -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(-100, 100)
    s.setValue(0)
    # shrinks with the canvas (a fixed 110 px pair made the window too wide for 1366 px screens)
    s.setMinimumWidth(50)
    s.setMaximumWidth(110)
    return s
