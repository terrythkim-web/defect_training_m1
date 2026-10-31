"""Brush tool UI: the stroke being painted, the size ring under the cursor and the options bar.
The label edit itself is Qt-free (`core/brush.py`); the canvas hands it the stroke on release."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from defect_inspection.core import brush
from defect_inspection.platform_utils import IS_MAC

ERASE_COLOR = QColor(255, 255, 255, 150)
PAINT_ALPHA = 110
_MIN_STEP_PX = 0.5  # image px: mouse moves shorter than this add no point


class BrushStroke:
    """Points of one stroke and its preview: a round-capped path as wide as the brush."""

    def __init__(self, scene: QGraphicsScene, start: QPointF, size: int, color: str, erase: bool) -> None:
        self.erase = erase
        self.points: list[tuple[float, float]] = [(start.x(), start.y())]
        fill = ERASE_COLOR if erase else QColor(color)
        if not erase:
            fill.setAlpha(PAINT_ALPHA)
        pen = QPen(fill, size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self._path = QPainterPath(start)
        self._path.lineTo(start)  # a zero-length segment draws a dot for a single click
        self.item: QGraphicsPathItem = scene.addPath(self._path, pen)
        self.item.setZValue(20)
        self._min_step = max(_MIN_STEP_PX, size / 8)

    def add(self, p: QPointF) -> None:
        """Extend the stroke; points closer than ~1/8 brush apart add nothing (the dabs overlap anyway),
        which keeps a long stroke's preview path short."""
        x, y = self.points[-1]
        if abs(p.x() - x) + abs(p.y() - y) >= self._min_step:
            self.points.append((p.x(), p.y()))
            self._path.lineTo(p)
            self.item.setPath(self._path)

    def remove(self) -> None:
        scene = self.item.scene()
        if scene is not None:
            scene.removeItem(self.item)


def make_ring(scene: QGraphicsScene) -> QGraphicsEllipseItem:
    """Brush outline under the cursor (image-pixel size, so it scales with the zoom)."""
    ring = QGraphicsEllipseItem()
    pen = QPen(QColor("white"), 1, Qt.PenStyle.DashLine)
    pen.setCosmetic(True)
    ring.setPen(pen)
    ring.setZValue(40)
    ring.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
    ring.setVisible(False)
    scene.addItem(ring)
    return ring


def place_ring(ring: QGraphicsEllipseItem, center: QPointF, size: int) -> None:
    r = size / 2
    ring.setRect(QRectF(-r, -r, size, size))
    ring.setPos(center)


class BrushBar(QWidget):
    """Options shown while the brush tool is active."""

    size_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(brush.MIN_SIZE, brush.MAX_SIZE)
        self.size_spin.setSuffix(" px")
        self.size_spin.setValue(brush.DEFAULT_SIZE)
        self.size_spin.setToolTip(self.tr("브러시 지름 (원본 이미지 픽셀) — [ / ] 키로 조절"))
        self.size_spin.valueChanged.connect(self.size_changed.emit)
        hint = QLabel(self.tr("왼쪽 드래그: 칠하기 (같은 클래스 라벨에 합쳐짐) · 오른쪽 또는 {0}+드래그: 지우기 · "
                              "[ ]: 크기").format("⌥" if IS_MAC else "Alt"))
        hint.setStyleSheet("color: gray")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel(self.tr("<b>브러시</b>")))
        layout.addWidget(QLabel(self.tr("크기")))
        layout.addWidget(self.size_spin)
        layout.addWidget(hint)
        layout.addStretch(1)

    def set_size(self, size: int) -> None:
        self.size_spin.blockSignals(True)
        self.size_spin.setValue(size)
        self.size_spin.blockSignals(False)
