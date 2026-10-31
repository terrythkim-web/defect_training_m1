"""Graphics items for label shapes. Items live in scene coordinates == image pixel coordinates.

Items only preview edits while the mouse is down; on release they report the result to their
controller (the canvas), which commits it to the LabelSession. The session then rebuilds the
items, so the model is always the single source of truth.
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem, QStyleOptionGraphicsItem

from defect_inspection.core.annotation import BBox, Mask, Polygon
from defect_inspection.core.label_edit import clamp_point, clamp_translation

Shape = BBox | Polygon | Mask

HANDLE_SIZE = 8  # screen pixels
_tag_font: list[QFont] = []


def tag_font() -> QFont:
    """Font of shape tags. Created on first use, after the QApplication exists: a QFont made at
    import time has no family, so on macOS it measured 129 px for text painted 93 px wide in the
    system font (tag background too wide, glyphs overlapping)."""
    if not _tag_font:
        font = QFont()
        font.setPointSize(9)
        _tag_font.append(font)
    return _tag_font[0]


class ShapeController(Protocol):
    image_width: int
    image_height: int

    def shape_moved(self, shape_id: str, dx: float, dy: float) -> None: ...
    def vertex_moved(self, shape_id: str, index: int, x: float, y: float) -> None: ...


def shape_points(shape: Shape) -> list[tuple[float, float]]:
    return shape.to_points()


def mask_path(shape: Mask) -> QPainterPath:
    """Every outline and hole of a mask label, filled with the even-odd rule (holes stay open)."""
    from defect_inspection.core.mask_codec import edge_rings

    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for ring in edge_rings(shape.pixels()):
        path.addPolygon(QPolygonF([QPointF(x + shape.x, y + shape.y) for x, y in ring]))
        path.closeSubpath()
    return path


class HandleItem(QGraphicsRectItem):
    """Vertex handle. Constant on-screen size (ignores the view zoom)."""

    def __init__(self, parent: ShapeItem, index: int) -> None:
        h = HANDLE_SIZE / 2
        super().__init__(-h, -h, HANDLE_SIZE, HANDLE_SIZE, parent)
        self.index = index
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.setPen(QPen(parent.color, 1.5))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(10)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        parent: ShapeItem = self.parentItem()
        p = parent.mapFromScene(event.scenePos())
        parent.preview_vertex(self.index, p.x(), p.y())

    def mouseReleaseEvent(self, event) -> None:
        parent: ShapeItem = self.parentItem()
        parent.commit_vertex(self.index)


class TagItem(QGraphicsItem):
    """Small label above a shape: class name, and source/score for suggestions."""

    def __init__(self, parent: QGraphicsItem, text: str, color: QColor, dashed: bool) -> None:
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.text, self.color, self.dashed = text, color, dashed
        from PySide6.QtGui import QFontMetricsF

        self.font = tag_font()
        fm = QFontMetricsF(self.font)
        self._rect = QRectF(0, -fm.height() - 4, fm.horizontalAdvance(text) + 8, fm.height() + 4)
        self.setZValue(5)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        bg = QColor(self.color)
        # Nearly opaque for suggestions too: a suggestion often sits on an existing label, and a
        # see-through tag mixed both tags' text. The tag text already says "제안".
        bg.setAlpha(235)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRect(self._rect)
        painter.setPen(QColor("white"))
        painter.setFont(self.font)
        painter.drawText(self._rect.adjusted(4, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, self.text)


class ShapeItem(QGraphicsPathItem):
    def __init__(self, shape: Shape, color: str, class_name: str, controller: ShapeController) -> None:
        super().__init__()
        self.shape_model = shape
        self.shape_id = shape.id
        self.color = QColor(color)
        self.controller = controller
        self.is_suggestion = shape.is_suggestion
        self._points = shape_points(shape)
        self._dragging_vertex = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(2 if self.is_suggestion else 1)
        self._apply_style(selected=False)
        self._update_path()

        tag = class_name
        if self.is_suggestion:
            score = "" if shape.score is None else f" {shape.score:.2f}"
            tag += f" · 제안({shape.source}{score})"
        self.tag = TagItem(self, tag, self.color, self.is_suggestion)
        x1, y1, _, _ = shape.bounds()
        self.tag.setPos(x1, y1)

        # A mask has no vertices to drag: select, move, class change, brush
        self.handles = [] if isinstance(shape, Mask) else [HandleItem(self, i) for i in range(len(self._points))]
        self._place_handles()
        self._show_handles(False)

    # ---- appearance ------------------------------------------------------------------
    def _apply_style(self, selected: bool) -> None:
        pen = QPen(self.color, 3 if selected else 2)
        pen.setCosmetic(True)  # constant width at any zoom
        fill = QColor(self.color)
        if self.is_suggestion:
            pen.setStyle(Qt.PenStyle.DashLine)
            c = QColor(self.color)
            c.setAlpha(200)
            pen.setColor(c)
            fill.setAlpha(25)
        else:
            fill.setAlpha(70 if selected else 40)
        self.setPen(pen)
        self.setBrush(QBrush(fill))

    def _update_path(self) -> None:
        if isinstance(self.shape_model, Mask):
            self.setPath(mask_path(self.shape_model))
            return
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in self._points]))
        path.closeSubpath()
        self.setPath(path)

    def _place_handles(self) -> None:
        for handle, (x, y) in zip(self.handles, self._points):
            handle.setPos(x, y)

    def _show_handles(self, visible: bool) -> None:
        for h in self.handles:
            h.setVisible(visible)

    def set_editable(self, editable: bool) -> None:
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
        self._editable = editable
        self._show_handles(editable and self.isSelected())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply_style(bool(value))
            self._show_handles(bool(value) and getattr(self, "_editable", False))
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Keep the shape inside the image while dragging.
            dx, dy = clamp_translation(self.shape_model, value.x(), value.y(),
                                       self.controller.image_width, self.controller.image_height)
            return QPointF(dx, dy)
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        # Skip Qt's dotted selection rectangle; selection is shown by pen width and handles.
        from PySide6.QtWidgets import QStyle

        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    # ---- move whole shape ------------------------------------------------------------------
    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self.pos() != QPointF(0, 0):
            dx, dy = self.pos().x(), self.pos().y()
            self.controller.shape_moved(self.shape_id, dx, dy)

    # ---- vertex edit -------------------------------------------------------------------
    def preview_vertex(self, index: int, x: float, y: float) -> None:
        x, y = clamp_point(x, y, self.controller.image_width, self.controller.image_height)
        if isinstance(self.shape_model, BBox):
            ox, oy = self._points[(index + 2) % 4]
            # Rebuild the rectangle with the opposite corner fixed; keep corner order stable.
            pts = [None] * 4
            pts[index], pts[(index + 2) % 4] = (x, y), (ox, oy)
            # Corners are TL, TR, BR, BL: for TL/BR the next corner shares the dragged y,
            # for TR/BL it shares the dragged x.
            pts[(index + 1) % 4] = (ox, y) if index % 2 == 0 else (x, oy)
            pts[(index + 3) % 4] = (x, oy) if index % 2 == 0 else (ox, y)
            self._points = pts
        else:
            self._points[index] = (x, y)
        self._dragging_vertex = True
        self._update_path()
        self._place_handles()

    def commit_vertex(self, index: int) -> None:
        if not self._dragging_vertex:
            return
        self._dragging_vertex = False
        x, y = self._points[index]
        self.controller.vertex_moved(self.shape_id, index, x, y)
