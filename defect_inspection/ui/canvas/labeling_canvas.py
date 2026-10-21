"""LabelingCanvas: the image viewer plus drawing/editing tools (PRD F2.1, F2.5, F2.8)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem, QWidget

from defect_inspection.core import label_edit
from defect_inspection.core.roi import RoiOptions, move_roi, resize_roi, roi_from_drag
from defect_inspection.core.sam_assist import SamPrompt
from defect_inspection.ui.canvas.image_canvas import ImageCanvas
from defect_inspection.ui.canvas.roi_tool import RoiItem, RoiSession
from defect_inspection.ui.canvas.session import LabelSession
from defect_inspection.ui.canvas.shape_items import ShapeItem
from defect_inspection.ui.canvas.tools import Tool

_DRAW_TOOLS = {Tool.BBOX, Tool.POLYGON}
_SAM_DRAG_PX = 5  # screen pixels: below this a SAM press is a click, above it a box


class LabelingCanvas(ImageCanvas):
    selection_changed = Signal(list)  # selected shape ids
    message = Signal(str)  # short user-facing hints for the status bar
    sam_prompt_changed = Signal(object)  # SamPrompt (empty when the user removed every prompt)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session: LabelSession | None = None
        self.tool = Tool.SELECT
        self.current_class_id = 0
        self.class_colors: list[str] = []
        self.class_names: list[str] = []
        self._items: dict[str, ShapeItem] = {}
        self._syncing_selection = False

        # in-progress drawing
        self._box_start: QPointF | None = None
        self._box_preview: QGraphicsRectItem | None = None
        self._poly_points: list[tuple[float, float]] = []
        self._poly_preview: QGraphicsPathItem | None = None
        self._poly_cursor: QPointF | None = None

        # SAM assist: prompts for the object being segmented, and the suggestion it produced
        self.sam_prompt = SamPrompt()
        self.sam_shape_id: str | None = None
        self._sam_press: tuple[QPointF, QPointF] | None = None  # (scene, viewport) at press
        self._sam_box_preview: QGraphicsRectItem | None = None
        self._sam_markers: list[QGraphicsItem] = []

        # Crop ROI layer (PRD F7.1): separate from labels, editable only with the ROI tool
        self.roi_session: RoiSession | None = None
        self.roi_options = RoiOptions()
        self._roi_items: dict[str, RoiItem] = {}
        self._roi_start: QPointF | None = None
        self._roi_preview: QGraphicsRectItem | None = None

        self.scene().selectionChanged.connect(self._on_scene_selection)

    # ---- ShapeController ---------------------------------------------------------------
    @property
    def image_width(self) -> int:
        return self.session.width if self.session else 0

    @property
    def image_height(self) -> int:
        return self.session.height if self.session else 0

    def shape_moved(self, shape_id: str, dx: float, dy: float) -> None:
        shape = self.session.shape(shape_id) if self.session else None
        if shape is not None:
            self.session.replace_shape(label_edit.translate(shape, dx, dy, self.image_width, self.image_height),
                                       self.tr("라벨 이동"))

    def vertex_moved(self, shape_id: str, index: int, x: float, y: float) -> None:
        shape = self.session.shape(shape_id) if self.session else None
        if shape is None:
            return
        try:
            new = label_edit.move_vertex(shape, index, x, y, self.image_width, self.image_height)
        except ValueError:
            self.message.emit(self.tr("크기가 0인 라벨은 만들 수 없습니다"))
            self.rebuild()  # drop the preview
            return
        self.session.replace_shape(new, self.tr("꼭짓점 이동"))

    # ---- session / classes -------------------------------------------------------------------
    def set_session(self, session: LabelSession | None) -> None:
        self.cancel_drawing()
        self.reset_sam()
        if self.session is not None:
            self.session.changed.disconnect(self.rebuild)
            self.session.select_requested.disconnect(self.select_ids)
        self.session = session
        if session is not None:
            session.changed.connect(self.rebuild)
            session.select_requested.connect(self.select_ids)
        self.rebuild()

    def set_classes(self, names: list[str], colors: list[str]) -> None:
        self.class_names, self.class_colors = names, colors
        self.rebuild()

    def _class_color(self, class_id: int) -> str:
        return self.class_colors[class_id] if class_id < len(self.class_colors) else "#FF00FF"

    def _class_name(self, class_id: int) -> str:
        return self.class_names[class_id] if class_id < len(self.class_names) else f"class {class_id}"

    # ---- items ---------------------------------------------------------------------------
    def rebuild(self) -> None:
        selected = self.selected_ids()
        self._syncing_selection = True
        for item in self._items.values():
            self.scene().removeItem(item)
        self._items.clear()
        if self.session is not None:
            for shape in self.session.shapes:
                item = ShapeItem(shape, self._class_color(shape.class_id), self._class_name(shape.class_id), self)
                self.scene().addItem(item)
                item.set_editable(self.tool == Tool.SELECT)
                self._items[shape.id] = item
        for sid in selected:
            if sid in self._items:
                self._items[sid].setSelected(True)
        self._syncing_selection = False
        self.selection_changed.emit(self.selected_ids())

    def items_by_id(self) -> dict[str, ShapeItem]:
        return dict(self._items)

    def selected_ids(self) -> list[str]:
        return [sid for sid, item in self._items.items() if item.isSelected()]

    def select_ids(self, ids: list[str]) -> None:
        self._syncing_selection = True
        for sid, item in self._items.items():
            item.setSelected(sid in ids)
        self._syncing_selection = False
        self.selection_changed.emit(self.selected_ids())

    def _on_scene_selection(self) -> None:
        if not self._syncing_selection:
            self.selection_changed.emit(self.selected_ids())

    # ---- Crop ROI layer -------------------------------------------------------------------------
    def set_roi_session(self, session: RoiSession | None) -> None:
        if self.roi_session is not None:
            self.roi_session.changed.disconnect(self.rebuild_rois)
        self.roi_session = session
        if session is not None:
            session.changed.connect(self.rebuild_rois)
        self.rebuild_rois()

    def set_roi_options(self, options: RoiOptions) -> None:
        self.roi_options = options
        self._update_roi_editability()

    def rebuild_rois(self) -> None:
        selected = set(self.selected_roi_ids())
        for item in self._roi_items.values():
            self.scene().removeItem(item)
        self._roi_items = {}
        if self.roi_session is not None:
            for roi, common in self.roi_session.displayed():
                item = RoiItem(roi, common, self)
                self.scene().addItem(item)
                self._roi_items[roi.id] = item
            self._update_roi_editability()
            for rid in selected & self._roi_items.keys():
                self._roi_items[rid].setSelected(True)

    def _update_roi_editability(self) -> None:
        editable = self.tool == Tool.CROP_ROI
        for item in self._roi_items.values():
            item.set_editable(editable, resizable=self.roi_options.fixed_size is None)

    def roi_items(self) -> dict[str, RoiItem]:
        return dict(self._roi_items)

    def selected_roi_ids(self) -> list[str]:
        return [rid for rid, item in self._roi_items.items() if item.isSelected()]

    def roi_moved(self, roi_id: str, dx: float, dy: float) -> None:
        found = self.roi_session.find(roi_id) if self.roi_session else None
        if found is not None:
            self.roi_session.replace(move_roi(found[0], dx, dy, self.image_width, self.image_height))

    def roi_resized(self, roi_id: str, corner: int, x: float, y: float) -> None:
        found = self.roi_session.find(roi_id) if self.roi_session else None
        if found is None:
            return
        try:
            new = resize_roi(found[0], corner, x, y, self.image_width, self.image_height, self.roi_options.square)
        except ValueError:
            self.message.emit(self.tr("ROI가 너무 작습니다"))
            self.rebuild_rois()
            return
        self.roi_session.replace(new)

    # ---- tools -------------------------------------------------------------------------------
    def set_tool(self, tool: Tool) -> None:
        self.cancel_drawing()
        if tool != Tool.SAM:
            self.reset_sam()
        self.tool = tool
        editable = tool == Tool.SELECT
        for item in self._items.values():
            item.set_editable(editable)
        self._update_roi_editability()
        if tool != Tool.CROP_ROI:
            for item in self._roi_items.values():
                item.setSelected(False)
        if tool in _DRAW_TOOLS or tool in (Tool.ERASER, Tool.SAM, Tool.CROP_ROI):
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().unsetCursor()

    @property
    def is_drawing(self) -> bool:
        return self._box_start is not None or bool(self._poly_points) or self._roi_start is not None

    def cancel_drawing(self) -> bool:
        was = self.is_drawing
        self._box_start = None
        self._poly_points = []
        self._poly_cursor = None
        self._roi_start = None
        for attr in ("_box_preview", "_poly_preview", "_roi_preview"):
            item = getattr(self, attr)
            if item is not None:
                self.scene().removeItem(item)
                setattr(self, attr, None)
        return was

    def finish_polygon(self) -> bool:
        """Close the polygon being drawn. Returns True if a polygon was in progress."""
        if not self._poly_points:
            return False
        points = list(self._poly_points)
        self.cancel_drawing()
        if self.session is None:
            return True
        poly = label_edit.polygon_from_clicks(self.current_class_id, points, self.image_width, self.image_height)
        if poly is None:
            self.message.emit(self.tr("폴리곤은 점이 3개 이상이어야 합니다"))
        else:
            self.session.add_shape(poly)
        return True

    def remove_last_point(self) -> bool:
        if not self._poly_points:
            return False
        self._poly_points.pop()
        if not self._poly_points:
            self.cancel_drawing()
        else:
            self._update_poly_preview()
        return True

    def _can_draw(self) -> bool:
        if self.session is None:
            return False
        if not self.class_names:
            self.message.emit(self.tr("먼저 클래스를 추가하세요 (클래스 목록의 + 버튼)"))
            return False
        return True

    def _scene_point(self, event: QMouseEvent) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _clamped(self, p: QPointF) -> QPointF:
        x, y = label_edit.clamp_point(p.x(), p.y(), self.image_width, self.image_height)
        return QPointF(x, y)

    def _preview_pen(self) -> QPen:
        pen = QPen(QColor(self._class_color(self.current_class_id)), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        return pen

    # ---- mouse -------------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.tool == Tool.SAM and not self._space_down and event.button() in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._sam_press_event(event)
            return
        if event.button() != Qt.MouseButton.LeftButton or self._space_down or self.tool == Tool.SELECT:
            super().mousePressEvent(event)
            return
        p = self._scene_point(event)
        if self.tool == Tool.CROP_ROI:
            self._roi_press(event, p)
            return
        if self.tool == Tool.ERASER:
            for item in self.scene().items(p):
                target = item if isinstance(item, ShapeItem) else item.parentItem()
                if isinstance(target, ShapeItem) and self.session is not None:
                    self.session.delete([target.shape_id])
                    break
            return
        if self.tool not in _DRAW_TOOLS or not self._can_draw():
            return
        if self.image_pos(event.position().toPoint()) is None and not self._poly_points:
            return  # start drawing only on the image
        p = self._clamped(p)
        if self.tool == Tool.BBOX:
            self._box_start = p
            self._box_preview = self.scene().addRect(QRectF(p, p), self._preview_pen())
            self._box_preview.setZValue(20)
        elif self.tool == Tool.POLYGON:
            self._poly_points.append((p.x(), p.y()))
            self._update_poly_preview()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if self._sam_press is not None:
            self._sam_drag_preview(event)
        elif self._roi_start is not None:
            self._roi_drag_preview(event)
        elif self._box_start is not None and self._box_preview is not None:
            self._box_preview.setRect(QRectF(self._box_start, self._clamped(self._scene_point(event))).normalized())
        elif self._poly_points:
            self._poly_cursor = self._clamped(self._scene_point(event))
            self._update_poly_preview()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._roi_start is not None:
            self._roi_release(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._sam_press is not None:
            self._sam_release_event(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._box_start is not None:
            start, end = self._box_start, self._clamped(self._scene_point(event))
            self.cancel_drawing()
            box = label_edit.bbox_from_drag(self.current_class_id, start.x(), start.y(), end.x(), end.y(),
                                            self.image_width, self.image_height)
            if box is not None and self.session is not None:
                self.session.add_shape(box)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.tool == Tool.POLYGON and self._poly_points:
            self.finish_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def _update_poly_preview(self) -> None:
        path = QPainterPath()
        pts = [QPointF(x, y) for x, y in self._poly_points]
        if self._poly_cursor is not None:
            pts.append(self._poly_cursor)
        if pts:
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
        if self._poly_preview is None:
            self._poly_preview = self.scene().addPath(path, self._preview_pen())
            self._poly_preview.setZValue(20)
        else:
            self._poly_preview.setPath(path)

    # ---- SAM assist ------------------------------------------------------------------------
    @property
    def sam_active(self) -> bool:
        return self.tool == Tool.SAM and (not self.sam_prompt.is_empty or self.sam_shape_id is not None)

    def reset_sam(self) -> None:
        """Forget the prompts; the next click starts a new object. The suggestion stays."""
        self.sam_prompt = SamPrompt()
        self.sam_shape_id = None
        self._sam_press = None
        self._clear_sam_graphics()

    def remove_last_sam_point(self) -> bool:
        if self.tool != Tool.SAM or self.sam_prompt.is_empty:
            return False
        if self.sam_prompt.points:
            self.sam_prompt.points.pop()
            self.sam_prompt.labels.pop()
        else:
            self.sam_prompt.box = None
        self._sam_changed()
        return True

    def _sam_press_event(self, event: QMouseEvent) -> None:
        if not self._can_draw() or self.image_pos(event.position().toPoint()) is None:
            return
        p = self._scene_point(event)
        if event.button() == Qt.MouseButton.RightButton:
            self.sam_prompt.points.append((p.x(), p.y()))
            self.sam_prompt.labels.append(0)  # exclude
            self._sam_changed()
        else:
            self._sam_press = (p, event.position())

    def _sam_drag_preview(self, event: QMouseEvent) -> None:
        start, start_vp = self._sam_press
        if (event.position() - start_vp).manhattanLength() < _SAM_DRAG_PX:
            return
        rect = QRectF(start, self._clamped(self._scene_point(event))).normalized()
        if self._sam_box_preview is None:
            self._sam_box_preview = self.scene().addRect(rect, self._preview_pen())
            self._sam_box_preview.setZValue(20)
        else:
            self._sam_box_preview.setRect(rect)

    def _sam_release_event(self, event: QMouseEvent) -> None:
        start, start_vp = self._sam_press
        self._sam_press = None
        if self._sam_box_preview is not None:
            self.scene().removeItem(self._sam_box_preview)
            self._sam_box_preview = None
        end = self._clamped(self._scene_point(event))
        if (event.position() - start_vp).manhattanLength() >= _SAM_DRAG_PX:
            r = QRectF(self._clamped(start), end).normalized()
            if r.width() < 2 or r.height() < 2:
                return
            self.sam_prompt.box = (r.left(), r.top(), r.right(), r.bottom())
        else:
            p = self._clamped(start)
            self.sam_prompt.points.append((p.x(), p.y()))
            self.sam_prompt.labels.append(1)  # include
        self._sam_changed()

    def _sam_changed(self) -> None:
        self._draw_sam_markers()
        self.sam_prompt_changed.emit(self.sam_prompt)

    def _clear_sam_graphics(self) -> None:
        for item in self._sam_markers:
            self.scene().removeItem(item)
        self._sam_markers = []
        if self._sam_box_preview is not None:
            self.scene().removeItem(self._sam_box_preview)
            self._sam_box_preview = None

    def _draw_sam_markers(self) -> None:
        self._clear_sam_graphics()
        for (x, y), label in zip(self.sam_prompt.points, self.sam_prompt.labels):
            dot = QGraphicsEllipseItem(-5, -5, 10, 10)
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            dot.setPos(x, y)
            dot.setBrush(QColor("#2ecc71") if label == 1 else QColor("#e74c3c"))
            dot.setPen(QPen(QColor("white"), 1.5))
            dot.setZValue(30)
            self.scene().addItem(dot)
            self._sam_markers.append(dot)
        if self.sam_prompt.box is not None:
            x1, y1, x2, y2 = self.sam_prompt.box
            pen = QPen(QColor("#f1c40f"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            box = self.scene().addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            box.setZValue(30)
            self._sam_markers.append(box)

    # ---- Crop ROI drawing ----------------------------------------------------------------------
    def _roi_under(self, p: QPointF) -> RoiItem | None:
        for item in self.scene().items(p):
            target = item if isinstance(item, RoiItem) else item.parentItem()
            if isinstance(target, RoiItem):
                return target
        return None

    def _roi_press(self, event: QMouseEvent, p: QPointF) -> None:
        if self.roi_session is None:
            return
        if self._roi_under(p) is not None:
            super().mousePressEvent(event)  # select / move / resize an existing ROI
            return
        if self.image_pos(event.position().toPoint()) is None:
            return
        for item in self._roi_items.values():
            item.setSelected(False)
        self._roi_start = self._clamped(p)
        pen = QPen(QColor("#6250d6"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._roi_preview = self.scene().addRect(QRectF(self._roi_start, self._roi_start), pen)
        self._roi_preview.setZValue(20)
        self._roi_drag_preview(event)

    def _roi_candidate(self, event: QMouseEvent):
        end = self._clamped(self._scene_point(event))
        s = self._roi_start
        return roi_from_drag(s.x(), s.y(), end.x(), end.y(), self.image_width, self.image_height, self.roi_options)

    def _roi_drag_preview(self, event: QMouseEvent) -> None:
        roi = self._roi_candidate(event)
        if self._roi_preview is not None:
            self._roi_preview.setRect(QRectF() if roi is None else QRectF(roi.x1, roi.y1, roi.width, roi.height))

    def _roi_release(self, event: QMouseEvent) -> None:
        roi = self._roi_candidate(event)
        self.cancel_drawing()
        if roi is None:
            self.message.emit(self.tr("ROI가 너무 작습니다. 드래그하거나 고정 크기를 고르세요"))
        elif self.roi_session is not None:
            self.roi_session.add(roi)
