"""Crop ROI tool (PRD F7.1): undoable ROI session, dashed ROI items, options bar.

ROIs are a separate layer under the labels. They are only editable with the Crop ROI tool (C);
in every other tool clicks pass through them to the labels.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsItem,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOptionGraphicsItem,
    QWidget,
)

from defect_inspection.core.roi import FIXED_SIZES, Roi, RoiOptions, RoiStore, clip_roi
from defect_inspection.ui.canvas.shape_items import HandleItem, TagItem

IMAGE_ROI_COLOR = "#6250d6"  # violet: this image's ROIs
COMMON_ROI_COLOR = "#1baf7a"  # aqua: common ROIs (every image)
ROI_Z = 0.5  # below labels (1-2), above the image


class _RoiCommand(QUndoCommand):
    def __init__(self, session: RoiSession, before: tuple, after: tuple, text: str) -> None:
        super().__init__(text)
        self._s, self._before, self._after = session, before, after

    def redo(self) -> None:
        self._s._apply(*self._after)

    def undo(self) -> None:
        self._s._apply(*self._before)


class RoiSession(QObject):
    """ROIs of the open image + the common ROIs; every change is undoable and saved at once."""

    changed = Signal()

    def __init__(self, store: RoiStore, image_path: str, size: tuple[int, int], undo_stack: QUndoStack,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.store, self.image_path = store, image_path
        self.width, self.height = size
        self.undo_stack = undo_stack

    @property
    def image_rois(self) -> list[Roi]:
        return self.store.image_rois(self.image_path)

    @property
    def common(self) -> list[Roi]:
        return self.store.common

    def displayed(self) -> list[tuple[Roi, bool]]:
        """(roi, is_common) as shown on this image; common ROIs clipped to it."""
        out = [(r, True) for r in (clip_roi(c, self.width, self.height) for c in self.common) if r is not None]
        return out + [(r, False) for r in self.image_rois]

    def find(self, roi_id: str) -> tuple[Roi, bool] | None:
        for r in self.common:
            if r.id == roi_id:
                return r, True
        for r in self.image_rois:
            if r.id == roi_id:
                return r, False
        return None

    # ---- edits ----------------------------------------------------------------------------
    def add(self, roi: Roi) -> None:
        """A newly drawn ROI replaces this image's earlier ones (one ROI per image; common ROIs
        stay). Undo brings the earlier ROIs back."""
        text = self.tr("ROI 다시 지정") if self.image_rois else self.tr("ROI 추가")
        self._push([roi], self.common, text)

    def replace(self, roi: Roi) -> None:
        image = [roi if r.id == roi.id else r for r in self.image_rois]
        common = [roi if r.id == roi.id else r for r in self.common]
        self._push(image, common, self.tr("ROI 수정"))

    def delete(self, ids: list[str]) -> None:
        image = [r for r in self.image_rois if r.id not in ids]
        common = [r for r in self.common if r.id not in ids]
        if len(image) + len(common) != len(self.image_rois) + len(self.common):
            self._push(image, common, self.tr("ROI 삭제"))

    def make_common(self) -> int:
        """Move this image's ROIs to the common list (fixed jig: same ROI on every image)."""
        moved = self.image_rois
        if moved:
            self._push([], self.common + moved, self.tr("공통 ROI로 지정"))
        return len(moved)

    def clear_common(self) -> int:
        n = len(self.common)
        if n:
            self._push(self.image_rois, [], self.tr("공통 ROI 삭제"))
        return n

    def _push(self, image: list[Roi], common: list[Roi], text: str) -> None:
        before = (self.image_rois, self.common)
        self.undo_stack.push(_RoiCommand(self, before, (list(image), list(common)), text))

    def _apply(self, image: list[Roi], common: list[Roi]) -> None:
        self.store.set_rois(self.image_path, list(image), list(common))
        self.changed.emit()


class RoiItem(QGraphicsRectItem):
    """Dashed ROI rectangle. Same handle protocol as ShapeItem (preview_vertex / commit_vertex)."""

    def __init__(self, roi: Roi, is_common: bool, controller) -> None:
        super().__init__(roi.x1, roi.y1, roi.width, roi.height)
        self.roi, self.roi_id, self.is_common, self.controller = roi, roi.id, is_common, controller
        self.color = QColor(COMMON_ROI_COLOR if is_common else IMAGE_ROI_COLOR)
        self._corners = [(roi.x1, roi.y1), (roi.x2, roi.y1), (roi.x2, roi.y2), (roi.x1, roi.y2)]
        self._dragging_vertex = False
        self._editable = False
        self.setZValue(ROI_Z)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self._style(False)
        label = ("공통 ROI" if is_common else "ROI") + f" {roi.width:.0f}×{roi.height:.0f}"  # graphics item: no tr()
        self.tag = TagItem(self, label, self.color, dashed=True)
        self.tag.setPos(roi.x1, roi.y1)
        self.handles = [HandleItem(self, i) for i in range(4)]
        for h, (x, y) in zip(self.handles, self._corners):
            h.setPos(x, y)
            h.setVisible(False)

    def _style(self, selected: bool) -> None:
        pen = QPen(self.color, 3 if selected else 2, Qt.PenStyle.DashDotLine if self.is_common else Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(self.color)
        fill.setAlpha(25 if selected else 0)
        self.setBrush(QBrush(fill))

    def set_editable(self, editable: bool, resizable: bool) -> None:
        self._editable, self._resizable = editable, resizable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton)
        self._show_handles(self.isSelected())

    def _show_handles(self, selected: bool) -> None:
        for h in self.handles:
            h.setVisible(selected and self._editable and getattr(self, "_resizable", True))

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        option.state &= ~QStyle.StateFlag.State_Selected  # no Qt dotted box; pen + handles show it
        super().paint(painter, option, widget)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._style(bool(value))
            self._show_handles(bool(value))
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            r = self.roi
            dx = min(max(value.x(), -r.x1), self.controller.image_width - r.x2)
            dy = min(max(value.y(), -r.y1), self.controller.image_height - r.y2)
            return QPointF(dx, dy)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self.pos() != QPointF(0, 0):
            self.controller.roi_moved(self.roi_id, self.pos().x(), self.pos().y())

    # handle protocol
    def preview_vertex(self, index: int, x: float, y: float) -> None:
        ox, oy = self._corners[(index + 2) % 4]
        x = min(max(x, 0), self.controller.image_width)
        y = min(max(y, 0), self.controller.image_height)
        self.setRect(min(x, ox), min(y, oy), abs(x - ox), abs(y - oy))
        self._preview_corner = (x, y)
        self._dragging_vertex = True

    def commit_vertex(self, index: int) -> None:
        if self._dragging_vertex:
            self._dragging_vertex = False
            self.controller.roi_resized(self.roi_id, index, *self._preview_corner)


class RoiBar(QWidget):
    """Options shown while the Crop ROI tool is active."""

    options_changed = Signal(object)  # RoiOptions
    make_common_requested = Signal()
    clear_common_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.size_combo = QComboBox()
        self.size_combo.addItem(self.tr("자유 크기 (드래그)"), None)
        for s in (*FIXED_SIZES, 1280):
            self.size_combo.addItem(self.tr("{0}×{0} (클릭)").format(s), s)
        self.square = QCheckBox(self.tr("정사각"))
        self.btn_common = QPushButton(self.tr("모든 이미지에 적용"))
        self.btn_common.setToolTip(self.tr("이 이미지의 ROI를 공통 ROI로 바꿔 모든 이미지에 같은 위치로 적용합니다 (고정 지그)"))
        self.btn_clear_common = QPushButton(self.tr("공통 ROI 삭제"))
        self.count_label = QLabel()
        self.size_combo.currentIndexChanged.connect(self._emit)
        self.square.toggled.connect(self._emit)
        self.btn_common.clicked.connect(self.make_common_requested.emit)
        self.btn_clear_common.clicked.connect(self.clear_common_requested.emit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel(self.tr("<b>Crop ROI</b>")))
        for w in (self.size_combo, self.square, self.btn_common, self.btn_clear_common):
            layout.addWidget(w)
        layout.addStretch(1)
        layout.addWidget(self.count_label)

    def options(self) -> RoiOptions:
        return RoiOptions(fixed_size=self.size_combo.currentData(), square=self.square.isChecked())

    def set_options(self, options: RoiOptions) -> None:
        self.blockSignals(True)
        idx = self.size_combo.findData(options.fixed_size)
        if idx < 0 and options.fixed_size:
            self.size_combo.addItem(self.tr("{0}×{0} (클릭)").format(options.fixed_size), options.fixed_size)
            idx = self.size_combo.count() - 1
        self.size_combo.setCurrentIndex(max(idx, 0))
        self.square.setChecked(options.square)
        self.square.setEnabled(options.fixed_size is None)
        self.blockSignals(False)

    def set_counts(self, image: int, common: int) -> None:
        self.count_label.setText(self.tr("이 이미지 ROI {0} · 공통 {1}").format(image, common))
        self.btn_common.setEnabled(image > 0)
        self.btn_clear_common.setEnabled(common > 0)

    def _emit(self) -> None:
        self.square.setEnabled(self.size_combo.currentData() is None)
        self.options_changed.emit(self.options())
