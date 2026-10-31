"""Label list for the current image, synced with the canvas selection, plus the suggestion bar
(accept / reject / accept all above a score, PRD F2.8)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.annotation import BBox, Mask
from defect_inspection.ui.canvas.session import LabelSession

ID_ROLE = Qt.ItemDataRole.UserRole


def describe(shape, class_name: str) -> str:
    if isinstance(shape, Mask):
        kind = f"Mask {shape.area} px"
    else:
        kind = "BBox" if isinstance(shape, BBox) else f"Polygon {len(shape.points)}점"
    text = f"{class_name} · {kind}"
    if shape.is_suggestion:
        score = "" if shape.score is None else f" {shape.score:.2f}"
        text += f"  [제안 {shape.source}{score}]"
    return text


class LabelListPanel(QWidget):
    selection_changed = Signal(list)  # shape ids selected in the list

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session: LabelSession | None = None
        self.class_names: list[str] = []
        self.class_colors: list[str] = []
        self._syncing = False

        self.list = QListWidget()
        self.list.setObjectName("label_list")
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._on_list_selection)

        self.suggestion_label = QLabel()
        self.btn_accept = QPushButton(self.tr("수락 (Enter)"))
        self.btn_reject = QPushButton(self.tr("거절 (Backspace)"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        self.threshold.setPrefix("score ≥ ")
        self.btn_accept_above = QPushButton(self.tr("모두 수락"))
        self.btn_reject_all = QPushButton(self.tr("이 이미지 제안 모두 거절"))

        self.btn_accept.clicked.connect(self.accept_selected)
        self.btn_reject.clicked.connect(self.reject_selected)
        self.btn_accept_above.clicked.connect(lambda: self.session and self.session.accept_above(self.threshold.value()))
        self.btn_reject_all.clicked.connect(lambda: self.session and self.session.reject_all())

        row1 = QHBoxLayout()
        row1.addWidget(self.btn_accept)
        row1.addWidget(self.btn_reject)
        row2 = QHBoxLayout()
        row2.addWidget(self.threshold)
        row2.addWidget(self.btn_accept_above)

        self.suggestion_bar = QWidget()
        bar = QVBoxLayout(self.suggestion_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.suggestion_label)
        bar.addLayout(row1)
        bar.addLayout(row2)
        bar.addWidget(self.btn_reject_all)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addWidget(self.suggestion_bar)
        self._refresh()

    # ---- data ------------------------------------------------------------------------------
    def set_session(self, session: LabelSession | None) -> None:
        if self.session is not None:
            self.session.changed.disconnect(self._refresh)
        self.session = session
        if session is not None:
            session.changed.connect(self._refresh)
        self._refresh()

    def set_classes(self, names: list[str], colors: list[str]) -> None:
        self.class_names, self.class_colors = names, colors
        self._refresh()

    def _refresh(self) -> None:
        selected = set(self.selected_ids())
        self._syncing = True
        self.list.clear()
        shapes = self.session.shapes if self.session else []
        # Confirmed labels first, then suggestions (highest score first).
        ordered = [s for s in shapes if not s.is_suggestion] + sorted(
            (s for s in shapes if s.is_suggestion), key=lambda s: -(s.score or 0))
        for shape in ordered:
            name = self.class_names[shape.class_id] if shape.class_id < len(self.class_names) else f"class {shape.class_id}"
            color = self.class_colors[shape.class_id] if shape.class_id < len(self.class_colors) else "#FF00FF"
            item = QListWidgetItem(_swatch(color), describe(shape, name))
            item.setData(ID_ROLE, shape.id)
            if shape.is_suggestion:
                font = QFont()
                font.setItalic(True)
                item.setFont(font)
                item.setForeground(QColor("gray"))
            self.list.addItem(item)
            item.setSelected(shape.id in selected)
        self._syncing = False

        n = len([s for s in shapes if s.is_suggestion])
        self.suggestion_label.setText(self.tr("제안 {0}개").format(n) if n else self.tr("제안 없음"))
        for w in (self.btn_accept, self.btn_reject, self.threshold, self.btn_accept_above, self.btn_reject_all):
            w.setEnabled(n > 0)

    # ---- selection --------------------------------------------------------------------------
    def selected_ids(self) -> list[str]:
        return [item.data(ID_ROLE) for item in self.list.selectedItems()]

    def select_ids(self, ids: list[str]) -> None:
        """Mirror the canvas selection without echoing it back."""
        self._syncing = True
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setSelected(item.data(ID_ROLE) in ids)
        if ids:
            for i in range(self.list.count()):
                if self.list.item(i).data(ID_ROLE) in ids:
                    self.list.scrollToItem(self.list.item(i))
                    break
        self._syncing = False

    def _on_list_selection(self) -> None:
        if not self._syncing:
            self.selection_changed.emit(self.selected_ids())

    # ---- suggestion actions --------------------------------------------------------------------
    def accept_selected(self) -> int:
        return self.session.accept(self.selected_ids()) if self.session else 0

    def reject_selected(self) -> int:
        return self.session.reject(self.selected_ids()) if self.session else 0


def _swatch(color: str) -> QIcon:
    pix = QPixmap(12, 12)
    pix.fill(QColor(color))
    return QIcon(pix)
