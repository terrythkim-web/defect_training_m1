"""Image list (PRD F1.3): thumbnails, label status, filter and next/previous navigation."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from defect_inspection.core.project import LabelStatus, Project
from defect_inspection.ui.thumbnails import THUMB_SIZE, ThumbnailLoader

PATH_ROLE = Qt.ItemDataRole.UserRole
STATUS_ROLE = Qt.ItemDataRole.UserRole + 1

# (key, label) in combo order
FILTERS = [
    ("all", "전체"),
    ("unlabeled", "미라벨"),
    ("labeled", "라벨됨"),
    ("suggested", "제안 대기"),
    ("ok", "OK 이미지"),
]


def matches_filter(status: LabelStatus, key: str) -> bool:
    return {
        "all": True,
        "unlabeled": not status.is_labeled,
        "labeled": status.is_labeled,
        "suggested": status.has_suggestions,
        "ok": status.is_ok,
    }[key]


def status_text(status: LabelStatus) -> str:
    if status.is_ok:
        main = "OK 이미지"
    elif status.confirmed:
        main = f"라벨 {status.confirmed}"
    else:
        main = "미라벨"
    if status.suggested:
        main += f" · 제안 {status.suggested}"
    return main


class ImageListPanel(QWidget):
    current_changed = Signal(object)  # image path relative to images/, or None

    def __init__(self, thumbnails: ThumbnailLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self.thumbnails = thumbnails
        self.thumbnails.ready.connect(self._on_thumbnail)
        self._items_by_abspath: dict[str, QListWidgetItem] = {}

        self.filter_combo = QComboBox()
        for key, label in FILTERS:
            self.filter_combo.addItem(self.tr(label), key)
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)

        self.list = QListWidget()
        self.list.setObjectName("image_list")
        self.list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.setUniformItemSizes(True)
        self.list.currentItemChanged.connect(self._on_current_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.filter_combo)
        layout.addWidget(self.list)

        self._placeholder_icon = _placeholder_icon()

    # ---- content -------------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.list.blockSignals(True)
        self.list.clear()
        self._items_by_abspath.clear()
        if project is not None:
            for entry in project.config.images:
                item = QListWidgetItem(self._placeholder_icon, "")
                item.setData(PATH_ROLE, entry.path)
                self.list.addItem(item)
                self._set_status(item, project.label_status(entry))
                abspath = str(project.image_abspath(entry))
                self._items_by_abspath[abspath] = item
                self.thumbnails.request(abspath)
        self.list.blockSignals(False)
        self.apply_filter()
        if self.list.currentItem() is None:
            self.current_changed.emit(None)

    def refresh_status(self, path: str) -> None:
        """Re-read the label status of one image, e.g. after its labels were saved."""
        if self.project is None:
            return
        item = self._item(path)
        if item is not None:
            self._set_status(item, self.project.label_status(path))
            item.setHidden(not matches_filter(item.data(STATUS_ROLE), self.filter_key))

    def set_status(self, path: str, status: LabelStatus) -> None:
        """Show an in-memory status (unsaved edits) without reading the label file."""
        item = self._item(path)
        if item is not None:
            self._set_status(item, status)

    def _set_status(self, item: QListWidgetItem, status: LabelStatus) -> None:
        item.setData(STATUS_ROLE, status)
        item.setText(f"{item.data(PATH_ROLE)}\n{self.tr(status_text(status))}")
        # Labeled: clear the role so the palette's text colour applies. setForeground(QColor())
        # painted an invalid (= black) brush, unreadable on the dark-mode list background.
        item.setData(Qt.ItemDataRole.ForegroundRole, None if status.is_labeled else QColor("gray"))

    def _on_thumbnail(self, abspath: str, image: QImage) -> None:
        item = self._items_by_abspath.get(abspath)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(image)))

    # ---- filter / selection ----------------------------------------------------------
    @property
    def filter_key(self) -> str:
        return self.filter_combo.currentData()

    def set_filter(self, key: str) -> None:
        self.filter_combo.setCurrentIndex(self.filter_combo.findData(key))

    def apply_filter(self) -> None:
        key = self.filter_key
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(not matches_filter(item.data(STATUS_ROLE), key))
        current = self.list.currentItem()
        if current is None or current.isHidden():
            visible = self.visible_rows()
            self.list.setCurrentRow(visible[0] if visible else -1)

    def visible_rows(self) -> list[int]:
        return [i for i in range(self.list.count()) if not self.list.item(i).isHidden()]

    def current_path(self) -> str | None:
        item = self.list.currentItem()
        return None if item is None else item.data(PATH_ROLE)

    def select(self, path: str) -> None:
        item = self._item(path)
        if item is not None:
            self.list.setCurrentItem(item)

    def step(self, delta: int) -> None:
        """Move to the next (delta=1) or previous (delta=-1) visible image; stops at the ends."""
        visible = self.visible_rows()
        if not visible:
            return
        row = self.list.currentRow()
        if row not in visible:
            target = visible[0]
        else:
            idx = min(max(visible.index(row) + delta, 0), len(visible) - 1)
            target = visible[idx]
        self.list.setCurrentRow(target)

    def neighbours(self, radius: int = 2) -> list[str]:
        """Paths of visible images around the current one, nearest first (for prefetch)."""
        visible = self.visible_rows()
        row = self.list.currentRow()
        if row not in visible:
            return []
        idx = visible.index(row)
        out = []
        for d in range(1, radius + 1):
            for j in (idx + d, idx - d):
                if 0 <= j < len(visible):
                    out.append(self.list.item(visible[j]).data(PATH_ROLE))
        return out

    def _item(self, path: str) -> QListWidgetItem | None:
        for i in range(self.list.count()):
            if self.list.item(i).data(PATH_ROLE) == path:
                return self.list.item(i)
        return None

    def _on_current_item(self, current: QListWidgetItem | None, _previous) -> None:
        self.current_changed.emit(None if current is None else current.data(PATH_ROLE))


def _placeholder_icon() -> QIcon:
    pix = QPixmap(THUMB_SIZE, THUMB_SIZE)
    pix.fill(QColor("#cccccc"))
    return QIcon(pix)
