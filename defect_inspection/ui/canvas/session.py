"""LabelSession: the labels of the image being edited. Every change is a snapshot command on the
shared QUndoStack, so Undo/Redo covers drawing, editing, class changes and suggestion
accept/reject alike."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon
from defect_inspection.core.project import Project

Shape = BBox | Polygon


def _copy(shapes: list[Shape]) -> list[Shape]:
    return [s.model_copy(deep=True) for s in shapes]


class _ShapesCommand(QUndoCommand):
    def __init__(self, session: LabelSession, before: list[Shape], after: list[Shape], text: str,
                 select: list[str]) -> None:
        super().__init__(text)
        self._session, self._before, self._after, self._select = session, before, after, select

    def redo(self) -> None:
        self._session._set_shapes(_copy(self._after), self._select)

    def undo(self) -> None:
        self._session._set_shapes(_copy(self._before), [])


class LabelSession(QObject):
    changed = Signal()  # shapes changed; views rebuild
    select_requested = Signal(list)  # shape ids to select after a change

    def __init__(self, project: Project, image_path: str, undo_stack: QUndoStack,
                 size: tuple[int, int] | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.image_path = image_path
        self.undo_stack = undo_stack
        self.annotations: ImageAnnotations = project.load_labels(image_path, size)
        self.dirty = False

    # ---- read ----------------------------------------------------------------------
    @property
    def width(self) -> int:
        return self.annotations.width

    @property
    def height(self) -> int:
        return self.annotations.height

    @property
    def shapes(self) -> list[Shape]:
        return self.annotations.shapes

    def shape(self, shape_id: str) -> Shape | None:
        return next((s for s in self.shapes if s.id == shape_id), None)

    # ---- edits (each one undoable) ---------------------------------------------------
    def add_shape(self, shape: Shape) -> None:
        self._commit(self.shapes + [shape], self.tr("라벨 추가"), select=[shape.id])

    def replace_shape(self, new: Shape, text: str | None = None) -> None:
        """Swap in an edited version of an existing shape (same id)."""
        shapes = [new if s.id == new.id else s for s in self.shapes]
        self._commit(shapes, text or self.tr("라벨 수정"), select=[new.id])

    def set_shapes(self, shapes: list[Shape], text: str, select: list[str] | None = None) -> None:
        """Replace every shape at once (a brush stroke may grow, split, join and delete labels): one undo step."""
        self._commit(list(shapes), text, select=select)

    def delete(self, ids: list[str]) -> None:
        shapes = [s for s in self.shapes if s.id not in ids]
        if len(shapes) != len(self.shapes):
            self._commit(shapes, self.tr("라벨 삭제"))

    def set_class(self, ids: list[str], class_id: int) -> None:
        if not any(s.id in ids and s.class_id != class_id for s in self.shapes):
            return
        shapes = [s.model_copy(update={"class_id": class_id}) if s.id in ids else s for s in self.shapes]
        self._commit(shapes, self.tr("클래스 변경"), select=list(ids))

    def accept(self, ids: list[str]) -> int:
        return self._suggestion_edit(lambda a: a.accept(ids), self.tr("제안 수락"), select=list(ids))

    def accept_above(self, min_score: float) -> int:
        return self._suggestion_edit(lambda a: a.accept_above(min_score), self.tr("제안 일괄 수락"))

    def reject(self, ids: list[str]) -> int:
        return self._suggestion_edit(lambda a: a.reject(ids), self.tr("제안 거절"))

    def reject_all(self) -> int:
        return self._suggestion_edit(lambda a: a.clear_suggestions(), self.tr("제안 모두 거절"))

    def _suggestion_edit(self, op, text: str, select: list[str] | None = None) -> int:
        trial = self.annotations.model_copy(update={"shapes": _copy(self.shapes)})
        count = op(trial)
        if count:
            self._commit(trial.shapes, text, select=select or [])
        return count

    def _commit(self, after: list[Shape], text: str, select: list[str] | None = None) -> None:
        self.undo_stack.push(_ShapesCommand(self, _copy(self.shapes), _copy(after), text, select or []))

    def _set_shapes(self, shapes: list[Shape], select: list[str]) -> None:
        self.annotations.shapes = shapes
        self.dirty = True
        self.changed.emit()
        self.select_requested.emit(select)

    # ---- persistence -------------------------------------------------------------------
    def save(self) -> bool:
        """Write JSON + YOLO txt if anything changed. Returns True if it wrote."""
        if not self.dirty:
            return False
        self.project.save_labels(self.annotations)
        self.dirty = False
        return True
