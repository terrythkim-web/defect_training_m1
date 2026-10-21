"""Marking tools in the real window: drawing, editing, classes, suggestions, undo (PROMPTS step 6)."""

import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QUndoStack
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.canvas.session import LabelSession
from defect_inspection.ui.canvas.shape_items import HandleItem
from defect_inspection.ui.canvas.tools import Tool
from defect_inspection.ui.main_window import MainWindow

IMG_W, IMG_H = 400, 300


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "원본"
    src.mkdir()
    for name in ["a.png", "b.png"]:
        imwrite_unicode(src / name, np.full((IMG_H, IMG_W, 3), 120, np.uint8))
    p = Project.create(tmp_path / "라벨링 프로젝트")
    p.add_class("스크래치")
    p.add_class("찍힘")
    p.add_images([src])
    p.save()
    return p


@pytest.fixture
def window(qapp, project, tmp_path):
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1400, 900)
    w.show()
    w.load_project(project)
    assert wait_until(qapp, lambda: w.session is not None)
    w.canvas.zoom_to(1.0)
    w.canvas.centerOn(IMG_W / 2, IMG_H / 2)
    qapp.processEvents()
    yield w
    w.project_dirty = False
    w.close()


def vp(w, x, y) -> QPoint:
    """Viewport position of an image pixel coordinate."""
    return w.canvas.mapFromScene(QPointF(x, y))


def drag(w, start, end, button=Qt.MouseButton.LeftButton):
    view = w.canvas.viewport()
    QTest.mousePress(view, button, Qt.KeyboardModifier.NoModifier, vp(w, *start))
    for t in (0.33, 0.66, 1.0):
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        QTest.mouseMove(view, vp(w, x, y))
    QTest.mouseRelease(view, button, Qt.KeyboardModifier.NoModifier, vp(w, *end))


def click(w, x, y):
    QTest.mouseClick(w.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, x, y))


def shapes(w):
    return w.session.shapes


def key(w, k):
    QTest.keyClick(w, k)


# ---- LabelSession (no mouse) -------------------------------------------------------

def test_session_edits_are_undoable(project, qapp):
    stack = QUndoStack()
    s = LabelSession(project, "a.png", stack)
    box = BBox(class_id=0, x1=1, y1=1, x2=10, y2=10)
    s.add_shape(box)
    s.set_class([box.id], 1)
    s.delete([box.id])
    assert s.shapes == []
    stack.undo()
    assert s.shapes[0].class_id == 1
    stack.undo()
    assert s.shapes[0].class_id == 0
    stack.undo()
    assert s.shapes == []
    stack.redo()
    assert len(s.shapes) == 1


def test_session_suggestion_ops_undoable(project, qapp):
    stack = QUndoStack()
    s = LabelSession(project, "a.png", stack)
    s.annotations.add_suggestions([BBox(class_id=0, x1=1, y1=1, x2=9, y2=9, score=0.9),
                                   BBox(class_id=0, x1=20, y1=1, x2=29, y2=9, score=0.3)], source="model")
    assert s.accept_above(0.5) == 1
    assert s.reject_all() == 1
    assert len(s.annotations.confirmed()) == 1 and s.annotations.suggestions() == []
    stack.undo()
    stack.undo()
    assert len(s.annotations.suggestions()) == 2
    assert s.accept_above(0.95) == 0  # no-op pushes nothing
    assert stack.count() == 2


def test_session_save_only_when_dirty(project, qapp):
    s = LabelSession(project, "a.png", QUndoStack())
    assert not s.save()
    s.add_shape(BBox(class_id=0, x1=1, y1=1, x2=10, y2=10))
    assert s.save()
    assert project.label_status("a.png").confirmed == 1


# ---- drawing -----------------------------------------------------------------------

def test_draw_bbox_with_current_class(window):
    window.class_list.setCurrentRow(1)
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 60), (150, 160))

    assert len(shapes(window)) == 1
    box = shapes(window)[0]
    assert isinstance(box, BBox) and box.class_id == 1
    assert box.bounds() == pytest.approx((50, 60, 150, 160), abs=1.5)


def test_bbox_drawn_off_image_is_clamped(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (350, 250), (450, 350))
    assert shapes(window)[0].bounds() == pytest.approx((350, 250, IMG_W, IMG_H), abs=1.5)


def test_tiny_drag_makes_no_box(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (50.5, 50.5))
    assert shapes(window) == []


def test_draw_polygon_with_double_click(window):
    window.tool_actions[Tool.POLYGON].trigger()
    for x, y in [(100, 100), (200, 100), (150, 200)]:
        click(window, x, y)
    QTest.mouseDClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      vp(window, 150, 200))

    poly = shapes(window)[0]
    assert isinstance(poly, Polygon)
    assert np.allclose(poly.points, [(100, 100), (200, 100), (150, 200)], atol=1.5)


def test_polygon_enter_finishes_backspace_removes_escape_cancels(window):
    window.tool_actions[Tool.POLYGON].trigger()
    for x, y in [(10, 10), (100, 10), (100, 100), (10, 100)]:
        click(window, x, y)
    window.act_backspace.trigger()  # drop (10, 100)
    window.act_enter.trigger()
    assert len(shapes(window)[0].points) == 3

    click(window, 200, 200)
    click(window, 250, 200)
    window.act_escape.trigger()
    assert not window.canvas.is_drawing
    assert len(shapes(window)) == 1


def test_polygon_needs_three_points(window):
    window.tool_actions[Tool.POLYGON].trigger()
    click(window, 10, 10)
    click(window, 50, 50)
    window.act_enter.trigger()
    assert shapes(window) == []


def test_drawing_needs_a_class(window, qapp):
    window.project.config.classes.clear()
    window._refresh_classes()
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    assert shapes(window) == []
    assert "클래스" in window.statusBar().currentMessage()


def test_add_class_button(window):
    window.add_class("이물")
    assert window.class_list.count() == 3
    assert window.canvas.current_class_id == 2
    assert window.project_dirty


# ---- select / move / vertex / delete -------------------------------------------------

@pytest.fixture
def boxed(window):
    window.session.add_shape(BBox(class_id=0, x1=100, y1=100, x2=200, y2=180))
    window.tool_actions[Tool.SELECT].trigger()
    window.canvas.select_ids([])
    return window


def test_click_selects_and_syncs_label_list(boxed):
    click(boxed, 150, 140)
    sid = shapes(boxed)[0].id
    assert boxed.canvas.selected_ids() == [sid]
    assert boxed.label_panel.selected_ids() == [sid]


def test_label_list_selection_selects_on_canvas(boxed):
    boxed.label_panel.list.item(0).setSelected(True)
    assert boxed.canvas.selected_ids() == [shapes(boxed)[0].id]


def test_drag_moves_shape(boxed):
    click(boxed, 150, 140)
    drag(boxed, (150, 140), (190, 160))
    assert shapes(boxed)[0].bounds() == pytest.approx((140, 120, 240, 200), abs=1.5)


def test_move_is_clamped_to_image(boxed):
    click(boxed, 150, 140)
    drag(boxed, (150, 140), (380, 290))
    x1, y1, x2, y2 = shapes(boxed)[0].bounds()
    assert x2 == pytest.approx(IMG_W) and y2 == pytest.approx(IMG_H)


def test_drag_corner_handle_resizes(boxed, qapp):
    click(boxed, 150, 140)
    item = boxed.canvas.items_by_id()[shapes(boxed)[0].id]
    handle = next(h for h in item.handles if h.index == 2)  # bottom-right
    assert handle.isVisible()
    drag(boxed, (200, 180), (260, 220))
    assert shapes(boxed)[0].bounds() == pytest.approx((100, 100, 260, 220), abs=1.5)


def test_drag_polygon_vertex(window):
    window.session.add_shape(Polygon(class_id=0, points=[(100, 100), (200, 100), (150, 200)]))
    window.tool_actions[Tool.SELECT].trigger()
    click(window, 150, 130)
    drag(window, (150, 200), (150, 250))
    assert np.allclose(shapes(window)[0].points[2], (150, 250), atol=1.5)


def test_delete_key_and_eraser(boxed):
    click(boxed, 150, 140)
    boxed.act_delete.trigger()
    assert shapes(boxed) == []

    boxed.undo_stack.undo()
    boxed.tool_actions[Tool.ERASER].trigger()
    click(boxed, 150, 140)
    assert shapes(boxed) == []


def test_number_key_sets_class_of_selection(boxed):
    click(boxed, 150, 140)
    boxed.class_key_actions[1].trigger()
    assert shapes(boxed)[0].class_id == 1
    assert boxed.class_list.currentRow() == 1
    boxed.class_key_actions[8].trigger()  # no class 9: ignored
    assert shapes(boxed)[0].class_id == 1


def test_shape_color_follows_class(boxed):
    item = boxed.canvas.items_by_id()[shapes(boxed)[0].id]
    assert item.pen().color().name().upper() == boxed.project.config.classes[0].color


def test_undo_redo_drawing(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    window.act_undo.trigger()
    assert shapes(window) == []
    window.act_redo.trigger()
    assert len(shapes(window)) == 1


def test_undo_button_text_is_fixed(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    assert window.act_undo.text() == "실행 취소"
    assert window.act_undo.toolTip() == "실행 취소: 라벨 추가"
    assert window.act_undo.isEnabled() and not window.act_redo.isEnabled()


# ---- suggestions ---------------------------------------------------------------------

@pytest.fixture
def suggested(window):
    window.session.annotations.add_suggestions([
        BBox(class_id=0, x1=20, y1=20, x2=80, y2=80, score=0.92),
        BBox(class_id=1, x1=120, y1=20, x2=180, y2=80, score=0.61),
        Polygon(class_id=0, points=[(250, 50), (300, 50), (275, 120)], score=0.35),
    ], source="model")
    window.session.changed.emit()
    window.tool_actions[Tool.SELECT].trigger()
    return window


def test_suggestions_drawn_dashed_with_score(suggested):
    items = suggested.canvas.items_by_id()
    item = items[suggested.session.annotations.suggestions()[0].id]
    assert item.pen().style() == Qt.PenStyle.DashLine
    assert "0.92" in item.tag.text and "model" in item.tag.text
    assert "제안 3개" in suggested.label_panel.suggestion_label.text()


def test_enter_accepts_selected_suggestion(suggested):
    click(suggested, 50, 50)
    suggested.act_enter.trigger()
    ann = suggested.session.annotations
    assert len(ann.confirmed()) == 1 and ann.confirmed()[0].score == 0.92
    item = suggested.canvas.items_by_id()[ann.confirmed()[0].id]
    assert item.pen().style() == Qt.PenStyle.SolidLine


def test_backspace_rejects_selected_suggestion(suggested):
    click(suggested, 150, 50)
    suggested.act_backspace.trigger()
    assert [s.score for s in suggested.session.annotations.suggestions()] == [0.92, 0.35]


def test_edit_class_then_accept(suggested):
    click(suggested, 150, 50)
    suggested.class_key_actions[0].trigger()
    sid = suggested.canvas.selected_ids()[0]
    assert suggested.session.shape(sid).is_suggestion  # editing does not auto-accept
    suggested.act_enter.trigger()
    shape = suggested.session.shape(sid)
    assert not shape.is_suggestion and shape.class_id == 0


def test_accept_above_threshold_button(suggested):
    suggested.label_panel.threshold.setValue(0.6)
    suggested.label_panel.btn_accept_above.click()
    ann = suggested.session.annotations
    assert sorted(s.score for s in ann.confirmed()) == [0.61, 0.92]
    assert [s.score for s in ann.suggestions()] == [0.35]


def test_reject_all_button_and_undo(suggested):
    suggested.label_panel.btn_reject_all.click()
    assert suggested.session.annotations.suggestions() == []
    assert not suggested.label_panel.btn_accept_above.isEnabled()
    suggested.act_undo.trigger()
    assert len(suggested.session.annotations.suggestions()) == 3


def test_delete_key_removes_confirmed_but_backspace_does_not(boxed, monkeypatch):
    monkeypatch.setattr("defect_inspection.platform_utils.BACKSPACE_DELETES_SELECTION", False)  # Linux/Windows
    click(boxed, 150, 140)
    boxed.act_backspace.trigger()  # only rejects suggestions
    assert len(shapes(boxed)) == 1


def test_backspace_deletes_selection_on_mac(boxed, monkeypatch):
    """Mac keyboards have no Delete key: ⌫ deletes a confirmed label when nothing is rejected (PRD 5.2)."""
    monkeypatch.setattr("defect_inspection.platform_utils.BACKSPACE_DELETES_SELECTION", True)
    click(boxed, 150, 140)
    boxed.act_backspace.trigger()
    assert shapes(boxed) == []
    boxed.act_undo.trigger()
    assert len(shapes(boxed)) == 1


# ---- persistence -------------------------------------------------------------------------

def test_switching_image_saves_labels_and_clears_history(window, qapp):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    window.act_next_image.trigger()

    assert window.project.label_status("a.png").confirmed == 1
    assert "라벨 1" in window.image_list.item(0).text()
    assert window.undo_stack.count() == 0
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == "b.png")

    window.act_prev_image.trigger()
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == "a.png")
    assert len(shapes(window)) == 1


def test_list_status_updates_live(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    assert window.image_list.item(0).text() == "a.png\n라벨 1"
    assert window.status_labels.text() == "라벨 1"


def test_ctrl_s_saves_labels(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    window.act_save.trigger()
    lines = window.project.label_txt_path("a.png").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].startswith("0 ")


def test_close_saves_labels(window):
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (50, 50), (100, 100))
    window.close()
    assert window.project.label_status("a.png").confirmed == 1
