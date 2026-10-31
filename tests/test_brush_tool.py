"""Brush tool in the real window: left drag paints (merging into the touched label of the current
class), right drag erases, [ and ] change the size, one stroke = one undo step."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QGraphicsPathItem
from test_labeling import IMG_H, IMG_W, drag, project, qapp, vp, wait_until  # noqa: F401 (fixtures)

from defect_inspection.core import brush
from defect_inspection.core.annotation import Mask, Polygon
from defect_inspection.ui.canvas.tools import Tool
from defect_inspection.ui.main_window import MainWindow


def make_window(qapp, project, tmp_path, task):  # noqa: F811
    project.config.task = task
    project.save()
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1400, 900)
    w.show()
    w.load_project(project)
    assert wait_until(qapp, lambda: w.session is not None)
    w.canvas.zoom_to(1.0)
    w.canvas.centerOn(IMG_W / 2, IMG_H / 2)
    qapp.processEvents()
    return w


@pytest.fixture
def seg_window(qapp, project, tmp_path):  # noqa: F811
    w = make_window(qapp, project, tmp_path, "seg")
    w.set_tool(Tool.BRUSH)
    w.canvas.set_brush_size(10)
    yield w
    w.project_dirty = False
    w.close()


@pytest.fixture
def detect_window(qapp, project, tmp_path):  # noqa: F811
    w = make_window(qapp, project, tmp_path, "detect")
    w.set_tool(Tool.BRUSH)
    w.canvas.set_brush_size(10)
    yield w
    w.project_dirty = False
    w.close()


def stroke(w, points, button=Qt.MouseButton.LeftButton, modifier=Qt.KeyboardModifier.NoModifier):
    view = w.canvas.viewport()
    QTest.mousePress(view, button, modifier, vp(w, *points[0]))
    for p in points[1:]:
        QTest.mouseMove(view, vp(w, *p))
    QTest.mouseRelease(view, button, modifier, vp(w, *points[-1]))


def test_brush_tool_has_a_shortcut_and_shows_its_bar(seg_window):
    w = seg_window
    assert w.tool_actions[Tool.BRUSH].shortcut().toString() == "R"
    assert w.brush_bar.isVisible()
    w.set_tool(Tool.SELECT)
    assert not w.brush_bar.isVisible()


def test_paint_stroke_adds_a_mask_of_the_current_class(seg_window):
    w = seg_window
    w.choose_class(1)
    stroke(w, [(100, 150), (150, 150), (200, 150)])
    (s,) = w.session.shapes
    assert isinstance(s, Mask) and s.class_id == 1 and s.status == "confirmed"
    x1, y1, x2, y2 = s.bounds()
    assert 93 <= x1 <= 96 and 204 <= x2 <= 207 and y2 - y1 == pytest.approx(10, abs=1)
    assert w.canvas.selected_ids() == [s.id]
    assert w.canvas._stroke is None and not any(isinstance(i, QGraphicsPathItem) and i.zValue() == 20 for i in w.canvas.scene().items())


def test_detect_project_gets_a_mask_saved_as_a_box(detect_window):
    w = detect_window
    stroke(w, [(100, 150), (200, 150)])
    (s,) = w.session.shapes
    assert isinstance(s, Mask)
    w.session.save()
    (line,) = w.project.label_txt_path(w.session.image_path).read_text().splitlines()
    assert len(line.split()) == 5


def test_second_stroke_merges_and_undo_is_per_stroke(seg_window):
    w = seg_window
    stroke(w, [(100, 150), (150, 150)])
    (first,) = w.session.shapes
    stroke(w, [(150, 150), (150, 220)])
    (merged,) = w.session.shapes
    assert merged.id == first.id and merged.bounds()[3] > first.bounds()[3] + 50
    w.undo_stack.undo()
    assert w.session.shapes[0].bounds() == first.bounds()
    w.undo_stack.undo()
    assert w.session.shapes == []


def test_right_drag_erases(seg_window):
    w = seg_window
    w.session.add_shape(Polygon(class_id=0, points=[(50, 100), (250, 100), (250, 200), (50, 200)]))
    w.canvas.set_brush_size(30)
    stroke(w, [(150, 80), (150, 220)], button=Qt.MouseButton.RightButton)
    assert len(w.session.shapes) == 2  # cut in two
    w.undo_stack.undo()
    assert len(w.session.shapes) == 1


def test_alt_drag_erases_too(seg_window):
    """Mac trackpads: no right drag."""
    w = seg_window
    w.session.add_shape(Polygon(class_id=0, points=[(50, 100), (250, 100), (250, 200), (50, 200)]))
    w.canvas.set_brush_size(30)
    stroke(w, [(150, 80), (150, 220)], modifier=Qt.KeyboardModifier.AltModifier)
    assert len(w.session.shapes) == 2


def test_erasing_a_hole_keeps_it(seg_window):
    w = seg_window
    w.session.add_shape(Polygon(class_id=0, points=[(50, 50), (350, 50), (350, 250), (50, 250)]))
    stroke(w, [(200, 150), (210, 150)], button=Qt.MouseButton.RightButton)
    (s,) = w.session.shapes
    assert isinstance(s, Mask) and not s.full_pixels(IMG_W, IMG_H)[150, 205]


def test_painting_needs_a_class(seg_window, qapp):
    w = seg_window
    w.project.config.classes = []
    w.canvas.set_classes([], [])
    stroke(w, [(100, 150), (200, 150)])
    assert w.session.shapes == []
    assert "클래스" in w.statusBar().currentMessage()


def test_bracket_keys_change_the_size_and_it_is_remembered(seg_window, tmp_path):
    w = seg_window
    QTest.keyClick(w, Qt.Key.Key_BracketRight)
    assert w.canvas.brush_size == brush.step_size(10, up=True) > 10
    assert w.brush_bar.size_spin.value() == w.canvas.brush_size
    QTest.keyClick(w, Qt.Key.Key_BracketLeft)
    QTest.keyClick(w, Qt.Key.Key_BracketLeft)
    assert w.canvas.brush_size < 10
    w.brush_bar.size_spin.setValue(42)
    assert w.canvas.brush_size == 42
    assert int(w.settings.value("brush/size")) == 42


def test_escape_cancels_a_stroke(seg_window):
    w = seg_window
    view = w.canvas.viewport()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, 100, 150))
    QTest.mouseMove(view, vp(w, 150, 150))
    assert w.canvas.is_drawing
    w.on_escape()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, 150, 150))
    assert w.session.shapes == [] and not w.canvas.is_drawing


def test_ring_follows_the_cursor_with_the_brush_size(seg_window):
    w = seg_window
    QTest.mouseMove(w.canvas.viewport(), vp(w, 120, 90))
    ring = w.canvas._brush_ring
    assert ring is not None and ring.isVisible()
    assert ring.rect().width() == pytest.approx(10)
    c = ring.mapToScene(ring.rect().center())
    assert (c.x(), c.y()) == pytest.approx((120, 90), abs=1)
    w.set_tool(Tool.SELECT)
    assert not ring.isVisible()


def test_step_size_limits():
    assert brush.step_size(brush.MAX_SIZE, up=True) == brush.MAX_SIZE
    assert brush.step_size(brush.MIN_SIZE, up=False) == brush.MIN_SIZE
    assert brush.step_size(3, up=True) == 4 and brush.step_size(100, up=True) == 125


def test_other_button_mid_stroke_keeps_the_stroke(seg_window):
    """Review fix: touching the right button while painting no longer drops the paint stroke."""
    w = seg_window
    view = w.canvas.viewport()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, 100, 150))
    QTest.mouseMove(view, vp(w, 150, 150))
    QTest.mousePress(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, vp(w, 150, 150))
    QTest.mouseMove(view, vp(w, 200, 150))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, 200, 150))
    QTest.mouseRelease(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, vp(w, 200, 150))
    (s,) = w.session.shapes
    assert s.bounds()[2] >= 200


def test_stroke_may_start_just_off_the_image(seg_window):
    w = seg_window
    stroke(w, [(-3, 150), (60, 150)])
    (s,) = w.session.shapes
    assert s.bounds()[0] == 0


def test_damaged_size_setting_falls_back_to_the_default(qapp, tmp_path):  # noqa: F811
    settings = QSettings(str(tmp_path / "bad.ini"), QSettings.Format.IniFormat)
    settings.setValue("brush/size", "abc")
    w = MainWindow(settings, cache_root=tmp_path / "cache")
    assert w.canvas.brush_size == brush.DEFAULT_SIZE
    w.close()


def test_ring_is_hidden_after_switching_image(seg_window, qapp):
    w = seg_window
    QTest.mouseMove(w.canvas.viewport(), vp(w, 130, 100))  # QTest skips a move to where the cursor already is
    assert w.canvas._brush_ring.isVisible()
    w.image_panel.step(1)
    assert wait_until(qapp, lambda: w.session is not None and w.session.image_path.startswith("b"))
    assert not w.canvas._brush_ring.isVisible()


def test_mask_item_shows_the_hole_and_has_no_vertex_handles(seg_window):
    w = seg_window
    w.session.add_shape(Polygon(class_id=0, points=[(50, 50), (350, 50), (350, 250), (50, 250)]))
    stroke(w, [(200, 150), (210, 150)], button=Qt.MouseButton.RightButton)
    (s,) = w.session.shapes
    item = w.canvas.items_by_id()[s.id]
    assert item.handles == []
    assert item.path().contains(QPointF(100, 100)) and not item.path().contains(QPointF(205.5, 150.5))


def test_dragging_a_mask_moves_it_by_whole_pixels(seg_window):
    w = seg_window
    stroke(w, [(100, 150), (150, 150)])
    (s,) = w.session.shapes
    w.set_tool(Tool.SELECT)
    drag(w, (125, 150), (145, 170))
    (moved,) = w.session.shapes
    assert (moved.x - s.x, moved.y - s.y) == (20, 20) and moved.rle == s.rle
