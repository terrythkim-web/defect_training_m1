"""Autosave, class management and the SAM tool in the window (PROMPTS step 7).

SAM runs through a fake controller so tests need neither weights nor torch inference."""

import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, QPointF, QSettings, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.project import Project
from defect_inspection.core.sam_assist import SamPrompt
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.canvas.tools import Tool
from defect_inspection.ui.main_window import MainWindow

IMG_W, IMG_H = 400, 300


class FakeSam(QObject):
    """Mimics SamController: a disk of radius 20 around the first include point, or the box."""

    result_ready = Signal(int, object)
    failed = Signal(int, str)
    status = Signal(str)
    ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = "fake.pt"
        self.prepared, self.requests = [], []
        self.fail = False

    def prepare(self, image8, key):
        self.prepared.append(key)

    def request(self, generation, image8, key, prompt: SamPrompt):
        self.requests.append(prompt)
        if self.fail:
            self.failed.emit(generation, "boom")
            return
        mask = np.zeros(image8.shape[:2], np.uint8)
        if prompt.box is not None:
            x1, y1, x2, y2 = map(int, prompt.box)
            mask[y1:y2, x1:x2] = 1
        include = [p for p, l in zip(prompt.points, prompt.labels) if l == 1]
        if include:
            cv2.circle(mask, tuple(map(int, include[0])), 20, 1, -1)
        result = (mask.astype(bool), 0.9) if mask.any() else None
        self.result_ready.emit(generation, result)

    def set_model(self, model):
        self.model = model

    def shutdown(self):
        pass


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


def make_window(qapp, tmp_path, task="detect"):
    src = tmp_path / "원본"
    src.mkdir(exist_ok=True)
    for name in ["a.png", "b.png"]:
        imwrite_unicode(src / name, np.full((IMG_H, IMG_W, 3), 120, np.uint8))
    p = Project.create(tmp_path / f"프로젝트 {task}", task=task)
    p.add_class("스크래치")
    p.add_class("찍힘")
    p.add_images([src])
    p.save()
    w = MainWindow(QSettings(str(tmp_path / f"s_{task}.ini"), QSettings.Format.IniFormat),
                   cache_root=tmp_path / "cache")
    fake = FakeSam()
    w.set_sam_controller(fake)
    w.resize(1400, 900)
    w.show()
    w.load_project(p)
    assert wait_until(qapp, lambda: w.session is not None)
    w.canvas.zoom_to(1.0)
    w.canvas.centerOn(IMG_W / 2, IMG_H / 2)
    qapp.processEvents()
    return w, fake


@pytest.fixture
def window(qapp, tmp_path):
    w, fake = make_window(qapp, tmp_path)
    w.fake_sam = fake
    yield w
    w.project_dirty = False
    w.close()


def vp(w, x, y):
    return w.canvas.mapFromScene(QPointF(x, y))


def click(w, x, y, button=Qt.MouseButton.LeftButton):
    QTest.mouseClick(w.canvas.viewport(), button, Qt.KeyboardModifier.NoModifier, vp(w, x, y))


def drag(w, start, end):
    view = w.canvas.viewport()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, *start))
    for t in (0.5, 1.0):
        QTest.mouseMove(view, vp(w, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, *end))


# ---- autosave ------------------------------------------------------------------------------

def test_autosave_writes_labels_and_project(window):
    window.session.add_shape(BBox(class_id=0, x1=10, y1=10, x2=50, y2=50))
    window.add_class("이물")
    assert window.autosave()
    assert window.project.label_status("a.png").confirmed == 1
    assert len(Project.open(window.project.root).config.classes) == 3
    assert not window.project_dirty and not window.session.dirty
    assert "자동 저장됨" in window.statusBar().currentMessage()
    assert not window.autosave()  # nothing left to write


def test_autosave_timer_interval_persisted(window, qapp):
    window.set_autosave_interval(1)
    assert window.autosave_timer.isActive() and window.autosave_timer.interval() == 1000
    window.session.add_shape(BBox(class_id=0, x1=10, y1=10, x2=50, y2=50))
    assert wait_until(qapp, lambda: not window.session.dirty, timeout=3)
    window.set_autosave_interval(0)
    assert not window.autosave_timer.isActive()
    assert int(window.settings.value("autosave/interval_sec")) == 0


def test_default_autosave_is_30s(window):
    window.settings.remove("autosave/interval_sec")
    window.set_autosave_interval(30)
    assert window.autosave_timer.interval() == 30_000


# ---- class management -------------------------------------------------------------------------

def test_rename_class_updates_lists_and_tags(window):
    window.session.add_shape(BBox(class_id=0, x1=10, y1=10, x2=50, y2=50))
    window.rename_class(0, "긁힘")
    assert window.class_list.item(0).text() == "1: 긁힘"
    assert window.label_panel.list.item(0).text().startswith("긁힘")
    item = next(iter(window.canvas.items_by_id().values()))
    assert item.tag.text == "긁힘"
    assert window.project_dirty


def test_rename_to_existing_name_rejected(window, monkeypatch):
    warnings = []
    monkeypatch.setattr("defect_inspection.ui.main_window.QMessageBox.warning", lambda *a: warnings.append(a))
    window.rename_class(0, "찍힘")
    assert window.project.config.classes[0].name == "스크래치" and warnings


def test_class_color_recolors_shapes(window):
    window.session.add_shape(BBox(class_id=1, x1=10, y1=10, x2=50, y2=50))
    window.set_class_color(1, "#123abc")
    item = next(iter(window.canvas.items_by_id().values()))
    assert item.pen().color().name() == "#123abc"
    assert window.project.config.classes[1].color == "#123ABC"


def test_rename_keeps_current_class(window):
    window.class_list.setCurrentRow(1)
    window.rename_class(0, "긁힘")
    assert window.class_list.currentRow() == 1 and window.canvas.current_class_id == 1


# ---- SAM tool -------------------------------------------------------------------------------

def sam_suggestions(w):
    return [s for s in w.session.annotations.suggestions() if s.source == "sam"]


def test_selecting_sam_prepares_image(window):
    window.tool_actions[Tool.SAM].trigger()
    assert window.fake_sam.prepared == [str(window.project.image_abspath("a.png"))]


def test_click_creates_sam_suggestion_with_current_class(window):
    window.class_list.setCurrentRow(1)
    window.tool_actions[Tool.SAM].trigger()
    click(window, 200, 150)

    sugg = sam_suggestions(window)
    assert len(sugg) == 1
    s = sugg[0]
    assert isinstance(s, BBox) and s.class_id == 1 and s.score == 0.9  # detect project -> box
    assert s.bounds() == pytest.approx((180, 130, 221, 171), abs=1.5)
    assert window.fake_sam.requests[-1].labels == [1]


def test_more_clicks_refine_the_same_suggestion(window):
    window.tool_actions[Tool.SAM].trigger()
    click(window, 200, 150)
    first_id = window.canvas.sam_shape_id
    click(window, 50, 50, Qt.MouseButton.RightButton)  # exclude point

    assert window.fake_sam.requests[-1].labels == [1, 0]
    assert len(sam_suggestions(window)) == 1 and sam_suggestions(window)[0].id == first_id


def test_box_drag_prompt(window):
    window.tool_actions[Tool.SAM].trigger()
    drag(window, (100, 100), (160, 140))
    prompt = window.fake_sam.requests[-1]
    assert prompt.box == pytest.approx((100, 100, 160, 140), abs=1.5) and prompt.points == []
    assert sam_suggestions(window)[0].bounds() == pytest.approx((100, 100, 160, 140), abs=1.5)


def test_enter_accepts_and_next_click_starts_new_object(window):
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    window.act_enter.trigger()
    assert window.session.annotations.confirmed()[0].source == "sam"
    assert window.canvas.sam_shape_id is None and window.canvas.sam_prompt.is_empty

    click(window, 300, 200)
    assert len(sam_suggestions(window)) == 1
    assert len(window.session.annotations.confirmed()) == 1


def test_escape_rejects_current_object(window):
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    window.act_escape.trigger()
    assert sam_suggestions(window) == [] and window.canvas.sam_prompt.is_empty


def test_backspace_removes_last_point_then_suggestion(window):
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    click(window, 50, 50, Qt.MouseButton.RightButton)
    window.act_backspace.trigger()
    assert window.canvas.sam_prompt.labels == [1] and len(sam_suggestions(window)) == 1
    window.act_backspace.trigger()
    assert window.canvas.sam_prompt.is_empty and sam_suggestions(window) == []


def test_seg_project_gets_polygon(qapp, tmp_path):
    w, fake = make_window(qapp, tmp_path, task="seg")
    try:
        w.tool_actions[Tool.SAM].trigger()
        click(w, 200, 150)
        s = sam_suggestions(w)[0]
        assert isinstance(s, Polygon) and len(s.points) >= 8
    finally:
        w.project_dirty = False
        w.close()


def test_stale_result_ignored(window):
    window.tool_actions[Tool.SAM].trigger()
    window._sam_generation = 5
    window._on_sam_result(4, (np.ones((IMG_H, IMG_W), bool), 0.9))
    assert sam_suggestions(window) == []


def test_sam_result_ignored_after_switching_tool(window):
    window.tool_actions[Tool.SAM].trigger()
    window._sam_generation = 1
    window.tool_actions[Tool.BBOX].trigger()
    window._on_sam_result(1, (np.ones((IMG_H, IMG_W), bool), 0.9))
    assert sam_suggestions(window) == []


def test_empty_sam_result_shows_hint(window):
    window.tool_actions[Tool.SAM].trigger()
    window._sam_generation = 1
    window._on_sam_result(1, None)
    assert "찾지 못했습니다" in window.statusBar().currentMessage()


def test_sam_failure_shows_dialog(window, monkeypatch):
    shown = []
    monkeypatch.setattr("defect_inspection.ui.main_window.QMessageBox.warning", lambda *a: shown.append(a))
    window.fake_sam.fail = True
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    assert shown and "boom" in shown[0][2]


def test_sam_suggestion_undo(window):
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    window.act_enter.trigger()
    window.act_undo.trigger()  # un-accept
    assert len(sam_suggestions(window)) == 1
    window.act_undo.trigger()  # remove the suggestion
    assert window.session.shapes == []


def test_switching_image_resets_sam(window, qapp):
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    window.act_next_image.trigger()
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == "b.png")
    assert window.canvas.sam_prompt.is_empty and window.canvas.sam_shape_id is None
    assert window.fake_sam.prepared[-1].endswith("b.png")
    # the pending suggestion on a.png was saved with it
    assert window.project.label_status("a.png").suggested == 1


def test_sam_model_setting_persisted(window):
    window.set_sam_model("sam2.1_b.pt")
    assert window.fake_sam.model == "sam2.1_b.pt"
    assert window.settings.value("sam/model") == "sam2.1_b.pt"


def test_sam_needs_a_class(window):
    window.project.config.classes.clear()
    window._refresh_classes()
    window.tool_actions[Tool.SAM].trigger()
    click(window, 100, 100)
    assert window.fake_sam.requests == []
