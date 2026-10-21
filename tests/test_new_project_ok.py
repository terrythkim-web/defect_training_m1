"""New project dialog (task, colour mode) and OK image marking (PRD F1.1, F1.5, F1.6)."""

import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from defect_inspection.core.annotation import BBox
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.main_window import MainWindow
from defect_inspection.ui.new_project_dialog import NewProjectDialog


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
    return cond()


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.show()
    yield w
    w.train_panel.shutdown()
    w.project_dirty = False
    w.close()


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "원본"
    src.mkdir()
    for name in ("a.png", "b.png"):
        imwrite_unicode(src / name, np.zeros((16, 16, 3), np.uint8))
    p = Project.create(tmp_path / "검사 프로젝트")
    p.add_class("스크래치")
    p.add_images([src])
    p.save()
    return p


def test_dialog_values_and_validation(qapp, tmp_path):
    d = NewProjectDialog()
    ok = d.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()  # no folder yet
    d.folder.setText(str(tmp_path / "새 프로젝트"))
    assert ok.isEnabled()
    d.task.setCurrentIndex(d.task.findData("seg"))
    d.color_mode.setCurrentIndex(d.color_mode.findData("grayscale"))
    assert "그레이" in d.note.text()
    assert d.values() == {"root": tmp_path / "새 프로젝트", "name": None, "task": "seg", "color_mode": "grayscale"}
    d.name.setText("PCB 라인")
    assert d.values()["name"] == "PCB 라인"
    Project.create(tmp_path / "있음")
    d.folder.setText(str(tmp_path / "있음"))
    assert not ok.isEnabled() and "이미 프로젝트" in d.note.text()


def test_new_project_uses_dialog_choices(window, tmp_path, monkeypatch):
    def fake_exec(dialog):
        dialog.folder.setText(str(tmp_path / "그레이 seg"))
        dialog.task.setCurrentIndex(dialog.task.findData("seg"))
        dialog.color_mode.setCurrentIndex(dialog.color_mode.findData("grayscale"))
        return NewProjectDialog.DialogCode.Accepted

    monkeypatch.setattr(NewProjectDialog, "exec", fake_exec)
    window.new_project_dialog()
    assert window.project is not None and window.project.root == tmp_path / "그레이 seg"
    assert window.project.config.task == "seg" and window.project.channels == 1


def test_toggle_ok_marks_and_clears(window, project, qapp):
    window.load_project(project)
    window.image_panel.select("a.png")
    assert wait_until(qapp, lambda: window.session is not None and window.current_image == "a.png")
    window.act_toggle_ok.trigger()
    assert project.image("a.png").is_ok and window.project_dirty
    assert "OK 이미지" in window.image_panel._item("a.png").text()
    window.image_panel.filter_combo.setCurrentIndex(window.image_panel.filter_combo.findData("ok"))
    assert not window.image_panel._item("a.png").isHidden() and window.image_panel._item("b.png").isHidden()
    window.act_toggle_ok.trigger()
    assert not project.image("a.png").is_ok
    assert window.act_toggle_ok.shortcut().toString() == "O"
    assert window.act_toggle_ok in window.image_list.actions()  # right-click menu


def test_ok_refused_with_labels_and_cleared_by_drawing(window, project, qapp):
    window.load_project(project)
    window.image_panel.select("a.png")
    assert wait_until(qapp, lambda: window.session is not None and window.current_image == "a.png")
    window.session.add_shape(BBox(class_id=0, x1=1, y1=1, x2=5, y2=5))
    window.toggle_ok()
    assert not project.image("a.png").is_ok and "OK로 표시할 수 없습니다" in window.statusBar().currentMessage()

    window.image_panel.select("b.png")
    assert wait_until(qapp, lambda: window.session is not None and window.current_image == "b.png")
    window.toggle_ok()
    assert project.image("b.png").is_ok
    window.session.add_shape(BBox(class_id=0, x1=1, y1=1, x2=5, y2=5))  # a defect after all
    assert not project.image("b.png").is_ok
