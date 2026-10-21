"""images/ folder sync: files copied in or deleted outside the app show up in the project."""

import os
import shutil
import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox
from defect_inspection.core.project import REMOVED_LABELS_DIR, Project
from defect_inspection.platform_utils import imread_unicode, imwrite_unicode
from defect_inspection.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def make_project(tmp_path, color_mode="color"):
    src = tmp_path / "원본"
    src.mkdir()
    for name in ("a.png", "b.png"):
        imwrite_unicode(src / name, np.full((16, 16, 3), 120, np.uint8))
    p = Project.create(tmp_path / "동기화 프로젝트", color_mode=color_mode)
    p.add_class("스크래치")
    p.add_images([src])
    p.save()
    return p


def drop_image(folder, name, color=(10, 200, 30), age=10.0):
    """Write an image as if copied in by Finder `age` seconds ago."""
    path = folder / name
    imwrite_unicode(path, np.full((16, 16, 3), color, np.uint8))
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def test_new_files_registered_and_hidden_files_ignored(tmp_path):
    p = make_project(tmp_path)
    drop_image(p.images_dir, "새 이미지.png")
    drop_image(p.images_dir, ".hidden.png")
    (p.images_dir / "notes.txt").write_text("x", encoding="utf-8")
    changes = p.sync_images_dir()
    assert [e.path for e in changes.added] == ["새 이미지.png"] and not changes.removed
    assert [e.path for e in p.config.images] == ["a.png", "b.png", "새 이미지.png"]
    assert not p.sync_images_dir().changed  # idempotent


def test_recent_file_is_pending_until_settled(tmp_path):
    p = make_project(tmp_path)
    drop_image(p.images_dir, "c.png", age=0)
    changes = p.sync_images_dir(settle_sec=5)
    assert changes.pending == ["c.png"] and not changes.added
    assert [e.path for e in p.sync_images_dir(settle_sec=5, now=time.time() + 10).added] == ["c.png"]


def test_stem_clash_renames_file(tmp_path):
    p = make_project(tmp_path)
    drop_image(p.images_dir, "a.jpg")
    changes = p.sync_images_dir()
    assert changes.renamed == [("a.jpg", "a_1.jpg")] and [e.path for e in changes.added] == ["a_1.jpg"]
    assert (p.images_dir / "a_1.jpg").is_file() and not (p.images_dir / "a.jpg").exists()


def test_grayscale_project_converts_new_colour_file(tmp_path):
    p = make_project(tmp_path, color_mode="grayscale")
    drop_image(p.images_dir, "c.png")
    assert [e.path for e in p.sync_images_dir().added] == ["c.png"]
    assert imread_unicode(p.images_dir / "c.png").ndim == 2


def test_removed_file_unregistered_and_labels_come_back(tmp_path):
    p = make_project(tmp_path)
    ann = p.load_labels("a.png")
    ann.shapes.append(BBox(class_id=0, x1=1, y1=1, x2=5, y2=5))
    p.save_labels(ann)
    backup = tmp_path / "a_backup.png"
    shutil.move(p.images_dir / "a.png", backup)

    changes = p.sync_images_dir()
    assert changes.removed == ["a.png"] and [e.path for e in p.config.images] == ["b.png"]
    assert not p.label_json_path("a.png").exists()
    assert (p.labels_dir / REMOVED_LABELS_DIR / "a.json").is_file()

    drop_image(p.images_dir, "b_other.png")  # unrelated new file: no labels
    shutil.move(backup, p.images_dir / "a.png")
    os.utime(p.images_dir / "a.png", (time.time() - 10, time.time() - 10))
    p.sync_images_dir()
    assert p.label_status("a.png").confirmed == 1  # restored
    assert p.label_status("b_other.png").confirmed == 0


def test_orphan_label_for_other_image_not_attached(tmp_path):
    p = make_project(tmp_path)
    ann = p.load_labels("a.png")
    ann.shapes.append(BBox(class_id=0, x1=1, y1=1, x2=5, y2=5))
    p.save_labels(ann)
    (p.labels_dir / "c.json").write_text(
        p.label_json_path("a.png").read_text(encoding="utf-8"), encoding="utf-8")  # says image a.png
    drop_image(p.images_dir, "c.png")
    changes = p.sync_images_dir()
    assert [e.path for e in changes.added] == ["c_1.png"]
    assert p.label_status("c_1.png").confirmed == 0


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.show()
    yield w
    w.train_panel.shutdown()
    w.project_dirty = False
    w.close()


def list_paths(window):
    lst = window.image_panel.list
    return [lst.item(i).data(0x0100) for i in range(lst.count())]


def test_window_follows_folder_changes(window, tmp_path, qapp):
    p = make_project(tmp_path)
    window.load_project(p)
    window.image_panel.select("b.png")
    assert wait_until(qapp, lambda: window.session is not None and window.current_image == "b.png")
    session = window.session

    drop_image(p.images_dir, "c.png")
    assert wait_until(qapp, lambda: "c.png" in list_paths(window))  # through QFileSystemWatcher
    assert window.session is session and window.current_image == "b.png"  # kept open
    assert window.project_dirty and "추가 1장" in window.statusBar().currentMessage()

    (p.images_dir / "b.png").unlink()
    assert wait_until(qapp, lambda: "b.png" not in list_paths(window))
    assert window.current_image != "b.png" and window.session is not session
    assert not p.label_json_path("b.png").exists()  # the closed session was not saved back


def test_changes_while_closed_picked_up_on_open(window, tmp_path, qapp):
    p = make_project(tmp_path)
    (p.images_dir / "a.png").unlink()
    drop_image(p.images_dir, "d.png")
    window.load_project(Project.open(p.root))
    assert list_paths(window) == ["b.png", "d.png"]
    assert window.project_dirty


def test_sync_waits_for_dataset_build(window, tmp_path, qapp, monkeypatch):
    p = make_project(tmp_path)
    window.load_project(p)
    monkeypatch.setattr(type(window.augment_panel), "is_building", property(lambda self: True))
    drop_image(p.images_dir, "c.png")
    assert not window.sync_images_folder() and window.images_sync_timer.isActive()
    assert "c.png" not in list_paths(window)
    monkeypatch.undo()
    assert window.sync_images_folder() and "c.png" in list_paths(window)
