"""Crop ROI tool (PROMPTS step 16, PRD F7.1)."""

import json
import time

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox
from defect_inspection.core.project import Project
from defect_inspection.core.roi import (
    CROPS_FILE,
    Roi,
    RoiOptions,
    RoiStore,
    clip_roi,
    fixed_roi_at,
    move_roi,
    resize_roi,
    roi_from_drag,
)
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.canvas.roi_tool import RoiItem
from defect_inspection.ui.canvas.tools import Tool
from defect_inspection.ui.main_window import MainWindow

W, H = 400, 300


# ---- geometry ------------------------------------------------------------------------------

def test_fixed_roi_centred_and_clamped():
    assert fixed_roi_at(200, 150, 100, W, H).bounds() == (150, 100, 250, 200)
    assert fixed_roi_at(10, 10, 100, W, H).bounds() == (0, 0, 100, 100)  # pushed inside
    assert fixed_roi_at(390, 290, 100, W, H).bounds() == (300, 200, 400, 300)
    assert fixed_roi_at(200, 150, 640, W, H).bounds() == (0, 0, W, H)  # image smaller than size


def test_free_drag_and_click():
    assert roi_from_drag(300, 250, 100, 50, W, H).bounds() == (100, 50, 300, 250)
    assert roi_from_drag(-50, -50, 100, 100, W, H).bounds() == (0, 0, 100, 100)
    assert roi_from_drag(10, 10, 12, 50, W, H) is None  # too thin


def test_fixed_size_click_or_drag():
    opts = RoiOptions(fixed_size=100)
    assert roi_from_drag(200, 150, 200, 150, W, H, opts).bounds() == (150, 100, 250, 200)  # click
    assert roi_from_drag(100, 100, 200, 200, W, H, opts).bounds() == (100, 100, 200, 200)  # drag centre


def test_square_drag_stays_inside():
    opts = RoiOptions(square=True)
    r = roi_from_drag(50, 50, 150, 90, W, H, opts)
    assert (r.width, r.height) == (100, 100)
    r = roi_from_drag(350, 250, 600, 400, W, H, opts)  # limited by the image corner
    assert r.width == r.height == 50 and r.x2 == W


def test_move_and_resize():
    roi = Roi(x1=100, y1=100, x2=200, y2=180)
    assert move_roi(roi, 500, -500, W, H).bounds() == (300, 0, 400, 80)
    assert resize_roi(roi, 2, 250, 260, W, H).bounds() == (100, 100, 250, 260)
    assert resize_roi(roi, 0, 0, 0, W, H).bounds() == (0, 0, 200, 180)
    sq = resize_roi(roi, 2, 300, 200, W, H, square=True)
    assert sq.width == sq.height
    with pytest.raises(ValueError):
        resize_roi(roi, 2, 102, 260, W, H)


def test_clip_roi():
    roi = Roi(x1=300, y1=200, x2=600, y2=500)
    assert clip_roi(roi, W, H).bounds() == (300, 200, W, H)
    assert clip_roi(Roi(x1=500, y1=0, x2=600, y2=50), W, H) is None


def test_roi_validation():
    with pytest.raises(ValueError):
        Roi(x1=5, y1=5, x2=5, y2=10)
    assert Roi(x1=10, y1=10, x2=0, y2=0).bounds() == (0, 0, 10, 10)


# ---- storage --------------------------------------------------------------------------------

def test_store_roundtrip_and_effective(tmp_path):
    store = RoiStore(tmp_path)
    a, b = Roi(x1=0, y1=0, x2=50, y2=50), Roi(x1=380, y1=280, x2=500, y2=400)
    store.set_rois("이미지 1.png", [a], [b])
    store.set_options(RoiOptions(fixed_size=320))
    raw = json.loads((tmp_path / CROPS_FILE).read_text(encoding="utf-8"))
    assert "이미지 1.png" in raw["images"] and raw["options"]["fixed_size"] == 320

    again = RoiStore(tmp_path)
    assert again.image_rois("이미지 1.png") == [a] and again.common == [b]
    eff = again.effective("이미지 1.png", W, H)
    assert [r.bounds() for r in eff] == [(380, 280, 400, 300), (0, 0, 50, 50)]  # common clipped first
    assert again.effective("other.png", 300, 200) == []  # common ROI entirely outside a small image
    again.set_rois("이미지 1.png", [], [])
    assert not again.has_any() and "이미지 1.png" not in json.loads((tmp_path / CROPS_FILE).read_text("utf-8"))["images"]


def test_store_missing_file(tmp_path):
    store = RoiStore(tmp_path)
    assert store.common == [] and store.options == RoiOptions() and not (tmp_path / CROPS_FILE).exists()


# ---- UI ---------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def window(qapp, tmp_path):
    src = tmp_path / "원본"
    src.mkdir()
    imwrite_unicode(src / "a.png", np.full((H, W, 3), 120, np.uint8))
    imwrite_unicode(src / "b.png", np.full((H, W, 3), 120, np.uint8))
    imwrite_unicode(src / "small.png", np.full((150, 200, 3), 120, np.uint8))
    p = Project.create(tmp_path / "ROI 프로젝트")
    p.add_class("dent")
    p.add_images([src])
    p.save()
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1400, 900)
    w.show()
    w.load_project(p)
    assert wait_until(qapp, lambda: w.roi_session is not None)
    w.canvas.zoom_to(1.0)
    w.canvas.centerOn(W / 2, H / 2)
    w.tool_actions[Tool.CROP_ROI].trigger()
    qapp.processEvents()
    yield w
    w.project_dirty = False
    w.close()


def vp(w, x, y):
    return w.canvas.mapFromScene(QPointF(x, y))


def drag(w, start, end):
    view = w.canvas.viewport()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, *start))
    for t in (0.5, 1.0):
        QTest.mouseMove(view, vp(w, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, *end))


def click(w, x, y):
    QTest.mouseClick(w.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(w, x, y))


def store_on_disk(w):
    return RoiStore(w.project.root)


def test_bar_visible_only_with_roi_tool(window):
    assert window.roi_bar.isVisible()
    window.tool_actions[Tool.SELECT].trigger()
    assert not window.roi_bar.isVisible()


def test_drag_creates_roi_saved_immediately(window):
    drag(window, (50, 60), (250, 200))
    rois = store_on_disk(window).image_rois("a.png")
    assert len(rois) == 1 and rois[0].bounds() == pytest.approx((50, 60, 250, 200), abs=1.5)
    assert "ROI 1" in window.roi_bar.count_label.text()
    assert window.session.annotations.shapes == []  # labels untouched


def test_fixed_size_click(window):
    window.roi_bar.size_combo.setCurrentIndex(window.roi_bar.size_combo.findData(320))
    click(window, 200, 150)
    (roi,) = store_on_disk(window).image_rois("a.png")
    assert (roi.width, roi.height) == (320, 300)  # image is only 300 px tall
    assert store_on_disk(window).options.fixed_size == 320
    assert not window.roi_bar.square.isEnabled()


def test_square_option(window):
    window.roi_bar.square.setChecked(True)
    drag(window, (50, 50), (200, 100))
    (roi,) = store_on_disk(window).image_rois("a.png")
    assert roi.width == pytest.approx(roi.height, abs=1.5)


def test_move_resize_and_delete(window, qapp):
    drag(window, (50, 50), (150, 150))
    click(window, 100, 100)
    rid = window.canvas.selected_roi_ids()[0]
    drag(window, (100, 100), (160, 130))
    assert store_on_disk(window).image_rois("a.png")[0].bounds() == pytest.approx((110, 80, 210, 180), abs=1.5)
    click(window, 160, 130)
    item = window.canvas.roi_items()[rid]
    assert any(h.isVisible() for h in item.handles)
    drag(window, (210, 180), (260, 240))  # bottom-right handle
    assert store_on_disk(window).image_rois("a.png")[0].bounds() == pytest.approx((110, 80, 260, 240), abs=1.5)
    click(window, 150, 150)
    window.act_delete.trigger()
    assert store_on_disk(window).image_rois("a.png") == []


def test_fixed_size_rois_cannot_be_resized(window):
    window.roi_bar.size_combo.setCurrentIndex(window.roi_bar.size_combo.findData(320))
    click(window, 200, 150)
    click(window, 200, 150)
    item = next(iter(window.canvas.roi_items().values()))
    assert item.isSelected() and not any(h.isVisible() for h in item.handles)


def test_undo_redo(window):
    drag(window, (50, 50), (150, 150))
    window.act_undo.trigger()
    assert store_on_disk(window).image_rois("a.png") == []
    window.act_redo.trigger()
    assert len(store_on_disk(window).image_rois("a.png")) == 1


def test_common_roi_on_every_image(window, qapp):
    drag(window, (300, 200), (400, 300))
    window.roi_bar.btn_common.click()
    store = store_on_disk(window)
    assert store.image_rois("a.png") == [] and len(store.common) == 1
    assert "공통 1" in window.roi_bar.count_label.text()
    item = next(iter(window.canvas.roi_items().values()))
    assert item.is_common and item.pen().style() == Qt.PenStyle.DashDotLine

    window.image_panel.select("b.png")
    assert wait_until(qapp, lambda: window.roi_session is not None and window.roi_session.image_path == "b.png")
    assert [i.is_common for i in window.canvas.roi_items().values()] == [True]

    window.image_panel.select("small.png")  # 200x150: the common ROI lies outside
    assert wait_until(qapp, lambda: window.roi_session is not None and window.roi_session.image_path == "small.png")
    assert window.canvas.roi_items() == {}

    window.image_panel.select("a.png")
    assert wait_until(qapp, lambda: window.roi_session is not None and window.roi_session.image_path == "a.png")
    window.roi_bar.btn_clear_common.click()
    assert store_on_disk(window).common == []


def test_rois_do_not_interfere_with_labels(window, qapp):
    drag(window, (50, 50), (250, 250))  # ROI
    window.tool_actions[Tool.BBOX].trigger()
    drag(window, (100, 100), (150, 150))  # label drawn inside the ROI
    assert len(window.session.annotations.shapes) == 1
    window.tool_actions[Tool.SELECT].trigger()
    click(window, 125, 125)
    assert window.canvas.selected_ids() == [window.session.annotations.shapes[0].id]
    assert window.canvas.selected_roi_ids() == []
    window.act_delete.trigger()  # deletes the label, not the ROI
    assert window.session.annotations.shapes == [] and len(store_on_disk(window).image_rois("a.png")) == 1
    window.tool_actions[Tool.ERASER].trigger()
    click(window, 60, 60)  # eraser ignores ROIs
    assert len(store_on_disk(window).image_rois("a.png")) == 1


def test_labels_and_yolo_txt_ignore_rois(window):
    drag(window, (50, 50), (250, 250))
    window.session.add_shape(BBox(class_id=0, x1=10, y1=10, x2=40, y2=40))
    window.save_project()
    lines = window.project.label_txt_path("a.png").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_escape_cancels_roi_drag(window):
    view = window.canvas.viewport()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(window, 50, 50))
    QTest.mouseMove(view, vp(window, 150, 150))
    window.act_escape.trigger()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(window, 150, 150))
    assert store_on_disk(window).image_rois("a.png") == []


def test_rois_survive_reopen(window, qapp, tmp_path):
    drag(window, (50, 50), (150, 150))
    root = window.project.root
    window.project_dirty = False
    window.close()
    w2 = MainWindow(QSettings(str(tmp_path / "s2.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "c2")
    w2.show()
    w2.load_project(Project.open(root))
    try:
        assert wait_until(qapp, lambda: w2.roi_session is not None)
        assert len(w2.canvas.roi_items()) == 1
        assert isinstance(next(iter(w2.canvas.roi_items().values())), RoiItem)
    finally:
        w2.project_dirty = False
        w2.close()


def test_new_roi_replaces_the_previous_one(window, qapp):
    drag(window, (300, 200), (400, 280))
    window.roi_bar.btn_common.click()  # a common ROI must survive
    drag(window, (50, 60), (150, 160))
    first = store_on_disk(window).image_rois("a.png")
    drag(window, (200, 50), (330, 150))
    store = store_on_disk(window)
    (roi,) = store.image_rois("a.png")
    assert roi.bounds() == pytest.approx((200, 50, 330, 150), abs=1.5) and len(store.common) == 1
    assert "이전 ROI 1개를 새 ROI로 바꿨습니다" in window.statusBar().currentMessage()
    assert len(window.canvas.roi_items()) == 2  # the new ROI + the common one
    window.act_undo.trigger()
    assert store_on_disk(window).image_rois("a.png") == first
    window.image_panel.select("b.png")  # other images keep their own ROIs
    assert wait_until(qapp, lambda: window.roi_session is not None and window.roi_session.image_path == "b.png")
    drag(window, (20, 20), (120, 120))
    assert store_on_disk(window).image_rois("a.png") == first
