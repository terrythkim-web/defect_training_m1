"""Augment tab (PROMPTS step 10): settings, presets, preview, dataset build from the UI."""

import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox, ImageAnnotations
from defect_inspection.core.augment_config import BUILTIN_PRESETS, AugmentConfig, TransformSetting
from defect_inspection.core.preview import augment_previews, draw_labels, scale_for_preview
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.main_window import MainWindow
from defect_inspection.ui.panels.augment_panel import CUSTOM, AugmentPanel

W, H = 120, 90


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_project(root, color_mode="color", labeled=4):
    src = root / "원본"
    src.mkdir(parents=True)
    for i in range(labeled + 1):
        img = np.full((H, W, 3), 200, np.uint8)
        cv2.circle(img, (40 + i * 5, 40), 10, (40, 40, 40), -1)
        imwrite_unicode(src / f"img{i}.png", img)
    p = Project.create(root / "증강 프로젝트", color_mode=color_mode)
    p.add_class("dent")
    p.add_images([src])
    for i in range(labeled):
        ann = p.load_labels(f"img{i}.png")
        ann.shapes.append(BBox(class_id=0, x1=30 + i * 5, y1=30, x2=51 + i * 5, y2=51))
        p.save_labels(ann)
    p.config.augment.multiplier = 2
    p.save()
    return p


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1500, 950)
    w.show()
    w.load_project(make_project(tmp_path))
    assert wait_until(qapp, lambda: w.session is not None)
    yield w
    w.project_dirty = False
    w.close()


@pytest.fixture
def panel(window, qapp):
    window.tabs.setCurrentWidget(window.augment_panel)
    qapp.processEvents()
    return window.augment_panel


# ---- core preview helpers --------------------------------------------------------------

def test_scale_for_preview_scales_labels():
    ann = ImageAnnotations(image="a.png", width=1000, height=500,
                           shapes=[BBox(class_id=0, x1=100, y1=50, x2=300, y2=150)])
    small, out = scale_for_preview(np.zeros((500, 1000), np.uint8), ann, max_side=500)
    assert small.shape == (250, 500) and (out.width, out.height) == (500, 250)
    assert out.shapes[0].bounds() == pytest.approx((50, 25, 150, 75))


def test_scale_for_preview_drops_suggestions():
    ann = ImageAnnotations(image="a.png", width=10, height=10)
    ann.add_suggestions([BBox(class_id=0, x1=1, y1=1, x2=5, y2=5, score=0.9)], source="sam")
    assert scale_for_preview(np.zeros((10, 10), np.uint8), ann)[1].shapes == []


def test_draw_labels_uses_class_color():
    ann = ImageAnnotations(image="a.png", width=40, height=40,
                           shapes=[BBox(class_id=0, x1=5, y1=5, x2=30, y2=30)])
    out = draw_labels(np.zeros((40, 40), np.uint8), ann, ["#FF0000"])
    assert out.shape == (40, 40, 3)
    edge = out[4:6, 15]  # top edge at y = 5 (continuous) -> pixel rows 4/5
    assert (edge[:, 2] > 100).any() and (edge[:, :2] < 50).all()  # red, BGR


def test_augment_previews_count_and_min_area_scaling():
    img = np.full((2000, 2000, 3), 200, np.uint8)
    # 30x30 label = 900 px² full size, ~54 px² at preview scale; min_area 100 must still keep it.
    ann = ImageAnnotations(image="a.png", width=2000, height=2000,
                           shapes=[BBox(class_id=0, x1=1000, y1=1000, x2=1030, y2=1030)])
    cfg = AugmentConfig(min_area_px=100)
    tiles = augment_previews(img, ann, cfg, 3, ["#00FF00"], n=9, seed=0)
    assert len(tiles) == 9 and all(t.shape == (480, 480, 3) for t in tiles)
    assert all((t[:, :, 1] == 255).any() for t in tiles)  # label drawn in every tile


# ---- settings -------------------------------------------------------------------------------

def test_tab_is_augment_panel(window):
    assert isinstance(window.panels["Augment"], AugmentPanel)


def test_panel_shows_project_settings(panel, window):
    cfg = window.project.config.augment
    assert panel.multiplier.value() == 2
    assert panel.rows["rotate90"].check.isChecked() == cfg.setting("rotate90").enabled
    assert panel.min_visibility.value() == 40


def test_edit_updates_project_and_marks_dirty(panel, window):
    window.project_dirty = False
    panel.rows["defocus"].check.setChecked(True)
    panel.rows["defocus"].strength.setValue(80)
    s = window.project.config.augment.setting("defocus")
    assert s.enabled and s.strength == pytest.approx(0.8)
    assert window.project_dirty
    panel.rows["defocus"].check.setChecked(False)
    assert window.project.config.augment.setting("defocus").strength == pytest.approx(0.8)  # remembered


def test_preset_combo_tracks_settings(panel, window):
    panel.apply_preset("보통")
    assert panel.preset_combo.currentText() == "보통"
    panel.rows["defocus"].check.setChecked(True)
    assert panel.preset_combo.currentText() == CUSTOM
    panel.apply_preset("강함")
    assert panel.preset_combo.currentText() == "강함"
    assert window.project.config.augment.enabled_keys() == BUILTIN_PRESETS["강함"].enabled_keys()


def test_save_user_preset(panel, window):
    panel.rows["defocus"].check.setChecked(True)
    path = panel.save_current_as("우리 라인")
    assert path.parent == window.presets_dir and path.is_file()
    items = [panel.preset_combo.itemText(i) for i in range(panel.preset_combo.count())]
    assert "우리 라인" in items and panel.preset_combo.currentText() == "우리 라인"


def test_grayscale_project_disables_color_transforms(qapp, tmp_path):
    p = make_project(tmp_path, color_mode="grayscale")
    panel = AugmentPanel(tmp_path / "presets")
    panel.set_project(p)
    assert not panel.rows["hue_saturation"].check.isEnabled()
    assert panel.rows["brightness_contrast"].check.isEnabled()
    assert not panel.current_config().setting("hue_saturation").enabled


def test_no_project_disables_panel(qapp, tmp_path):
    panel = AugmentPanel(tmp_path / "presets")
    assert not panel.btn_build.isEnabled() and not panel.rows["hflip"].check.isEnabled()


# ---- preview ---------------------------------------------------------------------------

def test_preview_fills_nine_tiles(panel, qapp):
    panel.refresh_preview()
    assert wait_until(qapp, lambda: all(not p.isNull() for p in panel.preview_pixmaps))
    assert "무작위" in panel.preview_status.text()


def test_preview_refreshes_after_setting_change(panel, qapp):
    panel.refresh_preview()
    assert wait_until(qapp, lambda: all(not p.isNull() for p in panel.preview_pixmaps))
    gen = panel._preview_gen
    panel.rows["defocus"].check.setChecked(True)
    assert wait_until(qapp, lambda: panel._preview_gen > gen)


def test_preview_uses_unsaved_labels(window, panel):
    window.session.add_shape(BBox(class_id=0, x1=1, y1=1, x2=20, y2=20))
    provided = window._current_annotations(window.current_image)
    assert len(provided.shapes) == 2  # includes the unsaved edit
    assert window._current_annotations("img3.png") is None


def test_preview_error_shown(panel, window, qapp):
    window.project.image_abspath(window.current_image).write_bytes(b"broken")
    panel.refresh_preview()
    assert wait_until(qapp, lambda: "실패" in panel.preview_status.text())


# ---- build -------------------------------------------------------------------------------

def test_build_from_ui(panel, window, qapp):
    reports = []
    panel.build_finished.connect(reports.append)
    window.session.add_shape(BBox(class_id=0, x1=60, y1=60, x2=80, y2=80))  # unsaved edit
    panel.btn_build.click()
    assert panel.is_building and not panel.btn_build.isEnabled()
    assert wait_until(qapp, lambda: not panel.is_building, timeout=60)

    (report,) = reports
    assert report.augmented == report.train_images * 2
    assert "완료" in panel.build_status.text() and "제외" in panel.build_status.text()  # img4 unlabeled
    assert (window.project.dataset_dir / "full" / "data.yaml").is_file()
    assert window.project.label_status("img0.png").confirmed == 2  # saved before building
    # is_building clears before build_finished; the button returns when the thread is cleaned up
    assert wait_until(qapp, panel.btn_build.isEnabled)


def test_build_cancel(panel, window, qapp):
    window.project.config.augment.multiplier = 200
    panel.btn_build.click()
    assert wait_until(qapp, lambda: panel.progress.value() > 0, timeout=30)
    panel.btn_cancel.click()
    assert wait_until(qapp, lambda: not panel.is_building, timeout=60)
    assert "취소" in panel.build_status.text()
    assert not (window.project.dataset_dir / "full.building").exists()


def test_build_error_message(qapp, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr("defect_inspection.ui.panels.augment_panel.QMessageBox.warning",
                        lambda *a: shown.append(a[2]))
    p = make_project(tmp_path, labeled=1)  # one labeled image: no Val defect
    panel = AugmentPanel(tmp_path / "presets")
    panel.set_project(p)
    panel.start_build()
    assert wait_until(qapp, lambda: not panel.is_building, timeout=30)
    assert shown and "2장 이상" in shown[0]
    assert "실패" in panel.build_status.text()


def test_close_during_build_cancels(qapp, tmp_path):
    w = MainWindow(QSettings(str(tmp_path / "s2.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "c2")
    w.show()
    p = make_project(tmp_path / "x")
    p.config.augment = AugmentConfig(multiplier=300, transforms={"hflip": TransformSetting(enabled=True)})
    w.load_project(p)
    w.augment_panel.start_build()
    assert wait_until(qapp, lambda: w.augment_panel.progress.value() > 0, timeout=30)
    w.close()
    assert not w.augment_panel.is_building or w.augment_panel.wait_build(1)
    assert not (p.dataset_dir / "full.building").exists()


def test_not_building_when_finished_signal_arrives(panel, qapp):
    """Follow-up actions connected to build_finished must see the build as over."""
    seen = []
    panel.build_finished.connect(lambda _r: seen.append(panel.is_building))
    panel.start_build()
    assert wait_until(qapp, lambda: seen, timeout=60)
    assert seen == [False]
