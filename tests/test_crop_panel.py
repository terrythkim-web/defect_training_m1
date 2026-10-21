"""Crop settings and Crop build in the Augment tab (PROMPTS step 17, PRD F7.2-F7.5)."""

import time

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from defect_inspection.core.crop_config import CROP_PRESETS, CropConfig
from defect_inspection.core.project import Project
from defect_inspection.ui.main_window import MainWindow
from test_crop import make_project


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def window(qapp, tmp_path):
    p, _ = make_project(tmp_path)
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1500, 950)
    w.show()
    w.load_project(p)
    w.tabs.setCurrentWidget(w.augment_panel)
    qapp.processEvents()
    yield w
    w.augment_panel.wait_build()
    w.train_panel.shutdown()
    w.project_dirty = False
    w.close()


@pytest.fixture
def panel(window):
    return window.augment_panel


def test_crop_settings_loaded(panel, window):
    cfg = window.project.config.crop
    assert panel.crop_size.value() == cfg.crop_size == 128
    assert panel.use_defect.isChecked() and not panel.use_tiles.isChecked()
    assert not panel.tile_size.isEnabled() and not panel.overlap.isEnabled()
    assert panel.crop_multiplier.value() == 2


def test_edits_write_project_config(panel, window):
    panel.use_tiles.setChecked(True)
    assert panel.tile_size.isEnabled()
    panel.tile_size.setValue(512)
    panel.jitter.setValue(30)
    panel.bg_ratio.setValue(2.0)
    cfg = window.project.config.crop
    assert cfg.use_tiles and cfg.tile_size == 512 and cfg.jitter == pytest.approx(0.3)
    assert cfg.background_ratio == 2.0 and window.project_dirty
    panel.use_defect.setChecked(False)
    assert not panel.per_defect.isEnabled() and not panel.jitter.isEnabled()


def test_overlap_percent_and_px(panel, window):
    panel.use_tiles.setChecked(True)
    panel.overlap.setValue(25)
    assert window.project.config.crop.overlap_percent == 25
    panel.overlap_unit.setCurrentIndex(panel.overlap_unit.findData("px"))
    assert panel.overlap.value() == 32  # the stored px value
    panel.overlap.setValue(64)
    cfg = window.project.config.crop
    assert cfg.overlap_unit == "px" and cfg.overlap_px == 64 and cfg.overlap_percent == 25
    assert cfg.overlap_pixels() == 64
    panel.overlap_unit.setCurrentIndex(panel.overlap_unit.findData("percent"))
    assert panel.overlap.value() == 25


def test_att_yolo_preset(panel, window):
    name = "ATT-YOLO (1280 타일, overlap 32px)"
    panel.crop_preset.setCurrentIndex(panel.crop_preset.findText(name))
    panel.crop_preset.activated.emit(panel.crop_preset.currentIndex())
    assert window.project.config.crop == CROP_PRESETS[name]
    assert panel.tile_size.value() == 1280 and panel.overlap.value() == 32
    assert panel.overlap_unit.currentData() == "px" and not panel.use_defect.isChecked()
    assert panel.build_mode == "crop"
    panel.tile_size.setValue(1024)
    assert panel.crop_preset.currentText() == "직접 설정"


def test_crop_config_persists(panel, window):
    panel.crop_size.setValue(256)
    window.save_project()
    assert Project.open(window.project.root).config.crop.crop_size == 256


def test_crop_build_from_panel(panel, window, qapp):
    reports = []
    panel.build_finished.connect(reports.append)
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("crop"))
    panel.start_build()
    assert wait_until(qapp, lambda: reports and not panel.is_building)
    assert (window.project.dataset_dir / "crop" / "data.yaml").is_file()
    assert not (window.project.dataset_dir / "full").exists()
    assert "Crop 완료" in panel.build_status.text() and "결함 중심" in panel.build_status.text()
    tp = window.train_panel
    i = tp.dataset_combo.findData("crop")
    assert tp.dataset_combo.model().item(i).isEnabled()


def test_crop_build_error_shown(panel, window, qapp, monkeypatch):
    monkeypatch.setattr("defect_inspection.ui.panels.augment_panel.QMessageBox.warning", lambda *a: None)
    window.project.config.crop = CropConfig(use_roi=False, use_defect=False, use_background=False)
    panel.load_crop_config(window.project.config.crop)
    stopped = []
    panel.build_stopped.connect(stopped.append)
    panel.start_build("crop")
    assert wait_until(qapp, lambda: stopped and not panel.is_building)
    assert "하나 이상" in panel.build_status.text()


def test_full_build_still_default(panel, window, qapp):
    reports = []
    panel.build_finished.connect(reports.append)
    assert panel.build_mode == "full"
    panel.start_build()
    assert wait_until(qapp, lambda: reports and not panel.is_building)
    assert (window.project.dataset_dir / "full" / "data.yaml").is_file()
    assert "완료" in panel.build_status.text() and "Crop" not in panel.build_status.text()
