"""Compare tab (PROMPTS step 19, PRD F8.1-F8.4)."""

import time
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from defect_inspection.core.trainer import list_runs
from defect_inspection.ui.main_window import MainWindow
from test_compare import blob_factory, make_project_with_runs


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
    p, runs = make_project_with_runs(tmp_path)
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1500, 1000)
    w.show()
    w.compare_panel.make_predictor = blob_factory()
    w.compare_panel.devices = [("CPU", "cpu")]
    w.load_project(p)
    w.tabs.setCurrentWidget(w.compare_panel)
    assert wait_until(qapp, lambda: w.session is not None)
    w.test_runs = [str(r) for r in runs]
    yield w
    w.compare_panel.shutdown()
    w.augment_panel.wait_build()
    w.train_panel.shutdown(5000)
    w.project_dirty = False
    w.close()


@pytest.fixture
def panel(window):
    return window.compare_panel


def compared(panel, qapp, runs):
    panel.set_checked(runs)
    assert panel.btn_compare.isEnabled()
    panel.compare_checked()
    assert wait_until(qapp, lambda: panel.results and not panel.busy)


def test_runs_listed_and_two_needed(panel, window):
    assert panel.run_list.count() == 2 and "crop" in panel.run_list.item(0).text() + panel.run_list.item(1).text()
    assert any("타일 128" in panel.run_list.item(i).text() for i in range(2))
    assert not panel.btn_compare.isEnabled()
    panel.set_checked(window.test_runs[:1])
    assert not panel.btn_compare.isEnabled()
    panel.set_checked(window.test_runs)
    assert panel.btn_compare.isEnabled()


def test_compare_fills_table_charts_and_summary(panel, window, qapp):
    compared(panel, qapp, window.test_runs)
    assert len(panel.results) == 2
    labels = [panel.table.verticalHeaderItem(r).text() for r in range(panel.table.rowCount())]
    for needed in ("Precision", "Recall", "F1", "mAP50", "mAP50-95", "OK/NG 판정 정확도", "미검 이미지 (NG→OK)",
                   "학습 시간", "ONNX 크기", "타일링"):
        assert any(needed in lb for lb in labels), needed
    assert any("dent mAP50-95" in l for l in labels) and any("작은 결함" in l for l in labels)
    assert any(l.startswith("추론 시간 CPU") for l in labels)
    tiling_row = next(i for i, l in enumerate(labels) if l == "타일링")
    texts = {panel.table.item(tiling_row, c).text() for c in range(2)}
    assert texts == {"-", "128px, overlap 26px"}
    assert panel.table.columnCount() == 2 and panel.table.horizontalHeaderItem(0).text().startswith("A · ")
    assert "추천:" in panel.summary.text() and "Val 원본" in panel.summary.text()
    assert len(panel.bars.listDataItems()) + len(panel.bars.getPlotItem().items) > 0
    assert len(panel.pr_plot.listDataItems()) == 2 and len(panel.map_plot.listDataItems()) == 2
    assert "A:" in panel.legend.text() and "B:" in panel.legend.text()


def test_priority_changes_summary(panel, window, qapp):
    compared(panel, qapp, window.test_runs)
    panel.priority.setCurrentText("Recall")
    assert "Recall" in panel.summary.text()


def test_side_by_side_and_open_image(panel, window, qapp):
    compared(panel, qapp, window.test_runs)
    panel.result_tabs.setCurrentIndex(2)
    assert panel.diff_list.count() == panel.results[0].metrics.images
    assert not panel.view_a.pixmap().isNull() and not panel.view_b.pixmap().isNull()
    assert "맞음" in panel.caption_a.text() and "맞음" in panel.caption_b.text()
    item = panel.diff_list.item(0)
    panel.diff_list.itemDoubleClicked.emit(item)
    assert window.current_image == item.data(Qt.ItemDataRole.UserRole)


def test_compare_error_is_shown(panel, window, qapp):
    window.project.add_class("scratch")
    panel.set_checked(window.test_runs)
    panel.compare_checked()
    assert wait_until(qapp, lambda: not panel.busy)
    assert "비교 실패" in panel.summary.text() and not panel.results


def test_full_crop_experiment(panel, window, qapp, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "2:0.01")
    before = {r.run_id for r in list_runs(window.project)}
    window.train_panel.epochs.setValue(2)
    assert window.start_compare_experiment()
    assert not window.start_compare_experiment()  # already running
    assert wait_until(qapp, lambda: window._exp_stage is None and panel.results and not panel.busy, 120)
    new = [r for r in list_runs(window.project) if r.run_id not in before]
    assert sorted(r.spec.train.dataset for r in new) == ["crop", "full"]
    full, crop = sorted(new, key=lambda r: r.spec.train.dataset != "full")
    assert full.spec.train.model_dump(exclude={"dataset"}) == crop.spec.train.model_dump(exclude={"dataset"})
    assert crop.spec.crop is not None and full.spec.crop is None
    assert {r.run_id for r in panel.results} == {full.run_id, crop.run_id}
    assert window.tabs.currentWidget() is panel and "비교" in panel.exp_status.text()


def test_experiment_stops_when_build_fails(panel, window, qapp, monkeypatch):
    from defect_inspection.core.crop_config import CropConfig

    monkeypatch.setattr("defect_inspection.ui.panels.augment_panel.QMessageBox.warning", lambda *a: None)
    window.project.config.crop = CropConfig(use_roi=False, use_defect=False, use_background=False)
    assert window.start_compare_experiment()
    assert wait_until(qapp, lambda: window._exp_stage is None, 60)
    assert "빌드 실패" in panel.exp_status.text() and not window.train_panel.running


# ---- HTML report and history (step 20, F8.5, F8.6) -------------------------------------------------

def test_save_report_and_history(panel, window, qapp, monkeypatch):
    assert not panel.btn_report.isEnabled()
    compared(panel, qapp, window.test_runs)
    assert panel.btn_report.isEnabled()
    panel.memo.setPlainText("Crop이 작은 결함에 유리")
    assert panel.save_report()
    assert wait_until(qapp, lambda: panel.last_report is not None and not panel.report_busy)
    path = panel.last_report
    assert path.parent == window.project.reports_dir and path.name.endswith("_full_vs_crop.html")
    html = path.read_text(encoding="utf-8")
    assert "Crop이 작은 결함에 유리" in html and "저장:" in panel.report_status.text()
    assert panel.report_list.count() == 1 and panel.selected_report().path == path
    assert "full·yolo11n" in panel.report_list.item(0).text()

    opened = []
    panel.open_url = opened.append
    panel.btn_open_report.click()
    assert opened and Path(opened[0].toLocalFile()) == path  # Windows: QUrl gives '/', str(path) '\\'

    monkeypatch.setattr("defect_inspection.ui.panels.compare_panel.QMessageBox.question",
                        lambda *a: __import__("PySide6.QtWidgets").QtWidgets.QMessageBox.StandardButton.No)
    assert not panel.delete_report() and path.exists()
    monkeypatch.setattr("defect_inspection.ui.panels.compare_panel.QMessageBox.question",
                        lambda *a: __import__("PySide6.QtWidgets").QtWidgets.QMessageBox.StandardButton.Yes)
    assert panel.delete_report() and not path.exists() and panel.report_list.count() == 0


def test_rerun_from_report(panel, window, qapp, monkeypatch):
    compared(panel, qapp, window.test_runs)
    panel.save_report()
    assert wait_until(qapp, lambda: panel.last_report is not None and not panel.report_busy)
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "1:0.01")
    before = {r.run_id for r in list_runs(window.project)}
    panel.result_tabs.setCurrentIndex(3)
    assert panel.btn_rerun.isEnabled()
    panel.btn_rerun.click()
    assert window._exp_stage is not None
    assert wait_until(qapp, lambda: window._exp_stage is None and not panel.busy
                      and {r.run_id for r in panel.results} - before, 120)
    new = [r for r in list_runs(window.project) if r.run_id not in before]
    old = {r.spec.train.dataset: r.spec.train for r in list_runs(window.project) if r.run_id in before}
    assert sorted(r.spec.train.dataset for r in new) == ["crop", "full"]
    for r in new:
        assert r.spec.train == old[r.spec.train.dataset]  # exactly the reported settings
