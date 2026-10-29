"""Train tab (PROMPTS step 12): form, start/stop/resume through the real worker process (fake
training mode), live charts, log and history."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from defect_inspection.core.annotation import BBox
from defect_inspection.core.augment_config import AugmentConfig, TransformSetting
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import LOG_FILE, EpochSeries, list_runs
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.main_window import MainWindow
from defect_inspection.ui.widgets.training_charts import SERIES, TrainingCharts, loss_style, metric_style


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_project(root, color_mode="color", build=True):
    src = root / "원본"
    src.mkdir(parents=True)
    for i in range(4):
        img = np.full((96, 96, 3), 190, np.uint8)
        cv2.circle(img, (40 + i * 4, 40), 10, (40, 40, 40), -1)
        imwrite_unicode(src / f"img{i}.png", img)
    p = Project.create(root / "학습 탭 프로젝트", color_mode=color_mode)
    p.add_class("dent")
    p.add_images([src])
    for i, e in enumerate(p.config.images):
        ann = p.load_labels(e)
        ann.shapes.append(BBox(class_id=0, x1=30 + i * 4, y1=30, x2=51 + i * 4, y2=51))
        p.save_labels(ann)
    p.config.augment = AugmentConfig(multiplier=1, transforms={"hflip": TransformSetting(enabled=True)})
    p.save()
    if build:
        build_full_dataset(p)
    return p


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "3:0.05")
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1500, 1000)
    w.show()
    w.load_project(make_project(tmp_path))
    w.tabs.setCurrentWidget(w.train_panel)
    qapp.processEvents()
    yield w
    w.train_panel.shutdown(5000)
    w.project_dirty = False
    w.close()


@pytest.fixture
def panel(window):
    return window.train_panel


def log_text(panel):
    return panel.log_view.toPlainText()


# ---- chart styling (dataviz rules) -------------------------------------------------------

def test_loss_colour_follows_component_and_style_follows_split():
    name, color, style = loss_style("train/box_loss")
    assert (name, color, style) == ("box (train)", SERIES[0], Qt.PenStyle.SolidLine)
    name, color, style = loss_style("val/box_loss")
    assert color == SERIES[0] and style == Qt.PenStyle.DashLine
    assert loss_style("train/cls_loss")[1] == SERIES[1] and loss_style("val/seg_loss")[1] == SERIES[3]
    assert loss_style("metrics/mAP50(B)") is None


def test_metric_styles_fixed_order():
    assert [metric_style(f"metrics/{k}(B)")[1] for k in ("precision", "recall", "mAP50", "mAP50-95")] == SERIES[:4]
    name, color, style = metric_style("metrics/mAP50(M)")
    assert name == "mAP50 (mask)" and color == SERIES[2] and style == Qt.PenStyle.DashLine


def test_charts_draw_series_and_hover(qapp):
    charts = TrainingCharts()
    charts.resize(500, 500)
    charts.show()
    for e in range(1, 4):
        charts.add_event({"type": "epoch", "epoch": e, "losses": {"train/box_loss": 1 / e, "train/cls_loss": 2 / e},
                          "metrics": {"val/box_loss": 1.2 / e, "metrics/mAP50(B)": e / 10}})
    assert set(charts.loss.curves) == {"train/box_loss", "train/cls_loss", "val/box_loss"}
    assert set(charts.metrics.curves) == {"metrics/mAP50(B)"}
    x, y = charts.metrics.curves["metrics/mAP50(B)"].getData()
    assert list(x) == [1, 2, 3] and list(y) == pytest.approx([0.1, 0.2, 0.3])
    legend = charts.loss_legend.text()
    assert "box (train)" in legend and "box (val)" in legend and "cls (train)" in legend
    assert SERIES[0] in legend and "╺╺╺" in legend  # colour only on the dashed/solid sample
    text = charts.loss.hover_text(2, charts.series)
    assert "epoch 2" in text and "box (train): 0.5000" in text
    charts.clear()
    assert charts.loss.curves == {} and charts.series.epochs == []


def test_series_replay_handles_resume_overlap():
    events = [{"type": "epoch", "epoch": e, "losses": {"train/box_loss": e}} for e in (1, 2, 3, 2, 3)]
    s = EpochSeries.from_events(events)
    assert s.epochs == [1, 2, 3] and s.latest("train/box_loss") == 3


# ---- form ---------------------------------------------------------------------------------

def test_model_combo_from_registry_and_defaults(panel):
    keys = [panel.model_combo.itemData(i) for i in range(panel.model_combo.count())]
    assert keys == ["yolo11n", "yolo11s", "yolo11m", "att-yolo-n", "att-yolo-s"]
    assert panel.model_combo.currentData() == "yolo11s" and not panel.model_note.isVisibleTo(panel)
    panel.select_model("att-yolo-s")
    assert (panel.epochs.value(), panel.imgsz.value()) == (300, 1280)
    assert panel.model_note.isVisibleTo(panel) and "Compare" in panel.model_note.text()
    panel.select_model("yolo11n")
    assert (panel.epochs.value(), panel.imgsz.value()) == (100, 640)


def test_current_config(panel):
    panel.paper_hyp.setChecked(True)
    panel.online_aug.setChecked(False)
    panel.lr0.setValue(0)
    panel.batch.setValue(-1)
    cfg = panel.current_config()
    assert cfg.paper_hyp and not cfg.online_augment and cfg.lr0 is None and cfg.batch == -1
    assert cfg.dataset == "full" and cfg.device == "auto"


def test_online_augment_warning(panel):
    assert panel.online_aug.isChecked() and panel.online_warning.isVisibleTo(panel)
    assert "1종" in panel.online_warning.text()
    panel.online_aug.setChecked(False)
    assert not panel.online_warning.isVisibleTo(panel)


def test_task_label_and_channels(panel, qapp, tmp_path):
    assert "3채널" in panel.task_label.text()
    gray = make_project(tmp_path / "g", color_mode="grayscale", build=False)
    panel.set_project(gray)
    assert "1채널" in panel.task_label.text()


def test_dataset_availability(panel, qapp, tmp_path):
    assert panel.dataset_combo.model().item(0).isEnabled()
    assert not panel.dataset_combo.model().item(1).isEnabled()  # crop not built
    assert "빌드 안 됨" in panel.dataset_combo.itemText(1)
    unbuilt = make_project(tmp_path / "u", build=False)
    panel.set_project(unbuilt)
    assert not panel.btn_start.isEnabled()


def test_device_detection_enables_gpu(panel):
    from defect_inspection.platform_utils import gpu_device_choice

    gpu = panel.device.findData(gpu_device_choice()[1])  # "0" (CUDA) or "mps" on a Mac
    assert not panel.device.model().item(gpu).isEnabled() or panel.gpu_available
    panel.set_detected_device("GPU: NVIDIA RTX 4090")
    assert panel.device.model().item(gpu).isEnabled() and "RTX" in panel.device.itemText(0)


# ---- running ---------------------------------------------------------------------------------

def test_training_run_updates_ui(panel, qapp):
    done = []
    panel.run_finished.connect(lambda run, state: done.append(state))
    assert panel.start()
    assert panel.running and not panel.btn_start.isEnabled() and panel.btn_stop.isEnabled()
    assert not panel.settings_box.isEnabled()
    assert wait_until(qapp, lambda: done)
    assert done == ["finished"]
    assert panel.progress.value() == 3 and panel.progress.maximum() == 3
    assert "metrics/mAP50(B)" in panel.charts.metrics.curves
    log = log_text(panel)
    assert "학습 시작" in log and "모델 입력 채널: 3 (컬러)" in log and "학습 완료" in log
    assert panel.history.rowCount() == 1 and panel.history.item(0, 2).text() == "완료"
    assert panel.btn_start.isEnabled() and panel.settings_box.isEnabled()


def test_grayscale_run_logs_one_channel(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "1:0.01")
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "c")
    w.load_project(make_project(tmp_path, color_mode="grayscale"))
    try:
        assert w.train_panel.start()
        assert wait_until(qapp, lambda: not w.train_panel.running and "학습 완료" in log_text(w.train_panel))
        assert "모델 입력 채널: 1 (그레이스케일)" in log_text(w.train_panel)
    finally:
        w.project_dirty = False
        w.close()


def test_stop_and_resume_from_history(panel, qapp, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "50:0.1")
    states = []
    panel.run_finished.connect(lambda run, state: states.append(state))
    panel.start()
    assert wait_until(qapp, lambda: panel.progress.value() >= 1)
    panel.btn_stop.click()
    assert wait_until(qapp, lambda: states == ["stopped"])
    assert "중지됨" in panel.history.item(0, 2).text() and "재개 가능" in panel.history.item(0, 2).text()

    panel.history.selectRow(0)
    assert panel.btn_resume.isEnabled()
    stopped_at = panel.charts.series.epochs[-1]
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", f"{stopped_at + 2}:0.02")
    assert panel.resume_selected()
    assert wait_until(qapp, lambda: states == ["stopped", "finished"])
    assert panel.charts.series.epochs == list(range(1, stopped_at + 3))  # replayed + continued
    assert panel.history.item(0, 2).text() == "완료"


def test_log_saved_per_run_and_fresh_for_each_run(panel, qapp, monkeypatch):
    done = []
    panel.run_finished.connect(lambda run, state: done.append(run))
    panel.log_view.setPlainText("이전 내용")
    assert panel.start()
    assert wait_until(qapp, lambda: len(done) == 1)
    first = log_text(panel)
    assert "이전 내용" not in first and "학습 완료" in first
    saved = (Path(done[0]) / LOG_FILE).read_text(encoding="utf-8")
    assert saved.startswith("===== 시작 ") and "학습 완료" in saved and "epoch 3/3" in saved

    assert panel.start()
    assert wait_until(qapp, lambda: len(done) == 2)
    assert done[0] != done[1]
    assert log_text(panel).count("학습 완료") == 1  # the second run's log starts empty
    assert (Path(done[0]) / LOG_FILE).read_text(encoding="utf-8") == saved  # untouched

    rows = {panel.history.runs[i].path: i for i in range(len(panel.history.runs))}
    panel.history.selectRow(rows[Path(done[0])])
    assert log_text(panel) == saved.rstrip("\n")


def test_resume_appends_to_the_run_log(panel, qapp, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "50:0.1")
    states = []
    panel.run_finished.connect(lambda run, state: states.append((run, state)))
    panel.start()
    assert wait_until(qapp, lambda: panel.progress.value() >= 1)
    panel.btn_stop.click()
    assert wait_until(qapp, lambda: len(states) == 1)
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "2:0.01")
    panel.history.selectRow(0)
    assert "중지됨" in log_text(panel)
    assert panel.resume_selected()
    assert wait_until(qapp, lambda: len(states) == 2)
    saved = (Path(states[0][0]) / LOG_FILE).read_text(encoding="utf-8")
    assert saved.index("===== 시작") < saved.index("중지됨") < saved.index("===== 재개")
    assert "===== 시작" in log_text(panel) and "===== 재개" in log_text(panel)


def test_old_run_without_log(panel, qapp):
    panel.start()
    assert wait_until(qapp, lambda: not panel.running and panel.history.rowCount() == 1)
    (panel.history.runs[0].path / LOG_FILE).unlink()
    panel.history.clearSelection()
    panel.history.selectRow(0)
    assert "저장된 로그가 없습니다" in log_text(panel)


def test_selecting_history_replays_charts(panel, qapp):
    panel.start()
    assert wait_until(qapp, lambda: not panel.running and panel.history.rowCount() == 1)
    panel.charts.clear()
    panel.history.selectRow(0)
    assert panel.charts.series.epochs == [1, 2, 3]


def test_failed_run_shown(panel, qapp, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_CRASH", "1")
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "1:0.01")
    panel.start()
    assert wait_until(qapp, lambda: not panel.running and panel.history.rowCount() == 1)
    assert panel.history.item(0, 2).text().startswith("실패")
    assert "비정상 종료" in log_text(panel)


def test_interrupted_run_from_previous_session(panel, window):
    from defect_inspection.core.trainer import TrainConfig, prepare_run, write_status

    run = prepare_run(window.project, TrainConfig(), window.weights_dir)
    write_status(run, {"state": "running"})
    panel.refresh_history()
    assert panel.history.item(0, 2).text() == "중단됨"


def test_start_refused_during_dataset_build(window, panel, monkeypatch):
    monkeypatch.setattr(type(window.augment_panel), "is_building", property(lambda self: True))
    shown = []
    monkeypatch.setattr("defect_inspection.ui.main_window.QMessageBox.information", lambda *a: shown.append(a[2]))
    assert not panel.start() and shown and "빌드" in shown[0]


def test_start_saves_unsaved_labels(window, panel, qapp):
    assert wait_until(qapp, lambda: window.session is not None)
    window.session.add_shape(BBox(class_id=0, x1=1, y1=1, x2=9, y2=9))
    panel.start()
    assert window.project.label_status(window.session.image_path).confirmed == 2
    assert wait_until(qapp, lambda: not panel.running)


def test_project_switch_blocked_while_training(window, panel, qapp, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "50:0.1")
    shown = []
    monkeypatch.setattr("defect_inspection.ui.main_window.QMessageBox.information", lambda *a: shown.append(a[2]))
    panel.start()
    window.open_project_dialog()
    assert shown and "학습 중" in shown[0]
    panel.stop()
    assert wait_until(qapp, lambda: not panel.running)


def test_close_while_training_asks_and_stops(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "50:0.1")
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "c")
    w.show()
    w.load_project(make_project(tmp_path))
    w.train_panel.start()
    assert wait_until(qapp, lambda: w.train_panel.progress.value() >= 1)
    monkeypatch.setattr("defect_inspection.ui.main_window.QMessageBox.question",
                        lambda *a: QMessageBox.StandardButton.No)
    assert not w.close()  # user said no
    monkeypatch.setattr("defect_inspection.ui.main_window.QMessageBox.question",
                        lambda *a: QMessageBox.StandardButton.Yes)
    w.project_dirty = False
    assert w.close()
    assert not w.train_panel.running
    assert list_runs(w.project)[0].state == "stopped"


def test_refresh_after_dataset_build(window, panel, qapp):
    import shutil

    shutil.rmtree(window.project.dataset_dir / "full")
    panel.refresh_datasets()
    assert not panel.btn_start.isEnabled()
    window.augment_panel.start_build()
    assert wait_until(qapp, lambda: not window.augment_panel.is_building)
    assert panel.btn_start.isEnabled()


def test_dataset_warnings_logged_at_start(panel, qapp):
    done = []
    panel.run_finished.connect(lambda run, state: done.append(run))
    assert panel.start()
    assert wait_until(qapp, lambda: done)
    log = log_text(panel)
    assert "⚠ 데이터셋: Val 결함이 1개뿐입니다" in log  # 4 labeled images, 1 in Val
    assert "⚠ 데이터셋" in (Path(done[0]) / LOG_FILE).read_text(encoding="utf-8")
