"""Export tab and file naming (PROMPTS step 15, PRD F6.3)."""

import time
from datetime import datetime

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from defect_inspection.ui.main_window import MainWindow
from test_exporter import make_project, untrained_run


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


class FixedNow:
    """Stand-in for datetime in core.exporter so default names are predictable."""

    value = datetime(2026, 10, 13, 14, 30, 25)

    @classmethod
    def now(cls):
        return cls.value


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("defect_inspection.core.exporter.datetime", FixedNow)
    p = make_project(tmp_path)
    untrained_run(p)
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1400, 1000)
    w.show()
    w.load_project(p)
    w.tabs.setCurrentWidget(w.export_panel)
    qapp.processEvents()
    yield w
    w.project_dirty = False
    w.close()


@pytest.fixture
def panel(window):
    return window.export_panel


def test_run_listed_and_default_name(panel, window):
    assert panel.run_combo.count() == 1
    assert panel.name.text() == "PCB_라인_full_yolo11n_20261013_143025.onnx"
    assert panel.folder.text() == str(window.project.exports_dir)
    assert "metadata.json" in panel.sidecar_label.text() and panel.btn_export.isEnabled()


def test_forbidden_characters_cannot_be_typed(panel, qapp):
    panel.name.clear()
    QTest.keyClicks(panel.name, 'a?b:c*d"e<f>g|h/i\\j')
    assert panel.name.text() == "abcdefghij"


@pytest.mark.parametrize("name, fragment", [("CON", "예약어"), ("com1.onnx", "예약어"), ("   ", "입력")])
def test_reserved_and_bad_names_block_export(panel, name, fragment):
    panel.name.setText(name)
    assert fragment in panel.name_problem.text()
    assert not panel.btn_export.isEnabled()


def test_long_path_blocked(panel, tmp_path):
    panel.folder.setText(str(tmp_path / ("긴폴더" * 60)))
    assert "260" in panel.name_problem.text() and not panel.btn_export.isEnabled()


def test_onnx_suffix_added(panel, tmp_path):
    panel.set_target(tmp_path / "out" / "my model")
    assert panel.target_path().name == "my model.onnx"


def test_default_name_uses_export_time(panel, qapp, monkeypatch):
    FixedNow.value = datetime(2026, 10, 13, 15, 0, 0)  # time passed since the tab was opened
    try:
        assert panel.export()
        assert wait_until(qapp, lambda: not panel.busy)
        assert panel.last_result.onnx_path.name == "PCB_라인_full_yolo11n_20261013_150000.onnx"
    finally:
        FixedNow.value = datetime(2026, 10, 13, 14, 30, 25)


def test_edited_name_is_kept(panel, qapp, tmp_path):
    # QTest cannot synthesize Korean key events (aborts on the offscreen platform): set the text and
    # emit textEdited, which is exactly what typing does.
    panel.name.setText("라인A_최종")
    panel.name.textEdited.emit("라인A_최종")
    panel.folder.setText(str(tmp_path / "내보내기 폴더"))
    assert panel.export()
    assert wait_until(qapp, lambda: not panel.busy)
    r = panel.last_result
    assert r.onnx_path == tmp_path / "내보내기 폴더" / "라인A_최종.onnx"
    assert r.metadata_path.name == "라인A_최종.metadata.json" and r.classes_path.name == "라인A_최종.classes.txt"
    assert r.onnx_path.is_file() and r.metadata_path.is_file()
    text = panel.result_label.text()
    assert "✓ PASS" in text and "3채널" in text and panel.btn_open_folder.isVisible()


def test_overwrite_confirmation(panel, qapp, monkeypatch, tmp_path):
    panel.set_target(tmp_path / "same.onnx")
    assert panel.export()
    assert wait_until(qapp, lambda: not panel.busy)
    first_mtime = (tmp_path / "same.onnx").stat().st_mtime_ns

    asked = []
    monkeypatch.setattr("defect_inspection.ui.panels.export_panel.QMessageBox.question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.No)
    panel.set_target(tmp_path / "same.onnx")
    assert not panel.export()
    assert asked and "same.onnx" in asked[0] and "same.metadata.json" in asked[0]
    assert (tmp_path / "same.onnx").stat().st_mtime_ns == first_mtime

    monkeypatch.setattr("defect_inspection.ui.panels.export_panel.QMessageBox.question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert panel.export()
    assert wait_until(qapp, lambda: not panel.busy)
    assert (tmp_path / "same.onnx").stat().st_mtime_ns != first_mtime


def test_options_passed(panel):
    panel.imgsz.setValue(160)
    panel.dynamic.setChecked(True)
    panel.nms.setChecked(True)
    opts = panel.current_options()
    assert (opts.imgsz, opts.dynamic, opts.nms, opts.opset) == (160, True, True, None)
    panel.imgsz.setValue(0)
    panel.opset.setValue(17)
    opts = panel.current_options()
    assert opts.imgsz is None and opts.opset == 17


def test_fp16_needs_gpu(panel):
    assert not panel.half.isEnabled()
    panel.set_detected_device("GPU: NVIDIA RTX 4090")
    assert panel.half.isEnabled()


def test_att_yolo_locks_size_and_dynamic(window, panel, qapp):
    untrained_run(window.project, model="att-yolo-n")
    panel.refresh_runs()
    att_index = next(i for i, r in enumerate(panel.runs) if r.spec.train.model == "att-yolo-n")
    panel.run_combo.setCurrentIndex(att_index)
    assert not panel.dynamic.isEnabled() and not panel.imgsz.isEnabled()
    assert panel.att_note.isVisibleTo(panel) and "attention" in panel.att_note.text()
    assert "att-yolo-n" in panel.name.text()


def test_pattern_setting(window, panel):
    assert window.set_export_pattern("{model}-{date}")
    assert panel.name.text() == "yolo11n-20261013.onnx"
    assert window.settings.value("export/name_pattern") == "{model}-{date}"
    warnings = []
    import defect_inspection.ui.main_window as mw

    original = mw.QMessageBox.warning
    mw.QMessageBox.warning = lambda *a: warnings.append(a[2])
    try:
        assert not window.set_export_pattern("{projekt}")
    finally:
        mw.QMessageBox.warning = original
    assert warnings and panel.pattern == "{model}-{date}"


def test_pattern_restored_from_settings(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("defect_inspection.core.exporter.datetime", FixedNow)
    settings = QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)
    settings.setValue("export/name_pattern", "{project}-{time}")
    p = make_project(tmp_path)
    untrained_run(p)
    w = MainWindow(settings, cache_root=tmp_path / "c")
    w.load_project(p)
    try:
        assert w.export_panel.name.text() == "PCB_라인-143025.onnx"
    finally:
        w.project_dirty = False
        w.close()


def test_new_run_selected_after_training(window, panel):
    run = untrained_run(window.project, model="yolo11s")
    window._on_run_finished(str(run), "finished")
    assert panel.current_run.path == run and "yolo11s" in panel.name.text()
