"""CoreML (.mlpackage) export option (PROMPTS step 22, PRD 5.2).

Core ML conversion and inference need macOS, so here Ultralytics' converter is replaced by a fake
that writes an .mlpackage folder; everything around it (macOS gate, names, sidecars, metadata,
UI switching) is real. The conversion itself is checked on a Mac (docs/mac_checklist.md).
"""

import json

import pytest

from defect_inspection.core import exporter as X
from test_exporter import make_project, untrained_run


class FakeYOLO:
    """Stands in for ultralytics.YOLO in exporter: export(format="coreml") makes an .mlpackage folder."""

    calls: list = []

    def __init__(self, path, task=None):
        self.path, self.task = path, task

    def export(self, **kwargs):
        FakeYOLO.calls.append(kwargs)
        from pathlib import Path

        out = Path(self.path).with_suffix(".mlpackage")
        (out / "Data" / "com.apple.CoreML").mkdir(parents=True, exist_ok=True)
        (out / "Manifest.json").write_text("{}", encoding="utf-8")
        (out / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"x" * 4096)
        return str(out)


@pytest.fixture
def run(tmp_path):
    p = make_project(tmp_path)
    return p, untrained_run(p)


def test_coreml_needs_macos(run, monkeypatch):
    p, r = run
    monkeypatch.setattr(X, "COREML_EXPORT_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="macOS"):
        X.export_run(p, r, p.exports_dir / "m", X.ExportOptions(format="coreml"))


def test_coreml_export_files_and_metadata(run, monkeypatch):
    p, r = run
    monkeypatch.setattr(X, "COREML_EXPORT_AVAILABLE", True)
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    FakeYOLO.calls = []
    result = X.export_run(p, r, p.exports_dir / "라인 A.onnx",
                          X.ExportOptions(format="coreml", half=True, nms=True, dynamic=True, opset=17), verify=False)
    assert result.onnx_path == p.exports_dir / "라인 A.mlpackage" and result.onnx_path.is_dir()
    assert (result.onnx_path / "Manifest.json").is_file()
    call = FakeYOLO.calls[-1]
    assert call["format"] == "coreml" and call["half"] and call["nms"] and call["device"] == "cpu"
    assert any("dynamic" in n for n in result.notes)  # ONNX-only options are dropped with a note
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.metadata_path.name == "라인 A.metadata.json" and result.classes_path.name == "라인 A.classes.txt"
    assert meta["format"] == "coreml" and meta["input"]["weights"] == "float16"
    assert "ImageType" in meta["preprocess"]["normalize"] and "coordinates" in meta["output"]
    # exporting again replaces the folder instead of nesting a second package inside it
    result = X.export_run(p, r, p.exports_dir / "라인 A.mlpackage", X.ExportOptions(format="coreml"), verify=False)
    assert not (result.onnx_path / "라인 A.mlpackage").exists() and not (result.onnx_path / "best.mlpackage").exists()


def test_coreml_refuses_grayscale(tmp_path, monkeypatch):
    p = make_project(tmp_path, color_mode="grayscale")
    r = untrained_run(p)
    monkeypatch.setattr(X, "COREML_EXPORT_AVAILABLE", True)
    with pytest.raises(ValueError, match="3채널"):
        X.export_run(p, r, p.exports_dir / "g", X.ExportOptions(format="coreml"))


def test_export_tab_blocks_coreml_for_grayscale(qapp, tmp_path):
    from PySide6.QtCore import QSettings

    from defect_inspection.ui.main_window import MainWindow

    p = make_project(tmp_path, color_mode="grayscale")
    untrained_run(p)
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "c")
    try:
        w.load_project(p)
        panel = w.export_panel
        panel.set_coreml_available(True)
        assert panel.problem() is None and panel.btn_export.isEnabled()  # ONNX is fine
        panel.format_combo.setCurrentIndex(panel.format_combo.findData("coreml"))
        # told before clicking (it used to be an error dialog after the export started)
        assert "3채널" in panel.problem() and "3채널" in panel.name_problem.text()
        assert not panel.btn_export.isEnabled()
    finally:
        w.train_panel.shutdown()
        w.project_dirty = False
        w.close()


def test_suffix_helpers():
    assert X.with_export_suffix("a.onnx", "coreml") == "a.mlpackage"
    assert X.with_export_suffix("a.mlpackage", "onnx") == "a.onnx"
    assert X.with_export_suffix("a", "coreml") == "a.mlpackage"
    assert X.default_export_name("{model}", "p", "full", "yolo11n", fmt="coreml") == "yolo11n.mlpackage"


# ---- Export tab ------------------------------------------------------------------------------------

def test_export_tab_format_switch(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from defect_inspection.ui.main_window import MainWindow

    p = make_project(tmp_path)
    untrained_run(p)
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "c")
    try:
        w.load_project(p)
        panel = w.export_panel
        assert panel.format_combo.findData("coreml") == -1 or X.COREML_EXPORT_AVAILABLE  # Linux/Windows: ONNX only
        panel.set_coreml_available(True)
        assert panel.format_combo.findData("coreml") >= 0
        panel.format_combo.setCurrentIndex(panel.format_combo.findData("coreml"))
        assert panel.name.text().endswith(".mlpackage") and panel.btn_export.text() == "CoreML 내보내기"
        assert not panel.opset.isEnabled() and not panel.simplify.isEnabled() and not panel.dynamic.isEnabled()
        assert panel.half.isEnabled()  # CoreML FP16 needs no CUDA
        assert panel.current_options().format == "coreml"
        assert "metadata.json" in panel.sidecar_label.text()

        seen = []
        monkeypatch.setattr("defect_inspection.ui.panels.export_panel.export_run",
                            lambda project, run, path, options: seen.append((path, options)) or (_ for _ in ()).throw(RuntimeError("stop")))
        monkeypatch.setattr("defect_inspection.ui.panels.export_panel.QMessageBox.warning", lambda *a: None)
        assert panel.export()
        end = __import__("time").time() + 10
        while panel.busy and __import__("time").time() < end:
            qapp.processEvents()
        assert seen and seen[0][0].name.endswith(".mlpackage") and seen[0][1].format == "coreml"

        panel.format_combo.setCurrentIndex(panel.format_combo.findData("onnx"))
        assert panel.name.text().endswith(".onnx") and panel.opset.isEnabled() and not panel.half.isEnabled()
        panel.set_coreml_available(False)
        assert panel.format_combo.count() == 1 and not panel.format_combo.isVisible()
    finally:
        w.train_panel.shutdown()
        w.project_dirty = False
        w.close()


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
