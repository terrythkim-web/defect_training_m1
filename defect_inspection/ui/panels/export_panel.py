"""Export tab (PRD F6.1-F6.4): pick a run, set ONNX options and the file name, export and show the
onnxruntime verification. On macOS also CoreML (.mlpackage, PRD 5.2)."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRegularExpression, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.exporter import (
    DEFAULT_NAME_PATTERN,
    ExportOptions,
    ExportResult,
    default_export_name,
    export_run,
    sidecar_paths,
    with_export_suffix,
)
from defect_inspection.core.model_zoo import get_choice
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import RunInfo, list_runs
from defect_inspection.platform_utils import (
    COREML_EXPORT_AVAILABLE,
    FORBIDDEN_FILENAME_CHARS,
    invalid_filename_reason,
    is_cuda_label,
    path_too_long,
)

# Typing a forbidden character is simply refused (F6.3); control characters too.
_NAME_REGEX = "^[^" + QRegularExpression.escape(FORBIDDEN_FILENAME_CHARS) + "\\x00-\\x1f]*$"


def _size_bytes(path: Path) -> int:
    """Size of a file, or of a folder such as an .mlpackage."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


class ExportPanel(QWidget):
    exported = Signal(object)  # ExportResult
    _finished = Signal(object)  # ExportResult | Exception (from the worker thread)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self.runs: list[RunInfo] = []
        self.pattern = DEFAULT_NAME_PATTERN
        self.gpu_available = False
        self.last_result: ExportResult | None = None
        self._name_edited = False
        self._busy = False
        self._finished.connect(self._on_finished)
        self._build_ui()
        self.set_coreml_available(COREML_EXPORT_AVAILABLE)
        self.set_project(None)

    # ---- UI ------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        self.run_info = QLabel()
        self.run_info.setWordWrap(True)

        self.format_combo = QComboBox()
        self.format_combo.addItem("ONNX (.onnx)", "onnx")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(0, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setSpecialValueText(self.tr("학습 크기"))
        self.opset = QSpinBox()
        self.opset.setRange(11, 23)  # 11 = "자동"
        self.opset.setSpecialValueText(self.tr("자동"))
        self.dynamic = QCheckBox(self.tr("dynamic (입력 크기 가변)"))
        self.simplify = QCheckBox(self.tr("simplify (그래프 단순화)"))
        self.simplify.setChecked(True)
        self.half = QCheckBox(self.tr("FP16 (GPU 필요)"))
        self.nms = QCheckBox(self.tr("NMS 포함 (출력: 박스·점수·클래스)"))
        self.att_note = QLabel()
        self.att_note.setWordWrap(True)
        self.att_note.setStyleSheet("color: #8a5300;")
        opts = QFormLayout()
        opts.addRow(self.tr("형식"), self.format_combo)
        opts.addRow(self.tr("입력 크기"), self.imgsz)
        opts.addRow("opset", self.opset)
        for w in (self.dynamic, self.simplify, self.half, self.nms, self.att_note):
            opts.addRow(w)
        self.opts_box = opts_box = QGroupBox(self.tr("ONNX 옵션"))
        opts_box.setLayout(opts)

        self.folder = QLineEdit()
        self.folder.textChanged.connect(self._validate)
        self.btn_browse = QPushButton(self.tr("다른 이름으로…"))
        self.btn_browse.clicked.connect(self.choose_path_dialog)
        self.name = QLineEdit()
        self.name.setValidator(QRegularExpressionValidator(QRegularExpression(_NAME_REGEX), self.name))
        self.name.textEdited.connect(self._on_name_edited)
        self.name.textChanged.connect(self._validate)
        self.btn_new_name = QPushButton(self.tr("기본 이름"))
        self.btn_new_name.setToolTip(self.tr("패턴으로 이름을 다시 만들고, 내보낼 때 날짜·시간을 그 시점으로 맞춥니다"))
        self.btn_new_name.clicked.connect(self.reset_name)
        self.name_problem = QLabel()
        self.name_problem.setWordWrap(True)
        self.name_problem.setStyleSheet("color: #c0392b;")
        self.sidecar_label = QLabel()
        self.sidecar_label.setWordWrap(True)
        self.sidecar_label.setStyleSheet("color: gray;")
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.btn_browse)
        name_row = QHBoxLayout()
        name_row.addWidget(self.name, 1)
        name_row.addWidget(self.btn_new_name)
        dest = QFormLayout()
        dest.addRow(self.tr("폴더"), folder_row)
        dest.addRow(self.tr("파일 이름"), name_row)
        dest.addRow(self.name_problem)
        dest.addRow(self.sidecar_label)
        dest_box = QGroupBox(self.tr("저장 위치"))
        dest_box.setLayout(dest)

        self.btn_export = QPushButton(self.tr("ONNX 내보내기"))
        self.btn_export.clicked.connect(self.export)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # busy indicator
        self.progress.setVisible(False)
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_open_folder = QPushButton(self.tr("폴더 열기"))
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_open_folder.setVisible(False)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("모델(run)")))
        top.addWidget(self.run_combo, 1)
        layout.addLayout(top)
        layout.addWidget(self.run_info)
        layout.addWidget(opts_box)
        layout.addWidget(dest_box)
        layout.addWidget(self.btn_export)
        layout.addWidget(self.progress)
        layout.addWidget(self.result_label)
        layout.addWidget(self.btn_open_folder, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)

    # ---- state -------------------------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.folder.setText("" if project is None else str(project.exports_dir))
        self.result_label.clear()
        self.btn_open_folder.setVisible(False)
        self.refresh_runs()

    def set_pattern(self, pattern: str) -> None:
        default_export_name(pattern, "p", "full", "m")  # raises ValueError for a bad pattern
        self.pattern = pattern
        if not self._name_edited:
            self.reset_name()

    def set_detected_device(self, text: str) -> None:
        self.gpu_available = is_cuda_label(text)  # FP16 export needs CUDA (not Apple MPS)
        self._update_controls()

    def refresh_runs(self, select: str | Path | None = None) -> None:
        self.runs = [] if self.project is None else [r for r in list_runs(self.project) if r.best_weights]
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for r in self.runs:
            self.run_combo.addItem(f"{r.run_id}  ({r.spec.train.model if r.spec else '?'})", str(r.path))
        if select is not None:
            self.run_combo.setCurrentIndex(max(self.run_combo.findData(str(select)), 0))
        self.run_combo.blockSignals(False)
        self._on_run_changed()

    @property
    def current_run(self) -> RunInfo | None:
        idx = self.run_combo.currentIndex()
        return self.runs[idx] if 0 <= idx < len(self.runs) else None

    @property
    def is_att_yolo(self) -> bool:
        run = self.current_run
        return run is not None and run.spec is not None and get_choice(run.spec.train.model).family == "att-yolo"

    def _on_run_changed(self) -> None:
        run = self.current_run
        if run is None:
            self.run_info.setText(self.tr("학습이 끝난 run이 없습니다. Train 탭에서 학습하세요."))
        else:
            metrics = run.status.get("final_metrics") or run.status.get("metrics") or {}
            m = metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95(M)"))
            tiling = run.spec.tiling()
            self.run_info.setText(self.tr("{0} · 학습 크기 {1} · 데이터셋 {2}{3}").format(
                run.spec.train.model, run.spec.train.imgsz, run.spec.train.dataset,
                "" if m is None else f" · mAP50-95 {m:.3f}")
                + ("" if tiling is None else self.tr("\nCrop 모델: 타일 {0}px, overlap {1}px 타일링 추론 정보가 metadata에 들어갑니다").format(
                    tiling.tile_size, tiling.overlap_px)))
        self._name_edited = False
        self.reset_name()
        self._update_controls()

    # ---- format (ONNX / CoreML on macOS) --------------------------------------------------------
    def set_coreml_available(self, available: bool) -> None:
        """CoreML is offered on macOS only (coremltools converts and runs it there)."""
        idx = self.format_combo.findData("coreml")
        if available and idx < 0:
            self.format_combo.addItem(self.tr("CoreML (.mlpackage, macOS · Neural Engine)"), "coreml")
        elif not available and idx >= 0:
            self.format_combo.setCurrentIndex(0)
            self.format_combo.removeItem(idx)
        shown = self.format_combo.count() > 1
        self.format_combo.setVisible(shown)
        label = self.opts_box.layout().labelForField(self.format_combo)
        if label is not None:
            label.setVisible(shown)

    @property
    def export_format(self) -> str:
        return self.format_combo.currentData() or "onnx"

    def _on_format_changed(self) -> None:
        fmt = self.export_format
        coreml = fmt == "coreml"
        self.opts_box.setTitle(self.tr("CoreML 옵션") if coreml else self.tr("ONNX 옵션"))
        self.btn_export.setText(self.tr("CoreML 내보내기") if coreml else self.tr("ONNX 내보내기"))
        self.half.setText(self.tr("FP16 가중치") if coreml else self.tr("FP16 (GPU 필요)"))
        if self.name.text().strip():
            self.name.setText(with_export_suffix(self.name.text().strip(), fmt))
        self._update_controls()

    def _update_controls(self) -> None:
        has = self.project is not None and self.current_run is not None
        att = self.is_att_yolo
        for w in (self.imgsz, self.opset, self.simplify, self.nms, self.folder, self.name,
                  self.btn_browse, self.btn_new_name):
            w.setEnabled(has and not self._busy)
        coreml = self.export_format == "coreml"
        self.dynamic.setEnabled(has and not att and not coreml and not self._busy)
        self.imgsz.setEnabled(has and not att and not self._busy)
        for w in (self.opset, self.simplify):  # ONNX-only
            w.setEnabled(has and not coreml and not self._busy)
        if att or coreml:
            self.dynamic.setChecked(False)
        if att:
            self.imgsz.setValue(0)
        half_ok = self.gpu_available or coreml  # CoreML FP16 needs no CUDA
        self.half.setEnabled(has and half_ok and not self._busy)
        if not half_ok:
            self.half.setChecked(False)
        self.half.setToolTip("" if half_ok else self.tr("GPU가 없어 FP16 내보내기를 쓸 수 없습니다"))
        self.att_note.setText(self.tr("ATT-YOLO는 attention 때문에 학습 크기로 고정되고 dynamic을 쓸 수 없습니다") if att else "")
        self.att_note.setVisible(att)
        self._validate()

    # ---- name / path (F6.3) ------------------------------------------------------------------
    def default_name(self, now: datetime | None = None) -> str:
        run = self.current_run
        if self.project is None or run is None or run.spec is None:
            return ""
        return default_export_name(self.pattern, self.project.config.name, run.spec.train.dataset,
                                   run.spec.train.model, now, self.export_format)

    def reset_name(self) -> None:
        self.name.setText(self.default_name())
        self._name_edited = False

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True

    def target_path(self) -> Path | None:
        if not self.folder.text().strip() or not self.name.text().strip():
            return None
        return Path(self.folder.text().strip()) / with_export_suffix(self.name.text().strip(), self.export_format)

    def problem(self) -> str | None:
        """Why the current destination cannot be used, or None."""
        if self.current_run is None:
            return self.tr("내보낼 run이 없습니다")
        if self.export_format == "coreml" and self.project is not None and self.project.channels != 3:
            # same rule as exporter.export_run, shown before the user clicks
            return self.tr("CoreML은 컬러(3채널) 모델만 지원합니다. 그레이 모델은 ONNX로 내보내세요")
        path = self.target_path()
        if path is None:
            return self.tr("폴더와 파일 이름을 입력하세요")
        reason = invalid_filename_reason(path.name)
        if reason:
            return reason
        meta, classes = sidecar_paths(path)
        longest = max((path, meta, classes), key=lambda p: len(str(p.absolute())))
        if path_too_long(longest):
            return self.tr("경로가 너무 깁니다 ({0}자, Windows 한도 260자). 폴더나 이름을 줄이세요").format(
                len(str(longest.absolute())))
        if path.parent.exists() and not path.parent.is_dir():
            return self.tr("폴더 경로가 파일을 가리킵니다")
        return None

    def _validate(self) -> None:
        problem = self.problem() if self.project is not None else None
        self.name_problem.setText(problem or "")
        path = self.target_path()
        if path is not None and problem is None:
            meta, classes = sidecar_paths(path)
            self.sidecar_label.setText(self.tr("함께 저장: {0}, {1}").format(meta.name, classes.name))
        else:
            self.sidecar_label.setText("")
        self.btn_export.setEnabled(self.project is not None and problem is None and not self._busy)

    def choose_path_dialog(self) -> None:
        start = str(self.target_path() or (self.project.exports_dir if self.project else Path.home()))
        coreml = self.export_format == "coreml"
        path, _ = QFileDialog.getSaveFileName(self, self.tr("CoreML 저장") if coreml else self.tr("ONNX 저장"), start,
                                              "Core ML (*.mlpackage)" if coreml else "ONNX (*.onnx)",
                                              options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.set_target(Path(path))

    def set_target(self, path: Path) -> None:
        self.folder.setText(str(path.parent))
        self.name.setText(with_export_suffix(path.name, self.export_format))
        self._name_edited = True

    # ---- export ---------------------------------------------------------------------------------
    def current_options(self) -> ExportOptions:
        return ExportOptions(
            format=self.export_format,
            imgsz=self.imgsz.value() or None,
            opset=None if self.opset.value() == self.opset.minimum() else self.opset.value(),
            dynamic=self.dynamic.isChecked(),
            simplify=self.simplify.isChecked(),
            half=self.half.isChecked(),
            nms=self.nms.isChecked(),
        )

    @property
    def busy(self) -> bool:
        return self._busy

    def export(self) -> bool:
        if self.project is None or self._busy:
            return False
        if not self._name_edited:
            self.reset_name()  # the date and time in the default name are the export time (F6.3)
        if self.problem() is not None:
            return False
        path = self.target_path()
        existing = [p for p in (path, *sidecar_paths(path)) if p.exists()]
        if existing:
            answer = QMessageBox.question(
                self, self.tr("덮어쓰기"),
                self.tr("이미 있는 파일을 덮어씁니다:\n{0}\n\n계속할까요?").format("\n".join(p.name for p in existing)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return False
        project, run, options = self.project, self.current_run, self.current_options()
        self._busy = True
        self.progress.setVisible(True)
        self.result_label.setText(self.tr("내보내는 중… (변환 후 Core ML로 검증)") if options.format == "coreml"
                                  else self.tr("내보내는 중… (변환 후 onnxruntime으로 검증)"))
        self.btn_open_folder.setVisible(False)
        self._update_controls()

        def work() -> None:
            try:
                result = export_run(project, run.path, path, options)
            except Exception as e:  # noqa: BLE001 - shown to the user
                result = e
            try:
                self._finished.emit(result)
            except RuntimeError:
                pass

        threading.Thread(target=work, daemon=True, name="onnx-export").start()
        return True

    def _on_finished(self, result) -> None:
        self._busy = False
        self.progress.setVisible(False)
        if isinstance(result, Exception):
            self.result_label.setText(self.tr("내보내기 실패: {0}").format(result))
            QMessageBox.warning(self, self.tr("내보내기"), f"{type(result).__name__}: {result}")
        else:
            self.last_result = result
            self.result_label.setText(self.result_text(result))
            self.btn_open_folder.setVisible(True)
            self._name_edited = False  # next export gets a fresh default name
            self.exported.emit(result)
        self._update_controls()

    def result_text(self, r: ExportResult) -> str:
        size_mb = _size_bytes(r.onnx_path) / 1e6
        v = r.verification
        lines = [self.tr("저장: {0} ({1:.1f} MB, {2:.1f}초)").format(r.onnx_path, size_mb, r.seconds)]
        if v is not None:
            lines.append(self.tr("입력: {0} ({1}채널)").format(v.input_shape, v.input_channels))
            mark = "✓ PASS" if v.passed else "✗ FAIL"
            lines.append(f"{self.tr('검증')} {mark} — {v.message}")
            if v.max_score_diff is not None:
                lines.append(self.tr("점수 차이 최대 {0:.4f}").format(v.max_score_diff))
        lines += [self.tr("참고: {0}").format(n) for n in r.notes]
        meta, classes = sidecar_paths(r.onnx_path)
        lines.append(self.tr("함께 저장: {0}, {1}").format(meta.name, classes.name))
        return "\n".join(lines)

    def open_folder(self) -> None:
        if self.last_result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_result.onnx_path.parent)))
