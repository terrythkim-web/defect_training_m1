"""Train tab (PRD F4.1-F4.7): settings form, start/stop/resume, progress, live charts, log, history."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.model_zoo import DEFAULT_MODEL, MODEL_CHOICES, PAPER_HYP, get_choice
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import (
    EpochSeries,
    RunInfo,
    TrainConfig,
    list_runs,
    log_header,
    open_run_log,
    prepare_run,
    read_events,
    read_run_log,
)
from defect_inspection.platform_utils import default_dataloader_workers, gpu_device_choice
from defect_inspection.ui.train_process import TrainProcess
from defect_inspection.ui.widgets.training_charts import TrainingCharts

STATE_LABELS = {
    "queued": "대기",
    "running": "학습 중",
    "finished": "완료",
    "stopped": "중지됨",
    "failed": "실패",
    "interrupted": "중단됨",
}
MAIN_METRIC = "metrics/mAP50-95(B)"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class TrainPanel(QWidget):
    run_finished = Signal(str, str)  # run dir, final state

    def __init__(self, weights_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.weights_dir = Path(weights_dir)
        self.project: Project | None = None
        self.process = TrainProcess(self)
        self.process.event.connect(self._on_event)
        self.process.log.connect(self._log)
        self.process.finished.connect(self._on_finished)
        self.active_run: Path | None = None
        self._log_file = None  # active run's train.log (every line shown in the log tab is saved)
        self.gpu_available = False
        # Main window hook: save pending edits and refuse while a dataset build runs.
        self.before_start: Callable[[], bool] = lambda: True
        self._build_ui()
        self.set_project(None)

    # ---- UI --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.model_combo = QComboBox()
        for c in MODEL_CHOICES:
            self.model_combo.addItem(c.label, c.key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet("color: #8a5300;")
        self.task_label = QLabel()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Full (전체 이미지)", "full")
        self.dataset_combo.addItem("Crop (잘라낸 영역)", "crop")

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(64, 4096)
        self.imgsz.setSingleStep(32)
        self.batch = QSpinBox()
        self.batch.setRange(-1, 1024)
        self.batch.setSpecialValueText(self.tr("자동"))
        self.lr0 = QDoubleSpinBox()
        self.lr0.setRange(0.0, 1.0)
        self.lr0.setDecimals(5)
        self.lr0.setSingleStep(0.001)
        self.lr0.setSpecialValueText(self.tr("기본"))
        self.patience = QSpinBox()
        self.patience.setRange(0, 5000)
        self.patience.setSpecialValueText(self.tr("끔"))
        self.patience.setValue(50)
        self.device = QComboBox()
        self.device.addItem(self.tr("자동"), "auto")
        self.device.addItem("CPU", "cpu")
        self.device.addItem(*gpu_device_choice())  # "GPU 0" (CUDA) or "Apple GPU (MPS)" on a Mac
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setValue(default_dataloader_workers())
        self.online_aug = QCheckBox(self.tr("Ultralytics 내장 증강 (mosaic 등)"))
        self.online_aug.setChecked(True)
        self.online_aug.toggled.connect(self._update_online_warning)
        self.online_warning = QLabel()
        self.online_warning.setWordWrap(True)
        self.online_warning.setStyleSheet("color: #8a5300;")
        self.paper_hyp = QCheckBox(self.tr("논문 하이퍼파라미터 (ATT-YOLO Table 6)"))
        self.paper_hyp.setToolTip(", ".join(f"{k}={v}" for k, v in PAPER_HYP.items()))
        self.pretrained = QCheckBox(self.tr("사전학습 가중치 사용"))
        self.pretrained.setChecked(True)

        form = QFormLayout()
        form.addRow(self.tr("모델"), self.model_combo)
        form.addRow("", self.model_note)
        form.addRow(self.tr("태스크"), self.task_label)
        form.addRow(self.tr("데이터셋"), self.dataset_combo)
        grid = QHBoxLayout()
        left, right = QFormLayout(), QFormLayout()
        left.addRow("epochs", self.epochs)
        left.addRow("imgsz", self.imgsz)
        left.addRow("batch", self.batch)
        left.addRow("lr0", self.lr0)
        right.addRow("patience", self.patience)
        right.addRow(self.tr("장치"), self.device)
        right.addRow("workers", self.workers)
        grid.addLayout(left)
        grid.addLayout(right)
        form.addRow(grid)
        form.addRow(self.online_aug)
        form.addRow(self.online_warning)
        form.addRow(self.paper_hyp)
        form.addRow(self.pretrained)
        self.settings_box = QGroupBox(self.tr("학습 설정"))
        self.settings_box.setLayout(form)

        self.btn_start = QPushButton(self.tr("학습 시작"))
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton(self.tr("중지"))
        self.btn_stop.clicked.connect(self.stop)
        self.btn_resume = QPushButton(self.tr("선택한 run 재개"))
        self.btn_resume.clicked.connect(self.resume_selected)
        buttons = QHBoxLayout()
        for b in (self.btn_start, self.btn_stop, self.btn_resume):
            buttons.addWidget(b)

        self.progress = QProgressBar()
        self.progress.setFormat("epoch %v / %m")
        self.status_label = QLabel()

        self.charts = TrainingCharts()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.history = QTableWidget(0, 5)
        self.history.setHorizontalHeaderLabels([self.tr("run"), self.tr("모델"), self.tr("상태"), "epoch", "mAP50-95"])
        self.history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history.verticalHeader().setVisible(False)
        self.history.itemSelectionChanged.connect(self._on_history_selected)
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.log_view, self.tr("로그"))
        self.bottom_tabs.addTab(self.history, self.tr("학습 이력"))

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.charts)
        split.addWidget(self.bottom_tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.settings_box)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(split, 1)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(self.model_combo.findData(DEFAULT_MODEL))
        self.model_combo.blockSignals(False)
        self._on_model_changed()

    # ---- project / form -------------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        if project is not None:
            mode = self.tr("그레이스케일 → 1채널 입력 모델") if project.channels == 1 else self.tr("컬러 → 3채널 입력 모델")
            self.task_label.setText(f"{'detect (BBox)' if project.config.task == 'detect' else 'seg (Polygon)'} · {mode}")
        else:
            self.task_label.setText("-")
        self.refresh_datasets()
        self._update_online_warning()
        self.refresh_history()
        self._update_buttons()
        if not self.running:
            self.charts.clear()
            self.log_view.clear()

    def set_detected_device(self, text: str) -> None:
        self.gpu_available = text.startswith("GPU")
        idx = self.device.findData(gpu_device_choice()[1])
        self.device.model().item(idx).setEnabled(self.gpu_available)
        self.device.setItemText(0, self.tr("자동 ({0})").format(text))

    def refresh_datasets(self) -> None:
        for i in range(self.dataset_combo.count()):
            key = self.dataset_combo.itemData(i)
            ok = self.project is not None and (self.project.dataset_dir / key / "data.yaml").is_file()
            self.dataset_combo.model().item(i).setEnabled(ok)
            base = self.dataset_combo.itemText(i).split(" — ")[0]
            self.dataset_combo.setItemText(i, base if ok else base + self.tr(" — 빌드 안 됨"))
        self._update_buttons()

    def _on_model_changed(self) -> None:
        choice = get_choice(self.model_combo.currentData())
        self.epochs.setValue(choice.default_epochs)
        self.imgsz.setValue(choice.default_imgsz)
        self.model_note.setText(self.tr(choice.note))
        self.model_note.setVisible(bool(choice.note))
        self.paper_hyp.setChecked(False)

    def _update_online_warning(self) -> None:
        offline = [] if self.project is None else self.project.config.augment.enabled_keys()
        show = self.online_aug.isChecked() and bool(offline)
        self.online_warning.setText(self.tr(
            "오프라인 증강({0}종)이 이미 데이터셋에 들어 있어, 내장 증강까지 켜면 뒤집기·밝기 같은 변환이 "
            "두 번 적용됩니다. 결과가 나빠지면 끄고 비교해 보세요.").format(len(offline)))
        self.online_warning.setVisible(show)

    def current_config(self) -> TrainConfig:
        return TrainConfig(
            model=self.model_combo.currentData(),
            dataset=self.dataset_combo.currentData(),
            epochs=self.epochs.value(),
            imgsz=self.imgsz.value(),
            batch=self.batch.value(),
            lr0=self.lr0.value() or None,
            patience=self.patience.value(),
            device=self.device.currentData(),
            workers=self.workers.value(),
            online_augment=self.online_aug.isChecked(),
            paper_hyp=self.paper_hyp.isChecked(),
            pretrained=self.pretrained.isChecked(),
            seed=0 if self.project is None else self.project.config.split.seed,
        )

    def select_model(self, key: str) -> None:
        self.model_combo.setCurrentIndex(self.model_combo.findData(key))

    # ---- run control ----------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.process.running

    def _update_buttons(self) -> None:
        has, running = self.project is not None, self.running
        dataset_ok = has and self.dataset_combo.model().item(self.dataset_combo.currentIndex()).isEnabled()
        self.btn_start.setEnabled(has and not running and dataset_ok)
        self.btn_stop.setEnabled(running)
        sel = self.selected_run()
        self.btn_resume.setEnabled(has and not running and sel is not None and sel.resumable)
        self.settings_box.setEnabled(has and not running)

    def start(self, config: TrainConfig | None = None) -> bool:
        """Start a run with the form's settings, or with `config` (e.g. the quick suggestion run)."""
        if self.project is None or self.running or not self.before_start():
            return False
        try:
            run = prepare_run(self.project, config or self.current_config(), self.weights_dir)
        except ValueError as e:
            QMessageBox.warning(self, self.tr("학습 시작"), str(e))
            return False
        self._launch(run, resume=False)
        return True

    def resume_selected(self) -> bool:
        run = self.selected_run()
        if run is None or not run.resumable or self.running or not self.before_start():
            return False
        self._launch(run.path, resume=True)
        self.charts.show_series(EpochSeries.from_events(read_events(run.path)))
        return True

    def _launch(self, run: Path, resume: bool) -> None:
        self.active_run = run
        if not resume:
            self.charts.clear()
        # A new run starts with an empty log; a resume shows the run's earlier attempts first.
        self.log_view.setPlainText("" if not resume else (read_run_log(run) or "").rstrip("\n"))
        self._close_log_file()
        try:
            self._log_file = open_run_log(run)
        except OSError as e:
            self.log_view.appendPlainText(self.tr("로그 파일을 열 수 없습니다: {0}").format(e))
        self._log(log_header(resume))
        self.progress.setValue(0)
        self._log(self.tr("{0}: {1}").format(self.tr("재개") if resume else self.tr("시작"), run.name))
        self.process.start(run, resume=resume)
        self.status_label.setText(self.tr("학습 프로세스 시작 중…"))
        self.refresh_history()
        self._update_buttons()

    def stop(self) -> None:
        self.process.stop()
        self.status_label.setText(self.tr("중지하는 중… (현재 배치가 끝나면 멈춥니다)"))

    def shutdown(self, timeout_ms: int = 25_000) -> None:
        """Stop a running training and wait (used when the app closes)."""
        if self.running:
            self.process.stop()
            if not self.process.wait(timeout_ms):
                self.process._kill()
                self.process.wait(5000)

    # ---- events --------------------------------------------------------------------------
    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        if self._log_file is not None:
            try:
                self._log_file.write(text + "\n")
                self._log_file.flush()  # a crash of the app must not lose the log
            except OSError:
                self._close_log_file()

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def _on_event(self, e: dict) -> None:
        kind = e.get("type")
        if kind == "start":
            self._log(self.tr("학습 시작: {0} (epochs {1}, PID {2})").format(e.get("model"), e.get("epochs"), e.get("pid")))
            self.progress.setMaximum(int(e.get("epochs") or 1))
        elif kind == "model":
            ch = e.get("channels")
            mode = self.tr("그레이스케일") if ch == 1 else self.tr("컬러")
            params = e.get("params") or 0
            self._log(self.tr("모델 입력 채널: {0} ({1}), 파라미터 {2:.1f}M").format(ch, mode, params / 1e6))
            if self.project is not None and ch != self.project.channels:
                self._log(self.tr("경고: 프로젝트는 {0}채널인데 모델은 {1}채널입니다").format(self.project.channels, ch))
        elif kind == "transfer":
            self._log(self.tr("사전학습 가중치 전이: {0}").format(e.get("summary")))
        elif kind == "train_epoch":
            loss = e.get("losses", {}).get("train/box_loss")
            if loss is not None:
                self.status_label.setText(self.tr("epoch {0}/{1} 학습 완료, 검증 중… (box loss {2:.4f})")
                                          .format(e["epoch"], e["epochs"], loss))
        elif kind == "epoch":
            self.charts.add_event(e)
            self.progress.setMaximum(e["epochs"])
            self.progress.setValue(e["epoch"])
            m = e.get("metrics", {})
            main = m.get(MAIN_METRIC, m.get("metrics/mAP50-95(M)"))
            self.status_label.setText(self.tr("epoch {0}/{1} · 경과 {2} · 남은 시간 약 {3}{4}").format(
                e["epoch"], e["epochs"], fmt_duration(e.get("elapsed_s")), fmt_duration(e.get("eta_s")),
                "" if main is None else f" · mAP50-95 {main:.3f}"))
            self._log(self.tr("epoch {0}/{1}: {2}").format(
                e["epoch"], e["epochs"], ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in m.items()
                                                   if k.startswith("metrics/"))))
        elif kind == "final":
            m = e.get("metrics", {})
            self._log(self.tr("best.pt 최종 검증: {0}").format(
                ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in m.items() if k.startswith("metrics/"))))
        elif kind == "done":
            self._log(self.tr("학습 완료: {0}").format(e.get("best")))
        elif kind == "stopped":
            self._log(self.tr("중지됨 (재개 {0})").format(self.tr("가능") if e.get("resumable") else self.tr("불가")))
        elif kind == "error":
            self._log(self.tr("오류: {0}").format(e.get("message")))
            if e.get("traceback"):
                self._log(e["traceback"])

    def _on_finished(self, state: str) -> None:
        run = self.active_run
        self.status_label.setText(self.tr("학습 {0}").format(self.tr(STATE_LABELS.get(state, state))))
        self._close_log_file()
        self.active_run = None
        self.refresh_history()
        self._update_buttons()
        if run is not None:
            self.run_finished.emit(str(run), state)

    # ---- history (F4.7) -----------------------------------------------------------------------
    def display_state(self, run: RunInfo) -> str:
        if run.state == "running" and (self.active_run is None or run.path != self.active_run):
            return "interrupted"  # left "running" by a previous session that did not end cleanly
        return run.state

    def refresh_history(self) -> None:
        self._runs = [] if self.project is None else list_runs(self.project)
        self.history.blockSignals(True)
        self.history.setRowCount(len(self._runs))
        for row, run in enumerate(self._runs):
            status = run.status
            metric = (status.get("final_metrics") or status.get("metrics") or {}).get(MAIN_METRIC)
            state = self.display_state(run)
            label = self.tr(STATE_LABELS.get(state, state))
            if state in ("stopped", "interrupted", "failed") and run.resumable:
                label += self.tr(" (재개 가능)")
            cells = [run.run_id, run.spec.train.model if run.spec else "?", label,
                     f"{status.get('epoch', 0)}/{status.get('epochs', run.spec.train.epochs if run.spec else '?')}",
                     "-" if metric is None else f"{metric:.3f}"]
            for col, text in enumerate(cells):
                self.history.setItem(row, col, QTableWidgetItem(text))
        self.history.blockSignals(False)
        self._update_buttons()

    def selected_run(self) -> RunInfo | None:
        rows = sorted({i.row() for i in self.history.selectedIndexes()})
        runs = getattr(self, "_runs", [])
        return runs[rows[0]] if len(rows) == 1 and rows[0] < len(runs) else None

    def _on_history_selected(self) -> None:
        run = self.selected_run()
        if run is not None and not self.running:
            self.charts.show_series(EpochSeries.from_events(read_events(run.path)))
            log = read_run_log(run.path)
            self.log_view.setPlainText(log.rstrip("\n") if log is not None else self.tr("(이 학습의 저장된 로그가 없습니다)"))
        self._update_buttons()
