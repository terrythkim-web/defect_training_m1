"""Train tab (PRD F4.1-F4.7): settings form, start/stop/resume, progress, live charts, log, history."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.project import Project
from defect_inspection.core.trainer import (
    EpochSeries,
    RunInfo,
    TrainConfig,
    load_spec,
    log_header,
    prepare_run,
    read_events,
    read_run_log,
)
from defect_inspection.ui.train_events import MAIN_METRIC, RunLog, describe_event, fmt_duration
from defect_inspection.ui.train_process import TrainProcess
from defect_inspection.ui.widgets.run_history import STATE_LABELS, RunHistoryTable, display_state
from defect_inspection.ui.widgets.train_settings_form import TrainSettingsForm
from defect_inspection.ui.widgets.training_charts import TrainingCharts

__all__ = ["MAIN_METRIC", "STATE_LABELS", "TrainPanel", "fmt_duration"]


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
        self._run_log = RunLog()  # active run's train.log (every line shown in the log tab is saved)
        # Main window hook: save pending edits and refuse while a dataset build runs.
        self.before_start: Callable[[], bool] = lambda: True
        self._build_ui()
        self.set_project(None)

    # ---- UI --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.settings_box = TrainSettingsForm()
        self._alias_form_widgets()

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

        layout = QVBoxLayout(self)
        layout.addWidget(self.settings_box)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self._build_output(), 1)

    def _alias_form_widgets(self) -> None:
        """The form's widgets stay reachable as panel attributes (main window, tests)."""
        form = self.settings_box
        for name in ("model_combo", "model_note", "task_label", "dataset_combo", "epochs", "imgsz", "batch",
                     "lr0", "patience", "device", "workers", "online_aug", "online_warning", "paper_hyp",
                     "pretrained"):
            setattr(self, name, getattr(form, name))

    def _build_output(self) -> QSplitter:
        """Charts above, log / history tabs below."""
        self.charts = TrainingCharts()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.history = RunHistoryTable()
        self.history.itemSelectionChanged.connect(self._on_history_selected)
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.log_view, self.tr("로그"))
        self.bottom_tabs.addTab(self.history, self.tr("학습 이력"))

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.charts)
        split.addWidget(self.bottom_tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        return split

    # ---- project / form -------------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.settings_box.set_project(project)
        self.refresh_history()
        if not self.running:
            self.charts.clear()
            self.log_view.clear()

    @property
    def gpu_available(self) -> bool:
        return self.settings_box.gpu_available

    def set_detected_device(self, text: str) -> None:
        self.settings_box.set_detected_device(text)

    def refresh_datasets(self) -> None:
        self.settings_box.refresh_datasets()
        self._update_buttons()

    def current_config(self) -> TrainConfig:
        return self.settings_box.current_config()

    def select_model(self, key: str) -> None:
        self.settings_box.select_model(key)

    # ---- run control ----------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.process.running

    def _update_buttons(self) -> None:
        has, running = self.project is not None, self.running
        self.btn_start.setEnabled(has and not running and self.settings_box.dataset_available())
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
        try:
            self._run_log.open(run)
        except OSError as e:
            self.log_view.appendPlainText(self.tr("로그 파일을 열 수 없습니다: {0}").format(e))
        self._log(log_header(resume))
        if not resume:
            self._log_dataset_warnings(run)
        self.progress.setValue(0)
        self._log(self.tr("{0}: {1}").format(self.tr("재개") if resume else self.tr("시작"), run.name))
        self.process.start(run, resume=resume)
        self.status_label.setText(self.tr("학습 프로세스 시작 중…"))
        self.refresh_history()

    def _log_dataset_warnings(self, run: Path) -> None:
        """The dataset's Val reliability warnings (dataset.val_warnings), so they stay in train.log."""
        from defect_inspection.core.crop import read_build_info

        try:
            info = read_build_info(Path(load_spec(run).data_yaml).parent)
        except (OSError, ValueError):
            return
        for warning in (info.get("report") or {}).get("warnings") or []:
            self._log(self.tr("⚠ 데이터셋: {0}").format(warning))

    def stop(self) -> None:
        self.process.stop()
        self.status_label.setText(self.tr("중지하는 중… (현재 배치가 끝나면 멈춥니다)"))

    def shutdown(self, timeout_ms: int = 25_000) -> None:
        """Stop a running training and wait (used when the app closes)."""
        if self.running:
            self.process.stop()
            if not self.process.wait(timeout_ms):
                self.process.kill()
                self.process.wait(5000)

    # ---- events --------------------------------------------------------------------------
    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        self._run_log.write(text)

    def _on_event(self, e: dict) -> None:
        view = describe_event(e, None if self.project is None else self.project.channels)
        if view.chart:
            self.charts.add_event(e)
        if view.progress is not None:
            value, maximum = view.progress
            self.progress.setMaximum(maximum)
            if value is not None:
                self.progress.setValue(value)
        if view.status is not None:
            self.status_label.setText(view.status)
        for line in view.logs:
            self._log(line)

    def _on_finished(self, state: str) -> None:
        run = self.active_run
        self.status_label.setText(self.tr("학습 {0}").format(self.tr(STATE_LABELS.get(state, state))))
        self._run_log.close()
        self.active_run = None
        self.refresh_history()
        if run is not None:
            self.run_finished.emit(str(run), state)

    # ---- history (F4.7) -----------------------------------------------------------------------
    def display_state(self, run: RunInfo) -> str:
        return display_state(run, self.active_run)

    def refresh_history(self) -> None:
        self.history.refresh(self.project, self.active_run)
        self._update_buttons()

    def selected_run(self) -> RunInfo | None:
        return self.history.selected_run()

    def _on_history_selected(self) -> None:
        run = self.selected_run()
        if run is not None and not self.running:
            self.charts.show_series(EpochSeries.from_events(read_events(run.path)))
            log = read_run_log(run.path)
            self.log_view.setPlainText(log.rstrip("\n") if log is not None else self.tr("(이 학습의 저장된 로그가 없습니다)"))
        self._update_buttons()
