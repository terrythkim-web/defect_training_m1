"""Train-tab history table (PRD F4.7): one row per run with model, state, epoch and mAP50-95."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget

from defect_inspection.core.project import Project
from defect_inspection.core.trainer import RunInfo, list_runs
from defect_inspection.ui.train_events import MAIN_METRIC

STATE_LABELS = {
    "queued": "대기",
    "running": "학습 중",
    "finished": "완료",
    "stopped": "중지됨",
    "failed": "실패",
    "interrupted": "중단됨",
}


def display_state(run: RunInfo, active_run: Path | None) -> str:
    if run.state == "running" and (active_run is None or run.path != active_run):
        return "interrupted"  # left "running" by a previous session that did not end cleanly
    return run.state


class RunHistoryTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 5, parent)
        self.runs: list[RunInfo] = []
        self.setHorizontalHeaderLabels([self.tr("run"), self.tr("모델"), self.tr("상태"), "epoch", "mAP50-95"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)

    def refresh(self, project: Project | None, active_run: Path | None) -> None:
        """Reload the runs; selection signals are blocked while the rows are rewritten."""
        self.runs = [] if project is None else list_runs(project)
        self.blockSignals(True)
        self.setRowCount(len(self.runs))
        for row, run in enumerate(self.runs):
            for col, text in enumerate(self._cells(run, active_run)):
                self.setItem(row, col, QTableWidgetItem(text))
        self.blockSignals(False)

    def _cells(self, run: RunInfo, active_run: Path | None) -> list[str]:
        status = run.status
        metric = (status.get("final_metrics") or status.get("metrics") or {}).get(MAIN_METRIC)
        state = display_state(run, active_run)
        label = self.tr(STATE_LABELS.get(state, state))
        if state in ("stopped", "interrupted", "failed") and run.resumable:
            label += self.tr(" (재개 가능)")
        return [run.run_id, run.spec.train.model if run.spec else "?", label,
                f"{status.get('epoch', 0)}/{status.get('epochs', run.spec.train.epochs if run.spec else '?')}",
                "-" if metric is None else f"{metric:.3f}"]

    def selected_run(self) -> RunInfo | None:
        rows = sorted({i.row() for i in self.selectedIndexes()})
        return self.runs[rows[0]] if len(rows) == 1 and rows[0] < len(self.runs) else None
