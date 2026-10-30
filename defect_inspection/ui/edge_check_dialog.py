"""도구 → "테두리 검사" (PRD F9): every confirmed label judged by the edge rule. Labels are the truth,
so a failing label is a real defect the filter would hide — the user sees where the rule falls short
before turning it on. Runs `edge_filter.check_labels` on a worker thread (images are decoded there)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.edge_filter import LabelCheck, check_labels, settings_text
from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.core.project import Project

COLUMNS = ("이미지", "클래스", "박스 (x1, y1, x2, y2)", "결과", "찾은 선")


class EdgeCheckDialog(QDialog):
    open_image = Signal(str)  # double-clicked row → show the image on the canvas
    _done = Signal(object)  # list[LabelCheck] | Exception, from the worker thread

    def __init__(self, project: Project, cfg: EdgeFilterConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("테두리 검사 — 라벨 박스별 판정"))
        self.resize(760, 480)
        self.rows: list[LabelCheck] = []
        self._cancel = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edge-check")

        self.summary = QLabel(self.tr("라벨 박스 판정 중…"))
        self.summary.setWordWrap(True)
        self.only_failed = QCheckBox(self.tr("제외되는 라벨만"))
        self.only_failed.toggled.connect(self._fill)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([self.tr(c) for c in COLUMNS])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_double_click)

        head = QHBoxLayout()
        head.addWidget(self.summary, 1)
        head.addWidget(self.only_failed)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.tr("설정: {0}{1}").format(
            settings_text(cfg), "" if cfg.enabled else self.tr(" (필터는 꺼져 있음)"))))
        layout.addLayout(head)
        layout.addWidget(self.table, 1)
        layout.addWidget(QLabel(self.tr("더블클릭: 왼쪽 캔버스에서 이미지 보기")))

        self._done.connect(self._on_done)
        cfg, cancel = cfg.model_copy(), self._cancel
        self._pool.submit(self._run, project, cfg, cancel)

    def _run(self, project: Project, cfg: EdgeFilterConfig, cancel: threading.Event) -> None:
        try:
            payload = check_labels(project, cfg, cancel=cancel.is_set)
        except Exception as e:  # noqa: BLE001 - shown in the dialog
            payload = e
        try:
            self._done.emit(payload)
        except RuntimeError:
            pass  # dialog already deleted

    def _on_done(self, payload) -> None:
        if isinstance(payload, Exception):
            self.summary.setText(self.tr("검사 실패: {0}").format(payload))
            return
        self.rows = payload
        passed = sum(r.passed for r in payload)
        self.summary.setText(self.tr("테두리 위로 판정된 라벨 {0} / {1}개 · 제외되는 라벨은 필터를 켜면 숨겨지는 진짜 결함입니다")
                             .format(passed, len(payload)))
        self._fill()

    def shown_rows(self) -> list[LabelCheck]:
        return [r for r in self.rows if not (self.only_failed.isChecked() and r.passed)]

    def _fill(self) -> None:
        rows = self.shown_rows()
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            line = "-" if r.hit is None else self.tr("{0} {1:.0f} px").format(
                self.tr("가로선 y") if r.hit.axis == "h" else self.tr("세로선 x"), r.hit.position)
            texts = (r.path, r.class_name, ", ".join(f"{v:.0f}" for v in r.bounds),
                     self.tr("통과") if r.passed else self.tr("제외"), line)
            for c, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if c == 3 and not r.passed:
                    item.setForeground(Qt.GlobalColor.darkRed)
                self.table.setItem(i, c, item)

    def _on_double_click(self, row: int, _column: int) -> None:
        rows = self.shown_rows()
        if 0 <= row < len(rows):
            self.open_image.emit(rows[row].path)

    def closeEvent(self, event) -> None:
        self._cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
