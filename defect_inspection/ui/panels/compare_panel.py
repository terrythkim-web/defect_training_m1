"""Compare tab (PRD F8.1-F8.6): Full + Crop experiment, fair re-evaluation of 2+ runs on the same
original Val images, metrics table, overlaid curves, metric bars, side-by-side predictions, HTML
report and report history."""

from __future__ import annotations

import threading
from typing import Any

import pyqtgraph as pg
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core import compare as C
from defect_inspection.core import report as R
from defect_inspection.core.evaluate import load_training_image, match
from defect_inspection.core.preview import draw_comparison
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import RunInfo, list_runs
from defect_inspection.ui.canvas.image_canvas import numpy_to_qimage
from defect_inspection.ui.widgets.training_charts import GRID_ALPHA, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

BAR_METRICS = C.BAR_LABELS
SIDE = 420  # px of each side-by-side image


def _plot(title: str, x_label: str, y_label: str) -> pg.PlotWidget:
    w = pg.PlotWidget(background=SURFACE)
    w.setTitle(title, color=TEXT_PRIMARY, size="10pt")
    w.showGrid(x=True, y=True, alpha=GRID_ALPHA)
    for axis in ("left", "bottom"):
        ax = w.getAxis(axis)
        ax.setPen(pg.mkPen(TEXT_SECONDARY, width=1))
        ax.setTextPen(pg.mkPen(TEXT_SECONDARY))
        ax.enableAutoSIPrefix(False)  # 0.2..0.5 must not become "200..500 (x0.001)"
    w.setLabel("bottom", x_label, color=TEXT_SECONDARY)
    w.setLabel("left", y_label, color=TEXT_SECONDARY)
    w.setMouseEnabled(x=False, y=False)
    w.hideButtons()
    w.setMinimumHeight(180)
    return w


def _series_key(values: dict[str, list], kind: str) -> str | None:
    """Metric/loss key to overlay: box metrics first (seg runs also have mask (M) ones)."""
    candidates = {"map": ["metrics/mAP50-95(B)", "metrics/mAP50-95(M)"],
                  "loss": ["val/box_loss", "train/box_loss"]}[kind]
    return next((k for k in candidates if k in values), None)


class ComparePanel(QWidget):
    experiment_requested = Signal()  # main window: build full + crop, train both, then compare
    rerun_requested = Signal(list)  # list[TrainConfig] of a report: train them again and compare (F8.6)
    _report_done = Signal(int, object)  # generation, Path | Exception
    open_image = Signal(str)
    _done = Signal(int, object)  # generation, list[RunComparison] | Exception
    _progress = Signal(int, int, int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self.runs: list[RunInfo] = []
        self.results: list[C.RunComparison] = []
        self.result_dirs: list[str] = []
        self.reports: list[R.ReportInfo] = []
        self.last_report: Path | None = None
        self.open_url = QDesktopServices.openUrl  # test hook
        self._report_gen = 0
        self._report_busy = False
        # test hooks: how predictors are made, and which devices are timed (None = CPU + GPU if any)
        self.make_predictor: C.PredictorFactory = C.default_factory
        self.devices: list[tuple[str, str]] | None = None
        self._gen = 0
        self._cancel = threading.Event()
        self._busy = False
        self._done.connect(self._on_done)
        self._progress.connect(self._on_progress)
        self._report_done.connect(self._on_report_done)
        self._build_ui()
        self.set_project(None)

    # ---- UI ------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.btn_experiment = QPushButton(self.tr("Full + Crop 비교 학습"))
        self.btn_experiment.setToolTip(self.tr("Full·Crop 데이터셋을 다시 빌드하고, Train 탭의 모델·하이퍼파라미터·seed로 "
                                               "둘을 차례로 학습한 뒤 비교합니다"))
        self.btn_experiment.clicked.connect(self.experiment_requested.emit)
        self.exp_status = QLabel()
        self.exp_status.setWordWrap(True)
        exp = QHBoxLayout()
        exp.addWidget(self.btn_experiment)
        exp.addWidget(self.exp_status, 1)

        self.run_list = QListWidget()
        self.run_list.setMaximumHeight(110)
        self.run_list.itemChanged.connect(self._update_buttons)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.7)
        self.priority = QComboBox()
        self.priority.addItems(list(C.PRIORITY_METRICS))
        self.priority.currentIndexChanged.connect(self._update_summary)
        self.btn_compare = QPushButton(self.tr("비교"))
        self.btn_compare.clicked.connect(self.compare_checked)
        self.btn_cancel = QPushButton(self.tr("취소"))
        self.btn_cancel.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        opts = QHBoxLayout()
        for text, w in ((self.tr("conf"), self.conf), (self.tr("NMS IoU"), self.iou), (self.tr("우선 지표"), self.priority)):
            opts.addWidget(QLabel(text))
            opts.addWidget(w)
        opts.addStretch(1)
        actions = QHBoxLayout()  # own row: one row with the options was the widest part of the tab
        actions.addStretch(1)
        actions.addWidget(self.btn_compare)
        actions.addWidget(self.btn_cancel)
        pick = QVBoxLayout()
        pick_note = QLabel(self.tr("비교할 run (2개 이상 체크) — 모두 같은 원본 Val 이미지, 같은 conf/IoU로 다시 평가"))
        pick_note.setWordWrap(True)  # unwrapped, this sentence set the Compare tab's minimum width
        pick.addWidget(pick_note)
        pick.addWidget(self.run_list)
        pick.addLayout(opts)
        pick.addLayout(actions)
        pick.addWidget(self.progress)
        pick_box = QGroupBox(self.tr("run 비교"))
        pick_box.setLayout(pick)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)

        # metrics table
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(22)

        # charts: one legend row (run colours are the same in every chart)
        self.legend = QLabel()
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.bars = _plot(self.tr("지표 (같은 Val 원본, 같은 conf · AP = mAP)"), "", "")
        self.map_plot = _plot(self.tr("mAP50-95 (각 run 자체 val)"), "epoch", "mAP50-95")
        self.loss_plot = _plot(self.tr("box loss"), "epoch", "loss")
        self.pr_plot = _plot(self.tr("PR curve (IoU 0.5)"), "recall", "precision")
        self.pr_plot.setXRange(0, 1, padding=0.02)
        self.pr_plot.setYRange(0, 1, padding=0.02)
        charts = QWidget()
        cg = QGridLayout(charts)
        cg.setContentsMargins(0, 0, 0, 0)
        cg.addWidget(self.legend, 0, 0, 1, 2)
        cg.addWidget(self.bars, 1, 0)
        cg.addWidget(self.pr_plot, 1, 1)
        cg.addWidget(self.map_plot, 2, 0)
        cg.addWidget(self.loss_plot, 2, 1)

        # side by side
        self.side_a, self.side_b = QComboBox(), QComboBox()
        for c in (self.side_a, self.side_b):
            c.currentIndexChanged.connect(self._fill_diff_list)
        self.diff_list = QListWidget()
        self.diff_list.currentItemChanged.connect(self._show_side_by_side)
        self.diff_list.itemDoubleClicked.connect(
            lambda item: self.open_image.emit(item.data(Qt.ItemDataRole.UserRole)))
        self.view_a, self.view_b = QLabel(), QLabel()
        self.caption_a, self.caption_b = QLabel(), QLabel()
        for v in (self.view_a, self.view_b):
            v.setMinimumSize(140, 140)  # 2 × 200 px + the list made the tab too wide for 1366 px screens
            v.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        for c in (self.caption_a, self.caption_b):
            c.setWordWrap(True)
        heads = QHBoxLayout()
        heads.addWidget(QLabel("A"))
        heads.addWidget(self.side_a, 1)
        heads.addWidget(QLabel("B"))
        heads.addWidget(self.side_b, 1)
        pics = QGridLayout()
        pics.addWidget(self.caption_a, 0, 0)
        pics.addWidget(self.caption_b, 0, 1)
        pics.addWidget(self.view_a, 1, 0)
        pics.addWidget(self.view_b, 1, 1)
        pics.setRowStretch(1, 1)
        pics_w = QWidget()
        pics_w.setLayout(pics)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.diff_list)
        split.addWidget(pics_w)
        split.setStretchFactor(1, 3)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addLayout(heads)
        diff_note = QLabel(self.tr("차이가 큰 이미지부터 (파랑 = 정답, 주황 = 예측). 더블클릭하면 캔버스에서 엽니다"))
        diff_note.setWordWrap(True)
        sl.addWidget(diff_note)
        sl.addWidget(split, 1)

        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self.table, self.tr("지표"))
        self.result_tabs.addTab(charts, self.tr("그래프"))
        self.result_tabs.addTab(side, self.tr("예측 비교"))
        self.result_tabs.addTab(self._build_report_ui(), self.tr("리포트"))

        layout = QVBoxLayout(self)
        layout.addLayout(exp)
        layout.addWidget(pick_box)
        layout.addWidget(self.summary)
        layout.addWidget(self.result_tabs, 1)

    def _build_report_ui(self) -> QWidget:
        self.memo = QPlainTextEdit()
        self.memo.setPlaceholderText(self.tr("결론 / 메모 (리포트 ⑥에 들어갑니다)"))
        self.memo.setMaximumHeight(90)
        self.btn_report = QPushButton(self.tr("HTML 리포트 저장"))
        self.btn_report.clicked.connect(self.save_report)
        self.report_status = QLabel()
        self.report_status.setWordWrap(True)
        self.report_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        save = QHBoxLayout()
        save.addWidget(self.btn_report)
        save.addWidget(self.report_status, 1)

        self.report_list = QListWidget()
        self.report_list.currentRowChanged.connect(self._update_buttons)
        self.report_list.itemDoubleClicked.connect(lambda _item: self.open_report())
        self.btn_open_report = QPushButton(self.tr("열기"))
        self.btn_open_report.clicked.connect(self.open_report)
        self.btn_delete_report = QPushButton(self.tr("삭제"))
        self.btn_delete_report.clicked.connect(self.delete_report)
        self.btn_rerun = QPushButton(self.tr("같은 설정으로 재실험"))
        self.btn_rerun.setToolTip(self.tr("리포트에 기록된 run들의 학습 설정으로 데이터셋을 다시 빌드하고 다시 학습한 뒤 비교합니다"))
        self.btn_rerun.clicked.connect(self.rerun_report)
        buttons = QHBoxLayout()
        for b in (self.btn_open_report, self.btn_delete_report, self.btn_rerun):
            buttons.addWidget(b)
        buttons.addStretch(1)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.memo)
        lay.addLayout(save)
        lay.addWidget(QLabel(self.tr("<b>리포트 이력</b> (reports/ 폴더, 최신 순)")))
        lay.addWidget(self.report_list, 1)
        lay.addLayout(buttons)
        return w

    # ---- project / runs ------------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.cancel()
        self.project = project
        self.results = []
        self.summary.setText("" if project else self.tr("프로젝트를 열면 비교할 수 있습니다"))
        self._clear_results()
        self.refresh_runs()
        self.refresh_reports()

    def refresh_runs(self, check: list[str] | None = None) -> None:
        """List finished runs with best.pt; `check` = run dirs to tick (others keep their state)."""
        ticked = set(check or []) | {str(r.path) for r in self.checked_runs()}
        self.runs = [r for r in list_runs(self.project) if r.best_weights is not None and r.spec is not None] \
            if self.project is not None else []
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for r in self.runs:
            tiling = r.spec.tiling()
            extra = f" · 타일 {tiling.tile_size}" if tiling else ""
            item = QListWidgetItem(f"{r.run_id}  ({r.spec.train.dataset} · {r.spec.train.model} · "
                                   f"imgsz {r.spec.train.imgsz}{extra} · {r.state})")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if str(r.path) in ticked else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, str(r.path))
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)
        self._update_buttons()

    def checked_runs(self) -> list[RunInfo]:
        paths = {self.run_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.run_list.count())
                 if self.run_list.item(i).checkState() == Qt.CheckState.Checked}
        return [r for r in self.runs if str(r.path) in paths]

    def set_checked(self, run_dirs: list[str]) -> None:
        for i in range(self.run_list.count()):
            item = self.run_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in run_dirs
                               else Qt.CheckState.Unchecked)

    def _update_buttons(self) -> None:
        has = self.project is not None
        self.btn_compare.setEnabled(has and not self._busy and len(self.checked_runs()) >= 2)
        self.btn_cancel.setEnabled(self._busy)
        self.btn_experiment.setEnabled(has and not self._busy)
        self.btn_report.setEnabled(has and len(self.results) >= 2 and not self._busy and not self._report_busy)
        selected = self.selected_report() is not None
        self.btn_open_report.setEnabled(selected)
        self.btn_delete_report.setEnabled(selected)
        self.btn_rerun.setEnabled(selected and not self._busy and bool(self.selected_report().configs))

    @property
    def busy(self) -> bool:
        return self._busy

    # ---- compare (F8.2) ---------------------------------------------------------------------
    def compare_checked(self) -> bool:
        return self.compare([str(r.path) for r in self.checked_runs()])

    def compare(self, run_dirs: list[str]) -> bool:
        if self.project is None or self._busy or len(run_dirs) < 2:
            return False
        project, conf, iou = self.project, self.conf.value(), self.iou.value()
        factory, devices = self.make_predictor, self.devices
        self._pending_dirs = list(run_dirs)
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        cancel = self._cancel
        self._busy = True
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.summary.setText(self.tr("run {0}개를 같은 Val 원본 이미지로 다시 평가하는 중…").format(len(run_dirs)))
        self._update_buttons()

        def work() -> None:
            try:
                payload: Any = C.compare_runs(
                    project, run_dirs, conf, iou, factory, devices,
                    progress=lambda d, t, m: self._emit(self._progress, gen, d, t, m), cancel=cancel.is_set)
            except Exception as e:  # noqa: BLE001 - shown to the user
                payload = e
            self._emit(self._done, gen, payload)

        threading.Thread(target=work, daemon=True, name="compare").start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def shutdown(self) -> None:
        self.cancel()
        self._gen += 1

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # panel destroyed meanwhile
            pass

    def _on_progress(self, gen: int, done: int, total: int, name: str) -> None:
        if gen == self._gen:
            self.progress.setMaximum(max(total, 1))
            self.progress.setValue(done)
            self.progress.setFormat(f"%p%  {name}")

    def _on_done(self, gen: int, payload) -> None:
        if gen != self._gen:
            return
        self._busy = False
        self.progress.setVisible(False)
        self._update_buttons()
        if isinstance(payload, Exception):
            self.summary.setText(self.tr("비교 실패: {0}").format(payload))
            return
        if self._cancel.is_set():
            self.summary.setText(self.tr("취소했습니다"))
            return
        self.show_results(payload)

    # ---- results (F8.3, F8.4) -------------------------------------------------------------------
    def show_results(self, results: list[C.RunComparison], run_dirs: list[str] | None = None) -> None:
        self.results = results
        self.result_dirs = list(run_dirs if run_dirs is not None else getattr(self, "_pending_dirs", []))
        self._update_buttons()
        self._update_summary()
        self._fill_table()
        self._fill_charts()
        for combo, index in ((self.side_a, 0), (self.side_b, 1)):
            combo.blockSignals(True)
            combo.clear()
            for r in results:
                combo.addItem(r.label)
            combo.setCurrentIndex(min(index, len(results) - 1))
            combo.blockSignals(False)
        self._fill_diff_list()

    # ---- report (F8.5) and history (F8.6) --------------------------------------------------------
    def save_report(self) -> bool:
        if self.project is None or len(self.results) < 2 or self._report_busy:
            return False
        project, results, dirs = self.project, list(self.results), [Path(d) for d in self.result_dirs]
        priority, memo = self.priority.currentText(), self.memo.toPlainText()
        self._report_gen += 1
        gen = self._report_gen
        self._report_busy = True
        self.report_status.setText(self.tr("리포트 만드는 중…"))
        self._update_buttons()

        def work() -> None:
            try:
                payload: Any = R.build_report(project, results, dirs, priority, memo)
            except Exception as e:  # noqa: BLE001 - shown to the user
                payload = e
            self._emit(self._report_done, gen, payload)

        threading.Thread(target=work, daemon=True, name="report").start()
        return True

    def _on_report_done(self, gen: int, payload) -> None:
        if gen != self._report_gen:
            return
        self._report_busy = False
        if isinstance(payload, Exception):
            self.report_status.setText(self.tr("리포트 저장 실패: {0}").format(payload))
        else:
            self.last_report = payload
            self.report_status.setText(self.tr("저장: {0}").format(payload))
            self.refresh_reports(select=payload)
        self._update_buttons()

    @property
    def report_busy(self) -> bool:
        return self._report_busy

    def refresh_reports(self, select: Path | None = None) -> None:
        self.reports = R.list_reports(self.project) if self.project is not None else []
        self.report_list.blockSignals(True)
        self.report_list.clear()
        for info in self.reports:
            names = ", ".join(f"{r.get('train', {}).get('dataset', '?')}·{r.get('train', {}).get('model', '?')}"
                              for r in info.runs) or self.tr("(DefectLab 리포트 아님)")
            rec = ""
            if info.recommended is not None and info.recommended < len(info.runs):
                rec = self.tr(" · 추천 {0}").format(chr(65 + info.recommended))
            item = QListWidgetItem(f"{info.path.name}  —  {names}{rec}")
            item.setData(Qt.ItemDataRole.UserRole, str(info.path))
            self.report_list.addItem(item)
            if select is not None and Path(select) == info.path:
                self.report_list.setCurrentItem(item)
        self.report_list.blockSignals(False)
        self._update_buttons()

    def selected_report(self) -> R.ReportInfo | None:
        row = self.report_list.currentRow()
        return self.reports[row] if 0 <= row < len(self.reports) else None

    def open_report(self) -> None:
        info = self.selected_report()
        if info is not None:
            self.open_url(QUrl.fromLocalFile(str(info.path)))

    def delete_report(self) -> bool:
        info = self.selected_report()
        if info is None:
            return False
        answer = QMessageBox.question(self, self.tr("리포트 삭제"), self.tr("{0}을(를) 삭제할까요?").format(info.path.name))
        if answer != QMessageBox.StandardButton.Yes:
            return False
        R.delete_report(info.path)
        self.refresh_reports()
        return True

    def rerun_report(self) -> None:
        info = self.selected_report()
        if info is not None and info.configs:
            self.rerun_requested.emit(info.configs)

    def recommendation(self) -> tuple[int, str] | None:
        if not self.results:
            return None
        return C.recommend(self.results, self.priority.currentText())

    def _update_summary(self) -> None:
        rec = self.recommendation()
        if rec is None:
            return
        i, why = rec
        r = self.results[i]
        n = r.metrics.images
        self.summary.setText(self.tr("<b>추천: {0}</b> — {1}<br>"
                                     "<span style='color:gray'>Val 원본 {2}장 (NG {3}, OK {4}) · conf {5:.2f} · "
                                     "NMS IoU {6:.2f} · 박스 기준 IoU ≥ 0.5, 클래스 일치</span>").format(
            r.label, why, n, r.metrics.ng_images, r.metrics.ok_images, r.conf, r.iou))

    def table_rows(self) -> list[C.MetricRow]:
        return C.metric_rows(self.results)

    def _fill_table(self) -> None:
        rows = self.table_rows()
        self.table.clear()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(self.results))
        # short headers (A · Crop · yolo11s); the full label with the run id is the tooltip
        for i, r in enumerate(self.results):
            head = QTableWidgetItem(f"{chr(65 + i)} · {r.training.dataset.capitalize()} · {r.training.model}")
            head.setToolTip(r.label)
            self.table.setHorizontalHeaderItem(i, head)
        self.table.setVerticalHeaderLabels([row.label for row in rows])
        bold = QFont()
        bold.setBold(True)
        for r, row in enumerate(rows):
            best = C.best_cells(row)
            for col, text in enumerate(row.texts):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col in best:
                    item.setFont(bold)
                self.table.setItem(r, col, item)
        self.table.resizeColumnsToContents()

    def _fill_charts(self) -> None:
        for plot in (self.bars, self.map_plot, self.loss_plot, self.pr_plot):
            plot.clear()
        n = len(self.results)
        self.legend.setText("&nbsp;&nbsp;&nbsp; ".join(
            f'<span style="color:{SERIES[i % len(SERIES)]}; font-weight:bold">■</span>&nbsp;'
            f'<span style="color:{TEXT_SECONDARY}">{chr(65 + i)}: {r.label}</span>'
            for i, r in enumerate(self.results)))
        width = 0.8 / max(n, 1)
        for i, r in enumerate(self.results):
            color = SERIES[i % len(SERIES)]
            xs = [k - 0.4 + width * (i + 0.5) for k in range(len(BAR_METRICS))]
            hs = [getattr(r.metrics, attr) for attr, _ in BAR_METRICS]
            self.bars.addItem(pg.BarGraphItem(x=xs, height=hs, width=width * 0.9, brush=color, pen=pg.mkPen(None)))
            pen = pg.mkPen(color, width=2)
            epochs, values = r.series.get("epochs", []), r.series.get("values", {})
            for plot, kind in ((self.map_plot, "map"), (self.loss_plot, "loss")):
                key = _series_key(values, kind)
                if key:
                    pts = [(e, v) for e, v in zip(epochs, values[key]) if v is not None]
                    plot.plot([p[0] for p in pts], [p[1] for p in pts], pen=pen)
            rec, prec = r.metrics.pr_curve
            # few labels -> few points: show them as dots too, a 1-point line is invisible
            dots = dict(symbol="o", symbolSize=6, symbolBrush=color, symbolPen=None) if len(rec) < 30 else {}
            self.pr_plot.plot(rec, prec, pen=pen, **dots)
        axis = self.bars.getAxis("bottom")
        axis.setTicks([[(k, name) for k, (_, name) in enumerate(BAR_METRICS)]])
        small = QFont()
        small.setPointSize(8)
        axis.setStyle(tickFont=small)  # "mAP50" and "mAP50-95" side by side must not run together
        self.bars.setYRange(0, 1, padding=0.02)

    def _fill_diff_list(self) -> None:
        self.diff_list.clear()
        a, b = self.side_a.currentIndex(), self.side_b.currentIndex()
        if not self.results or a < 0 or b < 0:
            return
        for path, diff in C.differing_images(self.results[a], self.results[b]):
            item = QListWidgetItem(f"{path}  (차이 {diff})" if diff else path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.diff_list.addItem(item)
        if self.diff_list.count():
            self.diff_list.setCurrentRow(0)

    def _show_side_by_side(self, item: QListWidgetItem | None) -> None:
        if item is None or self.project is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        image = load_training_image(self.project, path)
        if image is None:
            return
        for idx, view, caption, letter in ((self.side_a.currentIndex(), self.view_a, self.caption_a, "A"),
                                           (self.side_b.currentIndex(), self.view_b, self.caption_b, "B")):
            r = self.results[idx]
            im = next((x for x in r.images if x.path == path), None)
            if im is None:
                continue
            preds = im.above(r.conf)
            matched, missed = match(im.ground_truth, preds)
            drawn = draw_comparison(image, im.ground_truth, [p.shape for p in preds], max_side=SIDE * 2)
            pix = QPixmap.fromImage(numpy_to_qimage(drawn))
            view.setPixmap(pix.scaled(SIDE, SIDE, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
            caption.setText(f"<b>{letter}</b> {r.training.dataset} · {r.training.model} — "
                            f"맞음 {sum(matched)} · 과검 {len(matched) - sum(matched)} · 미검 {sum(missed)}")

    def _clear_results(self) -> None:
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        for plot in (self.bars, self.map_plot, self.loss_plot, self.pr_plot):
            plot.clear()
        self.legend.clear()
        self.diff_list.clear()
        for w in (self.view_a, self.view_b, self.caption_a, self.caption_b):
            w.clear()
        self.side_a.clear()
        self.side_b.clear()
