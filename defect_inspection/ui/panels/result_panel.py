"""Result tab (PRD F5, F2.8 ①): Val prediction grid, Ultralytics plots, drag & drop test, and
model-based label suggestions."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.crop import TiledPredictor
from defect_inspection.core.edge_filter import (
    EdgeFilteredPredictor,
    apply_edge_filter,
    filter_predictions,
    settings_text,
    with_edge_filter,
)
from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.core.evaluate import (
    ImageResult,
    Predictor,
    SuggestionReport,
    evaluate_images,
    generate_suggestions,
    load_training_image,
    match,
    unlabeled_images,
    val_images,
)
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.preview import DROPPED_COLOR, EDGE_COLOR, GT_COLOR, PRED_COLOR, draw_comparison
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import RunInfo, list_runs, weights_paths
from defect_inspection.platform_utils import imread_unicode, is_supported_image
from defect_inspection.ui.canvas.image_canvas import numpy_to_qimage
from defect_inspection.ui.widgets.edge_filter_box import EdgeFilterBox

TILE = 180
EVAL_CONF = 0.01  # predict once at a low threshold; the conf slider then filters instantly
VIEW_JOBS = ("evaluate", "refilter", "test")  # results only shown for the run/project they were made for
REFILTER_DELAY_MS = 400  # edge filter settings settle this long before the cached results are re-filtered
FILTERS = [("all", "전체"), ("problem", "오검·미검"), ("missed", "미검 (놓친 결함)"), ("false", "오검 (잘못 검출)"), ("ok", "정상")]
PLOTS = [
    ("confusion_matrix_normalized.png", "Confusion matrix (정규화)"),
    ("confusion_matrix.png", "Confusion matrix"),
    ("BoxPR_curve.png", "PR curve (box)"),
    ("MaskPR_curve.png", "PR curve (mask)"),
    ("BoxF1_curve.png", "F1 curve"),
    ("results.png", "학습 곡선 (Ultralytics)"),
]


def _filter_ok(result: ImageResult, key: str) -> bool:
    kind = result.kind
    return {"all": True, "problem": kind != "ok", "missed": kind in ("missed", "both"),
            "false": kind in ("false", "both"), "ok": kind == "ok"}[key]


class ResultPanel(QWidget):
    open_image = Signal(str)  # image path -> show it on the canvas for relabeling (F5.4)
    labels_changed = Signal(list)  # image paths whose label files were changed by suggestions
    quick_train_requested = Signal()  # main window builds the dataset if needed and trains yolo11n
    settings_changed = Signal()  # project.config.edge_filter was edited (unsaved project)
    _done = Signal(int, str, object)  # generation, job kind, payload (from the worker thread)
    _progress = Signal(int, int, int)  # generation, done, total

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self.runs: list[RunInfo] = []
        self.raw_results: list[ImageResult] = []  # Val predictions before the edge filter
        self.results: list[ImageResult] = []
        self.results_edge: EdgeFilterConfig | None = None  # edge filter the results were predicted with
        self.before_suggest: Callable[[], None] = lambda: None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="result")
        self._predictors: dict[tuple[str, float], Predictor] = {}
        self._gen = 0
        self._view_gen = 0  # bumped when the project or run changes: late VIEW_JOBS results are dropped
        self._running_kind = ""
        self._cancel = threading.Event()
        self._busy = False
        self._done.connect(self._on_done)
        self._progress.connect(self._on_progress)
        self._build_ui()
        self.setAcceptDrops(True)
        self.set_project(None)

    # ---- UI ------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(1, 95)
        self.conf.setValue(25)
        self.conf.valueChanged.connect(self._on_conf_changed)
        self.conf_label = QLabel()
        self.iou = QSlider(Qt.Orientation.Horizontal)
        self.iou.setRange(10, 95)
        self.iou.setValue(70)
        self.iou.valueChanged.connect(lambda: self.iou_label.setText(f"NMS IoU {self.iou_value:.2f}"))
        self.iou.sliderReleased.connect(self.evaluate)
        self.iou_label = QLabel()

        top = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("모델(run)")))
        row.addWidget(self.run_combo, 1)
        top.addLayout(row)
        sliders = QHBoxLayout()
        sliders.addWidget(self.conf_label)
        sliders.addWidget(self.conf, 1)
        sliders.addWidget(self.iou_label)
        sliders.addWidget(self.iou, 1)
        top.addLayout(sliders)
        self.edge_box = EdgeFilterBox()
        self.edge_box.changed.connect(self._on_edge_changed)
        top.addWidget(self.edge_box)
        self._refilter_timer = QTimer(self)  # spin boxes fire per step: re-filter once they settle
        self._refilter_timer.setSingleShot(True)
        self._refilter_timer.setInterval(REFILTER_DELAY_MS)
        self._refilter_timer.timeout.connect(self._refilter)

        # Val grid
        self.filter_combo = QComboBox()
        for key, label in FILTERS:
            self.filter_combo.addItem(self.tr(label), key)
        self.filter_combo.currentIndexChanged.connect(self._fill_grid)
        self.btn_eval = QPushButton(self.tr("검증 이미지 평가"))
        self.btn_eval.clicked.connect(self.evaluate)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        def key(color: str, text: str) -> str:  # one legend entry, never broken across lines
            return (f'<span style="white-space:nowrap"><span style="color:{color}; font-weight:bold">━━</span> '
                    f'{text}</span>')

        legend = QLabel("&nbsp;&nbsp; ".join([
            key(GT_COLOR, self.tr("정답(라벨)")), key(PRED_COLOR, self.tr("예측 (점수)")),
            key(DROPPED_COLOR, self.tr("테두리 필터로 제외")), key(EDGE_COLOR, self.tr("찾은 테두리 선")),
            f'<span style="white-space:nowrap">{self.tr("더블클릭: 왼쪽에서 라벨 수정")}</span>']))
        legend.setWordWrap(True)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setIconSize(QSize(TILE, TILE))
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setSpacing(4)
        self.grid.setWordWrap(True)
        self.grid.itemDoubleClicked.connect(lambda item: self.open_image.emit(item.data(Qt.ItemDataRole.UserRole)))
        val_tab = QWidget()
        v = QVBoxLayout(val_tab)
        head = QHBoxLayout()
        head.addWidget(self.btn_eval)
        head.addWidget(self.filter_combo, 1)
        v.addLayout(head)
        v.addWidget(self.summary)
        v.addWidget(legend)
        v.addWidget(self.grid, 1)

        # Ultralytics plots
        self.plot_combo = QComboBox()
        self.plot_combo.currentIndexChanged.connect(self._show_plot)
        self.plot_view = QLabel()
        self.plot_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(self.plot_view)
        plots_tab = QWidget()
        p = QVBoxLayout(plots_tab)
        p.addWidget(self.plot_combo)
        p.addWidget(plot_scroll, 1)

        # drag & drop test
        self.test_view = QLabel(self.tr("이미지를 여기로 끌어다 놓거나 '파일 열기'를 누르세요"))
        self.test_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_view.setMinimumHeight(240)
        self.test_view.setStyleSheet("border: 2px dashed #b0b0b0; color: gray;")
        self.test_info = QLabel()
        self.test_info.setWordWrap(True)
        self.btn_open_test = QPushButton(self.tr("파일 열기…"))
        self.btn_open_test.clicked.connect(self._open_test_file)
        test_tab = QWidget()
        t = QVBoxLayout(test_tab)
        t.addWidget(self.btn_open_test, 0, Qt.AlignmentFlag.AlignLeft)
        t.addWidget(self.test_info)  # above the image so a tall image never pushes it out of view
        t.addWidget(self.test_view, 1)
        self.test_path: Path | None = None

        self.tabs = QTabWidget()
        self.tabs.addTab(val_tab, self.tr("검증 결과"))
        self.tabs.addTab(plots_tab, self.tr("그래프"))
        self.tabs.addTab(test_tab, self.tr("테스트 추론"))

        # suggestions (F2.8 ①)
        self.unlabeled_label = QLabel()
        self.btn_suggest = QPushButton(self.tr("미라벨 이미지에 제안 생성"))
        self.btn_suggest.clicked.connect(self.suggest)
        self.btn_quick = QPushButton(self.tr("제안용 빠른 학습 (yolo11n)"))
        self.btn_quick.setToolTip(self.tr("확정된 라벨만으로 yolo11n을 짧게(30 epoch) 학습한 뒤 미라벨 이미지에 제안을 만듭니다"))
        self.btn_quick.clicked.connect(self.quick_train_requested.emit)
        self.suggest_status = QLabel()
        self.suggest_status.setWordWrap(True)
        sg = QVBoxLayout()
        sg.addWidget(self.unlabeled_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_suggest)
        buttons.addWidget(self.btn_quick)
        sg.addLayout(buttons)
        sg.addWidget(self.suggest_status)
        self.suggest_box = QGroupBox(self.tr("자동 라벨 제안"))
        self.suggest_box.setLayout(sg)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.suggest_box)
        self._on_conf_changed()
        self.iou_label.setText(f"NMS IoU {self.iou_value:.2f}")

    # ---- state ----------------------------------------------------------------------------
    @property
    def conf_value(self) -> float:
        return self.conf.value() / 100

    @property
    def iou_value(self) -> float:
        return self.iou.value() / 100

    def _invalidate_view(self) -> None:
        """Results of running evaluate/refilter/test jobs no longer belong on screen. Suggestions are
        not dropped: they already changed label files, and the window must reload them."""
        self._view_gen += 1
        self._refilter_timer.stop()

    def set_project(self, project: Project | None) -> None:
        self._cancel.set()
        self._invalidate_view()
        self.project = project
        self.raw_results, self.results = [], []
        self.edge_box.set_config(project.config.edge_filter if project else EdgeFilterConfig())
        self.edge_box.setEnabled(project is not None)
        self.grid.clear()
        self.summary.setText("")
        self.refresh_runs()
        self.refresh_unlabeled()

    def refresh_runs(self, select: str | Path | None = None) -> None:
        """Runs with a best.pt, newest first."""
        self.runs = [] if self.project is None else [r for r in list_runs(self.project) if r.best_weights]
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for r in self.runs:
            self.run_combo.addItem(f"{r.run_id}  ({r.spec.train.model if r.spec else '?'})", str(r.path))
        if select is not None:
            idx = self.run_combo.findData(str(select))
            self.run_combo.setCurrentIndex(max(idx, 0))
        self.run_combo.blockSignals(False)
        self._on_run_changed()

    def refresh_unlabeled(self) -> None:
        n = 0 if self.project is None else len(unlabeled_images(self.project))
        self.unlabeled_label.setText(self.tr("미라벨 이미지 {0}장 (OK 이미지 제외)").format(n))
        self._update_buttons()

    @property
    def current_run(self) -> RunInfo | None:
        idx = self.run_combo.currentIndex()
        return self.runs[idx] if 0 <= idx < len(self.runs) else None

    def _update_buttons(self) -> None:
        has_run = self.current_run is not None and not self._busy
        self.btn_eval.setEnabled(has_run)
        self.btn_suggest.setEnabled(has_run and self.unlabeled_count() > 0)
        self.btn_quick.setEnabled(self.project is not None and not self._busy)
        self.btn_open_test.setEnabled(has_run)

    def unlabeled_count(self) -> int:
        return 0 if self.project is None else len(unlabeled_images(self.project))

    def _on_run_changed(self) -> None:
        self._invalidate_view()
        self.raw_results, self.results = [], []
        self.grid.clear()
        self.summary.setText("" if self.current_run else self.tr("학습이 끝난 run이 없습니다. Train 탭에서 학습하세요."))
        self._fill_plots()
        self._update_buttons()

    def _on_conf_changed(self) -> None:
        self.conf_label.setText(f"conf ≥ {self.conf_value:.2f}")
        if self.results:
            self._fill_grid()

    # ---- worker ---------------------------------------------------------------------------
    def _predictor(self, run: RunInfo, edge: EdgeFilterConfig | None = None
                   ) -> Predictor | TiledPredictor | EdgeFilteredPredictor:
        """The run's model wrapped in the edge filter (F9.3) when `edge` (default: the project's
        settings now) is on; jobs pass `_edge_snapshot()` taken on the UI thread when they start."""
        return with_edge_filter(self._model(run), edge or self._edge_snapshot())

    def _edge_snapshot(self) -> EdgeFilterConfig:
        return self.project.config.edge_filter.model_copy()

    def dropped_note(self, n: int) -> str:
        return self.tr(" · 테두리 필터로 제외 {0}개").format(n) if n else ""

    def edge_note(self, edge: EdgeFilterConfig | None) -> str:
        """' · 테두리 필터 …' when the shown results were filtered, '' otherwise."""
        if edge is None or not edge.enabled:
            return ""
        return self.tr(" · 테두리 필터 ({0})").format(settings_text(edge))

    def _model(self, run: RunInfo) -> Predictor | TiledPredictor:
        """Cached per best.pt; crop runs are wrapped in tiling inference (F7.6)."""
        weights = run.best_weights
        key = (str(weights), weights.stat().st_mtime)
        if key not in self._predictors:
            imgsz = run.spec.train.imgsz if run.spec else 640
            predictor = Predictor(weights, self.project.config.task, imgsz)
            tiling = run.spec.tiling() if run.spec else None
            self._predictors = {key: TiledPredictor(predictor, tiling) if tiling else predictor}
        return self._predictors[key]

    def tiling_note(self) -> str:
        """' · 타일링 추론 …' for crop runs, '' otherwise."""
        run = self.current_run
        tiling = run.spec.tiling() if run is not None and run.spec else None
        if tiling is None:
            return ""
        return self.tr(" · 타일링 추론 (타일 {0}px, overlap {1}px)").format(tiling.tile_size, tiling.overlap_px)

    def _submit(self, kind: str, job: Callable) -> int:
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        cancel = self._cancel
        view = self._view_gen
        self._busy = True
        self._running_kind = kind
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._update_buttons()

        def run() -> None:
            try:
                payload = job(lambda d, t: self._emit(self._progress, gen, d, t), cancel.is_set)
            except Exception as e:  # noqa: BLE001 - shown to the user
                payload = e
            self._emit(self._done, gen, kind, (view, payload))

        self._pool.submit(run)
        return gen

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            pass

    def _on_progress(self, gen: int, done: int, total: int) -> None:
        if gen == self._gen:
            self.progress.setMaximum(max(total, 1))
            self.progress.setValue(done)

    def _on_done(self, gen: int, kind: str, payload) -> None:
        if gen != self._gen:
            return
        self._busy = False
        self.progress.setVisible(False)
        self._update_buttons()
        view, payload = payload
        if kind in VIEW_JOBS and view != self._view_gen:
            return  # made for a run or project no longer shown
        if isinstance(payload, Exception):
            QMessageBox.warning(self, self.tr("결과"), f"{type(payload).__name__}: {payload}")
            return
        if kind == "evaluate":
            self.raw_results, self.results, self.results_edge = payload
            self._fill_grid()
        elif kind == "refilter":
            self.results, self.results_edge = payload
            self._fill_grid()
        elif kind == "test":
            self._show_test(*payload)
        elif kind == "suggest":
            self._suggest_done(*payload)

    @property
    def busy(self) -> bool:
        return self._busy

    def shutdown(self) -> None:
        self._cancel.set()
        self._pool.shutdown(wait=True, cancel_futures=True)

    # ---- Val grid (F5.1, F5.4) ------------------------------------------------------------------
    def evaluate(self) -> None:
        run, project = self.current_run, self.project
        if run is None or project is None or self._busy:
            return
        paths = val_images(project, run.spec.train.dataset if run.spec else "full")
        if not paths:
            self.summary.setText(self.tr("검증 이미지가 없습니다. 데이터셋을 빌드했는지 확인하세요."))
            return
        iou, edge = self.iou_value, self._edge_snapshot()

        def job(progress, cancel):
            # with the filter on, prediction fills the first half of the bar and filtering the second
            half = (lambda d, t: progress(d, 2 * t)) if edge.enabled else progress
            raw = evaluate_images(project, self._model(run), paths, EVAL_CONF, iou, half, cancel)
            return raw, apply_edge_filter(project, raw, edge, lambda d, t: progress(t + d, 2 * t), cancel), edge

        self.summary.setText(self.tr("검증 이미지 {0}장 예측 중…").format(len(paths)))
        self._submit("evaluate", job)

    def visible_results(self) -> list[ImageResult]:
        """Results re-matched at the current conf threshold."""
        out = []
        conf = self.conf_value
        for r in self.results:
            hits = r.edge_hits or [None] * len(r.predictions)
            kept = [(p, h) for p, h in zip(r.predictions, hits) if p.score >= conf]
            preds = [p for p, _ in kept]
            matched, missed = match(r.ground_truth, preds)
            out.append(ImageResult(r.path, r.width, r.height, r.ground_truth, preds, matched, missed,
                                   dropped=[p for p in r.dropped if p.score >= conf],
                                   edge_hits=[h for _, h in kept] if r.edge_hits else []))
        return out

    # ---- edge filter (F9) ---------------------------------------------------------------------
    def _on_edge_changed(self) -> None:
        if self.project is None:
            return
        self.project.config.edge_filter = self.edge_box.config()
        self.settings_changed.emit()
        self._refilter_timer.start()

    def _refilter(self) -> None:
        """Apply the current settings to the cached Val predictions (no new prediction)."""
        project, raw = self.project, self.raw_results
        if project is None or not raw:
            return
        if self._busy:
            self._refilter_timer.start()  # after the running job
            return
        edge = self._edge_snapshot()

        def job(progress, cancel):
            return apply_edge_filter(project, raw, edge, progress, cancel), edge

        self._submit("refilter", job)

    def _fill_grid(self) -> None:
        self.grid.clear()
        if not self.results or self.project is None:
            return
        shown = self.visible_results()
        tp, fp, fn = (sum(getattr(r, k) for r in shown) for k in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        self.summary.setText(self.tr(
            "검증 {0}장 · conf {1:.2f}: 맞음 {2}, 오검 {3}, 미검 {4} · precision {5:.2f}, recall {6:.2f} "
            "(IoU ≥ 0.5, 클래스 일치 기준)").format(len(shown), self.conf_value, tp, fp, fn, precision, recall)
            + self.tiling_note() + self.edge_note(self.results_edge)
            + self.dropped_note(sum(len(r.dropped) for r in shown)))
        key = self.filter_combo.currentData()
        for r in shown:
            if not _filter_ok(r, key):
                continue
            image = load_training_image(self.project, r.path)
            if image is None:
                continue
            tile = draw_comparison(image, r.ground_truth, [p.shape for p in r.predictions], max_side=TILE * 2,
                                   dropped=[p.shape for p in r.dropped],
                                   edges=[(h, p.shape) for p, h in zip(r.predictions, r.edge_hits)])
            item = QListWidgetItem(QIcon(QPixmap.fromImage(numpy_to_qimage(tile))),
                                   f"{r.path}\nTP {r.tp} · FP {r.fp} · FN {r.fn}")
            item.setData(Qt.ItemDataRole.UserRole, r.path)
            item.setToolTip(self.tr("더블클릭하면 왼쪽 캔버스에서 라벨을 고칠 수 있습니다"))
            self.grid.addItem(item)

    # ---- plots (F5.2) -----------------------------------------------------------------------------
    def _fill_plots(self) -> None:
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        run = self.current_run
        if run is not None:
            for name, label in PLOTS:
                path = run.path / "train" / name
                if path.is_file():
                    self.plot_combo.addItem(self.tr(label), str(path))
        self.plot_combo.blockSignals(False)
        self._show_plot()

    def _show_plot(self) -> None:
        path = self.plot_combo.currentData()
        if not path:
            self.plot_view.setText(self.tr("그래프가 없습니다"))
            return
        pix = QPixmap(path)
        self.plot_view.setPixmap(pix.scaledToWidth(max(300, self.plot_view.width() - 10),
                                                   Qt.TransformationMode.SmoothTransformation))

    # ---- drag & drop test (F5.3) ---------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if self.current_run is not None and urls and is_supported_image(urls[0].toLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self.tabs.setCurrentIndex(2)
        self.test_image(Path(event.mimeData().urls()[0].toLocalFile()))

    def _open_test_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("테스트 이미지"), "",
                                              self.tr("이미지 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"))
        if path:
            self.test_image(Path(path))

    def test_image(self, path: Path) -> None:
        run, project = self.current_run, self.project
        if run is None or project is None or self._busy:
            return
        self.test_path = path
        conf, iou, edge = self.conf_value, self.iou_value, self._edge_snapshot()

        def job(progress, cancel):
            raw = imread_unicode(path)
            if raw is None:
                raise OSError(f"이미지를 읽을 수 없습니다: {path}")
            image = to_training_8bit(raw, project.channels)
            preds = self._model(run).predict(image, conf, iou)
            if not edge.enabled:
                return image, preds, [], [], edge
            f = filter_predictions(image, preds, edge)
            return image, list(f.kept), list(f.dropped), list(f.hits), edge

        self.test_info.setText(self.tr("추론 중…"))
        self._submit("test", job)

    def _show_test(self, image, preds, dropped=(), hits=(), edge: EdgeFilterConfig | None = None) -> None:
        drawn = draw_comparison(image, [], [p.shape for p in preds], max_side=1200,
                                dropped=[p.shape for p in dropped], edges=[(h, p.shape) for p, h in zip(preds, hits)])
        pix = QPixmap.fromImage(numpy_to_qimage(drawn))
        self.test_view.setStyleSheet("")
        self.test_view.setPixmap(pix.scaled(self.test_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
        names = [c.name for c in self.project.config.classes] if self.project else []
        lines = [f"{names[p.shape.class_id] if p.shape.class_id < len(names) else p.shape.class_id}: {p.score:.2f}"
                 for p in sorted(preds, key=lambda p: -p.score)]
        self.test_info.setText(self.tr("{0} · 검출 {1}개 (conf ≥ {2:.2f})").format(
            self.test_path.name if self.test_path else "", len(preds), self.conf_value)
            + self.tiling_note() + self.edge_note(edge) + self.dropped_note(len(dropped))
            + ("\n" + ", ".join(lines) if lines else ""))
        self.last_test_predictions = preds

    # ---- suggestions (F2.8 ①, F5.5) ------------------------------------------------------------------
    def suggest(self, run: RunInfo | None = None) -> bool:
        run = run or self.current_run
        project = self.project
        if run is None or project is None or (self._busy and self._running_kind not in VIEW_JOBS):
            return False
        if self._busy:
            self._cancel.set()  # an evaluation or refilter only updates the view: suggestions go first
        self.before_suggest()  # save the open image's labels first
        paths = unlabeled_images(project)
        if not paths:
            self.suggest_status.setText(self.tr("미라벨 이미지가 없습니다"))
            return False
        conf, iou, edge = self.conf_value, self.iou_value, self._edge_snapshot()

        def job(progress, cancel):
            return generate_suggestions(project, self._predictor(run, edge), paths, conf, iou, progress, cancel), edge

        self.suggest_status.setText(self.tr("{0}장에 제안 생성 중… ({1})").format(len(paths), run.run_id))
        self._submit("suggest", job)
        return True

    def _suggest_done(self, report: SuggestionReport, edge: EdgeFilterConfig | None = None) -> None:
        self.suggest_status.setText(self.tr(
            "제안 완료: {0}장 중 {1}장에 제안 {2}개 (conf ≥ {3:.2f}). 이미지 목록의 '제안 대기' 필터로 검토하세요.")
            .format(report.images, report.with_suggestions, report.suggestions, self.conf_value)
            + self.edge_note(edge))
        self.labels_changed.emit(report.paths)
        self.refresh_unlabeled()
