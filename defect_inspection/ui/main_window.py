"""Main window layout (PRD 3): toolbar, left image/class lists, center canvas, right tabs."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QColor, QIcon, QKeySequence, QPixmap, QUndoStack
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from defect_inspection import APP_NAME, __version__
from defect_inspection.core.compare import experiment_configs
from defect_inspection.core.image_cache import ImageCache
from defect_inspection.core.project import LabelStatus, Project
from defect_inspection.core.sam_assist import DEFAULT_SAM_MODEL, SAM_MODEL_CHOICES, SamPrompt, mask_to_shape
from defect_inspection import platform_utils
from defect_inspection.platform_utils import OVERLAY_SCROLLBARS, cache_dir, config_dir, detect_device
from defect_inspection.ui.canvas.image_canvas import CanvasPanel
from defect_inspection.ui.edge_check_dialog import EdgeCheckDialog
from defect_inspection.ui.group_checker import GroupChecker
from defect_inspection.ui.new_project_dialog import NewProjectDialog
from defect_inspection.core.roi import RoiStore
from defect_inspection.core import brush
from defect_inspection.ui.canvas.brush_tool import BrushBar
from defect_inspection.ui.canvas.labeling_canvas import LabelingCanvas
from defect_inspection.ui.canvas.roi_tool import RoiBar, RoiSession
from defect_inspection.ui.canvas.session import LabelSession
from defect_inspection.ui.canvas.tools import Tool
from defect_inspection.ui.image_list import ImageListPanel
from defect_inspection.ui.label_list import LabelListPanel
from defect_inspection.core.evaluate import quick_suggest_config
from defect_inspection.core.exporter import DEFAULT_NAME_PATTERN, NAME_TOKENS
from defect_inspection.ui.panels.augment_panel import AugmentPanel
from defect_inspection.ui.panels.compare_panel import ComparePanel
from defect_inspection.ui.panels.export_panel import ExportPanel
from defect_inspection.ui.panels.result_panel import ResultPanel
from defect_inspection.ui.panels.train_panel import TrainPanel
from defect_inspection.ui.sam_controller import SamController
from defect_inspection.ui.thumbnails import ThumbnailLoader


# Right-side tabs in display order, with the PROMPTS step that fills each one.
TABS = [("Augment", 10), ("Train", 12), ("Result", 13), ("Compare", 19), ("Export", 15)]

_SETTINGS_GEOMETRY = "main_window/geometry"
_SETTINGS_MAIN_SPLITTER = "main_window/main_splitter"
_SETTINGS_LEFT_SPLITTER = "main_window/left_splitter"
_SETTINGS_SAM_MODEL = "sam/model"
_SETTINGS_BRUSH_SIZE = "brush/size"
_SETTINGS_AUTOSAVE = "autosave/interval_sec"
_SETTINGS_EXPORT_PATTERN = "export/name_pattern"
DEFAULT_AUTOSAVE_SEC = 30
IMAGES_SYNC_DELAY_MS = 700
IMAGES_SETTLE_SEC = 1.0  # a file changed more recently may still be copying


class MainWindow(QMainWindow):
    tool_changed = Signal(Tool)
    project_changed = Signal(object)  # Project | None
    _device_detected = Signal(str)
    _image_loaded = Signal(object, object)  # Path, np.ndarray | None (from cache worker threads)

    def __init__(self, settings: QSettings | None = None, cache_root: Path | None = None) -> None:
        super().__init__()
        self.settings = settings or QSettings()
        self.project: Project | None = None
        self.project_dirty = False
        self.current_tool = Tool.SELECT
        self.current_image: str | None = None
        self.session: LabelSession | None = None
        self.roi_store: RoiStore | None = None
        self.roi_session: RoiSession | None = None
        self.undo_stack = QUndoStack(self)
        self.image_cache = ImageCache(capacity=8)
        self.thumbnails = ThumbnailLoader((cache_root or cache_dir()) / "thumbs", self)
        self.weights_dir = (cache_root or cache_dir()) / "weights"
        # User augmentation presets are per user; tests (cache_root given) keep them in a temp dir.
        self.presets_dir = (cache_root / "augment_presets") if cache_root else config_dir() / "augment_presets"
        self._image_loaded.connect(self._show_loaded_image)
        self.sam_controller = SamController(
            str(self.settings.value(_SETTINGS_SAM_MODEL, DEFAULT_SAM_MODEL)),
            (cache_root or cache_dir()) / "weights", self)
        self._sam_generation = 0
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        # images/ watched for files copied in or deleted outside the app (Finder/Explorer).
        # Events come in bursts during a copy, so they are collected for a moment first.
        self.images_watcher = QFileSystemWatcher(self)
        self.images_watcher.directoryChanged.connect(lambda _path: self.images_sync_timer.start())
        self.images_sync_timer = QTimer(self)
        self.images_sync_timer.setSingleShot(True)
        self.images_sync_timer.setInterval(IMAGES_SYNC_DELAY_MS)
        self.images_sync_timer.timeout.connect(self.sync_images_folder)
        # Slices / copies of other project images (core/image_groups), checked in the background.
        self.group_checker = GroupChecker(self)
        self.edge_check_dialog: EdgeCheckDialog | None = None  # 도구 → 테두리 검사 (non-modal)
        self.group_checker.finished.connect(self._on_groups_found)
        self._group_generation = 0
        self._groups_applied = 0
        self._show_groups_when_done = False
        self._group_notice: QMessageBox | None = None

        self.resize(1600, 900)
        self._build_actions()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self._refresh_lists()
        self._restore_layout()
        self._update_title()
        self._build_menu()
        self._connect_sam()
        self.set_autosave_interval(int(self.settings.value(_SETTINGS_AUTOSAVE, DEFAULT_AUTOSAVE_SEC)))
        self._start_device_detection()

    # ---- actions / toolbar -----------------------------------------------------
    def _build_actions(self) -> None:
        def action(text: str, shortcut: QKeySequence | str | None, slot=None) -> QAction:
            act = QAction(text, self)
            if shortcut is not None:
                act.setShortcut(QKeySequence(shortcut))
            if slot is not None:
                act.triggered.connect(slot)
            return act

        self.act_new = action(self.tr("새 프로젝트"), QKeySequence.StandardKey.New, self.new_project_dialog)
        self.act_open = action(self.tr("프로젝트 열기"), QKeySequence.StandardKey.Open, self.open_project_dialog)
        self.act_add_folder = action(self.tr("이미지 폴더 추가"), "Ctrl+I", self.add_image_folder_dialog)
        self.act_save = action(self.tr("저장"), QKeySequence.StandardKey.Save, self.save_project)

        # Fixed button text (Qt's createUndoAction appends the command name and resizes the
        # toolbar); the command name goes into the tooltip instead.
        self.act_undo = action(self.tr("실행 취소"), QKeySequence.StandardKey.Undo, self.undo_stack.undo)
        self.act_redo = action(self.tr("다시 실행"), None, self.undo_stack.redo)
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence.StandardKey.Redo])
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)
        self.undo_stack.canUndoChanged.connect(self.act_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.act_redo.setEnabled)
        self.undo_stack.undoTextChanged.connect(
            lambda t: self.act_undo.setToolTip(self.tr("실행 취소: {0}").format(t) if t else self.tr("실행 취소")))
        self.undo_stack.redoTextChanged.connect(
            lambda t: self.act_redo.setToolTip(self.tr("다시 실행: {0}").format(t) if t else self.tr("다시 실행")))

        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions: dict[Tool, QAction] = {}
        for tool, text, key in [
            (Tool.SELECT, self.tr("선택"), "V"),
            (Tool.BBOX, self.tr("BBox"), "B"),
            (Tool.POLYGON, self.tr("Polygon"), "P"),
            (Tool.BRUSH, self.tr("브러시"), "R"),
            (Tool.CROP_ROI, self.tr("Crop ROI"), "C"),
            (Tool.SAM, self.tr("SAM 보조"), "S"),
            (Tool.ERASER, self.tr("지우개"), "E"),
        ]:
            act = action(f"{text} ({key})", key)
            act.setCheckable(True)
            act.setData(tool)
            act.triggered.connect(lambda _checked=False, t=tool: self.set_tool(t))
            self.tool_group.addAction(act)
            self.tool_actions[tool] = act
        self.tool_actions[Tool.SELECT].setChecked(True)

        # View / navigation (not on the toolbar; window-wide shortcuts)
        self.act_prev_image = action(self.tr("이전 이미지"), "A", lambda: self.image_panel.step(-1))
        self.act_next_image = action(self.tr("다음 이미지"), "D", lambda: self.image_panel.step(1))
        self.act_fit = action(self.tr("화면에 맞춤"), "Ctrl+0", lambda: self.canvas.fit())
        self.act_zoom_100 = action(self.tr("실제 크기"), "Ctrl+1", lambda: self.canvas.zoom_to(1.0))
        for act in (self.act_prev_image, self.act_next_image, self.act_fit, self.act_zoom_100):
            self.addAction(act)

        # Labeling keys (PRD F2.2, F2.5, F2.8). Window-wide so they work wherever focus is;
        # text inputs such as spin boxes still receive their own keys first.
        self.class_key_actions = []
        for i in range(9):
            act = action(self.tr("클래스 {0}").format(i + 1), str(i + 1), lambda _=False, i=i: self.choose_class(i))
            self.addAction(act)
            self.class_key_actions.append(act)
        self.act_enter = action(self.tr("수락 / 폴리곤 완료"), None, self.on_enter)
        self.act_enter.setShortcuts([QKeySequence(Qt.Key.Key_Return), QKeySequence(Qt.Key.Key_Enter)])
        self.act_backspace = action(self.tr("거절 / 마지막 점 삭제"), QKeySequence(Qt.Key.Key_Backspace), self.on_backspace)
        self.act_delete = action(self.tr("라벨 삭제"), QKeySequence.StandardKey.Delete, self.delete_selected)
        self.act_escape = action(self.tr("그리기 취소"), QKeySequence(Qt.Key.Key_Escape), self.on_escape)
        # F1.5: an image without defects trains as background (empty label)
        self.act_toggle_ok = action(self.tr("OK 이미지 (결함 없음) 표시/해제"), "O", self.toggle_ok)
        # Brush size ([ / ]): window-wide like the other labeling keys
        self.act_brush_smaller = action(self.tr("브러시 작게"), "[", lambda: self._step_brush(up=False))
        self.act_brush_larger = action(self.tr("브러시 크게"), "]", lambda: self._step_brush(up=True))
        for act in (self.act_enter, self.act_backspace, self.act_delete, self.act_escape, self.act_toggle_ok,
                    self.act_brush_smaller, self.act_brush_larger):
            self.addAction(act)

        # Train/Val by hand and the synthetic mark (image-list right-click, 편집 menu)
        self.act_split_train = action(self.tr("Train으로 지정"), None, lambda: self.set_current_split("train"))
        self.act_split_val = action(self.tr("Val로 지정"), None, lambda: self.set_current_split("val"))
        self.act_split_auto = action(self.tr("Train/Val 자동"), None, lambda: self.set_current_split("auto"))
        self.act_synthetic = action(self.tr("합성 이미지 (항상 Train)"), None, self.toggle_synthetic)
        self.act_synthetic.setCheckable(True)
        self._project_actions = [self.act_add_folder, self.act_save, self.act_toggle_ok, self.act_split_train,
                                 self.act_split_val, self.act_split_auto, self.act_synthetic]

    def _build_toolbar(self) -> None:
        tb = QToolBar(self.tr("도구"), self)
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for act in (self.act_new, self.act_open, self.act_add_folder, self.act_save):
            tb.addAction(act)
        tb.addSeparator()
        for tool in (Tool.BBOX, Tool.POLYGON, Tool.BRUSH, Tool.CROP_ROI, Tool.SAM, Tool.SELECT, Tool.ERASER):
            tb.addAction(self.tool_actions[tool])
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        self.addToolBar(tb)
        self.toolbar = tb

    # ---- central layout ----------------------------------------------------------
    def _build_central(self) -> None:
        self.image_panel = ImageListPanel(self.thumbnails)
        self.image_panel.current_changed.connect(self._on_current_image_changed)
        self.image_list = self.image_panel.list
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)  # right-click: OK
        self.image_list.addAction(self.act_toggle_ok)
        self.image_list.addAction(_separator(self))
        for act in (self.act_split_train, self.act_split_val, self.act_split_auto, self.act_synthetic):
            self.image_list.addAction(act)
        self.class_list = QListWidget()
        self.class_list.setObjectName("class_list")
        self.class_list.currentRowChanged.connect(self._on_class_row_changed)
        self.class_list.itemDoubleClicked.connect(lambda _item: self.rename_class_dialog())
        self.btn_add_class = QPushButton(self.tr("+ 클래스"))
        self.btn_add_class.clicked.connect(self.add_class_dialog)
        self.btn_rename_class = QPushButton(self.tr("이름 변경"))
        self.btn_rename_class.clicked.connect(self.rename_class_dialog)
        self.btn_class_color = QPushButton(self.tr("색상"))
        self.btn_class_color.clicked.connect(self.class_color_dialog)
        class_box = QWidget()
        class_layout = QVBoxLayout(class_box)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.addWidget(self.class_list)
        buttons = QHBoxLayout()
        for b in (self.btn_add_class, self.btn_rename_class, self.btn_class_color):
            buttons.addWidget(b)
        buttons.addStretch(1)
        class_layout.addLayout(buttons)

        self.label_panel = LabelListPanel()

        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setObjectName("left_splitter")
        self.left_splitter.addWidget(_titled(self.tr("이미지"), self.image_panel))
        self.left_splitter.addWidget(_titled(self.tr("라벨"), self.label_panel))
        self.left_splitter.addWidget(_titled(self.tr("클래스"), class_box))
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 2)
        self.left_splitter.setStretchFactor(2, 1)

        self.canvas_panel = CanvasPanel(LabelingCanvas())
        self.canvas: LabelingCanvas = self.canvas_panel.canvas
        # Crop ROI options bar above the canvas, shown only with the ROI tool (PRD F7.1)
        self.roi_bar = RoiBar()
        self.roi_bar.setVisible(False)
        self.canvas_panel.layout().insertWidget(1, self.roi_bar)
        self.roi_bar.options_changed.connect(self._on_roi_options)
        self.roi_bar.make_common_requested.connect(lambda: self.roi_session and self.roi_session.make_common())
        self.roi_bar.clear_common_requested.connect(lambda: self.roi_session and self.roi_session.clear_common())
        self.tool_changed.connect(lambda tool: self.roi_bar.setVisible(tool == Tool.CROP_ROI))
        # Brush options bar, shown only with the brush tool
        self.brush_bar = BrushBar()
        self.brush_bar.setVisible(False)
        self.canvas_panel.layout().insertWidget(1, self.brush_bar)
        self.tool_changed.connect(lambda tool: self.brush_bar.setVisible(tool == Tool.BRUSH))
        self.brush_bar.size_changed.connect(lambda size: self.canvas.set_brush_size(size))
        self.canvas.set_placeholder(self.tr("프로젝트를 열고 이미지를 추가하세요"))
        self.canvas.brush_size_changed.connect(self._on_brush_size)
        try:
            self.canvas.set_brush_size(int(self.settings.value(_SETTINGS_BRUSH_SIZE, brush.DEFAULT_SIZE)))
        except (TypeError, ValueError):
            pass  # a damaged setting: keep the default size
        self.brush_bar.set_size(self.canvas.brush_size)
        self.canvas.cursor_moved.connect(self.set_cursor_position)
        self.canvas.message.connect(lambda text: self.statusBar().showMessage(text, 4000))
        self.canvas.selection_changed.connect(self.label_panel.select_ids)
        self.label_panel.selection_changed.connect(self.canvas.select_ids)
        self.tool_changed.connect(self.canvas.set_tool)

        self.tabs = ScrollTabs()
        self.tabs.setObjectName("right_tabs")
        self.panels: dict[str, QWidget] = {}
        self.augment_panel = AugmentPanel(self.presets_dir)
        self.augment_panel.changed.connect(self.mark_dirty)
        self.augment_panel.annotations_provider = self._current_annotations
        self.augment_panel.before_build = self._before_build
        self.train_panel = TrainPanel(self.weights_dir)
        self.train_panel.before_start = self._before_training
        self.augment_panel.build_finished.connect(lambda _report: self.train_panel.refresh_datasets())
        self.result_panel = ResultPanel()
        self.export_panel = ExportPanel()
        try:
            self.export_panel.set_pattern(str(self.settings.value(_SETTINGS_EXPORT_PATTERN, DEFAULT_NAME_PATTERN)))
        except ValueError:
            self.export_panel.set_pattern(DEFAULT_NAME_PATTERN)
        self.result_panel.before_suggest = self._save_all
        self.result_panel.open_image.connect(self.open_image_for_labeling)
        self.result_panel.labels_changed.connect(self._on_labels_changed_externally)
        self.result_panel.quick_train_requested.connect(self.start_quick_suggest)
        self.result_panel.settings_changed.connect(self.mark_dirty)
        self.train_panel.run_finished.connect(self._on_run_finished)
        self.augment_panel.build_finished.connect(self._on_quick_build_finished)
        self.augment_panel.build_stopped.connect(self._on_quick_build_stopped)
        self._quick_stage: str | None = None  # None / "build" / "train"
        self.compare_panel = ComparePanel()
        self.compare_panel.open_image.connect(self.open_image_for_labeling)
        self.compare_panel.experiment_requested.connect(self.start_compare_experiment)
        self.augment_panel.build_finished.connect(self._on_exp_build_finished)
        self.augment_panel.build_stopped.connect(self._on_exp_build_stopped)
        self.compare_panel.rerun_requested.connect(self.start_experiment)
        # comparison experiment (F8.1, F8.6): None / "build" / "train"
        self._exp_stage: str | None = None
        self._exp_steps: list = []
        self._exp_total = 0
        self._exp_runs: list[str] = []
        for name, step in TABS:
            if name == "Augment":
                panel = self.augment_panel
            elif name == "Train":
                panel = self.train_panel
            elif name == "Result":
                panel = self.result_panel
            elif name == "Export":
                panel = self.export_panel
            elif name == "Compare":
                panel = self.compare_panel
            else:
                panel = _placeholder_panel(self.tr("{0} 패널 (PROMPTS {1}단계에서 구현)").format(name, step))
            panel.setObjectName(f"panel_{name.lower()}")
            self.panels[name] = panel
            self.tabs.addTab(panel, name)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("main_splitter")
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.canvas_panel)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setStretchFactor(2, 2)
        self.main_splitter.setSizes([260, 840, 500])
        self.setCentralWidget(self.main_splitter)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self.status_image = QLabel()
        self.status_labels = QLabel()
        self.status_coords = QLabel()
        self.status_tool = QLabel()
        self.status_device = QLabel(self.tr("장치 확인 중…"))
        for w in (self.status_image, self.status_labels, self.status_coords, self.status_tool):
            sb.addWidget(w)
        sb.addPermanentWidget(self.status_device)
        self._update_status()

    # ---- layout persistence ------------------------------------------------------
    def _restore_layout(self) -> None:
        if (geom := self.settings.value(_SETTINGS_GEOMETRY)) is not None:
            self.restoreGeometry(geom)
        if (state := self.settings.value(_SETTINGS_MAIN_SPLITTER)) is not None:
            self.main_splitter.restoreState(state)
        if (state := self.settings.value(_SETTINGS_LEFT_SPLITTER)) is not None:
            self.left_splitter.restoreState(state)

    def save_layout(self) -> None:
        self.settings.setValue(_SETTINGS_GEOMETRY, self.saveGeometry())
        self.settings.setValue(_SETTINGS_MAIN_SPLITTER, self.main_splitter.saveState())
        self.settings.setValue(_SETTINGS_LEFT_SPLITTER, self.left_splitter.saveState())
        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard_changes():
            event.ignore()
            return
        if self.train_panel.running:
            answer = QMessageBox.question(
                self, APP_NAME, self.tr("학습 중입니다. 중지(재개 가능)하고 종료할까요?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.train_panel.shutdown()
        if self.augment_panel.is_building:
            self.augment_panel.cancel_build()
        self.augment_panel.wait_build()  # also covers a thread that just finished
        self.result_panel.shutdown()
        self.compare_panel.shutdown()
        self._close_edge_check()
        self._close_session()
        self.save_layout()
        self.sam_controller.shutdown()
        self.group_checker.shutdown()
        self.image_cache.shutdown()
        self.thumbnails.shutdown()
        super().closeEvent(event)

    # ---- tools -------------------------------------------------------------------
    def set_tool(self, tool: Tool) -> None:
        self.tool_actions[tool].setChecked(True)
        if tool != self.current_tool:
            self.current_tool = tool
            self.tool_changed.emit(tool)
        self._update_status()

    def _step_brush(self, up: bool) -> None:
        self.canvas.set_brush_size(brush.step_size(self.canvas.brush_size, up))

    def _on_brush_size(self, size: int) -> None:
        self.brush_bar.set_size(size)
        self.settings.setValue(_SETTINGS_BRUSH_SIZE, size)
        if self.current_tool == Tool.BRUSH:
            self.statusBar().showMessage(self.tr("브러시 크기 {0} px").format(size), 2000)

    # ---- project -------------------------------------------------------------------
    def load_project(self, project: Project | None) -> None:
        self._close_session()
        self._close_edge_check()  # its rows belong to the old project
        self.project = project
        self.roi_store = None if project is None else RoiStore(project.root)
        if self.roi_store is not None:
            self.roi_bar.set_options(self.roi_store.options)
            self.canvas.set_roi_options(self.roi_store.options)
        self.project_dirty = False
        self.undo_stack.clear()
        self.image_cache.invalidate()
        self.canvas.set_placeholder(
            self.tr("프로젝트를 열고 이미지를 추가하세요") if project is None
            else self.tr("'이미지 폴더 추가'로 이미지를 넣으세요")
        )
        # Panel first: _refresh_lists selects the first image, which the panel must keep.
        self.augment_panel.set_project(project)
        self.train_panel.set_project(project)
        self.result_panel.set_project(project)
        self.export_panel.set_project(project)
        self.compare_panel.set_project(project)
        self._exp_stage = None
        self._watch_images_dir()
        changes = self._sync_images_registry() if project is not None else None
        self._refresh_lists()
        if changes is not None and changes.changed:
            self.mark_dirty()
            self._show_sync_message(changes)
        self._update_title()
        self._start_group_check()
        self.project_changed.emit(project)

    # ---- images/ folder sync ---------------------------------------------------------------
    def _watch_images_dir(self) -> None:
        if self.images_watcher.directories():
            self.images_watcher.removePaths(self.images_watcher.directories())
        self.images_sync_timer.stop()
        if self.project is not None:
            self.project.images_dir.mkdir(parents=True, exist_ok=True)
            self.images_watcher.addPath(str(self.project.images_dir))

    def _sync_images_registry(self):
        changes = self.project.sync_images_dir(IMAGES_SETTLE_SEC)
        if changes.pending:  # copies still running: look again shortly
            self.images_sync_timer.start()
        return changes

    def sync_images_folder(self) -> bool:
        """Pick up images added to or deleted from the project's images/ folder outside the app.
        Returns True if the image list changed."""
        if self.project is None:
            return False
        edge_check_busy = self.edge_check_dialog is not None and self.edge_check_dialog.busy
        if self.augment_panel.is_building or self.compare_panel.busy or self.result_panel.busy or edge_check_busy:
            self.images_sync_timer.start()  # those threads read the image list: wait for them
            return False
        if self.current_image is not None and not self.project.image_abspath(self.current_image).is_file():
            self._close_session(save=False)  # its image is gone; saving would recreate its labels
        changes = self._sync_images_registry()
        if not changes.changed:
            return False
        if changes.removed:
            self.image_cache.invalidate()
        self.image_panel.apply_sync([e.path for e in changes.added], changes.removed)
        self.mark_dirty()
        self._update_status()
        self._show_sync_message(changes)
        self._start_group_check()
        return True

    # ---- slices / copies of other images (core/image_groups) --------------------------------
    def _start_group_check(self, show_result: bool = False) -> None:
        if self.project is None:
            return
        self._show_groups_when_done = self._show_groups_when_done or show_result
        items = [(e.path, self.project.image_abspath(e)) for e in self.project.config.images]
        self._group_generation = self.group_checker.check(items)

    def check_edges_dialog(self) -> None:
        """Menu 도구 → 테두리 검사 (F9): every confirmed label judged by the edge rule, non-modal."""
        if self.project is None:
            return
        self._save_all()  # judge the labels as they are on screen
        self._close_edge_check()  # one check at a time, always with the current labels and settings
        dialog = EdgeCheckDialog(self.project, self.project.config.edge_filter, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open_image.connect(self.open_image_for_labeling)
        dialog.destroyed.connect(lambda: setattr(self, "edge_check_dialog", None))
        self.edge_check_dialog = dialog
        dialog.show()

    def _close_edge_check(self) -> None:
        if self.edge_check_dialog is not None:
            self.edge_check_dialog.close()  # cancels its check; deleted later (WA_DeleteOnClose)
            self.edge_check_dialog = None

    def check_image_groups_dialog(self) -> None:
        """Menu: check again and list every group, also when nothing new was found."""
        if self.project is not None:
            self.statusBar().showMessage(self.tr("중복·조각 이미지 검사 중…"), 5000)
            self._start_group_check(show_result=True)

    def _on_groups_found(self, generation: int, groups: dict) -> None:
        if generation == self._group_generation and generation != self._groups_applied:
            self._apply_groups(generation, groups)

    def _before_build(self) -> None:
        # The split keeps groups together, so a build must not start on a stale result.
        if self.group_checker.busy:
            self.statusBar().showMessage(self.tr("중복·조각 이미지 검사를 마치는 중…"))
            result = self.group_checker.wait()
            if result is not None:
                self._on_groups_found(*result)
        self._save_all()

    def _apply_groups(self, generation: int, groups: dict) -> None:
        self._groups_applied = generation
        if self.project is None:
            return
        known = {e.path for e in self.project.config.images}
        groups = {m: s for m, s in groups.items() if m in known and s in known}
        old = self.project.config.source_groups
        new_members = sorted(m for m, s in groups.items() if old.get(m) != s)
        if groups != old:
            self.project.config.source_groups = groups
            self.mark_dirty()
            self.image_panel.refresh_groups()
        show_all = self._show_groups_when_done
        self._show_groups_when_done = False
        if new_members or show_all:
            self._show_group_notice(groups, new_members)

    def _show_group_notice(self, groups: dict, new_members: list[str]) -> None:
        by_source: dict[str, list[str]] = {}
        for member, source in sorted(groups.items()):
            by_source.setdefault(source, []).append(member)
        if not groups:
            text = self.tr("다른 이미지에서 잘라냈거나 똑같은 이미지는 없습니다.")
        else:
            text = self.tr(
                "같은 원본에서 잘라낸(또는 똑같은) 이미지 {0}장을 찾았습니다 (원본 {1}장).\n\n"
                "같은 결함이 두 번 학습되고, 원본과 조각이 Train과 Val에 나뉘면 Val이 이미 본 결함으로 "
                "채점됩니다. 원본이나 조각 중 한 종류만 쓰는 것을 권장합니다.\n\n"
                "그대로 두면 묶인 이미지는 Train/Val 같은 쪽에 둡니다.").format(len(groups), len(by_source))
        if self._group_notice is not None:
            self._group_notice.close()
        box = QMessageBox(QMessageBox.Icon.Information if not groups else QMessageBox.Icon.Warning,
                          self.tr("중복·조각 이미지"), text, QMessageBox.StandardButton.Ok, self)
        if groups:
            box.setDetailedText("\n".join(f"{source} ← {', '.join(members)}" for source, members in by_source.items()))
        box.setModal(False)
        box.show()
        self._group_notice = box
        self.statusBar().showMessage(
            self.tr("중복·조각 이미지 {0}장 (새로 {1}장)").format(len(groups), len(new_members)), 8000)

    def _show_sync_message(self, changes) -> None:
        parts = []
        if changes.added:
            parts.append(self.tr("추가 {0}장").format(len(changes.added)))
        if changes.removed:
            parts.append(self.tr("삭제 {0}장").format(len(changes.removed)))
        if changes.renamed:
            parts.append(self.tr("이름 변경 {0}").format(", ".join(f"{a} → {b}" for a, b in changes.renamed)))
        self.statusBar().showMessage(self.tr("이미지 폴더 변경 반영: {0}").format(" · ".join(parts)), 8000)

    def mark_dirty(self) -> None:
        self.project_dirty = True
        self._update_title()

    def save_project(self) -> None:
        if self.project is None:
            return
        self._save_session()
        self.project.save()
        self.project_dirty = False
        self._update_title()
        self.statusBar().showMessage(self.tr("저장했습니다"), 3000)

    def add_image_folder(self, folder: str | Path) -> int:
        if self.project is None:
            return 0
        added = self.project.add_images([folder])
        if added:
            self.mark_dirty()
            self._refresh_lists()
            self._start_group_check()
        return len(added)

    def new_project_dialog(self) -> None:
        if self._training_blocks_project_change() or not self._confirm_discard_changes():
            return
        dialog = NewProjectDialog(self)  # folder, name, task (detect/seg), colour mode (F1.1, F1.6)
        if dialog.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        try:
            self.load_project(Project.create(**dialog.values()))
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.critical(self, self.tr("새 프로젝트"), str(e))

    def open_project_dialog(self) -> None:
        if self._training_blocks_project_change() or not self._confirm_discard_changes():
            return
        folder = QFileDialog.getExistingDirectory(self, self.tr("프로젝트 폴더 열기"))
        if not folder:
            return
        try:
            self.load_project(Project.open(folder))
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.critical(self, self.tr("프로젝트 열기"), self.tr("프로젝트를 열 수 없습니다:\n{0}").format(e))

    def add_image_folder_dialog(self) -> None:
        if self.project is None:
            return
        folder = QFileDialog.getExistingDirectory(self, self.tr("이미지 폴더 선택"))
        if folder:
            count = self.add_image_folder(folder)
            self.statusBar().showMessage(self.tr("이미지 {0}장을 추가했습니다").format(count), 5000)

    def _confirm_discard_changes(self) -> bool:
        if not self.project_dirty:
            return True
        answer = QMessageBox.question(
            self,
            APP_NAME,
            self.tr("저장하지 않은 변경 사항이 있습니다. 저장할까요?"),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            self.save_project()
            return True
        return answer == QMessageBox.StandardButton.Discard

    # ---- view updates --------------------------------------------------------------
    def _refresh_lists(self) -> None:
        has_project = self.project is not None
        for act in self._project_actions:
            act.setEnabled(has_project)
        for b in (self.btn_add_class, self.btn_rename_class, self.btn_class_color):
            b.setEnabled(has_project)
        self._refresh_classes()
        self.image_panel.set_project(self.project)
        self._update_status()

    def _refresh_classes(self) -> None:
        row = self.class_list.currentRow()
        self.class_list.blockSignals(True)
        self.class_list.clear()
        classes = self.project.config.classes if self.project else []
        for i, cls in enumerate(classes):
            self.class_list.addItem(QListWidgetItem(_color_icon(cls.color), f"{i + 1}: {cls.name}" if i < 9 else cls.name))
        self.class_list.blockSignals(False)
        if classes:
            self.class_list.setCurrentRow(min(max(row, 0), len(classes) - 1))
        names, colors = [c.name for c in classes], [c.color for c in classes]
        self.canvas.set_classes(names, colors)
        self.label_panel.set_classes(names, colors)

    # ---- classes / labeling keys ----------------------------------------------------------
    def _on_class_row_changed(self, row: int) -> None:
        self.canvas.current_class_id = max(row, 0)

    def choose_class(self, index: int) -> None:
        """Number key: make it the class for new labels and apply it to the selected labels."""
        if self.project is None or index >= len(self.project.config.classes):
            return
        self.class_list.setCurrentRow(index)
        ids = self.canvas.selected_ids()
        if ids and self.session is not None:
            self.session.set_class(ids, index)

    def add_class_dialog(self) -> None:
        if self.project is None:
            return
        name, ok = QInputDialog.getText(self, self.tr("클래스 추가"), self.tr("클래스 이름:"))
        if ok and name.strip():
            self.add_class(name)

    def add_class(self, name: str) -> None:
        try:
            self.project.add_class(name)
        except ValueError as e:
            QMessageBox.warning(self, self.tr("클래스 추가"), str(e))
            return
        self.mark_dirty()
        self._refresh_classes()
        self.class_list.setCurrentRow(len(self.project.config.classes) - 1)

    def rename_class_dialog(self) -> None:
        row = self.class_list.currentRow()
        if self.project is None or row < 0:
            return
        old = self.project.config.classes[row].name
        name, ok = QInputDialog.getText(self, self.tr("클래스 이름 변경"), self.tr("새 이름:"), text=old)
        if ok and name.strip():
            self.rename_class(row, name)

    def rename_class(self, index: int, name: str) -> None:
        """Labels refer to classes by index, so a rename never touches label files."""
        try:
            self.project.rename_class(index, name)
        except ValueError as e:
            QMessageBox.warning(self, self.tr("클래스 이름 변경"), str(e))
            return
        self.mark_dirty()
        self._refresh_classes()

    def class_color_dialog(self) -> None:
        row = self.class_list.currentRow()
        if self.project is None or row < 0:
            return
        from PySide6.QtGui import QColor

        color = QColorDialog.getColor(QColor(self.project.config.classes[row].color), self, self.tr("클래스 색상"))
        if color.isValid():
            self.set_class_color(row, color.name())

    def set_class_color(self, index: int, color: str) -> None:
        self.project.set_class_color(index, color)
        self.mark_dirty()
        self._refresh_classes()

    def on_enter(self) -> None:
        if self.canvas.sam_shape_id is not None:
            # Accept the SAM result and start a new object with the next click.
            if self.session is not None:
                self.session.accept([self.canvas.sam_shape_id])
            self.canvas.reset_sam()
            return
        if self.canvas.finish_polygon():
            return
        if self.session is not None:
            self.session.accept(self.canvas.selected_ids())

    def on_backspace(self) -> None:
        if self.canvas.remove_last_point() or self.canvas.remove_last_sam_point():
            return
        rejected = self.session.reject(self.canvas.selected_ids()) if self.session is not None else 0
        if not rejected and platform_utils.BACKSPACE_DELETES_SELECTION:
            self.delete_selected()

    def delete_selected(self) -> None:
        if self.current_tool == Tool.CROP_ROI and self.roi_session is not None and self.canvas.selected_roi_ids():
            self.roi_session.delete(self.canvas.selected_roi_ids())
            return
        if self.session is not None:
            self.session.delete(self.canvas.selected_ids())

    def on_escape(self) -> None:
        if self.canvas.sam_active:
            # Drop the object being segmented: its suggestion and its prompts.
            if self.canvas.sam_shape_id is not None and self.session is not None:
                self.session.reject([self.canvas.sam_shape_id])
            self.canvas.reset_sam()
            return
        if not self.canvas.cancel_drawing():
            self.canvas.select_ids([])

    # ---- label session ------------------------------------------------------------------
    # ---- SAM assist (PRD F2.8 ②) -------------------------------------------------------------
    def _connect_sam(self) -> None:
        self.canvas.sam_prompt_changed.connect(self._on_sam_prompt)
        self.tool_changed.connect(lambda tool: tool == Tool.SAM and self._prepare_sam())
        self.set_sam_controller(self.sam_controller)

    def set_sam_controller(self, controller) -> None:
        """Use another SAM backend (tests pass a fake with the same signals)."""
        old = getattr(self, "_connected_sam", None)
        if old is not None:
            for sig in (old.result_ready, old.failed, old.status):
                sig.disconnect()
        self.sam_controller = controller
        controller.result_ready.connect(self._on_sam_result)
        controller.failed.connect(self._on_sam_failed)
        controller.status.connect(self._show_sam_status)
        self._connected_sam = controller

    def _show_sam_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 6000)

    def _sam_image_key(self) -> str | None:
        if self.project is None or self.current_image is None:
            return None
        return str(self.project.image_abspath(self.current_image))

    def _prepare_sam(self) -> None:
        """Encode the current image in the background as soon as SAM can be used on it."""
        key = self._sam_image_key()
        if key is not None and self.canvas.has_image:
            self.sam_controller.prepare(self.canvas._base, key)

    def _on_sam_prompt(self, prompt: SamPrompt) -> None:
        if self.session is None:
            return
        self._sam_generation += 1
        if prompt.is_empty:
            if self.canvas.sam_shape_id is not None:
                self.session.reject([self.canvas.sam_shape_id])
                self.canvas.sam_shape_id = None
            return
        self.statusBar().showMessage(self.tr("SAM 계산 중…"))
        self.sam_controller.request(self._sam_generation, self.canvas._base, self._sam_image_key(),
                                    SamPrompt(list(prompt.points), list(prompt.labels), prompt.box))

    def _on_sam_result(self, generation: int, result) -> None:
        if generation != self._sam_generation or self.session is None or self.canvas.tool != Tool.SAM:
            return  # stale: the user clicked again, switched image or tool
        self.statusBar().clearMessage()
        shape = None
        if result is not None:
            mask, score = result
            shape = mask_to_shape(mask, self.project.config.task, self.canvas.current_class_id, score)
        if shape is None:
            self.statusBar().showMessage(self.tr("SAM이 영역을 찾지 못했습니다. 점을 옮기거나 박스로 지정해 보세요"), 5000)
            return
        sid = self.canvas.sam_shape_id
        if sid is not None and self.session.shape(sid) is not None:
            # Same object refined by another click: update its suggestion in place.
            self.session.replace_shape(shape.model_copy(update={"id": sid}), self.tr("SAM 제안 갱신"))
        else:
            self.session.add_shape(shape)
            self.canvas.sam_shape_id = shape.id

    def _on_sam_failed(self, generation: int, message: str) -> None:
        if generation in (0, self._sam_generation):
            QMessageBox.warning(
                self, self.tr("SAM"),
                self.tr("SAM을 실행할 수 없습니다:\n{0}\n\n오프라인이면 [설정 > SAM 모델]에서 가중치 파일(.pt)을 직접 지정하세요.")
                .format(message))

    def choose_sam_model_dialog(self) -> None:
        items = SAM_MODEL_CHOICES + [self.tr("파일 선택…")]
        current = self.sam_controller.model
        idx = items.index(current) if current in items else 0
        choice, ok = QInputDialog.getItem(self, self.tr("SAM 모델"), self.tr("모델:"), items, idx, False)
        if not ok:
            return
        if choice == items[-1]:
            choice, _ = QFileDialog.getOpenFileName(self, self.tr("SAM 가중치 파일"), "", "PyTorch (*.pt)")
            if not choice:
                return
        self.set_sam_model(choice)

    def set_sam_model(self, model: str) -> None:
        self.sam_controller.set_model(model)
        self.settings.setValue(_SETTINGS_SAM_MODEL, model)
        if self.current_tool == Tool.SAM:
            self._prepare_sam()

    # ---- autosave (PRD F2.4) ---------------------------------------------------------------
    def set_autosave_interval(self, seconds: int) -> None:
        """0 turns periodic autosave off (labels are still saved on image switch and close)."""
        self.autosave_interval = max(0, int(seconds))
        self.settings.setValue(_SETTINGS_AUTOSAVE, self.autosave_interval)
        if self.autosave_interval:
            self.autosave_timer.start(self.autosave_interval * 1000)
        else:
            self.autosave_timer.stop()

    def autosave(self) -> bool:
        """Save pending label edits and project changes. Returns True if anything was written."""
        if self.project is None:
            return False
        wrote = False
        if self.session is not None and self.session.dirty:
            self._save_session()
            wrote = True
        if self.project_dirty:
            self.project.save()
            self.project_dirty = False
            self._update_title()
            wrote = True
        if wrote:
            self.statusBar().showMessage(self.tr("자동 저장됨 {0}").format(time.strftime("%H:%M:%S")), 3000)
        return wrote

    def autosave_interval_dialog(self) -> None:
        sec, ok = QInputDialog.getInt(self, self.tr("자동 저장"), self.tr("자동 저장 간격(초, 0 = 끄기):"),
                                      self.autosave_interval, 0, 3600, 5)
        if ok:
            self.set_autosave_interval(sec)

    def _build_menu(self) -> None:
        # The same QActions as the toolbar / window shortcuts, so the menus show every shortcut
        # (⌘N, ⌘S, ⇧⌘Z … on a Mac, where users look for them in the menu bar).
        bar = self.menuBar()
        file_menu = bar.addMenu(self.tr("파일"))
        for act in (self.act_new, self.act_open, self.act_add_folder, self.act_save):
            file_menu.addAction(act)
        edit_menu = bar.addMenu(self.tr("편집"))
        for act in (self.act_undo, self.act_redo, self.act_delete, self.act_toggle_ok):
            edit_menu.addAction(act)
        edit_menu.addSeparator()
        for act in (self.act_split_train, self.act_split_val, self.act_split_auto, self.act_synthetic):
            edit_menu.addAction(act)
        view_menu = bar.addMenu(self.tr("보기"))
        for act in (self.act_fit, self.act_zoom_100, self.act_prev_image, self.act_next_image):
            view_menu.addAction(act)
        tool_menu = bar.addMenu(self.tr("도구"))
        for tool in (Tool.SELECT, Tool.BBOX, Tool.POLYGON, Tool.BRUSH, Tool.CROP_ROI, Tool.SAM, Tool.ERASER):
            tool_menu.addAction(self.tool_actions[tool])
        tool_menu.addSeparator()
        self.act_check_groups = tool_menu.addAction(self.tr("중복·조각 이미지 검사"), self.check_image_groups_dialog)
        self._project_actions.append(self.act_check_groups)
        self.act_check_groups.setEnabled(self.project is not None)
        self.act_check_edges = tool_menu.addAction(self.tr("테두리 검사 (라벨 박스)"), self.check_edges_dialog)
        self.act_check_edges.setToolTip(self.tr("확정 라벨마다 테두리 필터 규칙으로 판정 — 필터가 숨길 진짜 결함 확인"))
        self._project_actions.append(self.act_check_edges)
        self.act_check_edges.setEnabled(self.project is not None)
        menu = bar.addMenu(self.tr("설정"))
        self.act_sam_model = menu.addAction(self.tr("SAM 모델…"), self.choose_sam_model_dialog)
        self.act_autosave = menu.addAction(self.tr("자동 저장 간격…"), self.autosave_interval_dialog)
        self.act_export_pattern = menu.addAction(self.tr("내보내기 파일명 패턴…"), self.export_pattern_dialog)

    def export_pattern_dialog(self) -> None:
        tokens = ", ".join("{" + t + "}" for t in NAME_TOKENS)
        text, ok = QInputDialog.getText(
            self, self.tr("내보내기 파일명 패턴"),
            self.tr("사용 가능: {0}\n기본: {1}").format(tokens, DEFAULT_NAME_PATTERN), text=self.export_panel.pattern)
        if ok:
            self.set_export_pattern(text.strip() or DEFAULT_NAME_PATTERN)

    def set_export_pattern(self, pattern: str) -> bool:
        try:
            self.export_panel.set_pattern(pattern)
        except ValueError as e:
            QMessageBox.warning(self, self.tr("내보내기 파일명 패턴"), str(e))
            return False
        self.settings.setValue(_SETTINGS_EXPORT_PATTERN, pattern)
        return True

    def _open_session(self, path: str, size: tuple[int, int]) -> None:
        self.session = LabelSession(self.project, path, self.undo_stack, size=size, parent=self)
        self.session.changed.connect(self._on_session_changed)
        self.canvas.set_session(self.session)
        self.label_panel.set_session(self.session)
        if self.roi_store is not None:
            self.roi_session = RoiSession(self.roi_store, path, size, self.undo_stack, parent=self)
            self.roi_session.changed.connect(self._update_roi_counts)
            self.canvas.set_roi_session(self.roi_session)
            self._update_roi_counts()
        self._update_status()
        if self.current_tool == Tool.SAM:
            self._prepare_sam()

    def _current_annotations(self, path: str):
        """Unsaved labels of the image being edited (for previews), else None."""
        if self.session is not None and self.session.image_path == path:
            return self.session.annotations.model_copy(deep=True)
        return None

    # ---- Result tab: relabel loop and model suggestions (F5.4, F5.5, F2.8 ①) -------------------
    def open_image_for_labeling(self, path: str) -> None:
        """Show an image from the Result tab on the canvas so its labels can be fixed."""
        if self.image_panel._item(path) is None:
            return
        if self.image_panel._item(path).isHidden():
            self.image_panel.set_filter("all")
        self.image_panel.select(path)
        self.canvas.setFocus()

    def _on_labels_changed_externally(self, paths: list) -> None:
        """Label files were rewritten outside the session (model suggestions): refresh views and
        reload the open image so its session does not overwrite them with stale shapes."""
        for path in paths:
            self.image_panel.refresh_status(path)
        if self.session is not None and self.session.image_path in paths and self.canvas.has_image:
            size = self.canvas.image_size
            path = self.session.image_path
            self._close_session()
            self._open_session(path, size)

    def _on_run_finished(self, run: str, state: str) -> None:
        self.result_panel.refresh_runs(select=run)
        self.export_panel.refresh_runs(select=run)
        self.compare_panel.refresh_runs()
        self._on_exp_run_finished(run, state)
        if self._quick_stage == "train":
            self._quick_stage = None
            if state == "finished":
                info = next((r for r in self.result_panel.runs if str(r.path) == run), None)
                if info is not None:
                    self.tabs.setCurrentWidget(self.result_panel)
                    self.result_panel.suggest(info)
            else:
                self.result_panel.suggest_status.setText(self.tr("빠른 학습이 끝나지 않아 제안을 만들지 않았습니다"))

    def start_quick_suggest(self) -> None:
        """Rebuild the dataset from the confirmed labels, train yolo11n briefly, then suggest."""
        if self.project is None or self._quick_stage is not None or self._exp_stage is not None:
            return
        if self.train_panel.running or self.augment_panel.is_building:
            QMessageBox.information(self, APP_NAME, self.tr("진행 중인 학습/빌드가 끝난 뒤에 시도하세요."))
            return
        self._quick_stage = "build"
        self.result_panel.suggest_status.setText(self.tr("빠른 학습 준비: 데이터셋 빌드 중…"))
        self.augment_panel.start_build("full")  # quick suggest trains on full images

    def _on_quick_build_finished(self, _report) -> None:
        if self._quick_stage != "build":
            return
        self._quick_stage = "train"
        self.result_panel.suggest_status.setText(self.tr("빠른 학습 중 (yolo11n, 최대 30 epoch)… Train 탭에서 진행 확인"))
        if not self.train_panel.start(quick_suggest_config()):
            self._quick_stage = None
            self.result_panel.suggest_status.setText(self.tr("빠른 학습을 시작하지 못했습니다"))

    def _on_quick_build_stopped(self, message: str) -> None:
        if self._quick_stage == "build":
            self._quick_stage = None
            self.result_panel.suggest_status.setText(self.tr("데이터셋 빌드 실패로 빠른 학습을 취소했습니다: {0}").format(message))

    # ---- comparison experiments (F8.1, F8.6) -------------------------------------------------------
    # Steps run one after another: ("build", "full"|"crop") then ("train", TrainConfig) for every
    # config, then the trained runs are compared. Started by "Full + Crop 비교 학습" or by
    # "같은 설정으로 재실험" on a report.
    def start_compare_experiment(self) -> bool:
        """Rebuild dataset/full and dataset/crop, train both with the Train tab's model,
        hyperparameters and seed, then compare the two runs on the same Val images."""
        return self.start_experiment(list(experiment_configs(self.train_panel.current_config())))

    def start_experiment(self, configs: list) -> bool:
        if self.project is None or self._exp_stage is not None or self._quick_stage is not None or not configs:
            return False
        if self.train_panel.running or self.augment_panel.is_building or self.compare_panel.busy:
            QMessageBox.information(self, APP_NAME, self.tr("진행 중인 학습/빌드/비교가 끝난 뒤에 시도하세요."))
            return False
        datasets = [d for d in ("full", "crop") if any(c.dataset == d for c in configs)]
        self._exp_steps = [("build", d) for d in datasets] + [("train", c) for c in configs]
        self._exp_total = len(self._exp_steps)
        self._exp_runs = []
        self._exp_next()
        return True

    def _exp_status(self, text: str) -> None:
        self.compare_panel.exp_status.setText(text)

    def _exp_fail(self, text: str) -> None:
        self._exp_stage = None
        self._exp_steps = []
        self._exp_status(text)

    def _exp_next(self) -> None:
        if not self._exp_steps:
            self._exp_stage = None
            if len(self._exp_runs) >= 2:
                self._exp_status(self.tr("학습 완료 — run {0}개를 비교합니다").format(len(self._exp_runs)))
                self.compare_panel.refresh_runs()
                self.compare_panel.set_checked(self._exp_runs)
                self.tabs.setCurrentWidget(self.compare_panel)
                self.compare_panel.compare(self._exp_runs)
            else:
                self._exp_status(self.tr("학습 완료"))
            return
        kind, arg = self._exp_steps.pop(0)
        step = self._exp_total - len(self._exp_steps)
        if kind == "build":
            self._exp_stage = "build"
            self._exp_status(self.tr("{0}/{1} {2} 데이터셋 빌드 중…").format(step, self._exp_total, arg.capitalize()))
            self.augment_panel.start_build(arg)
        else:
            self._exp_stage = "train"
            self._exp_status(self.tr("{0}/{1} {2} · {3} 학습 중… (Train 탭에서 진행 확인)").format(
                step, self._exp_total, arg.dataset.capitalize(), arg.model))
            if not self.train_panel.start(arg):
                self._exp_fail(self.tr("학습을 시작하지 못해 비교 실험을 멈췄습니다"))

    def _on_exp_build_finished(self, _report) -> None:
        if self._exp_stage == "build":
            self._exp_next()

    def _on_exp_build_stopped(self, message: str) -> None:
        if self._exp_stage == "build":
            self._exp_fail(self.tr("데이터셋 빌드 실패로 비교 실험을 멈췄습니다: {0}").format(message))

    def _on_exp_run_finished(self, run: str, state: str) -> None:
        if self._exp_stage != "train":
            return
        if state != "finished":
            self._exp_fail(self.tr("학습이 끝나지 않아({0}) 비교 실험을 멈췄습니다").format(state))
            return
        self._exp_runs.append(run)
        self._exp_next()

    def _on_roi_options(self, options) -> None:
        self.canvas.set_roi_options(options)
        if self.roi_store is not None:
            self.roi_store.set_options(options)

    def _update_roi_counts(self) -> None:
        if self.roi_session is not None:
            self.roi_bar.set_counts(len(self.roi_session.image_rois), len(self.roi_session.common))

    def _before_training(self) -> bool:
        if self.augment_panel.is_building:
            QMessageBox.information(self, self.tr("학습 시작"), self.tr("데이터셋 빌드가 끝난 뒤에 학습을 시작하세요."))
            return False
        self._save_all()
        return True

    def _training_blocks_project_change(self) -> bool:
        if self.train_panel.running:
            QMessageBox.information(self, APP_NAME, self.tr("학습 중에는 프로젝트를 바꿀 수 없습니다. 먼저 학습을 중지하세요."))
            return True
        return False

    def _save_all(self) -> None:
        self._save_session()
        if self.project is not None and self.project_dirty:
            self.project.save()
            self.project_dirty = False
            self._update_title()

    def _save_session(self) -> None:
        if self.session is not None and self.session.save():
            self.image_panel.refresh_status(self.session.image_path)

    def _close_session(self, save: bool = True) -> None:
        """Save the current image's labels and start a fresh undo history for the next one."""
        if save:
            self._save_session()
        if self.roi_session is not None:
            self.canvas.set_roi_session(None)
            self.roi_session.deleteLater()
            self.roi_session = None
        if self.session is not None:
            self.canvas.set_session(None)
            self.label_panel.set_session(None)
            self.session.deleteLater()
            self.session = None
        self.undo_stack.clear()

    def _on_session_changed(self) -> None:
        if self.session is None:
            return
        ann = self.session.annotations
        entry = self.project.image(self.session.image_path)
        if entry.is_ok and ann.confirmed():  # drawing a defect on an OK image: it is not OK any more
            self.project.set_ok(entry.path, False)
            self.mark_dirty()
            self.statusBar().showMessage(self.tr("라벨을 그려 OK 이미지 표시를 해제했습니다"), 4000)
        self.image_panel.set_status(
            self.session.image_path, LabelStatus(len(ann.confirmed()), len(ann.suggestions()), entry.is_ok))
        self._update_status()

    def toggle_ok(self) -> None:
        """Mark the current image as OK (no defect) or clear the mark (F1.5). Refused while the
        image has confirmed labels: an OK image trains as background with an empty label."""
        if self.project is None or self.current_image is None:
            return
        path = self.current_image
        entry = self.project.image(path)
        if not entry.is_ok:
            confirmed = (len(self.session.annotations.confirmed()) if self.session is not None
                         and self.session.image_path == path else self.project.label_status(entry).confirmed)
            if confirmed:
                self.statusBar().showMessage(
                    self.tr("라벨이 있는 이미지는 OK로 표시할 수 없습니다. 라벨을 지운 뒤 다시 하세요"), 5000)
                return
        self.project.set_ok(path, not entry.is_ok)
        self.mark_dirty()
        if self.session is not None and self.session.image_path == path:
            self._on_session_changed()
        else:
            self.image_panel.refresh_status(path)
        self.statusBar().showMessage(
            self.tr("OK 이미지로 표시했습니다 (배경으로 학습)") if entry.is_ok else self.tr("OK 표시를 해제했습니다"), 3000)
        self._update_status()

    # ---- Train/Val by hand, synthetic images ----------------------------------------------------
    def _group_of(self, path: str) -> list[str]:
        """The image and every image cut from the same source (they share one split)."""
        groups = self.project.config.source_groups
        key = groups.get(path, path)
        return [e.path for e in self.project.config.images if groups.get(e.path, e.path) == key]

    def set_current_split(self, split: str) -> None:
        """Put the current image (and its source group) in Train / Val by hand, or back to auto."""
        if self.project is None or self.current_image is None:
            return
        group = self._group_of(self.current_image)
        if split == "val" and any(self.project.image(p).synthetic for p in group):
            self.statusBar().showMessage(
                self.tr("합성 이미지(또는 합성 이미지와 같은 원본 묶음)는 Val로 지정할 수 없습니다"), 5000)
            return
        for path in group:
            self.project.set_split(path, split)
        self.mark_dirty()
        self.image_panel.refresh_marks(group)
        names = {"train": "Train", "val": "Val", "auto": self.tr("자동")}
        extra = self.tr(" (같은 원본 묶음 {0}장 함께)").format(len(group) - 1) if len(group) > 1 else ""
        self.statusBar().showMessage(self.tr("{0}: {1}(으)로 지정했습니다{2}").format(
            self.current_image, names[split], extra), 5000)

    def toggle_synthetic(self) -> None:
        """Mark the current image as made by the user (pasted / moved defects): always Train."""
        if self.project is None or self.current_image is None:
            return
        entry = self.project.image(self.current_image)
        self.project.set_synthetic(entry.path, not entry.synthetic)
        if entry.synthetic and entry.split == "val":
            self.project.set_split(entry.path, "auto")  # a synthetic image is never measured on
        self.mark_dirty()
        self.act_synthetic.setChecked(entry.synthetic)
        self.image_panel.refresh_marks([entry.path])
        self.statusBar().showMessage(
            self.tr("합성 이미지로 표시했습니다: 같은 원본 묶음과 함께 항상 Train에 들어갑니다") if entry.synthetic
            else self.tr("합성 이미지 표시를 해제했습니다"), 5000)

    def _on_current_image_changed(self, path: str | None) -> None:
        self._close_session()
        self.current_image = path
        self.act_synthetic.setChecked(bool(path and self.project and self.project.image(path).synthetic))
        self.augment_panel.set_current_image(path)
        if self.project is None or path is None:
            self.canvas.set_image(None)
        else:
            # Never decode on the UI thread: a 4K PNG takes ~150 ms. Cached images show at once;
            # otherwise show a loading hint and display the image when the worker finishes.
            abspath = self.project.image_abspath(path)
            if not self.image_cache.get_async(abspath, self._emit_image_loaded):
                self.canvas.set_placeholder(self.tr("불러오는 중…"))
                self.canvas.set_image(None)
            self.image_cache.prefetch([self.project.image_abspath(p) for p in self.image_panel.neighbours()])
        self._update_status()

    def _emit_image_loaded(self, abspath: Path, image) -> None:
        try:
            self._image_loaded.emit(abspath, image)  # queued to the UI thread
        except RuntimeError:  # window destroyed while decoding
            pass

    def _show_loaded_image(self, abspath: Path, image) -> None:
        if self.project is None or self.current_image is None:
            return
        if abspath != self.project.image_abspath(self.current_image):
            return  # user already moved on to another image
        if image is None:
            self.canvas.set_placeholder(self.tr("이미지를 읽을 수 없습니다: {0}").format(self.current_image))
        self.canvas.set_image(image)
        if image is not None and self.session is None:
            self._open_session(self.current_image, (image.shape[1], image.shape[0]))

    def _update_title(self) -> None:
        title = f"{APP_NAME} {__version__}"
        if self.project is not None:
            title = f"{self.project.config.name}{' *' if self.project_dirty else ''} — {title}"
        self.setWindowTitle(title)

    def _update_status(self) -> None:
        total = self.image_list.count()
        row = self.image_list.currentRow()
        self.status_image.setText(self.tr("이미지 {0}/{1}").format(row + 1 if row >= 0 else 0, total))
        if self.session is not None:
            ann = self.session.annotations
            text = self.tr("라벨 {0}").format(len(ann.confirmed()))
            if ann.suggestions():
                text += self.tr(" · 제안 {0}").format(len(ann.suggestions()))
        else:
            text = self.tr("라벨 {0}").format(0)
        self.status_labels.setText(text)
        self.status_tool.setText(self.tr("도구: {0}").format(self.tool_actions[self.current_tool].text()))

    def set_cursor_position(self, x: int | None, y: int | None) -> None:
        """Called by the canvas with the image pixel under the cursor (None when outside)."""
        self.status_coords.setText("" if x is None else f"x {x}, y {y}")

    @property
    def image_ready(self) -> bool:
        """True once the current image is on the canvas (tests wait on this)."""
        return self.canvas.has_image

    def _start_device_detection(self) -> None:
        # torch import takes ~1 s, so detect off the UI thread; the signal is delivered queued.
        self._device_detected.connect(self.status_device.setText)
        self._device_detected.connect(self.train_panel.set_detected_device)
        self._device_detected.connect(self.export_panel.set_detected_device)

        def run() -> None:
            device = detect_device()
            try:
                self._device_detected.emit(device)
            except RuntimeError:  # window already destroyed
                pass

        threading.Thread(target=run, daemon=True).start()


SCROLL_TAB_MAX_MIN_WIDTH = 420  # px, Windows/Linux (see ScrollTabs)


class ScrollTabs(QTabWidget):
    """Tab widget whose pages scroll. Without it the tallest panel (Augment, ~810 px) set the
    window's minimum height to 893 px, more than a 13" MacBook's 870 px usable height, so the
    window could not fit on the screen. Callers keep using the panels themselves:
    setCurrentWidget(panel) / currentWidget() / indexOf(panel) map to and from the scroll areas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._areas: dict[QWidget, QScrollArea] = {}

    def addTab(self, page: QWidget, label: str) -> int:  # type: ignore[override]
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        needed = page.minimumSizeHint().width() + area.verticalScrollBar().sizeHint().width()
        if OVERLAY_SCROLLBARS:
            # macOS: vertical scrolling only. Scrollbars stay hidden until you scroll, so a panel
            # clipped sideways looks broken; the widths fit a 1440 px MacBook screen anyway.
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setMinimumWidth(needed)
        else:
            # Windows/Linux: the native styles are wider (the window needed 1674 px on Windows);
            # cap the tab width and let a wider panel scroll sideways with a visible scrollbar.
            area.setMinimumWidth(min(needed, SCROLL_TAB_MAX_MIN_WIDTH))
        area.setWidget(page)
        self._areas[page] = area
        return super().addTab(area, label)

    def setCurrentWidget(self, widget: QWidget) -> None:  # type: ignore[override]
        super().setCurrentWidget(self._areas.get(widget, widget))

    def currentWidget(self) -> QWidget | None:  # type: ignore[override]
        return self._page(super().currentWidget())

    def widget(self, index: int) -> QWidget | None:  # type: ignore[override]
        return self._page(super().widget(index))

    def indexOf(self, widget: QWidget) -> int:  # type: ignore[override]
        return super().indexOf(self._areas.get(widget, widget))

    def _page(self, w: QWidget | None) -> QWidget | None:
        return w.widget() if isinstance(w, QScrollArea) and w in self._areas.values() else w


def _separator(parent: QWidget) -> QAction:
    act = QAction(parent)
    act.setSeparator(True)
    return act


def _titled(title: str, widget: QWidget) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(QLabel(f"<b>{title}</b>"))
    layout.addWidget(widget)
    return box


def _placeholder_panel(text: str) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("color: gray;")
    layout.addWidget(label)
    return panel


def _color_icon(color: str) -> QIcon:
    pix = QPixmap(14, 14)
    pix.fill(QColor(color))
    return QIcon(pix)
