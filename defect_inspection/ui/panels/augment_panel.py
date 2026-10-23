"""Augment tab (PRD F3.3-F3.5, F3.2 build, F7.2-F7.5 crop): transform settings, presets, 3x3 preview,
crop settings, dataset build (Full or Crop)."""

from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.annotation import ImageAnnotations
from defect_inspection.core.augment_config import (
    TRANSFORM_INFO,
    AugmentConfig,
    TransformSetting,
    available_transform_keys,
    load_presets,
    save_preset,
)
from defect_inspection.core.crop import CropBuildReport, build_crop_dataset
from defect_inspection.core.crop_config import CROP_PRESETS, CropConfig
from defect_inspection.core.dataset import BuildCancelled, BuildReport, DatasetBuildError, build_full_dataset
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imread_unicode
from defect_inspection.ui.canvas.image_canvas import numpy_to_qimage

CUSTOM = "직접 설정"
PREVIEW_COUNT = 9
PREVIEW_TILE = 120


class TransformRow:
    """Checkbox + strength slider + probability spin box for one transform."""

    def __init__(self, key: str, label: str, enabled_for_project: bool) -> None:
        self.key = key
        self.check = QCheckBox(label)
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setToolTip("강도")
        self.prob = QDoubleSpinBox()
        self.prob.setRange(0.0, 1.0)
        self.prob.setSingleStep(0.05)
        self.prob.setDecimals(2)
        self.prob.setToolTip("적용 확률")
        self.available = enabled_for_project
        for w in (self.check, self.strength, self.prob):
            w.setEnabled(enabled_for_project)
        if not enabled_for_project:
            self.check.setToolTip("그레이스케일 프로젝트에서는 사용하지 않습니다")

    def load(self, s: TransformSetting) -> None:
        self.check.setChecked(s.enabled and self.available)
        self.strength.setValue(round(s.strength * 100))
        self.prob.setValue(s.p)

    def value(self) -> TransformSetting:
        return TransformSetting(enabled=self.check.isChecked() and self.available,
                                p=self.prob.value(), strength=self.strength.value() / 100)


class _BuildWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)  # BuildReport
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project, mode: str) -> None:
        super().__init__()
        self.project, self.mode = project, mode
        self.stop = threading.Event()

    def run(self) -> None:
        build = build_crop_dataset if self.mode == "crop" else build_full_dataset
        try:
            report = build(self.project, progress=self.progress.emit, cancel=self.stop.is_set)
        except BuildCancelled:
            self.cancelled.emit()
        except DatasetBuildError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 - shown to the user, never crash the UI
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(report)


class AugmentPanel(QWidget):
    changed = Signal()  # settings changed (project.yaml becomes dirty)
    build_finished = Signal(object)  # BuildReport (full) or CropBuildReport (crop)
    build_stopped = Signal(str)  # failed or cancelled: message
    _previews_ready = Signal(int, list)  # generation, list[np.ndarray] (from worker thread)
    _preview_failed = Signal(int, str)

    def __init__(self, presets_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.presets_dir = Path(presets_dir)
        self.project: Project | None = None
        self.image_path: str | None = None
        # Main window hooks: current (unsaved) labels of an image, and "save everything" before a build.
        self.annotations_provider: Callable[[str], ImageAnnotations | None] = lambda path: None
        self.before_build: Callable[[], None] = lambda: None
        self._loading = False
        self._preview_gen = 0
        self._thread: QThread | None = None
        self._worker: _BuildWorker | None = None
        self._building = False
        self._pending_build: str | None = None  # requested while the last build's thread was ending

        self._build_ui()
        self._previews_ready.connect(self._show_previews)
        self._preview_failed.connect(self._show_preview_error)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self.set_project(None)

    # ---- UI --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self._on_preset_chosen)
        self.btn_save_preset = QPushButton(self.tr("프리셋 저장…"))
        self.btn_save_preset.clicked.connect(self.save_preset_dialog)
        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("프리셋")))
        top.addWidget(self.preset_combo, 1)
        top.addWidget(self.btn_save_preset)

        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("<b>변환</b>")), 0, 0)
        grid.addWidget(QLabel(self.tr("<b>강도</b>")), 0, 1)
        grid.addWidget(QLabel(self.tr("<b>확률</b>")), 0, 2)
        self.rows: dict[str, TransformRow] = {}
        r, group = 1, None
        for info in TRANSFORM_INFO:
            if info.group != group:
                group = info.group
                head = QLabel(f"<i>{self.tr(group)}</i>")
                grid.addWidget(head, r, 0, 1, 3)
                r += 1
            row = TransformRow(info.key, self.tr(info.label), True)
            row.check.toggled.connect(self._on_edit)
            row.strength.valueChanged.connect(self._on_edit)
            row.prob.valueChanged.connect(self._on_edit)
            grid.addWidget(row.check, r, 0)
            grid.addWidget(row.strength, r, 1)
            grid.addWidget(row.prob, r, 2)
            self.rows[info.key] = row
            r += 1

        self.multiplier = QSpinBox()
        self.multiplier.setRange(1, 500)
        self.multiplier.setSuffix(self.tr(" 장/원본"))
        self.min_visibility = QSpinBox()
        self.min_visibility.setRange(0, 100)
        self.min_visibility.setSuffix(" %")
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100000)
        self.min_area.setSuffix(" px²")
        self.cp_enabled = QCheckBox(self.tr("Copy-Paste (결함을 OK 이미지에 합성)"))
        self.cp_prob = QDoubleSpinBox()
        self.cp_prob.setRange(0.0, 1.0)
        self.cp_prob.setSingleStep(0.1)
        self.cp_max = QSpinBox()
        self.cp_max.setRange(1, 10)
        for w in (self.multiplier, self.min_visibility, self.min_area, self.cp_max):
            w.valueChanged.connect(self._on_edit)
        self.cp_prob.valueChanged.connect(self._on_edit)
        self.cp_enabled.toggled.connect(self._on_edit)
        dt_box = self._build_defect_transform_ui()

        opts = QGridLayout()
        opts.addWidget(QLabel(self.tr("증강 개수")), 0, 0)
        opts.addWidget(self.multiplier, 0, 1)
        opts.addWidget(QLabel(self.tr("라벨 유지: 보이는 비율 ≥")), 1, 0)
        opts.addWidget(self.min_visibility, 1, 1)
        opts.addWidget(QLabel(self.tr("라벨 유지: 넓이 ≥")), 2, 0)
        opts.addWidget(self.min_area, 2, 1)
        opts.addWidget(self.cp_enabled, 3, 0, 1, 2)
        opts.addWidget(QLabel(self.tr("  확률 / 최대 개수")), 4, 0)
        cp_row = QHBoxLayout()
        cp_row.addWidget(self.cp_prob)
        cp_row.addWidget(self.cp_max)
        opts.addLayout(cp_row, 4, 1)
        opts_box = QGroupBox(self.tr("옵션"))
        opts_box.setLayout(opts)

        crop_box = self._build_crop_ui()

        # Transforms, options and crop settings scroll together, so none gets squeezed.
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addLayout(grid)
        settings_layout.addWidget(opts_box)
        settings_layout.addWidget(dt_box)
        settings_layout.addWidget(crop_box)
        settings_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        scroll.setMinimumHeight(200)

        # preview
        self.btn_preview = QPushButton(self.tr("다시 뽑기"))
        self.btn_preview.clicked.connect(self.refresh_preview)
        self.preview_status = QLabel()
        self.preview_status.setStyleSheet("color: gray;")
        self.preview_labels = []
        pgrid = QGridLayout()
        pgrid.setSpacing(2)
        for i in range(PREVIEW_COUNT):
            lbl = QLabel()
            lbl.setFixedSize(PREVIEW_TILE, PREVIEW_TILE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #333;")
            pgrid.addWidget(lbl, i // 3, i % 3)
            self.preview_labels.append(lbl)
        prev_head = QHBoxLayout()
        prev_head.addWidget(QLabel(self.tr("<b>미리보기</b> (현재 이미지)")))
        prev_head.addStretch(1)
        prev_head.addWidget(self.btn_preview)
        preview = QWidget()
        prev_box = QVBoxLayout(preview)
        prev_box.setContentsMargins(0, 4, 0, 0)
        prev_box.addLayout(prev_head)
        prev_box.addLayout(pgrid)
        prev_box.addWidget(self.preview_status)

        # build
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(self.tr("Full (전체 이미지)"), "full")
        self.mode_combo.addItem(self.tr("Crop (잘라낸 영역)"), "crop")
        self.mode_combo.setToolTip(self.tr("dataset/full 과 dataset/crop 은 따로 만들어지고 함께 보관됩니다"))
        self.btn_build = QPushButton(self.tr("데이터셋 빌드"))
        self.btn_build.clicked.connect(self.start_build)
        self.btn_cancel = QPushButton(self.tr("취소"))
        self.btn_cancel.clicked.connect(self.cancel_build)
        self.btn_cancel.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.build_status = QLabel()
        self.build_status.setWordWrap(True)
        build_row = QHBoxLayout()
        build_row.addWidget(self.mode_combo)
        build_row.addWidget(self.btn_build)
        build_row.addWidget(self.btn_cancel)
        build_row.addWidget(self.progress, 1)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(preview)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.splitter, 1)
        layout.addLayout(build_row)
        layout.addWidget(self.build_status)

    def _build_crop_ui(self) -> QGroupBox:
        """Crop dataset settings (F7.2-F7.4); stored in project.yaml as `crop`."""
        self.crop_preset = QComboBox()
        self.crop_preset.addItems(list(CROP_PRESETS) + [self.tr(CUSTOM)])
        self.crop_preset.activated.connect(self._on_crop_preset_chosen)
        self.use_roi = QCheckBox(self.tr("수동 ROI (Crop ROI 도구로 그린 영역)"))
        self.use_defect = QCheckBox(self.tr("결함 중심 crop"))
        self.use_background = QCheckBox(self.tr("배경 crop (결함 없는 영역)"))
        self.use_tiles = QCheckBox(self.tr("슬라이딩 타일 (이미지 전체 분할)"))
        self.crop_size = QSpinBox()
        self.crop_size.setRange(32, 8192)
        self.crop_size.setSingleStep(32)
        self.crop_size.setSuffix(" px")
        self.per_defect = QSpinBox()
        self.per_defect.setRange(1, 50)
        self.per_defect.setSuffix(self.tr(" 개/라벨"))
        self.jitter = QSpinBox()
        self.jitter.setRange(0, 100)
        self.jitter.setSuffix(" %")
        self.jitter.setToolTip(self.tr("0 % = 결함이 항상 중앙, 100 % = 결함이 crop 안 어디든"))
        self.bg_ratio = QDoubleSpinBox()
        self.bg_ratio.setRange(0.0, 10.0)
        self.bg_ratio.setSingleStep(0.5)
        self.bg_ratio.setDecimals(1)
        self.bg_ratio.setPrefix(self.tr("결함 1 : 배경 "))
        self.tile_size = QSpinBox()
        self.tile_size.setRange(32, 8192)
        self.tile_size.setSingleStep(32)
        self.tile_size.setSuffix(" px")
        self.overlap = QDoubleSpinBox()
        self.overlap.setDecimals(0)
        self.overlap_unit = QComboBox()
        self.overlap_unit.addItem("%", "percent")
        self.overlap_unit.addItem("px", "px")
        self.crop_multiplier = QSpinBox()
        self.crop_multiplier.setRange(0, 500)
        self.crop_multiplier.setSuffix(self.tr(" 장/crop"))
        self.crop_multiplier.setToolTip(self.tr("Train crop마다 만드는 증강 이미지 수 (위 변환 설정 사용, Val은 증강 안 함)"))
        for w in (self.use_roi, self.use_defect, self.use_background, self.use_tiles):
            w.toggled.connect(self._on_crop_edit)
        for w in (self.crop_size, self.per_defect, self.jitter, self.tile_size, self.crop_multiplier):
            w.valueChanged.connect(self._on_crop_edit)
        self.bg_ratio.valueChanged.connect(self._on_crop_edit)
        self.overlap.valueChanged.connect(self._on_crop_edit)
        self.overlap_unit.currentIndexChanged.connect(self._on_overlap_unit_changed)

        g = QGridLayout()
        g.addWidget(QLabel(self.tr("프리셋")), 0, 0)
        g.addWidget(self.crop_preset, 0, 1)
        g.addWidget(self.use_roi, 1, 0, 1, 2)
        g.addWidget(self.use_defect, 2, 0, 1, 2)
        g.addWidget(self.use_background, 3, 0, 1, 2)
        g.addWidget(QLabel(self.tr("  crop 크기")), 4, 0)
        g.addWidget(self.crop_size, 4, 1)
        g.addWidget(QLabel(self.tr("  라벨당 개수")), 5, 0)
        g.addWidget(self.per_defect, 5, 1)
        g.addWidget(QLabel(self.tr("  위치 흔들기")), 6, 0)
        g.addWidget(self.jitter, 6, 1)
        g.addWidget(QLabel(self.tr("  배경 비율")), 7, 0)
        g.addWidget(self.bg_ratio, 7, 1)
        g.addWidget(self.use_tiles, 8, 0, 1, 2)
        g.addWidget(QLabel(self.tr("  타일 크기")), 9, 0)
        g.addWidget(self.tile_size, 9, 1)
        g.addWidget(QLabel(self.tr("  겹침 (overlap)")), 10, 0)
        ov = QHBoxLayout()
        ov.addWidget(self.overlap, 1)
        ov.addWidget(self.overlap_unit)
        g.addLayout(ov, 10, 1)
        g.addWidget(QLabel(self.tr("crop 증강 개수")), 11, 0)
        g.addWidget(self.crop_multiplier, 11, 1)
        note = QLabel(self.tr("Train/Val은 원본 이미지 단위로 먼저 나눈 뒤 자릅니다. "
                              "경계에 걸린 라벨은 잘리고, 보이는 비율이 위 '라벨 유지' 기준 미만이면 제외됩니다."))
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        g.addWidget(note, 12, 0, 1, 2)
        box = QGroupBox(self.tr("Crop 데이터셋"))
        box.setLayout(g)
        return box

    # ---- project / config ----------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.image_path = None
        has = project is not None
        for w in self.findChildren(QWidget):
            if w is not self:
                w.setEnabled(has)
        self.btn_cancel.setEnabled(False)
        self._reload_presets()
        if project is not None:
            keys = set(available_transform_keys(project.channels))
            for key, row in self.rows.items():
                row.available = key in keys
                for w in (row.check, row.strength, row.prob):
                    w.setEnabled(row.available)
                row.check.setToolTip("" if row.available else self.tr("그레이스케일 프로젝트에서는 사용하지 않습니다"))
            self.load_config(project.config.augment)
            self.load_crop_config(project.config.crop)
        self._clear_previews(self.tr("프로젝트를 열면 미리보기가 표시됩니다") if not has else "")

    def load_config(self, cfg: AugmentConfig) -> None:
        self._loading = True
        for key, row in self.rows.items():
            row.load(cfg.setting(key))
        self.multiplier.setValue(cfg.multiplier)
        self.min_visibility.setValue(round(cfg.min_visibility * 100))
        self.min_area.setValue(round(cfg.min_area_px))
        self.cp_enabled.setChecked(cfg.copy_paste.enabled)
        self.cp_prob.setValue(cfg.copy_paste.p)
        self.cp_max.setValue(cfg.copy_paste.max_patches)
        dt = cfg.defect_transform
        self.dt_box.setChecked(dt.enabled)
        self.dt_mode.setCurrentIndex(self.dt_mode.findData(dt.mode))
        self.dt_prob.setValue(dt.p)
        self.dt_max.setValue(dt.max_defects)
        self.dt_scale_min.setValue(round(dt.scale_min * 100))
        self.dt_scale_max.setValue(round(dt.scale_max * 100))
        self.dt_angle.setValue(round(dt.max_angle))
        self.dt_direction.setCurrentIndex(self.dt_direction.findData(dt.direction))
        self.dt_shift.setValue(round(dt.max_shift * 100))
        self.dt_on_ok.setChecked(dt.on_ok_images)
        self._loading = False
        self._sync_preset_combo()

    def current_config(self) -> AugmentConfig:
        return AugmentConfig(
            transforms={k: row.value() for k, row in self.rows.items()},  # unchecked ones keep p/strength
            multiplier=self.multiplier.value(),
            min_visibility=self.min_visibility.value() / 100,
            min_area_px=float(self.min_area.value()),
            copy_paste={"enabled": self.cp_enabled.isChecked(), "p": self.cp_prob.value(),
                        "max_patches": self.cp_max.value()},
            defect_transform=self._defect_transform_value(),
        )

    # ---- defect transform ----------------------------------------------------------------------
    def _build_defect_transform_ui(self) -> QGroupBox:
        box = QGroupBox(self.tr("결함 변형 (결함 크기·각도·위치 랜덤 변경)"))
        box.setCheckable(True)
        box.setChecked(False)
        box.setToolTip(self.tr(
            "라벨된 결함을 잘라 크기와 각도를 바꾸고, 정한 방향으로 옮겨 붙입니다 (Train 증강본에만). "
            "Polygon 라벨이면 결함 모양대로, BBox 라벨이면 상자째 옮깁니다."))
        self.dt_box = box
        self.dt_mode = QComboBox()
        self.dt_mode.addItem(self.tr("추가 (원래 결함도 남김)"), "add")
        self.dt_mode.addItem(self.tr("이동 (원래 자리는 주변으로 채움)"), "move")
        self.dt_prob = QDoubleSpinBox()
        self.dt_prob.setRange(0.0, 1.0)
        self.dt_prob.setSingleStep(0.1)
        self.dt_max = QSpinBox()
        self.dt_max.setRange(1, 10)
        self.dt_scale_min, self.dt_scale_max = QSpinBox(), QSpinBox()
        for w in (self.dt_scale_min, self.dt_scale_max):
            w.setRange(20, 300)
            w.setSingleStep(5)
            w.setSuffix(" %")
        self.dt_angle = QSpinBox()
        self.dt_angle.setRange(0, 180)
        self.dt_angle.setPrefix("± ")
        self.dt_angle.setSuffix("°")
        self.dt_direction = QComboBox()
        self.dt_direction.addItem(self.tr("수평 + 수직"), "both")
        self.dt_direction.addItem(self.tr("수평만"), "horizontal")
        self.dt_direction.addItem(self.tr("수직만"), "vertical")
        self.dt_shift = QSpinBox()
        self.dt_shift.setRange(0, 100)
        self.dt_shift.setSuffix(self.tr(" % (이미지 크기 대비)"))
        self.dt_on_ok = QCheckBox(self.tr("OK 이미지에도 붙이기"))
        self.dt_prob.valueChanged.connect(self._on_edit)
        for w in (self.dt_max, self.dt_scale_min, self.dt_scale_max, self.dt_angle, self.dt_shift):
            w.valueChanged.connect(self._on_edit)
        for w in (self.dt_mode, self.dt_direction):
            w.currentIndexChanged.connect(self._on_edit)
        box.toggled.connect(self._on_edit)
        self.dt_on_ok.toggled.connect(self._on_edit)
        self.dt_scale_min.valueChanged.connect(lambda v: self.dt_scale_max.setMinimum(v))
        self.dt_scale_max.valueChanged.connect(lambda v: self.dt_scale_min.setMaximum(v))

        def pair(a: QWidget, b: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.addWidget(a)
            row.addWidget(QLabel("~"))
            row.addWidget(b)
            return row

        grid = QGridLayout(box)
        grid.addWidget(QLabel(self.tr("원래 결함")), 0, 0)
        grid.addWidget(self.dt_mode, 0, 1)
        grid.addWidget(QLabel(self.tr("확률 / 최대 개수")), 1, 0)
        prob_row = QHBoxLayout()
        prob_row.addWidget(self.dt_prob)
        prob_row.addWidget(self.dt_max)
        grid.addLayout(prob_row, 1, 1)
        grid.addWidget(QLabel(self.tr("크기")), 2, 0)
        grid.addLayout(pair(self.dt_scale_min, self.dt_scale_max), 2, 1)
        grid.addWidget(QLabel(self.tr("회전 각도")), 3, 0)
        grid.addWidget(self.dt_angle, 3, 1)
        grid.addWidget(QLabel(self.tr("이동 방향")), 4, 0)
        grid.addWidget(self.dt_direction, 4, 1)
        grid.addWidget(QLabel(self.tr("최대 이동")), 5, 0)
        grid.addWidget(self.dt_shift, 5, 1)
        grid.addWidget(self.dt_on_ok, 6, 0, 1, 2)
        return box

    def _defect_transform_value(self) -> dict:
        return {"enabled": self.dt_box.isChecked(), "mode": self.dt_mode.currentData(),
                "p": self.dt_prob.value(), "max_defects": self.dt_max.value(),
                "scale_min": self.dt_scale_min.value() / 100, "scale_max": self.dt_scale_max.value() / 100,
                "max_angle": float(self.dt_angle.value()), "direction": self.dt_direction.currentData(),
                "max_shift": self.dt_shift.value() / 100, "on_ok_images": self.dt_on_ok.isChecked()}

    def _on_edit(self) -> None:
        if self._loading or self.project is None:
            return
        self.project.config.augment = self.current_config()
        self._sync_preset_combo()
        self.changed.emit()
        self.schedule_preview()

    # ---- crop settings (F7.2-F7.4) ---------------------------------------------------------------
    def load_crop_config(self, cfg: CropConfig) -> None:
        self._loading = True
        self.use_roi.setChecked(cfg.use_roi)
        self.use_defect.setChecked(cfg.use_defect)
        self.use_background.setChecked(cfg.use_background)
        self.use_tiles.setChecked(cfg.use_tiles)
        self.crop_size.setValue(cfg.crop_size)
        self.per_defect.setValue(cfg.per_defect)
        self.jitter.setValue(round(cfg.jitter * 100))
        self.bg_ratio.setValue(cfg.background_ratio)
        self.tile_size.setValue(cfg.tile_size)
        self.overlap_unit.setCurrentIndex(self.overlap_unit.findData(cfg.overlap_unit))
        self._set_overlap_range(cfg.overlap_unit)
        self.overlap.setValue(cfg.overlap_px if cfg.overlap_unit == "px" else cfg.overlap_percent)
        self.crop_multiplier.setValue(cfg.multiplier)
        self._loading = False
        self._update_crop_enabled()
        self._sync_crop_preset()

    def current_crop_config(self) -> CropConfig:
        base = self.project.config.crop if self.project is not None else CropConfig()
        unit = self.overlap_unit.currentData()
        # the value of the unit not shown is kept, so switching % <-> px back and forth loses nothing
        overlap = {"overlap_px": int(self.overlap.value())} if unit == "px" else {"overlap_percent": self.overlap.value()}
        return base.model_copy(update={
            "use_roi": self.use_roi.isChecked(), "use_defect": self.use_defect.isChecked(),
            "use_background": self.use_background.isChecked(), "use_tiles": self.use_tiles.isChecked(),
            "crop_size": self.crop_size.value(), "per_defect": self.per_defect.value(),
            "jitter": self.jitter.value() / 100, "background_ratio": self.bg_ratio.value(),
            "tile_size": self.tile_size.value(), "overlap_unit": unit, "multiplier": self.crop_multiplier.value(),
            **overlap,
        })

    def _set_overlap_range(self, unit: str) -> None:
        if unit == "px":
            self.overlap.setRange(0, 8191)
            self.overlap.setSingleStep(8)
        else:
            self.overlap.setRange(0, 90)
            self.overlap.setSingleStep(5)

    def _on_overlap_unit_changed(self) -> None:
        if self._loading or self.project is None:
            return
        unit = self.overlap_unit.currentData()
        cfg = self.project.config.crop
        self._loading = True
        self._set_overlap_range(unit)
        self.overlap.setValue(cfg.overlap_px if unit == "px" else cfg.overlap_percent)
        self._loading = False
        self._on_crop_edit()

    def _on_crop_edit(self) -> None:
        self._update_crop_enabled()
        if self._loading or self.project is None:
            return
        self.project.config.crop = CropConfig.model_validate(self.current_crop_config().model_dump())
        self._sync_crop_preset()
        self.changed.emit()

    def _update_crop_enabled(self) -> None:
        if self.project is None:
            return
        windows = self.use_defect.isChecked() or self.use_background.isChecked()
        self.crop_size.setEnabled(windows)
        self.per_defect.setEnabled(self.use_defect.isChecked())
        self.jitter.setEnabled(self.use_defect.isChecked())
        self.bg_ratio.setEnabled(self.use_background.isChecked())
        for w in (self.tile_size, self.overlap, self.overlap_unit):
            w.setEnabled(self.use_tiles.isChecked())

    def _sync_crop_preset(self) -> None:
        if self.project is None:
            return
        current = self.project.config.crop.model_dump()
        name = next((n for n, c in CROP_PRESETS.items() if c.model_dump() == current), self.tr(CUSTOM))
        self.crop_preset.blockSignals(True)
        self.crop_preset.setCurrentText(name)
        self.crop_preset.blockSignals(False)

    def _on_crop_preset_chosen(self, index: int) -> None:
        name = self.crop_preset.itemText(index)
        if self.project is not None and name in CROP_PRESETS:
            self.apply_crop_preset(name)

    def apply_crop_preset(self, name: str) -> None:
        self.project.config.crop = CROP_PRESETS[name].model_copy(deep=True)
        self.load_crop_config(self.project.config.crop)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData("crop"))
        self.changed.emit()

    # ---- presets (F3.4) --------------------------------------------------------------------
    def _reload_presets(self) -> None:
        self.presets = load_presets(self.presets_dir)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(list(self.presets) + [self.tr(CUSTOM)])
        self.preset_combo.blockSignals(False)

    def _sync_preset_combo(self) -> None:
        """Show the preset matching the current settings, or 직접 설정."""
        if self.project is None:
            return
        current = _comparable(self.project.config.augment)
        name = next((n for n, c in self.presets.items() if _comparable(c) == current), self.tr(CUSTOM))
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(name)
        self.preset_combo.blockSignals(False)

    def _on_preset_chosen(self, index: int) -> None:
        name = self.preset_combo.itemText(index)
        if self.project is None or name not in self.presets:
            return
        self.apply_preset(name)

    def apply_preset(self, name: str) -> None:
        self.project.config.augment = self.presets[name].model_copy(deep=True)
        self.load_config(self.project.config.augment)
        self.changed.emit()
        self.schedule_preview()

    def save_preset_dialog(self) -> None:
        if self.project is None:
            return
        name, ok = QInputDialog.getText(self, self.tr("프리셋 저장"), self.tr("프리셋 이름:"))
        if ok and name.strip():
            try:
                self.save_current_as(name)
            except ValueError as e:
                QMessageBox.warning(self, self.tr("프리셋 저장"), str(e))

    def save_current_as(self, name: str) -> Path:
        path = save_preset(self.presets_dir, name, self.current_config())
        self._reload_presets()
        self._sync_preset_combo()
        return path

    # ---- preview (F3.5) ------------------------------------------------------------------
    def set_current_image(self, path: str | None) -> None:
        self.image_path = path
        self.schedule_preview()

    def schedule_preview(self) -> None:
        if self.project is not None and self.isVisible():
            self._preview_timer.start()

    def refresh_preview(self) -> None:
        """Compute 9 samples on a worker thread; only the newest request is shown."""
        if self.project is None or self.image_path is None:
            self._clear_previews(self.tr("이미지를 선택하세요"))
            return
        self._preview_gen += 1
        gen = self._preview_gen
        project, path = self.project, self.image_path
        config = project.config.augment.model_copy(deep=True)
        annotations = self.annotations_provider(path)
        colors = [c.color for c in project.config.classes]
        seed = random.randrange(1 << 30)
        self.preview_status.setText(self.tr("미리보기 계산 중…"))

        def work() -> None:
            try:
                from defect_inspection.core.preview import augment_previews

                raw = imread_unicode(project.image_abspath(path))
                if raw is None:
                    raise OSError(f"cannot read {path}")
                image = to_training_8bit(raw, project.channels)
                ann = annotations or project.load_labels(path, size=(image.shape[1], image.shape[0]))
                tiles = augment_previews(image, ann, config, project.channels, colors, PREVIEW_COUNT, seed)
                self._emit(self._previews_ready, gen, tiles)
            except Exception as e:  # noqa: BLE001 - shown in the panel
                self._emit(self._preview_failed, gen, str(e))

        threading.Thread(target=work, daemon=True, name="augment-preview").start()

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # panel destroyed meanwhile
            pass

    def _show_previews(self, gen: int, tiles: list) -> None:
        if gen != self._preview_gen:
            return
        for lbl, tile in zip(self.preview_labels, tiles):
            pix = QPixmap.fromImage(numpy_to_qimage(tile))
            lbl.setPixmap(pix.scaled(PREVIEW_TILE, PREVIEW_TILE, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
        self.preview_status.setText(self.tr("증강 {0}종 적용, 매번 무작위").format(
            len(self.project.config.augment.enabled_keys()) if self.project else 0))

    def _show_preview_error(self, gen: int, message: str) -> None:
        if gen == self._preview_gen:
            self._clear_previews(self.tr("미리보기 실패: {0}").format(message))

    def _clear_previews(self, text: str) -> None:
        for lbl in self.preview_labels:
            lbl.clear()
        self.preview_status.setText(text)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.schedule_preview()

    @property
    def preview_pixmaps(self) -> list[QPixmap]:
        return [lbl.pixmap() for lbl in self.preview_labels]

    # ---- dataset build (F3.2) -------------------------------------------------------------
    @property
    def is_building(self) -> bool:
        # A flag, not "thread exists": it must already be False when build_finished/build_stopped
        # are emitted (the thread is cleaned up a moment later), or a follow-up action such as the
        # quick-suggest training would be refused as "build still running".
        return self._building

    @property
    def build_mode(self) -> str:
        return self.mode_combo.currentData()

    def start_build(self, mode: str | None = None) -> None:
        """Build dataset/full or dataset/crop (default: the mode chosen next to the button)."""
        if self.project is None or self._building:
            return
        mode = mode or self.build_mode
        if self._thread is not None:
            # e.g. from a build_finished slot (Full -> Crop experiment): the finished thread is
            # cleaned up a moment later; start then.
            self._pending_build = mode
            return
        self.before_build()  # save labels + project.yaml so the build sees the latest edits
        self._worker = _BuildWorker(self.project, mode)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_build_done)
        self._worker.failed.connect(self._on_build_failed)
        self._worker.cancelled.connect(self._on_build_cancelled)
        for sig in (self._worker.finished, self._worker.failed, self._worker.cancelled):
            # Direct: QThread.quit is thread-safe. Queued, it would need the UI event loop, which
            # is blocked while closeEvent waits for the build -> deadlock.
            sig.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._cleanup_build)
        self.btn_build.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.build_status.setText(self.tr("{0} 데이터셋 빌드 중…").format("Crop" if mode == "crop" else "Full"))
        self._building = True
        self._thread.start()

    def cancel_build(self) -> None:
        if self._worker is not None:
            self._worker.stop.set()
            self.build_status.setText(self.tr("취소하는 중…"))

    def wait_build(self, timeout_ms: int = 60000) -> bool:
        thread = self._thread
        return thread is None or thread.wait(timeout_ms)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.progress.setFormat("%p%")  # the bar is narrow; the message goes to the status line
        self.build_status.setText(message)

    def _on_build_done(self, report: BuildReport | CropBuildReport) -> None:
        self._building = False
        lines = self._crop_report_lines(report) if isinstance(report, CropBuildReport) else self._full_report_lines(report)
        self.build_status.setText("\n".join(lines))
        self.progress.setValue(self.progress.maximum())
        self.build_finished.emit(report)

    def _crop_report_lines(self, r: CropBuildReport) -> list[str]:
        names = {"roi": self.tr("ROI"), "defect": self.tr("결함 중심"), "background": self.tr("배경"),
                 "tile": self.tr("타일")}
        kinds = ", ".join(f"{names[k]} {n}" for k, n in r.by_kind.items() if n)
        lines = [self.tr("Crop 완료 ({0:.1f}초): Train {1}장 (crop {2} + 증강 {3}), Val {4}장 (결함 포함 {5})").format(
                     r.elapsed_s, r.total_train, r.train_crops, r.augmented, r.val_crops, r.val_defect_crops),
                 self.tr("원본: Train {0} / Val {1} · 종류: {2}").format(r.train_images, r.val_images, kinds)]
        if r.defect_transformed:
            lines.append(self.tr("결함 변형: {0}장").format(r.defect_transformed))
        if r.dropped_labels:
            lines.append(self.tr("경계·증강으로 제외된 라벨: {0}개").format(r.dropped_labels))
        if r.no_windows:
            lines.append(self.tr("crop 없음: {0}").format(", ".join(r.no_windows)))
        if r.skipped_unlabeled:
            lines.append(self.tr("제외(미라벨·제안만): {0}").format(", ".join(r.skipped_unlabeled)))
        if r.skipped_unreadable:
            lines.append(self.tr("제외(읽기 실패·크기 불일치): {0}").format(", ".join(r.skipped_unreadable)))
        return lines + [f"⚠ {w}" for w in r.warnings]

    def _full_report_lines(self, report: BuildReport) -> list[str]:
        lines = [self.tr("완료 ({0:.1f}초): Train {1}장 (원본 {2} + 증강 {3}), Val {4}장").format(
            report.elapsed_s, report.total_train, report.train_images, report.augmented, report.val_images)]
        if report.ok_train or report.ok_val:
            lines.append(self.tr("OK 이미지: Train {0} / Val {1}").format(report.ok_train, report.ok_val))
        if report.copy_pasted:
            lines.append(self.tr("Copy-Paste 합성: {0}장").format(report.copy_pasted))
        if report.defect_transformed:
            lines.append(self.tr("결함 변형: {0}장").format(report.defect_transformed))
        if report.dropped_labels:
            lines.append(self.tr("증강 중 제거된 라벨: {0}개").format(report.dropped_labels))
        if report.skipped_unlabeled:
            lines.append(self.tr("제외(미라벨·제안만): {0}").format(", ".join(report.skipped_unlabeled)))
        if report.skipped_unreadable:
            lines.append(self.tr("제외(읽기 실패·크기 불일치): {0}").format(", ".join(report.skipped_unreadable)))
        return lines + [f"⚠ {w}" for w in report.warnings]

    def _on_build_failed(self, message: str) -> None:
        self._building = False
        self.build_status.setText(self.tr("빌드 실패: {0}").format(message))
        self.build_stopped.emit(message)
        QMessageBox.warning(self, self.tr("데이터셋 빌드"), message)

    def _on_build_cancelled(self) -> None:
        self._building = False
        self.build_status.setText(self.tr("취소했습니다. 이전 데이터셋은 그대로입니다."))
        self.build_stopped.emit("cancelled")
        self.progress.setValue(0)

    def _cleanup_build(self) -> None:
        self._worker.deleteLater()
        self._thread.deleteLater()
        self._worker = self._thread = None
        self.btn_build.setEnabled(self.project is not None)
        self.btn_cancel.setEnabled(False)
        pending, self._pending_build = self._pending_build, None
        if pending is not None:
            self.start_build(pending)


def _comparable(cfg: AugmentConfig) -> dict:
    """Settings that define a preset (only enabled transforms count)."""
    d = cfg.model_dump(mode="json")
    d["transforms"] = {k: v for k, v in d["transforms"].items() if v["enabled"]}
    if not d["defect_transform"]["enabled"]:  # its other values do not matter while it is off
        d["defect_transform"] = {"enabled": False}
    return d
