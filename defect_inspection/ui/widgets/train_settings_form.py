"""Train-tab settings form (PRD F4.1-F4.3): model, dataset and hyperparameters → `TrainConfig`."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from defect_inspection.core.model_zoo import DEFAULT_MODEL, MODEL_CHOICES, PAPER_HYP, get_choice, is_unet
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import TrainConfig
from defect_inspection.platform_utils import default_dataloader_workers, gpu_device_choice

WARNING_STYLE = "color: #8a5300;"


class TrainSettingsForm(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle(self.tr("학습 설정"))
        self.project: Project | None = None
        self.gpu_available = False
        self._build_ui()
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(self.model_combo.findData(DEFAULT_MODEL))
        self.model_combo.blockSignals(False)
        self._on_model_changed()

    def _build_ui(self) -> None:
        self.model_combo = QComboBox()
        for c in MODEL_CHOICES:
            self.model_combo.addItem(c.label, c.key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet(WARNING_STYLE)
        self.task_label = QLabel()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Full (전체 이미지)", "full")
        self.dataset_combo.addItem("Crop (잘라낸 영역)", "crop")

        self._build_hyperparameters()

        self.online_aug = QCheckBox(self.tr("Ultralytics 내장 증강 (mosaic 등)"))
        self.online_aug.setChecked(True)
        self.online_aug.toggled.connect(self._update_online_warning)
        self.online_warning = QLabel()
        self.online_warning.setWordWrap(True)
        self.online_warning.setStyleSheet(WARNING_STYLE)
        self.paper_hyp = QCheckBox(self.tr("논문 하이퍼파라미터 (ATT-YOLO Table 6)"))
        self.paper_hyp.setToolTip(", ".join(f"{k}={v}" for k, v in PAPER_HYP.items()))
        self.pretrained = QCheckBox(self.tr("사전학습 가중치 사용"))
        self.pretrained.setChecked(True)

        form = QFormLayout(self)
        form.addRow(self.tr("모델"), self.model_combo)
        form.addRow("", self.model_note)
        form.addRow(self.tr("태스크"), self.task_label)
        form.addRow(self.tr("데이터셋"), self.dataset_combo)
        form.addRow(self._hyperparameter_grid())
        form.addRow(self.online_aug)
        form.addRow(self.online_warning)
        form.addRow(self.paper_hyp)
        form.addRow(self.pretrained)

    def _build_hyperparameters(self) -> None:
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(64, 4096)
        self.imgsz.setSingleStep(32)
        self.batch = QSpinBox()
        self.batch.setRange(-1, 1024)
        self.batch.setSpecialValueText(self.tr("자동"))
        self.batch.setToolTip(self.tr(
            "자동(또는 0): CUDA GPU는 GPU 메모리를 재서 정하고, Apple GPU·CPU는 컴퓨터 메모리와 "
            "모델·imgsz로 추정합니다 (8 GB Mac, 640px: yolo11n 8, yolo11s 4). "
            "학습이 매우 느리거나 스왑이 많으면 직접 낮추세요."))
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

    def _hyperparameter_grid(self) -> QHBoxLayout:
        left, right = QFormLayout(), QFormLayout()
        left.addRow("epochs", self.epochs)
        left.addRow("imgsz", self.imgsz)
        left.addRow("batch", self.batch)
        left.addRow("lr0", self.lr0)
        right.addRow("patience", self.patience)
        right.addRow(self.tr("장치"), self.device)
        right.addRow("workers", self.workers)
        grid = QHBoxLayout()
        grid.addLayout(left)
        grid.addLayout(right)
        return grid

    # ---- project ---------------------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self._update_task_label()
        self.refresh_datasets()
        self._update_online_warning()

    def _is_unet(self) -> bool:
        return is_unet(self.model_combo.currentData())

    def _update_task_label(self) -> None:
        project = self.project
        if project is None:
            self.task_label.setText("-")
            return
        mode = self.tr("그레이스케일 → 1채널 입력 모델") if project.channels == 1 else self.tr("컬러 → 3채널 입력 모델")
        if is_unet(self.model_combo.currentData()):
            task = self.tr("시맨틱 분할 (픽셀 마스크)")
        else:
            task = "detect (BBox)" if project.config.task == "detect" else "seg (Polygon)"
        self.task_label.setText(f"{task} · {mode}")

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

    def dataset_available(self) -> bool:
        return self.project is not None and self.dataset_combo.model().item(self.dataset_combo.currentIndex()).isEnabled()

    # ---- form -----------------------------------------------------------------------------
    def _on_model_changed(self) -> None:
        choice = get_choice(self.model_combo.currentData())
        self.epochs.setValue(choice.default_epochs)
        self.imgsz.setValue(choice.default_imgsz)
        self.model_note.setText(self.tr(choice.note))
        self.model_note.setVisible(bool(choice.note))
        self.paper_hyp.setChecked(False)
        unet = is_unet(choice.key)  # its own training loop: no Ultralytics augmentation or hyperparameters
        self.online_aug.setEnabled(not unet)
        self.paper_hyp.setEnabled(not unet)
        self._update_task_label()
        self._update_online_warning()

    def _update_online_warning(self) -> None:
        offline = [] if self.project is None else self.project.config.augment.enabled_keys()
        self.online_warning.setText(self.tr(
            "오프라인 증강({0}종)이 이미 데이터셋에 들어 있어, 내장 증강까지 켜면 뒤집기·밝기 같은 변환이 "
            "두 번 적용됩니다. 결과가 나빠지면 끄고 비교해 보세요.").format(len(offline)))
        self.online_warning.setVisible(self.online_aug.isChecked() and not self._is_unet() and bool(offline))

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
            online_augment=self.online_aug.isChecked() and not self._is_unet(),
            paper_hyp=self.paper_hyp.isChecked(),
            pretrained=self.pretrained.isChecked(),
            seed=0 if self.project is None else self.project.config.split.seed,
        )

    def select_model(self, key: str) -> None:
        self.model_combo.setCurrentIndex(self.model_combo.findData(key))
