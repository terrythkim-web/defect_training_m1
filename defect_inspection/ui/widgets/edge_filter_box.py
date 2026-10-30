"""Result-tab edge filter settings (PRD F9): on/off, line kind, strip length, touch distance
→ `EdgeFilterConfig`. Settings without a widget (contrast, tilt, flatness) are kept as they are."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QSpinBox, QWidget

from defect_inspection.core.edge_filter_config import EdgeFilterConfig

POLARITIES = [("both", "어두운 선 + 밝은 선"), ("dark", "어두운 선만"), ("bright", "밝은 선만")]


class EdgeFilterBox(QGroupBox):
    """Checkable group: checked = filter on. `changed` fires on every user edit, never on `set_config`."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle(self.tr("테두리 필터 — 물체 최외각 선 위의 예측만 남김"))
        self.setCheckable(True)
        self.setToolTip(self.tr(
            "예측 박스마다 박스 아래(또는 허용 거리 안)에 물체의 가장 바깥 선이 있는지 봅니다.\n"
            "선 너머부터 이미지 끝까지 다른 선이 없고 평평해야 최외각입니다.\n"
            "Result·제안·비교에만 적용되고 내보낸 ONNX 모델에는 적용되지 않습니다."))
        self._base = EdgeFilterConfig()

        self.polarity = QComboBox()
        for key, label in POLARITIES:
            self.polarity.addItem(self.tr(label), key)
        self.polarity.setToolTip(self.tr("밝은 배경 위 어두운 선, 어두운 배경(디스플레이 등) 위 밝은 선"))
        self.along = QSpinBox()
        self.along.setRange(40, 4096)
        self.along.setSuffix(" px")
        self.along.setToolTip(self.tr("박스 양옆으로 이만큼의 선을 보고 테두리인지 판단합니다"))
        self.touch = QSpinBox()
        self.touch.setRange(0, 256)
        self.touch.setSuffix(" px")
        self.touch.setToolTip(self.tr("박스가 선에서 이만큼 떨어져 있어도 테두리 위로 봅니다 (안쪽으로 뻗은 크랙 등)"))

        form = QFormLayout(self)
        form.addRow(self.tr("선 종류"), self.polarity)
        form.addRow(self.tr("박스 양옆 길이"), self.along)
        form.addRow(self.tr("허용 거리"), self.touch)

        self.toggled.connect(self.changed)
        self.polarity.currentIndexChanged.connect(self.changed)
        self.along.valueChanged.connect(self.changed)
        self.touch.valueChanged.connect(self.changed)

    def set_config(self, cfg: EdgeFilterConfig) -> None:
        self._base = cfg.model_copy()
        widgets = (self, self.polarity, self.along, self.touch)
        for w in widgets:
            w.blockSignals(True)
        self.setChecked(cfg.enabled)
        self.polarity.setCurrentIndex(max(0, self.polarity.findData(cfg.line_polarity)))
        self.along.setValue(cfg.along_px)
        self.touch.setValue(cfg.touch_px)
        for w in widgets:
            w.blockSignals(False)

    def config(self) -> EdgeFilterConfig:
        return self._base.model_copy(update={
            "enabled": self.isChecked(), "line_polarity": self.polarity.currentData(),
            "along_px": self.along.value(), "touch_px": self.touch.value()})
