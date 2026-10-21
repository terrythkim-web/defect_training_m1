"""New project dialog (PRD F1.1, F1.6): folder, name, task (detect / seg) and colour mode.

The task decides the label shape written to YOLO txt (BBox or Polygon) and the colour mode the
number of input channels of every later stage, so both are chosen up front.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from defect_inspection.core.project import PROJECT_FILE

TASKS = [("detect", "detect — BBox (사각형 검출)"), ("seg", "seg — Polygon (영역 분할)")]
COLOR_MODES = [("color", "컬러 (3채널)"), ("grayscale", "그레이 (1채널, 흑백 카메라)")]


class NewProjectDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("새 프로젝트"))
        self.folder = QLineEdit()
        self.folder.setPlaceholderText(self.tr("프로젝트를 만들 폴더 (비어 있거나 새 폴더)"))
        self.folder.setMinimumWidth(340)  # long paths (Korean folder names) stay readable
        self.folder.textChanged.connect(self._on_folder_changed)
        browse = QPushButton(self.tr("찾아보기…"))
        browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        self.name = QLineEdit()
        self.name.setPlaceholderText(self.tr("비우면 폴더 이름"))
        self.task = QComboBox()
        for key, label in TASKS:
            self.task.addItem(self.tr(label), key)
        self.color_mode = QComboBox()
        for key, label in COLOR_MODES:
            self.color_mode.addItem(self.tr(label), key)
        self.color_mode.currentIndexChanged.connect(self._validate)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray;")

        form = QFormLayout()
        form.addRow(self.tr("폴더"), folder_row)
        form.addRow(self.tr("이름"), self.name)
        form.addRow(self.tr("작업(task)"), self.task)
        form.addRow(self.tr("색 모드"), self.color_mode)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.note)
        layout.addWidget(self.buttons)
        self.resize(max(600, self.sizeHint().width()), self.sizeHint().height())
        self._validate()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("새 프로젝트 폴더 선택"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _on_folder_changed(self, text: str) -> None:
        self.name.setPlaceholderText(Path(text).name if text.strip() else self.tr("비우면 폴더 이름"))
        self._validate()

    def problem(self) -> str | None:
        text = self.folder.text().strip()
        if not text:
            return self.tr("폴더를 고르세요")
        if (Path(text) / PROJECT_FILE).exists():
            return self.tr("이 폴더에는 이미 프로젝트가 있습니다. '프로젝트 열기'를 쓰세요")
        if Path(text).exists() and not Path(text).is_dir():
            return self.tr("폴더가 아니라 파일입니다")
        return None

    def _validate(self) -> None:
        problem = self.problem()
        notes = [problem] if problem else []
        if self.color_mode.currentData() == "grayscale":
            notes.append(self.tr("그레이: 이미지를 1채널로 저장하고 1채널 입력 모델을 학습합니다. CoreML 내보내기는 컬러만 됩니다."))
        self.note.setText("\n".join(notes))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(problem is None)

    def values(self) -> dict:
        """Keyword arguments for Project.create."""
        return {"root": Path(self.folder.text().strip()), "name": self.name.text().strip() or None,
                "task": self.task.currentData(), "color_mode": self.color_mode.currentData()}
