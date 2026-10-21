"""Models the user can train (PRD F4.1, F4.8). Trainer and Train tab create models only through
make_model(), so every model family gets the same weights location and task handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from defect_inspection.core.att_yolo import TransferReport, build_att_yolo
from defect_inspection.core.weights import resolve_weights

TaskMode = Literal["detect", "seg"]
Family = Literal["yolo11", "att-yolo"]


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    family: Family
    scale: str
    default_imgsz: int
    default_epochs: int
    note: str = ""


ATT_YOLO_NOTE = (
    "P2~P7 6개 헤드 + self-attention(ATT-YOLO 논문 구조). 사전학습 가중치가 없어 백본만 YOLOv5에서 "
    "가져오므로, 이미지가 적으면 yolo11보다 낮을 수 있습니다. Compare 탭에서 yolo11과 비교해 보세요."
)

MODEL_CHOICES: list[ModelChoice] = [
    ModelChoice("yolo11n", "YOLO11 n (가장 빠름)", "yolo11", "n", 640, 100),
    ModelChoice("yolo11s", "YOLO11 s (기본)", "yolo11", "s", 640, 100),
    ModelChoice("yolo11m", "YOLO11 m (정확도 우선)", "yolo11", "m", 640, 100),
    ModelChoice("att-yolo-n", "ATT-YOLO n (작은 결함, 고해상도)", "att-yolo", "n", 1280, 300, ATT_YOLO_NOTE),
    ModelChoice("att-yolo-s", "ATT-YOLO s (작은 결함, 고해상도)", "att-yolo", "s", 1280, 300, ATT_YOLO_NOTE),
]
DEFAULT_MODEL = "yolo11s"

# ATT-YOLO paper, Table 6. Offered as an optional preset; Ultralytics defaults otherwise.
PAPER_HYP = {
    "lr0": 0.0032,
    "lrf": 0.12,
    "momentum": 0.843,
    "weight_decay": 0.00036,
    "mosaic": 1.0,
    "mixup": 0.243,
}


def get_choice(key: str) -> ModelChoice:
    for choice in MODEL_CHOICES:
        if choice.key == key:
            return choice
    raise KeyError(f"unknown model {key!r}; choose from {[c.key for c in MODEL_CHOICES]}")


def yolo11_weights_name(scale: str, task: TaskMode) -> str:
    return f"yolo11{scale}{'-seg' if task == 'seg' else ''}.pt"


def make_model(key: str, task: TaskMode, nc: int, channels: int, run_dir: Path,
               weights_dir: Path, pretrained: bool = True):
    """Create the Ultralytics model for a choice. Returns (YOLO, TransferReport | None).

    yolo11: the official pretrained checkpoint (downloaded once into weights_dir). Ultralytics
    adapts the head to nc and, for grayscale data (channels=1), the first conv at train time.
    att-yolo: see core/att_yolo.build_att_yolo (YOLOv5 backbone transfer, rest fresh).
    """
    from ultralytics import YOLO

    choice = get_choice(key)
    yolo_task = "segment" if task == "seg" else "detect"
    if choice.family == "att-yolo":
        return build_att_yolo(task, nc, channels, choice.scale, Path(run_dir), weights_dir, pretrained)
    if pretrained:
        weights = resolve_weights(yolo11_weights_name(choice.scale, task), weights_dir)
        return YOLO(str(weights), task=yolo_task), None
    cfg = f"yolo11{choice.scale}{'-seg' if task == 'seg' else ''}.yaml"
    return YOLO(cfg, task=yolo_task), None


__all__ = ["MODEL_CHOICES", "DEFAULT_MODEL", "PAPER_HYP", "ModelChoice", "TransferReport",
           "get_choice", "make_model", "yolo11_weights_name"]
