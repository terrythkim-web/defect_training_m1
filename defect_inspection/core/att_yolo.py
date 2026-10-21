"""ATT-YOLO, adapted to Ultralytics (PRD F4.8).

Paper: Wang et al., "Toward surface defect detection in electronics manufacturing by an accurate
and lightweight YOLO-style object detector", Scientific Reports 13:7062 (2023).

What is kept from the paper:
- YOLOv5-small basis with a many-scale backbone and feature pyramid, P2..P7 (strides 4..128),
  and one detection head per level (six heads).
- One self-attention module (multi-head attention + MLP) right behind P7, after Conv+SPP+C3.
  Here it is Ultralytics' C3TR, a C3 block whose core is a TransformerBlock (MHA + MLP).

What differs:
- Heads are Ultralytics' anchor-free Detect/Segment, so the paper's GA auto-anchor (which fixed
  a poor best-possible-recall of anchors) is not needed. Segmentation also works.
- The stem is Conv(64, 6, 2, 2) as in YOLOv5 v6/"v5u" (equivalent to Focus) and backbone P2..P5
  match YOLOv5s layer for layer, so YOLOv5su pretrained weights can initialise them. The paper
  trained from scratch on 14k defects; with tens of images that is not an option.

No code from the paper's repository is used; the structure is rebuilt from the paper's text,
Fig. 7 and Table 8.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

TaskMode = Literal["detect", "seg"]
Scale = Literal["n", "s"]

# [depth, width] as in Ultralytics' yolov5.yaml, so P2..P5 shapes match YOLOv5su.
SCALES = {"n": (0.33, 0.25), "s": (0.33, 0.50)}
BACKBONE_TRANSFER_LAYERS = range(0, 9)  # layers 0..8 are identical to YOLOv5 (stem .. P5 C3)
SOURCE_WEIGHTS = {"n": "yolov5nu.pt", "s": "yolov5su.pt"}
HEAD_LEVEL_LAYERS = [34, 37, 40, 43, 46, 49]  # P2..P7 outputs feeding the head

_UP = [None, 2, "nearest"]

_BACKBONE = [
    # [from, repeats, module, args]
    [-1, 1, "Conv", [64, 6, 2, 2]],        # 0  P1/2 stem (Focus-equivalent)
    [-1, 1, "Conv", [128, 3, 2]],          # 1  P2/4
    [-1, 3, "C3", [128]],                  # 2
    [-1, 1, "Conv", [256, 3, 2]],          # 3  P3/8
    [-1, 6, "C3", [256]],                  # 4
    [-1, 1, "Conv", [512, 3, 2]],          # 5  P4/16
    [-1, 9, "C3", [512]],                  # 6
    [-1, 1, "Conv", [1024, 3, 2]],         # 7  P5/32
    [-1, 3, "C3", [1024]],                 # 8
    [-1, 1, "Conv", [768, 3, 2]],          # 9  P6/64   (ATT-YOLO)
    [-1, 3, "C3", [768]],                  # 10
    [-1, 1, "Conv", [768, 3, 2]],          # 11 P7/128  (ATT-YOLO)
    [-1, 1, "SPP", [768, [5, 9, 13]]],     # 12
    [-1, 3, "C3", [768, False]],           # 13
    [-1, 1, "C3TR", [768]],                # 14 self-attention: MHA + MLP
]
# P6/P7 use 768 channels like the official YOLOv5-P6 models; with the s scale this gives
# 22.4M parameters (paper: 23.1M). FLOPs are higher than the paper's 21.8 G@640 (~33.5 G)
# because Ultralytics' decoupled head is heavier, and it runs at stride 4 for P2.

# Top-down: (channels of the next lower level, its backbone layer). Bottom-up: (conv channels,
# matching top-down layer, output channels).
_TOP_DOWN = [(768, 10), (1024, 8), (512, 6), (256, 4), (128, 2)]
_BOTTOM_UP = [(128, 31, 256), (256, 27, 512), (512, 23, 1024), (1024, 19, 768), (768, 15, 768)]


def _neck() -> list:
    layers: list = []
    for ch, skip in _TOP_DOWN:  # layers 15..34, P2 output at 34
        layers += [[-1, 1, "Conv", [ch, 1, 1]], [-1, 1, "nn.Upsample", _UP],
                   [[-1, skip], 1, "Concat", [1]], [-1, 3, "C3", [ch, False]]]
    for ch, lateral, out in _BOTTOM_UP:  # layers 35..49, P3..P7 outputs at 37, 40, 43, 46, 49
        layers += [[-1, 1, "Conv", [ch, 3, 2]], [[-1, lateral], 1, "Concat", [1]],
                   [-1, 3, "C3", [out, False]]]
    return layers


def att_yolo_config(task: TaskMode, nc: int, channels: int = 3, scale: Scale = "s") -> dict:
    """Model definition dict (the same structure as an Ultralytics model YAML)."""
    if scale not in SCALES:
        raise ValueError(f"unknown ATT-YOLO scale {scale!r}; choose from {list(SCALES)}")
    if task not in ("detect", "seg"):
        raise ValueError(f"unknown task {task!r}")
    head_module, head_args = ("Detect", ["nc"]) if task == "detect" else ("Segment", ["nc", 32, 256])
    depth, width = SCALES[scale]
    # Explicit multiples instead of `scales` + `scale`: when loading a YAML file Ultralytics
    # overwrites `scale` with a guess from the file name (yolo<N><scale>), which would silently
    # turn att-yolo-s into n.
    return {
        "nc": nc,
        "channels": channels,
        "depth_multiple": depth,
        "width_multiple": width,
        "backbone": copy.deepcopy(_BACKBONE),
        "head": _neck() + [[HEAD_LEVEL_LAYERS, 1, head_module, head_args]],
    }


def write_att_yolo_yaml(path: str | Path, task: TaskMode, nc: int, channels: int = 3,
                        scale: Scale = "s") -> Path:
    """Save the definition next to a training run (Ultralytics loads models from files, and the
    file documents exactly what was trained)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = att_yolo_config(task, nc, channels, scale)
    header = "# ATT-YOLO (Ultralytics adaptation), see defect_inspection/core/att_yolo.py\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return path


@dataclass
class TransferReport:
    """Which pretrained tensors were copied into the new model."""

    source: str
    transferred: list[str] = field(default_factory=list)
    total: int = 0

    @property
    def layers(self) -> list[int]:
        return sorted({int(k.split(".")[1]) for k in self.transferred})

    def summary(self) -> str:
        return (f"{self.source}: {len(self.transferred)}/{self.total} tensors transferred "
                f"(layers {self.layers[0]}-{self.layers[-1]})" if self.transferred
                else f"{self.source}: nothing transferred")


def transfer_backbone(model, source_state: dict, source_name: str = "pretrained") -> TransferReport:
    """Copy tensors from source_state whose name AND shape match, limited to the YOLOv5-identical
    backbone layers (stem .. P5). P6/P7, attention, neck and head keep their fresh init.

    model is an Ultralytics DetectionModel/SegmentationModel (nn.Module with .model layers).
    With a 1-channel model the stem conv has a different shape and is skipped automatically.
    """
    target = model.state_dict()
    allowed = {f"model.{i}." for i in BACKBONE_TRANSFER_LAYERS}
    picked = {
        k: v for k, v in source_state.items()
        if any(k.startswith(prefix) for prefix in allowed)
        and k in target and v.shape == target[k].shape
    }
    model.load_state_dict(picked, strict=False)
    return TransferReport(source=source_name, transferred=sorted(picked), total=len(target))


def build_att_yolo(task: TaskMode, nc: int, channels: int, scale: Scale, run_dir: Path,
                   weights_dir: Path | None = None, pretrained: bool = True):
    """Create an ATT-YOLO model ready for `.train()`.

    Returns (YOLO, TransferReport | None). With pretrained=True the YOLOv5 backbone weights are
    copied in and the result is saved as a checkpoint in run_dir, because Ultralytics rebuilds
    a YAML-only model from scratch at train() and would drop the transferred weights.
    """
    from ultralytics import YOLO

    from ultralytics.nn.tasks import DetectionModel, SegmentationModel

    run_dir = Path(run_dir)
    yolo_task = "segment" if task == "seg" else "detect"
    yaml_path = write_att_yolo_yaml(run_dir / f"att-yolo-{scale}-{task}.yaml", task, nc, channels, scale)
    yolo = YOLO(str(yaml_path), task=yolo_task)
    # Ultralytics builds YAML models with ch=3 regardless of the file's `channels`; rebuild with
    # the project's channel count so grayscale models (and the transfer report) are right.
    model_cls = SegmentationModel if task == "seg" else DetectionModel
    yolo.model = model_cls(att_yolo_config(task, nc, channels, scale), ch=channels, nc=nc, verbose=False)
    yolo.model.yaml["yaml_file"] = str(yaml_path)
    if not pretrained:
        return yolo, None

    from ultralytics.nn.tasks import load_checkpoint

    from defect_inspection.core.weights import resolve_weights

    src_name = SOURCE_WEIGHTS[scale]
    src_path = resolve_weights(src_name, weights_dir or run_dir)
    src_model, _ = load_checkpoint(str(src_path))
    report = transfer_backbone(yolo.model, src_model.float().state_dict(), src_name)

    ckpt_path = run_dir / f"att-yolo-{scale}-{task}-init.pt"
    # A YAML-built model has no training args, so Ultralytics would reload this checkpoint with its
    # default task ("detect") and train a seg model with the detect trainer. Record the task on the
    # model object (kept by load_checkpoint) and in the checkpoint's train_args.
    yolo.model.task = yolo_task
    yolo.ckpt = {**(yolo.ckpt or {}), "train_args": {"task": yolo_task}}
    yolo.save(str(ckpt_path))
    return YOLO(str(ckpt_path), task=yolo_task), report
