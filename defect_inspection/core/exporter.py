"""ONNX export with verification (PRD F6.1-F6.4), and CoreML (.mlpackage) export on macOS (PRD 5.2).

- export_run(): best.pt of a training run -> <name>.onnx + <name>.metadata.json + <name>.classes.txt
- The input channel count follows the trained model (1 for grayscale projects, 3 otherwise) and is
  recorded in the metadata together with the exact preprocessing an external program must apply.
- ATT-YOLO contains an attention block whose shapes are traced as constants, so its export always
  uses a fixed input size (dynamic=False) equal to the training imgsz.
- Verification runs the exported file in onnxruntime and the checkpoint in PyTorch on the same
  images and requires every PyTorch detection to have an ONNX match of the same class with box
  IoU >= 0.99 (F6.2).
- Export runs on CPU (works for every model and avoids Ultralytics swapping the installed
  onnxruntime for onnxruntime-gpu); only FP16 needs a CUDA device.
- CoreML (format="coreml", macOS only, colour models): <name>.mlpackage (a folder) + the same
  sidecars. It runs on the Neural Engine / GPU in FP16, so its verification against PyTorch uses
  a looser IoU (COREML_VERIFY_IOU) than ONNX.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from defect_inspection import __version__
from defect_inspection.core.evaluate import box_iou, load_training_image, results_to_predictions, val_images
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.model_zoo import get_choice, is_unet
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import RunSpec, load_spec, read_status, weights_paths
from defect_inspection.platform_utils import (
    COREML_EXPORT_AVAILABLE,
    atomic_write_text,
    imread_unicode,
    resolve_device,
    sanitize_filename_part,
)

DEFAULT_NAME_PATTERN = "{project}_{mode}_{model}_{date}_{time}"
NAME_TOKENS = ("project", "mode", "model", "date", "time")
VERIFY_IOU = 0.99
VERIFY_CONF = 0.25
COREML_VERIFY_IOU = 0.9  # FP16 on the Neural Engine moves boxes slightly
SUFFIXES = {"onnx": ".onnx", "coreml": ".mlpackage"}


def with_export_suffix(name: str, fmt: str = "onnx") -> str:
    """`name` ending in the format's suffix (a different known suffix is replaced)."""
    for other in SUFFIXES.values():
        if name.lower().endswith(other):
            name = name[: -len(other)]
            break
    return name + SUFFIXES[fmt]


# ---- file names (F6.3) -------------------------------------------------------------------

def default_export_name(pattern: str, project: str, mode: str, model: str, now: datetime | None = None,
                        fmt: str = "onnx") -> str:
    """Default ONNX file name from a pattern with {project} {mode} {model} {date} {time}.

    The time is the export time (`now`). Every token value is made file-name safe, so a project
    named "PCB:라인" still gives a valid name on Windows, and spaces become "_" (spaces in file
    names trip up line-side scripts and command lines). Always ends in the format's suffix (".onnx" or ".mlpackage").
    """
    now = now or datetime.now()

    def token(text: str) -> str:
        return "_".join(sanitize_filename_part(text).split())

    values = {
        "project": token(project),
        "mode": token(mode),
        "model": token(model),
        "date": now.strftime("%Y%m%d"),
        "time": now.strftime("%H%M%S"),
    }
    try:
        name = pattern.format(**values)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"잘못된 파일명 패턴입니다 ({e}). 사용 가능: " + ", ".join("{" + t + "}" for t in NAME_TOKENS))
    return with_export_suffix(sanitize_filename_part(name), fmt)


def sidecar_paths(onnx_path: Path) -> tuple[Path, Path]:
    """<name>.metadata.json and <name>.classes.txt next to <name>.onnx / <name>.mlpackage."""
    stem = onnx_path.with_suffix("")
    return Path(f"{stem}.metadata.json"), Path(f"{stem}.classes.txt")


# ---- options --------------------------------------------------------------------------------

class ExportOptions(BaseModel):
    format: str = Field(default="onnx", pattern="^(onnx|coreml)$")  # coreml: macOS only
    imgsz: int | None = Field(default=None, ge=32, le=4096)  # None = training imgsz
    opset: int | None = Field(default=None, ge=12, le=23)  # None = Ultralytics' best for installed onnx
    dynamic: bool = False
    simplify: bool = True
    half: bool = False  # FP16, needs a CUDA device
    nms: bool = False  # NMS inside the ONNX graph


def effective_options(spec: RunSpec, options: ExportOptions) -> tuple[ExportOptions, list[str]]:
    """Options actually used, plus notes for every forced change."""
    notes = []
    opts = options.model_copy()
    if opts.format == "coreml" and (opts.dynamic or opts.opset is not None):
        notes.append("CoreML에는 dynamic·opset 옵션이 없어 쓰지 않았습니다")
        opts.dynamic, opts.opset = False, None
    if opts.imgsz is None:
        opts.imgsz = spec.train.imgsz
    if get_choice(spec.train.model).family == "att-yolo":
        if opts.dynamic:
            notes.append("ATT-YOLO의 attention 때문에 dynamic 입력을 끄고 크기를 고정했습니다")
            opts.dynamic = False
        if opts.imgsz != spec.train.imgsz:
            notes.append(f"ATT-YOLO는 학습 크기({spec.train.imgsz})로 고정합니다")
            opts.imgsz = spec.train.imgsz
    if opts.half and opts.format == "onnx":
        import torch

        if not torch.cuda.is_available():
            notes.append("FP16은 GPU가 필요해 FP32로 내보냈습니다")
            opts.half = False
    return opts, notes


# ---- verification (F6.2) --------------------------------------------------------------------

@dataclass
class Verification:
    passed: bool
    images: int = 0
    torch_detections: int = 0
    onnx_detections: int = 0
    min_iou: float | None = None  # worst match over all PyTorch detections
    max_score_diff: float | None = None
    input_name: str = ""
    input_shape: list = field(default_factory=list)
    input_channels: int = 0
    output_shapes: list = field(default_factory=list)
    message: str = ""


def compare_predictions(torch_preds: list, onnx_preds: list, iou_threshold: float = VERIFY_IOU) -> tuple[bool, float | None, float | None]:
    """(all matched, worst IoU, worst score difference) - each PyTorch detection needs an ONNX
    detection of the same class with IoU >= threshold, and the counts must agree."""
    used = [False] * len(onnx_preds)
    worst_iou, worst_diff = None, None
    ok = len(torch_preds) == len(onnx_preds)
    for t in sorted(torch_preds, key=lambda p: -p.score):
        best, best_iou = -1, -1.0
        for j, o in enumerate(onnx_preds):
            if used[j] or o.shape.class_id != t.shape.class_id:
                continue
            v = box_iou(t.shape.bounds(), o.shape.bounds())
            if v > best_iou:
                best, best_iou = j, v
        if best < 0:
            ok = False
            worst_iou = 0.0
            continue
        used[best] = True
        diff = abs(t.score - onnx_preds[best].score)
        worst_iou = best_iou if worst_iou is None else min(worst_iou, best_iou)
        worst_diff = diff if worst_diff is None else max(worst_diff, diff)
        ok = ok and best_iou >= iou_threshold
    return ok, worst_iou, worst_diff


def _yolo_task(task: str) -> str:
    return "segment" if task == "seg" else "detect"


def verify_coreml(package: Path, weights: Path, task: str, images: list[np.ndarray], imgsz: int) -> Verification:
    """Compare PyTorch (CPU) with the .mlpackage run by Core ML (macOS), through Ultralytics'
    predictor so pre- and post-processing are the same. Core ML computes in FP16 on the Neural
    Engine/GPU, hence COREML_VERIFY_IOU instead of the ONNX threshold."""
    from ultralytics import YOLO

    v = Verification(passed=False, input_name="image", input_shape=[1, 3, imgsz, imgsz], input_channels=3)
    pt_model = YOLO(str(weights), task=_yolo_task(task))
    ml_model = YOLO(str(package), task=_yolo_task(task))
    worst_iou, worst_diff, ok = None, None, True
    for image in images:
        tp = results_to_predictions(pt_model.predict(image, imgsz=imgsz, conf=VERIFY_CONF, verbose=False,
                                                     device=resolve_device("auto", "verify"))[0], task)
        mp = results_to_predictions(ml_model.predict(image, imgsz=imgsz, conf=VERIFY_CONF, verbose=False)[0], task)
        match_ok, iou, diff = compare_predictions(tp, mp, COREML_VERIFY_IOU)
        ok = ok and match_ok
        v.images += 1
        v.torch_detections += len(tp)
        v.onnx_detections += len(mp)
        if iou is not None:
            worst_iou = iou if worst_iou is None else min(worst_iou, iou)
        if diff is not None:
            worst_diff = diff if worst_diff is None else max(worst_diff, diff)
    v.passed, v.min_iou, v.max_score_diff = ok, worst_iou, worst_diff
    v.message = (f"PyTorch {v.torch_detections}개 / Core ML {v.onnx_detections}개 검출, "
                 f"같은 클래스 IoU ≥ {COREML_VERIFY_IOU}" + (f", 최소 IoU {worst_iou:.4f}" if worst_iou is not None else ""))
    return v


def verify_onnx(onnx_path: Path, weights: Path, task: str, images: list[np.ndarray], imgsz: int,
                channels: int) -> Verification:
    """Run the same images through PyTorch and onnxruntime (via Ultralytics' predictor, so pre-
    and post-processing are identical) and compare, then inspect the raw onnxruntime session."""
    import onnxruntime as ort
    from ultralytics import YOLO

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    v = Verification(passed=False, input_name=inp.name, input_shape=shape,
                     input_channels=int(shape[1]) if isinstance(shape[1], int) else 0,
                     output_shapes=[list(o.shape) for o in sess.get_outputs()])
    if v.input_channels != channels:
        v.message = f"ONNX 입력 채널 {v.input_channels} ≠ 프로젝트 {channels}"
        return v

    # Always pass the task: Ultralytics guesses it from the checkpoint otherwise, and a wrong guess
    # (detect post-processing on a seg model) breaks prediction.
    yolo_task = "segment" if task == "seg" else "detect"
    device = resolve_device("auto", "verify")  # CPU on purpose: compare like with like
    pt_model = YOLO(str(weights), task=yolo_task)
    onnx_model = YOLO(str(onnx_path), task=yolo_task)
    worst_iou, worst_diff, ok = None, None, True
    for image in images:
        tp = results_to_predictions(pt_model.predict(image, imgsz=imgsz, conf=VERIFY_CONF, verbose=False,
                                                     device=device)[0], task)
        op = results_to_predictions(onnx_model.predict(image, imgsz=imgsz, conf=VERIFY_CONF, verbose=False,
                                                       device=device)[0], task)
        match_ok, iou, diff = compare_predictions(tp, op)
        ok = ok and match_ok
        v.images += 1
        v.torch_detections += len(tp)
        v.onnx_detections += len(op)
        if iou is not None:
            worst_iou = iou if worst_iou is None else min(worst_iou, iou)
        if diff is not None:
            worst_diff = diff if worst_diff is None else max(worst_diff, diff)
    v.min_iou, v.max_score_diff, v.passed = worst_iou, worst_diff, ok and v.images > 0
    if v.passed:
        v.message = (f"일치: 이미지 {v.images}장, 검출 {v.torch_detections}개"
                     + (f", 최소 IoU {worst_iou:.4f}" if worst_iou is not None else " (검출 없음)"))
    else:
        v.message = (f"불일치: PyTorch {v.torch_detections}개 / ONNX {v.onnx_detections}개"
                     + (f", 최소 IoU {worst_iou:.4f}" if worst_iou is not None else ""))
    return v


def verification_images(project: Project, spec: RunSpec, limit: int = 3) -> list[np.ndarray]:
    """A few Val images (training-equivalent pixels); a synthetic one if none can be read."""
    images = []
    if spec.tiling() is not None:
        # crop model: verify on what it is fed, single tiles (the Val crops of the dataset)
        val_dir = project.dataset_dir / "crop" / "images" / "val"
        files = sorted(val_dir.iterdir()) if val_dir.is_dir() else []
        for f in files[:limit]:
            raw = imread_unicode(f)
            if raw is not None:
                images.append(to_training_8bit(raw, project.channels))
    else:
        for path in val_images(project, spec.train.dataset)[:limit]:
            img = load_training_image(project, path)
            if img is not None:
                images.append(img)
    if not images:
        rng = np.random.default_rng(0)
        shape = (spec.train.imgsz, spec.train.imgsz) + (() if project.channels == 1 else (3,))
        images.append(rng.integers(0, 255, shape, dtype=np.uint8))
    return images


# ---- metadata (F6.4) ------------------------------------------------------------------------

def crop_metadata(project: Project, spec: RunSpec) -> dict[str, Any] | None:
    """Crop models: how to run them on a whole image (F7.6) + crop size and fixed ROIs."""
    tiling = spec.tiling()
    if tiling is None:
        return None
    from defect_inspection.core.roi import RoiStore

    crop = tiling.crop
    return {
        "tiling": tiling.model_dump(mode="json"),
        "how": (f"전체 이미지를 {tiling.tile_size}px 타일로 자르되 이웃 타일과 {tiling.overlap_px}px 겹치게 하고 "
                "(마지막 타일은 이미지 오른쪽/아래 끝에 맞춤, 이미지가 타일보다 작으면 이미지 전체 1장), "
                "타일마다 추론한 좌표에 타일 원점(x1, y1)을 더해 원본 좌표로 바꾼 뒤, 같은 클래스끼리 점수 높은 순으로 "
                f"IoU ≥ {tiling.merge_iou} 또는 작은 박스 기준 겹침(IoS) ≥ {tiling.merge_ios}이면 낮은 쪽을 제거"),
        "crop_size": crop.get("crop_size"),
        "sources": [k for k in ("roi", "defect", "background", "tiles") if crop.get(f"use_{k}")],
        "common_rois": [[r.x1, r.y1, r.x2, r.y2] for r in RoiStore(project.root).common],
    }


def edge_filter_metadata(project: Project) -> dict[str, Any] | None:
    """The project's edge filter (F9), recorded so nobody expects it from the model: DefectLab drops
    predictions off the outermost edge after inference, the exported model does not."""
    cfg = project.config.edge_filter
    if not cfg.enabled:
        return None
    return {"applied_in_model": False,
            "note": "DefectLab 안(Result·제안·비교)에서만 예측 뒤에 적용됩니다. 이 모델의 출력에는 적용되지 않습니다.",
            "settings": cfg.model_dump(mode="json")}


def build_metadata(project: Project, spec: RunSpec, run_dir: Path, opts: ExportOptions,
                   verification: Verification | None, notes: list[str]) -> dict[str, Any]:
    status = read_status(run_dir)
    channels = project.channels
    seg = spec.task == "seg"
    if opts.nms:
        output = "(1, max_det, 6): x1, y1, x2, y2, score, class (NMS 적용됨, 입력 이미지 픽셀 좌표 = letterbox 좌표)"
    else:
        output = (f"(1, 4 + {spec.nc}" + (" + 32" if seg else "") + ", N): cx, cy, w, h, 클래스별 점수"
                  + (", 마스크 계수 32 (두 번째 출력이 마스크 프로토타입)" if seg else "") + " — NMS는 직접 적용")
    meta = {
        "app": {"name": "DefectLab", "version": __version__},
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "format": opts.format,
        "run_id": spec.run_id,
        "model": spec.train.model,
        "task": spec.task,
        "nc": spec.nc,
        "names": spec.names,
        "input": {
            "shape": [1, channels, opts.imgsz, opts.imgsz] if not opts.dynamic else [1, channels, "H", "W"],
            "channels": channels,
            "color_order": "GRAY" if channels == 1 else "RGB",
            "dtype": "float16" if opts.half else "float32",
            "dynamic": opts.dynamic,
        },
        "preprocess": {
            "source_bit_depth": "8bit 그대로, 16bit는 값 >> 8 (학습과 동일)",
            "color": "1채널 그레이" if channels == 1 else "BGR 이미지라면 RGB로 바꿔서 입력",
            "letterbox": {"size": opts.imgsz, "keep_ratio": True, "pad_value": 114, "center": True},
            "normalize": "0~255 → 0~1 (255로 나눔)",
            "layout": "NCHW",
        },
        "output": output,
        "recommended_thresholds": {"conf": 0.25, "iou": 0.7},
        "export_options": opts.model_dump(),
        "train": {"epochs": spec.train.epochs, "imgsz": spec.train.imgsz, "dataset": spec.train.dataset},
        "metrics": status.get("final_metrics") or status.get("metrics") or {},
        "crop": crop_metadata(project, spec),
        "edge_filter": edge_filter_metadata(project),
        "verification": None if verification is None else asdict(verification),
        "notes": notes,
    }
    if opts.format == "coreml":
        meta["input"] = {"type": "image (Core ML ImageType, RGB)", "size": [opts.imgsz, opts.imgsz], "channels": 3,
                         "weights": "float16" if opts.half else "float32"}
        meta["preprocess"] = {
            "source_bit_depth": "8bit 그대로, 16bit는 값 >> 8 (학습과 동일)",
            "color": "RGB 이미지(CGImage/CVPixelBuffer/PIL)로 입력",
            "letterbox": {"size": opts.imgsz, "keep_ratio": True, "pad_value": 114, "center": True},
            "normalize": "모델 안에서 처리 (ImageType scale 1/255)",
        }
        if opts.nms:
            meta["output"] = ("Core ML NMS 파이프라인: confidence (N, 클래스 수), coordinates (N, 4) "
                              "= 입력 크기 대비 정규화된 cx, cy, w, h")
    return meta


# ---- export ----------------------------------------------------------------------------------

@dataclass
class ExportResult:
    onnx_path: Path  # the exported model: .onnx file, or .mlpackage folder for CoreML
    metadata_path: Path
    classes_path: Path
    options: ExportOptions
    verification: Verification | None
    notes: list[str]
    seconds: float


def _export_unet(project: Project, spec: RunSpec, run_dir: Path, best: Path, out_path: Path,
                 options: ExportOptions, verify: bool, t0: float) -> ExportResult:
    """U-Net++ runs (core/unet.py): torch.onnx at the training input size, verified by pixel classes."""
    from defect_inspection.core import unet

    if options.format != "onnx":
        raise ValueError("U-Net++ 모델은 ONNX로만 내보낼 수 있습니다")
    meta = unet.export_onnx(best, out_path, options.opset)
    notes = []
    if options.imgsz not in (None, meta["imgsz"]):
        notes.append(f"U-Net++는 학습 입력 크기 {meta['imgsz']}로 내보냅니다 (요청 {options.imgsz} 무시)")
    if options.half or options.dynamic or options.nms:
        notes.append("U-Net++ 내보내기는 FP32 · 고정 크기 · NMS 없음만 지원합니다")
    opts = options.model_copy(update={"imgsz": meta["imgsz"], "half": False, "dynamic": False, "nms": False})
    verification = None
    if verify:
        verification = unet.verify_onnx(out_path, best, verification_images(project, spec))
    meta_path, classes_path = sidecar_paths(out_path)
    metadata = build_metadata(project, spec, run_dir, opts, verification, notes)
    metadata.update(unet.onnx_metadata(meta))
    metadata["task"] = "semantic_segmentation"
    atomic_write_text(meta_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    atomic_write_text(classes_path, "".join(f"{name}\n" for name in spec.names))
    return ExportResult(out_path, meta_path, classes_path, opts, verification, notes, round(time.monotonic() - t0, 2))


def export_run(project: Project, run_dir: Path, out_path: Path, options: ExportOptions | None = None,
               verify: bool = True) -> ExportResult:
    """Export a run's best.pt to `out_path` (+ sidecars) and verify it: ONNX with onnxruntime,
    CoreML (macOS) with Core ML itself."""
    from ultralytics import YOLO

    t0 = time.monotonic()
    options = options or ExportOptions()
    run_dir, out_path = Path(run_dir), Path(out_path)
    out_path = out_path.with_name(with_export_suffix(out_path.name, options.format))
    if options.format == "coreml":
        if not COREML_EXPORT_AVAILABLE:
            raise RuntimeError("CoreML(.mlpackage) 내보내기는 macOS에서만 할 수 있습니다")
        if project.channels != 3:
            raise ValueError("CoreML 내보내기는 컬러(3채널) 모델만 지원합니다. 그레이 모델은 ONNX로 내보내세요")
    spec = load_spec(run_dir)
    best, _ = weights_paths(run_dir)
    if not best.is_file():
        raise FileNotFoundError(f"best.pt가 없습니다: {best}")
    if spec.nc != len(project.config.classes) or spec.channels != project.channels:
        raise ValueError("이 run은 현재 프로젝트의 클래스/색 모드와 다르게 학습됐습니다")
    if is_unet(spec.train.model):
        return _export_unet(project, spec, run_dir, best, out_path, options, verify, t0)
    opts, notes = effective_options(spec, options)

    if opts.format == "coreml":
        # FP16 weights are fine here (the Neural Engine computes in FP16 anyway); no CUDA needed
        exported = Path(YOLO(str(best), task=_yolo_task(spec.task)).export(
            format="coreml", imgsz=opts.imgsz, half=opts.half, nms=opts.nms, batch=1, device="cpu", verbose=False))
    else:
        exported = Path(YOLO(str(best)).export(
            format="onnx", imgsz=opts.imgsz, opset=opts.opset, dynamic=opts.dynamic, simplify=opts.simplify,
            half=opts.half, nms=opts.nms, batch=1, device="0" if opts.half else "cpu", verbose=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_dir():  # an .mlpackage is a folder: replace it as a whole
        shutil.rmtree(out_path)
    shutil.move(str(exported), str(out_path))

    verification = None
    if verify:
        images = verification_images(project, spec)
        if opts.format == "coreml":
            verification = verify_coreml(out_path, best, spec.task, images, opts.imgsz)
        else:
            verification = verify_onnx(out_path, best, spec.task, images, opts.imgsz, project.channels)
    meta_path, classes_path = sidecar_paths(out_path)
    metadata = build_metadata(project, spec, run_dir, opts, verification, notes)
    atomic_write_text(meta_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    atomic_write_text(classes_path, "".join(f"{name}\n" for name in spec.names))
    return ExportResult(out_path, meta_path, classes_path, opts, verification, notes,
                        round(time.monotonic() - t0, 2))
