"""ONNX export (PROMPTS step 14, PRD F6.1-F6.4)."""

import json
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from defect_inspection.core import exporter as X
from defect_inspection.core.annotation import BBox
from defect_inspection.core.augment_config import AugmentConfig, TransformSetting
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.evaluate import Prediction
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import TrainConfig, prepare_run, weights_paths, write_status
from defect_inspection.platform_utils import imwrite_unicode, invalid_filename_reason, path_too_long

S = 128


def make_project(root, task="detect", color_mode="color", n=6):
    src = root / "원본"
    src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    centers = []
    for i in range(n):
        img = np.full((S, S, 3), 200, np.uint8)
        cx, cy = int(rng.integers(30, S - 30)), int(rng.integers(30, S - 30))
        cv2.circle(img, (cx, cy), 14, (40, 40, 40), -1)
        imwrite_unicode(src / f"img{i}.png", img)
        centers.append((cx, cy))
    p = Project.create(root / "내보내기 프로젝트", name="PCB:라인", task=task, color_mode=color_mode)
    p.add_class("dent")
    p.add_class("scratch")
    p.add_images([src])
    for (cx, cy), e in zip(centers, p.config.images):
        ann = p.load_labels(e)
        ann.shapes.append(BBox(class_id=0, x1=cx - 15, y1=cy - 15, x2=cx + 15, y2=cy + 15))
        p.save_labels(ann)
    p.config.augment = AugmentConfig(multiplier=1, transforms={"hflip": TransformSetting(enabled=True)})
    p.save()
    build_full_dataset(p)
    return p


def untrained_run(project, model="yolo11n", imgsz=S, dataset="full"):
    """A finished run whose best.pt is an untrained model of the right task/channels/classes."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel

    from defect_inspection.core.att_yolo import att_yolo_config

    run = prepare_run(project, TrainConfig(model=model, imgsz=imgsz, epochs=1, dataset=dataset), project.root / "w")
    seg = project.config.task == "seg"
    nc, ch = len(project.config.classes), project.channels
    if model.startswith("att-yolo"):
        cfg = att_yolo_config(project.config.task, nc, ch, model[-1])
    else:
        cfg = f"yolo11{model[-1]}{'-seg' if seg else ''}.yaml"
    yolo = YOLO(cfg if isinstance(cfg, str) else "yolo11n.yaml", task="segment" if seg else "detect")
    yolo.model = (SegmentationModel if seg else DetectionModel)(cfg, ch=ch, nc=nc, verbose=False)
    yolo.model.names = {i: c.name for i, c in enumerate(project.config.classes)}
    yolo.model.task = "segment" if seg else "detect"  # YAML-built: Ultralytics would assume detect
    best, _ = weights_paths(run)
    best.parent.mkdir(parents=True)
    yolo.save(str(best))
    write_status(run, {"state": "finished", "final_metrics": {"metrics/mAP50-95(B)": 0.5}})
    return run


# ---- names (F6.3) ---------------------------------------------------------------------

def test_default_export_name():
    now = datetime(2026, 10, 13, 14, 30, 25)
    name = X.default_export_name(X.DEFAULT_NAME_PATTERN, "pcb", "crop", "yolo11s", now)
    assert name == "pcb_crop_yolo11s_20261013_143025.onnx"


def test_default_export_name_sanitizes_and_keeps_korean():
    now = datetime(2026, 1, 2, 3, 4, 5)
    name = X.default_export_name(X.DEFAULT_NAME_PATTERN, "PCB:라인/1", "full", "att-yolo-s", now)
    assert name == "PCB_라인_1_full_att-yolo-s_20260102_030405.onnx"
    spaced = X.default_export_name(X.DEFAULT_NAME_PATTERN, "찍힘 검사  2라인", "full", "yolo11n", now)
    assert spaced == "찍힘_검사_2라인_full_yolo11n_20260102_030405.onnx"
    assert invalid_filename_reason(name) is None


def test_custom_pattern_and_extension():
    now = datetime(2026, 1, 2, 3, 4, 5)
    assert X.default_export_name("{model}-{date}", "p", "full", "yolo11n", now) == "yolo11n-20260102.onnx"
    assert X.default_export_name("{project}.onnx", "p", "full", "m", now) == "p.onnx"
    with pytest.raises(ValueError, match="패턴"):
        X.default_export_name("{projekt}", "p", "full", "m", now)


def test_name_uses_export_time():
    a = X.default_export_name(X.DEFAULT_NAME_PATTERN, "p", "full", "m", datetime(2026, 1, 1, 0, 0, 0))
    b = X.default_export_name(X.DEFAULT_NAME_PATTERN, "p", "full", "m", datetime(2026, 1, 1, 0, 0, 1))
    assert a != b


@pytest.mark.parametrize("name, ok", [
    ("model.onnx", True), ("모델 1.onnx", True), ("a?b.onnx", False), ("a:b", False),
    ("CON.onnx", False), ("com3", False), ("ends.", False), ("", False),
])
def test_invalid_filename_reason(name, ok):
    assert (invalid_filename_reason(name) is None) == ok


def test_path_too_long(tmp_path):
    assert not path_too_long(tmp_path / "a.onnx")
    assert path_too_long(tmp_path / ("x" * 300 + ".onnx"))


def test_sidecar_paths(tmp_path):
    meta, classes = X.sidecar_paths(tmp_path / "모델 v1.onnx")
    assert meta.name == "모델 v1.metadata.json" and classes.name == "모델 v1.classes.txt"


# ---- options -------------------------------------------------------------------------------

def spec(model="yolo11s", imgsz=640):
    from defect_inspection.core.trainer import RunSpec

    return RunSpec(run_id="r", project_root=".", task="detect", nc=1, names=["d"], channels=3,
                   data_yaml="d", weights_dir=".", created="now", train=TrainConfig(model=model, imgsz=imgsz))


def test_effective_options_defaults_to_training_imgsz():
    opts, notes = X.effective_options(spec(imgsz=512), X.ExportOptions(dynamic=True))
    assert opts.imgsz == 512 and opts.dynamic and notes == []


def test_att_yolo_forces_fixed_training_size():
    opts, notes = X.effective_options(spec("att-yolo-n", 1280), X.ExportOptions(imgsz=640, dynamic=True))
    assert opts.imgsz == 1280 and not opts.dynamic and len(notes) == 2


def test_half_without_gpu_falls_back(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    opts, notes = X.effective_options(spec(), X.ExportOptions(half=True))
    assert not opts.half and "FP16" in notes[0]


# ---- verification logic -------------------------------------------------------------------------

def P(x1, y1, x2, y2, s, c=0):
    return Prediction(BBox(class_id=c, x1=x1, y1=y1, x2=x2, y2=y2, status="suggested", source="model", score=s))


def test_compare_predictions():
    a = [P(0, 0, 10, 10, 0.9), P(20, 20, 30, 30, 0.8, c=1)]
    ok, iou, diff = X.compare_predictions(a, [P(0, 0, 10, 10, 0.89), P(20, 20, 30, 30, 0.8, c=1)])
    assert ok and iou == 1.0 and diff == pytest.approx(0.01)
    ok, iou, _ = X.compare_predictions(a, [P(0, 0, 10, 10.5, 0.9), P(20, 20, 30, 30, 0.8, c=1)])
    assert not ok and iou < 0.99  # box moved half a pixel
    assert not X.compare_predictions(a, [P(0, 0, 10, 10, 0.9)])[0]  # missing detection
    assert not X.compare_predictions(a, [P(0, 0, 10, 10, 0.9), P(20, 20, 30, 30, 0.8, c=0)])[0]  # wrong class
    assert X.compare_predictions([], []) == (True, None, None)


# ---- export (untrained checkpoints: mechanics, shapes, metadata) ---------------------------------

@pytest.mark.parametrize("task, color_mode, model, channels", [
    ("detect", "color", "yolo11n", 3),
    ("detect", "grayscale", "yolo11n", 1),
    ("seg", "color", "yolo11n", 3),
    ("detect", "grayscale", "att-yolo-n", 1),
])
def test_export_run(tmp_path, task, color_mode, model, channels):
    p = make_project(tmp_path, task=task, color_mode=color_mode)
    run = untrained_run(p, model=model)
    out = tmp_path / "내보내기" / X.default_export_name(X.DEFAULT_NAME_PATTERN, p.config.name, "full", model)
    result = X.export_run(p, run, out, X.ExportOptions(dynamic=True))

    assert result.onnx_path == out and out.is_file()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    assert shape[1] == channels
    if model.startswith("att-yolo"):
        assert shape == [1, channels, S, S] and result.notes  # attention -> fixed size
    assert result.verification.passed, result.verification.message
    assert result.verification.input_channels == channels

    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["input"]["channels"] == channels
    assert meta["input"]["color_order"] == ("GRAY" if channels == 1 else "RGB")
    assert meta["names"] == ["dent", "scratch"] and meta["task"] == task and meta["model"] == model
    assert meta["preprocess"]["letterbox"]["size"] == S and meta["metrics"]["metrics/mAP50-95(B)"] == 0.5
    assert meta["verification"]["passed"] is True
    assert result.classes_path.read_text(encoding="utf-8") == "dent\nscratch\n"
    best, _ = weights_paths(run)
    assert not best.with_suffix(".onnx").exists()  # moved, not left next to best.pt


def test_export_appends_extension_and_overwrites(tmp_path):
    p = make_project(tmp_path)
    run = untrained_run(p)
    out = tmp_path / "model"
    r1 = X.export_run(p, run, out, verify=False)
    assert r1.onnx_path.name == "model.onnx" and r1.verification is None
    r2 = X.export_run(p, run, r1.onnx_path, verify=False)
    assert r2.onnx_path == r1.onnx_path


def test_export_refuses_mismatched_run(tmp_path):
    p = make_project(tmp_path)
    run = untrained_run(p)
    p.add_class("stain")
    with pytest.raises(ValueError, match="클래스"):
        X.export_run(p, run, tmp_path / "x.onnx")


def test_export_without_best(tmp_path):
    p = make_project(tmp_path)
    run = prepare_run(p, TrainConfig(), tmp_path / "w")
    with pytest.raises(FileNotFoundError):
        X.export_run(p, run, tmp_path / "x.onnx")


# ---- trained model (slow): detections must match between PyTorch and onnxruntime ----------------

@pytest.mark.slow
@pytest.mark.parametrize("nms", [False, True])
def test_trained_model_export_matches(tmp_path, nms):
    from ultralytics import YOLO

    from defect_inspection.core.weights import resolve_weights
    from defect_inspection.platform_utils import cache_dir

    p = make_project(tmp_path, n=8)
    p.config.augment.multiplier = 6
    p.save()
    build_full_dataset(p)
    run = prepare_run(p, TrainConfig(model="yolo11n", imgsz=S, epochs=12), tmp_path / "w")
    yolo = YOLO(str(resolve_weights("yolo11n.pt", cache_dir() / "weights")))
    yolo.train(data=str(p.dataset_dir / "full" / "data.yaml"), epochs=12, imgsz=S, batch=8, workers=0,
               device="cpu", project=str(run), name="train", exist_ok=True, plots=False, verbose=False)
    write_status(run, {"state": "finished"})
    result = X.export_run(p, run, tmp_path / "trained.onnx", X.ExportOptions(nms=nms))
    v = result.verification
    assert v.torch_detections > 0, "model should detect the training blobs"
    assert v.passed, v.message
    assert v.min_iou >= X.VERIFY_IOU
