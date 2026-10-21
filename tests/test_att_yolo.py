"""ATT-YOLO adaptation (PRD F4.8) and the model registry (F4.1)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from defect_inspection.core import model_zoo
from defect_inspection.core.att_yolo import (
    BACKBONE_TRANSFER_LAYERS,
    HEAD_LEVEL_LAYERS,
    att_yolo_config,
    transfer_backbone,
    write_att_yolo_yaml,
)

pytest.importorskip("ultralytics")
from ultralytics.nn.modules import C3TR, SPP, Detect, Segment  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, SegmentationModel  # noqa: E402

MODEL = {"detect": DetectionModel, "seg": SegmentationModel}


def build(task="detect", channels=3, scale="s", nc=2):
    return MODEL[task](att_yolo_config(task, nc, channels, scale), ch=channels, verbose=False)


# ---- structure ---------------------------------------------------------------------

@pytest.mark.parametrize("task", ["detect", "seg"])
@pytest.mark.parametrize("channels", [3, 1])
@pytest.mark.parametrize("scale", ["n", "s"])
def test_builds_with_six_levels(task, channels, scale):
    m = build(task, channels, scale)
    assert m.stride.int().tolist() == [4, 8, 16, 32, 64, 128]
    head = m.model[-1]
    assert isinstance(head, Segment if task == "seg" else Detect)
    assert head.nl == 6 and head.nc == 2
    with torch.no_grad():
        m.eval()(torch.zeros(1, channels, 256, 256))


def test_attention_sits_behind_p7():
    m = build()
    assert isinstance(m.model[12], SPP)
    assert isinstance(m.model[14], C3TR)
    assert m.model[11].conv.stride == (2, 2)  # P7 downsample
    assert HEAD_LEVEL_LAYERS[-1] > 14


def test_size_close_to_paper():
    """Paper (s): 23.1 M params. Measured 22.4 M; FLOPs are higher (see att_yolo.py)."""
    params = sum(p.numel() for p in build(nc=80).parameters()) / 1e6
    assert 21.0 < params < 24.0
    n_params = sum(p.numel() for p in build(scale="n").parameters()) / 1e6
    assert n_params < params / 3


def test_invalid_arguments():
    with pytest.raises(ValueError):
        att_yolo_config("detect", 1, 3, "x")
    with pytest.raises(ValueError):
        att_yolo_config("classify", 1, 3, "s")


def test_yaml_roundtrip(tmp_path):
    path = write_att_yolo_yaml(tmp_path / "모델 설정" / "att.yaml", "seg", 3, channels=1, scale="n")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == att_yolo_config("seg", 3, 1, "n")
    # The trainer builds from the file with the dataset's channels
    m = SegmentationModel(str(path), ch=1, verbose=False)
    assert m.yaml["channels"] == 1 and m.model[-1].nc == 3


@pytest.mark.parametrize("scale", ["n", "s"])
def test_scale_survives_loading_from_file(tmp_path, scale):
    """Ultralytics guesses `scale` from the file name; our explicit multiples must win."""
    path = write_att_yolo_yaml(tmp_path / f"att-yolo-{scale}-detect.yaml", "detect", 1, scale=scale)
    from_file = DetectionModel(str(path), verbose=False)
    from_dict = build("detect", 3, scale, nc=1)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(from_file) == count(from_dict)


def test_make_model_grayscale_has_one_channel(tmp_path):
    yolo, _ = model_zoo.make_model("att-yolo-s", "detect", nc=1, channels=1,
                                   run_dir=tmp_path, weights_dir=tmp_path / "w", pretrained=False)
    assert yolo.model.model[0].conv.in_channels == 1
    assert yolo.model.yaml["channels"] == 1
    params = sum(p.numel() for p in yolo.model.parameters()) / 1e6
    assert params > 20  # s, not silently n


def test_config_is_a_fresh_copy():
    a = att_yolo_config("detect", 1)
    a["backbone"][0][3][0] = 999
    assert att_yolo_config("detect", 1)["backbone"][0][3][0] == 64


# ---- backbone transfer -------------------------------------------------------------

@pytest.fixture(scope="module")
def yolov5s_state():
    """Same architecture as yolov5su.pt (random weights, no download)."""
    torch.manual_seed(0)
    return DetectionModel("yolov5s.yaml", verbose=False).state_dict()


def test_transfer_copies_only_backbone_p2_to_p5(yolov5s_state):
    m = build(nc=1)
    before = copy.deepcopy(m.state_dict())
    report = transfer_backbone(m, yolov5s_state, "yolov5s")

    assert report.layers == list(BACKBONE_TRANSFER_LAYERS)
    after = m.state_dict()
    for key in report.transferred:
        assert torch.equal(after[key], yolov5s_state[key])
    # P6/P7, attention and head keep their own init
    for key in ("model.9.conv.weight", "model.14.cv1.conv.weight"):
        assert torch.equal(after[key], before[key])
    assert any(k.startswith("model.50.") for k in after)  # head exists, not transferred
    assert not any(k.startswith("model.50.") for k in report.transferred)


def test_transfer_skips_stem_for_grayscale(yolov5s_state):
    rgb = transfer_backbone(build(nc=1), yolov5s_state)
    gray = transfer_backbone(build(channels=1, nc=1), yolov5s_state)
    assert "model.0.conv.weight" in rgb.transferred
    assert "model.0.conv.weight" not in gray.transferred
    assert len(gray.transferred) == len(rgb.transferred) - 1


def test_transfer_report_summary(yolov5s_state):
    report = transfer_backbone(build(scale="s"), yolov5s_state, "yolov5su.pt")
    assert report.summary().startswith("yolov5su.pt: 186/")
    assert "layers 0-8" in report.summary()


# ---- registry ----------------------------------------------------------------------

def test_model_choices():
    keys = [c.key for c in model_zoo.MODEL_CHOICES]
    assert keys == ["yolo11n", "yolo11s", "yolo11m", "att-yolo-n", "att-yolo-s"]
    assert model_zoo.get_choice(model_zoo.DEFAULT_MODEL).family == "yolo11"
    att = model_zoo.get_choice("att-yolo-s")
    assert (att.default_imgsz, att.default_epochs) == (1280, 300) and "Compare" in att.note
    with pytest.raises(KeyError):
        model_zoo.get_choice("yolov9")


def test_paper_hyperparameters():
    assert model_zoo.PAPER_HYP == {"lr0": 0.0032, "lrf": 0.12, "momentum": 0.843,
                                   "weight_decay": 0.00036, "mosaic": 1.0, "mixup": 0.243}


def test_yolo11_weight_names():
    assert model_zoo.yolo11_weights_name("s", "detect") == "yolo11s.pt"
    assert model_zoo.yolo11_weights_name("n", "seg") == "yolo11n-seg.pt"


def test_make_model_att_without_pretrained(tmp_path):
    yolo, report = model_zoo.make_model("att-yolo-n", "seg", nc=2, channels=1,
                                        run_dir=tmp_path / "run", weights_dir=tmp_path / "w", pretrained=False)
    assert report is None
    assert yolo.task == "segment" and yolo.model.model[-1].nl == 6
    assert (tmp_path / "run" / "att-yolo-n-seg.yaml").is_file()
    assert not (tmp_path / "w").exists()  # nothing downloaded


def test_make_model_yolo11_without_pretrained(tmp_path):
    yolo, report = model_zoo.make_model("yolo11n", "detect", nc=1, channels=3,
                                        run_dir=tmp_path, weights_dir=tmp_path / "w", pretrained=False)
    assert report is None and yolo.task == "detect"


# ---- end to end: train -> ONNX (slow; run with `pytest -m slow`) -------------------

def _tiny_dataset(root: Path, channels: int) -> Path:
    import cv2

    from defect_inspection.platform_utils import imwrite_unicode

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(4):
            img = np.full((256, 256, 3), 170, np.uint8) + rng.integers(0, 10, (256, 256, 3), dtype=np.uint8)
            cx, cy, r = rng.integers(60, 196), rng.integers(60, 196), int(rng.integers(12, 30))
            cv2.circle(img, (int(cx), int(cy)), r, (50, 50, 50), -1)
            if channels == 1:
                img = img[..., 0]
            imwrite_unicode(root / "images" / split / f"결함 {i}.png", img)
            (root / "labels" / split / f"결함 {i}.txt").write_text(
                f"0 {cx / 256:.6f} {cy / 256:.6f} {2 * r / 256:.6f} {2 * r / 256:.6f}\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                    "names": {0: "dent"}, "channels": channels}, allow_unicode=True),
                    encoding="utf-8")
    return data


@pytest.mark.slow
@pytest.mark.parametrize("channels", [3, 1])
def test_train_and_export_onnx(tmp_path, channels):
    import onnxruntime as ort

    from defect_inspection.platform_utils import cache_dir

    data = _tiny_dataset(tmp_path / "데이터 셋", channels)
    yolo, report = model_zoo.make_model("att-yolo-n", "detect", nc=1, channels=channels,
                                        run_dir=tmp_path / "run", weights_dir=cache_dir() / "weights")
    assert report is not None and report.layers == list(range(0, 9))
    key = "model.2.cv1.conv.weight"
    transferred = report.transferred and yolo.model.state_dict()[key].clone()

    seen = {}

    def check_weights_survive(trainer):
        seen["equal"] = torch.allclose(trainer.model.state_dict()[key].float().cpu(), transferred.float(), atol=1e-3)

    yolo.add_callback("on_pretrain_routine_end", check_weights_survive)
    yolo.train(data=str(data), epochs=1, imgsz=256, batch=4, workers=0, device="cpu",
               project=str(tmp_path / "runs"), name="smoke", plots=False, amp=False, verbose=False)
    assert seen["equal"], "transferred backbone weights were discarded by the trainer"

    best = Path(yolo.trainer.best)
    assert best.is_file()
    from ultralytics import YOLO

    onnx_path = Path(YOLO(str(best)).export(format="onnx", imgsz=256, opset=12, simplify=False))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    assert inp.shape[1] == channels
    out = sess.run(None, {inp.name: np.zeros((1, channels, 256, 256), np.float32)})[0]
    # 4 box coords + 1 class, one prediction per grid cell over the six levels
    assert out.shape == (1, 5, sum((256 // s) ** 2 for s in (4, 8, 16, 32, 64, 128)))


@pytest.mark.parametrize("task, expected", [("detect", "detect"), ("seg", "segment")])
def test_pretrained_init_checkpoint_keeps_task(tmp_path, task, expected):
    """The saved init checkpoint must reload with the right task, or a seg ATT-YOLO would be
    trained with the detect trainer (regression)."""
    torch.manual_seed(0)
    torch.save({"model": DetectionModel("yolov5n.yaml", verbose=False)}, tmp_path / "yolov5nu.pt")
    yolo, report = model_zoo.make_model("att-yolo-n", task, nc=2, channels=3, run_dir=tmp_path / "run",
                                        weights_dir=tmp_path, pretrained=True)
    assert report.layers == list(BACKBONE_TRANSFER_LAYERS)
    assert yolo.task == expected
    from ultralytics import YOLO

    reloaded = YOLO(str(tmp_path / "run" / f"att-yolo-n-{task}-init.pt"))
    assert reloaded.task == expected
