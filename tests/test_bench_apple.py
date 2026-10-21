"""tools/bench_apple.py and the MLX YOLO11 port (PROMPTS step 22, PRD 5.2).

The MLX network must compute the same numbers as PyTorch before its speed means anything. MLX has
a CPU build for Linux, so outputs, the loss and gradients are compared here; speed is measured on
a real Mac (docs/mac_checklist.md). Skipped where MLX is not installed (pip install -e ".[bench]").
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bench_apple as B  # noqa: E402

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
import mlx_yolo as MY  # noqa: E402


def _torch_model(scale="n", seed=0):
    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    model = YOLO(f"yolo11{scale}.yaml", task="detect").model.float()
    for m in model.modules():  # BN statistics that are not the identity
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.1, 0.1)
    return model


def _nhwc(t):
    return mx.array(t.numpy().transpose(0, 2, 3, 1))


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


@pytest.mark.parametrize("train", [False, True])
def test_outputs_match_pytorch(train):
    import torch

    tm = _torch_model()
    tm.train(train)
    x = torch.rand(2, 3, 96, 96)
    with torch.no_grad():
        out = tm(x)
    if not train:  # eval mode returns (decoded, raw dict)
        out = out[1]
    boxes, scores = MY.YOLO(tm)(_nhwc(x), train=train)
    assert _rel(boxes, out["boxes"].transpose(1, 2).numpy()) < 1e-4
    assert _rel(scores, out["scores"].transpose(1, 2).numpy()) < 1e-4
    assert boxes.shape == (2, (12 * 12 + 6 * 6 + 3 * 3), 64) and scores.shape[-1] == 80


def test_loss_and_gradients_match_pytorch():
    import torch

    tm = _torch_model(seed=1)
    tm.train()
    for p in tm.parameters():
        p.requires_grad_(True)
    mm = MY.YOLO(tm)
    x = torch.rand(2, 3, 64, 64)
    n = 8 * 8 + 4 * 4 + 2 * 2
    bt, ct = torch.randn(2, n, 64), (torch.rand(2, n, 80) > 0.9).float()
    tm.zero_grad()
    lt = MY.surrogate_loss_torch(tm, x, bt, ct)
    lt.backward()
    lm, grads = nn.value_and_grad(mm, MY.surrogate_loss_mlx)(mm, _nhwc(x), mx.array(bt.numpy()), mx.array(ct.numpy()))
    assert float(lm) == pytest.approx(float(lt.detach()), rel=1e-4)
    pairs = [  # first conv, attention (C2PSA) BN, a DWConv in the class head, the last box conv
        (tm.model[0].conv.weight.grad, grads["layers"][0]["weight"], True),
        (tm.model[10].m[0].attn.qkv.bn.weight.grad, grads["layers"][10]["m"][0]["attn"]["qkv"]["gamma"], False),
        (tm.model[23].cv3[1][0][0].conv.weight.grad, grads["layers"][23]["cls"][1][0]["weight"], True),
        (tm.model[23].cv2[2][2].weight.grad, grads["layers"][23]["box"][2][-1]["weight"], True),
    ]
    for g_torch, g_mlx, conv in pairs:
        g = g_torch.numpy().transpose(0, 2, 3, 1) if conv else g_torch.numpy()
        # float32 through ~100 layers forward and back, summed in a different order: the first
        # layer's gradient differs by ~0.1 % (3 significant digits agree); a wrong port is off by far more
        assert _rel(g_mlx, g) < 5e-3


def test_all_yolo11_scales_convert():
    for scale in ("s", "m"):
        tm = _torch_model(scale)
        tm.eval()
        boxes, _ = MY.YOLO(tm)(mx.random.uniform(shape=(1, 64, 64, 3)), train=False)
        assert boxes.shape == (1, 8 * 8 + 4 * 4 + 2 * 2, 64)


def test_unsupported_model_is_refused():
    from defect_inspection.core.att_yolo import att_yolo_config
    from ultralytics.nn.tasks import DetectionModel

    with pytest.raises(NotImplementedError):
        MY.YOLO(DetectionModel(att_yolo_config("detect", 1, 3, "n"), verbose=False))  # C3TR not ported


# ---- benchmark script ---------------------------------------------------------------------------

def test_step_benchmark_runs():
    step = B.bench_step(batch=1, imgsz=64, steps=1)
    assert step["torch_cpu"] > 0 and step["mlx"] > 0 and step["mlx_compiled"] > 0


@pytest.mark.parametrize("mps, mlx, expect", [
    (300.0, 150.0, "계획"),  # 2x faster -> plan the engine
    (300.0, 250.0, "진행하지 않음"),  # 1.2x -> stay on MPS
    (None, 150.0, "판단 불가"),
])
def test_decision_rule(mps, mlx, expect):
    text, ratio = B.decide({"torch_mps": mps, "mlx": mlx, "mlx_compiled": None})
    assert expect in text
    assert ratio == (None if mps is None else pytest.approx(mps / mlx))


def test_report_format():
    result = {"environment": {"os": "Darwin 24", "machine": "arm64", "chip": "Apple M1", "python": "3.12",
                              "torch": "2.5", "mps": True, "mlx": "0.32", "mlx_device": "gpu"},
              "training": {"images": 64, "imgsz": 640, "batch": 16, "cpu_s_per_epoch": 90.0, "mps_s_per_epoch": 30.0},
              "infer_imgsz": 640, "inference": {"torch_cpu": 80.0, "torch_mps": 12.0, "coreml": 5.0},
              "step": {"batch": 8, "imgsz": 640, "torch_cpu": 4000.0, "torch_mps": 600.0, "mlx": 350.0}}
    result["decision"], _ = B.decide(result["step"])
    text = B.format_report(result)
    assert "Apple M1" in text and "MPS가 CPU보다 3.00배" in text and "| CoreML .mlpackage | 5.0 |" in text
    assert "1.71배" in text and "계획" in text
