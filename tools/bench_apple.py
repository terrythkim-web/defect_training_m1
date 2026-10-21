"""Apple Silicon acceleration benchmark (PRD 5.2, PROMPTS step 22).

    python tools/bench_apple.py                 # full run on a Mac (~10-20 min)
    python tools/bench_apple.py --quick         # small sizes, a few minutes (also works on Linux)
    python tools/bench_apple.py --json result.json

Measures on the same data:
  1. training: seconds per epoch of Ultralytics YOLO11n on a synthetic dataset, CPU vs MPS
  2. inference: ms per image — PyTorch CPU / MPS, ONNX Runtime (CPU and CoreML providers),
     CoreML .mlpackage (macOS only)
  3. MLX micro-benchmark: ONE training step (forward + backward + SGD update) of the YOLO11n
     network with the same surrogate loss in MLX and in PyTorch (CPU / MPS). The MLX network is
     tools/mlx_yolo.py, numerically checked against PyTorch in tests/test_bench_apple.py.

Decision rule (PRD 5.2): plan an MLX training engine only if the MLX step is >= 1.5x faster than
the MPS step. Anything missing on this machine (no MPS on Linux, no CoreML outside macOS) is
reported as "-", and the decision is then "판단 불가".
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import defect_inspection  # noqa: E402,F401  (sets YOLO_AUTOINSTALL=false, PYTORCH_ENABLE_MPS_FALLBACK=1)

MLX_THRESHOLD = 1.5


# ---- environment -----------------------------------------------------------------------------

def environment() -> dict:
    import torch

    info = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "cuda": torch.cuda.is_available()}
    if sys.platform == "darwin":
        try:
            info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                          text=True, timeout=5).stdout.strip()
        except OSError:
            pass
    try:
        import mlx.core as mx

        info["mlx"] = mx.__version__
        info["mlx_device"] = str(mx.default_device())
    except ImportError:
        info["mlx"] = None
    return info


def torch_devices() -> list[str]:
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _median_ms(fn, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# ---- 3. one training step: MLX vs PyTorch ---------------------------------------------------

def make_step_inputs(batch: int, imgsz: int, nc: int = 80, seed: int = 0):
    """(torch model on CPU, images NCHW, box target, cls target) with targets shaped like the head."""
    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    model = YOLO("yolo11n.yaml", task="detect").model.float()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    images = torch.rand(batch, 3, imgsz, imgsz)
    n = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    box_t = torch.randn(batch, n, 64)
    cls_t = (torch.rand(batch, n, nc) > 0.95).float()
    return model, images, box_t, cls_t


def torch_step_ms(device: str, batch: int, imgsz: int, steps: int) -> float:
    import torch

    from mlx_yolo import surrogate_loss_torch

    model, images, box_t, cls_t = make_step_inputs(batch, imgsz)
    model.to(device)
    images, box_t, cls_t = images.to(device), box_t.to(device), cls_t.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = surrogate_loss_torch(model, images, box_t, cls_t)
        loss.backward()
        opt.step()
        _sync(device)

    return _median_ms(step, steps)


def mlx_step_ms(batch: int, imgsz: int, steps: int, compiled: bool = False) -> float:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from mlx_yolo import YOLO, surrogate_loss_mlx

    tmodel, images, box_t, cls_t = make_step_inputs(batch, imgsz)
    model = YOLO(tmodel)
    x = mx.array(images.numpy().transpose(0, 2, 3, 1))
    bt, ct = mx.array(box_t.numpy()), mx.array(cls_t.numpy())
    opt = optim.SGD(learning_rate=1e-3, momentum=0.9)
    loss_and_grad = nn.value_and_grad(model, surrogate_loss_mlx)

    def raw_step(x, bt, ct):
        loss, grads = loss_and_grad(model, x, bt, ct)
        opt.update(model, grads)
        return loss

    state = [model.state, opt.state]
    fn = mx.compile(raw_step, inputs=state, outputs=state) if compiled else raw_step

    def step():
        loss = fn(x, bt, ct)
        mx.eval(loss, model.parameters(), opt.state)

    return _median_ms(step, steps)


def bench_step(batch: int, imgsz: int, steps: int) -> dict:
    """ms per training step for every available backend."""
    out: dict = {"batch": batch, "imgsz": imgsz}
    for device in torch_devices():
        out[f"torch_{device}"] = round(torch_step_ms(device, batch, imgsz, steps), 1)
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        out["mlx"] = None
        out["mlx_compiled"] = None
        return out
    out["mlx"] = round(mlx_step_ms(batch, imgsz, steps), 1)
    out["mlx_compiled"] = round(mlx_step_ms(batch, imgsz, steps, compiled=True), 1)
    return out


def decide(step: dict, threshold: float = MLX_THRESHOLD) -> tuple[str, float | None]:
    """(decision text, MPS/MLX speed ratio). Ratio > 1 means MLX is faster."""
    mps = step.get("torch_mps")
    mlx_times = [v for v in (step.get("mlx"), step.get("mlx_compiled")) if v]
    if not mps or not mlx_times:
        missing = "MPS" if not mps else "MLX"
        return f"판단 불가: 이 PC에 {missing}가 없습니다 (Apple Silicon Mac에서 실행하세요)", None
    ratio = mps / min(mlx_times)
    if ratio >= threshold:
        return (f"MLX가 MPS보다 {ratio:.2f}배 빠름 (기준 {threshold}배 이상) → MLX 학습 엔진을 별도 단계로 계획", ratio)
    return f"MLX가 MPS 대비 {ratio:.2f}배 (기준 {threshold}배 미만) → MPS 유지, MLX 학습 엔진은 진행하지 않음", ratio


# ---- 2. inference -------------------------------------------------------------------------------

def bench_inference(imgsz: int, repeats: int, workdir: Path) -> dict:
    """Raw network time per image (preprocessed input -> network outputs, no NMS) for every backend,
    so PyTorch, ONNX Runtime and CoreML are compared like for like."""
    import os

    import torch
    from ultralytics import YOLO

    out: dict = {}
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
    pt = YOLO("yolo11n.yaml", task="detect")
    net = pt.model.fuse().eval()
    for device in torch_devices():
        model, xt = net.to(device), torch.from_numpy(x).to(device)

        def run(m=model, t=xt, d=device):
            with torch.inference_mode():
                m(t)
            _sync(d)

        out[f"torch_{device}"] = round(_median_ms(run, repeats), 2)
    net.to("cpu")
    cwd = os.getcwd()
    os.chdir(workdir)  # a YAML-built model is exported into the working directory
    try:
        onnx = Path(pt.export(format="onnx", imgsz=imgsz, simplify=True, device="cpu", verbose=False)).resolve()
        import onnxruntime as ort

        for name, provider in (("onnxruntime_cpu", "CPUExecutionProvider"), ("onnxruntime_coreml", "CoreMLExecutionProvider")):
            if provider not in ort.get_available_providers():
                out[name] = None
                continue
            sess = ort.InferenceSession(str(onnx), providers=[provider])
            feed = {sess.get_inputs()[0].name: x}
            out[name] = round(_median_ms(lambda s=sess, f=feed: s.run(None, f), repeats), 2)
        out["coreml"] = None
        if sys.platform == "darwin":
            try:
                import coremltools as ct
                from PIL import Image

                package = Path(pt.export(format="coreml", imgsz=imgsz, device="cpu", verbose=False)).resolve()
                model = ct.models.MLModel(str(package))  # Neural Engine / GPU / CPU chosen by CoreML
                name = model.get_spec().description.input[0].name
                pil = Image.fromarray((x[0].transpose(1, 2, 0) * 255).astype(np.uint8))
                out["coreml"] = round(_median_ms(lambda: model.predict({name: pil}), repeats), 2)
            except Exception as e:  # noqa: BLE001 - reported, the rest of the benchmark still runs
                out["coreml_error"] = f"{type(e).__name__}: {e}"
    finally:
        os.chdir(cwd)
    return out


# ---- 1. training epochs -----------------------------------------------------------------------------

def synthetic_dataset(root: Path, n: int, imgsz: int) -> Path:
    """n images with dark rectangles + YOLO labels; returns data.yaml."""
    import cv2
    import yaml

    rng = np.random.default_rng(0)
    for split, count in (("train", n), ("val", max(2, n // 4))):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(count):
            img = (rng.normal(170, 10, (imgsz, imgsz, 3))).clip(0, 255).astype(np.uint8)
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                w, h = (int(v) for v in rng.integers(imgsz // 16, imgsz // 5, 2))
                x, y = int(rng.integers(0, imgsz - w)), int(rng.integers(0, imgsz - h))
                img[y:y + h, x:x + w] = 40
                lines.append(f"0 {(x + w / 2) / imgsz:.6f} {(y + h / 2) / imgsz:.6f} {w / imgsz:.6f} {h / imgsz:.6f}")
            cv2.imwrite(str(root / "images" / split / f"{i}.png"), img)
            (root / "labels" / split / f"{i}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                    "names": {0: "defect"}}), encoding="utf-8")
    return data


def train_epoch_seconds(data: Path, device: str, epochs: int, imgsz: int, batch: int, workdir: Path) -> float:
    """Median seconds of the epochs after the first (the first includes warm-up and caching)."""
    from ultralytics import YOLO

    stamps: list[float] = []
    model = YOLO("yolo11n.yaml", task="detect")
    model.add_callback("on_train_epoch_start", lambda trainer: stamps.append(time.perf_counter()))
    model.add_callback("on_train_epoch_end", lambda trainer: stamps.append(time.perf_counter()))
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=0,
                project=str(workdir), name=f"train_{device}", exist_ok=True, plots=False, val=False,
                verbose=False, amp=False, pretrained=False)
    durations = [stamps[i + 1] - stamps[i] for i in range(0, len(stamps) - 1, 2)]
    return statistics.median(durations[1:] if len(durations) > 1 else durations)


def bench_training(n: int, imgsz: int, epochs: int, batch: int, workdir: Path) -> dict:
    data = synthetic_dataset(workdir / "data", n, imgsz)
    out: dict = {"images": n, "imgsz": imgsz, "batch": batch}
    for device in torch_devices():
        out[f"{device}_s_per_epoch"] = round(train_epoch_seconds(data, device, epochs, imgsz, batch, workdir), 2)
    return out


# ---- report ---------------------------------------------------------------------------------------

def _cell(v) -> str:
    return "-" if v is None else str(v)


def format_report(result: dict) -> str:
    env = result["environment"]
    lines = [f"# Apple Silicon 가속 측정 ({env.get('chip', env['machine'])}, {env['os']})", "",
             f"Python {env['python']} · torch {env['torch']} · MPS {'있음' if env['mps'] else '없음'}"
             f" · MLX {env.get('mlx') or '없음'} ({env.get('mlx_device', '-')})", ""]
    if "training" in result:
        t = result["training"]
        lines += [f"## 1. 학습 (YOLO11n, 이미지 {t['images']}장, {t['imgsz']}px, batch {t['batch']})", "",
                  "| 장치 | epoch당 초 |", "| --- | --- |"]
        lines += [f"| {k.split('_')[0].upper()} | {v} |" for k, v in t.items() if k.endswith("_s_per_epoch")]
        cpu, mps = t.get("cpu_s_per_epoch"), t.get("mps_s_per_epoch")
        if cpu and mps:
            lines += ["", f"MPS가 CPU보다 {cpu / mps:.2f}배 빠름"]
        lines.append("")
    if "inference" in result:
        i = result["inference"]
        names = [("torch_cpu", "PyTorch CPU"), ("torch_mps", "PyTorch MPS"), ("torch_cuda", "PyTorch CUDA"),
                 ("onnxruntime_cpu", "ONNX Runtime CPU"), ("onnxruntime_coreml", "ONNX Runtime CoreML"),
                 ("coreml", "CoreML .mlpackage")]
        lines += [f"## 2. 추론 (1장, {result.get('infer_imgsz')}px, 네트워크만 — 전처리·NMS 제외, 중앙값)", "",
                  "| 방식 | ms/장 |", "| --- | --- |"]
        lines += [f"| {label} | {_cell(i.get(k))} |" for k, label in names if k in i]
        if i.get("coreml_error"):
            lines.append(f"\nCoreML 오류: {i['coreml_error']}")
        lines.append("")
    s = result["step"]
    lines += [f"## 3. 학습 1 step (forward + backward + SGD, batch {s['batch']}, {s['imgsz']}px)", "",
              "| 구현 | ms/step |", "| --- | --- |"]
    for k, label in (("torch_cpu", "PyTorch CPU"), ("torch_mps", "PyTorch MPS"), ("torch_cuda", "PyTorch CUDA"),
                     ("mlx", "MLX"), ("mlx_compiled", "MLX (mx.compile)")):
        if k in s:
            lines.append(f"| {label} | {_cell(s[k])} |")
    lines += ["", f"**판단**: {result['decision']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="작은 크기로 빠르게 (동작 확인용)")
    ap.add_argument("--skip-train", action="store_true", help="1. 학습 epoch 측정 생략")
    ap.add_argument("--skip-infer", action="store_true", help="2. 추론 측정 생략")
    ap.add_argument("--json", type=Path, help="결과를 JSON으로도 저장")
    args = ap.parse_args(argv)

    q = args.quick
    step_cfg = dict(batch=2 if q else 8, imgsz=128 if q else 640, steps=3 if q else 10)
    result: dict = {"environment": environment(), "quick": q}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if not args.skip_train:
            print("1/3 학습 epoch 측정 중…", flush=True)
            result["training"] = bench_training(n=16 if q else 64, imgsz=160 if q else 640, epochs=2 if q else 3,
                                                batch=4 if q else 16, workdir=work)
        if not args.skip_infer:
            print("2/3 추론 측정 중…", flush=True)
            result["infer_imgsz"] = 160 if q else 640
            result["inference"] = bench_inference(result["infer_imgsz"], 5 if q else 30, work)
        print("3/3 학습 1 step (MLX vs PyTorch) 측정 중…", flush=True)
        result["step"] = bench_step(**step_cfg)
    result["decision"], result["mps_over_mlx"] = decide(result["step"])
    report = format_report(result)
    print("\n" + report)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
