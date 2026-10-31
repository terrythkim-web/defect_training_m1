"""U-Net++ semantic segmentation (segmentation_models_pytorch): training in the worker process,
prediction for the Result tab and suggestions. Qt-free; torch/smp are imported on first use.

Data: the dataset build's `masks/{train,val}/<stem>.png` class maps (0 = background, k = class k-1,
`dataset.label_map`) next to `images/`. Every image is resized so its long side is `imgsz` and padded
(bottom/right) to imgsz x imgsz; the padding is ignored by the loss and the metrics (IGNORE).
Colour images go in as RGB (ImageNet encoders), gray as one channel; ImageNet mean/std.

Checkpoints (`train/weights/{best,last}.pt`, like Ultralytics runs) are dicts with
format = "defectlab-unetpp", the encoder, nc, channels, imgsz, names and the state_dict; last.pt also
holds the optimizer/scheduler for resume."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from defect_inspection.platform_utils import imread_unicode, resolve_device

CHECKPOINT_FORMAT = "defectlab-unetpp"
ENCODERS = {"r18": "resnet18", "r34": "resnet34"}
IGNORE = 255
MEAN_RGB, STD_RGB = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN_GRAY, STD_GRAY = 0.449, 0.226
DEFAULT_LR = 3e-4
BLOB_HIT = 0.5  # a ground-truth defect region counts as found when this share of it is predicted
MIN_BLOB_PX = 4  # predicted regions smaller than this are noise, not a defect


def input_size(imgsz: int) -> int:
    """Network input side: a multiple of 32 (five encoder stages)."""
    return max(32, int(math.ceil(imgsz / 32)) * 32)


def torch_device(requested: str, purpose: str) -> str:
    dev = resolve_device(requested, purpose)
    return f"cuda:{dev.split(',')[0]}" if dev[:1].isdigit() else dev


# ---- pre-processing ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Letterbox:
    """How an image was fitted into the square input: scaled by `scale`, content `w` x `h` at the top-left."""

    size: int
    scale: float
    w: int
    h: int


def letterbox_geometry(width: int, height: int, size: int) -> Letterbox:
    scale = size / max(width, height)
    return Letterbox(size, scale, max(1, round(width * scale)), max(1, round(height * scale)))


def to_model_pixels(image8: np.ndarray, channels: int) -> np.ndarray:
    """uint8 BGR/gray (OpenCV, as datasets and `load_training_image` give) → HxW (gray) or HxWx3 RGB."""
    if channels == 1:
        return image8 if image8.ndim == 2 else cv2.cvtColor(image8, cv2.COLOR_BGR2GRAY)
    if image8.ndim == 2:
        return cv2.cvtColor(image8, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image8, cv2.COLOR_BGR2RGB)


def prepare_image(image8: np.ndarray, channels: int, size: int) -> tuple[np.ndarray, Letterbox]:
    """Network input (C, size, size) float32, normalized, zero-padded; and its geometry."""
    pixels = to_model_pixels(image8, channels)
    h, w = pixels.shape[:2]
    box = letterbox_geometry(w, h, size)
    interp = cv2.INTER_AREA if box.scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(pixels, (box.w, box.h), interpolation=interp).astype(np.float32) / 255.0
    if channels == 1:
        norm = ((resized - MEAN_GRAY) / STD_GRAY)[None]
    else:
        norm = ((resized - np.array(MEAN_RGB, np.float32)) / np.array(STD_RGB, np.float32)).transpose(2, 0, 1)
    out = np.zeros((channels, size, size), np.float32)
    out[:, :box.h, :box.w] = norm
    return out, box


def prepare_mask(mask: np.ndarray, box: Letterbox) -> np.ndarray:
    out = np.full((box.size, box.size), IGNORE, np.uint8)
    out[:box.h, :box.w] = cv2.resize(mask, (box.w, box.h), interpolation=cv2.INTER_NEAREST)
    return out


def dataset_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    """(image, mask PNG) pairs of one split; images without a mask are skipped."""
    pairs = []
    for img in sorted((dataset_root / "images" / split).iterdir()):
        mask = dataset_root / "masks" / split / f"{img.stem}.png"
        if mask.is_file():
            pairs.append((img, mask))
    return pairs


def has_masks(dataset_root: Path) -> bool:
    return (dataset_root / "masks" / "train").is_dir() and (dataset_root / "masks" / "val").is_dir()


def load_pair(image_path: Path, mask_path: Path, channels: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = imread_unicode(image_path)
    mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise OSError(f"학습 이미지/마스크를 읽을 수 없습니다: {image_path.name}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    x, box = prepare_image(image, channels, size)
    return x, prepare_mask(mask, box)


# ---- metrics --------------------------------------------------------------------------------------

class SegMeter:
    """Validation metrics over a split: IoU and Dice per defect class (summed over images, averaged over
    classes that occur), and blob recall = share of ground-truth defect regions found (≥ BLOB_HIT of
    their pixels predicted as their class)."""

    def __init__(self, nc: int) -> None:
        self.nc = nc
        self.inter = np.zeros(nc + 1)
        self.pred = np.zeros(nc + 1)
        self.true = np.zeros(nc + 1)
        self.blobs = 0
        self.found = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        valid = target != IGNORE
        for c in range(1, self.nc + 1):
            p, t = (pred == c) & valid, target == c
            self.inter[c] += np.count_nonzero(p & t)
            self.pred[c] += np.count_nonzero(p)
            self.true[c] += np.count_nonzero(t)
            if t.any():
                n, labels = cv2.connectedComponents(t.astype(np.uint8), connectivity=8)
                for k in range(1, n):
                    region = labels == k
                    self.blobs += 1
                    self.found += np.count_nonzero(p & region) >= BLOB_HIT * np.count_nonzero(region)

    def metrics(self) -> dict[str, float]:
        present = [c for c in range(1, self.nc + 1) if self.true[c] + self.pred[c] > 0]
        iou = [self.inter[c] / (self.true[c] + self.pred[c] - self.inter[c]) for c in present]
        dice = [2 * self.inter[c] / (self.true[c] + self.pred[c]) for c in present]
        return {"metrics/IoU": round(float(np.mean(iou)) if iou else 0.0, 5),
                "metrics/Dice": round(float(np.mean(dice)) if dice else 0.0, 5),
                "metrics/recall(blob)": round(self.found / self.blobs, 5) if self.blobs else 0.0}


# ---- model / checkpoints ---------------------------------------------------------------------------

def encoder_for(model_key: str) -> str:
    from defect_inspection.core.model_zoo import get_choice

    return ENCODERS[get_choice(model_key).scale]


def build_model(encoder: str, channels: int, nc: int, pretrained: bool, weights_dir: Path | None = None):
    """(model, warning or None). ImageNet encoder weights are downloaded once into weights_dir; without
    a network the model starts from scratch and the warning says so."""
    import os

    if weights_dir is not None:
        # smp fetches encoder weights from the Hugging Face hub / torch hub. huggingface_hub reads these
        # when it is first imported, so they must be set before importing smp (the worker imports it here
        # first); an explicit HF_HUB_CACHE also wins over a user's own HF_HOME.
        os.environ["HF_HUB_CACHE"] = str(Path(weights_dir) / "hf")
        os.environ.setdefault("TORCH_HOME", str(Path(weights_dir) / "torch"))
    import segmentation_models_pytorch as smp
    if pretrained:
        try:
            return smp.UnetPlusPlus(encoder, encoder_weights="imagenet", in_channels=channels, classes=nc + 1), None
        except Exception as e:  # noqa: BLE001 - offline line PCs: train from scratch instead
            warning = f"사전학습 인코더 가중치를 받지 못해 처음부터 학습합니다 ({type(e).__name__}: {e})"
            return smp.UnetPlusPlus(encoder, encoder_weights=None, in_channels=channels, classes=nc + 1), warning
    return smp.UnetPlusPlus(encoder, encoder_weights=None, in_channels=channels, classes=nc + 1), None


def checkpoint_meta(spec, size: int) -> dict:
    return {"format": CHECKPOINT_FORMAT, "encoder": encoder_for(spec.train.model), "nc": spec.nc,
            "channels": spec.channels, "imgsz": size, "names": list(spec.names), "model": spec.train.model}


def is_unet_checkpoint(path: Path) -> bool:
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(ckpt, dict) and ckpt.get("format") == CHECKPOINT_FORMAT


def load_model(path: Path, device: str = "cpu"):
    """(model in eval mode, checkpoint dict) from best.pt / last.pt."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"U-Net++ 체크포인트가 아닙니다: {path}")
    model, _ = build_model(ckpt["encoder"], ckpt["channels"], ckpt["nc"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# ---- training -------------------------------------------------------------------------------------

class SegDataset:
    """(image, mask) pairs as tensors. Module level: DataLoader workers on Windows/macOS (spawn) pickle
    the dataset, and a class defined inside a function cannot be pickled."""

    def __init__(self, pairs, channels: int, size: int, flip: bool) -> None:
        self.pairs, self.channels, self.size, self.flip = pairs, channels, size, flip

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        import torch

        x, y = load_pair(*self.pairs[i], self.channels, self.size)
        if self.flip and torch.rand(()) < 0.5:  # light online flip on top of the offline augmentation
            x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def _loader(pairs, channels: int, size: int, batch: int, shuffle: bool, workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    gen = torch.Generator().manual_seed(seed)
    # spawn, not fork: the worker process reads stdin on a thread (stop requests), and a forked loader
    # worker closing its copy of sys.stdin waits forever on the lock that thread holds — training hung
    # before the first epoch. Persistent, so the spawn start-up is paid once per run, not per epoch.
    extra = {"multiprocessing_context": "spawn", "persistent_workers": True} if workers > 0 else {}
    return DataLoader(SegDataset(pairs, channels, size, shuffle), batch_size=batch, shuffle=shuffle,
                      num_workers=workers, generator=gen, drop_last=False, **extra)


def auto_batch(device: str) -> int:
    return 8 if device.startswith("cuda") else 4


def _save(obj: dict, path: Path) -> None:
    """torch.save via a temp file: a process killed mid-write must not leave a truncated last.pt that
    still looks resumable."""
    import os

    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def evaluate(model, loader, nc: int, device: str, loss_fn=None, should_stop: Callable[[], bool] | None = None,
             stop_exc: type[BaseException] | None = None) -> tuple[dict[str, float], float | None]:
    import torch

    meter, total, n = SegMeter(nc), 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            if should_stop is not None and should_stop():
                raise stop_exc()
            logits = model(x.to(device))
            if loss_fn is not None:
                total += loss_fn(logits, y.to(device)).item() * len(x)
                n += len(x)
            pred = logits.argmax(1).cpu().numpy()
            for p, t in zip(pred, y.numpy()):
                meter.update(p, t)
    return meter.metrics(), (total / n if n else None)


def train_unet(spec, run_dir: Path, emit: Callable[[dict], None], should_stop: Callable[[], bool],
               resume: bool, stop_exc: type[BaseException]) -> None:
    """The worker's training loop for a U-Net++ run (same events and files as an Ultralytics run)."""
    import torch
    from segmentation_models_pytorch.losses import DiceLoss

    from defect_inspection.core.trainer import weights_paths, write_status

    cfg = spec.train
    root = Path(spec.data_yaml).parent
    if not has_masks(root):
        raise FileNotFoundError("데이터셋에 마스크(masks/)가 없습니다. Augment 탭에서 데이터셋을 다시 빌드하세요.")
    train_pairs, val_pairs = dataset_pairs(root, "train"), dataset_pairs(root, "val")
    if not train_pairs or not val_pairs:
        raise ValueError("U-Net++ 학습에는 Train과 Val 이미지가 모두 필요합니다.")
    device = torch_device(cfg.device, "train")
    size = input_size(cfg.imgsz)
    best_path, last_path = weights_paths(run_dir)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    model, warning = build_model(encoder_for(cfg.model), spec.channels, spec.nc,
                                 cfg.pretrained and not resume, Path(spec.weights_dir))
    if warning:
        print(warning, flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr0 or DEFAULT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    dice = DiceLoss(mode="multiclass", ignore_index=IGNORE, from_logits=True)

    def loss_fn(logits, target):
        return ce(logits, target) + dice(logits, target)

    start, best_fitness, best_epoch = 0, -1.0, 0
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError("재개할 last.pt가 없습니다")
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best_fitness, best_epoch = ckpt["epoch"], ckpt["best_fitness"], ckpt.get("best_epoch", 0)

    # seed + start: a resumed run must not replay epoch 1's shuffle and flips
    torch.manual_seed(cfg.seed + start)
    np.random.seed(cfg.seed + start)
    batch = cfg.batch if cfg.batch > 0 else auto_batch(device)
    if cfg.batch <= 0:
        print(f"자동 batch: {batch} (장치 {device})", flush=True)
    workers = min(cfg.workers, 4)
    train_loader = _loader(train_pairs, spec.channels, size, batch, True, workers, cfg.seed + start)
    val_loader = _loader(val_pairs, spec.channels, size, batch, False, workers, cfg.seed)
    emit({"type": "model", "channels": spec.channels, "task": spec.task,
          "params": int(sum(p.numel() for p in model.parameters()))})
    meta = checkpoint_meta(spec, size)
    t0 = time.time()
    for epoch in range(start + 1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            if should_stop():
                raise stop_exc()
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        scheduler.step()
        train_loss = round(total / max(n, 1), 6)
        emit({"type": "train_epoch", "epoch": epoch, "epochs": cfg.epochs, "losses": {"train/seg_loss": train_loss}})

        model.eval()
        metrics, val_loss = evaluate(model, val_loader, spec.nc, device, loss_fn, should_stop, stop_exc)
        metrics["val/seg_loss"] = round(val_loss or 0.0, 6)
        fitness = metrics["metrics/IoU"]
        if fitness > best_fitness:
            best_fitness, best_epoch = fitness, epoch
            _save({**meta, "state_dict": model.state_dict(), "epoch": epoch, "best_fitness": best_fitness},
                  best_path)
        _save({**meta, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_fitness": best_fitness,
                    "best_epoch": best_epoch}, last_path)
        elapsed = time.time() - t0
        done_epochs = epoch - start
        emit({"type": "epoch", "epoch": epoch, "epochs": cfg.epochs, "losses": {"train/seg_loss": train_loss},
              "metrics": metrics, "fitness": fitness, "best_fitness": best_fitness,
              "lr": {"lr/pg0": round(optimizer.param_groups[0]["lr"], 8)}, "elapsed_s": round(elapsed, 2),
              "eta_s": round(elapsed / done_epochs * (cfg.epochs - epoch), 2)})
        write_status(run_dir, {"epoch": epoch, "epochs": cfg.epochs, "best_fitness": best_fitness,
                               "metrics": metrics})
        if cfg.patience and epoch - best_epoch >= cfg.patience:
            print(f"{cfg.patience} epoch 동안 개선이 없어 조기 종료합니다 (최고 epoch {best_epoch})", flush=True)
            break

    best_model, _ = load_model(best_path, device)
    final, _ = evaluate(best_model, val_loader, spec.nc, device, None, should_stop, stop_exc)
    emit({"type": "final", "metrics": final, "fitness": final["metrics/IoU"]})
    write_status(run_dir, {"final_metrics": final})


# ---- prediction -----------------------------------------------------------------------------------

class UnetPredictor:
    """Same interface as evaluate.Predictor: `predict(image8, conf, iou)` → Predictions, one Mask per
    connected region of a defect class (argmax), scored by the region's mean class probability;
    regions below `conf` are dropped. `iou` is unused (no NMS)."""

    def __init__(self, weights: str | Path, device: str | None = None) -> None:
        self.weights = str(weights)
        self.device = torch_device(device or "auto", "predict")
        self.model, ckpt = load_model(Path(weights), self.device)
        self.channels, self.nc, self.size = ckpt["channels"], ckpt["nc"], ckpt["imgsz"]
        self.task = "semseg"

    def probabilities(self, image8: np.ndarray) -> np.ndarray:
        """(nc + 1, h, w) softmax probabilities at NETWORK resolution (the letterboxed content)."""
        import torch

        x, box = prepare_image(image8, self.channels, self.size)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x)[None].to(self.device))
            return torch.softmax(logits, 1)[0, :, :box.h, :box.w].cpu().numpy()

    def predict(self, image8: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> list:
        """Regions are found at network resolution and scored there; only the uint8 label map is
        resized to the image (nearest) — full-size float maps per class took ~1.5 GB on an 8K image."""
        prob = self.probabilities(image8)
        h, w = image8.shape[:2]
        label = cv2.resize(prob.argmax(0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return label_map_to_predictions(label, prob, conf)


def probabilities_to_predictions(prob: np.ndarray, conf: float) -> list:
    """Predictions from (nc + 1, H, W) probabilities at the image's size."""
    return label_map_to_predictions(prob.argmax(0).astype(np.uint8), prob, conf)


def label_map_to_predictions(label: np.ndarray, prob: np.ndarray, conf: float) -> list:
    """One Mask per connected region of a defect class in `label` (image size). Its score is the mean
    class probability over the region, read from `prob` (any resolution: regions are mapped onto it)."""
    from defect_inspection.core.annotation import Mask
    from defect_inspection.core.evaluate import Prediction

    h, w = label.shape
    ph, pw = prob.shape[1:]
    sx, sy = pw / w, ph / h
    out = []
    for c in range(1, prob.shape[0]):
        cls = label == c
        if not cls.any():
            continue
        n, regions, stats, _ = cv2.connectedComponentsWithStats(cls.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < MIN_BLOB_PX:
                continue
            x, y, bw, bh = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                       cv2.CC_STAT_HEIGHT))
            region = regions[y:y + bh, x:x + bw] == k
            ys, xs = np.nonzero(region)
            py = np.minimum(((ys + y + 0.5) * sy).astype(int), ph - 1)
            px = np.minimum(((xs + x + 0.5) * sx).astype(int), pw - 1)
            score = float(prob[c, py, px].mean())
            if score < conf:
                continue
            shape = Mask.from_pixels(region, origin=(x, y), class_id=c - 1, status="suggested", source="model",
                                     score=min(max(score, 0.0), 1.0))
            out.append(Prediction(shape))
    out.sort(key=lambda p: -p.score)
    return out


# ---- ONNX export ----------------------------------------------------------------------------------

VERIFY_AGREEMENT = 0.995  # share of content pixels whose argmax class must match PyTorch


def export_onnx(weights: Path, out_path: Path, opset: int | None = None) -> dict:
    """best.pt → .onnx with softmax probabilities (1, nc + 1, S, S) for a (1, C, S, S) input; returns the
    checkpoint meta (encoder, nc, channels, imgsz). Exported on the CPU in float32."""
    import torch

    model, ckpt = load_model(weights, "cpu")

    class WithSoftmax(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.softmax(self.inner(x), dim=1)

    size, channels = ckpt["imgsz"], ckpt["channels"]
    dummy = torch.zeros(1, channels, size, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["images"], output_names=["probabilities"], opset_version=opset or 17)
    try:  # torch >= 2.5 defaults towards the dynamo exporter; the classic one is what we verify
        torch.onnx.export(WithSoftmax(model).eval(), dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:  # older torch: no `dynamo` keyword
        torch.onnx.export(WithSoftmax(model).eval(), dummy, str(out_path), **kwargs)
    return {k: ckpt[k] for k in ("encoder", "nc", "channels", "imgsz", "names")}


def verify_onnx(onnx_path: Path, weights: Path, images: list[np.ndarray]):
    """PyTorch vs ONNX Runtime on the same inputs: argmax agreement over the image content."""
    import onnxruntime as ort
    import torch

    from defect_inspection.core.exporter import Verification

    model, ckpt = load_model(weights, "cpu")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    agree = total = 0
    max_diff = 0.0
    for image in images:
        x, box = prepare_image(image, ckpt["channels"], ckpt["imgsz"])
        with torch.no_grad():
            ref = torch.softmax(model(torch.from_numpy(x)[None]), 1)[0, :, :box.h, :box.w].numpy()
        got = session.run(None, {inp.name: x[None]})[0][0, :, :box.h, :box.w]
        agree += int((ref.argmax(0) == got.argmax(0)).sum())
        total += ref.shape[1] * ref.shape[2]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    share = agree / total if total else 0.0
    return Verification(
        passed=share >= VERIFY_AGREEMENT, images=len(images), max_score_diff=round(max_diff, 6),
        input_name=inp.name, input_shape=list(inp.shape), input_channels=int(inp.shape[1]),
        output_shapes=[list(o.shape) for o in session.get_outputs()],
        message=f"픽셀 클래스 일치율 {share * 100:.2f}% (기준 {VERIFY_AGREEMENT * 100:.1f}%)")


def onnx_metadata(meta: dict) -> dict:
    """Input / pre-processing / output description for an exported U-Net++ (metadata.json)."""
    size, channels = meta["imgsz"], meta["channels"]
    mean = [MEAN_GRAY] if channels == 1 else list(MEAN_RGB)
    std = [STD_GRAY] if channels == 1 else list(STD_RGB)
    return {
        "input": {"shape": [1, channels, size, size], "channels": channels,
                  "color_order": "GRAY" if channels == 1 else "RGB", "dtype": "float32", "dynamic": False},
        "preprocess": {
            "source_bit_depth": "8bit 그대로, 16bit는 값 >> 8 (학습과 동일)",
            "color": "1채널 그레이" if channels == 1 else "BGR 이미지라면 RGB로 바꿔서 입력",
            "resize": {"size": size, "keep_ratio": True, "long_side": size, "pad": "오른쪽·아래쪽을 0으로 (정규화 후)",
                       "center": False},
            "normalize": {"divide": 255.0, "mean": mean, "std": std, "formula": "(x / 255 - mean) / std"},
            "layout": "NCHW",
        },
        "output": (f"(1, {meta['nc'] + 1}, {size}, {size}) 클래스별 확률(softmax). 채널 0 = 배경, k = 클래스 k-1. "
                   "패딩을 잘라낸 뒤 원본 크기로 늘리고 argmax → 같은 클래스의 연결된 덩어리 = 결함 하나"),
        "recommended_thresholds": {"conf": 0.25, "min_region_px": MIN_BLOB_PX},
        "encoder": meta["encoder"],
    }
