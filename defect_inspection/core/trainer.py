"""Training runs (PRD F4.1-F4.4, F4.7): run folders, configuration, event protocol, history.

A run lives in runs/<run_id>/:
  config.yaml   what to train (TrainConfig + resolved paths), written before the worker starts
  status.json   current state, updated by the worker (queued/running/finished/stopped/failed)
  events.jsonl  every event the worker emitted (one JSON object per line)
  train.log     the Train tab's log, one section per attempt (start, then each resume)
  train/        Ultralytics output (weights/best.pt, weights/last.pt, results.csv, plots)

The worker (defect_inspection.worker) runs in a separate process. It prints each event to stdout
as one line `@@DL {json}` so the UI can tell events from Ultralytics' own log output.

Stopping: a stop request ends training after the current batch WITHOUT Ultralytics' final step,
which would strip the optimizer from last.pt and make the run impossible to resume.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from defect_inspection.core.crop_config import TilingParams
from defect_inspection.core.model_zoo import DEFAULT_MODEL, PAPER_HYP, get_choice
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import (
    atomic_write_text,
    default_dataloader_workers,
    resolve_device,
    total_memory_bytes,
)

EVENT_PREFIX = "@@DL "
LOG_FILE = "train.log"
RunState = Literal["queued", "running", "finished", "stopped", "failed"]

# Ultralytics' built-in (online) augmentation, all switched off when online_augment=False.
ONLINE_AUGMENT_OFF = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0,
    "copy_paste": 0.0, "erasing": 0.0,
}


class TrainConfig(BaseModel):
    model: str = DEFAULT_MODEL
    dataset: Literal["full", "crop"] = "full"
    epochs: int = Field(default=100, ge=1, le=5000)
    imgsz: int = Field(default=640, ge=32, le=4096)
    batch: int = Field(default=-1, ge=-1, le=1024)  # -1 or 0 = auto (auto_batch), else fixed
    lr0: float | None = Field(default=None, gt=0, le=1)  # None = Ultralytics default
    patience: int = Field(default=50, ge=0)  # early stopping, 0 = off
    device: str = "auto"  # "auto" (CUDA GPU, Apple MPS or CPU), "cpu", "0", "0,1", "mps"
    workers: int = Field(default_factory=default_dataloader_workers, ge=0, le=64)
    online_augment: bool = True  # Ultralytics' own augmentation on top of the offline dataset
    paper_hyp: bool = False  # ATT-YOLO paper hyperparameters (Table 6)
    pretrained: bool = True
    seed: int = 0

    @classmethod
    def for_model(cls, model: str, **overrides: Any) -> TrainConfig:
        """Defaults of a model choice (e.g. ATT-YOLO: imgsz 1280, 300 epochs) plus overrides."""
        choice = get_choice(model)
        values = {"model": model, "imgsz": choice.default_imgsz, "epochs": choice.default_epochs}
        values.update(overrides)
        return cls(**values)


class RunSpec(BaseModel):
    """config.yaml: everything the worker needs, resolved when the run is prepared."""

    run_id: str
    project_root: str
    task: Literal["detect", "seg"]
    nc: int
    names: list[str]
    channels: int
    data_yaml: str
    weights_dir: str
    train: TrainConfig
    created: str
    # crop runs: how to tile whole images for inference (F7.6), frozen when the run is prepared
    # because the crop dataset may be rebuilt with other settings later
    crop: TilingParams | None = None

    def tiling(self) -> TilingParams | None:
        if self.train.dataset != "crop":
            return None
        return self.crop or TilingParams.from_window(self.train.imgsz, None)  # runs from before step 18


# Batch that trained without swapping on an 8 GB M1 at imgsz 640 (yolo11s at 16 thrashed the swap
# for minutes per epoch; 8 once took 680 s for the first epoch). Scaled by memory and imgsz².
_BATCH_8GB_640 = {"n": 8, "s": 4, "m": 2}
AUTO_BATCH_MAX = 16  # Ultralytics' own default when AutoBatch cannot measure the device


def auto_batch(model: str, imgsz: int, device: str, memory_bytes: int | None) -> int:
    """Batch for "자동". CUDA: -1, Ultralytics AutoBatch measures the GPU. Apple MPS / CPU:
    AutoBatch cannot measure them and falls back to 16, which overflows an 8 GB Mac, so estimate
    from total memory instead (never above 16, never below 1)."""
    if device not in ("mps", "cpu"):
        return -1
    if memory_bytes is None:
        return AUTO_BATCH_MAX
    choice = get_choice(model)
    base = _BATCH_8GB_640.get(choice.scale, 2)
    if choice.family != "yolo11":
        base = max(1, base // 2)  # ATT-YOLO: six heads incl. P2, about twice the activations
    estimate = base * (memory_bytes / 2**33) * (640 / imgsz) ** 2
    batch = 1
    while batch * 2 <= min(estimate, AUTO_BATCH_MAX):
        batch *= 2
    return batch


def train_args(spec: RunSpec, run_dir: Path) -> dict[str, Any]:
    """Keyword arguments for Ultralytics `YOLO.train()`."""
    cfg = spec.train
    args: dict[str, Any] = {
        "data": spec.data_yaml,
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "patience": cfg.patience,
        "workers": cfg.workers,
        "seed": cfg.seed,
        "project": str(run_dir),
        "name": "train",
        "exist_ok": True,
        "plots": True,
        "verbose": True,
        "pretrained": cfg.pretrained,
    }
    # Always explicit: Ultralytics never chooses MPS by itself (PRD 5.2). Resolved in the worker.
    args["device"] = resolve_device(cfg.device, "train")
    args["batch"] = cfg.batch if cfg.batch > 0 else auto_batch(cfg.model, cfg.imgsz, args["device"],
                                                               total_memory_bytes())
    if cfg.paper_hyp:
        args.update(PAPER_HYP)
    if cfg.lr0 is not None:
        args["lr0"] = cfg.lr0
    if not cfg.online_augment:
        args.update(ONLINE_AUGMENT_OFF)
    return args


# ---- run folders ------------------------------------------------------------------------

def new_run_id(model: str, now: float | None = None) -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{model}"


def prepare_run(project: Project, cfg: TrainConfig, weights_dir: Path, now: float | None = None) -> Path:
    """Create runs/<run_id>/ with config.yaml and a queued status. Fails early (ValueError with a
    user-facing message) if the dataset has not been built."""
    data_yaml = project.dataset_dir / cfg.dataset / "data.yaml"
    if not data_yaml.is_file():
        raise ValueError(f"데이터셋({cfg.dataset})이 없습니다. Augment 탭에서 먼저 데이터셋을 빌드하세요.")
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = [c.name for c in project.config.classes]
    if data.get("nc") != len(names) or data.get("channels", 3) != project.channels:
        raise ValueError("데이터셋이 현재 클래스/색 모드와 다릅니다. 데이터셋을 다시 빌드하세요.")

    from defect_inspection.core.model_zoo import is_unet

    if is_unet(cfg.model) and not ((data_yaml.parent / "masks" / "train").is_dir()
                                   and (data_yaml.parent / "masks" / "val").is_dir()):
        raise ValueError("U-Net++ 학습에는 마스크가 있는 데이터셋이 필요합니다. Augment 탭에서 데이터셋을 다시 빌드하세요.")

    tiling = None
    if cfg.dataset == "crop":
        info_path = data_yaml.parent / "build_info.yaml"
        info = {}
        if info_path.is_file():
            with open(info_path, encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
        window = info.get("window") or {}
        tiling = TilingParams.from_window(int(window.get("size") or cfg.imgsz), window.get("overlap_px"),
                                          info.get("crop"))

    run_id = new_run_id(cfg.model, now)
    run_dir = project.runs_dir / run_id
    n = 1
    while run_dir.exists():  # two runs in the same second
        n += 1
        run_dir = project.runs_dir / f"{run_id}-{n}"
    run_dir.mkdir(parents=True)
    spec = RunSpec(
        run_id=run_dir.name, project_root=str(project.root), task=project.config.task,
        nc=len(names), names=names, channels=project.channels, data_yaml=str(data_yaml),
        weights_dir=str(weights_dir), train=cfg, created=time.strftime("%Y-%m-%d %H:%M:%S"), crop=tiling,
    )
    atomic_write_text(run_dir / "config.yaml",
                      yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False))
    write_status(run_dir, {"state": "queued"})
    return run_dir


def load_spec(run_dir: Path) -> RunSpec:
    with open(Path(run_dir) / "config.yaml", encoding="utf-8") as f:
        return RunSpec.model_validate(yaml.safe_load(f))


def read_status(run_dir: Path) -> dict[str, Any]:
    try:
        with open(Path(run_dir) / "status.json", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"state": "failed", "error": "status.json missing or unreadable"}


def write_status(run_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    status = read_status(run_dir) if (Path(run_dir) / "status.json").exists() else {}
    status.update(updates)
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_text(Path(run_dir) / "status.json", json.dumps(status, ensure_ascii=False, indent=2))
    return status


def weights_paths(run_dir: Path) -> tuple[Path, Path]:
    w = Path(run_dir) / "train" / "weights"
    return w / "best.pt", w / "last.pt"


# ---- events ---------------------------------------------------------------------------------

def encode_event(event: dict[str, Any]) -> str:
    return EVENT_PREFIX + json.dumps(event, ensure_ascii=False)


def decode_event(line: str) -> dict[str, Any] | None:
    """The event in a worker stdout line, or None for ordinary log output."""
    line = line.strip()
    if not line.startswith(EVENT_PREFIX.strip()):
        return None
    try:
        event = json.loads(line[len(EVENT_PREFIX.strip()):].strip())
    except ValueError:
        return None
    return event if isinstance(event, dict) and "type" in event else None


def open_run_log(run_dir: Path):
    """Append handle for the run's train.log (each attempt adds its own section)."""
    return open(Path(run_dir) / LOG_FILE, "a", encoding="utf-8")


def log_header(resume: bool, now: float | None = None) -> str:
    """First line of an attempt's section in train.log."""
    return f"===== {'재개' if resume else '시작'} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} ====="


def read_run_log(run_dir: Path) -> str | None:
    """The saved log of a run (all attempts), or None for runs from before logs were saved."""
    try:
        return (Path(run_dir) / LOG_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue  # a line cut by a crash
    return events


def epoch_event(trainer) -> dict[str, Any]:
    """Losses and validation metrics of the epoch that just finished (from on_fit_epoch_end)."""
    def clean(d: dict | None) -> dict[str, float]:
        return {k: round(float(v), 6) for k, v in (d or {}).items()}

    epoch = int(trainer.epoch) + 1
    elapsed = time.time() - trainer.train_time_start
    done = max(1, epoch - int(getattr(trainer, "start_epoch", 0)))
    remaining = max(0, int(trainer.epochs) - epoch)
    return {
        "type": "epoch",
        "epoch": epoch,
        "epochs": int(trainer.epochs),
        "losses": clean(trainer.label_loss_items(trainer.tloss, prefix="train") if trainer.tloss is not None else {}),
        "metrics": clean(trainer.metrics),
        "fitness": None if trainer.fitness is None else round(float(trainer.fitness), 6),
        "best_fitness": None if trainer.best_fitness is None else round(float(trainer.best_fitness), 6),
        "lr": clean(getattr(trainer, "lr", {})),
        "elapsed_s": round(elapsed, 1),
        "eta_s": round(elapsed / done * remaining, 1),
    }


# ---- history (F4.7) --------------------------------------------------------------------

@dataclass
class RunInfo:
    run_id: str
    path: Path
    spec: RunSpec | None
    status: dict[str, Any]

    @property
    def state(self) -> str:
        return self.status.get("state", "failed")

    @property
    def best_weights(self) -> Path | None:
        best, _ = weights_paths(self.path)
        return best if best.is_file() else None

    @property
    def resumable(self) -> bool:
        _, last = weights_paths(self.path)
        return self.state in ("stopped", "failed", "running") and last.is_file()


def list_runs(project: Project) -> list[RunInfo]:
    """Runs of a project, newest first."""
    runs = []
    if project.runs_dir.is_dir():
        for d in project.runs_dir.iterdir():
            if not (d / "config.yaml").is_file():
                continue
            try:
                spec = load_spec(d)
            except (OSError, ValueError, yaml.YAMLError):
                spec = None
            runs.append(RunInfo(d.name, d, spec, read_status(d)))
    return sorted(runs, key=lambda r: r.run_id, reverse=True)


class EpochSeries:
    """Per-key value series from `epoch` events, for live charts and for replaying a past run."""

    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.values: dict[str, list[float | None]] = {}

    def add(self, event: dict[str, Any]) -> None:
        if event.get("type") != "epoch":
            return
        epoch = int(event["epoch"])
        if self.epochs and epoch <= self.epochs[-1]:
            # A resumed run repeats from its checkpoint: drop the overlapping tail.
            cut = next(i for i, e in enumerate(self.epochs) if e >= epoch)
            self.epochs = self.epochs[:cut]
            for v in self.values.values():
                del v[cut:]
        merged = {**event.get("losses", {}), **event.get("metrics", {})}
        for key in merged.keys() - self.values.keys():
            self.values[key] = [None] * len(self.epochs)
        self.epochs.append(epoch)
        for key, series in self.values.items():
            series.append(merged.get(key))

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> EpochSeries:
        s = cls()
        for e in events:
            s.add(e)
        return s

    def latest(self, key: str) -> float | None:
        vals = [v for v in self.values.get(key, []) if v is not None]
        return vals[-1] if vals else None

    def best(self, key: str) -> float | None:
        vals = [v for v in self.values.get(key, []) if v is not None]
        return max(vals) if vals else None
