"""Training worker process: `python -m defect_inspection.worker --run-dir <runs/id> [--resume]`.

Runs one training in its own process so a crash, an out-of-memory error or a hung DataLoader
can never take the UI down. Protocol (see core/trainer.py):
- stdout: `@@DL {json}` event lines (start, transfer, model, train_epoch, epoch, final, done, stopped,
  error; `final` = validation metrics of best.pt after training);
  everything else on stdout/stderr is plain log text.
- stdin: a line `stop` asks for a graceful stop after the current batch.

DEFECTLAB_FAKE_TRAIN="<epochs>:<seconds per epoch>[:hang]" replaces Ultralytics with a fake
training loop. Tests use it to exercise the protocol, stop and kill quickly; "hang" ignores stop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

from defect_inspection.core.model_zoo import is_unet
from defect_inspection.core.trainer import (
    encode_event,
    epoch_event,
    load_spec,
    train_args,
    weights_paths,
    write_status,
)
from defect_inspection.platform_utils import stdin_lines


class StopTraining(Exception):
    """Raised from a callback to end training without Ultralytics' final (optimizer-stripping) step."""


class Emitter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.lock = threading.Lock()

    def __call__(self, event: dict) -> None:
        event.setdefault("time", time.strftime("%H:%M:%S"))
        with self.lock:
            print(encode_event(event), flush=True)
            with open(self.run_dir / "events.jsonl", "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


def watch_stdin(stop: threading.Event) -> None:
    """Set `stop` when the parent writes `stop` (or closes stdin, i.e. the UI went away)."""
    for line in stdin_lines():
        if line.strip().lower() == "stop":
            stop.set()
            return
    stop.set()


def run_fake(spec, emit: Emitter, stop: threading.Event, run_dir: Path, start_epoch: int) -> None:
    epochs, per_epoch, *rest = os.environ["DEFECTLAB_FAKE_TRAIN"].split(":")
    epochs, per_epoch, hang = int(epochs), float(per_epoch), "hang" in rest
    best, last = weights_paths(run_dir)
    best.parent.mkdir(parents=True, exist_ok=True)
    emit({"type": "model", "channels": spec.channels, "task": spec.task, "params": 0})
    t0 = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        end = time.time() + per_epoch
        while time.time() < end:
            if stop.is_set() and not hang:
                raise StopTraining()
            time.sleep(0.01)
        loss = round(1.0 / epoch, 4)
        emit({"type": "train_epoch", "epoch": epoch, "epochs": epochs, "losses": {"train/box_loss": loss}})
        emit({"type": "epoch", "epoch": epoch, "epochs": epochs, "losses": {"train/box_loss": loss},
              "metrics": {"metrics/mAP50(B)": round(1 - loss, 4)}, "fitness": round(1 - loss, 4),
              "best_fitness": round(1 - loss, 4), "lr": {}, "elapsed_s": round(time.time() - t0, 2),
              "eta_s": round(per_epoch * (epochs - epoch), 2)})
        last.write_text(f"fake last {epoch}", encoding="utf-8")
        best.write_text(f"fake best {epoch}", encoding="utf-8")
        write_status(run_dir, {"epoch": epoch, "epochs": epochs})
    if os.environ.get("DEFECTLAB_FAKE_CRASH"):
        os._exit(3)  # simulate a hard crash (no done/error event)


def run_ultralytics(spec, emit: Emitter, stop: threading.Event, run_dir: Path, resume: bool) -> None:
    from ultralytics import YOLO

    from defect_inspection.core.model_zoo import make_model

    _, last = weights_paths(run_dir)
    if resume:
        if not last.is_file():
            raise FileNotFoundError("재개할 last.pt가 없습니다")
        model = YOLO(str(last))
    else:
        model, report = make_model(spec.train.model, spec.task, spec.nc, spec.channels, run_dir,
                                   Path(spec.weights_dir), spec.train.pretrained)
        if report is not None:
            emit({"type": "transfer", "summary": report.summary(), "layers": report.layers,
                  "transferred": len(report.transferred), "total": report.total})

    def on_pretrain_routine_end(trainer) -> None:
        # The model Ultralytics actually trains (rebuilt with the dataset's channels).
        import torch.nn as nn

        first = next(m for m in trainer.model.modules() if isinstance(m, nn.Conv2d))
        emit({"type": "model", "channels": int(first.in_channels), "task": spec.task,
              "params": int(sum(p.numel() for p in trainer.model.parameters()))})

    def on_train_batch_end(trainer) -> None:
        if stop.is_set():
            raise StopTraining()

    def on_train_epoch_end(trainer) -> None:
        losses = trainer.label_loss_items(trainer.tloss, prefix="train") if trainer.tloss is not None else {}
        emit({"type": "train_epoch", "epoch": int(trainer.epoch) + 1, "epochs": int(trainer.epochs),
              "losses": {k: round(float(v), 6) for k, v in losses.items()}})

    emitted: set[int] = set()

    def on_fit_epoch_end(trainer) -> None:
        event = epoch_event(trainer)
        # Ultralytics fires this callback once more from final_eval (validation of best.pt)
        # after the last epoch. Report that as the final result, not as an extra epoch.
        if event["epoch"] in emitted or event["epoch"] > event["epochs"]:
            final = {"type": "final", "metrics": event["metrics"], "fitness": event["fitness"]}
            emit(final)
            write_status(run_dir, {"final_metrics": event["metrics"]})
            return
        emitted.add(event["epoch"])
        emit(event)
        write_status(run_dir, {"epoch": event["epoch"], "epochs": event["epochs"],
                               "best_fitness": event["best_fitness"], "metrics": event["metrics"]})

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    # Training runs with torch.use_deterministic_algorithms(True, warn_only=True) (Ultralytics
    # deterministic=True). MPS lacks deterministic versions of a few backward ops and warns about
    # each one; harmless (results may differ in the last digits between runs) but alarming in the log.
    warnings.filterwarnings("ignore", message=r".*does not have a deterministic implementation.*")
    if resume:
        model.train(resume=True)
    else:
        args = train_args(spec, run_dir)
        if spec.train.batch <= 0 and args["batch"] > 0:
            print(f"자동 batch: {args['batch']} (장치 {args['device']}, 메모리 기준 추정. "
                  "느리거나 스왑이 많으면 batch를 직접 낮추세요)", flush=True)
        model.train(**args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="defect_inspection.worker")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    emit = Emitter(run_dir)
    stop = threading.Event()
    threading.Thread(target=watch_stdin, args=(stop,), daemon=True).start()

    try:
        spec = load_spec(run_dir)
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "message": f"config.yaml을 읽을 수 없습니다: {e}"})
        return 2
    status = write_status(run_dir, {"state": "running", "pid": os.getpid(), "error": None})
    emit({"type": "start", "run_id": spec.run_id, "model": spec.train.model, "resume": args.resume,
          "epochs": spec.train.epochs, "pid": os.getpid()})
    best, last = weights_paths(run_dir)
    try:
        if os.environ.get("DEFECTLAB_FAKE_TRAIN"):
            run_fake(spec, emit, stop, run_dir, int(status.get("epoch", 0)) if args.resume else 0)
        elif is_unet(spec.train.model):
            from defect_inspection.core.unet import train_unet

            train_unet(spec, run_dir, emit, stop.is_set, args.resume, StopTraining)
        else:
            run_ultralytics(spec, emit, stop, run_dir, args.resume)
    except StopTraining:
        resumable = last.is_file()
        write_status(run_dir, {"state": "stopped", "resumable": resumable})
        emit({"type": "stopped", "resumable": resumable, "best": str(best) if best.is_file() else None})
        return 0
    except BaseException as e:  # noqa: BLE001 - reported, then the process exits non-zero
        message = f"{type(e).__name__}: {e}"
        write_status(run_dir, {"state": "failed", "error": message})
        emit({"type": "error", "message": message, "traceback": traceback.format_exc(limit=8)})
        return 1
    write_status(run_dir, {"state": "finished", "best": str(best) if best.is_file() else None})
    emit({"type": "done", "best": str(best) if best.is_file() else None})
    return 0


if __name__ == "__main__":  # required on Windows (spawn) for DataLoader worker processes
    sys.exit(main())
