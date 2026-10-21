"""Training runs and the worker process (PROMPTS step 11, PRD F4.1-F4.4, F4.7)."""

import json
import sys
import time

import cv2
import numpy as np
import pytest
import yaml
from PySide6.QtWidgets import QApplication

from defect_inspection.core import trainer as T
from defect_inspection.core.annotation import BBox
from defect_inspection.core.augment_config import AugmentConfig, TransformSetting
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.train_process import TrainProcess, worker_command


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_project(root, n=6, size=128, multiplier=1):
    src = root / "원본"
    src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    centers = []
    for i in range(n):
        img = np.full((size, size, 3), 190, np.uint8)
        cx, cy = int(rng.integers(30, size - 30)), int(rng.integers(30, size - 30))
        cv2.circle(img, (cx, cy), 14, (40, 40, 40), -1)
        imwrite_unicode(src / f"결함 {i}.png", img)
        centers.append((cx, cy))
    p = Project.create(root / "학습 프로젝트")
    p.add_class("dent")
    p.add_images([src])
    for (cx, cy), e in zip(centers, p.config.images):
        ann = p.load_labels(e)
        ann.shapes.append(BBox(class_id=0, x1=cx - 15, y1=cy - 15, x2=cx + 15, y2=cy + 15))
        p.save_labels(ann)
    p.config.augment = AugmentConfig(multiplier=multiplier, transforms={"hflip": TransformSetting(enabled=True)})
    p.save()
    build_full_dataset(p)
    return p


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


# ---- configuration -------------------------------------------------------------------

def test_config_defaults_and_model_defaults():
    cfg = T.TrainConfig()
    assert (cfg.model, cfg.epochs, cfg.imgsz, cfg.batch, cfg.device) == ("yolo11s", 100, 640, -1, "auto")
    att = T.TrainConfig.for_model("att-yolo-s", epochs=5)
    assert (att.imgsz, att.epochs) == (1280, 5)
    with pytest.raises(ValueError):
        T.TrainConfig(epochs=0)
    with pytest.raises(KeyError):
        T.TrainConfig.for_model("yolov9")


def test_train_args(project, tmp_path):
    spec = T.RunSpec(run_id="r", project_root=".", task="detect", nc=1, names=["dent"], channels=3,
                     data_yaml="d.yaml", weights_dir=".", created="now",
                     train=T.TrainConfig(paper_hyp=True, lr0=0.01, online_augment=False, device="cpu"))
    args = T.train_args(spec, tmp_path)
    assert args["data"] == "d.yaml" and args["project"] == str(tmp_path) and args["name"] == "train"
    assert args["device"] == "cpu" and args["lr0"] == 0.01  # explicit lr0 wins over the paper preset
    assert args["momentum"] == 0.843 and args["mosaic"] == 0.0 and args["fliplr"] == 0.0
    auto = T.train_args(spec.model_copy(update={"train": T.TrainConfig()}), tmp_path)
    assert "lr0" not in auto and "mosaic" not in auto
    # "auto" is always made explicit, because Ultralytics never picks Apple MPS by itself (PRD 5.2)
    from defect_inspection.platform_utils import resolve_device

    assert auto["device"] == resolve_device("auto", "train") in ("0", "mps", "cpu")


def test_windows_uses_few_dataloader_workers(monkeypatch):
    from defect_inspection import platform_utils as pu

    monkeypatch.setattr(pu, "IS_MAC", False)
    monkeypatch.setattr(pu, "IS_WINDOWS", True)
    assert pu.default_dataloader_workers() <= 2
    monkeypatch.setattr(pu, "IS_WINDOWS", False)
    assert pu.default_dataloader_workers() >= 1
    monkeypatch.setattr(pu, "IS_MAC", True)  # MPS training runs with 0 workers (PRD 5.2)
    assert pu.default_dataloader_workers() == 0


def test_prepare_run(project, tmp_path):
    run = T.prepare_run(project, T.TrainConfig(model="yolo11n"), tmp_path / "w", now=0)
    spec = T.load_spec(run)
    assert run.parent == project.runs_dir and run.name.endswith("_yolo11n")
    assert spec.nc == 1 and spec.names == ["dent"] and spec.channels == 3 and spec.task == "detect"
    assert spec.data_yaml.endswith("data.yaml")
    assert T.read_status(run)["state"] == "queued"
    again = T.prepare_run(project, T.TrainConfig(model="yolo11n"), tmp_path / "w", now=0)
    assert again != run and again.name.startswith(run.name)  # same second -> unique


def test_prepare_run_requires_matching_dataset(project, tmp_path):
    with pytest.raises(ValueError, match="빌드"):
        T.prepare_run(project, T.TrainConfig(dataset="crop"), tmp_path)
    project.add_class("scratch")  # dataset now stale
    with pytest.raises(ValueError, match="다시 빌드"):
        T.prepare_run(project, T.TrainConfig(), tmp_path)


def test_event_codec():
    line = T.encode_event({"type": "epoch", "epoch": 3, "msg": "한글"})
    assert line.startswith("@@DL ") and T.decode_event(line) == {"type": "epoch", "epoch": 3, "msg": "한글"}
    assert T.decode_event("Ultralytics 8.4 🚀 Python") is None
    assert T.decode_event("@@DL {broken") is None
    assert T.decode_event('@@DL {"no_type": 1}') is None


def test_worker_command(monkeypatch, tmp_path):
    prog, args = worker_command(tmp_path, resume=True)
    assert prog == sys.executable and args[:3] == ["-u", "-m", "defect_inspection.worker"] and "--resume" in args
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _, args = worker_command(tmp_path, resume=False)
    assert args[0] == "--train-worker" and "--resume" not in args


def test_handle_line_splits_events_and_logs(qapp):
    tp = TrainProcess()
    events, logs = [], []
    tp.event.connect(events.append)
    tp.log.connect(logs.append)
    tp._handle_line('@@DL {"type": "epoch", "epoch": 1}')
    tp._handle_line("      Epoch    GPU_mem   box_loss")
    tp._handle_line("   ")
    assert events == [{"type": "epoch", "epoch": 1}] and logs == ["      Epoch    GPU_mem   box_loss"]


# ---- worker process (fake training) ------------------------------------------------------

class Recorder:
    def __init__(self, tp: TrainProcess):
        self.events, self.logs, self.final = [], [], None
        tp.event.connect(self.events.append)
        tp.log.connect(self.logs.append)
        tp.finished.connect(self._done)

    def _done(self, state):
        self.final = state

    def types(self):
        return [e["type"] for e in self.events]


def start(qapp, project, tmp_path, monkeypatch, fake, resume_run=None):
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", fake)
    run = resume_run or T.prepare_run(project, T.TrainConfig(), tmp_path / "w")
    tp = TrainProcess()
    rec = Recorder(tp)
    tp.start(run, resume=resume_run is not None)
    return tp, rec, run


def test_full_run(qapp, project, tmp_path, monkeypatch):
    tp, rec, run = start(qapp, project, tmp_path, monkeypatch, "3:0.05")
    assert wait_until(qapp, lambda: rec.final is not None)
    assert rec.final == "finished"
    assert rec.types() == ["start", "model"] + ["train_epoch", "epoch"] * 3 + ["done"]
    epochs = [e for e in rec.events if e["type"] == "epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2, 3] and epochs[-1]["eta_s"] == 0
    status = T.read_status(run)
    assert status["state"] == "finished" and status["epoch"] == 3 and status["best"].endswith("best.pt")
    assert [e["type"] for e in T.read_events(run)] == rec.types()


def test_stop_then_resume(qapp, project, tmp_path, monkeypatch):
    tp, rec, run = start(qapp, project, tmp_path, monkeypatch, "40:0.1")
    assert wait_until(qapp, lambda: "epoch" in rec.types())
    tp.stop()
    assert wait_until(qapp, lambda: rec.final is not None)
    assert rec.final == "stopped" and rec.events[-1] == {**rec.events[-1], "type": "stopped", "resumable": True}
    stopped_at = T.read_status(run)["epoch"]
    assert stopped_at < 40
    info = T.list_runs(project)[0]
    assert info.state == "stopped" and info.resumable

    tp2, rec2, _ = start(qapp, project, tmp_path, monkeypatch, f"{stopped_at + 2}:0.02", resume_run=run)
    assert wait_until(qapp, lambda: rec2.final is not None)
    assert rec2.final == "finished" and rec2.events[0]["resume"] is True
    assert [e["epoch"] for e in rec2.events if e["type"] == "epoch"] == [stopped_at + 1, stopped_at + 2]
    assert not T.list_runs(project)[0].resumable  # finished


def test_unresponsive_worker_is_killed(qapp, project, tmp_path, monkeypatch):
    tp, rec, run = start(qapp, project, tmp_path, monkeypatch, "40:0.1:hang")
    tp.stop_timeout_ms = 500
    assert wait_until(qapp, lambda: "start" in rec.types())
    tp.stop()
    assert wait_until(qapp, lambda: rec.final is not None, timeout=15)
    assert rec.final == "stopped"
    assert any("강제 종료" in line for line in rec.logs)
    assert T.read_status(run)["state"] == "stopped"


def test_crash_is_recorded_and_ui_survives(qapp, project, tmp_path, monkeypatch):
    monkeypatch.setenv("DEFECTLAB_FAKE_CRASH", "1")
    tp, rec, run = start(qapp, project, tmp_path, monkeypatch, "1:0.01")
    assert wait_until(qapp, lambda: rec.final is not None)
    assert rec.final == "failed" and "done" not in rec.types()
    status = T.read_status(run)
    assert status["state"] == "failed" and "exit code 3" in status["error"]
    assert not tp.running


def test_bad_config_reports_error(qapp, project, tmp_path, monkeypatch):
    run = T.prepare_run(project, T.TrainConfig(), tmp_path / "w")
    (run / "config.yaml").write_text("train: {epochs: -5}", encoding="utf-8")
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "1:0.01")
    tp2 = TrainProcess()
    rec2 = Recorder(tp2)
    tp2.start(run)
    assert wait_until(qapp, lambda: rec2.final is not None)
    assert rec2.final == "failed" and rec2.events[0]["type"] == "error" and "config.yaml" in rec2.events[0]["message"]


def test_only_one_run_per_process_object(qapp, project, tmp_path, monkeypatch):
    tp, rec, run = start(qapp, project, tmp_path, monkeypatch, "20:0.1")
    with pytest.raises(RuntimeError):
        tp.start(run)
    tp.stop()
    assert wait_until(qapp, lambda: rec.final is not None)


def test_list_runs_newest_first(project, tmp_path):
    a = T.prepare_run(project, T.TrainConfig(model="yolo11n"), tmp_path, now=1_000_000)
    b = T.prepare_run(project, T.TrainConfig(model="yolo11s"), tmp_path, now=2_000_000)
    (project.runs_dir / "junk").mkdir()
    runs = T.list_runs(project)
    assert [r.path for r in runs] == [b, a]
    assert runs[0].spec.train.model == "yolo11s" and runs[0].state == "queued" and not runs[0].resumable


# ---- real Ultralytics training (slow) -------------------------------------------------------

@pytest.mark.slow
def test_real_training_stop_resume_and_att_transfer(qapp, tmp_path):
    p = make_project(tmp_path, n=6, size=160, multiplier=2)
    from defect_inspection.platform_utils import cache_dir

    weights = cache_dir() / "weights"
    cfg = T.TrainConfig(model="yolo11n", epochs=3, imgsz=160, batch=4, workers=0, device="cpu")
    run = T.prepare_run(p, cfg, weights)
    tp = TrainProcess()
    rec = Recorder(tp)
    tp.start(run)
    assert wait_until(qapp, lambda: "epoch" in rec.types(), timeout=300), rec.logs[-20:]
    model_event = next(e for e in rec.events if e["type"] == "model")
    assert model_event["channels"] == 3 and model_event["params"] > 1_000_000
    first = next(e for e in rec.events if e["type"] == "epoch")
    assert "metrics/mAP50(B)" in first["metrics"] and "train/box_loss" in first["losses"]
    tp.stop()
    assert wait_until(qapp, lambda: rec.final is not None, timeout=120)
    assert rec.final == "stopped" and T.list_runs(p)[0].resumable

    tp2 = TrainProcess()
    rec2 = Recorder(tp2)
    tp2.start(run, resume=True)
    assert wait_until(qapp, lambda: rec2.final is not None, timeout=300), rec2.logs[-20:]
    assert rec2.final == "finished", rec2.logs[-20:]
    epochs = [e["epoch"] for e in rec2.events if e["type"] == "epoch"]
    assert epochs and epochs[-1] == 3 and len(epochs) == len(set(epochs))  # no phantom epoch 4
    final = [e for e in rec2.events if e["type"] == "final"]
    assert len(final) == 1 and "metrics/mAP50(B)" in final[0]["metrics"]
    assert "final_metrics" in T.read_status(run)
    best, _ = T.weights_paths(run)
    from ultralytics import YOLO

    assert YOLO(str(best)).predict(np.zeros((160, 160, 3), np.uint8), verbose=False)

    att = T.prepare_run(p, T.TrainConfig(model="att-yolo-n", epochs=1, imgsz=128, batch=4, workers=0,
                                         device="cpu"), weights)
    tp3 = TrainProcess()
    rec3 = Recorder(tp3)
    tp3.start(att)
    assert wait_until(qapp, lambda: rec3.final is not None, timeout=300), rec3.logs[-20:]
    assert rec3.final == "finished", rec3.logs[-20:]
    transfer = next(e for e in rec3.events if e["type"] == "transfer")
    assert transfer["layers"] == list(range(9)) and "yolov5nu.pt" in transfer["summary"]


def test_frozen_worker_exe_preferred(monkeypatch, tmp_path):
    """Windows builds ship DefectLab-worker(.exe) next to the windowed app; it is used when present."""
    from defect_inspection import platform_utils as pu

    app = tmp_path / ("DefectLab.exe" if pu.IS_WINDOWS else "DefectLab")
    app.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app))
    prog, args = worker_command(tmp_path, resume=True)
    assert prog == str(app) and args[0] == "--train-worker"
    worker = tmp_path / (pu.WORKER_EXE + (".exe" if pu.IS_WINDOWS else ""))
    worker.write_bytes(b"")
    prog, args = worker_command(tmp_path, resume=True)
    assert prog == str(worker) and args[:2] == ["--run-dir", str(tmp_path)] and "--resume" in args
