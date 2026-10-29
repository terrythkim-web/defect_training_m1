"""Training worker (defect_inspection/worker.py) called in-process: the stdin stop watcher, the
Ultralytics path up to model.train (model faked) and the failure reporting. The real Ultralytics
training runs in the slow tests (test_trainer.py, test_integration.py)."""

from __future__ import annotations

import io
import threading

import pytest

import defect_inspection.core.model_zoo as model_zoo
from defect_inspection import worker
from defect_inspection.core import trainer as T
from test_trainer import make_project


@pytest.mark.parametrize("text", ["noise\n  STOP \nlater\n", ""])
def test_stdin_stop_line_or_closed_stdin_stops(monkeypatch, text):
    """`stop` asks for a stop; a closed stdin means the app went away, which stops too."""
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(text))
    stop = threading.Event()
    worker.watch_stdin(stop)
    assert stop.is_set()


class FakeModel:
    def __init__(self):
        self.callbacks, self.train_kwargs = {}, None

    def add_callback(self, name, fn):
        self.callbacks[name] = fn

    def train(self, **kwargs):
        self.train_kwargs = kwargs


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A prepared run; the stdin watcher is disabled (pytest's stdin cannot be read)."""
    monkeypatch.delenv("DEFECTLAB_FAKE_TRAIN", raising=False)
    monkeypatch.setattr(worker, "watch_stdin", lambda stop: None)
    project = make_project(tmp_path)

    def prepare(**train):
        cfg = T.TrainConfig(model="yolo11n", epochs=2, imgsz=128, workers=0, device="cpu", **train)
        return T.prepare_run(project, cfg, tmp_path / "w")
    return prepare


def test_new_run_trains_with_train_args_and_logs_auto_batch(run, monkeypatch, capsys):
    model = FakeModel()
    monkeypatch.setattr(model_zoo, "make_model", lambda *a, **k: (model, None))
    run_dir = run(batch=-1)
    assert worker.main(["--run-dir", str(run_dir)]) == 0
    assert set(model.callbacks) == {"on_pretrain_routine_end", "on_train_batch_end",
                                    "on_train_epoch_end", "on_fit_epoch_end"}
    assert model.train_kwargs["batch"] > 0 and model.train_kwargs["device"] == "cpu"
    assert f"자동 batch: {model.train_kwargs['batch']}" in capsys.readouterr().out
    assert [e["type"] for e in T.read_events(run_dir)] == ["start", "done"]
    assert T.read_status(run_dir)["state"] == "finished"


def test_fixed_batch_is_not_logged(run, monkeypatch, capsys):
    monkeypatch.setattr(model_zoo, "make_model", lambda *a, **k: (FakeModel(), None))
    assert worker.main(["--run-dir", str(run(batch=4))]) == 0
    assert "자동 batch" not in capsys.readouterr().out


def test_resume_without_last_pt_fails(run):
    run_dir = run()
    assert worker.main(["--run-dir", str(run_dir), "--resume"]) == 1
    error = T.read_events(run_dir)[-1]
    assert error["type"] == "error" and "FileNotFoundError" in error["message"] and "last.pt" in error["message"]
    assert "Traceback" in error["traceback"]
    status = T.read_status(run_dir)
    assert status["state"] == "failed" and "last.pt" in status["error"]


def test_any_training_error_is_reported(run, monkeypatch):
    def boom(*a, **k):
        raise MemoryError("out of memory")

    monkeypatch.setattr(model_zoo, "make_model", boom)
    run_dir = run()
    assert worker.main(["--run-dir", str(run_dir)]) == 1
    assert T.read_events(run_dir)[-1]["message"] == "MemoryError: out of memory"
    assert T.read_status(run_dir)["state"] == "failed"
