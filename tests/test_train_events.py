"""Worker event → Train-tab view (ui/train_events.py) and the run's train.log writer."""

from __future__ import annotations

from defect_inspection.core.trainer import LOG_FILE
from defect_inspection.ui.train_events import EventView, RunLog, describe_event, fmt_duration, fmt_metrics


def test_fmt_duration():
    assert fmt_duration(None) == "-"
    assert fmt_duration(65.4) == "1:05"
    assert fmt_duration(3725) == "1:02:05"


def test_fmt_metrics_only_metrics_entries():
    text = fmt_metrics({"metrics/precision(B)": 0.5, "val/box_loss": 1.0, "metrics/mAP50(B)": 0.25})
    assert text == "precision(B)=0.5000, mAP50(B)=0.2500"


def test_start_sets_progress_maximum_only():
    view = describe_event({"type": "start", "model": "yolo11s", "epochs": 7, "pid": 42}, 3)
    assert view.progress == (None, 7) and "학습 시작: yolo11s (epochs 7, PID 42)" in view.logs[0]
    assert describe_event({"type": "start"}, 3).progress == (None, 1)


def test_model_warns_on_channel_mismatch():
    ok = describe_event({"type": "model", "channels": 3, "params": 2_500_000}, 3)
    assert ok.logs == ("모델 입력 채널: 3 (컬러), 파라미터 2.5M",)
    bad = describe_event({"type": "model", "channels": 3}, 1)
    assert len(bad.logs) == 2 and "프로젝트는 1채널인데 모델은 3채널" in bad.logs[1]
    assert len(describe_event({"type": "model", "channels": 1}, None).logs) == 1


def test_train_epoch_status_only_with_loss():
    assert describe_event({"type": "train_epoch", "epoch": 1, "epochs": 2, "losses": {}}, 3) == EventView()
    view = describe_event({"type": "train_epoch", "epoch": 1, "epochs": 2, "losses": {"train/box_loss": 0.5}}, 3)
    assert view.status == "epoch 1/2 학습 완료, 검증 중… (box loss 0.5000)" and not view.logs


def test_epoch_updates_everything():
    e = {"type": "epoch", "epoch": 2, "epochs": 5, "elapsed_s": 61, "eta_s": 90,
         "metrics": {"metrics/mAP50-95(B)": 0.4, "val/box_loss": 1.0}}
    view = describe_event(e, 3)
    assert view.chart and view.progress == (2, 5)
    assert view.status == "epoch 2/5 · 경과 1:01 · 남은 시간 약 1:30 · mAP50-95 0.400"
    assert view.logs == ("epoch 2/5: mAP50-95(B)=0.4000",)
    seg = describe_event({**e, "metrics": {"metrics/mAP50-95(M)": 0.3}}, 3)
    assert seg.status.endswith("mAP50-95 0.300")
    assert "mAP50-95" not in describe_event({**e, "metrics": {}}, 3).status


def test_terminal_events_log():
    assert describe_event({"type": "final", "metrics": {"metrics/recall(B)": 1}}, 3).logs == \
        ("best.pt 최종 검증: recall(B)=1.0000",)
    assert describe_event({"type": "done", "best": "x/best.pt"}, 3).logs == ("학습 완료: x/best.pt",)
    assert describe_event({"type": "stopped", "resumable": True}, 3).logs == ("중지됨 (재개 가능)",)
    assert describe_event({"type": "stopped"}, 3).logs == ("중지됨 (재개 불가)",)
    err = describe_event({"type": "error", "message": "boom", "traceback": "Traceback…"}, 3)
    assert err.logs == ("오류: boom", "Traceback…")
    assert describe_event({"type": "error", "message": "boom"}, 3).logs == ("오류: boom",)


def test_unknown_event_changes_nothing():
    assert describe_event({"type": "nope"}, 3) == EventView()
    assert describe_event({}, 3) == EventView()


def test_run_log_appends_and_closes(tmp_path):
    log = RunLog()
    log.write("ignored while closed")
    log.open(tmp_path)
    assert log.is_open
    log.write("a")
    log.open(tmp_path)  # reopening closes the earlier handle first
    log.write("b")
    log.close()
    log.close()
    assert not log.is_open
    assert (tmp_path / LOG_FILE).read_text(encoding="utf-8") == "a\nb\n"


def test_run_log_write_error_closes(tmp_path):
    class Broken:
        closed = False

        def write(self, text):
            raise OSError("disk full")

        def close(self):
            Broken.closed = True

    log = RunLog()
    log._file = Broken()
    log.write("x")
    assert not log.is_open and Broken.closed
