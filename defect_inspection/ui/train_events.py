"""Training worker events → what the Train tab shows (log lines, status, progress), and the run's train.log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QCoreApplication

from defect_inspection.core.trainer import open_run_log

MAIN_METRIC = "metrics/mAP50-95(B)"


def _tr(text: str) -> str:
    return QCoreApplication.translate("TrainPanel", text)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_metrics(metrics: dict) -> str:
    """`precision(B)=0.5000, …` for the `metrics/*` entries of an epoch/final event."""
    return ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items() if k.startswith("metrics/"))


@dataclass(frozen=True)
class EventView:
    """What one worker event changes in the Train tab. `progress` = (value or None, maximum)."""

    logs: tuple[str, ...] = ()
    status: str | None = None
    progress: tuple[int | None, int] | None = None
    chart: bool = False


def _start(e: dict, channels: int | None) -> EventView:
    return EventView(
        logs=(_tr("학습 시작: {0} (epochs {1}, PID {2})").format(e.get("model"), e.get("epochs"), e.get("pid")),),
        progress=(None, int(e.get("epochs") or 1)))


def _model(e: dict, channels: int | None) -> EventView:
    ch = e.get("channels")
    mode = _tr("그레이스케일") if ch == 1 else _tr("컬러")
    params = e.get("params") or 0
    logs = [_tr("모델 입력 채널: {0} ({1}), 파라미터 {2:.1f}M").format(ch, mode, params / 1e6)]
    if channels is not None and ch != channels:
        logs.append(_tr("경고: 프로젝트는 {0}채널인데 모델은 {1}채널입니다").format(channels, ch))
    return EventView(logs=tuple(logs))


def _transfer(e: dict, channels: int | None) -> EventView:
    return EventView(logs=(_tr("사전학습 가중치 전이: {0}").format(e.get("summary")),))


def _train_epoch(e: dict, channels: int | None) -> EventView:
    loss = e.get("losses", {}).get("train/box_loss")
    if loss is None:
        return EventView()
    return EventView(status=_tr("epoch {0}/{1} 학습 완료, 검증 중… (box loss {2:.4f})")
                     .format(e["epoch"], e["epochs"], loss))


def _epoch(e: dict, channels: int | None) -> EventView:
    m = e.get("metrics", {})
    main = m.get(MAIN_METRIC, m.get("metrics/mAP50-95(M)"))
    status = _tr("epoch {0}/{1} · 경과 {2} · 남은 시간 약 {3}{4}").format(
        e["epoch"], e["epochs"], fmt_duration(e.get("elapsed_s")), fmt_duration(e.get("eta_s")),
        "" if main is None else f" · mAP50-95 {main:.3f}")
    return EventView(logs=(_tr("epoch {0}/{1}: {2}").format(e["epoch"], e["epochs"], fmt_metrics(m)),),
                     status=status, progress=(e["epoch"], e["epochs"]), chart=True)


def _final(e: dict, channels: int | None) -> EventView:
    return EventView(logs=(_tr("best.pt 최종 검증: {0}").format(fmt_metrics(e.get("metrics", {}))),))


def _done(e: dict, channels: int | None) -> EventView:
    return EventView(logs=(_tr("학습 완료: {0}").format(e.get("best")),))


def _stopped(e: dict, channels: int | None) -> EventView:
    return EventView(logs=(_tr("중지됨 (재개 {0})").format(_tr("가능") if e.get("resumable") else _tr("불가")),))


def _error(e: dict, channels: int | None) -> EventView:
    logs = [_tr("오류: {0}").format(e.get("message"))]
    if e.get("traceback"):
        logs.append(e["traceback"])
    return EventView(logs=tuple(logs))


_HANDLERS: dict[str, Callable[[dict, int | None], EventView]] = {
    "start": _start, "model": _model, "transfer": _transfer, "train_epoch": _train_epoch, "epoch": _epoch,
    "final": _final, "done": _done, "stopped": _stopped, "error": _error,
}


def describe_event(e: dict, project_channels: int | None) -> EventView:
    """`project_channels` (None without a project) lets the `model` event warn about a channel mismatch."""
    handler = _HANDLERS.get(e.get("type"))
    return EventView() if handler is None else handler(e, project_channels)


class RunLog:
    """Append handle for the active run's train.log; every write is flushed (an app crash must not lose
    the log) and a failing write closes the file instead of raising."""

    def __init__(self) -> None:
        self._file = None

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def open(self, run: Path) -> None:
        """Raises OSError if the file cannot be opened."""
        self.close()
        self._file = open_run_log(run)

    def write(self, text: str) -> None:
        if self._file is None:
            return
        try:
            self._file.write(text + "\n")
            self._file.flush()
        except OSError:
            self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
