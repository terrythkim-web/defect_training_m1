"""Launch and supervise the training worker with QProcess (PRD F4.4)."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from defect_inspection.core.trainer import decode_event, weights_paths, write_status
from defect_inspection.platform_utils import frozen_worker_command, worker_environment

STOP_TIMEOUT_MS = 20_000  # graceful stop window before the worker is killed


def worker_command(run_dir: Path, resume: bool) -> tuple[str, list[str]]:
    """Program and arguments. A PyInstaller build has no `python -m`: it starts its worker
    executable (Windows) or itself with --train-worker (platform_utils.frozen_worker_command)."""
    args = ["--run-dir", str(run_dir)] + (["--resume"] if resume else [])
    if getattr(sys, "frozen", False):
        return frozen_worker_command(args)
    return sys.executable, ["-u", "-m", "defect_inspection.worker", *args]


class TrainProcess(QObject):
    """One training run in a child process. Emits parsed events and log lines; survives any
    worker crash (the UI only sees `finished` with a failure state)."""

    event = Signal(dict)
    log = Signal(str)
    finished = Signal(str)  # final state: finished / stopped / failed

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.run_dir: Path | None = None
        self.process: QProcess | None = None
        self._buffers = {"out": b"", "err": b""}
        self._final_event: str | None = None
        self._stop_requested = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill)
        self.stop_timeout_ms = STOP_TIMEOUT_MS

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, run_dir: Path, resume: bool = False) -> None:
        if self.running:
            raise RuntimeError("a training run is already active")
        self.run_dir = Path(run_dir)
        self._final_event = None
        self._stop_requested = False
        self._buffers = {"out": b"", "err": b""}
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        for key, value in worker_environment().items():  # incl. PYTORCH_ENABLE_MPS_FALLBACK on macOS
            env.insert(key, value)
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(self.run_dir))
        proc.readyReadStandardOutput.connect(lambda: self._read("out"))
        proc.readyReadStandardError.connect(lambda: self._read("err"))
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self.process = proc
        program, args = worker_command(self.run_dir, resume)
        proc.start(program, args)

    def stop(self) -> None:
        """Ask for a graceful stop (after the current batch); kill if it does not exit in time."""
        if not self.running:
            return
        self._stop_requested = True
        self.process.write(b"stop\n")
        self.process.waitForBytesWritten(1000)
        self._kill_timer.start(self.stop_timeout_ms)
        self.log.emit("중지 요청: 현재 배치가 끝나면 멈춥니다")

    def wait(self, msecs: int = 30_000) -> bool:
        return not self.running or self.process.waitForFinished(msecs)

    def _kill(self) -> None:
        if self.running:
            self.log.emit("응답이 없어 학습 프로세스를 강제 종료합니다")
            self.process.kill()

    def _read(self, which: str) -> None:
        data = (self.process.readAllStandardOutput() if which == "out" else self.process.readAllStandardError()).data()
        buf = self._buffers[which] + data
        *lines, rest = buf.split(b"\n")
        self._buffers[which] = rest
        for raw in lines:
            self._handle_line(raw.decode("utf-8", errors="replace").rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        event = decode_event(line)
        if event is None:
            if line.strip():
                self.log.emit(line)
            return
        if event["type"] in ("done", "stopped", "error"):
            self._final_event = event["type"]
        self.event.emit(event)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log.emit(f"학습 프로세스를 시작할 수 없습니다: {self.process.errorString()}")
            self._finish("failed", "worker failed to start")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        for which in ("out", "err"):
            if self._buffers[which]:
                self._handle_line(self._buffers[which].decode("utf-8", errors="replace"))
                self._buffers[which] = b""
        self._kill_timer.stop()
        state = {"done": "finished", "stopped": "stopped", "error": "failed"}.get(self._final_event)
        if state is not None:
            self.finished.emit(state)
            return
        # Exited without a final event (killed after a stop request, or crashed): record it so
        # the run history stays honest.
        if self._stop_requested:
            self._finish("stopped", "중지 요청 후 응답이 없어 강제 종료됨")
        else:
            self._finish("failed", f"비정상 종료 (exit code {exit_code})")

    def _finish(self, state: str, reason: str) -> None:
        if self.run_dir is not None:
            _, last = weights_paths(self.run_dir)
            write_status(self.run_dir, {"state": state, "error": None if state == "stopped" else reason,
                                        "resumable": last.is_file()})
        self.log.emit(f"학습 프로세스 종료: {reason}")
        self.finished.emit(state)
