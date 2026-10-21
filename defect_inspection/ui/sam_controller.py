"""Runs SamAssistant on one background thread and reports results as Qt signals."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from defect_inspection.core.sam_assist import SamAssistant, SamPrompt


class SamController(QObject):
    """One worker thread, so encode/predict calls never overlap. Each request carries a
    generation number; results for older generations are dropped (user clicked again)."""

    result_ready = Signal(int, object)  # generation, (mask, score) | None
    failed = Signal(int, str)  # generation, error message
    status = Signal(str)  # progress text for the status bar
    ready = Signal(str)  # image key whose embedding is cached

    def __init__(self, model: str, weights_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.assistant = SamAssistant(model, weights_dir)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam")
        self._latest = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.assistant.model

    def set_model(self, model: str) -> None:
        if model != self.assistant.model:
            self.assistant = SamAssistant(model, self.assistant.weights_dir)

    def prepare(self, image8: np.ndarray, key: str) -> None:
        """Load the model and encode the image ahead of the first click (~seconds on CPU)."""
        self._pool.submit(self._run_prepare, self.assistant, image8, key)

    def request(self, generation: int, image8: np.ndarray, key: str, prompt: SamPrompt) -> None:
        with self._lock:
            self._latest = generation
        self._pool.submit(self._run_predict, self.assistant, generation, image8, key, prompt)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- worker thread --------------------------------------------------------------
    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # controller destroyed while working
            pass

    def _ensure(self, assistant: SamAssistant, image8: np.ndarray, key: str) -> None:
        if not assistant.is_loaded:
            self._emit(self.status, self.tr("SAM 모델 불러오는 중… (처음 한 번은 다운로드)"))
            assistant.load()
        if assistant._image_key != key:
            self._emit(self.status, self.tr("SAM 이미지 분석 중…"))
            assistant.set_image(image8, key)
            self._emit(self.ready, key)

    def _run_prepare(self, assistant: SamAssistant, image8: np.ndarray, key: str) -> None:
        try:
            self._ensure(assistant, image8, key)
            self._emit(self.status, self.tr("SAM 준비 완료: 클릭(포함) / 우클릭(제외) / 드래그(박스)"))
        except Exception as e:  # noqa: BLE001 - reported to the user
            self._emit(self.failed, 0, str(e))

    def _run_predict(self, assistant: SamAssistant, generation: int, image8: np.ndarray, key: str,
                     prompt: SamPrompt) -> None:
        with self._lock:
            if generation != self._latest:
                return  # superseded before it started
        try:
            self._ensure(assistant, image8, key)
            result = assistant.predict(prompt)
        except Exception as e:  # noqa: BLE001 - reported to the user
            self._emit(self.failed, generation, str(e))
            return
        self._emit(self.result_ready, generation, result)
