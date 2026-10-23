"""Runs core/image_groups on one background thread and reports the groups as a Qt signal."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from defect_inspection.core.image_groups import ImageSignature, find_source_groups, image_signature


class GroupChecker(QObject):
    """Each check gets a generation number; a newer check cancels an older one, and results of
    older generations are never emitted. Signatures (1/8 gray copies) are cached per file
    path + mtime + size, so re-checking after adding a few images reads only the new ones."""

    finished = Signal(int, object)  # generation, {member: source}

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="groups")
        self._lock = threading.Lock()
        self._latest = 0
        self._future: Future | None = None
        self._cache: dict[Path, tuple[tuple[float, int], ImageSignature | None]] = {}

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def check(self, items: list[tuple[str, Path]]) -> int:
        """Start a check of (project path, absolute path) pairs; returns its generation."""
        with self._lock:
            self._latest += 1
            generation = self._latest
        self._future = self._pool.submit(self._run, generation, list(items))
        return generation

    def wait(self, timeout: float | None = 120.0) -> tuple[int, dict[str, str]] | None:
        """Block until the latest check is done; its (generation, groups) or None."""
        future = self._future
        if future is None:
            return None
        try:
            return future.result(timeout)
        except Exception:  # noqa: BLE001 - timeout or a failed check: build without groups
            return None

    def shutdown(self) -> None:
        with self._lock:
            self._latest += 1  # cancels a running check
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- worker thread --------------------------------------------------------------
    def _cancelled(self, generation: int) -> bool:
        with self._lock:
            return generation != self._latest

    def _signature(self, path: str, abspath: Path) -> ImageSignature | None:
        try:
            st = abspath.stat()
        except OSError:
            return None
        key = (st.st_mtime, st.st_size)
        cached = self._cache.get(abspath)
        if cached is not None and cached[0] == key:
            sig = cached[1]
            if sig is not None:
                sig.path = path
            return sig
        sig = image_signature(path, abspath)
        self._cache[abspath] = (key, sig)
        return sig

    def _run(self, generation: int, items: list[tuple[str, Path]]) -> tuple[int, dict[str, str]] | None:
        sigs = []
        for path, abspath in items:
            if self._cancelled(generation):
                return None
            sig = self._signature(path, abspath)
            if sig is not None:
                sigs.append(sig)
        groups = find_source_groups(sigs, cancel=lambda: self._cancelled(generation))
        if groups is None or self._cancelled(generation):
            return None
        try:
            self.finished.emit(generation, groups)
        except RuntimeError:  # checker destroyed while working
            pass
        return generation, groups
