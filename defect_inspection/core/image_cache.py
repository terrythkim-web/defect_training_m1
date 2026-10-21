"""Decoded-image LRU cache with background prefetch, so moving between images (A/D) does not
wait on decoding large files (PRD 5: 4K images, UI response < 100 ms)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from defect_inspection.core.imaging import to_display_8bit
from defect_inspection.platform_utils import imread_unicode


class ImageCache:
    """Caches display-ready (uint8) images keyed by path. Thread-safe.

    cv2 decoding releases the GIL, so prefetching on worker threads runs truly in parallel
    with the UI thread.
    """

    def __init__(self, capacity: int = 8, workers: int = 2) -> None:
        self.capacity = capacity
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._pending: dict[Path, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-cache")

    def get(self, path: str | Path) -> np.ndarray | None:
        """Display image for path, decoding it now if it is not cached or being prefetched.
        None if the file cannot be read."""
        path = Path(path)
        with self._lock:
            if path in self._items:
                self._items.move_to_end(path)
                return self._items[path]
            future = self._pending.get(path)
        if future is not None:
            return future.result()
        return self._load(path)

    def get_async(self, path: str | Path, callback) -> bool:
        """Deliver the image to callback(path, image) without blocking the caller.

        Returns True if it was cached and callback already ran synchronously. Otherwise the
        callback runs later on a worker thread (UI code should re-emit it as a Qt signal).
        """
        path = Path(path)
        with self._lock:
            if path in self._items:
                self._items.move_to_end(path)
                image = self._items[path]
            else:
                image = None
                future = self._pending.get(path)
                if future is None:
                    future = self._pool.submit(self._load, path)
                    self._pending[path] = future
        if image is not None:
            callback(path, image)
            return True
        future.add_done_callback(lambda f: callback(path, None if f.cancelled() else f.result()))
        return False

    def prefetch(self, paths: list[str | Path]) -> None:
        for path in map(Path, paths):
            with self._lock:
                if path in self._items or path in self._pending:
                    continue
                self._pending[path] = self._pool.submit(self._load, path)

    def contains(self, path: str | Path) -> bool:
        with self._lock:
            return Path(path) in self._items

    def invalidate(self, path: str | Path | None = None) -> None:
        with self._lock:
            if path is None:
                self._items.clear()
            else:
                self._items.pop(Path(path), None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _load(self, path: Path) -> np.ndarray | None:
        raw = imread_unicode(path)
        image = None if raw is None else to_display_8bit(raw)
        with self._lock:
            self._pending.pop(path, None)
            if image is not None:
                self._items[path] = image
                self._items.move_to_end(path)
                while len(self._items) > self.capacity:
                    self._items.popitem(last=False)
        return image
