"""Background thumbnail generation with a disk cache, so a 200-image project opens instantly
and only decodes each image once (PRD F1.3)."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from defect_inspection.core.imaging import make_thumbnail, to_display_8bit
from defect_inspection.platform_utils import imread_unicode, imwrite_unicode
from defect_inspection.ui.canvas.image_canvas import numpy_to_qimage

THUMB_SIZE = 64


class ThumbnailLoader(QObject):
    """Emits ready(path, QImage) from worker threads; Qt queues it to the receiver's thread."""

    ready = Signal(str, QImage)

    def __init__(self, cache_dir: Path, parent: QObject | None = None, workers: int = 2) -> None:
        super().__init__(parent)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="thumbs")

    def request(self, path: str | Path) -> None:
        self._pool.submit(self._work, str(path))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def cache_path(self, path: str | Path) -> Path:
        """Keyed by absolute path + mtime + size, so an edited image gets a fresh thumbnail."""
        p = Path(path)
        st = p.stat()
        key = f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}|{THUMB_SIZE}"
        return self.cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.png"

    def _work(self, path: str) -> None:
        try:
            cached = self.cache_path(path)
            thumb = imread_unicode(cached) if cached.is_file() else None
            if thumb is None:
                raw = imread_unicode(path)
                if raw is None:
                    return
                thumb = make_thumbnail(to_display_8bit(raw), THUMB_SIZE)
                imwrite_unicode(cached, thumb)
            self.ready.emit(path, numpy_to_qimage(thumb))
        except (OSError, RuntimeError):
            # File vanished, or the loader was destroyed while we worked.
            pass
