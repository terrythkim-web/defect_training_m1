"""Model weights location shared by SAM, YOLO11 and ATT-YOLO backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def resolve_weights(model: str, weights_dir: Path,
                    download: Callable[[str], str] | None = None) -> Path:
    """Local path of a weights file. An existing file path is used as is (offline PCs); a known
    Ultralytics asset name is downloaded once into weights_dir (never into the working directory)."""
    p = Path(model)
    if p.is_file():
        return p
    target = Path(weights_dir) / p.name
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if download is None:
        from ultralytics.utils.downloads import attempt_download_asset as download
    return Path(download(str(target)))
