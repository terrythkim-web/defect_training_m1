"""OS-dependent helpers. Keep every Linux/Windows/macOS difference in this module (PRD 5.1, 5.2),
including the choice of compute device (CUDA GPU, Apple MPS or CPU)."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from defect_inspection import APP_NAME

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
# Mac keyboards have no Delete key (only ⌫ = Backspace; forward delete is fn+⌫), so there ⌫ also
# deletes the selected label/ROI when there is no suggestion to reject (PRD 5.2).
BACKSPACE_DELETES_SELECTION = IS_MAC
# macOS hides scrollbars until you scroll, so a panel clipped sideways there just looks cut off.
OVERLAY_SCROLLBARS = IS_MAC
# CoreML (.mlpackage) export and its verification need macOS (coremltools' converter and runtime).
COREML_EXPORT_AVAILABLE = IS_MAC

# Input image formats (PRD F1.2). Compared case-insensitively, so .JPG/.JPEG also match.
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Korean-capable fonts for report figures, in order of preference per OS.
_KOREAN_FONTS = {
    "win": ["Malgun Gothic", "NanumGothic", "Gulim"],
    "mac": ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"],
    "other": ["Noto Sans CJK KR", "NanumGothic", "Noto Sans KR", "UnDotum"],
}


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """cv2.imread replacement that handles non-ASCII paths on Windows.

    Returns None when the file is missing or cannot be decoded, like cv2.imread.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    """cv2.imwrite replacement that handles non-ASCII paths on Windows."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text with LF newlines via temp file + os.replace, so a crash never leaves
    a half-written file (os.replace is atomic on both Linux and Windows)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def config_dir() -> Path:
    """Per-user config directory (Linux ~/.config/DefectLab, Windows %APPDATA%/DefectLab,
    macOS ~/Library/Application Support/DefectLab)."""
    from PySide6.QtCore import QStandardPaths

    # macOS: GenericConfigLocation is ~/Library/Preferences, which is meant for plist files only
    location = (QStandardPaths.StandardLocation.GenericDataLocation if IS_MAC
                else QStandardPaths.StandardLocation.GenericConfigLocation)
    base = QStandardPaths.writableLocation(location)
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Per-user cache directory for thumbnails and downloaded weights (Linux ~/.cache/DefectLab,
    Windows %LOCALAPPDATA%/cache/DefectLab, macOS ~/Library/Caches/DefectLab)."""
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericCacheLocation)
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def use_persistent_matplotlib_cache() -> None:
    """PyInstaller builds: keep matplotlib's font cache in <cache>/matplotlib. PyInstaller's runtime
    hook points MPLCONFIGDIR at a fresh temp folder on every start, so each app start and each
    training worker rebuilt the font cache (~15-20 s on an M1; once it stalled for minutes).
    Call before matplotlib is imported; no-op when running from source."""
    if not getattr(sys, "frozen", False):
        return
    path = cache_dir() / "matplotlib"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(path)


# ---- compute device (PRD 5.2: one place decides CUDA / MPS / CPU) -----------------------------
# Ultralytics only picks CUDA by itself; on a Mac it stays on the CPU unless "mps" is passed.
# Everything that runs a model asks resolve_device() instead of leaving the device empty.

_accelerator_cache: list[str | None] = []


def torch_accelerator() -> str | None:
    """"cuda", "mps" or None. Imports torch (~1 s) once; call it off the UI thread."""
    if _accelerator_cache:
        return _accelerator_cache[0]
    found = None
    try:
        import torch

        if torch.cuda.is_available():
            found = "cuda"
        else:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available() and _mps_works(torch):
                found = "mps"
    except ImportError:
        pass
    _accelerator_cache.append(found)
    return found


def _mps_works(torch) -> bool:
    """is_available() is not enough: GitHub's macos-14 runners (virtual GPU) report MPS but every
    allocation fails with "MPS backend out of memory (MPS allocated: 0 bytes …)". Run one tiny op."""
    try:
        x = torch.arange(6, dtype=torch.float32, device="mps").reshape(2, 3)
        return (x @ x.T).cpu().tolist() == [[5.0, 14.0], [14.0, 50.0]]
    except RuntimeError:
        return False


def resolve_device(requested: str | None = "auto", purpose: str = "predict") -> str:
    """Ultralytics `device` for a model run.

    requested: "auto" (or None/""), or an explicit "cpu" / "0" / "0,1" / "mps", which wins.
    purpose: "train", "predict", "sam" or "verify". "verify" (ONNX export check) is always "cpu":
    it compares PyTorch with ONNX Runtime on the CPU, and a GPU's different arithmetic would make
    the two disagree for reasons unrelated to the export.
    """
    if requested not in (None, "", "auto"):
        return str(requested)
    if purpose == "verify":
        return "cpu"
    accel = torch_accelerator()
    return {"cuda": "0", "mps": "mps"}.get(accel or "", "cpu")


def gpu_device_choice() -> tuple[str, str]:
    """(label, device) of the GPU entry in the Train tab's device list."""
    return ("Apple GPU (MPS)", "mps") if IS_MAC else ("GPU 0", "0")


def is_cuda_label(text: str) -> bool:
    """True for a detect_device() label of an NVIDIA GPU (FP16 export needs CUDA, not MPS)."""
    return text.startswith("GPU") and "MPS" not in text


def worker_environment() -> dict[str, str]:
    """Extra environment variables for the training worker process."""
    env = {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}  # Korean log text on Windows consoles
    if IS_MAC:
        # operators MPS does not implement run on the CPU instead of failing (slower, but works)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def detect_device() -> str:
    """Human-readable compute device, e.g. "GPU: NVIDIA RTX 4090", "GPU: Apple MPS" or "CPU".

    Imports torch (~1 s), so call it off the UI thread.
    """
    try:
        import torch
    except ImportError:
        return "CPU (PyTorch 없음)"
    accel = torch_accelerator()
    if accel == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if accel == "mps":
        return "GPU: Apple MPS"
    return "CPU"


WORKER_EXE = "DefectLab-worker"  # console exe next to the windowed app in Windows builds


def frozen_worker_command(args: list[str]) -> tuple[str, list[str]]:
    """Training worker command in a PyInstaller build. Windows builds ship a separate console
    executable for the worker: the windowed app exe may get no usable stdout, and the worker
    talks to the app through stdout. QProcess starts it without a console window. Elsewhere (or
    if it is missing) the app exe itself is started with --train-worker (handled in app.py)."""
    exe = Path(sys.executable)
    worker = exe.with_name(WORKER_EXE + (".exe" if IS_WINDOWS else ""))
    if worker.is_file() and worker != exe:
        return str(worker), list(args)
    return str(exe), ["--train-worker", *args]


def inference_devices() -> list[tuple[str, str]]:
    """(label, Ultralytics device) pairs to time inference on: always CPU, plus the GPU if any
    (CUDA, or Apple MPS). Imports torch, so call it off the UI thread."""
    devices = [("CPU", "cpu")]
    accel = torch_accelerator()
    if accel == "cuda":
        devices.append(("GPU", "0"))
    elif accel == "mps":
        devices.append(("GPU", "mps"))
    return devices


# File names that work on both Linux and Windows (PRD 5.1, F6.3).
FORBIDDEN_FILENAME_CHARS = '\\/:*?"<>|'
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                          *(f"LPT{i}" for i in range(1, 10))}
MAX_PATH = 260


def sanitize_filename_part(text: str, replacement: str = "_") -> str:
    """Make text safe inside a file name: forbidden and control characters -> replacement,
    surrounding spaces/dots trimmed. Korean and other letters are kept."""
    out = "".join(replacement if (c in FORBIDDEN_FILENAME_CHARS or ord(c) < 32) else c for c in text)
    return out.strip(" .") or replacement


def invalid_filename_reason(name: str) -> str | None:
    """Why `name` (a bare file name) is not allowed on Windows/Linux, or None if it is fine."""
    if not name or not name.strip():
        return "파일 이름이 비어 있습니다"
    bad = sorted({c for c in name if c in FORBIDDEN_FILENAME_CHARS or ord(c) < 32})
    if bad:
        return "사용할 수 없는 문자: " + " ".join(repr(c)[1:-1] if ord(c) < 32 else c for c in bad)
    if name != name.rstrip(" ."):
        return "파일 이름은 공백이나 마침표로 끝날 수 없습니다"
    if name.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        return f"Windows 예약어는 파일 이름으로 쓸 수 없습니다: {name.split('.')[0]}"
    return None


def path_too_long(path: str | Path) -> bool:
    """True if the absolute path exceeds the classic Windows limit (260 characters)."""
    return len(str(Path(path).absolute())) >= MAX_PATH


def default_dataloader_workers() -> int:
    """DataLoader worker processes for training. Windows spawns each worker as a fresh process
    (slow start, high memory), so keep it low there (PRD 5.1). macOS also spawns, and training
    on MPS runs with 0 workers anyway (Ultralytics forces it), so 0 (PRD 5.2)."""
    cpus = os.cpu_count() or 1
    if IS_MAC:
        return 0
    return min(2, cpus) if IS_WINDOWS else min(8, max(1, cpus - 1))


def korean_font_candidates() -> list[str]:
    return _KOREAN_FONTS["win" if IS_WINDOWS else "mac" if IS_MAC else "other"]


def pick_korean_font() -> str | None:
    """Return the first installed Korean-capable font for matplotlib, or None with a warning."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in korean_font_candidates():
        if name in installed:
            return name
    warnings.warn(
        "No Korean font found for report figures; Korean text may not render. "
        f"Install one of: {', '.join(korean_font_candidates())}",
        stacklevel=2,
    )
    return None
