"""Shared PyInstaller settings for defectlab_linux.spec and defectlab_windows.spec.

Collected explicitly because PyInstaller cannot see them through static imports:
- ultralytics: model YAMLs (cfg/), assets and its many lazily imported submodules,
- albumentations / pyqtgraph / onnxslim: submodules imported by name at run time,
- onnxruntime and torchvision: their native libraries (torchvision's operator library is not
  found by the stock hook — see native_libs),
- defect_inspection: modules the app imports lazily (augment, crop, compare, report, worker).
torch and PySide6 come through the hooks that ship with PyInstaller / pyinstaller-hooks-contrib.
"""

from pathlib import Path

import importlib.util

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

APP_NAME = "DefectLab"
WORKER_NAME = "DefectLab-worker"
BUNDLE_ID = "com.defectlab.app"

EXCLUDES = [
    "tkinter", "IPython", "jupyter", "notebook", "pytest", "sphinx",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D", "PySide6.QtMultimedia",
    "tensorflow", "tensorboard", "clearml", "comet_ml", "wandb", "mlflow", "dvclive", "neptune",
]


def native_libs(package: str) -> list[tuple[str, str]]:
    """Every compiled library inside a package, kept at its place in the package. torchvision
    loads its operators (nms, roi_align…) from `_C_stable.so` / `_C.pyd` next to its __init__;
    the stock hook does not pick that name up, and without it training fails with
    "operator torchvision::nms does not exist"."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return []
    root = Path(next(iter(spec.submodule_search_locations)))
    out = []
    for pattern in ("*.so", "*.so.*", "*.pyd", "*.dll", "*.dylib"):
        for lib in root.rglob(pattern):
            out.append((str(lib), str(Path(package) / lib.parent.relative_to(root))))
    return out


def collect(spec_dir: str) -> dict:
    root = Path(spec_dir).resolve().parent
    hidden = []
    for pkg in ("ultralytics", "albumentations", "pyqtgraph", "onnxslim", "defect_inspection"):
        hidden += collect_submodules(pkg, filter=lambda name: ".tests" not in name and "opengl" not in name)
    datas = []
    for pkg in ("ultralytics", "albumentations", "pyqtgraph"):
        datas += collect_data_files(pkg, excludes=["**/tests/**", "**/*.md"])
    binaries = collect_dynamic_libs("onnxruntime") + native_libs("torchvision")
    # macOS: CoreML export (Export tab) needs coremltools' Python modules and native libraries
    if importlib.util.find_spec("coremltools") is not None:
        hidden += collect_submodules("coremltools", filter=lambda name: ".test" not in name)
        datas += collect_data_files("coremltools")
        binaries += native_libs("coremltools")
    return {
        "root": str(root),
        "app_script": str(root / "packaging" / "launch_app.py"),
        "worker_script": str(root / "packaging" / "launch_worker.py"),
        "analysis": dict(pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hidden,
                         excludes=EXCLUDES, noarchive=False),
    }
