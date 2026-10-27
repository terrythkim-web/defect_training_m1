"""DefectLab: defect labeling, augmentation, YOLO11 training and ONNX export."""

import os

# Ultralytics pip-installs missing packages at run time by default (e.g. during export). An
# offline line PC would fail and the app must not change its own environment: dependencies
# are declared in pyproject.toml instead. Child processes (training worker) inherit this.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# macOS (PRD 5.2): PyTorch operators that MPS does not implement fall back to the CPU instead of
# raising. Must be set before torch is imported; harmless elsewhere (only read on macOS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.3"
APP_NAME = "DefectLab"
