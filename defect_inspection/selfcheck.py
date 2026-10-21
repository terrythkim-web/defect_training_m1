"""`DefectLab --self-check`: verify an installation (PyInstaller build or source) without a display.

Checks that every heavy dependency imports, that Ultralytics' model YAMLs were bundled (a YOLO11n
model is built from YAML, offline), that torchvision's compiled NMS works, that ONNX Runtime runs, that an ONNX export works, that a Korean font is available for
report figures, and that the main window can be created (offscreen). Prints one line per check
and returns 0 only if all passed — used by the build scripts and the install checklist in README.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable


def _check_imports() -> str:
    import albumentations
    import cv2
    import jinja2  # noqa: F401
    import matplotlib
    import onnx
    import onnxruntime
    import PySide6
    import pyqtgraph  # noqa: F401
    import torch
    import ultralytics

    return (f"torch {torch.__version__}, ultralytics {ultralytics.__version__}, onnx {onnx.__version__}, "
            f"onnxruntime {onnxruntime.__version__}, albumentations {albumentations.__version__}, "
            f"opencv {cv2.__version__}, PySide6 {PySide6.__version__}, matplotlib {matplotlib.__version__}")


def _check_model() -> str:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    from defect_inspection.core.att_yolo import att_yolo_config

    model = YOLO("yolo11n.yaml", task="detect")  # needs Ultralytics' bundled cfg files
    n = sum(p.numel() for p in model.model.parameters())
    att = DetectionModel(att_yolo_config("detect", 1, 3, "n"), verbose=False)  # P2-P7 heads + C3TR
    m = sum(p.numel() for p in att.parameters())
    return f"yolo11n {n / 1e6:.1f}M, att-yolo-n {m / 1e6:.1f}M parameters"


def _check_torch_ops() -> str:
    """torchvision's compiled operators: NMS runs in every Ultralytics validation and prediction."""
    import torch
    import torchvision

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2], keep
    return f"torchvision {torchvision.__version__} nms"


def _check_onnxruntime() -> str:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 9)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out = session.run(None, {"x": np.array([[-1.0, 0.0, 2.0]], np.float32)})[0]
    assert out.tolist() == [[0.0, 0.0, 2.0]]
    return f"providers {', '.join(ort.get_available_providers())}"


def _check_export() -> str:
    """Export an (untrained) YOLO11n to ONNX the way the Export tab does (onnx + onnxslim) and run it."""
    import numpy as np
    import onnxruntime as ort
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)  # a YAML-built model is exported into the working directory
        try:
            onnx_path = Path(YOLO("yolo11n.yaml", task="detect").export(
                format="onnx", imgsz=64, simplify=True, device="cpu", verbose=False))
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            out = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 64, 64), np.float32)})[0]
            size = onnx_path.stat().st_size
        finally:
            os.chdir(cwd)
    return f"yolo11n → ONNX {size / 1e6:.1f} MB, 출력 {list(out.shape)}"


def _check_font() -> str:
    from defect_inspection.core.report import setup_matplotlib

    font = setup_matplotlib()
    if not font:
        raise RuntimeError("한글 폰트 없음 — 리포트 그래프의 한글이 깨집니다 (README 참고)")
    return font


def _check_device() -> str:
    from defect_inspection.platform_utils import detect_device, resolve_device

    return f"{detect_device()} → 자동 선택 시 학습·추론 장치 '{resolve_device('auto', 'train')}'"


def _check_apple() -> str:
    """macOS: a tensor operation on the Mac GPU (MPS) and the CoreML converter (Export tab)."""
    import coremltools

    from defect_inspection.platform_utils import torch_accelerator

    detail = f"coremltools {coremltools.__version__}"
    if torch_accelerator() == "mps":  # probes a real MPS tensor op
        return detail + ", MPS 연산 OK"
    return detail + ", MPS 없음 또는 사용 불가 (가상 머신이거나 Intel Mac) — CPU로 동작"


def _check_window() -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from defect_inspection.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        window = MainWindow(QSettings(str(Path(tmp) / "s.ini"), QSettings.Format.IniFormat), cache_root=Path(tmp))
        tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        window.train_panel.shutdown()
        window.close()
        app.processEvents()
    return "탭 " + " / ".join(tabs)


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("라이브러리", _check_imports),
    ("모델 생성", _check_model),
    ("torchvision 연산", _check_torch_ops),
    ("ONNX Runtime", _check_onnxruntime),
    ("ONNX 내보내기", _check_export),
    ("한글 폰트", _check_font),
    ("계산 장치", _check_device),
    ("메인 창", _check_window),
]
if sys.platform == "darwin":
    CHECKS.insert(-1, ("Mac (MPS·CoreML)", _check_apple))


def run_self_check() -> int:
    from defect_inspection import __version__

    frozen = "PyInstaller" if getattr(sys, "frozen", False) else "source"
    print(f"DefectLab {__version__} self-check ({frozen}, Python {sys.version.split()[0]}, {sys.platform})", flush=True)
    failed = 0
    for name, check in CHECKS:
        t0 = time.monotonic()
        try:
            detail = check()
            print(f"[OK]   {name}: {detail} ({time.monotonic() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 - every failure is reported
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=3)
    print("모든 확인 통과" if not failed else f"{failed}개 실패", flush=True)
    return 0 if not failed else 1
