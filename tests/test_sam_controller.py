"""SamController's worker thread (ui/sam_controller.py) with a fake SamAssistant: status texts,
encode-ahead for small images, dropping superseded requests and reporting errors."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from defect_inspection.core.sam_assist import SAM_WINDOW, SamPrompt
from defect_inspection.ui.sam_controller import SamController


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


class FakeAssistant:
    def __init__(self, model="mobile_sam.pt", weights_dir=None, fail=None):
        self.model, self.weights_dir = model, weights_dir
        self.is_loaded = False
        self.fail = fail
        self.calls: list[str] = []
        self.gate = threading.Event()
        self.gate.set()

    def load(self):
        self.calls.append("load")
        if self.fail == "load":
            raise RuntimeError("download failed")
        self.is_loaded = True

    def prepare(self, image8, key):
        self.gate.wait(5)
        self.calls.append(f"prepare {key}")
        return True

    def needs_encoding(self, image8, key, prompt):
        return key == "new"

    def predict_tiled(self, image8, key, prompt):
        self.calls.append(f"predict {key}")
        if self.fail == "predict":
            raise ValueError("bad prompt")
        return np.ones((4, 4), bool), 0.9


@pytest.fixture
def controller(qapp, tmp_path):
    c = SamController("mobile_sam.pt", tmp_path)
    c.assistant = FakeAssistant(weights_dir=tmp_path)
    c.log = {"status": [], "ready": [], "result": [], "failed": []}
    c.status.connect(c.log["status"].append)
    c.ready.connect(c.log["ready"].append)
    c.result_ready.connect(lambda g, r: c.log["result"].append((g, r)))
    c.failed.connect(lambda g, m: c.log["failed"].append((g, m)))
    yield c
    c.shutdown()


PROMPT = SamPrompt(points=[(2.0, 2.0)], labels=[1])


def test_model_switch_makes_a_new_assistant(controller, tmp_path):
    first = controller.assistant
    assert controller.model == "mobile_sam.pt"
    controller.set_model("mobile_sam.pt")
    assert controller.assistant is first  # same model: keep the loaded one
    controller.set_model("sam2.1_b.pt")
    assert controller.assistant is not first and controller.model == "sam2.1_b.pt"
    assert controller.assistant.weights_dir == tmp_path


def test_prepare_small_image_encodes_ahead(controller):
    controller.prepare(np.zeros((64, 64, 3), np.uint8), "img")
    assert wait_until(lambda: controller.log["ready"] == ["img"])
    assert controller.assistant.calls == ["load", "prepare img"]
    status = controller.log["status"]
    assert "불러오는 중" in status[0] and "이미지 분석 중" in status[1] and "SAM 준비 완료" in status[-1]


def test_prepare_large_image_waits_for_clicks(controller):
    controller.assistant.is_loaded = True
    controller.prepare(np.zeros((8, SAM_WINDOW + 1, 3), np.uint8), "big")
    assert wait_until(lambda: controller.log["status"])
    assert controller.log["ready"] == [] and controller.assistant.calls == []  # loaded already, nothing encoded
    assert str(SAM_WINDOW) in controller.log["status"][0]


def test_prepare_failure_reported(controller):
    controller.assistant.fail = "load"
    controller.prepare(np.zeros((64, 64, 3), np.uint8), "img")
    assert wait_until(lambda: controller.log["failed"])
    assert controller.log["failed"] == [(0, "download failed")]


def test_request_returns_result_with_generation(controller):
    controller.request(3, np.zeros((64, 64, 3), np.uint8), "new", PROMPT)
    assert wait_until(lambda: controller.log["result"])
    generation, (mask, score) = controller.log["result"][0]
    assert generation == 3 and mask.shape == (4, 4) and score == 0.9
    assert any("영역 분석 중" in s for s in controller.log["status"])  # "new" needed an encoding


def test_superseded_request_is_skipped(controller):
    assistant = controller.assistant
    assistant.gate.clear()  # hold the worker inside prepare() while two clicks queue up
    image = np.zeros((64, 64, 3), np.uint8)
    controller.prepare(image, "img")
    controller.request(1, image, "img", PROMPT)
    controller.request(2, image, "img", PROMPT)
    assistant.gate.set()
    assert wait_until(lambda: controller.log["result"])
    assert wait_until(lambda: False, timeout=0.2) is False  # let a late result arrive, if any
    assert [g for g, _ in controller.log["result"]] == [2]
    assert assistant.calls.count("predict img") == 1


def test_predict_failure_reported_with_generation(controller):
    controller.assistant.fail = "predict"
    controller.request(5, np.zeros((64, 64, 3), np.uint8), "img", PROMPT)
    assert wait_until(lambda: controller.log["failed"])
    assert controller.log["failed"] == [(5, "bad prompt")] and controller.log["result"] == []


def test_emit_after_controller_deleted_is_ignored(controller):
    class Dead:
        def emit(self, *args):
            raise RuntimeError("Internal C++ object already deleted")

    controller._emit(Dead(), "x")  # must not raise
