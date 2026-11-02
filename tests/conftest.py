import os

import pytest

# Headless Qt for CI (Linux and Windows) and local test runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _cpu_only_main_window(monkeypatch):
    """MainWindow detects the compute device on a thread and enables GPU-only options (FP16 export,
    the Train tab's GPU entry) when the signal arrives. UI tests assume a CPU machine and fake a GPU
    with set_detected_device("GPU: …"); on a machine with a real CUDA GPU the late real result made
    them fail. platform_utils.detect_device itself is still tested in test_app.py."""
    monkeypatch.setattr("defect_inspection.ui.main_window.detect_device", lambda: "CPU")
