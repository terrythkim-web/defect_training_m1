import io
import os
import sys

import numpy as np
import pytest

from defect_inspection import platform_utils as pu


@pytest.fixture
def unicode_dir(tmp_path):
    # Spaces and Korean characters: cv2.imread/imwrite fail on these paths on Windows.
    d = tmp_path / "결함 이미지 폴더"
    d.mkdir()
    return d


@pytest.mark.parametrize("suffix", [".png", ".bmp", ".tif"])
def test_imwrite_imread_roundtrip_unicode_path(unicode_dir, suffix):
    image = (np.arange(48 * 64 * 3) % 256).astype(np.uint8).reshape(48, 64, 3)
    path = unicode_dir / f"샘플 01{suffix}"

    assert pu.imwrite_unicode(path, image)
    loaded = pu.imread_unicode(path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, image)


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".JPG", ".JPEG"])
def test_jpeg_roundtrip_unicode_path(unicode_dir, suffix):
    # JPEG is lossy: use a smooth gradient and compare with a tolerance.
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    image = np.dstack([np.tile(ramp, (48, 1))] * 3)
    path = unicode_dir / f"샘플 01{suffix}"

    assert pu.imwrite_unicode(path, image)
    loaded = pu.imread_unicode(path)

    assert loaded is not None
    assert loaded.shape == image.shape
    assert np.abs(loaded.astype(int) - image.astype(int)).max() <= 8


@pytest.mark.parametrize(
    "name, expected",
    [
        ("a.png", True), ("a.jpg", True), ("a.jpeg", True), ("a.JPEG", True), ("a.Jpg", True),
        ("a.bmp", True), ("a.tif", True), ("a.TIFF", True),
        ("a.gif", False), ("a.txt", False), ("jpeg", False),
    ],
)
def test_is_supported_image(name, expected):
    assert pu.is_supported_image(name) is expected


def test_imread_keeps_16bit_grayscale(unicode_dir):
    image = np.linspace(0, 65535, 32 * 32, dtype=np.uint16).reshape(32, 32)
    path = unicode_dir / "그레이 16bit.tif"

    assert pu.imwrite_unicode(path, image)
    loaded = pu.imread_unicode(path)

    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, image)


def test_imread_missing_file_returns_none(unicode_dir):
    assert pu.imread_unicode(unicode_dir / "없는 파일.png") is None


def test_config_and_cache_dirs_exist():
    assert pu.config_dir().is_dir()
    assert pu.cache_dir().is_dir()
    assert pu.config_dir().name == "DefectLab"


def test_frozen_build_keeps_matplotlib_cache(monkeypatch, tmp_path):
    # PyInstaller's runtime hook sets a fresh temp MPLCONFIGDIR per start (font cache rebuilt each time)
    monkeypatch.setattr(pu, "cache_dir", lambda: tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/_MEI_temp")
    pu.use_persistent_matplotlib_cache()
    assert pu.os.environ["MPLCONFIGDIR"] == "/tmp/_MEI_temp"  # from source: untouched
    monkeypatch.setattr(pu.sys, "frozen", True, raising=False)
    pu.use_persistent_matplotlib_cache()
    assert pu.os.environ["MPLCONFIGDIR"] == str(tmp_path / "matplotlib")
    assert (tmp_path / "matplotlib").is_dir()


def test_korean_font_candidates_match_os():
    candidates = pu.korean_font_candidates()
    if pu.IS_WINDOWS:
        assert candidates[0] == "Malgun Gothic"
    else:
        assert "Noto Sans CJK KR" in candidates


# ---- macOS and device selection (PROMPTS step 21, PRD 5.2) ----------------------------------------

@pytest.fixture
def accelerator(monkeypatch):
    """Pretend the machine has the given torch accelerator ("cuda", "mps" or None)."""
    from defect_inspection import platform_utils as pu

    def set_to(value):
        monkeypatch.setattr(pu, "_accelerator_cache", [value])

    return set_to


@pytest.mark.parametrize("op_works, expected", [(True, "mps"), (False, None)])
def test_mps_needs_a_working_tensor_op(monkeypatch, op_works, expected):
    # GitHub's macos-14 runners report MPS as available, but every allocation fails with
    # "MPS backend out of memory (MPS allocated: 0 bytes …)"; then everything must use the CPU
    import sys
    import types

    from defect_inspection import platform_utils as pu

    class FakeTensor:
        def reshape(self, *_):
            return self

        def __matmul__(self, _):
            return self

        @property
        def T(self):  # noqa: N802
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[5.0, 14.0], [14.0, 50.0]]

    def arange(*_a, **_k):
        if not op_works:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 0 bytes)")
        return FakeTensor()

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        arange=arange, float32="float32")
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(pu, "_accelerator_cache", [])
    assert pu.torch_accelerator() == expected


@pytest.mark.parametrize("accel, train, predict", [("cuda", "0", "0"), ("mps", "mps", "mps"), (None, "cpu", "cpu")])
def test_auto_device_per_machine(accelerator, accel, train, predict):
    from defect_inspection.platform_utils import inference_devices, resolve_device

    accelerator(accel)
    assert resolve_device("auto", "train") == train
    assert resolve_device(None, "predict") == predict and resolve_device("", "sam") == predict
    assert resolve_device("auto", "verify") == "cpu"  # ONNX check always compares on the CPU
    assert resolve_device("cpu", "train") == "cpu" and resolve_device("mps", "verify") == "mps"  # explicit wins
    assert [d for _, d in inference_devices()] == ["cpu"] + ([train] if accel else [])


def test_mac_fonts_workers_and_gpu_choice(monkeypatch):
    from defect_inspection import platform_utils as pu

    monkeypatch.setattr(pu, "IS_MAC", True)
    monkeypatch.setattr(pu, "IS_WINDOWS", False)
    assert pu.korean_font_candidates()[0] == "Apple SD Gothic Neo"
    assert pu.default_dataloader_workers() == 0
    assert pu.gpu_device_choice() == ("Apple GPU (MPS)", "mps")
    assert pu.worker_environment()["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    monkeypatch.setattr(pu, "IS_MAC", False)
    assert pu.gpu_device_choice() == ("GPU 0", "0")
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in pu.worker_environment()
    assert pu.worker_environment()["PYTHONUTF8"] == "1"


def test_cuda_label():
    from defect_inspection.platform_utils import is_cuda_label

    assert is_cuda_label("GPU: NVIDIA RTX 4090")
    assert not is_cuda_label("GPU: Apple MPS") and not is_cuda_label("CPU")


def test_mps_fallback_set_before_torch():
    import os

    import defect_inspection  # noqa: F401

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_predictor_and_sam_use_resolved_device(accelerator, monkeypatch):
    from defect_inspection.core import evaluate as E
    from defect_inspection.core.sam_assist import SamAssistant

    accelerator("mps")
    monkeypatch.setattr("ultralytics.YOLO", lambda *a, **k: object())
    assert E.Predictor("x.pt", "detect", 64).device == "mps"
    assert E.Predictor("x.pt", "detect", 64, device="cpu").device == "cpu"
    assert SamAssistant().requested_device is None  # resolved to mps on load()


def test_accelerator_probe_runs_once_across_threads(monkeypatch):
    # several threads asked at start-up and ran the first MPS op at once: Metal aborted the process
    import threading
    import time

    from defect_inspection import platform_utils as pu

    calls = []

    def slow_probe():
        calls.append(1)
        time.sleep(0.05)
        return "mps"

    monkeypatch.setattr(pu, "_accelerator_cache", [])
    monkeypatch.setattr(pu, "_probe_accelerator", slow_probe)
    results = []
    threads = [threading.Thread(target=lambda: results.append(pu.torch_accelerator())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1] and results == ["mps"] * 8


# ---- stdin stop requests without a blocking pipe read (Windows) ---------------------------------

def test_polled_lines_split_reads_into_lines():
    """Windows reads the worker's stdin pipe only when bytes are waiting: a thread blocked in
    ReadFile on the pipe made spawning DataLoader workers (they inherit it) hang forever."""
    r, w = os.pipe()
    os.write(w, "noise\r\nST".encode() + "OP\n끝".encode())
    os.close(w)
    waits = iter([0, 9, 0, 0, 6, None])  # nothing yet, then the bytes in two chunks, then EOF
    slept = []
    lines = list(pu._polled_lines(r, lambda: next(waits), 0.2, slept.append))
    os.close(r)
    assert lines == ["noise\n", "STOP\n", "끝"]
    assert slept == [0.2, 0.2, 0.2]


def test_stdin_lines_reads_a_plain_stream():
    assert list(pu.stdin_lines(io.StringIO("a\nstop\n"))) == ["a\n", "stop\n"]


def test_stdin_lines_without_stdin(monkeypatch):
    monkeypatch.setattr(pu.sys, "stdin", None)
    assert list(pu.stdin_lines()) == []


def run_script(tmp_path, code):
    import subprocess

    script = tmp_path / "script.py"
    script.write_text(code, encoding="utf-8")
    return subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


SPAWN_WHILE_WATCHING = """
import multiprocessing, os, threading, time
from defect_inspection import platform_utils

if __name__ == "__main__":
    threading.Thread(target=lambda: list(platform_utils.stdin_lines()), daemon=True).start()
    time.sleep(0.5)  # the watcher is now waiting for stdin
    child = multiprocessing.get_context("spawn").Process(target=os.getpid)
    child.start()
    child.join(60)
    print("child exit", child.exitcode, flush=True)
"""


def test_spawning_while_stdin_is_watched(tmp_path):
    """The training worker's case: a thread waits for a stop line on the stdin PIPE (kept open)
    while DataLoader workers are spawned. With a blocking read, Windows never started the child."""
    import subprocess

    proc = run_script(tmp_path, SPAWN_WHILE_WATCHING)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("spawning a process hung while stdin was being read")
    finally:
        proc.stdin.close()
    assert proc.stdout.read().strip() == "child exit 0"


@pytest.mark.skipif(not pu.IS_WINDOWS, reason="PeekNamedPipe is Windows only")
def test_windows_pipe_is_polled(tmp_path):
    proc = run_script(tmp_path, "import sys\nfrom defect_inspection import platform_utils as p\n"
                                "print(p._pipe_peeker(sys.stdin) is not None, flush=True)\n"
                                "print(list(p.stdin_lines()), flush=True)\n")
    try:
        assert proc.stdout.readline().strip() == "True"
        proc.stdin.write("x\nstop\n")
        proc.stdin.close()
        assert proc.stdout.readline().strip() == "['x\\n', 'stop\\n']"
    finally:
        proc.kill()
        proc.wait(timeout=30)
