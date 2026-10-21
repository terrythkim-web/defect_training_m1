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
