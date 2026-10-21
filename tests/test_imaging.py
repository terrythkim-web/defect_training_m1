import numpy as np
import pytest

from defect_inspection.core.imaging import is_grayscale, to_grayscale


def test_bgr_to_gray_uses_luminance_weights():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 1] = 200  # pure green
    gray = to_grayscale(bgr)
    assert gray.shape == (2, 2)
    assert gray[0, 0] == round(0.587 * 200)


def test_bgra_to_gray():
    assert to_grayscale(np.full((3, 4, 4), 100, np.uint8)).shape == (3, 4)


def test_16bit_depth_is_kept():
    bgr = np.full((4, 4, 3), 40000, np.uint16)
    gray = to_grayscale(bgr)
    assert gray.dtype == np.uint16
    assert gray[0, 0] == 40000


def test_gray_inputs_pass_through():
    g = np.zeros((5, 5), np.uint8)
    assert to_grayscale(g) is g
    assert to_grayscale(g[..., None]).shape == (5, 5)
    assert is_grayscale(g) and is_grayscale(g[..., None])
    assert not is_grayscale(np.zeros((5, 5, 3), np.uint8))


def test_unsupported_shape():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 2, 2), np.uint8))
