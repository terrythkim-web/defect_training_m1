"""Display helpers, image cache and thumbnails (Qt-free parts of PROMPTS step 5)."""

import threading

import numpy as np
import pytest

from defect_inspection.core.image_cache import ImageCache
from defect_inspection.core.imaging import (
    apply_brightness_contrast,
    brightness_contrast_lut,
    make_thumbnail,
    to_display_8bit,
)
from defect_inspection.platform_utils import imwrite_unicode


# ---- to_display_8bit -----------------------------------------------------------

def test_8bit_passes_through_unchanged():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert to_display_8bit(img) is img


def test_16bit_is_stretched_to_full_range():
    # A faint 16-bit image (values 1000..1255) would look black if simply divided by 257.
    img = (np.arange(256, dtype=np.uint16) + 1000).reshape(16, 16)
    out = to_display_8bit(img)
    assert out.dtype == np.uint8
    assert out.min() <= 2 and out.max() >= 253


def test_16bit_flat_image():
    assert to_display_8bit(np.full((4, 4), 500, np.uint16)).max() == 0


def test_16bit_percentile_ignores_hot_pixel():
    img = np.tile(np.arange(1000, 2000, dtype=np.uint16), (100, 1))
    img[0, 0] = 65535  # single outlier must not compress the rest
    out = to_display_8bit(img)
    assert out[50, 500] in range(110, 145)


def test_bgra_and_single_channel_shapes():
    assert to_display_8bit(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4, 3)
    assert to_display_8bit(np.zeros((4, 4, 1), np.uint8)).shape == (4, 4)
    assert to_display_8bit(np.zeros((4, 4, 3), np.uint16)).shape == (4, 4, 3)


# ---- brightness / contrast -----------------------------------------------------

def test_identity_lut():
    assert np.array_equal(brightness_contrast_lut(0, 0), np.arange(256, dtype=np.uint8))


def test_brightness_raises_and_lowers():
    img = np.full((2, 2), 100, np.uint8)
    assert apply_brightness_contrast(img, 50, 0)[0, 0] > 100
    assert apply_brightness_contrast(img, -50, 0)[0, 0] < 100


def test_contrast_spreads_around_mid_gray():
    img = np.array([[64, 192]], np.uint8)
    out = apply_brightness_contrast(img, 0, 60)
    assert out[0, 0] < 64 and out[0, 1] > 192
    flat = apply_brightness_contrast(img, 0, -100)
    assert flat[0, 0] == flat[0, 1] == 128


def test_adjustment_does_not_modify_input():
    img = np.full((3, 3, 3), 77, np.uint8)
    apply_brightness_contrast(img, 80, 80)
    assert (img == 77).all()


def test_thumbnail_keeps_aspect():
    assert make_thumbnail(np.zeros((400, 800, 3), np.uint8), 64).shape == (32, 64, 3)
    small = np.zeros((10, 10), np.uint8)
    assert make_thumbnail(small, 64) is small


# ---- ImageCache ----------------------------------------------------------------

@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "캐시 테스트"
    d.mkdir()
    for i in range(5):
        imwrite_unicode(d / f"이미지 {i}.png", np.full((20, 30, 3), i * 40, np.uint8))
    imwrite_unicode(d / "deep.tif", (np.arange(600, dtype=np.uint16) * 50).reshape(20, 30))
    return d


def test_cache_get_and_lru_eviction(image_dir):
    cache = ImageCache(capacity=2)
    paths = [image_dir / f"이미지 {i}.png" for i in range(3)]
    for p in paths:
        assert cache.get(p).shape == (20, 30, 3)
    assert not cache.contains(paths[0])
    assert cache.contains(paths[1]) and cache.contains(paths[2])
    cache.shutdown()


def test_cache_returns_display_8bit(image_dir):
    cache = ImageCache()
    img = cache.get(image_dir / "deep.tif")
    assert img.dtype == np.uint8 and img.ndim == 2
    cache.shutdown()


def test_cache_missing_file(image_dir):
    cache = ImageCache()
    assert cache.get(image_dir / "없음.png") is None
    cache.shutdown()


def test_prefetch_loads_in_background(image_dir):
    cache = ImageCache(capacity=8)
    paths = [image_dir / f"이미지 {i}.png" for i in range(4)]
    cache.prefetch(paths)
    # get() on a pending path waits for that prefetch instead of decoding twice
    assert all(cache.get(p) is not None for p in paths)
    assert all(cache.contains(p) for p in paths)
    cache.shutdown()


def test_cache_is_thread_safe(image_dir):
    cache = ImageCache(capacity=3)
    paths = [image_dir / f"이미지 {i}.png" for i in range(5)]
    errors = []

    def worker():
        try:
            for _ in range(20):
                for p in paths:
                    assert cache.get(p) is not None
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    cache.shutdown()


def test_get_async_cached_runs_synchronously(image_dir):
    cache = ImageCache()
    path = image_dir / "이미지 0.png"
    cache.get(path)
    got = []
    assert cache.get_async(path, lambda p, img: got.append((p, img.shape)))
    assert got == [(path, (20, 30, 3))]
    cache.shutdown()


def test_get_async_uncached_calls_back_from_worker(image_dir):
    cache = ImageCache()
    done = threading.Event()
    got = []

    def cb(p, img):
        got.append((p.name, None if img is None else img.shape, threading.current_thread().name))
        done.set()

    assert not cache.get_async(image_dir / "이미지 1.png", cb)
    assert done.wait(5)
    name, shape, thread = got[0]
    assert (name, shape) == ("이미지 1.png", (20, 30, 3))
    assert thread.startswith("image-cache")
    cache.shutdown()


def test_get_async_missing_file_gives_none(image_dir):
    cache = ImageCache()
    done = threading.Event()
    got = []
    cache.get_async(image_dir / "없음.png", lambda p, img: (got.append(img), done.set()))
    assert done.wait(5) and got == [None]
    cache.shutdown()
