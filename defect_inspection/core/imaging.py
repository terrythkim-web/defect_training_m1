"""Pixel-level helpers shared by import, dataset build and inference."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel (H, W) image, keeping the bit depth (8 or 16 bit).

    Accepts gray (H, W) / (H, W, 1), BGR (H, W, 3) and BGRA (H, W, 4) as loaded by OpenCV.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape {image.shape}")


def is_grayscale(image: np.ndarray) -> bool:
    return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)


def to_display_8bit(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Convert any loaded image to uint8 gray (H, W) or BGR (H, W, 3) for on-screen display only.

    8-bit images pass through (alpha dropped). 16-bit and float images are contrast-stretched
    between the given percentiles so faint detail stays visible; stored data is never changed.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.dtype == np.uint8:
        return image
    sample = image[::4, ::4] if image.size > 1_000_000 else image  # percentiles on a subsample
    lo, hi = np.percentile(sample, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
        if hi <= lo:
            return np.zeros(image.shape, np.uint8)
    scaled = (image.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def brightness_contrast_lut(brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """256-entry LUT. brightness and contrast are in [-100, 100]; 0/0 is the identity."""
    b = np.clip(brightness, -100, 100) * 2.55
    c = np.clip(contrast, -100, 100)
    factor = (100 + c) / 100 if c <= 0 else 100 / (100 - c * 0.99)
    x = np.arange(256, dtype=np.float32)
    return np.clip((x - 128) * factor + 128 + b, 0, 255).astype(np.uint8)


def apply_brightness_contrast(image8: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Display-only adjustment of a uint8 image (labels and stored pixels are unaffected)."""
    if brightness == 0 and contrast == 0:
        return image8
    return cv2.LUT(image8, brightness_contrast_lut(brightness, contrast))


def make_thumbnail(image8: np.ndarray, max_side: int = 96) -> np.ndarray:
    h, w = image8.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image8
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image8, size, interpolation=cv2.INTER_AREA)


def to_training_8bit(image: np.ndarray, channels: int) -> np.ndarray:
    """Pixels for a training dataset: uint8 with the project's channel count.

    16-bit images are scaled by value >> 8 (the same mapping for every image, so relative
    brightness between images is kept; unlike the display stretch, which is per image).
    Float images in [0, 1] are scaled by 255, otherwise clipped. BGRA drops alpha.
    channels=1 converts colour to gray, channels=3 converts gray to BGR.
    """
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        img = image.astype(np.float32)
        if img.size and float(img.max()) <= 1.0:
            img = img * 255.0
        image = np.clip(img, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if channels == 1:
        return to_grayscale(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
