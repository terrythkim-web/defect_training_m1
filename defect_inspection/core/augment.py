"""Offline augmentation (PRD F3.1, F3.3, F3.4, F3.6).

Every shape - boxes included - goes through Albumentations as polygon keypoints. After the
transform each shape is clipped to the image; it is dropped when too little of it is still
visible or when it became too small. Boxes come back as the bounding box of their transformed
corners (as YOLO expects after rotation/perspective), polygons stay exact.

Coordinates: labels use continuous pixel coordinates (a full-image box is 0..W). Albumentations
keypoints use pixel indices (a flip maps x to W-1-x), so points are shifted by -0.5 before and
+0.5 after the transform; without that every flipped label would be off by one pixel.

Images must be uint8 (gray HxW or BGR HxWx3); 16-bit sources are converted when the dataset is
built (PROMPTS step 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import albumentations as A
import numpy as np

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon, fit_shape
from defect_inspection.core.augment_config import (  # noqa: F401 (re-exported)
    BUILTIN_PRESETS,
    DEFAULT_PRESET,
    TRANSFORM_INFO,
    TRANSFORM_KEYS,
    AugmentConfig,
    CopyPasteSetting,
    TransformSetting,
    available_transform_keys,
    load_presets,
    save_preset,
)

Shape = BBox | Polygon


# ---- transform registry --------------------------------------------------------------

@dataclass(frozen=True)
class TransformSpec:
    key: str
    label: str  # Korean UI label
    group: str  # 기하 / 광학 / 노이즈·블러 / 환경
    build: Callable[[float], A.BasicTransform]  # strength in [0, 1] -> transform (p set later)
    color_only: bool = False  # needs a colour image (disabled for grayscale projects)
    geometric: bool = False


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


TRANSFORMS: list[TransformSpec] = [
    # 기하
    TransformSpec("hflip", "좌우 뒤집기", "기하", lambda s: A.HorizontalFlip(), geometric=True),
    TransformSpec("vflip", "상하 뒤집기", "기하", lambda s: A.VerticalFlip(), geometric=True),
    TransformSpec("rotate90", "90° 회전", "기하", lambda s: A.RandomRotate90(), geometric=True),
    TransformSpec("rotate", "임의 회전", "기하",
                  lambda s: A.Affine(rotate=(-45 * s, 45 * s), border_mode=0), geometric=True),
    TransformSpec("scale", "확대/축소", "기하",
                  lambda s: A.Affine(scale=(1 - 0.4 * s, 1 + 0.4 * s), border_mode=0), geometric=True),
    TransformSpec("translate", "이동", "기하",
                  lambda s: A.Affine(translate_percent=(-0.2 * s, 0.2 * s), border_mode=0), geometric=True),
    TransformSpec("perspective", "원근", "기하",
                  lambda s: A.Perspective(scale=(0.01, 0.01 + 0.09 * s), keep_size=True, border_mode=0),
                  geometric=True),
    # 광학
    TransformSpec("brightness_contrast", "밝기/대비", "광학",
                  lambda s: A.RandomBrightnessContrast(brightness_limit=0.05 + 0.35 * s,
                                                       contrast_limit=0.05 + 0.35 * s)),
    TransformSpec("gamma", "감마", "광학",
                  lambda s: A.RandomGamma(gamma_limit=(100 - 40 * s, 100 + 40 * s))),
    TransformSpec("hue_saturation", "색조/채도", "광학",
                  lambda s: A.HueSaturationValue(hue_shift_limit=20 * s, sat_shift_limit=30 * s,
                                                 val_shift_limit=0), color_only=True),
    TransformSpec("clahe", "CLAHE", "광학", lambda s: A.CLAHE(clip_limit=(1, 1 + 3 * s))),
    # 노이즈·블러
    TransformSpec("gauss_noise", "가우시안 노이즈", "노이즈·블러",
                  lambda s: A.GaussNoise(std_range=(0.005, 0.01 + 0.07 * s))),
    TransformSpec("motion_blur", "모션 블러", "노이즈·블러",
                  lambda s: A.MotionBlur(blur_limit=(3, _odd(3 + 8 * s)))),
    TransformSpec("gaussian_blur", "가우시안 블러", "노이즈·블러",
                  lambda s: A.GaussianBlur(blur_limit=(3, _odd(3 + 6 * s)))),
    TransformSpec("jpeg", "JPEG 압축", "노이즈·블러",
                  lambda s: A.ImageCompression(quality_range=(int(95 - 60 * s), 100))),
    # 환경
    TransformSpec("uneven_light", "조명 불균일", "환경",
                  lambda s: A.Illumination(mode="linear", intensity_range=(0.01, 0.02 + 0.18 * s))),
    TransformSpec("specular", "반사광", "환경",
                  lambda s: A.Illumination(mode="gaussian", effect_type="brighten",
                                           intensity_range=(0.01, 0.02 + 0.18 * s))),  # library max 0.2
    TransformSpec("defocus", "초점 이탈", "환경", lambda s: A.Defocus(radius=(1, 1 + max(1, int(4 * s))))),
]
assert [t.key for t in TRANSFORMS] == TRANSFORM_KEYS, "keep augment.py and augment_config.py in sync"


# ---- pipeline ------------------------------------------------------------------------

def available_transforms(channels: int) -> list[TransformSpec]:
    """Transforms offered for a project; colour-only ones are hidden for grayscale (F1.6)."""
    keys = set(available_transform_keys(channels))
    return [t for t in TRANSFORMS if t.key in keys]


@dataclass
class AugmentResult:
    image: np.ndarray
    annotations: ImageAnnotations
    dropped: int  # labels removed by the visibility/size rule


class Augmenter:
    """Applies one AugmentConfig to images with their confirmed labels."""

    def __init__(self, config: AugmentConfig, channels: int = 3, seed: int | None = None) -> None:
        self.config = config
        self.channels = channels
        steps = []
        for spec in available_transforms(channels):
            s = config.setting(spec.key)
            if s.enabled and s.p > 0:
                t = spec.build(s.strength)
                t.p = s.p
                steps.append(t)
        self.keys = [t.key for t in available_transforms(channels) if config.setting(t.key).enabled]
        self.pipeline = A.Compose(
            steps,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            seed=seed,
        )

    def __call__(self, image: np.ndarray, annotations: ImageAnnotations, image_name: str | None = None) -> AugmentResult:
        if image.dtype != np.uint8:
            raise ValueError(f"augmentation needs uint8 images, got {image.dtype}")
        if self.channels == 1 and image.ndim == 3:
            raise ValueError("grayscale project but the image has colour channels")
        h, w = image.shape[:2]
        if (w, h) != (annotations.width, annotations.height):
            raise ValueError(f"image size {(w, h)} does not match labels {(annotations.width, annotations.height)}")

        shapes = annotations.confirmed()  # suggestions are never training data
        keypoints, owners = [], []
        for i, shape in enumerate(shapes):
            for x, y in shape.to_points():
                keypoints.append((x - 0.5, y - 0.5))  # continuous -> pixel-index convention
                owners.append(i)

        out = self.pipeline(image=image, keypoints=keypoints)
        new_image = out["image"]
        if image.ndim == 2 and new_image.ndim == 3:
            new_image = new_image[:, :, 0]
        nh, nw = new_image.shape[:2]

        moved: list[list[tuple[float, float]]] = [[] for _ in shapes]
        for (x, y), owner in zip(out["keypoints"], owners):
            moved[owner].append((float(x) + 0.5, float(y) + 0.5))

        kept, dropped = [], 0
        for shape, pts in zip(shapes, moved):
            new = self._fit(shape, pts, nw, nh)
            if new is None:
                dropped += 1
            else:
                kept.append(new)
        result = ImageAnnotations(image=image_name or annotations.image, width=nw, height=nh, shapes=kept)
        return AugmentResult(new_image, result, dropped)

    def _fit(self, shape: Shape, points: list[tuple[float, float]], w: int, h: int) -> Shape | None:
        """Clip a transformed shape to the image and apply the F3.6 keep/drop rule."""
        return fit_shape(shape, points, w, h, self.config.min_visibility, self.config.min_area_px)
