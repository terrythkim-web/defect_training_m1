"""Augmentation settings (PRD F3.3, F3.4): pure data, no Albumentations import, so project.yaml
can hold them without slowing down project loading (Albumentations takes ~1 s to import).
The transforms themselves are built in core/augment.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class TransformInfo:
    key: str
    label: str  # Korean UI label
    group: str  # 기하 / 광학 / 노이즈·블러 / 환경
    color_only: bool = False  # needs a colour image (disabled for grayscale projects)
    geometric: bool = False


TRANSFORM_INFO: list[TransformInfo] = [
    TransformInfo("hflip", "좌우 뒤집기", "기하", geometric=True),
    TransformInfo("vflip", "상하 뒤집기", "기하", geometric=True),
    TransformInfo("rotate90", "90° 회전", "기하", geometric=True),
    TransformInfo("rotate", "임의 회전", "기하", geometric=True),
    TransformInfo("scale", "확대/축소", "기하", geometric=True),
    TransformInfo("translate", "이동", "기하", geometric=True),
    TransformInfo("perspective", "원근", "기하", geometric=True),
    TransformInfo("brightness_contrast", "밝기/대비", "광학"),
    TransformInfo("gamma", "감마", "광학"),
    TransformInfo("hue_saturation", "색조/채도", "광학", color_only=True),
    TransformInfo("clahe", "CLAHE", "광학"),
    TransformInfo("gauss_noise", "가우시안 노이즈", "노이즈·블러"),
    TransformInfo("motion_blur", "모션 블러", "노이즈·블러"),
    TransformInfo("gaussian_blur", "가우시안 블러", "노이즈·블러"),
    TransformInfo("jpeg", "JPEG 압축", "노이즈·블러"),
    TransformInfo("uneven_light", "조명 불균일", "환경"),
    TransformInfo("specular", "반사광", "환경"),
    TransformInfo("defocus", "초점 이탈", "환경"),
]
TRANSFORM_KEYS = [t.key for t in TRANSFORM_INFO]


def available_transform_keys(channels: int) -> list[str]:
    """Transforms offered for a project; colour-only ones are hidden for grayscale (F1.6)."""
    return [t.key for t in TRANSFORM_INFO if channels == 3 or not t.color_only]


class CopyPasteSetting(BaseModel):
    """F3.3 advanced: paste defect patches from training images onto OK (normal) images."""

    enabled: bool = False
    p: float = Field(default=0.5, ge=0.0, le=1.0)  # chance per augmented copy of an OK image
    max_patches: int = Field(default=2, ge=1, le=10)


# ---- configuration ---------------------------------------------------------------------

class TransformSetting(BaseModel):
    enabled: bool = False
    p: float = Field(default=0.5, ge=0.0, le=1.0)  # probability per generated image
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class AugmentConfig(BaseModel):
    version: int = 1
    transforms: dict[str, TransformSetting] = Field(default_factory=dict)
    multiplier: int = Field(default=20, ge=1, le=500)  # augmented images per training image (F3.2)
    min_visibility: float = Field(default=0.4, ge=0.0, le=1.0)  # F3.6: visible fraction to keep a label
    min_area_px: float = Field(default=16.0, ge=0.0)  # F3.6: smaller labels are dropped
    copy_paste: CopyPasteSetting = Field(default_factory=CopyPasteSetting)

    def setting(self, key: str) -> TransformSetting:
        if key not in TRANSFORM_KEYS:
            raise KeyError(f"unknown transform {key!r}")
        return self.transforms.get(key, TransformSetting())

    def enabled_keys(self) -> list[str]:
        return [k for k in TRANSFORM_KEYS if self.setting(k).enabled]


def _preset(multiplier: int = 20, **settings: tuple[float, float]) -> AugmentConfig:
    """settings: key=(p, strength) for each enabled transform."""
    return AugmentConfig(
        multiplier=multiplier,
        transforms={k: TransformSetting(enabled=True, p=p, strength=s) for k, (p, s) in settings.items()},
    )


BUILTIN_PRESETS: dict[str, AugmentConfig] = {
    "약함": _preset(
        hflip=(0.5, 0.5), vflip=(0.5, 0.5), brightness_contrast=(0.5, 0.2), gauss_noise=(0.2, 0.2),
    ),
    "보통": _preset(
        hflip=(0.5, 0.5), vflip=(0.5, 0.5), rotate90=(0.5, 0.5), rotate=(0.3, 0.3), scale=(0.3, 0.4),
        translate=(0.3, 0.4), brightness_contrast=(0.5, 0.4), gamma=(0.3, 0.4), gauss_noise=(0.3, 0.4),
        gaussian_blur=(0.2, 0.3), jpeg=(0.2, 0.4),
    ),
    "강함": _preset(
        hflip=(0.5, 0.5), vflip=(0.5, 0.5), rotate90=(0.5, 0.5), rotate=(0.5, 0.7), scale=(0.5, 0.7),
        translate=(0.5, 0.6), perspective=(0.3, 0.6), brightness_contrast=(0.7, 0.7), gamma=(0.4, 0.6),
        hue_saturation=(0.3, 0.5), clahe=(0.2, 0.5), gauss_noise=(0.4, 0.7), motion_blur=(0.2, 0.6),
        gaussian_blur=(0.2, 0.5), jpeg=(0.3, 0.7), uneven_light=(0.3, 0.6), defocus=(0.2, 0.5),
    ),
    # Metal: orientation-free, strong lighting changes and reflections, no colour shifts.
    "금속 표면": _preset(
        hflip=(0.5, 0.5), vflip=(0.5, 0.5), rotate90=(0.5, 0.5), rotate=(0.3, 0.4),
        brightness_contrast=(0.6, 0.6), gamma=(0.4, 0.5), clahe=(0.3, 0.5), gauss_noise=(0.3, 0.4),
        uneven_light=(0.5, 0.6), specular=(0.4, 0.6), defocus=(0.2, 0.4),
    ),
    # PCB: boards are imaged aligned, so small rotations only; mild colour and focus changes.
    "PCB": _preset(
        hflip=(0.5, 0.5), vflip=(0.5, 0.5), rotate=(0.2, 0.1), scale=(0.3, 0.3), translate=(0.3, 0.3),
        brightness_contrast=(0.5, 0.4), hue_saturation=(0.3, 0.3), gaussian_blur=(0.2, 0.3),
        jpeg=(0.3, 0.4), defocus=(0.2, 0.3),
    ),
}
DEFAULT_PRESET = "보통"


# ---- user presets (F3.4) ---------------------------------------------------------------

def save_preset(directory: str | Path, name: str, config: AugmentConfig) -> Path:
    """Save a user preset as <directory>/<name>.yaml (UTF-8, readable/editable by hand)."""
    from defect_inspection.platform_utils import atomic_write_text

    name = name.strip()
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise ValueError(f"invalid preset name {name!r}")
    if name in BUILTIN_PRESETS:
        raise ValueError(f"{name!r} is a built-in preset")
    path = Path(directory) / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False))
    return path


def load_presets(directory: str | Path | None = None) -> dict[str, AugmentConfig]:
    """Built-in presets followed by the user's presets (invalid files are skipped)."""
    presets = {k: v.model_copy(deep=True) for k, v in BUILTIN_PRESETS.items()}
    if directory is not None and Path(directory).is_dir():
        for path in sorted(Path(directory).glob("*.yaml")):
            try:
                with open(path, encoding="utf-8") as f:
                    presets[path.stem] = AugmentConfig.model_validate(yaml.safe_load(f) or {})
            except (OSError, ValueError, yaml.YAMLError):
                continue
    return presets
