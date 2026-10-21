"""Crop dataset settings (PRD F7.2-F7.4): pure data, stored in project.yaml as `crop`.
The crop generation itself lives in core/crop.py."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CropConfig(BaseModel):
    version: int = 1
    # Sources of crop windows; each can be switched on independently.
    use_roi: bool = True  # manual ROIs from crops.json (F7.1)
    use_defect: bool = True  # around each confirmed label, position jittered (F7.2)
    use_background: bool = True  # random windows without defects (F7.2)
    use_tiles: bool = False  # sliding tiles over the whole image (F7.2)

    crop_size: int = Field(default=640, ge=32, le=8192)  # defect / background windows (square)
    per_defect: int = Field(default=2, ge=1, le=50)  # windows per label
    jitter: float = Field(default=0.5, ge=0.0, le=1.0)  # 0 = defect centred, 1 = anywhere it still fits
    background_ratio: float = Field(default=1.0, ge=0.0, le=10.0)  # background : defect windows

    tile_size: int = Field(default=640, ge=32, le=8192)
    overlap_unit: Literal["percent", "px"] = "percent"
    overlap_percent: float = Field(default=20.0, ge=0.0, lt=100.0)
    overlap_px: int = Field(default=32, ge=0)

    multiplier: int = Field(default=5, ge=0, le=500)  # augmented copies per Train crop (0 = none)

    def overlap_pixels(self) -> int:
        """Tile overlap in pixels (always smaller than the tile)."""
        px = self.overlap_px if self.overlap_unit == "px" else round(self.tile_size * self.overlap_percent / 100)
        return max(0, min(int(px), self.tile_size - 1))

    def any_source(self) -> bool:
        return self.use_roi or self.use_defect or self.use_background or self.use_tiles


CROP_PRESETS: dict[str, CropConfig] = {
    "기본 (결함 중심 640)": CropConfig(),
    # ATT-YOLO paper: 5000x5000 originals cut into 1280 tiles overlapping by 32 px.
    "ATT-YOLO (1280 타일, overlap 32px)": CropConfig(
        use_roi=False, use_defect=False, use_background=False, use_tiles=True,
        tile_size=1280, overlap_unit="px", overlap_px=32, multiplier=2,
    ),
    "슬라이딩 타일 (640, 20%)": CropConfig(
        use_roi=False, use_defect=False, use_background=False, use_tiles=True, tile_size=640, multiplier=3,
    ),
    "수동 ROI만": CropConfig(use_defect=False, use_background=False),
}
DEFAULT_CROP_PRESET = "기본 (결함 중심 640)"


class TilingParams(BaseModel):
    """How a crop model is applied to a whole image (F7.6): tiles of the size it was trained on,
    overlapping so a defect cut by one tile border is whole in the neighbour. Recorded in the run
    (RunSpec.crop) when training starts and in the ONNX metadata (F6.4)."""

    tile_size: int = Field(ge=16, le=8192)
    overlap_px: int = Field(ge=0)
    merge_iou: float = Field(default=0.5, ge=0.0, le=1.0)  # cross-tile NMS: same class and IoU >= this
    merge_ios: float = Field(default=0.7, ge=0.0, le=1.0)  # ... or a box mostly inside another (cut piece)
    crop: dict = Field(default_factory=dict)  # crop settings the dataset was built with

    @classmethod
    def from_window(cls, size: int, overlap_px: int | None, crop: dict | None = None) -> TilingParams:
        """Window of build_info.yaml; crops made without tiles get a 20 % overlap."""
        overlap = round(size * 0.2) if overlap_px is None else overlap_px
        return cls(tile_size=size, overlap_px=min(max(0, overlap), size - 1), crop=crop or {})
