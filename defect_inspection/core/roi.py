"""Manual crop regions (PRD F7.1): ROIs per image plus common ROIs for every image (fixed jig),
stored in <project>/crops.json. Qt-free.

Coordinates are the label convention: continuous pixel coordinates, top-left origin. Every
geometry helper returns a new Roi clamped to the image.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from defect_inspection.platform_utils import atomic_write_text

CROPS_FILE = "crops.json"
MIN_ROI = 8.0  # px: smaller drags are treated as clicks
FIXED_SIZES = (320, 640)


def _new_id() -> str:
    return uuid.uuid4().hex


class Roi(BaseModel):
    id: str = Field(default_factory=_new_id)
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _normalize(self) -> Roi:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            raise ValueError("ROI must have positive width and height")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def bounds(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


class RoiOptions(BaseModel):
    """How new ROIs are drawn (remembered per project)."""

    fixed_size: int | None = Field(default=None, ge=16, le=8192)  # None = free drag
    square: bool = False


class CropsFile(BaseModel):
    version: int = 1
    options: RoiOptions = Field(default_factory=RoiOptions)
    common: list[Roi] = Field(default_factory=list)  # applied to every image
    images: dict[str, list[Roi]] = Field(default_factory=dict)  # image path -> its own ROIs


# ---- geometry ------------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def fixed_roi_at(cx: float, cy: float, size: int, width: int, height: int) -> Roi:
    """A size x size ROI centred on (cx, cy), shifted to stay inside the image (and shrunk if
    the image is smaller than `size`)."""
    w, h = min(size, width), min(size, height)
    x1 = _clamp(cx - w / 2, 0, width - w)
    y1 = _clamp(cy - h / 2, 0, height - h)
    return Roi(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)


def roi_from_drag(x1: float, y1: float, x2: float, y2: float, width: int, height: int,
                  options: RoiOptions | None = None) -> Roi | None:
    """New ROI from a mouse drag (or click, for fixed size). None if too small."""
    options = options or RoiOptions()
    if options.fixed_size:
        return fixed_roi_at((x1 + x2) / 2 if abs(x2 - x1) >= MIN_ROI else x1,
                            (y1 + y2) / 2 if abs(y2 - y1) >= MIN_ROI else y1,
                            options.fixed_size, width, height)
    x1, y1 = _clamp(x1, 0, width), _clamp(y1, 0, height)
    x2, y2 = _clamp(x2, 0, width), _clamp(y2, 0, height)
    if options.square:
        side = max(abs(x2 - x1), abs(y2 - y1))
        sx, sy = (1 if x2 >= x1 else -1), (1 if y2 >= y1 else -1)
        # largest square from the anchor that stays inside the image
        side = min(side, (width - x1) if sx > 0 else x1, (height - y1) if sy > 0 else y1)
        x2, y2 = x1 + sx * side, y1 + sy * side
    if abs(x2 - x1) < MIN_ROI or abs(y2 - y1) < MIN_ROI:
        return None
    return Roi(x1=x1, y1=y1, x2=x2, y2=y2)


def move_roi(roi: Roi, dx: float, dy: float, width: int, height: int) -> Roi:
    dx = _clamp(dx, -roi.x1, width - roi.x2)
    dy = _clamp(dy, -roi.y1, height - roi.y2)
    return roi.model_copy(update={"x1": roi.x1 + dx, "y1": roi.y1 + dy, "x2": roi.x2 + dx, "y2": roi.y2 + dy})


def resize_roi(roi: Roi, corner: int, x: float, y: float, width: int, height: int, square: bool = False) -> Roi:
    """Drag one corner (0 TL, 1 TR, 2 BR, 3 BL); the opposite corner stays. Raises ValueError if
    the result would be smaller than MIN_ROI."""
    corners = [(roi.x1, roi.y1), (roi.x2, roi.y1), (roi.x2, roi.y2), (roi.x1, roi.y2)]
    ox, oy = corners[(corner + 2) % 4]
    x, y = _clamp(x, 0, width), _clamp(y, 0, height)
    if square:
        side = max(abs(x - ox), abs(y - oy))
        sx, sy = (1 if x >= ox else -1), (1 if y >= oy else -1)
        side = min(side, (width - ox) if sx > 0 else ox, (height - oy) if sy > 0 else oy)
        x, y = ox + sx * side, oy + sy * side
    if abs(x - ox) < MIN_ROI or abs(y - oy) < MIN_ROI:
        raise ValueError("ROI too small")
    return Roi.model_validate({**roi.model_dump(), "x1": ox, "y1": oy, "x2": x, "y2": y})


def clip_roi(roi: Roi, width: int, height: int) -> Roi | None:
    """The part of an ROI inside an image of this size (common ROIs on a smaller image)."""
    x1, y1 = _clamp(roi.x1, 0, width), _clamp(roi.y1, 0, height)
    x2, y2 = _clamp(roi.x2, 0, width), _clamp(roi.y2, 0, height)
    if x2 - x1 < MIN_ROI or y2 - y1 < MIN_ROI:
        return None
    return roi.model_copy(update={"x1": x1, "y1": y1, "x2": x2, "y2": y2})


# ---- storage ------------------------------------------------------------------------------

class RoiStore:
    """crops.json of a project. Every change is written immediately (the file is small)."""

    def __init__(self, project_root: str | Path) -> None:
        self.path = Path(project_root) / CROPS_FILE
        self.data = self._load()

    def _load(self) -> CropsFile:
        if not self.path.is_file():
            return CropsFile()
        with open(self.path, encoding="utf-8") as f:
            return CropsFile.model_validate(json.load(f))

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(self.data.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n")

    @property
    def options(self) -> RoiOptions:
        return self.data.options

    def set_options(self, options: RoiOptions) -> None:
        self.data.options = options
        self.save()

    def image_rois(self, image: str) -> list[Roi]:
        return list(self.data.images.get(image, []))

    @property
    def common(self) -> list[Roi]:
        return list(self.data.common)

    def set_rois(self, image: str, image_rois: list[Roi], common: list[Roi]) -> None:
        if image_rois:
            self.data.images[image] = list(image_rois)
        else:
            self.data.images.pop(image, None)
        self.data.common = list(common)
        self.save()

    def effective(self, image: str, width: int, height: int) -> list[Roi]:
        """ROIs used for cropping this image: common ones (clipped to it) + its own."""
        common = [r for r in (clip_roi(c, width, height) for c in self.data.common) if r is not None]
        return common + self.image_rois(image)

    def has_any(self) -> bool:
        return bool(self.data.common) or any(self.data.images.values())

    def forget_image(self, image: str) -> None:
        if self.data.images.pop(image, None) is not None:
            self.save()
