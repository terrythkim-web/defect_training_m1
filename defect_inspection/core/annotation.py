"""Label data: BBox/Polygon shapes in original pixel coordinates, JSON storage and YOLO txt
conversion (PRD F2.6, F2.7).

Coordinates are continuous pixel coordinates with the origin at the top-left corner of the
image, so a shape covering the whole image is (0, 0)-(width, height).

JSON is the source of truth. YOLO txt is always derived from it for the project's task:
- detect: "cls cx cy w h" (normalized). Polygons become their bounding box.
- seg:    "cls x1 y1 x2 y2 ..." (normalized). Boxes become a 4-point polygon.

Auto-label suggestions (PRD F2.8) live in the same JSON with status="suggested". They are
never written to YOLO txt, so they never reach training until the user accepts them.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from defect_inspection.platform_utils import atomic_write_text

LABEL_SCHEMA_VERSION = 1
TaskMode = Literal["detect", "seg"]
ShapeStatus = Literal["confirmed", "suggested"]
ShapeSource = Literal["manual", "model", "sam"]

_PRECISION = 6  # decimals in YOLO txt


def _new_id() -> str:
    return uuid.uuid4().hex


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("coordinates must be finite")
    return v


class _ShapeBase(BaseModel):
    id: str = Field(default_factory=_new_id)
    class_id: int = Field(ge=0)
    status: ShapeStatus = "confirmed"
    source: ShapeSource = "manual"
    score: float | None = Field(default=None, ge=0.0, le=1.0)  # model confidence of a suggestion

    @property
    def is_suggestion(self) -> bool:
        return self.status == "suggested"


class BBox(_ShapeBase):
    type: Literal["bbox"] = "bbox"
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _normalize(self) -> BBox:
        for v in (self.x1, self.y1, self.x2, self.y2):
            _finite(v)
        # Accept boxes dragged in any direction.
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            raise ValueError("bbox must have positive width and height")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        return self

    def bounds(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def to_points(self) -> list[tuple[float, float]]:
        """Corners clockwise from top-left."""
        return [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)]


class Polygon(_ShapeBase):
    type: Literal["polygon"] = "polygon"
    points: list[tuple[float, float]]

    @field_validator("points")
    @classmethod
    def _check_points(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(pts) < 3:
            raise ValueError("polygon needs at least 3 points")
        for x, y in pts:
            _finite(x)
            _finite(y)
        if polygon_area(pts) <= 0:
            raise ValueError("polygon must have a non-zero area")
        return pts

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def to_points(self) -> list[tuple[float, float]]:
        return list(self.points)


class Mask(_ShapeBase):
    """Brush-painted pixel mask (F2.9), holes kept: run lengths (`mask_codec.rle_encode`, row-major,
    starting with a background run) over its tight pixel box x, y, w, h. `bounds()` is that box;
    `to_points()` is the outer outline along pixel edges, so code that needs a polygon (YOLO txt, keypoint
    augmentation, drawing) gets one — holes reach only mask-aware code (brush, mask PNGs, U-Net++).
    OpenCV is imported on first use only: loading a project must stay light."""

    type: Literal["mask"] = "mask"
    x: int
    y: int
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    rle: list[int]
    _cache: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check_rle(self) -> Mask:
        if any(v < 0 for v in self.rle) or sum(self.rle) != self.w * self.h:
            raise ValueError("mask run lengths must be >= 0 and cover the box exactly")
        if not any(self.rle[1::2]):
            raise ValueError("mask has no painted pixel")
        return self

    @classmethod
    def from_pixels(cls, pixels, origin: tuple[int, int] = (0, 0), **fields) -> Mask | None:
        """Mask of the True pixels of a 2-D array whose top-left pixel is `origin` (None if empty)."""
        import numpy as np

        from defect_inspection.core.mask_codec import rle_encode

        arr = np.asarray(pixels, bool)
        rows, cols = np.flatnonzero(arr.any(axis=1)), np.flatnonzero(arr.any(axis=0))
        if rows.size == 0:
            return None
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        return cls(x=origin[0] + x0, y=origin[1] + y0, w=x1 - x0, h=y1 - y0,
                   rle=rle_encode(arr[y0:y1, x0:x1]), **fields)

    def _cached(self, name: str, make):
        """Per-pixel results cached against the current `rle` object (a copy with other pixels has
        another list, so it never sees this one's cache)."""
        hit = self._cache.get(name)
        if hit is None or hit[0] is not self.rle or hit[1] != (self.w, self.h):
            hit = (self.rle, (self.w, self.h), make())
            self._cache[name] = hit
        return hit[2]

    def pixels(self):
        """bool array (h, w) of the box; do not modify it (shared cache)."""
        from defect_inspection.core.mask_codec import rle_decode

        def decode():
            arr = rle_decode(self.rle, self.w, self.h)
            arr.flags.writeable = False  # shared cache: a write would corrupt every copy
            return arr

        return self._cached("pixels", decode)

    @property
    def area(self) -> int:
        return int(sum(self.rle[1::2]))

    def full_pixels(self, width: int, height: int):
        """The mask in a `width` x `height` image (pixels outside it dropped)."""
        import numpy as np

        out = np.zeros((height, width), bool)
        x0, y0 = max(self.x, 0), max(self.y, 0)
        x1, y1 = min(self.x + self.w, width), min(self.y + self.h, height)
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1] = self.pixels()[y0 - self.y:y1 - self.y, x0 - self.x:x1 - self.x]
        return out

    def bounds(self) -> tuple[float, float, float, float]:
        return float(self.x), float(self.y), float(self.x + self.w), float(self.y + self.h)

    def to_points(self) -> list[tuple[float, float]]:
        from defect_inspection.core.mask_codec import mask_outline

        local = self._cached("outline", lambda: mask_outline(self.pixels()) or [
            (0.0, 0.0), (float(self.w), 0.0), (float(self.w), float(self.h)), (0.0, float(self.h))])
        return [(x + self.x, y + self.y) for x, y in local]

    def pieces(self) -> list[Polygon]:
        """One outline polygon per connected piece (largest first), in image coordinates."""
        from defect_inspection.core.mask_codec import outer_outlines

        local = self._cached("pieces", lambda: outer_outlines(self.pixels()))
        out = []
        for ring in local:
            try:
                out.append(Polygon(class_id=self.class_id, points=[(x + self.x, y + self.y) for x, y in ring]))
            except ValueError:
                continue  # degenerate piece
        return out

    def moved(self, dx: int, dy: int) -> Mask:
        """Shifted by whole pixels (same id and pixels)."""
        return self.model_copy(update={"x": self.x + int(dx), "y": self.y + int(dy)})


Shape = Annotated[Union[BBox, Polygon, Mask], Field(discriminator="type")]


class ImageAnnotations(BaseModel):
    """All labels of one image, as stored in labels/<image stem>.json."""

    version: int = LABEL_SCHEMA_VERSION
    image: str  # image path relative to the project's images/ folder
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    shapes: list[Shape] = Field(default_factory=list)

    def confirmed(self) -> list[BBox | Polygon]:
        return [s for s in self.shapes if not s.is_suggestion]

    def suggestions(self) -> list[BBox | Polygon]:
        return [s for s in self.shapes if s.is_suggestion]

    def to_yolo_lines(self, task: TaskMode) -> list[str]:
        """Confirmed shapes only: suggestions are not training data until accepted. A Mask in pieces
        gives one seg line per piece (its box covers all pieces in detect)."""
        lines = []
        for s in self.confirmed():
            if isinstance(s, Mask) and task == "seg":
                lines += [line for piece in s.pieces()
                          if (line := shape_to_yolo(piece, self.width, self.height, task))]
            elif line := shape_to_yolo(s, self.width, self.height, task):
                lines.append(line)
        return lines

    # ---- suggestions (PRD F2.8) ---------------------------------------------
    def add_suggestions(self, shapes: list[BBox | Polygon], source: ShapeSource, replace: bool = True) -> None:
        """Add shapes as pending suggestions. By default earlier suggestions from the same source
        are replaced, so re-running a model does not pile up duplicates."""
        if replace:
            self.clear_suggestions(source)
        for shape in shapes:
            shape.status, shape.source = "suggested", source
            self.shapes.append(shape)

    def accept(self, ids: list[str]) -> int:
        """Turn suggestions into labels. Source and score are kept for traceability."""
        count = 0
        for shape in self.suggestions():
            if shape.id in ids:
                shape.status = "confirmed"
                count += 1
        return count

    def accept_above(self, min_score: float) -> int:
        return self.accept([s.id for s in self.suggestions() if s.score is not None and s.score >= min_score])

    def reject(self, ids: list[str]) -> int:
        """Drop suggestions. Confirmed labels are never removed here."""
        before = len(self.shapes)
        self.shapes = [s for s in self.shapes if not (s.is_suggestion and s.id in ids)]
        return before - len(self.shapes)

    def clear_suggestions(self, source: ShapeSource | None = None) -> int:
        return self.reject([s.id for s in self.suggestions() if source is None or s.source == source])


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Absolute shoelace area."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def clip_polygon(points: list[tuple[float, float]], w: float, h: float) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a polygon to the image rectangle [0, w] x [0, h]."""

    def clip(pts, inside, intersect):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
        return out

    def at_x(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def at_y(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    pts = list(points)
    for inside, intersect in (
        (lambda p: p[0] >= 0, at_x(0)),
        (lambda p: p[0] <= w, at_x(w)),
        (lambda p: p[1] >= 0, at_y(0)),
        (lambda p: p[1] <= h, at_y(h)),
    ):
        if not pts:
            break
        pts = clip(pts, inside, intersect)
    return pts


def fit_mask(mask: Mask, dx: int, dy: int, w: int, h: int, min_visibility: float,
             min_area_px: float) -> Mask | None:
    """`mask` shifted by whole pixels and clipped to the w x h image, like `fit_shape`."""
    full = mask.area
    moved = mask.moved(dx, dy)
    clipped = Mask.from_pixels(moved.full_pixels(int(w), int(h)),
                               **moved.model_dump(include={"id", "class_id", "status", "source", "score"}))
    if clipped is None or clipped.area / full < min_visibility or clipped.area < min_area_px:
        return None
    if clipped.bounds() == moved.bounds():
        return moved  # nothing cut off: keep the same pixel list (and its caches)
    return clipped


def fit_shape(shape: BBox | Polygon | Mask, points: list[tuple[float, float]], w: float, h: float,
              min_visibility: float, min_area_px: float) -> BBox | Polygon | Mask | None:
    """`shape` moved to `points` (new coordinates), clipped to [0, w] x [0, h]; None when less than
    `min_visibility` of it is still visible or it is smaller than `min_area_px` (F3.6 / F7.3).
    A Mask can only be shifted here (taken from its first outline point); warped masks go through
    the augmenter's mask path."""
    if isinstance(shape, Mask):
        ox, oy = shape.to_points()[0]
        return fit_mask(shape, round(points[0][0] - ox), round(points[0][1] - oy), int(w), int(h),
                        min_visibility, min_area_px)
    full_area = polygon_area(points)
    clipped = clip_polygon(points, w, h)
    if len(clipped) < 3 or full_area <= 0:
        return None
    visible = polygon_area(clipped)
    if visible / full_area < min_visibility:
        return None
    try:
        if isinstance(shape, BBox):
            xs, ys = [p[0] for p in clipped], [p[1] for p in clipped]
            new = BBox.model_validate({**shape.model_dump(), "x1": min(xs), "y1": min(ys),
                                       "x2": max(xs), "y2": max(ys)})
            area = (new.x2 - new.x1) * (new.y2 - new.y1)
        else:
            new = Polygon.model_validate({**shape.model_dump(), "points": clipped})
            area = visible
    except ValueError:
        return None
    return new if area >= min_area_px else None


def _fmt(v: float) -> str:
    return f"{v:.{_PRECISION}f}"


def shape_to_yolo(shape: BBox | Polygon, width: int, height: int, task: TaskMode) -> str | None:
    """One YOLO txt line, clipped to the image. None if nothing of the shape is inside."""
    if task == "detect":
        x1, y1, x2, y2 = shape.bounds()
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            return None
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        return " ".join([str(shape.class_id), _fmt(cx), _fmt(cy), _fmt(bw), _fmt(bh)])
    if task == "seg":
        pts = clip_polygon(shape.to_points(), width, height)
        if len(pts) < 3 or polygon_area(pts) <= 0:
            return None
        coords = []
        for x, y in pts:
            coords += [_fmt(x / width), _fmt(y / height)]
        return " ".join([str(shape.class_id), *coords])
    raise ValueError(f"unknown task {task!r}")


def shapes_from_yolo(lines: list[str], width: int, height: int, task: TaskMode) -> list[BBox | Polygon]:
    """Parse YOLO txt lines back into pixel-coordinate shapes (for import and pre-labels)."""
    shapes: list[BBox | Polygon] = []
    for n, raw in enumerate(lines, 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            cls_id, values = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError as e:
            raise ValueError(f"line {n}: not a YOLO label: {raw!r}") from e
        if task == "detect":
            if len(values) != 4:
                raise ValueError(f"line {n}: detect label needs 4 values, got {len(values)}")
            cx, cy, bw, bh = values
            shapes.append(BBox(
                class_id=cls_id,
                x1=(cx - bw / 2) * width, y1=(cy - bh / 2) * height,
                x2=(cx + bw / 2) * width, y2=(cy + bh / 2) * height,
            ))
        elif task == "seg":
            if len(values) < 6 or len(values) % 2:
                raise ValueError(f"line {n}: seg label needs an even number (>= 6) of values")
            pts = [(values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)]
            shapes.append(Polygon(class_id=cls_id, points=pts))
        else:
            raise ValueError(f"unknown task {task!r}")
    return shapes


# ---- files -------------------------------------------------------------------

def save_annotations(path: str | Path, annotations: ImageAnnotations) -> None:
    text = json.dumps(annotations.model_dump(mode="json"), ensure_ascii=False, indent=2)
    atomic_write_text(path, text + "\n")


def load_annotations(path: str | Path) -> ImageAnnotations:
    with open(path, encoding="utf-8") as f:
        return ImageAnnotations.model_validate(json.load(f))


def write_yolo_txt(path: str | Path, annotations: ImageAnnotations, task: TaskMode) -> None:
    """Write the derived YOLO label file. Images without shapes get an empty file, which YOLO
    treats as a background (negative) image."""
    lines = annotations.to_yolo_lines(task)
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
