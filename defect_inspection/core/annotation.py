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

from pydantic import BaseModel, Field, field_validator, model_validator

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


Shape = Annotated[Union[BBox, Polygon], Field(discriminator="type")]


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
        """Confirmed shapes only: suggestions are not training data until accepted."""
        return [line for s in self.confirmed() if (line := shape_to_yolo(s, self.width, self.height, task))]

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


def fit_shape(shape: BBox | Polygon, points: list[tuple[float, float]], w: float, h: float,
              min_visibility: float, min_area_px: float) -> BBox | Polygon | None:
    """`shape` moved to `points` (new coordinates), clipped to [0, w] x [0, h]; None when less than
    `min_visibility` of it is still visible or it is smaller than `min_area_px` (F3.6 / F7.3)."""
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
