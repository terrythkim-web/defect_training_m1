"""Qt-free geometry edits on label shapes. Every function returns a new shape (same id); the
original is never mutated, so before/after snapshots stay valid for undo."""

from __future__ import annotations

from defect_inspection.core.annotation import BBox, Mask, Polygon

Shape = BBox | Polygon | Mask


def clamp_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))


def clamp_translation(shape: Shape, dx: float, dy: float, width: int, height: int) -> tuple[float, float]:
    """Limit a move so the whole shape stays inside the image."""
    x1, y1, x2, y2 = shape.bounds()
    dx = min(max(dx, -x1), width - x2)
    dy = min(max(dy, -y1), height - y2)
    return dx, dy


def translate(shape: Shape, dx: float, dy: float, width: int, height: int) -> Shape:
    dx, dy = clamp_translation(shape, dx, dy, width, height)
    if isinstance(shape, Mask):  # whole pixels only, still inside the image
        return shape.moved(min(max(round(dx), -shape.x), width - shape.x - shape.w),
                           min(max(round(dy), -shape.y), height - shape.y - shape.h))
    if isinstance(shape, BBox):
        return shape.model_copy(update={"x1": shape.x1 + dx, "y1": shape.y1 + dy,
                                         "x2": shape.x2 + dx, "y2": shape.y2 + dy})
    return shape.model_copy(update={"points": [(x + dx, y + dy) for x, y in shape.points]})


def move_vertex(shape: Shape, index: int, x: float, y: float, width: int, height: int) -> Shape:
    """Move one vertex. For a box, index is the corner (0 top-left, 1 top-right, 2 bottom-right,
    3 bottom-left) and the opposite corner stays fixed; dragging past it flips the box.

    Raises ValueError if the result would be degenerate (zero width/height or area); callers
    keep the old shape in that case.
    """
    x, y = clamp_point(x, y, width, height)
    if isinstance(shape, BBox):
        corners = shape.to_points()
        ox, oy = corners[(index + 2) % 4]  # opposite corner
        return BBox.model_validate({**shape.model_dump(), "x1": ox, "y1": oy, "x2": x, "y2": y})
    points = list(shape.points)
    points[index] = (x, y)
    return Polygon.model_validate({**shape.model_dump(), "points": points})


def with_class(shape: Shape, class_id: int) -> Shape:
    return shape.model_copy(update={"class_id": class_id})


def bbox_from_drag(class_id: int, x1: float, y1: float, x2: float, y2: float,
                   width: int, height: int, min_size: float = 2.0) -> BBox | None:
    """New box from a mouse drag, clamped to the image. None if smaller than min_size pixels."""
    x1, y1 = clamp_point(x1, y1, width, height)
    x2, y2 = clamp_point(x2, y2, width, height)
    if abs(x2 - x1) < min_size or abs(y2 - y1) < min_size:
        return None
    return BBox(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2)


def polygon_from_clicks(class_id: int, points: list[tuple[float, float]],
                        width: int, height: int) -> Polygon | None:
    """New polygon from clicked points (clamped, consecutive duplicates removed). None if it
    has fewer than 3 distinct points or no area."""
    clean: list[tuple[float, float]] = []
    for x, y in points:
        p = clamp_point(x, y, width, height)
        if not clean or (abs(p[0] - clean[-1][0]) > 1e-6 or abs(p[1] - clean[-1][1]) > 1e-6):
            clean.append(p)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    try:
        return Polygon(class_id=class_id, points=clean)
    except ValueError:
        return None
