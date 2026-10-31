"""Brush marking tool (Qt-free): a stroke is a mask of round dabs along the mouse path.

Painting adds the stroke to the confirmed label of the current class it touches (several touched
labels become one, keeping the first one's id), or makes a new label. Erasing cuts the stroke out of
every label under it, of any class or status; a label cut in two becomes two labels.

Every brush result is a `Mask` label (pixels kept as painted, holes included), whatever the task;
YOLO txt uses its outline (seg) or box (detect). A Polygon or BBox the brush touches becomes a Mask.
Pixel (i, j) belongs to a shape when its centre (i + 0.5, j + 0.5) does — the same continuous
coordinates as the labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from defect_inspection.core.annotation import BBox, Mask, Polygon
from defect_inspection.core.mask_codec import largest_component, mask_outline  # noqa: F401 (re-exported)

Shape = BBox | Polygon | Mask

MIN_SIZE, MAX_SIZE, DEFAULT_SIZE = 1, 300, 16  # brush diameter in image pixels
MIN_PIECE_PX = 9  # leftovers of an erase smaller than this (or thinner than MIN_PIECE_SIDE) are dropped
MIN_PIECE_SIDE = 3


def step_size(size: int, up: bool) -> int:
    """Next brush size for the [ / ] keys: about 25 % per step, at least 1 px."""
    new = max(size + 1, round(size * 1.25)) if up else min(size - 1, round(size / 1.25))
    return int(min(max(new, MIN_SIZE), MAX_SIZE))


@dataclass(frozen=True)
class StrokeResult:
    shapes: list[Shape]  # every shape of the image after the stroke
    selected: list[str] = field(default_factory=list)  # ids to select (the painted label)
    changed: bool = False
    touched: bool = False  # the stroke met an existing label


# ---- masks ----------------------------------------------------------------------------------

def stroke_mask(points: list[tuple[float, float]], size: float, width: int, height: int) -> np.ndarray:
    """Pixels whose centre lies within size/2 of the path through `points` (image coordinates)."""
    mask = np.zeros((height, width), bool)
    if not points:
        return mask
    r = max(float(size), MIN_SIZE) / 2
    pts = list(points) if len(points) > 1 else [points[0], points[0]]
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        x0, x1 = max(0, int(np.floor(min(ax, bx) - r))), min(width, int(np.ceil(max(ax, bx) + r)) + 1)
        y0, y1 = max(0, int(np.floor(min(ay, by) - r))), min(height, int(np.ceil(max(ay, by) + r)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        px = np.arange(x0, x1) + 0.5
        py = (np.arange(y0, y1) + 0.5)[:, None]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0, 1)
        dist2 = (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2
        mask[y0:y1, x0:x1] |= dist2 <= r * r
    return mask


def shape_mask(shape: Shape, width: int, height: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Pixels whose centre lies inside the shape (even-odd scanline, half-open edges, so a box
    (10, 20)-(30, 40) is exactly pixels 10..29 x 20..39). cv2.fillPoly also fills the right and
    bottom edge pixels, which made every repaint grow the label by one pixel.
    `origin`: image pixel of the mask's top-left corner, for a `width` x `height` window.
    A Mask gives its own pixels."""
    if isinstance(shape, Mask):
        return shape.moved(-origin[0], -origin[1]).full_pixels(width, height)
    pts = np.asarray(shape.to_points(), np.float64) - np.asarray(origin, np.float64)
    mask = np.zeros((height, width), bool)
    y0 = max(0, int(np.floor(pts[:, 1].min() - 0.5)))
    y1 = min(height, int(np.ceil(pts[:, 1].max())) + 1)
    if y0 >= y1 or width <= 0:
        return mask
    a, b = pts, np.roll(pts, -1, axis=0)
    yc = np.arange(y0, y1)[:, None] + 0.5
    ya, yb = a[:, 1][None], b[:, 1][None]
    cross = ((ya <= yc) & (yc < yb)) | ((yb <= yc) & (yc < ya))
    rows, edges = np.nonzero(cross)
    if rows.size == 0:
        return mask
    ay, by = a[edges, 1], b[edges, 1]
    x = a[edges, 0] + (yc[rows, 0] - ay) / (by - ay) * (b[edges, 0] - a[edges, 0])
    start = np.clip(np.ceil(x - 0.5).astype(np.int64), 0, width)
    toggles = np.zeros((y1 - y0, width + 1), np.int32)
    np.add.at(toggles, (rows, start), 1)
    mask[y0:y1] = (np.cumsum(toggles, axis=1)[:, :width] % 2).astype(bool)
    return mask


def mask_to_shape(mask: np.ndarray, fields: dict, origin: tuple[int, int] = (0, 0)) -> Mask | None:
    """Largest region of `mask` as a Mask label carrying `fields` (id, class_id, …); `origin` = image
    pixel of the mask's top-left corner. One label = one connected region."""
    region = largest_component(mask)
    return None if region is None else Mask.from_pixels(region, origin=origin, **fields)


def _fields(shape: Shape, new_id: bool = False) -> dict:
    keep = {"class_id", "status", "source", "score"} | (set() if new_id else {"id"})
    return shape.model_dump(include=keep)


Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels, end exclusive


def _mask_box(mask: np.ndarray) -> Box | None:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _shape_box(shape: Shape) -> Box:
    x1, y1, x2, y2 = shape.bounds()
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def _meets(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class _Window:
    """The part of the image a stroke edit works on (the stroke and the labels it touches, plus a
    1 px border), so a stroke on a 12 MP image does not run whole-image passes per label (an erase
    across 14 labels took 3.5 s on 4000x3000 before)."""

    def __init__(self, boxes: list[Box], width: int, height: int) -> None:
        self.x0 = max(0, min(b[0] for b in boxes) - 1)
        self.y0 = max(0, min(b[1] for b in boxes) - 1)
        self.x1 = min(width, max(b[2] for b in boxes) + 1)
        self.y1 = min(height, max(b[3] for b in boxes) + 1)
        self.origin = (self.x0, self.y0)

    def crop(self, mask: np.ndarray) -> np.ndarray:
        return mask[self.y0:self.y1, self.x0:self.x1]

    def mask(self, shape: Shape) -> np.ndarray:
        return shape_mask(shape, self.x1 - self.x0, self.y1 - self.y0, self.origin)

    def to_shape(self, mask: np.ndarray, fields: dict) -> Mask | None:
        return mask_to_shape(mask, fields, self.origin)


# ---- paint / erase ----------------------------------------------------------------------------

def paint(shapes: list[Shape], stroke: np.ndarray, class_id: int, width: int, height: int) -> StrokeResult:
    sbox = _mask_box(stroke)
    if sbox is None:
        return StrokeResult(list(shapes))
    near = [s for s in shapes
            if s.class_id == class_id and s.status == "confirmed" and _meets(_shape_box(s), sbox)]
    win = _Window([sbox] + [_shape_box(s) for s in near], width, height)
    local = win.crop(stroke)
    touched = [(s, m) for s in near if ((m := win.mask(s)) & local).any()]
    if not touched:
        new = win.to_shape(local, {"class_id": class_id})
        if new is None:
            return StrokeResult(list(shapes))
        return StrokeResult(list(shapes) + [new], [new.id], True, touched=False)
    union = local.copy()
    for _, m in touched:
        union |= m
    first = touched[0][0]
    if len(touched) == 1 and (union == touched[0][1]).all():
        return StrokeResult(list(shapes), [first.id], touched=True)  # painted inside the label
    merged = win.to_shape(union, _fields(first))
    if merged is None:
        return StrokeResult(list(shapes))
    gone = {s.id for s, _ in touched[1:]}
    out = [merged if s.id == first.id else s for s in shapes if s.id not in gone]
    return StrokeResult(out, [merged.id], True, touched=True)


def _pieces(mask: np.ndarray) -> list[np.ndarray]:
    """Connected regions big enough to keep, largest first."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= MIN_PIECE_PX
            and min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= MIN_PIECE_SIDE]
    keep.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [labels == i for i in keep]


def _erase_one(shape: Shape, win: _Window, m: np.ndarray, stroke: np.ndarray) -> list[Mask]:
    """What is left of `shape` (largest piece keeps the id)."""
    out = []
    for i, piece in enumerate(_pieces(m & ~stroke)):
        s = win.to_shape(piece, _fields(shape, new_id=i > 0))
        if s is not None:
            out.append(s)
    return out


def erase(shapes: list[Shape], stroke: np.ndarray, width: int, height: int) -> StrokeResult:
    sbox = _mask_box(stroke)
    if sbox is None:
        return StrokeResult(list(shapes))
    out: list[Shape] = []
    changed = touched = False
    for s in shapes:
        rest = None
        if _meets(_shape_box(s), sbox):
            win = _Window([_shape_box(s)], width, height)  # only the label's own area can change
            local, m = win.crop(stroke), win.mask(s)
            if (m & local).any():
                touched = True
                rest = _erase_one(s, win, m, local)
        if rest is None:
            out.append(s)
        else:
            out.extend(rest)
            changed = True
    return StrokeResult(out if changed else list(shapes), [], changed, touched)
