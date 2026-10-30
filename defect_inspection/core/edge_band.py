"""Is a predicted box on the object's outermost edge? (PRD F9.1, F9.2)

The object's outer edge shows up as a long, thin, dark, nearly straight line. Each box is checked in
a strip around it — long along the expected edge (`along_px`), narrow across it (`across_px`): the
box must sit on (or within `touch_px` of) such a line, the strip beyond the line must be flat
background, and no further dark line may lie beyond it. A whole-image search was tried first and
picked inner lines when a flat dark display area sat beyond them; a narrow local strip avoids that.
Limit: a parallel line inside the edge whose own beyond-side is flat over `across_px` looks the same
as the edge. A defect cutting the line is allowed: the line is only measured outside the box.
Assumes the edge line is darker than the background on both sides; on a black background every box fails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.core.imaging import to_grayscale

Axis = Literal["h", "v"]

RIDGE_PX = 4  # a line pixel is darker than the pixels this far away on both sides (lines up to ~5 px thick)
ON_LINE_PX = 2  # a fitted line counts as dark at a position if a ridge pixel is within this distance
MIN_SEGMENT_PX = 40  # Hough pieces shorter than this are noise; also the least strip length beside the box
MERGE_TOL_PX = 6  # pieces this close to a line belong to it; lines closer than this are one line
TEXTURE_LEVEL = 6.0  # |Laplacian| above this counts as structure (JPEG noise stays below)
LINE_CLEARANCE_PX = 8  # start of the "beyond the line" region, past the line's own pixels
MIN_BEYOND_PX = 6  # less room than this beyond the line is only accepted at the image border
HOUGH_ANGLE_STEP = np.pi / 720  # 0.25 degree
HOUGH_MAX_GAP_PX = 20  # bridges ridge dropouts and small breaks within one Hough piece

Box = tuple[float, float, float, float]
Piece = tuple[float, float, float, float]


@dataclass(frozen=True)
class EdgeHit:
    """The edge line found under a box. `position` = y (axis h) or x (axis v) of the line at the box's
    middle in image pixels; `outward` (+1 / -1) = background direction along y (h) or x (v);
    `busy` = textured share of the strip beyond the line."""

    axis: Axis
    position: float
    outward: int
    busy: float


@dataclass(frozen=True)
class _Strip:
    """A strip cut out of an image oriented so the expected edge runs along its rows (axis h) — for
    axis v the image was transposed. Box and offsets are in strip coordinates."""

    pixels: np.ndarray
    across0: int
    box_along: tuple[float, float]
    box_across: tuple[float, float]
    open_before: bool  # the strip reaches the image border before row 0
    open_after: bool  # ... after its last row


def edge_under_box(image8: np.ndarray, box: Box, cfg: EdgeFilterConfig) -> EdgeHit | None:
    """The outermost edge line under the box (x1, y1, x2, y2), or None if the box is not on one.
    `image8`: uint8 gray or OpenCV BGR(A), like the predictors' input (`imaging.to_training_8bit`)."""
    if image8.dtype != np.uint8:
        raise ValueError(f"edge filter expects uint8 pixels, got {image8.dtype}")
    if not np.all(np.isfinite(box)):
        return None
    gray = to_grayscale(image8)
    hits = [hit for axis in ("h", "v") if (hit := _check_axis(gray, box, axis, cfg)) is not None]
    return min(hits, key=lambda h: h.busy, default=None)


def _check_axis(gray: np.ndarray, box: Box, axis: Axis, cfg: EdgeFilterConfig) -> EdgeHit | None:
    strip = _cut_strip(gray if axis == "h" else gray.T, box, axis, cfg)
    if strip is None:
        return None
    ridge = _ridge_rows(strip.pixels, cfg.dark_contrast)
    beside = _beside_box(strip)
    lines = _lines(ridge, beside, cfg)
    texture = _texture(strip.pixels)
    mid = sum(strip.box_along) / 2
    c0, c1 = strip.box_across
    best: EdgeHit | None = None
    for a, b in lines:
        pos = a * mid + b
        if not c0 - cfg.touch_px <= pos <= c1 + cfg.touch_px:
            continue
        for side in (-1, 1):
            busy = _beyond(texture, beside, strip, lines, (a, b), side)
            if busy is not None and busy <= cfg.flat_max and (best is None or busy < best.busy):
                best = EdgeHit(axis, strip.across0 + pos, side, busy)
    return best


def _cut_strip(g: np.ndarray, box: Box, axis: Axis, cfg: EdgeFilterConfig) -> _Strip | None:
    depth, length = g.shape
    x1, y1, x2, y2 = box
    a0, a1, c0, c1 = (x1, x2, y1, y2) if axis == "h" else (y1, y2, x1, x2)
    a0, a1 = max(0.0, a0), min(float(length), a1)
    c0, c1 = max(0.0, c0), min(float(depth), c1)
    if a1 <= a0 or c1 <= c0:
        return None  # box outside the image
    # Wide enough that a line `touch_px` from the box still has room beyond it to be judged.
    across = max(cfg.across_px, cfg.touch_px + LINE_CLEARANCE_PX + MIN_BEYOND_PX)
    s0, s1 = max(0, int(a0) - cfg.along_px), min(length, int(np.ceil(a1)) + cfg.along_px)
    r0, r1 = max(0, int(c0) - across), min(depth, int(np.ceil(c1)) + across)
    if r1 - r0 < 2 * RIDGE_PX + 2:
        return None
    return _Strip(np.ascontiguousarray(g[r0:r1, s0:s1]), r0, (a0 - s0, a1 - s0), (c0 - r0, c1 - r0),
                  open_before=r0 == 0, open_after=r1 == depth)


def _ridge_rows(g: np.ndarray, contrast: float) -> np.ndarray:
    """Thin dark lines along the rows: darker than the pixels RIDGE_PX above and below by `contrast`.
    Wide dark regions are not lines."""
    v = g.astype(np.int16)
    r = RIDGE_PX
    mask = np.zeros(v.shape, bool)
    mask[r:-r, :] = (v[r:-r, :] <= v[:-2 * r, :] - contrast) & (v[r:-r, :] <= v[2 * r:, :] - contrast)
    return mask


def _beside_box(strip: _Strip) -> np.ndarray:
    """Columns of the strip outside the box: the line and the background are judged there only, so a
    defect breaking the line or reaching into the background does not count against it. A box spanning
    almost the whole strip leaves too little beside it: then all columns count."""
    beside = np.ones(strip.pixels.shape[1], bool)
    beside[int(strip.box_along[0]):int(np.ceil(strip.box_along[1]))] = False
    return beside if beside.sum() >= MIN_SEGMENT_PX else np.ones_like(beside)


def _texture(g: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    return np.abs(cv2.Laplacian(blurred, cv2.CV_32F)) > TEXTURE_LEVEL


def _lines(ridge: np.ndarray, beside: np.ndarray, cfg: EdgeFilterConfig) -> list[tuple[float, float]]:
    """Straight lines row = a * column + b covering at least `min_length` of the columns beside the box."""
    max_slope = np.tan(np.radians(cfg.max_tilt_deg))
    out: list[tuple[float, float]] = []
    for group in _merge(_pieces(ridge, cfg.max_tilt_deg)):
        a, b = _fit(group)
        if abs(a) <= max_slope and _coverage(ridge, a, b)[beside].mean() >= cfg.min_length:
            out.append((a, b))
    return out


def _pieces(mask: np.ndarray, max_tilt_deg: float) -> list[Piece]:
    """Near-horizontal Hough pieces as (x1, y1, x2, y2)."""
    found = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, HOUGH_ANGLE_STEP, threshold=MIN_SEGMENT_PX,
                            minLineLength=MIN_SEGMENT_PX, maxLineGap=HOUGH_MAX_GAP_PX)
    pieces = [] if found is None else found.reshape(-1, 4).astype(float)  # OpenCV 5: (N, 4)
    return [tuple(p) for p in pieces
            if np.degrees(np.arctan2(abs(p[3] - p[1]), abs(p[2] - p[0]))) <= max_tilt_deg]


def _fit(pieces: list[Piece]) -> tuple[float, float]:
    """Least-squares line row = a * column + b through all piece end points."""
    xs = [p[0] for p in pieces] + [p[2] for p in pieces]
    ys = [p[1] for p in pieces] + [p[3] for p in pieces]
    if max(xs) - min(xs) < 1:
        return 0.0, float(np.mean(ys))
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def _merge(pieces: list[Piece]) -> list[list[Piece]]:
    """Group pieces lying on one straight line, longest pieces first. Each group's fit is kept and
    only redone when a piece joins (refitting every group per piece took ~0.5 s on busy images)."""
    groups: list[tuple[list[Piece], tuple[float, float]]] = []
    for p in sorted(pieces, key=lambda p: -abs(p[2] - p[0])):
        mid_x, mid_y = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        i = next((i for i, (_, (a, b)) in enumerate(groups) if abs(a * mid_x + b - mid_y) <= MERGE_TOL_PX), None)
        if i is None:
            groups.append(([p], _fit([p])))
        else:
            members = [*groups[i][0], p]
            groups[i] = (members, _fit(members))
    return [members for members, _ in groups]


def _coverage(mask: np.ndarray, a: float, b: float) -> np.ndarray:
    """Per column: is there a ridge pixel within ON_LINE_PX of the line row = a * column + b?
    Checked along the fitted line itself, so pieces from different structures never add up to a line."""
    depth, length = mask.shape
    cols = np.arange(length)
    rows = np.rint(a * cols + b).astype(int)
    hit = np.zeros(length, bool)
    for d in range(-ON_LINE_PX, ON_LINE_PX + 1):
        r = rows + d
        ok = (r >= 0) & (r < depth)
        hit[ok] |= mask[r[ok], cols[ok]]
    return hit


def _beyond(texture: np.ndarray, beside: np.ndarray, strip: _Strip, lines: list[tuple[float, float]],
            line: tuple[float, float],
            side: int) -> float | None:
    """Textured share of the strip beyond the line on `side` (columns beside the box), or None when
    that side cannot be background: another line lies beyond, or there is no room and no image border."""
    a, b = line
    mid = sum(strip.box_along) / 2
    pos = a * mid + b
    if any((a2 * mid + b2 - pos) * side > MERGE_TOL_PX for a2, b2 in lines):
        return None
    depth, length = texture.shape
    offset = (np.arange(depth)[:, None] - (a * np.arange(length) + b)[None, :]) * side
    region = (offset > LINE_CLEARANCE_PX) & beside[None, :]
    room = depth - 1 - pos if side == 1 else pos
    if room - LINE_CLEARANCE_PX < MIN_BEYOND_PX or not region.any():
        return 0.0 if (strip.open_after if side == 1 else strip.open_before) else None
    return float(texture[region].mean())
