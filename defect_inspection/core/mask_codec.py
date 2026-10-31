"""Pixel-mask helpers for Mask labels and the brush (Qt-free): run-length codec, connected regions
and outlines along pixel EDGES (continuous label coordinates: pixel i spans [i, i + 1])."""

from __future__ import annotations

import cv2
import numpy as np

OUTLINE_TOLERANCE = 0.7  # Douglas-Peucker deviation (px) when a mask becomes a polygon outline


def rle_encode(mask: np.ndarray) -> list[int]:
    """Run lengths of a 2-D bool mask in row-major order, starting with a background run (may be 0)."""
    flat = np.asarray(mask, bool).ravel()
    if flat.size == 0:
        return [0]
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    return ([0] + runs) if flat[0] else runs


def rle_decode(rle: list[int], width: int, height: int) -> np.ndarray:
    values = np.zeros(len(rle), bool)
    values[1::2] = True
    return np.repeat(values, rle).reshape(height, width)


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _edge_points(contour: np.ndarray) -> np.ndarray:
    """A contour traced on the 2x copy → points on pixel edges. Boundary pixels 2i (left/top) and
    2i + 1 (right/bottom) map to the edges i and i + 1 with ceil(u / 2)."""
    edge = np.ceil(contour.reshape(-1, 2) / 2.0)
    return edge[np.any(edge != np.roll(edge, 1, axis=0), axis=1)]


def _upscaled(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def mask_outline(mask: np.ndarray, tolerance: float = OUTLINE_TOLERANCE) -> list[tuple[float, float]] | None:
    """Outer outline of the largest region along the pixel edges, so the polygon covers exactly the
    painted pixels (`brush.shape_mask` gives them back before simplification). sam_assist's
    mask_to_polygon traces pixel centres, which leaves a label half a pixel short on every side."""
    region = largest_component(mask)
    if region is None:
        return None
    contours, _ = cv2.findContours(_upscaled(region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    edge = _edge_points(max(contours, key=cv2.contourArea))
    if tolerance > 0 and len(edge) > 3:
        edge = cv2.approxPolyDP(edge.astype(np.float32).reshape(-1, 1, 2), tolerance, True).reshape(-1, 2)
    if len(edge) < 3:
        return None
    return [(float(x), float(y)) for x, y in edge]


def outer_outlines(mask: np.ndarray, tolerance: float = OUTLINE_TOLERANCE) -> list[list[tuple[float, float]]]:
    """Outer outline of EVERY region (largest first), along pixel edges: a mask in pieces (a thin
    crack after a nearest-neighbour warp often is) becomes one polygon per piece in YOLO seg labels."""
    if not mask.any():
        return []
    contours, _ = cv2.findContours(_upscaled(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        edge = _edge_points(c)
        if tolerance > 0 and len(edge) > 3:
            edge = cv2.approxPolyDP(edge.astype(np.float32).reshape(-1, 1, 2), tolerance, True).reshape(-1, 2)
        if len(edge) >= 3:
            out.append([(float(x), float(y)) for x, y in edge])
    return out


def edge_rings(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Every outer outline and hole of `mask` along the pixel edges (for drawing with the even-odd rule)."""
    if not mask.any():
        return []
    contours, _ = cv2.findContours(_upscaled(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for c in contours:
        edge = _edge_points(c)
        if len(edge) >= 3:
            rings.append([(float(x), float(y)) for x, y in edge])
    return rings
