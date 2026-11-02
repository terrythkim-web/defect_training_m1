"""Defect transform along the edge line (direction "edge", 테두리선 따라).

A defect on an edge line must stay on a line of the same kind: moved onto a flat surface it is
a defect that cannot exist. Instead of estimating the line's angle, the defect's surroundings
(its box plus a margin) are used as a template, with the defect itself masked out of the score,
and the defect is only placed where `cv2.matchTemplate` finds the same surroundings again. A
straight line therefore matches all along itself at the same perpendicular offset, tilted lines
and lines that end are handled for free, and a defect whose surroundings are unique (a bump in
the line) only moves as far as they repeat. The same template places patches onto other (OK)
images.

Choices measured on real samples (512x1024 edge photos):
- Masking, not inpainting: an inpainted large defect left a gap in the line, which then matched
  an intact line badly (NCC 0.82 for a 40 px blob on a 3 px line).
- Gradient magnitude, not intensity: a tall template is mostly flat surface, so an intensity
  match still scored >= 0.85 with the defect slid 34 px across the edge; on gradients flat areas
  count for nothing and the drift was 0.
- Ridge only (`RIDGE_PX`): a position must score within RIDGE_EPS of the best in its
  neighbourhood, which keeps the line itself and drops the flanks beside it.
Flat surroundings (no line) are matched by intensity difference instead.

Large images are matched at most MATCH_MAX_SIDE px wide (looser threshold, a thin line blurs
there) and a picked position is refined at full size, so a defect on a 2-3 px line does not end
up beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CONTEXT_FRACTION = 0.5  # margin around the defect box, of its longer side
CONTEXT_MIN_PX = 16
DEFECT_GROW_PX = 3  # defect mask grown by this before it is left out of the score
MATCH_SCORE = 0.7  # TM_CCOEFF_NORMED of gradient magnitudes needed for "the same surroundings"
RIDGE_PX = 4
RIDGE_EPS = 0.02
FLAT_STD = 3.0  # surroundings flatter than this have no line: matched by difference instead
FLAT_RMS = 10.0  # gray levels, for flat surroundings
MATCH_MAX_SIDE = 1024
COARSE_SLACK = 0.25  # looser on a downscaled image; `refine` applies MATCH_SCORE


@dataclass(frozen=True)
class Context:
    template: np.ndarray  # uint8 gray surroundings of the defect (defect included)
    gradient: np.ndarray  # float32 gradient magnitude of the same window (taken from the whole image)
    mask: np.ndarray  # uint8, 255 = compared, 0 = the (grown) defect
    center: tuple[float, float]  # defect box centre in template coordinates

    @property
    def flat(self) -> bool:
        return float(self.template[self.mask > 0].std()) < FLAT_STD


def gray8(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gradient(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude after a light blur (JPEG noise), float32."""
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))


def context_for(image: np.ndarray, shape) -> Context | None:
    """The defect's surroundings as a template (None if they fill the whole image)."""
    from defect_inspection.core.defect_patches import defect_mask

    h, w = image.shape[:2]
    x1, y1, x2, y2 = shape.bounds()
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    m = max(CONTEXT_MIN_PX, int(CONTEXT_FRACTION * max(x2 - x1, y2 - y1)))
    cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
    if cx2 - cx1 >= w or cy2 - cy1 >= h:
        return None
    keep = 255 - defect_mask(shape, w, h, DEFECT_GROW_PX)[cy1:cy2, cx1:cx2]
    if not keep.any():
        return None
    g = gray8(image)
    return Context(g[cy1:cy2, cx1:cx2].copy(), gradient(g)[cy1:cy2, cx1:cx2].copy(), keep,
                   ((x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1))


def _scores(ctx: Context, gray: np.ndarray, grad: np.ndarray, template: np.ndarray, tgrad: np.ndarray,
            mask: np.ndarray, slack: float = 0.0, ridge_px: int = RIDGE_PX) -> np.ndarray:
    """Higher = more alike, threshold 0, at matchTemplate top-left positions; `slack` loosens it.
    Positions off the ridge (below the best around them) are set to -1."""
    if ctx.flat:
        sq = cv2.matchTemplate(gray.astype(np.float32), template.astype(np.float32), cv2.TM_SQDIFF, mask=mask)
        return FLAT_RMS * (1 + slack) - np.sqrt(np.maximum(sq, 0) / max(1, np.count_nonzero(mask)))
    score = cv2.matchTemplate(grad, tgrad, cv2.TM_CCOEFF_NORMED, mask=mask)
    score = np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if ridge_px > 0:
        best = cv2.dilate(score, np.ones((2 * ridge_px + 1, 2 * ridge_px + 1), np.uint8))
        score = np.where(score >= best - RIDGE_EPS, score, -1.0)
    return score - (MATCH_SCORE - slack)


def match_centers(image: np.ndarray, ctx: Context) -> np.ndarray:
    """(N, 2) defect centres in `image` where the surroundings match; coarse (± a few px) on
    images larger than MATCH_MAX_SIDE — `refine` a picked one."""
    g = gray8(image)
    th, tw = ctx.template.shape
    if th >= g.shape[0] or tw >= g.shape[1]:
        return np.empty((0, 2))
    f = min(1.0, MATCH_MAX_SIDE / max(g.shape))
    if f < 1.0:
        size = (max(2, round(tw * f)), max(2, round(th * f)))
        g = cv2.resize(g, (max(2, round(g.shape[1] * f)), max(2, round(g.shape[0] * f))), interpolation=cv2.INTER_AREA)
        tpl = cv2.resize(ctx.template, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(ctx.mask, size, interpolation=cv2.INTER_NEAREST)
        if not mask.any() or tpl.shape[0] >= g.shape[0] or tpl.shape[1] >= g.shape[1]:
            return np.empty((0, 2))
        score = _scores(ctx, g, gradient(g), tpl, gradient(tpl), mask, COARSE_SLACK, max(1, round(RIDGE_PX * f)))
    else:
        score = _scores(ctx, g, gradient(g), ctx.template, ctx.gradient, ctx.mask)
    ys, xs = np.nonzero(score >= 0)
    return np.column_stack([xs / f + ctx.center[0], ys / f + ctx.center[1]]).astype(np.float64)


def refine(gray: np.ndarray, grad: np.ndarray, ctx: Context, cx: float, cy: float,
           radius: int) -> tuple[float, float] | None:
    """The best full-size match within `radius` px of a coarse centre (None if none qualifies);
    `gray`/`grad` = the whole image and its `gradient`."""
    th, tw = ctx.template.shape
    x0, y0 = round(cx - ctx.center[0]), round(cy - ctx.center[1])
    ax, ay = max(0, x0 - radius), max(0, y0 - radius)
    bx, by = min(gray.shape[1], x0 + radius + tw), min(gray.shape[0], y0 + radius + th)
    if bx - ax < tw or by - ay < th:
        return None
    score = _scores(ctx, gray[ay:by, ax:bx], grad[ay:by, ax:bx], ctx.template, ctx.gradient, ctx.mask, ridge_px=0)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[iy, ix] < 0:
        return None
    return ax + ix + ctx.center[0], ay + iy + ctx.center[1]


def coarse_radius(image: np.ndarray) -> int:
    """How far a `match_centers` result can be off (0 when the image was matched at full size)."""
    f = min(1.0, MATCH_MAX_SIDE / max(image.shape[:2]))
    return 0 if f >= 1.0 else math.ceil(1 / f) + 1
