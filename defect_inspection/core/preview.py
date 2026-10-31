"""Augmentation preview (PRD F3.5): a few augmented samples of one image with labels drawn on."""

from __future__ import annotations

import cv2
import numpy as np

from defect_inspection.core.annotation import BBox, ImageAnnotations, Mask
from defect_inspection.core.augment_config import AugmentConfig


def scale_for_preview(image: np.ndarray, annotations: ImageAnnotations,
                      max_side: int = 480) -> tuple[np.ndarray, ImageAnnotations]:
    """Shrink a large image and its confirmed labels together, so a 4K preview stays fast."""
    h, w = image.shape[:2]
    s = min(1.0, max_side / max(h, w))
    confirmed = annotations.confirmed()
    if s >= 1.0:
        return image, annotations.model_copy(update={"shapes": confirmed})
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = nw / w, nh / h
    shapes = []
    for shape in confirmed:
        if isinstance(shape, Mask):
            scaled = scale_mask(shape, sx, sy, nw, nh)
            if scaled is not None:
                shapes.append(scaled)
        elif isinstance(shape, BBox):
            shapes.append(shape.model_copy(update={"x1": shape.x1 * sx, "y1": shape.y1 * sy,
                                                   "x2": shape.x2 * sx, "y2": shape.y2 * sy}))
        else:
            shapes.append(shape.model_copy(update={"points": [(x * sx, y * sy) for x, y in shape.points]}))
    return small, annotations.model_copy(update={"width": nw, "height": nh, "shapes": shapes})


def scale_mask(shape: Mask, sx: float, sy: float, width: int, height: int) -> Mask | None:
    """A mask label resized by (sx, sy) into a width x height image (nearest pixel; None if it vanishes)."""
    x0, y0 = int(np.floor(shape.x * sx)), int(np.floor(shape.y * sy))
    x1 = max(x0 + 1, int(np.ceil((shape.x + shape.w) * sx)))
    y1 = max(y0 + 1, int(np.ceil((shape.y + shape.h) * sy)))
    small = cv2.resize(shape.pixels().astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    fields = shape.model_dump(include={"id", "class_id", "status", "source", "score"})
    full = np.zeros((height, width), bool)
    cx1, cy1 = min(x1, width), min(y1, height)
    full[y0:cy1, x0:cx1] = small[:cy1 - y0, :cx1 - x0].astype(bool)
    return Mask.from_pixels(full, **fields)


def outline_rings(shape, scale: float = 1.0) -> list[np.ndarray]:
    """Outlines to draw with cv2.polylines(shift=4): a mask gives every outline and hole."""
    if isinstance(shape, Mask):
        from defect_inspection.core.mask_codec import edge_rings

        rings = [[(x + shape.x, y + shape.y) for x, y in r] for r in edge_rings(shape.pixels())]
    else:
        rings = [shape.to_points()]
    # Sub-pixel (shift=4) so outlines sit exactly on continuous label coordinates.
    return [np.round((np.array(r, np.float32) * scale - 0.5) * 16).astype(np.int32) for r in rings]


def _bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return b, g, r


def draw_labels(image8: np.ndarray, annotations: ImageAnnotations, colors: list[str]) -> np.ndarray:
    """BGR copy of the image with boxes / polygons outlined in their class colour."""
    out = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR) if image8.ndim == 2 else image8.copy()
    thickness = max(1, round(max(out.shape[:2]) / 300))
    for shape in annotations.shapes:
        color = _bgr(colors[shape.class_id]) if shape.class_id < len(colors) else (255, 0, 255)
        cv2.polylines(out, outline_rings(shape), True, color, thickness, cv2.LINE_AA, shift=4)
    return out


def augment_previews(image8: np.ndarray, annotations: ImageAnnotations, config: AugmentConfig,
                     channels: int, colors: list[str], n: int = 9, seed: int | None = None,
                     max_side: int = 480) -> list[np.ndarray]:
    """n augmented samples (BGR, labels drawn) using exactly the dataset-build pipeline."""
    from defect_inspection.core.augment import Augmenter  # Albumentations: lazy import

    small, ann = scale_for_preview(image8, annotations, max_side)
    # The min-area rule is in full-size pixels; scale it so the preview drops what the build drops.
    s = small.shape[1] / image8.shape[1]
    config = config.model_copy(update={"min_area_px": config.min_area_px * s * s})
    augmenter = Augmenter(config, channels=channels, seed=seed)
    dt = config.defect_transform
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        image, labels = small, ann
        if dt.enabled and rng.random() < dt.p:  # same-image defect transform, as in the build
            from defect_inspection.core.defect_patches import transform_defects

            image, labels, _ = transform_defects(small, ann, dt, rng)
        r = augmenter(image, labels)
        out.append(draw_labels(r.image, r.annotations, colors))
    return out


# Ground truth vs prediction (Result tab, PRD F5.1): two fixed hues from the validated palette,
# named in the UI legend; predictions also carry their score as text.
GT_COLOR = "#2a78d6"  # blue
PRED_COLOR = "#eb6834"  # orange
DROPPED_COLOR = "#9a9a9a"  # grey: predictions removed by the edge filter (F9)
EDGE_COLOR = "#1baf7a"  # green: the outermost edge line a kept prediction sits on (F9)


def draw_comparison(image8: np.ndarray, ground_truth: list, predictions: list, max_side: int | None = None,
                    dropped: list = (), edges: list = ()) -> np.ndarray:
    """BGR image with ground-truth outlines (blue) and predictions (orange + score); edge filter (F9):
    `dropped` shapes in grey, `edges` = (EdgeHit, shape) pairs as a green line under the shape."""
    out = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR) if image8.ndim == 2 else image8.copy()
    scale = 1.0
    if max_side is not None and max(out.shape[:2]) > max_side:
        scale = max_side / max(out.shape[:2])
        out = cv2.resize(out, (round(out.shape[1] * scale), round(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    thickness = max(1, round(max(out.shape[:2]) / 250))
    font_scale = max(0.35, max(out.shape[:2]) / 900)

    def outline(shape, color):
        cv2.polylines(out, outline_rings(shape, scale), True, color, thickness, cv2.LINE_AA, shift=4)

    for shape in dropped:
        outline(shape, _bgr(DROPPED_COLOR))
    for hit, shape in edges:
        x1, y1, x2, y2 = shape.bounds()
        reach = max(x2 - x1, y2 - y1) / 2  # the line beside the box, where the rule measured it
        if hit.axis == "h":
            a, b = (x1 - reach, hit.position), (x2 + reach, hit.position)
        else:
            a, b = (hit.position, y1 - reach), (hit.position, y2 + reach)
        cv2.line(out, (round(a[0] * scale), round(a[1] * scale)), (round(b[0] * scale), round(b[1] * scale)),
                 _bgr(EDGE_COLOR), thickness, cv2.LINE_AA)
    for shape in ground_truth:
        outline(shape, _bgr(GT_COLOR))
    for shape in predictions:
        outline(shape, _bgr(PRED_COLOR))
        if shape.score is not None:
            x1, y1, _, _ = shape.bounds()
            org = (int(x1 * scale), max(int(y1 * scale) - 4, 10))
            text = f"{shape.score:.2f}"
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, _bgr(PRED_COLOR), thickness, cv2.LINE_AA)
    return out
