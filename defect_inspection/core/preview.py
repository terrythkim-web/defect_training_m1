"""Augmentation preview (PRD F3.5): a few augmented samples of one image with labels drawn on."""

from __future__ import annotations

import cv2
import numpy as np

from defect_inspection.core.annotation import BBox, ImageAnnotations
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
        if isinstance(shape, BBox):
            shapes.append(shape.model_copy(update={"x1": shape.x1 * sx, "y1": shape.y1 * sy,
                                                   "x2": shape.x2 * sx, "y2": shape.y2 * sy}))
        else:
            shapes.append(shape.model_copy(update={"points": [(x * sx, y * sy) for x, y in shape.points]}))
    return small, annotations.model_copy(update={"width": nw, "height": nh, "shapes": shapes})


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
        # Sub-pixel (shift=4) so outlines sit exactly on continuous label coordinates.
        pts = np.round((np.array(shape.to_points(), np.float32) - 0.5) * 16).astype(np.int32)
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA, shift=4)
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
    return [draw_labels(r.image, r.annotations, colors) for r in (augmenter(small, ann) for _ in range(n))]


# Ground truth vs prediction (Result tab, PRD F5.1): two fixed hues from the validated palette,
# named in the UI legend; predictions also carry their score as text.
GT_COLOR = "#2a78d6"  # blue
PRED_COLOR = "#eb6834"  # orange


def draw_comparison(image8: np.ndarray, ground_truth: list, predictions: list, max_side: int | None = None) -> np.ndarray:
    """BGR image with ground-truth outlines (blue) and predictions (orange + score)."""
    out = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR) if image8.ndim == 2 else image8.copy()
    scale = 1.0
    if max_side is not None and max(out.shape[:2]) > max_side:
        scale = max_side / max(out.shape[:2])
        out = cv2.resize(out, (round(out.shape[1] * scale), round(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    thickness = max(1, round(max(out.shape[:2]) / 250))
    font_scale = max(0.35, max(out.shape[:2]) / 900)

    def outline(shape, color):
        pts = np.round((np.array(shape.to_points(), np.float32) * scale - 0.5) * 16).astype(np.int32)
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA, shift=4)

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
