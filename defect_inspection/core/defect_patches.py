"""Defect patches: Copy-Paste (F3.3) and defect transform (size, angle, position of each defect).

A patch is a labeled defect cut out of a training image: pixels, a soft (feathered) mask and
the label in patch coordinates. Copy-Paste blends patches onto OK images unchanged. The defect
transform (`DefectTransformSetting`) scales and rotates a patch by a random amount and places
it shifted horizontally / vertically / both from where it was, inside the same image ("add"
keeps the original defect, "move" fills its place from the surroundings with cv2.inpaint), and
onto OK images. Labels travel with the pixels: a rotated defect gets the rotated outline as a
polygon (detect tasks then train on its bounding box).

Coordinates: labels are continuous (pixel i covers [i, i+1)), OpenCV warps use pixel indices,
so points are mapped as q = M·(p − 0.5) + 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon
from defect_inspection.core.augment_config import DefectTransformSetting

Shape = BBox | Polygon
PLACE_TRIES = 20


@dataclass
class Patch:
    pixels: np.ndarray
    mask: np.ndarray  # float32 0..1, feathered edges
    shape: Shape  # in patch coordinates
    origin: tuple[int, int] = (0, 0)  # top-left in the source image
    source_size: tuple[int, int] = (0, 0)  # (width, height) of the source image


def patch_for(image: np.ndarray, shape: Shape, max_fraction: float = 0.5) -> Patch | None:
    """Cut one labeled defect out of an image; None if it is tiny or covers most of the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = shape.bounds()
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    pw, ph = x2 - x1, y2 - y1
    if pw < 2 or ph < 2 or pw > w * max_fraction or ph > h * max_fraction:
        return None
    mask = np.zeros((ph, pw), np.uint8)
    local = [(px - x1, py - y1) for px, py in shape.to_points()]
    pts = np.round((np.array(local, np.float32) - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, shift=4)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0) if min(pw, ph) >= 8 else mask / 255.0
    if isinstance(shape, BBox):
        local_shape = shape.model_copy(update={"x1": shape.x1 - x1, "y1": shape.y1 - y1,
                                               "x2": shape.x2 - x1, "y2": shape.y2 - y1})
    else:
        local_shape = shape.model_copy(update={"points": local})
    return Patch(image[y1:y2, x1:x2].copy(), soft.astype(np.float32), local_shape, (x1, y1), (w, h))


def extract_patches(image: np.ndarray, annotations: ImageAnnotations, max_fraction: float = 0.5) -> list[Patch]:
    """Cut every confirmed defect out of a training image (pixels + soft mask + its label)."""
    patches = (patch_for(image, s, max_fraction) for s in annotations.confirmed())
    return [p for p in patches if p is not None]


def transform_patch(patch: Patch, scale: float, angle: float) -> Patch:
    """The patch scaled and rotated (degrees, counter-clockwise) about its centre, on a canvas
    just large enough; the label becomes the transformed outline."""
    ph, pw = patch.pixels.shape[:2]
    m = cv2.getRotationMatrix2D((pw / 2 - 0.5, ph / 2 - 0.5), angle, scale)

    def apply(points):
        pts = np.asarray(points, np.float64) - 0.5
        return pts @ m[:, :2].T + m[:, 2] + 0.5

    corners = apply([(0, 0), (pw, 0), (pw, ph), (0, ph)])
    # 1e-6: cos(90°) is not exactly 0, which would add a pixel row
    (minx, miny), (maxx, maxy) = np.floor(corners.min(0) + 1e-6), np.ceil(corners.max(0) - 1e-6)
    m[:, 2] -= (minx, miny)
    size = (max(1, int(maxx - minx)), max(1, int(maxy - miny)))
    pixels = cv2.warpAffine(patch.pixels, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(patch.mask, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    points = [(float(x), float(y)) for x, y in apply(patch.shape.to_points())]  # m already shifted
    s = patch.shape
    if isinstance(s, BBox) and angle % 360 == 0:
        shape = BBox(class_id=s.class_id, x1=points[0][0], y1=points[0][1], x2=points[2][0], y2=points[2][1])
    else:
        shape = Polygon(class_id=s.class_id, points=points)
    # where the transformed patch sits if it keeps the original centre
    cx, cy = patch.origin[0] + pw / 2, patch.origin[1] + ph / 2
    origin = (int(round(cx - size[0] / 2)), int(round(cy - size[1] / 2)))
    return Patch(pixels, mask.astype(np.float32), shape, origin, patch.source_size)


def random_transform(patch: Patch, cfg: DefectTransformSetting, rng: np.random.Generator) -> Patch:
    scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
    angle = float(rng.uniform(-cfg.max_angle, cfg.max_angle)) if cfg.max_angle > 0 else 0.0
    return transform_patch(patch, scale, angle)


def _overlaps(box: tuple[float, float, float, float], shapes: list[Shape], margin: float = 2.0) -> bool:
    x1, y1, x2, y2 = box
    for s in shapes:
        a1, b1, a2, b2 = s.bounds()
        if x1 < a2 + margin and a1 < x2 + margin and y1 < b2 + margin and b1 < y2 + margin:
            return True
    return False


def shifted_origin(patch: Patch, width: int, height: int, avoid: list[Shape], cfg: DefectTransformSetting,
                   rng: np.random.Generator) -> tuple[int, int] | None:
    """A free position near the patch's own: shifted by up to max_shift × image size along the
    allowed direction(s), kept inside the image and off `avoid`. Without a position in this
    image's frame (other image size) anywhere free."""
    ph, pw = patch.pixels.shape[:2]
    if pw >= width or ph >= height:
        return None
    same_frame = patch.source_size == (width, height)
    for _ in range(PLACE_TRIES):
        if same_frame:
            dx = rng.uniform(-1, 1) * cfg.max_shift * width if cfg.direction in ("both", "horizontal") else 0.0
            dy = rng.uniform(-1, 1) * cfg.max_shift * height if cfg.direction in ("both", "vertical") else 0.0
            ox = int(np.clip(round(patch.origin[0] + dx), 0, width - pw))
            oy = int(np.clip(round(patch.origin[1] + dy), 0, height - ph))
        else:
            ox, oy = int(rng.integers(0, width - pw + 1)), int(rng.integers(0, height - ph + 1))
        if not _overlaps((ox, oy, ox + pw, oy + ph), avoid):
            return ox, oy
    return None


def blend(image: np.ndarray, patch: Patch, ox: int, oy: int) -> Shape:
    """Blend the patch into `image` in place at (ox, oy); returns its label in image coordinates."""
    ph, pw = patch.pixels.shape[:2]
    region = image[oy:oy + ph, ox:ox + pw].astype(np.float32)
    alpha = patch.mask if image.ndim == 2 else patch.mask[:, :, None]
    image[oy:oy + ph, ox:ox + pw] = np.clip(alpha * patch.pixels + (1 - alpha) * region, 0, 255).astype(np.uint8)
    s = patch.shape
    if isinstance(s, BBox):
        return BBox(class_id=s.class_id, x1=s.x1 + ox, y1=s.y1 + oy, x2=s.x2 + ox, y2=s.y2 + oy)
    return Polygon(class_id=s.class_id, points=[(x + ox, y + oy) for x, y in s.points])


def erase(image: np.ndarray, shape: Shape, grow: int = 3) -> np.ndarray:
    """Fill a defect's place from its surroundings (cv2.inpaint), for the "move" mode."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = np.round((np.array(shape.to_points(), np.float32) - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, shift=4)
    mask = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def transform_defects(image: np.ndarray, annotations: ImageAnnotations, cfg: DefectTransformSetting,
                      rng: np.random.Generator) -> tuple[np.ndarray, ImageAnnotations, int]:
    """Same-image defect transform: up to cfg.max_defects defects get a random size, angle and
    shifted position; "add" keeps the original, "move" erases it. Returns the new image, its
    labels and how many defects were placed."""
    shapes = list(annotations.confirmed())
    if not shapes:
        return image, annotations, 0
    h, w = image.shape[:2]
    out = image.copy()
    kept = list(shapes)
    placed = 0
    for i in rng.permutation(len(shapes))[:cfg.max_defects]:
        source = shapes[int(i)]
        patch = patch_for(image, source)
        if patch is None:
            continue
        moved = random_transform(patch, cfg, rng)
        avoid = [s for s in kept if s is not source] if cfg.mode == "move" else kept
        where = shifted_origin(moved, w, h, avoid, cfg, rng)
        if where is None:
            continue
        if cfg.mode == "move":
            out = erase(out, source)
            kept.remove(source)
        kept.append(blend(out, moved, *where))
        placed += 1
    return out, annotations.model_copy(update={"shapes": kept}), placed


def paste_patches(image: np.ndarray, annotations: ImageAnnotations, patches: list[Patch],
                  rng: np.random.Generator, max_patches: int, tries: int = PLACE_TRIES,
                  transform: DefectTransformSetting | None = None) -> tuple[np.ndarray, ImageAnnotations]:
    """Blend 1..max_patches random patches at free spots and add their labels. With `transform`
    each patch first gets a random size and angle and is placed shifted from its own position
    (defect transform onto OK images)."""
    out = image.copy()
    shapes = list(annotations.confirmed())
    h, w = image.shape[:2]
    for _ in range(int(rng.integers(1, max_patches + 1))):
        patch = patches[int(rng.integers(len(patches)))]
        if transform is not None:
            patch = random_transform(patch, transform, rng)
            where = shifted_origin(patch, w, h, shapes, transform, rng)
            if where is not None:
                shapes.append(blend(out, patch, *where))
            continue
        ph, pw = patch.pixels.shape[:2]
        if pw >= w or ph >= h:
            continue
        for _ in range(tries):
            ox, oy = int(rng.integers(0, w - pw + 1)), int(rng.integers(0, h - ph + 1))
            if not _overlaps((ox, oy, ox + pw, oy + ph), shapes):
                break
        else:
            continue
        shapes.append(blend(out, patch, ox, oy))
    return out, annotations.model_copy(update={"shapes": shapes})
