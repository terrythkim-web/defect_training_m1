"""Images cut from other project images (slices, crops, exact duplicates).

If a 3072×1024 photo and its six 512×1024 slices are all in the project, every defect is
labeled twice and the split can put a defect in Train and the same pixels in Val, so Val only
measures memorization. `find_source_groups` finds such images; `dataset.split_images` keeps
each group on one side.

Matching: a coarse search on 1/8 downscaled, lightly blurred copies (`cv2.matchTemplate`,
correlation; the blur keeps a slice that starts between two 1/8 pixels above 0.9), then a
check at full resolution in a ±8 px window: mean absolute difference ≤ MAX_MEAN_ABS_DIFF
(allows JPEG re-encoding). Candidate pairs are only images that fit inside another one, so a
project of same-size photos only compares those of equal size. Near-uniform images are skipped
(any flat patch would "match" them), as are images smaller than MIN_SIDE.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.platform_utils import imread_unicode

SCALE = 8
MIN_SIDE = 16 * SCALE  # smaller images: too few pixels for a reliable coarse match
MIN_STD = 4.0  # gray levels; flatter images are not compared
COARSE_MIN_CORR = 0.9
MAX_MEAN_ABS_DIFF = 3.0
SEARCH_PX = SCALE

Progress = Callable[[int, int], None]


@dataclass
class ImageSignature:
    path: str  # project image path (relative to images/)
    abspath: Path
    width: int
    height: int
    small: np.ndarray  # float32 gray, 1/SCALE size
    std: float

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def comparable(self) -> bool:
        return min(self.width, self.height) >= MIN_SIDE and self.std >= MIN_STD


def load_gray(abspath: Path) -> np.ndarray | None:
    image = imread_unicode(abspath)
    if image is None:
        return None
    return to_training_8bit(image, 1)


def image_signature(path: str, abspath: Path) -> ImageSignature | None:
    gray = load_gray(abspath)
    if gray is None:
        return None
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, w // SCALE), max(1, h // SCALE)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.0)  # sub-pixel offsets: corr 0.75 -> 0.94
    return ImageSignature(path, Path(abspath), w, h, small, float(gray.std()))


class _GrayCache:
    """Full-resolution grays for the verification step (few images, re-read on demand)."""

    def __init__(self, capacity: int = 6) -> None:
        self.capacity = capacity
        self._items: OrderedDict[Path, np.ndarray | None] = OrderedDict()

    def get(self, abspath: Path) -> np.ndarray | None:
        if abspath in self._items:
            self._items.move_to_end(abspath)
            return self._items[abspath]
        gray = load_gray(abspath)
        self._items[abspath] = gray
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return gray


def contains(small: ImageSignature, big: ImageSignature, grays: _GrayCache | None = None) -> bool:
    """True if `small`'s pixels appear unchanged somewhere inside `big` (or equal it)."""
    if small.width > big.width or small.height > big.height or not (small.comparable and big.comparable):
        return False
    t, img = small.small, big.small
    if t.shape[0] > img.shape[0] or t.shape[1] > img.shape[1]:
        return False
    corr = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    _, best, _, (cx, cy) = cv2.minMaxLoc(corr)
    if best < COARSE_MIN_CORR:
        return False
    grays = grays or _GrayCache()
    full_t, full_img = grays.get(small.abspath), grays.get(big.abspath)
    if full_t is None or full_img is None or full_t.shape != (small.height, small.width):
        return False
    x0, y0 = max(0, cx * SCALE - SEARCH_PX), max(0, cy * SCALE - SEARCH_PX)
    x1 = min(big.width, cx * SCALE + SEARCH_PX + small.width)
    y1 = min(big.height, cy * SCALE + SEARCH_PX + small.height)
    region = full_img[y0:y1, x0:x1]
    if region.shape[0] < small.height or region.shape[1] < small.width:
        return False
    sq = cv2.matchTemplate(region.astype(np.float32), full_t.astype(np.float32), cv2.TM_SQDIFF)
    _, _, (bx, by), _ = cv2.minMaxLoc(sq)
    window = region[by:by + small.height, bx:bx + small.width]
    return float(np.abs(window.astype(np.int16) - full_t.astype(np.int16)).mean()) <= MAX_MEAN_ABS_DIFF


def find_source_groups(signatures: list[ImageSignature], progress: Progress | None = None,
                       cancel: Callable[[], bool] | None = None) -> dict[str, str] | None:
    """{member path: source path} for every image that is a slice or copy of another one.
    The source of a group is its largest image (ties: first path). None if cancelled."""
    sigs = sorted((s for s in signatures if s.comparable), key=lambda s: (-s.area, s.path))
    parent = {s.path: s.path for s in sigs}
    order = {s.path: k for k, s in enumerate(sigs)}  # larger first: the earlier root wins

    def find(p: str) -> str:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    grays = _GrayCache()
    for i, big in enumerate(sigs):
        if cancel is not None and cancel():
            return None
        for small in sigs[i + 1:]:
            if find(small.path) == find(big.path):
                continue
            if contains(small, big, grays):
                a, b = sorted((find(big.path), find(small.path)), key=order.__getitem__)
                parent[b] = a
        if progress is not None:
            progress(i + 1, len(sigs))
    return {p: find(p) for p in parent if find(p) != p}
