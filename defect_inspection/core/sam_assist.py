"""SAM click assist (PRD F2.8 ②): point/box prompts -> SAM mask -> polygon or box suggestion.

Qt-free. Heavy imports (torch / ultralytics) happen lazily on first use, and the image
embedding is cached so extra clicks on the same area only run the light mask decoder.

Tiled SAM: SAM encodes a 1024×1024 input, so a whole 3072×1024 photo was shrunk 3× and a 24 px
wide crack became 8 px. Images larger than SAM_WINDOW are therefore cut to a window of that size
around the prompt (`sam_window`), at native resolution. Windows snap to a WINDOW_GRID grid so
clicks close together reuse one embedding; a prompt (box) larger than the window gets a larger
window (SAM then downscales it). If the mask is cut by a window border that is not an image
border, the window is grown around the mask once and SAM runs again.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.weights import resolve_weights  # noqa: F401 (re-exported)

DEFAULT_SAM_MODEL = "mobile_sam.pt"  # ~40 MB, fast enough on CPU
SAM_MODEL_CHOICES = ["mobile_sam.pt", "sam2.1_b.pt"]  # second = more precise, slower

TaskMode = Literal["detect", "seg"]


@dataclass
class SamPrompt:
    """Prompts for one object: include (label 1) / exclude (label 0) points and/or a box."""

    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    box: tuple[float, float, float, float] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.points and self.box is None

    def to_kwargs(self) -> dict:
        """Arguments for the Ultralytics SAM predictor. Points are wrapped as ONE object with
        n points, shape (1, n, 2) / labels (1, n); a flat (n, 2) list would mean n objects."""
        kwargs: dict = {}
        if self.points:
            kwargs["points"] = [[list(p) for p in self.points]]
            kwargs["labels"] = [list(self.labels)]
        if self.box is not None:
            kwargs["bboxes"] = [list(self.box)]
        return kwargs


# ---- mask -> shape -------------------------------------------------------------------

def largest_component(mask: np.ndarray, min_area: int = 4) -> np.ndarray | None:
    """The biggest connected region of a boolean mask (drops SAM speckles), or None."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return None
    return labels == idx


def mask_to_polygon(mask: np.ndarray, tolerance: float = 1.0) -> list[tuple[float, float]] | None:
    """Outer contour of the largest region, simplified with Douglas-Peucker.

    tolerance is the max deviation in pixels (larger -> fewer vertices). Points are pixel
    centres (+0.5) in the label coordinate system.
    """
    region = largest_component(mask)
    if region is None:
        return None
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return [(float(x) + 0.5, float(y) + 0.5) for x, y in approx]


def mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight box around the largest region, covering whole pixels: (x1, y1, x2+1, y2+1)."""
    region = largest_component(mask)
    if region is None:
        return None
    ys, xs = np.nonzero(region)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mask_to_shape(mask: np.ndarray, task: TaskMode, class_id: int, score: float | None,
                  tolerance: float = 1.0) -> BBox | Polygon | None:
    """SAM mask -> a pending suggestion (status=suggested, source=sam) for the project's task:
    a polygon for seg, a box for detect. None if the mask is empty."""
    score = None if score is None else float(min(max(score, 0.0), 1.0))
    common = dict(class_id=class_id, status="suggested", source="sam", score=score)
    if task == "seg":
        points = mask_to_polygon(mask, tolerance)
        if points is not None:
            try:
                return Polygon(points=points, **common)
            except ValueError:
                pass  # degenerate polygon: fall back to a box
    box = mask_to_bbox(mask)
    if box is None:
        return None
    try:
        return BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], **common)
    except ValueError:
        return None


# ---- tiled SAM window ----------------------------------------------------------------

SAM_WINDOW = 1024  # SAM's input size: a window this large keeps native resolution
WINDOW_MARGIN = 64  # context kept around the prompt
WINDOW_GRID = 256  # window origins snap to this grid, so nearby clicks share an embedding

Window = tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels


def prompt_bounds(prompt: SamPrompt) -> tuple[float, float, float, float]:
    xs = [p[0] for p in prompt.points]
    ys = [p[1] for p in prompt.points]
    if prompt.box is not None:
        xs += [prompt.box[0], prompt.box[2]]
        ys += [prompt.box[1], prompt.box[3]]
    return min(xs), min(ys), max(xs), max(ys)


def _axis_window(lo: float, hi: float, extent: int, size: int, margin: int, grid: int) -> tuple[int, int]:
    if extent <= size:
        return 0, extent
    lo, hi = max(0, int(np.floor(lo - margin))), min(extent, int(np.ceil(hi + margin)))
    if hi - lo >= size:
        return lo, hi  # prompt larger than a window: cover it (SAM downscales)
    start = int(round(((lo + hi) / 2 - size / 2) / grid)) * grid
    start = min(max(start, hi - size, 0), lo, extent - size)  # contains [lo, hi], inside the image
    return start, start + size


def sam_window(bounds: tuple[float, float, float, float], width: int, height: int,
               size: int = SAM_WINDOW, margin: int = WINDOW_MARGIN, grid: int = WINDOW_GRID) -> Window:
    """Image window SAM runs on for a prompt with these bounds: the whole image if it fits,
    else a size×size window around the prompt (larger only if the prompt itself is)."""
    x0, x1 = _axis_window(bounds[0], bounds[2], width, size, margin, grid)
    y0, y1 = _axis_window(bounds[1], bounds[3], height, size, margin, grid)
    return x0, y0, x1, y1


def _grown_window(mask: np.ndarray, window: Window, prompt: SamPrompt, width: int, height: int) -> Window | None:
    """A window that also covers the mask if it is cut by a window border inside the image."""
    region = largest_component(mask)
    if region is None:
        return None
    x0, y0, x1, y1 = window
    ys, xs = np.nonzero(region)
    h, w = mask.shape
    left, top = xs.min() == 0 and x0 > 0, ys.min() == 0 and y0 > 0
    right, bottom = xs.max() == w - 1 and x1 < width, ys.max() == h - 1 and y1 < height
    if not (left or top or right or bottom):
        return None
    # the object goes on past a cut border: look one more window further on that side
    px0, py0, px1, py1 = prompt_bounds(prompt)
    bounds = (min(px0, x0 + xs.min() - SAM_WINDOW * left), min(py0, y0 + ys.min() - SAM_WINDOW * top),
              max(px1, x0 + xs.max() + 1 + SAM_WINDOW * right), max(py1, y0 + ys.max() + 1 + SAM_WINDOW * bottom))
    grown = sam_window(bounds, width, height)
    return grown if grown != window else None


def shift_prompt(prompt: SamPrompt, dx: float, dy: float) -> SamPrompt:
    box = None if prompt.box is None else (prompt.box[0] + dx, prompt.box[1] + dy, prompt.box[2] + dx, prompt.box[3] + dy)
    return SamPrompt([(x + dx, y + dy) for x, y in prompt.points], list(prompt.labels), box)


# ---- model ---------------------------------------------------------------------------

class SamAssistant:
    """Wraps an Ultralytics SAM predictor. Thread-safe; meant to run on one worker thread."""

    def __init__(self, model: str = DEFAULT_SAM_MODEL, weights_dir: Path | None = None,
                 device: str | None = None) -> None:
        self.model = model
        self.requested_device = device  # None = resolve_device: CUDA GPU, Apple MPS or CPU
        self.device: str | None = None  # resolved when the model is loaded
        self.weights_dir = Path(weights_dir) if weights_dir else Path.home() / ".cache" / "DefectLab" / "weights"
        self._predictor = None
        self._image_key: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        with self._lock:
            if self._predictor is not None:
                return
            from ultralytics.models.sam import Predictor, SAM2Predictor

            from defect_inspection.platform_utils import resolve_device

            self.device = resolve_device(self.requested_device, "sam")
            weights = resolve_weights(self.model, self.weights_dir)
            cls = SAM2Predictor if "sam2" in Path(weights).stem else Predictor
            overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                             model=str(weights), save=False, verbose=False, device=self.device)
            self._predictor = cls(overrides=overrides)

    def set_image(self, image8: np.ndarray, key: str) -> None:
        """Encode the image once per key (the expensive step); later prompts reuse it."""
        self.load()
        with self._lock:
            if key == self._image_key:
                return
            bgr = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR) if image8.ndim == 2 else image8
            self._predictor.set_image(np.ascontiguousarray(bgr))
            self._image_key = key

    def predict(self, prompt: SamPrompt) -> tuple[np.ndarray, float | None] | None:
        """(boolean mask at image size, score) for the prompt, or None if SAM found nothing."""
        if prompt.is_empty:
            return None
        with self._lock:
            results = self._predictor(**prompt.to_kwargs())
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None
        r = results[0]
        mask = r.masks.data[0].cpu().numpy().astype(bool)
        score = float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes.conf) else None
        return mask, score

    @staticmethod
    def window_key(key: str, window: Window) -> str:
        return f"{key}#{window[0]},{window[1]},{window[2]},{window[3]}"

    def prepare(self, image8: np.ndarray, key: str) -> bool:
        """Encode ahead of the first click when one window covers the whole image (a larger
        image is encoded per click window). Returns True if it encoded something."""
        h, w = image8.shape[:2]
        if w > SAM_WINDOW or h > SAM_WINDOW:
            self.load()
            return False
        self.set_image(image8, self.window_key(key, (0, 0, w, h)))
        return True

    def needs_encoding(self, image8: np.ndarray, key: str, prompt: SamPrompt) -> bool:
        h, w = image8.shape[:2]
        return self.window_key(key, sam_window(prompt_bounds(prompt), w, h)) != self._image_key

    def predict_tiled(self, image8: np.ndarray, key: str, prompt: SamPrompt) -> tuple[np.ndarray, float | None] | None:
        """(boolean mask at image size, score): SAM on the native-resolution window around the
        prompt (the whole image when it is no larger than SAM_WINDOW)."""
        if prompt.is_empty:
            return None
        h, w = image8.shape[:2]
        window = sam_window(prompt_bounds(prompt), w, h)
        best: tuple[tuple[np.ndarray, float | None], Window] | None = None
        for _ in range(2):  # the first window, then once more if the mask was cut by its border
            x0, y0, x1, y1 = window
            self.set_image(np.ascontiguousarray(image8[y0:y1, x0:x1]), self.window_key(key, window))
            result = self.predict(shift_prompt(prompt, -x0, -y0))
            if result is None:
                break
            best = (result, window)
            window = _grown_window(result[0], window, prompt, w, h)
            if window is None:
                break
        if best is None:
            return None
        (mask, score), (x0, y0, x1, y1) = best
        full = np.zeros((h, w), bool)
        full[y0:y1, x0:x1] = mask[:y1 - y0, :x1 - x0]
        return full, score
