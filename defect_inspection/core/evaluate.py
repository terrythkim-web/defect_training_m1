"""Model results (PRD F5) and model-based label suggestions (F2.8 ①, F5.5). Qt-free.

- Predictions are returned as the app's own shapes (BBox for detect, Polygon for seg) in original
  image pixel coordinates, so they can be drawn, compared and stored like labels.
- Images go through the same to_training_8bit conversion as the dataset, so inference sees what
  training saw (16-bit >> 8, project channel count).
- Matching for the Val grid is greedy by score: a prediction is a true positive if it has the same
  class as an unmatched ground-truth shape and their boxes overlap with IoU >= 0.5.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

import numpy as np

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imread_unicode, resolve_device

Shape = BBox | Polygon
TaskMode = Literal["detect", "seg"]
MATCH_IOU = 0.5


@dataclass
class Prediction:
    shape: Shape  # status=suggested, source=model, score set

    @property
    def score(self) -> float:
        return float(self.shape.score or 0.0)


class Predictor:
    """Loads a trained checkpoint once and predicts on uint8 images. Thread-safe."""

    def __init__(self, weights: str | Path, task: TaskMode, imgsz: int = 640, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        self.task = task
        self.imgsz = imgsz
        # None/"auto" = platform_utils.resolve_device: CUDA GPU, Apple MPS or CPU (PRD 5.2)
        self.device = resolve_device(device, "predict")
        self.model = YOLO(str(self.weights), task="segment" if task == "seg" else "detect")  # never guess
        self._lock = threading.Lock()

    def predict(self, image8: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> list[Prediction]:
        with self._lock:
            result = self.model.predict(image8, imgsz=self.imgsz, conf=conf, iou=iou, verbose=False,
                                        device=self.device)[0]
        return results_to_predictions(result, self.task)


def results_to_predictions(result, task: TaskMode) -> list[Prediction]:
    """Ultralytics Results -> Predictions (masks become polygons for seg; boxes otherwise)."""
    out: list[Prediction] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    polys = result.masks.xy if (task == "seg" and result.masks is not None) else None
    for i in range(len(xyxy)):
        common = dict(class_id=int(cls[i]), status="suggested", source="model",
                      score=float(min(max(conf[i], 0.0), 1.0)))
        shape: Shape | None = None
        if polys is not None and len(polys[i]) >= 3:
            try:
                shape = Polygon(points=[(float(x), float(y)) for x, y in polys[i]], **common)
            except ValueError:
                shape = None
        if shape is None:
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            try:
                shape = BBox(x1=x1, y1=y1, x2=x2, y2=y2, **common)
            except ValueError:
                continue
        out.append(Prediction(shape))
    return out


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ImageResult:
    path: str  # image path relative to images/
    width: int
    height: int
    ground_truth: list[Shape]
    predictions: list[Prediction]
    matched: list[bool] = field(default_factory=list)  # per prediction: true positive?
    missed: list[bool] = field(default_factory=list)  # per ground truth: not found?
    dropped: list[Prediction] = field(default_factory=list)  # removed by the edge filter (F9)
    edge_hits: list = field(default_factory=list)  # edge_band.EdgeHit per kept prediction (F9)

    @property
    def tp(self) -> int:
        return sum(self.matched)

    @property
    def fp(self) -> int:
        return len(self.matched) - self.tp

    @property
    def fn(self) -> int:
        return sum(self.missed)

    @property
    def kind(self) -> str:
        """ok / missed / false / both (for the 오검·미검 filter)."""
        if self.fn and self.fp:
            return "both"
        return "missed" if self.fn else "false" if self.fp else "ok"


def match(ground_truth: list[Shape], predictions: list[Prediction], iou: float = MATCH_IOU) -> tuple[list[bool], list[bool]]:
    """(matched per prediction, missed per ground truth), greedy by prediction score."""
    used = [False] * len(ground_truth)
    matched = [False] * len(predictions)
    for pi in sorted(range(len(predictions)), key=lambda i: -predictions[i].score):
        pred = predictions[pi].shape
        best, best_iou = -1, iou
        for gi, gt in enumerate(ground_truth):
            if used[gi] or gt.class_id != pred.class_id:
                continue
            v = box_iou(gt.bounds(), pred.bounds())
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used[best] = True
            matched[pi] = True
    return matched, [not u for u in used]


def load_training_image(project: Project, path: str) -> np.ndarray | None:
    raw = imread_unicode(project.image_abspath(path))
    return None if raw is None else to_training_8bit(raw, project.channels)


def val_images(project: Project, dataset: str = "full") -> list[str]:
    """Project image paths in the built dataset's Val split (dataset files keep the image stems;
    crop files are named <stem>__<kind><n>, so a crop dataset maps back to its ORIGINAL images,
    which crop models are evaluated on with tiling inference)."""
    val_dir = project.dataset_dir / dataset / "images" / "val"
    if not val_dir.is_dir():
        return []
    stems = {p.stem.split("__")[0] if dataset == "crop" else p.stem for p in val_dir.iterdir() if p.is_file()}
    return [e.path for e in project.config.images if PurePosixPath(e.path).stem in stems]


def evaluate_images(project: Project, predictor: Predictor, paths: list[str], conf: float, iou: float,
                    progress: Callable[[int, int], None] | None = None,
                    cancel: Callable[[], bool] | None = None) -> list[ImageResult]:
    results = []
    for i, path in enumerate(paths, 1):
        if cancel is not None and cancel():
            break
        image = load_training_image(project, path)
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = project.load_labels(path, size=(w, h)).confirmed()
        preds = predictor.predict(image, conf, iou)
        m, missed = match(gt, preds)
        results.append(ImageResult(path, w, h, gt, preds, m, missed))
        if progress is not None:
            progress(i, len(paths))
    return results


# ---- label suggestions (F2.8 ①) ------------------------------------------------------------

def unlabeled_images(project: Project) -> list[str]:
    """Images without confirmed labels that are not marked OK: the ones that need labeling."""
    return [e.path for e in project.config.images
            if not e.is_ok and project.label_status(e).confirmed == 0]


@dataclass
class SuggestionReport:
    images: int = 0  # images predicted
    with_suggestions: int = 0
    suggestions: int = 0
    paths: list[str] = field(default_factory=list)  # images whose labels changed


def generate_suggestions(project: Project, predictor: Predictor, paths: list[str], conf: float = 0.25,
                         iou: float = 0.7, progress: Callable[[int, int], None] | None = None,
                         cancel: Callable[[], bool] | None = None) -> SuggestionReport:
    """Predict and store results as pending `source="model"` suggestions. Earlier model suggestions
    of those images are replaced; confirmed labels and SAM suggestions are kept."""
    report = SuggestionReport()
    for i, path in enumerate(paths, 1):
        if cancel is not None and cancel():
            break
        image = load_training_image(project, path)
        if image is None:
            continue
        h, w = image.shape[:2]
        ann: ImageAnnotations = project.load_labels(path, size=(w, h))
        preds = predictor.predict(image, conf, iou)
        had = any(s.source == "model" for s in ann.suggestions())
        ann.add_suggestions([p.shape for p in preds], source="model", replace=True)
        report.images += 1
        if preds:
            report.with_suggestions += 1
            report.suggestions += len(preds)
        if preds or had:
            project.save_labels(ann)
            report.paths.append(path)
        if progress is not None:
            progress(i, len(paths))
    return report


def predictor_for_run(weights: str | Path, spec, device: str | None = None) -> Predictor:
    """Predictor of a training run (RunSpec): crop runs get tiling inference (F7.6)."""
    predictor = Predictor(weights, spec.task, spec.train.imgsz, device)
    tiling = spec.tiling()
    if tiling is None:
        return predictor
    from defect_inspection.core.crop import TiledPredictor

    return TiledPredictor(predictor, tiling)


def predictor_for_onnx(onnx_path: str | Path) -> Predictor:
    """Predictor of an exported .onnx, configured from its <name>.metadata.json (task, input size,
    and tiling for crop models), the way an external program would use it."""
    import json

    from defect_inspection.core.crop_config import TilingParams
    from defect_inspection.core.exporter import sidecar_paths

    meta_path, _ = sidecar_paths(Path(onnx_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    predictor = Predictor(onnx_path, meta["task"], int(meta["preprocess"]["letterbox"]["size"]))
    if not meta.get("crop"):
        return predictor
    from defect_inspection.core.crop import TiledPredictor

    return TiledPredictor(predictor, TilingParams.model_validate(meta["crop"]["tiling"]))


def quick_suggest_config():
    """Short yolo11n run on the confirmed labels, only to bootstrap suggestions (F2.8 ①)."""
    from defect_inspection.core.trainer import TrainConfig

    return TrainConfig(model="yolo11n", epochs=30, imgsz=640, patience=10)
