"""Edge filter on predictions (PRD F9.3): keep only predictions on the object's outermost edge.

Applied inside DefectLab only — Result tab, label suggestions, comparison and report — by wrapping a
predictor (`Predictor`, `TiledPredictor`, test fakes: anything with `predict(image8, conf, iou)` and
`task`). Exported ONNX/CoreML models are not filtered. Wrap after tiling, so boxes are judged in
original-image coordinates. Training is unchanged.

The filter runs after the model's NMS (and the tile merge): a higher-scored box reaching off the edge
can suppress a tighter one on the line, and then both are gone. Filtering before NMS changed nothing
on the sample project (3 trained runs, 26 labels), so it is not done. Cost: ~10-16 ms per box on
the samples (0.14 s per image at conf 0.01); up to `max_det` boxes per image at very low conf."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from defect_inspection.core.edge_band import EdgeHit, edge_under_box
from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.core.evaluate import ImageResult, load_training_image, match
from defect_inspection.core.project import Project


@dataclass(frozen=True)
class EdgeFilterResult:
    kept: tuple  # Predictions on the edge, in input order
    dropped: tuple  # the others
    hits: tuple[EdgeHit, ...]  # one per kept prediction


def filter_predictions(image8: np.ndarray, predictions: list, cfg: EdgeFilterConfig) -> EdgeFilterResult:
    """Split `predictions` (evaluate.Prediction) by whether their box lies on the outermost edge."""
    kept, dropped, hits = [], [], []
    for p in predictions:
        hit = edge_under_box(image8, p.shape.bounds(), cfg)
        if hit is None:
            dropped.append(p)
        else:
            kept.append(p)
            hits.append(hit)
    return EdgeFilterResult(tuple(kept), tuple(dropped), tuple(hits))


class EdgeFilteredPredictor:
    """Same interface as evaluate.Predictor; predictions off the outermost edge are removed."""

    def __init__(self, predictor, cfg: EdgeFilterConfig) -> None:
        self.predictor = predictor
        self.cfg = cfg.model_copy()  # project edits must not change a running job
        self.task = predictor.task
        self.weights = getattr(predictor, "weights", None)

    def predict(self, image8: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> list:
        return list(filter_predictions(image8, self.predictor.predict(image8, conf, iou), self.cfg).kept)

    def describe(self) -> str:
        return "테두리 필터: " + settings_text(self.cfg)


def settings_text(cfg: EdgeFilterConfig) -> str:
    polarity = {"dark": "어두운 선", "bright": "밝은 선", "both": "어두운·밝은 선"}[cfg.line_polarity]
    return f"최외각 {polarity}, 박스 양옆 {cfg.along_px}px, 허용 거리 {cfg.touch_px}px"


def with_edge_filter(predictor, cfg: EdgeFilterConfig):
    """`predictor` wrapped in the edge filter if `cfg.enabled`, else unchanged."""
    return EdgeFilteredPredictor(predictor, cfg) if cfg.enabled else predictor


def apply_edge_filter(project: Project, results: list[ImageResult], cfg: EdgeFilterConfig,
                      progress: Callable[[int, int], None] | None = None,
                      cancel: Callable[[], bool] | None = None) -> list[ImageResult]:
    """Unfiltered Val results → new results with `cfg` applied (kept / dropped / edge hits), without
    predicting again: the Result tab keeps the raw predictions and re-filters on a settings change.
    `cfg` off → `results` unchanged. Cancelled → []."""
    if not cfg.enabled:
        return results
    out = []
    for i, r in enumerate(results, 1):
        if cancel is not None and cancel():
            return []
        image = load_training_image(project, r.path)
        if image is None:
            out.append(r)  # deleted since it was predicted: shown as it was
        else:
            f = filter_predictions(image, r.predictions, cfg)
            matched, missed = match(r.ground_truth, list(f.kept))
            out.append(replace(r, predictions=list(f.kept), matched=matched, missed=missed,
                               dropped=list(f.dropped), edge_hits=list(f.hits)))
        if progress is not None:
            progress(i, len(results))
    return out


@dataclass(frozen=True)
class LabelCheck:
    """One confirmed label judged by the edge rule ("테두리 검사"): labels are the truth, so a label
    that fails shows where the rule would hide a real defect."""

    path: str
    class_name: str
    bounds: tuple[float, float, float, float]
    hit: EdgeHit | None

    @property
    def passed(self) -> bool:
        return self.hit is not None


def check_labels(project: Project, cfg: EdgeFilterConfig,
                 progress: Callable[[int, int], None] | None = None,
                 cancel: Callable[[], bool] | None = None) -> list[LabelCheck]:
    """Every confirmed label of every project image, judged with `cfg` (also when it is off)."""
    names = [c.name for c in project.config.classes]
    entries = [e for e in project.config.images if project.label_status(e).confirmed]
    out = []
    for i, entry in enumerate(entries, 1):
        if cancel is not None and cancel():
            break
        image = load_training_image(project, entry.path)
        if image is not None:
            for shape in project.load_labels(entry).confirmed():
                name = names[shape.class_id] if shape.class_id < len(names) else str(shape.class_id)
                out.append(LabelCheck(entry.path, name, shape.bounds(), edge_under_box(image, shape.bounds(), cfg)))
        if progress is not None:
            progress(i, len(entries))
    return out
