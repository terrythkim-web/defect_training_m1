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

from dataclasses import dataclass

import numpy as np

from defect_inspection.core.edge_band import EdgeHit, edge_under_box
from defect_inspection.core.edge_filter_config import EdgeFilterConfig


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
