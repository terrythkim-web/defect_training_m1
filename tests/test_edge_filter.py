"""Edge filter applied to predictions (PRD F9.3): `EdgeFilteredPredictor` keeps only predictions on the
object's outermost edge; `with_edge_filter` wraps a predictor when the project turns it on."""

from __future__ import annotations

import numpy as np
import pytest
from test_compare import BlobModel, make_project_with_runs
from test_edge_band import EDGE_ROW, panel

from defect_inspection.core import compare as C
from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.edge_filter import EdgeFilteredPredictor, filter_predictions, with_edge_filter
from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.core.evaluate import Prediction

ON = EdgeFilterConfig(enabled=True)


def pred(x1, y1, x2, y2, score=0.9):
    return Prediction(BBox(class_id=0, x1=x1, y1=y1, x2=x2, y2=y2, status="suggested", source="model",
                           score=score))


ON_EDGE = pred(180, EDGE_ROW - 15, 220, EDGE_ROW + 15)
INSIDE = pred(180, 385, 220, 415, score=0.95)


class FixedModel:
    """Fake model returning the same predictions for every image."""

    def __init__(self, preds, task="detect"):
        self.preds, self.task, self.weights = preds, task, "best.pt"
        self.calls: list[tuple[float, float]] = []

    def predict(self, image8, conf=0.25, iou=0.7):
        self.calls.append((conf, iou))
        return list(self.preds)


def test_filter_keeps_only_boxes_on_the_edge():
    result = filter_predictions(panel(), [INSIDE, ON_EDGE], ON)
    assert result.kept == (ON_EDGE,) and result.dropped == (INSIDE,)
    assert len(result.hits) == 1 and abs(result.hits[0].position - EDGE_ROW) <= 2


def test_filter_judges_seg_polygons_by_their_box():
    poly = Prediction(Polygon(class_id=0, points=[(180, EDGE_ROW - 10), (220, EDGE_ROW - 10), (200, EDGE_ROW + 10)],
                              status="suggested", source="model", score=0.8))
    assert filter_predictions(panel(), [poly], ON).kept == (poly,)


def test_filter_without_predictions():
    result = filter_predictions(panel(), [], ON)
    assert (result.kept, result.dropped, result.hits) == ((), (), ())


def test_wrapped_predictor_passes_thresholds_and_attributes():
    model = FixedModel([INSIDE, ON_EDGE], task="seg")
    wrapped = EdgeFilteredPredictor(model, ON)
    assert wrapped.predict(panel(), 0.1, 0.5) == [ON_EDGE]
    assert model.calls == [(0.1, 0.5)]
    assert (wrapped.task, wrapped.weights, wrapped.predictor) == ("seg", "best.pt", model)
    assert "테두리 필터" in wrapped.describe()


def test_with_edge_filter_only_when_enabled():
    model = FixedModel([])
    assert with_edge_filter(model, EdgeFilterConfig()) is model
    wrapped = with_edge_filter(model, ON)
    assert isinstance(wrapped, EdgeFilteredPredictor) and wrapped.cfg == ON


def test_settings_are_frozen_at_wrapping():
    """Later edits of the project's settings do not change a predictor already running a job."""
    cfg = EdgeFilterConfig(enabled=True)
    wrapped = with_edge_filter(FixedModel([]), cfg)
    cfg.touch_px = 0
    assert wrapped.cfg.touch_px == 12


# ---- fair comparison (F8) -----------------------------------------------------------------------

@pytest.fixture
def project_with_full_run(tmp_path):
    p, runs = make_project_with_runs(tmp_path)
    return p, runs[0]


def test_comparison_applies_and_records_the_filter(project_with_full_run):
    """The blob images have no edge line: with the filter on, every prediction is dropped."""
    p, run = project_with_full_run

    def factory(weights, spec, device):
        return BlobModel([])

    off = C.evaluate_run(p, run, C.fair_val_images(p), make_predictor=factory, devices=[])
    assert off.metrics.recall == pytest.approx(1.0) and off.edge_filter is None

    p.config.edge_filter = EdgeFilterConfig(enabled=True, touch_px=5)
    on = C.evaluate_run(p, run, C.fair_val_images(p), make_predictor=factory, devices=[])
    assert on.metrics.recall == 0.0 and on.metrics.fp == 0
    assert on.edge_filter["enabled"] and on.edge_filter["touch_px"] == 5
    assert on.summary()["edge_filter"]["touch_px"] == 5


def test_blank_image_drops_everything():
    model = FixedModel([ON_EDGE])
    assert EdgeFilteredPredictor(model, ON).predict(np.full((300, 400), 128, np.uint8)) == []


def test_report_lists_the_filter(tmp_path):
    from defect_inspection.core import report as R
    from defect_inspection.core.edge_filter_config import EdgeFilterConfig as Cfg

    p, runs = make_project_with_runs(tmp_path)
    p.config.edge_filter = Cfg(enabled=True, line_polarity="bright", touch_px=7)
    results = C.compare_runs(p, runs, make_predictor=lambda w, s, d: BlobModel([]), devices=[])
    html = R.build_report(p, results, runs).read_text(encoding="utf-8")
    assert "테두리 필터" in html and "최외각 밝은 선" in html and "허용 거리 7px" in html
    assert R.read_report_data(R.list_reports(p)[0].path)["runs"][0]["summary"]["edge_filter"]["touch_px"] == 7


# ---- export metadata: the filter is not part of the exported model ---------------------------------

def test_export_metadata_notes_the_filter(tmp_path):
    from defect_inspection.core.exporter import edge_filter_metadata
    from defect_inspection.core.project import Project

    p = Project.create(tmp_path / "p")
    assert edge_filter_metadata(p) is None
    p.config.edge_filter = EdgeFilterConfig(enabled=True, touch_px=9)
    meta = edge_filter_metadata(p)
    assert meta["applied_in_model"] is False and "DefectLab" in meta["note"]
    assert meta["settings"]["touch_px"] == 9


def test_comparison_uses_one_snapshot_for_every_run(tmp_path):
    """Settings edited while a comparison runs must not filter the runs differently."""
    p, runs = make_project_with_runs(tmp_path)
    p.config.edge_filter = EdgeFilterConfig(enabled=True, touch_px=5)

    def edit_midway(done, total, name):
        p.config.edge_filter = EdgeFilterConfig(enabled=False)

    results = C.compare_runs(p, runs, make_predictor=lambda w, s, d: BlobModel([]), devices=[],
                             progress=edit_midway)
    assert [r.edge_filter["touch_px"] for r in results] == [5, 5]
