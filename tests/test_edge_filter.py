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


# ---- step 4: refiltering cached results, label check, overlay ------------------------------------

def edge_project(tmp_path):
    """Project with one panel image: one label on the edge, one inside."""
    from defect_inspection.core.annotation import ImageAnnotations
    from defect_inspection.core.project import Project
    from defect_inspection.platform_utils import imwrite_unicode

    src = tmp_path / "src"
    src.mkdir()
    imwrite_unicode(src / "panel.png", panel())
    p = Project.create(tmp_path / "p", color_mode="grayscale")
    p.add_class("crack")
    (entry,) = p.add_images([src / "panel.png"])
    gt = [BBox(class_id=0, x1=180, y1=EDGE_ROW - 15, x2=220, y2=EDGE_ROW + 15),
          BBox(class_id=0, x1=180, y1=385, x2=220, y2=415)]
    p.save_labels(ImageAnnotations(image=entry.path, width=512, height=1024, shapes=gt))
    return p, entry.path, gt


def test_apply_edge_filter_refilters_cached_results(tmp_path):
    from defect_inspection.core.edge_filter import apply_edge_filter
    from defect_inspection.core.evaluate import ImageResult

    p, path, gt = edge_project(tmp_path)
    raw = [ImageResult(path, 512, 1024, gt, [INSIDE, ON_EDGE])]
    (r,) = apply_edge_filter(p, raw, ON)
    assert r.predictions == [ON_EDGE] and r.dropped == [INSIDE] and len(r.edge_hits) == 1
    assert r.matched == [True] and r.missed == [False, True]  # inside label is now a miss
    assert raw[0].predictions == [INSIDE, ON_EDGE]  # the cached raw results are not changed
    assert apply_edge_filter(p, raw, EdgeFilterConfig()) == raw  # off: unfiltered


def test_apply_edge_filter_progress_and_cancel(tmp_path):
    from defect_inspection.core.edge_filter import apply_edge_filter
    from defect_inspection.core.evaluate import ImageResult

    p, path, gt = edge_project(tmp_path)
    raw = [ImageResult(path, 512, 1024, gt, [ON_EDGE])] * 3
    seen = []
    assert len(apply_edge_filter(p, raw, ON, progress=lambda d, t: seen.append((d, t)))) == 3
    assert seen[-1] == (3, 3)
    assert apply_edge_filter(p, raw, ON, cancel=lambda: True) == []


def test_check_labels_judges_every_confirmed_label(tmp_path):
    from defect_inspection.core.edge_filter import check_labels

    p, path, gt = edge_project(tmp_path)
    rows = check_labels(p, EdgeFilterConfig())  # judged whether or not the filter is on
    assert [(r.path, r.class_name, r.passed) for r in rows] == [(path, "crack", True), (path, "crack", False)]
    assert rows[0].bounds == gt[0].bounds() and abs(rows[0].hit.position - EDGE_ROW) <= 2
    assert rows[1].hit is None


def test_overlay_draws_dropped_boxes_and_edge_lines():
    from defect_inspection.core.preview import draw_comparison

    img = np.full((200, 200), 200, np.uint8)
    hit = filter_predictions(panel(), [ON_EDGE], ON).hits[0]
    shifted = Prediction(BBox(class_id=0, x1=80, y1=80, x2=120, y2=120, status="suggested", source="model", score=0.5))
    plain = draw_comparison(img, [], [])
    dropped = draw_comparison(img, [], [], dropped=[shifted.shape])
    assert (plain != dropped).any() and (dropped[80:121, 80:121] != 200).any()
    edged = draw_comparison(np.full((1024, 512), 200, np.uint8), [], [ON_EDGE.shape], edges=[(hit, ON_EDGE.shape)])
    no_edge = draw_comparison(np.full((1024, 512), 200, np.uint8), [], [ON_EDGE.shape])
    assert (edged != no_edge).any()


def test_check_labels_on_a_snapshot_skips_deleted_images(tmp_path):
    """Review fix: the dialog passes the image list taken on the UI thread; an image deleted in the
    meantime (folder sync) is skipped instead of failing the whole check."""
    from defect_inspection.core.edge_filter import check_labels

    p, path, _ = edge_project(tmp_path)
    entries = list(p.config.images)
    p.image_abspath(path).unlink()
    assert check_labels(p, EdgeFilterConfig(), entries=entries) == []
