"""Tiling inference for crop models (PROMPTS step 18, PRD F7.6, F6.4)."""

import json

import cv2
import numpy as np
import pytest

from defect_inspection.core import evaluate as E
from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.crop import (
    TiledPredictor,
    build_crop_dataset,
    merge_predictions,
    tile_windows,
    tiled_predict,
)
from defect_inspection.core.crop_config import CropConfig, TilingParams
from defect_inspection.core.trainer import RunSpec, TrainConfig, load_spec, prepare_run


def pred(x1, y1, x2, y2, score=0.9, cls=0):
    return E.Prediction(BBox(class_id=cls, x1=x1, y1=y1, x2=x2, y2=y2, status="suggested", source="model",
                             score=score))


def blob_predict(tile, min_area=30):
    """Fake model in TILE coordinates: every dark blob is a detection; score grows with size, so a
    blob cut by the tile border scores lower than the whole blob in the neighbouring tile."""
    gray = tile if tile.ndim == 2 else tile[..., 0]
    n, _, stats, _ = cv2.connectedComponentsWithStats((gray < 100).astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append(pred(x, y, x + w, y + h, score=min(0.99, area / 1000)))
    return out


def blobs_image(w, h, rects):
    img = np.full((h, w, 3), 200, np.uint8)
    for x1, y1, x2, y2 in rects:
        img[y1:y2, x1:x2] = 30
    return img


# ---- coordinate restore and merge ---------------------------------------------------------------

def test_tiles_restore_original_coordinates():
    # 1000x700 image, 400 px tiles with 100 px overlap -> tile origins x 0/300/600, y 0/300
    rects = [(20, 30, 60, 70), (650, 620, 700, 690), (930, 100, 990, 160), (450, 350, 500, 380)]
    image = blobs_image(1000, 700, rects)
    params = TilingParams(tile_size=400, overlap_px=100)
    got = sorted(p.shape.bounds() for p in tiled_predict(blob_predict, image, params))
    assert got == sorted(tuple(float(v) for v in r) for r in rects)


def test_defect_on_tile_border_merged_into_one():
    # x 380..420 straddles the first tile's right edge (400); the second tile (300..700) has it whole
    image = blobs_image(1000, 400, [(380, 100, 420, 140)])
    params = TilingParams(tile_size=400, overlap_px=100)
    raw = []
    for w in tile_windows(1000, 400, 400, 100):
        raw += blob_predict(image[w.y1:w.y2, w.x1:w.x2])
    assert len(raw) == 2  # a cut piece + the whole blob
    (merged,) = tiled_predict(blob_predict, image, params)
    assert merged.shape.bounds() == (380, 100, 420, 140)


def test_merge_rules():
    a, dup = pred(0, 0, 100, 100, 0.9), pred(2, 2, 100, 100, 0.8)
    other_class = pred(0, 0, 100, 100, 0.7, cls=1)
    piece = pred(60, 0, 100, 100, 0.95)  # inside `a`: IoU 0.4 but IoS 1.0
    separate = pred(200, 200, 260, 260, 0.5)
    kept = merge_predictions([dup, a, other_class, separate], 0.5, 0.7)
    assert {(k.shape.class_id, k.score) for k in kept} == {(0, 0.9), (1, 0.7), (0, 0.5)}
    assert [k.score for k in merge_predictions([a, piece], 0.5, 0.7)] == [0.95]  # higher score wins
    assert len(merge_predictions([a, piece], 0.5, 1.01)) == 2  # IoS rule off


def test_polygons_restored():
    def poly_predict(tile):
        return [E.Prediction(Polygon(class_id=0, points=[(10, 10), (50, 10), (30, 40)], status="suggested",
                                     source="model", score=0.8))] if tile.shape[1] == 300 else []

    image = np.zeros((300, 800, 3), np.uint8)
    out = tiled_predict(poly_predict, image, TilingParams(tile_size=300, overlap_px=50))
    xs = sorted(p.shape.points[0][0] for p in out)
    assert xs == [10, 260, 510]  # origins 0, 250, 500 (last tile flush: 800 - 300 = 500)
    assert all(isinstance(p.shape, Polygon) for p in out)


def test_image_smaller_than_tile_is_one_call():
    calls = []
    image = np.zeros((100, 150), np.uint8)
    tiled_predict(lambda t: calls.append(t.shape) or [], image, TilingParams(tile_size=640, overlap_px=64))
    assert calls == [(100, 150)]


def test_tiled_predictor_interface():
    class Inner:
        task, weights = "detect", "x.pt"

        def __init__(self):
            self.seen = []

        def predict(self, image8, conf=0.25, iou=0.7):
            self.seen.append((image8.shape, conf, iou))
            return blob_predict(image8)

    inner = Inner()
    tp = TiledPredictor(inner, TilingParams(tile_size=256, overlap_px=32))
    preds = tp.predict(blobs_image(500, 256, [(300, 50, 340, 90)]), 0.4, 0.6)
    assert [p.shape.bounds() for p in preds] == [(300, 50, 340, 90)]
    assert {s for s in inner.seen} == {((256, 256, 3), 0.4, 0.6)} and len(inner.seen) == 3
    assert "256" in tp.describe() and tp.task == "detect"


# ---- runs remember their tiling --------------------------------------------------------------------

@pytest.fixture
def crop_project(tmp_path):
    from test_crop import make_project

    p, _ = make_project(tmp_path)
    p.config.crop = CropConfig(use_roi=False, use_defect=False, use_background=False, use_tiles=True,
                               tile_size=256, overlap_unit="px", overlap_px=48, multiplier=0)
    p.save()
    build_crop_dataset(p)
    return p


def test_run_records_tiling(crop_project):
    run = prepare_run(crop_project, TrainConfig(model="yolo11n", imgsz=256, dataset="crop"), crop_project.root / "w")
    spec = load_spec(run)
    assert spec.tiling() == spec.crop
    assert (spec.crop.tile_size, spec.crop.overlap_px) == (256, 48) and spec.crop.crop["use_tiles"]
    # rebuilding the dataset with other settings does not change an existing run
    crop_project.config.crop.tile_size = 128
    build_crop_dataset(crop_project)
    assert load_spec(run).crop.tile_size == 256


def test_defect_crop_run_gets_20_percent_overlap(tmp_path):
    from test_crop import make_project

    p, _ = make_project(tmp_path)  # defect + background crops of 128 px
    build_crop_dataset(p)
    spec = load_spec(prepare_run(p, TrainConfig(model="yolo11n", imgsz=128, dataset="crop"), p.root / "w"))
    assert (spec.crop.tile_size, spec.crop.overlap_px) == (128, 26)


def test_full_run_has_no_tiling_and_old_crop_run_falls_back(crop_project):
    from defect_inspection.core.dataset import build_full_dataset

    build_full_dataset(crop_project)
    full = load_spec(prepare_run(crop_project, TrainConfig(model="yolo11n"), crop_project.root / "w"))
    assert full.crop is None and full.tiling() is None
    old = RunSpec.model_validate({**full.model_dump(), "train": {**full.train.model_dump(), "dataset": "crop",
                                                                   "imgsz": 320}})
    assert (old.tiling().tile_size, old.tiling().overlap_px) == (320, 64)


def test_crop_val_images_are_originals(crop_project):
    paths = E.val_images(crop_project, "crop")
    assert paths and all("__" not in p and p.endswith(".png") for p in paths)
    assert set(paths) <= {e.path for e in crop_project.config.images}
    assert set(paths) == set(E.val_images(crop_project, "full")) or not (crop_project.dataset_dir / "full").exists()


def test_predictor_for_run_wraps_crop_runs(crop_project, monkeypatch):
    monkeypatch.setattr(E, "Predictor", lambda w, task, imgsz, device=None: type("P", (), {"task": task, "weights": w})())
    run = prepare_run(crop_project, TrainConfig(model="yolo11n", imgsz=256, dataset="crop"), crop_project.root / "w")
    p = E.predictor_for_run("best.pt", load_spec(run))
    assert isinstance(p, TiledPredictor) and p.params.tile_size == 256


# ---- ONNX: metadata and the same tiling on the exported model ----------------------------------------

def test_onnx_crop_export_metadata_and_tiled_inference(crop_project):
    from defect_inspection.core import exporter as X
    from defect_inspection.core.roi import Roi, RoiStore
    from test_exporter import untrained_run

    RoiStore(crop_project.root).set_rois("d0.png", [], [Roi(x1=10, y1=20, x2=210, y2=220)])
    run = untrained_run(crop_project, imgsz=256, dataset="crop")
    result = X.export_run(crop_project, run, crop_project.root / "exports" / "crop.onnx")
    assert result.verification.passed
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    crop = meta["crop"]
    assert crop["tiling"]["tile_size"] == 256 and crop["tiling"]["overlap_px"] == 48
    assert crop["sources"] == ["tiles"] and crop["common_rois"] == [[10, 20, 210, 220]]
    assert "256px" in crop["how"] and meta["train"]["dataset"] == "crop"

    onnx_pred = E.predictor_for_onnx(result.onnx_path)
    pt_pred = E.predictor_for_run(run / "train" / "weights" / "best.pt", load_spec(run))
    assert isinstance(onnx_pred, TiledPredictor) and onnx_pred.params == pt_pred.params
    image = E.load_training_image(crop_project, "d0.png")  # 400x300 -> 2x2 tiles of 256
    a, b = pt_pred.predict(image, 0.001, 0.7), onnx_pred.predict(image, 0.001, 0.7)
    ok, _, _ = X.compare_predictions(a, b)
    assert ok


def test_full_export_has_no_crop_metadata(tmp_path):
    from defect_inspection.core import exporter as X
    from test_exporter import make_project as make_full, untrained_run

    p = make_full(tmp_path)
    run = untrained_run(p)
    assert X.crop_metadata(p, load_spec(run)) is None


def test_export_panel_mentions_tiling(crop_project):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from defect_inspection.ui.main_window import MainWindow
    from test_exporter import untrained_run

    app = QApplication.instance() or QApplication([])
    run = untrained_run(crop_project, imgsz=256, dataset="crop")
    w = MainWindow(QSettings(str(crop_project.root / "s.ini"), QSettings.Format.IniFormat),
                   cache_root=crop_project.root / "cache")
    try:
        w.load_project(crop_project)
        panel = w.export_panel
        panel.refresh_runs()
        assert panel.current_run.path == run
        assert "타일 256px, overlap 48px" in panel.run_info.text()
        assert "_crop_yolo11n_" in panel.name.text()
        app.processEvents()
    finally:
        w.train_panel.shutdown()
        w.project_dirty = False
        w.close()
