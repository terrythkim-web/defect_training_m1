"""Full vs Crop comparison (PROMPTS step 19, PRD F8.1-F8.4)."""

import json

import cv2
import numpy as np
import pytest

from defect_inspection.core import compare as C
from defect_inspection.core.annotation import BBox
from defect_inspection.core.crop import TiledPredictor, build_crop_dataset
from defect_inspection.core.crop_config import CropConfig
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.evaluate import Prediction
from defect_inspection.core.trainer import TrainConfig, prepare_run, weights_paths, write_status


def box(x1, y1, x2, y2, cls=0):
    return BBox(class_id=cls, x1=x1, y1=y1, x2=x2, y2=y2)


def pred(x1, y1, x2, y2, score=0.9, cls=0):
    return Prediction(BBox(class_id=cls, x1=x1, y1=y1, x2=x2, y2=y2, status="suggested", source="model",
                           score=score))


def im(path, gt, preds, w=200, h=200):
    return C.ImagePredictions(path, w, h, gt, preds)


# ---- metrics ------------------------------------------------------------------------------------

def test_interpolated_ap():
    assert C.interpolated_ap([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert C.interpolated_ap([0.5, 0.5], [1.0, 0.5]) == pytest.approx(51 / 101)
    assert C.interpolated_ap([], []) == 0.0


def test_perfect_and_shifted_map():
    perfect = [im("a", [box(10, 10, 50, 50)], [pred(10, 10, 50, 50)])]
    m = C.compute_metrics(perfect, ["dent"], 0.25)
    assert (m.precision, m.recall, m.f1, m.map50, m.map50_95) == (1, 1, 1, pytest.approx(1), pytest.approx(1))
    # IoU of 40x40 vs 40x(40*0.75... ) -> 0.75: AP is 1 at IoU 0.5..0.75 (6 of 10 thresholds), 0 above
    shifted = [im("a", [box(0, 0, 40, 40)], [pred(0, 10, 40, 40)])]
    m = C.compute_metrics(shifted, ["dent"], 0.25)
    assert m.map50 == pytest.approx(1) and m.map50_95 == pytest.approx(0.6)


def test_counts_sizes_and_image_judgement():
    images = [
        im("hit", [box(0, 0, 20, 20)], [pred(0, 0, 20, 20, 0.9)]),  # small, found
        im("miss", [box(0, 0, 50, 50)], [pred(0, 0, 50, 50, 0.1)]),  # medium, below conf -> NG judged OK
        im("false", [], [pred(5, 5, 30, 30, 0.8)]),  # OK image judged NG
        im("ok", [], []),
        im("big", [box(0, 0, 150, 150, cls=1)], [pred(0, 0, 150, 150, 0.7, cls=0)]),  # wrong class
    ]
    m = C.compute_metrics(images, ["dent", "scratch"], 0.25)
    assert (m.tp, m.fp, m.fn) == (1, 2, 2)
    assert m.precision == pytest.approx(1 / 3) and m.recall == pytest.approx(1 / 3)
    assert m.size_counts == {"small": 1, "medium": 1, "large": 1}
    assert m.size_recall == {"small": 1.0, "medium": 0.0, "large": 0.0}
    assert (m.ng_images, m.ok_images, m.missed_ng_images, m.false_ng_images) == (3, 2, 1, 1)
    assert m.judge_accuracy == pytest.approx(3 / 5)  # hit, ok, big (NG judged NG, despite the class)
    assert (m.fn_images, m.fp_images) == (2, 2)
    assert m.per_class["dent"]["gt"] == 2 and m.per_class["scratch"]["recall"] == 0.0
    assert m.per_class["scratch"]["ap50"] == 0.0
    r, p = m.pr_curve
    assert len(r) == len(p) == 4 and r[-1] == pytest.approx(2 / 3)  # the 0.1 one counts below conf too


def test_class_without_labels_not_in_map():
    images = [im("a", [box(10, 10, 50, 50)], [pred(10, 10, 50, 50), pred(100, 100, 120, 120, 0.5, cls=1)])]
    m = C.compute_metrics(images, ["dent", "scratch"], 0.25)
    assert m.map50 == pytest.approx(1.0) and m.per_class["scratch"]["ap50"] is None


def test_recommend_rule():
    def fake(map5095, cpu):
        m = C.compute_metrics([], ["dent"], 0.25)
        m.map50_95, m.recall = map5095, map5095 / 2
        return C.RunComparison("r", "r", 0.25, 0.7, m, {}, {"CPU": cpu}, None, {})

    a, b = fake(0.600, 50.0), fake(0.595, 20.0)
    i, why = C.recommend([a, b])
    assert i == 1 and "빠른" in why  # within 1 %p: the faster one
    i, why = C.recommend([fake(0.70, 50.0), fake(0.60, 5.0)])
    assert i == 0 and "가장 높음" in why
    assert C.recommend([fake(0.5, 1), fake(0.9, 1)], priority="Recall")[0] == 1


def test_differing_images_order():
    gt = [box(0, 0, 20, 20)]
    a = C.RunComparison("a", "a", 0.25, 0.7, None, {}, {}, None, {},
                        images=[im("same", gt, [pred(0, 0, 20, 20)]), im("diff", gt, [pred(0, 0, 20, 20)])])
    b = C.RunComparison("b", "b", 0.25, 0.7, None, {}, {}, None, {},
                        images=[im("same", gt, [pred(0, 0, 20, 20)]), im("diff", gt, [pred(90, 90, 99, 99)])])
    assert C.differing_images(a, b) == [("diff", 2), ("same", 0)]


def test_experiment_configs_share_everything_but_dataset():
    base = TrainConfig(model="yolo11n", epochs=7, imgsz=320, seed=3, lr0=0.02)
    full, crop = C.experiment_configs(base)
    assert full.dataset == "full" and crop.dataset == "crop"
    assert full.model_dump(exclude={"dataset"}) == crop.model_dump(exclude={"dataset"})


def test_file_size_text():
    assert C.file_size_text(None) == "-" and C.file_size_text(500) == "500 B"
    assert C.file_size_text(10 * 1024 * 1024) == "10.0 MB"


# ---- runs: fair evaluation with a fake model ---------------------------------------------------------

def blob_factory(seen=None):
    """make_predictor for compare: BlobModel, tiled for crop runs."""
    def factory(weights, spec, device):
        model = BlobModel([] if seen is None else seen.setdefault(spec.train.dataset, []))
        tiling = spec.tiling()
        return TiledPredictor(model, tiling) if tiling else model

    return factory


class BlobModel:
    """Fake trained model: dark blobs are 'dent'. Records the image sizes it was given."""

    def __init__(self, seen):
        self.seen, self.task, self.weights = seen, "detect", None

    def predict(self, image8, conf=0.25, iou=0.7):
        self.seen.append(image8.shape[:2])
        gray = image8[..., 0]
        n, _, stats, _ = cv2.connectedComponentsWithStats((gray < 100).astype(np.uint8))
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            score = min(0.99, area / 400)
            if score >= conf:
                out.append(pred(x, y, x + w, y + h, score))
        return out


def make_project_with_runs(tmp_path):
    """Project with full + crop datasets and one finished (fake) run on each."""
    from test_crop import make_project

    p, _ = make_project(tmp_path)
    p.config.crop = CropConfig(use_roi=False, crop_size=128, per_defect=1, background_ratio=0.0, multiplier=0)
    p.config.augment.multiplier = 1
    build_full_dataset(p)
    build_crop_dataset(p)
    runs = []
    for dataset, imgsz in (("full", 400), ("crop", 128)):
        run = prepare_run(p, TrainConfig(model="yolo11n", imgsz=imgsz, epochs=3, dataset=dataset), p.root / "w")
        best, _ = weights_paths(run)
        best.parent.mkdir(parents=True)
        best.write_bytes(b"x" * 1000)
        with open(run / "events.jsonl", "w", encoding="utf-8") as f:
            for e, fit, t in ((1, 0.2, 10.0), (2, 0.5, 20.0), (3, 0.4, 30.0)):
                f.write(json.dumps({"type": "epoch", "epoch": e, "epochs": 3, "fitness": fit, "elapsed_s": t,
                                    "losses": {"train/box_loss": 1 / e},
                                    "metrics": {"metrics/mAP50-95(B)": fit}}) + "\n")
        write_status(run, {"state": "finished", "final_metrics": {"metrics/mAP50-95(B)": 0.5}})
        runs.append(run)
    return p, runs


@pytest.fixture
def project_with_runs(tmp_path):
    return make_project_with_runs(tmp_path)


def test_fair_val_images_same_for_both_datasets(project_with_runs):
    from defect_inspection.core.evaluate import val_images

    p, _ = project_with_runs
    paths = C.fair_val_images(p)
    # no background crops here, so the OK Val image has no crop at all: the fair set comes from the
    # project's split, not from what happens to be in a dataset folder
    assert paths and set(paths) == set(val_images(p, "full")) > set(val_images(p, "crop"))


def test_compare_runs_full_and_crop(project_with_runs):
    p, (full_run, crop_run) = project_with_runs
    seen: dict[str, list] = {"full": [], "crop": []}
    made = []

    def factory(weights, spec, device):
        model = BlobModel(seen[spec.train.dataset])
        tiling = spec.tiling()
        made.append((spec.train.dataset, device))
        return TiledPredictor(model, tiling) if tiling else model

    progress = []
    results = C.compare_runs(p, [full_run, crop_run], conf=0.25, iou=0.7, make_predictor=factory,
                             devices=[("CPU", "cpu")], progress=lambda d, t, m: progress.append((d, t)))
    full, crop = results
    assert {s for s in seen["full"]} == {(300, 400)}  # whole images
    assert {s for s in seen["crop"]} == {(128, 128)}  # tiles only
    assert [im.path for im in full.images] == [im.path for im in crop.images] == C.fair_val_images(p)
    for r in results:
        assert r.metrics.recall == pytest.approx(1.0) and r.metrics.map50 == pytest.approx(1.0)
        assert r.metrics.judge_accuracy == pytest.approx(1.0)
        assert set(r.speed_ms) == {"CPU"} and r.speed_ms["CPU"] >= 0
        assert r.training.best_epoch == 2 and r.training.train_seconds == 30.0
        assert r.training.weights_bytes == 1000 and r.training.onnx_bytes is None
        assert r.own_metrics == {"metrics/mAP50-95(B)": 0.5}
        assert r.series["epochs"] == [1, 2, 3]
    assert full.training.dataset == "full" and full.training.tiling is None
    assert crop.training.tiling == {"tile_size": 128, "overlap_px": 26}
    assert crop.training.dataset_train_images < full.training.dataset_train_images * 10
    assert ("crop", "cpu") in made and ("crop", None) in made
    assert progress[-1][0] == progress[-1][1]
    summary = crop.summary()
    assert "images" not in summary and json.dumps(summary)


def test_resumed_training_time_adds_up(project_with_runs):
    p, (full_run, _) = project_with_runs
    with open(full_run / "events.jsonl", "a", encoding="utf-8") as f:  # resumed from epoch 3: clock restarts
        f.write(json.dumps({"type": "epoch", "epoch": 4, "epochs": 4, "fitness": 0.9, "elapsed_s": 5.0}) + "\n")
    info = C.training_info(p, full_run, C.load_spec(full_run))
    assert info.train_seconds == 35.0 and info.best_epoch == 4


def test_onnx_size_found_from_exports(project_with_runs):
    p, (full_run, _) = project_with_runs
    spec = C.load_spec(full_run)
    p.exports_dir.mkdir(parents=True, exist_ok=True)
    (p.exports_dir / "m.onnx").write_bytes(b"o" * 2048)
    (p.exports_dir / "m.metadata.json").write_text(json.dumps({"run_id": spec.run_id}), encoding="utf-8")
    (p.exports_dir / "other.onnx").write_bytes(b"o")
    (p.exports_dir / "other.metadata.json").write_text(json.dumps({"run_id": "nope"}), encoding="utf-8")
    info = C.training_info(p, full_run, spec)
    assert info.onnx_bytes == 2048 and info.onnx_path.endswith("m.onnx")


def test_mismatched_run_refused(project_with_runs):
    p, (full_run, _) = project_with_runs
    p.add_class("scratch")
    with pytest.raises(ValueError, match="클래스"):
        C.compare_runs(p, [full_run], make_predictor=lambda w, s, d: BlobModel([]), devices=[])
