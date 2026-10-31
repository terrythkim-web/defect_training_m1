"""U-Net++ semantic segmentation (core/unet.py): pre-processing, metrics, predictions and a tiny real
training run on the CPU (resnet18 from scratch, 64 px)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("segmentation_models_pytorch")

from defect_inspection.core import unet  # noqa: E402
from defect_inspection.core.annotation import Mask  # noqa: E402
from defect_inspection.core.trainer import RunSpec, TrainConfig, read_status, weights_paths  # noqa: E402
from defect_inspection.platform_utils import imwrite_unicode  # noqa: E402


def test_input_size_is_a_multiple_of_32():
    assert unet.input_size(500) == 512 and unet.input_size(64) == 64 and unet.input_size(10) == 32


def test_prepare_image_letterboxes_and_normalizes():
    img = np.full((50, 100), 255, np.uint8)
    x, box = unet.prepare_image(img, 1, 64)
    assert x.shape == (1, 64, 64) and (box.w, box.h) == (64, 32)
    assert x[0, 0, 0] == pytest.approx((1 - unet.MEAN_GRAY) / unet.STD_GRAY)
    assert (x[0, 40:] == 0).all()  # padding
    bgr = np.zeros((10, 10, 3), np.uint8)
    bgr[..., 2] = 255  # red in OpenCV order
    xc, _ = unet.prepare_image(bgr, 3, 32)
    assert xc[0, 0, 0] > xc[2, 0, 0]  # channel 0 is R after the BGR → RGB swap
    mask = np.ones((50, 100), np.uint8)
    y = unet.prepare_mask(mask, box)
    assert (y[:32] == 1).all() and (y[32:] == unet.IGNORE).all()


def test_seg_meter():
    meter = unet.SegMeter(nc=2)
    target = np.zeros((10, 10), np.uint8)
    target[0:4, 0:4] = 1  # 16 px of class 1
    target[6:8, 6:8] = 2  # a second region (class 2), missed
    target[9, :] = unet.IGNORE
    pred = np.zeros((10, 10), np.int64)
    pred[0:4, 0:2] = 1  # half of the class-1 region
    pred[9, :] = 2  # on ignored pixels: not counted
    meter.update(pred, target)
    m = meter.metrics()
    assert m["metrics/IoU"] == pytest.approx((8 / 16 + 0) / 2)
    assert m["metrics/recall(blob)"] == pytest.approx(1 / 2)  # half of region 1 is found (≥ 0.5)


def test_probabilities_to_predictions():
    prob = np.zeros((3, 20, 30), np.float32)
    prob[0] = 1.0
    prob[:, 2:6, 2:8] = [[[0.1]], [[0.8]], [[0.1]]]  # class 0 region, score 0.8
    prob[:, 10:14, 20:24] = [[[0.3]], [[0.0]], [[0.7]]]  # class 1 region, score 0.7
    prob[:, 18:19, 0:2] = [[[0.0]], [[1.0]], [[0.0]]]  # 2 px: noise
    preds = unet.probabilities_to_predictions(prob, conf=0.5)
    assert [(p.shape.class_id, round(p.score, 2)) for p in preds] == [(0, 0.8), (1, 0.7)]
    assert isinstance(preds[0].shape, Mask) and preds[0].shape.bounds() == (2, 2, 8, 6)
    assert preds[0].shape.status == "suggested" and preds[0].shape.source == "model"
    assert len(unet.probabilities_to_predictions(prob, conf=0.75)) == 1


# ---- a tiny real training run ------------------------------------------------------------------

def tiny_dataset(root, n_train=4, n_val=2, size=64):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for i in range(n):
            img = np.full((size, size), 90, np.uint8)
            mask = np.zeros((size, size), np.uint8)
            y = int(rng.integers(8, size - 8))
            img[y - 2:y + 2, 8:size - 8] = 230
            mask[y - 2:y + 2, 8:size - 8] = 1
            imwrite_unicode(root / "images" / split / f"im{i}.png", img)
            imwrite_unicode(root / "masks" / split / f"im{i}.png", mask)
    (root / "data.yaml").write_text("nc: 1\nchannels: 1\n", encoding="utf-8")


def tiny_spec(tmp_path, epochs=2, patience=50, workers=0):
    ds = tmp_path / "dataset" / "full"
    tiny_dataset(ds)
    cfg = TrainConfig(model="unetpp-r18", dataset="full", epochs=epochs, imgsz=64, batch=2, device="cpu",
                      workers=workers, pretrained=False, patience=patience, online_augment=False)
    return RunSpec(run_id="r", project_root=str(tmp_path), task="seg", nc=1, names=["crack"], channels=1,
                   data_yaml=str(ds / "data.yaml"), weights_dir=str(tmp_path / "w"), train=cfg, created="")


class Stop(Exception):
    pass


def test_training_writes_events_checkpoints_and_predicts(tmp_path):
    spec = tiny_spec(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = []
    unet.train_unet(spec, run_dir, events.append, lambda: False, False, Stop)
    kinds = [e["type"] for e in events]
    assert kinds == ["model", "train_epoch", "epoch", "train_epoch", "epoch", "final"]
    epoch = events[2]
    assert {"metrics/IoU", "metrics/Dice", "metrics/recall(blob)", "val/seg_loss"} <= set(epoch["metrics"])
    assert "train/seg_loss" in events[1]["losses"]
    best, last = weights_paths(run_dir)
    assert best.is_file() and last.is_file() and unet.is_unet_checkpoint(best)
    assert read_status(run_dir)["epoch"] == 2

    predictor = unet.UnetPredictor(best, "cpu")
    preds = predictor.predict(np.full((80, 120), 90, np.uint8), conf=0.0)
    assert all(isinstance(p.shape, Mask) for p in preds)
    assert predictor.probabilities(np.zeros((80, 120), np.uint8)).shape == (2, 43, 64)  # network resolution


def test_stop_then_resume(tmp_path):
    spec = tiny_spec(tmp_path, epochs=3)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events, stop = [], threading.Event()

    def emit(e):
        events.append(e)
        if e["type"] == "epoch":
            stop.set()  # stop after the first epoch

    with pytest.raises(Stop):
        unet.train_unet(spec, run_dir, emit, stop.is_set, False, Stop)
    assert weights_paths(run_dir)[1].is_file()
    events.clear()
    unet.train_unet(spec, run_dir, events.append, lambda: False, True, Stop)
    assert [e["epoch"] for e in events if e["type"] == "epoch"] == [2, 3]


def test_missing_masks_are_explained(tmp_path):
    spec = tiny_spec(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "dataset" / "full" / "masks")
    with pytest.raises(FileNotFoundError, match="다시 빌드"):
        unet.train_unet(spec, tmp_path, lambda e: None, lambda: False, False, Stop)


def test_worker_process_trains_a_unet_run(tmp_path):
    """The real worker entry point picks the U-Net++ loop from the model key. With DataLoader workers:
    forked loader workers hung on the stdin lock held by the worker's stop-watcher thread."""
    import subprocess
    import sys

    import yaml

    spec = tiny_spec(tmp_path, epochs=1, workers=2)
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "defect_inspection.worker", "--run-dir", str(run_dir)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout.read()  # stdin stays open: a closed stdin means "app gone, stop"
    proc.wait(timeout=300)
    proc.stdin.close()
    assert proc.returncode == 0, out
    assert read_status(run_dir)["state"] == "finished"
    assert '"type": "final"' in out


# ---- step 4: Result tab / suggestions -----------------------------------------------------------

class PaintedLine:
    """Fake U-Net++ output: every bright pixel is class 0 with probability 0.9."""

    task = "semseg"

    def predict(self, image8, conf=0.25, iou=0.7):
        prob = np.zeros((2,) + image8.shape[:2], np.float32)
        prob[1] = np.where(image8 > 200, 0.9, 0.0)
        prob[0] = 1 - prob[1]
        return unet.probabilities_to_predictions(prob, conf)


def mask_project(tmp_path):
    from defect_inspection.core.annotation import ImageAnnotations
    from defect_inspection.core.project import Project

    src = tmp_path / "src"
    src.mkdir()
    for name in ("a.png", "b.png"):
        img = np.full((60, 80), 90, np.uint8)
        img[20:24, 10:70] = 230
        imwrite_unicode(src / name, img)
    p = Project.create(tmp_path / "p", color_mode="grayscale", task="seg")
    p.add_class("crack")
    a, b = p.add_images([src / "a.png", src / "b.png"])
    line = np.zeros((60, 80), bool)
    line[20:24, 10:70] = True
    p.save_labels(ImageAnnotations(image=a.path, width=80, height=60, shapes=[Mask.from_pixels(line, class_id=0)]))
    return p, a.path, b.path


def test_mask_predictions_match_mask_labels_by_pixels(tmp_path):
    from defect_inspection.core.evaluate import evaluate_images

    p, a, _ = mask_project(tmp_path)
    (r,) = evaluate_images(p, PaintedLine(), [a], conf=0.25, iou=0.7)
    assert r.matched == [True] and r.missed == [False]


def test_suggestions_are_masks(tmp_path):
    from defect_inspection.core.evaluate import generate_suggestions

    p, _, b = mask_project(tmp_path)
    generate_suggestions(p, PaintedLine(), [b])
    (s,) = p.load_labels(b).suggestions()
    assert isinstance(s, Mask) and s.source == "model" and s.score == pytest.approx(0.9)


def test_predictor_for_run_picks_unet(tmp_path):
    from defect_inspection.core.evaluate import predictor_for_run

    spec = tiny_spec(tmp_path, epochs=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    unet.train_unet(spec, run_dir, lambda e: None, lambda: False, False, Stop)
    assert isinstance(predictor_for_run(weights_paths(run_dir)[0], spec, "cpu"), unet.UnetPredictor)


# ---- step 5: ONNX -------------------------------------------------------------------------------

def test_onnx_export_matches_pytorch(tmp_path):
    spec = tiny_spec(tmp_path, epochs=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    unet.train_unet(spec, run_dir, lambda e: None, lambda: False, False, Stop)
    best = weights_paths(run_dir)[0]
    out = tmp_path / "exp" / "m.onnx"
    meta = unet.export_onnx(best, out)
    assert out.is_file() and meta["imgsz"] == 64 and meta["channels"] == 1
    images = [np.full((50, 70), 90, np.uint8), np.random.default_rng(0).integers(0, 255, (64, 64), np.uint8)]
    v = unet.verify_onnx(out, best, images)
    assert v.passed and v.images == 2 and v.input_shape == [1, 1, 64, 64] and v.output_shapes == [[1, 2, 64, 64]]
    md = unet.onnx_metadata(meta)
    assert md["input"]["shape"] == [1, 1, 64, 64] and md["preprocess"]["normalize"]["mean"] == [unet.MEAN_GRAY]


def test_unet_export_refuses_coreml(tmp_path):
    from defect_inspection.core.exporter import ExportOptions, _export_unet

    with pytest.raises(ValueError, match="ONNX"):
        _export_unet(None, None, tmp_path, tmp_path / "best.pt", tmp_path / "m.mlpackage",
                     ExportOptions(format="coreml"), False, 0.0)


def test_dataset_pickles_for_spawned_workers(tmp_path):
    """Review fix: Windows/macOS DataLoader workers (spawn) pickle the dataset."""
    import torch

    tiny_dataset(tmp_path / "ds")
    pairs = unet.dataset_pairs(tmp_path / "ds", "train")
    loader = torch.utils.data.DataLoader(unet.SegDataset(pairs, 1, 64, True), batch_size=2, num_workers=1,
                                         multiprocessing_context="spawn")
    x, y = next(iter(loader))
    assert x.shape == (2, 1, 64, 64) and y.dtype == torch.int64


def test_low_resolution_scores_map_onto_the_image():
    """Prediction finds regions in the full-size label map and scores them from network-size probabilities."""
    prob = np.zeros((2, 10, 20), np.float32)
    prob[0] = 1.0
    prob[:, 2:4, 5:10] = [[[0.2]], [[0.8]]]
    label = np.zeros((40, 80), np.uint8)
    label[8:16, 20:40] = 1
    (p,) = unet.label_map_to_predictions(label, prob, conf=0.5)
    assert p.score == pytest.approx(0.8) and p.shape.bounds() == (20, 8, 40, 16)
