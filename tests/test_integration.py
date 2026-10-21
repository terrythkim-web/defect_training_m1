"""End-to-end (PROMPTS step 20, PRD 10장): 5 synthetic defect images -> label -> full and crop
datasets -> real YOLO11 training (epochs=3 each, through the real worker process) -> ONNX export
and onnxruntime inference -> fair comparison -> HTML report whose numbers match the comparison.

Slow (minutes on CPU): .venv/bin/python -m pytest -m slow tests/test_integration.py
The project path contains a space and Korean characters on purpose (PRD 5.1).
"""

import json
import re
import subprocess
import sys

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from defect_inspection.core import compare as C
from defect_inspection.core import report as R
from defect_inspection.core.annotation import BBox
from defect_inspection.core.augment_config import AugmentConfig, TransformSetting
from defect_inspection.core.crop import build_crop_dataset
from defect_inspection.core.crop_config import CropConfig
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.evaluate import load_training_image, predictor_for_onnx
from defect_inspection.core.exporter import ExportOptions, export_run
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import TrainConfig, list_runs, prepare_run, read_status
from defect_inspection.platform_utils import cache_dir, imwrite_unicode

SIZE = 320


def synthetic_project(root):
    """5 defect images (dark disks and scratches on a textured background) + 1 OK image."""
    src = root / "합성 원본"
    src.mkdir(parents=True)
    rng = np.random.default_rng(7)
    shapes = {}
    for i in range(5):
        img = rng.normal(170, 8, (SIZE, SIZE, 3)).clip(0, 255).astype(np.uint8)
        cx, cy = (int(v) for v in rng.integers(50, SIZE - 50, 2))
        r = int(rng.integers(10, 18))
        cv2.circle(img, (cx, cy), r, (45, 45, 45), -1)
        imwrite_unicode(src / f"결함 {i}.png", img)
        shapes[f"결함 {i}.png"] = (cx - r, cy - r, cx + r + 1, cy + r + 1)
    imwrite_unicode(src / "정상.png", rng.normal(170, 8, (SIZE, SIZE, 3)).clip(0, 255).astype(np.uint8))
    p = Project.create(root / "통합 테스트 프로젝트")
    p.add_class("dent")
    p.add_images([src])
    for name, (x1, y1, x2, y2) in shapes.items():
        ann = p.load_labels(name)
        ann.shapes.append(BBox(class_id=0, x1=x1, y1=y1, x2=x2, y2=y2))
        p.save_labels(ann)
    p.set_ok("정상.png")
    p.config.split.val_ratio = 0.4  # 2 of the 5 defect images in Val
    p.config.augment = AugmentConfig(multiplier=4, transforms={"hflip": TransformSetting(enabled=True),
                                                               "vflip": TransformSetting(enabled=True)})
    p.config.crop = CropConfig(use_roi=False, crop_size=160, per_defect=2, background_ratio=1.0, multiplier=3)
    p.save()
    return p


def train(project, cfg):
    run = prepare_run(project, cfg, cache_dir() / "weights")
    log = run / "worker.log"
    # stdin stays open like with the app's QProcess: the worker stops when its stdin closes
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "defect_inspection.worker", "--run-dir", str(run)],
                                stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=1800)
        finally:
            proc.stdin.close()
    status = read_status(run)
    assert code == 0 and status["state"] == "finished", log.read_text(encoding="utf-8", errors="replace")[-4000:]
    return run


@pytest.mark.slow
def test_full_and_crop_to_onnx_and_report(tmp_path):
    p = synthetic_project(tmp_path)
    full_report = build_full_dataset(p)
    crop_report = build_crop_dataset(p)
    assert full_report.val_images >= 2 and crop_report.val_defect_crops >= 1

    base = TrainConfig(model="yolo11n", epochs=3, batch=4, workers=0, device="cpu", seed=0)
    full_cfg, crop_cfg = C.experiment_configs(base)
    runs = [train(p, full_cfg.model_copy(update={"imgsz": SIZE})), train(p, crop_cfg.model_copy(update={"imgsz": 160}))]
    assert [r.spec.train.dataset for r in sorted(list_runs(p), key=lambda r: r.run_id)] == ["full", "crop"]

    # ONNX export + verification against PyTorch, then plain onnxruntime inference
    exports = []
    for run in runs:
        out = p.exports_dir / f"{run.name}.onnx"
        result = export_run(p, run, out, ExportOptions())
        assert result.verification.passed, result.verification
        meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(result.onnx_path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        size = meta["input"]["shape"][2]
        outputs = session.run(None, {inp.name: np.zeros((1, 3, size, size), np.float32)})
        assert outputs and outputs[0].shape[0] == 1
        exports.append((result, meta))
    assert exports[0][1]["crop"] is None
    assert exports[1][1]["crop"]["tiling"]["tile_size"] == 160

    # the exported crop model tiles a whole image just like the .pt (F7.6)
    onnx_crop = predictor_for_onnx(exports[1][0].onnx_path)
    assert onnx_crop.params.tile_size == 160
    onnx_crop.predict(load_training_image(p, "결함 0.png"), 0.25, 0.7)

    # fair comparison on the same original Val images (crop model tiled), then the HTML report
    results = C.compare_runs(p, runs, conf=0.25, iou=0.7, devices=[("CPU", "cpu")])
    assert [r.training.dataset for r in results] == ["full", "crop"]
    for r in results:
        assert r.metrics.images == len(C.fair_val_images(p)) and r.metrics.ng_images >= 2
        assert 0.0 <= r.metrics.map50_95 <= r.metrics.map50 <= 1.0
        assert r.training.epochs_done == 3 and r.training.onnx_bytes > 100_000
        assert r.speed_ms["CPU"] > 0
    path = R.build_report(p, results, runs, memo="통합 테스트")
    html = path.read_text(encoding="utf-8")
    assert re.fullmatch(r"\d{8}_\d{4}_full_vs_crop\.html", path.name) and path.parent == p.reports_dir
    for row in C.metric_rows(results):
        if row.label.startswith(("Precision", "Recall", "F1", "mAP50", "OK/NG", "미검", "과검")):
            found = re.search(r"<tr><th>" + re.escape(row.label.replace(">", "&gt;")) + r"</th>(.*?)</tr>", html)
            assert found, row.label
            assert [c.replace("&gt;", ">") for c in re.findall(r"<td[^>]*>(.*?)</td>", found.group(1))] == row.texts
    data = R.read_report_data(path)
    assert [x["run_id"] for x in data["runs"]] == [r.run_id for r in results]
    assert data["recommended"] == C.recommend(results)[0]
