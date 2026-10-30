"""Result tab and model suggestions (PROMPTS step 13, PRD F5, F2.8 ①)."""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PySide6.QtCore import QMimeData, QPointF, QSettings, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from defect_inspection.core import evaluate as E
from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.augment_config import AugmentConfig, TransformSetting
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import TrainConfig, prepare_run, weights_paths, write_status
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.main_window import MainWindow

S = 96


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


class BlobPredictor:
    """Stand-in for a trained model: every dark blob of >= 60 px is a 'dent' (class 0) with a score
    that grows with its size. Small blobs are missed on purpose."""

    def __init__(self, weights=None, task="detect", imgsz=640):
        self.task = task

    def predict(self, image8, conf=0.25, iou=0.7):
        gray = image8 if image8.ndim == 2 else image8[..., 0]
        n, _, stats, _ = cv2.connectedComponentsWithStats((gray < 100).astype(np.uint8))
        preds = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            score = min(0.99, area / 400)
            if area >= 60 and score >= conf:
                preds.append(E.Prediction(BBox(class_id=0, x1=x, y1=y, x2=x + w, y2=y + h,
                                               status="suggested", source="model", score=score)))
        return preds


def make_project(root, n_labeled=6, n_unlabeled=3):
    src = root / "원본"
    src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    truth = {}
    for i in range(n_labeled + n_unlabeled):
        img = np.full((S, S, 3), 200, np.uint8)
        cx, cy = int(rng.integers(25, S - 25)), int(rng.integers(25, S - 25))
        r = 10 if i % 3 else 3  # every third defect is tiny -> the blob predictor misses it
        cv2.circle(img, (cx, cy), r, (40, 40, 40), -1)
        name = f"{'lab' if i < n_labeled else 'new'}{i}.png"
        imwrite_unicode(src / name, img)
        truth[name] = (cx, cy, r)
    p = Project.create(root / "결과 프로젝트")
    p.add_class("dent")
    p.add_images([src])
    for name, (cx, cy, r) in truth.items():
        if name.startswith("lab"):
            ann = p.load_labels(name)
            ann.shapes.append(BBox(class_id=0, x1=cx - r, y1=cy - r, x2=cx + r + 1, y2=cy + r + 1))
            p.save_labels(ann)
    p.config.augment = AugmentConfig(multiplier=1, transforms={"hflip": TransformSetting(enabled=True)})
    p.config.split.val_ratio = 0.5
    p.save()
    build_full_dataset(p)
    return p


def fake_finished_run(project, model="yolo11n"):
    run = prepare_run(project, TrainConfig(model=model, epochs=1), project.root / "w")
    best, last = weights_paths(run)
    best.parent.mkdir(parents=True)
    best.write_bytes(b"fake")
    last.write_bytes(b"fake")
    (run / "train" / "confusion_matrix_normalized.png").write_bytes(
        cv2.imencode(".png", np.full((40, 40, 3), 255, np.uint8))[1].tobytes())
    write_status(run, {"state": "finished", "epoch": 1, "epochs": 1})
    return run


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("defect_inspection.ui.panels.result_panel.Predictor", BlobPredictor)
    p = make_project(tmp_path)
    fake_finished_run(p)
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.resize(1500, 1000)
    w.show()
    w.load_project(p)
    w.tabs.setCurrentWidget(w.result_panel)
    assert wait_until(qapp, lambda: w.session is not None)
    yield w
    w.train_panel.shutdown(5000)  # never reach the "학습 중" close dialog in teardown
    w.project_dirty = False
    w.close()


@pytest.fixture
def panel(window):
    return window.result_panel


# ---- core ------------------------------------------------------------------------------------

def P(x1, y1, x2, y2, score, cls=0):
    return E.Prediction(BBox(class_id=cls, x1=x1, y1=y1, x2=x2, y2=y2, status="suggested", source="model", score=score))


def test_box_iou():
    assert E.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert E.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert E.box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_match_greedy_by_score_and_class():
    gt = [BBox(class_id=0, x1=0, y1=0, x2=10, y2=10), BBox(class_id=1, x1=20, y1=20, x2=30, y2=30)]
    preds = [P(0, 0, 10, 10, 0.6), P(1, 1, 10, 10, 0.9), P(20, 20, 30, 30, 0.8, cls=0), P(50, 50, 60, 60, 0.7)]
    matched, missed = E.match(gt, preds)
    assert matched == [False, True, False, False]  # higher score wins the box; wrong class never matches
    assert missed == [False, True]


def test_image_result_kind():
    r = E.ImageResult("a.png", 10, 10, [], [], matched=[True, False], missed=[True])
    assert (r.tp, r.fp, r.fn, r.kind) == (1, 1, 1, "both")
    assert E.ImageResult("a", 1, 1, [], [], [True], [False]).kind == "ok"


def test_results_to_predictions_boxes_and_masks():
    boxes = SimpleNamespace(xyxy=torch.tensor([[1.0, 2.0, 11.0, 12.0], [5.0, 5.0, 5.0, 9.0]]),
                            cls=torch.tensor([1.0, 0.0]), conf=torch.tensor([0.9, 0.4]))
    boxes.__len__ = lambda: 2
    det = SimpleNamespace(boxes=_Sized(boxes, 2), masks=None)
    preds = E.results_to_predictions(det, "detect")
    assert len(preds) == 1  # zero-width box dropped
    assert preds[0].shape.bounds() == (1, 2, 11, 12) and preds[0].shape.class_id == 1
    assert (preds[0].shape.status, preds[0].shape.source, preds[0].score) == ("suggested", "model", pytest.approx(0.9))
    seg = SimpleNamespace(boxes=_Sized(boxes, 2),
                          masks=SimpleNamespace(xy=[np.array([[1, 2], [11, 2], [11, 12]], np.float32), np.zeros((0, 2))]))
    preds = E.results_to_predictions(seg, "seg")
    assert isinstance(preds[0].shape, Polygon) and len(preds[0].shape.points) == 3


class _Sized:
    def __init__(self, ns, n):
        self.__dict__.update(ns.__dict__)
        self._n = n

    def __len__(self):
        return self._n


def test_val_and_unlabeled_images(tmp_path):
    p = make_project(tmp_path)
    val = E.val_images(p)
    assert val and all(v.startswith("lab") for v in val)
    assert sorted(E.unlabeled_images(p)) == ["new6.png", "new7.png", "new8.png"]
    p.set_ok("new6.png")
    assert "new6.png" not in E.unlabeled_images(p)


def test_generate_suggestions_replaces_model_suggestions_only(tmp_path):
    p = make_project(tmp_path)
    ann = p.load_labels("new7.png")
    ann.add_suggestions([BBox(class_id=0, x1=1, y1=1, x2=5, y2=5, score=0.5)], source="sam")
    p.save_labels(ann)
    report = E.generate_suggestions(p, BlobPredictor(), ["new6.png", "new7.png", "new8.png"])
    assert report.images == 3 and report.with_suggestions == 2 and report.suggestions == 2  # new6 is tiny
    sources = sorted(s.source for s in p.load_labels("new7.png").suggestions())
    assert sources == ["model", "sam"]
    again = E.generate_suggestions(p, BlobPredictor(), ["new7.png"])
    assert again.suggestions == 1 and len(p.load_labels("new7.png").suggestions()) == 2  # replaced, not stacked
    assert p.label_txt_path("new7.png").read_bytes() == b""  # suggestions never reach YOLO txt


def test_quick_suggest_config():
    cfg = E.quick_suggest_config()
    assert cfg.model == "yolo11n" and cfg.epochs == 30


# ---- panel ---------------------------------------------------------------------------------

def test_runs_listed(panel):
    assert panel.run_combo.count() == 1 and "yolo11n" in panel.run_combo.itemText(0)
    assert panel.btn_eval.isEnabled()
    assert "3장" in panel.unlabeled_label.text()


def test_evaluate_fills_grid_with_problems(panel, qapp):
    panel.evaluate()
    assert wait_until(qapp, lambda: panel.results)
    assert panel.grid.count() == len(panel.results) > 0
    assert "precision" in panel.summary.text()
    first = panel.grid.item(0)
    assert "TP" in first.text() and not first.icon().isNull()
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("missed"))
    missed = [panel.grid.item(i).data(Qt.ItemDataRole.UserRole) for i in range(panel.grid.count())]
    expected = [r.path for r in panel.visible_results() if r.fn]
    assert missed == expected


def test_conf_slider_refilters_without_predicting(panel, qapp):
    panel.evaluate()
    assert wait_until(qapp, lambda: panel.results)
    calls = []
    panel._submit = lambda *a: calls.append(a)
    panel.conf.setValue(95)
    assert calls == []
    assert all(not r.predictions for r in panel.visible_results()) or all(
        p.score >= 0.95 for r in panel.visible_results() for p in r.predictions)
    assert "conf 0.95" in panel.summary.text()


def test_double_click_opens_image_for_relabeling(window, panel, qapp):
    panel.evaluate()
    assert wait_until(qapp, lambda: panel.results)
    item = panel.grid.item(panel.grid.count() - 1)
    path = item.data(Qt.ItemDataRole.UserRole)
    window.image_panel.set_filter("unlabeled")  # target hidden by the filter
    panel.grid.itemDoubleClicked.emit(item)
    assert window.image_panel.current_path() == path
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == path)


def test_plots_tab(panel):
    assert panel.plot_combo.count() == 1 and "정규화" in panel.plot_combo.itemText(0)
    assert panel.plot_view.pixmap() is not None and not panel.plot_view.pixmap().isNull()


def test_drop_image_runs_test_inference(panel, qapp, tmp_path):
    img = np.full((S, S, 3), 200, np.uint8)
    cv2.circle(img, (40, 40), 12, (30, 30, 30), -1)
    path = tmp_path / "드롭 테스트.png"
    imwrite_unicode(path, img)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    panel.dropEvent(ev)
    assert wait_until(qapp, lambda: "검출 1개" in panel.test_info.text())
    assert panel.tabs.currentIndex() == 2 and not panel.test_view.pixmap().isNull()


def test_suggest_marks_images_and_reloads_open_session(window, panel, qapp):
    window.image_panel.select("new7.png")
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == "new7.png")
    # unsaved SAM suggestion: the image stays unlabeled, so it still gets model suggestions
    window.session.add_shape(BBox(class_id=0, x1=0, y1=0, x2=4, y2=4, status="suggested", source="sam", score=0.5))
    assert panel.suggest()
    assert wait_until(qapp, lambda: "제안 완료" in panel.suggest_status.text())
    # the unsaved edit was saved first, and the open session was reloaded with the model suggestion
    sources = sorted(s.source for s in window.session.annotations.suggestions())
    assert sources == ["model", "sam"]
    texts = [window.image_list.item(i).text() for i in range(window.image_list.count())]
    assert any(t.startswith("new8.png") and "제안 1" in t for t in texts)


def test_labeled_images_never_get_model_suggestions(window, panel, qapp):
    panel.suggest()
    assert wait_until(qapp, lambda: "제안 완료" in panel.suggest_status.text())
    for e in window.project.config.images:
        if e.path.startswith("lab"):
            assert not window.project.load_labels(e).suggestions()


def test_suggest_then_accept_in_canvas(window, panel, qapp):
    panel.suggest()
    assert wait_until(qapp, lambda: "제안 완료" in panel.suggest_status.text())
    window.image_panel.select("new8.png")
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == "new8.png")
    sid = window.session.annotations.suggestions()[0].id
    window.canvas.select_ids([sid])
    window.act_enter.trigger()
    assert window.session.annotations.confirmed()[0].source == "model"


def test_quick_suggest_chain(window, panel, qapp, monkeypatch):
    """Build -> quick yolo11n run (fake worker) -> suggestions with the new run."""
    monkeypatch.setenv("DEFECTLAB_FAKE_TRAIN", "2:0.02")
    panel.btn_quick.click()
    assert window._quick_stage in ("build", "train")
    assert wait_until(qapp, lambda: "제안 완료" in panel.suggest_status.text(), timeout=60)
    assert window._quick_stage is None
    assert panel.run_combo.count() == 2 and window.tabs.currentWidget() is panel
    newest = panel.runs[0]
    assert newest.spec.train.model == "yolo11n" and newest.spec.train.epochs == 30


def test_quick_suggest_reports_failed_build(window, panel, qapp, monkeypatch):
    shown = []
    monkeypatch.setattr("defect_inspection.ui.panels.augment_panel.QMessageBox.warning", lambda *a: shown.append(a))
    for e in list(window.project.config.images):
        ann = window.project.load_labels(e)
        ann.shapes.clear()
        window.project.save_labels(ann)
    window.session.annotations.shapes.clear()
    panel.btn_quick.click()
    assert wait_until(qapp, lambda: window._quick_stage is None, timeout=30)
    assert "취소" in panel.suggest_status.text()


# ---- real model (slow) ----------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_evaluate_and_suggest(tmp_path):
    from ultralytics import YOLO

    from defect_inspection.platform_utils import cache_dir
    from defect_inspection.core.weights import resolve_weights

    p = make_project(tmp_path)
    p.config.augment.multiplier = 8
    p.save()
    build_full_dataset(p)
    model = YOLO(str(resolve_weights("yolo11n.pt", cache_dir() / "weights")))
    model.train(data=str(p.dataset_dir / "full" / "data.yaml"), epochs=15, imgsz=96, batch=8, workers=0,
                device="cpu", project=str(tmp_path / "r"), name="t", plots=False, verbose=False)
    predictor = E.Predictor(model.trainer.best, "detect", imgsz=96)
    results = E.evaluate_images(p, predictor, E.val_images(p), conf=0.05, iou=0.7)
    assert results and all(r.width == S for r in results)
    report = E.generate_suggestions(p, predictor, E.unlabeled_images(p), conf=0.05)
    assert report.images == 3
    for path in report.paths:
        for s in p.load_labels(path).suggestions():
            assert s.source == "model" and 0 <= s.score <= 1


# ---- crop runs: tiling inference (step 18, F7.6) --------------------------------------------------

def test_crop_run_uses_tiling(window, panel, qapp, monkeypatch, tmp_path):
    from defect_inspection.core.crop import TiledPredictor, build_crop_dataset
    from defect_inspection.core.crop_config import CropConfig

    p = window.project
    p.config.crop = CropConfig(use_roi=False, crop_size=64, per_defect=1, background_ratio=0.0, multiplier=0)
    build_crop_dataset(p)
    run = prepare_run(p, TrainConfig(model="yolo11n", imgsz=64, dataset="crop"), p.root / "w")
    best, _ = weights_paths(run)
    best.parent.mkdir(parents=True)
    best.write_bytes(b"fake")
    write_status(run, {"state": "finished", "epoch": 1, "epochs": 1})
    seen = []
    original = BlobPredictor.predict
    monkeypatch.setattr(BlobPredictor, "predict", lambda self, img, conf=0.25, iou=0.7: seen.append(img.shape)
                        or original(self, img, conf, iou))
    panel.refresh_runs(select=run)
    assert panel.current_run.path == run

    panel.evaluate()
    assert wait_until(qapp, lambda: panel.results)
    assert isinstance(panel._predictor(panel.current_run), TiledPredictor)
    assert seen and {s[:2] for s in seen} == {(64, 64)}  # the 96 px originals went in as 64 px tiles
    assert all("__" not in r.path and r.width == S for r in panel.results)  # evaluated on originals
    assert "타일링 추론 (타일 64px, overlap 13px)" in panel.summary.text()
    found = [p.shape.bounds() for r in panel.results for p in r.predictions]
    assert found and all(x2 <= S and y2 <= S for _, _, x2, y2 in found)

    img = np.full((S, S, 3), 200, np.uint8)
    cv2.circle(img, (70, 70), 10, (40, 40, 40), -1)  # inside the last (flush) tile 32..96
    imwrite_unicode(tmp_path / "t.png", img)
    panel.test_image(tmp_path / "t.png")
    assert wait_until(qapp, lambda: "검출" in panel.test_info.text())
    assert "타일링 추론" in panel.test_info.text()
    (box,) = [p.shape.bounds() for p in panel.last_test_predictions]
    assert box == pytest.approx((60, 60, 81, 81), abs=1)


def test_edge_filter_applies_to_val_grid_and_suggestions(window, panel, qapp):
    """F9.3: the blobs lie on no edge line, so with the filter on nothing is found or suggested."""
    from defect_inspection.core.edge_filter import EdgeFilteredPredictor
    from defect_inspection.core.edge_filter_config import EdgeFilterConfig

    window.project.config.edge_filter = EdgeFilterConfig(enabled=True)
    assert isinstance(panel._predictor(panel.current_run), EdgeFilteredPredictor)
    panel.evaluate()
    assert wait_until(qapp, lambda: panel.results)
    assert all(not r.predictions for r in panel.results)
    panel.suggest()
    assert wait_until(qapp, lambda: "제안 완료" in panel.suggest_status.text())
    assert not any(window.project.load_labels(e).suggestions() for e in window.project.config.images)
    window.project.config.edge_filter = EdgeFilterConfig(enabled=False)
    assert isinstance(panel._predictor(panel.current_run), BlobPredictor)


def test_results_show_the_filter_they_were_predicted_with(window, panel, qapp):
    from defect_inspection.core.edge_filter_config import EdgeFilterConfig

    window.project.config.edge_filter = EdgeFilterConfig(enabled=True, touch_px=7)
    panel.evaluate()
    window.project.config.edge_filter = EdgeFilterConfig(enabled=False)  # edited after the click
    assert wait_until(qapp, lambda: panel.results)
    assert panel.results_edge.touch_px == 7 and "테두리 필터" in panel.summary.text()
    panel.suggest()
    assert wait_until(qapp, lambda: "제안 완료" in panel.suggest_status.text())
    assert "테두리 필터" not in panel.suggest_status.text()
