"""Full vs Crop (or any runs) comparison (PRD F8.1-F8.4). Qt-free.

Fair evaluation (F8.2): every model is re-evaluated on the SAME original, full-size Val images
(the project's per-image split, identical for the full and crop datasets) with the SAME conf and
NMS IoU. Full models predict the whole image; crop models use tiling inference (F7.6). Each run's
own Ultralytics val metrics are kept next to it, but they were measured on different images
(whole images vs crops) and are not comparable.

Metrics are box-based for both tasks (a seg prediction is compared by its bounding box):
- Precision / Recall / F1 at `conf`, IoU >= 0.5, same class (greedy by score, as in the Result tab).
- mAP50 and mAP50-95: COCO-style 101-point AP per class from predictions down to conf 0.001,
  averaged over classes that have ground truth.
- Recall by defect size (COCO areas: small < 32², medium < 96², large).
- Image level: an image is NG if it has a label; it is judged NG if any prediction >= conf.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.dataset import select_images, split_images
from defect_inspection.core.evaluate import MATCH_IOU, Prediction, box_iou, load_training_image, match  # noqa: F401 (match re-exported)
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import EpochSeries, RunSpec, TrainConfig, load_spec, read_events, read_status, weights_paths

Shape = BBox | Polygon
FLOOR_CONF = 0.001  # predictions kept for mAP / PR curves; P/R/F1 use the chosen conf
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
SIZE_BINS = (("small", 0.0, 32.0 ** 2), ("medium", 32.0 ** 2, 96.0 ** 2), ("large", 96.0 ** 2, float("inf")))
SPEED_IMAGES = 5  # images timed per device (after one warm-up)
PRIORITY_METRICS = {"mAP50-95": "map50_95", "mAP50": "map50", "Recall": "recall", "Precision": "precision",
                    "F1": "f1"}


# ---- data ------------------------------------------------------------------------------------

@dataclass
class ImagePredictions:
    path: str
    width: int
    height: int
    ground_truth: list[Shape]
    predictions: list[Prediction]  # down to FLOOR_CONF

    def above(self, conf: float) -> list[Prediction]:
        return [p for p in self.predictions if p.score >= conf]


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    tp: int
    fp: int
    fn: int
    per_class: dict[str, dict[str, float | int | None]]  # name -> precision/recall/ap50/ap50_95/gt
    size_recall: dict[str, float | None]  # small/medium/large -> recall (None: no labels of that size)
    size_counts: dict[str, int]
    images: int
    ng_images: int
    ok_images: int
    judge_accuracy: float  # OK/NG decision
    missed_ng_images: int  # NG judged OK (미검)
    false_ng_images: int  # OK judged NG (과검)
    fn_images: int  # images with at least one missed label
    fp_images: int  # images with at least one false detection
    pr_curve: tuple[list[float], list[float]]  # recall, precision at IoU 0.5, all classes pooled


@dataclass
class TrainingInfo:
    model: str
    dataset: str
    imgsz: int
    epochs_done: int
    best_epoch: int | None
    train_seconds: float | None
    dataset_train_images: int | None  # images in dataset/<mode>/images/train (incl. augmented)
    dataset_val_images: int | None
    dataset_labels: int | None  # label lines in the training split
    weights_bytes: int | None
    onnx_bytes: int | None  # newest export of this run, if it was exported
    onnx_path: str | None
    tiling: dict[str, Any] | None


@dataclass
class RunComparison:
    run_id: str
    label: str
    conf: float
    iou: float
    metrics: Metrics
    own_metrics: dict[str, float]  # the run's own val metrics (not comparable across datasets)
    speed_ms: dict[str, float]  # device label -> ms per image (tiling included)
    training: TrainingInfo
    series: dict[str, Any]  # {"epochs": [...], "values": {key: [...]}} for curve overlays
    images: list[ImagePredictions] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly without the per-image predictions (for reports / logs)."""
        d = asdict(self)
        d.pop("images")
        return d


# ---- Val set (F8.2) -----------------------------------------------------------------------------

def fair_val_images(project: Project) -> list[str]:
    """Original images in the Val split (defect + OK), the same set for full and crop models."""
    defect, ok, _ = select_images(project)
    split = split_images(project, defect, ok)
    return [e.path for e in defect + ok if split[e.path] == "val"]


def predict_images(project: Project, predictor, paths: list[str], iou: float,
                   progress: Callable[[int, int], None] | None = None,
                   cancel: Callable[[], bool] | None = None) -> list[ImagePredictions]:
    out = []
    for i, path in enumerate(paths, 1):
        if cancel is not None and cancel():
            break
        image = load_training_image(project, path)
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = project.load_labels(path, size=(w, h)).confirmed()
        out.append(ImagePredictions(path, w, h, gt, predictor.predict(image, FLOOR_CONF, iou)))
        if progress is not None:
            progress(i, len(paths))
    return out


# ---- metrics (F8.3) -------------------------------------------------------------------------------

def interpolated_ap(recall: list[float], precision: list[float]) -> float:
    """COCO 101-point AP from a PR curve ordered by descending score."""
    if not recall:
        return 0.0
    r = np.asarray(recall, float)
    p = np.maximum.accumulate(np.asarray(precision, float)[::-1])[::-1]  # precision envelope
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(r, points, side="left")
    return float(np.mean([p[i] if i < len(p) else 0.0 for i in idx]))


def pr_points(images: list[ImagePredictions], class_id: int | None, iou_thr: float) -> tuple[list[float], list[float], int]:
    """(recall, precision, number of labels) for one class (None = all classes, matched within
    the same class), predictions taken in descending score over all images."""
    def same(c: int) -> bool:
        return class_id is None or c == class_id

    n_gt = sum(1 for im in images for g in im.ground_truth if same(g.class_id))
    preds = [(p.score, k, p) for k, im in enumerate(images) for p in im.predictions if same(p.shape.class_id)]
    preds.sort(key=lambda t: -t[0])
    used = [[False] * len(im.ground_truth) for im in images]
    tp = fp = 0
    recall, precision = [], []
    for _, k, p in preds:
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(images[k].ground_truth):
            if used[k][gi] or g.class_id != p.shape.class_id:
                continue
            v = box_iou(g.bounds(), p.shape.bounds())
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used[k][best] = True
            tp += 1
        else:
            fp += 1
        recall.append(tp / n_gt if n_gt else 0.0)
        precision.append(tp / (tp + fp))
    return recall, precision, n_gt


def _area(shape: Shape) -> float:
    x1, y1, x2, y2 = shape.bounds()
    return (x2 - x1) * (y2 - y1)


def size_bin(shape: Shape) -> str:
    a = _area(shape)
    return next(name for name, lo, hi in SIZE_BINS if lo <= a < hi)


def compute_metrics(images: list[ImagePredictions], class_names: list[str], conf: float) -> Metrics:
    tp = fp = fn = 0
    per_class_counts = {c: [0, 0, 0] for c in range(len(class_names))}  # tp, fp, fn
    size_hit = {name: [0, 0] for name, _, _ in SIZE_BINS}  # found, total
    ng = ok = correct = missed_ng = false_ng = fn_images = fp_images = 0
    for im in images:
        preds = im.above(conf)
        matched, missed = match(im.ground_truth, preds, MATCH_IOU)
        tp += sum(matched)
        fp += len(matched) - sum(matched)
        fn += sum(missed)
        for p, m in zip(preds, matched):
            if p.shape.class_id in per_class_counts:
                per_class_counts[p.shape.class_id][0 if m else 1] += 1
        for g, miss in zip(im.ground_truth, missed):
            if g.class_id in per_class_counts:
                per_class_counts[g.class_id][2] += miss
            hit = size_hit[size_bin(g)]
            hit[0] += not miss
            hit[1] += 1
        is_ng, judged_ng = bool(im.ground_truth), bool(preds)
        ng += is_ng
        ok += not is_ng
        correct += is_ng == judged_ng
        missed_ng += is_ng and not judged_ng
        false_ng += judged_ng and not is_ng
        fn_images += any(missed)
        fp_images += not all(matched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_class, ap50s, ap_all = {}, [], []
    for c, name in enumerate(class_names):
        aps = []
        n_gt = 0
        for thr in IOU_THRESHOLDS:
            r, p, n_gt = pr_points(images, c, thr)
            aps.append(interpolated_ap(r, p))
        c_tp, c_fp, c_fn = per_class_counts[c]
        per_class[name] = {
            "gt": n_gt,
            "precision": c_tp / (c_tp + c_fp) if c_tp + c_fp else None,
            "recall": c_tp / (c_tp + c_fn) if c_tp + c_fn else None,
            "ap50": aps[0] if n_gt else None,
            "ap50_95": float(np.mean(aps)) if n_gt else None,
        }
        if n_gt:
            ap50s.append(aps[0])
            ap_all.append(float(np.mean(aps)))
    r, p, _ = pr_points(images, None, MATCH_IOU)
    return Metrics(
        precision=precision, recall=recall, f1=f1,
        map50=float(np.mean(ap50s)) if ap50s else 0.0,
        map50_95=float(np.mean(ap_all)) if ap_all else 0.0,
        tp=tp, fp=fp, fn=fn, per_class=per_class,
        size_recall={k: (v[0] / v[1] if v[1] else None) for k, v in size_hit.items()},
        size_counts={k: v[1] for k, v in size_hit.items()},
        images=len(images), ng_images=ng, ok_images=ok,
        judge_accuracy=correct / len(images) if images else 0.0,
        missed_ng_images=missed_ng, false_ng_images=false_ng, fn_images=fn_images, fp_images=fp_images,
        pr_curve=_thin(r, p),
    )


def _thin(recall: list[float], precision: list[float], n: int = 200) -> tuple[list[float], list[float]]:
    """At most n points of a PR curve (plots and reports do not need thousands)."""
    if len(recall) <= n:
        return list(recall), list(precision)
    idx = np.linspace(0, len(recall) - 1, n).round().astype(int)
    return [recall[i] for i in idx], [precision[i] for i in idx]


# ---- speed, training, model (F8.3) ------------------------------------------------------------------

def measure_speed(predictor, images: list[np.ndarray], conf: float, iou: float) -> float:
    """Median ms per image (tiling included) after one warm-up prediction."""
    if not images:
        return 0.0
    predictor.predict(images[0], conf, iou)
    times = []
    for image in images:
        t0 = time.perf_counter()
        predictor.predict(image, conf, iou)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _dataset_counts(project: Project, mode: str) -> tuple[int | None, int | None, int | None]:
    root = project.dataset_dir / mode
    if not (root / "data.yaml").is_file():
        return None, None, None
    train = [f for f in (root / "images" / "train").iterdir() if f.is_file()]
    val = [f for f in (root / "images" / "val").iterdir() if f.is_file()]
    labels = 0
    for f in (root / "labels" / "train").glob("*.txt"):
        labels += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    return len(train), len(val), labels


def newest_export(project: Project, run_id: str) -> Path | None:
    """The newest .onnx exported from this run (found through its metadata.json)."""
    import json

    found = []
    if project.exports_dir.is_dir():
        for meta in project.exports_dir.glob("*.metadata.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            onnx = meta.with_name(meta.name[: -len(".metadata.json")] + ".onnx")
            if data.get("run_id") == run_id and onnx.is_file():
                found.append(onnx)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def training_info(project: Project, run_dir: Path, spec: RunSpec) -> TrainingInfo:
    events = read_events(run_dir)
    epochs = [e for e in events if e.get("type") == "epoch"]
    best_epoch = None
    scored = [e for e in epochs if e.get("fitness") is not None]
    if scored:
        best_epoch = int(max(scored, key=lambda e: e["fitness"])["epoch"])
    # elapsed_s restarts after a resume: add up the last value of every segment
    seconds, last = 0.0, None
    for e in epochs:
        t = float(e.get("elapsed_s") or 0.0)
        if last is not None and t < last:
            seconds += last
        last = t
    train_seconds = seconds + last if last is not None else None
    n_train, n_val, n_labels = _dataset_counts(project, spec.train.dataset)
    best, _ = weights_paths(run_dir)
    onnx = newest_export(project, spec.run_id)
    tiling = spec.tiling()
    return TrainingInfo(
        model=spec.train.model, dataset=spec.train.dataset, imgsz=spec.train.imgsz,
        epochs_done=int(epochs[-1]["epoch"]) if epochs else 0, best_epoch=best_epoch,
        train_seconds=None if train_seconds is None else round(train_seconds, 1),
        dataset_train_images=n_train, dataset_val_images=n_val, dataset_labels=n_labels,
        weights_bytes=best.stat().st_size if best.is_file() else None,
        onnx_bytes=onnx.stat().st_size if onnx else None, onnx_path=str(onnx) if onnx else None,
        tiling=None if tiling is None else {"tile_size": tiling.tile_size, "overlap_px": tiling.overlap_px},
    )


def run_label(spec: RunSpec) -> str:
    return f"{spec.train.dataset.capitalize()} · {spec.train.model} · {spec.run_id}"


# ---- one run ------------------------------------------------------------------------------------

PredictorFactory = Callable[[Path, RunSpec, str | None], Any]  # (weights, spec, device) -> predictor


def default_factory(weights: Path, spec: RunSpec, device: str | None):
    from defect_inspection.core.evaluate import predictor_for_run

    return predictor_for_run(weights, spec, device)


def evaluate_run(project: Project, run_dir: Path, paths: list[str], conf: float = 0.25, iou: float = 0.7,
                 make_predictor: PredictorFactory = default_factory,
                 devices: list[tuple[str, str]] | None = None,
                 progress: Callable[[int, int], None] | None = None,
                 cancel: Callable[[], bool] | None = None) -> RunComparison:
    """Fair re-evaluation of one run on `paths` (original images), plus speed and training info."""
    from defect_inspection.platform_utils import inference_devices

    run_dir = Path(run_dir)
    spec = load_spec(run_dir)
    best, _ = weights_paths(run_dir)
    if not best.is_file():
        raise FileNotFoundError(f"best.pt가 없습니다: {run_dir.name}")
    if spec.nc != len(project.config.classes) or spec.channels != project.channels:
        raise ValueError(f"{spec.run_id}: 현재 프로젝트의 클래스/색 모드와 다르게 학습된 run입니다")
    images = predict_images(project, make_predictor(best, spec, None), paths, iou, progress, cancel)
    metrics = compute_metrics(images, [c.name for c in project.config.classes], conf)

    speed: dict[str, float] = {}
    sample = [img for img in (load_training_image(project, p) for p in paths[:SPEED_IMAGES]) if img is not None]
    for label, device in devices if devices is not None else inference_devices():
        if cancel is not None and cancel():
            break
        speed[label] = round(measure_speed(make_predictor(best, spec, device), sample, conf, iou), 2)

    status = read_status(run_dir)
    series = EpochSeries.from_events(read_events(run_dir))
    return RunComparison(
        run_id=spec.run_id, label=run_label(spec), conf=conf, iou=iou, metrics=metrics,
        own_metrics=status.get("final_metrics") or status.get("metrics") or {},
        speed_ms=speed, training=training_info(project, run_dir, spec),
        series={"epochs": series.epochs, "values": series.values}, images=images,
    )


def compare_runs(project: Project, run_dirs: list[Path], conf: float = 0.25, iou: float = 0.7,
                 make_predictor: PredictorFactory = default_factory,
                 devices: list[tuple[str, str]] | None = None,
                 progress: Callable[[int, int, str], None] | None = None,
                 cancel: Callable[[], bool] | None = None) -> list[RunComparison]:
    """Evaluate every run on the same Val images with the same thresholds (F8.2)."""
    paths = fair_val_images(project)
    if not paths:
        raise ValueError("검증(Val) 이미지가 없습니다. 라벨된 이미지가 2장 이상 필요합니다.")
    out = []
    for k, run_dir in enumerate(run_dirs):
        def step(done: int, total: int, k=k, name=Path(run_dir).name) -> None:
            if progress is not None:
                progress(k * total + done, len(run_dirs) * total, name)

        out.append(evaluate_run(project, run_dir, paths, conf, iou, make_predictor, devices, step, cancel))
        if cancel is not None and cancel():
            break
    return out


# ---- reading the results -------------------------------------------------------------------------

def differing_images(a: RunComparison, b: RunComparison) -> list[tuple[str, int]]:
    """(path, difference) of images where two models disagree most, largest difference first.
    Difference = |found_a - found_b| + |false_a - false_b| at each run's conf."""
    by_path = {im.path: im for im in b.images}
    out = []
    for im_a in a.images:
        im_b = by_path.get(im_a.path)
        if im_b is None:
            continue
        ma, _ = match(im_a.ground_truth, im_a.above(a.conf))
        mb, _ = match(im_b.ground_truth, im_b.above(b.conf))
        diff = abs(sum(ma) - sum(mb)) + abs((len(ma) - sum(ma)) - (len(mb) - sum(mb)))
        out.append((im_a.path, diff))
    return sorted(out, key=lambda t: -t[1])


def recommend(results: list[RunComparison], priority: str = "mAP50-95", tie: float = 0.01) -> tuple[int, str]:
    """(index of the recommended run, reason) — F8.5: the priority metric wins; within `tie`
    (1 %p) of the best, the faster model (CPU ms per image) is recommended."""
    if not results:
        raise ValueError("no results")
    attr = PRIORITY_METRICS[priority]
    values = [getattr(r.metrics, attr) for r in results]
    best = max(values)
    close = [i for i, v in enumerate(values) if best - v <= tie]

    def speed(i: int) -> float:
        s = results[i].speed_ms
        return s.get("CPU", min(s.values())) if s else float("inf")

    pick = min(close, key=speed) if len(close) > 1 else values.index(best)
    if len(close) > 1:
        reason = (f"{priority} 차이가 {tie * 100:.0f}%p 이내({', '.join(f'{values[i]:.3f}' for i in close)})라 "
                  f"추론이 더 빠른 쪽({speed(pick):.1f} ms/장)을 추천")
    else:
        reason = f"{priority}가 가장 높음 ({values[pick]:.3f})"
    return pick, reason


def experiment_configs(base: TrainConfig) -> tuple[TrainConfig, TrainConfig]:
    """F8.1: the same model, hyperparameters and seed for the full and the crop run."""
    return base.model_copy(update={"dataset": "full"}), base.model_copy(update={"dataset": "crop"})


BAR_LABELS = [("precision", "P"), ("recall", "R"), ("f1", "F1"), ("map50", "AP50"), ("map50_95", "AP50-95")]  # AP = mAP

# ---- metrics table (Compare tab and HTML report share it) ----------------------------------------------

@dataclass
class MetricRow:
    label: str
    texts: list[str]  # one per run
    values: list[float | None]  # for highlighting the best run
    better: str | None  # "high", "low" or None (not ranked)


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _int(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f} ms"


def seconds_text(v: float | None) -> str:
    if v is None:
        return "-"
    m, s = divmod(int(round(v)), 60)
    return f"{m}분 {s}초" if m else f"{s}초"


def own_map(r: RunComparison) -> float | None:
    m = r.own_metrics
    return m.get("metrics/mAP50-95(B)", m.get("metrics/mAP50-95(M)"))


def metric_rows(results: list[RunComparison]) -> list[MetricRow]:
    """Rows of the comparison table (F8.3), in display order."""
    rows: list[MetricRow] = []

    def metric(label: str, get: Callable[[RunComparison], float | None], better: str | None = "high",
               fmt: Callable[[float | None], str] = _num) -> None:
        vals = [get(r) for r in results]
        rows.append(MetricRow(label, [fmt(v) for v in vals], vals, better))

    def text(label: str, get: Callable[[RunComparison], str]) -> None:
        rows.append(MetricRow(label, [get(r) for r in results], [None] * len(results), None))

    metric("Precision", lambda r: r.metrics.precision)
    metric("Recall", lambda r: r.metrics.recall)
    metric("F1", lambda r: r.metrics.f1)
    metric("mAP50", lambda r: r.metrics.map50)
    metric("mAP50-95", lambda r: r.metrics.map50_95)
    for name in (list(results[0].metrics.per_class) if results else []):
        metric(f"  {name} mAP50-95", lambda r, n=name: r.metrics.per_class[n]["ap50_95"])
        metric(f"  {name} Recall", lambda r, n=name: r.metrics.per_class[n]["recall"])
    for key, label in (("small", "작은"), ("medium", "중간"), ("large", "큰")):
        count = results[0].metrics.size_counts[key] if results else 0
        metric(f"Recall · {label} 결함 ({count}개)", lambda r, k=key: r.metrics.size_recall[k])
    metric("OK/NG 판정 정확도", lambda r: r.metrics.judge_accuracy)
    metric("미검 이미지 (NG→OK)", lambda r: r.metrics.missed_ng_images, "low", _int)
    metric("과검 이미지 (OK→NG)", lambda r: r.metrics.false_ng_images, "low", _int)
    metric("미검 라벨 (FN)", lambda r: r.metrics.fn, "low", _int)
    metric("과검 검출 (FP)", lambda r: r.metrics.fp, "low", _int)
    devices = sorted({d for r in results for d in r.speed_ms}, key=lambda d: d != "CPU")
    for d in devices:
        metric(f"추론 시간 {d} (1장, 타일링 포함)", lambda r, d=d: r.speed_ms.get(d), "low", _ms)
    metric("학습 시간", lambda r: r.training.train_seconds, None, seconds_text)
    text("best / 진행 epoch", lambda r: f"{r.training.best_epoch or '-'} / {r.training.epochs_done}")
    text("데이터셋", lambda r: r.training.dataset)
    text("Train 이미지 / 라벨", lambda r: f"{r.training.dataset_train_images or '-'} / {r.training.dataset_labels or '-'}")
    text("모델 / 입력 크기", lambda r: f"{r.training.model} / {r.training.imgsz}")
    text("타일링", lambda r: "-" if not r.training.tiling else
         f"{r.training.tiling['tile_size']}px, overlap {r.training.tiling['overlap_px']}px")
    text("best.pt 크기", lambda r: file_size_text(r.training.weights_bytes))
    text("ONNX 크기", lambda r: file_size_text(r.training.onnx_bytes) if r.training.onnx_bytes else "내보내기 전")
    metric("참고: 자체 val mAP50-95 (데이터셋별, 직접 비교 불가)", own_map, None)
    return rows


def best_cells(row: MetricRow) -> set[int]:
    """Columns holding the best value of a ranked row (empty when all are equal)."""
    known = [v for v in row.values if v is not None]
    if not row.better or len(set(known)) < 2:
        return set()
    best = max(known) if row.better == "high" else min(known)
    return {i for i, v in enumerate(row.values) if v == best}


def file_size_text(n: int | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"

