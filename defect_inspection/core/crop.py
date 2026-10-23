"""Crop-based dataset: dataset/crop/ (PRD F7.2-F7.5). Qt-free.

Crop windows come from four sources (each optional, see CropConfig):
- roi: manual ROIs of crops.json (common ROIs clipped to each image + the image's own),
- defect: one or more windows around every confirmed label, shifted at random (jitter) so the
  defect is not always in the centre,
- background: random windows that touch no label (background_ratio x the defect windows; OK
  images get the same number a single label would),
- tile: sliding tiles over the whole image (overlap in % or px; the last tile is aligned to the
  image edge, so every pixel is covered).

Labels go original -> crop coordinates and are clipped to the crop; what is left must keep
`augment.min_visibility` of its area (default 40 %) and `augment.min_area_px`, the F3.6 rule.

Leakage: Train/Val is decided per ORIGINAL image first (same split as the full dataset), then
the image is cropped, so crops of one image never span both splits. Only Train crops are
augmented. Windows are reproducible: each image's random generator is seeded from the project
seed and the image path.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

import numpy as np
import yaml

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon, fit_shape, write_yolo_txt
from defect_inspection.core.crop_config import CropConfig
from defect_inspection.core.dataset import (
    BuildCancelled,
    DatasetBuildError,
    Progress,
    _out_suffix,
    _write_image,
    select_images,
    split_images,
    val_warnings,
    write_data_yaml,
)
from defect_inspection.core.defect_patches import Patch, extract_patches, paste_patches, transform_defects
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.project import Project
from defect_inspection.core.roi import RoiStore
from defect_inspection.platform_utils import atomic_write_text, imread_unicode

Shape = BBox | Polygon
WindowKind = Literal["roi", "defect", "background", "tile"]
KINDS: tuple[WindowKind, ...] = ("roi", "defect", "background", "tile")
BACKGROUND_TRIES = 50


@dataclass(frozen=True)
class CropWindow:
    """Integer pixel window [x1, x2) x [y1, y2) of an original image."""

    x1: int
    y1: int
    x2: int
    y2: int
    kind: WindowKind = "tile"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def key(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


# ---- coordinates (F7.3) ------------------------------------------------------------------

def shift_shape(shape: Shape, dx: float, dy: float) -> Shape:
    """Same shape (same id) moved by (dx, dy), not clipped."""
    if isinstance(shape, BBox):
        return shape.model_copy(update={"x1": shape.x1 + dx, "y1": shape.y1 + dy,
                                        "x2": shape.x2 + dx, "y2": shape.y2 + dy})
    return shape.model_copy(update={"points": [(x + dx, y + dy) for x, y in shape.points]})


def to_crop(shape: Shape, window: CropWindow) -> Shape:
    return shift_shape(shape, -window.x1, -window.y1)


def to_original(shape: Shape, window: CropWindow) -> Shape:
    return shift_shape(shape, window.x1, window.y1)


def crop_annotations(annotations: ImageAnnotations, window: CropWindow, min_visibility: float,
                     min_area_px: float, image_name: str | None = None) -> tuple[ImageAnnotations, int]:
    """Confirmed labels of the original image as seen in the crop: (crop labels, dropped count).
    Labels completely outside the window are not counted as dropped."""
    kept, dropped = [], 0
    for shape in annotations.confirmed():
        a1, b1, a2, b2 = shape.bounds()
        if a2 <= window.x1 or a1 >= window.x2 or b2 <= window.y1 or b1 >= window.y2:
            continue  # not in this crop at all
        moved = to_crop(shape, window)
        new = fit_shape(moved, moved.to_points(), window.width, window.height, min_visibility, min_area_px)
        if new is None:
            dropped += 1
        else:
            kept.append(new)
    out = ImageAnnotations(image=image_name or annotations.image, width=window.width,
                           height=window.height, shapes=kept)
    return out, dropped


def crop_pixels(image: np.ndarray, window: CropWindow) -> np.ndarray:
    return np.ascontiguousarray(image[window.y1:window.y2, window.x1:window.x2])


# ---- windows (F7.2) ------------------------------------------------------------------------

def _clamp_start(start: float, size: int, limit: int) -> int:
    return int(min(max(round(start), 0), limit - size))


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start positions along one axis: step tile - overlap, last tile flush with the edge."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def tile_windows(width: int, height: int, tile: int, overlap: int) -> list[CropWindow]:
    """Sliding tiles covering the whole image (an image smaller than a tile is one window)."""
    tw, th = min(tile, width), min(tile, height)
    return [CropWindow(x, y, x + tw, y + th, "tile")
            for y in tile_starts(height, tile, overlap) for x in tile_starts(width, tile, overlap)]


def defect_windows(annotations: ImageAnnotations, width: int, height: int, size: int, per_defect: int,
                   jitter: float, rng: np.random.Generator) -> list[CropWindow]:
    """`per_defect` windows per confirmed label; the label's centre is moved off the window centre
    by up to `jitter` of the free space, so it stays inside the window when it fits."""
    sw, sh = min(size, width), min(size, height)
    out = []
    for shape in annotations.confirmed():
        a1, b1, a2, b2 = shape.bounds()
        cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
        free_x, free_y = max(0.0, (sw - (a2 - a1)) / 2), max(0.0, (sh - (b2 - b1)) / 2)
        for _ in range(per_defect):
            dx = rng.uniform(-1, 1) * jitter * free_x
            dy = rng.uniform(-1, 1) * jitter * free_y
            x1 = _clamp_start(cx + dx - sw / 2, sw, width)
            y1 = _clamp_start(cy + dy - sh / 2, sh, height)
            out.append(CropWindow(x1, y1, x1 + sw, y1 + sh, "defect"))
    return out


def _touches(window: CropWindow, shapes: list[Shape]) -> bool:
    for s in shapes:
        a1, b1, a2, b2 = s.bounds()
        if a1 < window.x2 and window.x1 < a2 and b1 < window.y2 and window.y1 < b2:
            return True
    return False


def background_windows(annotations: ImageAnnotations, width: int, height: int, size: int, count: int,
                       rng: np.random.Generator, tries: int = BACKGROUND_TRIES) -> list[CropWindow]:
    """Up to `count` random windows that overlap no confirmed label (fewer if there is no room)."""
    sw, sh = min(size, width), min(size, height)
    shapes = annotations.confirmed()
    out: list[CropWindow] = []
    for _ in range(count):
        for _ in range(tries):
            x1 = int(rng.integers(0, width - sw + 1))
            y1 = int(rng.integers(0, height - sh + 1))
            w = CropWindow(x1, y1, x1 + sw, y1 + sh, "background")
            if not _touches(w, shapes) and w.key() not in {o.key() for o in out}:
                out.append(w)
                break
    return out


def roi_windows(store: RoiStore | None, image: str, width: int, height: int) -> list[CropWindow]:
    """Manual ROIs (F7.1) as integer windows: outward rounding, clamped to the image."""
    if store is None:
        return []
    out = []
    for r in store.effective(image, width, height):
        x1, y1 = max(0, int(np.floor(r.x1))), max(0, int(np.floor(r.y1)))
        x2, y2 = min(width, int(np.ceil(r.x2))), min(height, int(np.ceil(r.y2)))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append(CropWindow(x1, y1, x2, y2, "roi"))
    return out


def image_rng(seed: int, path: str) -> np.random.Generator:
    """Per-image generator: same windows for the same project seed and image, on every OS."""
    digest = int(hashlib.sha1(path.encode("utf-8")).hexdigest()[:12], 16)
    return np.random.default_rng([seed, digest])


def plan_windows(cfg: CropConfig, annotations: ImageAnnotations, width: int, height: int, is_ok: bool,
                 rois: list[CropWindow], rng: np.random.Generator) -> list[CropWindow]:
    """All crop windows of one image in a fixed order (roi, defect, background, tile), duplicates
    removed."""
    windows: list[CropWindow] = []
    if cfg.use_roi:
        windows += rois
    n_defect = 0
    if cfg.use_defect and not is_ok:
        found = defect_windows(annotations, width, height, cfg.crop_size, cfg.per_defect, cfg.jitter, rng)
        n_defect = len(found)
        windows += found
    if cfg.use_background and cfg.background_ratio > 0:
        base = n_defect if not is_ok else cfg.per_defect  # an OK image counts like one label
        count = round(base * cfg.background_ratio)
        windows += background_windows(annotations, width, height, cfg.crop_size, count, rng)
    if cfg.use_tiles:
        windows += tile_windows(width, height, cfg.tile_size, cfg.overlap_pixels())
    seen, unique = set(), []
    for w in windows:
        if w.key() not in seen:
            seen.add(w.key())
            unique.append(w)
    return unique


# ---- build (F7.4, F7.5) ----------------------------------------------------------------------

@dataclass
class CropBuildReport:
    out_dir: str
    train_images: int = 0  # original images in Train / Val
    val_images: int = 0
    train_crops: int = 0
    val_crops: int = 0
    val_defect_crops: int = 0  # Val crops that contain at least one label
    augmented: int = 0
    defect_transformed: int = 0  # augmented crops with moved / resized / rotated defects
    dropped_labels: int = 0  # F7.3 rule at the crop border (+ F3.6 during augmentation)
    by_kind: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})
    skipped_unlabeled: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)
    no_windows: list[str] = field(default_factory=list)  # images that produced no crop
    warnings: list[str] = field(default_factory=list)  # dataset.val_warnings
    elapsed_s: float = 0.0

    @property
    def total_train(self) -> int:
        return self.train_crops + self.augmented


def build_crop_dataset(project: Project, progress: Progress | None = None,
                       cancel: Callable[[], bool] | None = None) -> CropBuildReport:
    """Build dataset/crop/ from the project's images, labels, crops.json and `crop` settings."""
    from defect_inspection.core.augment import Augmenter  # Albumentations: import only when building

    started = time.monotonic()
    cfg, ccfg = project.config, project.config.crop
    aug_cfg = cfg.augment
    if not cfg.classes:
        raise DatasetBuildError("클래스가 없습니다. 클래스를 추가하고 라벨링하세요.")
    if not ccfg.any_source():
        raise DatasetBuildError("Crop 방식을 하나 이상 선택하세요 (ROI / 결함 중심 / 배경 / 슬라이딩 타일).")
    store = RoiStore(project.root)
    if ccfg.use_roi and not store.has_any() and not (ccfg.use_defect or ccfg.use_background or ccfg.use_tiles):
        raise DatasetBuildError("그려진 ROI가 없습니다. 캔버스에서 Crop ROI 도구(C)로 영역을 그리세요.")
    defect, ok, skipped = select_images(project)
    if not defect:
        raise DatasetBuildError("라벨된 이미지가 없습니다. 결함 이미지를 라벨링하세요.")
    split = split_images(project, defect, ok)  # per original image, before cropping (leakage rule)
    if not any(split[e.path] == "val" for e in defect):
        raise DatasetBuildError("검증(Val)용 결함 이미지가 없습니다. 라벨된 결함 이미지가 2장 이상 필요합니다.")

    final_dir = project.dataset_dir / "crop"
    work_dir = project.dataset_dir / "crop.building"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (work_dir / sub).mkdir(parents=True)

    report = CropBuildReport(out_dir=str(final_dir), skipped_unlabeled=skipped,
                             warnings=val_warnings(project, split, defect))
    channels = project.channels
    augmenter = Augmenter(aug_cfg, channels=channels, seed=cfg.split.seed) if ccfg.multiplier else None
    defect_paths = {e.path for e in defect}
    entries = defect + ok
    dt = aug_cfg.defect_transform
    use_dt = dt.enabled and ccfg.multiplier > 0
    steps = 100  # progress resolution per image

    def tick(index: int, fraction: float, message: str) -> None:
        if cancel is not None and cancel():
            raise BuildCancelled()
        if progress is not None:
            progress(index * steps + int(fraction * steps), len(entries) * steps, message)

    try:
        # Defect transform onto crops without labels: patches from Train defect images.
        patches: list[Patch] = []
        if use_dt and dt.on_ok_images:
            for entry in defect:
                if split[entry.path] != "train":
                    continue
                raw = imread_unicode(project.image_abspath(entry))
                if raw is not None:
                    image = to_training_8bit(raw, channels)
                    patches.extend(extract_patches(image, project.load_labels(entry, size=image.shape[1::-1])))
        for index, entry in enumerate(entries):
            tick(index, 0.0, f"읽는 중: {entry.path}")
            src = project.image_abspath(entry)
            raw = imread_unicode(src)
            if raw is None:
                report.skipped_unreadable.append(entry.path)
                continue
            image = to_training_8bit(raw, channels)
            h, w = image.shape[:2]
            ann = project.load_labels(entry, size=(w, h))
            if (ann.width, ann.height) != (w, h):
                report.skipped_unreadable.append(entry.path)  # image replaced after labeling
                continue
            part = split[entry.path]
            is_ok = entry.path not in defect_paths
            windows = plan_windows(ccfg, ann, w, h, is_ok, roi_windows(store, entry.path, w, h),
                                   image_rng(cfg.split.seed, entry.path))
            if part == "train":
                report.train_images += 1
            else:
                report.val_images += 1
            if not windows:
                report.no_windows.append(entry.path)
                continue

            stem, suffix = PurePosixPath(entry.path).stem, _out_suffix(src)
            img_dir, lbl_dir = work_dir / "images" / part, work_dir / "labels" / part
            counters = {k: 0 for k in KINDS}
            copies = ccfg.multiplier if part == "train" else 0
            dt_rng = image_rng(cfg.split.seed + 1, entry.path) if use_dt and copies else None
            for n, window in enumerate(windows):
                counters[window.kind] += 1
                name = f"{stem}__{window.kind}{counters[window.kind]:03d}"
                pixels = crop_pixels(image, window)
                labels, dropped = crop_annotations(ann, window, aug_cfg.min_visibility, aug_cfg.min_area_px,
                                                   image_name=name + suffix)
                _write_image(img_dir / f"{name}{suffix}", pixels)
                write_yolo_txt(lbl_dir / f"{name}.txt", labels, cfg.task)
                report.dropped_labels += dropped
                report.by_kind[window.kind] += 1
                if part == "train":
                    report.train_crops += 1
                else:
                    report.val_crops += 1
                    report.val_defect_crops += bool(labels.shapes)
                for k in range(1, copies + 1):
                    aug_name = f"{name}_aug{k:02d}"
                    src_pixels, src_labels = pixels, labels
                    if dt_rng is not None and dt_rng.random() < dt.p:
                        if labels.shapes:
                            src_pixels, src_labels, placed = transform_defects(pixels, labels, dt, dt_rng)
                            report.defect_transformed += placed > 0
                        elif patches and dt.on_ok_images:
                            src_pixels, src_labels = paste_patches(pixels, labels, patches, dt_rng,
                                                                   dt.max_defects, transform=dt)
                            report.defect_transformed += len(src_labels.shapes) > 0
                    result = augmenter(src_pixels, src_labels, image_name=aug_name + suffix)
                    _write_image(img_dir / f"{aug_name}{suffix}", result.image)
                    write_yolo_txt(lbl_dir / f"{aug_name}.txt", result.annotations, cfg.task)
                    report.augmented += 1
                    report.dropped_labels += result.dropped
                tick(index, (n + 1) / len(windows), f"{part}: {entry.path} ({n + 1}/{len(windows)})")

        if report.train_crops == 0:
            raise DatasetBuildError("Train crop이 하나도 만들어지지 않았습니다. Crop 설정을 확인하세요.")
        if report.val_defect_crops == 0:
            raise DatasetBuildError("결함이 들어 있는 검증(Val) crop이 없습니다. "
                                    "결함 중심 crop을 켜거나 ROI가 결함을 포함하게 그리세요.")
        write_data_yaml(work_dir / "data.yaml", final_dir, [c.name for c in cfg.classes], channels)
        report.elapsed_s = round(time.monotonic() - started, 2)
        info = {"mode": "crop", "report": asdict(report), "task": cfg.task, "channels": channels,
                "seed": cfg.split.seed, "val_ratio": cfg.split.val_ratio,
                "crop": ccfg.model_dump(mode="json"), "augment": aug_cfg.model_dump(mode="json"),
                # read by tiling inference / ONNX metadata (F7.6, F6): the window size the model saw
                "window": {"size": ccfg.tile_size if ccfg.use_tiles else ccfg.crop_size,
                           "overlap_px": ccfg.overlap_pixels() if ccfg.use_tiles else None}}
        atomic_write_text(work_dir / "build_info.yaml", yaml.safe_dump(info, allow_unicode=True, sort_keys=False))
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    if final_dir.exists():
        shutil.rmtree(final_dir)
    work_dir.rename(final_dir)
    return report


def read_build_info(dataset_dir: Path) -> dict:
    """build_info.yaml of a built dataset ({} if missing)."""
    path = Path(dataset_dir) / "build_info.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---- tiling inference (F7.6) ------------------------------------------------------------------
# A crop model has only seen windows of `tile_size`; on a whole image it is run tile by tile
# (the same sliding tiles as the dataset, last tile flush with the edge), every prediction is
# moved back to original coordinates, and duplicates from overlapping tiles are merged.

def box_ios(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over the smaller box: 1.0 when one box lies inside the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def merge_predictions(predictions: list, iou: float = 0.5, ios: float = 0.7) -> list:
    """Cross-tile NMS: greedy by score, per class. A prediction is dropped when a kept one of the
    same class overlaps it with IoU >= `iou`, or covers most of it (IoS >= `ios`: the piece of a
    defect that a tile border cut off, while the neighbouring tile saw it whole)."""
    from defect_inspection.core.evaluate import box_iou

    kept: list = []
    for p in sorted(predictions, key=lambda p: -p.score):
        b = p.shape.bounds()
        if any(k.shape.class_id == p.shape.class_id
               and (box_iou(b, k.shape.bounds()) >= iou or box_ios(b, k.shape.bounds()) >= ios)
               for k in kept):
            continue
        kept.append(p)
    return kept


def tiled_predict(predict: Callable[[np.ndarray], list], image: np.ndarray, params) -> list:
    """Run `predict(tile) -> [Prediction]` on every tile and return merged predictions in
    original image coordinates. `params` is a TilingParams."""
    from defect_inspection.core.evaluate import Prediction

    h, w = image.shape[:2]
    found = []
    for window in tile_windows(w, h, params.tile_size, params.overlap_px):
        for p in predict(crop_pixels(image, window)):
            found.append(Prediction(to_original(p.shape, window)))
    return merge_predictions(found, params.merge_iou, params.merge_ios)


class TiledPredictor:
    """Same interface as evaluate.Predictor (predict(image8, conf, iou)), for crop models: whole
    images are tiled, predicted per tile and merged. Works for .pt and .onnx models alike."""

    def __init__(self, predictor, params) -> None:
        self.predictor, self.params = predictor, params
        self.task = predictor.task
        self.weights = getattr(predictor, "weights", None)

    def predict(self, image8: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> list:
        return tiled_predict(lambda tile: self.predictor.predict(tile, conf, iou), image8, self.params)

    def describe(self) -> str:
        return f"타일링 추론: 타일 {self.params.tile_size}px, overlap {self.params.overlap_px}px"
