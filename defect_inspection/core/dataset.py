"""Build the full-image YOLO dataset: dataset/full/ (PRD F1.5, F3.2, F3.3 Copy-Paste, F7.5).

Rules
- Only images that are ready for training go in: images with confirmed labels, and OK (normal)
  images as background with an empty label file. Unlabeled images and images with nothing but
  unaccepted suggestions are skipped: they may contain unmarked defects, and training on them
  would teach the model that those defects are background.
- Leakage: Train/Val is decided per original image first (defect and OK images split
  separately, so Val always has real defects); only Train is augmented; Copy-Paste patches come
  from Train images only.
- Pixels: uint8 with the project's channel count; 16-bit is scaled by >> 8 (to_training_8bit).
- The dataset is written to a temporary folder and swapped in at the end, so a cancelled or
  failed build never leaves a half-written dataset/full.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2
import numpy as np
import yaml

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon, write_yolo_txt
from defect_inspection.core.augment_config import CopyPasteSetting, DefectTransformSetting
from defect_inspection.core.defect_patches import (  # noqa: F401 - extract/paste re-exported
    Patch,
    extract_patches,
    paste_patches,
    transform_defects,
)
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.project import ImageEntry, Project, Split, compute_split
from defect_inspection.platform_utils import atomic_write_text, imread_unicode, imwrite_unicode

Shape = BBox | Polygon
Progress = Callable[[int, int, str], None]  # done, total, message
JPEG_QUALITY = 95


class DatasetBuildError(Exception):
    """The project cannot produce a usable dataset (message is shown to the user)."""


class BuildCancelled(Exception):
    pass


@dataclass
class BuildReport:
    out_dir: str
    train_images: int = 0  # originals in train (defect + OK)
    val_images: int = 0
    ok_train: int = 0
    ok_val: int = 0
    augmented: int = 0  # extra training images generated
    copy_pasted: int = 0  # augmented OK images that received pasted defects
    defect_transformed: int = 0  # augmented images with moved / resized / rotated defects
    dropped_labels: int = 0  # labels removed by the F3.6 rule during augmentation
    skipped_unlabeled: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # val_warnings: why metrics may mislead
    elapsed_s: float = 0.0

    @property
    def total_train(self) -> int:
        return self.train_images + self.augmented


# ---- selection and split ---------------------------------------------------------------

def select_images(project: Project) -> tuple[list[ImageEntry], list[ImageEntry], list[str]]:
    """(defect images, OK images, skipped paths). Labels win over the OK flag."""
    defect, ok, skipped = [], [], []
    for entry in project.config.images:
        status = project.label_status(entry)
        if status.confirmed > 0:
            defect.append(entry)
        elif entry.is_ok:
            ok.append(entry)
        else:
            skipped.append(entry.path)
    return defect, ok, skipped


def split_units(project: Project, entries: list[ImageEntry]) -> dict[str, list[ImageEntry]]:
    """Split units: an image on its own, or all images cut from one source image
    (project.config.source_groups, found by core/image_groups), keyed by the source path."""
    groups = project.config.source_groups
    units: dict[str, list[ImageEntry]] = {}
    for entry in entries:
        units.setdefault(groups.get(entry.path, entry.path), []).append(entry)
    return units


def split_images(project: Project, defect: list[ImageEntry], ok: list[ImageEntry]) -> dict[str, Split]:
    """Stratified split: defect and OK units each follow the project's ratio, seed and manual
    overrides, so Val is never only OK images. A unit is a group of images cut from the same
    source (a defect in a slice must not be in Train while the full photo is in Val); a group
    with any defect image counts as a defect unit, and the first manual split of a member
    applies to the whole group. Without groups every image is its own unit (same result as
    splitting the images directly). A unit with a synthetic image (made by the user from real
    ones) always goes to Train: Val measures real defects only, and the real image it was made
    from must not sit in Val while its copy trains."""
    defect_paths = {e.path for e in defect}
    defect_units, ok_units = [], []
    units = split_units(project, defect + ok)
    unit_split: dict[str, Split] = {}
    for key, members in units.items():
        if any(m.synthetic for m in members):
            unit_split[key] = "train"
            continue
        manual = next((m.split for m in members if m.split != "auto"), "auto")
        unit = ImageEntry(path=key, split=manual)
        (defect_units if any(m.path in defect_paths for m in members) else ok_units).append(unit)
    unit_split.update(compute_split(defect_units, project.config.split))
    unit_split.update(compute_split(ok_units, project.config.split))
    return {m.path: unit_split[key] for key, members in units.items() for m in members}


VAL_MIN_DEFECTS = 10  # below this one missed defect moves recall by > 10 %p
MIN_UNIQUE_PER_CLASS = 10


def val_warnings(project: Project, split: dict[str, Split], defect: list[ImageEntry]) -> list[str]:
    """Why the Val metrics of a dataset may mislead, counted on the original images' labels
    (not crops; a defect seen in several crops is one defect). Images cut from one source count
    once: per group and class the largest member count. Synthetic images count no defect."""
    names = [c.name for c in project.config.classes]
    counts = {"train": [0] * len(names), "val": [0] * len(names)}
    for key, members in split_units(project, defect).items():
        per_class = [0] * len(names)
        for m in members:
            if m.synthetic:
                continue
            member = [0] * len(names)
            for shape in project.load_labels(m).confirmed():
                if 0 <= shape.class_id < len(names):
                    member[shape.class_id] += 1
            per_class = [max(a, b) for a, b in zip(per_class, member)]
        side = split[members[0].path]
        counts[side] = [a + b for a, b in zip(counts[side], per_class)]
    grouped = sum(1 for member in project.config.source_groups if member in split)  # defect + OK images
    synthetic = [e.path for e in project.config.images if e.synthetic and e.path in split]
    warnings = []
    for i, name in enumerate(names):
        if counts["train"][i] + counts["val"][i] == 0:
            continue  # class without labels: nothing to say about its metrics
        if counts["val"][i] == 0:
            warnings.append(f"Val에 '{name}' 결함이 없습니다. 이 클래스의 성능은 측정되지 않습니다 "
                            "(라벨된 이미지를 늘리거나 이미지를 Val로 지정하세요).")
        if counts["train"][i] == 0:
            warnings.append(f"Train에 '{name}' 결함이 없습니다. 이 클래스는 학습되지 않습니다.")
    n_val = sum(counts["val"])
    if 0 < n_val < VAL_MIN_DEFECTS:
        warnings.append(f"Val 결함이 {n_val}개뿐입니다. 결함 1개 차이로 recall이 약 {100 / n_val:.0f}%p 바뀌므로 "
                        "mAP·recall 숫자를 믿기 어렵습니다. Result 탭에서 이미지를 직접 확인하세요.")
    totals = [t + v for t, v in zip(counts["train"], counts["val"])]
    few = [f"{name} {n}" for name, n in zip(names, totals) if 0 < n < MIN_UNIQUE_PER_CLASS]
    if few:
        warnings.append(f"서로 다른 결함 수가 적습니다: {', '.join(few)} (클래스당 20개 이상 권장). "
                        "증강은 실제 결함 수를 대신하지 못합니다.")
    if synthetic:
        warnings.append(f"합성 이미지 {len(synthetic)}장은 같은 원본 묶음과 함께 Train에만 두었고, "
                        "위 결함 수에서 뺐습니다.")
    if grouped:
        warnings.append(f"같은 원본에서 잘라낸 이미지 {grouped}장을 원본과 같은 쪽(Train/Val)에 두었습니다. "
                        "같은 결함이 두 번 학습되므로 원본이나 조각 중 한 종류만 쓰는 것을 권장합니다.")
    return warnings


# ---- build -------------------------------------------------------------------------------

def _out_suffix(src: Path) -> str:
    return ".jpg" if src.suffix.lower() in (".jpg", ".jpeg") else ".png"


def label_map(annotations: ImageAnnotations) -> np.ndarray:
    """uint8 class map of the confirmed labels for semantic segmentation (U-Net++): 0 = background,
    class_id + 1 = defect; later labels win where they overlap. Every shape type is painted
    (`brush.shape_mask`: pixel centres inside); Mask labels keep their holes."""
    from defect_inspection.core.brush import shape_mask

    w, h = annotations.width, annotations.height
    out = np.zeros((h, w), np.uint8)
    for shape in annotations.confirmed():
        x1, y1, x2, y2 = shape.bounds()
        x0, y0 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x1i, y1i = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x0 >= x1i or y0 >= y1i:
            continue
        m = shape_mask(shape, x1i - x0, y1i - y0, (x0, y0))
        out[y0:y1i, x0:x1i][m] = min(shape.class_id + 1, 255)
    return out


def write_labels(label_dir: Path, mask_dir: Path, stem: str, annotations: ImageAnnotations, task: str) -> None:
    """YOLO txt (labels/) and the class-map PNG (masks/) of one dataset image."""
    write_yolo_txt(label_dir / f"{stem}.txt", annotations, task)
    if not imwrite_unicode(mask_dir / f"{stem}.png", label_map(annotations)):
        raise DatasetBuildError(f"마스크를 쓸 수 없습니다: {mask_dir / stem}.png")


def _write_image(path: Path, image: np.ndarray) -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if path.suffix == ".jpg" else None
    if not imwrite_unicode(path, image, params):
        raise OSError(f"cannot write {path}")


def write_data_yaml(path: Path, dataset_root: Path, class_names: list[str], channels: int) -> None:
    data = {
        "path": PurePosixPath(dataset_root.resolve()).as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": dict(enumerate(class_names)),
        "channels": channels,
    }
    atomic_write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def build_full_dataset(project: Project, progress: Progress | None = None,
                       cancel: Callable[[], bool] | None = None) -> BuildReport:
    """Build dataset/full/ from the project's images, labels and augmentation settings."""
    from defect_inspection.core.augment import Augmenter  # Albumentations: import only when building

    started = time.monotonic()
    cfg = project.config
    if not cfg.classes:
        raise DatasetBuildError("클래스가 없습니다. 클래스를 추가하고 라벨링하세요.")
    defect, ok, skipped = select_images(project)
    if not defect:
        raise DatasetBuildError("라벨된 이미지가 없습니다. 결함 이미지를 라벨링하세요.")
    split = split_images(project, defect, ok)
    if not any(split[e.path] == "val" for e in defect):
        raise DatasetBuildError("검증(Val)용 결함 이미지가 없습니다. 라벨된 결함 이미지가 2장 이상 필요합니다.")

    final_dir = project.dataset_dir / "full"
    work_dir = project.dataset_dir / "full.building"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "masks/train", "masks/val"):
        (work_dir / sub).mkdir(parents=True)

    report = BuildReport(out_dir=str(final_dir), skipped_unlabeled=skipped,
                         warnings=val_warnings(project, split, defect))
    channels, aug_cfg = project.channels, cfg.augment
    entries = defect + ok
    total = len(entries) * (1 + aug_cfg.multiplier) + len(defect)  # + patch extraction pass
    done = 0

    def tick(message: str) -> None:
        nonlocal done
        done += 1
        if cancel is not None and cancel():
            raise BuildCancelled()
        if progress is not None:
            progress(done, total, message)

    try:
        # Load and convert every selected image once.
        # path -> (training pixels, labels, source file, source can be copied byte for byte)
        loaded: dict[str, tuple[np.ndarray, ImageAnnotations, Path, bool]] = {}
        for entry in entries:
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
            copyable = (raw.dtype == np.uint8 and raw.shape == image.shape
                         and src.suffix.lower() in (".png", ".jpg", ".jpeg"))
            loaded[entry.path] = (image, ann, src, copyable)

        # Copy-Paste / defect transform patches: from training defect images only.
        patches: list[Patch] = []
        cp: CopyPasteSetting = aug_cfg.copy_paste
        dt: DefectTransformSetting = aug_cfg.defect_transform
        dt_ok = dt.enabled and dt.on_ok_images
        for entry in defect:
            if (cp.enabled or dt_ok) and entry.path in loaded and split[entry.path] == "train":
                image, ann, _, _ = loaded[entry.path]
                patches.extend(extract_patches(image, ann))
            tick("결함 패치 수집")

        augmenter = Augmenter(aug_cfg, channels=channels, seed=cfg.split.seed)
        rng = np.random.default_rng(cfg.split.seed)
        defect_paths = {e.path for e in defect}
        for entry in entries:
            if entry.path not in loaded:
                done += 1 + aug_cfg.multiplier
                continue
            image, ann, src, copyable = loaded[entry.path]
            part = split[entry.path]
            is_ok = entry.path not in defect_paths
            stem, suffix = PurePosixPath(entry.path).stem, _out_suffix(src)
            img_dir, lbl_dir, mask_dir = (work_dir / sub / part for sub in ("images", "labels", "masks"))
            confirmed = ann.model_copy(update={"shapes": ann.confirmed()})

            # original
            dst = img_dir / f"{stem}{suffix}"
            if copyable:
                shutil.copy2(src, dst)  # no re-encoding (no JPEG quality loss)
            else:
                _write_image(dst, image)
            write_labels(lbl_dir, mask_dir, stem, confirmed, cfg.task)
            if part == "train":
                report.train_images += 1
                report.ok_train += is_ok
            else:
                report.val_images += 1
                report.ok_val += is_ok
            tick(f"{part}: {entry.path}")

            if part != "train":
                done += aug_cfg.multiplier
                continue
            for k in range(1, aug_cfg.multiplier + 1):
                src_img, src_ann = image, confirmed
                # rng is drawn for the defect transform only when it is on, so builds without it
                # stay identical to earlier ones
                if is_ok and patches and cp.enabled and rng.random() < cp.p:
                    src_img, src_ann = paste_patches(image, confirmed, patches, rng, cp.max_patches,
                                                     transform=dt if dt.enabled else None)
                    report.copy_pasted += 1
                elif is_ok and patches and dt_ok and rng.random() < dt.p:
                    src_img, src_ann = paste_patches(image, confirmed, patches, rng, dt.max_defects, transform=dt)
                    report.defect_transformed += 1
                elif not is_ok and dt.enabled and rng.random() < dt.p:
                    src_img, src_ann, placed = transform_defects(image, confirmed, dt, rng)
                    report.defect_transformed += placed > 0
                name = f"{stem}_aug{k:03d}{suffix}"
                result = augmenter(src_img, src_ann, image_name=name)
                _write_image(img_dir / name, result.image)
                write_labels(lbl_dir, mask_dir, f"{stem}_aug{k:03d}", result.annotations, cfg.task)
                report.augmented += 1
                report.dropped_labels += result.dropped
                tick(f"증강 {entry.path} #{k}")

        if report.val_images == 0:
            raise DatasetBuildError("검증(Val) 이미지를 읽을 수 없습니다.")
        write_data_yaml(work_dir / "data.yaml", final_dir, [c.name for c in cfg.classes], channels)
        report.elapsed_s = round(time.monotonic() - started, 2)
        info = {"report": asdict(report), "task": cfg.task, "channels": channels,
                "seed": cfg.split.seed, "val_ratio": cfg.split.val_ratio,
                "augment": aug_cfg.model_dump(mode="json")}
        atomic_write_text(work_dir / "build_info.yaml", yaml.safe_dump(info, allow_unicode=True, sort_keys=False))
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    if final_dir.exists():
        shutil.rmtree(final_dir)
    work_dir.rename(final_dir)
    return report

