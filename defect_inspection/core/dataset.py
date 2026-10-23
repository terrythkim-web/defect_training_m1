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
from defect_inspection.core.augment_config import CopyPasteSetting
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
    splitting the images directly)."""
    defect_paths = {e.path for e in defect}
    defect_units, ok_units = [], []
    units = split_units(project, defect + ok)
    for key, members in units.items():
        manual = next((m.split for m in members if m.split != "auto"), "auto")
        unit = ImageEntry(path=key, split=manual)
        (defect_units if any(m.path in defect_paths for m in members) else ok_units).append(unit)
    unit_split = compute_split(defect_units, project.config.split)
    unit_split.update(compute_split(ok_units, project.config.split))
    return {m.path: unit_split[key] for key, members in units.items() for m in members}


VAL_MIN_DEFECTS = 10  # below this one missed defect moves recall by > 10 %p
MIN_UNIQUE_PER_CLASS = 10


def val_warnings(project: Project, split: dict[str, Split], defect: list[ImageEntry]) -> list[str]:
    """Why the Val metrics of a dataset may mislead, counted on the original images' labels
    (not crops; a defect seen in several crops is one defect). Images cut from one source count
    once: per group and class the largest member count."""
    names = [c.name for c in project.config.classes]
    counts = {"train": [0] * len(names), "val": [0] * len(names)}
    for key, members in split_units(project, defect).items():
        per_class = [0] * len(names)
        for m in members:
            member = [0] * len(names)
            for shape in project.load_labels(m).confirmed():
                if 0 <= shape.class_id < len(names):
                    member[shape.class_id] += 1
            per_class = [max(a, b) for a, b in zip(per_class, member)]
        side = split[members[0].path]
        counts[side] = [a + b for a, b in zip(counts[side], per_class)]
    grouped = sum(1 for member in project.config.source_groups if member in split)  # defect + OK images
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
    if grouped:
        warnings.append(f"같은 원본에서 잘라낸 이미지 {grouped}장을 원본과 같은 쪽(Train/Val)에 두었습니다. "
                        "같은 결함이 두 번 학습되므로 원본이나 조각 중 한 종류만 쓰는 것을 권장합니다.")
    return warnings


# ---- Copy-Paste (F3.3) ------------------------------------------------------------------

@dataclass
class Patch:
    pixels: np.ndarray
    mask: np.ndarray  # float32 0..1, feathered edges
    shape: Shape  # in patch coordinates


def extract_patches(image: np.ndarray, annotations: ImageAnnotations, max_fraction: float = 0.5) -> list[Patch]:
    """Cut every confirmed defect out of a training image (pixels + soft mask + its label)."""
    h, w = image.shape[:2]
    patches = []
    for shape in annotations.confirmed():
        x1, y1, x2, y2 = shape.bounds()
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        pw, ph = x2 - x1, y2 - y1
        if pw < 2 or ph < 2 or pw > w * max_fraction or ph > h * max_fraction:
            continue
        mask = np.zeros((ph, pw), np.uint8)
        local = [(px - x1, py - y1) for px, py in shape.to_points()]
        pts = np.round((np.array(local, np.float32) - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255, shift=4)
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0) if min(pw, ph) >= 8 else mask / 255.0
        if isinstance(shape, BBox):
            local_shape = shape.model_copy(update={"x1": shape.x1 - x1, "y1": shape.y1 - y1,
                                                   "x2": shape.x2 - x1, "y2": shape.y2 - y1})
        else:
            local_shape = shape.model_copy(update={"points": local})
        patches.append(Patch(image[y1:y2, x1:x2].copy(), soft.astype(np.float32), local_shape))
    return patches


def _overlaps(box: tuple[float, float, float, float], shapes: list[Shape], margin: float = 2.0) -> bool:
    x1, y1, x2, y2 = box
    for s in shapes:
        a1, b1, a2, b2 = s.bounds()
        if x1 < a2 + margin and a1 < x2 + margin and y1 < b2 + margin and b1 < y2 + margin:
            return True
    return False


def paste_patches(image: np.ndarray, annotations: ImageAnnotations, patches: list[Patch],
                  rng: np.random.Generator, max_patches: int, tries: int = 20) -> tuple[np.ndarray, ImageAnnotations]:
    """Blend 1..max_patches random patches at free spots and add their labels."""
    out = image.copy()
    shapes = list(annotations.confirmed())
    h, w = image.shape[:2]
    for _ in range(int(rng.integers(1, max_patches + 1))):
        patch = patches[int(rng.integers(len(patches)))]
        ph, pw = patch.pixels.shape[:2]
        if pw >= w or ph >= h:
            continue
        for _ in range(tries):
            ox, oy = int(rng.integers(0, w - pw + 1)), int(rng.integers(0, h - ph + 1))
            if not _overlaps((ox, oy, ox + pw, oy + ph), shapes):
                break
        else:
            continue
        region = out[oy:oy + ph, ox:ox + pw].astype(np.float32)
        alpha = patch.mask if out.ndim == 2 else patch.mask[:, :, None]
        out[oy:oy + ph, ox:ox + pw] = np.clip(alpha * patch.pixels + (1 - alpha) * region, 0, 255).astype(np.uint8)
        s = patch.shape
        if isinstance(s, BBox):
            placed = BBox(class_id=s.class_id, x1=s.x1 + ox, y1=s.y1 + oy, x2=s.x2 + ox, y2=s.y2 + oy)
        else:
            placed = Polygon(class_id=s.class_id, points=[(x + ox, y + oy) for x, y in s.points])
        shapes.append(placed)
    return out, annotations.model_copy(update={"shapes": shapes})


# ---- build -------------------------------------------------------------------------------

def _out_suffix(src: Path) -> str:
    return ".jpg" if src.suffix.lower() in (".jpg", ".jpeg") else ".png"


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
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
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

        # Copy-Paste patches: from training defect images only.
        patches: list[Patch] = []
        cp: CopyPasteSetting = aug_cfg.copy_paste
        for entry in defect:
            if cp.enabled and entry.path in loaded and split[entry.path] == "train":
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
            img_dir, lbl_dir = work_dir / "images" / part, work_dir / "labels" / part
            confirmed = ann.model_copy(update={"shapes": ann.confirmed()})

            # original
            dst = img_dir / f"{stem}{suffix}"
            if copyable:
                shutil.copy2(src, dst)  # no re-encoding (no JPEG quality loss)
            else:
                _write_image(dst, image)
            write_yolo_txt(lbl_dir / f"{stem}.txt", confirmed, cfg.task)
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
                if is_ok and patches and cp.enabled and rng.random() < cp.p:
                    src_img, src_ann = paste_patches(image, confirmed, patches, rng, cp.max_patches)
                    report.copy_pasted += 1
                name = f"{stem}_aug{k:03d}{suffix}"
                result = augmenter(src_img, src_ann, image_name=name)
                _write_image(img_dir / name, result.image)
                write_yolo_txt(lbl_dir / f"{stem}_aug{k:03d}.txt", result.annotations, cfg.task)
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

