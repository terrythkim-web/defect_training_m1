"""Project model: project.yaml schema, image registry and Train/Val split (PRD F1)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal

import cv2
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from defect_inspection.core.annotation import (
    ImageAnnotations,
    load_annotations,
    save_annotations,
    write_yolo_txt,
)
from defect_inspection.core.augment_config import BUILTIN_PRESETS, DEFAULT_PRESET, AugmentConfig
from defect_inspection.core.crop_config import CropConfig
from defect_inspection.core.imaging import is_grayscale, to_grayscale
from defect_inspection.platform_utils import (
    atomic_write_text,
    imread_unicode,
    imwrite_unicode,
    is_supported_image,
)

PROJECT_FILE = "project.yaml"
SCHEMA_VERSION = 1

Split = Literal["train", "val"]
SplitAssignment = Literal["auto", "train", "val"]
TaskMode = Literal["detect", "seg"]
ColorMode = Literal["color", "grayscale"]

_JPEG_QUALITY = 95

_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_DEFAULT_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4",
]


class ClassDef(BaseModel):
    """A defect class. Its index in ProjectConfig.classes is the YOLO class id."""

    name: str = Field(min_length=1)
    color: str = "#E6194B"

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("class name must not be blank")
        return v

    @field_validator("color")
    @classmethod
    def _check_color(cls, v: str) -> str:
        if not _COLOR_RE.match(v):
            raise ValueError(f"color must be #RRGGBB, got {v!r}")
        return v.upper()


class ImageEntry(BaseModel):
    """An image inside the project's images/ folder."""

    model_config = ConfigDict(validate_assignment=True)

    path: str  # relative to images/, POSIX separators so project.yaml is portable across OSes
    split: SplitAssignment = "auto"  # manual override (F1.4)
    is_ok: bool = False  # normal image used as a background/negative sample (F1.5)

    @field_validator("path")
    @classmethod
    def _check_path(cls, v: str) -> str:
        v = v.replace("\\", "/")
        p = PurePosixPath(v)
        if not v or p.is_absolute() or re.match(r"^[A-Za-z]:", v) or ".." in p.parts:
            raise ValueError(f"image path must be relative to images/, got {v!r}")
        return str(p)


class SplitConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42


class ProjectConfig(BaseModel):
    """Schema of project.yaml."""

    version: int = SCHEMA_VERSION
    name: str = Field(min_length=1)
    task: TaskMode = "detect"
    # grayscale: images are stored single-channel to cut disk use, and datasets/models use 1 channel (F1.6)
    color_mode: ColorMode = "color"
    classes: list[ClassDef] = Field(default_factory=list)
    images: list[ImageEntry] = Field(default_factory=list)
    split: SplitConfig = Field(default_factory=SplitConfig)
    # Offline augmentation used when building datasets (F3); starts from the "보통" preset.
    augment: AugmentConfig = Field(default_factory=lambda: BUILTIN_PRESETS[DEFAULT_PRESET].model_copy(deep=True))
    # Crop dataset (F7.2-F7.4); manual ROIs themselves live in crops.json.
    crop: CropConfig = Field(default_factory=CropConfig)

    @model_validator(mode="after")
    def _check_unique(self) -> ProjectConfig:
        names = [c.name for c in self.classes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate class names: {names}")
        paths = [i.path for i in self.images]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate image paths in project")
        return self


@dataclass(frozen=True)
class LabelStatus:
    confirmed: int
    suggested: int
    is_ok: bool

    @property
    def is_labeled(self) -> bool:
        """Done for training: has confirmed labels, or is a normal (OK) image with none needed."""
        return self.confirmed > 0 or self.is_ok

    @property
    def has_suggestions(self) -> bool:
        return self.suggested > 0


@dataclass
class GrayscaleConversion:
    converted: int
    already_gray: int
    failed: list[str]
    bytes_before: int
    bytes_after: int


@dataclass
class ImageSync:
    """Result of Project.sync_images_dir: what changed in images/ outside the app."""

    added: list[ImageEntry] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)  # (found as, registered as)
    pending: list[str] = field(default_factory=list)  # still being written / unreadable: retry later

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


REMOVED_LABELS_DIR = "removed"  # labels/removed/: labels of images deleted from images/


def compute_split(images: Iterable[ImageEntry], split: SplitConfig) -> dict[str, Split]:
    """Assign every image to train or val, reproducibly.

    Manual assignments are kept. Remaining images are ranked by a seeded SHA-1 hash of their
    path (stable across runs, OSes and Python's hash randomization) and the lowest-ranked ones
    go to val until the total val count reaches round(n * val_ratio). With 2+ images there is
    always at least one train and one val image. Adding an image moves at most one other image.
    """
    images = list(images)
    result: dict[str, Split] = {i.path: i.split for i in images if i.split != "auto"}
    auto = [i.path for i in images if i.split == "auto"]
    n = len(images)
    if not auto:
        return result

    target_val = round(n * split.val_ratio)
    if n >= 2:
        target_val = min(max(target_val, 1), n - 1)
    manual_val = sum(1 for s in result.values() if s == "val")
    k = min(max(target_val - manual_val, 0), len(auto))

    def rank(path: str) -> str:
        return hashlib.sha1(f"{split.seed}:{path}".encode("utf-8")).hexdigest()

    for idx, path in enumerate(sorted(auto, key=rank)):
        result[path] = "val" if idx < k else "train"
    return result


class Project:
    """A project folder: project.yaml + images/ + labels/ + generated outputs (PRD 6)."""

    def __init__(self, root: Path, config: ProjectConfig) -> None:
        self.root = Path(root)
        self.config = config

    # ---- folders -------------------------------------------------------------
    @property
    def channels(self) -> int:
        """Image channels for datasets, training and ONNX input."""
        return 1 if self.config.color_mode == "grayscale" else 3

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    # ---- create / open / save ------------------------------------------------
    @classmethod
    def create(
        cls,
        root: str | Path,
        name: str | None = None,
        task: TaskMode = "detect",
        color_mode: ColorMode = "color",
    ) -> Project:
        root = Path(root)
        if (root / PROJECT_FILE).exists():
            raise FileExistsError(f"project already exists: {root / PROJECT_FILE}")
        root.mkdir(parents=True, exist_ok=True)
        project = cls(root, ProjectConfig(name=name or root.name, task=task, color_mode=color_mode))
        project.images_dir.mkdir(exist_ok=True)
        project.labels_dir.mkdir(exist_ok=True)
        project.save()
        return project

    @classmethod
    def open(cls, root: str | Path) -> Project:
        root = Path(root)
        with open(root / PROJECT_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(root, ProjectConfig.model_validate(data))

    def save(self) -> None:
        """Write project.yaml atomically (temp file + replace) in UTF-8."""
        text = yaml.safe_dump(
            self.config.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        )
        atomic_write_text(self.project_file, text)

    # ---- classes -------------------------------------------------------------
    def add_class(self, name: str, color: str | None = None) -> ClassDef:
        if color is None:
            color = _DEFAULT_COLORS[len(self.config.classes) % len(_DEFAULT_COLORS)]
        new = ClassDef(name=name, color=color)
        if any(c.name == new.name for c in self.config.classes):
            raise ValueError(f"class already exists: {new.name}")
        self.config.classes.append(new)
        return new

    def rename_class(self, index: int, name: str) -> None:
        renamed = ClassDef(name=name, color=self.config.classes[index].color)
        if any(c.name == renamed.name for i, c in enumerate(self.config.classes) if i != index):
            raise ValueError(f"class already exists: {renamed.name}")
        self.config.classes[index] = renamed

    def set_class_color(self, index: int, color: str) -> None:
        self.config.classes[index] = ClassDef(name=self.config.classes[index].name, color=color)

    # ---- images --------------------------------------------------------------
    def image(self, path: str) -> ImageEntry:
        for entry in self.config.images:
            if entry.path == path:
                return entry
        raise KeyError(path)

    def image_abspath(self, entry: ImageEntry | str) -> Path:
        rel = entry.path if isinstance(entry, ImageEntry) else entry
        return self.images_dir.joinpath(*PurePosixPath(rel).parts)

    def add_images(self, sources: Iterable[str | Path]) -> list[ImageEntry]:
        """Copy image files (or every supported image in given folders) into images/.

        Unsupported files are skipped. Clashing names get a numeric suffix ("a.png" -> "a_1.png").
        Stems must be unique too ("a.png" + "a.jpg" -> "a_1.jpg") because labels are stored as
        labels/<stem>.json / .txt, matching the YOLO image/label naming rule.
        In grayscale mode color images are converted on import; files that cannot be decoded
        are skipped with a warning.
        """
        files: list[Path] = []
        for src in map(Path, sources):
            if src.is_dir():
                files.extend(p for p in sorted(src.iterdir()) if p.is_file() and is_supported_image(p))
            elif src.is_file() and is_supported_image(src):
                files.append(src)

        self.images_dir.mkdir(parents=True, exist_ok=True)
        added: list[ImageEntry] = []
        for src in files:
            dest = self._unique_dest(src.name)
            if self.config.color_mode == "grayscale":
                image = imread_unicode(src)
                if image is None:
                    warnings.warn(f"cannot decode image, skipped: {src}", stacklevel=2)
                    continue
                if is_grayscale(image):
                    shutil.copy2(src, dest)  # no re-encode, no JPEG quality loss
                elif not _write_image(dest, to_grayscale(image)):
                    warnings.warn(f"cannot write image, skipped: {dest}", stacklevel=2)
                    continue
            else:
                shutil.copy2(src, dest)
            entry = ImageEntry(path=dest.name)
            self.config.images.append(entry)
            added.append(entry)
        return added

    def sync_images_dir(self, settle_sec: float = 1.0, now: float | None = None) -> ImageSync:
        """Match the image registry to what is actually in images/ (files copied in or deleted
        with Finder/Explorer while the project is open or closed).

        New supported files are registered like add_images (unique stems, renamed on disk if
        needed; converted to single channel in grayscale mode). Files changed less than
        `settle_sec` ago, or that cannot be decoded yet, are reported as pending — a copy may
        still be running. Entries whose file is gone are removed and their labels moved to
        labels/removed/, from where they come back if the same file name reappears.
        """
        now = time.time() if now is None else now
        result = ImageSync()
        removed_dir = self.labels_dir / REMOVED_LABELS_DIR
        for entry in list(self.config.images):
            if not self.image_abspath(entry).is_file():
                self.config.images.remove(entry)
                for label in (self.label_json_path(entry), self.label_txt_path(entry)):
                    if label.is_file():
                        removed_dir.mkdir(parents=True, exist_ok=True)
                        os.replace(label, removed_dir / label.name)
                result.removed.append(entry.path)
        if not self.images_dir.is_dir():
            return result

        registered = {e.path for e in self.config.images}
        for src in sorted(self.images_dir.iterdir()):
            if (src.name in registered or src.name.startswith((".", "~$")) or not src.is_file()
                    or not is_supported_image(src)):
                continue
            try:
                if now - src.stat().st_mtime < settle_sec:
                    result.pending.append(src.name)
                    continue
            except OSError:
                continue
            if self.config.color_mode == "grayscale":
                image = imread_unicode(src)
                if image is None:
                    result.pending.append(src.name)
                    continue
                if not is_grayscale(image) and not _write_image(src, to_grayscale(image)):
                    result.pending.append(src.name)
                    continue
            dest = src
            stem = src.stem.lower()
            if any(PurePosixPath(e.path).stem.lower() == stem for e in self.config.images) \
                    or not self._labels_free_for(src.name):
                dest = self._unique_dest(src.name)
                try:
                    os.replace(src, dest)
                except OSError:  # Windows: still locked by the copying program
                    result.pending.append(src.name)
                    continue
                result.renamed.append((src.name, dest.name))
            entry = ImageEntry(path=dest.name)
            self.config.images.append(entry)
            registered.add(dest.name)
            self._restore_removed_labels(entry)
            result.added.append(entry)
        return result

    def _labels_free_for(self, rel: str) -> bool:
        """False if labels/<stem>.json exists for a different image (would be attached wrongly)."""
        path = self.label_json_path(rel)
        if not path.is_file():
            return True
        try:
            return load_annotations(path).image == rel
        except (OSError, ValueError):
            return False

    def _restore_removed_labels(self, entry: ImageEntry) -> None:
        removed_dir = self.labels_dir / REMOVED_LABELS_DIR
        json_path = removed_dir / self.label_json_path(entry).name
        if self.has_labels(entry) or not json_path.is_file():
            return
        try:
            if load_annotations(json_path).image != entry.path:
                return
        except (OSError, ValueError):
            return
        os.replace(json_path, self.label_json_path(entry))
        ann = load_annotations(self.label_json_path(entry))
        write_yolo_txt(self.label_txt_path(entry), ann, self.config.task)  # task may have changed
        (removed_dir / self.label_txt_path(entry).name).unlink(missing_ok=True)

    def _unique_dest(self, filename: str) -> Path:
        # Case-insensitive: Windows file systems treat A.png and a.png as the same file.
        taken_stems = {PurePosixPath(e.path).stem.lower() for e in self.config.images}
        taken_stems |= {p.stem.lower() for p in self.images_dir.iterdir()} if self.images_dir.exists() else set()
        stem, suffix = Path(filename).stem, Path(filename).suffix
        candidate, n = stem, 0
        while candidate.lower() in taken_stems:
            n += 1
            candidate = f"{stem}_{n}"
        return self.images_dir / f"{candidate}{suffix}"

    def convert_to_grayscale(self) -> GrayscaleConversion:
        """Switch the project to grayscale and convert stored color images in place.

        Only the copies in images/ are changed; files the user imported from are untouched.
        JPEGs are re-encoded at quality 95. Call save() afterwards to persist color_mode.
        """
        result = GrayscaleConversion(0, 0, [], 0, 0)
        for entry in self.config.images:
            path = self.image_abspath(entry)
            size = path.stat().st_size
            result.bytes_before += size
            image = imread_unicode(path)
            if image is None:
                result.failed.append(entry.path)
                result.bytes_after += size
                continue
            if is_grayscale(image):
                result.already_gray += 1
                result.bytes_after += size
                continue
            tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
            if not _write_image(tmp, to_grayscale(image)):
                tmp.unlink(missing_ok=True)
                result.failed.append(entry.path)
                result.bytes_after += size
                continue
            os.replace(tmp, path)
            result.converted += 1
            result.bytes_after += path.stat().st_size
        self.config.color_mode = "grayscale"
        return result

    # ---- labels (PRD F2.6) ---------------------------------------------------
    def label_json_path(self, entry: ImageEntry | str) -> Path:
        rel = entry.path if isinstance(entry, ImageEntry) else entry
        return self.labels_dir / f"{PurePosixPath(rel).stem}.json"

    def label_txt_path(self, entry: ImageEntry | str) -> Path:
        rel = entry.path if isinstance(entry, ImageEntry) else entry
        return self.labels_dir / f"{PurePosixPath(rel).stem}.txt"

    def has_labels(self, entry: ImageEntry | str) -> bool:
        return self.label_json_path(entry).is_file()

    def load_labels(self, entry: ImageEntry | str, size: tuple[int, int] | None = None) -> ImageAnnotations:
        """Labels of an image; an empty set if none were saved.

        For the empty set the image size is needed: pass size=(width, height) when the image is
        already decoded, otherwise the file is read here.
        """
        rel = entry.path if isinstance(entry, ImageEntry) else entry
        path = self.label_json_path(rel)
        if path.is_file():
            return load_annotations(path)
        if size is not None:
            return ImageAnnotations(image=rel, width=size[0], height=size[1])
        image = imread_unicode(self.image_abspath(rel))
        if image is None:
            raise FileNotFoundError(f"cannot read image: {self.image_abspath(rel)}")
        return ImageAnnotations(image=rel, width=image.shape[1], height=image.shape[0])

    def label_status(self, entry: ImageEntry | str) -> LabelStatus:
        """Counts for the image list (F1.3): confirmed labels and pending suggestions."""
        entry = entry if isinstance(entry, ImageEntry) else self.image(entry)
        confirmed = suggested = 0
        path = self.label_json_path(entry)
        if path.is_file():
            ann = load_annotations(path)
            confirmed, suggested = len(ann.confirmed()), len(ann.suggestions())
        return LabelStatus(confirmed=confirmed, suggested=suggested, is_ok=entry.is_ok)

    def save_labels(self, annotations: ImageAnnotations) -> None:
        """Save the JSON (source of truth) and regenerate the YOLO txt for the project's task."""
        self.image(annotations.image)  # must be a project image
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        save_annotations(self.label_json_path(annotations.image), annotations)
        write_yolo_txt(self.label_txt_path(annotations.image), annotations, self.config.task)

    def regenerate_yolo_labels(self) -> int:
        """Rewrite every YOLO txt from JSON, e.g. after switching task between detect and seg."""
        count = 0
        for entry in self.config.images:
            if self.has_labels(entry):
                write_yolo_txt(self.label_txt_path(entry), load_annotations(self.label_json_path(entry)), self.config.task)
                count += 1
        return count

    def set_split(self, path: str, split: SplitAssignment) -> None:
        self.image(path).split = split

    def set_ok(self, path: str, is_ok: bool = True) -> None:
        self.image(path).is_ok = is_ok

    def splits(self) -> dict[str, Split]:
        return compute_split(self.config.images, self.config.split)


def _write_image(path: Path, image) -> bool:
    params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY] if path.suffix.lower() in (".jpg", ".jpeg") else None
    return imwrite_unicode(path, image, params)
