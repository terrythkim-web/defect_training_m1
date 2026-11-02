"""What a built dataset was made from, to warn before training on an outdated one.

A build stores `dataset_inputs(project, mode)` in build_info.yaml (`inputs`); `dataset_changes`
compares it with the project now and says in Korean what changed: labels (confirmed shapes only,
without ids — suggestions are never training data), Train/Val sides, images added or left out,
image files replaced (size), classes/task/colour mode, augmentation settings and, for the crop
dataset, crop settings and ROIs. Training on a dataset built before such a change silently
ignored the change (seen: four runs after re-splitting and re-labeling, all on the old build).
Qt-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from defect_inspection.core.project import Project

LABEL_FIELDS_IGNORED = {"id", "status", "source", "score"}


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _label_digest(project: Project, path: str) -> str:
    shapes = [{k: v for k, v in s.model_dump(mode="json").items() if k not in LABEL_FIELDS_IGNORED}
              for s in project.load_labels(path).confirmed()]
    return _digest(shapes)


def _file_size(project: Project, path: str) -> int:
    try:
        return project.image_abspath(path).stat().st_size
    except OSError:
        return -1


def _crop_digest(project: Project) -> str:
    from defect_inspection.core.roi import CROPS_FILE

    rois = project.root / CROPS_FILE
    try:
        roi_data = json.loads(rois.read_text(encoding="utf-8")) if rois.is_file() else {}
    except (OSError, ValueError):
        roi_data = {"unreadable": True}
    roi_data.pop("options", None)  # drawing options only
    return _digest({"crop": project.config.crop.model_dump(mode="json"), "rois": roi_data})


def dataset_inputs(project: Project, mode: str) -> dict[str, Any]:
    """JSON/YAML-friendly record of everything the `mode` ("full"/"crop") build depends on."""
    from defect_inspection.core.dataset import select_images, split_images

    cfg = project.config
    defect, ok, _ = select_images(project)
    split = split_images(project, defect, ok)
    entries = defect + ok
    out: dict[str, Any] = {
        "version": 1,
        "classes": _digest([[c.name for c in cfg.classes], cfg.task, project.channels]),
        "augment": _digest(cfg.augment.model_dump(mode="json")),
        "split": {e.path: split[e.path] for e in entries},
        "labels": {e.path: _label_digest(project, e.path) for e in defect},
        "files": {e.path: _file_size(project, e.path) for e in entries},
    }
    if mode == "crop":
        out["crop"] = _crop_digest(project)
    return out


def _count_changed(before: dict, now: dict, keys: set) -> int:
    return sum(1 for k in keys if before.get(k) != now.get(k))


def dataset_changes(project: Project, mode: str) -> list[str] | None:
    """What changed since dataset/<mode> was built ([] = up to date). None when there is no
    dataset or it was not built by the app (no build_info.yaml): nothing to judge."""
    from defect_inspection.core.crop import read_build_info

    root = Path(project.dataset_dir) / mode
    if not (root / "data.yaml").is_file() or not (root / "build_info.yaml").is_file():
        return None
    try:
        before = read_build_info(root).get("inputs")
    except (OSError, ValueError):
        before = None
    if not isinstance(before, dict) or before.get("version") != 1:
        return ["이전 버전에서 만든 데이터셋이라 비교할 수 없음"]
    now = dataset_inputs(project, mode)
    changes = []
    if before.get("classes") != now["classes"]:
        changes.append("클래스·작업 종류·색 모드")
    old_paths, new_paths = set(before.get("split", {})), set(now["split"])
    if old_paths != new_paths:
        changes.append(f"추가되거나 빠진 이미지 {len(old_paths ^ new_paths)}장")
    both = old_paths & new_paths
    n = _count_changed(before.get("labels", {}), now["labels"], both)
    if n:
        changes.append(f"라벨이 바뀐 이미지 {n}장")
    n = _count_changed(before.get("split", {}), now["split"], both)
    if n:
        changes.append(f"Train/Val이 바뀐 이미지 {n}장")
    n = _count_changed(before.get("files", {}), now["files"], both)
    if n:
        changes.append(f"파일이 바뀐 이미지 {n}장")
    if before.get("augment") != now["augment"]:
        changes.append("증강 설정")
    if mode == "crop" and before.get("crop") != now.get("crop"):
        changes.append("Crop 설정·ROI")
    return changes
