"""Full dataset build (PROMPTS step 9, PRD F1.5, F3.2, F3.3 Copy-Paste, F7.5)."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from defect_inspection.core.annotation import BBox, Polygon, shapes_from_yolo
from defect_inspection.core.augment_config import AugmentConfig, CopyPasteSetting, TransformSetting
from defect_inspection.core.dataset import (
    BuildCancelled,
    DatasetBuildError,
    build_full_dataset,
    extract_patches,
    paste_patches,
    select_images,
    split_images,
)
from defect_inspection.core.imaging import to_training_8bit
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imread_unicode, imwrite_unicode

W, H = 96, 64
BG = 200


def defect_image(cx, cy, r, value=40):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.circle(img, (cx, cy), r, (value, value, value), -1)
    return img


def make_project(tmp_path, n_defect=6, n_ok=2, task="detect", color_mode="color", multiplier=3,
                 copy_paste=False, val_value=40):
    src = tmp_path / "원본 이미지"
    src.mkdir()
    p = Project.create(tmp_path / "데이터셋 프로젝트", task=task, color_mode=color_mode)
    p.add_class("dent")
    p.add_class("scratch")
    rng = np.random.default_rng(0)
    centers = {}
    for i in range(n_defect):
        cx, cy = int(rng.integers(20, W - 20)), int(rng.integers(16, H - 16))
        centers[f"d{i}.png"] = (cx, cy)
        imwrite_unicode(src / f"d{i}.png", defect_image(cx, cy, 8))
    for i in range(n_ok):
        imwrite_unicode(src / f"ok{i}.png", np.full((H, W, 3), BG, np.uint8))
    imwrite_unicode(src / "unlabeled.png", defect_image(30, 30, 8))
    imwrite_unicode(src / "sugg.png", defect_image(30, 30, 8))
    p.add_images([src])
    for name, (cx, cy) in centers.items():
        ann = p.load_labels(name)
        if name.endswith(("0.png", "2.png", "4.png")):
            ann.shapes.append(BBox(class_id=0, x1=cx - 8, y1=cy - 8, x2=cx + 9, y2=cy + 9))
        else:
            ann.shapes.append(Polygon(class_id=1, points=[(cx - 8, cy), (cx, cy - 8), (cx + 9, cy), (cx, cy + 9)]))
        p.save_labels(ann)
    for i in range(n_ok):
        p.set_ok(f"ok{i}.png")
    ann = p.load_labels("sugg.png")
    ann.add_suggestions([BBox(class_id=0, x1=22, y1=22, x2=39, y2=39, score=0.9)], source="model")
    p.save_labels(ann)
    p.config.augment = AugmentConfig(
        multiplier=multiplier,
        transforms={"hflip": TransformSetting(enabled=True, p=0.5), "rotate90": TransformSetting(enabled=True, p=0.5),
                    "brightness_contrast": TransformSetting(enabled=True, p=0.5, strength=0.3)},
        copy_paste=CopyPasteSetting(enabled=copy_paste, p=1.0, max_patches=2),
    )
    p.save()
    return p


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def files(d: Path, pattern="*"):
    return sorted(p.name for p in d.glob(pattern))


# ---- selection / split -----------------------------------------------------------------

def test_selection_skips_unlabeled_and_suggestion_only(project):
    defect, ok, skipped = select_images(project)
    assert len(defect) == 6 and [e.path for e in ok] == ["ok0.png", "ok1.png"]
    assert sorted(skipped) == ["sugg.png", "unlabeled.png"]


def test_stratified_split_gives_val_defects(project):
    defect, ok, _ = select_images(project)
    split = split_images(project, defect, ok)
    assert sum(split[e.path] == "val" for e in defect) >= 1
    assert sum(split[e.path] == "train" for e in defect) >= 1
    assert sorted(set(split.values())) == ["train", "val"]


# ---- build ---------------------------------------------------------------------------

def test_build_structure_and_counts(project):
    calls = []
    report = build_full_dataset(project, progress=lambda d, t, m: calls.append((d, t)))
    root = project.dataset_dir / "full"

    assert report.out_dir == str(root) and not (project.dataset_dir / "full.building").exists()
    assert report.train_images + report.val_images == 8
    assert report.augmented == report.train_images * 3
    assert sorted(report.skipped_unlabeled) == ["sugg.png", "unlabeled.png"]
    # only train is augmented
    assert not files(root / "images" / "val", "*_aug*")
    assert len(files(root / "images" / "train", "*_aug*")) == report.augmented
    # every image has a label file and vice versa
    for part in ("train", "val"):
        imgs = {Path(n).stem for n in files(root / "images" / part)}
        lbls = {Path(n).stem for n in files(root / "labels" / part)}
        assert imgs == lbls
    assert calls[-1][0] == calls[-1][1]  # progress reaches 100%


def test_data_yaml(project):
    build_full_dataset(project)
    root = project.dataset_dir / "full"
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert data == {"path": root.resolve().as_posix(), "train": "images/train", "val": "images/val",
                    "nc": 2, "names": {0: "dent", 1: "scratch"}, "channels": 3}
    info = yaml.safe_load((root / "build_info.yaml").read_text(encoding="utf-8"))
    assert info["augment"]["multiplier"] == 3 and info["report"]["augmented"] > 0


def test_ultralytics_accepts_dataset(project):
    from ultralytics.data.utils import check_det_dataset, verify_image_label

    build_full_dataset(project)
    root = project.dataset_dir / "full"
    data = check_det_dataset(str(root / "data.yaml"))
    assert data["nc"] == 2 and data["channels"] == 3
    for img in (root / "images" / "train").glob("*"):
        lbl = root / "labels" / "train" / f"{img.stem}.txt"
        res = verify_image_label((str(img), str(lbl), "", False, 2, 0, 3, False))
        assert res[8] == 0, res[9]  # not corrupt


def test_original_labels_and_ok_background(project):
    build_full_dataset(project)
    root = project.dataset_dir / "full"
    for part in ("train", "val"):
        for name in files(root / "labels" / part, "d?.txt"):
            stem = Path(name).stem
            expected = project.load_labels(f"{stem}.png").to_yolo_lines("detect")
            assert (root / "labels" / part / name).read_text(encoding="utf-8").splitlines() == expected
        for name in files(root / "labels" / part, "ok?.txt"):
            assert (root / "labels" / part / name).read_bytes() == b""


def test_augmented_labels_sit_on_defects(project):
    """Each augmented box covers the dark defect: label/pixel agreement end to end."""
    build_full_dataset(project)
    root = project.dataset_dir / "full"
    checked = 0
    for img_path in (root / "images" / "train").glob("d*_aug*.png"):
        img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        lines = (root / "labels" / "train" / f"{img_path.stem}.txt").read_text(encoding="utf-8").splitlines()
        for shape in shapes_from_yolo(lines, w, h, "detect"):
            x1, y1, x2, y2 = (int(round(v)) for v in shape.bounds())
            assert img[y1:y2, x1:x2].min() < 120  # dark defect inside the box
            checked += 1
    assert checked >= 6


def test_unchanged_png_copied_bytewise(project):
    build_full_dataset(project)
    root = project.dataset_dir / "full"
    for part in ("train", "val"):
        for name in files(root / "images" / part, "d?.png"):
            src = project.image_abspath(name)
            assert (root / "images" / part / name).read_bytes() == src.read_bytes()


def test_rebuild_replaces_old_dataset(project):
    build_full_dataset(project)
    stale = project.dataset_dir / "full" / "images" / "train" / "stale.png"
    stale.write_bytes(b"x")
    build_full_dataset(project)
    assert not stale.exists()


def test_build_is_reproducible(project, tmp_path):
    def digest():
        h = hashlib.sha256()
        for f in sorted((project.dataset_dir / "full").rglob("*")):
            if f.is_file() and f.name != "build_info.yaml":
                h.update(f.relative_to(project.dataset_dir).as_posix().encode())
                h.update(f.read_bytes())
        return h.hexdigest()

    build_full_dataset(project)
    first = digest()
    build_full_dataset(project)
    assert digest() == first


def test_cancel_keeps_previous_dataset(project):
    build_full_dataset(project)
    marker = project.dataset_dir / "full" / "data.yaml"
    before = marker.read_bytes()
    count = {"n": 0}

    def cancel():
        count["n"] += 1
        return count["n"] > 5

    with pytest.raises(BuildCancelled):
        build_full_dataset(project, cancel=cancel)
    assert marker.read_bytes() == before
    assert not (project.dataset_dir / "full.building").exists()


# ---- pixels: 16-bit, grayscale, colour ------------------------------------------------

def test_to_training_8bit():
    assert to_training_8bit(np.full((2, 2), 40000, np.uint16), 1)[0, 0] == 40000 >> 8
    assert to_training_8bit(np.full((2, 2), 0.5, np.float32), 1)[0, 0] == 127
    assert to_training_8bit(np.zeros((2, 2), np.uint8), 3).shape == (2, 2, 3)
    assert to_training_8bit(np.zeros((2, 2, 4), np.uint8), 3).shape == (2, 2, 3)
    assert to_training_8bit(np.zeros((2, 2, 3), np.uint8), 1).shape == (2, 2)
    # same mapping for every image: relative brightness is kept
    dark, bright = np.full((1, 1), 10000, np.uint16), np.full((1, 1), 50000, np.uint16)
    assert to_training_8bit(dark, 1)[0, 0] < to_training_8bit(bright, 1)[0, 0]


def test_16bit_tiff_scaled_by_256(tmp_path):
    p = make_project(tmp_path)
    tif = tmp_path / "deep.tif"
    img = np.full((H, W), 40000, np.uint16)
    img[20:40, 30:50] = 8000
    imwrite_unicode(tif, img)
    p.add_images([tif])
    ann = p.load_labels("deep.tif")
    ann.shapes.append(BBox(class_id=0, x1=30, y1=20, x2=50, y2=40))
    p.save_labels(ann)
    p.config.split.val_ratio = 0.01  # keep deep.tif likely in train; we only check the original
    build_full_dataset(p)
    out = next((p.dataset_dir / "full" / "images").rglob("deep.png"))
    px = imread_unicode(out)
    assert px.dtype == np.uint8 and px.shape == (H, W, 3)
    assert px[0, 0, 0] == 40000 >> 8 and px[30, 40, 0] == 8000 >> 8


def test_grayscale_project_dataset(tmp_path):
    p = make_project(tmp_path, color_mode="grayscale")
    build_full_dataset(p)
    root = p.dataset_dir / "full"
    assert yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["channels"] == 1
    for f in (root / "images" / "train").glob("*"):
        assert imread_unicode(f).ndim == 2


def test_seg_task_writes_polygons(tmp_path):
    p = make_project(tmp_path, task="seg")
    build_full_dataset(p)
    lines = []
    for f in (p.dataset_dir / "full" / "labels").rglob("d?.txt"):
        lines += f.read_text(encoding="utf-8").splitlines()
    assert lines and all(len(line.split()) >= 7 for line in lines)  # class + >= 3 points


# ---- Copy-Paste --------------------------------------------------------------------------

def test_extract_and_paste_patch():
    img = defect_image(40, 30, 8, value=20)
    from defect_inspection.core.annotation import ImageAnnotations

    ann = ImageAnnotations(image="a.png", width=W, height=H,
                           shapes=[BBox(class_id=1, x1=32, y1=22, x2=49, y2=39)])
    patches = extract_patches(img, ann)
    assert len(patches) == 1 and patches[0].pixels.shape == (17, 17, 3)
    target = ImageAnnotations(image="ok.png", width=W, height=H)
    out, out_ann = paste_patches(np.full((H, W, 3), BG, np.uint8), target, patches,
                                 np.random.default_rng(0), max_patches=1)
    (s,) = out_ann.shapes
    x1, y1, x2, y2 = (int(v) for v in s.bounds())
    assert s.class_id == 1 and (x2 - x1, y2 - y1) == (17, 17)
    assert out[y1:y2, x1:x2].min() < 60  # defect pixels pasted inside the new label
    assert out[:, :, 0].min() == out[y1:y2, x1:x2, 0].min()


def test_paste_avoids_existing_labels():
    from defect_inspection.core.annotation import ImageAnnotations

    patch_src = ImageAnnotations(image="a.png", width=W, height=H, shapes=[BBox(class_id=0, x1=0, y1=0, x2=10, y2=10)])
    patches = extract_patches(defect_image(5, 5, 4), patch_src)
    existing = BBox(class_id=0, x1=0, y1=0, x2=W, y2=H / 2)
    target = ImageAnnotations(image="b.png", width=W, height=H, shapes=[existing])
    for seed in range(10):
        _, out = paste_patches(np.full((H, W, 3), BG, np.uint8), target, patches, np.random.default_rng(seed), 3)
        for s in out.shapes[1:]:
            assert s.bounds()[1] >= H / 2  # never on top of the existing label


def test_copy_paste_build_uses_train_patches_only(tmp_path):
    p = make_project(tmp_path, copy_paste=True, multiplier=4)
    defect, ok, _ = select_images(p)
    split = split_images(p, defect, ok)
    # Make val defect pixels uniquely recognisable (value 7) to detect leakage.
    for e in defect:
        if split[e.path] == "val":
            img = imread_unicode(p.image_abspath(e))
            img[img < 100] = 7
            imwrite_unicode(p.image_abspath(e), img)
    report = build_full_dataset(p)
    root = p.dataset_dir / "full"
    assert report.copy_pasted == report.ok_train * 4 > 0
    pasted_labels = 0
    for f in (root / "images" / "train").glob("ok*_aug*.png"):
        img = imread_unicode(f)
        assert not (img == 7).any(), "pixels from a Val image leaked into Train"
        pasted_labels += len((root / "labels" / "train" / f"{f.stem}.txt").read_text(encoding="utf-8").splitlines())
    assert pasted_labels > 0


# ---- errors ----------------------------------------------------------------------------

def test_errors(tmp_path):
    empty = Project.create(tmp_path / "빈 프로젝트")
    with pytest.raises(DatasetBuildError, match="클래스"):
        build_full_dataset(empty)
    empty.add_class("dent")
    with pytest.raises(DatasetBuildError, match="라벨된 이미지가 없습니다"):
        build_full_dataset(empty)


def test_single_labeled_image_needs_more(tmp_path):
    p = make_project(tmp_path, n_defect=1, n_ok=0)
    with pytest.raises(DatasetBuildError, match="2장 이상"):
        build_full_dataset(p)
    assert not (p.dataset_dir / "full").exists()


def test_augment_settings_saved_in_project(project):
    reopened = Project.open(project.root)
    assert reopened.config.augment.multiplier == 3
    assert reopened.config.augment.setting("hflip").enabled
    fresh = Project.create(project.root.parent / "새 프로젝트")
    assert fresh.config.augment.multiplier == 20 and "rotate90" in fresh.config.augment.enabled_keys()
