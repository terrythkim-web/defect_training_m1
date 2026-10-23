"""Defect transform (core/defect_patches): defects get a random size, angle and shifted position;
labels must keep following the defect pixels."""

import numpy as np
import pytest
import yaml
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon
from defect_inspection.core.augment_config import AugmentConfig, DefectTransformSetting
from defect_inspection.core.crop import build_crop_dataset
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.defect_patches import (
    extract_patches,
    paste_patches,
    patch_for,
    transform_defects,
    transform_patch,
)
from defect_inspection.core.preview import augment_previews
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imread_unicode, imwrite_unicode

W, H, BG, DARK = 240, 160, 200, 30
import cv2  # noqa: E402


def blob_image(centres, r=10):
    img = np.full((H, W, 3), BG, np.uint8)
    for cx, cy in centres:
        cv2.circle(img, (cx, cy), r, (DARK, DARK, DARK), -1)
    return img


def octagon(cx, cy, r=11):
    return Polygon(class_id=0, points=[(cx + r * np.cos(a), cy + r * np.sin(a))
                                        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])


def inside_mean(image, shape):
    mask = np.zeros(image.shape[:2], np.uint8)
    pts = np.round((np.array(shape.to_points(), np.float32) - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, shift=4)
    gray = image if image.ndim == 2 else image[:, :, 0]
    return float(gray[mask > 0].mean())


def cfg(**kw):
    base = dict(enabled=True, p=1.0, scale_min=1.0, scale_max=1.0, max_angle=0.0, max_shift=0.5)
    base.update(kw)
    return DefectTransformSetting(**base)


def test_transform_patch_scale_and_rotation():
    img = blob_image([(60, 50)])
    patch = patch_for(img, BBox(class_id=0, x1=40, y1=40, x2=80, y2=60))
    big = transform_patch(patch, 2.0, 0.0)
    assert big.pixels.shape[:2] == (40, 80) and isinstance(big.shape, BBox)
    assert big.shape.bounds() == pytest.approx((0, 0, 80, 40), abs=0.6)
    turned = transform_patch(patch, 1.0, 90.0)
    assert turned.pixels.shape[:2] == (40, 20) and isinstance(turned.shape, Polygon)
    x1, y1, x2, y2 = turned.shape.bounds()
    assert (x2 - x1, y2 - y1) == pytest.approx((20, 40), abs=0.6)


def test_setting_validation():
    with pytest.raises(ValueError):
        DefectTransformSetting(scale_min=1.5, scale_max=1.0)


@pytest.mark.parametrize("seed", range(8))
def test_labels_follow_moved_scaled_rotated_defects(seed):
    img = blob_image([(60, 50), (170, 110)])
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[octagon(60, 50), octagon(170, 110)])
    setting = cfg(scale_min=0.6, scale_max=1.6, max_angle=180.0, max_defects=2)
    out, out_ann, placed = transform_defects(img, ann, setting, np.random.default_rng(seed))
    assert placed >= 1 and len(out_ann.shapes) == 2 + placed  # "add" keeps the originals
    for s in out_ann.shapes:
        assert inside_mean(out, s) < 110, s  # every label sits on dark defect pixels


def test_move_mode_erases_the_original():
    img = blob_image([(60, 50)])
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[octagon(60, 50)])
    out, out_ann, placed = transform_defects(img, ann, cfg(mode="move", max_shift=1.0), np.random.default_rng(3))
    assert placed == 1 and len(out_ann.shapes) == 1
    (moved,) = out_ann.shapes
    assert moved.bounds() != pytest.approx(octagon(60, 50).bounds(), abs=1)
    assert inside_mean(out, moved) < 110
    assert inside_mean(out, octagon(60, 50, r=8)) > BG - 30  # filled from the surroundings


@pytest.mark.parametrize("direction, keep", [("horizontal", 1), ("vertical", 0)])
def test_direction_keeps_the_other_axis(direction, keep):
    img = blob_image([(60, 50)])
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[octagon(60, 50)])
    for seed in range(6):
        _, out_ann, placed = transform_defects(img, ann, cfg(direction=direction, max_shift=1.0),
                                               np.random.default_rng(seed))
        if placed:
            new = out_ann.shapes[-1].bounds()
            old = ann.shapes[0].bounds()
            assert new[keep] == pytest.approx(old[keep], abs=1.0)  # y for horizontal, x for vertical


def test_paste_onto_ok_image_with_transform():
    src = blob_image([(60, 50)])
    patches = extract_patches(src, ImageAnnotations(image="a.png", width=W, height=H, shapes=[octagon(60, 50)]))
    ok = np.full((H, W, 3), BG, np.uint8)
    target = ImageAnnotations(image="ok.png", width=W, height=H)
    for seed in range(5):
        out, out_ann = paste_patches(ok, target, patches, np.random.default_rng(seed), 1,
                                     transform=cfg(direction="horizontal", max_angle=45.0, scale_max=1.3))
        for s in out_ann.shapes:
            assert inside_mean(out, s) < 120
            cy = (s.bounds()[1] + s.bounds()[3]) / 2
            assert cy == pytest.approx(50, abs=1.5)  # same row as in its own image


def make_project(tmp_path, polygon=True):
    src = tmp_path / "원본"
    src.mkdir(parents=True)
    for i in range(6):
        imwrite_unicode(src / f"d{i}.png", blob_image([(50 + i * 20, 60)]))
    for i in range(3):
        imwrite_unicode(src / f"ok{i}.png", np.full((H, W, 3), BG, np.uint8))
    p = Project.create(tmp_path / "변형 프로젝트")
    p.add_class("dent")
    p.add_images([src])
    for e in p.config.images:
        if e.path.startswith("ok"):
            p.set_ok(e.path)
            continue
        i = int(e.path[1])
        ann = p.load_labels(e)
        ann.shapes.append(octagon(50 + i * 20, 60) if polygon else BBox(class_id=0, x1=38 + i * 20, y1=48,
                                                                          x2=62 + i * 20, y2=72))
        p.save_labels(ann)
    p.config.augment = AugmentConfig(multiplier=3, transforms={},
                                     defect_transform=cfg(max_angle=90.0, scale_min=0.8, scale_max=1.2))
    p.save()
    return p


def label_files(folder):
    return sorted(folder.glob("*.txt"))


def test_full_build_transforms_train_defects_and_ok_images(tmp_path):
    p = make_project(tmp_path)
    report = build_full_dataset(p)
    assert report.defect_transformed > 0
    out = p.dataset_dir / "full"
    # augmented OK images got defects; every label lies on dark pixels
    ok_aug = [f for f in label_files(out / "labels" / "train") if f.stem.startswith("ok") and "_aug" in f.stem]
    assert ok_aug and any(f.read_text(encoding="utf-8").strip() for f in ok_aug)
    for f in label_files(out / "labels" / "train"):
        img = imread_unicode(out / "images" / "train" / f"{f.stem}.png")
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                _, cx, cy, w, h = map(float, line.split())
                box = BBox(class_id=0, x1=(cx - w / 4) * W, y1=(cy - h / 4) * H, x2=(cx + w / 4) * W, y2=(cy + h / 4) * H)
                assert inside_mean(img, box) < 120, f.name  # centre half of each box is defect
    info = yaml.safe_load((out / "build_info.yaml").read_text(encoding="utf-8"))
    assert info["augment"]["defect_transform"]["enabled"] is True
    assert not any("_aug" in f.name for f in (out / "images" / "val").iterdir())  # Val untouched


def test_disabled_transform_leaves_builds_unchanged(tmp_path):
    a = make_project(tmp_path / "a")
    a.config.augment.defect_transform = DefectTransformSetting()  # off, defaults
    a.save()
    b = make_project(tmp_path / "b")
    b.config.augment = AugmentConfig(multiplier=3, transforms={})  # project from before the feature
    b.save()
    build_full_dataset(a)
    build_full_dataset(b)
    fa = sorted((a.dataset_dir / "full" / "labels" / "train").glob("*.txt"))
    fb = sorted((b.dataset_dir / "full" / "labels" / "train").glob("*.txt"))
    assert [f.read_text() for f in fa] == [f.read_text() for f in fb]


def test_crop_build_with_transform(tmp_path):
    p = make_project(tmp_path, polygon=False)
    p.config.crop = p.config.crop.model_copy(update={"crop_size": 96, "multiplier": 3})
    p.save()
    report = build_crop_dataset(p)
    assert report.defect_transformed > 0


def test_preview_applies_transform():
    img = blob_image([(60, 50)])
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[octagon(60, 50)])
    config = AugmentConfig(multiplier=1, transforms={}, defect_transform=cfg(max_shift=1.0))
    previews = augment_previews(img, ann, config, channels=3, colors=["#ff0000"], n=4, seed=1)
    assert len(previews) == 4
    plain = augment_previews(img, ann, AugmentConfig(multiplier=1, transforms={}), channels=3,
                             colors=["#ff0000"], n=1, seed=1)[0]
    assert any(not np.array_equal(p, plain) for p in previews)


def test_augment_panel_round_trip(tmp_path):
    from defect_inspection.ui.panels.augment_panel import AugmentPanel

    QApplication.instance() or QApplication([])
    p = make_project(tmp_path)
    panel = AugmentPanel(tmp_path / "presets")
    panel.set_project(p)
    assert panel.dt_box.isChecked() and panel.dt_angle.value() == 90
    panel.dt_mode.setCurrentIndex(panel.dt_mode.findData("move"))
    panel.dt_direction.setCurrentIndex(panel.dt_direction.findData("vertical"))
    panel.dt_shift.setValue(30)
    dt = p.config.augment.defect_transform
    assert (dt.mode, dt.direction, dt.max_shift, dt.scale_min) == ("move", "vertical", 0.3, 0.8)
    panel.dt_box.setChecked(False)
    assert not p.config.augment.defect_transform.enabled
