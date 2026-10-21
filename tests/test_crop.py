"""Crop dataset (PROMPTS step 17, PRD F7.2-F7.5)."""

from pathlib import PurePosixPath

import cv2
import numpy as np
import pytest
import yaml

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon, shapes_from_yolo
from defect_inspection.core.crop import (
    CropWindow,
    background_windows,
    build_crop_dataset,
    crop_annotations,
    defect_windows,
    image_rng,
    plan_windows,
    read_build_info,
    tile_starts,
    tile_windows,
    to_crop,
    to_original,
)
from defect_inspection.core.crop_config import CROP_PRESETS, CropConfig
from defect_inspection.core.dataset import BuildCancelled, DatasetBuildError
from defect_inspection.core.project import Project
from defect_inspection.core.roi import Roi, RoiStore
from defect_inspection.platform_utils import imread_unicode, imwrite_unicode


def ann(*shapes, w=1000, h=800):
    return ImageAnnotations(image="x.png", width=w, height=h, shapes=list(shapes))


# ---- coordinates and clipping (F7.3) ---------------------------------------------------------

@pytest.mark.parametrize("shape", [
    BBox(class_id=0, x1=120.5, y1=40.25, x2=180, y2=95),
    Polygon(class_id=1, points=[(110, 50), (150, 30.5), (190, 70), (130, 99)]),
])
def test_round_trip(shape):
    win = CropWindow(100, 20, 300, 220, "defect")
    local = to_crop(shape, win)
    back = to_original(local, win)
    assert back.id == shape.id
    assert np.allclose(back.to_points(), shape.to_points())
    assert local.bounds()[0] == pytest.approx(shape.bounds()[0] - 100)


def test_label_inside_kept_exactly():
    box = BBox(class_id=0, x1=150, y1=60, x2=200, y2=100)
    out, dropped = crop_annotations(ann(box), CropWindow(100, 50, 300, 250), 0.4, 0)
    assert dropped == 0 and out.width == 200 and out.height == 200
    assert out.shapes[0].bounds() == (50, 10, 100, 50)


def test_border_label_clipped_and_40_percent_rule():
    win = CropWindow(100, 100, 300, 300)
    half = BBox(class_id=0, x1=80, y1=150, x2=120, y2=170)  # 50 % inside
    third = BBox(class_id=0, x1=70, y1=200, x2=115, y2=220)  # 15/45 = 33 % inside
    outside = BBox(class_id=0, x1=0, y1=0, x2=50, y2=50)
    out, dropped = crop_annotations(ann(half, third, outside), win, 0.4, 0)
    assert [s.bounds() for s in out.shapes] == [(0, 50, 20, 70)]
    assert dropped == 1  # `outside` is simply not in the crop, not "dropped"


def test_polygon_clipped_to_crop():
    tri = Polygon(class_id=0, points=[(50, 150), (250, 150), (150, 250)])
    out, _ = crop_annotations(ann(tri), CropWindow(100, 100, 400, 400), 0.4, 0)
    xs = [p[0] for p in out.shapes[0].points]
    assert min(xs) == pytest.approx(0) and isinstance(out.shapes[0], Polygon)


def test_suggestions_never_cropped():
    s = BBox(class_id=0, x1=150, y1=150, x2=170, y2=170, status="suggested", source="model", score=0.9)
    out, _ = crop_annotations(ann(s), CropWindow(100, 100, 300, 300), 0.4, 0)
    assert out.shapes == []


# ---- windows (F7.2) ---------------------------------------------------------------------------

def test_tile_starts_cover_and_end_at_edge():
    assert tile_starts(1000, 400, 80) == [0, 320, 600]
    assert tile_starts(400, 640, 32) == [0]
    starts = tile_starts(5000, 1280, 32)
    assert starts[-1] == 5000 - 1280 and all(b - a <= 1280 - 32 for a, b in zip(starts, starts[1:]))


def test_tile_windows_cover_every_pixel():
    wins = tile_windows(1000, 700, 300, 60)
    cover = np.zeros((700, 1000), bool)
    for w in wins:
        assert w.width == 300 and w.height == 300
        cover[w.y1:w.y2, w.x1:w.x2] = True
    assert cover.all()


def test_overlap_percent_and_px():
    assert CropConfig(tile_size=640, overlap_percent=20).overlap_pixels() == 128
    assert CropConfig(tile_size=640, overlap_unit="px", overlap_px=32).overlap_pixels() == 32
    assert CropConfig(tile_size=64, overlap_unit="px", overlap_px=500).overlap_pixels() == 63
    att = CROP_PRESETS["ATT-YOLO (1280 타일, overlap 32px)"]
    assert att.use_tiles and att.tile_size == 1280 and att.overlap_pixels() == 32 and not att.use_defect


def test_defect_windows_contain_defect_and_jitter():
    box = BBox(class_id=0, x1=480, y1=380, x2=520, y2=420)
    rng = np.random.default_rng(0)
    wins = defect_windows(ann(box), 1000, 800, 200, 50, 1.0, rng)
    assert len(wins) == 50
    for w in wins:
        assert (w.width, w.height) == (200, 200)
        assert w.x1 <= 480 and w.x2 >= 520 and w.y1 <= 380 and w.y2 >= 420
    assert len({w.x1 for w in wins}) > 10  # not always centred
    centred = defect_windows(ann(box), 1000, 800, 200, 3, 0.0, rng)
    assert {w.key() for w in centred} == {(400, 300, 600, 500)}


def test_defect_window_near_border_stays_inside():
    box = BBox(class_id=0, x1=0, y1=0, x2=10, y2=10)
    for w in defect_windows(ann(box), 1000, 800, 300, 20, 1.0, np.random.default_rng(1)):
        assert w.x1 >= 0 and w.y1 >= 0 and w.x2 <= 1000 and w.y2 <= 800


def test_small_image_window_is_whole_image():
    box = BBox(class_id=0, x1=10, y1=10, x2=20, y2=20)
    (w,) = defect_windows(ann(box, w=100, h=80), 100, 80, 640, 1, 0.5, np.random.default_rng(0))
    assert w.key() == (0, 0, 100, 80)


def test_background_windows_avoid_labels():
    box = BBox(class_id=0, x1=400, y1=300, x2=600, y2=500)
    wins = background_windows(ann(box), 1000, 800, 200, 10, np.random.default_rng(0))
    assert len(wins) == 10
    for w in wins:
        assert w.x2 <= 400 or w.x1 >= 600 or w.y2 <= 300 or w.y1 >= 500
    inside = BBox(class_id=0, x1=50, y1=50, x2=60, y2=60)
    assert background_windows(ann(inside, w=200, h=200), 200, 200, 200, 3, np.random.default_rng(0)) == []


def test_plan_ratio_order_and_reproducible():
    a = ann(BBox(class_id=0, x1=100, y1=100, x2=120, y2=120), BBox(class_id=0, x1=700, y1=500, x2=720, y2=530))
    cfg = CropConfig(crop_size=200, per_defect=2, background_ratio=0.5)
    rois = [CropWindow(0, 0, 50, 50, "roi")]
    wins = plan_windows(cfg, a, 1000, 800, False, rois, image_rng(0, "a.png"))
    kinds = [w.kind for w in wins]
    assert kinds == ["roi"] + ["defect"] * 4 + ["background"] * 2
    again = plan_windows(cfg, a, 1000, 800, False, rois, image_rng(0, "a.png"))
    assert wins == again
    ok = plan_windows(cfg, ann(), 1000, 800, True, [], image_rng(0, "ok.png"))
    assert [w.kind for w in ok] == ["background"]  # OK image: like one label (2 x 0.5)


# ---- dataset build (F7.4, F7.5) ------------------------------------------------------------------

W, H = 400, 300


def make_project(tmp_path, n_defect=6, n_ok=2, task="detect"):
    src = tmp_path / "원본"
    src.mkdir()
    p = Project.create(tmp_path / "Crop 프로젝트", task=task)
    p.add_class("dent")
    rng = np.random.default_rng(0)
    centers = {}
    for i in range(n_defect):
        cx, cy = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        img = np.full((H, W, 3), 200, np.uint8)
        cv2.circle(img, (cx, cy), 10, (40, 40, 40), -1)
        centers[f"d{i}.png"] = (cx, cy)
        imwrite_unicode(src / f"d{i}.png", img)
    for i in range(n_ok):
        imwrite_unicode(src / f"ok{i}.png", np.full((H, W, 3), 200, np.uint8))
    imwrite_unicode(src / "unlabeled.png", np.full((H, W, 3), 200, np.uint8))
    p.add_images([src])
    for name, (cx, cy) in centers.items():
        a = p.load_labels(name)
        if task == "seg":
            a.shapes.append(Polygon(class_id=0, points=[(cx - 10, cy), (cx, cy - 10), (cx + 10, cy), (cx, cy + 10)]))
        else:
            a.shapes.append(BBox(class_id=0, x1=cx - 10, y1=cy - 10, x2=cx + 10, y2=cy + 10))
        p.save_labels(a)
    for i in range(n_ok):
        p.set_ok(f"ok{i}.png")
    p.config.augment.multiplier = 1
    p.config.crop = CropConfig(use_roi=False, crop_size=128, per_defect=2, multiplier=2)
    p.save()
    return p, centers


def files(root, part):
    return sorted(q.name for q in (root / "images" / part).iterdir())


def source_of(name):
    return name.split("__")[0]


def test_build_crop_dataset(tmp_path):
    p, centers = make_project(tmp_path)
    calls = []
    report = build_crop_dataset(p, progress=lambda d, t, m: calls.append((d, t)))
    root = p.dataset_dir / "crop"
    assert (root / "data.yaml").is_file() and not (p.dataset_dir / "crop.building").exists()
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert data["channels"] == 3 and data["names"] == {0: "dent"}

    train, val = files(root, "train"), files(root, "val")
    # leakage: one original image never appears on both sides
    assert not ({source_of(n) for n in train} & {source_of(n) for n in val})
    assert not any("_aug" in n for n in val)  # Val is never augmented
    assert any("_aug" in n for n in train)
    assert report.train_crops + report.val_crops == sum(report.by_kind.values())
    assert report.augmented == report.train_crops * 2
    assert len(train) == report.total_train and len(val) == report.val_crops
    assert "unlabeled" in report.skipped_unlabeled[0]
    assert report.by_kind["defect"] == 12 and report.by_kind["background"] > 0
    assert calls[-1][0] == calls[-1][1]

    for name in val + [n for n in train if "_aug" not in n]:
        img = imread_unicode(root / "images" / ("val" if name in val else "train") / name)
        assert img.shape[:2] == (128, 128)
    info = read_build_info(root)
    assert info["mode"] == "crop" and info["window"] == {"size": 128, "overlap_px": None}


def test_crop_labels_match_pixels(tmp_path):
    """Every defect crop's label covers the dark disk exactly where it is in the crop."""
    p, centers = make_project(tmp_path)
    p.config.crop.multiplier = 0
    build_crop_dataset(p)
    root = p.dataset_dir / "crop"
    checked = 0
    for part in ("train", "val"):
        for img_path in (root / "images" / part).glob("*__defect*.png"):
            img = imread_unicode(img_path)
            lines = (root / "labels" / part / f"{img_path.stem}.txt").read_text(encoding="utf-8").splitlines()
            (box,) = shapes_from_yolo(lines, 128, 128, "detect")
            ys, xs = np.nonzero(img[:, :, 0] < 100)
            assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == pytest.approx(box.bounds(), abs=1.01)
            checked += 1
    assert checked == 12


def test_background_crops_have_no_labels(tmp_path):
    p, _ = make_project(tmp_path)
    p.config.crop.multiplier = 0
    build_crop_dataset(p)
    root = p.dataset_dir / "crop"
    bg = list((root / "labels").rglob("*__background*.txt"))
    assert bg and all(f.read_text(encoding="utf-8") == "" for f in bg)


def test_roi_crops_from_crops_json(tmp_path):
    p, centers = make_project(tmp_path)
    store = RoiStore(p.root)
    store.set_rois("d0.png", [Roi(x1=0.4, y1=0.6, x2=100.2, y2=80)], [Roi(x1=300, y1=200, x2=500, y2=400)])
    p.config.crop = CropConfig(use_defect=True, use_background=False, crop_size=128, per_defect=1, multiplier=0)
    report = build_crop_dataset(p)
    root = p.dataset_dir / "crop"
    roi_files = sorted(f.name for f in (root / "images").rglob("*__roi*.png"))
    assert "d0__roi001.png" in roi_files and "d0__roi002.png" in roi_files
    assert report.by_kind["roi"] == 6 + 2 + 1  # common ROI on all 8 images + d0's own
    d0 = next((root / "images").rglob("d0__roi002.png"))  # own ROI comes after the common one
    assert imread_unicode(d0).shape[:2] == (80, 101)  # outward rounding: (0,0)-(101,80)
    common = next((root / "images").rglob("d1__roi001.png"))
    assert imread_unicode(common).shape[:2] == (100, 100)  # clipped to the 400x300 image


def test_tiles_att_yolo_preset_and_seg(tmp_path):
    p, _ = make_project(tmp_path, task="seg")
    p.config.crop = CropConfig(use_roi=False, use_defect=False, use_background=False, use_tiles=True,
                               tile_size=256, overlap_unit="px", overlap_px=32, multiplier=0)
    report = build_crop_dataset(p)
    assert report.by_kind["tile"] == 8 * 4  # 400x300 -> 2x2 tiles of 256
    root = p.dataset_dir / "crop"
    info = read_build_info(root)
    assert info["window"] == {"size": 256, "overlap_px": 32}
    lines = [ln for f in (root / "labels").rglob("*.txt") for ln in f.read_text(encoding="utf-8").splitlines()]
    assert lines and all(len(ln.split()) >= 7 for ln in lines)  # polygons


def test_errors(tmp_path):
    p, _ = make_project(tmp_path)
    p.config.crop = CropConfig(use_roi=False, use_defect=False, use_background=False, use_tiles=False)
    with pytest.raises(DatasetBuildError, match="하나 이상"):
        build_crop_dataset(p)
    p.config.crop = CropConfig(use_roi=True, use_defect=False, use_background=False)
    with pytest.raises(DatasetBuildError, match="ROI가 없습니다"):
        build_crop_dataset(p)
    RoiStore(p.root).set_rois("d0.png", [], [Roi(x1=0, y1=0, x2=20, y2=20)])  # ROI that misses every defect
    with pytest.raises(DatasetBuildError, match="Val"):
        build_crop_dataset(p)
    assert not (p.dataset_dir / "crop").exists() and not (p.dataset_dir / "crop.building").exists()


def test_cancel_keeps_previous_dataset(tmp_path):
    p, _ = make_project(tmp_path)
    build_crop_dataset(p)
    before = files(p.dataset_dir / "crop", "val")
    count = [0]

    def cancel():
        count[0] += 1
        return count[0] > 5

    with pytest.raises(BuildCancelled):
        build_crop_dataset(p, cancel=cancel)
    assert files(p.dataset_dir / "crop", "val") == before
    assert not (p.dataset_dir / "crop.building").exists()


def test_full_and_crop_coexist(tmp_path):
    from defect_inspection.core.dataset import build_full_dataset

    p, _ = make_project(tmp_path)
    build_full_dataset(p)
    build_crop_dataset(p)
    assert (p.dataset_dir / "full" / "data.yaml").is_file() and (p.dataset_dir / "crop" / "data.yaml").is_file()
    full_val = {PurePosixPath(n).stem for n in files(p.dataset_dir / "full", "val")}
    crop_val = {source_of(n) for n in files(p.dataset_dir / "crop", "val")}
    assert crop_val <= full_val  # same per-image split as the full dataset


def test_crop_config_saved_in_project(tmp_path):
    p, _ = make_project(tmp_path)
    p.config.crop.tile_size = 1280
    p.save()
    assert Project.open(p.root).config.crop.tile_size == 1280
