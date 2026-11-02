"""Defect transform direction "edge" (테두리선 따라): a defect is only placed where its surroundings
look like where it came from, so a defect on an edge line moves along that line."""

import cv2
import numpy as np
import pytest

from defect_inspection.core.annotation import ImageAnnotations, Polygon
from defect_inspection.core.augment_config import DefectTransformSetting
from defect_inspection.core.defect_patches import extract_patches, paste_patches, patch_for, transform_defects
from defect_inspection.core.edge_place import context_for, match_centers

W, H = 400, 300
BG, LINE, DEFECT = 160, 40, 90


def noisy(img, seed=0):
    noise = np.random.default_rng(seed).normal(0, 2, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def line_image(p0, p1, thickness=3, w=W, h=H):
    img = np.full((h, w), BG, np.float64)
    cv2.line(img, p0, p1, LINE, thickness)
    return img


def blob(cx, cy, r=8):
    return Polygon(class_id=0, points=[(cx + r * np.cos(a), cy + r * np.sin(a))
                                        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)])


def with_defect(img, shape):
    pts = np.round((np.array(shape.to_points(), np.float32) - 0.5) * 16).astype(np.int32)
    out = img.copy()
    cv2.fillPoly(out, [pts], DEFECT, shift=4)
    return noisy(out)


def ann(*shapes, w=W, h=H):
    return ImageAnnotations(image="a.png", width=w, height=h, shapes=list(shapes))


def cfg(**kw):
    base = dict(enabled=True, p=1.0, scale_min=1.0, scale_max=1.0, max_angle=0.0, max_shift=1.0,
                direction="edge", mode="add")
    base.update(kw)
    return DefectTransformSetting(**base)


def centre(shape):
    x1, y1, x2, y2 = shape.bounds()
    return (x1 + x2) / 2, (y1 + y2) / 2


def placed_centres(image, shapes, setting, seeds=range(8)):
    out = []
    for seed in seeds:
        _, labels, placed = transform_defects(image, ann(*shapes), setting, np.random.default_rng(seed))
        if placed:
            out.append(centre(labels.shapes[-1]))
    return out


def test_setting_accepts_edge():
    assert DefectTransformSetting(direction="edge").direction == "edge"


def test_moves_along_a_horizontal_line():
    d = blob(60, 100)
    img = with_defect(line_image((0, 100), (W, 100)), d)
    got = placed_centres(img, [d], cfg())
    assert len(got) >= 6
    assert all(abs(y - 100) <= 1.5 for _, y in got)
    assert max(abs(x - 60) for x, _ in got) > 40  # it really moves


def test_moves_along_a_vertical_line():
    d = blob(200, 50)
    img = with_defect(line_image((200, 0), (200, H)), d)
    got = placed_centres(img, [d], cfg())
    assert got and all(abs(x - 200) <= 1.5 for x, _ in got)


def test_follows_a_tilted_line():
    a, b = (0, 60), (W, 60 + int(W * np.tan(np.radians(20))))
    d = blob(100, 60 + 100 * np.tan(np.radians(20)))
    img = with_defect(line_image(a, b), d)
    got = placed_centres(img, [d], cfg())
    assert got
    ux, uy = np.subtract(b, a) / np.hypot(*np.subtract(b, a))
    for x, y in got:  # distance from the line
        assert abs((x - a[0]) * uy - (y - a[1]) * ux) <= 2.0


def test_tall_defect_across_an_edge_does_not_slide_across_it():
    """Real sample: a tall defect across a dark/light edge in a mostly flat template slid 34 px
    across the edge with intensity matching; gradient matching keeps it on the edge."""
    img = np.full((H, W), 160.0)
    img[:150] = 50.0
    cv2.line(img, (0, 200), (W, 200), 120, 1)
    d = Polygon(class_id=0, points=[(95, 110), (105, 110), (105, 190), (95, 190)])
    img = with_defect(img, d)
    got = placed_centres(img, [d], cfg(mode="move"))
    assert len(got) >= 6 and all(abs(y - 150) <= 1.5 for _, y in got)
    assert max(abs(x - 100) for x, _ in got) > 40


def test_stays_on_a_line_that_ends():
    d = blob(40, 100)
    img = with_defect(line_image((0, 100), (220, 100)), d)
    got = placed_centres(img, [d], cfg())
    assert got and all(x <= 220 for x, _ in got)


def test_max_shift_limits_the_distance():
    d = blob(200, 100)
    img = with_defect(line_image((0, 100), (W, 100)), d)
    got = placed_centres(img, [d], cfg(max_shift=0.1))
    assert got and all(abs(x - 200) <= 0.1 * W + 1 for x, _ in got)


def test_unique_surroundings_are_not_moved():
    rng = np.random.default_rng(5)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W)).astype(np.uint8), (0, 0), 3)
    d = blob(150, 150)
    img = with_defect(img.astype(np.float64), d)
    out, labels, placed = transform_defects(img, ann(d), cfg(), np.random.default_rng(0))
    assert placed == 0 and labels.shapes == [d] and np.array_equal(out, img)


def test_context_only_when_asked():
    d = blob(60, 100)
    img = with_defect(line_image((0, 100), (W, 100)), d)
    assert patch_for(img, d).context is None
    assert patch_for(img, d, context=True).context is not None
    assert all(p.context is not None for p in extract_patches(img, ann(d), context=True))


def test_context_leaves_the_defect_out_of_the_score():
    d = blob(60, 100)
    img = with_defect(line_image((0, 100), (W, 100)), d)
    ctx = context_for(img, d)
    cx, cy = (round(v) for v in ctx.center)
    assert ctx.mask[cy, cx] == 0 and ctx.mask[cy - 11, cx] == 0  # defect (r 8) grown by 3
    assert ctx.mask[0, 0] == 255 and ctx.mask[cy, 0] == 255  # surroundings, the line too
    assert not ctx.flat


def test_a_large_defect_matches_an_intact_line():
    """An inpainted 40 px blob left a gap in a 3 px line and scored 0.82 against the intact line."""
    d = blob(100, 150, r=20)
    img = with_defect(line_image((0, 150), (W, 150)), d)
    got = placed_centres(img, [d], cfg(), seeds=range(4))
    assert len(got) == 4 and all(abs(y - 150) <= 1.5 for _, y in got)


def test_onto_another_image_lands_on_its_line():
    d = blob(60, 100)
    src = with_defect(line_image((0, 100), (W, 100)), d)
    patches = extract_patches(src, ann(d), context=True)
    ok = noisy(line_image((0, 220), (W, 220)))
    for seed in range(5):
        _, labels = paste_patches(ok, ann(), patches, np.random.default_rng(seed), 1, transform=cfg())
        assert len(labels.shapes) == 1 and abs(centre(labels.shapes[0])[1] - 220) <= 1.5


def test_colour_images_work():
    d = blob(60, 100)
    img = cv2.cvtColor(with_defect(line_image((0, 100), (W, 100)), d), cv2.COLOR_GRAY2BGR)
    got = placed_centres(img, [d], cfg(), seeds=range(3))
    assert got and all(abs(y - 100) <= 1.5 for _, y in got)


def test_large_images_are_matched_downscaled_then_refined():
    w, h = 3000, 1200
    d = blob(300, 700, r=20)
    img = with_defect(line_image((0, 700), (w, 700), w=w, h=h), d)
    centres = match_centers(img, context_for(img, d))
    assert len(centres) and np.all(np.abs(centres[:, 1] - 700) <= 6)  # coarse
    _, labels, placed = transform_defects(img, ann(d, w=w, h=h), cfg(), np.random.default_rng(1))
    assert placed and abs(centre(labels.shapes[-1])[1] - 700) <= 1.5  # refined at full size


@pytest.mark.parametrize("mode", ["add", "move"])
def test_label_follows_the_pixels(mode):
    d = blob(60, 100)
    img = with_defect(line_image((0, 100), (W, 100)), d)
    out, labels, placed = transform_defects(img, ann(d), cfg(mode=mode), np.random.default_rng(2))
    assert placed
    new = labels.shapes[-1]
    mask = np.zeros((H, W), np.uint8)
    pts = np.round((np.array(new.to_points(), np.float32) - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, shift=4)
    assert abs(float(out[mask > 0].mean()) - DEFECT) < 12


# ---- through the dataset builds and the Augment tab ------------------------------------------------

def make_line_project(tmp_path):
    from defect_inspection.core.augment_config import AugmentConfig
    from defect_inspection.core.project import Project
    from defect_inspection.platform_utils import imwrite_unicode

    src = tmp_path / "원본"
    src.mkdir(parents=True)
    defects = {}
    for i in range(6):
        d = blob(60 + i * 40, 100)
        imwrite_unicode(src / f"d{i}.png", with_defect(line_image((0, 100), (W, 100)), d))
        defects[f"d{i}.png"] = d
    for i in range(3):
        imwrite_unicode(src / f"ok{i}.png", noisy(line_image((0, 180), (W, 180)), seed=i))
    p = Project.create(tmp_path / "테두리 프로젝트")
    p.add_class("dent")
    p.add_images([src])
    for e in p.config.images:
        if e.path.startswith("ok"):
            p.set_ok(e.path)
            continue
        a = p.load_labels(e)
        a.shapes.append(defects[e.path])
        p.save_labels(a)
    p.config.augment = AugmentConfig(multiplier=4, transforms={}, defect_transform=cfg(on_ok_images=True))
    p.save()
    return p


def yolo_centres(folder, prefix):
    out = []
    for f in sorted(folder.glob(f"{prefix}*_aug*.txt")):
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                _, cx, cy, _, _ = map(float, line.split())
                out.append((cx * W, cy * H))
    return out


def test_full_build_places_along_the_lines(tmp_path):
    from defect_inspection.core.dataset import build_full_dataset

    p = make_line_project(tmp_path)
    report = build_full_dataset(p)
    assert report.defect_transformed > 0
    labels = p.dataset_dir / "full" / "labels" / "train"
    on_defect_images = yolo_centres(labels, "d")
    on_ok_images = yolo_centres(labels, "ok")
    assert on_defect_images and all(abs(y - 100) <= 1.5 for _, y in on_defect_images)
    assert on_ok_images and all(abs(y - 180) <= 1.5 for _, y in on_ok_images)


def test_augment_panel_offers_edge(tmp_path):
    from PySide6.QtWidgets import QApplication

    from defect_inspection.ui.panels.augment_panel import AugmentPanel

    QApplication.instance() or QApplication([])
    p = make_line_project(tmp_path)
    panel = AugmentPanel(tmp_path / "presets")
    panel.set_project(p)
    assert panel.dt_direction.currentData() == "edge"
    panel.dt_direction.setCurrentIndex(panel.dt_direction.findData("both"))
    assert p.config.augment.defect_transform.direction == "both"
    panel.dt_direction.setCurrentIndex(panel.dt_direction.findData("edge"))
    assert p.config.augment.defect_transform.direction == "edge"
