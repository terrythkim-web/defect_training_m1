"""Mask labels (PRD F2.9): brush-painted pixel masks stored as they are (holes kept), next to BBox/Polygon."""

from __future__ import annotations

import json

import numpy as np
import pytest

from defect_inspection.core import mask_codec
from defect_inspection.core.annotation import (
    BBox,
    ImageAnnotations,
    Mask,
    Polygon,
    fit_shape,
    load_annotations,
    save_annotations,
    shape_to_yolo,
)
from defect_inspection.core.label_edit import translate

W, H = 120, 80


def ring_mask() -> np.ndarray:
    """A 40x30 block at (20, 10) with a 10x10 hole."""
    m = np.zeros((H, W), bool)
    m[10:40, 20:60] = True
    m[20:30, 35:45] = False
    return m


# ---- codec ------------------------------------------------------------------------------------

def test_rle_round_trip():
    rng = np.random.default_rng(1)
    m = rng.random((17, 23)) > 0.6
    rle = mask_codec.rle_encode(m)
    assert sum(rle) == m.size and rle[0] >= 0
    assert (mask_codec.rle_decode(rle, 23, 17) == m).all()
    assert mask_codec.rle_encode(np.ones((2, 3), bool)) == [0, 6]


def test_edge_rings_include_holes():
    rings = mask_codec.edge_rings(ring_mask())
    assert len(rings) == 2
    xs = sorted(min(x for x, _ in r) for r in rings)
    assert xs == [20, 35]  # outer ring and hole, on pixel edges


# ---- Mask shape -------------------------------------------------------------------------------

def test_from_pixels_crops_to_the_tight_box_and_keeps_holes():
    m = Mask.from_pixels(ring_mask(), class_id=1)
    assert (m.x, m.y, m.w, m.h) == (20, 10, 40, 30)
    assert m.bounds() == (20.0, 10.0, 60.0, 40.0)
    assert m.area == 40 * 30 - 100
    assert (m.full_pixels(W, H) == ring_mask()).all()
    assert Mask.from_pixels(np.zeros((H, W), bool), class_id=0) is None


def test_from_pixels_with_an_origin():
    local = np.ones((5, 4), bool)
    m = Mask.from_pixels(local, class_id=0, origin=(100, 70))
    assert m.bounds() == (100.0, 70.0, 104.0, 75.0)


def test_outline_is_the_outer_edge():
    m = Mask.from_pixels(ring_mask(), class_id=0)
    pts = m.to_points()
    assert min(p[0] for p in pts) == 20 and max(p[0] for p in pts) == 60
    assert min(p[1] for p in pts) == 10 and max(p[1] for p in pts) == 40


def test_validation():
    with pytest.raises(ValueError):
        Mask(class_id=0, x=0, y=0, w=2, h=2, rle=[0, 3])  # sums to 3, box has 4 pixels
    with pytest.raises(ValueError):
        Mask(class_id=0, x=0, y=0, w=2, h=2, rle=[4])  # nothing painted
    with pytest.raises(ValueError):
        Mask(class_id=0, x=0, y=0, w=2, h=2, rle=[2, -1, 3])


def test_json_round_trip(tmp_path):
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[
        Mask.from_pixels(ring_mask(), class_id=1),
        BBox(class_id=0, x1=1, y1=1, x2=5, y2=5),
    ])
    save_annotations(tmp_path / "a.json", ann)
    raw = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert raw["shapes"][0]["type"] == "mask" and "rle" in raw["shapes"][0]
    again = load_annotations(tmp_path / "a.json")
    assert isinstance(again.shapes[0], Mask)
    assert (again.shapes[0].full_pixels(W, H) == ring_mask()).all()


def test_yolo_lines_use_the_outline_or_box():
    m = Mask.from_pixels(ring_mask(), class_id=1)
    det = shape_to_yolo(m, W, H, "detect").split()
    assert det[0] == "1" and float(det[3]) == pytest.approx(40 / W)
    seg = shape_to_yolo(m, W, H, "seg").split()
    assert seg[0] == "1" and len(seg) >= 9


def test_translate_moves_by_whole_pixels_inside_the_image():
    m = Mask.from_pixels(ring_mask(), class_id=0)
    moved = translate(m, 10.4, -3.6, W, H)
    assert (moved.x, moved.y) == (30, 6) and moved.id == m.id and moved.rle == m.rle
    edge = translate(m, 500, 0, W, H)
    assert edge.bounds()[2] == W


def test_fit_shape_clips_a_mask():
    m = Mask.from_pixels(ring_mask(), class_id=0)
    shifted = m.model_copy(update={"x": 100})  # 40 wide at x=100: 20 px outside a 120 px image
    fitted = fit_shape(shifted, shifted.to_points(), W, H, min_visibility=0.3, min_area_px=1)
    assert isinstance(fitted, Mask) and fitted.bounds()[2] == W and fitted.id == m.id
    assert fit_shape(shifted, shifted.to_points(), W, H, min_visibility=0.9, min_area_px=1) is None


def test_mask_cache_follows_copies():
    """A copy with other pixels must not reuse the original's cached pixels or outline."""
    a = Mask.from_pixels(ring_mask(), class_id=0)
    _ = a.to_points()
    small = np.zeros((H, W), bool)
    small[0:4, 0:4] = True
    b_src = Mask.from_pixels(small, class_id=0)
    b = a.model_copy(update={"w": b_src.w, "h": b_src.h, "rle": b_src.rle, "x": 0, "y": 0})
    assert b.area == 16 and b.bounds() == (0.0, 0.0, 4.0, 4.0) and max(p[0] for p in b.to_points()) == 4


def test_polygon_and_box_unchanged():
    p = Polygon(class_id=0, points=[(0, 0), (4, 0), (4, 4)])
    assert p.type == "polygon" and BBox(class_id=0, x1=0, y1=0, x2=1, y2=1).type == "bbox"


# ---- preview drawing ---------------------------------------------------------------------------

def test_preview_draws_the_hole_and_scales_masks():
    from defect_inspection.core.preview import draw_comparison, outline_rings, scale_for_preview

    m = Mask.from_pixels(ring_mask(), class_id=0)
    assert len(outline_rings(m)) == 2
    img = draw_comparison(np.full((H, W), 200, np.uint8), [m], [])
    assert (img[25, 35] != 200).any()  # the hole's edge is drawn
    big = np.full((800, 1200), 100, np.uint8)
    ann = ImageAnnotations(image="a.png", width=1200, height=800,
                           shapes=[Mask.from_pixels(np.kron(ring_mask(), np.ones((10, 10), bool)), class_id=0)])
    small, sann = scale_for_preview(big, ann, max_side=120)
    (s,) = sann.shapes
    assert small.shape == (80, 120) and isinstance(s, Mask)
    assert (s.full_pixels(120, 80) == ring_mask()).all()


def test_shape_iou_uses_pixels_only_between_masks():
    from defect_inspection.core.evaluate import shape_iou

    a = Mask.from_pixels(ring_mask(), class_id=0)
    box = BBox(class_id=0, x1=20, y1=10, x2=60, y2=40)
    assert shape_iou(a, box) == 1.0  # a detect box against a brush label: box IoU
    full = Mask.from_pixels(np.pad(np.ones((30, 40), bool), ((10, H - 40), (20, W - 60))), class_id=0)
    assert shape_iou(a, full) == pytest.approx((1200 - 100) / 1200)
    assert shape_iou(a, a) == 1.0
    assert shape_iou(box, BBox(class_id=0, x1=40, y1=10, x2=80, y2=40)) == pytest.approx(1 / 3)
    assert shape_iou(a, BBox(class_id=0, x1=100, y1=50, x2=110, y2=60)) == 0.0


# ---- step 2: masks through augmentation, patches, crops and the dataset build ------------------------

def painted(mask: np.ndarray) -> np.ndarray:
    """Grey image whose defect pixels (the mask) are white: lets a test check labels follow pixels."""
    img = np.full(mask.shape, 60, np.uint8)
    img[mask] = 255
    return img


def _geometric():
    from test_augment import GEOMETRIC

    return GEOMETRIC


@pytest.mark.parametrize("key", _geometric())
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_augmented_mask_follows_the_pixels(key, seed):
    """Like test_augment.test_labels_follow_pixels, for Mask labels (holes included)."""
    from defect_inspection.core.augment import Augmenter
    from defect_inspection.core.augment_config import AugmentConfig, TransformSetting

    m = Mask.from_pixels(ring_mask(), class_id=0)
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[m])
    cfg = AugmentConfig(transforms={key: TransformSetting(enabled=True, p=1.0, strength=0.6)},
                        min_visibility=0.0, min_area_px=0)
    r = Augmenter(cfg, channels=1, seed=seed)(painted(ring_mask()), ann)
    bright = r.image > 157
    if not r.annotations.shapes:
        assert bright.sum() == 0  # moved fully off-image
        return
    (s,) = r.annotations.shapes
    assert isinstance(s, Mask) and s.id == m.id
    lab = s.full_pixels(r.annotations.width, r.annotations.height)
    assert (bright & lab).sum() / (bright | lab).sum() > 0.9


def test_mask_moved_mostly_out_of_view_is_dropped():
    from defect_inspection.core.augment import Augmenter
    from defect_inspection.core.augment_config import AugmentConfig

    m = Mask.from_pixels(ring_mask(), class_id=0)
    aug = Augmenter(AugmentConfig(min_visibility=0.5), channels=1, seed=0)
    shifted = m.moved(70, 0)  # 40 wide at x=90 in a 120 px image: 30/40 visible
    r = aug(painted(shifted.full_pixels(W, H)), ImageAnnotations(image="a.png", width=W, height=H, shapes=[shifted]))
    assert len(r.annotations.shapes) == 1  # no transform enabled: the visible part stays
    aug9 = Augmenter(AugmentConfig(min_visibility=0.95), channels=1, seed=0)
    clipped = Mask(class_id=0, x=90, y=10, w=30, h=30, rle=Mask.from_pixels(ring_mask()[10:40, 20:50], class_id=0).rle)
    r9 = aug9(painted(clipped.full_pixels(W, H)), ImageAnnotations(image="a.png", width=W, height=H, shapes=[clipped]))
    assert len(r9.annotations.shapes) == 1  # fully visible already


def test_patch_cut_transform_and_blend_keep_mask_pixels():
    from defect_inspection.core.defect_patches import blend, erase, patch_for, transform_patch

    m = Mask.from_pixels(ring_mask(), class_id=1)
    img = painted(ring_mask())
    patch = patch_for(img, m, max_fraction=0.9)
    assert isinstance(patch.shape, Mask) and patch.mask[15, 20] < 0.5  # the hole is not cut out
    turned = transform_patch(patch, 1.0, 90)
    assert isinstance(turned.shape, Mask) and turned.shape.area == m.area
    target = np.full((H, W), 60, np.uint8)
    label = blend(target, turned, 5, 5)
    assert isinstance(label, Mask) and label.class_id == 1 and label.bounds()[0] == 5
    cleaned = erase(img, m)
    assert cleaned[12, 22] < 200  # the defect is painted over


def test_crop_keeps_a_mask_and_clips_it():
    from defect_inspection.core.crop import CropWindow, crop_annotations

    m = Mask.from_pixels(ring_mask(), class_id=0)
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[m])
    out, dropped = crop_annotations(ann, CropWindow(30, 0, 94, 64, "defect"), 0.3, 1)
    (s,) = out.shapes
    assert dropped == 0 and isinstance(s, Mask) and s.bounds() == (0.0, 10.0, 30.0, 40.0)
    assert (s.full_pixels(64, 64)[10:40, 0:30] == ring_mask()[10:40, 30:60]).all()


def test_label_map_paints_every_shape_type():
    from defect_inspection.core.dataset import label_map

    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[
        Mask.from_pixels(ring_mask(), class_id=1),
        BBox(class_id=0, x1=100, y1=60, x2=110, y2=70),
        BBox(class_id=0, x1=0, y1=0, x2=5, y2=5, status="suggested", source="model", score=0.5),
    ])
    lm = label_map(ann)
    assert lm.dtype == np.uint8 and lm[15, 25] == 2 and lm[25, 40] == 0  # hole stays background
    assert (lm[60:70, 100:110] == 1).all() and lm[2, 2] == 0  # suggestions are not training data


def test_mask_in_pieces_gives_one_seg_line_per_piece():
    """Review fix: a thin crack broken by a warp must not lose all but its largest piece in YOLO seg."""
    m = np.zeros((H, W), bool)
    m[10:25, 10:25] = True
    m[50:60, 80:90] = True
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[Mask.from_pixels(m, class_id=0)])
    assert len(ann.to_yolo_lines("seg")) == 2
    (det,) = ann.to_yolo_lines("detect")
    assert float(det.split()[3]) == pytest.approx(80 / W)


def test_cached_pixels_are_read_only():
    m = Mask.from_pixels(ring_mask(), class_id=0)
    with pytest.raises(ValueError):
        m.pixels()[0, 0] = False
