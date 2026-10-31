"""Brush tool logic (Qt-free): strokes become label masks, painted onto the same-class label they touch
or erased from any label; the result is a Mask label (pixels as painted, holes kept)."""

from __future__ import annotations

import numpy as np
import pytest

from defect_inspection.core import brush
from defect_inspection.core.annotation import BBox, Mask, Polygon

W, H = 200, 150


def area(shape) -> int:
    return int(brush.shape_mask(shape, W, H).sum())


def line(x1, y1, x2, y2, size=10):
    return brush.stroke_mask([(x1, y1), (x2, y2)], size, W, H)


def square(x1, y1, x2, y2, class_id=0, **kw):
    return Polygon(class_id=class_id, points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)], **kw)


# ---- masks ------------------------------------------------------------------------------------

def test_stroke_mask_covers_the_line_with_round_ends():
    m = line(50, 50, 100, 50, size=10)
    assert m.shape == (H, W) and m.dtype == bool
    assert m[50, 50] and m[50, 100] and m[46, 75] and m[54, 75]
    assert not m[40, 75] and not m[50, 110]


def test_single_click_paints_a_dot():
    m = brush.stroke_mask([(30, 30)], 9, W, H)
    assert m[30, 30] and m[30, 33] and not m[30, 35]
    assert 50 < m.sum() < 90  # about pi * 4.5^2


def test_stroke_is_clipped_to_the_image():
    m = brush.stroke_mask([(-20, 10), (5, 10)], 10, W, H)
    assert m[10, 0] and m.shape == (H, W)


def test_shape_mask_matches_label_coordinates():
    """A box (10,20)-(30,40) covers exactly pixels 10..29 x 20..39."""
    m = brush.shape_mask(BBox(class_id=0, x1=10, y1=20, x2=30, y2=40), W, H)
    assert m.sum() == 400 and m[20, 10] and m[39, 29] and not m[40, 30]
    assert brush.shape_mask(square(10, 20, 30, 40), W, H).sum() == 400


def test_mask_to_shape_keeps_every_pixel():
    m = line(40, 40, 120, 90, size=12)
    shape = brush.mask_to_shape(m, {"class_id": 0})
    assert isinstance(shape, Mask) and (brush.shape_mask(shape, W, H) == m).all()
    ys, xs = np.nonzero(m)
    assert shape.bounds() == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


# ---- painting ---------------------------------------------------------------------------------

def test_paint_on_empty_image_adds_a_mask():
    r = brush.paint([], line(50, 50, 100, 50), class_id=1, width=W, height=H)
    assert r.changed and len(r.shapes) == 1
    (s,) = r.shapes
    assert isinstance(s, Mask) and s.class_id == 1 and s.status == "confirmed" and s.source == "manual"
    assert r.selected == [s.id]


def test_painted_mask_bounds():
    r = brush.paint([], line(50, 50, 100, 50), class_id=0, width=W, height=H)
    (s,) = r.shapes
    assert s.bounds() == pytest.approx((45, 45, 106, 56), abs=1)


def test_paint_grows_the_touched_label_of_the_same_class():
    old = square(20, 20, 60, 60)
    r = brush.paint([old], line(55, 40, 120, 40), class_id=0, width=W, height=H)
    (s,) = r.shapes
    assert isinstance(s, Mask) and s.id == old.id and area(s) > area(old) + 300
    assert s.bounds()[2] >= 124


def test_paint_does_not_merge_other_classes():
    other = square(20, 20, 60, 60, class_id=1)
    r = brush.paint([other], line(55, 40, 120, 40), class_id=0, width=W, height=H)
    assert len(r.shapes) == 2 and r.shapes[0] == other


def test_paint_joins_two_labels_into_one():
    a, b = square(10, 30, 50, 60), square(100, 30, 140, 60)
    r = brush.paint([a, b], line(45, 45, 105, 45), class_id=0, width=W, height=H)
    (s,) = r.shapes
    assert s.id == a.id and s.bounds()[0] <= 10 and s.bounds()[2] >= 140


def test_paint_does_not_merge_suggestions():
    sug = square(20, 20, 60, 60, status="suggested", source="model", score=0.6)
    r = brush.paint([sug], line(55, 40, 120, 40), class_id=0, width=W, height=H)
    assert len(r.shapes) == 2 and r.shapes[0] == sug


def test_paint_inside_a_label_changes_nothing():
    old = square(20, 20, 80, 80)
    r = brush.paint([old], line(40, 50, 60, 50, size=6), class_id=0, width=W, height=H)
    assert not r.changed and r.shapes == [old]


def test_paint_merges_into_a_detect_box():
    old = BBox(class_id=0, x1=20, y1=20, x2=60, y2=60)
    r = brush.paint([old], line(55, 40, 120, 40), class_id=0, width=W, height=H)
    (s,) = r.shapes
    assert s.id == old.id and isinstance(s, Mask) and s.bounds()[0] == 20 and s.bounds()[2] >= 124


def test_empty_stroke_changes_nothing():
    r = brush.paint([], np.zeros((H, W), bool), class_id=0, width=W, height=H)
    assert not r.changed and r.shapes == []


# ---- erasing ----------------------------------------------------------------------------------

def test_erase_cuts_the_label():
    old = square(20, 20, 100, 60)
    r = brush.erase([old], line(80, 0, 80, 149, size=60), width=W, height=H)
    (s,) = r.shapes
    assert r.changed and s.id == old.id and s.bounds()[2] <= 51


def test_erase_through_the_middle_splits_the_label():
    old = square(20, 20, 140, 60, class_id=1)
    r = brush.erase([old], line(80, 0, 80, 149, size=20), width=W, height=H)
    assert len(r.shapes) == 2
    assert {s.class_id for s in r.shapes} == {1}
    assert old.id in [s.id for s in r.shapes]
    assert sorted(s.bounds()[0] for s in r.shapes)[1] >= 89


def test_erase_everything_deletes_the_label():
    old = square(20, 20, 40, 40)
    r = brush.erase([old], line(30, 0, 30, 149, size=60), width=W, height=H)
    assert r.changed and r.shapes == []


def test_erase_works_on_any_class_and_suggestions_and_keeps_their_fields():
    sug = square(20, 20, 100, 60, class_id=1, status="suggested", source="sam", score=0.7)
    r = brush.erase([sug], line(90, 0, 90, 149, size=40), width=W, height=H)
    (s,) = r.shapes
    assert (s.id, s.class_id, s.status, s.source, s.score) == (sug.id, 1, "suggested", "sam", 0.7)


def test_erase_shrinks_a_box():
    old = BBox(class_id=0, x1=20, y1=20, x2=100, y2=60)
    r = brush.erase([old], line(90, 0, 90, 149, size=40), width=W, height=H)
    (s,) = r.shapes
    assert isinstance(s, Mask) and s.bounds() == pytest.approx((20, 20, 70, 60), abs=1)


def test_erase_a_hole_keeps_the_hole():
    old = square(20, 20, 120, 120)
    hole = brush.stroke_mask([(70, 70)], 10, W, H)
    r = brush.erase([old], hole, width=W, height=H)
    (s,) = r.shapes
    assert r.changed and s.id == old.id and isinstance(s, Mask)
    assert not brush.shape_mask(s, W, H)[70, 70] and s.area == area(old) - hole.sum()


def test_erase_leaves_untouched_labels_alone():
    a, b = square(10, 10, 40, 40), square(120, 10, 160, 40)
    r = brush.erase([a, b], line(25, 0, 25, 149, size=60), width=W, height=H)
    assert r.shapes == [b]


def test_tiny_leftovers_are_dropped():
    old = square(20, 20, 60, 60)
    r = brush.erase([old], line(40, 0, 40, 149, size=39), width=W, height=H)
    assert r.shapes == []  # 1 px slivers on either side


# ---- exactness and speed ---------------------------------------------------------------------

def test_thin_stroke_keeps_its_width():
    """A 4 px stroke is a 4 px label, also in its outline (pixel-centre outlines gave 3 px)."""
    r = brush.paint([], line(40, 80, 160, 80, size=4), class_id=0, width=W, height=H)
    (s,) = r.shapes
    x1, y1, x2, y2 = s.bounds()
    assert y2 - y1 == 4 and x2 - x1 == 124
    ys = [y for _, y in s.to_points()]
    assert max(ys) - min(ys) == 4


def test_outline_without_simplification_is_exact():
    m = line(30, 30, 150, 110, size=7) | brush.stroke_mask([(90, 70)], 25, W, H)
    poly = Polygon(class_id=0, points=brush.mask_outline(m, tolerance=0))
    assert (brush.shape_mask(poly, W, H) == m).all()


def test_repainting_does_not_drift():
    r = brush.paint([], line(40, 40, 120, 90, size=12), class_id=0, width=W, height=H)
    first = area(r.shapes[0])
    for x in (60, 80, 100):
        r = brush.paint(r.shapes, brush.stroke_mask([(x, 40 + (x - 40) * 50 / 80)], 4, W, H), class_id=0,
                        width=W, height=H)
    assert abs(area(r.shapes[0]) - first) <= 0.02 * first


def test_stroke_on_a_large_image_with_many_labels_is_fast():
    import time

    w, h = 3072, 1024
    labels = [square(100 + 140 * i, 100, 200 + 140 * i, 300) for i in range(20)]
    stroke = brush.stroke_mask([(x, 500 + 200 * np.sin(x / 300)) for x in range(100, 2900, 10)], 20, w, h)
    start = time.perf_counter()
    brush.paint(labels, stroke, class_id=0, width=w, height=h)
    brush.erase(labels, stroke, width=w, height=h)
    assert time.perf_counter() - start < 0.5


def test_erasing_a_box_corner_keeps_the_l_shape():
    old = BBox(class_id=0, x1=10, y1=10, x2=50, y2=50)
    r = brush.erase([old], brush.stroke_mask([(10, 10)], 10, W, H), width=W, height=H)
    (s,) = r.shapes
    assert r.changed and r.touched and s.bounds() == (10, 10, 50, 50) and s.area < 1600


def test_touched_is_false_when_the_stroke_misses_every_label():
    r = brush.erase([square(10, 10, 30, 30)], line(150, 100, 180, 100), width=W, height=H)
    assert not r.changed and not r.touched


def test_label_partly_outside_the_image_is_edited_in_place():
    old = square(-20, 20, 40, 60)
    r = brush.paint([old], line(35, 40, 80, 40), class_id=0, width=W, height=H)
    (s,) = r.shapes
    assert s.id == old.id and s.bounds()[0] == 0 and s.bounds()[2] >= 84


def test_erase_across_many_labels_on_a_12_mp_image_is_fast():
    """Review fix: an erase across 14 labels took 3.5 s on 4000x3000 with whole-image masks."""
    import time

    w, h = 4000, 3000
    labels = [square(200 + 250 * i, 1300, 400 + 250 * i, 1700) for i in range(14)]
    stroke = brush.stroke_mask([(x, 1500.0) for x in range(100, 3900, 8)], 30, w, h)
    start = time.perf_counter()
    r = brush.erase(labels, stroke, width=w, height=h)
    assert time.perf_counter() - start < 0.6
    assert len(r.shapes) == 28  # every label cut in two
