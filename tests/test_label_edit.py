import pytest

from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.label_edit import (
    bbox_from_drag,
    clamp_translation,
    move_vertex,
    polygon_from_clicks,
    translate,
    with_class,
)

W, H = 100, 80
BOX = BBox(class_id=0, x1=10, y1=10, x2=30, y2=40)
TRI = Polygon(class_id=1, points=[(10, 10), (50, 10), (30, 50)])


def test_translate_keeps_id_and_does_not_mutate():
    moved = translate(BOX, 5, 7, W, H)
    assert moved.id == BOX.id
    assert moved.bounds() == (15, 17, 35, 47)
    assert BOX.bounds() == (10, 10, 30, 40)


def test_translate_clamped_to_image():
    assert translate(BOX, -50, 100, W, H).bounds() == (0, 50, 20, 80)
    assert translate(TRI, 1000, 0, W, H).bounds()[2] == W


def test_clamp_translation():
    assert clamp_translation(BOX, 200, -200, W, H) == (70, -10)


@pytest.mark.parametrize("index, point, expected", [
    (0, (5, 5), (5, 5, 30, 40)),      # top-left
    (1, (50, 0), (10, 0, 50, 40)),    # top-right
    (2, (60, 70), (10, 10, 60, 70)),  # bottom-right
    (3, (0, 60), (0, 10, 30, 60)),    # bottom-left
    (2, (0, 0), (0, 0, 10, 10)),      # dragged past the opposite corner -> flips
])
def test_move_box_corner(index, point, expected):
    assert move_vertex(BOX, index, *point, W, H).bounds() == expected


def test_move_vertex_clamps_to_image():
    assert move_vertex(BOX, 2, 500, 500, W, H).bounds() == (10, 10, W, H)


def test_move_polygon_vertex():
    moved = move_vertex(TRI, 2, 30, 70, W, H)
    assert moved.points == [(10, 10), (50, 10), (30, 70)]
    assert TRI.points[2] == (30, 50)


def test_degenerate_vertex_move_raises():
    with pytest.raises(ValueError):
        move_vertex(BOX, 2, 10, 60, W, H)  # zero width
    with pytest.raises(ValueError):
        move_vertex(TRI, 2, 30, 10, W, H)  # collinear


def test_with_class_keeps_suggestion_state():
    s = BOX.model_copy(update={"status": "suggested", "source": "model", "score": 0.7})
    c = with_class(s, 3)
    assert (c.class_id, c.status, c.score) == (3, "suggested", 0.7)


def test_bbox_from_drag():
    assert bbox_from_drag(2, 50, 60, 10, 20, W, H).bounds() == (10, 20, 50, 60)
    assert bbox_from_drag(0, -10, -10, 500, 500, W, H).bounds() == (0, 0, W, H)
    assert bbox_from_drag(0, 10, 10, 11, 30, W, H) is None  # narrower than 2 px


def test_polygon_from_clicks():
    p = polygon_from_clicks(1, [(10, 10), (10, 10), (50, 10), (30, 50), (30, 50)], W, H)
    assert p.points == [(10, 10), (50, 10), (30, 50)]  # double-click duplicates removed
    assert polygon_from_clicks(1, [(10, 10), (50, 10)], W, H) is None
    assert polygon_from_clicks(1, [(0, 0), (10, 10), (20, 20)], W, H) is None  # collinear
    assert polygon_from_clicks(1, [(-5, -5), (500, 0), (0, 500)], W, H).bounds() == (0, 0, W, H)
