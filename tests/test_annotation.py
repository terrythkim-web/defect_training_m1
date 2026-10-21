import json

import numpy as np
import pytest
from pydantic import ValidationError

from defect_inspection.core.annotation import (
    BBox,
    ImageAnnotations,
    Polygon,
    load_annotations,
    save_annotations,
    shape_to_yolo,
    shapes_from_yolo,
    write_yolo_txt,
)
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imwrite_unicode

W, H = 640, 480
TRIANGLE = [(100.0, 50.0), (300.0, 120.0), (150.0, 400.0)]


def sample() -> ImageAnnotations:
    return ImageAnnotations(
        image="스크래치 01.png",
        width=W,
        height=H,
        shapes=[
            BBox(class_id=0, x1=10.5, y1=20.25, x2=110.5, y2=220.75),
            Polygon(class_id=1, points=TRIANGLE),
        ],
    )


def values(line: str) -> list[float]:
    return [float(v) for v in line.split()[1:]]


# ---- shapes ------------------------------------------------------------------

def test_bbox_normalizes_drag_direction():
    b = BBox(class_id=0, x1=200, y1=150, x2=100, y2=50)
    assert b.bounds() == (100, 50, 200, 150)


@pytest.mark.parametrize("kwargs", [
    dict(x1=10, y1=10, x2=10, y2=50),          # zero width
    dict(x1=10, y1=10, x2=50, y2=10),          # zero height
    dict(x1=float("nan"), y1=0, x2=5, y2=5),
])
def test_bbox_rejects_degenerate(kwargs):
    with pytest.raises(ValidationError):
        BBox(class_id=0, **kwargs)


@pytest.mark.parametrize("points", [
    [(0, 0), (1, 1)],                       # too few points
    [(0, 0), (5, 5), (10, 10)],             # collinear, zero area
    [(0, 0), (float("inf"), 0), (0, 5)],
])
def test_polygon_rejects_invalid(points):
    with pytest.raises(ValidationError):
        Polygon(class_id=0, points=points)


def test_negative_class_id_rejected():
    with pytest.raises(ValidationError):
        BBox(class_id=-1, x1=0, y1=0, x2=1, y2=1)


def test_shapes_get_unique_ids():
    a, b = BBox(class_id=0, x1=0, y1=0, x2=1, y2=1), BBox(class_id=0, x1=0, y1=0, x2=1, y2=1)
    assert a.id and a.id != b.id


# ---- JSON --------------------------------------------------------------------

def test_json_roundtrip_exact(tmp_path):
    ann = sample()
    path = tmp_path / "라벨 폴더" / "a.json"
    path.parent.mkdir()
    save_annotations(path, ann)

    assert load_annotations(path) == ann
    raw = path.read_text(encoding="utf-8")
    assert "스크래치" in raw  # UTF-8, not \u escapes
    assert json.loads(raw)["shapes"][1]["type"] == "polygon"


def test_json_rejects_unknown_shape_type(tmp_path):
    data = sample().model_dump(mode="json")
    data["shapes"][0]["type"] = "circle"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_annotations(path)


# ---- YOLO detect -------------------------------------------------------------

def test_detect_bbox_line():
    line = shape_to_yolo(BBox(class_id=2, x1=0, y1=0, x2=320, y2=120), W, H, "detect")
    assert line == "2 0.250000 0.125000 0.500000 0.250000"


def test_detect_polygon_becomes_bounding_box():
    line = shape_to_yolo(Polygon(class_id=1, points=TRIANGLE), W, H, "detect")
    cx, cy, bw, bh = values(line)
    assert (cx * W, cy * H, bw * W, bh * H) == pytest.approx((200, 225, 200, 350))


def test_detect_roundtrip():
    ann = sample()
    back = shapes_from_yolo(ann.to_yolo_lines("detect"), W, H, "detect")

    assert [s.class_id for s in back] == [0, 1]
    assert back[0].bounds() == pytest.approx(ann.shapes[0].bounds(), abs=1e-3)
    assert back[1].bounds() == pytest.approx(ann.shapes[1].bounds(), abs=1e-3)


# ---- YOLO seg ----------------------------------------------------------------

def test_seg_bbox_becomes_four_point_polygon():
    line = shape_to_yolo(BBox(class_id=0, x1=64, y1=48, x2=320, y2=240), W, H, "seg")
    assert values(line) == pytest.approx([0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5])


def test_seg_roundtrip():
    ann = sample()
    back = shapes_from_yolo(ann.to_yolo_lines("seg"), W, H, "seg")

    assert all(isinstance(s, Polygon) for s in back)
    assert np.allclose(back[0].points, ann.shapes[0].to_points(), atol=1e-3)
    assert np.allclose(back[1].points, TRIANGLE, atol=1e-3)


# ---- clipping ----------------------------------------------------------------

def test_detect_clips_to_image():
    line = shape_to_yolo(BBox(class_id=0, x1=-100, y1=-50, x2=320, y2=240), W, H, "detect")
    cx, cy, bw, bh = values(line)
    assert (cx, cy, bw, bh) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_seg_clips_polygon_to_image():
    # Triangle poking out of the right edge.
    poly = Polygon(class_id=0, points=[(600, 100), (700, 200), (600, 300)])
    vals = values(shape_to_yolo(poly, W, H, "seg"))
    xs = vals[0::2]
    assert max(xs) == pytest.approx(1.0)
    assert len(xs) == 4  # tip cut off -> quadrilateral


@pytest.mark.parametrize("task", ["detect", "seg"])
def test_shape_fully_outside_is_dropped(task):
    outside = BBox(class_id=0, x1=700, y1=10, x2=800, y2=50)
    assert shape_to_yolo(outside, W, H, task) is None
    ann = ImageAnnotations(image="a.png", width=W, height=H, shapes=[outside])
    assert ann.to_yolo_lines(task) == []


# ---- parsing errors ----------------------------------------------------------

@pytest.mark.parametrize("task, line", [
    ("detect", "0 0.5 0.5 0.1"),
    ("detect", "x 0.5 0.5 0.1 0.1"),
    ("seg", "0 0.1 0.1 0.2 0.2"),
    ("seg", "0 0.1 0.1 0.2 0.2 0.3"),
])
def test_invalid_yolo_lines(task, line):
    with pytest.raises(ValueError):
        shapes_from_yolo([line], W, H, task)


def test_blank_yolo_lines_ignored():
    assert shapes_from_yolo(["", "   "], W, H, "detect") == []


def test_empty_label_file_for_background(tmp_path):
    path = tmp_path / "ok.txt"
    write_yolo_txt(path, ImageAnnotations(image="ok.png", width=W, height=H), "detect")
    assert path.read_bytes() == b""


def test_yolo_txt_uses_lf(tmp_path):
    path = tmp_path / "a.txt"
    write_yolo_txt(path, sample(), "detect")
    data = path.read_bytes()
    assert b"\r\n" not in data and data.count(b"\n") == 2


# ---- project integration -----------------------------------------------------

@pytest.fixture
def project(tmp_path):
    src = tmp_path / "원본"
    src.mkdir()
    imwrite_unicode(src / "스크래치 01.png", np.zeros((H, W, 3), np.uint8))
    imwrite_unicode(src / "b.png", np.zeros((10, 20, 3), np.uint8))
    p = Project.create(tmp_path / "검사 프로젝트")
    p.add_images([src])
    return p


def test_project_load_labels_empty_uses_image_size(project):
    ann = project.load_labels("b.png")
    assert (ann.width, ann.height, ann.shapes) == (20, 10, [])
    assert not project.has_labels("b.png")


def test_project_save_labels_writes_json_and_txt(project):
    ann = sample()
    project.save_labels(ann)

    assert project.has_labels("스크래치 01.png")
    assert project.label_json_path("스크래치 01.png").name == "스크래치 01.json"
    assert project.load_labels("스크래치 01.png") == ann
    lines = project.label_txt_path("스크래치 01.png").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and len(values(lines[1])) == 4  # detect: polygon -> box


def test_project_regenerate_after_task_switch(project):
    project.save_labels(sample())
    project.config.task = "seg"
    assert project.regenerate_yolo_labels() == 1

    lines = project.label_txt_path("스크래치 01.png").read_text(encoding="utf-8").splitlines()
    assert len(values(lines[0])) == 8  # bbox -> 4-point polygon
    assert len(values(lines[1])) == 6  # triangle


def test_project_save_labels_rejects_unknown_image(project):
    with pytest.raises(KeyError):
        project.save_labels(ImageAnnotations(image="missing.png", width=1, height=1))


def test_import_avoids_label_stem_clash(project, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    imwrite_unicode(other / "b.jpg", np.zeros((4, 4, 3), np.uint8))
    added = project.add_images([other])

    assert added[0].path == "b_1.jpg"
    assert project.label_json_path("b_1.jpg") != project.label_json_path("b.png")


# ---- auto-label suggestions (F2.8) -------------------------------------------

def suggested(score, class_id=0, x=0):
    return BBox(class_id=class_id, x1=x, y1=0, x2=x + 50, y2=50, score=score)


def test_suggestions_excluded_from_yolo_until_accepted():
    ann = sample()
    ann.add_suggestions([suggested(0.9), suggested(0.4, x=100)], source="model")

    assert len(ann.suggestions()) == 2
    assert len(ann.to_yolo_lines("detect")) == 2  # only the 2 confirmed shapes

    accepted = ann.accept([ann.suggestions()[0].id])

    assert accepted == 1
    assert len(ann.to_yolo_lines("detect")) == 3
    assert len(ann.suggestions()) == 1


def test_accepted_suggestion_keeps_source_and_score():
    ann = sample()
    ann.add_suggestions([suggested(0.8)], source="model")
    sid = ann.suggestions()[0].id
    ann.accept([sid])
    shape = next(s for s in ann.shapes if s.id == sid)
    assert (shape.status, shape.source, shape.score) == ("confirmed", "model", 0.8)


def test_accept_above_threshold():
    ann = sample()
    ann.add_suggestions([suggested(0.95), suggested(0.5, x=100), suggested(0.7, x=200)], source="model")
    assert ann.accept_above(0.7) == 2
    assert [s.score for s in ann.suggestions()] == [0.5]


def test_reject_only_removes_suggestions():
    ann = sample()
    ann.add_suggestions([suggested(0.9)], source="sam")
    confirmed_id = ann.shapes[0].id
    removed = ann.reject([ann.suggestions()[0].id, confirmed_id])

    assert removed == 1
    assert ann.suggestions() == []
    assert any(s.id == confirmed_id for s in ann.shapes)


def test_rerunning_a_source_replaces_its_suggestions():
    ann = sample()
    ann.add_suggestions([suggested(0.9), suggested(0.8, x=100)], source="model")
    ann.add_suggestions([suggested(0.6, x=200)], source="sam")
    ann.add_suggestions([suggested(0.7, x=300)], source="model")

    assert sorted((s.source, s.score) for s in ann.suggestions()) == [("model", 0.7), ("sam", 0.6)]
    assert ann.clear_suggestions() == 2


def test_suggestions_persist_in_json(tmp_path):
    ann = sample()
    ann.add_suggestions([Polygon(class_id=1, points=TRIANGLE, score=0.66)], source="sam")
    path = tmp_path / "a.json"
    save_annotations(path, ann)

    loaded = load_annotations(path)
    assert loaded == ann
    assert loaded.suggestions()[0].source == "sam"


def test_old_json_without_status_loads_as_confirmed(tmp_path):
    data = sample().model_dump(mode="json")
    for shape in data["shapes"]:
        for key in ("status", "source", "score"):
            shape.pop(key)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_annotations(path)
    assert loaded.suggestions() == [] and len(loaded.confirmed()) == 2


def test_score_must_be_probability():
    with pytest.raises(ValidationError):
        suggested(1.5)


def test_project_txt_ignores_pending_suggestions(project):
    ann = project.load_labels("b.png")
    ann.add_suggestions([BBox(class_id=0, x1=1, y1=1, x2=5, y2=5, score=0.9)], source="model")
    project.save_labels(ann)

    assert project.label_txt_path("b.png").read_bytes() == b""
    assert len(project.load_labels("b.png").suggestions()) == 1
