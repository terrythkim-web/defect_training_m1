from pathlib import Path

import cv2
import numpy as np
import pytest

from defect_inspection.core.annotation import BBox, Polygon
from defect_inspection.core.sam_assist import (
    SamAssistant,
    SamPrompt,
    largest_component,
    mask_to_bbox,
    mask_to_polygon,
    mask_to_shape,
    resolve_weights,
)

CACHED_WEIGHTS = Path.home() / ".cache" / "DefectLab" / "weights" / "mobile_sam.pt"


def disk(h=200, w=300, cx=150, cy=100, r=40):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


# ---- prompts -------------------------------------------------------------------------

def test_prompt_points_are_one_object():
    p = SamPrompt(points=[(10, 20), (30, 40)], labels=[1, 0])
    assert p.to_kwargs() == {"points": [[[10, 20], [30, 40]]], "labels": [[1, 0]]}


def test_prompt_box_and_points():
    p = SamPrompt(points=[(5, 5)], labels=[1], box=(1, 2, 3, 4))
    kw = p.to_kwargs()
    assert kw["bboxes"] == [[1, 2, 3, 4]] and kw["labels"] == [[1]]


def test_empty_prompt():
    assert SamPrompt().is_empty and SamPrompt().to_kwargs() == {}
    assert not SamPrompt(box=(0, 0, 1, 1)).is_empty


# ---- mask -> shape -------------------------------------------------------------------

def test_largest_component_drops_speckles():
    m = disk()
    m[5, 5] = True  # speckle
    region = largest_component(m)
    assert not region[5, 5] and region[100, 150]


def test_empty_mask():
    empty = np.zeros((50, 50), bool)
    assert largest_component(empty) is None
    assert mask_to_polygon(empty) is None and mask_to_bbox(empty) is None
    assert mask_to_shape(empty, "seg", 0, 0.9) is None


def test_mask_to_bbox_covers_whole_pixels():
    m = np.zeros((50, 60), bool)
    m[10:20, 30:45] = True
    assert mask_to_bbox(m) == (30, 10, 45, 20)


def test_mask_to_polygon_area_close_to_mask():
    m = disk(r=40)
    pts = mask_to_polygon(m, tolerance=1.0)
    area = cv2.contourArea(np.array(pts, np.float32))
    assert abs(area - m.sum()) / m.sum() < 0.05
    assert 8 <= len(pts) <= 60  # simplified, not one point per pixel


def test_tolerance_controls_vertex_count():
    m = disk(r=60)
    assert len(mask_to_polygon(m, tolerance=0.5)) > len(mask_to_polygon(m, tolerance=4.0))


def test_mask_to_shape_by_task():
    m = disk()
    seg = mask_to_shape(m, "seg", 2, 0.93)
    det = mask_to_shape(m, "detect", 2, 0.93)
    assert isinstance(seg, Polygon) and isinstance(det, BBox)
    for s in (seg, det):
        assert (s.status, s.source, s.score, s.class_id) == ("suggested", "sam", 0.93, 2)
    assert det.bounds() == pytest.approx((110, 60, 191, 141))


def test_tiny_mask_falls_back_to_box_for_seg():
    m = np.zeros((20, 20), bool)
    m[5:7, 5:7] = True  # 2x2: contour degenerates
    shape = mask_to_shape(m, "seg", 0, None)
    assert isinstance(shape, BBox) and shape.bounds() == (5, 5, 7, 7)


def test_score_clamped():
    assert mask_to_shape(disk(), "detect", 0, 1.2).score == 1.0


# ---- weights -------------------------------------------------------------------------

def test_resolve_existing_file_used_as_is(tmp_path):
    f = tmp_path / "my_sam.pt"
    f.write_bytes(b"x")
    assert resolve_weights(str(f), tmp_path / "weights") == f


def test_resolve_downloads_into_weights_dir(tmp_path):
    calls = []

    def fake_download(target):
        calls.append(target)
        Path(target).write_bytes(b"w")
        return target

    wdir = tmp_path / "weights"
    path = resolve_weights("mobile_sam.pt", wdir, download=fake_download)
    assert path == wdir / "mobile_sam.pt" and calls == [str(wdir / "mobile_sam.pt")]
    # second call finds it without downloading
    assert resolve_weights("mobile_sam.pt", wdir, download=fake_download) == path and len(calls) == 1


# ---- real model (only where the weights are already cached; CI skips) -------------------

@pytest.mark.skipif(not CACHED_WEIGHTS.is_file(), reason="mobile_sam.pt not cached locally")
def test_real_sam_segments_a_blob():
    img = np.full((400, 600, 3), 170, np.uint8)
    cv2.circle(img, (300, 200), 40, (50, 50, 50), -1)
    sam = SamAssistant(weights_dir=CACHED_WEIGHTS.parent)
    sam.set_image(img, "blob")
    mask, score = sam.predict(SamPrompt(points=[(300, 200)], labels=[1]))
    box = mask_to_shape(mask, "detect", 0, score)
    assert box.bounds() == pytest.approx((260, 160, 341, 241), abs=4)
    assert score > 0.8
