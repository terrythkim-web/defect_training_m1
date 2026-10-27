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


# ---- tiled SAM (native-resolution window around the prompt) -------------------------------

from defect_inspection.core.sam_assist import SAM_WINDOW, WINDOW_GRID, prompt_bounds, sam_window  # noqa: E402


class FakeSam(SamAssistant):
    """Encodes nothing; the 'mask' is the dark region of the current window that contains the
    first include point (or the box centre), like SAM on a clean defect."""

    def __init__(self):
        super().__init__()
        self.encoded = []

    def load(self):
        self._predictor = object()

    def set_image(self, image8, key):
        if key != self._image_key:
            self.encoded.append(key)
            self._tile, self._image_key = image8.copy(), key

    def predict(self, prompt):
        import cv2

        x, y = prompt.points[0] if prompt.points else ((prompt.box[0] + prompt.box[2]) / 2, (prompt.box[1] + prompt.box[3]) / 2)
        dark = (self._tile < 100).astype(np.uint8)
        n, labels = cv2.connectedComponents(dark)
        lab = labels[int(y), int(x)]
        return (labels == lab, 0.9) if lab else None


def line_image(w=3072, h=1024, x=1500, y0=300, y1=500, width=6):
    img = np.full((h, w), 200, np.uint8)
    img[y0:y1, x:x + width] = 30
    return img


def test_window_whole_image_when_it_fits():
    assert sam_window((10, 10, 20, 20), 800, 600) == (0, 0, 800, 600)
    assert sam_window((10, 10, 20, 20), 1024, 1024) == (0, 0, 1024, 1024)


@pytest.mark.parametrize("px, py", [(5, 5), (1500, 400), (3070, 1020), (700, 900)])
def test_window_contains_prompt_and_stays_inside(px, py):
    x0, y0, x1, y1 = sam_window((px, py, px, py), 3072, 2048)
    assert (x1 - x0, y1 - y0) == (SAM_WINDOW, SAM_WINDOW)
    assert 0 <= x0 <= px - 0 and px <= x1 <= 3072 and 0 <= y0 <= py and py <= y1 <= 2048
    assert x0 % WINDOW_GRID == 0 or x1 == 3072  # snapped (or flush with the image edge)


def test_nearby_clicks_share_a_window_and_big_boxes_get_a_bigger_one():
    a = sam_window((1500, 400, 1500, 400), 3072, 1024)
    b = sam_window((1530, 420, 1530, 420), 3072, 1024)
    assert a == b
    big = sam_window((100, 100, 2500, 900), 3072, 1024)
    assert big[0] <= 100 and big[2] >= 2500


def test_tiled_prediction_maps_mask_back_to_the_image():
    img = line_image()
    sam = FakeSam()
    mask, score = sam.predict_tiled(img, "img", SamPrompt(points=[(1502, 400)], labels=[1]))
    assert mask.shape == img.shape and score == 0.9
    assert np.array_equal(mask, img < 100)  # exact pixels, full resolution
    assert len(sam.encoded) == 1 and sam.encoded[0].startswith("img#")
    sam.predict_tiled(img, "img", SamPrompt(points=[(1503, 450)], labels=[1]))
    assert len(sam.encoded) == 1  # same window: embedding reused
    assert not sam.needs_encoding(img, "img", SamPrompt(points=[(1503, 450)], labels=[1]))


def test_mask_cut_by_the_window_grows_it_once():
    img = line_image(y0=0, y1=1024, x=100, width=2200)  # wider than one window
    img[:, :] = 200
    img[480:520, 100:2300] = 30  # 2200 px long horizontal scratch
    sam = FakeSam()
    mask, _ = sam.predict_tiled(img, "img", SamPrompt(points=[(1200, 500)], labels=[1]))
    assert np.array_equal(mask, img < 100) and len(sam.encoded) == 2


def test_small_image_prepared_ahead_and_reused():
    img = line_image(w=800, h=600, x=400, y0=100, y1=300)
    sam = FakeSam()
    assert sam.prepare(img, "small")
    sam.predict_tiled(img, "small", SamPrompt(points=[(402, 200)], labels=[1]))
    assert sam.encoded == ["small#0,0,800,600"]
    assert not FakeSam().prepare(line_image(), "big")  # large images encode per click window


def test_prompt_bounds():
    assert prompt_bounds(SamPrompt(points=[(5, 6), (1, 9)], labels=[1, 0], box=(0, 2, 3, 4))) == (0, 2, 5, 9)
