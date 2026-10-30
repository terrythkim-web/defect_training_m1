"""Edge filter (PRD F9): a prediction is kept only if it sits on the object's outermost edge —
a thin dark line in a strip around the box, with flat background beyond it and no further line.
Synthetic panels: textured interior with extra dark lines, flat background beyond the edge."""

from __future__ import annotations

import time

import numpy as np
import pytest

from defect_inspection.core.edge_band import edge_under_box
from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.core.project import Project

H, W = 1024, 512
CFG = EdgeFilterConfig()
EDGE_ROW = 880


def panel(edge_row=EDGE_ROW, inner_rows=(200, 600), tilt=0, gap=None, flat_rows=None, seed=0):
    """Panel above `edge_row` (textured, with extra dark lines), flat background below.
    `tilt` = how many px the outer line drops from left to right; `gap` = (x0, x1) left unbroken
    by the line (a defect cutting the edge); `flat_rows` = (y0, y1) of the panel left untextured."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(160, 2, (H, W)), 0, 255)
    xs = np.arange(W)
    edge_y = edge_row + tilt * xs / (W - 1)
    rows = np.arange(H)[:, None]
    inside = rows < edge_y[None, :]
    if flat_rows is not None:
        inside &= ~((rows >= flat_rows[0]) & (rows < flat_rows[1]))
    texture = rng.integers(0, 2, (H // 8, W // 8)).repeat(8, 0).repeat(8, 1) * 60 + 70
    img = np.where(inside, texture + rng.normal(0, 2, (H, W)), img)
    for r in inner_rows:
        img[r:r + 3, :] = 20
    for x in xs:
        if gap is None or not gap[0] <= x < gap[1]:
            y = int(round(edge_y[x]))
            img[y - 1:y + 2, x] = 20
    return np.clip(img, 0, 255).astype(np.uint8)


def on_row(y, x=200, half=15):
    return (x - 20, y - half, x + 20, y + half)


# ---- config --------------------------------------------------------------------------------

def test_config_defaults_and_limits():
    assert (CFG.enabled, CFG.along_px, CFG.across_px, CFG.touch_px) == (False, 200, 30, 12)
    with pytest.raises(ValueError):
        EdgeFilterConfig(across_px=-1)
    with pytest.raises(ValueError):
        EdgeFilterConfig(min_length=0)


def test_config_saved_in_project(tmp_path):
    p = Project.create(tmp_path / "p")
    p.config.edge_filter = EdgeFilterConfig(enabled=True, along_px=150, across_px=40, touch_px=5)
    p.save()
    again = Project.open(tmp_path / "p").config.edge_filter
    assert (again.enabled, again.along_px, again.across_px, again.touch_px) == (True, 150, 40, 5)


# ---- boxes on the edge are kept ---------------------------------------------------------------

def test_box_on_the_outer_edge():
    hit = edge_under_box(panel(), on_row(EDGE_ROW), CFG)
    assert hit is not None and hit.axis == "h" and hit.outward == 1
    assert abs(hit.position - EDGE_ROW) <= 2


def test_edge_broken_under_the_box():
    """A defect cutting the line: the line is only checked outside the box."""
    assert edge_under_box(panel(gap=(170, 230)), on_row(EDGE_ROW), CFG) is not None


def test_tilted_edge():
    img = panel(edge_row=860, tilt=20)
    assert edge_under_box(img, on_row(868, x=200), CFG) is not None


def test_background_on_the_other_side():
    img = np.ascontiguousarray(panel()[::-1])
    hit = edge_under_box(img, on_row(H - 1 - EDGE_ROW), CFG)
    assert hit is not None and hit.outward == -1


def test_vertical_edge():
    img = np.ascontiguousarray(panel(edge_row=420, inner_rows=(100, 250)).T)  # 512 x 1024, background right
    hit = edge_under_box(img, (405, 500, 435, 540), CFG)
    assert hit is not None and hit.axis == "v" and hit.outward == 1


def test_colour_image_accepted():
    rgb = np.repeat(panel()[:, :, None], 3, axis=2)
    assert edge_under_box(rgb, on_row(EDGE_ROW), CFG) is not None


def test_bgr_order_like_the_predictor_input():
    """Images reach the filter as OpenCV BGR (imaging.to_training_8bit): a pure blue line is gray 29
    read as BGR (blue weighs 0.114), dark enough; read as RGB it would be 76, lighter than the panel."""
    img = np.repeat(panel()[:, :, None], 3, axis=2)
    img[EDGE_ROW - 1:EDGE_ROW + 2] = (255, 0, 0)  # B, G, R
    assert edge_under_box(img, on_row(EDGE_ROW), CFG) is not None


def test_single_channel_with_axis():
    assert edge_under_box(panel()[:, :, None], on_row(EDGE_ROW), CFG) is not None


def test_only_8_bit_pixels():
    with pytest.raises(ValueError):
        edge_under_box(panel().astype(np.uint16) * 256, on_row(EDGE_ROW), CFG)


@pytest.mark.parametrize("gap, kept", [(10, True), (11, True), (20, False)])
def test_box_near_the_line_within_touch_distance(gap, kept):
    """A crack hanging into the object from the edge: its box may stop short of the line."""
    box = (180, EDGE_ROW - gap - 40, 220, EDGE_ROW - gap)
    assert (edge_under_box(panel(), box, CFG) is not None) is kept


def test_box_along_most_of_the_edge():
    """A chip as long as the image leaves no line beside it: the line is measured under the box too."""
    assert edge_under_box(panel(), (10, EDGE_ROW - 15, W - 10, EDGE_ROW + 15), CFG) is not None


def test_narrow_strip_still_reaches_the_touch_distance():
    """`across_px` below what `touch_px` needs is widened, not a silent rejection."""
    cfg = EdgeFilterConfig(across_px=8)
    assert edge_under_box(panel(), (180, EDGE_ROW - 42, 220, EDGE_ROW - 2), cfg) is not None


def test_line_along_the_image_border():
    """Nothing beyond a line at the very border to check: it counts as the edge."""
    img = panel(edge_row=H - 6)
    assert edge_under_box(img, on_row(H - 6, half=5), CFG) is not None


# ---- other boxes are dropped -----------------------------------------------------------------

def test_box_on_an_inner_line():
    assert edge_under_box(panel(), on_row(600), CFG) is None


def test_line_with_another_line_just_outside_is_not_the_edge():
    """Parallel line 30 px inside the edge, flat strip between: the edge shows up beyond it.
    (Closer than `touch_px` to the edge, a box would count as on the edge itself.)"""
    img = panel(inner_rows=(EDGE_ROW - 30,), flat_rows=(EDGE_ROW - 28, EDGE_ROW))
    assert edge_under_box(img, on_row(EDGE_ROW - 30, half=8), CFG) is None
    assert edge_under_box(img, on_row(EDGE_ROW, half=8), CFG) is not None


def test_line_tilted_beyond_the_limit():
    img = panel(edge_row=860, tilt=60)  # ~6.7 degrees
    assert edge_under_box(img, on_row(890, x=256), CFG) is None
    assert edge_under_box(img, on_row(890, x=256), EdgeFilterConfig(max_tilt_deg=10)) is not None


def test_box_in_the_textured_interior():
    assert edge_under_box(panel(), on_row(400), CFG) is None


def test_box_in_the_flat_background():
    assert edge_under_box(panel(), on_row(960), CFG) is None


def test_box_outside_the_image_is_clipped():
    assert edge_under_box(panel(), (-50, EDGE_ROW - 15, 30, EDGE_ROW + 15), CFG) is not None
    assert edge_under_box(panel(), (W + 10, 0, W + 50, 40), CFG) is None
    assert edge_under_box(panel(), (float("nan"), EDGE_ROW - 15, 30, EDGE_ROW + 15), CFG) is None


def test_many_line_pieces_stay_fast():
    """Striped noisy picture: line grouping must not refit every group per piece."""
    rng = np.random.default_rng(3)
    img = rng.normal(150, 20, (1024, 2048)).clip(0, 255).astype(np.uint8)
    img[::7, :] = 40
    start = time.perf_counter()
    edge_under_box(img, (800, 400, 1200, 700), CFG)
    assert time.perf_counter() - start < 1.0
