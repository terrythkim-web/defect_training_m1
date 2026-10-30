"""Edge filter (PRD F9): a prediction is kept only if it sits on the object's outermost edge —
a thin dark or bright line under the box, with flat background and no further line from it to the image border.
Synthetic panels: textured interior with extra dark lines, flat background beyond the edge.
Inverted (255 - panel) they show a bright edge line on a dark background, like a display panel."""

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
    assert (CFG.enabled, CFG.along_px, CFG.touch_px) == (False, 200, 12)
    assert CFG.line_polarity == "both"
    with pytest.raises(ValueError):
        EdgeFilterConfig(line_polarity="grey")
    with pytest.raises(ValueError):
        EdgeFilterConfig(along_px=-1)
    with pytest.raises(ValueError):
        EdgeFilterConfig(min_length=0)


def test_config_saved_in_project(tmp_path):
    p = Project.create(tmp_path / "p")
    p.config.edge_filter = EdgeFilterConfig(enabled=True, along_px=150, touch_px=5, line_polarity="dark")
    p.save()
    again = Project.open(tmp_path / "p").config.edge_filter
    assert (again.enabled, again.along_px, again.touch_px, again.line_polarity) == (True, 150, 5, "dark")


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


def test_line_along_the_image_border():
    """Nothing beyond a line at the very border to check: it counts as the edge."""
    img = panel(edge_row=H - 6)
    assert edge_under_box(img, on_row(H - 6, half=5), CFG) is not None


def test_bright_edge_line_on_dark_background():
    hit = edge_under_box(255 - panel(), on_row(EDGE_ROW), CFG)
    assert hit is not None and hit.axis == "h" and hit.outward == 1
    assert abs(hit.position - EDGE_ROW) <= 2


@pytest.mark.parametrize("polarity, dark_kept, bright_kept",
                         [("both", True, True), ("dark", True, False), ("bright", False, True)])
def test_line_polarity_setting(polarity, dark_kept, bright_kept):
    cfg = EdgeFilterConfig(line_polarity=polarity)
    assert (edge_under_box(panel(), on_row(EDGE_ROW), cfg) is not None) is dark_kept
    assert (edge_under_box(255 - panel(), on_row(EDGE_ROW), cfg) is not None) is bright_kept


# ---- other boxes are dropped -----------------------------------------------------------------

def test_box_on_an_inner_line():
    assert edge_under_box(panel(), on_row(600), CFG) is None


def test_line_with_another_line_just_outside_is_not_the_edge():
    """Parallel line 30 px inside the edge, flat strip between: the edge shows up beyond it.
    (Closer than `touch_px` to the edge, a box would count as on the edge itself.)"""
    img = panel(inner_rows=(EDGE_ROW - 30,), flat_rows=(EDGE_ROW - 28, EDGE_ROW))
    assert edge_under_box(img, on_row(EDGE_ROW - 30, half=8), CFG) is None
    assert edge_under_box(img, on_row(EDGE_ROW, half=8), CFG) is not None


def test_bright_inner_line_with_the_edge_beyond_is_not_the_edge():
    img = 255 - panel(inner_rows=(EDGE_ROW - 30,), flat_rows=(EDGE_ROW - 28, EDGE_ROW))
    assert edge_under_box(img, on_row(EDGE_ROW - 30, half=8), CFG) is None
    assert edge_under_box(img, on_row(EDGE_ROW, half=8), CFG) is not None


def test_line_of_the_other_polarity_beyond_is_still_a_line():
    """A dark edge with a bright line just outside it: the bright line counts as 'another line beyond'."""
    img = panel()
    img[EDGE_ROW + 21:EDGE_ROW + 24, :] = 255  # beyond `touch_px`, else it would be the edge itself
    assert edge_under_box(img, on_row(EDGE_ROW, half=8), CFG) is None


@pytest.mark.parametrize("gap", [30, 100, 300])
def test_inner_line_with_flat_panel_beyond_is_not_the_edge(gap):
    """Strictly the outermost line: however wide the flat stretch beyond an inner line, the edge
    further out disqualifies it."""
    img = panel(inner_rows=(EDGE_ROW - gap,), flat_rows=(EDGE_ROW - gap + 2, EDGE_ROW))
    assert edge_under_box(img, on_row(EDGE_ROW - gap, half=8), CFG) is None
    assert edge_under_box(img, on_row(EDGE_ROW, half=8), CFG) is not None


def test_bright_inner_line_on_a_flat_dark_display_is_not_the_edge():
    """A pad border inside a dark display: flat dark beyond it, but the panel edge lies further out."""
    img = 255 - panel(inner_rows=(EDGE_ROW - 150,), flat_rows=(EDGE_ROW - 148, EDGE_ROW))
    assert edge_under_box(img, on_row(EDGE_ROW - 150, half=8), CFG) is None


def test_dust_far_out_in_the_background_is_allowed():
    img = panel()
    img[1000:1006, 100:106] = 0
    assert edge_under_box(img, on_row(EDGE_ROW), CFG) is not None


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
