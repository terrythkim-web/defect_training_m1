"""Canvas, image list and main-window navigation (Qt parts of PROMPTS step 5)."""

import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from defect_inspection.core.annotation import BBox
from defect_inspection.core.project import Project
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.canvas.image_canvas import CanvasPanel, ImageCanvas, numpy_to_qimage
from defect_inspection.ui.image_list import status_text
from defect_inspection.ui.main_window import MainWindow
from defect_inspection.core.project import LabelStatus


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def canvas(qapp):
    c = ImageCanvas()
    c.resize(800, 600)
    c.show()
    qapp.processEvents()
    yield c
    c.close()


# ---- canvas --------------------------------------------------------------------

def test_numpy_to_qimage_formats():
    gray = numpy_to_qimage(np.zeros((5, 7), np.uint8))
    bgr = np.zeros((5, 7, 3), np.uint8)
    bgr[..., 2] = 255  # red in BGR
    color = numpy_to_qimage(bgr)
    assert (gray.width(), gray.height()) == (7, 5)
    assert color.pixelColor(0, 0).red() == 255 and color.pixelColor(0, 0).blue() == 0


def test_set_image_fits_view(canvas):
    canvas.set_image(np.zeros((1000, 2000, 3), np.uint8))
    assert canvas.image_size == (2000, 1000)
    # 2000 px wide image in an ~800 px viewport -> zoom below 0.4
    assert 0.3 < canvas.zoom < 0.41


def test_grayscale_image_displays(canvas):
    canvas.set_image(np.full((50, 60), 200, np.uint8))
    assert canvas.has_image and canvas.image_size == (60, 50)


def test_zoom_limits_and_fit(canvas):
    canvas.set_image(np.zeros((100, 100), np.uint8))
    canvas.zoom_to(1000)
    assert canvas.zoom == pytest.approx(64)
    canvas.zoom_to(0.0001)
    assert canvas.zoom == pytest.approx(0.02)
    canvas.fit()
    assert canvas.zoom == pytest.approx(min(canvas.viewport().width(), canvas.viewport().height()) / 100, rel=0.05)


def test_wheel_zooms_in_and_out(canvas, qapp):
    canvas.set_image(np.zeros((400, 400), np.uint8))
    before = canvas.zoom
    center = QPointF(canvas.viewport().rect().center())
    for delta in (120, -240):
        ev = QWheelEvent(center, canvas.mapToGlobal(center), QPoint(), QPoint(0, delta),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(canvas.viewport(), ev)
        if delta > 0:
            assert canvas.zoom == pytest.approx(before * 1.25)
    assert canvas.zoom == pytest.approx(before / 1.25)


def test_image_pos_maps_view_to_pixels(canvas):
    canvas.set_image(np.zeros((100, 200), np.uint8))
    canvas.zoom_to(1.0)
    canvas.centerOn(100, 50)
    pos = canvas.image_pos(canvas.mapFromScene(QPointF(10.5, 20.5)))
    assert pos == pytest.approx((10.5, 20.5), abs=1)
    assert canvas.image_pos(canvas.mapFromScene(QPointF(-5, 20))) is None


def test_adjustment_is_display_only(canvas):
    base = np.full((10, 10), 100, np.uint8)
    canvas.set_image(base)
    canvas.set_adjustment(80, 50)
    shown = canvas._pixmap_item.pixmap().toImage()
    assert shown.pixelColor(0, 0).red() > 100
    assert (base == 100).all()


def test_clear_image(canvas):
    canvas.set_image(np.zeros((10, 10), np.uint8))
    canvas.set_image(None)
    assert not canvas.has_image and canvas.image_size is None


def test_canvas_panel_sliders_apply(qapp):
    panel = CanvasPanel()
    panel.canvas.set_image(np.zeros((10, 10), np.uint8))
    panel.brightness.setValue(30)
    panel.contrast.setValue(-20)
    assert (panel.canvas.brightness, panel.canvas.contrast) == (30, -20)
    panel.reset_adjustment()
    assert (panel.canvas.brightness, panel.canvas.contrast) == (0, 0)


def test_4k_display_and_zoom_under_100ms(canvas):
    """PRD 5: UI response < 100 ms for 4K images (cached decode; display and zoom)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (2160, 3840, 3), dtype=np.uint8)

    def median_ms(fn, runs=9):  # 9 runs: robust to background load from other tests
        fn()  # warm-up: first call pays one-time Qt/numpy setup
        times = []
        for i in range(runs):
            t = time.perf_counter()
            fn(i)
            times.append((time.perf_counter() - t) * 1000)
        return sorted(times)[runs // 2]

    set_ms = median_ms(lambda i=0: canvas.set_image(img))
    zoom_ms = median_ms(lambda i=0: (canvas.zoom_by(1.25 if i % 2 else 0.8), canvas.viewport().repaint()))
    adjust_ms = median_ms(lambda i=0: canvas.set_adjustment(10 + i, 20 - i))

    assert set_ms < 100, set_ms
    assert zoom_ms < 100, zoom_ms
    assert adjust_ms < 100, adjust_ms


# ---- image list + navigation in the main window --------------------------------

@pytest.fixture
def window(qapp, tmp_path):
    src = tmp_path / "원본 이미지"
    src.mkdir()
    for i, name in enumerate(["a.png", "b.png", "c.jpg", "d.tif"]):
        imwrite_unicode(src / name, np.full((40, 60, 3), 50 * i, np.uint8))
    project = Project.create(tmp_path / "검사 프로젝트")
    project.add_class("scratch")
    project.add_images([src])
    project.set_ok("d.tif")
    ann = project.load_labels("b.png")
    ann.shapes.append(BBox(class_id=0, x1=1, y1=1, x2=10, y2=10))
    ann.add_suggestions([BBox(class_id=0, x1=20, y1=20, x2=30, y2=30, score=0.9)], source="model")
    project.save_labels(ann)
    ann = project.load_labels("c.jpg")
    ann.add_suggestions([BBox(class_id=0, x1=2, y1=2, x2=8, y2=8, score=0.5)], source="sam")
    project.save_labels(ann)

    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.show()
    w.load_project(project)
    qapp.processEvents()
    yield w
    w.project_dirty = False
    w.close()


def texts(w):
    return [w.image_list.item(i).text() for i in range(w.image_list.count())]


def visible(w):
    return [w.image_list.item(i).text().split("\n")[0] for i in w.image_panel.visible_rows()]


def test_status_text():
    assert status_text(LabelStatus(0, 0, False)) == "미라벨"
    assert status_text(LabelStatus(3, 1, False)) == "라벨 3 · 제안 1"
    assert status_text(LabelStatus(0, 0, True)) == "OK 이미지"


def test_list_shows_label_status(window):
    assert texts(window) == ["a.png\n미라벨", "b.png\n라벨 1 · 제안 1", "c.jpg\n미라벨 · 제안 1", "d.tif\nOK 이미지"]


@pytest.mark.parametrize("key, expected", [
    ("all", ["a.png", "b.png", "c.jpg", "d.tif"]),
    ("unlabeled", ["a.png", "c.jpg"]),
    ("labeled", ["b.png", "d.tif"]),
    ("suggested", ["b.png", "c.jpg"]),
    ("ok", ["d.tif"]),
])
def test_filters(window, key, expected):
    window.image_panel.set_filter(key)
    assert visible(window) == expected
    assert window.image_panel.current_path() == expected[0]


def test_first_image_shown_on_load(window, qapp):
    assert window.current_image == "a.png"
    assert wait_until(qapp, lambda: window.image_ready)
    assert window.canvas.image_size == (60, 40)


def test_uncached_image_loads_without_blocking(window, qapp):
    window.image_cache.invalidate()
    window.image_panel.select("c.jpg")
    # Selection returns before decoding; the image arrives via the queued signal.
    assert wait_until(qapp, lambda: window.image_ready)
    assert window.current_image == "c.jpg"


def test_stale_load_result_is_ignored(window, qapp):
    assert wait_until(qapp, lambda: window.image_ready)
    window.image_panel.select("b.png")
    assert wait_until(qapp, lambda: window.image_ready)
    shown = window.canvas._base
    # A late result for another image must not replace what is on screen.
    window._show_loaded_image(window.project.image_abspath("a.png"), np.zeros((5, 5), np.uint8))
    assert window.canvas._base is shown


def test_unreadable_image_shows_message(window, qapp):
    path = window.project.image_abspath("a.png")
    path.write_bytes(b"broken")
    window.image_cache.invalidate()
    window.image_panel.select("b.png")
    window.image_panel.select("a.png")
    assert wait_until(qapp, lambda: "읽을 수 없습니다" in window.canvas._placeholder.toPlainText())
    assert not window.image_ready


def test_a_d_navigation_respects_filter(window):
    window.act_next_image.trigger()
    assert window.current_image == "b.png"
    window.image_panel.set_filter("unlabeled")  # current b.png hidden -> jumps to a.png
    assert window.current_image == "a.png"
    window.act_next_image.trigger()
    assert window.current_image == "c.jpg"
    window.act_next_image.trigger()  # stays at the end
    assert window.current_image == "c.jpg"
    window.act_prev_image.trigger()
    assert window.current_image == "a.png"


def test_status_bar_label_count(window, qapp):
    window.image_panel.select("b.png")
    assert wait_until(qapp, lambda: window.session is not None and window.session.image_path == "b.png")
    assert window.status_labels.text() == "라벨 1 · 제안 1"


def test_neighbours_prefetched(window, qapp):
    window.image_panel.select("b.png")
    assert window.image_panel.neighbours() == ["c.jpg", "a.png", "d.tif"]
    assert wait_until(qapp, lambda: window.image_cache.contains(window.project.image_abspath("d.tif")))


def test_thumbnails_arrive_and_are_cached(window, qapp, tmp_path):
    def all_icons_loaded():
        return all(window.image_list.item(i).icon().pixmap(64).toImage().pixelColor(0, 0).name() != "#cccccc"
                   or i == 0  # a.png is black, indistinguishable from the placeholder check
                   for i in range(window.image_list.count()))

    assert wait_until(qapp, lambda: len(list((tmp_path / "cache" / "thumbs").glob("*.png"))) == 4)
    assert wait_until(qapp, all_icons_loaded)


def test_refresh_status_after_labels_change(window):
    ann = window.project.load_labels("a.png")
    ann.shapes.append(BBox(class_id=0, x1=0, y1=0, x2=5, y2=5))
    window.project.save_labels(ann)
    window.image_panel.refresh_status("a.png")
    assert texts(window)[0] == "a.png\n라벨 1"


def test_cursor_signal_updates_status(window):
    window.canvas.cursor_moved.emit(5, 7)
    assert window.status_coords.text() == "x 5, y 7"
