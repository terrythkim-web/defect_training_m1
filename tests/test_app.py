import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QTabWidget

from defect_inspection.core.project import Project
from defect_inspection.platform_utils import detect_device, imwrite_unicode
from defect_inspection.ui.main_window import TABS, MainWindow, Tool


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qapp, settings, tmp_path):
    w = MainWindow(settings, cache_root=tmp_path / "cache")
    w.show()
    qapp.processEvents()
    yield w
    w.project_dirty = False  # skip the save prompt
    w.close()


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "원본 이미지"
    src.mkdir()
    for name in ["a.png", "b.png", "c.jpg"]:
        imwrite_unicode(src / name, np.zeros((8, 8, 3), np.uint8))
    p = Project.create(tmp_path / "검사 프로젝트", name="PCB 라인")
    p.add_class("스크래치")
    p.add_class("찍힘")
    p.add_images([src])
    p.save()
    return p


def test_main_window_opens(window):
    assert window.isVisible()
    assert window.windowTitle().startswith("DefectLab")


def test_layout_left_canvas_right(window):
    widgets = [window.main_splitter.widget(i) for i in range(window.main_splitter.count())]
    assert widgets == [window.left_splitter, window.canvas_panel, window.tabs]
    assert isinstance(window.tabs, QTabWidget)


def test_right_tabs_in_order(window):
    names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert names == [name for name, _ in TABS] == ["Augment", "Train", "Result", "Compare", "Export"]


def test_toolbar_contains_all_actions(window):
    actions = set(window.toolbar.actions())
    for act in [window.act_new, window.act_open, window.act_add_folder, window.act_save,
                window.act_undo, window.act_redo, *window.tool_actions.values()]:
        assert act in actions


@pytest.mark.parametrize("tool, key", [
    (Tool.SELECT, "V"), (Tool.BBOX, "B"), (Tool.POLYGON, "P"),
    (Tool.CROP_ROI, "C"), (Tool.SAM, "S"), (Tool.ERASER, "E"),
])
def test_tool_shortcuts(window, tool, key):
    assert window.tool_actions[tool].shortcut().toString() == key


def test_tools_are_exclusive_and_emit(window):
    seen = []
    window.tool_changed.connect(seen.append)

    window.tool_actions[Tool.BBOX].trigger()
    window.tool_actions[Tool.POLYGON].trigger()

    assert window.current_tool == Tool.POLYGON
    assert [a for a in window.tool_actions.values() if a.isChecked()] == [window.tool_actions[Tool.POLYGON]]
    assert seen == [Tool.BBOX, Tool.POLYGON]
    assert "Polygon" in window.status_tool.text()


def test_project_actions_disabled_without_project(window):
    assert not window.act_add_folder.isEnabled()
    assert not window.act_save.isEnabled()
    assert window.act_open.isEnabled() and window.act_new.isEnabled()


def test_load_project_fills_lists_and_title(window, project):
    window.load_project(project)

    assert [window.image_list.item(i).text() for i in range(window.image_list.count())] == [
        "a.png\n미라벨", "b.png\n미라벨", "c.jpg\n미라벨"]
    assert [window.class_list.item(i).text() for i in range(window.class_list.count())] == ["1: 스크래치", "2: 찍힘"]
    assert window.windowTitle().startswith("PCB 라인 — DefectLab")
    assert window.status_image.text() == "이미지 1/3"
    assert window.act_save.isEnabled()


def test_image_selection_updates_status(window, project):
    window.load_project(project)
    window.image_list.setCurrentRow(2)
    assert window.status_image.text() == "이미지 3/3"


def test_add_folder_marks_dirty_and_save_clears(window, project, tmp_path):
    window.load_project(project)
    more = tmp_path / "추가 폴더"
    more.mkdir()
    imwrite_unicode(more / "d.bmp", np.zeros((4, 4, 3), np.uint8))

    assert window.add_image_folder(more) == 1
    assert window.image_list.count() == 4
    assert window.windowTitle().startswith("PCB 라인 * —")

    window.save_project()
    assert not window.project_dirty
    assert len(Project.open(project.root).config.images) == 4


def test_splitter_layout_persists(qapp, settings, tmp_path):
    w = MainWindow(settings, cache_root=tmp_path)
    w.show()
    qapp.processEvents()
    w.main_splitter.setSizes([400, 700, 500])
    qapp.processEvents()
    saved, size = w.main_splitter.sizes(), w.size()
    w.close()

    w2 = MainWindow(settings, cache_root=tmp_path)
    w2.resize(size)  # the offscreen test platform ignores restoreGeometry
    w2.show()
    qapp.processEvents()
    assert w2.main_splitter.sizes() == saved
    w2.close()


def test_cursor_position_in_status(window):
    window.set_cursor_position(12, 99)
    assert window.status_coords.text() == "x 12, y 99"
    window.set_cursor_position(None, None)
    assert window.status_coords.text() == ""


def test_detect_device_returns_label():
    assert detect_device().startswith(("GPU", "CPU"))


def test_self_check_passes(capsys):
    """`DefectLab --self-check` (used by the build scripts and the README install checklist)."""
    from defect_inspection.platform_utils import pick_korean_font
    from defect_inspection.selfcheck import run_self_check

    code = run_self_check()
    out = capsys.readouterr().out
    assert "[OK]   메인 창" in out and "[OK]   모델 생성" in out and "[OK]   ONNX Runtime" in out
    if pick_korean_font():
        assert code == 0 and "모든 확인 통과" in out
    else:
        assert code == 1 and "[FAIL] 한글 폰트" in out


# ---- found on a real M1 (docs/mac_checklist_tested.md) -----------------------------------------

def test_menus_show_every_shortcut(window):
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    assert list(menus) == ["파일", "편집", "보기", "도구", "설정"]
    assert window.act_save in menus["파일"].actions() and window.act_new in menus["파일"].actions()
    assert window.act_undo in menus["편집"].actions() and window.act_redo in menus["편집"].actions()
    assert window.act_fit in menus["보기"].actions()
    assert set(window.tool_actions.values()) <= set(menus["도구"].actions())


def test_window_fits_a_13_inch_mac_screen(window, qapp):
    # the Augment tab alone wanted ~810 px; the window's minimum height was 893 > 870 px usable
    assert window.minimumSizeHint().height() < 700
    # 1366 px laptop screens; Windows' native style needed 1674 px before (fixed sliders, unwrapped
    # Compare-tab text, uncapped tab width). The parts are in the message for CI logs.
    parts = {"left": window.left_splitter.minimumSizeHint().width(),
             "canvas": window.canvas_panel.minimumSizeHint().width(),
             "tabs": window.tabs.minimumSizeHint().width(),
             **{n: p.minimumSizeHint().width() for n, p in window.panels.items()}}
    assert window.minimumSizeHint().width() < 1366, parts
    window.tabs.setCurrentWidget(window.augment_panel)
    assert window.tabs.currentWidget() is window.augment_panel
    assert window.tabs.indexOf(window.export_panel) == 4 and window.tabs.widget(4) is window.export_panel


def test_labeled_items_use_palette_text_colour(window, project):
    from PySide6.QtCore import Qt

    from defect_inspection.core.annotation import BBox

    ann = project.load_labels("a.png")
    ann.shapes.append(BBox(class_id=0, x1=1, y1=1, x2=4, y2=4))
    project.save_labels(ann)
    window.load_project(project)
    labeled, unlabeled = window.image_panel._item("a.png"), window.image_panel._item("b.png")
    # no forced colour: an invalid QColor() brush painted black text on the dark-mode list
    assert labeled.data(Qt.ItemDataRole.ForegroundRole) is None
    assert unlabeled.data(Qt.ItemDataRole.ForegroundRole) is not None


def test_tag_is_measured_with_the_painted_font(qapp):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QGraphicsRectItem

    from defect_inspection.ui.canvas.shape_items import TagItem

    tag = TagItem(QGraphicsRectItem(), "dent · 제안(sam 0.91)", QColor("red"), dashed=True)
    image = QImage(400, 40, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    painter.setFont(tag.font)
    painted = painter.boundingRect(QRectF(0, 0, 400, 40), Qt.AlignmentFlag.AlignVCenter, tag.text).width()
    painter.end()
    assert abs(tag.boundingRect().width() - 8 - painted) < 1.5


def test_tabs_scroll_sideways_where_scrollbars_are_visible(qapp, monkeypatch, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    from defect_inspection.ui import main_window as mw

    wide = QLabel("x" * 400)  # a panel much wider than the cap
    monkeypatch.setattr(mw, "OVERLAY_SCROLLBARS", False)  # Windows / Linux
    tabs = mw.ScrollTabs()
    tabs.addTab(wide, "W")
    area = tabs._areas[wide]
    assert area.minimumWidth() <= mw.SCROLL_TAB_MAX_MIN_WIDTH
    assert area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    monkeypatch.setattr(mw, "OVERLAY_SCROLLBARS", True)  # macOS: never clipped sideways
    tabs2 = mw.ScrollTabs()
    wide2 = QLabel("x" * 400)
    tabs2.addTab(wide2, "W")
    assert tabs2._areas[wide2].minimumWidth() >= wide2.minimumSizeHint().width()
    assert tabs2._areas[wide2].horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
