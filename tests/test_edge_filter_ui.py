"""Edge filter UI (PRD F9, step 4): the settings box in the Result tab and the "테두리 검사" dialog."""

from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QApplication
from test_edge_filter import edge_project

from defect_inspection.core.edge_filter_config import EdgeFilterConfig
from defect_inspection.ui.edge_check_dialog import EdgeCheckDialog
from defect_inspection.ui.widgets.edge_filter_box import EdgeFilterBox


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_box_shows_and_edits_settings(qapp):
    box = EdgeFilterBox()
    box.set_config(EdgeFilterConfig(enabled=True, line_polarity="bright", along_px=150, touch_px=7))
    assert box.isChecked() and box.polarity.currentData() == "bright"
    assert (box.along.value(), box.touch.value()) == (150, 7)

    changes = []
    box.changed.connect(lambda: changes.append(box.config()))
    box.touch.setValue(9)
    box.setChecked(False)
    assert [c.touch_px for c in changes] == [9, 9] and changes[-1].enabled is False
    cfg = box.config()
    assert (cfg.line_polarity, cfg.along_px, cfg.flat_max) == ("bright", 150, EdgeFilterConfig().flat_max)


def test_set_config_does_not_emit(qapp):
    box = EdgeFilterBox()
    changes = []
    box.changed.connect(lambda: changes.append(1))
    box.set_config(EdgeFilterConfig(enabled=True, touch_px=3))
    assert changes == []


def test_box_keeps_settings_without_widgets(qapp):
    """Fields the box has no widget for (contrast, tilt, …) survive an edit."""
    box = EdgeFilterBox()
    box.set_config(EdgeFilterConfig(dark_contrast=33.0, max_tilt_deg=8.0))
    box.along.setValue(300)
    assert (box.config().dark_contrast, box.config().max_tilt_deg) == (33.0, 8.0)


def test_check_dialog_lists_labels_and_opens_images(qapp, tmp_path):
    p, path, _ = edge_project(tmp_path)
    dialog = EdgeCheckDialog(p, EdgeFilterConfig())
    assert wait_until(qapp, lambda: dialog.table.rowCount() == 2)
    assert "1 / 2" in dialog.summary.text()
    cells = [[dialog.table.item(r, c).text() for c in range(dialog.table.columnCount())] for r in range(2)]
    assert cells[0][0] == path and cells[0][1] == "crack"
    assert "통과" in cells[0][3] and "제외" in cells[1][3]
    opened = []
    dialog.open_image.connect(opened.append)
    dialog.table.cellDoubleClicked.emit(1, 0)
    assert opened == [path]
    dialog.close()


def test_check_dialog_only_failures(qapp, tmp_path):
    p, _, _ = edge_project(tmp_path)
    dialog = EdgeCheckDialog(p, EdgeFilterConfig())
    assert wait_until(qapp, lambda: dialog.table.rowCount() == 2)
    dialog.only_failed.setChecked(True)
    assert dialog.table.rowCount() == 1 and "제외" in dialog.table.item(0, 3).text()
    dialog.close()
