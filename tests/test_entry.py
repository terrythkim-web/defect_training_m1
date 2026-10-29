"""Entry points in app.py: --self-check, --train-worker (frozen builds) and the GUI start."""

from __future__ import annotations

import sys

import pytest

from defect_inspection import app


@pytest.fixture
def argv(monkeypatch):
    def set_argv(*args):
        monkeypatch.setattr(sys, "argv", ["DefectLab", *args])
    return set_argv


def test_self_check_flag(argv, monkeypatch):
    import defect_inspection.selfcheck as selfcheck

    monkeypatch.setattr(selfcheck, "run_self_check", lambda: 7)
    argv("--self-check")
    assert app.main() == 7


def test_train_worker_flag_passes_the_rest(argv, monkeypatch):
    import defect_inspection.worker as worker

    seen = []
    monkeypatch.setattr(worker, "main", lambda args: seen.append(args) or 0)
    argv("--train-worker", "--run-dir", "/tmp/run 1", "--resume")
    assert app.main() == 0
    assert seen == [["--run-dir", "/tmp/run 1", "--resume"]]


def test_worker_entry(argv, monkeypatch):
    import defect_inspection.worker as worker

    seen = []
    monkeypatch.setattr(worker, "main", lambda args: seen.append(args) or 3)
    argv("--run-dir", "r")
    assert app.worker_entry() == 3 and seen == [["--run-dir", "r"]]


def test_gui_start_shows_main_window(argv, monkeypatch):
    import PySide6.QtWidgets as widgets

    import defect_inspection.ui.main_window as mw

    events = []

    class FakeApp:
        def __init__(self, args):
            events.append(("app", list(args)))

        def setApplicationName(self, name):
            events.append(("name", name))

        def setOrganizationName(self, name):
            events.append(("org", name))

        def exec(self):
            return 5

    class FakeWindow:
        def show(self):
            events.append("show")

    monkeypatch.setattr(widgets, "QApplication", FakeApp)
    monkeypatch.setattr(mw, "MainWindow", FakeWindow)
    argv()
    assert app.main() == 5
    assert events == [("app", ["DefectLab"]), ("name", "DefectLab"), ("org", "DefectLab"), "show"]
