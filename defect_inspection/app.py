"""Entry point: python -m defect_inspection.app (and the PyInstaller executables)."""

import multiprocessing
import sys


def main() -> int:
    # Frozen Windows builds: DataLoader worker processes re-run this executable; let
    # multiprocessing take over before anything else starts (no-op otherwise).
    multiprocessing.freeze_support()
    from defect_inspection.platform_utils import use_persistent_matplotlib_cache

    use_persistent_matplotlib_cache()  # before anything imports matplotlib (report, self-check, worker)
    if "--self-check" in sys.argv:  # verify an installation without a display (README checklist)
        from defect_inspection.selfcheck import run_self_check

        return run_self_check()
    if "--train-worker" in sys.argv:  # PyInstaller build: the app exe doubles as the training worker
        from defect_inspection.worker import main as worker_main

        args = [a for a in sys.argv[1:] if a != "--train-worker"]
        return worker_main(args)

    from PySide6.QtWidgets import QApplication

    from defect_inspection import APP_NAME
    from defect_inspection.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


def worker_entry() -> int:
    """Entry of the console worker executable of Windows builds (DefectLab-worker.exe)."""
    multiprocessing.freeze_support()
    from defect_inspection.platform_utils import use_persistent_matplotlib_cache

    use_persistent_matplotlib_cache()
    from defect_inspection.worker import main as worker_main

    return worker_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
