"""PyInstaller entry of DefectLab-worker.exe (Windows console training worker)."""

import sys

from defect_inspection.app import worker_entry

if __name__ == "__main__":
    sys.exit(worker_entry())
