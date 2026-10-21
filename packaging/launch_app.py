"""PyInstaller entry of the DefectLab app (also the training worker on Linux: --train-worker)."""

import sys

from defect_inspection.app import main

if __name__ == "__main__":  # required: Windows spawns child processes by re-running the executable
    sys.exit(main())
