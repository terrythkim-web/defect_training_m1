# PyInstaller spec — Linux (folder build: dist/DefectLab/DefectLab).
# Build: packaging/build_linux.sh  (or: python -m PyInstaller --noconfirm packaging/defectlab_linux.spec)
# One executable: the training worker is the same program started with --train-worker.
import sys

sys.path.insert(0, SPECPATH)
from spec_common import APP_NAME, collect  # noqa: E402

c = collect(SPECPATH)
a = Analysis([c["app_script"]], **c["analysis"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, console=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
