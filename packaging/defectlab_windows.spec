# PyInstaller spec — Windows 10/11 64-bit (folder build: dist\DefectLab\DefectLab.exe).
# Build: packaging\build_windows.ps1  (or: python -m PyInstaller --noconfirm packaging\defectlab_windows.spec)
# Two executables in one folder: DefectLab.exe (windowed, no console) and DefectLab-worker.exe
# (console) for training, because a windowed exe may get no usable stdout and the worker reports
# progress through stdout. The app starts it via platform_utils.frozen_worker_command; QProcess
# opens no console window for it.
import sys

sys.path.insert(0, SPECPATH)
from spec_common import APP_NAME, WORKER_NAME, collect  # noqa: E402

c = collect(SPECPATH)
a = Analysis([c["app_script"]], **c["analysis"])
w = Analysis([c["worker_script"]], **c["analysis"])
app_exe = EXE(PYZ(a.pure), a.scripts, [], exclude_binaries=True, name=APP_NAME, console=False, upx=False)
worker_exe = EXE(PYZ(w.pure), w.scripts, [], exclude_binaries=True, name=WORKER_NAME, console=True, upx=False)
coll = COLLECT(app_exe, worker_exe, a.binaries, a.datas, w.binaries, w.datas, upx=False, name=APP_NAME)
