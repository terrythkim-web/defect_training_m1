#!/usr/bin/env bash
# Build the Linux folder distribution and check it: dist/DefectLab/ + dist/DefectLab-<version>-linux-<arch>.tar.gz
#   packaging/build_linux.sh [python]   (default: .venv/bin/python, else python — e.g. the Docker dev container)
# Needs: pip install -e ".[package]" in that environment (PyInstaller).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
fi
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' defect_inspection/__init__.py)"
TAR="dist/DefectLab-$VERSION-linux-$(uname -m).tar.gz"

"$PY" -m PyInstaller --version >/dev/null || { echo "PyInstaller가 없습니다: $PY -m pip install -e '.[package]'"; exit 1; }
"$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging/defectlab_linux.spec

echo "== self-check (설치본) =="
QT_QPA_PLATFORM=offscreen dist/DefectLab/DefectLab --self-check
echo "== training worker 진입 확인 =="
dist/DefectLab/DefectLab --train-worker --help >/dev/null && echo "[OK]   --train-worker"

tar -C dist -czf "$TAR" DefectLab
echo "완료: dist/DefectLab/DefectLab, $TAR"
