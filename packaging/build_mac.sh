#!/usr/bin/env bash
# Build the macOS app and disk image on an Apple Silicon Mac:
#   dist/DefectLab-<version>.app, dist/DefectLab-<version>-macos-arm64.dmg  (version: defect_inspection.__version__,
#   raised before each build with tools/bump_version.py; earlier versions are left in dist/)
#   packaging/build_mac.sh [python]   (default: .venv/bin/python)
# Needs: pip install torch torchvision && pip install -e ".[package]" in that environment.
#
# Signing (optional, environment variables):
#   DEVELOPER_ID="Developer ID Application: Name (TEAMID)"  sign with your certificate (hardened runtime)
#   NOTARY_PROFILE=<keychain profile>                       also notarize + staple (xcrun notarytool store-credentials)
# Without them the app gets an ad-hoc signature: it runs, but Gatekeeper warns on the first start
# on another Mac (README "Mac 설치본" explains how to open it).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-.venv/bin/python}"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' defect_inspection/__init__.py)"
[ -n "$VERSION" ] || { echo "defect_inspection/__init__.py에서 버전을 읽지 못했습니다"; exit 1; }
APP="dist/DefectLab-$VERSION.app"
DMG="dist/DefectLab-$VERSION-macos-arm64.dmg"

[ "$(uname -s)" = "Darwin" ] || { echo "macOS에서 실행하세요"; exit 1; }
[ "$(uname -m)" = "arm64" ] || echo "경고: Apple Silicon(arm64)이 아닙니다 — arm64 설치본을 만들 수 없습니다"
"$PY" -m PyInstaller --version >/dev/null || { echo "PyInstaller가 없습니다: $PY -m pip install -e '.[package]'"; exit 1; }
"$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging/defectlab_mac.spec
rm -rf "$APP"
mv dist/DefectLab.app "$APP"   # the spec builds DefectLab.app; keep each version side by side

if [ -n "${DEVELOPER_ID:-}" ]; then
  echo "== 서명: $DEVELOPER_ID =="
  codesign --force --deep --options runtime --timestamp --sign "$DEVELOPER_ID" "$APP"
else
  echo "== ad-hoc 서명 (개발자 인증서 없음) =="
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP" && echo "[OK]   서명 확인"

echo "== self-check (설치본) =="
QT_QPA_PLATFORM=offscreen "$APP/Contents/MacOS/DefectLab" --self-check
"$APP/Contents/MacOS/DefectLab" --train-worker --help >/dev/null && echo "[OK]   --train-worker"

echo "== 디스크 이미지 =="
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/DefectLab.app"   # installed as DefectLab.app: a new version replaces the old one
ln -s /Applications "$STAGE/Applications"   # drag the app onto it to install
rm -f "$DMG"
hdiutil create -volname "DefectLab" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
if [ -n "${DEVELOPER_ID:-}" ]; then
  codesign --force --sign "$DEVELOPER_ID" "$DMG"
fi
if [ -n "${DEVELOPER_ID:-}" ] && [ -n "${NOTARY_PROFILE:-}" ]; then
  echo "== 공증 (몇 분 걸림) =="
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi
echo "완료: $APP, $DMG"
