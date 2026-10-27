---
name: versioned-app-builds
description: "Every DefectLab app build gets a bumped version in its name, and ask before deleting older builds"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 72db448d-c62c-4a5b-8199-93e46bc192cb
  modified: 2026-10-27T03:37:11.099Z
---

Before every app build, raise the patch version with `python tools/bump_version.py` (0.1.1 → 0.1.2 …; `.venv/bin/python` on the Mac, plain `python` in the Docker dev container) and commit the bump with the build's changes. The build scripts name their outputs after the version:
- macOS `packaging/build_mac.sh` → `dist/DefectLab-<version>.app`, `dist/DefectLab-<version>-macos-arm64.dmg`
- Linux `packaging/build_linux.sh` → `dist/DefectLab/` + `dist/DefectLab-<version>-linux-<arch>.tar.gz`

After the build, list the older builds in dist/ (earlier versioned files, and old unversioned ones such as `DefectLab-macos-arm64.dmg` / `DefectLab-linux-<arch>.tar.gz`) and ask the user whether to delete them — never delete without asking.

**Why:** the user asked (2026-10-23): "앞으로 앱을 만들때, 버젼을 뒤에 부쳐주고, 이전버젼을 지울지 물어봐줘". They chose "빌드마다 버전 올리기" over date or commit-hash suffixes.

**How to apply:** whenever the user says 빌드/재빌드/커밋 푸시 빌드 — bump, commit, build, then ask about deleting previous versions (AskUserQuestion, multi-select of the old files).
