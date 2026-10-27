"""Raise the app version before a build: 0.1.0 -> 0.1.1 (patch), or --minor / --major.

The version lives in pyproject.toml and defect_inspection/__init__.py (window title, About,
bundle Info.plist, build file names); both are rewritten together.

    .venv/bin/python tools/bump_version.py           # prints the new version
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    ROOT / "pyproject.toml": re.compile(r'^(version = ")(\d+)\.(\d+)\.(\d+)(")', re.M),
    ROOT / "defect_inspection" / "__init__.py": re.compile(r'^(__version__ = ")(\d+)\.(\d+)\.(\d+)(")', re.M),
}


def bump(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--minor", action="store_true")
    group.add_argument("--major", action="store_true")
    args = parser.parse_args(argv)
    part = "major" if args.major else "minor" if args.minor else "patch"

    texts = {path: path.read_text(encoding="utf-8") for path in FILES}
    versions = set()
    for path, pattern in FILES.items():
        match = pattern.search(texts[path])
        if match is None:
            print(f"버전을 찾지 못했습니다: {path}", file=sys.stderr)
            return 1
        versions.add(".".join(match.group(2, 3, 4)))
    if len(versions) != 1:
        print(f"두 파일의 버전이 다릅니다: {sorted(versions)}", file=sys.stderr)
        return 1
    new = bump(versions.pop(), part)
    for path, pattern in FILES.items():
        path.write_text(pattern.sub(lambda m: f"{m.group(1)}{new}{m.group(5)}", texts[path], count=1), encoding="utf-8")
    print(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
