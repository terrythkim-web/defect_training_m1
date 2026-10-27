"""App version: one value in pyproject.toml and defect_inspection/__init__.py (tools/bump_version.py)."""

import re
import sys
from pathlib import Path

import defect_inspection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import bump_version  # noqa: E402


def test_pyproject_and_package_agree():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "(.+)"', pyproject, re.M).group(1) == defect_inspection.__version__


def test_bump():
    assert bump_version.bump("0.1.9", "patch") == "0.1.10"
    assert bump_version.bump("0.1.9", "minor") == "0.2.0"
    assert bump_version.bump("0.1.9", "major") == "1.0.0"


def test_bump_rewrites_both_files(tmp_path, monkeypatch, capsys):
    toml, init = tmp_path / "pyproject.toml", tmp_path / "__init__.py"
    toml.write_text('[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")
    init.write_text('APP = "x"\n__version__ = "1.2.3"\n', encoding="utf-8")
    monkeypatch.setattr(bump_version, "FILES", {toml: bump_version.FILES[ROOT / "pyproject.toml"],
                                                init: bump_version.FILES[ROOT / "defect_inspection" / "__init__.py"]})
    assert bump_version.main([]) == 0 and capsys.readouterr().out.strip() == "1.2.4"
    assert 'version = "1.2.4"' in toml.read_text(encoding="utf-8")
    assert '__version__ = "1.2.4"' in init.read_text(encoding="utf-8")
    init.write_text('__version__ = "9.9.9"\n', encoding="utf-8")
    assert bump_version.main([]) == 1  # out of sync: refuse
