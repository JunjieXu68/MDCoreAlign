"""Release-contract tests for the mandatory-I/O product installation."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_mdanalysis_and_gemmi_are_base_dependencies() -> None:
    dependencies = _project_metadata()["dependencies"]
    assert "numpy>=1.24" in dependencies
    assert "MDAnalysis>=2.6" in dependencies
    assert "gemmi>=0.6.7" in dependencies


def test_legacy_io_extras_are_compatibility_noop_aliases() -> None:
    extras = _project_metadata()["optional-dependencies"]
    assert extras["io"] == []
    assert extras["external"] == []


def test_formats_reports_standard_io_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "from mdcorealign.cli import main; raise SystemExit(main())", "formats"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Standard runtime dependencies" in completed.stdout
    assert "available in standard installation" in completed.stdout
    assert "MDAnalysis via [external]" not in completed.stdout
    assert "Gemmi via [external]" not in completed.stdout


def test_apply_help_describes_standard_mmcif_and_xtc() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "from mdcorealign.cli import main; raise SystemExit(main())", "apply", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "available in the standard installation" in completed.stdout
    assert "optional external I/O" not in completed.stdout


def test_built_wheel_declares_mandatory_io_dependencies() -> None:
    configured_wheel = os.environ.get("MDCOREALIGN_WHEEL_UNDER_TEST")
    if configured_wheel is None:
        pytest.skip("set MDCOREALIGN_WHEEL_UNDER_TEST during the DI6 build validation")
    wheel = Path(configured_wheel)
    assert wheel.is_file()
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    requires = metadata.get_all("Requires-Dist", [])
    assert any(value.lower().startswith("mdanalysis") for value in requires)
    assert any(value.lower().startswith("gemmi") for value in requires)


def test_support_table_schema_remains_machine_readable(tmp_path: Path) -> None:
    output = tmp_path / "formats.tsv"
    subprocess.run(
        [sys.executable, "-c", "from mdcorealign.cli import main; raise SystemExit(main())", "formats", "--out", str(output)],
        cwd=ROOT,
        check=True,
    )
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows
    assert {"category", "format_or_pair", "status", "notes"}.issubset(rows[0])
