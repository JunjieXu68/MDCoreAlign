"""Release gates for the public-stable polish candidate."""

from __future__ import annotations

import csv
import inspect
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import mdcorealign
from mdcorealign.cli import run
from mdcorealign.output_schema import FRAME_METRIC_COLUMNS, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
PUBLIC_API = [
    "ArrayCoordinateSource",
    "ContractExecutionResult",
    "ExactAlignmentResult",
    "ExportAtomRecord",
    "ExportStatistics",
    "GPUAlignmentResult",
    "InputReadiness",
    "MDAnalysisCoordinateSource",
    "SCHEMA_VERSION",
    "__version__",
    "build_coordinate_writer",
    "classify_input",
    "gpu_available",
    "gpu_environment",
    "parse_frame_indices",
    "resolve_export_contract",
    "solve_contract",
    "solve_exact_contract",
    "stream_apply_transforms",
]


def _public_text_files() -> list[Path]:
    roots = [ROOT / "src" / "mdcorealign", ROOT / "examples", ROOT / ".github"]
    paths = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "CITATION.cff"]
    paths.extend(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".sh", ".json", ".tsv"}
    )
    paths.extend(
        path
        for path in (ROOT / "docs").glob("*.md")
        if not path.name.startswith("CODEX_")
        and path.name != "private_alpha_advisor_test.md"
    )
    return sorted(set(paths))


def test_public_namespace_matches_snapshot() -> None:
    """The root namespace must remain explicit, small and reproducible."""

    assert mdcorealign.__all__ == PUBLIC_API
    assert len(mdcorealign.__all__) < 25
    assert "solve_fast_global_alignment" not in mdcorealign.__all__
    assert "run_mdlovofit_on_files" not in mdcorealign.__all__


def test_public_functions_are_documented_and_typed() -> None:
    """Every stable public function must have an English typed signature."""

    for name in mdcorealign.__all__:
        value = getattr(mdcorealign, name)
        if not inspect.isfunction(value):
            continue
        assert inspect.getdoc(value), name
        assert not CJK.search(inspect.getdoc(value) or ""), name
        signature = inspect.signature(value)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ), name


def test_shipped_public_text_and_filenames_are_english_only() -> None:
    """Public product files must not contain CJK characters."""

    offenders: list[str] = []
    for path in _public_text_files():
        relative = path.relative_to(ROOT).as_posix()
        if CJK.search(relative):
            offenders.append(f"filename:{relative}")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if CJK.search(text):
            offenders.append(f"content:{relative}")
    assert offenders == []


def test_canonical_throughput_schema_is_consistent(tmp_path: Path) -> None:
    """All canonical JSON records must agree on product terminology."""

    output = tmp_path / "analysis"
    assert run(
        [
            "align",
            "--input",
            str(TOY),
            "--input-format",
            "ca-pdb",
            "--mode",
            "throughput",
            "--backend",
            "cpu",
            "--output-dir",
            str(output),
        ]
    ) == 0
    records = [
        json.loads((output / "summary.json").read_text()),
        json.loads((output / "diagnostics" / "diagnostics.json").read_text()),
        json.loads((output / "diagnostics" / "run_metadata.json").read_text()),
    ]
    expected = {
        "method": "fixed-budget continuous-weight alignment",
        "mode": "throughput",
        "execution_contract": "throughput",
        "experimental_mode": False,
    }
    for record in records:
        assert {key: record[key] for key in expected} == expected
        assert record["schema_version"] == SCHEMA_VERSION
        assert record["software_version"] == mdcorealign.__version__
    timing = json.loads((output / "diagnostics" / "timing.json").read_text())
    assert {key: timing[key] for key in expected} == expected
    assert timing["schema_version"] == SCHEMA_VERSION
    assert timing["software_version"] == mdcorealign.__version__
    assert set(timing["timing"]) == {
        "weight_estimation_seconds",
        "all_frame_fit_seconds",
        "rmsd_seconds",
        "rmsf_seconds",
        "diagnostics_seconds",
        "coordinate_materialization_seconds",
        "end_to_end_seconds",
    }
    assert "exact_all_frame_total" not in timing["timing"]
    assert "limited_global_refinement" not in timing["timing"]
    assert timing["legacy_timing_schema"] == "internal-v5"
    summary = records[0]
    assert "core_threshold" not in summary
    assert "n_iterations" not in summary
    assert summary["fixed_update_rounds"] == 4
    assert summary["input_readiness_recommended_action"].find("align/flexibility") < 0
    with (output / "frame_metrics.tsv").open(newline="") as handle:
        assert tuple(csv.DictReader(handle, delimiter="\t").fieldnames or ()) == FRAME_METRIC_COLUMNS


def test_basic_help_and_documented_commands_are_clean() -> None:
    """First-screen commands must run and avoid legacy product terminology."""

    commands = [
        ["--help"],
        ["align", "--help"],
        ["apply", "--help"],
        ["local-deformation", "--help"],
        ["doctor", "--help"],
        ["formats", "--help"],
    ]
    for arguments in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "mdcorealign", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        combined = completed.stdout + completed.stderr
        assert not CJK.search(combined)
        assert "private-alpha" not in combined.lower()
        if arguments in (["--help"], ["align", "--help"]):
            assert "fast-global" not in combined


def test_wheel_surface_when_built() -> None:
    """A built wheel must expose the product package but no benchmark package."""

    explicit_wheel = os.environ.get("MDCOREALIGN_WHEEL_UNDER_TEST")
    if explicit_wheel:
        wheel = Path(explicit_wheel)
        assert wheel.is_file(), f"Requested wheel does not exist: {wheel}"
    else:
        wheels = sorted((ROOT / "dist").glob("mdcorealign-*.whl"))
        if not wheels:
            return
        wheel = wheels[-1]
        if wheel.stat().st_mtime < (ROOT / "src/mdcorealign/output_schema.py").stat().st_mtime:
            return
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
    assert any(name == "mdcorealign/output_schema.py" for name in members)
    assert not any(name.startswith("benchmark/") for name in members)
    assert not any(name.startswith("tests/") for name in members)
    excluded = {
        "mdcorealign/benchmark_minimal.py",
        "mdcorealign/experimental_gpu_batch.py",
        "mdcorealign/experimental_stream.py",
        "mdcorealign/experimental_throughput_v2.py",
        "mdcorealign/guarded.py",
    }
    assert excluded.isdisjoint(members)
    expected_modules = {
        path.relative_to(ROOT / "src").as_posix()
        for package in ("mdcorealign", "mdsuperpose")
        for path in (ROOT / "src" / package).rglob("*.py")
    } - excluded
    actual_modules = {
        name for name in members
        if name.endswith(".py") and name.split("/", 1)[0] in {"mdcorealign", "mdsuperpose"}
    }
    assert actual_modules == expected_modules
