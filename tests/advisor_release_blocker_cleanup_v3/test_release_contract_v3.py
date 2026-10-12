"""Release-blocker regression tests for the RF-v3 advisor package."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from mdcorealign.cli import build_parser, run

ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"


@pytest.fixture(autouse=True)
def _selected_distribution(request, tmp_path, monkeypatch, record_property) -> None:
    """Run the original assembly assertions on explicitly selected real artifacts."""
    if request.node.originalname not in {
        "test_shipped_tests_do_not_import_excluded_mdstablecore",
        "test_assembled_distribution_is_complete_when_present",
    }:
        return
    wheel = os.environ.get("MDCOREALIGN_WHEEL_UNDER_TEST")
    sdist = os.environ.get("MDCOREALIGN_SDIST_UNDER_TEST")
    if not wheel and not sdist:
        return  # Preserve the original assembled-checkout fallback.
    if not wheel or not sdist:
        raise pytest.UsageError("Assembly checks require both selected wheel and sdist paths")
    assembly = tmp_path / "selected_distribution"
    source = assembly / "source"
    source.mkdir(parents=True)
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        roots = set()
        destinations = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError(f"Unsafe source archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Non-regular source archive member: {member.name}")
            roots.add(path.parts[0])
            relative = Path(*path.parts[1:])
            if relative in destinations:
                raise ValueError(f"Duplicate source archive path: {member.name}")
            destinations.add(relative)
        if len(roots) != 1:
            raise ValueError("The selected sdist must contain one source root")
        for member in members:
            destination = source.joinpath(*PurePosixPath(member.name).parts[1:])
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable source archive member: {member.name}")
                with stream, destination.open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
    dist = assembly / "dist"
    dist.mkdir()
    for artifact in (Path(wheel), Path(sdist)):
        shutil.copyfile(artifact, dist / artifact.name)
        record_property(f"selected_{'wheel' if artifact.suffix == '.whl' else 'sdist'}", str(artifact))
    record_property("assembly_source", "selected_sdist_extracted_to_temporary_root")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", source)


def _align(tmp_path: Path, *extra: str) -> Path:
    output = tmp_path / ("run_" + str(len(list(tmp_path.iterdir()))))
    options = build_parser().parse_args(["align", "--input", str(TOY), *extra])
    mode = options.contract or options.mode
    representative_flag = (
        "--exact-representative-count" if mode == "exact" else "--fast-representative-count"
    )
    arguments = [
        "align",
        "--input",
        str(TOY),
        "--input-format",
        "ca-pdb",
        representative_flag,
        "5",
        "--output-dir",
        str(output),
        *extra,
    ]
    assert run(arguments) == 0
    return output


def test_default_is_canonical_throughput() -> None:
    args = build_parser().parse_args(["align", "--input", str(TOY)])
    assert args.mode == "throughput"
    assert args.backend == "cpu"


def test_default_throughput_metadata_and_layout(tmp_path: Path) -> None:
    output = _align(tmp_path)
    assert {path.name for path in output.iterdir()} == {
        "diagnostics",
        "flexible_segments.tsv",
        "frame_metrics.tsv",
        "residue_profile.tsv",
        "summary.json",
        "transforms.npz",
    }
    assert {path.name for path in (output / "diagnostics").iterdir()} == {
        "chain_mapping_report.tsv",
        "diagnostics.json",
        "input_audit.tsv",
            "run_metadata.json",
            "selected_frame_indices.tsv",
            "source_manifest.json",
            "timing.json",
    }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["method"] == "fixed-budget continuous-weight alignment"
    assert summary["mode"] == "throughput"
    assert summary["execution_contract"] == "throughput"
    assert summary["experimental_mode"] is False
    assert summary["converged"] is None
    assert summary["converged_fraction"] is None
    assert summary["convergence_interpretation"] == "not_applicable_fixed_budget"
    assert summary["quality_status"] == "completed"
    assert summary["fixed_budget_estimator"] is True
    assert summary["quality_audit_performed"] is False
    assert "fast_global_confidence" not in summary
    assert not (output / "weights.csv").exists()
    assert not (output / "fast_global_confidence.json").exists()
    for relative in summary["primary_outputs"].values():
        assert (output / relative).is_file()
    for relative in summary["diagnostic_outputs"].values():
        assert (output / relative).is_file()


def test_exact_has_real_convergence_metadata(tmp_path: Path) -> None:
    output = _align(tmp_path, "--mode", "exact")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["mode"] == "exact"
    assert isinstance(summary["converged_fraction"], float)
    assert isinstance(summary["converged"], bool)
    assert summary["fixed_budget_estimator"] is False


def test_optional_local_deformation_stays_in_diagnostics(tmp_path: Path) -> None:
    output = _align(tmp_path, "--local-deformation")
    diagnostics = output / "diagnostics"
    assert (diagnostics / "local_deformation.tsv").is_file()
    assert (diagnostics / "locally_deforming_segments.tsv").is_file()
    assert not (output / "local_deformation.tsv").exists()


def test_ordinary_rmsf_is_opt_in(tmp_path: Path) -> None:
    basic = _align(tmp_path)
    enabled = _align(tmp_path, "--ordinary-rmsf")
    assert "ordinary_RMSF" not in (basic / "residue_profile.tsv").read_text().splitlines()[0]
    assert "ordinary_RMSF" in (enabled / "residue_profile.tsv").read_text().splitlines()[0]


def test_legacy_artifacts_are_explicit(tmp_path: Path) -> None:
    output = _align(tmp_path, "--legacy-artifacts")
    assert (output / "weights.csv").is_file()
    assert (output / "input_audit.json").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["compatibility_outputs"]["weights_csv"] == "weights.csv"


def test_basic_help_hides_research_modes(capsys) -> None:
    try:
        build_parser().parse_args(["align", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    help_text = capsys.readouterr().out
    assert "{throughput,exact}" in help_text
    assert "--ordinary-rmsf" in help_text
    assert "--local-deformation" in help_text
    assert "fast-global" not in help_text
    assert "balanced-lite" not in help_text
    assert "--gpu-rotation-solver" not in help_text


def test_advanced_help_keeps_legacy_discoverable(capsys) -> None:
    try:
        build_parser().parse_args(["align", "--help-advanced"])
    except SystemExit as exc:
        assert exc.code == 0
    help_text = capsys.readouterr().out
    assert "fast-global" in help_text
    assert "balanced-lite" in help_text
    assert "docs/LEGACY_AND_ADVANCED.md" in help_text


def test_shipped_tests_do_not_import_excluded_mdstablecore() -> None:
    if not (ROOT.parent / "dist").is_dir():
        pytest.skip("shipped-test import policy is checked in assembled packages")
    import_pattern = re.compile(r"^\s*(?:from|import)\s+mdstablecore\b", re.MULTILINE)
    offenders = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if import_pattern.search(path.read_text()):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_assembled_distribution_is_complete_when_present() -> None:
    package_root = ROOT.parent
    dist = package_root / "dist"
    if not dist.is_dir():
        pytest.skip("distribution completeness is checked in assembled packages")

    wheels = sorted(dist.glob("mdcorealign-*.whl"))
    sdists = sorted(dist.glob("mdcorealign-*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.namelist()
    assert any(name.startswith("mdcorealign/") for name in members)
    assert any(name.startswith("mdsuperpose/") for name in members)
    assert not any(name.startswith("mdstablecore/") for name in members)

    for name in ["INSTALL.md", "QUICK_START.md", "source/pyproject.toml"]:
        candidate = package_root / name
        if package_root.name == "MDCoreAlign_supervisor_test_package":
            assert candidate.is_file()
