#!/usr/bin/env python3
"""Build the RF-v3 advisor package without touching scientific evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "advisor_release_blocker_cleanup_v3"
SOURCE = OUT / "02_packaging" / "release_source"
DIST = OUT / "02_packaging" / "dist"
PACKAGE = OUT / "09_package" / "MDCoreAlign_supervisor_test_package"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = datetime.now().strftime("%Y%m%d")

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}


def write(path: Path, text: str) -> None:
    """Write a normalized UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def copy_file(source: Path, target: Path) -> None:
    """Copy one file with metadata and parent creation."""

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    """Copy a source tree while excluding generated caches and distributions."""

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name.endswith("_offsets.npz") or path.name.endswith("_offsets.lock"):
            continue
        copy_file(path, target / relative)


def copy_distribution_artifacts(source: Path, target: Path) -> list[Path]:
    """Copy tested wheel/sdist artifacts through a dedicated allow-list path."""

    artifacts = sorted(source.glob("*.whl")) + sorted(source.glob("*.tar.gz"))
    if not any(path.suffix == ".whl" for path in artifacts):
        raise RuntimeError("no tested wheel found")
    if not any(path.name.endswith(".tar.gz") for path in artifacts):
        raise RuntimeError("no tested sdist found")
    copied: list[Path] = []
    for artifact in artifacts:
        destination = target / artifact.name
        copy_file(artifact, destination)
        copied.append(destination)
    return copied


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, output: Path) -> int:
    """Write deterministic size and SHA-256 records for a payload tree."""

    rows: list[tuple[str, int, str]] = []
    excluded_names = {"file_manifest.tsv", "sha256_manifest.tsv"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded_names:
            rows.append(
                (
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    sha256(path),
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)
    return len(rows)


def selected_tests() -> list[Path]:
    """Return the shipped product tests, excluding historical mdstablecore tests."""

    root_names = {
        "test_fast_kabsch_equivalence.py",
        "test_formats_cli.py",
        "test_input_format_policy.py",
        "test_mdcorealign_release_readiness.py",
        "test_mdsuperpose_alignment.py",
        "test_mdsuperpose_contacts.py",
        "test_mdsuperpose_core_selection.py",
        "test_mdsuperpose_export.py",
        "test_mdsuperpose_flexibility.py",
        "test_mdsuperpose_protein_input.py",
        "test_mdsuperpose_robust.py",
        "test_mdsuperpose_smoke.py",
        "test_mdsuperpose_trajectory_mode.py",
        "test_mdsuperpose_weighted_alignment.py",
        "test_mdsuperpose_weights.py",
        "test_non_protein_policy.py",
        "test_product_contract_cli.py",
    }
    directories = {
        "advisor_release_candidate_v2",
        "advisor_release_blocker_cleanup_v3",
        "purpose_based_coordinate_export_v1",
        "cpu_optimisation_v1",
        "throughput_gpu_v1",
    }
    selected: list[Path] = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        relative = path.relative_to(ROOT / "tests")
        if path.name in root_names or relative.parts[0] in directories:
            selected.append(path)
    return selected


def prepare_source() -> None:
    """Create one clean installable source copy for all subsequent validation."""

    if SOURCE.exists():
        raise SystemExit(f"refusing to overwrite prepared source: {SOURCE}")
    for name in [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "CHANGELOG.md",
    ]:
        copy_file(ROOT / name, SOURCE / name)
    for package in ["mdcorealign", "mdsuperpose"]:
        copy_tree(ROOT / "src" / package, SOURCE / "src" / package)
    docs = [
        "installation.md",
        "cpu_installation.md",
        "gpu_installation.md",
        "quickstart.md",
        "recommended_workflow.md",
        "cli_reference.md",
        "modes_and_quality_contracts.md",
        "LEGACY_AND_ADVANCED.md",
        "python_api.md",
        "input_formats.md",
        "support_matrix.md",
        "coordinate_export.md",
        "output_schema.md",
        "output_formats.md",
        "xtc_output.md",
        "mmcif_output.md",
        "pdb_compatibility.md",
        "md_preprocessing.md",
        "rmsf_flexible_segments.md",
        "local_deformation.md",
        "troubleshooting.md",
        "user_testing_protocol.md",
        "release.md",
        "mdlovofit_comparison.md",
        "non_protein_systems.md",
    ]
    for name in docs:
        source = ROOT / "docs" / name
        if source.exists():
            copy_file(source, SOURCE / "docs" / name)
    copy_tree(ROOT / "examples" / "toy_md", SOURCE / "examples" / "toy_md")
    v2_examples = ROOT / "results" / "advisor_release_candidate_v2" / "04_examples"
    for name in [
        "01_gromacs_xtc_cpu",
        "02_pdb_xtc_cpu",
        "03_multimodel_pdb",
        "04_exact_audit",
        "05_gpu",
        "06_local_deformation",
        "07_mmcif_export",
        "08_xtc_export",
        "09_pdb_compatibility",
    ]:
        copy_tree(v2_examples / name, SOURCE / "examples" / name)
    for source, destination in [
        (ROOT / "scripts" / "advisor_compare_throughput_exact.py", SOURCE / "examples" / "compare_throughput_exact.py"),
        (ROOT / "scripts" / "run_advisor_smoke_test.sh", SOURCE / "run_advisor_smoke_test.sh"),
        (ROOT / "scripts" / "run_gpu_smoke_test.sh", SOURCE / "run_gpu_smoke_test.sh"),
    ]:
        copy_file(source, destination)
    for path in selected_tests():
        copy_file(path, SOURCE / "tests" / path.relative_to(ROOT / "tests"))
    write(
        SOURCE / "THIRD_PARTY_NOTICES.md",
        """# Third-party notices

NumPy is the base dependency. MDAnalysis and Gemmi are optional external-I/O
dependencies; CuPy is an optional CUDA backend. Their source and binaries are
not bundled. MDLovoFit and THESEUS were external research comparators and are
not redistributed. The base package remains CPU-only and NumPy-only.
""",
    )
    write(
        OUT / "03_tests" / "legacy_test_decision.md",
        """# Legacy test decision

The historical files `tests/test_alignment.py`,
`tests/test_soft_superposition.py`, and `tests/test_synthetic_pipeline.py`
exercise the non-distributed `mdstablecore` research namespace. They remain in
the working tree and canonical research suite, but are intentionally excluded
from the shipped product suite. No broad pytest ignore rule is used; all
shipped tests target distributed `mdcorealign`/`mdsuperpose` APIs.
""",
    )
    write_manifest(SOURCE, OUT / "02_packaging" / "release_source_manifest.tsv")


def advisor_documents() -> dict[str, str]:
    """Return concise advisor entry documents with one consistent workflow."""

    return {
        "00_START_HERE.md": """# Start Here

Install the tested wheel from `dist/`, then run `bash
examples/run_advisor_smoke_test.sh`. Production analysis is `mdcorealign
align`; omitting `--mode` selects throughput CPU. Use `--mode exact` only for
the slower chronological numerical audit. Open `summary.json` first.

Analysis is compact. Use `mdcorealign apply` later for mmCIF structures, XTC
trajectories or compatibility PDB. No scientific benchmark was rerun for this
release cleanup.
""",
        "PROJECT_OVERVIEW.md": """# Project Overview

MDCoreAlign estimates continuous stable-residue weights and then performs one
explicit weighted rigid fit for every trajectory frame. Throughput is the
production weight estimator; sequential exact is the numerical audit path.
W5 is an optional alignment-free local-deformation diagnostic.
""",
        "CURRENT_EVIDENCE.md": """# Current Evidence

Frozen evidence only is included. The primary external runtime statement uses
the matched no-materialisation C1 workflow: the same centred CA multi-model PDB
is supplied and neither method writes aligned coordinates. MDCoreAlign and
MDLovoFit identify related but non-identical, method-dependent stable-reference
solutions. No universal-fastest or biological-equivalence claim is made.
""",
        "ADVISOR_TEST_PLAN.md": """# Advisor Test Plan

1. Install the wheel from `dist/`.
2. Run `bash examples/run_advisor_smoke_test.sh`.
3. Inspect throughput `summary.json` and `residue_profile.tsv`.
4. Compare explicit exact and optional W5 outputs.
5. Test mmCIF, XTC and PDB export and record usability feedback.
""",
        "ADVISOR_FEEDBACK_TEMPLATE.md": """# Advisor Feedback

- Installation environment:
- Command tested:
- Input type:
- Outputs inspected:
- Scientific interpretation questions:
- Errors or usability friction:
""",
        "INSTALL_CPU.md": (SOURCE / "docs" / "cpu_installation.md").read_text(),
        "INSTALL_GPU.md": (SOURCE / "docs" / "gpu_installation.md").read_text(),
        "QUICK_START.md": (SOURCE / "docs" / "quickstart.md").read_text(),
        "INPUT_FORMATS.md": (SOURCE / "docs" / "input_formats.md").read_text(),
        "COORDINATE_EXPORT.md": (SOURCE / "docs" / "coordinate_export.md").read_text(),
        "OUTPUTS_AND_INTERPRETATION.md": (SOURCE / "docs" / "output_formats.md").read_text(),
        "ALGORITHM_IMPLEMENTATION_MAP.md": """# Algorithm Implementation Map

- CLI and product output contract: `source/src/mdcorealign/cli.py`
- Exact/throughput dispatch: `source/src/mdcorealign/contracts.py`
- CPU all-frame fitting: `source/src/mdsuperpose/all_frame_engine.py`
- GPU backend: `source/src/mdcorealign/gpu_backend.py`
- W5 local deformation: `source/src/mdsuperpose/window_kinematics.py`
- Streaming coordinate application: `source/src/mdcorealign/coordinate_export.py`
""",
        "TROUBLESHOOTING.md": (SOURCE / "docs" / "troubleshooting.md").read_text(),
        "KNOWN_LIMITATIONS.md": """# Known Limitations

- PBC/nojump and centering remain upstream responsibilities.
- RMSF and stable-reference interpretation are reference-aware.
- Exact is a numerical reference, not biological ground truth.
- W5 is dimensionless local deformation, not RMSD or a mature hard domain label.
- Formal validation is centred on protein CA workflows.
- External tools are not bundled.
""",
        "LEGACY_AND_ADVANCED.md": (SOURCE / "docs" / "LEGACY_AND_ADVANCED.md").read_text(),
        "THIRD_PARTY_NOTICES.md": (SOURCE / "THIRD_PARTY_NOTICES.md").read_text(),
    }


def assemble_package() -> None:
    """Assemble the advisor payload, including the exact tested distributions."""

    if PACKAGE.exists():
        raise SystemExit(f"refusing to overwrite package staging: {PACKAGE}")
    if not SOURCE.exists():
        raise SystemExit("prepare_source has not run")
    copy_tree(SOURCE, PACKAGE / "source")
    artifacts = copy_distribution_artifacts(DIST, PACKAGE / "dist")
    copy_tree(SOURCE / "examples", PACKAGE / "examples")
    copy_file(SOURCE / "run_advisor_smoke_test.sh", PACKAGE / "examples" / "run_advisor_smoke_test.sh")
    copy_file(SOURCE / "run_gpu_smoke_test.sh", PACKAGE / "examples" / "run_gpu_smoke_test.sh")
    copy_tree(SOURCE / "tests", PACKAGE / "tests")
    copy_tree(SOURCE / "docs", PACKAGE / "docs")
    for name, content in advisor_documents().items():
        write(PACKAGE / name, content)
    for source, target_name in [
        (OUT / "00_freeze" / "source_manifest_before.tsv", "source_manifest.tsv"),
        (ROOT / "results" / "advisor_release_candidate_v2" / "00_freeze" / "frozen_evidence_registry.tsv", "frozen_evidence_registry.tsv"),
    ]:
        copy_file(source, PACKAGE / target_name)
    copy_tree(OUT / "08_validation", PACKAGE / "validation")
    with (OUT / "02_packaging" / "dist_artifact_manifest.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["filename", "size_bytes", "sha256"])
        for artifact in artifacts:
            writer.writerow([artifact.name, artifact.stat().st_size, sha256(artifact)])
    copy_file(
        OUT / "00_freeze" / "environment_manifest.tsv",
        PACKAGE / "environment_manifest.tsv",
    )
    write_manifest(PACKAGE, PACKAGE / "file_manifest.tsv")
    copy_file(PACKAGE / "file_manifest.tsv", PACKAGE / "sha256_manifest.tsv")


def make_advisor_archive() -> Path:
    """Create the canonical advisor TAR.GZ from the assembled package."""

    archive_path = OUT / f"MDCoreAlign_supervisor_test_package_v3_{DATE}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(PACKAGE, arcname="MDCoreAlign_supervisor_test_package")
    return archive_path


def make_review_archive() -> Path:
    """Create the required compact RF-v3 review bundle."""

    if HANDOFF.exists() and any(HANDOFF.iterdir()):
        raise SystemExit(f"refusing to overwrite review handoff: {HANDOFF}")
    HANDOFF.mkdir(parents=True, exist_ok=True)
    mapping = {
        "final_release_blocker_cleanup_report.md": OUT / "08_validation" / "validation_summary.md",
        "phase_status.tsv": OUT / "phase_status.tsv",
        "key_metrics.tsv": OUT / "08_validation" / "key_metrics.tsv",
        "release_blockers.tsv": OUT / "01_reproduction" / "release_blockers.tsv",
        "release_blocker_checklist.tsv": OUT / "08_validation" / "release_blocker_checklist.tsv",
        "package_builder_fix.md": OUT / "02_packaging" / "package_builder_fix.md",
        "dist_artifact_manifest.tsv": OUT / "02_packaging" / "dist_artifact_manifest.tsv",
        "shipped_pytest_output.txt": OUT / "08_validation" / "full_test_output.txt",
        "legacy_test_decision.md": OUT / "03_tests" / "legacy_test_decision.md",
        "metadata_contract.md": OUT / "04_metadata_outputs" / "metadata_contract.md",
        "summary_before_after.tsv": OUT / "04_metadata_outputs" / "summary_before_after.tsv",
        "default_output_contract.tsv": OUT / "04_metadata_outputs" / "default_output_contract.tsv",
        "default_output_inventory_after.tsv": OUT / "04_metadata_outputs" / "default_output_inventory_after.tsv",
        "cli_design.md": OUT / "05_cli" / "cli_design.md",
        "help_before_after.txt": OUT / "05_cli" / "help_before_after.txt",
        "documentation_review.md": OUT / "06_docs" / "documentation_review.md",
        "contradiction_scan_before_after.tsv": OUT / "06_docs" / "contradiction_scan_before_after.tsv",
        "command_execution_audit.tsv": OUT / "06_docs" / "command_execution_audit.tsv",
        "static_quality_summary.md": OUT / "07_static_quality" / "static_quality_summary.md",
        "technical_debt.md": OUT / "07_static_quality" / "technical_debt.md",
        "wheel_module_classification.tsv": OUT / "07_static_quality" / "wheel_module_classification.tsv",
        "build_report.md": OUT / "08_validation" / "build_report.md",
        "package_contents.tsv": OUT / "08_validation" / "package_contents.tsv",
        "cpu_clean_install_report.md": OUT / "08_validation" / "cpu_clean_install_report.md",
        "external_io_clean_install_report.md": OUT / "08_validation" / "external_io_clean_install_report.md",
        "gpu_clean_install_report.md": OUT / "08_validation" / "gpu_clean_install_report.md",
        "scientific_regression.tsv": OUT / "08_validation" / "scientific_regression.tsv",
        "advisor_smoke_test_output.txt": OUT / "08_validation" / "advisor_smoke_test_output.txt",
        "validation_summary.md": OUT / "08_validation" / "validation_summary.md",
        "unified_source_diff.patch": OUT / "08_validation" / "unified_source_diff.patch",
    }
    write(
        HANDOFF / "00_README_FIRST.md",
        """# RF-v3 Review Bundle

RF0-RF10 repair the advisor release package and product presentation only.
Scientific algorithms, frozen benchmarks, thresholds and manuscript claims
were not changed. Start with `final_release_blocker_cleanup_report.md` and
`release_blocker_checklist.tsv`.
""",
    )
    for target, source in mapping.items():
        if not source.exists():
            raise RuntimeError(f"missing review input: {source}")
        copy_file(source, HANDOFF / target)
    for source, target in [
        (ROOT / "src" / "mdcorealign" / "cli.py", HANDOFF / "changed_source" / "src" / "mdcorealign" / "cli.py"),
        (ROOT / "src" / "mdcorealign" / "contracts.py", HANDOFF / "changed_source" / "src" / "mdcorealign" / "contracts.py"),
        (ROOT / "scripts" / "build_advisor_release_blocker_cleanup_v3.py", HANDOFF / "changed_source" / "scripts" / "build_advisor_release_blocker_cleanup_v3.py"),
        (ROOT / "tests" / "advisor_release_blocker_cleanup_v3" / "test_release_contract_v3.py", HANDOFF / "changed_tests" / "test_release_contract_v3.py"),
        (ROOT / "README.md", HANDOFF / "changed_docs" / "README.md"),
    ]:
        copy_file(source, target)
    for relative in [
        "test_product_contract_cli.py",
        "advisor_release_candidate_v2/test_advisor_product_contract.py",
        "test_mdcorealign_release_readiness.py",
        "cpu_fast_path_v2/test_parallel_weights.py",
        "cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
        "mainline_reintegration_v1/test_experimental_modes.py",
        "test_input_format_policy.py",
        "test_private_alpha_hardening.py",
    ]:
        source = ROOT / "tests" / relative
        copy_file(source, HANDOFF / "changed_tests" / relative)
    for name in [
        "quickstart.md",
        "recommended_workflow.md",
        "cli_reference.md",
        "modes_and_quality_contracts.md",
        "coordinate_export.md",
        "output_schema.md",
        "output_formats.md",
        "installation.md",
    ]:
        copy_file(ROOT / "docs" / name, HANDOFF / "changed_docs" / "docs" / name)
    individual_diffs = OUT / "08_validation" / "individual_diffs"
    if individual_diffs.exists():
        copy_tree(individual_diffs, HANDOFF / "individual_diffs")
    write_manifest(HANDOFF, HANDOFF / "file_manifest.tsv")
    copy_file(HANDOFF / "file_manifest.tsv", HANDOFF / "sha256_manifest.tsv")
    archive_path = OUT / f"advisor_release_blocker_cleanup_v3_review_bundle_{DATE}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(HANDOFF, arcname="handoff_to_chatgpt")
    return archive_path


def main() -> None:
    """Dispatch non-destructive package-building phases."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["prepare", "assemble", "advisor-archive", "review-archive"],
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_source()
    elif args.phase == "assemble":
        assemble_package()
    elif args.phase == "advisor-archive":
        print(make_advisor_archive())
    else:
        print(make_review_archive())


if __name__ == "__main__":
    main()
