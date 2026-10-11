#!/usr/bin/env python3
"""Assemble the MDCoreAlign v5 single full-product release artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "single_full_product_v5"
SOURCE = OUT / "02_packaging" / "release_source"
DIST = OUT / "02_packaging" / "dist"
PACKAGE = OUT / "06_release" / "MDCoreAlign_supervisor_test_package"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = "20260829"

ROOT_FILES = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-tested.txt",
    "requirements-dev.txt",
]

USER_DOCS = [
    "LEGACY_AND_ADVANCED.md",
    "cli_reference.md",
    "container.md",
    "conversion_guidance.md",
    "coordinate_export.md",
    "gpu_installation.md",
    "input_formats.md",
    "installation.md",
    "large_trajectory_workflow.md",
    "local_deformation.md",
    "md_preprocessing.md",
    "mdlovofit_comparison.md",
    "mmcif_output.md",
    "modes_and_quality_contracts.md",
    "non_protein_systems.md",
    "output_formats.md",
    "output_schema.md",
    "pdb_compatibility.md",
    "python_api.md",
    "quickstart.md",
    "recommended_workflow.md",
    "release.md",
    "rmsf_flexible_segments.md",
    "support_matrix.md",
    "troubleshooting.md",
    "user_testing_protocol.md",
    "xtc_output.md",
]

SHIPPED_TESTS = [
    "advisor_release_blocker_cleanup_v3/test_release_contract_v3.py",
    "advisor_release_candidate_v2/test_advisor_product_contract.py",
    "cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
    "cpu_optimisation_v1/test_fast_kabsch_cpu_v1.py",
    "cpu_optimisation_v1/test_sequential_mean_cpu_v1.py",
    "mandatory_io_full_product_v4/test_mandatory_io_contract.py",
    "purpose_based_coordinate_export_v1/test_coordinate_export.py",
    "single_full_product_v5/test_single_product_contract.py",
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
    "throughput_gpu_v1/test_contracts.py",
    "throughput_gpu_v1/test_gpu_backend.py",
]

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "output",
}

VALIDATION_FILES = [
    "archive_advisor_smoke_output.txt",
    "archive_cpu_pip_check.txt",
    "archive_doctor_cpu.txt",
    "archive_doctor_gpu.txt",
    "archive_gpu_pip_check.txt",
    "archive_gpu_smoke_output.txt",
    "archive_manifest_verification.txt",
    "archive_only_negative_result.md",
    "archive_only_validation.md",
    "archive_shipped_pytest_output.txt",
    "archive_source_build_output.txt",
    "archive_source_twine_check.txt",
    "advisor_smoke_output.txt",
    "canonical_pytest_output.txt",
    "compileall_output.txt",
    "focused_test_output.txt",
    "gpu_parity_pytest_output.txt",
    "gpu_pip_check.txt",
    "gpu_smoke_output.txt",
    "io_focused_pytest_output.txt",
    "io_regression.tsv",
    "pip_check.txt",
    "release_builder_ruff_output.txt",
    "ruff_output.txt",
    "ruff_scope_note.md",
    "scientific_regression.tsv",
    "scientific_regression_output.txt",
    "security_scan.txt",
    "shipped_pytest_output.txt",
    "twine_check.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def include_file(path: Path) -> bool:
    return (
        path.is_file()
        and not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
        and ".xtc_offsets." not in path.name
        and not path.name.endswith("~")
    )


def copy_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if include_file(path):
            copy_file(path, target / path.relative_to(source))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def sanitize_text(path: Path) -> None:
    """Replace machine-local paths in copied review and validation text."""

    if path.suffix.lower() not in {
        "",
        ".cff",
        ".json",
        ".md",
        ".patch",
        ".py",
        ".sh",
        ".toml",
        ".tsv",
        ".txt",
    }:
        return
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return
    text = text.replace(str(ROOT), "<repository_root>")
    user_home = Path.home()
    text = text.replace(str(user_home / "miniconda3" / "bin" / "python"), "<canonical_python>")
    text = text.replace(str(user_home), "<user_home>")
    text = re.sub(r"/tmp/[A-Za-z0-9._/+-]+", "<temporary_path>", text)
    path.write_text(text)


def sanitize_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            sanitize_text(path)


def write_manifest(root: Path) -> None:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            continue
        if (
            any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in relative.parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        rows.append((path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))
    for name in ("file_manifest.tsv", "sha256_manifest.tsv"):
        with (root / name).open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["path", "size_bytes", "sha256"])
            writer.writerows(rows)


def prepare() -> None:
    if SOURCE.exists():
        raise SystemExit(f"refusing to overwrite release source: {SOURCE}")
    SOURCE.mkdir(parents=True)
    for name in ROOT_FILES:
        copy_file(ROOT / name, SOURCE / name)
    for package in ("mdcorealign", "mdsuperpose"):
        copy_tree(ROOT / "src" / package, SOURCE / "src" / package)
    for name in USER_DOCS:
        copy_file(ROOT / "docs" / name, SOURCE / "docs" / name)
    copy_tree(ROOT / "examples", SOURCE / "examples")
    for relative in SHIPPED_TESTS:
        copy_file(ROOT / "tests" / relative, SOURCE / "tests" / relative)
    write_manifest(SOURCE)


def advisor_documents() -> dict[str, str]:
    return {
        "00_START_HERE.md": """# Start Here

This archive contains one complete MDCoreAlign product. Install the tested
wheel from `dist/`, run `mdcorealign doctor`, then run `bash
examples/run_advisor_smoke_test.sh`. The standard wheel includes CPU analysis,
complete validated I/O and the GPU backend. GPU execution additionally needs a
compatible CuPy/CUDA runtime; it is not a separate MDCoreAlign edition.
""",
        "PROJECT_OVERVIEW.md": """# Project Overview

MDCoreAlign estimates continuous stable-residue weights and then explicitly
fits every frame with weighted rigid-body alignment. Throughput is the
production default, exact is the chronological float64 numerical reference,
and W5 is an optional alignment-free local-deformation analysis. Compact
analysis and coordinate materialisation are intentionally separate.
""",
        "INSTALL.md": (ROOT / "docs" / "installation.md").read_text(),
        "GPU_SETUP.md": (ROOT / "docs" / "gpu_installation.md").read_text(),
        "QUICK_START.md": (ROOT / "docs" / "quickstart.md").read_text(),
        "INPUT_FORMATS.md": (ROOT / "docs" / "input_formats.md").read_text(),
        "COORDINATE_EXPORT.md": (ROOT / "docs" / "coordinate_export.md").read_text(),
        "OUTPUTS_AND_INTERPRETATION.md": (ROOT / "docs" / "output_formats.md").read_text(),
        "TROUBLESHOOTING.md": (ROOT / "docs" / "troubleshooting.md").read_text(),
        "ADVISOR_TEST_PLAN.md": """# Advisor Test Plan

1. Install `dist/mdcorealign-0.1.0rc1-py3-none-any.whl` in a clean environment.
2. Run `mdcorealign doctor` and `bash examples/run_advisor_smoke_test.sh`.
3. Inspect compact CPU throughput, exact, W5 and ordinary-RMSF outputs.
4. Inspect mmCIF, XTC and compatibility-PDB round trips.
5. On a compatible GPU host, run `bash examples/run_gpu_smoke_test.sh` and
   verify that `backend_used` is `gpu`.
""",
        "KNOWN_LIMITATIONS.md": """# Known Limitations

- PBC/nojump and centering are upstream responsibilities and are not proven by
  coordinate export.
- RMSF and stable-reference interpretation are alignment-frame dependent.
- Exact is a numerical reference, not biological ground truth.
- W5 is dimensionless local deformation, not RMSD or a mature hard domain label.
- Formal product validation is centred on protein C-alpha workflows.
- CUDA 12 with CuPy 13 is validated; CUDA 13 is not claimed.
- The archive is not an offline bundle of third-party dependency wheels.
""",
        "THIRD_PARTY_NOTICES.md": (ROOT / "THIRD_PARTY_NOTICES.md").read_text(),
    }


def assemble() -> None:
    if not SOURCE.exists() or not DIST.exists():
        raise SystemExit("release source and distributions must exist before assembly")
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    copy_tree(SOURCE, PACKAGE / "source")
    artifacts = sorted(
        path
        for path in DIST.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if len(artifacts) != 2:
        raise RuntimeError(f"expected one wheel and one sdist, found: {artifacts}")
    for artifact in artifacts:
        copy_file(artifact, PACKAGE / "dist" / artifact.name)
    copy_tree(ROOT / "examples", PACKAGE / "examples")
    copy_tree(SOURCE / "tests", PACKAGE / "tests")
    copy_tree(SOURCE / "docs", PACKAGE / "docs")
    for name in ("pyproject.toml", "README.md"):
        copy_file(ROOT / name, PACKAGE / name)
    for name in ("LICENSE", "CITATION.cff", "CHANGELOG.md"):
        copy_file(ROOT / name, PACKAGE / name)
    for name, content in advisor_documents().items():
        write(PACKAGE / name, content)
    validation = OUT / "05_tests"
    for name in VALIDATION_FILES:
        source = validation / name
        if source.exists():
            copy_file(source, PACKAGE / "validation" / name)
    copy_file(OUT / "00_freeze" / "environment_manifest.tsv", PACKAGE / "environment_manifest.tsv")
    copy_file(OUT / "00_freeze" / "source_manifest.tsv", PACKAGE / "source_manifest.tsv")
    sanitize_text(PACKAGE / "environment_manifest.tsv")
    write_manifest(PACKAGE)


def advisor_archive() -> Path:
    if not PACKAGE.exists():
        raise SystemExit("assemble the package before creating the archive")
    target = OUT / f"MDCoreAlign_supervisor_test_package_v5_{DATE}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        archive.add(PACKAGE, arcname="MDCoreAlign_supervisor_test_package")
    return target


def review_archive() -> Path:
    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    mapping = {
        "final_single_product_report.md": OUT / "06_release" / "final_single_product_report.md",
        "single_product_contract.md": OUT / "01_product_contract" / "single_product_contract.md",
        "capability_matrix.tsv": OUT / "01_product_contract" / "capability_matrix.tsv",
        "dependency_contract.tsv": OUT / "01_product_contract" / "dependency_contract.tsv",
        "wheel_metadata.txt": OUT / "02_packaging" / "wheel_metadata.txt",
        "wheel_requires_dist.tsv": OUT / "02_packaging" / "wheel_requires_dist.tsv",
        "doctor_output_cpu.txt": OUT / "03_gpu_integration" / "doctor_output_cpu.txt",
        "doctor_output_gpu.txt": OUT / "03_gpu_integration" / "doctor_output_gpu.txt",
        "advisor_smoke_output.txt": OUT / "05_tests" / "advisor_smoke_output.txt",
        "gpu_smoke_output.txt": OUT / "05_tests" / "gpu_smoke_output.txt",
        "scientific_regression.tsv": OUT / "05_tests" / "scientific_regression.tsv",
        "io_regression.tsv": OUT / "05_tests" / "io_regression.tsv",
        "shipped_pytest_output.txt": OUT / "05_tests" / "shipped_pytest_output.txt",
        "canonical_pytest_output.txt": OUT / "05_tests" / "canonical_pytest_output.txt",
        "ruff_output.txt": OUT / "05_tests" / "ruff_output.txt",
        "compileall_output.txt": OUT / "05_tests" / "compileall_output.txt",
        "twine_check.txt": OUT / "05_tests" / "twine_check.txt",
        "pip_check.txt": OUT / "05_tests" / "pip_check.txt",
        "archive_only_validation.md": OUT / "05_tests" / "archive_only_validation.md",
        "documented_command_audit.tsv": OUT / "04_cli_docs" / "documented_command_audit.tsv",
        "release_checklist.tsv": OUT / "06_release" / "release_checklist.tsv",
        "phase_status.tsv": OUT / "phase_status.tsv",
        "key_metrics.tsv": OUT / "key_metrics.tsv",
        "archive_manifest_verification.txt": OUT / "05_tests" / "archive_manifest_verification.txt",
        "security_scan.txt": OUT / "05_tests" / "security_scan.txt",
        "documentation_consistency_audit.md": OUT / "04_cli_docs" / "documentation_consistency_audit.md",
        "package_content_audit.tsv": OUT / "02_packaging" / "package_content_audit.tsv",
    }
    write(
        HANDOFF / "00_README_FIRST.md",
        """# MDCoreAlign Single Full-Product v5 Review

This bundle documents SP0-SP9 only. Scientific algorithms, frozen thresholds,
benchmark evidence, version and Git metadata were not changed. Start with
`final_single_product_report.md` and `release_checklist.tsv`.
""",
    )
    for target, source in mapping.items():
        if not source.exists():
            raise RuntimeError(f"missing review input: {source}")
        copy_file(source, HANDOFF / target)
    changed_sources = [
        "src/mdcorealign/cli.py",
        "src/mdcorealign/gpu_backend.py",
        "pyproject.toml",
        "scripts/build_single_full_product_v5.py",
    ]
    changed_tests = [
        "tests/single_full_product_v5/test_single_product_contract.py",
        "tests/test_product_contract_cli.py",
        "tests/advisor_release_candidate_v2/test_advisor_product_contract.py",
    ]
    changed_docs = [
        "README.md",
        "docs/installation.md",
        "docs/gpu_installation.md",
        "docs/quickstart.md",
        "docs/cli_reference.md",
        "docs/troubleshooting.md",
        "examples/throughput_gpu/README.md",
        "examples/run_advisor_smoke_test.sh",
    ]
    for relative in changed_sources:
        copy_file(ROOT / relative, HANDOFF / "changed_source" / relative)
    for relative in changed_tests:
        copy_file(ROOT / relative, HANDOFF / "changed_tests" / relative)
    for relative in changed_docs:
        copy_file(ROOT / relative, HANDOFF / "changed_docs" / relative)
    patch = OUT / "04_cli_docs" / "unified_source_diff.patch"
    if patch.exists():
        copy_file(patch, HANDOFF / "unified_source_diff.patch")
    individual = OUT / "04_cli_docs" / "individual_diffs"
    if individual.exists():
        copy_tree(individual, HANDOFF / "individual_diffs")
    sanitize_tree(HANDOFF)
    write_manifest(HANDOFF)
    target = OUT / f"single_full_product_v5_review_bundle_{DATE}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        archive.add(HANDOFF, arcname="handoff_to_chatgpt")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "assemble", "advisor-archive", "review-archive"])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "assemble":
        assemble()
    elif args.phase == "advisor-archive":
        print(advisor_archive())
    else:
        print(review_archive())


if __name__ == "__main__":
    main()
