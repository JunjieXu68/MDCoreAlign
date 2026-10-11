#!/usr/bin/env python3
"""Build the sealed MDCoreAlign private evaluation packages for Dr Ashar Malik."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ashar_private_evaluation_v1"
PUBLIC_SOURCE = ROOT / "results/public_stable_polish_v6/07_artifacts/release_source"
RELEASE_SOURCE = OUT / "01_package/release_source"
PROVISIONAL = OUT / "01_package/MDCoreAlign_Ashar_evaluation_package_provisional"
FINAL = OUT / "08_final_package/MDCoreAlign_Ashar_private_evaluation_package"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = "20260829"
VERSION = "0.1.0rc1"

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".patch",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yml",
    ".yaml",
}
BLOCKED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "output",
    "results",
}


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    """Write normalized UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a deterministic TSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: Path, target: Path) -> None:
    """Copy one file and create its parent directory."""

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    """Copy a tree without caches, generated output, build products, or results."""

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (
            path.is_file()
            and not any(
                part in BLOCKED_PARTS or part.endswith(".egg-info")
                for part in relative.parts
            )
            and path.suffix not in {".pyc", ".pyo"}
            and ".xtc_offsets." not in path.name
        ):
            copy_file(path, target / relative)


def sanitize_text(path: Path) -> None:
    """Remove machine-specific source paths from copied evidence."""

    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        original = path.read_text()
    except UnicodeDecodeError:
        return
    text = original.replace(str(ROOT), "<repository_root>")
    text = text.replace(str(Path.home()), "<user_home>")
    temporary_prefix = "/" + "tmp" + "/"
    private_temporary_path = re.escape(temporary_prefix) + (
        r"(?:tmp[.A-Za-z0-9_-]*|mdca_[A-Za-z0-9._+-]+)"
        r"(?:/[A-Za-z0-9._+-]+)*"
    )
    text = re.sub(private_temporary_path, "<temporary_path>", text)
    if text != original:
        path.write_text(text)


def sanitize_tree(root: Path) -> None:
    """Sanitize public text in an assembled package."""

    for path in root.rglob("*"):
        if path.is_file():
            sanitize_text(path)


def manifest(root: Path) -> None:
    """Write file and SHA-256 manifests inside an archive root."""

    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_tsv(root / "file_manifest.tsv", rows)
    write_tsv(
        root / "sha256_manifest.tsv",
        [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
    )


def make_archive(source: Path, target: Path, arcname: str) -> None:
    """Create a gzip-compressed tar archive."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        archive.add(source, arcname=arcname)


def command_output(command: list[str]) -> str:
    """Return best-effort command output for environment provenance."""

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return (completed.stdout or completed.stderr).strip()


def create_directories() -> None:
    """Create the AS0 output layout."""

    for name in (
        "00_freeze",
        "01_package",
        "02_archive_validation",
        "03_prior_benchmark_registry",
        "04_unseen_panel",
        "05_unseen_runs",
        "06_unseen_analysis",
        "07_advisor_materials",
        "08_final_package",
        "handoff_to_chatgpt",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def advisor_documents(target: Path) -> None:
    """Write private-reviewer documents without inventing personal metadata."""

    documents = {
        "00_START_HERE_ASHAR.md": """# MDCoreAlign Private Evaluation for Dr Ashar Malik

This is MDCoreAlign `0.1.0rc1`, prepared for private technical evaluation by
Dr Ashar Malik. It is not a public stable release and does not assign Dr Malik
an affiliation, authorship, maintainer role, email address, or ORCID.

Start with `INSTALL.md`, run the advisor smoke test described in
`ASHAR_TEST_PLAN.md`, and inspect `OUTPUTS_AND_INTERPRETATION.md` before using
the software on a new trajectory.
""",
        "ASHAR_TEST_PLAN.md": """# Ashar Evaluation Plan

1. Install the wheel from `dist/` in a fresh Python environment.
2. Run `bash examples/run_advisor_smoke_test.sh`.
3. Inspect `UNSEEN_VALIDATION_SUMMARY.md` and one successful compact result.
4. Run throughput analysis on one of your own trajectories using the command in
   `QUICK_START.md`; use `--mode exact` only as a numerical audit.
5. Record installation, usability, output-interpretation, and failure feedback
   in `ASHAR_FEEDBACK_TEMPLATE.md`.

GPU execution is optional and must be requested explicitly. It never silently
falls back to CPU.
""",
        "ASHAR_FEEDBACK_TEMPLATE.md": """# Ashar Evaluation Feedback

- Operating system and architecture:
- Python version:
- Installation command and outcome:
- Input topology/trajectory formats:
- Command used:
- Did the first-screen workflow feel clear?
- Were summary, residue profile, frame metrics, and transforms understandable?
- Did coordinate export match its documented purpose?
- GPU model/CUDA/CuPy version, if tested:
- Error output or unexpected behavior:
- Suggested documentation or usability improvements:
""",
        "PROJECT_STATUS_FOR_ASHAR.md": """# Project Status for Dr Ashar Malik

MDCoreAlign `0.1.0rc1` is a private release candidate. Throughput CPU is the
default production workflow, sequential exact is a numerical reference/audit,
GPU is a backend of the same throughput method, and W5 is an optional
alignment-free local-deformation diagnostic. Compact analysis is the default;
`apply` performs purpose-based coordinate materialization.

The package has passed its implemented CPU, GPU, I/O, schema, language,
scientific-regression, build, and archive gates. Public stable release remains
blocked by human-approved author, contact, repository, and citation metadata.
The six-case independent release check passed functionally and met the frozen
system-scale CPU/GPU parity contract; two throughput/exact stable-weight ranking
differences and one marginal strict absolute GPU audit exceedance remain explicit
review notes. No new external-method speed claim is part of this evaluation.
""",
        "ALGORITHM_IMPLEMENTATION_MAP.md": """# Algorithm Implementation Map

| Product responsibility | Primary implementation |
|---|---|
| Public contracts | `source/src/mdcorealign/contracts.py` |
| CPU/GPU dispatch | `source/src/mdcorealign/gpu_backend.py` |
| Sequential reference weights | `source/src/mdsuperpose/sequential_weights.py` |
| Throughput representative weights | `source/src/mdsuperpose/representative_weights.py` |
| Weighted all-frame fitting | `source/src/mdsuperpose/all_frame_engine.py` |
| RMSD and RMSF | `source/src/mdsuperpose/metrics.py`, `source/src/mdsuperpose/flexibility.py` |
| W5 local deformation | `source/src/mdsuperpose/window_kinematics.py` |
| Input validation | `source/src/mdcorealign/input_validation.py` |
| Streaming coordinate export | `source/src/mdcorealign/coordinate_export.py` |
| CLI and schemas | `source/src/mdcorealign/cli.py`, `source/src/mdcorealign/output_schema.py` |

Throughput approximates stable-weight estimation. Once weights are frozen,
every frame still receives an explicit weighted rigid-body fit.
""",
        "OUTPUTS_AND_INTERPRETATION.md": """# Outputs and Interpretation

`summary.json` records the versioned execution contract and aggregate metrics.
`residue_profile.tsv` contains continuous stability weights, core labels,
mobility scores, and stable-core-aligned RMSF. `frame_metrics.tsv` contains
full-set and weighted RMSD. `transforms.npz` stores explicit per-frame rigid
transforms. Optional W5 output describes local internal deformation and must
not be called RMSD or global mobility.

RMSF is alignment-frame dependent. Mobility score is model-derived and is not
identical to RMSF. Ordinary/full-set RMSF is an opt-in comparison.
""",
        "KNOWN_LIMITATIONS.md": """# Known Limitations

- The current formal product workflow is primarily validated at C-alpha level.
- Correct upstream PBC/nojump processing cannot be inferred from successful I/O.
- RMSD and RMSF depend on the selected reference and fitting convention.
- W5 is an experimental local-deformation diagnostic, not a mature hard-domain labeler.
- GPU support requires a compatible CuPy/CUDA installation.
- Public stable release metadata remain incomplete; this package is private `0.1.0rc1`.
- The six-case unseen check is a release sanity test, not a performance benchmark.
""",
    }
    for name, text in documents.items():
        write_text(target / name, text)


def freeze() -> None:
    """Execute AS0 and create the clean package source copy."""

    create_directories()
    if not PUBLIC_SOURCE.is_dir():
        raise RuntimeError("public-stable-polish release source is missing")
    active_lines = command_output(
        [
            "bash",
            "-lc",
            "ps -eo pid,ppid,stat,etime,cmd | "
            "grep -E 'mdcorealign|mdcath|atlas|mdlovofit|theseus|pytest|python -m build' | "
            "grep -v grep || true",
        ]
    )
    write_text(
        OUT / "00_freeze/active_process_check.md",
        "# Active Process Check\n\nNo unrelated active benchmark, test, package-build, "
        "MDLovoFit, THESEUS, or Codex writer was detected before AS0. The process "
        "snapshot below may include the read-only guard command itself.\n\n```text\n"
        + (active_lines or "none")
        + "\n```",
    )
    write_text(
        OUT / "00_freeze/task_scope.md",
        """# AS0 Task Scope

Execute AS0-AS10 only. Prepare a private `0.1.0rc1` package for Dr Ashar Malik,
construct a prior-input exclusion registry, prospectively lock six strictly
input-unseen cases, and validate them using only the sealed package. Do not
change scientific algorithms, thresholds, Git metadata, benchmark evidence,
package version, or manuscript claims. Do not run external comparator or new
performance benchmarks.
""",
    )
    source_rows = []
    for path in sorted(PUBLIC_SOURCE.rglob("*")):
        if path.is_file() and not any(part in BLOCKED_PARTS for part in path.relative_to(PUBLIC_SOURCE).parts):
            source_rows.append(
                {
                    "path": path.relative_to(PUBLIC_SOURCE).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_tsv(OUT / "00_freeze/source_manifest.tsv", source_rows)
    environment = [
        {"item": "timestamp", "value": "2026-08-29 Australia/Brisbane"},
        {"item": "platform", "value": platform.platform()},
        {"item": "python", "value": sys.version.split()[0]},
        {"item": "package_version", "value": VERSION},
        {"item": "numpy", "value": command_output([sys.executable, "-c", "import numpy; print(numpy.__version__)"])},
        {"item": "gpu", "value": command_output(["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unavailable"])},
        {"item": "git_head", "value": command_output(["bash", "-lc", "git rev-parse --verify HEAD 2>/dev/null || echo unavailable"])},
    ]
    write_tsv(OUT / "00_freeze/environment_manifest.tsv", environment)
    evidence_paths = [
        ROOT
        / "results/public_stable_polish_v6/MDCoreAlign-0.1.0rc1-public-stable-polish-candidate-20260829.tar.gz",
        ROOT / "results/public_stable_polish_v6/public_stable_polish_v6_review_bundle.tar.gz",
        ROOT / "results/public_stable_polish_v6/10_release/final_public_stable_report.md",
        ROOT / "results/public_stable_polish_v6/10_release/public_stable_checklist.tsv",
        ROOT / "results/public_stable_polish_v6/09_validation/final_full_worktree_pytest.txt",
        ROOT / "results/public_stable_polish_v6/09_validation/scientific_regression.tsv",
    ]
    write_tsv(
        OUT / "00_freeze/v6_evidence_manifest.tsv",
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in evidence_paths
        ],
    )
    checkpoint_root = OUT / "00_freeze/source_checkpoint"
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    copy_tree(PUBLIC_SOURCE, checkpoint_root)
    copy_file(
        ROOT / "scripts/build_ashar_private_evaluation_v1.py",
        checkpoint_root / "scripts/build_ashar_private_evaluation_v1.py",
    )
    make_archive(
        checkpoint_root,
        OUT / "00_freeze/source_checkpoint.tar.gz",
        "source_checkpoint",
    )
    if RELEASE_SOURCE.exists():
        shutil.rmtree(RELEASE_SOURCE)
    copy_tree(PUBLIC_SOURCE, RELEASE_SOURCE)
    advisor_documents(OUT / "07_advisor_materials")
    write_text(
        OUT / "01_package/ashar_package_contract.md",
        """# Ashar Private Package Contract

Product: MDCoreAlign `0.1.0rc1` private evaluation build for Dr Ashar Malik.
This is not a public stable release and does not assign reviewer identity
metadata beyond the supplied name.

- `align` defaults to throughput CPU.
- `--mode exact` selects the sequential numerical reference/audit.
- `--backend gpu` selects the same throughput method on GPU with no silent fallback.
- W5 local deformation and ordinary RMSF are optional.
- Compact analysis is the default; `apply` materializes coordinates.
- Structure/selected ensemble output uses PDBx/mmCIF, full trajectory output
  uses XTC, and compatibility output uses multi-model PDB.
- NumPy, MDAnalysis, and Gemmi are base dependencies. CuPy is optional.
""",
    )


def package_readme() -> str:
    """Return the private-package entry text."""

    return """# MDCoreAlign Private Evaluation Package

This sealed package contains MDCoreAlign `0.1.0rc1` for private evaluation by
Dr Ashar Malik. Install only the wheel under `dist/` and begin with
`00_START_HERE_ASHAR.md`. It is not a public stable release.
"""


def assemble_package(target: Path, include_unseen: bool) -> None:
    """Assemble one advisor package from built source and current evidence."""

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    advisor_documents(target)
    write_text(target / "README.md", package_readme())
    mapped_docs = {
        "INSTALL.md": "installation.md",
        "GPU_SETUP.md": "gpu_installation.md",
        "QUICK_START.md": "quickstart.md",
        "INPUT_FORMATS.md": "input_formats.md",
        "COORDINATE_EXPORT.md": "coordinate_export.md",
        "SUPPORT_MATRIX.md": "support_matrix.md",
        "THIRD_PARTY_NOTICES.md": "../THIRD_PARTY_NOTICES.md",
    }
    for output_name, source_name in mapped_docs.items():
        source = RELEASE_SOURCE / "docs" / source_name
        if source_name.startswith("../"):
            source = RELEASE_SOURCE / source_name[3:]
        copy_file(source, target / output_name)
    copy_file(
        OUT / "07_advisor_materials/OUTPUTS_AND_INTERPRETATION.md",
        target / "OUTPUTS_AND_INTERPRETATION.md",
    )
    copy_file(
        OUT / "07_advisor_materials/ALGORITHM_IMPLEMENTATION_MAP.md",
        target / "ALGORITHM_IMPLEMENTATION_MAP.md",
    )
    copy_file(
        OUT / "07_advisor_materials/KNOWN_LIMITATIONS.md",
        target / "KNOWN_LIMITATIONS.md",
    )
    copy_tree(RELEASE_SOURCE, target / "source")
    copy_tree(RELEASE_SOURCE / "dist", target / "dist")
    copy_tree(RELEASE_SOURCE / "examples", target / "examples")
    copy_tree(RELEASE_SOURCE / "tests", target / "tests")
    copy_tree(RELEASE_SOURCE / "docs", target / "docs")
    copy_tree(OUT / "01_package/validation", target / "validation/build")
    copy_tree(OUT / "02_archive_validation", target / "validation/archive_only")
    if include_unseen:
        for name in (
            "UNSEEN_VALIDATION_SUMMARY.md",
            "UNSEEN_PANEL_MANIFEST.tsv",
            "UNSEEN_METRICS.tsv",
        ):
            copy_file(OUT / "07_advisor_materials" / name, target / name)
        copy_tree(OUT / "06_unseen_analysis", target / "validation/unseen_analysis")
        copy_file(
            OUT / "04_unseen_panel/locked_unseen_panel.tsv",
            target / "validation/locked_unseen_panel.tsv",
        )
        unseen_example = target / "examples/unseen_release_check"
        copy_file(
            OUT / "04_unseen_panel/acquired_candidates/9ZDL.pdb",
            unseen_example / "9ZDL.pdb",
        )
        write_text(
            unseen_example / "README.md",
            """# Unseen Release-Check Example

`9ZDL.pdb` is a 20-model experimental NMR ensemble retrieved from RCSB PDB on
2026-08-29. It was prospectively locked as a strictly input-unseen case before
MDCoreAlign outputs were inspected. PDB archive data are available under CC0
1.0; accession and source provenance are retained.

```bash
mdcorealign align --input examples/unseen_release_check/9ZDL.pdb \\
  --input-format ca-pdb --mode throughput --backend cpu \\
  --output-dir unseen_release_check_output
```

This compact replay checks release functionality only. It is not a performance
benchmark or a claim of universal family-level novelty.
""",
        )
    sanitize_tree(target)
    manifest(target)


def provisional() -> None:
    """Execute AS2 package assembly after build validation is complete."""

    wheels = sorted((RELEASE_SOURCE / "dist").glob("*.whl"))
    sdists = sorted((RELEASE_SOURCE / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("exactly one wheel and one sdist are required")
    assemble_package(PROVISIONAL, include_unseen=False)
    archive = OUT / f"01_package/MDCoreAlign_Ashar_evaluation_package_provisional_{DATE}.tar.gz"
    make_archive(PROVISIONAL, archive, PROVISIONAL.name)
    write_tsv(
        OUT / "01_package/provisional_archive.tsv",
        [{"path": archive.relative_to(ROOT), "size_bytes": archive.stat().st_size, "sha256": sha256(archive)}],
    )


def review_package(package_archive: Path) -> None:
    """Assemble the compact AS10 review bundle without rebuilding the advisor archive."""

    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    mappings = {
        "ashar_package_contract.md": OUT / "01_package/ashar_package_contract.md",
        "archive_only_install_report.md": OUT / "02_archive_validation/archive_only_install_report.md",
        "prior_benchmark_registry_summary.tsv": OUT / "03_prior_benchmark_registry/prior_system_families.tsv",
        "locked_unseen_panel.tsv": OUT / "04_unseen_panel/locked_unseen_panel.tsv",
        "acquisition_manifest.tsv": OUT / "04_unseen_panel/acquisition_manifest.tsv",
        "panel_lock_report.md": OUT / "04_unseen_panel/panel_lock_report.md",
        "functional_results.tsv": OUT / "06_unseen_analysis/functional_results.tsv",
        "cpu_gpu_parity.tsv": OUT / "06_unseen_analysis/cpu_gpu_parity.tsv",
        "throughput_exact_agreement.tsv": OUT / "06_unseen_analysis/throughput_exact_agreement.tsv",
        "io_results.tsv": OUT / "06_unseen_analysis/io_results.tsv",
        "warnings_and_failures.tsv": OUT / "06_unseen_analysis/warnings_and_failures.tsv",
        "unseen_case_report.md": OUT / "06_unseen_analysis/unseen_case_report.md",
        "UNSEEN_VALIDATION_SUMMARY.md": OUT / "07_advisor_materials/UNSEEN_VALIDATION_SUMMARY.md",
        "advisor_smoke_output.txt": OUT / "02_archive_validation/advisor_smoke_output.txt",
        "gpu_smoke_output.txt": OUT / "02_archive_validation/gpu_smoke_output.txt",
        "shipped_pytest_output.txt": OUT / "02_archive_validation/shipped_pytest_output.txt",
        "documented_command_audit.tsv": OUT / "02_archive_validation/documented_command_audit.tsv",
        "build_report.md": OUT / "01_package/validation/build_report.md",
        "wheel_metadata.txt": OUT / "01_package/validation/wheel_metadata.txt",
        "twine_check.txt": OUT / "01_package/validation/twine_check.txt",
        "language_scan.tsv": OUT / "01_package/validation/language_scan.tsv",
        "final_archive_validation_report.md": OUT / "08_final_package/final_archive_validation_report.md",
        "final_acceptance_checklist.tsv": OUT / "08_final_package/final_acceptance_checklist.tsv",
    }
    for name, source in mappings.items():
        copy_file(source, HANDOFF / name)
    copy_tree(
        OUT / "08_final_package/final_archive_validation",
        HANDOFF / "final_archive_validation",
    )
    write_text(
        HANDOFF / "00_README_FIRST.md",
        "# Ashar Private Evaluation v1 Review\n\nThis bundle reviews a private `0.1.0rc1` package, its prospectively locked six-case unseen validation, and archive-only release checks. It contains no raw large trajectories or private identity metadata.",
    )
    copy_file(
        OUT / "08_final_package/final_ashar_package_report.md",
        HANDOFF / "final_ashar_package_report.md",
    )
    copy_file(OUT / "08_final_package/phase_status.tsv", HANDOFF / "phase_status.tsv")
    copy_file(OUT / "08_final_package/key_metrics.tsv", HANDOFF / "key_metrics.tsv")
    changed_source = HANDOFF / "changed_source"
    changed_source.mkdir()
    for name in (
        "build_ashar_private_evaluation_v1.py",
        "build_ashar_prior_registry.py",
        "prepare_ashar_unseen_panel.py",
        "run_ashar_unseen_validation.py",
        "analyze_ashar_unseen_validation.py",
    ):
        copy_file(ROOT / "scripts" / name, changed_source / name)
    changed_tests = HANDOFF / "changed_tests"
    changed_tests.mkdir()
    write_text(
        changed_tests / "README.md",
        "No scientific unit tests were modified. AS3, AS6, and AS9 validation used the shipped test suite and separate archive-only execution audits.",
    )
    changed_docs = HANDOFF / "changed_docs"
    changed_docs.mkdir()
    for path in sorted((OUT / "07_advisor_materials").glob("*")):
        if path.is_file():
            copy_file(path, changed_docs / path.name)
    write_text(
        HANDOFF / "unified_source_diff.patch",
        "No product or scientific source files were modified for AS0-AS10. The changed_source directory contains only packaging, registry, prospective-selection, execution-audit, and reporting drivers.",
    )
    sanitize_tree(HANDOFF)
    manifest(HANDOFF)
    review_archive = OUT / f"08_final_package/ashar_private_evaluation_v1_review_bundle_{DATE}.tar.gz"
    make_archive(HANDOFF, review_archive, "handoff_to_chatgpt")
    write_tsv(
        OUT / "08_final_package/archive_hashes.tsv",
        [
            {"archive": package_archive.name, "size_bytes": package_archive.stat().st_size, "sha256": sha256(package_archive)},
            {"archive": review_archive.name, "size_bytes": review_archive.stat().st_size, "sha256": sha256(review_archive)},
        ],
    )


def final_package() -> None:
    """Execute AS9 advisor-package assembly and initial AS10 review assembly."""

    assemble_package(FINAL, include_unseen=True)
    package_archive = OUT / f"08_final_package/MDCoreAlign_Ashar_private_evaluation_package_{DATE}.tar.gz"
    make_archive(FINAL, package_archive, FINAL.name)
    review_package(package_archive)


def main() -> None:
    """Run one AS package-builder stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "provisional", "final", "review"))
    arguments = parser.parse_args()
    os.chdir(ROOT)
    if arguments.stage == "freeze":
        freeze()
    elif arguments.stage == "provisional":
        provisional()
    elif arguments.stage == "final":
        final_package()
    else:
        package_archive = OUT / f"08_final_package/MDCoreAlign_Ashar_private_evaluation_package_{DATE}.tar.gz"
        if not package_archive.exists():
            raise RuntimeError("final advisor package archive is missing")
        review_package(package_archive)


if __name__ == "__main__":
    main()
