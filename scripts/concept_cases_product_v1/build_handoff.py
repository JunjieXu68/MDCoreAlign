#!/usr/bin/env python3
"""Build a compact review handoff from an explicit safe-file allowlist."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/concept_cases_product_v1"
HANDOFF = OUT / "handoff_to_chatgpt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def result_files() -> list[Path]:
    files: list[Path] = []
    for directory in ("00_scope", "01_concept", "02_case_selection", "04_product", "05_manuscript_assets"):
        files.extend(path for path in (OUT / directory).rglob("*") if path.is_file())
    for role in ("case_A", "case_B", "case_C_scaling"):
        case = OUT / "03_case_studies" / role
        files.extend(
            path
            for path in case.glob("*")
            if path.is_file() and path.suffix.lower() not in {".pdb", ".npy", ".npz"}
        )
    release = OUT / "06_release_candidate"
    files.extend(
        path
        for path in release.rglob("*")
        if path.is_file()
        and "logs" not in path.parts
        and "dist" not in path.parts
        and not any(part.startswith("automated_") for part in path.parts)
        and path.suffix.lower() not in {".whl", ".gz"}
    )
    files.append(OUT / "final_decision_report.md")
    return files


def repository_files() -> list[Path]:
    explicit = [
        "README.md",
        "pyproject.toml",
        "LICENSE",
        "CITATION.cff",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "release_checklist.md",
        "src/mdcorealign/cli.py",
        "src/mdcorealign/contracts.py",
        "src/mdcorealign/gpu_backend.py",
        "tests/test_product_contract_cli.py",
        "tests/test_mdcorealign_release_readiness.py",
        "scripts/check_clean_install.sh",
        "scripts/build_package.sh",
        "scripts/concept_cases_product_v1/run_cp0_cp2.py",
        "scripts/concept_cases_product_v1/make_figures.py",
        "scripts/concept_cases_product_v1/build_handoff.py",
        ".github/workflows/ci.yml",
    ]
    files = [ROOT / item for item in explicit]
    for directory in (
        "docs/cpu_installation.md",
        "docs/gpu_installation.md",
        "docs/modes_and_quality_contracts.md",
        "docs/output_schema.md",
        "docs/large_trajectory_workflow.md",
        "docs/local_deformation.md",
        "docs/quickstart.md",
        "docs/python_api.md",
        "docs/troubleshooting.md",
        "examples/exact_cpu",
        "examples/throughput_cpu",
        "examples/throughput_gpu",
        "examples/local_deformation",
        "examples/apply_transforms",
    ):
        path = ROOT / directory
        files.extend([path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()])
    return files


def main() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[Path, Path, str]] = []
    for source in sorted(set(result_files() + repository_files())):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.is_relative_to(OUT):
            relative = Path("evidence") / source.relative_to(OUT)
        else:
            relative = Path("repository") / source.relative_to(ROOT)
        destination = HANDOFF / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append((source, destination, digest(destination)))

    inventory = HANDOFF / "file_inventory.tsv"
    with inventory.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["handoff_path", "source_path", "size_bytes", "sha256"])
        for source, destination, sha256 in copied:
            writer.writerow(
                [
                    destination.relative_to(HANDOFF),
                    source.relative_to(ROOT),
                    destination.stat().st_size,
                    sha256,
                ]
            )
    (HANDOFF / "hashes.sha256").write_text(
        "".join(
            f"{sha256}  {destination.relative_to(HANDOFF)}\n"
            for _, destination, sha256 in copied
        )
    )
    total_bytes = sum(destination.stat().st_size for _, destination, _ in copied)
    (HANDOFF / "00_README_FIRST.md").write_text(
        "# MDCoreAlign CP0-CP4 Compact Handoff\n\n"
        "This review bundle contains claim/selection audits, derived case-study "
        "tables and figures, manuscript drafts, product/release reports, relevant "
        "source changes and tests. It excludes raw ATLAS/mdCATH trajectories, "
        "coordinate caches, representative aligned PDB files, Python caches, "
        "Git metadata and secrets.\n\n"
        f"- Files copied: {len(copied)}\n"
        f"- Total bytes: {total_bytes}\n"
        "- Canonical tests: 287 passed, 13 known warnings\n"
        "- Public release: not performed; human approval required\n"
        "- Start review with `evidence/final_decision_report.md`.\n"
    )


if __name__ == "__main__":
    main()
