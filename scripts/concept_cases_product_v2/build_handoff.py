#!/usr/bin/env python3
"""Build the compact, allowlisted CP v2 review handoff."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/concept_cases_product_v2"
HANDOFF = OUT / "handoff_to_chatgpt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def evidence_files() -> list[Path]:
    files: list[Path] = []
    for directory in (
        "00_audit",
        "01_concept_figure",
        "02_case_A",
        "03_case_B",
        "04_manuscript_assets",
        "05_product_docs",
        "06_release_validation",
    ):
        files.extend(
            path
            for path in (OUT / directory).rglob("*")
            if path.is_file()
            and "dist" not in path.parts
            and path.suffix.lower() not in {".whl", ".gz"}
        )
    files.append(OUT / "final_revision_report.md")
    return files


def repository_files() -> list[Path]:
    explicit = [
        "README.md",
        "pyproject.toml",
        "scripts/concept_cases_product_v2/__init__.py",
        "scripts/concept_cases_product_v2/case_interpretation.py",
        "scripts/concept_cases_product_v2/generate_revision.py",
        "scripts/concept_cases_product_v2/finalize_revision.py",
        "scripts/concept_cases_product_v2/build_handoff.py",
        "tests/test_concept_cases_product_v2.py",
        "docs/quickstart.md",
        "docs/modes_and_quality_contracts.md",
        "docs/local_deformation.md",
        "docs/large_trajectory_workflow.md",
        "docs/output_schema.md",
        "docs/cpu_installation.md",
        "docs/gpu_installation.md",
    ]
    return [ROOT / path for path in explicit]


def main() -> None:
    existing = [path for path in HANDOFF.rglob("*") if path.is_file()]
    if existing:
        raise RuntimeError("handoff is non-empty; refusing a potentially stale destructive rebuild")
    copied: list[tuple[Path, Path, str]] = []
    for source in sorted(set(evidence_files() + repository_files())):
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
            writer.writerow([destination.relative_to(HANDOFF), source.relative_to(ROOT), destination.stat().st_size, sha256])
    (HANDOFF / "hashes.sha256").write_text(
        "".join(f"{sha256}  {destination.relative_to(HANDOFF)}\n" for _, destination, sha256 in copied)
    )
    total_bytes = sum(destination.stat().st_size for _, destination, _ in copied)
    (HANDOFF / "00_README_FIRST.md").write_text(
        "# MDCoreAlign CP v2 Compact Handoff\n\n"
        "This allowlisted bundle contains only revised figures/source tables, manuscript assets, documentation diffs, focused tests and release-validation reports. It excludes raw trajectories, coordinate caches, historical benchmark tables, aligned trajectory archives, package binaries, Python caches, Git metadata and secrets.\n\n"
        f"- Files copied: {len(copied)}\n"
        f"- Total bytes: {total_bytes}\n"
        "- Selected cases: 3nci_A and 6sup_A (unchanged)\n"
        "- Public release: not performed; human approval required\n"
        "- Start with `evidence/final_revision_report.md`.\n"
    )
    archive = OUT / "handoff_to_chatgpt.zip"
    if archive.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(item for item in HANDOFF.rglob("*") if item.is_file()):
            handle.write(path, Path("handoff_to_chatgpt") / path.relative_to(HANDOFF))


if __name__ == "__main__":
    main()
