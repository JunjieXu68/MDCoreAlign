#!/usr/bin/env python3
"""Copy only compact evidence files required by the source regression suite."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "results/source_code_handoff_v1/MDCoreAlign_source_code_handoff"

COMPACT_TREES = (
    "results/final_statistics",
    "results/ablation/regularization",
    "results/diagnostics",
    "results/sensitivity",
    "results/md_trajectory_focus",
    "results/final_submission_upgrade",
    "results/final_quality_upgrade",
    "results/submission_polish",
    "results/submission_critical/regularization_rerun",
    "results/submission_critical/diagnostics_rerun",
    "results/submission_critical/theseus_reproducibility",
    "results/submission_critical/lambda_sensitivity_rerun",
    "results/submission_critical/yiip_biology_proxy",
    "results/submission_critical/readiness_audit",
    "results/flexibility/figure_data",
    "results/flexibility/figures",
)

COMPACT_FILES = (
    "results/flexibility/nhaa_flexibility_profile.csv",
    "results/flexibility/yiip_short_flexibility_profile.csv",
    "results/higher_tier_external_baseline_theseus_formal/combined_theseus_formal_summary.csv",
    "results/final_multi_system_summary_20260603/final_baseline_breadth_hardtrim_summary.csv",
    "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv",
    "results/database_scale_benchmark_v2/mdlovofit_comparison/manifests/mdlovofit_real_long_prefix_manifest_locked.tsv",
    "docs/current_main_results_summary.md",
    "docs/current_main_results_summary.txt",
    "docs/rmsf_flexible_segments.md",
    "results/concept_cases_product_v1/02_case_selection/case_candidate_table.tsv",
    "results/concept_cases_product_v1/02_case_selection/selection_decision.json",
    "results/concept_cases_product_v1/03_case_studies/case_A/residue_decomposition.tsv",
    "results/concept_cases_product_v1/03_case_studies/case_B/residue_decomposition.tsv",
    "results/concept_cases_product_v2/02_case_A/case_A_classification_track_v2.tsv",
    "results/concept_cases_product_v2/02_case_A/case_A_key_regions_v2.tsv",
    "results/concept_cases_product_v2/03_case_B/case_B_classification_track_v2.tsv",
    "results/concept_cases_product_v2/03_case_B/case_B_key_regions_v2.tsv",
    "results/concept_cases_product_v2/03_case_B/case_B_region_summary_v2.tsv",
    "results/concept_cases_product_v2/04_manuscript_assets/figure_captions_v2.md",
    "results/concept_cases_product_v2/05_product_docs/documentation_consistency_audit.tsv",
)

FIGURE_STEMS = (
    "results/concept_cases_product_v2/01_concept_figure/fig1_method_concept_v2",
    "results/concept_cases_product_v2/02_case_A/fig2_case_A_3nci_A_v2",
    "results/concept_cases_product_v2/03_case_B/fig3_case_B_6sup_A_v2",
)

SOURCE_ADDITIONS = (
    "scripts/source_code_handoff_v1/prepare_source_handoff.py",
    "scripts/source_code_handoff_v1/copy_test_fixtures.py",
    "scripts/source_code_handoff_v1/finalize_source_handoff.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_file(relative: Path, rows: list[dict[str, object]], category: str) -> None:
    source = ROOT / relative
    destination = SNAPSHOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = digest(source)
    copied_hash = digest(destination)
    if source_hash != copied_hash:
        raise RuntimeError(f"fixture copy hash mismatch: {relative}")
    rows.append(
        {
            "relative_path": relative.as_posix(),
            "category": category,
            "size_bytes": source.stat().st_size,
            "source_sha256": source_hash,
            "snapshot_sha256": copied_hash,
            "match": True,
        }
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sync_incremental_source() -> None:
    """Copy newly included benchmark/packaging sources and extend parity."""

    parity_path = SNAPSHOT / "validation/source_hash_parity.tsv"
    with parity_path.open(newline="") as handle:
        parity = list(csv.DictReader(handle, delimiter="\t"))
    by_path = {row["relative_path"]: row for row in parity}
    additions = [ROOT / name for name in SOURCE_ADDITIONS]
    additions.extend(
        path
        for path in sorted((ROOT / "benchmark").rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".py", ".sh", ".yaml", ".yml", ".json", ".toml"}
        and "__pycache__" not in path.parts
    )
    for source in additions:
        relative = source.relative_to(ROOT)
        destination = SNAPSHOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = digest(source)
        copied_hash = digest(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"source copy hash mismatch: {relative}")
        by_path[relative.as_posix()] = {
            "relative_path": relative.as_posix(),
            "source_sha256": source_hash,
            "snapshot_sha256": copied_hash,
            "match": "True",
            "size_bytes": str(source.stat().st_size),
        }
    ordered = [by_path[key] for key in sorted(by_path)]
    with parity_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(ordered)


def main() -> None:
    sync_incremental_source()
    rows: list[dict[str, object]] = []
    for tree in COMPACT_TREES:
        for source in sorted((ROOT / tree).rglob("*")):
            if source.is_file():
                copy_file(source.relative_to(ROOT), rows, "compact_test_fixture")
    for name in COMPACT_FILES:
        copy_file(Path(name), rows, "compact_test_fixture")
    for stem in FIGURE_STEMS:
        for suffix in (".pdf", ".svg", ".png"):
            copy_file(Path(stem + suffix), rows, "compact_test_fixture")

    # The regression test needs a hash manifest, not the complete historical
    # release bundle. This derivative manifest covers only copied v1 fixtures.
    v1_rows = [
        row
        for row in rows
        if str(row["relative_path"]).startswith("results/concept_cases_product_v1/")
    ]
    compact_manifest = SNAPSHOT / "results/concept_cases_product_v2/00_audit/v1_baseline_manifest.tsv"
    compact_manifest.parent.mkdir(parents=True, exist_ok=True)
    with compact_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"], delimiter="\t")
        writer.writeheader()
        for row in v1_rows:
            writer.writerow(
                {
                    "path": row["relative_path"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["source_sha256"],
                }
            )
    rows.append(
        {
            "relative_path": compact_manifest.relative_to(SNAPSHOT).as_posix(),
            "category": "derived_compact_test_manifest",
            "size_bytes": compact_manifest.stat().st_size,
            "source_sha256": "derived_not_historical_evidence",
            "snapshot_sha256": digest(compact_manifest),
            "match": True,
        }
    )
    write_rows(SNAPSHOT / "validation/test_fixture_manifest.tsv", rows)
    (SNAPSHOT / "validation/TEST_FIXTURES.md").write_text(
        "# Compact Regression Fixtures\n\n"
        "These files are compact, read-only copies used by evidence-schema and "
        "figure-regression tests. They are not a replacement for the frozen "
        "historical result packages. The v1 baseline manifest is a derivative "
        "fixture manifest containing only files included here; no historical "
        "evidence was modified. Raw trajectories, coordinate caches and nested "
        "release archives are excluded.\n"
    )
    print(f"copied {len(rows)} compact fixture files")


if __name__ == "__main__":
    main()
