#!/usr/bin/env python3
"""Correct B3 derived tails and generate independent confidence metadata v2."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import statistics
from typing import Any

from scripts.post_b3_optimisation_v1.common import (
    B2,
    B3,
    MDL,
    OUT,
    domain_split,
    finite,
    iter_tsv,
    number,
    read_tsv,
    truth,
    write_json,
    write_text,
    write_tsv,
)


def registered_reasons(row: dict[str, Any], method: str) -> list[str]:
    """Apply only the catastrophic criteria registered before B3."""

    reasons = []
    if row.get("status") != "success" or not truth(
        row.get(f"{method}_exact_finite_outputs")
    ):
        reasons.append("non_finite_or_failed_output")
    effective = number(row, f"{method}_exact_effective_core_fraction")
    if finite(effective) and effective < 0.10:
        reasons.append("effective_core_fraction_below_0.10")
    core = number(row, f"{method}_exact_core_jaccard_vs_exact")
    if finite(core) and core < 0.50:
        reasons.append("core_jaccard_below_0.50")
    rmsf = number(row, f"{method}_exact_rmsf_spearman_vs_exact")
    if finite(rmsf) and rmsf < 0.90:
        reasons.append("RMSF_spearman_below_0.90")
    if row.get("failure_class") in {
        "unit_integrity_failure",
        "mapping_integrity_failure",
    }:
        reasons.append(str(row["failure_class"]))
    return reasons


def geometric_tail(row: dict[str, Any], method: str) -> bool:
    """Return the separate post-hoc median RMSD > 1 A review flag."""

    value = number(
        row,
        f"{method}_exact_median_absolute_rmsd_difference_vs_exact_A",
    )
    return finite(value) and value > 1.0


def sampling_status(row: dict[str, Any]) -> str:
    """Classify representative and temporal reproducibility independently."""

    fields = (
        ("representative_subset_weight_spearman", 0.90),
        ("representative_subset_core_jaccard", 0.80),
        ("temporal_block_weight_spearman", 0.85),
        ("temporal_block_core_jaccard", 0.75),
    )
    observed = [(number(row, field), threshold) for field, threshold in fields]
    available = [(value, threshold) for value, threshold in observed if finite(value)]
    if not available:
        return "not_evaluated"
    failed = sum(value < threshold for value, threshold in available)
    if failed == 0:
        return "reproducible_at_registered_internal_thresholds"
    if failed < len(available):
        return "mixed_reproducibility"
    return "reproducibility_review"


def solver_status(row: dict[str, Any]) -> str:
    """Classify solver state without conflating it with scientific quality."""

    if row.get("status") == "explicitly_unavailable":
        return "not_evaluated_input_unavailable"
    if row.get("status") != "success":
        if "too_few_frames" in row.get("failure_reason", ""):
            return "unsupported_short_input"
        return "solver_or_input_failure"
    strict = row.get("strict_fixed_budget_status", "")
    if strict == "fixed_budget_stable":
        return "fixed_budget_stable"
    if strict == "fixed_budget_complete_not_converged":
        return "fixed_budget_complete_not_strictly_converged"
    return "successful_unclassified_solver_status"


def approximation_status(row: dict[str, Any]) -> str:
    """Classify exact-relative review separately from solver and sampling."""

    if not truth(row.get("exact_completed")):
        if row.get("status") != "success":
            return "not_evaluated"
        if sampling_status(row).startswith("reproducible"):
            return "not_exact_audited_internal_checks_pass"
        return "not_exact_audited_review"
    registered = bool(registered_reasons(row, "cpu")) or bool(
        registered_reasons(row, "gpu")
    )
    if registered:
        return "registered_catastrophic_quality_failure"
    if geometric_tail(row, "cpu") or geometric_tail(row, "gpu"):
        return "large_geometric_tail_review"
    return "exact_audit_pass"


def median(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [number(row, field) for row in rows]
    finite_values = [value for value in values if finite(value)]
    return statistics.median(finite_values) if finite_values else ""


def main() -> None:
    exact = read_tsv(B3 / "b3_exact_quality_subset.tsv")
    corrected = []
    for row in exact:
        cpu_reasons = registered_reasons(row, "cpu")
        gpu_reasons = registered_reasons(row, "gpu")
        corrected.append(
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["domain_id"],
                "domain_split": domain_split(row["domain_id"]),
                "legacy_cpu_catastrophic_quality_failure": row[
                    "cpu_exact_catastrophic_quality_failure"
                ],
                "legacy_gpu_catastrophic_quality_failure": row[
                    "gpu_exact_catastrophic_quality_failure"
                ],
                "registered_catastrophic_quality_failure": bool(
                    cpu_reasons or gpu_reasons
                ),
                "registered_catastrophic_reasons": ";".join(
                    sorted(set(cpu_reasons + gpu_reasons))
                ),
                "large_geometric_tail_median_RMSD_gt_1A": bool(
                    geometric_tail(row, "cpu") or geometric_tail(row, "gpu")
                ),
                "cpu_median_absolute_RMSD_difference_A": row[
                    "cpu_exact_median_absolute_rmsd_difference_vs_exact_A"
                ],
                "gpu_median_absolute_RMSD_difference_A": row[
                    "gpu_exact_median_absolute_rmsd_difference_vs_exact_A"
                ],
                "cpu_core_jaccard": row["cpu_exact_core_jaccard_vs_exact"],
                "gpu_core_jaccard": row["gpu_exact_core_jaccard_vs_exact"],
                "cpu_RMSF_spearman": row["cpu_exact_rmsf_spearman_vs_exact"],
                "gpu_RMSF_spearman": row["gpu_exact_rmsf_spearman_vs_exact"],
            }
        )
    registered_count = sum(
        truth(row["registered_catastrophic_quality_failure"]) for row in corrected
    )
    large_count = sum(
        truth(row["large_geometric_tail_median_RMSD_gt_1A"])
        for row in corrected
    )
    if registered_count != 8 or large_count != 54:
        raise RuntimeError(
            f"frozen exact-tail audit mismatch: registered={registered_count}, "
            f"large={large_count}"
        )
    write_tsv(OUT / "p1_corrected_reporting/corrected_exact_quality_subset.tsv", corrected)

    mdl_failures = read_tsv(MDL / "mdlovofit_failures_timeouts.tsv")
    failure_rows = []
    for row in mdl_failures:
        derived = (
            "unsupported_short_input"
            if "too_few_frames" in row.get("failure_reason", "")
            else row.get("failure_class", "")
        )
        failure_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["domain_id"],
                "legacy_failure_class": row.get("failure_class", ""),
                "derived_failure_class": derived,
                "failure_reason": row.get("failure_reason", ""),
                "n_frames": row.get("n_frames", ""),
            }
        )
    short_count = sum(
        row["derived_failure_class"] == "unsupported_short_input"
        for row in failure_rows
    )
    if short_count != 14:
        raise RuntimeError(f"expected 14 unsupported short inputs, observed {short_count}")
    write_tsv(
        OUT / "p1_corrected_reporting/corrected_failure_classification.tsv",
        failure_rows,
    )
    summary_rows = [
        {"category": "registered_catastrophic_quality_failure", "count": 8, "denominator": 10000, "definition": "prospectively registered finite/transform/effective-core/core-Jaccard/RMSF-Spearman/integrity criteria"},
        {"category": "large_geometric_tail_median_RMSD_gt_1A", "count": 54, "denominator": 10000, "definition": "post-hoc median absolute RMSD difference versus exact above 1 A; review flag, not a registered catastrophe"},
        {"category": "unsupported_short_input", "count": 14, "denominator": len(mdl_failures), "definition": "fewer than 20 frames; renamed in derived reporting only"},
        {"category": "PDB_coordinate_range_failure", "count": len(mdl_failures) - short_count, "denominator": len(mdl_failures), "definition": "PDB 8.3 coordinate field overflow in the original uncentered MDLovoFit preparation"},
    ]
    write_tsv(OUT / "p1_corrected_reporting/corrected_benchmark_summary.tsv", summary_rows)
    write_text(
        OUT / "p1_corrected_reporting/corrected_benchmark_interpretation_report.md",
        "\n".join(
            [
                "# Corrected B3 benchmark interpretation",
                "",
                "The frozen raw TSVs were not modified. The original helper combined a post-hoc `median absolute RMSD difference > 1 A` rule with the prospectively registered catastrophic criteria.",
                "",
                "- Registered catastrophic quality failures: **8/10,000**.",
                "- Separate large-geometric-tail review cases: **54/10,000**.",
                "- The same eight registered cases are identified for CPU and GPU; all are triggered by exact-relative RMSF Spearman below 0.9.",
                "- MDLovoFit failures are **14 unsupported short inputs** and **45 original PDB coordinate-range failures**.",
                "- `unsupported_short_input` is a derived reporting label only; historical failure rows remain unchanged.",
            ]
        ),
    )

    metadata_rows = []
    exact_by_id = {row["trajectory_id"]: row for row in corrected}
    for row in iter_tsv(B3 / "b3_all_trajectory_results.tsv"):
        corrected_exact = exact_by_id.get(row["trajectory_id"], {})
        metadata_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["domain_id"],
                "domain_split": domain_split(row["domain_id"]),
                "legacy_confidence_category": row.get("confidence_category", ""),
                "legacy_warning_reason": row.get("warning_reason", ""),
                "solver_status": solver_status(row),
                "sampling_reproducibility_status": sampling_status(row),
                "approximation_review_status": approximation_status(row),
                "exact_completed": row.get("exact_completed", ""),
                "registered_catastrophic_quality_failure": corrected_exact.get(
                    "registered_catastrophic_quality_failure", ""
                ),
                "large_geometric_tail_median_RMSD_gt_1A": corrected_exact.get(
                    "large_geometric_tail_median_RMSD_gt_1A", ""
                ),
                "strict_fixed_budget_status": row.get("strict_fixed_budget_status", ""),
                "representative_subset_weight_spearman": row.get("representative_subset_weight_spearman", ""),
                "representative_subset_core_jaccard": row.get("representative_subset_core_jaccard", ""),
                "temporal_block_weight_spearman": row.get("temporal_block_weight_spearman", ""),
                "temporal_block_core_jaccard": row.get("temporal_block_core_jaccard", ""),
            }
        )
    write_tsv(OUT / "p2_confidence_v2/confidence_metadata_v2.tsv", metadata_rows)

    source_by_id = {
        row["trajectory_id"]: row
        for row in iter_tsv(B3 / "b3_exact_quality_subset.tsv")
    }
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for metadata in metadata_rows:
        source = source_by_id.get(metadata["trajectory_id"])
        if source is None:
            continue
        key = (
            metadata["domain_split"],
            metadata["solver_status"],
            metadata["sampling_reproducibility_status"],
            metadata["approximation_review_status"],
        )
        groups[key].append(source)
    calibration = []
    for key, rows in sorted(groups.items()):
        ids = {row["trajectory_id"] for row in rows}
        corrected_group = [
            row for row in corrected if row["trajectory_id"] in ids
        ]
        calibration.append(
            {
                "domain_split": key[0],
                "solver_status": key[1],
                "sampling_reproducibility_status": key[2],
                "approximation_review_status": key[3],
                "n_trajectories": len(rows),
                "n_domains": len({row["domain_id"] for row in rows}),
                "registered_catastrophic_count": sum(
                    truth(row["registered_catastrophic_quality_failure"])
                    for row in corrected_group
                ),
                "large_geometric_tail_count": sum(
                    truth(row["large_geometric_tail_median_RMSD_gt_1A"])
                    for row in corrected_group
                ),
                "median_cpu_core_jaccard": median(rows, "cpu_exact_core_jaccard_vs_exact"),
                "median_cpu_weight_spearman": median(rows, "cpu_exact_weight_spearman_vs_exact"),
                "median_cpu_RMSF_spearman": median(rows, "cpu_exact_rmsf_spearman_vs_exact"),
                "median_cpu_RMSF_MAE_A": median(rows, "cpu_exact_rmsf_mae_vs_exact_A"),
                "median_cpu_RMSD_difference_A": median(rows, "cpu_exact_median_absolute_rmsd_difference_vs_exact_A"),
            }
        )
    write_tsv(OUT / "p2_confidence_v2/confidence_v2_calibration.tsv", calibration)

    domain_assignments = defaultdict(set)
    for row in metadata_rows:
        domain_assignments[row["domain_id"]].add(row["domain_split"])
    leakage = [domain for domain, splits in domain_assignments.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"domain split leakage detected for {len(leakage)} domains")
    counts = Counter(
        (
            row["solver_status"],
            row["sampling_reproducibility_status"],
            row["approximation_review_status"],
        )
        for row in metadata_rows
    )
    write_json(
        OUT / "p2_confidence_v2/confidence_v2_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(metadata_rows),
            "domains": len(domain_assignments),
            "domain_split_leakage": 0,
            "exact_calibration_rows": len(source_by_id),
            "complex_black_box_classifier_used": False,
            "legacy_fields_preserved": True,
            "status_combination_counts": {
                "|".join(key): value for key, value in sorted(counts.items())
            },
        },
    )
    write_text(
        OUT / "p2_confidence_v2/confidence_v2_report.md",
        "\n".join(
            [
                "# Confidence metadata v2",
                "",
                f"- Derived rows: `{len(metadata_rows):,}` across `{len(domain_assignments):,}` domains.",
                "- Development/validation assignment is deterministic by domain; replicate leakage is zero.",
                "- Solver completion, sampling reproducibility and exact-relative approximation review are independent fields.",
                "- Legacy confidence and warning fields are retained verbatim for provenance.",
                "- No classifier was trained around the eight registered catastrophic cases.",
                "- Numerical or temporal warnings are not automatically relabelled as globally low confidence.",
            ]
        ),
    )


if __name__ == "__main__":
    main()
