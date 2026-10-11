#!/usr/bin/env python3
"""Audit the locked B3 pilot against the prospectively registered gates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.database_scale_benchmark_v2.common import (
    B2_ROOT,
    B3_ROOT,
    read_tsv,
    write_json,
    write_tsv,
)


def _number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def classify_catastrophic(row: dict[str, Any]) -> list[str]:
    """Apply only the catastrophic criteria registered before B3."""

    reasons: list[str] = []
    if row.get("status") != "success" or not _truth(row.get("cpu_exact_finite_outputs")):
        reasons.append("non_finite_or_failed_output")
    effective = _number(row, "cpu_exact_effective_core_fraction")
    if np.isfinite(effective) and effective < 0.10:
        reasons.append("effective_core_fraction_below_0.10")
    core_jaccard = _number(row, "cpu_exact_core_jaccard_vs_exact")
    if np.isfinite(core_jaccard) and core_jaccard < 0.50:
        reasons.append("core_jaccard_below_0.50")
    rmsf_spearman = _number(row, "cpu_exact_rmsf_spearman_vs_exact")
    if np.isfinite(rmsf_spearman) and rmsf_spearman < 0.90:
        reasons.append("RMSF_spearman_below_0.90")
    if row.get("failure_class") in {"unit_integrity_failure", "mapping_integrity_failure"}:
        reasons.append(str(row["failure_class"]))
    return reasons


def audit_rows(
    rows: list[dict[str, Any]], gates: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return summary, gate rows, catastrophic rows, and CPU/GPU mismatches."""

    success = [row for row in rows if row.get("status") == "success"]
    integrity = [
        row
        for row in rows
        if row.get("failure_class") in {"unit_integrity_failure", "mapping_integrity_failure"}
    ]
    numerical = [
        row for row in rows if row.get("failure_class") == "numerical_or_implementation_failure"
    ]
    catastrophic: list[dict[str, Any]] = []
    for row in rows:
        reasons = classify_catastrophic(row)
        if reasons:
            catastrophic.append(
                {
                    "trajectory_id": row.get("trajectory_id", ""),
                    "domain_id": row.get("domain_id", ""),
                    "reasons": ";".join(reasons),
                    "cpu_exact_core_jaccard": row.get("cpu_exact_core_jaccard_vs_exact", ""),
                    "cpu_exact_RMSF_spearman": row.get("cpu_exact_rmsf_spearman_vs_exact", ""),
                    "cpu_exact_RMSF_MAE_A": row.get("cpu_exact_rmsf_mae_vs_exact_A", ""),
                    "cpu_exact_median_absolute_RMSD_difference_A": row.get(
                        "cpu_exact_median_absolute_rmsd_difference_vs_exact_A", ""
                    ),
                    "cpu_exact_effective_core_fraction": row.get(
                        "cpu_exact_effective_core_fraction", ""
                    ),
                    "confidence_category": row.get("confidence_category", ""),
                    "warning_reason": row.get("warning_reason", ""),
                }
            )
    mismatches = [
        {
            "trajectory_id": row.get("trajectory_id", ""),
            "domain_id": row.get("domain_id", ""),
            "cpu_gpu_core_jaccard": row.get("cpu_gpu_core_jaccard", ""),
            "cpu_gpu_maximum_weight_difference": row.get(
                "cpu_gpu_maximum_weight_difference", ""
            ),
            "cpu_gpu_core_cutoff_minimum_margin": row.get(
                "cpu_gpu_core_cutoff_minimum_margin", ""
            ),
            "cpu_gpu_core_difference_explained_by_numerical_tie": row.get(
                "cpu_gpu_core_difference_explained_by_numerical_tie", ""
            ),
        }
        for row in success
        if _number(row, "cpu_gpu_core_jaccard") < 1.0
    ]

    core_values = np.asarray(
        [_number(row, "cpu_exact_core_jaccard_vs_exact") for row in success], dtype=float
    )
    rmsf_values = np.asarray(
        [_number(row, "cpu_exact_rmsf_spearman_vs_exact") for row in success], dtype=float
    )
    gpu_values = np.asarray(
        [_number(row, "cpu_gpu_core_jaccard") for row in success], dtype=float
    )
    catastrophic_fraction = len(catastrophic) / len(success) if success else float("nan")
    metrics = {
        "valid_input_data_mapping_numerical_failures": len(integrity) + len(numerical),
        "cpu_gpu_core_jaccard_min": float(np.nanmin(gpu_values)) if gpu_values.size else float("nan"),
        "catastrophic_exact_relative_failure_count": len(catastrophic),
        "catastrophic_exact_relative_failure_fraction": catastrophic_fraction,
        "median_exact_relative_core_jaccard": (
            float(np.nanmedian(core_values)) if core_values.size else float("nan")
        ),
        "median_exact_relative_RMSF_spearman": (
            float(np.nanmedian(rmsf_values)) if rmsf_values.size else float("nan")
        ),
    }
    gate_specs = [
        (
            "valid_input_data_mapping_numerical_failures",
            metrics["valid_input_data_mapping_numerical_failures"],
            "<=",
            gates["valid_input_data_mapping_numerical_failures_max"],
        ),
        (
            "CPU_GPU_core_jaccard",
            metrics["cpu_gpu_core_jaccard_min"],
            ">=",
            gates["CPU_GPU_core_jaccard_required"],
        ),
        (
            "catastrophic_exact_relative_failure_fraction",
            metrics["catastrophic_exact_relative_failure_fraction"],
            "<",
            gates["catastrophic_exact_relative_failure_fraction_max"],
        ),
        (
            "median_exact_relative_core_jaccard",
            metrics["median_exact_relative_core_jaccard"],
            ">=",
            gates["median_exact_relative_core_jaccard_min"],
        ),
        (
            "median_exact_relative_RMSF_spearman",
            metrics["median_exact_relative_RMSF_spearman"],
            ">=",
            gates["median_exact_relative_RMSF_spearman_min"],
        ),
    ]
    gate_rows = []
    for name, observed, operator, threshold in gate_specs:
        passed = (
            observed <= threshold
            if operator == "<="
            else observed < threshold
            if operator == "<"
            else observed >= threshold
        )
        gate_rows.append(
            {
                "gate": name,
                "observed": observed,
                "operator": operator,
                "registered_threshold": threshold,
                "status": "pass" if passed else "fail",
                "blocks_full_run": name == "valid_input_data_mapping_numerical_failures",
            }
        )
    summary = {
        "pilot_rows": len(rows),
        "success": len(success),
        "integrity_failures": len(integrity),
        "numerical_or_implementation_failures": len(numerical),
        **metrics,
        "cpu_gpu_core_mismatch_count": len(mismatches),
        "all_prospective_quality_gates_pass": all(row["status"] == "pass" for row in gate_rows),
        "full_run_authorized_under_registered_stop_rule": not integrity and not numerical,
        "authorization_rationale": (
            "Continue because no systematic input-integrity, mapping, or numerical failure was "
            "observed; quality-gate failures remain prospectively recorded as warnings."
        ),
        "post_hoc_threshold_tuning": False,
    }
    return summary, gate_rows, catastrophic, mismatches


def main() -> None:
    rows = read_tsv(B3_ROOT / "b3_exact_pilot_results.tsv")
    gates = json.loads((B2_ROOT / "prospective_quality_gates.json").read_text())
    summary, gate_rows, catastrophic, mismatches = audit_rows(rows, gates)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **summary}
    write_json(B3_ROOT / "b3_pilot_quality_gate_audit.json", payload)
    write_tsv(B3_ROOT / "b3_pilot_quality_gate_audit.tsv", gate_rows)
    write_tsv(B3_ROOT / "b3_pilot_catastrophic_tail_atlas.tsv", catastrophic)
    write_tsv(B3_ROOT / "b3_pilot_cpu_gpu_core_mismatches.tsv", mismatches)

    failures = [row for row in gate_rows if row["status"] == "fail"]
    report = [
        "# B3 Exact-Pilot Prospective Gate Audit",
        "",
        f"- Pilot rows: {summary['pilot_rows']:,}",
        f"- Successful rows: {summary['success']:,}",
        f"- Input/mapping integrity failures: {summary['integrity_failures']}",
        f"- Numerical or implementation failures: {summary['numerical_or_implementation_failures']}",
        f"- Prospective quality gates passed: {len(gate_rows) - len(failures)}/{len(gate_rows)}",
        "- Thresholds were not changed after inspecting the pilot.",
        "",
        "## Decision",
        "",
        summary["authorization_rationale"],
        "The full deployment is therefore authorised under the registered stop rule, but the "
        "pilot must not be described as passing every practical quality gate.",
        "",
        "## Failed prospective quality gates",
        "",
    ]
    report.extend(
        f"- `{row['gate']}`: observed {row['observed']}, required {row['operator']} "
        f"{row['registered_threshold']}."
        for row in failures
    )
    report.extend(
        [
            "",
            "## Tail interpretation",
            "",
            f"- Catastrophic exact-relative tails: {len(catastrophic)}/{len(rows)} "
            f"({100 * summary['catastrophic_exact_relative_failure_fraction']:.3f}%).",
            f"- CPU/GPU hard-core mismatches: {len(mismatches)}/{len(rows)}.",
            "- These rows remain in the database benchmark and are stratified rather than hidden.",
            "- Only systematic input-integrity, mapping, or numerical failures stop deployment.",
            "",
        ]
    )
    (B3_ROOT / "b3_pilot_quality_gate_audit.md").write_text("\n".join(report))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
