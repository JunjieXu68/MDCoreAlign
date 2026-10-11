#!/usr/bin/env python
"""Derive and irreversibly lock v3 gates from calibration evidence only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    destination = OUT / "03_gate_calibration/v3_gate_registry.json"
    if destination.exists():
        raise RuntimeError("v3 gate is already locked")
    summary = {
        row["perturbation_class"]: row
        for row in read_tsv(OUT / "02_self_variability/self_variability_summary.tsv")
    }
    mild_p95 = float(summary["mild_path"]["p95_across_case_level_p95_rmsd_difference_A"])
    exact_threshold = min(0.15, max(0.05, mild_p95))
    balanced_threshold = min(0.25, max(0.10, 1.5 * mild_p95))
    gates = {
        "registry_version": "cpu_chronology_v3_locked_1",
        "calibration_source": "02_self_variability/self_variability_summary.tsv",
        "calibration_panel_sha256": json.loads(
            (OUT / "02_self_variability/calibration_panel_lock.json").read_text()
        )["panel_sha256"],
        "mild_path_p95_across_case_level_p95_rmsd_difference_A": mild_p95,
        "contextual_sensitivity_used_for_threshold": False,
        "exact_like": {
            "catastrophic_mismatch_count_max": 0,
            "median_core_jaccard_min": 0.99,
            "p5_core_jaccard_min": 0.97,
            "minimum_non_tail_core_jaccard_min": 0.90,
            "median_weight_spearman_min": 0.9995,
            "p5_weight_spearman_min": 0.997,
            "median_rmsd_absolute_difference_A_max": 0.01,
            "p95_of_case_p95_rmsd_difference_A_max": exact_threshold,
            "median_stable_core_rmsf_spearman_min": 0.9995,
            "p5_stable_core_rmsf_spearman_min": 0.998,
            "median_rmsf_mae_A_max": 0.01,
            "non_finite_transform_count_max": 0,
            "fresh_full_panel_median_speedup_min": 2.0,
            "fresh_long_median_speedup_min": 1.5,
        },
        "balanced": {
            "catastrophic_mismatch_count_max": 0,
            "median_core_jaccard_min": 0.98,
            "p5_core_jaccard_min": 0.90,
            "minimum_core_jaccard_min": 0.80,
            "median_weight_spearman_min": 0.999,
            "p5_weight_spearman_min": 0.99,
            "median_rmsd_absolute_difference_A_max": 0.02,
            "p95_of_case_p95_rmsd_difference_A_max": balanced_threshold,
            "median_stable_core_rmsf_spearman_min": 0.999,
            "p5_stable_core_rmsf_spearman_min": 0.995,
            "median_rmsf_mae_A_max": 0.02,
            "non_finite_transform_count_max": 0,
            "fresh_full_panel_median_speedup_min": 3.0,
            "fresh_long_median_speedup_min": 2.0,
        },
        "adaptive_additional": {
            "fresh_long_median_speedup_min": 2.5,
            "D64_refinement_rate_max": 0.25,
            "unresolved_high_risk_fraction_max": 0.05,
            "must_improve_pareto_vs_current_throughput": True,
        },
    }
    destination.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    rows = []
    for target in ("exact_like", "balanced"):
        for metric, value in gates[target].items():
            rows.append({"target_class": target, "metric": metric, "locked_value": value})
    for metric, value in gates["adaptive_additional"].items():
        rows.append({"target_class": "adaptive_additional", "metric": metric, "locked_value": value})
    tsv = OUT / "03_gate_calibration/v3_gate_registry.tsv"
    with tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (OUT / "03_gate_calibration/gate_registry.sha256").write_text(
        f"{digest}  v3_gate_registry.json\n"
    )
    (OUT / "03_gate_calibration/gate_derivation_report.md").write_text(
        f"""# V3 gate derivation report

- Source: locked V1 calibration panel only.
- Mild-path P95 across case-level P95 absolute RMSD differences: `{mild_p95:.12g} A`.
- Locked exact-like RMSD tail threshold: `{exact_threshold:.6g} A`.
- Locked balanced RMSD tail threshold: `{balanced_threshold:.6g} A`.
- Numerical repeatability, core, weight and RMSF protections remain fixed as registered.
- Contextual reference/subset sensitivity was reported but did not inflate either threshold.
- The JSON registry was hashed before fresh panel selection or inspection.
"""
    )
    print(json.dumps({"gate_sha256": digest, "T_exact_rmsd_A": exact_threshold, "T_balanced_rmsd_A": balanced_threshold}, indent=2))


if __name__ == "__main__":
    main()
