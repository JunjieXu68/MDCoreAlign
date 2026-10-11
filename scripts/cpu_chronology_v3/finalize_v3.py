#!/usr/bin/env python
"""Aggregate locked v3 quality/performance evidence and apply decision rules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3"
METHODS = (
    "throughput_compact_f64",
    "D_block_64_v3",
    "C_correction_2_v3",
    "adaptive_C1_C2_D64_v3",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def numbers(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def aggregate(rows: list[dict[str, str]], method: str, panel: str) -> dict[str, object]:
    subset = [row for row in rows if row["method"] == method]
    if not subset:
        raise ValueError(f"no rows for {panel}/{method}")
    return {
        "panel": panel,
        "method": method,
        "n_cases": len(subset),
        "median_core_jaccard": float(np.median(numbers(subset, "core_jaccard"))),
        "p5_core_jaccard": float(np.quantile(numbers(subset, "core_jaccard"), 0.05)),
        "minimum_core_jaccard": float(np.min(numbers(subset, "core_jaccard"))),
        "median_weight_spearman": float(np.median(numbers(subset, "weight_spearman"))),
        "p5_weight_spearman": float(np.quantile(numbers(subset, "weight_spearman"), 0.05)),
        "median_case_median_rmsd_difference_A": float(
            np.median(numbers(subset, "rmsd_absolute_difference_median_A"))
        ),
        "p95_across_case_p95_rmsd_difference_A": float(
            np.quantile(numbers(subset, "rmsd_absolute_difference_p95_A"), 0.95)
        ),
        "maximum_case_p95_rmsd_difference_A": float(
            np.max(numbers(subset, "rmsd_absolute_difference_p95_A"))
        ),
        "median_rmsf_spearman": float(
            np.median(numbers(subset, "stable_core_rmsf_spearman"))
        ),
        "p5_rmsf_spearman": float(
            np.quantile(numbers(subset, "stable_core_rmsf_spearman"), 0.05)
        ),
        "median_rmsf_mae_A": float(np.median(numbers(subset, "stable_core_rmsf_mae_A"))),
        "catastrophic_mismatch_count": sum(row["catastrophic_mismatch"] == "True" for row in subset),
        "non_finite_transform_count": sum(row["non_finite_transform"] == "True" for row in subset),
    }


def quality_pass(row: dict[str, object], gate: dict[str, object], kind: str) -> tuple[bool, list[str]]:
    value = lambda key: float(row[key])
    checks = {
        "catastrophic mismatch": value("catastrophic_mismatch_count") <= gate["catastrophic_mismatch_count_max"],
        "median core Jaccard": value("median_core_jaccard") >= gate["median_core_jaccard_min"],
        "P5 core Jaccard": value("p5_core_jaccard") >= gate["p5_core_jaccard_min"],
        "median weight Spearman": value("median_weight_spearman") >= gate["median_weight_spearman_min"],
        "P5 weight Spearman": value("p5_weight_spearman") >= gate["p5_weight_spearman_min"],
        "median RMSD difference": value("median_case_median_rmsd_difference_A") <= gate["median_rmsd_absolute_difference_A_max"],
        "P95-of-case-P95 RMSD": value("p95_across_case_p95_rmsd_difference_A") <= gate["p95_of_case_p95_rmsd_difference_A_max"],
        "median RMSF Spearman": value("median_rmsf_spearman") >= gate["median_stable_core_rmsf_spearman_min"],
        "P5 RMSF Spearman": value("p5_rmsf_spearman") >= gate["p5_stable_core_rmsf_spearman_min"],
        "median RMSF MAE": value("median_rmsf_mae_A") <= gate["median_rmsf_mae_A_max"],
        "finite transforms": value("non_finite_transform_count") <= gate["non_finite_transform_count_max"],
    }
    minimum_key = "minimum_non_tail_core_jaccard_min" if kind == "exact_like" else "minimum_core_jaccard_min"
    checks["minimum core Jaccard"] = value("minimum_core_jaccard") >= gate[minimum_key]
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def paired_speedups(path: Path, candidate: str) -> np.ndarray:
    rows = [row for row in read_tsv(path) if row["timing_kind"] == "loaded_in_process"]
    case_ids = sorted({row["case_id"] for row in rows})
    output = []
    for case_id in case_ids:
        exact = np.median(
            [float(row["loaded_runtime_seconds"]) for row in rows if row["case_id"] == case_id and row["method"] == "exact_sequential_f64_compact"]
        )
        value = np.median(
            [float(row["loaded_runtime_seconds"]) for row in rows if row["case_id"] == case_id and row["method"] == candidate]
        )
        output.append(exact / value)
    return np.asarray(output)


def throughput_relative_speed(path: Path, candidate: str) -> float:
    rows = [row for row in read_tsv(path) if row["timing_kind"] == "loaded_in_process"]
    values = []
    for case_id in sorted({row["case_id"] for row in rows}):
        baseline = np.median([float(row["loaded_runtime_seconds"]) for row in rows if row["case_id"] == case_id and row["method"] == "throughput_compact_f64"])
        candidate_time = np.median([float(row["loaded_runtime_seconds"]) for row in rows if row["case_id"] == case_id and row["method"] == candidate])
        values.append(baseline / candidate_time)
    return float(np.median(values))


def quality_phase() -> None:
    panels = {
        "fresh_mdcath_512": read_tsv(OUT / "06_validation/fresh_mdcath_quality.tsv"),
        "fresh_long_R2_R3": read_tsv(OUT / "06_validation/fresh_long_quality.tsv"),
        "historical_stress": read_tsv(OUT / "06_validation/historical_stress_quality.tsv"),
    }
    summary = [aggregate(rows, method, panel) for panel, rows in panels.items() for method in METHODS]
    write_tsv(OUT / "06_validation/quality_summary.tsv", summary)
    fresh_rows = panels["fresh_mdcath_512"] + panels["fresh_long_R2_R3"]
    combined = [aggregate(fresh_rows, method, "fresh_combined") for method in METHODS]
    write_tsv(OUT / "06_validation/fresh_combined_quality_summary.tsv", combined)
    path_rows = read_tsv(OUT / "06_validation/adaptive_refinement_paths.tsv")
    fresh_paths = [row for row in path_rows if row["panel"] in {"fresh_mdcath", "fresh_long"}]
    route_counts = {level: sum(row["accepted_level"] == level for row in fresh_paths) for level in ("C1", "C2", "D64")}
    report = [
        "# V3 fresh quality validation",
        "",
        f"- Fresh mdCATH domains: `{len(panels['fresh_mdcath_512']) // 5}`.",
        f"- Fresh ATLAS R2/R3 trajectories: `{len(panels['fresh_long_R2_R3']) // 5}` from `{len({row['family_id'] for row in panels['fresh_long_R2_R3']})}` families.",
        f"- Historical stress cases (context only): `{len(panels['historical_stress']) // 5}`.",
        f"- Adaptive routes on fresh panels: C1 `{route_counts['C1']}`, C2 `{route_counts['C2']}`, D64 `{route_counts['D64']}`.",
        "- Locked gate thresholds were not modified after fresh results were opened.",
        "",
        "Final gate decisions also require V6 paired timing and are written only after performance completion.",
    ]
    (OUT / "06_validation/quality_validation_report.md").write_text("\n".join(report) + "\n")


def final_phase() -> None:
    gates = json.loads((OUT / "03_gate_calibration/v3_gate_registry.json").read_text())
    fresh = read_tsv(OUT / "06_validation/fresh_combined_quality_summary.tsv")
    long_path = OUT / "07_performance/fresh_long_repeats.tsv"
    mdcath_path = OUT / "07_performance/fresh_mdcath_128_repeats.tsv"
    gate_rows: list[dict[str, object]] = []
    pareto_rows: list[dict[str, object]] = []
    path_rows = [row for row in read_tsv(OUT / "06_validation/adaptive_refinement_paths.tsv") if row["panel"] in {"fresh_mdcath", "fresh_long"}]
    for row in fresh:
        method = row["method"]
        exact_quality, exact_failures = quality_pass(row, gates["exact_like"], "exact_like")
        balanced_quality, balanced_failures = quality_pass(row, gates["balanced"], "balanced")
        full_speed = float(np.median(paired_speedups(mdcath_path, method)))
        long_speed = float(np.median(paired_speedups(long_path, method)))
        exact_pass = bool(
            exact_quality
            and full_speed >= gates["exact_like"]["fresh_full_panel_median_speedup_min"]
            and long_speed >= gates["exact_like"]["fresh_long_median_speedup_min"]
        )
        balanced_pass = bool(
            balanced_quality
            and full_speed >= gates["balanced"]["fresh_full_panel_median_speedup_min"]
            and long_speed >= gates["balanced"]["fresh_long_median_speedup_min"]
        )
        throughput_speed = throughput_relative_speed(mdcath_path, method)
        throughput_row = next(item for item in fresh if item["method"] == "throughput_compact_f64")
        pareto_new = bool(
            method != "throughput_compact_f64"
            and full_speed > 1.0
            and (
                float(row["p95_across_case_p95_rmsd_difference_A"]) < float(throughput_row["p95_across_case_p95_rmsd_difference_A"])
                or float(row["minimum_core_jaccard"]) > float(throughput_row["minimum_core_jaccard"])
            )
        )
        gate_rows.append(
            {
                **row,
                "exact_like_quality_only_pass": exact_quality,
                "exact_like_quality_failures": "; ".join(exact_failures),
                "balanced_quality_only_pass": balanced_quality,
                "balanced_quality_failures": "; ".join(balanced_failures),
                "fresh_mdcath_128_median_speedup_vs_exact": full_speed,
                "fresh_long_median_speedup_vs_exact": long_speed,
                "exact_like_full_pass": exact_pass,
                "balanced_full_pass": balanced_pass,
                "median_runtime_ratio_vs_throughput": throughput_speed,
                "new_pareto_position_vs_throughput": pareto_new,
            }
        )
        pareto_rows.append(
            {
                "method": method,
                "fresh_mdcath_speedup_vs_exact": full_speed,
                "fresh_long_speedup_vs_exact": long_speed,
                "runtime_ratio_vs_throughput": throughput_speed,
                "median_core_jaccard": row["median_core_jaccard"],
                "minimum_core_jaccard": row["minimum_core_jaccard"],
                "p95_of_case_p95_rmsd_difference_A": row["p95_across_case_p95_rmsd_difference_A"],
                "median_rmsf_spearman": row["median_rmsf_spearman"],
                "new_pareto_position_vs_throughput": pareto_new,
            }
        )
    write_tsv(OUT / "06_validation/v3_gate_results.tsv", gate_rows)
    write_tsv(OUT / "08_decision/pareto_comparison.tsv", pareto_rows)
    long_rows = [
        {
            "method": method,
            "median_speedup_vs_exact": float(np.median(paired_speedups(long_path, method))),
            "median_runtime_ratio_vs_throughput": throughput_relative_speed(long_path, method),
            **next(row for row in fresh if row["method"] == method),
        }
        for method in METHODS
    ]
    write_tsv(OUT / "08_decision/long_trajectory_pareto.tsv", long_rows)

    all_quality = read_tsv(OUT / "06_validation/fresh_mdcath_quality.tsv") + read_tsv(OUT / "06_validation/fresh_long_quality.tsv") + read_tsv(OUT / "06_validation/historical_stress_quality.tsv")
    tails = [
        {
            "case_id": row["case_id"],
            "panel": row["panel"],
            "method": row["method"],
            "core_jaccard": row["core_jaccard"],
            "weight_spearman": row["weight_spearman"],
            "rmsd_p95_A": row["rmsd_absolute_difference_p95_A"],
            "rmsf_spearman": row["stable_core_rmsf_spearman"],
            "catastrophic_mismatch": row["catastrophic_mismatch"],
        }
        for row in all_quality
        if row["method"] != "exact_sequential_f64_compact"
        and (
            row["catastrophic_mismatch"] == "True"
            or float(row["core_jaccard"]) < 0.8
            or float(row["rmsd_absolute_difference_p95_A"]) > gates["balanced"]["p95_of_case_p95_rmsd_difference_A_max"]
            or float(row["stable_core_rmsf_spearman"]) < gates["balanced"]["p5_stable_core_rmsf_spearman_min"]
        )
    ]
    write_tsv(OUT / "08_decision/tail_case_summary.tsv", tails)

    by_method = {row["method"]: row for row in gate_rows}
    adaptive_paths = [row for row in path_rows if row["panel"] in {"fresh_mdcath", "fresh_long"}]
    d64_rate = sum(row["accepted_level"] == "D64" for row in adaptive_paths) / len(adaptive_paths)
    adaptive_quality_rows = [row for row in all_quality if row["method"] == "adaptive_C1_C2_D64_v3" and row["panel"] in {"fresh_mdcath", "fresh_long"}]
    unresolved_rate = sum(
        row["catastrophic_mismatch"] == "True"
        or row["non_finite_transform"] == "True"
        or float(row["core_jaccard"]) < gates["balanced"]["minimum_core_jaccard_min"]
        or float(row["rmsd_absolute_difference_p95_A"]) > gates["balanced"]["p95_of_case_p95_rmsd_difference_A_max"]
        for row in adaptive_quality_rows
    ) / len(adaptive_quality_rows)
    adaptive = by_method["adaptive_C1_C2_D64_v3"]
    adaptive_review = bool(
        adaptive["balanced_full_pass"]
        and float(adaptive["fresh_long_median_speedup_vs_exact"]) >= gates["adaptive_additional"]["fresh_long_median_speedup_min"]
        and d64_rate <= gates["adaptive_additional"]["D64_refinement_rate_max"]
        and unresolved_rate <= gates["adaptive_additional"]["unresolved_high_risk_fraction_max"]
        and adaptive["new_pareto_position_vs_throughput"]
    )
    decisions = [
        {
            "candidate": "D_block_64_v3",
            "decision": "exact-like blockwise estimator for human review" if by_method["D_block_64_v3"]["exact_like_full_pass"] else "negative_or_near_pass",
            "exact_like_pass": by_method["D_block_64_v3"]["exact_like_full_pass"],
            "balanced_pass": by_method["D_block_64_v3"]["balanced_full_pass"],
        },
        {
            "candidate": "C_correction_2_v3",
            "decision": "candidate fast estimator for human review" if (by_method["C_correction_2_v3"]["exact_like_full_pass"] or by_method["C_correction_2_v3"]["balanced_full_pass"]) else "rejected_under_locked_gate",
            "exact_like_pass": by_method["C_correction_2_v3"]["exact_like_full_pass"],
            "balanced_pass": by_method["C_correction_2_v3"]["balanced_full_pass"],
        },
        {
            "candidate": "adaptive_C1_C2_D64_v3",
            "decision": "promote_for_human_review" if adaptive_review else "rejected_under_locked_gate",
            "exact_like_pass": adaptive["exact_like_full_pass"],
            "balanced_pass": adaptive["balanced_full_pass"],
            "D64_refinement_rate": d64_rate,
            "unresolved_high_risk_fraction": unresolved_rate,
        },
    ]
    write_tsv(OUT / "08_decision/candidate_decisions.tsv", decisions)
    passed = [row for row in decisions if "review" in row["decision"]]
    final = "At least one internal candidate passed its complete locked v3 decision rule." if passed else "No candidate passed the complete locked v3 decision rules. Chronology optimisation stops without another block-size or correction-count grid."
    (OUT / "08_decision/final_v3_decision.md").write_text(
        "# Final chronology-calibrated v3 decision\n\n"
        "> The v2 result is unchanged: no candidate passed the original pre-registered v2 gate.\n\n"
        f"{final}\n\n"
        f"Adaptive D64 refinement rate: `{d64_rate:.4f}`; unresolved high-risk fraction: `{unresolved_rate:.4f}`.\n\n"
        "No internal candidate is exposed as a public user mode automatically. Sequential float64 exact remains unchanged.\n"
    )
    (OUT / "08_decision/product_integration_proposal.md").write_text(
        "# Product integration proposal\n\nSequential float64 exact remains the reference/default. Any passing v3 candidate requires explicit human approval and full product regression before internal substitution; D64, C2 and adaptive names are not public modes.\n"
    )
    (OUT / "08_decision/manuscript_impact.md").write_text(
        "# Manuscript impact\n\nThis v3 study is a separate prospective experiment and does not alter the historical v2 negative conclusion or the manuscript automatically. Only locked-gate, matched-contract results may be considered after author review.\n"
    )
    (OUT / "08_decision/next_branch_if_failed.md").write_text(
        "# Next branch if v3 fails\n\nIf no candidate passes, stop chronology tuning and evaluate the algorithm-changing stream-alignment branch (sparse landmarks, adaptive keyframes, approximate RMSF sampling, or local interval refinement) under a new prospective protocol.\n"
    )
    (OUT / "07_performance/performance_report.md").write_text(
        "# V3 performance report\n\nAll timing used loaded float64 coordinates, one CPU thread, ordinary RMSF disabled, no aligned-coordinate materialisation, one warmup and five measured repeats. Fresh-process timings are reported separately for ATLAS R2/R3. No external tool was rerun.\n"
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("quality", "final"))
    args = parser.parse_args()
    quality_phase() if args.phase == "quality" else final_phase()


if __name__ == "__main__":
    main()
