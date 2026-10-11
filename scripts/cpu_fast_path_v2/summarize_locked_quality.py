#!/usr/bin/env python
"""Summarize locked held-out and challenge quality before performance runs."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_fast_path_v2"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def aggregate(
    configuration: str,
    heldout: list[dict[str, str]],
    challenge: list[dict[str, str]],
) -> dict[str, object]:
    non_tail = [row for row in heldout if row["configuration"] == configuration]
    tails = [row for row in challenge if row["configuration"] == configuration]
    combined = non_tail + tails
    return {
        "configuration": configuration,
        "n_heldout": len(non_tail),
        "n_challenge": len(tails),
        "median_core_jaccard": float(np.median(values(combined, "core_jaccard"))),
        "p5_core_jaccard": float(np.quantile(values(combined, "core_jaccard"), 0.05)),
        "minimum_core_jaccard": float(np.min(values(combined, "core_jaccard"))),
        "minimum_non_tail_core_jaccard": float(np.min(values(non_tail, "core_jaccard"))),
        "median_weight_spearman": float(np.median(values(combined, "weight_spearman"))),
        "p5_weight_spearman": float(np.quantile(values(combined, "weight_spearman"), 0.05)),
        "median_rmsd_absolute_difference_A": float(
            np.median(values(combined, "rmsd_absolute_difference_median_A"))
        ),
        "p95_rmsd_absolute_difference_A": float(
            np.quantile(values(combined, "rmsd_absolute_difference_p95_A"), 0.95)
        ),
        "median_robust_rmsf_spearman": float(
            np.median(values(combined, "robust_rmsf_spearman"))
        ),
        "p5_robust_rmsf_spearman": float(
            np.quantile(values(combined, "robust_rmsf_spearman"), 0.05)
        ),
        "median_robust_rmsf_mae_A": float(
            np.median(values(combined, "robust_rmsf_mae_A"))
        ),
        "heldout_median_speedup": float(
            np.median(values(non_tail, "speedup_vs_reference"))
        ),
        "combined_median_speedup": float(
            np.median(values(combined, "speedup_vs_reference"))
        ),
        "catastrophic_count": sum(
            row["catastrophic_mismatch"] == "True" for row in combined
        ),
        "non_finite_count": sum(row["non_finite_output"] == "True" for row in combined),
    }


def exact_like_pass(row: dict[str, object]) -> bool:
    return bool(
        row["median_core_jaccard"] >= 0.99
        and row["p5_core_jaccard"] >= 0.97
        and row["minimum_non_tail_core_jaccard"] >= 0.90
        and row["median_weight_spearman"] >= 0.9995
        and row["p5_weight_spearman"] >= 0.997
        and row["median_rmsd_absolute_difference_A"] <= 0.01
        and row["p95_rmsd_absolute_difference_A"] <= 0.05
        and row["median_robust_rmsf_spearman"] >= 0.9995
        and row["p5_robust_rmsf_spearman"] >= 0.998
        and row["heldout_median_speedup"] >= 2.0
        and row["catastrophic_count"] == 0
    )


def balanced_pass(row: dict[str, object]) -> bool:
    return bool(
        row["median_core_jaccard"] >= 0.98
        and row["p5_core_jaccard"] >= 0.90
        and row["minimum_core_jaccard"] >= 0.80
        and row["median_weight_spearman"] >= 0.999
        and row["p5_weight_spearman"] >= 0.99
        and row["median_rmsd_absolute_difference_A"] <= 0.02
        and row["p95_rmsd_absolute_difference_A"] <= 0.10
        and row["median_robust_rmsf_spearman"] >= 0.999
        and row["p5_robust_rmsf_spearman"] >= 0.995
        and row["heldout_median_speedup"] >= 3.0
        and row["catastrophic_count"] == 0
    )


def main() -> None:
    heldout = read_tsv(OUT / "04_parallel_weights/heldout_results.tsv")
    challenge = read_tsv(OUT / "04_parallel_weights/challenge_results.tsv")
    configurations = sorted({row["configuration"] for row in heldout})
    summary = [aggregate(config, heldout, challenge) for config in configurations]
    exact_rows: list[dict[str, object]] = []
    balanced_rows: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in summary:
        exact_pass = exact_like_pass(row)
        balanced = balanced_pass(row)
        exact_rows.append({**row, "exact_like_pass": exact_pass})
        balanced_rows.append({**row, "balanced_pass": balanced})
        decision = {
            "configuration": row["configuration"],
            "exact_like_pass": exact_pass,
            "balanced_pass": balanced,
            "target_class": "exact_like" if exact_pass else "balanced" if balanced else "none",
        }
        (accepted if exact_pass or balanced else rejected).append(decision)
    write_tsv(OUT / "05_quality/exact_like_summary.tsv", exact_rows)
    write_tsv(OUT / "05_quality/balanced_summary.tsv", balanced_rows)
    write_tsv(OUT / "07_decision/accepted_changes.tsv", accepted)
    write_tsv(OUT / "07_decision/rejected_candidates.tsv", rejected)

    tail_rows: list[dict[str, object]] = []
    for row in challenge:
        tail_rows.append({
            "case_id": row["case_id"],
            "configuration": row["configuration"],
            "core_jaccard": row["core_jaccard"],
            "weight_spearman": row["weight_spearman"],
            "rmsd_absolute_difference_median_A": row["rmsd_absolute_difference_median_A"],
            "rmsd_absolute_difference_p95_A": row["rmsd_absolute_difference_p95_A"],
            "robust_rmsf_spearman": row["robust_rmsf_spearman"],
            "catastrophic_mismatch": row["catastrophic_mismatch"],
            "interpretation": (
                "large exact-relative disagreement"
                if row["catastrophic_mismatch"] == "True"
                else "bounded exact-relative difference"
            ),
        })
    write_tsv(OUT / "05_quality/tail_atlas.tsv", tail_rows)

    refinement_rows: list[dict[str, object]] = []
    for source, split in ((heldout, "heldout"), (challenge, "challenge")):
        for config in configurations:
            subset = [row for row in source if row["configuration"] == config]
            refinement_rows.append({
                "split": split,
                "configuration": config,
                "median_refinement_fraction": float(
                    np.median(values(subset, "refinement_fraction"))
                ),
                "minimum_refinement_fraction": float(
                    np.min(values(subset, "refinement_fraction"))
                ),
                "maximum_refinement_fraction": float(
                    np.max(values(subset, "refinement_fraction"))
                ),
            })
    write_tsv(OUT / "04_parallel_weights/refinement_rates.tsv", refinement_rows)

    natural_source = (
        ROOT
        / "results/scaling_optimization_v4/phase7a_rmsd_audit/exact_method_natural_variability.tsv"
    )
    natural_target = OUT / "05_quality/self_variability_comparison.tsv"
    shutil.copyfile(natural_source, natural_target)

    precision_lock = read_tsv(OUT / "03_precision_compact/locked_precision_candidate.tsv")[0]
    (OUT / "03_precision_compact/final_precision_decision.md").write_text(
        "# Final precision decision\n\n"
        "No float32 or mixed-precision candidate reached the prospectively "
        "registered 1.15x development speed gate under the matched compact "
        "contract. Float64 throughput is retained. Numerical agreement alone "
        "was not treated as sufficient for promotion.\n\n"
        f"Locked candidate: `{precision_lock['configuration']}`.\n"
    )


if __name__ == "__main__":
    main()
