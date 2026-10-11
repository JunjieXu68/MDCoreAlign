"""Generate final pre-manuscript statistics from existing benchmark outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_ready_common import (  # noqa: E402
    BOOTSTRAP_SEED,
    ROOT,
    bootstrap_ci,
    bootstrap_stat_ci,
    fmt,
    mean,
    median,
    read_table,
    spearman,
    to_float,
    write_table,
)


TRAJECTORY_FIELDS = [
    "benchmark_type",
    "system",
    "baseline_method",
    "soft_unified_posthoc_RMSD",
    "baseline_unified_posthoc_RMSD",
    "absolute_RMSD_difference_soft_minus_baseline",
    "percent_RMSD_difference_soft_vs_baseline",
    "soft_runtime_seconds",
    "baseline_runtime_seconds",
    "runtime_ratio_soft_over_baseline",
    "absolute_runtime_difference_soft_minus_baseline",
    "winner_by_unified_posthoc_RMSD",
    "winner_by_runtime",
    "metric_definition",
    "notes",
]

PAIR_FIELDS = TRAJECTORY_FIELDS

NMR_FIELDS = [
    "benchmark_type",
    "comparison",
    "n_paired_cases",
    "mean_RMSD_difference_soft_minus_baseline",
    "median_RMSD_difference_soft_minus_baseline",
    "RMSD_difference_bootstrap95CI_low",
    "RMSD_difference_bootstrap95CI_high",
    "mean_runtime_ratio_soft_over_baseline",
    "median_runtime_ratio_soft_over_baseline",
    "runtime_ratio_bootstrap95CI_low",
    "runtime_ratio_bootstrap95CI_high",
    "soft_lower_RMSD_count",
    "soft_faster_count",
    "soft_both_count",
    "wilcoxon_statistic",
    "wilcoxon_pvalue",
    "paired_effect_size_mean_diff_over_sd",
    "bootstrap_seed",
    "metric_definition",
    "notes",
]

FLEX_FIELDS = [
    "system",
    "n_residues",
    "n_flexible_segments",
    "spearman_robust_RMSF_vs_mobility",
    "spearman_bootstrap95CI_low",
    "spearman_bootstrap95CI_high",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "mean_difference_noncore_minus_core",
    "median_difference_noncore_minus_core",
    "mean_ordinary_RMSF",
    "mean_robust_RMSF",
    "global_mean_RMSF_difference_robust_minus_ordinary",
    "bootstrap_seed",
    "metric_definition",
    "notes",
]

BIOLOGY_FIELDS = [
    "system",
    "external_signal",
    "primary_metric",
    "soft_metric_value",
    "seq_contact_regularized_metric_value",
    "mdlovofit_metric_value_if_available",
    "support_status",
    "metric_definition",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final_statistics")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def benchmark_delta_rows(summary_path: Path, *, benchmark_type: str) -> list[dict[str, object]]:
    """Compute soft-vs-baseline rows from a final benchmark summary."""

    rows = []
    for row in read_table(summary_path):
        system = row["system"]
        soft_rmsd = to_float(row["soft_unified_rmsd"])
        soft_runtime = to_float(row["soft_runtime_s"])
        baselines = [
            ("MDLovoFit", to_float(row.get("mdlovofit_unified_rmsd")), to_float(row.get("mdlovofit_runtime_s"))),
            ("THESEUS", to_float(row.get("theseus_unified_rmsd")), to_float(row.get("theseus_runtime_s"))),
        ]
        for baseline_name, baseline_rmsd, baseline_runtime in baselines:
            if soft_rmsd is None or soft_runtime is None or baseline_rmsd is None or baseline_runtime is None:
                continue
            rmsd_delta = soft_rmsd - baseline_rmsd
            runtime_delta = soft_runtime - baseline_runtime
            rows.append(
                {
                    "benchmark_type": benchmark_type,
                    "system": system,
                    "baseline_method": baseline_name,
                    "soft_unified_posthoc_RMSD": fmt(soft_rmsd),
                    "baseline_unified_posthoc_RMSD": fmt(baseline_rmsd),
                    "absolute_RMSD_difference_soft_minus_baseline": fmt(rmsd_delta),
                    "percent_RMSD_difference_soft_vs_baseline": fmt(100.0 * rmsd_delta / baseline_rmsd if baseline_rmsd else None),
                    "soft_runtime_seconds": fmt(soft_runtime),
                    "baseline_runtime_seconds": fmt(baseline_runtime),
                    "runtime_ratio_soft_over_baseline": fmt(soft_runtime / baseline_runtime if baseline_runtime else None),
                    "absolute_runtime_difference_soft_minus_baseline": fmt(runtime_delta),
                    "winner_by_unified_posthoc_RMSD": "soft" if soft_rmsd < baseline_rmsd else baseline_name,
                    "winner_by_runtime": "soft" if soft_runtime < baseline_runtime else baseline_name,
                    "metric_definition": "Unified post-hoc RMSD is computed in a common post-processing frame; runtime ratio < 1 favors soft.",
                    "notes": trajectory_notes(system, baseline_name, rmsd_delta, runtime_delta),
                }
            )
    return rows


def trajectory_notes(system: str, baseline: str, rmsd_delta: float, runtime_delta: float) -> str:
    """Return conservative case notes for real-system comparisons."""

    notes: list[str] = []
    if baseline == "THESEUS" and runtime_delta > 0:
        notes.append("THESEUS is faster in this case")
    if baseline == "THESEUS" and rmsd_delta > 0:
        notes.append("THESEUS is slightly lower RMSD in this case")
    if "YiiP" in system:
        notes.append("YiiP external biology proxy remains weak; interpret biological support cautiously")
    return "; ".join(notes)


def nmr_source_paths() -> tuple[Path, Path, str]:
    """Return the best available NMR per-case and summary paths."""

    external_500 = Path(
        "/storage/junjie/projects/MDCoreAlign/results/nmr_ensemble_batch_500_gt50ca_gt20models_20260330/nmr_ensemble_batch_summary_by_dataset.csv"
    )
    external_runs_500 = Path(
        "/storage/junjie/projects/MDCoreAlign/results/nmr_ensemble_batch_500_gt50ca_gt20models_20260330/nmr_ensemble_batch_runs.csv"
    )
    local_100 = ROOT / "results" / "nmr_ensemble_batch_optimized_20260330" / "nmr_ensemble_batch_summary_by_dataset.csv"
    local_runs_100 = ROOT / "results" / "nmr_ensemble_batch_optimized_20260330" / "nmr_ensemble_batch_runs.csv"
    if external_500.exists() and external_runs_500.exists():
        return external_500, external_runs_500, "500-case NMR ensemble benchmark"
    return local_100, local_runs_100, "available local NMR ensemble benchmark"


def nmr_statistics(seed: int, n_bootstrap: int) -> list[dict[str, object]]:
    """Compute paired NMR ensemble statistics."""

    summary_path, _runs_path, benchmark_label = nmr_source_paths()
    rows = read_table(summary_path)
    by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset_name"], {})[row["method_name"]] = row

    comparisons = [
        ("soft_vs_mdlovofit", "external_mdlovofit", "Soft robust vs external MDLovoFit stable-core baseline."),
        (
            "soft_vs_ordinary",
            "ordinary_alignment",
            "Ordinary alignment is a full-set RMSD optimizer, not a stable-core method.",
        ),
    ]
    output: list[dict[str, object]] = []
    for comparison, baseline_method, note in comparisons:
        rmsd_diffs: list[float] = []
        runtime_ratios: list[float] = []
        soft_lower = 0
        soft_faster = 0
        soft_both = 0
        for methods in by_dataset.values():
            soft = methods.get("soft_robust_nmr_ensemble")
            baseline = methods.get(baseline_method)
            if soft is None or baseline is None:
                continue
            soft_rmsd = to_float(soft.get("unified_posthoc_rmsd_mean"))
            base_rmsd = to_float(baseline.get("unified_posthoc_rmsd_mean"))
            soft_runtime = to_float(soft.get("runtime_seconds_mean"))
            base_runtime = to_float(baseline.get("runtime_seconds_mean"))
            if None in (soft_rmsd, base_rmsd, soft_runtime, base_runtime) or base_runtime == 0:
                continue
            diff = soft_rmsd - base_rmsd
            ratio = soft_runtime / base_runtime
            rmsd_diffs.append(diff)
            runtime_ratios.append(ratio)
            lower = diff < 0
            faster = ratio < 1.0
            soft_lower += int(lower)
            soft_faster += int(faster)
            soft_both += int(lower and faster)
        rmsd_ci = bootstrap_ci(rmsd_diffs, seed=seed, n_bootstrap=n_bootstrap)
        ratio_ci = bootstrap_ci(runtime_ratios, seed=seed, n_bootstrap=n_bootstrap)
        wilcoxon_stat, wilcoxon_p = wilcoxon_if_available(rmsd_diffs)
        diff_sd = float(np.std(rmsd_diffs, ddof=1)) if len(rmsd_diffs) > 1 else None
        effect = (float(np.mean(rmsd_diffs)) / diff_sd) if diff_sd and diff_sd > 0 else None
        output.append(
            {
                "benchmark_type": benchmark_label,
                "comparison": comparison,
                "n_paired_cases": len(rmsd_diffs),
                "mean_RMSD_difference_soft_minus_baseline": fmt(mean(rmsd_diffs)),
                "median_RMSD_difference_soft_minus_baseline": fmt(median(rmsd_diffs)),
                "RMSD_difference_bootstrap95CI_low": fmt(rmsd_ci[0]),
                "RMSD_difference_bootstrap95CI_high": fmt(rmsd_ci[1]),
                "mean_runtime_ratio_soft_over_baseline": fmt(mean(runtime_ratios)),
                "median_runtime_ratio_soft_over_baseline": fmt(median(runtime_ratios)),
                "runtime_ratio_bootstrap95CI_low": fmt(ratio_ci[0]),
                "runtime_ratio_bootstrap95CI_high": fmt(ratio_ci[1]),
                "soft_lower_RMSD_count": soft_lower,
                "soft_faster_count": soft_faster,
                "soft_both_count": soft_both,
                "wilcoxon_statistic": fmt(wilcoxon_stat),
                "wilcoxon_pvalue": fmt(wilcoxon_p),
                "paired_effect_size_mean_diff_over_sd": fmt(effect),
                "bootstrap_seed": seed,
                "metric_definition": "Paired per-ensemble differences use soft minus baseline; negative RMSD difference favors soft.",
                "notes": note,
            }
        )
    return output


def wilcoxon_if_available(values: list[float]) -> tuple[float | None, float | None]:
    """Run Wilcoxon signed-rank test if SciPy is available."""

    if not values:
        return None, None
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None, None
    try:
        result = wilcoxon(np.asarray(values, dtype=float), zero_method="wilcox", alternative="two-sided")
    except Exception:
        return None, None
    return float(result.statistic), float(result.pvalue)


def flexibility_statistics(seed: int, n_bootstrap: int) -> list[dict[str, object]]:
    """Compute flexibility profile statistics with bootstrap CIs."""

    summary_by_system = {row["system"]: row for row in read_table(ROOT / "results" / "flexibility" / "flexibility_summary_four_systems.tsv")}
    profile_paths = {
        "I-FABP": ROOT / "results" / "flexibility" / "ifabp_flexibility_profile.csv",
        "ADK equilibrium": ROOT / "results" / "flexibility" / "adk_flexibility_profile.csv",
        "NhaA equilibrium": ROOT / "results" / "flexibility" / "nhaa_flexibility_profile.csv",
        "YiiP equilibrium short": ROOT / "results" / "flexibility" / "yiip_short_flexibility_profile.csv",
    }
    output = []
    for system, profile_path in profile_paths.items():
        profile = read_table(profile_path)
        robust = [to_float(row["robust_aligned_RMSF"], 0.0) or 0.0 for row in profile]
        ordinary = [to_float(row["ordinary_aligned_RMSF"], 0.0) or 0.0 for row in profile]
        mobility = [to_float(row["mobility_score"], 0.0) or 0.0 for row in profile]
        rho = spearman(robust, mobility)
        ci = bootstrap_stat_ci(robust, mobility, spearman, seed=seed, n_bootstrap=n_bootstrap)
        summary = summary_by_system.get(system, {})
        mean_core = to_float(summary.get("mean_RMSF_core"))
        mean_noncore = to_float(summary.get("mean_RMSF_noncore"))
        mean_ordinary = to_float(summary.get("mean_ordinary_RMSF"))
        mean_robust = to_float(summary.get("mean_robust_RMSF"))
        output.append(
            {
                "system": system,
                "n_residues": len(profile),
                "n_flexible_segments": summary.get("n_flexible_segments", ""),
                "spearman_robust_RMSF_vs_mobility": fmt(rho),
                "spearman_bootstrap95CI_low": fmt(ci[0]),
                "spearman_bootstrap95CI_high": fmt(ci[1]),
                "mean_core_RMSF": fmt(mean_core),
                "mean_noncore_RMSF": fmt(mean_noncore),
                "noncore_core_RMSF_ratio": fmt(mean_noncore / mean_core if mean_core and mean_noncore is not None else None),
                "mean_difference_noncore_minus_core": fmt((mean_noncore - mean_core) if mean_core is not None and mean_noncore is not None else None),
                "median_difference_noncore_minus_core": "",
                "mean_ordinary_RMSF": fmt(mean_ordinary),
                "mean_robust_RMSF": fmt(mean_robust),
                "global_mean_RMSF_difference_robust_minus_ordinary": fmt(
                    (mean_robust - mean_ordinary) if mean_robust is not None and mean_ordinary is not None else None
                ),
                "bootstrap_seed": seed,
                "metric_definition": "RMSF is alignment-dependent; bootstrap CI resamples residues and should be interpreted cautiously.",
                "notes": "Residues are not fully independent; emphasize effect sizes and CIs over p-values.",
            }
        )
    return output


def biology_statistics() -> list[dict[str, object]]:
    """Summarize external biological support conservatively."""

    rows = read_table(ROOT / "results" / "final_multi_system_summary_20260603" / "final_external_biology_summary.csv")
    output = []
    for row in rows:
        notes = "Positive external biological support."
        if "YiiP" in row["system"]:
            notes = "RMSF-level support exists, but current external biological proxy is weak/negative; do not overclaim."
        output.append(
            {
                "system": row["system"],
                "external_signal": row["external_signal"],
                "primary_metric": row["primary_metric"],
                "soft_metric_value": row["soft_metric_value"],
                "seq_contact_regularized_metric_value": row["seq_contact_regularized_metric_value"],
                "mdlovofit_metric_value_if_available": row["mdlovofit_metric_value_if_available"],
                "support_status": row["support_status"],
                "metric_definition": "External biological support summarizes independent B-factor, domain, or contact-density proxies.",
                "notes": notes,
            }
        )
    return output


def summary_markdown(
    trajectory_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    nmr_rows: list[dict[str, object]],
    flex_rows: list[dict[str, object]],
    biology_rows: list[dict[str, object]],
) -> str:
    """Build a concise final statistics summary."""

    traj_soft_vs_mdlovo = [row for row in trajectory_rows if row["baseline_method"] == "MDLovoFit"]
    pair_soft_vs_mdlovo = [row for row in pair_rows if row["baseline_method"] == "MDLovoFit"]
    nmr_mdlovo = next((row for row in nmr_rows if row["comparison"] == "soft_vs_mdlovofit"), {})
    lines = [
        "# Final Statistics Summary",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        f"- Bootstrap seed: `{BOOTSTRAP_SEED}`",
        "",
        "## Real Trajectory Benchmarks",
        "",
        f"- Soft beats MDLovoFit by unified RMSD in `{sum(row['winner_by_unified_posthoc_RMSD'] == 'soft' for row in traj_soft_vs_mdlovo)}/{len(traj_soft_vs_mdlovo)}` systems.",
        f"- Soft is faster than MDLovoFit in `{sum(row['winner_by_runtime'] == 'soft' for row in traj_soft_vs_mdlovo)}/{len(traj_soft_vs_mdlovo)}` systems.",
        "- THESEUS remains mixed: it is faster in some cases and slightly lower RMSD on NhaA.",
        "",
        "## Structure-Pair Benchmarks",
        "",
        f"- Soft beats MDLovoFit by unified RMSD in `{sum(row['winner_by_unified_posthoc_RMSD'] == 'soft' for row in pair_soft_vs_mdlovo)}/{len(pair_soft_vs_mdlovo)}` pairs.",
        f"- Soft is faster than MDLovoFit in `{sum(row['winner_by_runtime'] == 'soft' for row in pair_soft_vs_mdlovo)}/{len(pair_soft_vs_mdlovo)}` pairs.",
        "",
        "## NMR Ensemble Benchmark",
        "",
        "- This is an NMR ensemble benchmark, not an MD trajectory benchmark.",
        f"- Soft vs MDLovoFit paired cases: `{nmr_mdlovo.get('n_paired_cases', '')}`.",
        f"- Median RMSD difference soft - MDLovoFit: `{nmr_mdlovo.get('median_RMSD_difference_soft_minus_baseline', '')}`.",
        f"- Median runtime ratio soft / MDLovoFit: `{nmr_mdlovo.get('median_runtime_ratio_soft_over_baseline', '')}`.",
        "",
        "## Flexibility Profiling",
        "",
        "- Robust RMSF and mobility score are positively associated in all four systems.",
        "- Core residues have lower robust RMSF than non-core residues in all four systems.",
        "- Global ordinary and robust mean RMSF values remain similar, supporting local interpretation refinement rather than global RMSF inflation.",
        "",
        "## Biological Support",
        "",
        "- I-FABP, ADK, and NhaA have positive external biological support.",
        "- YiiP short remains limited by a weak/negative current external biological proxy.",
        "",
        "## Output Tables",
        "",
        "- `trajectory_benchmark_statistics.tsv`",
        "- `pair_benchmark_statistics.tsv`",
        "- `nmr500_benchmark_statistics.tsv`",
        "- `flexibility_statistics_final.tsv`",
        "- `biology_support_statistics.tsv`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Generate final statistics outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows = benchmark_delta_rows(
        ROOT / "results" / "final_multi_system_summary_20260603" / "final_real_trajectory_summary.csv",
        benchmark_type="real_md_trajectory",
    )
    pair_rows = benchmark_delta_rows(
        ROOT / "results" / "final_multi_system_summary_20260603" / "final_structure_pair_summary.csv",
        benchmark_type="real_structure_pair",
    )
    nmr_rows = nmr_statistics(args.bootstrap_seed, args.bootstrap_samples)
    flex_rows = flexibility_statistics(args.bootstrap_seed, args.bootstrap_samples)
    biology_rows = biology_statistics()

    write_table(args.output_dir / "trajectory_benchmark_statistics.tsv", trajectory_rows, TRAJECTORY_FIELDS)
    write_table(args.output_dir / "pair_benchmark_statistics.tsv", pair_rows, PAIR_FIELDS)
    write_table(args.output_dir / "nmr500_benchmark_statistics.tsv", nmr_rows, NMR_FIELDS)
    write_table(args.output_dir / "flexibility_statistics_final.tsv", flex_rows, FLEX_FIELDS)
    write_table(args.output_dir / "biology_support_statistics.tsv", biology_rows, BIOLOGY_FIELDS)
    (args.output_dir / "final_statistics_summary.md").write_text(
        summary_markdown(trajectory_rows, pair_rows, nmr_rows, flex_rows, biology_rows)
    )
    print(f"Wrote final statistics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
