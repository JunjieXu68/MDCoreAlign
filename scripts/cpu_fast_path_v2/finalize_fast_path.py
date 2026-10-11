#!/usr/bin/env python
"""Create F6/F7 summaries and publication-style review figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_fast_path_v2"
FIG = OUT / "handoff_to_chatgpt/figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
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
        if not fields:
            handle.write("status\tnote\nnot_applicable\tno rows\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def median_iqr(values: list[float]) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
        float(np.min(array)),
        float(np.max(array)),
    )


def build_performance_summary() -> tuple[list[dict[str, object]], dict[str, float]]:
    rows = read_tsv(OUT / "06_performance/real_system_repeats.tsv")
    full = [row for row in rows if row["full_length"] == "True"]
    systems = sorted({row["case_id"] for row in full})
    methods = sorted({row["method"] for row in full})
    medians: dict[tuple[str, str], float] = {}
    summary: list[dict[str, object]] = []
    for system in systems:
        for method in methods:
            subset = [
                float(row["loaded_runtime_seconds"])
                for row in full
                if row["case_id"] == system and row["method"] == method
            ]
            if not subset:
                continue
            median, q1, q3, minimum, maximum = median_iqr(subset)
            medians[(system, method)] = median
            template = next(row for row in full if row["case_id"] == system)
            summary.append({
                "case_id": system,
                "family_id": template["family_id"],
                "n_frames": int(template["n_frames"]),
                "n_residues": int(template["n_residues"]),
                "method": method,
                "median_loaded_runtime_seconds": median,
                "q1_loaded_runtime_seconds": q1,
                "q3_loaded_runtime_seconds": q3,
                "minimum_loaded_runtime_seconds": minimum,
                "maximum_loaded_runtime_seconds": maximum,
                "repeat_count": len(subset),
            })
    speedups: dict[str, list[float]] = {method: [] for method in methods}
    for row in summary:
        exact = medians[(str(row["case_id"]), "exact_sequential_f64_compact")]
        speedup = exact / float(row["median_loaded_runtime_seconds"])
        row["speedup_vs_exact"] = speedup
        speedups[str(row["method"])].append(speedup)
    headline = {method: float(np.median(values)) for method, values in speedups.items()}
    write_tsv(OUT / "06_performance/real_system_summary.tsv", summary)
    return summary, headline


def figures(summary: list[dict[str, object]], headline: dict[str, float]) -> None:
    exact_like = read_tsv(OUT / "05_quality/exact_like_summary.tsv")
    labels = [row["configuration"] for row in exact_like]
    speed = [float(row["heldout_median_speedup"]) for row in exact_like]
    quality = [float(row["median_core_jaccard"]) for row in exact_like]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.scatter(speed, quality, s=65, color="#1f6f5f")
    for x, y, label in zip(speed, quality, labels, strict=True):
        axis.annotate(label.replace("_", " "), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.axhline(0.99, color="#b84a3a", linestyle="--", linewidth=1)
    axis.axvline(2.0, color="#b84a3a", linestyle="--", linewidth=1)
    axis.set(xlabel="Held-out speedup vs chronological exact", ylabel="Combined median core Jaccard", title="Quality-speed Pareto of locked candidates")
    save_figure(figure, "fig1_quality_speed_pareto")

    methods = sorted({str(row["method"]) for row in summary})
    runtime_data = [
        [float(row["median_loaded_runtime_seconds"]) for row in summary if row["method"] == method]
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.boxplot(runtime_data, labels=[method.replace("_", "\n") for method in methods], showfliers=False)
    axis.set_yscale("log")
    axis.set(ylabel="Full-length loaded runtime (s, log scale)", title="Runtime across 20 independent real MD families")
    axis.tick_params(axis="x", labelsize=7)
    save_figure(figure, "fig2_real_system_runtime_distributions")

    prefix = read_tsv(OUT / "06_performance/prefix_scaling.tsv")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for method in methods:
        method_rows = [row for row in prefix if row["method"] == method]
        grouped: dict[int, list[float]] = {}
        for row in method_rows:
            workload = int(row["n_frames"]) * int(row["n_residues"])
            grouped.setdefault(workload, []).append(float(row["loaded_runtime_seconds"]))
        x = np.asarray(sorted(grouped), dtype=float)
        y = np.asarray([np.median(grouped[int(value)]) for value in x])
        axes[0].plot(x, y, ".", alpha=0.65, label=method)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Residue-frames", ylabel="Runtime (s)", title="Runtime scaling")
    candidate_methods = [method for method in methods if method != "exact_sequential_f64_compact"]
    axes[1].barh([method.replace("_", " ") for method in candidate_methods], [headline[method] for method in candidate_methods], color="#d28b3c")
    axes[1].axvline(1.0, color="black", linewidth=1)
    axes[1].set(xlabel="Median full-length speedup vs exact", title="System-level speedup")
    save_figure(figure, "fig3_speedup_vs_frames_residues")

    metrics = ("median_core_jaccard", "median_weight_spearman", "median_robust_rmsf_spearman")
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True)
    for axis, metric in zip(axes, metrics, strict=True):
        axis.bar(range(len(labels)), [float(row[metric]) for row in exact_like], color="#5c7ea6")
        axis.set_xticks(range(len(labels)), [label.split("_")[0] for label in labels])
        axis.set_ylim(0.5, 1.005)
        axis.set_title(metric.replace("median_", "").replace("_", " "), fontsize=9)
    axes[0].set_ylabel("Agreement with exact")
    save_figure(figure, "fig4_weight_core_rmsf_agreement")

    stage = read_tsv(OUT / "06_performance/stage_decomposition.tsv")
    stage_names = ["weight_estimation", "all_frame_fit", "stable_core_RMSF", "result_construction"]
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    bottoms = np.zeros(len(methods))
    for stage_name in stage_names:
        stage_values = []
        for method in methods:
            values = [float(row["seconds"]) for row in stage if row["method"] == method and row["stage"] == stage_name]
            stage_values.append(float(np.median(values)) if values else 0.0)
        axis.bar(range(len(methods)), stage_values, bottom=bottoms, label=stage_name)
        bottoms += np.asarray(stage_values)
    axis.set_xticks(range(len(methods)), [method.replace("_", "\n") for method in methods], fontsize=7)
    axis.set(ylabel="Median stage time (s)", title="Stage decomposition across measured workloads")
    axis.legend(fontsize=7)
    save_figure(figure, "fig5_stage_decomposition")

    ordinary = read_tsv(OUT.parent / "cpu_optimisation_v1/03_all_frame_engine/ordinary_rmsf_benchmark.tsv")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(ordinary))
    axis.bar(x - 0.18, [float(row["ordinary_on_seconds"]) for row in ordinary], width=0.36, label="ordinary RMSF on")
    axis.bar(x + 0.18, [float(row["ordinary_off_seconds"]) for row in ordinary], width=0.36, label="compact, ordinary RMSF off")
    axis.set_xticks(x, [row["case"] for row in ordinary], rotation=25, ha="right")
    axis.set(ylabel="All-frame engine time (s)", title="Output-contract cost of ordinary RMSF")
    axis.legend(fontsize=8)
    save_figure(figure, "fig6_ordinary_rmsf_contract")

    thread = read_tsv(OUT / "06_performance/thread_scaling.tsv")
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for method in sorted({row["method"] for row in thread}):
        xs, ys = [], []
        for count in (1, 2, 4, 8):
            vals = [float(row["loaded_runtime_seconds"]) for row in thread if row["method"] == method and int(row["cpu_threads"]) == count]
            xs.append(count)
            ys.append(float(np.median(vals)))
        axis.plot(xs, ys, marker="o", label=method)
    axis.set(xlabel="CPU thread limit", ylabel="Median runtime (s)", title="Thread scaling on the locked subset", xticks=[1, 2, 4, 8])
    axis.legend(fontsize=7)
    save_figure(figure, "fig7_thread_scaling")

    tails = read_tsv(OUT / "05_quality/tail_atlas.tsv")
    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    tail_methods = sorted({row["configuration"] for row in tails})
    data = [[float(row["rmsd_absolute_difference_p95_A"]) for row in tails if row["configuration"] == method] for method in tail_methods]
    axis.boxplot(data, labels=[method.replace("_", "\n") for method in tail_methods], showfliers=True)
    axis.axhline(0.10, color="#b84a3a", linestyle="--", label="balanced P95 gate")
    axis.set_yscale("log")
    axis.set(ylabel="Per-case P95 RMSD difference (A, log)", title="Registered challenge-tail behaviour")
    axis.tick_params(axis="x", labelsize=7)
    axis.legend(fontsize=8)
    save_figure(figure, "fig8_challenge_tail_cases")


def decisions(headline: dict[str, float]) -> None:
    rejected = read_tsv(OUT / "07_decision/rejected_candidates.tsv")
    accepted = [{
        "change": "ordinary_rmsf_opt_in",
        "status": "accepted_for_release_candidate",
        "evidence": "focused schema tests and compact-contract benchmark",
        "notes": "Python API compatibility defaults remain unchanged; CLI metadata records omission explicitly.",
    }]
    write_tsv(OUT / "07_decision/accepted_changes.tsv", accepted)
    (OUT / "07_decision/default_output_decision.md").write_text(
        "# Default output decision\n\nOrdinary/full-set RMSF is opt-in for the "
        "release-candidate CLI via `--ordinary-rmsf`. The default compact "
        "analysis retains weights, core labels, transforms, RMSD, weighted "
        "RMSD, robust RMSF and flexible segments. Missing ordinary RMSF is "
        "omitted and recorded as `ordinary_rmsf_computed=false`.\n"
    )
    (OUT / "07_decision/precision_decision.md").write_text(
        "# Precision decision\n\nFloat64 throughput is retained. Float32 and "
        "mixed candidates preserved the development outputs closely but did "
        "not reach the locked 1.15x end-to-end compact speed gate.\n"
    )
    rejected_names = ", ".join(row["configuration"] for row in rejected)
    speed_text = ", ".join(f"{method}={value:.2f}x" for method, value in headline.items())
    (OUT / "07_decision/parallel_estimator_decision.md").write_text(
        "# Parallel estimator decision\n\nNo candidate passed the complete locked "
        "exact-like or balanced contract after the registered challenge tails "
        "were included. Chronological float64 exact therefore remains the "
        "reference/default implementation, and current throughput remains "
        "experimental. Candidates not promoted: " + rejected_names + ".\n\n"
        "Full-length 20-family median speedups: " + speed_text + ".\n"
    )
    (OUT / "07_decision/final_fast_path_decision.md").write_text(
        "# Final CPU fast-path decision\n\n"
        "The output-contract simplification is accepted, but no precision or "
        "parallel-weight candidate is promoted as a replacement for the "
        "chronological float64 estimator. Strong median speedups were real, "
        "yet the challenge-tail RMSD gates exposed non-negligible worst-case "
        "differences. This negative result is retained in full.\n"
    )
    (OUT / "07_decision/user_interface_impact.md").write_text(
        "# User-interface impact\n\nThe public mode list is unchanged. The only "
        "release-candidate change is that ordinary RMSF requires "
        "`--ordinary-rmsf`. No candidate-specific public modes were added.\n"
    )
    (OUT / "07_decision/manuscript_impact.md").write_text(
        "# Manuscript impact\n\nThe study supports reporting a compact output "
        "contract and substantial median CPU acceleration potential, but it "
        "does not support claiming an exact-like or balanced universal "
        "parallel replacement. Challenge-tail limitations must accompany any "
        "median speed result.\n"
    )


def placeholder_precision_tables() -> None:
    row = [{
        "status": "not_run",
        "reason": "No precision candidate passed the prospectively registered development speed gate; held-out precision data remained locked.",
    }]
    for name in ("heldout_quality.tsv", "heldout_runtime.tsv", "challenge_cases.tsv"):
        path = OUT / "03_precision_compact" / name
        if not path.exists() or path.stat().st_size == 0:
            write_tsv(path, row)


def main() -> None:
    placeholder_precision_tables()
    summary, headline = build_performance_summary()
    figures(summary, headline)
    decisions(headline)


if __name__ == "__main__":
    main()
