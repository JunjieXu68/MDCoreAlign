#!/usr/bin/env python
"""Generate the nine required chronology-v3 review figures from frozen tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3"
FIGURES = OUT / "handoff_to_chatgpt/figures"
COLORS = {
    "throughput_compact_f64": "#4C78A8",
    "D_block_64_v3": "#59A14F",
    "C_correction_2_v3": "#F28E2B",
    "adaptive_C1_C2_D64_v3": "#E15759",
    "exact_sequential_f64_compact": "#6B6B6B",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def variability() -> None:
    rows = read_tsv(OUT / "02_self_variability/self_variability_summary.tsv")
    labels = [row["perturbation_class"].replace("_", " ") for row in rows]
    p95 = [float(row["p95_across_case_level_p95_rmsd_difference_A"]) for row in rows]
    maximum = [float(row["maximum_case_level_p95_rmsd_difference_A"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(rows))
    axis.bar(x - 0.18, np.maximum(p95, 1e-12), 0.36, label="P95 across case P95", color="#4C78A8")
    axis.bar(x + 0.18, np.maximum(maximum, 1e-12), 0.36, label="Maximum case P95", color="#F28E2B")
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Absolute RMSD difference (A; log scale)")
    axis.set_title("Exact-reference variability depends on perturbation class")
    axis.legend(frameon=False)
    save(figure, "fig1_exact_self_variability")


def gate_derivation() -> None:
    gate = json.loads((OUT / "03_gate_calibration/v3_gate_registry.json").read_text())
    values = [
        gate["mild_path_p95_across_case_level_p95_rmsd_difference_A"],
        gate["exact_like"]["p95_of_case_p95_rmsd_difference_A_max"],
        gate["balanced"]["p95_of_case_p95_rmsd_difference_A_max"],
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    bars = axis.bar(
        ["Mild exact path\nenvelope", "Locked exact-like\nthreshold", "Locked balanced\nthreshold"],
        values,
        color=["#76B7B2", "#59A14F", "#F28E2B"],
    )
    axis.set_ylabel("P95-of-case-P95 RMSD difference (A)")
    axis.set_title("V3 thresholds were locked from calibration only")
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4g}", ha="center", va="bottom")
    save(figure, "fig2_v3_gate_derivation")


def pareto() -> None:
    rows = read_tsv(OUT / "08_decision/pareto_comparison.tsv")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        method = row["method"]
        x = float(row["fresh_mdcath_speedup_vs_exact"])
        y = float(row["p95_of_case_p95_rmsd_difference_A"])
        axis.scatter(x, y, s=75, color=COLORS[method])
        axis.annotate(method.replace("_v3", ""), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.axhline(0.05, color="#59A14F", linestyle="--", linewidth=1, label="Exact-like RMSD gate")
    axis.axhline(0.10, color="#F28E2B", linestyle=":", linewidth=1, label="Balanced RMSD gate")
    axis.set_xlabel("Median speedup vs sequential exact")
    axis.set_ylabel("P95 across case-level P95 RMSD difference (A)")
    axis.set_title("Locked candidate quality-speed positions")
    axis.legend(frameon=False)
    save(figure, "fig3_quality_speed_pareto")


def runtime_distributions() -> None:
    rows = [row for row in read_tsv(OUT / "07_performance/fresh_long_repeats.tsv") if row["timing_kind"] == "loaded_in_process"]
    methods = ["exact_sequential_f64_compact", *[method for method in COLORS if method != "exact_sequential_f64_compact"]]
    values = [[float(row["loaded_runtime_seconds"]) for row in rows if row["method"] == method] for method in methods]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    boxes = axis.boxplot(values, labels=[method.replace("_compact_f64", "").replace("_v3", "") for method in methods], patch_artist=True, showfliers=False)
    for patch, method in zip(boxes["boxes"], methods, strict=True):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.8)
    axis.set_yscale("log")
    axis.set_ylabel("Loaded runtime (s; log scale)")
    axis.set_title("Fresh ATLAS R2/R3 full-length runtime distributions")
    axis.tick_params(axis="x", rotation=18)
    save(figure, "fig4_fresh_long_runtime_distributions")


def rmsd_tails() -> None:
    rows = read_tsv(OUT / "06_validation/fresh_mdcath_quality.tsv") + read_tsv(OUT / "06_validation/fresh_long_quality.tsv")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for method in COLORS:
        if method == "exact_sequential_f64_compact":
            continue
        values = sorted(float(row["rmsd_absolute_difference_p95_A"]) for row in rows if row["method"] == method)
        axis.plot(np.arange(1, len(values) + 1), values, label=method.replace("_v3", ""), color=COLORS[method], linewidth=1.3)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    axis.axhline(0.10, color="black", linestyle=":", linewidth=0.9)
    axis.set_yscale("symlog", linthresh=1e-5)
    axis.set_xlabel("Fresh cases sorted within method")
    axis.set_ylabel("Case-level P95 absolute RMSD difference (A)")
    axis.set_title("Fresh-panel RMSD tails")
    axis.legend(frameon=False, fontsize=8)
    save(figure, "fig5_case_level_p95_rmsd_tails")


def core_rmsf() -> None:
    rows = read_tsv(OUT / "06_validation/fresh_mdcath_quality.tsv") + read_tsv(OUT / "06_validation/fresh_long_quality.tsv")
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    for method in COLORS:
        if method == "exact_sequential_f64_compact":
            continue
        subset = [row for row in rows if row["method"] == method]
        axis.scatter(
            [float(row["core_jaccard"]) for row in subset],
            [float(row["stable_core_rmsf_spearman"]) for row in subset],
            s=10,
            alpha=0.42,
            label=method.replace("_v3", ""),
            color=COLORS[method],
        )
    axis.set_xlabel("Core Jaccard vs sequential exact")
    axis.set_ylabel("Robust RMSF Spearman vs sequential exact")
    axis.set_title("Core and RMSF agreement on fresh validation")
    axis.legend(frameon=False, fontsize=8)
    save(figure, "fig6_core_rmsf_agreement")


def adaptive_routes() -> None:
    rows = [row for row in read_tsv(OUT / "06_validation/adaptive_refinement_paths.tsv") if row["panel"] in {"fresh_mdcath", "fresh_long"}]
    panels = ("fresh_mdcath", "fresh_long")
    levels = ("C1", "C2", "D64")
    counts = np.asarray([[sum(row["panel"] == panel and row["accepted_level"] == level for row in rows) for level in levels] for panel in panels], dtype=float)
    fractions = counts / counts.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(len(panels))
    for index, level in enumerate(levels):
        axis.bar(panels, fractions[:, index], bottom=bottom, label=level, color=("#59A14F", "#F28E2B", "#E15759")[index])
        bottom += fractions[:, index]
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of trajectories")
    axis.set_title("Locked adaptive refinement routes")
    axis.legend(frameon=False, ncol=3)
    save(figure, "fig7_adaptive_refinement_paths")


def stage_decomposition() -> None:
    rows = read_tsv(OUT / "07_performance/stage_decomposition_v3.tsv")
    methods = list(COLORS)
    stages = ("weight_estimation_total_seconds", "all_frame_total_seconds", "result_construction_seconds")
    values = np.asarray([[np.median([float(row["seconds"]) for row in rows if row["method"] == method and row["stage"] == stage]) for stage in stages] for method in methods])
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    bottom = np.zeros(len(methods))
    for index, stage in enumerate(stages):
        axis.bar(np.arange(len(methods)), values[:, index], bottom=bottom, label=stage.replace("_seconds", ""), color=("#4C78A8", "#F28E2B", "#BAB0AC")[index])
        bottom += values[:, index]
    axis.set_xticks(np.arange(len(methods)), [method.replace("_compact_f64", "").replace("_v3", "") for method in methods], rotation=18)
    axis.set_ylabel("Median stage time (s)")
    axis.set_title("Corrected additive stage decomposition")
    axis.legend(frameon=False, fontsize=8)
    save(figure, "fig8_corrected_stage_decomposition")


def external_comparison() -> None:
    row = read_tsv(OUT / "07_performance/frozen_mdlovofit_direct_comparison.tsv")[0]
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    axis.axis("off")
    axis.text(0.5, 0.68, "No direct frozen MDLovoFit speedup reported", ha="center", va="center", fontsize=15, weight="bold")
    axis.text(0.5, 0.38, row["reason"], ha="center", va="center", fontsize=9, wrap=True, transform=axis.transAxes)
    axis.text(0.5, 0.12, "This is a contract-boundary limitation, not a missing positive estimate.", ha="center", fontsize=9, style="italic")
    save(figure, "fig9_frozen_mdlovofit_contract_check")


def main() -> None:
    plt.rcParams.update({"font.family": "Liberation Sans", "axes.spines.top": False, "axes.spines.right": False})
    variability()
    gate_derivation()
    pareto()
    runtime_distributions()
    rmsd_tails()
    core_rmsf()
    adaptive_routes()
    stage_decomposition()
    external_comparison()
    (FIGURES / "figure_captions.md").write_text(
        "# V3 review figure captions\n\n"
        "1. Exact self-variability is negligible numerically, bounded under mild path changes, and much larger under contextual choices.\n"
        "2. Exact-like and balanced RMSD thresholds were prospectively locked from calibration only.\n"
        "3. Fresh-panel candidate quality-speed positions under the matched compact contract.\n"
        "4. Loaded one-thread runtimes on locally available ATLAS R2/R3 trajectories.\n"
        "5. Case-level P95 RMSD-difference tails, including negative results.\n"
        "6. Core-label and stable-core RMSF agreement relative to sequential exact.\n"
        "7. C1/C2/D64 routes selected by the frozen adaptive risk rule.\n"
        "8. Corrected, non-double-counted top-level stage decomposition.\n"
        "9. Direct frozen MDLovoFit comparison is withheld because the frozen and v3 output/timing contracts are not identical.\n"
    )


if __name__ == "__main__":
    main()
