#!/usr/bin/env python3
"""Generate AT7 analysis, AT8 figures, and AT9 report from frozen rows."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    METHOD_MDCA,
    METHOD_MDLOVOFIT,
    OUT,
    read_tsv,
    write_text,
    write_tsv,
)

# Retain the frozen reference. This is a large-population 60% approximation,
# not the exact expected Jaccard for each finite, integer-valued matched k.
CHANCE_JACCARD = 0.60 / 1.40


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def iqr(values: list[float]) -> float:
    quartiles = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    return float(quartiles[1] - quartiles[0])


def cv(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / np.mean(values)) if len(values) > 1 else 0.0


def timing_analysis() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    panel_manifest = {f"{row['system_id']}__{row['prefix_id']}": row for row in panel}
    prefixes_by_system: dict[str, set[str]] = defaultdict(set)
    for row in panel:
        prefixes_by_system[row["system_id"]].add(row["prefix_id"])
    if (
        len(panel) != 48
        or len(panel_manifest) != 48
        or len(prefixes_by_system) != 8
        or any(len(prefixes) != 6 for prefixes in prefixes_by_system.values())
        or len({tuple(sorted(prefixes)) for prefixes in prefixes_by_system.values()}) != 1
    ):
        raise ValueError("AT1 must contain 8 systems with the same 6 unique prefixes")
    raw = read_tsv(OUT / "05_timing/raw_timing_records.tsv")
    measured = [
        row
        for row in raw
        if row["phase"] == "measured"
        and row["invalidated"] == "False"
        and row["exit_code"] == "0"
        and row["timeout"] == "False"
    ]
    if len(measured) != 480:
        raise ValueError(f"AT5 is incomplete: {len(measured)}/480 measured commands")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in measured:
        grouped[(row["setting_id"], row["method"])].append(row)
    expected_keys = {
        (sid, method) for sid in panel_manifest for method in (METHOD_MDCA, METHOD_MDLOVOFIT)
    }
    if set(grouped) != expected_keys:
        raise ValueError("AT5 setting/method coverage differs from the locked panel")
    for (sid, method), rows in grouped.items():
        if len(rows) != 5 or {int(row["repeat"]) for row in rows} != set(range(1, 6)):
            raise ValueError(f"AT5 requires exactly five distinct repeats for {sid}/{method}")
        if any(
            row["system_id"] != panel_manifest[sid]["system_id"]
            or row["prefix_id"] != panel_manifest[sid]["prefix_id"]
            or not np.isfinite(float(row["wall_seconds"]))
            or float(row["wall_seconds"]) <= 0
            for row in rows
        ):
            raise ValueError(f"AT5 contains inconsistent metadata or invalid runtime for {sid}")
    method_rows = []
    for (sid, method), rows in sorted(grouped.items()):
        walls = [float(row["wall_seconds"]) for row in rows]
        cpus = [float(row["user_cpu_seconds"]) + float(row["system_cpu_seconds"]) for row in rows]
        first = [float(row["wall_seconds"]) for row in rows if row["order_position"] == "1"]
        second = [float(row["wall_seconds"]) for row in rows if row["order_position"] == "2"]
        method_rows.append(
            {
                "setting_id": sid,
                "system_id": rows[0]["system_id"],
                "prefix_id": rows[0]["prefix_id"],
                "method": method,
                "n_repeats": len(rows),
                "median_wall_seconds": median(walls),
                "iqr_wall_seconds": iqr(walls),
                "cv_wall_time": cv(walls),
                "min_wall_seconds": min(walls),
                "max_wall_seconds": max(walls),
                "median_process_cpu_seconds": median(cpus),
                "median_when_first_seconds": median(first) if first else "",
                "median_when_second_seconds": median(second) if second else "",
            }
        )
    write_tsv(OUT / "07_analysis/repeat_variability.tsv", method_rows)
    by_key = {(row["setting_id"], row["method"]): row for row in method_rows}
    settings = []
    for sid in sorted(panel_manifest):
        item = panel_manifest[sid]
        mdca = by_key[(sid, METHOD_MDCA)]
        mdlovo = by_key[(sid, METHOD_MDLOVOFIT)]
        mdca_time = float(mdca["median_wall_seconds"])
        mdlovo_time = float(mdlovo["median_wall_seconds"])
        settings.append(
            {
                "setting_id": sid,
                "system_id": item["system_id"],
                "prefix_id": item["prefix_id"],
                "n_frames": int(item["n_frames"]),
                "n_ca": int(item["n_ca"]),
                "frames_times_ca": int(item["frames_times_ca"]),
                "mdcorealign_median_wall_seconds": mdca_time,
                "mdlovofit_median_wall_seconds": mdlovo_time,
                "absolute_wall_difference_seconds": mdlovo_time - mdca_time,
                "paired_speedup_mdlovofit_over_mdcorealign": mdlovo_time / mdca_time,
                "mdcorealign_cv": mdca["cv_wall_time"],
                "mdlovofit_cv": mdlovo["cv_wall_time"],
            }
        )
    write_tsv(OUT / "07_analysis/timing_summary_per_setting.tsv", settings)
    system_rows = []
    for system in sorted({row["system_id"] for row in settings}):
        selected = [row for row in settings if row["system_id"] == system]
        system_rows.append(
            {
                "system_id": system,
                "n_settings": len(selected),
                "median_mdcorealign_seconds": median(
                    [row["mdcorealign_median_wall_seconds"] for row in selected]
                ),
                "median_mdlovofit_seconds": median(
                    [row["mdlovofit_median_wall_seconds"] for row in selected]
                ),
                "median_speedup": median(
                    [row["paired_speedup_mdlovofit_over_mdcorealign"] for row in selected]
                ),
                "faster_setting_fraction": np.mean(
                    [row["paired_speedup_mdlovofit_over_mdcorealign"] > 1.0 for row in selected]
                ),
            }
        )
    write_tsv(OUT / "07_analysis/timing_summary_per_system.tsv", system_rows)
    prefix_rows = []
    for prefix in sorted({row["prefix_id"] for row in settings}):
        selected = [row for row in settings if row["prefix_id"] == prefix]
        prefix_speeds = [row["paired_speedup_mdlovofit_over_mdcorealign"] for row in selected]
        prefix_rows.append(
            {
                "prefix_id": prefix,
                "n_frames": selected[0]["n_frames"],
                "n_systems": len(selected),
                "median_mdcorealign_seconds": median(
                    [row["mdcorealign_median_wall_seconds"] for row in selected]
                ),
                "median_mdlovofit_seconds": median(
                    [row["mdlovofit_median_wall_seconds"] for row in selected]
                ),
                "median_speedup": median(prefix_speeds),
                "minimum_speedup": min(prefix_speeds),
                "maximum_speedup": max(prefix_speeds),
                "faster_setting_count": sum(value > 1 for value in prefix_speeds),
            }
        )
    write_tsv(OUT / "07_analysis/timing_summary_per_prefix.tsv", prefix_rows)

    speeds = [row["paired_speedup_mdlovofit_over_mdcorealign"] for row in settings]
    systems = sorted({row["system_id"] for row in settings})
    rng = np.random.default_rng(20260902)
    cluster_values = []
    for _ in range(10_000):
        selected_systems = rng.choice(systems, size=len(systems), replace=True)
        sample = []
        for system in selected_systems:
            sample.extend(
                row["paired_speedup_mdlovofit_over_mdcorealign"]
                for row in settings
                if row["system_id"] == system
            )
        cluster_values.append(float(np.median(sample)))
    setting_values = [
        float(np.median(rng.choice(speeds, size=len(speeds), replace=True))) for _ in range(10_000)
    ]
    bootstrap = [
        {
            "analysis": "system_clustered_primary",
            "n_resamples": 10_000,
            "seed": 20260902,
            "estimate": median(speeds),
            "ci_2_5": float(np.quantile(cluster_values, 0.025)),
            "ci_97_5": float(np.quantile(cluster_values, 0.975)),
        },
        {
            "analysis": "setting_level_sensitivity",
            "n_resamples": 10_000,
            "seed": 20260902,
            "estimate": median(speeds),
            "ci_2_5": float(np.quantile(setting_values, 0.025)),
            "ci_97_5": float(np.quantile(setting_values, 0.975)),
        },
    ]
    write_tsv(OUT / "07_analysis/bootstrap_results.tsv", bootstrap)
    full = [row for row in settings if row["prefix_id"] == "f10001"]
    summary = {
        "n_settings": len(settings),
        "mdcorealign_absolute_median_seconds": median(
            [row["mdcorealign_median_wall_seconds"] for row in settings]
        ),
        "mdlovofit_absolute_median_seconds": median(
            [row["mdlovofit_median_wall_seconds"] for row in settings]
        ),
        "median_speedup": median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "cluster_ci_low": bootstrap[0]["ci_2_5"],
        "cluster_ci_high": bootstrap[0]["ci_97_5"],
        "faster_fraction": float(np.mean(np.asarray(speeds) > 1.0)),
        "full_length_median_speedup": median(
            [row["paired_speedup_mdlovofit_over_mdcorealign"] for row in full]
        ),
        "mdcorealign_median_cv": median([float(row["mdcorealign_cv"]) for row in settings]),
        "mdlovofit_median_cv": median([float(row["mdlovofit_cv"]) for row in settings]),
        "invalidated_command_count": sum(row["invalidated"] == "True" for row in raw),
        "failure_or_timeout_count": sum(
            row["exit_code"] != "0" or row["timeout"] == "True" for row in raw
        ),
    }
    order_rows = []
    for method in (METHOD_MDCA, METHOD_MDLOVOFIT):
        selected = [row for row in measured if row["method"] == method]
        first = [float(row["wall_seconds"]) for row in selected if row["order_position"] == "1"]
        second = [float(row["wall_seconds"]) for row in selected if row["order_position"] == "2"]
        repeats = defaultdict(list)
        for row in selected:
            repeats[int(row["repeat"])].append(float(row["wall_seconds"]))
        order_rows.append(
            {
                "method": method,
                "n_first": len(first),
                "n_second": len(second),
                "median_first_seconds": median(first),
                "median_second_seconds": median(second),
                "second_over_first_ratio": median(second) / median(first),
                "median_within_setting_second_over_first_ratio": median(
                    [
                        float(row["median_when_second_seconds"])
                        / float(row["median_when_first_seconds"])
                        for row in method_rows
                        if row["method"] == method
                    ]
                ),
                "repeat_1_median_seconds": median(repeats[1]),
                "repeats_2_to_5_median_seconds": median(
                    [value for key in (2, 3, 4, 5) for value in repeats[key]]
                ),
            }
        )
    write_tsv(OUT / "07_analysis/order_effects.tsv", order_rows)
    summary["mdcorealign_second_over_first_ratio"] = next(
        row["second_over_first_ratio"] for row in order_rows if row["method"] == METHOD_MDCA
    )
    summary["mdlovofit_second_over_first_ratio"] = next(
        row["second_over_first_ratio"] for row in order_rows if row["method"] == METHOD_MDLOVOFIT
    )
    for method, label in ((METHOD_MDCA, "mdcorealign"), (METHOD_MDLOVOFIT, "mdlovofit")):
        summary[f"{label}_median_within_setting_second_over_first_ratio"] = next(
            row["median_within_setting_second_over_first_ratio"]
            for row in order_rows
            if row["method"] == method
        )
    wall = np.asarray([float(row["wall_seconds"]) for row in measured])
    process_cpu = np.asarray(
        [float(row["user_cpu_seconds"]) + float(row["system_cpu_seconds"]) for row in measured]
    )
    summary["wall_process_cpu_spearman"] = float(spearmanr(wall, process_cpu).statistic)
    write_tsv(OUT / "07_analysis/panel_speedup_summary.tsv", [summary])
    return settings, summary


def core_analysis(settings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = read_tsv(OUT / "06_core_extraction/core_agreement_per_setting.tsv")
    expected_ids = {row["setting_id"] for row in settings}
    if len(core) != 48 or {row["setting_id"] for row in core} != expected_ids:
        raise ValueError("AT6 must contain exactly one core-agreement row per locked setting")
    mapping = read_tsv(OUT / "06_core_extraction/core_mapping_audit.tsv")
    if (
        len(mapping) != 48
        or {row["setting_id"] for row in mapping} != expected_ids
        or any(
            row["status"] != "pass"
            or any(
                row[field] != "True"
                for field in ("atom_order_match", "model_count_match", "core_count_valid")
            )
            for row in mapping
        )
    ):
        raise ValueError("AT6 requires a passing mapping/count audit for every locked setting")
    core.sort(key=lambda row: (row["system_id"], int(row["n_frames"])))
    values = [float(row["matched_k_jaccard_across_extractions_median"]) for row in core]
    dice = [float(row["matched_k_dice_across_extractions_median"]) for row in core]
    full = [
        float(row["matched_k_jaccard_across_extractions_median"])
        for row in core
        if row["prefix_id"] == "f10001"
    ]
    summary = {
        "n_settings": len(core),
        "median_matched_k_jaccard": median(values),
        "p5_matched_k_jaccard": float(np.quantile(values, 0.05)),
        "minimum_matched_k_jaccard": min(values),
        "full_length_median_matched_k_jaccard": median(full),
        "median_matched_k_dice": median(dice),
        "exact_match_count": sum(row["matched_k_exact_match"] == "True" for row in core),
        "above_chance_fraction": float(np.mean(np.asarray(values) > CHANCE_JACCARD)),
        "chance_reference": CHANCE_JACCARD,
        "chance_reference_interpretation": "preregistered approximate random 60% reference",
        "median_native_jaccard": median([float(row["native_jaccard"]) for row in core]),
        "minimum_native_jaccard": min(float(row["native_jaccard"]) for row in core),
        "median_native_dice": median([float(row["native_dice"]) for row in core]),
    }
    write_tsv(OUT / "07_analysis/core_agreement_summary.tsv", [summary])
    setting_map = {row["setting_id"]: row for row in settings}
    association_rows = []
    joined = [(row, setting_map[row["setting_id"]]) for row in core]
    for predictor, source in (
        ("prefix_frames", "n_frames"),
        ("n_ca", "n_ca"),
        ("frames_times_ca", "frames_times_ca"),
        ("speedup", "paired_speedup_mdlovofit_over_mdcorealign"),
    ):
        x = [float(setting[source]) for _, setting in joined]
        y = [float(row["matched_k_jaccard_across_extractions_median"]) for row, _ in joined]
        result = spearmanr(x, y)
        association_rows.append(
            {
                "predictor": predictor,
                "outcome": "matched_k_core_jaccard",
                "spearman_rho": result.statistic,
                "p_value_descriptive": result.pvalue,
                "interpretation": (
                    "secondary descriptive association; 48 settings nested in 8 systems; "
                    "p value is unadjusted for clustering and is not inferential evidence"
                ),
            }
        )
    write_tsv(OUT / "07_analysis/speed_agreement_associations.tsv", association_rows)
    return core, summary


def make_figures(settings: list[dict[str, Any]], core: list[dict[str, Any]]) -> None:
    destination = OUT / "08_figures"
    data = destination / "figure_data"
    data.mkdir(parents=True, exist_ok=True)
    write_tsv(data / "figure1_runtime.tsv", settings)
    write_tsv(data / "figure2_speedup_workload.tsv", settings)
    write_tsv(data / "figure3_core_agreement.tsv", core)
    colors = {METHOD_MDCA: "#16697A", METHOD_MDLOVOFIT: "#C84C09"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(settings))
    ax.scatter(
        x,
        [row["mdcorealign_median_wall_seconds"] for row in settings],
        s=18,
        label="MDCoreAlign throughput CPU",
        color=colors[METHOD_MDCA],
    )
    ax.scatter(
        x,
        [row["mdlovofit_median_wall_seconds"] for row in settings],
        s=18,
        label="stock MDLovoFit",
        color=colors[METHOD_MDLOVOFIT],
    )
    ax.set_yscale("log")
    ax.set_xlabel("ATLAS setting (system and prefix)")
    ax.set_ylabel("Median wall time (s, log scale)")
    ax.set_title("Strict one-logical-CPU no-materialisation runtime")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"figure1_single_thread_runtime.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for system in sorted({row["system_id"] for row in settings}):
        rows = [row for row in settings if row["system_id"] == system]
        ax.plot(
            [row["frames_times_ca"] for row in rows],
            [row["paired_speedup_mdlovofit_over_mdcorealign"] for row in rows],
            marker="o",
            linewidth=1.2,
            markersize=3.5,
            label=system,
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frames x C-alpha residues")
    ax.set_ylabel("Wall-time ratio: MDLovoFit / MDCoreAlign")
    ax.set_title("Speedup versus common-input workload")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"figure2_speedup_workload.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [0.8, 1.4]})
    jaccard = np.asarray(
        [float(row["matched_k_jaccard_across_extractions_median"]) for row in core]
    )
    axes[0].boxplot(jaccard, widths=0.45, showmeans=True)
    axes[0].axhline(CHANCE_JACCARD, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xticks([1], ["48 settings"])
    axes[0].set_ylabel("Matched-k core Jaccard")
    axes[0].set_ylim(0, 1.02)
    for system in sorted({row["system_id"] for row in core}):
        rows = [row for row in core if row["system_id"] == system]
        axes[1].plot(
            [int(row["n_frames"]) for row in rows],
            [float(row["matched_k_jaccard_across_extractions_median"]) for row in rows],
            marker="o",
            linewidth=1.1,
            markersize=3,
            label=system,
        )
    axes[1].axhline(
        CHANCE_JACCARD, color="#777777", linestyle="--", linewidth=1,
        label="approx. random 60% reference"
    )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Matched-k core Jaccard")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Core agreement at matched k (requested fraction 0.60)")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"figure3_core_agreement.{suffix}", dpi=300)
    plt.close(fig)


def write_reports(timing: dict[str, Any], core: dict[str, Any]) -> None:
    self_consistency = read_tsv(OUT / "06_core_extraction/mdlovofit_self_consistency.tsv")
    expected_full_ids = {
        f"{row['system_id']}__{row['prefix_id']}"
        for row in read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
        if row["prefix_id"] == "f10001"
    }
    if (
        len(self_consistency) != 8
        or len(expected_full_ids) != 8
        or {row["setting_id"] for row in self_consistency} != expected_full_ids
        or any(row["exact_match"] not in ("True", "False") for row in self_consistency)
    ):
        raise ValueError("AT6 requires a duplicate-extraction audit for all 8 full-length systems")
    mdlovo_self_consistent = all(row["exact_match"] == "True" for row in self_consistency)
    report = f"""# Final strict single-thread ATLAS CPU benchmark report

## Contract

The locked panel contained 8 ATLAS R1 systems and 6 prefixes per system (48
settings). Both programs consumed the same pre-generated, per-frame-centred,
CA-only multi-model PDB. The timed no-materialisation boundary began with PDB
parsing and ended after each program's normal stable-reference estimation and
all-frame fit/RMSD workflow. Coordinate writing, raw-XTC decode, RMSF, W5, GPU,
and optional diagnostics were excluded. Both commands used one logical CPU,
one active numerical compute thread, the same affinity, warm cache, one
unmeasured warm-up, five measured repeats, and a counterbalanced order.
PBC/nojump preprocessing status is unknown: the available input metadata does
not establish it, and per-frame centring does not demonstrate PBC repair.

## Runtime

MDCoreAlign's median setting-level absolute runtime was
{timing["mdcorealign_absolute_median_seconds"]:.6g} s versus
{timing["mdlovofit_absolute_median_seconds"]:.6g} s for stock MDLovoFit. The
median paired wall-time ratio was {timing["median_speedup"]:.6g} (system-clustered
95% CI {timing["cluster_ci_low"]:.6g}–{timing["cluster_ci_high"]:.6g}); MDCoreAlign
was faster in {timing["faster_fraction"]:.1%} of settings. The full-length-only
median ratio was {timing["full_length_median_speedup"]:.6g}. Setting ratios ranged
from {timing["minimum_speedup"]:.6g} to {timing["maximum_speedup"]:.6g}.
Absolute runtime and speedup summaries for each prefix, including full length,
are in `07_analysis/timing_summary_per_prefix.tsv`.

Median repeat CVs were {timing["mdcorealign_median_cv"]:.4g} for MDCoreAlign and
{timing["mdlovofit_median_cv"]:.4g} for MDLovoFit. Median within-setting ratios
of second-position to first-position runtime were
{timing["mdcorealign_median_within_setting_second_over_first_ratio"]:.4g} and
{timing["mdlovofit_median_within_setting_second_over_first_ratio"]:.4g}, respectively.
These are descriptive order checks; pooled ratios are also retained in the table.
Wall and process
CPU times had Spearman rho {timing["wall_process_cpu_spearman"]:.4g}. The run
retained {timing["invalidated_command_count"]} invalidated command records and
{timing["failure_or_timeout_count"]} failures/timeouts; none were silently removed.

## Core agreement

Core extraction was untimed. A persistent stock export established that this
MDLovoFit build closes each frame with END, rather than ENDMDL; the benchmark
parser was corrected to validate those frames. Independent exports were
prefetched with one numerical thread per process on CPUs 17–24 and retained as
validated checkpoints. This scheduling did not rerun or modify the CPU-17 AT5
timing. Format, command, hash and scheduling evidence is retained under
`06_core_extraction/`.

At matched k, the median Jaccard was {core["median_matched_k_jaccard"]:.6g},
P5 was {core["p5_matched_k_jaccard"]:.6g}, and the minimum was
{core["minimum_matched_k_jaccard"]:.6g}. Median Dice was
{core["median_matched_k_dice"]:.6g}. Exact core identity occurred in
{core["exact_match_count"]}/48 settings, and {core["above_chance_fraction"]:.1%}
exceeded the preregistered approximate random-60% reference ({CHANCE_JACCARD:.6f}).
This reference is not a finite-sample significance threshold; actual matched k
uses MDLovoFit's integer core size. Native-label median Jaccard was
{core["median_native_jaccard"]:.6g} (minimum {core["minimum_native_jaccard"]:.6g}),
and native-label median Dice was {core["median_native_dice"]:.6g}.
These are method-agreement statistics, not accuracy or biological ground truth.
Core overlap alone does not establish biological equivalence.
The duplicate full-length MDLovoFit extraction audit was
{"self-consistent in all 8 systems" if mdlovo_self_consistent else "not self-consistent in at least one system; extraction ranges are reported"}.
Matched-k exact identity and native-label summaries refer to the primary extraction;
matched-k Jaccard and Dice use each setting's median across retained extractions.
Per-setting native/matched-k values and chain-aware disagreement segments are in
`06_core_extraction/core_agreement_per_setting.tsv`.

## Evidence grade

Historical wall-clock observations used differing or unresolved timing/output
and thread contracts and are superseded for formal CPU timing claims by the present strict
single-thread ATLAS benchmark. They remain archived and are not described as
invalid or fraudulent.
"""
    write_text(OUT / "09_report/final_atlas_single_thread_report.md", report)
    write_text(
        OUT / "09_report/methods_text.md",
        """## Strict CPU timing and core-agreement methods

We used the prospectively locked 48-setting ATLAS panel and an identical centred
CA-only multi-model PDB for both methods. One unmeasured warm-up and five measured
runs were executed under a counterbalanced order, warm-cache policy, one logical
CPU affinity, and verified one-thread numerical pools (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `RAYON_NUM_THREADS` all set to
1; dynamic OpenMP/MKL disabled). The primary per-setting
estimand was median wall time; uncertainty used a system-clustered bootstrap that
retained all six prefixes. Stable cores were extracted separately without timing.
At fraction 0.60, MDCoreAlign weights were deterministically ranked to the actual
MDLovoFit core cardinality before Jaccard and Dice calculation; ties used ascending
input index. Native production labels were also compared separately. The frozen
0.428571 random-60% reference is approximate, not a finite-sample hypothesis test.
PBC/nojump preprocessing status is unknown from the available input metadata;
per-frame centring is not evidence that periodic-boundary jumps were repaired.
""",
    )
    write_text(
        OUT / "09_report/results_text.md",
        f"""## Strict CPU timing and core agreement

Under the verified one-logical-CPU, single-compute-thread, matched
no-materialisation contract, the median setting-level wall-time ratio
(stock MDLovoFit / production MDCoreAlign throughput) was
{timing["median_speedup"]:.3g} across 48 ATLAS settings (system-clustered 95% CI
{timing["cluster_ci_low"]:.3g}–{timing["cluster_ci_high"]:.3g}).
At matched k using MDLovoFit's actual core size at requested fraction 0.60,
the median Jaccard index was {core["median_matched_k_jaccard"]:.3g} and exact
core identity occurred in {core["exact_match_count"]}/48 primary extractions.
These overlap measurements quantify method agreement and do not establish
biological equivalence or core accuracy.
""",
    )
    write_text(
        OUT / "09_report/limitations_text.md",
        """## Limitations

This benchmark measures a matched software-level no-materialisation workflow,
not raw trajectory decode, coordinate writing, GPU performance, or a pure
mathematical kernel. The eight systems are one ATLAS R1 panel rather than an
independent biological population. A fixed 60% core supports method comparison
but is not universally optimal. Core overlap measures agreement; neither method
defines biological ground truth. Runtime findings are specific to the recorded
host, executable, affinity, thread environment, and common PDB representation.
PBC/nojump preprocessing status is unknown because the available input metadata
does not establish it. Centring the coordinates does not demonstrate PBC repair.
""",
    )
    numbers = [{"metric": key, "value": value} for key, value in {**timing, **core}.items()]
    write_tsv(OUT / "09_report/manuscript_numbers.tsv", numbers)
    write_tsv(
        OUT / "09_report/claim_evidence_matrix.tsv",
        [
            {
                "claim": "strict matched no-materialisation CPU runtime comparison",
                "supported": True,
                "evidence": "07_analysis/panel_speedup_summary.tsv",
                "wording_caution": "Do not call the result universal or pure-kernel speedup.",
            },
            {
                "claim": "measured core overlap at matched cardinality",
                "supported": True,
                "evidence": "07_analysis/core_agreement_summary.tsv",
                "wording_caution": "Agreement is not accuracy or biological equivalence.",
            },
            {
                "claim": "biologically equivalent or equally accurate stable cores",
                "supported": False,
                "evidence": "not established by method-overlap measurements",
                "wording_caution": "Neither method is biological ground truth.",
            },
            {
                "claim": "universal fastest",
                "supported": False,
                "evidence": "none",
                "wording_caution": "Prohibited.",
            },
        ],
    )


def main() -> None:
    settings, timing = timing_analysis()
    core_rows, core = core_analysis(settings)
    make_figures(settings, core_rows)
    write_reports(timing, core)
    print(json.dumps({"timing": timing, "core": core}, sort_keys=True))


if __name__ == "__main__":
    main()
