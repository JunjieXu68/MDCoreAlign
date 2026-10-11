#!/usr/bin/env python3
"""Finalize the locked, untimed MA3-MA7 method-agreement analyses."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.no_materialisation_benchmark_v1.common import OUT, ROOT, write_json, write_text, write_tsv


MA = OUT / "method_agreement_v2"
REPORTING = MA / "08_reporting"
MANIFESTS = MA / "00_contracts/panel_manifests"
FIGURES = REPORTING / "figures"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def describe(values: list[float], prefix: str) -> dict[str, object]:
    array = finite(values)
    if array.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_median": float("nan"),
            f"{prefix}_Q1": float("nan"),
            f"{prefix}_Q3": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": int(array.size),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_Q1": float(np.quantile(array, 0.25)),
        f"{prefix}_Q3": float(np.quantile(array, 0.75)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def group_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    output: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[key] for key in keys)].append(row)
    return output


def summarize_core(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for (method,), group in sorted(group_rows(rows, ("MDCoreAlign_method",)).items()):
        output.append(
            {
                "MDCoreAlign_method": method,
                "n_cases": len(group),
                **describe([numeric(row, "Jaccard") for row in group], "Jaccard"),
                **describe([numeric(row, "Dice") for row in group], "Dice"),
                **describe([numeric(row, "chainwise_Jaccard_mean") for row in group], "chainwise_Jaccard"),
                **describe([numeric(row, "mean_boundary_distance_residues") for row in group], "boundary_distance_residues"),
                "fraction_above_equal_cardinality_chance": float(
                    np.mean([str(row["above_equal_cardinality_chance"]).lower() == "true" for row in group])
                ),
            }
        )
    return output


def summarize_continuous(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for (method,), group in sorted(group_rows(rows, ("MDCoreAlign_method",)).items()):
        output.append(
            {
                "MDCoreAlign_method": method,
                "n_cases": len(group),
                **describe([numeric(row, "selection_frequency_Spearman") for row in group], "selection_frequency_Spearman"),
                **describe([numeric(row, "selection_frequency_Kendall") for row in group], "selection_frequency_Kendall"),
                **describe([numeric(row, "selection_frequency_RBO_p0.9") for row in group], "selection_frequency_RBO_p0.9"),
                **describe([numeric(row, "internal_evidence_Spearman") for row in group], "internal_evidence_Spearman"),
            }
        )
    return output


def bootstrap_median_ci(values: list[float], seed: int = 20260820) -> tuple[float, float, float]:
    array = finite(values)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(array, size=(10000, array.size), replace=True), axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(np.median(array)), float(low), float(high)


def join_fraction_rows(
    core: list[dict[str, str]],
    continuous: list[dict[str, str]],
    geometry: list[dict[str, str]],
) -> list[dict[str, object]]:
    key_names = ("panel", "case_id", "prefix_label", "fraction", "MDCoreAlign_method")
    continuous_index = {tuple(row[key] for key in key_names): row for row in continuous}
    geometry_index = {
        tuple(row[key] for key in key_names): row
        for row in geometry
        if row["comparison_layer"] == "C_native_method_alignment"
    }
    output = []
    for row in core:
        key = tuple(row[name] for name in key_names)
        joined: dict[str, object] = dict(row)
        for source in (continuous_index.get(key, {}), geometry_index.get(key, {})):
            for name, value in source.items():
                if name not in joined:
                    joined[name] = value
        output.append(joined)
    return output


def self_variability() -> dict[str, dict[str, object]]:
    mapping = {"I-FABP": "ifabp", "ADK": "adk", "NhaA": "nhaa", "YiiP short": "yiip_short"}
    output: dict[str, dict[str, object]] = defaultdict(dict)
    sources = [
        (
            ROOT / "results/md_trajectory_focus/temporal_block_stability/temporal_block_stability.tsv",
            "jaccard_core_vs_full",
            "temporal_block_core_Jaccard",
        ),
        (
            ROOT / "results/md_trajectory_focus/reference_sensitivity/reference_sensitivity.tsv",
            "jaccard_core_vs_default_reference",
            "reference_core_Jaccard",
        ),
        (
            ROOT / "results/md_trajectory_focus/stride_sensitivity/stride_sensitivity.tsv",
            "jaccard_core_vs_stride1",
            "stride_core_Jaccard",
        ),
    ]
    for path, source_key, output_key in sources:
        if not path.is_file():
            continue
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in read_tsv(path):
            if row["system"] in mapping:
                value = numeric(row, source_key)
                if math.isfinite(value):
                    grouped[mapping[row["system"]]].append(value)
        for system, values in grouped.items():
            output[system][f"{output_key}_min"] = min(values)
            output[system][f"{output_key}_max"] = max(values)
    return output


def adjudicate(
    core: list[dict[str, str]],
    heldout: list[dict[str, str]],
) -> list[dict[str, object]]:
    core_index = {
        (row["case_id"], row["fraction"], row["MDCoreAlign_method"]): row
        for row in core
    }
    heldout_groups = group_rows(heldout, ("case_id", "fraction"))
    variability = self_variability()
    output = []
    for (case_id, fraction), rows in sorted(heldout_groups.items()):
        if abs(float(fraction) - 0.60) > 1e-9:
            continue
        by_method = {row["method"]: row for row in rows}
        mdl = by_method.get("MDLovoFit")
        if mdl is None:
            continue
        for method in ("exact", "throughput"):
            mdca = by_method.get(f"MDCoreAlign_{method}")
            core_row = core_index.get((case_id, fraction, method))
            if mdca is None or core_row is None:
                continue
            jaccard = numeric(core_row, "Jaccard")
            mdl_var = numeric(mdl, "heldout_pairwise_distance_variance_A2")
            mdca_var = numeric(mdca, "heldout_pairwise_distance_variance_A2")
            mdl_contact = numeric(mdl, "heldout_contact_persistence")
            mdca_contact = numeric(mdca, "heldout_contact_persistence")
            if jaccard >= 0.80:
                category = "both_methods_stable_broad_agreement"
            elif math.isfinite(mdl_var) and math.isfinite(mdca_var) and mdca_var < 0.9 * mdl_var and mdca_contact >= mdl_contact - 0.02:
                category = "MDCoreAlign_more_reproducible_on_heldout_geometry"
            elif math.isfinite(mdl_var) and math.isfinite(mdca_var) and mdl_var < 0.9 * mdca_var and mdl_contact >= mdca_contact - 0.02:
                category = "MDLovoFit_more_reproducible_on_heldout_geometry"
            elif min(mdl_contact, mdca_contact) < 0.50:
                category = "both_unstable_or_undersampled"
            else:
                category = "both_stable_but_different_reference_subsets"
            output.append(
                {
                    "panel": core_row["panel"],
                    "case_id": case_id,
                    "MDCoreAlign_method": method,
                    "core_Jaccard": jaccard,
                    "MDLovoFit_pairwise_distance_variance_A2": mdl_var,
                    "MDCoreAlign_pairwise_distance_variance_A2": mdca_var,
                    "MDLovoFit_contact_persistence": mdl_contact,
                    "MDCoreAlign_contact_persistence": mdca_contact,
                    "adjudication": category,
                    "ground_truth_caveat": "Neither method is treated as biological ground truth.",
                    **variability.get(case_id, {}),
                }
            )
    return output


def adaptive_expansion(core_60: list[dict[str, str]]) -> dict[str, object]:
    primary = [
        row for row in core_60
        if row["panel"] in {"mdCATH_primary_512", "mdCATH_adaptive_expansion_512"}
        and row["MDCoreAlign_method"] == "exact"
    ]
    median, low, high = bootstrap_median_ci([numeric(row, "Jaccard") for row in primary])
    manifest_path = MANIFESTS / ("mdcath_adaptive_1024.tsv" if len(primary) > 512 else "mdcath_primary_512.tsv")
    manifest = {row["domain_id"]: row for row in read_tsv(manifest_path)}
    stratum_values: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        metadata = manifest[row["case_id"]]
        labels = [
            f"CATH:{metadata['cath_class']}",
            f"residues:{metadata['residue_count_bin']}",
            f"frames:{metadata['frame_count_bin']}",
        ]
        for label in labels:
            stratum_values[label].append(numeric(row, "Jaccard"))
    stratum_rows = [
        {"stratum": label, "n": len(values), "median_Jaccard": float(np.median(values)), "above_chance": float(np.median(values)) > 3.0 / 7.0}
        for label, values in sorted(stratum_values.items())
    ]
    contradictory = len({bool(row["above_chance"]) for row in stratum_rows}) > 1
    divergent_count = sum(numeric(row, "Jaccard") < 0.60 for row in primary)
    reasons = []
    if high - low > 0.05:
        reasons.append("bootstrap_CI_width_gt_0.05")
    if contradictory:
        reasons.append("contradictory_major_strata")
    if divergent_count < 20:
        reasons.append("fewer_than_20_divergent_cases")
    initial_triggered = bool(reasons) if len(primary) == 512 else True
    return {
        "n_cases": len(primary),
        "median_core_Jaccard": median,
        "bootstrap_95CI_low": low,
        "bootstrap_95CI_high": high,
        "bootstrap_95CI_width": high - low,
        "divergent_case_count_Jaccard_lt_0.60": divergent_count,
        "contradictory_major_strata": contradictory,
        "mapping_uncertainty": False,
        "adaptive_expansion_triggered": initial_triggered,
        "expansion_to_1024_completed": len(primary) == 1024,
        "further_full_5397_expansion_requires_explicit_human_approval": len(primary) == 1024,
        "trigger_reasons": reasons if len(primary) == 512 else ["initial_512_had_fewer_than_20_divergent_cases"],
        "strata": stratum_rows,
    }


def _plot_box(data: list[list[float]], labels: list[str], ylabel: str, path: Path, *, chance: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.boxplot(data, labels=labels, showfliers=False)
    if chance is not None:
        axis.axhline(chance, color="#8c3b2f", linestyle="--", linewidth=1.2, label="equal-cardinality chance")
        axis.legend(frameon=False)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(figure)


def make_figures(
    core_60: list[dict[str, str]],
    continuous_60: list[dict[str, str]],
    fraction_rows: list[dict[str, object]],
    geometry: list[dict[str, str]],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    methods = ("exact", "throughput")
    _plot_box(
        [[numeric(row, "Jaccard") for row in core_60 if row["MDCoreAlign_method"] == method] for method in methods],
        list(methods),
        "Core Jaccard at matched 60%",
        FIGURES / "core_jaccard_dice_distributions",
        chance=3.0 / 7.0,
    )
    _plot_box(
        [[numeric(row, "selection_frequency_Spearman") for row in continuous_60 if row["MDCoreAlign_method"] == method] for method in methods],
        list(methods),
        "Per-case Spearman: stable weight vs selection frequency",
        FIGURES / "continuous_stability_score_agreement",
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for method, color in (("exact", "#1b6ca8"), ("throughput", "#d2691e")):
        xs, ys = [], []
        for fraction in (0.4, 0.5, 0.6, 0.7, 0.8):
            values = [numeric(row, "Jaccard") for row in fraction_rows if row["MDCoreAlign_method"] == method and abs(numeric(row, "fraction") - fraction) < 1e-9]
            xs.append(fraction)
            ys.append(float(np.median(finite(values))))
        axis.plot(xs, ys, marker="o", label=method, color=color)
    axis.set(xlabel="Matched core fraction", ylabel="Median core Jaccard")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "fraction_sensitivity_curves.pdf")
    figure.savefig(FIGURES / "fraction_sensitivity_curves.png", dpi=240)
    plt.close(figure)

    native = [row for row in geometry if row["comparison_layer"] == "C_native_method_alignment" and abs(numeric(row, "fraction") - 0.60) < 1e-9]
    figure, axis = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    axis.scatter(
        [numeric(row, "median_rotation_geodesic_degrees") for row in native],
        [numeric(row, "median_aligned_coordinate_difference_A") for row in native],
        s=12,
        alpha=0.55,
    )
    axis.set(xlabel="Median rotation disagreement (degrees)", ylabel="Median aligned-coordinate difference (A)")
    axis.grid(alpha=0.22)
    figure.savefig(FIGURES / "aligned_coordinate_rotation_agreement.pdf")
    figure.savefig(FIGURES / "aligned_coordinate_rotation_agreement.png", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].scatter([numeric(row, "RMSD_MAE_A") for row in native], [numeric(row, "RMSD_Spearman") for row in native], s=12, alpha=0.55)
    axes[0].set(xlabel="RMSD MAE (A)", ylabel="RMSD Spearman")
    axes[1].scatter([numeric(row, "RMSF_MAE_A") for row in native], [numeric(row, "RMSF_Spearman") for row in native], s=12, alpha=0.55)
    axes[1].set(xlabel="RMSF MAE (A)", ylabel="RMSF Spearman")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.savefig(FIGURES / "rmsd_rmsf_agreement.pdf")
    figure.savefig(FIGURES / "rmsd_rmsf_agreement.png", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    values = [numeric(row, "Jaccard") for row in core_60 if row["MDCoreAlign_method"] == "exact"]
    axis.hist(values, bins=30, color="#39796b", alpha=0.85)
    axis.axvline(3.0 / 7.0, color="#8c3b2f", linestyle="--", label="equal-cardinality chance")
    axis.set(xlabel="Core Jaccard", ylabel="Cases")
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "agreement_vs_equal_cardinality_chance.pdf")
    figure.savefig(FIGURES / "agreement_vs_equal_cardinality_chance.png", dpi=240)
    plt.close(figure)

    exact = sorted([row for row in core_60 if row["MDCoreAlign_method"] == "exact"], key=lambda row: numeric(row, "Jaccard"))
    selected = exact[:15] + exact[-5:]
    figure, axis = plt.subplots(figsize=(10.0, 5.2), constrained_layout=True)
    axis.bar(np.arange(len(selected)), [numeric(row, "Jaccard") for row in selected], color=["#b7513c"] * 15 + ["#39796b"] * 5)
    axis.axhline(3.0 / 7.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(selected)), [row["case_id"] for row in selected], rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("Core Jaccard")
    figure.savefig(FIGURES / "representative_disagreement_cases.pdf")
    figure.savefig(FIGURES / "representative_disagreement_cases.png", dpi=240)
    plt.close(figure)


def main() -> None:
    raw_status = REPORTING / "ma3_ma7_raw_status.json"
    if not raw_status.is_file() or json.loads(raw_status.read_text()).get("cases") != 530:
        raise RuntimeError("MA3-MA7 raw run is incomplete; finalization requires exactly 530 completed cases")
    core = read_tsv(REPORTING / "combined_core_agreement.tsv")
    continuous = read_tsv(REPORTING / "combined_continuous_agreement.tsv")
    geometry = read_tsv(REPORTING / "combined_geometry_agreement.tsv")
    heldout = read_tsv(REPORTING / "combined_heldout_geometry.tsv")
    expansion_status = REPORTING / "ma_adaptive_expansion_raw_status.json"
    if expansion_status.is_file() and json.loads(expansion_status.read_text()).get("status") == "pass":
        core.extend(read_tsv(REPORTING / "adaptive_expansion_core_agreement.tsv"))
        continuous.extend(read_tsv(REPORTING / "adaptive_expansion_continuous_agreement.tsv"))
        heldout.extend(read_tsv(REPORTING / "adaptive_expansion_heldout_geometry.tsv"))
    core_60 = [row for row in core if abs(numeric(row, "fraction") - 0.60) < 1e-9]
    continuous_60 = [row for row in continuous if abs(numeric(row, "fraction") - 0.60) < 1e-9]

    historical_core = [row for row in core_60 if row["panel"] == "historical_four_systems"]
    historical_geometry = [row for row in geometry if row["panel"] == "historical_four_systems" and abs(numeric(row, "fraction") - 0.60) < 1e-9]
    write_tsv(MA / "02_legacy_four_systems/four_system_core_agreement.tsv", historical_core)
    write_tsv(MA / "02_legacy_four_systems/four_system_alignment_agreement.tsv", historical_geometry)
    write_text(
        MA / "02_legacy_four_systems/four_system_interpretation.md",
        "# Historical Four-System Interpretation\n\nThis sanity panel reconnects the agreement analysis to I-FABP, ADK, NhaA and YiiP short. It is not the primary evidence. Similar native RMSD values, stable-core overlap and MDCoreAlign exact/throughput agreement are reported separately; none establishes biological equivalence.\n",
    )

    atlas_panels = {"ATLAS_full_8", "ATLAS_short_tails_6"}
    atlas_core = [row for row in core_60 if row["panel"] in atlas_panels]
    atlas_cont = [row for row in continuous_60 if row["panel"] in atlas_panels]
    mdcath_panels = {"mdCATH_primary_512", "mdCATH_adaptive_expansion_512"}
    mdcath_core = [row for row in core_60 if row["panel"] in mdcath_panels]
    mdcath_cont = [row for row in continuous_60 if row["panel"] in mdcath_panels]
    for directory, prefix, selected_core, selected_cont in (
        (MA / "03_atlas_60pct", "atlas", atlas_core, atlas_cont),
        (MA / "04_mdcath_60pct", "mdcath", mdcath_core, mdcath_cont),
    ):
        write_tsv(directory / f"{prefix}_core_agreement_60pct.tsv", selected_core)
        write_tsv(directory / f"{prefix}_continuous_stability_agreement_60pct.tsv", selected_cont)
        write_tsv(
            directory / f"{prefix}_disagreement_regions.tsv",
            [
                {key: row[key] for key in ("panel", "case_id", "system", "prefix_label", "MDCoreAlign_method", "Jaccard", "mean_boundary_distance_residues", "maximum_boundary_distance_residues", "disagreement_ranges")}
                for row in selected_core
            ],
        )
        summary = summarize_core(selected_core) + [dict(row, summary_type="continuous") for row in summarize_continuous(selected_cont)]
        for row in summary:
            row.setdefault("summary_type", "binary_core")
        write_tsv(directory / f"{prefix}_agreement_summary.tsv", summary)

    detailed_case_ids = {row["case_id"] for row in geometry}
    fraction_rows = [row for row in join_fraction_rows(core, continuous, geometry) if row["case_id"] in detailed_case_ids]
    write_tsv(MA / "05_fraction_sensitivity/fraction_sensitivity_per_case.tsv", fraction_rows)
    fraction_summary = []
    for (fraction, method), rows in sorted(group_rows([{key: str(value) for key, value in row.items()} for row in fraction_rows], ("fraction", "MDCoreAlign_method")).items()):
        fraction_summary.append(
            {
                "fraction": fraction,
                "MDCoreAlign_method": method,
                "n_cases": len(rows),
                **describe([numeric(row, "Jaccard") for row in rows], "Jaccard"),
                **describe([numeric(row, "selection_frequency_Spearman") for row in rows], "selection_frequency_Spearman"),
                **describe([numeric(row, "median_aligned_coordinate_difference_A") for row in rows], "aligned_coordinate_difference_A"),
                **describe([numeric(row, "RMSF_Spearman") for row in rows], "RMSF_Spearman"),
            }
        )
    write_tsv(MA / "05_fraction_sensitivity/fraction_sensitivity_summary.tsv", fraction_summary)
    write_tsv(MA / "05_fraction_sensitivity/fraction_agreement_curves.tsv", fraction_summary)
    write_text(
        MA / "05_fraction_sensitivity/fraction_stability_report.md",
        "# Fraction-Sensitivity Report\n\nAgreement was evaluated prospectively at 40%, 50%, 60%, 70% and 80% with exactly `K = floor(fraction * N)` residues in both methods. No fraction was selected retrospectively as globally optimal. See the structured summary for the observed distributions.\n",
    )

    native = [row for row in geometry if row["comparison_layer"] == "C_native_method_alignment"]
    write_tsv(MA / "06_alignment_outputs/transform_agreement.tsv", [{key: row[key] for key in row if key.startswith("median_rotation") or key.startswith("P95_rotation") or key.startswith("median_translation") or key in {"panel", "case_id", "system", "prefix_label", "fraction", "MDCoreAlign_method", "comparison_layer"}} for row in geometry])
    write_tsv(MA / "06_alignment_outputs/aligned_coordinate_agreement.tsv", [{key: row[key] for key in row if "coordinate_difference" in key or key in {"panel", "case_id", "system", "prefix_label", "fraction", "MDCoreAlign_method", "comparison_layer"}} for row in geometry])
    write_tsv(MA / "06_alignment_outputs/rmsd_agreement.tsv", [{key: row[key] for key in row if key.startswith("RMSD_") or key in {"panel", "case_id", "system", "prefix_label", "fraction", "MDCoreAlign_method", "comparison_layer"}} for row in geometry])
    write_tsv(MA / "06_alignment_outputs/rmsf_agreement.tsv", [{key: row[key] for key in row if key.startswith("RMSF_") or "RMSF" in key or key in {"panel", "case_id", "system", "prefix_label", "fraction", "MDCoreAlign_method", "comparison_layer"}} for row in geometry])
    write_tsv(MA / "06_alignment_outputs/mobile_region_agreement.tsv", [{key: row[key] for key in row if key in {"panel", "case_id", "system", "prefix_label", "fraction", "MDCoreAlign_method", "comparison_layer", "mobile_mask_Jaccard", "high_RMSF_Jaccard", "descriptive_agreement_category"}} for row in geometry])
    alignment_summary = []
    for (layer, method), rows in sorted(group_rows(geometry, ("comparison_layer", "MDCoreAlign_method")).items()):
        alignment_summary.append(
            {
                "comparison_layer": layer,
                "MDCoreAlign_method": method,
                "n_cases": len(rows),
                **describe([numeric(row, "median_aligned_coordinate_difference_A") for row in rows], "coordinate_difference_A"),
                **describe([numeric(row, "median_rotation_geodesic_degrees") for row in rows], "rotation_degrees"),
                **describe([numeric(row, "RMSD_Spearman") for row in rows], "RMSD_Spearman"),
                **describe([numeric(row, "RMSF_Spearman") for row in rows], "RMSF_Spearman"),
                **describe([numeric(row, "RMSF_MAE_A") for row in rows], "RMSF_MAE_A"),
            }
        )
    write_tsv(MA / "06_alignment_outputs/alignment_output_summary.tsv", alignment_summary)

    adjudication = adjudicate(core, heldout)
    write_tsv(MA / "07_independent_stability/disagreement_adjudication.tsv", adjudication)
    stability_summary = []
    for (method, category), rows in sorted(group_rows([{key: str(value) for key, value in row.items()} for row in adjudication], ("MDCoreAlign_method", "adjudication")).items()):
        stability_summary.append({"MDCoreAlign_method": method, "adjudication": category, "n_cases": len(rows), "fraction": len(rows) / sum(item["MDCoreAlign_method"] == method for item in adjudication)})
    write_tsv(MA / "07_independent_stability/independent_stability_summary.tsv", stability_summary)

    exact_detailed = sorted(
        [row for row in core_60 if row["MDCoreAlign_method"] == "exact" and any(item["case_id"] == row["case_id"] for item in adjudication)],
        key=lambda row: numeric(row, "Jaccard"),
    )
    disagreement_cases = exact_detailed[:25]
    controls = exact_detailed[-5:]
    atlas_lines = ["# Disagreement Case Atlas", "", "Cases were selected quantitatively as the 25 lowest-Jaccard detailed cases plus five highest-agreement controls; visual attractiveness was not used.", "", "| Role | Case | Panel | Jaccard | Boundary distance | Disagreement ranges |", "|---|---|---:|---:|---:|---|"]
    for role, rows in (("disagreement", disagreement_cases), ("high-agreement control", controls)):
        for row in rows:
            atlas_lines.append(f"| {role} | {row['case_id']} | {row['panel']} | {numeric(row, 'Jaccard'):.4f} | {numeric(row, 'mean_boundary_distance_residues'):.3f} | {row['disagreement_ranges']} |")
    atlas_lines.extend(["", "These are method-agreement descriptions, not biological-equivalence labels."])
    write_text(MA / "07_independent_stability/disagreement_case_atlas.md", "\n".join(atlas_lines) + "\n")

    expansion = adaptive_expansion(core_60)
    write_json(REPORTING / "adaptive_expansion_decision.json", expansion)
    make_figures(core_60, continuous_60, fraction_rows, geometry)

    pooled_exact_core = [row for row in core_60 if row["MDCoreAlign_method"] == "exact"]
    pooled_exact_cont = [row for row in continuous_60 if row["MDCoreAlign_method"] == "exact"]
    native_exact_60 = [
        row for row in native
        if row["MDCoreAlign_method"] == "exact" and abs(numeric(row, "fraction") - 0.60) < 1e-9
    ]
    med_core = float(np.median([numeric(row, "Jaccard") for row in pooled_exact_core]))
    med_score = float(np.nanmedian([numeric(row, "selection_frequency_Spearman") for row in pooled_exact_cont]))
    med_coord = float(np.median([numeric(row, "median_aligned_coordinate_difference_A") for row in native_exact_60]))
    med_rmsd_spearman = float(np.nanmedian([numeric(row, "RMSD_Spearman") for row in native_exact_60]))
    med_rmsf_spearman = float(np.nanmedian([numeric(row, "RMSF_Spearman") for row in native_exact_60]))
    if med_core >= 0.80 and med_score >= 0.80 and med_coord <= 0.10 and med_rmsf_spearman >= 0.98:
        outcome = "A_high_agreement"
        wording = "Despite their different optimisation formulations, MDCoreAlign and MDLovoFit identified substantially overlapping least-mobile regions and produced closely related aligned-coordinate and RMSF profiles."
    elif med_core >= 0.60 and med_score >= 0.60 and med_coord <= 0.25 and med_rmsf_spearman >= 0.95:
        outcome = "B_broad_scaffold_agreement_with_boundary_differences"
        wording = "The methods agreed on the broad stable scaffold but differed at core boundaries and in some multi-region trajectories."
    else:
        outcome = "C_method_dependent_stable_reference_solutions"
        wording = "MDCoreAlign and MDLovoFit produced method-dependent stable-reference solutions whose agreement varied by system, sampling and selected core fraction."

    key_metrics = [
        {"metric": "locked_primary_mdcath_domains", "value": len({row["case_id"] for row in mdcath_core}), "scope": "60% primary panel after registered adaptive decision"},
        {"metric": "locked_detailed_mdcath_domains", "value": 128, "scope": "fraction/alignment panel"},
        {"metric": "atlas_full_length_cases", "value": 8, "scope": "10,001-frame full-length"},
        {"metric": "atlas_short_prefix_tail_cases", "value": 6, "scope": "registered short-prefix tails"},
        {"metric": "historical_cases", "value": 4, "scope": "sanity panel"},
        {"metric": "pooled_exact_core_Jaccard_median", "value": med_core, "scope": "matched 60%"},
        {"metric": "pooled_exact_selection_frequency_Spearman_median", "value": med_score, "scope": "matched 60%"},
        {"metric": "detailed_exact_coordinate_difference_A_median", "value": med_coord, "scope": "native alignment, matched 60%"},
        {"metric": "detailed_exact_RMSD_Spearman_median", "value": med_rmsd_spearman, "scope": "independently recomputed"},
        {"metric": "detailed_exact_RMSF_Spearman_median", "value": med_rmsf_spearman, "scope": "independently recomputed"},
        {"metric": "equal_cardinality_chance_Jaccard_at_60pct", "value": 3.0 / 7.0, "scope": "analytic context"},
        {"metric": "adaptive_expansion_triggered", "value": expansion["adaptive_expansion_triggered"], "scope": "registered 512-to-1024 decision"},
        {"metric": "wording_outcome", "value": outcome, "scope": "descriptive, not biological equivalence"},
    ]
    write_tsv(REPORTING / "key_metrics.tsv", key_metrics)
    questions = [
        ("At 60%, do the methods identify substantially overlapping stable regions?", f"Median exact core Jaccard was {med_core:.4f}; interpretation follows outcome {outcome}."),
        ("Is agreement above equal-cardinality chance?", f"The analytic equal-cardinality chance at 60% is {3.0/7.0:.4f}; case distributions and above-chance fractions are reported in the panel summaries."),
        ("Are continuous rankings correlated?", f"Median exact selection-frequency Spearman was {med_score:.4f}. Internal aggregate evidence is reported separately and is not called per-frame frequency."),
        ("Are differences concentrated at boundaries or entire regions?", "Boundary distances and chain-aware disagreement ranges are reported per case; the case atlas was selected quantitatively."),
        ("Are transforms and aligned coordinates similar?", f"Median native exact aligned-coordinate difference was {med_coord:.4f} A. Transform identity was not required."),
        ("Are RMSD and RMSF profiles similar?", f"Median independently recomputed exact RMSD Spearman was {med_rmsd_spearman:.4f}; RMSF Spearman was {med_rmsf_spearman:.4f}."),
        ("Is agreement robust from 40% to 80%?", "The prospectively fixed five-fraction distributions are reported without selecting a post-hoc optimum."),
        ("How does disagreement compare with exact self-variability?", "Temporal-block, reference and stride envelopes are linked where frozen historical-system data exist; broad panels use held-out pair-distance variance and contact persistence."),
        ("Can the manuscript state broadly similar stable scaffolds?", wording),
        ("What caution is required?", "Agreement is method- and system-dependent, and overlap is a method-agreement analysis rather than proof of biological equivalence."),
    ]
    report_lines = [
        "# Final MDCoreAlign-MDLovoFit Method-Agreement Report",
        "",
        "This untimed analysis is scientifically separate from the NM no-materialisation timing benchmark. Exact identity was required only for the MDLovoFit exporter self-consistency check, which passed 8/8 locked cases.",
        "",
        f"## Registered wording outcome: {outcome}",
        "",
        wording,
        "",
        "## Required scientific questions",
        "",
    ]
    for index, (question, answer) in enumerate(questions, start=1):
        report_lines.extend([f"{index}. **{question}** {answer}", ""])
    report_lines.extend(
        [
            "## Scope",
            "",
            f"- Primary mdCATH panel: {len({row['case_id'] for row in mdcath_core})} corrected domains after the registered adaptive rule, with all registered tails.",
            "- Detailed mdCATH panel: 128 domains.",
            "- ATLAS: eight full-length systems and six registered short-prefix tails.",
            "- Historical panel: I-FABP, ADK, NhaA and YiiP short.",
            "- The full 5,397-domain method-agreement analysis was not run; it requires explicit human approval.",
            "",
            "## Interpretation boundary",
            "",
            "Neither method is treated as biological ground truth. Lower full-selection RMSD alone is not used to infer biological correctness. W5 and THESEUS are outside this agreement analysis.",
        ]
    )
    write_text(REPORTING / "final_method_agreement_report.md", "\n".join(report_lines))
    write_text(
        REPORTING / "manuscript_wording_recommendations.md",
        "# Manuscript Wording Recommendations\n\n"
        f"Recommended result wording: {wording}\n\n"
        "Do not state that MDCoreAlign and MDLovoFit are biologically equivalent, transform-identical or expected to return the same core. State that 60% is an equal-cardinality operating point, not a universal biological core fraction. Keep stock no-output workflow speed and instrumented core-compute speed as separate claims.\n",
    )
    write_tsv(
        REPORTING / "claim_evidence_matrix.tsv",
        [
            {"claim": "MDLovoFit exporter reproduces stock -t subset", "status": "supported", "evidence": "01_mdlovofit_export_validation/exporter_self_consistency.tsv", "caution": "Self-consistency only; not cross-method identity."},
            {"claim": "Stable-region agreement at matched 60%", "status": "supported descriptively", "evidence": "03_atlas_60pct and 04_mdcath_60pct summaries", "caution": "Method agreement is not biological equivalence."},
            {"claim": "Continuous rankings are correlated", "status": "supported descriptively", "evidence": "continuous_stability_agreement tables", "caution": "Score scales and semantics differ."},
            {"claim": "Alignment/RMSD/RMSF agreement", "status": "supported on locked detailed panel", "evidence": "06_alignment_outputs", "caution": "Transform identity is neither expected nor required."},
            {"claim": "Independent stability resolves all disagreement", "status": "partially supported", "evidence": "07_independent_stability", "caution": "Most broad cases lack biological ground truth and some frozen diagnostics exist only for historical systems."},
            {"claim": "Full 5,397-domain method agreement", "status": "not run by registered scope", "evidence": "00_contracts/panel_manifests/panel_lock_report.md", "caution": "Do not generalise the reduced agreement panel as a full-domain agreement run."},
        ],
    )
    write_text(
        REPORTING / "limitations.md",
        "# Limitations\n\n"
        "- Equal-cardinality 60% masks enable a controlled comparison but do not define a universal biological core size.\n"
        "- MDLovoFit hard subsets and MDCoreAlign soft weights solve different optimisation problems.\n"
        "- Per-frame MDLovoFit selection frequency and internal aggregate improvement evidence have different semantics and are reported separately.\n"
        "- Independent held-out geometry evaluates structural consistency, not biological truth.\n"
        "- Temporal, reference and stride self-variability envelopes are available mainly for the historical four-system panel.\n"
        "- The superseding scope uses a locked 512-domain primary panel and 128-domain detailed panel, not all 5,397 domains.\n"
        "- No THESEUS, W5 or speed benchmark was rerun for this untimed analysis.\n",
    )
    write_tsv(
        REPORTING / "phase_status.tsv",
        [
            {"phase": "MA0", "status": "PASS", "evidence": "00_contracts"},
            {"phase": "MA1", "status": "PASS", "evidence": "01_mdlovofit_export_validation/mdlovofit_subset_semantics.md"},
            {"phase": "MA2", "status": "PASS", "evidence": "01_mdlovofit_export_validation/export_validation_report.md"},
            {"phase": "MA3", "status": "PASS", "evidence": "02_legacy_four_systems"},
            {"phase": "MA4", "status": "PASS", "evidence": "03_atlas_60pct; 04_mdcath_60pct"},
            {"phase": "MA5", "status": "PASS", "evidence": "05_fraction_sensitivity"},
            {"phase": "MA6", "status": "PASS", "evidence": "06_alignment_outputs"},
            {"phase": "MA7", "status": "PASS", "evidence": "07_independent_stability"},
            {"phase": "MA8", "status": "PENDING", "evidence": "review archive not yet built"},
        ],
    )
    write_json(
        REPORTING / "ma3_ma7_final_status.json",
        {
            "status": "pass",
            "raw_cases": 530,
            "historical_cases": len({row["case_id"] for row in historical_core}),
            "atlas_cases": len({row["case_id"] for row in atlas_core}),
            "mdcath_cases": len({row["case_id"] for row in mdcath_core}),
            "detailed_cases": len({row["case_id"] for row in native}),
            "adaptive_expansion": expansion,
        },
    )
    print(json.dumps({"status": "pass", "adaptive_expansion": expansion["adaptive_expansion_triggered"]}, indent=2))


if __name__ == "__main__":
    main()
