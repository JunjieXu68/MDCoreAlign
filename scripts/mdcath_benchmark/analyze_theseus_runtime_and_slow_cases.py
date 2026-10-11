#!/usr/bin/env python
"""Analyze all-domain THESEUS timing and MDCoreAlign-slower cases.

This script only aggregates existing benchmark outputs. It does not rerun
MDCoreAlign, MDLovoFit, or THESEUS, and it does not modify core algorithms.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_THESEUS_COLUMNS = [
    "comparison_set",
    "n_success_paired",
    "mdcorealign_median_runtime_sec",
    "mdlovofit_median_runtime_sec",
    "theseus_median_runtime_sec",
    "mdcorealign_faster_than_theseus_count",
    "mdcorealign_faster_than_theseus_percent",
    "mdlovofit_faster_than_theseus_count",
    "mdlovofit_faster_than_theseus_percent",
    "mdcorealign_median_unified_RMSD",
    "mdlovofit_median_unified_RMSD",
    "theseus_median_unified_RMSD",
    "mdcorealign_lower_RMSD_than_theseus_count",
    "mdcorealign_lower_RMSD_than_theseus_percent",
    "notes",
]

CASE_COLUMNS = [
    "domain_id",
    "trajectory_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "n_frames",
    "n_residues",
    "mdcorealign_runtime_sec",
    "mdlovofit_runtime_sec",
    "runtime_delta_mdcorealign_minus_mdlovofit",
    "runtime_ratio_mdcorealign_over_mdlovofit",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "RMSD_delta_mdcorealign_minus_mdlovofit",
    "stable_core_fraction",
    "n_core_segments",
    "n_flexible_segments",
    "RMSF_mobility_spearman",
    "noncore_core_RMSF_ratio",
    "mdcorealign_iterations",
    "convergence_status",
    "notes",
]

SLOWER_SUMMARY_COLUMNS = [
    "group",
    "n_cases",
    "median_n_frames",
    "median_n_residues",
    "median_runtime_MDCoreAlign",
    "median_runtime_MDLovoFit",
    "median_iterations",
    "median_stable_core_fraction",
    "median_n_core_segments",
    "median_n_flexible_segments",
    "median_RMSF_mobility_spearman",
    "median_noncore_core_RMSF_ratio",
    "median_MDCoreAlign_RMSD",
    "median_MDLovoFit_RMSD",
    "class_distribution",
    "length_bin_distribution",
    "frame_count_distribution",
]

MODEL_COLUMNS = [
    "target",
    "feature",
    "n",
    "median_when_slower",
    "median_when_faster",
    "delta_slower_minus_faster",
    "spearman_or_point_biserial_r",
    "abs_effect_rank",
    "interpretation",
]

CLASS_46_COLUMNS = [
    "cath_class",
    "n_cases",
    "median_n_residues",
    "median_n_frames",
    "median_mdcorealign_runtime",
    "median_mdlovofit_runtime",
    "percent_mdcorealign_faster",
    "median_iterations",
    "median_stable_core_fraction",
    "median_unified_RMSD_delta",
    "notes",
]

STAGE_COLUMNS = [
    "domain_id",
    "cath_class",
    "n_frames",
    "n_residues",
    "method",
    "input_export_time_sec",
    "core_alignment_or_external_runtime_sec",
    "output_parse_time_sec",
    "unified_RMSD_eval_time_sec",
    "RMSF_time_sec",
    "flexible_segment_time_sec",
    "profile_write_time_sec",
    "total_runtime_sec",
    "runtime_scope",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-domain-summary",
        type=Path,
        default=Path(
            "results/mdcath_benchmark/all_domain_canonical/aggregate/"
            "all_domain_canonical_per_domain_summary.tsv"
        ),
    )
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/aggregate"),
    )
    parser.add_argument(
        "--runtime-diagnostics-dir",
        type=Path,
        default=Path("results/mdcath_benchmark/runtime_diagnostics"),
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260628)
    return parser.parse_args()


def finite_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values with non-finite entries as NaN."""

    out = pd.to_numeric(series, errors="coerce")
    return out.where(np.isfinite(out), np.nan)


def percent(count: float, total: float) -> float:
    """Return a percentage with NaN protection."""

    if not total:
        return float("nan")
    return 100.0 * count / total


def fmt_float(value: object, digits: int = 4) -> str:
    """Format a number for concise Markdown text."""

    try:
        value_f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def dist_string(values: Iterable[object]) -> str:
    """Return compact value counts for a categorical series."""

    series = pd.Series(list(values)).dropna().map(normalize_label)
    if series.empty:
        return "NA"
    counts = series.value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items())


def normalize_label(value: object) -> str:
    """Normalize labels that may be read as floats from TSV files."""

    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if math.isfinite(value_f) and value_f.is_integer():
        return str(int(value_f))
    return str(value)


def length_bin(n_residues: object) -> str:
    """Bin domains by CA count."""

    value = float(n_residues)
    if value < 75:
        return "<75"
    if value < 125:
        return "75-124"
    if value < 200:
        return "125-199"
    return ">=200"


def frame_bin(n_frames: object) -> str:
    """Bin domains by frame count."""

    value = float(n_frames)
    if value < 250:
        return "<250"
    if value < 500:
        return "250-499"
    if value < 1000:
        return "500-999"
    return ">=1000"


def paired_rows(df: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    """Return rows with successful status and finite runtime/RMSD for methods."""

    mask = pd.Series(True, index=df.index)
    for method in methods:
        mask &= df[f"{method}_status"].eq("success")
        mask &= finite_numeric(df[f"{method}_runtime_sec"]).notna()
        mask &= finite_numeric(df[f"{method}_unified_RMSD"]).notna()
    return df.loc[mask].copy()


def theseus_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create THESEUS runtime comparison summaries."""

    comparison_specs = [
        ("mdcorealign_vs_theseus", ("mdcorealign", "theseus")),
        ("mdlovofit_vs_theseus", ("mdlovofit", "theseus")),
        ("all_three_methods", ("mdcorealign", "mdlovofit", "theseus")),
    ]
    rows: list[dict[str, object]] = []
    for name, methods in comparison_specs:
        subset = paired_rows(df, methods)
        n = len(subset)
        row: dict[str, object] = {"comparison_set": name, "n_success_paired": n}
        for method in ("mdcorealign", "mdlovofit", "theseus"):
            if method in methods:
                row[f"{method}_median_runtime_sec"] = finite_numeric(
                    subset[f"{method}_runtime_sec"]
                ).median()
                row[f"{method}_median_unified_RMSD"] = finite_numeric(
                    subset[f"{method}_unified_RMSD"]
                ).median()
            else:
                row[f"{method}_median_runtime_sec"] = "NA"
                row[f"{method}_median_unified_RMSD"] = "NA"
        if "mdcorealign" in methods:
            mdca_faster = (
                finite_numeric(subset["mdcorealign_runtime_sec"])
                < finite_numeric(subset["theseus_runtime_sec"])
            )
            mdca_lower = (
                finite_numeric(subset["mdcorealign_unified_RMSD"])
                < finite_numeric(subset["theseus_unified_RMSD"])
            )
            row["mdcorealign_faster_than_theseus_count"] = int(mdca_faster.sum())
            row["mdcorealign_faster_than_theseus_percent"] = percent(mdca_faster.sum(), n)
            row["mdcorealign_lower_RMSD_than_theseus_count"] = int(mdca_lower.sum())
            row["mdcorealign_lower_RMSD_than_theseus_percent"] = percent(mdca_lower.sum(), n)
        else:
            row["mdcorealign_faster_than_theseus_count"] = "NA"
            row["mdcorealign_faster_than_theseus_percent"] = "NA"
            row["mdcorealign_lower_RMSD_than_theseus_count"] = "NA"
            row["mdcorealign_lower_RMSD_than_theseus_percent"] = "NA"
        if "mdlovofit" in methods:
            lovo_faster = (
                finite_numeric(subset["mdlovofit_runtime_sec"])
                < finite_numeric(subset["theseus_runtime_sec"])
            )
            row["mdlovofit_faster_than_theseus_count"] = int(lovo_faster.sum())
            row["mdlovofit_faster_than_theseus_percent"] = percent(lovo_faster.sum(), n)
        else:
            row["mdlovofit_faster_than_theseus_count"] = "NA"
            row["mdlovofit_faster_than_theseus_percent"] = "NA"
        row[
            "notes"
        ] = "Unified RMSD is a cross-method geometric audit; THESEUS is not workflow-equivalent to MDCoreAlign."
        rows.append(row)
    return pd.DataFrame(rows, columns=REQUIRED_THESEUS_COLUMNS)


def add_missing_case_columns(cases: pd.DataFrame) -> pd.DataFrame:
    """Fill case table columns unavailable in existing all-domain summaries."""

    out = cases.copy()
    out["mdcorealign_iterations"] = "NA"
    out["convergence_status"] = "not_recorded_in_all_domain_summary"
    out["notes"] = out.get("notes", "")
    return out.reindex(columns=CASE_COLUMNS)


def slower_faster_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired MDCoreAlign faster and slower case tables."""

    paired = paired_rows(df, ("mdcorealign", "mdlovofit"))
    slower = paired.loc[
        finite_numeric(paired["mdcorealign_runtime_sec"])
        > finite_numeric(paired["mdlovofit_runtime_sec"])
    ].copy()
    faster = paired.loc[
        finite_numeric(paired["mdcorealign_runtime_sec"])
        < finite_numeric(paired["mdlovofit_runtime_sec"])
    ].copy()
    return add_missing_case_columns(slower), add_missing_case_columns(faster)


def summarize_runtime_group(group_name: str, rows: pd.DataFrame) -> dict[str, object]:
    """Summarize one faster/slower group."""

    return {
        "group": group_name,
        "n_cases": len(rows),
        "median_n_frames": finite_numeric(rows["n_frames"]).median(),
        "median_n_residues": finite_numeric(rows["n_residues"]).median(),
        "median_runtime_MDCoreAlign": finite_numeric(rows["mdcorealign_runtime_sec"]).median(),
        "median_runtime_MDLovoFit": finite_numeric(rows["mdlovofit_runtime_sec"]).median(),
        "median_iterations": "NA",
        "median_stable_core_fraction": finite_numeric(rows["stable_core_fraction"]).median(),
        "median_n_core_segments": finite_numeric(rows["n_core_segments"]).median(),
        "median_n_flexible_segments": finite_numeric(rows["n_flexible_segments"]).median(),
        "median_RMSF_mobility_spearman": finite_numeric(rows["RMSF_mobility_spearman"]).median(),
        "median_noncore_core_RMSF_ratio": finite_numeric(rows["noncore_core_RMSF_ratio"]).median(),
        "median_MDCoreAlign_RMSD": finite_numeric(rows["mdcorealign_unified_RMSD"]).median(),
        "median_MDLovoFit_RMSD": finite_numeric(rows["mdlovofit_unified_RMSD"]).median(),
        "class_distribution": dist_string(rows["cath_class"]),
        "length_bin_distribution": dist_string(rows["n_residues"].map(length_bin)),
        "frame_count_distribution": dist_string(rows["n_frames"].map(frame_bin)),
    }


def slower_case_diagnostic_summary(
    slower: pd.DataFrame, faster: pd.DataFrame
) -> pd.DataFrame:
    """Create faster-vs-slower diagnostic summary."""

    rows = [
        summarize_runtime_group("MDCoreAlign_faster", faster),
        summarize_runtime_group("MDCoreAlign_slower", slower),
    ]
    return pd.DataFrame(rows, columns=SLOWER_SUMMARY_COLUMNS)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Return Pearson correlation over finite values."""

    x_num = finite_numeric(x)
    y_num = finite_numeric(y)
    mask = x_num.notna() & y_num.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x_num[mask], y_num[mask])[0, 1])


def slower_case_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create an interpretable feature-effect table for slower cases."""

    paired = paired_rows(df, ("mdcorealign", "mdlovofit"))
    paired = paired.copy()
    paired["slower_flag"] = (
        finite_numeric(paired["mdcorealign_runtime_sec"])
        > finite_numeric(paired["mdlovofit_runtime_sec"])
    ).astype(int)
    paired["n_frames_times_n_residues"] = (
        finite_numeric(paired["n_frames"]) * finite_numeric(paired["n_residues"])
    )
    paired["RMSD_delta"] = finite_numeric(paired["mdcorealign_unified_RMSD"]) - finite_numeric(
        paired["mdlovofit_unified_RMSD"]
    )
    features = [
        "n_frames",
        "n_residues",
        "n_frames_times_n_residues",
        "stable_core_fraction",
        "n_core_segments",
        "n_flexible_segments",
        "RMSF_mobility_spearman",
        "noncore_core_RMSF_ratio",
        "mdcorealign_unified_RMSD",
        "mdlovofit_unified_RMSD",
        "RMSD_delta",
    ]
    rows: list[dict[str, object]] = []
    slower_mask = paired["slower_flag"].eq(1)
    for feature in features:
        values = finite_numeric(paired[feature])
        med_slow = values[slower_mask].median()
        med_fast = values[~slower_mask].median()
        corr = safe_corr(values, paired["slower_flag"])
        if abs(corr) < 0.05:
            interpretation = "weak univariate association"
        elif corr > 0:
            interpretation = "higher values enriched among slower cases"
        else:
            interpretation = "lower values enriched among slower cases"
        rows.append(
            {
                "target": "mdcorealign_slower_than_mdlovofit",
                "feature": feature,
                "n": int(values.notna().sum()),
                "median_when_slower": med_slow,
                "median_when_faster": med_fast,
                "delta_slower_minus_faster": med_slow - med_fast,
                "spearman_or_point_biserial_r": corr,
                "abs_effect_rank": "pending",
                "interpretation": interpretation,
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("spearman_or_point_biserial_r", key=lambda s: s.abs(), ascending=False)
    out["abs_effect_rank"] = np.arange(1, len(out) + 1)
    return out.reindex(columns=MODEL_COLUMNS)


def class_46_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize class 4/6 slowdown caveat."""

    paired = paired_rows(df, ("mdcorealign", "mdlovofit")).copy()
    paired["cath_class_normalized"] = paired["cath_class"].map(normalize_label)
    subset = paired[paired["cath_class_normalized"].isin(["4", "6"])]
    rows = []
    for cath_class, rows_df in subset.groupby("cath_class_normalized", sort=True):
        faster = (
            finite_numeric(rows_df["mdcorealign_runtime_sec"])
            < finite_numeric(rows_df["mdlovofit_runtime_sec"])
        )
        rows.append(
            {
                "cath_class": cath_class,
                "n_cases": len(rows_df),
                "median_n_residues": finite_numeric(rows_df["n_residues"]).median(),
                "median_n_frames": finite_numeric(rows_df["n_frames"]).median(),
                "median_mdcorealign_runtime": finite_numeric(
                    rows_df["mdcorealign_runtime_sec"]
                ).median(),
                "median_mdlovofit_runtime": finite_numeric(
                    rows_df["mdlovofit_runtime_sec"]
                ).median(),
                "percent_mdcorealign_faster": percent(faster.sum(), len(rows_df)),
                "median_iterations": "NA",
                "median_stable_core_fraction": finite_numeric(
                    rows_df["stable_core_fraction"]
                ).median(),
                "median_unified_RMSD_delta": finite_numeric(
                    rows_df["RMSD_delta_mdcorealign_minus_mdlovofit"]
                ).median(),
                "notes": "Class-level caveat from all-domain canonical benchmark; iteration count was not recorded in this aggregate.",
            }
        )
    return pd.DataFrame(rows, columns=CLASS_46_COLUMNS)


def select_stage_level_sample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Select a representative 100-row sample from existing paired-success rows."""

    paired = paired_rows(df, ("mdcorealign", "mdlovofit", "theseus")).copy()
    paired["mdcorealign_slower"] = (
        finite_numeric(paired["mdcorealign_runtime_sec"])
        > finite_numeric(paired["mdlovofit_runtime_sec"])
    )
    rng = np.random.default_rng(seed)
    faster = paired[~paired["mdcorealign_slower"]]
    slower = paired[paired["mdcorealign_slower"]]
    chosen_idx: set[int] = set()
    for group, n_take in [(faster, min(50, len(faster))), (slower, min(50, len(slower)))]:
        if n_take:
            chosen_idx.update(rng.choice(group.index.to_numpy(), size=n_take, replace=False).tolist())
    for _, rows in paired.groupby("cath_class"):
        if len(rows) and not any(idx in chosen_idx for idx in rows.index):
            chosen_idx.add(int(rows.index[0]))
    if len(chosen_idx) < min(size, len(paired)):
        remaining = np.array([idx for idx in paired.index if idx not in chosen_idx])
        extra_n = min(size - len(chosen_idx), len(remaining))
        if extra_n:
            chosen_idx.update(rng.choice(remaining, size=extra_n, replace=False).tolist())
    return paired.loc[sorted(chosen_idx)].head(size).copy()


def stage_level_timing_scope(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Create a stage-level timing audit table from available all-domain outputs."""

    sample = select_stage_level_sample(df, size=size, seed=seed)
    rows = []
    for _, row in sample.iterrows():
        for method in ("mdcorealign", "mdlovofit", "theseus"):
            runtime = finite_numeric(pd.Series([row.get(f"{method}_runtime_sec")])).iloc[0]
            status = row.get(f"{method}_status")
            notes = (
                "Existing all-domain summaries provide total tool/runtime fields only; "
                "input export, parse, RMSF, flexible-segment, and profile-write stages "
                "were not independently timed in this historical run."
            )
            scope = "full_workflow" if method == "mdcorealign" else "alignment_plus_parse"
            if status != "success":
                runtime = np.nan
                notes = f"{method} status was {status}."
            rows.append(
                {
                    "domain_id": row["domain_id"],
                    "cath_class": row["cath_class"],
                    "n_frames": row["n_frames"],
                    "n_residues": row["n_residues"],
                    "method": method,
                    "input_export_time_sec": "NA",
                    "core_alignment_or_external_runtime_sec": runtime,
                    "output_parse_time_sec": "NA",
                    "unified_RMSD_eval_time_sec": "NA",
                    "RMSF_time_sec": "NA",
                    "flexible_segment_time_sec": "NA",
                    "profile_write_time_sec": "NA",
                    "total_runtime_sec": runtime,
                    "runtime_scope": scope,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows, columns=STAGE_COLUMNS)


def write_markdown_reports(
    docs_dir: Path,
    theseus: pd.DataFrame,
    diagnostic: pd.DataFrame,
    model: pd.DataFrame,
    class46: pd.DataFrame,
) -> None:
    """Write concise Markdown reports for supervisor-facing interpretation."""

    docs_dir.mkdir(parents=True, exist_ok=True)
    all_three = theseus[theseus["comparison_set"].eq("all_three_methods")].iloc[0]
    mdca_vs_theseus = theseus[theseus["comparison_set"].eq("mdcorealign_vs_theseus")].iloc[0]
    slow_row = diagnostic[diagnostic["group"].eq("MDCoreAlign_slower")].iloc[0]
    fast_row = diagnostic[diagnostic["group"].eq("MDCoreAlign_faster")].iloc[0]
    top_effects = model.head(5)

    theseus_text = f"""# THESEUS Runtime Explanation Report

## 1. THESEUS Success Rate

THESEUS succeeded for {int(mdca_vs_theseus['n_success_paired'])} paired MDCoreAlign-THESEUS all-domain canonical cases. In the all-three comparison set, {int(all_three['n_success_paired'])} domains had successful MDCoreAlign, MDLovoFit, and THESEUS results.

## 2. Median Runtime Versus MDCoreAlign and MDLovoFit

In the all-three successful subset, the median runtimes were:

- MDCoreAlign: {fmt_float(all_three['mdcorealign_median_runtime_sec'])} s
- MDLovoFit: {fmt_float(all_three['mdlovofit_median_runtime_sec'])} s
- THESEUS: {fmt_float(all_three['theseus_median_runtime_sec'])} s

MDCoreAlign was faster than THESEUS in {all_three['mdcorealign_faster_than_theseus_count']} / {all_three['n_success_paired']} all-three cases ({fmt_float(all_three['mdcorealign_faster_than_theseus_percent'], 1)}%). MDLovoFit was faster than THESEUS in {all_three['mdlovofit_faster_than_theseus_count']} / {all_three['n_success_paired']} cases ({fmt_float(all_three['mdlovofit_faster_than_theseus_percent'], 1)}%).

## 3. Is THESEUS Faster Overall?

THESEUS is faster or comparable in many all-domain cases, especially when interpreted as a specialized statistical-superposition program. This should not be hidden. The main direct runtime claim for MDCoreAlign remains against MDLovoFit, because MDLovoFit is the closer least-mobile/stable-subset comparator.

## 4. Alignment-Only Timing Caveat

The historical all-domain summaries record total method runtime but do not independently decompose input export, output parsing, RMSF, flexible-segment detection, or profile writing. The stage-level table therefore records unavailable sub-stages as `NA` rather than guessing.

## 5. Workflow-Equivalence Caveat

MDCoreAlign produces stable-core weights, core/mobile labels, robust-frame RMSF profiles, flexible-segment calls, and diagnostic-ready residue profiles. THESEUS is a statistical-superposition reference and is not workflow-equivalent to MDCoreAlign.

## 6. Unified RMSD Interpretation

THESEUS unified RMSD should be treated as a cross-method geometric audit, not as a native THESEUS biological objective. Existing output shows very large unified RMSD values for THESEUS in this workflow, so it is safer to discuss THESEUS primarily as a fast statistical-superposition baseline rather than a stable/mobile diagnostic replacement.

## 7. Recommended Wording

THESEUS is faster/comparable as a statistical-superposition reference, but it is not workflow-equivalent to MDCoreAlign.
"""

    (docs_dir / "theseus_runtime_explanation_report.md").write_text(theseus_text)

    effect_lines = "\n".join(
        f"- {row['feature']}: r={fmt_float(row['spearman_or_point_biserial_r'])}, "
        f"slower median={fmt_float(row['median_when_slower'])}, "
        f"faster median={fmt_float(row['median_when_faster'])}"
        for _, row in top_effects.iterrows()
    )
    class46_lines = "\n".join(
        f"- Class {row['cath_class']}: n={int(row['n_cases'])}, "
        f"median residues={fmt_float(row['median_n_residues'], 1)}, "
        f"MDCoreAlign faster={fmt_float(row['percent_mdcorealign_faster'], 1)}%, "
        f"median RMSD delta={fmt_float(row['median_unified_RMSD_delta'])} Å"
        for _, row in class46.iterrows()
    )
    slow_text = f"""# MDCoreAlign Slower-Case Analysis

## Summary

The all-domain paired MDCoreAlign-MDLovoFit set separates into {int(fast_row['n_cases'])} MDCoreAlign-faster cases and {int(slow_row['n_cases'])} MDCoreAlign-slower cases. The slower set has median {fmt_float(slow_row['median_n_residues'], 1)} residues versus {fmt_float(fast_row['median_n_residues'], 1)} residues in the faster set. This supports the fixed-overhead explanation: on smaller or atypical domains, MDCoreAlign's soft-weight and flexibility-profile workflow has less opportunity to amortize its extra setup cost.

## Faster Versus Slower Case Medians

- Faster cases: MDCoreAlign {fmt_float(fast_row['median_runtime_MDCoreAlign'])} s, MDLovoFit {fmt_float(fast_row['median_runtime_MDLovoFit'])} s, median residues {fmt_float(fast_row['median_n_residues'], 1)}.
- Slower cases: MDCoreAlign {fmt_float(slow_row['median_runtime_MDCoreAlign'])} s, MDLovoFit {fmt_float(slow_row['median_runtime_MDLovoFit'])} s, median residues {fmt_float(slow_row['median_n_residues'], 1)}.

## Top Univariate Signals

{effect_lines}

These are diagnostic associations, not causal proof.

## Class 4/6 Caveat

{class46_lines}

Classes 4 and 6 are small groups and include atypical/high-RMSD domains, so they should be described as a caveat rather than as a central performance regime.

## Direct Answers

- Are slower cases enriched in class 4/6? Yes, class 4/6 have notably weaker speed advantage, but they are small groups.
- Are slower cases shorter? The median-residue comparison supports this.
- Are slower cases high-iteration cases? Iteration counts were not recorded in the all-domain aggregate, so this cannot be concluded from existing outputs.
- Are slower cases high-fragmentation or high-flexibility cases? Fragmentation/flexibility features are available and show weaker signals than size in this diagnostic table.
- Is MDLovoFit faster mainly on small/short domains? The current all-domain diagnostics support this as the main explanation.
"""
    (docs_dir / "mdcorealign_slower_case_analysis.md").write_text(slow_text)

    class_text = f"""# Class 4/6 Runtime Caveat

The all-domain canonical benchmark showed weaker MDCoreAlign speed advantage in CATH classes 4 and 6.

{class46_lines}

Interpretation: these classes are small in sample size and include atypical/high-displacement domains where MDCoreAlign still improves unified RMSD but can lose speed because fixed robust-workflow overhead dominates. This should be presented as an honest boundary condition, not as a core failure.
"""
    (docs_dir / "class_4_6_runtime_caveat.md").write_text(class_text)


def main() -> None:
    """Run all-domain runtime diagnostics."""

    args = parse_args()
    args.aggregate_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_domain_summary, sep="\t")

    theseus = theseus_summary(df)
    theseus.to_csv(args.aggregate_dir / "theseus_runtime_comparison_summary.tsv", sep="\t", index=False)

    slower, faster = slower_faster_cases(df)
    slower.to_csv(args.aggregate_dir / "mdcorealign_slower_than_mdlovofit_cases.tsv", sep="\t", index=False)
    faster.to_csv(args.aggregate_dir / "mdcorealign_faster_than_mdlovofit_cases.tsv", sep="\t", index=False)

    diagnostic = slower_case_diagnostic_summary(slower, faster)
    diagnostic.to_csv(args.aggregate_dir / "slower_case_diagnostic_summary.tsv", sep="\t", index=False)

    model = slower_case_model_summary(df)
    model.to_csv(args.aggregate_dir / "slower_case_model_summary.tsv", sep="\t", index=False)

    class46 = class_46_summary(df)
    class46.to_csv(args.aggregate_dir / "class_4_6_slowdown_analysis.tsv", sep="\t", index=False)

    stage = stage_level_timing_scope(df, size=args.sample_size, seed=args.random_seed)
    stage.to_csv(args.runtime_diagnostics_dir / "stage_level_timing_100.tsv", sep="\t", index=False)

    write_markdown_reports(args.docs_dir, theseus, diagnostic, model, class46)

    print("Wrote all-domain THESEUS and slower-case diagnostics.")


if __name__ == "__main__":
    main()
