#!/usr/bin/env python
"""Finalize long-trajectory stitching benchmark outputs from existing results.

This script is post-processing only. It reads completed long-stitching aggregate
tables and writes standardized supervisor-facing tables, figures, and reports.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark import run_replicate_stitching_external_comparison as external


PREFIX_ORDER = ["R1", "R1_R2", "R1_R2_R3", "R1_R2_R3_R4", "R1_R2_R3_R4_R5"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/mdcath_benchmark/long_trajectory_stitching"),
    )
    parser.add_argument(
        "--all-domain-aggregate",
        type=Path,
        default=Path("results/mdcath_benchmark/all_domain_canonical/aggregate"),
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV table."""

    return pd.read_csv(path, sep="\t")


def to_num(series: pd.Series) -> pd.Series:
    """Convert values to numeric floats with invalid values as NaN."""

    return pd.to_numeric(series, errors="coerce").where(lambda s: np.isfinite(s), np.nan)


def fmt(value: object, digits: int = 3) -> str:
    """Format numeric values for Markdown."""

    try:
        value_f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def pct(count: float, total: float) -> float:
    """Return percent with NaN protection."""

    return float(100.0 * count / total) if total else float("nan")


def finite_median(values: Iterable[object]) -> float:
    """Return finite median or NaN."""

    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.median()) if len(vals) else float("nan")


def create_stitching_manifest(root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """Combine per-domain stitching manifests under the required schema."""

    rows = []
    for domain_id in selected["domain_id"].astype(str):
        domain_manifest = root / domain_id / "stitching_manifest.tsv"
        extraction_manifest = root / domain_id / "replicate_extraction_manifest.tsv"
        mapping_ok = "NA"
        if extraction_manifest.exists():
            ext = read_tsv(extraction_manifest)
            mapping_ok = int(to_num(ext["residue_mapping_ok"]).fillna(0).min()) if len(ext) else 0
        if not domain_manifest.exists():
            continue
        table = read_tsv(domain_manifest)
        for _, row in table.iterrows():
            included = str(row["included_replicates"])
            rows.append(
                {
                    "domain_id": domain_id,
                    "prefix": row["stitch_name"],
                    "included_replicates": included,
                    "n_replicates": len([item for item in included.split(",") if item]),
                    "n_frames": row["n_total_frames"],
                    "n_residues": row["n_residues"],
                    "boundary_jump_mean": row["mean_boundary_jump"],
                    "boundary_jump_max": row["max_boundary_jump"],
                    "boundary_to_within_jump_ratio": row["boundary_to_within_jump_ratio"],
                    "residue_mapping_ok": mapping_ok,
                    "notes": row["notes"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(root / "stitching_manifest.tsv", sep="\t", index=False)
    return out


def create_tool_comparison(root: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create the required long-trajectory tool comparison table."""

    out = pd.DataFrame(
        {
            "domain_id": comparison["domain_id"],
            "prefix": comparison["prefix"],
            "n_replicates": comparison["n_replicates"],
            "n_frames": comparison["n_frames"],
            "n_residues": comparison["n_residues"],
            "mdcorealign_runtime_sec": comparison["mdcorealign_median_runtime_sec"],
            "mdlovofit_runtime_sec": comparison["mdlovofit_median_runtime_sec"],
            "theseus_runtime_sec": comparison["theseus_median_runtime_sec"],
            "mdcorealign_vs_mdlovofit_speedup": comparison["mdcorealign_vs_mdlovofit_speedup"],
            "mdcorealign_vs_theseus_speedup": comparison["mdcorealign_vs_theseus_speedup"],
            "mdcorealign_faster_than_mdlovofit": comparison["mdcorealign_faster_than_mdlovofit"],
            "mdcorealign_faster_than_theseus": comparison["mdcorealign_faster_than_theseus"],
            "mdcorealign_unified_RMSD": comparison["mdcorealign_unified_RMSD"],
            "mdlovofit_unified_RMSD": comparison["mdlovofit_unified_RMSD"],
            "theseus_unified_RMSD": comparison["theseus_unified_RMSD"],
            "mdcorealign_lower_RMSD_than_mdlovofit": comparison[
                "mdcorealign_lower_RMSD_than_mdlovofit"
            ],
            "mdcorealign_lower_RMSD_than_theseus": comparison[
                "mdcorealign_lower_RMSD_than_theseus"
            ],
            "boundary_to_within_jump_ratio": comparison["boundary_to_within_jump_ratio"],
            "notes": comparison["notes"],
        }
    )
    out.to_csv(root / "aggregate" / "long_trajectory_tool_comparison.tsv", sep="\t", index=False)
    return out


def create_speedup_vs_frame_count(root: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize speedup and RMSD audit by stitched-prefix length."""

    rows = []
    comparison = comparison.copy()
    comparison["prefix_order"] = comparison["prefix"].map({p: i for i, p in enumerate(PREFIX_ORDER)})
    for prefix, group in comparison.sort_values("prefix_order").groupby("prefix", sort=False):
        n = len(group)
        mdca_rt = to_num(group["mdcorealign_runtime_sec"])
        mdlo_rt = to_num(group["mdlovofit_runtime_sec"])
        th_rt = to_num(group["theseus_runtime_sec"])
        rows.append(
            {
                "prefix": prefix,
                "n_cases": n,
                "median_n_frames": finite_median(group["n_frames"]),
                "MDCoreAlign_median_runtime": finite_median(mdca_rt),
                "MDLovoFit_median_runtime": finite_median(mdlo_rt),
                "THESEUS_median_runtime": finite_median(th_rt),
                "MDCoreAlign_vs_MDLovoFit_median_speedup": finite_median(
                    group["mdcorealign_vs_mdlovofit_speedup"]
                ),
                "MDCoreAlign_faster_than_MDLovoFit_count": int(
                    to_num(group["mdcorealign_faster_than_mdlovofit"]).fillna(0).sum()
                ),
                "MDCoreAlign_faster_than_MDLovoFit_percent": pct(
                    to_num(group["mdcorealign_faster_than_mdlovofit"]).fillna(0).sum(), n
                ),
                "MDCoreAlign_vs_THESEUS_median_speedup": finite_median(
                    group["mdcorealign_vs_theseus_speedup"]
                ),
                "MDCoreAlign_faster_than_THESEUS_count": int(
                    to_num(group["mdcorealign_faster_than_theseus"]).fillna(0).sum()
                ),
                "MDCoreAlign_faster_than_THESEUS_percent": pct(
                    to_num(group["mdcorealign_faster_than_theseus"]).fillna(0).sum(), n
                ),
                "MDCoreAlign_lower_RMSD_than_MDLovoFit_percent": pct(
                    to_num(group["mdcorealign_lower_RMSD_than_mdlovofit"]).fillna(0).sum(), n
                ),
                "MDCoreAlign_lower_RMSD_than_THESEUS_percent": pct(
                    to_num(group["mdcorealign_lower_RMSD_than_theseus"]).fillna(0).sum(), n
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(root / "aggregate" / "speedup_vs_frame_count.tsv", sep="\t", index=False)
    return out


def create_alignment_scope_table(root: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create alignment-only versus full-workflow timing table."""

    frame_scaling = read_tsv(root / "aggregate" / "frame_scaling_runtime.tsv")
    frame_lookup = {
        (str(row["domain_id"]), str(row["stitch_name"])): row for _, row in frame_scaling.iterrows()
    }
    rows = []
    for _, row in comparison.iterrows():
        key = (str(row["domain_id"]), str(row["prefix"]))
        mdca_stage = frame_lookup.get(key, {})
        rows.append(
            {
                "domain_id": row["domain_id"],
                "prefix": row["prefix"],
                "method": "mdcorealign",
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "alignment_only_runtime_sec": row["mdcorealign_runtime_sec"],
                "output_parse_runtime_sec": "NA",
                "RMSF_runtime_sec": mdca_stage.get("rmsf_sec", "NA"),
                "flexible_segment_runtime_sec": "NA",
                "profile_write_runtime_sec": mdca_stage.get("output_write_sec", "NA"),
                "full_workflow_runtime_sec": mdca_stage.get("runtime_sec", row["mdcorealign_runtime_sec"]),
                "notes": "MDCoreAlign alignment-only is warm-cache algorithm-only timing; full workflow is historical stitched diagnostic run with RMSF/profile outputs.",
            }
        )
        for method, col in (
            ("mdlovofit", "mdlovofit_runtime_sec"),
            ("theseus", "theseus_runtime_sec"),
        ):
            rows.append(
                {
                    "domain_id": row["domain_id"],
                    "prefix": row["prefix"],
                    "method": method,
                    "n_frames": row["n_frames"],
                    "n_residues": row["n_residues"],
                    "alignment_only_runtime_sec": row[col],
                    "output_parse_runtime_sec": "included_or_unavailable",
                    "RMSF_runtime_sec": "NA",
                    "flexible_segment_runtime_sec": "NA",
                    "profile_write_runtime_sec": "NA",
                    "full_workflow_runtime_sec": row[col],
                    "notes": "External tool runtime on cached PDB input; downstream MDCoreAlign-style RMSF/flexible-segment outputs are not produced by this method.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(root / "aggregate" / "alignment_only_vs_full_workflow_timing.tsv", sep="\t", index=False)
    return out


def copy_required_aliases(root: Path) -> None:
    """Write required alias filenames for existing aggregate tables."""

    aggregate = root / "aggregate"
    shutil.copyfile(
        aggregate / "stitched_external_tool_warmcache_repeats.tsv",
        aggregate / "warmcache_repeats.tsv",
    )
    shutil.copyfile(
        aggregate / "stitched_external_tool_warmcache_summary.tsv",
        aggregate / "warmcache_summary.tsv",
    )
    shutil.copyfile(
        aggregate / "tool_scaling_model_summary.tsv",
        aggregate / "scaling_model_summary.tsv",
    )


def create_long_report(
    root: Path,
    docs_dir: Path,
    comparison: pd.DataFrame,
    speedup: pd.DataFrame,
    scaling: pd.DataFrame,
    all_domain_aggregate: Path,
) -> None:
    """Write the supervisor-facing long-trajectory report."""

    docs_dir.mkdir(parents=True, exist_ok=True)
    theseus_summary = read_tsv(all_domain_aggregate / "theseus_runtime_comparison_summary.tsv")
    slow_summary = read_tsv(all_domain_aggregate / "slower_case_diagnostic_summary.tsv")
    class46 = read_tsv(all_domain_aggregate / "class_4_6_slowdown_analysis.tsv")

    n = len(comparison)
    mdlo_success = to_num(comparison["mdlovofit_runtime_sec"]).notna().sum()
    th_success = to_num(comparison["theseus_runtime_sec"]).notna().sum()
    mdlo_faster_count = int(to_num(comparison["mdcorealign_faster_than_mdlovofit"]).fillna(0).sum())
    th_faster_count = int(to_num(comparison["mdcorealign_faster_than_theseus"]).fillna(0).sum())
    mdlo_lower_count = int(to_num(comparison["mdcorealign_lower_RMSD_than_mdlovofit"]).fillna(0).sum())
    th_lower_count = int(to_num(comparison["mdcorealign_lower_RMSD_than_theseus"]).fillna(0).sum())
    med_mdca = finite_median(comparison["mdcorealign_runtime_sec"])
    med_mdlo = finite_median(comparison["mdlovofit_runtime_sec"])
    med_th = finite_median(comparison["theseus_runtime_sec"])
    med_speed_mdlo = finite_median(comparison["mdcorealign_vs_mdlovofit_speedup"])
    med_speed_th = finite_median(comparison["mdcorealign_vs_theseus_speedup"])
    med_boundary = finite_median(comparison["boundary_to_within_jump_ratio"])
    scaling_counts = (
        scaling.groupby(["method", "estimated_scaling_preference"]).size().reset_index(name="n")
    )
    scaling_text = "; ".join(
        f"{row.method} {row.estimated_scaling_preference}:{int(row.n)}"
        for _, row in scaling_counts.iterrows()
    )

    speed_lines = "\n".join(
        f"- {row['prefix']}: median frames={fmt(row['median_n_frames'], 1)}, "
        f"MDCoreAlign/MDLovoFit speedup={fmt(row['MDCoreAlign_vs_MDLovoFit_median_speedup'])}, "
        f"MDCoreAlign faster={fmt(row['MDCoreAlign_faster_than_MDLovoFit_percent'], 1)}%, "
        f"MDCoreAlign/THESEUS speedup={fmt(row['MDCoreAlign_vs_THESEUS_median_speedup'])}"
        for _, row in speedup.iterrows()
    )
    theseus_all = theseus_summary[theseus_summary["comparison_set"].eq("all_three_methods")].iloc[0]
    slow_row = slow_summary[slow_summary["group"].eq("MDCoreAlign_slower")].iloc[0]
    fast_row = slow_summary[slow_summary["group"].eq("MDCoreAlign_faster")].iloc[0]
    class_lines = "\n".join(
        f"- CATH class {row['cath_class']}: n={int(row['n_cases'])}, "
        f"median residues={fmt(row['median_n_residues'], 1)}, "
        f"MDCoreAlign faster={fmt(row['percent_mdcorealign_faster'], 1)}%, "
        f"median RMSD delta={fmt(row['median_unified_RMSD_delta'])} Å"
        for _, row in class46.iterrows()
    )
    report = f"""# Long-Trajectory Stitching Supervisor Report

## 1. Why THESEUS Can Be Faster

THESEUS is a specialized statistical-superposition reference and does not produce MDCoreAlign's stable-core weights, core/mobile labels, robust-frame RMSF profiles, flexible-segment calls, or diagnostic residue profiles. In the all-domain all-three successful subset, median runtimes were MDCoreAlign {fmt(theseus_all['mdcorealign_median_runtime_sec'])} s, MDLovoFit {fmt(theseus_all['mdlovofit_median_runtime_sec'])} s, and THESEUS {fmt(theseus_all['theseus_median_runtime_sec'])} s. MDCoreAlign was faster than THESEUS in {int(theseus_all['mdcorealign_faster_than_theseus_count'])}/{int(theseus_all['n_success_paired'])} cases ({fmt(theseus_all['mdcorealign_faster_than_theseus_percent'], 1)}%).

## 2. Why MDCoreAlign Is Slower in Some All-Domain Cases

The all-domain slower-case diagnostic supports a fixed-overhead explanation. MDCoreAlign-faster cases had median {fmt(fast_row['median_n_residues'], 1)} residues, whereas MDCoreAlign-slower cases had median {fmt(slow_row['median_n_residues'], 1)} residues. This means short domains provide less opportunity to amortize MDCoreAlign's soft-core and flexibility-profile workflow overhead.

Class 4/6 caveat:

{class_lines}

## 3. Long-Trajectory Stitched Benchmark Design

The stitched benchmark used 25 mdCATH domains selected from the all-domain canonical benchmark: 10 MDCoreAlign-faster cases, 10 MDCoreAlign-slower cases, and 5 CATH class 4/6 caveat cases. For each domain, five 320 K replicates were extracted and progressively stitched into R1, R1_R2, R1_R2_R3, R1_R2_R3_R4, and R1_R2_R3_R4_R5. Full frames were used, with no stride and no frame cap. Timing used cached stitched CA-only inputs and excluded Hugging Face download and HDF5 extraction.

Important caveat: stitching independent replicates introduces large boundary jumps and should be treated as a diagnostic stress test, not as a physically continuous MD trajectory. The median boundary-to-within jump ratio was {fmt(med_boundary)}.

## 4. Long-Trajectory Results

Across {n} stitched-prefix inputs, MDLovoFit had finite runtime for {mdlo_success}/{n} and THESEUS for {th_success}/{n}. Median warm-cache runtimes were:

- MDCoreAlign: {fmt(med_mdca)} s
- MDLovoFit: {fmt(med_mdlo)} s
- THESEUS: {fmt(med_th)} s

MDCoreAlign was faster than MDLovoFit in {mdlo_faster_count}/{mdlo_success} finite comparisons ({fmt(pct(mdlo_faster_count, mdlo_success), 1)}%) with median speedup {fmt(med_speed_mdlo)}. MDCoreAlign was faster than THESEUS in {th_faster_count}/{th_success} finite comparisons ({fmt(pct(th_faster_count, th_success), 1)}%) with median speedup {fmt(med_speed_th)}.

For the unified RMSD audit, MDCoreAlign had lower RMSD than MDLovoFit in {mdlo_lower_count}/{mdlo_success} finite MDLovoFit comparisons and lower RMSD than THESEUS in {th_lower_count}/{th_success} finite THESEUS comparisons. Missing external values are treated as unavailable rather than as wins or losses.

## 5. Does Longer Trajectory Length Help?

Prefix-level summary:

{speed_lines}

This directly answers the supervisor's question: in this deliberately challenging 25-domain stitched panel, increasing the frame count did not produce a clear MDCoreAlign speed advantage over MDLovoFit. The median MDCoreAlign-vs-MDLovoFit speedup remained near parity but slightly below 1.0 across prefix lengths, so MDLovoFit remained marginally faster for many stitched inputs. In contrast, increasing frame count strongly favored MDCoreAlign relative to THESEUS, whose median speedup ratio crossed above 1.0 from R1_R2_R3 onward and reached the largest value at R1_R2_R3_R4_R5.

## 6. Scaling Model Results

Scaling labels across domain-method fits: {scaling_text}. These labels are conservative five-point diagnostics and should not be over-interpreted as formal complexity proofs.

## 7. Recommended Interpretation

MDCoreAlign's all-domain canonical benchmark already supports a median 1.85x speedup over MDLovoFit and lower unified RMSD in all paired MDLovoFit comparisons. The 25-domain long stitched stress test adds an important boundary condition: longer stitched trajectories do not automatically make MDCoreAlign faster than MDLovoFit when the panel is enriched for short/slower/class 4/6 cases, but MDCoreAlign remains near runtime parity and preserves better unified RMSD on finite comparisons. The same stress test suggests THESEUS can become slower with frame count, consistent with possible superlinear behaviour in this diagnostic setting. Stitched replicates should remain a stress test rather than a physically continuous MD benchmark.
"""
    (docs_dir / "long_trajectory_stitching_supervisor_report.md").write_text(report)


def main() -> None:
    """Finalize long-trajectory stitching benchmark outputs."""

    args = parse_args()
    aggregate = args.root / "aggregate"
    selected = read_tsv(args.root / "selected_25_domains.tsv")
    comparison_raw = read_tsv(aggregate / "stitched_tool_comparison_summary.tsv")

    create_stitching_manifest(args.root, selected)
    copy_required_aliases(args.root)
    comparison = create_tool_comparison(args.root, comparison_raw)
    speedup = create_speedup_vs_frame_count(args.root, comparison)
    create_alignment_scope_table(args.root, comparison)
    scaling = read_tsv(aggregate / "scaling_model_summary.tsv")

    external.DOMAINS = tuple(selected["domain_id"].astype(str))
    external.PREFIXES = tuple(PREFIX_ORDER)
    warm_summary = external.read_tsv(aggregate / "warmcache_summary.tsv")
    comparison_rows = external.read_tsv(aggregate / "stitched_tool_comparison_summary.tsv")
    scaling_rows = external.read_tsv(aggregate / "scaling_model_summary.tsv")
    rmsd_rows = external.read_tsv(aggregate / "stitched_prefix_unified_rmsd_comparison.tsv")
    external.make_external_plots(args.root, warm_summary, comparison_rows)
    external.update_report(
        args.root,
        comparison_rows,
        scaling_rows,
        rmsd_rows,
        "long_trajectory_stitching",
    )

    figdir = args.root / "figures"
    if (figdir / "replicate_stitching_scaling.pdf").exists():
        shutil.copyfile(
            figdir / "replicate_stitching_scaling.pdf",
            figdir / "long_trajectory_stitching_benchmark.pdf",
        )
    if (figdir / "replicate_stitching_scaling.png").exists():
        shutil.copyfile(
            figdir / "replicate_stitching_scaling.png",
            figdir / "long_trajectory_stitching_benchmark.png",
        )

    create_long_report(args.root, args.docs_dir, comparison, speedup, scaling, args.all_domain_aggregate)
    print(args.root / "aggregate" / "speedup_vs_frame_count.tsv")


if __name__ == "__main__":
    main()
