#!/usr/bin/env python
"""Create mdCATH benchmark plots from saved TSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULT_ROOT / "aggregate/mdcath_full_trajectory_summary.tsv")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "figures")
    return parser.parse_args()


def as_float(value: object) -> float | None:
    """Convert a value to float."""

    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    """Generate plots."""

    args = parse_args()
    rows = [row for row in read_tsv(args.summary) if row.get("status") == "success"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        note = args.output_dir / "mdcath_plot_status.md"
        note.write_text("No successful mdCATH trajectories are available for plotting yet.\n")
        print(note)
        return

    import matplotlib.pyplot as plt

    plot_rows = []
    for row in rows:
        plot_rows.append(
            {
                "trajectory_id": row.get("trajectory_id"),
                "n_residues": as_float(row.get("n_residues")),
                "runtime_seconds": as_float(row.get("runtime_seconds")),
                "stable_core_fraction": as_float(row.get("stable_core_fraction")),
                "RMSF_mobility_spearman": as_float(row.get("RMSF_mobility_spearman")),
                "noncore_core_RMSF_ratio": as_float(row.get("noncore_core_RMSF_ratio")),
            }
        )
    write_tsv(RESULT_ROOT / "manuscript_tables/mdcath_main_table_coverage.tsv", plot_rows)

    n_res = np.asarray([row["n_residues"] for row in plot_rows if row["n_residues"] is not None and row["runtime_seconds"] is not None])
    runtime = np.asarray([row["runtime_seconds"] for row in plot_rows if row["n_residues"] is not None and row["runtime_seconds"] is not None])
    stable = np.asarray([row["stable_core_fraction"] for row in plot_rows if row["stable_core_fraction"] is not None])
    spearman = np.asarray([row["RMSF_mobility_spearman"] for row in plot_rows if row["RMSF_mobility_spearman"] is not None])
    ratio = np.asarray([row["noncore_core_RMSF_ratio"] for row in plot_rows if row["noncore_core_RMSF_ratio"] is not None])

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    if n_res.size:
        axes[0, 0].scatter(n_res, runtime, s=12, alpha=0.6)
    axes[0, 0].set_xlabel("CA residues")
    axes[0, 0].set_ylabel("runtime (s)")
    axes[0, 0].set_title("Runtime scaling")
    if stable.size:
        axes[0, 1].hist(stable, bins=20, color="#4477AA", alpha=0.85)
    axes[0, 1].set_xlabel("stable core fraction")
    axes[0, 1].set_title("Stable-core distribution")
    if spearman.size:
        axes[1, 0].hist(spearman, bins=20, color="#228833", alpha=0.85)
    axes[1, 0].set_xlabel("RMSF vs mobility Spearman")
    axes[1, 0].set_title("RMSF/mobility consistency")
    if ratio.size:
        axes[1, 1].hist(ratio, bins=20, color="#CC6677", alpha=0.85)
    axes[1, 1].set_xlabel("noncore/core RMSF ratio")
    axes[1, 1].set_title("Core vs mobile RMSF")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"mdcath_summary_panels.{suffix}", dpi=300)
        fig.savefig(args.output_dir / f"domain_canonical_coverage_runtime.{suffix}", dpi=300)
    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3.5))
    if stable.size:
        axes2[0].hist(stable, bins=20, color="#4477AA", alpha=0.85)
    axes2[0].set_xlabel("stable core fraction")
    axes2[0].set_ylabel("trajectory count")
    axes2[0].set_title("Stable-core fraction")
    if spearman.size:
        axes2[1].hist(spearman, bins=20, color="#228833", alpha=0.85)
    axes2[1].set_xlabel("RMSF/mobility Spearman")
    axes2[1].set_title("Flexibility consistency")
    fig2.tight_layout()
    for suffix in ("pdf", "png"):
        fig2.savefig(args.output_dir / f"domain_canonical_core_flexibility.{suffix}", dpi=300)
    print(args.output_dir / "mdcath_summary_panels.pdf")


if __name__ == "__main__":
    main()
