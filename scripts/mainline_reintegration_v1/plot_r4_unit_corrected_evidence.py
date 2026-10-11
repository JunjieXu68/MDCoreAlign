#!/usr/bin/env python3
"""Plot paper-candidate R4 unit-corrected evidence without changing tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/project_completion_v2_unit_corrected"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, name: str) -> None:
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_tsv(
        RESULTS / "aggregate/all_domain_unit_corrected_per_domain.tsv"
    )
    paired = [
        row
        for row in rows
        if row["mdcorealign_status"] == "success"
        and row["mdlovofit_status"] == "success"
    ]
    all_three = [
        row for row in paired if row["theseus_status"] == "success"
    ]
    figure_rows: list[dict[str, object]] = []
    for row in paired:
        for method, runtime_field, rmsd_field in (
            (
                "MDCoreAlign exact",
                "mdcorealign_runtime_sec",
                "mdcorealign_unified_RMSD",
            ),
            (
                "MDLovoFit",
                "mdlovofit_runtime_sec",
                "mdlovofit_unified_RMSD",
            ),
        ):
            figure_rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "method": method,
                    "runtime_seconds": row[runtime_field],
                    "unified_RMSD_A": row[rmsd_field],
                }
            )
    for row in all_three:
        figure_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "method": "THESEUS",
                "runtime_seconds": row["theseus_runtime_sec"],
                "unified_RMSD_A": row["theseus_unified_RMSD"],
            }
        )
    write_tsv(RESULTS / "figure_data/fig_r4_exact_methods.tsv", figure_rows)

    methods = ("MDCoreAlign exact", "MDLovoFit", "THESEUS")
    colors = ("#176B87", "#D35400", "#477A45")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for position, (method, color) in enumerate(zip(methods, colors)):
        sample = [row for row in figure_rows if row["method"] == method]
        runtime = np.asarray(
            [float(row["runtime_seconds"]) for row in sample]
        )
        rmsd = np.asarray([float(row["unified_RMSD_A"]) for row in sample])
        axes[0].boxplot(
            runtime,
            positions=[position + 1],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.75},
            medianprops={"color": "black"},
        )
        axes[1].boxplot(
            rmsd,
            positions=[position + 1],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.75},
            medianprops={"color": "black"},
        )
    for axis in axes:
        axis.set_xticks(range(1, 4), methods, rotation=18, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Runtime (s, log scale)")
    axes[0].set_title("Formal all-domain runtime")
    axes[1].set_ylabel("Unified post-hoc RMSD (A)")
    axes[1].set_title("Unit-corrected geometric comparison")
    fig.suptitle("Corrected mdCATH all-domain evidence", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_r4_all_domain_runtime_rmsd")

    fast = read_tsv(RESULTS / "fast_global/fast_global_all_domain.tsv")
    fast_success = [row for row in fast if row["status"] == "success"]
    fast_figure_rows = [
        {
            "trajectory_id": row["trajectory_id"],
            "core_jaccard": row["core_jaccard"],
            "weight_spearman": row["weight_spearman"],
            "RMSF_spearman": row["RMSF_spearman"],
            "absolute_RMSD_difference_A": row[
                "absolute_RMSD_difference_A"
            ],
            "catastrophic_failure": row["catastrophic_failure"],
        }
        for row in fast_success
    ]
    write_tsv(
        RESULTS / "figure_data/fig_r4_fast_global_quality.tsv",
        fast_figure_rows,
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    metrics = (
        ("core_jaccard", "Core Jaccard", "#176B87"),
        ("weight_spearman", "Weight Spearman", "#477A45"),
        ("RMSF_spearman", "RMSF Spearman", "#B06A2C"),
    )
    for axis, (field, label, color) in zip(axes, metrics):
        values = np.asarray([float(row[field]) for row in fast_figure_rows])
        axis.hist(values, bins=40, color=color, alpha=0.82)
        axis.axvline(np.median(values), color="black", linestyle="--")
        axis.set_xlabel(label)
        axis.set_ylabel("Systems")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Experimental fast-global agreement with exact/default",
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig_r4_fast_global_quality")

    (RESULTS / "figures/figure_captions.md").write_text(
        """# R4 figure captions

**Figure R4-1. Unit-corrected all-domain evidence.** Runtime and unified
post-hoc RMSD distributions for the exact/default MDCoreAlign path, MDLovoFit
and THESEUS. THESEUS ensembles were oriented with one common
consensus-to-reference rigid transform; frames were not independently refit.

**Figure R4-2. Experimental fast-global quality.** Distributions of core
Jaccard, soft-weight Spearman and RMSF Spearman relative to exact/default over
5,397 successfully processed domains. Strong medians coexist with measurable
tail failures, so fast-global remains experimental.
"""
    )


if __name__ == "__main__":
    main()
