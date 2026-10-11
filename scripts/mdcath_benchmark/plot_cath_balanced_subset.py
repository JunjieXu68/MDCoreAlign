#!/usr/bin/env python
"""Generate figures, tables, and conservative text for the CATH-balanced subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, write_tsv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=Path,
        default=RESULT_ROOT / "subsets/mdcath_cath_balanced_5_per_class.tsv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULT_ROOT / "cath_balanced_subset/aggregate/cath_balanced_5_per_class_summary.tsv",
    )
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT / "cath_balanced_subset")
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    """Convert a cell to float if possible."""

    if value in {None, "", "NA", "None"}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def coverage_table(subset_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Summarize selected hierarchy coverage by CATH class."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in subset_rows:
        grouped[row["cath_class"]].append(row)
    rows: list[dict[str, object]] = []
    for class_id in sorted(grouped, key=str):
        group = grouped[class_id]
        rows.append(
            {
                "cath_class": class_id,
                "n_domains": len(group),
                "n_architectures": len({row["cath_architecture"] for row in group}),
                "n_topologies": len({row["cath_topology"] for row in group}),
                "n_homologous_superfamilies": len({row["cath_homologous_superfamily"] for row in group}),
                "domains": ";".join(row["domain_id"] for row in group),
            }
        )
    return rows


def baseline_table(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Create long-format method comparison rows."""

    rows: list[dict[str, object]] = []
    method_map = [
        ("MDCoreAlign", "mdcorealign_status", "mdcorealign_runtime_sec", "mdcorealign_unified_RMSD"),
        ("ordinary_fullset", "fullset_status", "fullset_runtime_sec", "fullset_unified_RMSD"),
        ("MDLovoFit", "mdlovofit_status", "mdlovofit_runtime_sec", "mdlovofit_unified_RMSD"),
        ("THESEUS", "theseus_status", "theseus_runtime_sec", "theseus_unified_RMSD"),
    ]
    for row in summary_rows:
        for method, status_key, runtime_key, rmsd_key in method_map:
            rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "domain_id": row["domain_id"],
                    "cath_class": row["cath_class"],
                    "method": method,
                    "status": row.get(status_key, "NA"),
                    "runtime_sec": row.get(runtime_key, "NA"),
                    "unified_RMSD": row.get(rmsd_key, "NA"),
                }
            )
    return rows


def flexibility_table(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Create flexibility-consistency summary rows."""

    return [
        {
            "trajectory_id": row["trajectory_id"],
            "domain_id": row["domain_id"],
            "cath_class": row["cath_class"],
            "stable_core_fraction": row.get("stable_core_fraction", "NA"),
            "n_core_segments": row.get("n_core_segments", "NA"),
            "n_flexible_segments": row.get("n_flexible_segments", "NA"),
            "robust_RMSF_mean": row.get("robust_RMSF_mean", "NA"),
            "noncore_core_RMSF_ratio": row.get("noncore_core_RMSF_ratio", "NA"),
            "RMSF_mobility_spearman": row.get("RMSF_mobility_spearman", "NA"),
        }
        for row in summary_rows
    ]


def plot_coverage(rows: list[dict[str, object]], output: Path) -> None:
    """Plot CATH class coverage."""

    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    classes = [str(row["cath_class"]) for row in rows]
    domains = [int(row["n_domains"]) for row in rows]
    topologies = [int(row["n_topologies"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = range(len(classes))
    ax.bar([i - 0.18 for i in x], domains, width=0.36, label="domains")
    ax.bar([i + 0.18 for i in x], topologies, width=0.36, label="topologies")
    ax.set_xticks(list(x), classes)
    ax.set_xlabel("CATH class")
    ax.set_ylabel("Count")
    ax.set_title("CATH-balanced mdCATH subset coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_baseline(rows: list[dict[str, object]], output: Path) -> None:
    """Plot runtime and RMSD comparison for completed methods."""

    import matplotlib.pyplot as plt

    completed = [row for row in rows if row["status"] == "success" and as_float(str(row["unified_RMSD"])) is not None]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    if completed:
        methods = sorted({str(row["method"]) for row in completed})
        rmsd_means = []
        runtime_means = []
        for method in methods:
            method_rows = [row for row in completed if row["method"] == method]
            rmsds = [as_float(str(row["unified_RMSD"])) for row in method_rows]
            runtimes = [as_float(str(row["runtime_sec"])) for row in method_rows]
            rmsd_means.append(sum(v for v in rmsds if v is not None) / max(sum(v is not None for v in rmsds), 1))
            runtime_means.append(sum(v for v in runtimes if v is not None) / max(sum(v is not None for v in runtimes), 1))
        axes[0].bar(methods, rmsd_means)
        axes[0].set_ylabel("Mean unified RMSD")
        axes[1].bar(methods, runtime_means)
        axes[1].set_ylabel("Mean runtime (s)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
    else:
        axes[0].text(0.5, 0.5, "No completed baseline rows", ha="center", va="center")
        axes[1].axis("off")
    axes[0].set_title("RMSD")
    axes[1].set_title("Runtime")
    fig.suptitle("CATH-balanced subset baseline comparison")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_flexibility(rows: list[dict[str, object]], output: Path) -> None:
    """Plot stable-core and RMSF consistency metrics."""

    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [as_float(str(row["RMSF_mobility_spearman"])) for row in rows]
    ys = [as_float(str(row["noncore_core_RMSF_ratio"])) for row in rows]
    labels = [str(row["domain_id"]) for row in rows]
    plotted = False
    for x, y, label in zip(xs, ys, labels):
        if x is None or y is None:
            continue
        ax.scatter(x, y)
        ax.text(x, y, label, fontsize=7)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No completed flexibility rows", ha="center", va="center")
    ax.set_xlabel("Spearman: RMSF vs mobility score")
    ax.set_ylabel("Non-core / core RMSF ratio")
    ax.set_title("Stable-core and RMSF consistency")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_manuscript_text(
    output_root: Path,
    subset_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    coverage_rows: list[dict[str, object]],
) -> None:
    """Write conservative manuscript insert drafts."""

    text_dir = output_root / "manuscript_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    class_counts = Counter(row["cath_class"] for row in subset_rows)
    n_success = sum(1 for row in summary_rows if row.get("mdcorealign_status") == "success")
    n_mdlo = sum(1 for row in summary_rows if row.get("mdlovofit_status") == "success")
    n_the = sum(1 for row in summary_rows if row.get("theseus_status") == "success")
    methods = [
        "We constructed a CATH-balanced public mdCATH subset by querying the Hugging Face mdCATH repository for available domain files, joining mdCATH domain identifiers to official CATH domain-list hierarchy annotations, and sorting available domains by Class, Architecture, Topology and Homologous superfamily.",
        "To avoid over-representing replicate simulations of the same structure, one canonical trajectory was selected per domain, using 320 K replicate 1.",
        f"The primary balanced subset manifest contains {len(subset_rows)} domains across {len(class_counts)} observed CATH classes ({'; '.join(f'{k}:{v}' for k, v in sorted(class_counts.items()))}).",
        "This subset is not a full mdCATH trajectory benchmark.",
    ]
    results = [
        f"The current execution table contains {len(summary_rows)} processed trajectories from the CATH-balanced subset manifest.",
        f"MDCoreAlign completed successfully for {n_success}/{len(summary_rows)} processed trajectories; MDLovoFit completed for {n_mdlo}/{len(summary_rows)}; THESEUS completed for {n_the}/{len(summary_rows)}.",
        "If this table was generated with a frame cap, the frame cap should be reported explicitly and the results treated as a bounded subset execution rather than a full-length mdCATH run.",
    ]
    discussion = [
        "This CATH-balanced subset is a hierarchy-balanced public-database stress test designed to complement the four formal trajectories.",
        "It should not be described as a full mdCATH database benchmark.",
        "A full all-domain or all-trajectory mdCATH benchmark remains a separate large-scale computing task, primarily limited by HDF5 download throughput and storage.",
    ]
    (text_dir / "mdcath_cath_balanced_methods_insert.md").write_text("\n".join(methods) + "\n")
    (text_dir / "mdcath_cath_balanced_results_insert.md").write_text("\n".join(results) + "\n")
    (text_dir / "mdcath_cath_balanced_discussion_insert.md").write_text("\n".join(discussion) + "\n")
    (text_dir / "mdcath_subset_methods_insert.md").write_text("\n".join(methods) + "\n")
    (text_dir / "mdcath_subset_results_insert.md").write_text("\n".join(results) + "\n")
    (text_dir / "mdcath_subset_discussion_insert.md").write_text("\n".join(discussion) + "\n")


def main() -> None:
    """Generate all subset plot/data outputs."""

    args = parse_args()
    subset_rows = read_tsv(args.subset)
    summary_rows = read_tsv(args.summary)
    table_dir = args.output_root / "manuscript_tables"
    figure_dir = args.output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cov = coverage_table(subset_rows)
    baseline = baseline_table(summary_rows)
    flex = flexibility_table(summary_rows)
    write_tsv(table_dir / "cath_balanced_subset_coverage.tsv", cov)
    write_tsv(table_dir / "cath_balanced_subset_baseline_comparison.tsv", baseline)
    write_tsv(table_dir / "cath_balanced_subset_flexibility_summary.tsv", flex)
    plot_coverage(cov, figure_dir / "cath_balanced_subset_coverage.pdf")
    plot_baseline(baseline, figure_dir / "cath_balanced_subset_baseline_comparison.pdf")
    plot_flexibility(flex, figure_dir / "cath_balanced_subset_flexibility_consistency.pdf")
    write_manuscript_text(args.output_root, subset_rows, summary_rows, cov)
    print(figure_dir)


if __name__ == "__main__":
    main()
