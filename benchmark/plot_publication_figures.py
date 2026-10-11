"""Generate publication-friendly bar plots from benchmark summary tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_METRICS = [
    "runtime_seconds_mean",
    "rmsd_mean",
    "weighted_rmsd_mean",
    "iteration_count_mean",
    "core_fraction_mean",
]


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_figures")
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    """Load and concatenate summary rows from CSV files."""

    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def plot_metric(rows: list[dict[str, str]], metric: str, output_dir: Path) -> Path:
    """Plot one benchmark metric grouped by dataset and method."""

    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        if not row.get(metric):
            continue
        grouped.setdefault(row["dataset_name"], []).append((row["method_name"], float(row[metric])))

    if not grouped:
        raise ValueError(f"metric {metric} was not found in the provided summaries")

    datasets = list(grouped)
    methods = sorted({method for values in grouped.values() for method, _ in values})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), squeeze=False)
    color_map = plt.get_cmap("tab10")

    for axis, dataset in zip(axes[0], datasets, strict=True):
        values = {method: value for method, value in grouped[dataset]}
        plotted_methods = [method for method in methods if method in values]
        plotted_values = [values[method] for method in plotted_methods]
        axis.bar(
            range(len(plotted_methods)),
            plotted_values,
            color=[color_map(index % 10) for index in range(len(plotted_methods))],
            edgecolor="black",
            linewidth=0.8,
        )
        axis.set_title(dataset)
        axis.set_xticks(range(len(plotted_methods)))
        axis.set_xticklabels(plotted_methods, rotation=35, ha="right")
        axis.set_ylabel(metric)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    output_path = output_dir / f"{metric}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate the requested publication-style summary plots."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.summary_csv)
    metrics = args.metric or DEFAULT_METRICS

    print("Generating plots")
    for metric in metrics:
        output_path = plot_metric(rows, metric, args.output_dir)
        print(f"{metric}: {output_path}")


if __name__ == "__main__":
    main()
