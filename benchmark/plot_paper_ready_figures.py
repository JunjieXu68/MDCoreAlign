"""Generate clean paper-ready figures from the current benchmark tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


METHOD_ORDER = [
    "Ordinary alignment",
    "Soft robust",
    "Soft + sequence prior",
    "Soft + contact prior",
    "Soft + seq + contact",
    "MDLovoFit",
]

METHOD_COLORS = {
    "Ordinary alignment": "#7f7f7f",
    "Soft robust": "#1f77b4",
    "Soft + sequence prior": "#17becf",
    "Soft + contact prior": "#ff7f0e",
    "Soft + seq + contact": "#2ca02c",
    "MDLovoFit": "#d62728",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for figure generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ROOT / "results" / "paper_ready_20260326",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read one CSV file into memory."""

    with path.open() as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    """Convert a table field to float when available."""

    if value in (None, ""):
        return None
    return float(value)


def set_axis_style(axis: plt.Axes, ylabel: str) -> None:
    """Apply a simple paper-like visual style to one axis."""

    axis.set_ylabel(ylabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", linestyle=":", alpha=0.35)


def ordered_methods(rows: list[dict[str, str]]) -> list[str]:
    """Return human labels in a consistent order."""

    present = {row["method_label"] for row in rows}
    ordered = [method for method in METHOD_ORDER if method in present]
    return ordered + sorted(present - set(ordered))


def metric_arrays(
    rows: list[dict[str, str]],
    case_key: str,
    cases: list[str],
    methods: list[str],
    mean_field: str,
    std_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mean/std matrices for grouped bar plotting."""

    value_map: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for row in rows:
        value_map[(row[case_key], row["method_label"])] = (
            to_float(row.get(mean_field)),
            to_float(row.get(std_field)),
        )

    means = np.full((len(methods), len(cases)), np.nan, dtype=float)
    stds = np.full((len(methods), len(cases)), np.nan, dtype=float)
    for method_index, method in enumerate(methods):
        for case_index, case in enumerate(cases):
            mean_value, std_value = value_map.get((case, method), (None, None))
            if mean_value is not None:
                means[method_index, case_index] = mean_value
            if std_value is not None:
                stds[method_index, case_index] = std_value
    return means, stds


def grouped_bars(
    axis: plt.Axes,
    cases: list[str],
    methods: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    *,
    ylabel: str,
    title: str | None = None,
) -> None:
    """Draw grouped bars with stable colors and optional error bars."""

    x = np.arange(len(cases), dtype=float)
    width = 0.82 / max(len(methods), 1)
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width

    for method_index, method in enumerate(methods):
        series = means[method_index]
        errors = stds[method_index]
        valid = ~np.isnan(series)
        axis.bar(
            x[valid] + offsets[method_index],
            series[valid],
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, "#4c4c4c"),
            edgecolor="black",
            linewidth=0.6,
            yerr=None if np.all(np.isnan(errors[valid])) else errors[valid],
            capsize=2.5,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(cases, rotation=20, ha="right")
    if title is not None:
        axis.set_title(title)
    set_axis_style(axis, ylabel)


def plot_synthetic(summary_path: Path, output_dir: Path) -> Path:
    """Plot synthetic runtime and unified RMSD across all scenarios."""

    rows = read_rows(summary_path)
    methods = ordered_methods(rows)
    cases = [case for case in [
        "Clean rigid",
        "Moderate outliers",
        "Many outliers",
        "Two quasi-rigid domains",
    ] if case in {row["case_label"] for row in rows}]

    runtime_mean, runtime_std = metric_arrays(
        rows,
        "case_label",
        cases,
        methods,
        "runtime_seconds_mean",
        "runtime_seconds_std",
    )
    rmsd_mean, rmsd_std = metric_arrays(
        rows,
        "case_label",
        cases,
        methods,
        "unified_posthoc_rmsd_mean",
        "unified_posthoc_rmsd_std",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    grouped_bars(axes[0], cases, methods, runtime_mean, runtime_std, ylabel="Runtime (s)", title="Synthetic runtime")
    grouped_bars(
        axes[1],
        cases,
        methods,
        rmsd_mean,
        rmsd_std,
        ylabel="Unified post-hoc RMSD",
        title="Synthetic geometric quality",
    )
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)
    fig.tight_layout()
    path = output_dir / "synthetic_runtime_rmsd.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ifabp(summary_path: Path, output_dir: Path) -> Path:
    """Plot I-FABP runtime and unified RMSD."""

    rows = read_rows(summary_path)
    methods = ordered_methods(rows)
    means_runtime, std_runtime = metric_arrays(
        rows,
        "case_label",
        ["I-FABP"],
        methods,
        "runtime_seconds_mean",
        "runtime_seconds_std",
    )
    means_rmsd, std_rmsd = metric_arrays(
        rows,
        "case_label",
        ["I-FABP"],
        methods,
        "unified_posthoc_rmsd_mean",
        "unified_posthoc_rmsd_std",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    grouped_bars(axes[0], ["I-FABP"], methods, means_runtime, std_runtime, ylabel="Runtime (s)", title="I-FABP runtime")
    grouped_bars(
        axes[1],
        ["I-FABP"],
        methods,
        means_rmsd,
        std_rmsd,
        ylabel="Unified post-hoc RMSD",
        title="I-FABP geometric quality",
    )
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)
    fig.tight_layout()
    path = output_dir / "ifabp_runtime_rmsd.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(summary_path: Path, output_dir: Path) -> Path:
    """Plot runtime, RMSD, and core smoothness for the ablation study."""

    rows = read_rows(summary_path)
    methods = [
        method for method in [
            "Soft robust",
            "Soft + sequence prior",
            "Soft + contact prior",
            "Soft + seq + contact",
            "MDLovoFit",
            "Ordinary alignment",
        ]
        if method in {row["method_label"] for row in rows}
    ]
    case = ["I-FABP"]
    runtime_mean, runtime_std = metric_arrays(rows, "case_label", case, methods, "runtime_seconds_mean", "runtime_seconds_std")
    rmsd_mean, rmsd_std = metric_arrays(rows, "case_label", case, methods, "unified_posthoc_rmsd_mean", "unified_posthoc_rmsd_std")

    smooth_methods = [
        method for method in [
            "Soft robust",
            "Soft + sequence prior",
            "Soft + contact prior",
            "Soft + seq + contact",
        ]
        if method in {row["method_label"] for row in rows}
    ]
    smooth_mean, smooth_std = metric_arrays(
        rows,
        "case_label",
        case,
        smooth_methods,
        "core_transition_count_mean",
        "core_transition_count_std",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    grouped_bars(axes[0], case, methods, runtime_mean, runtime_std, ylabel="Runtime (s)", title="Ablation runtime")
    grouped_bars(axes[1], case, methods, rmsd_mean, rmsd_std, ylabel="Unified post-hoc RMSD", title="Ablation RMSD")
    grouped_bars(
        axes[2],
        case,
        smooth_methods,
        smooth_mean,
        smooth_std,
        ylabel="Core transitions",
        title="Ablation core smoothness",
    )
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)
    fig.tight_layout()
    path = output_dir / "ablation_runtime_rmsd_core_smoothness.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_biological(summary_path: Path, output_dir: Path) -> Path:
    """Plot the main biological plausibility metrics."""

    rows = read_rows(summary_path)
    methods = [
        method for method in [
            "Soft robust",
            "Soft + sequence prior",
            "Soft + contact prior",
            "Soft + seq + contact",
        ]
        if method in {row["method_label"] for row in rows}
    ]
    case = ["I-FABP"]
    transition_mean, transition_std = metric_arrays(
        rows,
        "case_label",
        case,
        methods,
        "core_transition_count_mean",
        "core_transition_count_std",
    )
    variation_mean, variation_std = metric_arrays(
        rows,
        "case_label",
        case,
        methods,
        "weight_total_variation_mean",
        "weight_total_variation_std",
    )
    disagreement_mean, disagreement_std = metric_arrays(
        rows,
        "case_label",
        case,
        methods,
        "contact_graph_disagreement_mean",
        "contact_graph_disagreement_std",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    grouped_bars(
        axes[0],
        case,
        methods,
        transition_mean,
        transition_std,
        ylabel="Core transitions",
        title="Sequence smoothness",
    )
    grouped_bars(
        axes[1],
        case,
        methods,
        variation_mean,
        variation_std,
        ylabel="Weight total variation",
        title="Weight smoothness",
    )
    grouped_bars(
        axes[2],
        case,
        methods,
        disagreement_mean,
        disagreement_std,
        ylabel="Contact disagreement",
        title="Contact consistency",
    )
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)
    fig.tight_layout()
    path = output_dir / "biological_plausibility.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Generate the requested paper-ready figures."""

    args = parse_args()
    tables_dir = args.paper_root / "tables"
    output_dir = args.paper_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_synthetic(tables_dir / "synthetic_benchmark_summary.csv", output_dir),
        plot_ifabp(tables_dir / "ifabp_trajectory_summary.csv", output_dir),
        plot_ablation(tables_dir / "ablation_summary.csv", output_dir),
        plot_biological(tables_dir / "biological_plausibility_summary.csv", output_dir),
    ]

    print("Paper-ready figures complete")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
