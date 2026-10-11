"""Generate publication-ready candidate flexibility-profile figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROFILES = [
    ("ifabp", "I-FABP", ROOT / "results" / "flexibility" / "ifabp_flexibility_profile.csv"),
    ("adk", "ADK", ROOT / "results" / "flexibility" / "adk_flexibility_profile.csv"),
    ("nhaa", "NhaA", ROOT / "results" / "flexibility" / "nhaa_flexibility_profile.csv"),
    ("yiip_short", "YiiP short", ROOT / "results" / "flexibility" / "yiip_short_flexibility_profile.csv"),
]

PROFILE_REQUIRED_COLUMNS = [
    "residue_id",
    "resname",
    "chain",
    "robust_aligned_RMSF",
    "ordinary_aligned_RMSF",
    "soft_stability_weight",
    "mobility_score",
    "flexible_segment_id",
    "rank",
    "percentile",
    "core_label",
]

RMSF_FIGURE_FIELDS = [
    "system",
    "system_key",
    "panel_label",
    "chain",
    "residue_id",
    "resname",
    "residue_order",
    "plot_position",
    "alignment",
    "rmsf",
    "mobility_score",
    "stability_weight",
    "flexible_segment_id",
    "core_label",
    "is_flexible",
    "is_terminal",
]

MOBILITY_FIGURE_FIELDS = [
    "system",
    "system_key",
    "chain",
    "residue_id",
    "resname",
    "robust_aligned_RMSF",
    "ordinary_aligned_RMSF",
    "mobility_score",
    "stability_weight",
    "core_label",
    "flexible_segment_id",
]

CORE_NONCORE_FIELDS = [
    "system",
    "system_key",
    "chain",
    "residue_id",
    "resname",
    "core_group",
    "robust_aligned_RMSF",
    "mobility_score",
    "stability_weight",
    "flexible_segment_id",
]

STATISTICS_FIELDS = [
    "system",
    "system_key",
    "n_residues",
    "n_chains",
    "n_flexible_segments",
    "spearman_robust_RMSF_vs_mobility",
    "spearman_ordinary_RMSF_vs_mobility",
    "mean_robust_RMSF_core",
    "mean_robust_RMSF_noncore",
    "median_robust_RMSF_core",
    "median_robust_RMSF_noncore",
    "delta_mean_noncore_minus_core",
    "delta_median_noncore_minus_core",
]


@dataclass(frozen=True)
class SystemProfile:
    """One loaded flexibility profile with normalized numeric fields."""

    system_key: str
    system_label: str
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class ProfilePanel:
    """One plotting panel, usually one system-chain pair."""

    system_key: str
    system_label: str
    chain: str
    panel_label: str
    rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results" / "flexibility",
        help="Directory containing formal flexibility-profile inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "flexibility" / "figures",
        help="Directory for PDF/PNG figure outputs.",
    )
    parser.add_argument(
        "--figure-data-dir",
        type=Path,
        default=ROOT / "results" / "flexibility" / "figure_data",
        help="Directory for plot-ready figure data tables.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "results" / "flexibility" / "flexibility_summary_four_systems.tsv",
        help="Optional formal flexibility summary table.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=ROOT / "results" / "flexibility" / "summary.json",
        help="Optional formal run metadata JSON.",
    )
    parser.add_argument(
        "--profile-files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit profile files for testing or custom inputs.",
    )
    parser.add_argument(
        "--system-labels",
        nargs="*",
        default=None,
        help="Optional labels matching --profile-files.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def read_table(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a CSV/TSV table into dictionaries."""

    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str], *, delimiter: str = "\t") -> None:
    """Write a table with a stable schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_profiles(
    *,
    profile_files: list[Path] | None = None,
    system_labels: list[str] | None = None,
) -> list[SystemProfile]:
    """Load formal or custom flexibility profile files."""

    if profile_files is None:
        entries = DEFAULT_PROFILES
    else:
        if system_labels is not None and len(system_labels) != len(profile_files):
            raise ValueError("--system-labels must match --profile-files length")
        entries = []
        for index, path in enumerate(profile_files):
            key = path.stem.replace("_flexibility_profile", "")
            label = system_labels[index] if system_labels is not None else key.replace("_", " ").title()
            entries.append((key, label, path))

    profiles: list[SystemProfile] = []
    for key, label, path in entries:
        rows = normalize_profile_rows(read_table(path, delimiter=","))
        profiles.append(SystemProfile(system_key=key, system_label=label, rows=rows))
    return profiles


def normalize_profile_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Validate and normalize one flexibility profile table."""

    if not rows:
        raise ValueError("profile table is empty")
    missing = [column for column in PROFILE_REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"profile table is missing required columns: {missing}")

    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        normalized.append(
            {
                "residue_id": row["residue_id"],
                "resname": row["resname"],
                "chain": row["chain"] or "_",
                "robust_aligned_RMSF": float(row["robust_aligned_RMSF"]),
                "ordinary_aligned_RMSF": float(row["ordinary_aligned_RMSF"]),
                "soft_stability_weight": float(row["soft_stability_weight"]),
                "mobility_score": float(row["mobility_score"]),
                "flexible_segment_id": int(float(row["flexible_segment_id"])),
                "rank": int(float(row["rank"])),
                "percentile": float(row["percentile"]),
                "core_label": int(float(row["core_label"])),
                "is_flexible": int(float(row.get("is_flexible", 1 if int(float(row["flexible_segment_id"])) > 0 else 0))),
                "is_terminal": int(float(row.get("is_terminal", 0))),
                "input_order": index,
            }
        )
    return normalized


def split_profile_panels(profiles: list[SystemProfile]) -> list[ProfilePanel]:
    """Split profiles by chain so multi-chain systems are not visually merged."""

    panels: list[ProfilePanel] = []
    for profile in profiles:
        chain_order = list(dict.fromkeys(str(row["chain"]) for row in profile.rows))
        multi_chain = len(chain_order) > 1
        for chain in chain_order:
            chain_rows = [row for row in profile.rows if str(row["chain"]) == chain]
            panel_label = f"{profile.system_label} chain {chain}" if multi_chain else profile.system_label
            panels.append(
                ProfilePanel(
                    system_key=profile.system_key,
                    system_label=profile.system_label,
                    chain=chain,
                    panel_label=panel_label,
                    rows=chain_rows,
                )
            )
    return panels


def flexible_segment_intervals(panel: ProfilePanel) -> list[dict[str, object]]:
    """Return shading intervals for contiguous flexible segments within one panel."""

    intervals: list[dict[str, object]] = []
    rows = panel.rows
    start_index: int | None = None
    current_segment: int | None = None
    for position, row in enumerate(rows, start=1):
        segment_id = int(row["flexible_segment_id"])
        if segment_id <= 0:
            if current_segment is not None and start_index is not None:
                intervals.append(_build_interval(panel, current_segment, start_index, position - 1))
            current_segment = None
            start_index = None
            continue
        if current_segment is None:
            current_segment = segment_id
            start_index = position
        elif segment_id != current_segment:
            intervals.append(_build_interval(panel, current_segment, start_index, position - 1))
            current_segment = segment_id
            start_index = position
    if current_segment is not None and start_index is not None:
        intervals.append(_build_interval(panel, current_segment, start_index, len(rows)))
    return intervals


def _build_interval(panel: ProfilePanel, segment_id: int, start_position: int, end_position: int) -> dict[str, object]:
    """Build one interval record from panel-local positions."""

    start_row = panel.rows[start_position - 1]
    end_row = panel.rows[end_position - 1]
    return {
        "system": panel.system_label,
        "system_key": panel.system_key,
        "panel_label": panel.panel_label,
        "chain": panel.chain,
        "segment_id": segment_id,
        "start_position": start_position,
        "end_position": end_position,
        "start_residue": start_row["residue_id"],
        "end_residue": end_row["residue_id"],
    }


def build_rmsf_figure_rows(profiles: list[SystemProfile]) -> list[dict[str, object]]:
    """Build plot-ready long-format rows for Figure 1."""

    rows: list[dict[str, object]] = []
    for panel in split_profile_panels(profiles):
        for position, row in enumerate(panel.rows, start=1):
            for alignment, rmsf_key in (
                ("ordinary", "ordinary_aligned_RMSF"),
                ("robust", "robust_aligned_RMSF"),
            ):
                rows.append(
                    {
                        "system": panel.system_label,
                        "system_key": panel.system_key,
                        "panel_label": panel.panel_label,
                        "chain": panel.chain,
                        "residue_id": row["residue_id"],
                        "resname": row["resname"],
                        "residue_order": position,
                        "plot_position": position,
                        "alignment": alignment,
                        "rmsf": row[rmsf_key],
                        "mobility_score": row["mobility_score"],
                        "stability_weight": row["soft_stability_weight"],
                        "flexible_segment_id": row["flexible_segment_id"],
                        "core_label": row["core_label"],
                        "is_flexible": row["is_flexible"],
                        "is_terminal": row["is_terminal"],
                    }
                )
    return rows


def build_mobility_figure_rows(profiles: list[SystemProfile]) -> list[dict[str, object]]:
    """Build Figure 2 scatter data."""

    rows: list[dict[str, object]] = []
    for profile in profiles:
        for row in profile.rows:
            rows.append(
                {
                    "system": profile.system_label,
                    "system_key": profile.system_key,
                    "chain": row["chain"],
                    "residue_id": row["residue_id"],
                    "resname": row["resname"],
                    "robust_aligned_RMSF": row["robust_aligned_RMSF"],
                    "ordinary_aligned_RMSF": row["ordinary_aligned_RMSF"],
                    "mobility_score": row["mobility_score"],
                    "stability_weight": row["soft_stability_weight"],
                    "core_label": row["core_label"],
                    "flexible_segment_id": row["flexible_segment_id"],
                }
            )
    return rows


def build_core_noncore_rows(profiles: list[SystemProfile]) -> list[dict[str, object]]:
    """Build Figure 3 core/non-core RMSF data."""

    rows: list[dict[str, object]] = []
    for profile in profiles:
        for row in profile.rows:
            core_label = int(row["core_label"])
            rows.append(
                {
                    "system": profile.system_label,
                    "system_key": profile.system_key,
                    "chain": row["chain"],
                    "residue_id": row["residue_id"],
                    "resname": row["resname"],
                    "core_group": "core" if core_label == 1 else "noncore",
                    "robust_aligned_RMSF": row["robust_aligned_RMSF"],
                    "mobility_score": row["mobility_score"],
                    "stability_weight": row["soft_stability_weight"],
                    "flexible_segment_id": row["flexible_segment_id"],
                }
            )
    return rows


def build_statistics_rows(profiles: list[SystemProfile]) -> list[dict[str, object]]:
    """Build per-system statistics for figure annotations and reporting."""

    rows: list[dict[str, object]] = []
    for profile in profiles:
        robust = np.asarray([float(row["robust_aligned_RMSF"]) for row in profile.rows], dtype=float)
        ordinary = np.asarray([float(row["ordinary_aligned_RMSF"]) for row in profile.rows], dtype=float)
        mobility = np.asarray([float(row["mobility_score"]) for row in profile.rows], dtype=float)
        core_mask = np.asarray([int(row["core_label"]) == 1 for row in profile.rows], dtype=bool)
        noncore_mask = ~core_mask
        chain_count = len(set(str(row["chain"]) for row in profile.rows))
        segment_ids = {int(row["flexible_segment_id"]) for row in profile.rows if int(row["flexible_segment_id"]) > 0}
        rows.append(
            {
                "system": profile.system_label,
                "system_key": profile.system_key,
                "n_residues": len(profile.rows),
                "n_chains": chain_count,
                "n_flexible_segments": len(segment_ids),
                "spearman_robust_RMSF_vs_mobility": spearman_correlation(robust, mobility),
                "spearman_ordinary_RMSF_vs_mobility": spearman_correlation(ordinary, mobility),
                "mean_robust_RMSF_core": mean_or_blank(robust[core_mask]),
                "mean_robust_RMSF_noncore": mean_or_blank(robust[noncore_mask]),
                "median_robust_RMSF_core": median_or_blank(robust[core_mask]),
                "median_robust_RMSF_noncore": median_or_blank(robust[noncore_mask]),
                "delta_mean_noncore_minus_core": delta_mean(robust, core_mask, noncore_mask),
                "delta_median_noncore_minus_core": delta_median(robust, core_mask, noncore_mask),
            }
        )
    return rows


def mean_or_blank(values: np.ndarray) -> float | str:
    """Return the mean or an empty value for empty arrays."""

    if values.size == 0:
        return ""
    return float(np.mean(values))


def median_or_blank(values: np.ndarray) -> float | str:
    """Return the median or an empty value for empty arrays."""

    if values.size == 0:
        return ""
    return float(np.median(values))


def delta_mean(values: np.ndarray, core_mask: np.ndarray, noncore_mask: np.ndarray) -> float | str:
    """Return mean(noncore) - mean(core)."""

    if not np.any(core_mask) or not np.any(noncore_mask):
        return ""
    return float(np.mean(values[noncore_mask]) - np.mean(values[core_mask]))


def delta_median(values: np.ndarray, core_mask: np.ndarray, noncore_mask: np.ndarray) -> float | str:
    """Return median(noncore) - median(core)."""

    if not np.any(core_mask) or not np.any(noncore_mask):
        return ""
    return float(np.median(values[noncore_mask]) - np.median(values[core_mask]))


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | str:
    """Compute Spearman correlation without SciPy."""

    if x.size < 2 or y.size < 2:
        return ""
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator == 0.0:
        return ""
    return float(np.sum(x_centered * y_centered) / denominator)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and np.isclose(values[order[end]], values[order[start]]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def configure_matplotlib() -> None:
    """Configure compact publication-friendly figure defaults."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def plot_rmsf_profiles(profiles: list[SystemProfile], output_pdf: Path, output_png: Path, *, dpi: int) -> None:
    """Plot ordinary-vs-robust RMSF curves with flexible segment shading."""

    panels = split_profile_panels(profiles)
    n_panels = len(panels)
    fig = plt.figure(figsize=(8.4, 1.9 * n_panels), constrained_layout=False)
    ratios: list[float] = []
    for _ in panels:
        ratios.extend([3.0, 0.55])
    grid = GridSpec(n_panels * 2, 1, figure=fig, height_ratios=ratios, hspace=0.08)

    ordinary_color = "#7f7f7f"
    robust_color = "#005f73"
    mobility_color = "#ca6702"
    shade_color = "#f2cc8f"

    legend_handles = None
    for panel_index, panel in enumerate(panels):
        main_axis = fig.add_subplot(grid[2 * panel_index, 0])
        track_axis = fig.add_subplot(grid[2 * panel_index + 1, 0], sharex=main_axis)

        x = np.arange(1, len(panel.rows) + 1, dtype=float)
        ordinary = np.asarray([row["ordinary_aligned_RMSF"] for row in panel.rows], dtype=float)
        robust = np.asarray([row["robust_aligned_RMSF"] for row in panel.rows], dtype=float)
        mobility = np.asarray([row["mobility_score"] for row in panel.rows], dtype=float)

        for interval in flexible_segment_intervals(panel):
            start = float(interval["start_position"]) - 0.5
            end = float(interval["end_position"]) + 0.5
            main_axis.axvspan(start, end, color=shade_color, alpha=0.32, linewidth=0)
            track_axis.axvspan(start, end, color=shade_color, alpha=0.32, linewidth=0)

        ordinary_line = main_axis.plot(x, ordinary, color=ordinary_color, linewidth=1.15, label="Ordinary-fit RMSF")
        robust_line = main_axis.plot(x, robust, color=robust_color, linewidth=1.25, label="Robust-fit RMSF")
        main_axis.set_ylabel("RMSF")
        main_axis.text(
            0.0,
            0.98,
            panel.panel_label,
            transform=main_axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )
        main_axis.spines["top"].set_visible(False)
        main_axis.spines["right"].set_visible(False)
        main_axis.grid(axis="y", linestyle=":", alpha=0.32)

        track_axis.fill_between(x, 0.0, mobility, color=mobility_color, alpha=0.55, linewidth=0)
        track_axis.plot(x, mobility, color=mobility_color, linewidth=0.8)
        track_axis.set_ylim(0.0, 1.0)
        track_axis.set_ylabel("Mob.", rotation=0, labelpad=17, va="center")
        track_axis.spines["top"].set_visible(False)
        track_axis.spines["right"].set_visible(False)
        track_axis.grid(axis="y", linestyle=":", alpha=0.2)
        if panel_index < n_panels - 1:
            plt.setp(track_axis.get_xticklabels(), visible=False)
        else:
            track_axis.set_xlabel("Residue order within chain")
        track_axis.set_xlim(0.5, len(panel.rows) + 0.5)

        annotate_residue_range(track_axis, panel)
        if legend_handles is None:
            legend_handles = [ordinary_line[0], robust_line[0]]

    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            labels=["Ordinary-fit RMSF", "Robust-fit RMSF"],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.998),
        )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.05, top=0.955, hspace=0.28)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def annotate_residue_range(axis: plt.Axes, panel: ProfilePanel) -> None:
    """Mark first/last residue IDs so residue identity comes from metadata."""

    if not panel.rows:
        return
    left = panel.rows[0]
    right = panel.rows[-1]
    axis.text(
        0.01,
        -0.65,
        f"{left['resname']}{left['residue_id']}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#3a3a3a",
    )
    axis.text(
        0.99,
        -0.65,
        f"{right['resname']}{right['residue_id']}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#3a3a3a",
    )


def plot_mobility_vs_rmsf(profiles: list[SystemProfile], statistics_rows: list[dict[str, object]], output_pdf: Path) -> None:
    """Plot mobility score against robust RMSF, faceted by system."""

    n = len(profiles)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 3.25 * n_rows), squeeze=False)
    stat_map = {row["system_key"]: row for row in statistics_rows}

    for axis, profile in zip(axes.flat, profiles, strict=False):
        mobility = np.asarray([row["mobility_score"] for row in profile.rows], dtype=float)
        robust = np.asarray([row["robust_aligned_RMSF"] for row in profile.rows], dtype=float)
        core = np.asarray([int(row["core_label"]) for row in profile.rows], dtype=int)
        colors = np.where(core == 1, "#2a9d8f", "#e76f51")
        axis.scatter(mobility, robust, c=colors, s=14, alpha=0.72, edgecolors="none")
        stat = stat_map.get(profile.system_key, {})
        rho = stat.get("spearman_robust_RMSF_vs_mobility", "")
        rho_text = "" if rho == "" else f"Spearman r = {float(rho):.3f}"
        axis.text(0.04, 0.94, rho_text, transform=axis.transAxes, ha="left", va="top")
        axis.set_title(profile.system_label, loc="left", fontweight="bold")
        axis.set_xlabel("Mobility score")
        axis.set_ylabel("Robust-aligned RMSF")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(linestyle=":", alpha=0.28)
    for axis in axes.flat[len(profiles):]:
        axis.axis("off")
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_core_vs_noncore(profiles: list[SystemProfile], output_pdf: Path) -> None:
    """Compare robust RMSF between core and non-core residues."""

    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    positions: list[float] = []
    labels: list[str] = []
    data: list[np.ndarray] = []
    colors: list[str] = []
    for system_index, profile in enumerate(profiles):
        base = system_index * 3.0
        for offset, group, color in ((0.0, "core", "#2a9d8f"), (0.9, "noncore", "#e76f51")):
            values = np.asarray(
                [
                    row["robust_aligned_RMSF"]
                    for row in profile.rows
                    if ("core" if int(row["core_label"]) == 1 else "noncore") == group
                ],
                dtype=float,
            )
            positions.append(base + offset)
            labels.append(group)
            data.append(values)
            colors.append(color)

    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)
        patch.set_edgecolor("black")

    rng = np.random.default_rng(42)
    for position, values, color in zip(positions, data, colors, strict=True):
        if values.size == 0:
            continue
        sample = values if values.size <= 180 else rng.choice(values, size=180, replace=False)
        jitter = rng.uniform(-0.16, 0.16, size=sample.shape[0])
        axis.scatter(np.full(sample.shape[0], position) + jitter, sample, s=7, color=color, alpha=0.36, edgecolors="none")
        axis.scatter([position], [np.mean(values)], marker="D", s=24, color="white", edgecolors="black", linewidths=0.7, zorder=4)

    system_centers = [index * 3.0 + 0.45 for index in range(len(profiles))]
    axis.set_xticks(system_centers)
    axis.set_xticklabels([profile.system_label for profile in profiles], rotation=20, ha="right")
    axis.set_ylabel("Robust-aligned RMSF")
    axis.set_title("Core residues have lower robust-aligned RMSF than non-core residues", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", linestyle=":", alpha=0.32)
    axis.text(0.01, 0.98, "Diamond = mean; box median/IQR", transform=axis.transAxes, ha="left", va="top")
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_caption_draft(path: Path) -> None:
    """Write draft figure captions for manuscript planning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Draft Captions for Formal Flexibility Figures",
                "",
                "## Figure 1. Robust-core alignment changes flexibility profiles across four systems.",
                "",
                "Ordinary-fit and soft-core robust-fit RMSF profiles are shown for I-FABP, ADK, NhaA, and YiiP short. NhaA and YiiP short are split by chain to avoid visually merging independent chains. Shaded regions denote automatically detected flexible segments, and the lower track shows the soft mobility score derived from the learned stability weights.",
                "",
                "## Figure 2. Soft mobility scores are associated with robust-aligned RMSF.",
                "",
                "Each point is one residue. The x-axis shows the model-derived mobility score and the y-axis shows robust-aligned RMSF. Spearman correlations are annotated per system. Core and non-core residues are colored separately.",
                "",
                "## Figure 3. Non-core residues show higher robust-aligned RMSF.",
                "",
                "Robust-aligned RMSF distributions are compared between soft-core and non-core residues for each system. Diamonds show group means and boxes show median/interquartile range. The underlying plot-ready table is saved in `figure_data/fig_core_noncore_rmsf.tsv`.",
                "",
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run figure-data generation and plotting."""

    args = parse_args() if argv is None else parse_args_from_list(argv)
    configure_matplotlib()

    profiles = load_profiles(profile_files=args.profile_files, system_labels=args.system_labels)
    rmsf_rows = build_rmsf_figure_rows(profiles)
    mobility_rows = build_mobility_figure_rows(profiles)
    core_rows = build_core_noncore_rows(profiles)
    statistics_rows = build_statistics_rows(profiles)

    args.figure_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_table(args.figure_data_dir / "fig_rmsf_profiles.tsv", rmsf_rows, RMSF_FIGURE_FIELDS)
    write_table(args.figure_data_dir / "fig_mobility_vs_rmsf.tsv", mobility_rows, MOBILITY_FIGURE_FIELDS)
    write_table(args.figure_data_dir / "fig_core_noncore_rmsf.tsv", core_rows, CORE_NONCORE_FIELDS)
    write_table(args.input_dir / "flexibility_statistics_four_systems.tsv", statistics_rows, STATISTICS_FIELDS)

    plot_rmsf_profiles(
        profiles,
        args.output_dir / "fig_rmsf_profiles_four_systems.pdf",
        args.output_dir / "fig_rmsf_profiles_four_systems.png",
        dpi=args.dpi,
    )
    plot_mobility_vs_rmsf(profiles, statistics_rows, args.output_dir / "fig_mobility_vs_robust_rmsf.pdf")
    plot_core_vs_noncore(profiles, args.output_dir / "fig_core_vs_noncore_rmsf.pdf")
    write_caption_draft(args.output_dir / "figure_captions_draft.md")

    print(f"Wrote figures to {args.output_dir}")
    print(f"Wrote figure data to {args.figure_data_dir}")
    return 0


def parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    """Parse an explicit argv list for tests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "flexibility")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "flexibility" / "figures")
    parser.add_argument("--figure-data-dir", type=Path, default=ROOT / "results" / "flexibility" / "figure_data")
    parser.add_argument("--summary-path", type=Path, default=ROOT / "results" / "flexibility" / "flexibility_summary_four_systems.tsv")
    parser.add_argument("--metadata-json", type=Path, default=ROOT / "results" / "flexibility" / "summary.json")
    parser.add_argument("--profile-files", type=Path, nargs="*", default=None)
    parser.add_argument("--system-labels", nargs="*", default=None)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
