#!/usr/bin/env python3
"""Create conceptual and case-study figures from frozen CP inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/concept_cases_product_v1"
ASSETS = OUT / "05_manuscript_assets"
CASE_ROOT = OUT / "03_case_studies"

COLORS = {
    "ink": "#17324D",
    "exact": "#145A75",
    "throughput": "#D97706",
    "w5": "#2A7F62",
    "light": "#EAF0F3",
    "mobile": "#C23B22",
    "stable": "#2878B5",
    "ordinary": "#777777",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _box(ax, xy, width, height, text, color, fontsize=9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])


def conceptual_figure() -> None:
    """Draw the two-stage method and complementary W5 branch."""

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.03, 0.66), 0.15, 0.14, "Protein\ntrajectory", COLORS["ink"], 11)
    _box(ax, (0.26, 0.76), 0.22, 0.12, "Temporally stratified\nrepresentative frames", COLORS["ink"])
    _box(ax, (0.56, 0.82), 0.18, 0.10, "Exact sequential\nweight estimation", COLORS["exact"])
    _box(ax, (0.56, 0.66), 0.18, 0.10, "Throughput shared-weight\nestimation (CPU / GPU)", COLORS["throughput"])
    _box(ax, (0.80, 0.72), 0.16, 0.14, "One frozen\nstability-weight\nvector", COLORS["ink"])
    _box(ax, (0.63, 0.43), 0.26, 0.13, "Explicit weighted rigid fit\nfor every frame", COLORS["ink"], 11)
    _box(ax, (0.57, 0.18), 0.38, 0.14, "Transforms + RMSD + robust/ordinary RMSF\n+ optional aligned coordinates", COLORS["ink"])
    _box(ax, (0.25, 0.30), 0.23, 0.13, "W5 local deformation\n(alignment-free)", COLORS["w5"], 11)
    _box(ax, (0.04, 0.09), 0.39, 0.12, "Local deformation index + local rigidity\n+ segments + temporal reproducibility", COLORS["w5"])
    arrows = [
        ((0.18, 0.73), (0.26, 0.82)),
        ((0.48, 0.82), (0.56, 0.87)),
        ((0.48, 0.82), (0.56, 0.71)),
        ((0.74, 0.87), (0.80, 0.81)),
        ((0.74, 0.71), (0.80, 0.77)),
        ((0.88, 0.72), (0.76, 0.56)),
        ((0.76, 0.43), (0.76, 0.32)),
        ((0.10, 0.66), (0.30, 0.43)),
        ((0.36, 0.30), (0.28, 0.21)),
    ]
    for start, stop in arrows:
        ax.annotate("", xy=stop, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": COLORS["ink"]})
    ax.text(0.51, 0.96, "Weight estimation is decoupled from exact all-frame fitting", ha="center", fontsize=15, weight="bold", color=COLORS["ink"])
    ax.text(0.08, 0.47, "Parallel diagnostic branch", fontsize=9, color=COLORS["w5"], weight="bold")
    ax.text(0.75, 0.59, "same fitting stage", fontsize=8, color=COLORS["ink"], ha="center")
    fig.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(ASSETS / f"fig1_method_concept.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _segments(
    values: np.ndarray,
    chains: np.ndarray,
    mask: np.ndarray,
    *,
    merge_gap: int = 0,
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for chain in dict.fromkeys(chains.tolist()):
        indices = np.flatnonzero((chains == chain) & mask)
        if not indices.size:
            continue
        group = [int(indices[0])]
        for index in indices[1:]:
            if int(index) - group[-1] <= merge_gap + 1:
                group.append(int(index))
            else:
                result.append(np.asarray(group))
                group = [int(index)]
        result.append(np.asarray(group))
    return result


def _region_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    chains = np.asarray([row["chain"] for row in rows])
    global_score = np.asarray([float(row["global_displacement_score"]) for row in rows])
    local_score = np.asarray([float(row["local_deformation_score"]) for row in rows])
    weights = np.asarray([float(row["exact_stable_weight"]) for row in rows])
    definitions = {
        "stable_anchor": ((weights >= 0.70) & (global_score <= 0.30), 3, 1),
        "coherently_displaced": ((global_score >= 0.70) & (local_score <= 0.30), 8, 2),
        "locally_deforming": (local_score >= 0.70, 3, 1),
        "putative_transition": ((global_score >= 0.50) & (local_score >= 0.50), 3, 1),
    }
    output: list[dict[str, Any]] = []
    segment_id = 0
    for region_type, (mask, minimum_length, merge_gap) in definitions.items():
        for segment in _segments(
            global_score, chains, mask, merge_gap=merge_gap
        ):
            if segment.size < minimum_length:
                continue
            segment_id += 1
            first, last = rows[int(segment[0])], rows[int(segment[-1])]
            output.append(
                {
                    "region_id": segment_id,
                    "region_type": region_type,
                    "chain": first["chain"],
                    "start_residue_id": first["residue_id"],
                    "end_residue_id": last["residue_id"],
                    "start_residue_index": int(segment[0]) + 1,
                    "end_residue_index": int(segment[-1]) + 1,
                    "length": int(segment.size),
                    "mean_global_displacement_score": float(np.mean(global_score[segment])),
                    "mean_local_deformation_score": float(np.mean(local_score[segment])),
                    "mean_stable_weight": float(np.mean(weights[segment])),
                    "interpretation_scope": "geometric_only_no_biological_domain_annotation",
                }
            )
    return output


def _load_models(path: Path) -> np.ndarray:
    frames: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL") and current:
            frames.append(current)
            current = []
    return np.asarray(frames)


def case_figure(role: str, panel_letter: str) -> dict[str, Any]:
    """Draw one selected case from its immutable decomposition table."""

    case_dir = CASE_ROOT / role
    rows = _rows(case_dir / "residue_decomposition.tsv")
    summary = json.loads((case_dir / "case_summary.json").read_text())
    positions = np.arange(1, len(rows) + 1)
    weights = np.asarray([float(row["exact_stable_weight"]) for row in rows])
    throughput_weights = np.asarray([float(row["throughput_stable_weight"]) for row in rows])
    robust = np.asarray([float(row["robust_RMSF_A"]) for row in rows])
    ordinary = np.asarray([float(row["ordinary_RMSF_A"]) for row in rows])
    global_score = np.asarray([float(row["global_displacement_score"]) for row in rows])
    local = np.asarray([float(row["local_deformation_score"]) for row in rows])
    confidence = np.asarray([float(row["local_deformation_confidence"]) for row in rows])
    models = _load_models(case_dir / "representative_stable_core_aligned_CA.pdb")
    regions = _region_rows(rows)
    _write_tsv(case_dir / "region_summary.tsv", regions)
    key_regions: list[dict[str, Any]] = []
    for region_type in (
        "stable_anchor",
        "coherently_displaced",
        "locally_deforming",
        "putative_transition",
    ):
        matching = [row for row in regions if row["region_type"] == region_type]
        matching.sort(
            key=lambda row: (
                -int(row["length"]),
                -float(row["mean_global_displacement_score"]),
                int(row["start_residue_index"]),
            )
        )
        key_regions.extend(matching[:3])
    if key_regions:
        _write_tsv(case_dir / "key_region_summary.tsv", key_regions)

    fig = plt.figure(figsize=(12, 8.2))
    grid = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.42)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    for frame, alpha, width in ((models[0], 0.45, 0.8), (models[-1], 0.95, 1.3)):
        ax.plot(frame[:, 0], frame[:, 1], frame[:, 2], color=COLORS["ink"], alpha=alpha, lw=width)
    points = models[-1]
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=weights, cmap="viridis", s=6, vmin=0, vmax=1)
    ax.set_title("a  Aligned conformations", loc="left", weight="bold")
    ax.set_axis_off()
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.030, pad=0.015)
    colorbar.ax.set_title("Stable\nweight", fontsize=7, pad=5)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(positions, weights, color=COLORS["exact"], lw=1.4, label="Exact")
    ax.plot(positions, throughput_weights, color=COLORS["throughput"], lw=1.0, alpha=0.8, label="Throughput")
    ax.axhline(0.7, color="#AAAAAA", ls="--", lw=0.8)
    ax.set(title="b  Frozen stability weights", xlabel="Residue index", ylabel="Weight")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    ax.plot(positions, ordinary, color=COLORS["ordinary"], lw=1.0, label="Ordinary RMSF")
    ax.plot(positions, robust, color=COLORS["mobile"], lw=1.25, label="Robust RMSF")
    ax.set(title="c  Alignment-dependent flexibility", xlabel="Residue index", ylabel=r"RMSF ($\AA$)")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(positions, local, color=COLORS["w5"], lw=1.25)
    ax.fill_between(positions, 0, local, color=COLORS["w5"], alpha=0.15)
    ax.plot(positions, confidence, color=COLORS["ink"], lw=0.7, alpha=0.65, label="Confidence")
    ax.set(title="d  Alignment-free local deformation", xlabel="Residue index", ylabel="Dimensionless score")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    ax.scatter(global_score, local, c=weights, cmap="coolwarm_r", s=12, alpha=0.75, vmin=0, vmax=1)
    ax.axvline(0.5, color="#BBBBBB", lw=0.7)
    ax.axhline(0.5, color="#BBBBBB", lw=0.7)
    ax.text(0.03, 0.96, "local strain", va="top", fontsize=7)
    ax.text(0.97, 0.96, "global + local", va="top", ha="right", fontsize=7)
    ax.text(0.03, 0.04, "stable", va="bottom", fontsize=7)
    ax.text(0.97, 0.04, "coherent displacement", va="bottom", ha="right", fontsize=7)
    ax.set(title="e  Joint interpretation", xlabel="Global displacement", ylabel="Local deformation", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))

    ax = fig.add_subplot(grid[1, 2])
    class_codes = np.select(
        [(global_score >= 0.5) & (local >= 0.5), global_score >= 0.5, local >= 0.5],
        [3, 2, 1],
        default=0,
    )
    ax.imshow(class_codes[None, :], aspect="auto", interpolation="nearest", cmap=plt.matplotlib.colors.ListedColormap(["#DCE7EC", "#63B995", "#F2A65A", "#D1495B"]), vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_xlabel("Residue index")
    ax.set_title("f  Interpretation track", loc="left", weight="bold")
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=8, label=label) for color, label in zip(["#DCE7EC", "#63B995", "#F2A65A", "#D1495B"], ["low/low", "local deformation", "coherent displacement", "high/high"], strict=True)]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    fig.suptitle(f"{panel_letter}. {summary['system']}: geometric decomposition of global displacement and local deformation", fontsize=14, weight="bold", color=COLORS["ink"])
    for axis in fig.axes:
        if hasattr(axis, "spines") and axis.name != "3d":
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    stem = f"fig{2 if role == 'case_A' else 3}_{role}_{summary['system']}"
    for suffix in ("pdf", "png"):
        fig.savefig(ASSETS / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "role": role,
        "system": summary["system"],
        "n_CA": summary["n_CA"],
        "n_frames": summary["n_frames"],
        "mean_exact_throughput_weight_abs_difference": float(np.mean(np.abs(weights - throughput_weights))),
        "max_exact_throughput_weight_abs_difference": float(np.max(np.abs(weights - throughput_weights))),
        "throughput_exact_core_Jaccard": summary["throughput_exact_core_Jaccard"],
        "throughput_exact_RMSF_Spearman": summary["throughput_exact_RMSF_Spearman"],
        "W5_temporal_reproducibility": summary["W5_temporal_reproducibility"],
        "mean_robust_RMSF_A": float(np.mean(robust)),
        "mean_ordinary_RMSF_A": float(np.mean(ordinary)),
        "n_geometric_regions": len(regions),
        "n_key_regions_reported": len(key_regions),
    }


def main() -> None:
    conceptual_figure()
    summaries = [case_figure("case_A", "Case A"), case_figure("case_B", "Case B")]
    _write_tsv(ASSETS / "case_study_summary.tsv", summaries)
    decision = json.loads((OUT / "02_case_selection/selection_decision.json").read_text())
    (ASSETS / "case_study_results_draft.md").write_text(
        "# Case-Study Results Draft\n\n"
        f"The prospectively locked geometric screen selected `{decision['case_A']}` as Case A and "
        f"`{decision['case_B']}` as Case B. Case A contains a stable anchor and contiguous "
        "high-global-displacement/low-local-deformation region with a boundary-local-deformation "
        "peak. Case B contains a localised W5 deformation peak with limited coherent global "
        "displacement. These are geometric interpretations from existing ATLAS R1 data; no "
        "biological domain or mechanism annotation is asserted. Exact and throughput consistency "
        "statistics are reported in `case_study_summary.tsv`.\n"
    )
    (ASSETS / "case_study_captions.md").write_text(
        "# Case-Study Figure Captions\n\n"
        "**Figure 1. MDCoreAlign conceptual workflow.** Stable-residue weights are estimated from "
        "temporally stratified representatives, after which every frame receives an explicit "
        "weighted rigid-body fit. W5 independently measures local internal deformation.\n\n"
        f"**Figure 2. Case A ({decision['case_A']}).** Stable-core-aligned geometry and residue "
        "profiles distinguish coherent global displacement from local internal deformation. "
        "Region labels are geometric and do not assert a biological domain mechanism.\n\n"
        f"**Figure 3. Case B ({decision['case_B']}).** A local-deformation example showing the "
        "joint interpretation of W5, robust/ordinary RMSF and frozen stability weights. The "
        "2,128-CA size is reported separately as scaling context.\n"
    )


if __name__ == "__main__":
    main()
