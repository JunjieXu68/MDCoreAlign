"""Compute per-frame residual time series after ordinary and robust alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from md_trajectory_focus_common import (  # noqa: E402
    FORMAL_SYSTEMS,
    MD_FOCUS_ROOT,
    align_trajectory_ordinary,
    common_manifest,
    load_formal_system,
    residual_distances,
    solve_default_soft,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "per_frame_residuals"

FIELDS = [
    "system",
    "frame_index",
    "time_ps_if_available",
    "ordinary_frame_RMSD",
    "robust_frame_unified_RMSD",
    "robust_frame_weighted_RMSD",
    "mean_core_residual",
    "mean_noncore_residual",
    "max_residual",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def rmsd_from_distances(distances: np.ndarray) -> float:
    """Return RMSD from per-residue distances."""

    return float(np.sqrt(np.mean(np.asarray(distances, dtype=float) ** 2)))


def run_analysis(systems: list[str]) -> list[dict[str, object]]:
    """Run per-frame residual analysis."""

    rows: list[dict[str, object]] = []
    for key in systems:
        system = load_formal_system(key)
        reference = system.trajectory.coords[0]
        ordinary = align_trajectory_ordinary(system.trajectory)
        robust = solve_default_soft(system)
        core = robust.core_mask
        for frame in robust.trajectory_result.frame_results:
            index = frame.frame_index
            ordinary_dist = residual_distances(ordinary[index], reference)
            robust_dist = residual_distances(frame.result.aligned_coords, reference)
            rows.append(
                {
                    "system": system.label,
                    "frame_index": index,
                    "time_ps_if_available": "",
                    "ordinary_frame_RMSD": rmsd_from_distances(ordinary_dist),
                    "robust_frame_unified_RMSD": rmsd_from_distances(robust_dist),
                    "robust_frame_weighted_RMSD": frame.result.weighted_rmsd,
                    "mean_core_residual": float(np.mean(robust_dist[core])) if np.any(core) else "",
                    "mean_noncore_residual": float(np.mean(robust_dist[~core])) if np.any(~core) else "",
                    "max_residual": float(np.max(robust_dist)),
                    "notes": "time unavailable from formal in-memory trajectory; frame_index is over formal loaded frames",
                }
            )
        print(f"[per-frame] {system.label}")
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    deltas = [float(row["ordinary_frame_RMSD"]) - float(row["robust_frame_unified_RMSD"]) for row in rows]
    high_distortion = [row for row, delta in zip(rows, deltas, strict=True) if delta > 0.5]
    core_gap = [
        float(row["mean_noncore_residual"]) - float(row["mean_core_residual"])
        for row in rows
        if row["mean_noncore_residual"] != "" and row["mean_core_residual"] != ""
    ]
    lines = [
        "# MD Per-Frame Residual Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Are there transient periods where ordinary fitting is strongly distorted? `{len(high_distortion)}` frame rows have ordinary RMSD more than 0.5 above robust unified RMSD.",
        f"2. Does robust alignment stabilize per-frame residual behavior? Mean ordinary-minus-robust frame RMSD difference is `{float(np.mean(deltas)):.4g}`.",
        f"3. Do high-residual periods correspond to mobile/flexible segments? Mean noncore-minus-core residual gap is `{float(np.mean(core_gap)):.4g}` when core/noncore are both defined.",
        "4. Are there trajectory segments that should be excluded as non-equilibrated? This table flags residual transients, but exclusion requires biological/provenance review rather than an automatic decision.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run analysis and write outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args.systems)
    write_table(args.output_dir / "per_frame_residuals.tsv", rows, FIELDS)
    write_table(args.output_dir / "per_frame_residuals_plot_data.tsv", rows, FIELDS)
    (args.output_dir / "per_frame_residuals_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_per_frame_residuals.py",
        output_dir=args.output_dir,
        systems=args.systems,
        output_files=[
            args.output_dir / "per_frame_residuals.tsv",
            args.output_dir / "per_frame_residuals_summary.md",
            args.output_dir / "per_frame_residuals_plot_data.tsv",
        ],
    )
    print(f"Wrote per-frame residual outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
