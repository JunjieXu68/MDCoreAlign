"""Benchmark trajectory mode with and without warm start on a smooth toy trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    Trajectory,
    benchmark_results_to_rows,
    compare_trajectory_warm_start,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)


def build_toy_trajectory() -> Trajectory:
    """Build a smooth MD-like toy trajectory with one flexible region."""

    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.5, 2.0, 0.0],
            [1.5, 2.0, 0.0],
            [2.5, 2.0, 0.0],
            [3.5, 2.0, 0.0],
        ],
        dtype=float,
    )
    frames = []
    for time_index in range(18):
        theta = 0.02 * time_index
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.03 * time_index, -0.015 * time_index, 0.0], dtype=float)
        frame = base @ rotation.T + translation
        frame[[8, 9, 10, 11]] += np.array([0.0, 0.15 * np.sin(0.35 * time_index), 0.0], dtype=float)
        frame[[10, 11]] += np.array([0.0, 0.18 * np.cos(0.25 * time_index), 0.0], dtype=float)
        frames.append(frame)
    return Trajectory(coords=np.asarray(frames, dtype=float))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the warm-start toy demo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-mode", choices=["first_frame", "previous_frame"], default="first_frame")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_benchmarks_test")
    parser.add_argument("--delta", type=float, default=0.16)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    """Run the warm-start comparison and save CSV and JSON outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = build_toy_trajectory()
    results, difference = compare_trajectory_warm_start(
        trajectory,
        reference_mode=args.reference_mode,
        core_fraction=0.5,
        rule="huber",
        delta=args.delta,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    rows = benchmark_results_to_rows(
        results,
        benchmark_mode="trajectory",
        dataset_name=f"toy_{args.reference_mode}",
        repeat_index=0,
        preprocessing_steps=[
            "source=synthetic_toy",
            f"reference_mode={args.reference_mode}",
            "warm_start_comparison=cold_vs_warm",
        ],
    )
    summary_rows = summarize_benchmark_rows(rows)

    prefix = args.output_dir / f"trajectory_warm_start_{args.reference_mode}"
    write_benchmark_csv(Path(f"{prefix}_runs.csv"), rows)
    write_benchmark_csv(Path(f"{prefix}_summary.csv"), summary_rows)
    write_benchmark_json(
        Path(f"{prefix}_comparison.json"),
        {
            "reference_mode": args.reference_mode,
            "results": rows,
            "summary_rows": summary_rows,
            "solution_difference": difference,
        },
    )

    print("Trajectory warm-start demo complete")
    for row in summary_rows:
        print(
            f"{row['method_name']}: avg_iterations={row['iteration_count_mean']:.3f}, "
            f"runtime_per_frame={row['runtime_seconds_mean'] / trajectory.n_frames:.6f}s, "
            f"total_runtime={row['runtime_seconds_mean']:.6f}s, rmsd={row['rmsd_mean']:.6f}"
        )
    print(f"solution_difference={difference}")
    print(f"summary_csv={prefix}_summary.csv")


if __name__ == "__main__":
    main()
