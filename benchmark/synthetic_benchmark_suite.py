"""Synthetic benchmark suite for internal methods and external MDLovoFit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    Trajectory,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    compare_methods_on_trajectory,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)


def _base_reference(n_atoms: int = 24) -> np.ndarray:
    """Build a simple CA-like backbone reference shape."""

    x = np.linspace(-3.0, 3.0, n_atoms)
    return np.column_stack([x, np.sin(0.8 * x), 0.5 * np.cos(0.5 * x)])


def _global_transform(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a gentle frame-dependent rigid transform."""

    angle = 0.03 * frame_index
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.array([0.04 * frame_index, -0.015 * frame_index, 0.01 * frame_index], dtype=float)
    return rotation, translation


def _rigid_frames(reference: np.ndarray, n_frames: int, rng: np.random.Generator, noise_scale: float) -> np.ndarray:
    """Generate a rigidly transformed noisy trajectory from one reference."""

    frames = []
    for frame_index in range(n_frames):
        rotation, translation = _global_transform(frame_index)
        frame = reference @ rotation.T + translation
        if noise_scale > 0.0:
            frame = frame + rng.normal(scale=noise_scale, size=frame.shape)
        frames.append(frame)
    return np.asarray(frames, dtype=float)


def build_clean_rigid_transform_only(seed: int) -> tuple[Trajectory, list[str]]:
    """Scenario A1: rigid transform only with small observational noise."""

    rng = np.random.default_rng(seed)
    reference = _base_reference()
    trajectory = Trajectory(coords=_rigid_frames(reference, n_frames=15, rng=rng, noise_scale=0.005))
    steps = [
        "scenario=clean_rigid_transform_only",
        "reference_shape=single_backbone_curve",
        "global_motion=rigid_transform_only",
        "noise_scale=0.005",
    ]
    return trajectory, steps


def build_rigid_transform_with_moderate_outliers(seed: int) -> tuple[Trajectory, list[str]]:
    """Scenario A2: rigid motion with a few moderately displaced atoms."""

    rng = np.random.default_rng(seed)
    reference = _base_reference()
    coords = _rigid_frames(reference, n_frames=15, rng=rng, noise_scale=0.008)
    outlier_indices = np.sort(rng.choice(reference.shape[0], size=4, replace=False))
    direction = rng.normal(size=(outlier_indices.shape[0], 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    for frame_index in range(coords.shape[0]):
        scale = 0.35 + 0.10 * np.sin(0.7 * frame_index)
        coords[frame_index, outlier_indices] += scale * direction
    steps = [
        "scenario=rigid_transform_moderate_outliers",
        f"outlier_indices={outlier_indices.tolist()}",
        "outlier_scale~0.35-0.45",
        "noise_scale=0.008",
    ]
    return Trajectory(coords=coords), steps


def build_rigid_transform_with_many_outliers(seed: int) -> tuple[Trajectory, list[str]]:
    """Scenario A3: rigid motion with a larger set of stronger outliers."""

    rng = np.random.default_rng(seed)
    reference = _base_reference()
    coords = _rigid_frames(reference, n_frames=15, rng=rng, noise_scale=0.010)
    outlier_indices = np.sort(rng.choice(reference.shape[0], size=9, replace=False))
    direction = rng.normal(size=(outlier_indices.shape[0], 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    for frame_index in range(coords.shape[0]):
        scale = 0.80 + 0.25 * np.cos(0.5 * frame_index)
        coords[frame_index, outlier_indices] += scale * direction
    steps = [
        "scenario=rigid_transform_many_outliers",
        f"outlier_indices={outlier_indices.tolist()}",
        "outlier_scale~0.55-1.05",
        "noise_scale=0.010",
    ]
    return Trajectory(coords=coords), steps


def build_two_quasi_rigid_domains(seed: int) -> tuple[Trajectory, list[str]]:
    """Scenario A4: two quasi-rigid domains connected by a moving region."""

    rng = np.random.default_rng(seed)
    reference = _base_reference(n_atoms=24)
    coords = _rigid_frames(reference, n_frames=15, rng=rng, noise_scale=0.006)

    domain_a = np.arange(0, 10)
    linker = np.arange(10, 14)
    domain_b = np.arange(14, 24)

    for frame_index in range(coords.shape[0]):
        hinge = coords[frame_index, 13]
        local_angle = 0.05 * frame_index
        local_rotation = np.array(
            [
                [np.cos(local_angle), 0.0, np.sin(local_angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(local_angle), 0.0, np.cos(local_angle)],
            ],
            dtype=float,
        )
        coords[frame_index, domain_b] = (coords[frame_index, domain_b] - hinge) @ local_rotation.T + hinge
        coords[frame_index, linker, 1] += np.linspace(0.0, 0.6, linker.shape[0]) * np.sin(0.8 * frame_index)
        coords[frame_index, linker, 2] += np.linspace(0.0, 0.4, linker.shape[0]) * np.cos(0.6 * frame_index)
        coords[frame_index, domain_a] += rng.normal(scale=0.003, size=(domain_a.shape[0], 3))
    steps = [
        "scenario=two_quasi_rigid_domains",
        f"domain_a={domain_a.tolist()}",
        f"linker={linker.tolist()}",
        f"domain_b={domain_b.tolist()}",
        "domain_b_has_additional_hinge_rotation",
        "linker_has_internal_motion",
    ]
    return Trajectory(coords=coords), steps


SCENARIOS = {
    "clean_rigid_transform_only": build_clean_rigid_transform_only,
    "rigid_transform_moderate_outliers": build_rigid_transform_with_moderate_outliers,
    "rigid_transform_many_outliers": build_rigid_transform_with_many_outliers,
    "two_quasi_rigid_domains": build_two_quasi_rigid_domains,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the synthetic benchmark suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_benchmarks")
    parser.add_argument("--workdir", type=Path, default=ROOT / "_tmp_mdsuperpose_synthetic")
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the requested synthetic benchmark suite."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for scenario_name, builder in SCENARIOS.items():
        for repeat_index in range(args.repeats):
            seed = args.seed_base + repeat_index
            trajectory, preprocessing_steps = builder(seed)
            results = compare_methods_on_trajectory(
                trajectory,
                mdlovofit_fraction=args.fraction,
                workdir=args.workdir / f"{scenario_name}_repeat{repeat_index}",
                mdlovofit_executable=args.mdlovofit_executable,
                mdlovofit_config=args.mdlovofit_config,
                robust_rule="huber",
                delta=0.25,
            )
            run_rows.extend(
                benchmark_results_to_rows(
                    results,
                    benchmark_mode="synthetic",
                    dataset_name=scenario_name,
                    repeat_index=repeat_index,
                    preprocessing_steps=preprocessing_steps,
                )
            )
            run_records.append(
                {
                    "benchmark_mode": "synthetic",
                    "dataset_name": scenario_name,
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "preprocessing_steps": preprocessing_steps,
                    "results": [benchmark_result_to_record(result) for result in results],
                }
            )

    summary_rows = summarize_benchmark_rows(run_rows)
    runs_csv = args.output_dir / "synthetic_runs.csv"
    summary_csv = args.output_dir / "synthetic_summary.csv"
    payload_json = args.output_dir / "synthetic_runs.json"
    write_benchmark_csv(runs_csv, run_rows)
    write_benchmark_csv(summary_csv, summary_rows)
    write_benchmark_json(
        payload_json,
        {
            "benchmark_mode": "synthetic",
            "fraction": args.fraction,
            "repeats": args.repeats,
            "records": run_records,
            "summary_rows": summary_rows,
        },
    )

    print("Synthetic benchmark complete")
    print(f"runs_csv={runs_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"payload_json={payload_json}")


if __name__ == "__main__":
    main()
