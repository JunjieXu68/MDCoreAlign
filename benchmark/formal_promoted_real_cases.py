"""Promote NhaA and YiiP short from smoke validation to formal repeated benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (
    BenchmarkResult,
    RigidTransform,
    Trajectory,
    benchmark_ordinary_alignment,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    compare_methods_with_soft_configurations,
    load_paired_protein_coords_from_arrays,
    run_mdlovofit_on_coordinates,
    select_core_by_target_fraction,
    solve_iterative_robust_alignment,
    solve_rigid_alignment,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)
from mdstablecore.trajectory_io import load_mdanalysis_trajectory


FORMAL_DATASETS: dict[str, dict[str, object]] = {
    "nhaa_equilibrium": {
        "label": "NhaA equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
        "stride": 25,
        "max_frames": 201,
    },
    "yiip_equilibrium_short": {
        "label": "YiiP equilibrium short",
        "topology": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc"),
        "selection": "protein and name CA",
        "stride": 5,
        "max_frames": 181,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the promoted formal real-case benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(FORMAL_DATASETS),
        default=sorted(FORMAL_DATASETS),
    )
    parser.add_argument("--fraction", type=float, default=0.60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--robust-rule", type=str, default="huber")
    parser.add_argument("--robust-delta", type=float, default=0.25)
    parser.add_argument("--robust-max-iterations", type=int, default=20)
    parser.add_argument("--robust-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "higher_tier_promoted_real_cases_formal",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/dev/shm/mdca_promoted_real_cases_formal"),
    )
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def _pair_rmsd(aligned_mobile: np.ndarray, reference: np.ndarray) -> float:
    """Compute RMSD between one aligned mobile structure and one reference."""

    squared = np.sum((aligned_mobile - reference) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def load_dataset(name: str) -> tuple[Trajectory, dict[str, object], list[str]]:
    """Load one promoted real dataset using its formal benchmark settings."""

    config = FORMAL_DATASETS[name]
    coords = load_mdanalysis_trajectory(
        topology_path=Path(config["topology"]),
        trajectory_path=Path(config["trajectory"]),
        selection=str(config["selection"]),
        stride=int(config["stride"]),
        max_frames=int(config["max_frames"]),
    ).coords
    preprocessing_steps = [
        "source_format=mdanalysis_named_dataset",
        f"named_dataset={name}",
        f"topology_path={config['topology']}",
        f"trajectory_path={config['trajectory']}",
        f"selection={config['selection']}",
        f"stride={config['stride']}",
        f"max_frames={config['max_frames']}",
        "reference_mode=first_frame",
        "atom_order=as_provided",
    ]
    return Trajectory(coords=coords), config, preprocessing_steps


def benchmark_pair_methods(
    trajectory: Trajectory,
    *,
    dataset_name: str,
    fraction: float,
    rule: str,
    delta: float,
    max_iterations: int,
    tolerance: float,
    workdir: Path,
    mdlovofit_executable: Path | None,
    mdlovofit_config: Path | None,
) -> tuple[list[BenchmarkResult], list[str]]:
    """Benchmark ordinary, soft, and external MDLovoFit on a first-vs-last pair."""

    paired = load_paired_protein_coords_from_arrays(
        trajectory.coords[-1],
        trajectory.coords[0],
    )
    mobile = paired.mobile_coords
    reference = paired.reference_coords
    preprocessing_steps = [
        "source=named_real_ca_pair",
        f"named_pair={dataset_name}_first_last_ca",
        "reference_frame_index=0",
        f"mobile_frame_index={trajectory.n_frames - 1}",
        "correspondence=trajectory_ca_order",
    ]

    ordinary_start = time.perf_counter()
    ordinary = solve_rigid_alignment(mobile, reference)
    ordinary_runtime = time.perf_counter() - ordinary_start

    robust_start = time.perf_counter()
    robust = solve_iterative_robust_alignment(
        mobile,
        reference,
        rule=rule,
        delta=delta,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    robust_runtime = time.perf_counter() - robust_start
    hard_core = select_core_by_target_fraction(robust.weights, fraction)

    external_run = run_mdlovofit_on_coordinates(
        np.stack([reference, mobile], axis=0),
        fraction=fraction,
        workdir=workdir,
        executable=mdlovofit_executable,
        config_path=mdlovofit_config,
    )
    external_rmsd = _pair_rmsd(external_run.aligned_coords[1], reference)
    external_overhead = external_run.input_write_seconds + external_run.output_read_seconds
    parsed_values = [
        external_run.parsed_output.average_rmsd_least_mobile,
        external_run.parsed_output.average_rmsd_most_mobile,
        external_run.parsed_output.average_rmsd_all_atoms,
    ]
    parsed_count = sum(value is not None for value in parsed_values)

    results = [
        BenchmarkResult(
            method_name="ordinary_alignment_pair",
            runtime_seconds=ordinary_runtime,
            algorithm_runtime_seconds=ordinary_runtime,
            overhead_seconds=0.0,
            rmsd=ordinary.rmsd,
            weighted_rmsd=ordinary.weighted_rmsd,
            iteration_count=None,
            core_size=None,
            core_fraction=None,
            parsing_success=None,
            transform_if_available=RigidTransform(rotation=ordinary.rotation, translation=ordinary.translation),
            extra_metadata={"benchmark_mode": "structure_pair"},
        ),
        BenchmarkResult(
            method_name="soft_weight_robust_alignment_pair",
            runtime_seconds=robust_runtime,
            algorithm_runtime_seconds=robust_runtime,
            overhead_seconds=0.0,
            rmsd=robust.rmsd,
            weighted_rmsd=robust.weighted_rmsd,
            iteration_count=float(len(robust.history)),
            core_size=hard_core.core_size,
            core_fraction=hard_core.core_fraction,
            parsing_success=None,
            transform_if_available=RigidTransform(rotation=robust.rotation, translation=robust.translation),
            extra_metadata={
                "benchmark_mode": "structure_pair",
                "selected_core_indices": hard_core.core_indices.tolist(),
                "converged": robust.converged,
            },
        ),
        BenchmarkResult(
            method_name="external_mdlovofit_pair",
            runtime_seconds=external_run.total_runtime_seconds,
            algorithm_runtime_seconds=None,
            overhead_seconds=external_overhead,
            rmsd=external_rmsd,
            weighted_rmsd=None,
            iteration_count=None,
            core_size=None,
            core_fraction=None,
            parsing_success=parsed_count > 0,
            transform_if_available=None,
            stdout=external_run.stdout,
            stderr=external_run.stderr,
            return_code=external_run.return_code,
            extra_metadata={
                "benchmark_mode": "structure_pair",
                "requested_fraction": external_run.requested_fraction,
                "subprocess_runtime_seconds": external_run.subprocess_runtime_seconds,
                "reported_fraction_of_best_aligned_atoms": external_run.parsed_output.fraction_of_best_aligned_atoms,
                "reported_total_number_of_atoms": external_run.parsed_output.total_number_of_atoms,
                "reported_number_of_atoms_of_selection": external_run.parsed_output.number_of_atoms_of_selection,
                "reported_total_number_of_atoms_found": external_run.parsed_output.total_number_of_atoms_found,
                "reported_different_assignment_for_each_frame": external_run.parsed_output.different_assignment_for_each_frame,
                "reported_number_of_trials_for_each_alignment": external_run.parsed_output.number_of_trials_for_each_alignment,
                "reported_frame_rmsd_row_count": external_run.parsed_output.frame_rmsd_row_count,
                "stdout_parse_status": "none" if parsed_count == 0 else "partial" if parsed_count < 3 else "full",
                "reported_average_rmsd_all_atoms": external_run.parsed_output.average_rmsd_all_atoms,
                "reported_average_rmsd_least_mobile": external_run.parsed_output.average_rmsd_least_mobile,
                "reported_average_rmsd_most_mobile": external_run.parsed_output.average_rmsd_most_mobile,
                "output_subset_atom_count": external_run.output_subset_atom_count,
                "output_subset_atom_fraction": external_run.output_subset_atom_fraction,
                "output_subset_mask_binary": external_run.output_subset_mask_binary,
                "output_subset_mask_constant_across_frames": external_run.output_subset_mask_constant_across_frames,
            },
        ),
    ]
    return results, preprocessing_steps


def build_note(trajectory_summary: list[dict[str, object]], pair_summary: list[dict[str, object]]) -> str:
    """Build a concise markdown note for the promoted formal real cases."""

    def add_rows(lines: list[str], rows: list[dict[str, object]], title: str) -> None:
        lines.extend([title, ""])
        for row in rows:
            runtime = row.get("runtime_seconds_mean")
            rmsd = row.get("unified_posthoc_rmsd_mean")
            dataset_name = row["dataset_name"]
            method_name = row["method_name"]
            if runtime is not None and rmsd is not None:
                lines.append(
                    f"- {dataset_name} / {method_name}: runtime={runtime:.4f}s, unified_posthoc_rmsd={rmsd:.4f}"
                )
            else:
                lines.append(f"- {dataset_name} / {method_name}")
        lines.append("")

    lines = [
        "# Promoted Formal Real-Case Summary",
        "",
        "This run promotes NhaA and YiiP short from smoke validation to repeated formal benchmarks.",
        "",
        "Trajectory block:",
        "",
        "- ordinary alignment (supplementary reference)",
        "- soft robust alignment",
        "- external MDLovoFit",
        "",
    ]
    add_rows(lines, trajectory_summary, "Trajectory formal summary rows:")
    add_rows(lines, pair_summary, "Structure-pair formal summary rows:")
    lines.extend(
        [
            "Notes:",
            "",
            "- unified post-hoc RMSD remains the main cross-method comparison metric",
            "- MDLovoFit-native metrics are reported separately",
            "- the promoted datasets are NhaA equilibrium and YiiP equilibrium short",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run repeated formal trajectory and pair benchmarks for promoted real cases."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    trajectory_rows_all: list[dict[str, object]] = []
    trajectory_records_all: list[dict[str, object]] = []
    pair_rows_all: list[dict[str, object]] = []
    pair_records_all: list[dict[str, object]] = []

    for dataset_name in args.datasets:
        trajectory, config, trajectory_preprocessing = load_dataset(dataset_name)
        dataset_output_dir = args.output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        dataset_trajectory_rows: list[dict[str, object]] = []
        dataset_trajectory_records: list[dict[str, object]] = []
        dataset_pair_rows: list[dict[str, object]] = []
        dataset_pair_records: list[dict[str, object]] = []

        soft_configurations = [
            {
                "method_name": "soft_weight_no_regularization",
                "core_fraction": args.fraction,
                "rule": args.robust_rule,
                "delta": args.robust_delta,
                "max_iterations": args.robust_max_iterations,
                "tolerance": args.robust_tolerance,
                "sequence_continuity_strength": 0.0,
                "sequence_continuity_passes": 1,
                "contact_consistency_strength": 0.0,
                "contact_consistency_passes": 1,
                "contact_distance_cutoff": 8.0,
                "contact_min_sequence_separation": 2,
                "contact_max_neighbors": 8,
            }
        ]

        for repeat_index in range(args.repeats):
            trajectory_results = compare_methods_with_soft_configurations(
                trajectory,
                mdlovofit_fraction=args.fraction,
                workdir=args.workdir / dataset_name / "trajectory" / f"repeat{repeat_index}",
                soft_configurations=soft_configurations,
                mdlovofit_executable=args.mdlovofit_executable,
                mdlovofit_config=args.mdlovofit_config,
            )
            dataset_trajectory_rows.extend(
                benchmark_results_to_rows(
                    trajectory_results,
                    benchmark_mode="promoted_real_trajectory",
                    dataset_name=dataset_name,
                    repeat_index=repeat_index,
                    preprocessing_steps=trajectory_preprocessing,
                )
            )
            dataset_trajectory_records.append(
                {
                    "benchmark_mode": "promoted_real_trajectory",
                    "dataset_name": dataset_name,
                    "repeat_index": repeat_index,
                    "preprocessing_steps": trajectory_preprocessing,
                    "results": [benchmark_result_to_record(result) for result in trajectory_results],
                }
            )

            pair_results, pair_preprocessing = benchmark_pair_methods(
                trajectory,
                dataset_name=dataset_name,
                fraction=args.fraction,
                rule=args.robust_rule,
                delta=args.robust_delta,
                max_iterations=args.robust_max_iterations,
                tolerance=args.robust_tolerance,
                workdir=args.workdir / dataset_name / "pair" / f"repeat{repeat_index}",
                mdlovofit_executable=args.mdlovofit_executable,
                mdlovofit_config=args.mdlovofit_config,
            )
            full_pair_preprocessing = trajectory_preprocessing + pair_preprocessing
            dataset_pair_rows.extend(
                benchmark_results_to_rows(
                    pair_results,
                    benchmark_mode="promoted_real_structure_pair",
                    dataset_name=f"{dataset_name}_first_last_ca_pair",
                    repeat_index=repeat_index,
                    preprocessing_steps=full_pair_preprocessing,
                )
            )
            dataset_pair_records.append(
                {
                    "benchmark_mode": "promoted_real_structure_pair",
                    "dataset_name": f"{dataset_name}_first_last_ca_pair",
                    "repeat_index": repeat_index,
                    "preprocessing_steps": full_pair_preprocessing,
                    "results": [benchmark_result_to_record(result) for result in pair_results],
                }
            )

        trajectory_summary = summarize_benchmark_rows(dataset_trajectory_rows)
        pair_summary = summarize_benchmark_rows(dataset_pair_rows)

        write_benchmark_csv(
            dataset_output_dir / f"{dataset_name}_formal_trajectory_runs.csv",
            dataset_trajectory_rows,
        )
        write_benchmark_csv(
            dataset_output_dir / f"{dataset_name}_formal_trajectory_summary.csv",
            trajectory_summary,
        )
        write_benchmark_json(
            dataset_output_dir / f"{dataset_name}_formal_trajectory_runs.json",
            {
                "benchmark_mode": "promoted_real_trajectory",
                "dataset_name": dataset_name,
                "fraction": args.fraction,
                "repeats": args.repeats,
                "dataset_config": {key: str(value) for key, value in config.items()},
                "runs": dataset_trajectory_records,
                "summary_rows": trajectory_summary,
            },
        )

        write_benchmark_csv(
            dataset_output_dir / f"{dataset_name}_formal_pair_runs.csv",
            dataset_pair_rows,
        )
        write_benchmark_csv(
            dataset_output_dir / f"{dataset_name}_formal_pair_summary.csv",
            pair_summary,
        )
        write_benchmark_json(
            dataset_output_dir / f"{dataset_name}_formal_pair_runs.json",
            {
                "benchmark_mode": "promoted_real_structure_pair",
                "dataset_name": f"{dataset_name}_first_last_ca_pair",
                "fraction": args.fraction,
                "repeats": args.repeats,
                "dataset_config": {key: str(value) for key, value in config.items()},
                "runs": dataset_pair_records,
                "summary_rows": pair_summary,
            },
        )

        trajectory_rows_all.extend(dataset_trajectory_rows)
        trajectory_records_all.extend(dataset_trajectory_records)
        pair_rows_all.extend(dataset_pair_rows)
        pair_records_all.extend(dataset_pair_records)

    combined_trajectory_summary = summarize_benchmark_rows(trajectory_rows_all)
    combined_pair_summary = summarize_benchmark_rows(pair_rows_all)
    combined_multi_system_summary = []
    for row in combined_trajectory_summary:
        row_copy = dict(row)
        row_copy["formal_case_block"] = "trajectory"
        combined_multi_system_summary.append(row_copy)
    for row in combined_pair_summary:
        row_copy = dict(row)
        row_copy["formal_case_block"] = "structure_pair"
        combined_multi_system_summary.append(row_copy)

    write_benchmark_csv(
        args.output_dir / "combined_formal_trajectory_runs.csv",
        trajectory_rows_all,
    )
    write_benchmark_csv(
        args.output_dir / "combined_formal_trajectory_summary.csv",
        combined_trajectory_summary,
    )
    write_benchmark_json(
        args.output_dir / "combined_formal_trajectory_runs.json",
        {
            "benchmark_mode": "promoted_real_trajectory",
            "datasets": args.datasets,
            "fraction": args.fraction,
            "repeats": args.repeats,
            "runs": trajectory_records_all,
            "summary_rows": combined_trajectory_summary,
        },
    )

    write_benchmark_csv(
        args.output_dir / "combined_formal_pair_runs.csv",
        pair_rows_all,
    )
    write_benchmark_csv(
        args.output_dir / "combined_formal_pair_summary.csv",
        combined_pair_summary,
    )
    write_benchmark_json(
        args.output_dir / "combined_formal_pair_runs.json",
        {
            "benchmark_mode": "promoted_real_structure_pair",
            "datasets": args.datasets,
            "fraction": args.fraction,
            "repeats": args.repeats,
            "runs": pair_records_all,
            "summary_rows": combined_pair_summary,
        },
    )

    write_benchmark_csv(
        args.output_dir / "combined_promoted_real_cases_summary.csv",
        combined_multi_system_summary,
    )
    (args.output_dir / "promoted_real_cases_note.md").write_text(
        build_note(combined_trajectory_summary, combined_pair_summary)
    )

    print("Promoted formal real-case benchmark complete")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
