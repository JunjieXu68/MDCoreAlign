"""Benchmark one real or protein-like structure pair against external MDLovoFit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    BenchmarkResult,
    RigidTransform,
    apply_transform,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    load_paired_ca_coords_from_pdb,
    load_paired_protein_coords_from_arrays,
    run_mdlovofit_on_coordinates,
    run_theseus_on_coordinates,
    solve_rigid_alignment as solve_rigid_alignment_pair_map,
    select_core_by_target_fraction,
    solve_iterative_robust_alignment,
    solve_rigid_alignment,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)
from mdstablecore.trajectory_io import load_mdanalysis_trajectory


EXAMPLE_MOBILE_PDB = """
ATOM      1  CA  MET A   1      0.300  -0.100   0.200  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.800   0.100   0.200  1.00 20.00           C
ATOM      3  CA  LEU A   3      3.200   0.300   0.200  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.400   0.700   0.300  1.00 20.00           C
END
"""

EXAMPLE_REFERENCE_PDB = """
ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLU A   2      1.500   0.200   0.000  1.00 20.00           C
ATOM      3  CA  LEU A   3      2.900   0.400   0.000  1.00 20.00           C
ATOM      4  CA  ARG A   4      4.100   0.800   0.100  1.00 20.00           C
END
"""


NAMED_TRAJECTORY_PAIRS = {
    "ifabp_first_last_ca": {
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "reference_frame_index": 0,
        "mobile_frame_index": -1,
    },
    "adk_first_last_ca": {
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "reference_frame_index": 0,
        "mobile_frame_index": -1,
        "stride": 1,
        "max_frames": 500,
    },
    "nhaa_first_last_ca": {
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
        "reference_frame_index": 0,
        "mobile_frame_index": -1,
        "stride": 25,
        "max_frames": 201,
    },
    "yiip_short_first_last_ca": {
        "topology": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc"),
        "selection": "protein and name CA",
        "reference_frame_index": 0,
        "mobile_frame_index": -1,
        "stride": 5,
        "max_frames": 181,
    },
}


def _pair_rmsd(aligned_mobile: np.ndarray, reference: np.ndarray) -> float:
    """Compute RMSD between one aligned mobile structure and one reference."""

    squared = np.sum((aligned_mobile - reference) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def _load_example_pair():
    """Load the built-in tiny example structure pair."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mobile_path = tmp / "mobile_example.pdb"
        reference_path = tmp / "reference_example.pdb"
        mobile_path.write_text(EXAMPLE_MOBILE_PDB.strip() + "\n")
        reference_path.write_text(EXAMPLE_REFERENCE_PDB.strip() + "\n")
        paired = load_paired_ca_coords_from_pdb(mobile_path, reference_path)
    return paired


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the structure-pair benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--named-pair", choices=sorted(NAMED_TRAJECTORY_PAIRS), default=None)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--mobile-pdb", type=Path, default=None)
    parser.add_argument("--reference-pdb", type=Path, default=None)
    parser.add_argument("--mobile-npy", type=Path, default=None)
    parser.add_argument("--reference-npy", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default="structure_pair")
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--rule", type=str, default="huber")
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_structure_pairs")
    parser.add_argument("--workdir", type=Path, default=ROOT / "_tmp_mdsuperpose_structure_pairs")
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    parser.add_argument("--theseus-executable", type=Path, default=None)
    parser.add_argument("--theseus-config", type=Path, default=None)
    return parser.parse_args()


def load_pair(args: argparse.Namespace):
    """Load one paired coordinate problem from arrays, PDBs, or the example."""

    if args.named_pair is not None:
        pair = NAMED_TRAJECTORY_PAIRS[args.named_pair]
        coords = load_mdanalysis_trajectory(
            topology_path=Path(pair["topology"]),
            trajectory_path=Path(pair["trajectory"]),
            selection=str(pair["selection"]),
            stride=int(pair.get("stride", 1)),
            max_frames=(
                int(pair["max_frames"])
                if pair.get("max_frames") is not None
                else None
            ),
        ).coords
        n_frames = coords.shape[0]
        reference_frame_index = int(pair["reference_frame_index"])
        mobile_frame_index = int(pair["mobile_frame_index"])
        if reference_frame_index < 0:
            reference_frame_index += n_frames
        if mobile_frame_index < 0:
            mobile_frame_index += n_frames
        if not (0 <= reference_frame_index < n_frames and 0 <= mobile_frame_index < n_frames):
            raise ValueError("named pair frame indices must be valid after normalization")
        return (
            load_paired_protein_coords_from_arrays(
                coords[mobile_frame_index],
                coords[reference_frame_index],
            ),
            [
                "source=named_real_ca_pair",
                f"named_pair={args.named_pair}",
                f"topology_path={pair['topology']}",
                f"trajectory_path={pair['trajectory']}",
                f"selection={pair['selection']}",
                f"stride={pair.get('stride', 1)}",
                f"max_frames={pair.get('max_frames', 'all')}",
                f"reference_frame_index={reference_frame_index}",
                f"mobile_frame_index={mobile_frame_index}",
                "correspondence=trajectory_ca_order",
            ],
        )
    if args.example:
        return _load_example_pair(), ["source=example_pdb_pair", "correspondence=residue_labels"]
    if args.mobile_pdb is not None or args.reference_pdb is not None:
        if args.mobile_pdb is None or args.reference_pdb is None:
            raise ValueError("both --mobile-pdb and --reference-pdb are required together")
        return (
            load_paired_ca_coords_from_pdb(args.mobile_pdb, args.reference_pdb),
            [
                "source=paired_ca_pdb",
                f"mobile_path={args.mobile_pdb}",
                f"reference_path={args.reference_pdb}",
                "correspondence=residue_labels",
            ],
        )
    if args.mobile_npy is not None or args.reference_npy is not None:
        if args.mobile_npy is None or args.reference_npy is None:
            raise ValueError("both --mobile-npy and --reference-npy are required together")
        return (
            load_paired_protein_coords_from_arrays(np.load(args.mobile_npy), np.load(args.reference_npy)),
            [
                "source=paired_numpy_arrays",
                f"mobile_path={args.mobile_npy}",
                f"reference_path={args.reference_npy}",
                "correspondence=order",
            ],
        )
    raise ValueError("please provide --example, a pair of PDBs, or a pair of NumPy arrays")


def benchmark_pair_methods(args: argparse.Namespace):
    """Run ordinary, robust, and external MDLovoFit on one structure pair."""

    paired, preprocessing_steps = load_pair(args)
    mobile = paired.mobile_coords
    reference = paired.reference_coords

    ordinary_start = time.perf_counter()
    ordinary = solve_rigid_alignment(mobile, reference)
    ordinary_runtime = time.perf_counter() - ordinary_start

    robust_start = time.perf_counter()
    robust = solve_iterative_robust_alignment(
        mobile,
        reference,
        rule=args.rule,
        delta=args.delta,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    robust_runtime = time.perf_counter() - robust_start
    hard_core = select_core_by_target_fraction(robust.weights, args.fraction)

    external_run = run_mdlovofit_on_coordinates(
        np.stack([reference, mobile], axis=0),
        fraction=args.fraction,
        workdir=args.workdir / args.dataset_name,
        executable=args.mdlovofit_executable,
        config_path=args.mdlovofit_config,
    )
    external_rmsd = _pair_rmsd(external_run.aligned_coords[1], reference)
    external_overhead = external_run.input_write_seconds + external_run.output_read_seconds
    parsed_values = [
        external_run.parsed_output.average_rmsd_least_mobile,
        external_run.parsed_output.average_rmsd_most_mobile,
        external_run.parsed_output.average_rmsd_all_atoms,
    ]
    parsed_count = sum(value is not None for value in parsed_values)

    theseus_result: BenchmarkResult | None = None
    if args.theseus_executable is not None or args.theseus_config is not None:
        theseus_run = run_theseus_on_coordinates(
            np.stack([reference, mobile], axis=0),
            workdir=args.workdir / f"{args.dataset_name}_theseus",
            executable=args.theseus_executable,
            config_path=args.theseus_config,
            output_root="theseus_pair",
            input_name="pair_input.pdb",
        )
        aligned_coords = theseus_run.aligned_coords
        if aligned_coords is not None:
            first_frame_fit = solve_rigid_alignment_pair_map(aligned_coords[0], reference)
            aligned_coords = apply_transform(
                aligned_coords,
                RigidTransform(
                    rotation=first_frame_fit.rotation,
                    translation=first_frame_fit.translation,
                ),
            )
            theseus_rmsd = _pair_rmsd(aligned_coords[1], reference)
        else:
            theseus_rmsd = None
        theseus_overhead = theseus_run.input_write_seconds + theseus_run.output_read_seconds
        theseus_parsed_values = [
            theseus_run.parsed_output.classical_pairwise_rmsd,
            theseus_run.parsed_output.maximum_likelihood_rmsd,
            theseus_run.parsed_output.maximum_likelihood_sigma,
        ]
        theseus_parsed_count = sum(value is not None for value in theseus_parsed_values)
        theseus_result = BenchmarkResult(
            method_name="external_theseus_pair",
            runtime_seconds=theseus_run.total_runtime_seconds,
            algorithm_runtime_seconds=None,
            overhead_seconds=theseus_overhead,
            rmsd=theseus_rmsd,
            weighted_rmsd=None,
            iteration_count=None,
            core_size=theseus_run.output_core_atom_count,
            core_fraction=theseus_run.output_core_atom_fraction,
            parsing_success=theseus_parsed_count > 0,
            transform_if_available=None,
            stdout=theseus_run.stdout,
            stderr=theseus_run.stderr,
            return_code=theseus_run.return_code,
            extra_metadata={
                "benchmark_mode": "structure_pair",
                "subprocess_runtime_seconds": theseus_run.subprocess_runtime_seconds,
                "reported_classical_pairwise_rmsd": theseus_run.parsed_output.classical_pairwise_rmsd,
                "reported_least_squares_sigma": theseus_run.parsed_output.least_squares_sigma,
                "reported_maximum_likelihood_rmsd": theseus_run.parsed_output.maximum_likelihood_rmsd,
                "reported_maximum_likelihood_sigma": theseus_run.parsed_output.maximum_likelihood_sigma,
                "reported_marginal_log_likelihood": theseus_run.parsed_output.marginal_log_likelihood,
                "reported_aic": theseus_run.parsed_output.aic,
                "reported_bic": theseus_run.parsed_output.bic,
                "reported_total_rounds": theseus_run.parsed_output.total_rounds,
                "reported_converged_fractional_precision": theseus_run.parsed_output.converged_fractional_precision,
                "reported_median_structure_index": theseus_run.parsed_output.median_structure_index,
                "reported_n_total": theseus_run.parsed_output.n_total,
                "reported_n_atoms": theseus_run.parsed_output.n_atoms,
                "reported_n_structures": theseus_run.parsed_output.n_structures,
                "output_core_atom_count": theseus_run.output_core_atom_count,
                "output_core_atom_fraction": theseus_run.output_core_atom_fraction,
                "output_variance_row_count": len(theseus_run.variance_records),
                "stdout_parse_status": (
                    "none"
                    if theseus_parsed_count == 0
                    else "full" if theseus_parsed_count == len(theseus_parsed_values) else "partial"
                ),
            },
        )

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
    if theseus_result is not None:
        results.append(theseus_result)
    return results, preprocessing_steps


def main() -> None:
    """Run the structure-pair benchmark and save machine-friendly outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    results, preprocessing_steps = benchmark_pair_methods(args)
    rows = benchmark_results_to_rows(
        results,
        benchmark_mode="structure_pair",
        dataset_name=args.dataset_name,
        repeat_index=0,
        preprocessing_steps=preprocessing_steps,
    )
    summary = summarize_benchmark_rows(rows)
    prefix = args.output_dir / args.dataset_name
    runs_csv = Path(f"{prefix}_structure_pair_runs.csv")
    summary_csv = Path(f"{prefix}_structure_pair_summary.csv")
    payload_json = Path(f"{prefix}_structure_pair_runs.json")
    write_benchmark_csv(runs_csv, rows)
    write_benchmark_csv(summary_csv, summary)
    write_benchmark_json(
        payload_json,
        {
            "benchmark_mode": "structure_pair",
            "dataset_name": args.dataset_name,
            "fraction": args.fraction,
            "preprocessing_steps": preprocessing_steps,
            "results": [benchmark_result_to_record(result) for result in results],
            "summary_rows": summary,
        },
    )

    print("Real structure-pair benchmark complete")
    print(f"runs_csv={runs_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"payload_json={payload_json}")


if __name__ == "__main__":
    main()
