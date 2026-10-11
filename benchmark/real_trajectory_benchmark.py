"""Real-input benchmark for internal methods and external MDLovoFit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (
    Trajectory,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    compare_methods_with_soft_configurations,
    load_trajectory_from_array_file,
    load_trajectory_from_multimodel_pdb,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)

from mdstablecore.trajectory_io import load_mdanalysis_trajectory


NAMED_DATASETS: dict[str, dict[str, object]] = {
    "ifabp": {
        "label": "I-FABP",
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
    },
    "adk_equilibrium": {
        "label": "Adk_equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
    },
    "nhaa_equilibrium": {
        "label": "NhaA_equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
    },
    "yiip_equilibrium_short": {
        "label": "YiiP_equilibrium_short",
        "topology": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc"),
        "selection": "protein and name CA",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the real-input benchmark mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--named-dataset", choices=sorted(NAMED_DATASETS), default=None)
    source.add_argument("--coords-npy", type=Path, default=None)
    source.add_argument("--coords-npz", type=Path, default=None)
    source.add_argument("--coords-pdb", type=Path, default=None)
    parser.add_argument("--coords-key", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--robust-rule", type=str, default="huber")
    parser.add_argument("--robust-delta", type=float, default=0.25)
    parser.add_argument("--robust-max-iterations", type=int, default=20)
    parser.add_argument("--robust-tolerance", type=float, default=1e-6)
    parser.add_argument("--sequence-continuity-strengths", type=float, nargs="+", default=[0.0])
    parser.add_argument("--sequence-continuity-passes", type=int, default=1)
    parser.add_argument("--contact-consistency-strengths", type=float, nargs="+", default=[0.0])
    parser.add_argument("--contact-consistency-passes", type=int, default=1)
    parser.add_argument("--contact-distance-cutoff", type=float, default=8.0)
    parser.add_argument("--contact-min-sequence-separation", type=int, default=2)
    parser.add_argument("--contact-max-neighbors", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_benchmarks")
    parser.add_argument("--workdir", type=Path, default=ROOT / "_tmp_mdsuperpose_real")
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def load_real_trajectory(args: argparse.Namespace) -> tuple[Trajectory, str, list[str]]:
    """Load one real trajectory input and report explicit preprocessing steps."""

    preprocessing_steps: list[str] = []
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    if args.named_dataset is not None:
        dataset = NAMED_DATASETS[args.named_dataset]
        trajectory = Trajectory(
            coords=load_mdanalysis_trajectory(
                topology_path=Path(dataset["topology"]),
                trajectory_path=Path(dataset["trajectory"]),
                selection=str(dataset["selection"]),
                stride=args.stride,
                max_frames=args.max_frames,
            ).coords
        )
        dataset_name = args.dataset_name or args.named_dataset
        preprocessing_steps.extend([
            "source_format=mdanalysis_named_dataset",
            f"named_dataset={args.named_dataset}",
            f"topology_path={dataset['topology']}",
            f"trajectory_path={dataset['trajectory']}",
            f"selection={dataset['selection']}",
        ])
        original_frames = trajectory.n_frames
        coords = trajectory.coords
    elif args.coords_npy is not None:
        trajectory = load_trajectory_from_array_file(args.coords_npy)
        dataset_name = args.dataset_name or args.coords_npy.stem
        preprocessing_steps.extend([
            f"source_format=npy",
            f"source_path={args.coords_npy}",
        ])
        original_frames = trajectory.n_frames
        coords = trajectory.coords[:: args.stride]
    elif args.coords_npz is not None:
        trajectory = load_trajectory_from_array_file(args.coords_npz, key=args.coords_key)
        dataset_name = args.dataset_name or args.coords_npz.stem
        preprocessing_steps.extend([
            f"source_format=npz",
            f"source_path={args.coords_npz}",
            f"coords_key={args.coords_key if args.coords_key is not None else 'auto'}",
        ])
        original_frames = trajectory.n_frames
        coords = trajectory.coords[:: args.stride]
    else:
        trajectory = load_trajectory_from_multimodel_pdb(args.coords_pdb)
        dataset_name = args.dataset_name or args.coords_pdb.stem
        preprocessing_steps.extend([
            f"source_format=multimodel_pdb",
            f"source_path={args.coords_pdb}",
        ])
        original_frames = trajectory.n_frames
        coords = trajectory.coords[:: args.stride]

    preprocessing_steps.append(f"stride={args.stride}")
    if args.max_frames is not None and args.named_dataset is None:
        if args.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        coords = coords[: args.max_frames]
        preprocessing_steps.append(f"max_frames={args.max_frames}")
    elif args.max_frames is not None:
        preprocessing_steps.append(f"max_frames={args.max_frames}")
    preprocessing_steps.extend([
        f"frames_before_preprocessing={original_frames}",
        f"frames_after_preprocessing={coords.shape[0]}",
        "reference_mode=first_frame",
        "atom_order=as_provided",
    ])
    return Trajectory(coords=coords), dataset_name, preprocessing_steps


def build_soft_configurations(args: argparse.Namespace) -> list[dict[str, object]]:
    """Build the requested soft-method comparison configurations."""

    configurations: list[dict[str, object]] = []
    if args.contact_distance_cutoff <= 0.0:
        raise ValueError("contact_distance_cutoff must be positive")
    if args.contact_min_sequence_separation < 0:
        raise ValueError("contact_min_sequence_separation must be non-negative")
    if args.contact_max_neighbors <= 0:
        raise ValueError("contact_max_neighbors must be positive")
    if args.contact_consistency_passes < 0:
        raise ValueError("contact_consistency_passes must be non-negative")

    for sequence_strength in args.sequence_continuity_strengths:
        if sequence_strength < 0.0 or sequence_strength > 1.0:
            raise ValueError("sequence continuity strengths must lie in [0, 1]")
        for contact_strength in args.contact_consistency_strengths:
            if contact_strength < 0.0 or contact_strength > 1.0:
                raise ValueError("contact consistency strengths must lie in [0, 1]")

            if abs(sequence_strength) < 1e-12 and abs(contact_strength) < 1e-12:
                method_name = "soft_weight_no_regularization"
            elif abs(contact_strength) < 1e-12:
                method_name = f"soft_weight_sequence_reg_s{sequence_strength:.2f}"
            elif abs(sequence_strength) < 1e-12:
                method_name = f"soft_weight_contact_reg_c{contact_strength:.2f}"
            else:
                method_name = (
                    f"soft_weight_seq_s{sequence_strength:.2f}"
                    f"_contact_c{contact_strength:.2f}"
                )

            configurations.append(
                {
                    "method_name": method_name,
                    "core_fraction": args.fraction,
                    "rule": args.robust_rule,
                    "delta": args.robust_delta,
                    "max_iterations": args.robust_max_iterations,
                    "tolerance": args.robust_tolerance,
                    "sequence_continuity_strength": float(sequence_strength),
                    "sequence_continuity_passes": args.sequence_continuity_passes,
                    "contact_consistency_strength": float(contact_strength),
                    "contact_consistency_passes": args.contact_consistency_passes,
                    "contact_distance_cutoff": args.contact_distance_cutoff,
                    "contact_min_sequence_separation": args.contact_min_sequence_separation,
                    "contact_max_neighbors": args.contact_max_neighbors,
                }
            )
    return configurations


def main() -> None:
    """Run the real-input benchmark and save machine-friendly outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    trajectory, dataset_name, preprocessing_steps = load_real_trajectory(args)
    if trajectory.n_frames < 2:
        raise ValueError("real-input benchmark requires at least two frames")
    soft_configurations = build_soft_configurations(args)
    preprocessing_steps.extend([
        f"robust_rule={args.robust_rule}",
        f"robust_delta={args.robust_delta}",
        f"robust_max_iterations={args.robust_max_iterations}",
        f"robust_tolerance={args.robust_tolerance}",
        f"sequence_continuity_strengths={args.sequence_continuity_strengths}",
        f"sequence_continuity_passes={args.sequence_continuity_passes}",
        f"contact_consistency_strengths={args.contact_consistency_strengths}",
        f"contact_consistency_passes={args.contact_consistency_passes}",
        f"contact_distance_cutoff={args.contact_distance_cutoff}",
        f"contact_min_sequence_separation={args.contact_min_sequence_separation}",
        f"contact_max_neighbors={args.contact_max_neighbors}",
    ])

    run_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for repeat_index in range(args.repeats):
        results = compare_methods_with_soft_configurations(
            trajectory,
            mdlovofit_fraction=args.fraction,
            workdir=args.workdir / f"{dataset_name}_repeat{repeat_index}",
            soft_configurations=soft_configurations,
            mdlovofit_executable=args.mdlovofit_executable,
            mdlovofit_config=args.mdlovofit_config,
        )
        run_rows.extend(
            benchmark_results_to_rows(
                results,
                benchmark_mode="real",
                dataset_name=dataset_name,
                repeat_index=repeat_index,
                preprocessing_steps=preprocessing_steps,
            )
        )
        run_records.append(
            {
                "benchmark_mode": "real",
                "dataset_name": dataset_name,
                "repeat_index": repeat_index,
                "preprocessing_steps": preprocessing_steps,
                "results": [benchmark_result_to_record(result) for result in results],
            }
        )

    summary_rows = summarize_benchmark_rows(run_rows)
    runs_csv = args.output_dir / f"{dataset_name}_real_runs.csv"
    summary_csv = args.output_dir / f"{dataset_name}_real_summary.csv"
    payload_json = args.output_dir / f"{dataset_name}_real_runs.json"
    write_benchmark_csv(runs_csv, run_rows)
    write_benchmark_csv(summary_csv, summary_rows)
    write_benchmark_json(
        payload_json,
        {
            "benchmark_mode": "real",
            "dataset_name": dataset_name,
            "fraction": args.fraction,
            "repeats": args.repeats,
            "preprocessing_steps": preprocessing_steps,
            "records": run_records,
            "summary_rows": summary_rows,
        },
    )

    print("Real-input benchmark complete")
    print(f"runs_csv={runs_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"payload_json={payload_json}")


if __name__ == "__main__":
    main()
