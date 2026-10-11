"""Run a moderate ablation study for the internal soft-weight method."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (
    benchmark_result_to_record,
    benchmark_results_to_rows,
    compare_methods_with_soft_configurations,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)

from mdstablecore.trajectory_io import load_mdanalysis_trajectory
from mdsuperpose.trajectory import Trajectory


NAMED_DATASETS = {
    "ifabp": {
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
    },
    "adk_equilibrium": {
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse arguments for the ablation study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--named-dataset", choices=sorted(NAMED_DATASETS), default="ifabp")
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mdsuperpose_ablation")
    parser.add_argument("--workdir", type=Path, default=ROOT / "_tmp_mdsuperpose_ablation")
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def load_named_trajectory(dataset_name: str, *, stride: int, max_frames: int | None) -> Trajectory:
    """Load one named trajectory for ablation runs."""

    dataset = NAMED_DATASETS[dataset_name]
    coords = load_mdanalysis_trajectory(
        topology_path=Path(dataset["topology"]),
        trajectory_path=Path(dataset["trajectory"]),
        selection=str(dataset["selection"]),
        stride=stride,
        max_frames=max_frames,
    ).coords
    return Trajectory(coords=coords)


def main() -> None:
    """Run a moderate ablation set around regularization choices."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    trajectory = load_named_trajectory(args.named_dataset, stride=args.stride, max_frames=args.max_frames)

    configurations = [
        {
            "method_name": "ablation_base_no_regularization",
            "core_fraction": args.fraction,
            "rule": "huber",
            "delta": 0.25,
        },
        {
            "method_name": "ablation_sequence_only",
            "core_fraction": args.fraction,
            "rule": "huber",
            "delta": 0.25,
            "sequence_continuity_strength": 0.6,
            "sequence_continuity_passes": 2,
        },
        {
            "method_name": "ablation_contact_only",
            "core_fraction": args.fraction,
            "rule": "huber",
            "delta": 0.25,
            "contact_consistency_strength": 0.5,
            "contact_consistency_passes": 2,
        },
        {
            "method_name": "ablation_sequence_plus_contact",
            "core_fraction": args.fraction,
            "rule": "huber",
            "delta": 0.25,
            "sequence_continuity_strength": 0.6,
            "sequence_continuity_passes": 2,
            "contact_consistency_strength": 0.5,
            "contact_consistency_passes": 2,
        },
    ]

    run_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for repeat_index in range(args.repeats):
        results = compare_methods_with_soft_configurations(
            trajectory,
            mdlovofit_fraction=args.fraction,
            workdir=args.workdir / f"{args.named_dataset}_repeat{repeat_index}",
            soft_configurations=configurations,
            mdlovofit_executable=args.mdlovofit_executable,
            mdlovofit_config=args.mdlovofit_config,
        )
        rows = benchmark_results_to_rows(
            results,
            benchmark_mode="ablation",
            dataset_name=args.named_dataset,
            repeat_index=repeat_index,
            preprocessing_steps=[
                f"named_dataset={args.named_dataset}",
                f"stride={args.stride}",
                f"max_frames={args.max_frames}",
                "selection=protein and name CA",
            ],
        )
        run_rows.extend(rows)
        run_records.append(
            {
                "repeat_index": repeat_index,
                "results": [benchmark_result_to_record(result) for result in results],
            }
        )

    summary_rows = summarize_benchmark_rows(run_rows)
    prefix = args.output_dir / args.named_dataset
    runs_csv = Path(f"{prefix}_ablation_runs.csv")
    summary_csv = Path(f"{prefix}_ablation_summary.csv")
    payload_json = Path(f"{prefix}_ablation_runs.json")
    write_benchmark_csv(runs_csv, run_rows)
    write_benchmark_csv(summary_csv, summary_rows)
    write_benchmark_json(
        payload_json,
        {
            "benchmark_mode": "ablation",
            "dataset_name": args.named_dataset,
            "fraction": args.fraction,
            "records": run_records,
            "summary_rows": summary_rows,
        },
    )

    print("Ablation study complete")
    print(f"runs_csv={runs_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"payload_json={payload_json}")


if __name__ == "__main__":
    main()
