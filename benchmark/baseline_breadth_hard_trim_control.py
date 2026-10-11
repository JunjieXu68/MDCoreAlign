"""Run a minimal baseline-breadth benchmark with one hard-trim control."""

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
    benchmark_external_mdlovofit,
    benchmark_hard_trim_control_alignment,
    benchmark_ordinary_alignment,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    benchmark_soft_weight_robust_alignment,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)
from mdstablecore.trajectory_io import load_mdanalysis_trajectory


DATASETS: dict[str, dict[str, object]] = {
    "ifabp": {
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
    },
    "adk_equilibrium": {
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
    },
    "nhaa_equilibrium": {
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
        "stride": 25,
        "max_frames": 201,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline-breadth run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--fraction", type=float, default=0.60)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "baseline_breadth_hard_trim_control",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/dev/shm/mdca_baseline_breadth_hard_trim"),
    )
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def load_dataset(name: str) -> tuple[Trajectory, list[str]]:
    """Load one real CA-only trajectory subset for the breadth benchmark."""

    config = DATASETS[name]
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
    return Trajectory(coords=coords), preprocessing_steps


def run_one_dataset(
    dataset_name: str,
    *,
    fraction: float,
    repeats: int,
    output_dir: Path,
    workdir: Path,
    mdlovofit_executable: Path | None,
    mdlovofit_config: Path | None,
) -> None:
    """Run one compact baseline-breadth benchmark and write structured outputs."""

    trajectory, preprocessing_steps = load_dataset(dataset_name)
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for repeat_index in range(repeats):
        results = [
            benchmark_ordinary_alignment(trajectory),
            benchmark_soft_weight_robust_alignment(
                trajectory,
                method_name="soft_weight_no_regularization",
                core_fraction=fraction,
                rule="huber",
                delta=0.25,
                max_iterations=20,
                tolerance=1e-6,
            ),
            benchmark_hard_trim_control_alignment(
                trajectory,
                method_name="hard_trim_control_alignment",
                core_fraction=fraction,
                max_iterations=20,
                tolerance=1e-6,
            ),
            benchmark_external_mdlovofit(
                trajectory,
                fraction=fraction,
                workdir=workdir / dataset_name / f"repeat{repeat_index}",
                executable=mdlovofit_executable,
                config_path=mdlovofit_config,
            ),
        ]
        run_rows.extend(
            benchmark_results_to_rows(
                results,
                benchmark_mode="baseline_breadth_real",
                dataset_name=dataset_name,
                repeat_index=repeat_index,
                preprocessing_steps=preprocessing_steps,
            )
        )
        run_records.append(
            {
                "benchmark_mode": "baseline_breadth_real",
                "dataset_name": dataset_name,
                "repeat_index": repeat_index,
                "preprocessing_steps": preprocessing_steps,
                "results": [benchmark_result_to_record(result) for result in results],
            }
        )

    summary_rows = summarize_benchmark_rows(run_rows)
    write_benchmark_csv(dataset_output_dir / f"{dataset_name}_baseline_breadth_runs.csv", run_rows)
    write_benchmark_csv(dataset_output_dir / f"{dataset_name}_baseline_breadth_summary.csv", summary_rows)
    write_benchmark_json(
        dataset_output_dir / f"{dataset_name}_baseline_breadth_runs.json",
        {
            "benchmark_mode": "baseline_breadth_real",
            "dataset_name": dataset_name,
            "fraction": fraction,
            "repeats": repeats,
            "preprocessing_steps": preprocessing_steps,
            "runs": run_records,
        },
    )


def main() -> None:
    """Run the compact baseline-breadth benchmark on the requested datasets."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        run_one_dataset(
            dataset_name,
            fraction=args.fraction,
            repeats=args.repeats,
            output_dir=args.output_dir,
            workdir=args.workdir,
            mdlovofit_executable=args.mdlovofit_executable,
            mdlovofit_config=args.mdlovofit_config,
        )
    print("Baseline-breadth benchmark complete")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
