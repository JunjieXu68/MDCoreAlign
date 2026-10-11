"""Run a formal repeated benchmark including the external THESEUS baseline."""

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
    benchmark_external_theseus,
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
        "label": "I-FABP",
        "topology": Path("/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
    },
    "adk_equilibrium": {
        "label": "ADK equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
    },
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
    """Parse command-line arguments for the THESEUS formal benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--fraction", type=float, default=0.60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "higher_tier_external_baseline_theseus_formal",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/dev/shm/mdca_external_baseline_theseus_formal"),
    )
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    parser.add_argument("--theseus-executable", type=Path, default=None)
    parser.add_argument("--theseus-config", type=Path, default=None)
    return parser.parse_args()


def load_dataset(name: str) -> tuple[Trajectory, list[str]]:
    """Load one real CA-only trajectory subset for the formal THESEUS comparison."""

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


def build_note(summary_rows: list[dict[str, object]]) -> str:
    """Build a concise markdown note summarizing the formal THESEUS run."""

    lines = [
        "# Formal THESEUS Baseline Summary",
        "",
        "This benchmark compares:",
        "",
        "- ordinary alignment (supplementary reference)",
        "- soft robust alignment",
        "- external MDLovoFit",
        "- external THESEUS",
        "",
        "Primary comparison focus:",
        "",
        "- soft robust alignment vs real external MDLovoFit",
        "- soft robust alignment vs real external THESEUS",
        "",
        "Key summary rows:",
        "",
    ]
    for row in summary_rows:
        dataset_name = row["dataset_name"]
        method_name = row["method_name"]
        runtime = row.get("runtime_seconds_mean")
        rmsd = row.get("unified_posthoc_rmsd_mean")
        lines.append(
            f"- {dataset_name} / {method_name}: "
            f"runtime={runtime:.4f}s, unified_posthoc_rmsd={rmsd:.4f}"
            if runtime is not None and rmsd is not None
            else f"- {dataset_name} / {method_name}"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- unified post-hoc RMSD remains the primary cross-method metric",
            "- MDLovoFit-native and THESEUS-native metrics are reported separately when available",
            "- external baseline I/O overhead remains separate from the main internal algorithm runtime",
            "",
        ]
    )
    return "\n".join(lines)


def run_one_dataset(
    dataset_name: str,
    *,
    fraction: float,
    repeats: int,
    output_dir: Path,
    workdir: Path,
    mdlovofit_executable: Path | None,
    mdlovofit_config: Path | None,
    theseus_executable: Path | None,
    theseus_config: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run one repeated formal benchmark including THESEUS."""

    trajectory, preprocessing_steps = load_dataset(dataset_name)
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for repeat_index in range(repeats):
        repeat_workdir = workdir / dataset_name / f"repeat{repeat_index}"
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
            benchmark_external_mdlovofit(
                trajectory,
                fraction=fraction,
                workdir=repeat_workdir / "mdlovofit",
                executable=mdlovofit_executable,
                config_path=mdlovofit_config,
            ),
            benchmark_external_theseus(
                trajectory,
                workdir=repeat_workdir / "theseus",
                executable=theseus_executable,
                config_path=theseus_config,
            ),
        ]
        run_rows.extend(
            benchmark_results_to_rows(
                results,
                benchmark_mode="external_baseline_theseus_real",
                dataset_name=dataset_name,
                repeat_index=repeat_index,
                preprocessing_steps=preprocessing_steps,
            )
        )
        run_records.append(
            {
                "benchmark_mode": "external_baseline_theseus_real",
                "dataset_name": dataset_name,
                "repeat_index": repeat_index,
                "preprocessing_steps": preprocessing_steps,
                "results": [benchmark_result_to_record(result) for result in results],
            }
        )

    summary_rows = summarize_benchmark_rows(run_rows)
    write_benchmark_csv(
        dataset_output_dir / f"{dataset_name}_theseus_formal_runs.csv",
        run_rows,
    )
    write_benchmark_csv(
        dataset_output_dir / f"{dataset_name}_theseus_formal_summary.csv",
        summary_rows,
    )
    write_benchmark_json(
        dataset_output_dir / f"{dataset_name}_theseus_formal_runs.json",
        {
            "benchmark_mode": "external_baseline_theseus_real",
            "dataset_name": dataset_name,
            "fraction": fraction,
            "repeats": repeats,
            "preprocessing_steps": preprocessing_steps,
            "runs": run_records,
            "summary_rows": summary_rows,
        },
    )
    return run_rows, summary_rows, run_records


def main() -> None:
    """Run the formal THESEUS baseline benchmark and save structured outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    all_run_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    all_run_records: list[dict[str, object]] = []

    for dataset_name in args.datasets:
        run_rows, summary_rows, run_records = run_one_dataset(
            dataset_name,
            fraction=args.fraction,
            repeats=args.repeats,
            output_dir=args.output_dir,
            workdir=args.workdir,
            mdlovofit_executable=args.mdlovofit_executable,
            mdlovofit_config=args.mdlovofit_config,
            theseus_executable=args.theseus_executable,
            theseus_config=args.theseus_config,
        )
        all_run_rows.extend(run_rows)
        all_summary_rows.extend(summary_rows)
        all_run_records.extend(run_records)

    write_benchmark_csv(
        args.output_dir / "combined_theseus_formal_runs.csv",
        all_run_rows,
    )
    write_benchmark_csv(
        args.output_dir / "combined_theseus_formal_summary.csv",
        all_summary_rows,
    )
    write_benchmark_json(
        args.output_dir / "combined_theseus_formal_runs.json",
        {
            "benchmark_mode": "external_baseline_theseus_real",
            "fraction": args.fraction,
            "repeats": args.repeats,
            "datasets": args.datasets,
            "runs": all_run_records,
            "summary_rows": all_summary_rows,
        },
    )
    (args.output_dir / "theseus_formal_note.md").write_text(build_note(all_summary_rows))

    print("Formal THESEUS baseline benchmark complete")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
