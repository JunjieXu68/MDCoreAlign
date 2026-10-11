"""Inspect local datasets and run minimal smoke benchmarks on runnable cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (  # noqa: E402
    Trajectory,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    compare_methods_with_soft_configurations,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)
from mdstablecore.trajectory_io import load_mdanalysis_trajectory  # noqa: E402


DATASETS: list[dict[str, object]] = [
    {
        "dataset_name": "Adk_equilibrium",
        "local_dir": "data/Adk_equilibrium",
        "topology_path": "/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf",
        "trajectory_path": "/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd",
        "suggested_selection": "protein and name CA",
        "smoke_stride": 8,
        "smoke_max_frames": 500,
        "suggested_smallest_entry": "reference-vs-trajectory smoke benchmark",
        "pair_useful": "yes, first-vs-last CA pair",
        "motion_regime": "equilibrium single-domain with known conformational flexibility",
    },
    {
        "dataset_name": "Adk_transitions_DIMS",
        "local_dir": "data/Adk_transitions_DIMS",
        "topology_path": "/home/jjxu/MDAnalysis_data/adk_transitions_DIMS/DIMS/topologies/adk4ake.psf",
        "trajectory_path": "/home/jjxu/MDAnalysis_data/adk_transitions_DIMS/DIMS/trajectories/dims0001_fit-core.dcd",
        "suggested_selection": "protein and name CA",
        "smoke_stride": 1,
        "smoke_max_frames": 200,
        "suggested_smallest_entry": "reference-vs-trajectory smoke benchmark if topology is restored",
        "pair_useful": "unclear until topology issue is resolved",
        "motion_regime": "transition-path ensemble",
    },
    {
        "dataset_name": "I-FABP with water",
        "local_dir": "data/I-FABP with water",
        "topology_path": "/home/jjxu/MDAnalysis_data/ifabp_water/ifabp_water.psf",
        "trajectory_path": "/home/jjxu/MDAnalysis_data/ifabp_water/rmsfit_ifabp_water_1.dcd",
        "suggested_selection": "protein and name CA",
        "smoke_stride": 1,
        "smoke_max_frames": 500,
        "suggested_smallest_entry": "reference-vs-trajectory smoke benchmark",
        "pair_useful": "yes, first-vs-last CA pair already useful",
        "motion_regime": "small soluble protein equilibrium",
    },
    {
        "dataset_name": "NhaA_equilibrium",
        "local_dir": "data/NhaA_equilibrium",
        "topology_path": "/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro",
        "trajectory_path": "/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc",
        "suggested_selection": "protein and name CA",
        "smoke_stride": 25,
        "smoke_max_frames": 220,
        "suggested_smallest_entry": "reference-vs-trajectory smoke benchmark",
        "pair_useful": "yes, first-vs-last CA pair after smoke pass",
        "motion_regime": "membrane transporter equilibrium",
    },
    {
        "dataset_name": "YiiP_equilibrium_short",
        "local_dir": "data/YiiP_equilibrium",
        "topology_path": "/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb",
        "trajectory_path": "/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_9ns_center.xtc",
        "suggested_selection": "protein and name CA",
        "smoke_stride": 5,
        "smoke_max_frames": 220,
        "suggested_smallest_entry": "reference-vs-trajectory smoke benchmark",
        "pair_useful": "possibly, but trajectory benchmark is the primary minimal entry",
        "motion_regime": "membrane dimer equilibrium, shorter run",
    },
    {
        "dataset_name": "YiiP_equilibrium_long",
        "local_dir": "data/YiiP_equilibrium",
        "topology_path": "/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system.pdb",
        "trajectory_path": "/home/jjxu/MDAnalysis_data/yiip_equilibrium/YiiP_system_90ns_center.xtc",
        "suggested_selection": "protein and name CA",
        "smoke_stride": 50,
        "smoke_max_frames": 220,
        "suggested_smallest_entry": "reference-vs-trajectory smoke benchmark",
        "pair_useful": "low priority because short and long are structurally redundant for first-pass validation",
        "motion_regime": "membrane dimer equilibrium, longer run",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the local smoke benchmark runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "local_dataset_smoke_20260326",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT / "_tmp_local_dataset_smoke",
    )
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--robust-rule", type=str, default="huber")
    parser.add_argument("--robust-delta", type=float, default=0.25)
    parser.add_argument("--robust-max-iterations", type=int, default=20)
    parser.add_argument("--robust-tolerance", type=float, default=1e-6)
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset_name values to inspect/run.",
    )
    return parser.parse_args()


def _match_requested_datasets(names: list[str] | None) -> list[dict[str, object]]:
    """Filter the dataset registry by optional requested names."""

    if not names:
        return DATASETS
    selected = []
    allowed = {name for name in names}
    for dataset in DATASETS:
        if str(dataset["dataset_name"]) in allowed:
            selected.append(dataset)
    missing = sorted(allowed - {str(dataset["dataset_name"]) for dataset in selected})
    if missing:
        raise ValueError(f"unknown dataset_name values: {missing}")
    return selected


def _inspect_dataset(dataset: dict[str, object]) -> dict[str, object]:
    """Inspect one dataset and return a compact inventory row."""

    topology_path = Path(str(dataset["topology_path"]))
    trajectory_path = Path(str(dataset["trajectory_path"]))
    row: dict[str, object] = {
        "dataset_name": dataset["dataset_name"],
        "local_dir": dataset["local_dir"],
        "topology_path": str(topology_path),
        "trajectory_path": str(trajectory_path),
        "suggested_selection": dataset["suggested_selection"],
        "smoke_stride": dataset["smoke_stride"],
        "smoke_max_frames": dataset["smoke_max_frames"],
        "suggested_smallest_entry": dataset["suggested_smallest_entry"],
        "pair_useful": dataset["pair_useful"],
        "motion_regime": dataset["motion_regime"],
        "estimated_frame_count": None,
        "protein_ca_atom_count": None,
        "runnable_status": None,
        "notes": "",
    }

    if not topology_path.exists():
        row["runnable_status"] = "blocked_missing_topology"
        row["notes"] = "topology file is missing on disk"
        return row
    if not trajectory_path.exists():
        row["runnable_status"] = "blocked_missing_trajectory"
        row["notes"] = "trajectory file is missing on disk"
        return row

    try:
        universe = mda.Universe(str(topology_path), str(trajectory_path))
        selection = universe.select_atoms(str(dataset["suggested_selection"]))
    except Exception as exc:
        row["runnable_status"] = "blocked_loader_error"
        row["notes"] = repr(exc)
        return row

    row["estimated_frame_count"] = int(len(universe.trajectory))
    row["protein_ca_atom_count"] = int(selection.n_atoms)
    if selection.n_atoms <= 0:
        row["runnable_status"] = "blocked_empty_selection"
        row["notes"] = "selection produced zero atoms"
        return row

    smoke_frames = min(
        int(dataset["smoke_max_frames"]),
        int((len(universe.trajectory) + int(dataset["smoke_stride"]) - 1) // int(dataset["smoke_stride"])),
    )
    row["smoke_frame_count"] = smoke_frames
    row["runnable_status"] = "runnable"
    row["notes"] = "CA selection loads cleanly"
    return row


def _load_smoke_trajectory(dataset: dict[str, object]) -> tuple[Trajectory, list[str]]:
    """Load one smoke-scale trajectory subset for benchmarking."""

    data = load_mdanalysis_trajectory(
        topology_path=Path(str(dataset["topology_path"])),
        trajectory_path=Path(str(dataset["trajectory_path"])),
        selection=str(dataset["suggested_selection"]),
        stride=int(dataset["smoke_stride"]),
        max_frames=int(dataset["smoke_max_frames"]),
    )
    preprocessing_steps = [
        "source_format=local_mdanalysis_dataset",
        f"local_dir={dataset['local_dir']}",
        f"topology_path={dataset['topology_path']}",
        f"trajectory_path={dataset['trajectory_path']}",
        f"selection={dataset['suggested_selection']}",
        f"stride={dataset['smoke_stride']}",
        f"max_frames={dataset['smoke_max_frames']}",
        f"frames_after_preprocessing={data.coords.shape[0]}",
        f"atoms_after_selection={data.coords.shape[1]}",
        "reference_mode=first_frame",
        "benchmark_tier=smoke",
    ]
    return Trajectory(coords=data.coords), preprocessing_steps


def _inventory_to_markdown(rows: list[dict[str, object]]) -> str:
    """Render dataset inventory rows as markdown."""

    lines = [
        "# Local Dataset Inventory",
        "",
        "| dataset_name | topology_path | trajectory_path | suggested_selection | estimated_frame_count | protein_ca_atom_count | runnable_status | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset_name", "")),
                    str(row.get("topology_path", "")),
                    str(row.get("trajectory_path", "")),
                    str(row.get("suggested_selection", "")),
                    str(row.get("estimated_frame_count", "")),
                    str(row.get("protein_ca_atom_count", "")),
                    str(row.get("runnable_status", "")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _smoke_summary_to_markdown(rows: list[dict[str, object]]) -> str:
    """Render smoke summary rows as markdown."""

    lines = [
        "# Smoke Benchmark Summary",
        "",
        "| dataset_name | method_name | runtime_seconds | algorithm_runtime_seconds | overhead_seconds | unified_posthoc_rmsd | mdlovofit_native_all_atom_rmsd | mdlovofit_native_least_mobile_rmsd | weighted_rmsd | iteration_count | core_fraction | parsing_success_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset_name", "")),
                    str(row.get("method_name", "")),
                    str(row.get("runtime_seconds_mean", "")),
                    str(row.get("algorithm_runtime_seconds_mean", "")),
                    str(row.get("overhead_seconds_mean", "")),
                    str(row.get("unified_posthoc_rmsd_mean", "")),
                    str(row.get("mdlovofit_native_all_atom_rmsd_mean", "")),
                    str(row.get("mdlovofit_native_least_mobile_rmsd_mean", "")),
                    str(row.get("weighted_rmsd_mean", "")),
                    str(row.get("iteration_count_mean", "")),
                    str(row.get("core_fraction_mean", "")),
                    str(row.get("parsing_success_rate", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _recommendation_note(inventory_rows: list[dict[str, object]], smoke_rows: list[dict[str, object]]) -> str:
    """Write one concise recommendation note for the next formal dataset set."""

    runnable_names = [str(row["dataset_name"]) for row in inventory_rows if row["runnable_status"] == "runnable"]
    promoted: list[str] = []
    notes: list[str] = []

    for preferred in ["I-FABP with water", "Adk_equilibrium", "NhaA_equilibrium"]:
        if preferred in runnable_names:
            promoted.append(preferred)

    if "YiiP_equilibrium_short" in runnable_names:
        notes.append(
            "`YiiP_equilibrium_short` is the preferred YiiP representative because it is less redundant and cheaper than the long run."
        )
    if "YiiP_equilibrium_long" in runnable_names:
        notes.append(
            "`YiiP_equilibrium_long` should stay secondary unless long-timescale behavior is specifically needed; it is largely redundant with the short YiiP smoke case at this stage."
        )
    if any(str(row["dataset_name"]) == "Adk_transitions_DIMS" and row["runnable_status"] != "runnable" for row in inventory_rows):
        notes.append(
            "`Adk_transitions_DIMS` should not be promoted yet because the local topology file is missing and a safe topology substitution is not established."
        )

    lines = [
        "# Recommended Formal Multi-Dataset Benchmark Plan",
        "",
        "Recommended next formal dataset set:",
        "",
    ]
    for dataset_name in promoted:
        lines.append(f"- {dataset_name}")
    lines.extend(
        [
            "",
            "Rationale:",
            "",
            "- keep I-FABP as the continuity anchor because it is already the most validated case",
            "- keep Adk_equilibrium because it covers a distinct motion regime and already has interpretable results",
            "- add NhaA_equilibrium as the next membrane-protein external-validity case if the smoke run is clean",
            "- treat YiiP as optional extra breadth rather than part of the minimal formal set",
            "",
            "Dataset-specific notes:",
            "",
        ]
    )
    for note in notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "Suggested minimal formal progression:",
            "",
            "1. promote I-FABP, Adk_equilibrium, and NhaA_equilibrium to the next formal trajectory benchmark set",
            "2. add one CA-only structure-pair case from Adk_equilibrium",
            "3. add one CA-only structure-pair case from NhaA_equilibrium after the trajectory smoke pass is confirmed",
            "4. keep YiiP_short as a reserve case if one more membrane-protein example is needed",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Inspect local datasets and run smoke benchmarks on runnable cases."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    datasets = _match_requested_datasets(args.datasets)
    inventory_rows = [_inspect_dataset(dataset) for dataset in datasets]

    inventory_csv_path = args.output_dir / "dataset_inventory.csv"
    inventory_md_path = args.output_dir / "dataset_inventory.md"
    with inventory_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)
    inventory_md_path.write_text(_inventory_to_markdown(inventory_rows))

    soft_configuration = [
        {
            "method_name": "soft_robust_smoke",
            "core_fraction": args.fraction,
            "rule": args.robust_rule,
            "delta": args.robust_delta,
            "max_iterations": args.robust_max_iterations,
            "tolerance": args.robust_tolerance,
        }
    ]

    run_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for dataset, inventory_row in zip(datasets, inventory_rows, strict=True):
        if inventory_row["runnable_status"] != "runnable":
            continue
        trajectory, preprocessing_steps = _load_smoke_trajectory(dataset)
        results = compare_methods_with_soft_configurations(
            trajectory,
            mdlovofit_fraction=args.fraction,
            workdir=args.workdir / str(dataset["dataset_name"]),
            soft_configurations=soft_configuration,
            mdlovofit_executable=args.mdlovofit_executable,
            mdlovofit_config=args.mdlovofit_config,
        )
        run_rows.extend(
            benchmark_results_to_rows(
                results,
                benchmark_mode="local_smoke",
                dataset_name=str(dataset["dataset_name"]),
                repeat_index=0,
                preprocessing_steps=preprocessing_steps,
            )
        )
        run_records.append(
            {
                "benchmark_mode": "local_smoke",
                "dataset_name": str(dataset["dataset_name"]),
                "preprocessing_steps": preprocessing_steps,
                "results": [benchmark_result_to_record(result) for result in results],
            }
        )

    smoke_runs_csv_path = args.output_dir / "smoke_runs.csv"
    smoke_runs_json_path = args.output_dir / "smoke_runs.json"
    smoke_summary_csv_path = args.output_dir / "smoke_summary.csv"
    smoke_summary_md_path = args.output_dir / "smoke_summary.md"
    recommendation_md_path = args.output_dir / "recommended_formal_benchmark_set.md"

    if run_rows:
        summary_rows = summarize_benchmark_rows(run_rows)
        write_benchmark_csv(smoke_runs_csv_path, run_rows)
        write_benchmark_csv(smoke_summary_csv_path, summary_rows)
        write_benchmark_json(
            smoke_runs_json_path,
            {
                "inventory": inventory_rows,
                "records": run_records,
            },
        )
        smoke_summary_md_path.write_text(_smoke_summary_to_markdown(summary_rows))
        recommendation_md_path.write_text(_recommendation_note(inventory_rows, summary_rows))
    else:
        smoke_summary_md_path.write_text("# Smoke Benchmark Summary\n\nNo runnable datasets completed.\n")
        recommendation_md_path.write_text(
            "# Recommended Formal Multi-Dataset Benchmark Plan\n\nNo runnable datasets completed.\n"
        )

    print(f"dataset_inventory_csv={inventory_csv_path}")
    print(f"dataset_inventory_md={inventory_md_path}")
    print(f"smoke_runs_csv={smoke_runs_csv_path}")
    print(f"smoke_runs_json={smoke_runs_json_path}")
    print(f"smoke_summary_csv={smoke_summary_csv_path}")
    print(f"smoke_summary_md={smoke_summary_md_path}")
    print(f"recommendation_md={recommendation_md_path}")


if __name__ == "__main__":
    main()
