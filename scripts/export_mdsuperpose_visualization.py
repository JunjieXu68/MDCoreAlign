"""Generate minimal visualization-ready exports for mdsuperpose and MDLovoFit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (
    Trajectory,
    export_external_mdlovofit_visualization,
    export_soft_trajectory_visualization,
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
        "paper_raw_json": ROOT / "results" / "paper_ready_20260326" / "raw" / "trajectory_ifabp" / "ifabp_real_runs.json",
    },
    "adk_equilibrium": {
        "label": "ADK equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/adk4AKE.psf"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"),
        "selection": "protein and name CA",
        "stride": 1,
        "max_frames": 500,
        "paper_raw_json": ROOT / "results" / "paper_ready_20260326" / "raw" / "trajectory_adk" / "adk_equilibrium_real_runs.json",
    },
    "nhaa_equilibrium": {
        "label": "NhaA equilibrium",
        "topology": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.gro"),
        "trajectory": Path("/home/jjxu/MDAnalysis_data/nhaa_equilibrium/NhaA_non_water.xtc"),
        "selection": "protein and name CA",
        "stride": 25,
        "max_frames": 201,
        "paper_raw_json": ROOT
        / "results"
        / "higher_tier_validation_batch1_smoke"
        / "nhaa_trajectory"
        / "nhaa_equilibrium_smoke_real_runs.json",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse export-script arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *sorted(DATASETS)], default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper_ready_20260326" / "exports",
    )
    parser.add_argument("--core-fraction", type=float, default=0.6)
    return parser.parse_args()


def load_external_run(raw_json_path: Path) -> dict[str, object]:
    """Load the first external MDLovoFit result from one raw benchmark payload."""

    payload = json.loads(raw_json_path.read_text())
    records = payload.get("records", [])
    for record in records:
        for result in record["results"]:
            if result["method_name"] == "external_mdlovofit":
                return result
    raise ValueError(f"no external_mdlovofit result was found in {raw_json_path}")


def write_external_inspection_markdown(path: Path, dataset_rows: list[dict[str, object]]) -> None:
    """Write one compact markdown summary of external export inspection findings."""

    def summarize_values(values: list[object]) -> str:
        """Return a compact summary string for one numeric value list."""

        if not values:
            return "[]"
        if len(values) <= 8:
            return str(values)
        return (
            f"count={len(values)}, min={values[0]}, median~={values[len(values) // 2]}, "
            f"max={values[-1]}"
        )

    lines = [
        "# MDLovoFit Export Inspection Summary",
        "",
        "The observations below come from the preserved MDLovoFit aligned output PDB files used for export.",
        "",
        "| Dataset | Occupancy unique values | Binary occupancy | Constant across frames | Reconstructed subset count | Reconstructed subset fraction | B-factor summary | B-factor constant across frames |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in dataset_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["occupancy_unique_values"]),
                    str(row["occupancy_mask_binary"]),
                    str(row["occupancy_mask_constant_across_frames"]),
                    str(row["output_subset_atom_count"]),
                    str(row["output_subset_atom_fraction"]),
                    summarize_values(list(row["bfactor_unique_values"])),
                    str(row["bfactor_constant_across_frames"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation used in this project:",
            "",
            "- occupancy is preserved directly from MDLovoFit output and used as the stable-subset mask when binary and constant",
            "- B-factor is preserved as-is; no additional semantics are assigned unless MDLovoFit itself documents them",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Generate the requested visualization exports for one or both named datasets."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]

    inspection_rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]
        export_dir = args.output_dir / dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)

        data = load_mdanalysis_trajectory(
            topology_path=Path(dataset["topology"]),
            trajectory_path=Path(dataset["trajectory"]),
            selection=str(dataset["selection"]),
            stride=int(dataset.get("stride", 1)),
            max_frames=dataset.get("max_frames"),
        )
        trajectory = Trajectory(coords=data.coords)

        soft_artifacts = export_soft_trajectory_visualization(
            trajectory,
            export_dir / dataset_name,
            core_fraction=args.core_fraction,
            residue_labels=data.residue_ids,
            atom_names=data.atom_names,
            group_ids=data.atom_to_residue,
            rule="huber",
            delta=0.25,
            reference_mode="first_frame",
            warm_start=False,
            bfactor_field="mobility_score",
        )

        external_run = load_external_run(Path(dataset["paper_raw_json"]))
        external_artifacts = export_external_mdlovofit_visualization(
            external_run["extra_metadata"]["output_path"],
            export_dir / dataset_name,
            n_atoms=trajectory.n_atoms,
            residue_labels=data.residue_ids,
            atom_names=data.atom_names,
            algorithm_runtime_seconds=external_run["runtime_seconds"],
            benchmark_overhead_seconds=external_run["overhead_seconds"],
        )

        inspection = json.loads(external_artifacts.summary_json_path.read_text())
        inspection_rows.append(
            {
                "dataset": dataset["label"],
                "occupancy_unique_values": inspection["occupancy_unique_values"],
                "occupancy_mask_binary": inspection["occupancy_mask_binary"],
                "occupancy_mask_constant_across_frames": inspection["occupancy_mask_constant_across_frames"],
                "output_subset_atom_count": inspection["output_subset_atom_count"],
                "output_subset_atom_fraction": inspection["output_subset_atom_fraction"],
                "bfactor_unique_values": inspection["bfactor_unique_values"],
                "bfactor_constant_across_frames": inspection["bfactor_constant_across_frames"],
            }
        )

        print(f"{dataset_name}_soft_occupancy_pdb={soft_artifacts.occupancy_pdb_path}")
        print(f"{dataset_name}_soft_bfactor_pdb={soft_artifacts.bfactor_pdb_path}")
        print(f"{dataset_name}_soft_csv={soft_artifacts.csv_path}")
        print(f"{dataset_name}_external_occupancy_pdb={external_artifacts.occupancy_pdb_path}")
        print(f"{dataset_name}_external_csv={external_artifacts.csv_path}")
        print(f"{dataset_name}_external_summary={external_artifacts.summary_json_path}")

    inspection_summary_path = args.output_dir / "mdlovofit_export_inspection_summary.md"
    write_external_inspection_markdown(inspection_summary_path, inspection_rows)
    print(f"mdlovofit_export_inspection_summary={inspection_summary_path}")


if __name__ == "__main__":
    main()
