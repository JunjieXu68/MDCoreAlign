"""Proof-of-feasibility smoke benchmark for one CA-only NMR ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import MDAnalysis as mda
import numpy as np

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
    load_trajectory_from_multimodel_pdb,
    run_mdlovofit_on_files,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
    write_multimodel_pdb,
)


DEFAULT_PDB_ID = "1D3Z"
DEFAULT_URL = "https://files.rcsb.org/download/1D3Z.pdb"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the NMR ensemble smoke runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-id", type=str, default=DEFAULT_PDB_ID)
    parser.add_argument("--pdb-url", type=str, default=DEFAULT_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nmr_ensemble_feasibility_20260326",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT / "_tmp_nmr_ensemble_smoke",
    )
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--robust-rule", type=str, default="huber")
    parser.add_argument("--robust-delta", type=float, default=0.25)
    parser.add_argument("--robust-max-iterations", type=int, default=20)
    parser.add_argument("--robust-tolerance", type=float, default=1e-6)
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def _download_pdb(path: Path, url: str) -> None:
    """Download one PDB file unless it is already present."""

    if path.exists():
        return
    path.write_bytes(urlopen(url).read())


def _inspect_nmr_ensemble(pdb_path: Path) -> dict[str, object]:
    """Inspect one NMR multi-model PDB for model consistency and CA usability."""

    universe = mda.Universe(str(pdb_path))
    ca = universe.select_atoms("protein and name CA")
    first_signature = [
        (atom.segid, int(atom.resid), atom.resname, atom.name)
        for atom in universe.atoms
    ]
    atom_order_consistent = True
    model_atom_counts: list[int] = []
    for _ts in universe.trajectory:
        current_signature = [
            (atom.segid, int(atom.resid), atom.resname, atom.name)
            for atom in universe.atoms
        ]
        if current_signature != first_signature:
            atom_order_consistent = False
        model_atom_counts.append(universe.atoms.n_atoms)

    ca_coords = []
    for _ts in universe.trajectory:
        ca_coords.append(ca.positions.copy())
    ca_coords = np.asarray(ca_coords, dtype=float)

    return {
        "pdb_id": pdb_path.stem,
        "source_pdb_path": str(pdb_path),
        "n_models": int(len(universe.trajectory)),
        "n_atoms_total": int(universe.atoms.n_atoms),
        "n_ca_atoms": int(ca.n_atoms),
        "ca_selection_clean": bool(ca.n_atoms > 0 and np.isfinite(ca_coords).all()),
        "atom_order_consistent": bool(atom_order_consistent),
        "all_models_same_atom_count": bool(len(set(model_atom_counts)) == 1),
        "ca_coords_shape": list(ca_coords.shape),
        "chain_ids": sorted({str(segid) for segid in ca.segids}),
        "resid_first": int(ca.residues.resids[0]),
        "resid_last": int(ca.residues.resids[-1]),
    }


def _extract_ca_coords(pdb_path: Path) -> np.ndarray:
    """Extract CA coordinates from each model of one multi-model PDB."""

    universe = mda.Universe(str(pdb_path))
    ca = universe.select_atoms("protein and name CA")
    return np.asarray([ca.positions.copy() for _ts in universe.trajectory], dtype=float)


def _write_markdown_feasibility_note(
    path: Path,
    *,
    inspection: dict[str, object],
    ca_multimodel_path: Path,
    benchmark_summary_rows: list[dict[str, object]],
    direct_mdlovofit_result: dict[str, object],
) -> None:
    """Write one concise NMR ensemble feasibility note."""

    lines = [
        "# NMR Ensemble Feasibility Note",
        "",
        "This test treats an NMR ensemble as a multi-conformation structure set, not as an MD trajectory.",
        "",
        "## Tested case",
        "",
        f"- PDB ID: `{inspection['pdb_id']}`",
        f"- Source PDB: `{inspection['source_pdb_path']}`",
        f"- Models: `{inspection['n_models']}`",
        f"- Total atoms per model: `{inspection['n_atoms_total']}`",
        f"- CA atoms per model: `{inspection['n_ca_atoms']}`",
        f"- Atom ordering consistent across models: `{inspection['atom_order_consistent']}`",
        f"- CA-only selection clean: `{inspection['ca_selection_clean']}`",
        f"- CA-only multi-model PDB used for smoke benchmark: `{ca_multimodel_path}`",
        "",
        "## Technical interpretation",
        "",
        "- The current internal pipeline can accept a multi-model PDB directly through `load_trajectory_from_multimodel_pdb(...)`.",
        "- For the first NMR feasibility pass, CA-only preprocessing is still recommended because the current internal multi-model PDB loader does not apply atom selections by itself.",
        "- External MDLovoFit can be run on the CA-only multi-model PDB as a PDB-trajectory-like input.",
        "- This input should be treated as a separate `NMR ensemble benchmark` or `multi-conformation ensemble benchmark`, not as an MD trajectory benchmark.",
        "",
        "## Direct MDLovoFit file check",
        "",
        f"- direct_run_return_code: `{direct_mdlovofit_result['return_code']}`",
        f"- direct_run_runtime_seconds: `{direct_mdlovofit_result['runtime_seconds']}`",
        f"- direct_run_native_all_atom_rmsd: `{direct_mdlovofit_result['native_all_atom_rmsd']}`",
        f"- direct_run_native_least_mobile_rmsd: `{direct_mdlovofit_result['native_least_mobile_rmsd']}`",
        "",
        "## Smoke benchmark summary",
        "",
        "| method_name | runtime_seconds | algorithm_runtime_seconds | overhead_seconds | unified_posthoc_rmsd | mdlovofit_native_all_atom_rmsd | mdlovofit_native_least_mobile_rmsd | weighted_rmsd | iteration_count | core_fraction | parsing_success_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in benchmark_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
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
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use NMR ensembles as a separate `NMR ensemble benchmark` class.",
            "- Keep the first pass CA-only and multi-model-PDB based.",
            "- Do not fold these cases into the MD trajectory benchmark group because model order is not physical time order.",
            "- If later expanded, compare them against both the trajectory-style benchmark and structure-pair benchmark narratives, but keep the reporting category distinct.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run a proof-of-feasibility NMR ensemble smoke benchmark."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    source_pdb_path = args.output_dir / f"{args.pdb_id}.pdb"
    ca_multimodel_path = args.output_dir / f"{args.pdb_id}_ca_ensemble.pdb"
    inspection_json_path = args.output_dir / "nmr_ensemble_inspection.json"
    test_config_json_path = args.output_dir / "nmr_ensemble_test_config.json"
    smoke_runs_csv_path = args.output_dir / "nmr_ensemble_smoke_runs.csv"
    smoke_runs_json_path = args.output_dir / "nmr_ensemble_smoke_runs.json"
    smoke_summary_csv_path = args.output_dir / "nmr_ensemble_smoke_summary.csv"
    smoke_summary_md_path = args.output_dir / "nmr_ensemble_smoke_summary.md"
    feasibility_note_path = args.output_dir / "nmr_ensemble_feasibility_note.md"

    _download_pdb(source_pdb_path, args.pdb_url)
    inspection = _inspect_nmr_ensemble(source_pdb_path)
    write_benchmark_json(inspection_json_path, inspection)

    ca_coords = _extract_ca_coords(source_pdb_path)
    write_multimodel_pdb(ca_coords, ca_multimodel_path)
    reloaded = load_trajectory_from_multimodel_pdb(ca_multimodel_path)
    if reloaded.coords.shape != ca_coords.shape:
        raise RuntimeError("reloaded CA-only multi-model PDB shape did not match extracted CA coordinates")

    direct_run = run_mdlovofit_on_files(
        [ca_multimodel_path],
        fraction=args.fraction,
        output_path=args.workdir / "nmr_direct_aligned.pdb",
        executable=args.mdlovofit_executable,
        config_path=args.mdlovofit_config,
    )
    direct_mdlovofit_result = {
        "return_code": direct_run.return_code,
        "runtime_seconds": direct_run.total_runtime_seconds,
        "native_all_atom_rmsd": direct_run.parsed_output.average_rmsd_all_atoms,
        "native_least_mobile_rmsd": direct_run.parsed_output.average_rmsd_least_mobile,
        "reported_fraction": direct_run.parsed_output.fraction_of_best_aligned_atoms,
        "output_path": str(direct_run.output_path),
    }

    results = compare_methods_with_soft_configurations(
        Trajectory(coords=ca_coords),
        mdlovofit_fraction=args.fraction,
        workdir=args.workdir / "benchmark_compare",
        soft_configurations=[
            {
                "method_name": "soft_robust_nmr_ensemble",
                "core_fraction": args.fraction,
                "rule": args.robust_rule,
                "delta": args.robust_delta,
                "max_iterations": args.robust_max_iterations,
                "tolerance": args.robust_tolerance,
            }
        ],
        mdlovofit_executable=args.mdlovofit_executable,
        mdlovofit_config=args.mdlovofit_config,
    )

    preprocessing_steps = [
        "benchmark_class=nmr_ensemble",
        "interpretation=multi_conformation_ensemble_not_md_trajectory",
        f"source_pdb={source_pdb_path}",
        "selection=protein and name CA",
        "input_format=multi_model_pdb",
        f"n_models={inspection['n_models']}",
        f"n_ca_atoms={inspection['n_ca_atoms']}",
        "reference_mode=first_model",
    ]
    run_rows = benchmark_results_to_rows(
        results,
        benchmark_mode="nmr_ensemble",
        dataset_name=args.pdb_id,
        repeat_index=0,
        preprocessing_steps=preprocessing_steps,
    )
    summary_rows = summarize_benchmark_rows(run_rows)

    write_benchmark_csv(smoke_runs_csv_path, run_rows)
    write_benchmark_csv(smoke_summary_csv_path, summary_rows)
    write_benchmark_json(
        smoke_runs_json_path,
        {
            "inspection": inspection,
            "direct_mdlovofit_result": direct_mdlovofit_result,
            "preprocessing_steps": preprocessing_steps,
            "results": [benchmark_result_to_record(result) for result in results],
        },
    )
    smoke_summary_md_path.write_text(
        "\n".join(
            [
                "# NMR Ensemble Smoke Summary",
                "",
                f"- dataset: `{args.pdb_id}`",
                "- benchmark class: `nmr_ensemble`",
                "- interpretation: `multi-conformation ensemble, not an MD trajectory`",
                "",
            ]
        )
        + _write_table_string(summary_rows)
    )
    _write_markdown_feasibility_note(
        feasibility_note_path,
        inspection=inspection,
        ca_multimodel_path=ca_multimodel_path,
        benchmark_summary_rows=summary_rows,
        direct_mdlovofit_result=direct_mdlovofit_result,
    )
    write_benchmark_json(
        test_config_json_path,
        {
            "pdb_id": args.pdb_id,
            "source_pdb_path": str(source_pdb_path),
            "ca_multimodel_path": str(ca_multimodel_path),
            "selection": "protein and name CA",
            "benchmark_class": "nmr_ensemble",
            "interpretation": "multi_conformation_ensemble_not_md_trajectory",
            "fraction": args.fraction,
            "soft_method": {
                "rule": args.robust_rule,
                "delta": args.robust_delta,
                "max_iterations": args.robust_max_iterations,
                "tolerance": args.robust_tolerance,
            },
        },
    )

    print(f"source_pdb={source_pdb_path}")
    print(f"ca_multimodel_pdb={ca_multimodel_path}")
    print(f"inspection_json={inspection_json_path}")
    print(f"test_config_json={test_config_json_path}")
    print(f"smoke_summary_csv={smoke_summary_csv_path}")
    print(f"smoke_summary_md={smoke_summary_md_path}")
    print(f"feasibility_note={feasibility_note_path}")


def _write_table_string(summary_rows: list[dict[str, object]]) -> str:
    """Render one summary-row list as a markdown table string."""

    lines = [
        "| method_name | runtime_seconds | algorithm_runtime_seconds | overhead_seconds | unified_posthoc_rmsd | mdlovofit_native_all_atom_rmsd | mdlovofit_native_least_mobile_rmsd | weighted_rmsd | iteration_count | core_fraction | parsing_success_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
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


if __name__ == "__main__":
    main()
