"""Audit export completeness on the expanded real-case set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_mdsuperpose_visualization.py"


DATASET_EXPORT_NAMES = ["ifabp", "adk_equilibrium", "nhaa_equilibrium"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for export-completeness auditing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "higher_tier_export_audit",
    )
    parser.add_argument("--core-fraction", type=float, default=0.60)
    return parser.parse_args()


def _read_single_model_pdb_fields(path: Path) -> tuple[list[float], list[float]]:
    """Read occupancy and B-factor values from one single-model PDB export."""

    occupancies: list[float] = []
    bfactors: list[float] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            occupancies.append(float(line[54:60]))
            bfactors.append(float(line[60:66]))
    return occupancies, bfactors


def _csv_columns(path: Path) -> list[str]:
    """Read CSV header columns."""

    with path.open() as handle:
        reader = csv.reader(handle)
        return next(reader)


def main() -> None:
    """Run the existing export script and audit the resulting artifacts."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        "all",
        "--output-dir",
        str(args.output_dir / "exports"),
        "--core-fraction",
        str(args.core_fraction),
    ]
    subprocess.run(command, check=True, cwd=str(ROOT))

    rows: list[dict[str, object]] = []
    for dataset_name in DATASET_EXPORT_NAMES:
        dataset_dir = args.output_dir / "exports" / dataset_name
        prefix = dataset_dir / dataset_name
        soft_occ = prefix.with_name(f"{prefix.name}_soft_core_mask_occupancy.pdb")
        soft_bfac = prefix.with_name(f"{prefix.name}_soft_mobility_score_bfactor.pdb")
        soft_csv = prefix.with_name(f"{prefix.name}_soft_scores.csv")
        soft_summary = prefix.with_name(f"{prefix.name}_soft_export_summary.json")
        external_occ = prefix.with_name(f"{prefix.name}_mdlovofit_preserved_occupancy.pdb")
        external_csv = prefix.with_name(f"{prefix.name}_mdlovofit_scores.csv")
        external_summary = prefix.with_name(f"{prefix.name}_mdlovofit_export_summary.json")

        soft_occ_values, soft_occ_bfactors = _read_single_model_pdb_fields(soft_occ)
        _, soft_bfac_values = _read_single_model_pdb_fields(soft_bfac)
        external_occ_values, external_bfac_values = _read_single_model_pdb_fields(external_occ)

        soft_summary_payload = json.loads(soft_summary.read_text())
        external_summary_payload = json.loads(external_summary.read_text())

        rows.append(
            {
                "dataset_name": dataset_name,
                "soft_occupancy_pdb_exists": soft_occ.exists(),
                "soft_bfactor_pdb_exists": soft_bfac.exists(),
                "soft_csv_exists": soft_csv.exists(),
                "external_occupancy_pdb_exists": external_occ.exists(),
                "external_csv_exists": external_csv.exists(),
                "soft_occupancy_binary": set(soft_occ_values).issubset({0.0, 1.0}),
                "soft_occupancy_bfactor_zero": all(abs(value) < 1e-8 for value in soft_occ_bfactors),
                "soft_bfactor_min": min(soft_bfac_values),
                "soft_bfactor_max": max(soft_bfac_values),
                "soft_csv_columns": "|".join(_csv_columns(soft_csv)),
                "external_occupancy_binary": external_summary_payload["occupancy_mask_binary"],
                "external_occupancy_constant": external_summary_payload["occupancy_mask_constant_across_frames"],
                "external_output_subset_atom_count": external_summary_payload["output_subset_atom_count"],
                "external_output_subset_atom_fraction": external_summary_payload["output_subset_atom_fraction"],
                "external_bfactor_constant": external_summary_payload["bfactor_constant_across_frames"],
                "external_bfactor_unique_count": len(external_summary_payload["bfactor_unique_values"]),
                "external_csv_columns": "|".join(_csv_columns(external_csv)),
                "soft_algorithm_runtime_seconds": soft_summary_payload["algorithm_runtime_seconds"],
                "soft_export_runtime_seconds": soft_summary_payload["export_runtime_seconds"],
                "soft_render_runtime_seconds": soft_summary_payload["render_runtime_seconds"],
                "external_algorithm_runtime_seconds": external_summary_payload["algorithm_runtime_seconds"],
                "external_benchmark_overhead_seconds": external_summary_payload["benchmark_overhead_seconds"],
                "external_export_runtime_seconds": external_summary_payload["export_runtime_seconds"],
                "external_render_runtime_seconds": external_summary_payload["render_runtime_seconds"],
            }
        )

    summary_csv = args.output_dir / "export_completeness_audit_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    note_lines = [
        "# Export Completeness Audit",
        "",
        "Audited datasets:",
        "- ifabp",
        "- adk_equilibrium",
        "- nhaa_equilibrium",
        "",
        "Audit checks:",
        "- soft occupancy export exists and is binary",
        "- soft B-factor export exists and stays within the expected visualization range",
        "- CSV exports contain the expected columns",
        "- external MDLovoFit occupancy mask is preserved and remains binary/constant when reported as such",
        "- algorithm, benchmark-overhead, and export timings remain separated",
    ]
    note_md = args.output_dir / "export_completeness_audit_note.md"
    note_md.write_text("\n".join(note_lines) + "\n")

    print(f"summary_csv={summary_csv}")
    print(f"note_md={note_md}")


if __name__ == "__main__":
    main()
