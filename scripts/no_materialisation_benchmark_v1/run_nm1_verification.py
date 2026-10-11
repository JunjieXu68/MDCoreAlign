#!/usr/bin/env python3
"""Verify that removing -t changes only stock MDLovoFit file materialisation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.no_materialisation_benchmark_v1.common import (
    OUT,
    ROOT,
    run_external,
    sha256,
    stock_command,
    write_json,
    write_text,
    write_tsv,
)


LOCKED_CASES = (
    {"case": "153lA00", "role": "small_mdCATH", "kind": "mdcath", "domain": "153lA00"},
    {"case": "1jnr_C_f00500", "role": "medium_ATLAS", "kind": "atlas", "system": "1jnr_C", "prefix": "f00500"},
    {"case": "6sup_A_f00250", "role": "large_ATLAS", "kind": "atlas", "system": "6sup_A", "prefix": "f00250"},
    {"case": "1hm7B00", "role": "prior_PDB_range_related", "kind": "mdcath", "domain": "1hm7B00"},
    {"case": "1fzrA00", "role": "locked_geometric_tail", "kind": "mdcath", "domain": "1fzrA00"},
)
METRICS = (
    "native_least_mobile_RMSD_A",
    "native_most_mobile_RMSD_A",
    "native_all_atom_RMSD_A",
    "native_fraction",
    "native_total_atoms",
    "native_selection_atoms",
    "frame_rows",
)


def _mdcath_input(domain: str, destination: Path) -> tuple[int, int, str]:
    checkpoint = (
        ROOT
        / "results/database_scale_benchmark_v2/mdlovofit_comparison/all_domain_checkpoints"
        / f"{domain}__T320__R1.json"
    )
    task = json.loads(checkpoint.read_text())
    coords, _, _, metadata = extract_ca_trajectory(
        ROOT / task["source_file"],
        coordinate_path=task["coordinate_path"],
        direct_contiguous_read=True,
    )
    values = np.asarray(coords, dtype=np.float64)
    # Registered PDB-safe convention removes translation only and is shared by
    # both verification commands.
    values = values - np.mean(values, axis=1, keepdims=True)
    write_multimodel_ca_pdb_streaming(values, destination)
    return values.shape[0], values.shape[1], metadata["unit_status"]


def _input(case: dict[str, str], workspace: Path) -> tuple[Path, int, int, str]:
    if case["kind"] == "atlas":
        path = (
            ROOT
            / "results/atlas_large_long_v1/a3_external/centered_inputs"
            / case["system"]
            / f"{case['prefix']}_centered_CA.pdb"
        )
        manifest = {
            (row.split("\t")[0], row.split("\t")[1]): row.split("\t")
            for row in (
                ROOT / "results/atlas_large_long_v1/a3_external/centered_input_manifest.tsv"
            ).read_text().splitlines()[1:]
        }
        row = manifest[(case["system"], case["prefix"])]
        return path, int(row[2]), int(row[3]), "verified_existing_centered_PDB"
    path = workspace / f"{case['case']}_centered_CA.pdb"
    frames, residues, unit = _mdcath_input(case["domain"], path)
    return path, frames, residues, f"deterministic_reconstruction_{unit}"


def main() -> None:
    write_tsv(OUT / "01_stock_verification/locked_verification_panel.tsv", LOCKED_CASES)
    rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nm1-stock-") as temporary:
        root = Path(temporary)
        for case in LOCKED_CASES:
            case_root = root / case["case"]
            case_root.mkdir()
            input_path, frames, residues, provenance = _input(case, case_root)
            output_path = case_root / "aligned_output.pdb"
            before = {item.name for item in case_root.iterdir()}
            with_output = run_external(
                stock_command(input_path, output_path), cwd=case_root, timeout=1800
            )
            output_exists = output_path.is_file()
            output_bytes = output_path.stat().st_size if output_exists else 0
            output_path.unlink(missing_ok=True)
            after_cleanup = {item.name for item in case_root.iterdir()}
            no_output = run_external(
                stock_command(input_path, None), cwd=case_root, timeout=1800
            )
            after_no_output = {item.name for item in case_root.iterdir()}
            hidden_created = sorted(after_no_output - after_cleanup)
            differences = {}
            identical = True
            for metric in METRICS:
                left = with_output.get(metric)
                right = no_output.get(metric)
                if isinstance(left, float) and isinstance(right, float):
                    difference = abs(left - right)
                    same = difference <= 5e-5
                else:
                    difference = 0 if left == right else 1
                    same = left == right
                differences[metric] = difference
                identical &= same
            identical &= with_output["status"] == no_output["status"] == "success"
            identical &= with_output["return_code"] == no_output["return_code"]
            identical &= output_exists and not hidden_created
            rows.append(
                {
                    **case,
                    "n_frames": frames,
                    "n_residues": residues,
                    "input_provenance": provenance,
                    "input_size_bytes": input_path.stat().st_size,
                    "input_sha256": sha256(input_path),
                    "with_t_status": with_output["status"],
                    "without_t_status": no_output["status"],
                    "with_t_return_code": with_output["return_code"],
                    "without_t_return_code": no_output["return_code"],
                    "with_t_wall_seconds": with_output["wall_seconds"],
                    "without_t_wall_seconds": no_output["wall_seconds"],
                    "least_mobile_RMSD_with_t_A": with_output.get("native_least_mobile_RMSD_A"),
                    "least_mobile_RMSD_without_t_A": no_output.get("native_least_mobile_RMSD_A"),
                    "most_mobile_RMSD_with_t_A": with_output.get("native_most_mobile_RMSD_A"),
                    "most_mobile_RMSD_without_t_A": no_output.get("native_most_mobile_RMSD_A"),
                    "all_atom_RMSD_with_t_A": with_output.get("native_all_atom_RMSD_A"),
                    "all_atom_RMSD_without_t_A": no_output.get("native_all_atom_RMSD_A"),
                    "frame_rows_with_t": with_output.get("frame_rows"),
                    "frame_rows_without_t": no_output.get("frame_rows"),
                    "maximum_numeric_difference": max(differences.values()),
                    "no_output_numerical_identity": identical,
                }
            )
            file_rows.append(
                {
                    "case": case["case"],
                    "aligned_output_created_with_t": output_exists,
                    "aligned_output_bytes": output_bytes,
                    "files_created_without_t": ";".join(hidden_created),
                    "hidden_NONE_file_created": "NONE" in hidden_created,
                    "only_coordinate_output_behavior_changed": identical,
                    "workspace_initial_files": ";".join(sorted(before)),
                }
            )
    write_tsv(OUT / "01_stock_verification/no_output_verification.tsv", rows)
    write_tsv(OUT / "01_stock_verification/file_creation_audit.tsv", file_rows)
    passed = all(bool(row["no_output_numerical_identity"]) for row in rows)
    write_json(
        OUT / "01_stock_verification/nm1_status.json",
        {"status": "pass" if passed else "fail", "cases": len(rows), "all_identical": passed},
    )
    write_text(
        OUT / "01_stock_verification/no_output_verification_report.md",
        f"""# Stock MDLovoFit No-output Verification

Status: **{'PASS' if passed else 'FAIL'}** across {len(rows)} prospectively
locked cases. Removing `-t` created no aligned or hidden trajectory file and
left printed least-mobile, most-mobile and all-selection RMSDs, selected
fraction, atom counts, frame rows and exit status unchanged within the
program's printed precision. The stock executable was not modified.
""",
    )
    if not passed:
        raise RuntimeError("NM1 identity gate failed; do not continue")


if __name__ == "__main__":
    main()
