#!/usr/bin/env python3
"""Register MA contracts and validate the untimed MDLovoFit exporter."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

from scripts.no_materialisation_benchmark_v1.common import (
    OUT,
    ROOT,
    STOCK,
    run_external,
    sha256,
    write_json,
    write_text,
    write_tsv,
)
from scripts.no_materialisation_benchmark_v1.run_nm1_verification import _input


MA = OUT / "method_agreement_v2"
EXPORTER = MA / "01_mdlovofit_export_validation/bin/mdlovofit_export"
FRACTION = 0.60
SUBDIRS = (
    "00_contracts/panel_manifests",
    "01_mdlovofit_export_validation",
    "02_legacy_four_systems",
    "03_atlas_60pct",
    "04_mdcath_60pct",
    "05_fraction_sensitivity",
    "06_alignment_outputs",
    "07_independent_stability",
    "08_reporting/figures",
    "handoff_to_chatgpt",
)

VALIDATION_CASES = (
    {"case": "153lA00", "role": "small_mdCATH", "kind": "mdcath", "domain": "153lA00"},
    {"case": "1jnr_C_f00500", "role": "medium_ATLAS", "kind": "atlas", "system": "1jnr_C", "prefix": "f00500"},
    {"case": "6sup_A_f00250", "role": "large_ATLAS", "kind": "atlas", "system": "6sup_A", "prefix": "f00250"},
    {"case": "ifabp_historical", "role": "historical_four_system", "kind": "existing_pdb", "path": "results/reviewer2_hardening/theseus_rerun/inputs/ifabp_ca_multimodel.pdb"},
    {"case": "1fzrA00", "role": "locked_geometric_tail", "kind": "mdcath", "domain": "1fzrA00"},
    {"case": "1hm7B00", "role": "PDB_range_remediated", "kind": "mdcath", "domain": "1hm7B00"},
    {"case": "nhaa_historical", "role": "multi_chain_membrane", "kind": "existing_pdb", "path": "results/reviewer2_hardening/theseus_rerun/inputs/nhaa_ca_multimodel.pdb"},
    {"case": "yiip_short_historical", "role": "multi_chain_edge", "kind": "existing_pdb", "path": "results/reviewer2_hardening/theseus_rerun/inputs/yiip_short_ca_multimodel.pdb"},
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _parse_pdb(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, str]]]:
    frames: list[list[list[float]]] = []
    occupancies: list[list[float]] = []
    metadata: list[tuple[str, str, str]] = []
    current: list[list[float]] = []
    current_occupancy: list[float] = []
    for line in path.open():
        if line.startswith(("ATOM", "HETATM")):
            current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            current_occupancy.append(float(line[54:60]))
            if not metadata:
                # Metadata are captured below after the first frame closes.
                pass
        elif line.startswith(("END", "ENDMDL")) and current:
            if not metadata:
                first_lines = []
                # Re-read exactly the first frame to preserve fixed-column metadata.
                for first in path.open():
                    if first.startswith(("ATOM", "HETATM")):
                        first_lines.append(first)
                    elif first.startswith(("END", "ENDMDL")) and first_lines:
                        break
                metadata = [
                    (line_[21:22].strip(), line_[22:27].strip(), line_[17:20].strip())
                    for line_ in first_lines
                ]
            frames.append(current)
            occupancies.append(current_occupancy)
            current = []
            current_occupancy = []
    if current:
        frames.append(current)
        occupancies.append(current_occupancy)
    coords = np.asarray(frames, dtype=np.float64)
    occ = np.asarray(occupancies, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"invalid PDB coordinate shape for {path}: {coords.shape}")
    if not all(len(frame) == coords.shape[1] for frame in frames):
        raise ValueError(f"inconsistent atom count in {path}")
    return coords, occ, metadata


def _existing_input(case: dict[str, str]) -> tuple[Path, int, int, str]:
    path = ROOT / case["path"]
    coords, _, _ = _parse_pdb(path)
    return path, coords.shape[0], coords.shape[1], "frozen_existing_CA_multimodel_PDB"


def _case_input(case: dict[str, str], workspace: Path) -> tuple[Path, int, int, str]:
    if case["kind"] == "existing_pdb":
        return _existing_input(case)
    return _input(case, workspace)


def _native_differences(stock: dict[str, object], exported: dict[str, object]) -> tuple[float, bool]:
    fields = (
        "native_least_mobile_RMSD_A",
        "native_most_mobile_RMSD_A",
        "native_all_atom_RMSD_A",
        "native_fraction",
        "native_total_atoms",
        "native_selection_atoms",
        "frame_rows",
    )
    differences: list[float] = []
    passed = True
    for field in fields:
        left = stock.get(field)
        right = exported.get(field)
        if isinstance(left, float) and isinstance(right, float):
            difference = abs(left - right)
            same = difference <= 5e-5
        else:
            difference = float(left != right)
            same = left == right
        differences.append(difference)
        passed &= same
    return max(differences), passed


def _register_contracts() -> None:
    for subdir in SUBDIRS:
        (MA / subdir).mkdir(parents=True, exist_ok=True)
    write_text(
        MA / "00_contracts/method_agreement_contract.md",
        """# MDCoreAlign-MDLovoFit Method-agreement Contract

The primary matched operating point is fraction 0.60 with exactly
`K = floor(0.60 * N)` selected residues. It is an equal-cardinality method
comparison, not a universal biological core-size assumption.

Layer A compares stable-residue rankings and matched top-K masks. Layer B
compares explicit all-frame hard-subset refits at equal cardinality. Layer C
compares native MDLovoFit hard-subset alignment with native MDCoreAlign exact
and throughput soft-weight alignments. Cross-method identity is not a gate.

Exporter identity is a separate self-consistency requirement against stock
MDLovoFit `-t` for identical input and parameters. All MA runs are untimed.
""",
    )
    write_tsv(
        MA / "00_contracts/fraction_registry.tsv",
        [
            {
                "fraction": fraction,
                "integer_rule": "floor(fraction*N)",
                "primary_operating_point": fraction == FRACTION,
                "registered_before_results": True,
            }
            for fraction in (0.40, 0.50, 0.60, 0.70, 0.80)
        ],
    )
    metrics = [
        ("core_Jaccard", "intersection/union of equal-cardinality hard masks", "dimensionless"),
        ("core_Dice", "2*intersection/(size_A+size_B)", "dimensionless"),
        ("weight_Spearman", "rank correlation of continuous residue evidence", "dimensionless"),
        ("aligned_coordinate_RMSD", "RMS coordinate difference after applying both method transforms", "Angstrom"),
        ("rotation_geodesic", "geodesic angle between canonical rotations", "degrees"),
        ("RMSF_Spearman", "rank correlation of independently recomputed RMSF", "dimensionless"),
        ("RMSF_MAE", "mean absolute difference of independently recomputed RMSF", "Angstrom"),
        ("chance_Jaccard", "f/(2-f) for random equal-cardinality masks", "dimensionless"),
    ]
    write_tsv(
        MA / "00_contracts/metric_dictionary.tsv",
        [{"metric": name, "definition": definition, "unit": unit} for name, definition, unit in metrics],
    )
    write_text(
        MA / "00_contracts/agreement_category_dictionary.md",
        """# Descriptive Agreement Categories

- High practical agreement: Jaccard >= 0.80, score Spearman >= 0.80, median aligned-coordinate difference <= 0.10 A, and RMSF Spearman >= 0.98.
- Broad/moderate agreement: Jaccard >= 0.60, score Spearman >= 0.60, median aligned-coordinate difference <= 0.25 A, and RMSF Spearman >= 0.95.
- Method-dependent result: does not meet the broad/moderate category.

These are descriptive categories, not biological-equivalence or quality gates.
""",
    )
    write_tsv(
        MA / "00_contracts/environment_manifest.tsv",
        [
            {"item": "generated_at", "value": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()},
            {"item": "platform", "value": platform.platform()},
            {"item": "python", "value": platform.python_version()},
            {"item": "stock_mdlovofit", "value": "/usr/local/bin/mdlovofit"},
            {"item": "stock_sha256", "value": sha256(STOCK)},
            {"item": "exporter", "value": "01_mdlovofit_export_validation/bin/mdlovofit_export"},
            {"item": "exporter_sha256", "value": sha256(EXPORTER)},
            {"item": "timing_status", "value": "untimed_scientific_extraction_only"},
        ],
    )
    write_tsv(MA / "00_contracts/panel_manifests/exporter_validation_panel.tsv", list(VALIDATION_CASES))


def _write_semantics() -> None:
    rows = [
        {"variable": "scores", "role": "Per-residue squared residuals for the current fit; sorted during LOVO updates.", "frequency_semantics": "none"},
        {"variable": "bije_new", "role": "Residue indices ordered by current residual; after aggregate sorting it defines final fixed-subset rank.", "frequency_semantics": "none"},
        {"variable": "bije_old", "role": "Inverse rank mapping used by occupancy assignment and final-pass residual lookup.", "frequency_semantics": "none"},
        {"variable": "best_bije", "role": "Best final residue ordering retained for the current frame across trials.", "frequency_semantics": "one final subset source per frame"},
        {"variable": "av_best_bije", "role": "Decremented whenever a trial improves the current frame best; sorted to construct the final fixed subset.", "frequency_semantics": "internal improvement-event evidence, not per-frame frequency"},
        {"variable": "frame_selection_count", "role": "Exporter-only counter incremented once after all trials for each frame using best_bije.", "frequency_semantics": "true per-frame final selection count"},
        {"variable": "n_consider", "role": "Hard subset size min(N, floor(fraction*N)); final occupancy is one when inverse rank <= n_consider.", "frequency_semantics": "none"},
    ]
    write_tsv(MA / "01_mdlovofit_export_validation/source_variable_dictionary.tsv", rows)
    write_text(
        MA / "01_mdlovofit_export_validation/mdlovofit_subset_semantics.md",
        """# MDLovoFit 20.1.1 Fixed-subset Semantics

For each frame and trial, MDLovoFit alternates rigid fitting and sorting of
per-residue squared residuals. `best_bije` stores the best final ordering found
for that frame. In the stock source, `av_best_bije` is decremented whenever a
trial improves the current frame best; it therefore mixes frames with a
variable number of improvement events and must not be called selection
frequency. The non-perframe fixed subset is obtained by sorting this aggregate
evidence, and the second pass fits the first `n_consider` ranked residues.

In the final stock `-t` pass, occupancy is set to 1.00 exactly when the inverse
fixed rank is no greater than `n_consider`. The separate exporter adds
`frame_selection_count`, incremented once per frame after all trials, so true
per-frame final selection frequency and internal aggregate evidence remain
distinct outputs.

The exporter captures transforms during the final fixed-subset pass using the
column-vector convention `x_aligned = R @ x_raw + t`, where
`t = reference_core_centroid - R @ frame_core_centroid`. Diagnostic files are
written only after every `NM_TIMING` record has been emitted. This binary is
prohibited from use in timing comparisons.

The fixed reference frame receives a final-pass transform but is not an
eligible per-frame LOVO-selection event. Consequently, the exported selection
frequency denominator is `n_frames - 1`, while the transform table contains
all `n_frames` rows.
""",
    )


def _run_validation() -> None:
    if not EXPORTER.is_file():
        raise FileNotFoundError(f"build exporter first: {EXPORTER}")
    rows: list[dict[str, object]] = []
    subset_rows: list[dict[str, object]] = []
    transform_rows: list[dict[str, object]] = []
    numerical_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ma2_exporter_") as temporary:
        temp_root = Path(temporary)
        for case in VALIDATION_CASES:
            case_root = temp_root / case["case"]
            case_root.mkdir()
            input_path, n_frames, n_residues, provenance = _case_input(case, case_root)
            stock_output = case_root / "stock_aligned.pdb"
            stock = run_external(
                [str(STOCK), "-f", f"{FRACTION:.6f}", "-t", str(stock_output), str(input_path)],
                cwd=case_root,
                timeout=3600,
            )
            prefix = case_root / "mdlovofit"
            exported = run_external(
                [str(EXPORTER), "-f", f"{FRACTION:.6f}", "-ma-export", str(prefix), str(input_path)],
                cwd=case_root,
                timeout=3600,
            )
            required = {
                "mask": case_root / "mdlovofit_final_fixed_subset.tsv",
                "frequency": case_root / "mdlovofit_per_frame_final_selection_frequency.tsv",
                "evidence": case_root / "mdlovofit_internal_selection_evidence.tsv",
                "transforms": case_root / "mdlovofit_final_transforms.tsv",
            }
            files_exist = all(path.is_file() and path.stat().st_size > 0 for path in required.values())
            if not files_exist:
                raise RuntimeError(f"missing exporter output for {case['case']}: {required}")
            mask_rows = _read_tsv(required["mask"])
            frequency_rows = _read_tsv(required["frequency"])
            transform_data = _read_tsv(required["transforms"])
            raw, _, raw_metadata = _parse_pdb(input_path)
            aligned, occupancy, aligned_metadata = _parse_pdb(stock_output)
            mask = np.asarray([int(row["selected_in_final_fixed_subset"]) for row in mask_rows], dtype=np.int8)
            stock_masks = occupancy == 1.0
            stock_mask = stock_masks[0].astype(np.int8)
            k = math.floor(FRACTION * n_residues)
            count_pass = int(mask.sum()) == k == int(stock_mask.sum())
            occupancy_constant = bool(np.all(stock_masks == stock_masks[0]))
            mask_identity = bool(np.array_equal(mask, stock_mask))
            eligible_frames = max(n_frames - 1, 0)
            selected_frame_counts = [int(row["selected_frame_count"]) for row in frequency_rows]
            frequency_count_pass = (
                len(frequency_rows) == n_residues
                and all(int(row["eligible_frame_count"]) == eligible_frames for row in frequency_rows)
                and all(0 <= count <= eligible_frames for count in selected_frame_counts)
                and sum(selected_frame_counts) == k * eligible_frames
            )
            rotations = np.asarray(
                [
                    [[float(row[f"rotation_{i}{j}"]) for j in range(3)] for i in range(3)]
                    for row in transform_data
                ],
                dtype=np.float64,
            )
            translations = np.asarray(
                [[float(row["translation_x"]), float(row["translation_y"]), float(row["translation_z"])] for row in transform_data],
                dtype=np.float64,
            )
            reconstructed = np.einsum("fij,faj->fai", rotations, raw) + translations[:, None, :]
            coordinate_errors = np.linalg.norm(reconstructed - aligned, axis=2)
            maximum_coordinate_error = float(np.max(coordinate_errors))
            median_coordinate_error = float(np.median(coordinate_errors))
            transform_pass = maximum_coordinate_error <= 0.0010
            metadata_pass = raw_metadata == aligned_metadata and len(mask_rows) == n_residues
            native_difference, native_pass = _native_differences(stock, exported)
            overall = (
                stock["status"] == exported["status"] == "success"
                and count_pass
                and occupancy_constant
                and mask_identity
                and frequency_count_pass
                and transform_pass
                and metadata_pass
                and native_pass
            )
            common = {
                **case,
                "n_frames": n_frames,
                "n_residues": n_residues,
                "input_provenance": provenance,
                "K": k,
            }
            rows.append(
                {
                    **common,
                    "stock_status": stock["status"],
                    "exporter_status": exported["status"],
                    "export_files_complete": files_exist,
                    "selected_count_pass": count_pass,
                    "mask_identity_pass": mask_identity,
                    "transform_reconstruction_pass": transform_pass,
                    "metadata_mapping_pass": metadata_pass,
                    "frequency_counter_bounds_pass": frequency_count_pass,
                    "native_numerical_identity_pass": native_pass,
                    "overall_pass": overall,
                }
            )
            subset_rows.append(
                {
                    **common,
                    "export_selected_count": int(mask.sum()),
                    "stock_occupancy_selected_count": int(stock_mask.sum()),
                    "stock_mask_constant_across_frames": occupancy_constant,
                    "mismatched_residue_count": int(np.count_nonzero(mask != stock_mask)),
                    "exact_mask_identity": mask_identity,
                }
            )
            transform_rows.append(
                {
                    **common,
                    "exported_transform_rows": len(transform_data),
                    "median_coordinate_reconstruction_error_A": median_coordinate_error,
                    "maximum_coordinate_reconstruction_error_A": maximum_coordinate_error,
                    "PDB_precision_tolerance_A": 0.001,
                    "transform_reconstruction_pass": transform_pass,
                }
            )
            numerical_rows.append(
                {
                    **common,
                    "stock_least_mobile_RMSD_A": stock.get("native_least_mobile_RMSD_A"),
                    "exporter_least_mobile_RMSD_A": exported.get("native_least_mobile_RMSD_A"),
                    "stock_all_atom_RMSD_A": stock.get("native_all_atom_RMSD_A"),
                    "exporter_all_atom_RMSD_A": exported.get("native_all_atom_RMSD_A"),
                    "maximum_printed_numeric_difference": native_difference,
                    "native_numerical_identity_pass": native_pass,
                }
            )
    destination = MA / "01_mdlovofit_export_validation"
    write_tsv(destination / "exporter_self_consistency.tsv", rows)
    write_tsv(destination / "subset_vs_occupancy.tsv", subset_rows)
    write_tsv(destination / "transform_reconstruction.tsv", transform_rows)
    write_tsv(destination / "numerical_identity.tsv", numerical_rows)
    passed = all(bool(row["overall_pass"]) for row in rows)
    write_text(
        destination / "export_validation_report.md",
        f"""# MDLovoFit Exporter Self-consistency Validation

Status: **{'PASS' if passed else 'FAIL'}** on {len(rows)}/{len(VALIDATION_CASES)}
prospectively locked cases. This is exporter-versus-stock-MDLovoFit
self-consistency, not MDCoreAlign-versus-MDLovoFit identity. The exporter was
run once per case outside all NM timing measurements.

The final exported mask was compared exactly with stock `-t` occupancy 1.00,
native printed RMSD fields were compared within printed precision, and exported
transforms were independently applied to the same raw coordinates and compared
with the rounded stock aligned PDB at 0.001 A precision.
""",
    )
    write_json(
        destination / "ma2_status.json",
        {"cases": len(rows), "passed": sum(bool(row["overall_pass"]) for row in rows), "status": "pass" if passed else "fail"},
    )
    if not passed:
        raise RuntimeError("MA2 exporter self-consistency gate failed; MA3-MA8 must not proceed")


def main() -> None:
    _register_contracts()
    _write_semantics()
    _run_validation()
    write_json(MA / "00_contracts/ma0_ma2_status.json", {"MA0": "pass", "MA1": "pass", "MA2": "pass"})
    print(json.dumps({"MA0": "pass", "MA1": "pass", "MA2": "pass", "cases": len(VALIDATION_CASES)}, indent=2))


if __name__ == "__main__":
    main()
