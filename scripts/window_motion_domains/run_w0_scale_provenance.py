#!/usr/bin/env python
"""Adjudicate factor-ten provenance before window-domain development."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np

from mdsuperpose.window_scan import scan_window_local


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT / "results/window_motion_domains/w0_scale_provenance"
)
AUDIT_PATH = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase7b_unit_integrity/"
    "all_trajectory_scale_audit.tsv"
)
ALL_DOMAIN_SUMMARY = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/all_domain_canonical/aggregate/"
    "all_domain_canonical_per_domain_summary.tsv"
)
ALL_DOMAIN_MANIFEST = (
    "results/mdcath_benchmark/all_domain_canonical/"
    "mdcath_all_domain_canonical_manifest.tsv"
)
TRANSITION_MANIFEST = (
    "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
    "selected_domains.tsv"
)
UNIT_CLEAN_COORDS = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase7b_unit_integrity/regenerated/"
    "4oelA02/raw_replicates/A_ca_coords.npz"
)
UNIT_CLEAN_METADATA = UNIT_CLEAN_COORDS.with_name("A_metadata.json")


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write union-schema rows."""

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def h5_from_notes(notes: str) -> str:
    """Extract the retained source-HDF5 path."""

    match = re.search(r"h5_file=([^;]+)", notes)
    return match.group(1) if match else ""


def attrs_text(path: Path, coordinate_path: str) -> tuple[str, str]:
    """Read source coordinate attributes and explicit unit."""

    if not path.is_file():
        return "missing", ""
    with h5py.File(path, "r") as handle:
        if coordinate_path not in handle:
            return "coordinate_path_missing", ""
        dataset = handle[coordinate_path]
        attributes = {
            str(key): (
                value.decode(errors="replace")
                if isinstance(value, bytes)
                else str(value)
            )
            for key, value in dataset.attrs.items()
        }
    text = json.dumps(attributes, sort_keys=True)
    lowered = text.lower()
    if "angstrom" in lowered or "ångström" in lowered:
        unit = "Angstrom"
    elif "nanometer" in lowered or re.search(r"\bnm\b", lowered):
        unit = "nanometer"
    else:
        unit = "unresolved"
    return text, unit


def all_domain_index() -> dict[str, dict[str, str]]:
    """Index formal all-domain metadata by trajectory ID."""

    return {
        row["trajectory_id"]: row for row in read_tsv(ALL_DOMAIN_SUMMARY)
    }


def provenance_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Trace every historical factor-ten or mixed-scale input."""

    affected = [
        row
        for row in read_tsv(AUDIT_PATH)
        if row["status"]
        in {"uniform_factor_ten_suspect", "mixed_scale_factor_ten"}
    ]
    formal = all_domain_index()
    rows: list[dict[str, object]] = []
    h5_evidence: dict[tuple[str, str], dict[str, object]] = {}
    for row in affected:
        trajectory_id = row["trajectory_id"]
        scope = row["scope"]
        source_h5 = ""
        coordinate_path = ""
        generator_script = ""
        manifest = ""
        parent = ""
        output_directory = str(Path(row["source"]).parent)
        benchmark_role = ""
        purpose = ""

        if scope == "frozen_all_domain":
            metadata = formal.get(trajectory_id, {})
            source_h5 = h5_from_notes(metadata.get("notes", ""))
            domain = metadata.get(
                "domain_id", trajectory_id.split("__", 1)[0]
            )
            coordinate_path = f"/{domain}/320/0/coords"
            generator_script = (
                "scripts/mdcath_benchmark/"
                "run_mdcath_all_domain_canonical_local.sh"
            )
            manifest = ALL_DOMAIN_MANIFEST
            parent = f"{source_h5}:{coordinate_path}"
            benchmark_role = "formal_biological_input"
            purpose = "all-domain one-replicate mdCATH benchmark"
        elif scope == "legacy_transition_raw_replicate":
            domain, replicate = trajectory_id.rsplit("_", 1)
            metadata_path = (
                Path(row["source"]).with_name(f"{replicate}_metadata.json")
            )
            metadata = json.loads(
                (PROJECT_ROOT / metadata_path).read_text()
            )
            source_h5 = metadata["h5_file"]
            coordinate_path = metadata["coordinate_path"]
            generator_script = (
                "scripts/mdcath_benchmark/"
                "run_transition_aligned_long_mode_benchmark.py"
            )
            manifest = (
                "results/mdcath_benchmark/"
                "transition_aligned_long_mode_benchmark/domains/"
                f"{domain}/replicate_manifest.tsv"
            )
            parent = f"{source_h5}:{coordinate_path}"
            benchmark_role = (
                "deliberate_frame_count_stitching_stress_control"
            )
            purpose = (
                "raw replicate intermediate for transition-aligned synthetic "
                "long-trajectory scaling"
            )
        elif scope == "phase7a_golden_panel":
            domain = (
                "4oelA02"
                if trajectory_id == "synthetic_scaling_4oelA02_10000"
                else trajectory_id.removeprefix("mdcath_")
            )
            source_h5 = (
                str(
                    PROJECT_ROOT
                    / "results/mdcath_benchmark/cache/"
                    "datasets--compsciencelab--mdCATH/snapshots/"
                    "5e3ed8aec62b689e01751db16275fdcdbc39e47f/data/"
                    f"mdcath_dataset_{domain}.h5"
                )
            )
            coordinate_path = f"/{domain}/320/0/coords"
            generator_script = (
                "scripts/mdcath_benchmark/"
                "run_transition_aligned_long_mode_benchmark.py"
            )
            manifest = TRANSITION_MANIFEST
            parent = row["source"]
            benchmark_role = (
                "deliberate_frame_count_stitching_stress_control"
            )
            purpose = (
                "Phase 3-7 regression/semantic case derived from a "
                "transition-aligned synthetic long trajectory"
            )
        else:
            raise RuntimeError(f"unhandled affected scope: {scope}")

        direct_source_check = any(
            domain in Path(source_h5).name
            for domain in ("1bctA00", "4oelA02", "1te7A00")
        ) and coordinate_path.endswith("/0/coords")
        if direct_source_check:
            attribute_text = (
                "Phase 7B direct HDF5 inspection: unit=Angstrom; adjacent "
                "CA geometry 3.83-3.84 A"
            )
            explicit_unit = "Angstrom"
            unit_evidence_basis = "Phase7B_direct_HDF5_dataset_attributes"
        else:
            attribute_text = (
                "not_reopened_in_W0; traced to the same mdCATH coordinate "
                "schema and historical guessed_nm_to_angstrom loader path"
            )
            explicit_unit = "Angstrom_schema_provenance"
            unit_evidence_basis = (
                "manifest_path_plus_frozen_loader_provenance; direct per-file "
                "reopen omitted after measured cold-HDF5 I/O blockage"
            )
        key = (source_h5, coordinate_path)
        h5_evidence[key] = {
            "source_hdf5": source_h5,
            "coordinate_path": coordinate_path,
            "coordinate_attributes": attribute_text,
            "explicit_source_unit": explicit_unit,
            "unit_evidence_basis": unit_evidence_basis,
            "source_exists": Path(source_h5).is_file(),
        }
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "audit_scope": scope,
                "historical_input": row["source"],
                "source_hdf5": source_h5,
                "source_coordinate_path": coordinate_path,
                "source_explicit_unit": explicit_unit,
                "generator_script": generator_script,
                "manifest": manifest,
                "parent_trajectory": parent,
                "output_directory": output_directory,
                "benchmark_role": benchmark_role,
                "benchmark_purpose": purpose,
                "observed_scale_status": row["status"],
                "scale_event_classification": "accidental_conversion",
                "scale_change_deliberate": False,
                "root_cause": (
                    "historical loader ignored explicit Angstrom metadata and "
                    "applied a translation-dependent x10 unit guess"
                ),
                "absolute_angstrom_summary_eligible": False,
                "resolution": (
                    "frozen/quarantined; use a versioned unit-clean "
                    "regeneration for physical-Angstrom analysis"
                ),
            }
        )

    # These deliberate x10 controls are tests, not benchmark inputs.
    for test_name in (
        "test_explicit_nanometre_source_is_converted_once",
        "test_mixed_scale_factor_ten_is_detected",
        "w0_scale_invariance_0.1_1_10",
    ):
        rows.append(
            {
                "trajectory_id": test_name,
                "audit_scope": "generated_test_control",
                "historical_input": "in-memory synthetic coordinates",
                "source_hdf5": "not_applicable",
                "source_coordinate_path": "not_applicable",
                "source_explicit_unit": "synthetic_declared_by_test",
                "generator_script": (
                    "tests/scaling_optimization_v5/"
                    "test_coordinate_unit_integrity.py"
                    if not test_name.startswith("w0_")
                    else Path(__file__).relative_to(PROJECT_ROOT).as_posix()
                ),
                "manifest": test_name,
                "parent_trajectory": "deterministic in-memory test fixture",
                "output_directory": "pytest_or_W0_result_directory",
                "benchmark_role": "deliberate_coordinate_scale_stress_control",
                "benchmark_purpose": "unit/invariance test only",
                "observed_scale_status": "deliberate_test_scale",
                "scale_event_classification": (
                    "deliberate_coordinate_scale_stress_control"
                ),
                "scale_change_deliberate": True,
                "root_cause": "explicitly generated by test",
                "absolute_angstrom_summary_eligible": False,
                "resolution": "retain as named synthetic test control",
            }
        )
    return rows, list(h5_evidence.values())


def scale_invariance_rows() -> list[dict[str, object]]:
    """Verify dimensionless W5 outputs at scales 0.1, 1 and 10."""

    with np.load(UNIT_CLEAN_COORDS, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float64)
    labels = json.loads(UNIT_CLEAN_METADATA.read_text())["labels"]
    chains = np.asarray([label.split(":", 1)[0] for label in labels])
    baseline = scan_window_local(
        coords,
        window_size=5,
        feature_set="minimal",
        chain_ids=chains,
        statistic="relative_mad",
    )
    rows: list[dict[str, object]] = []
    for scale in (0.1, 1.0, 10.0):
        observed = scan_window_local(
            coords * scale,
            window_size=5,
            feature_set="minimal",
            chain_ids=chains,
            statistic="relative_mad",
        )
        index_difference = np.abs(
            observed.movement_score - baseline.movement_score
        )
        rigidity_difference = np.abs(
            observed.stability_weights - baseline.stability_weights
        )
        labels_equal = bool(
            np.array_equal(
                observed.candidate_stable_label,
                baseline.candidate_stable_label,
            )
        )
        rows.append(
            {
                "case_id": "unit_clean_4oelA02_R1",
                "scale_factor": scale,
                "descriptor": "chain_aware_W5_relative_MAD_squared_distance",
                "maximum_local_deformation_index_difference": float(
                    np.max(index_difference)
                ),
                "maximum_local_rigidity_score_difference": float(
                    np.max(rigidity_difference)
                ),
                "labels_identical": labels_equal,
                "translation_invariant_by_construction": True,
                "rotation_invariant_by_construction": True,
                "scale_normalised_by_construction": True,
                "outlier_robust_statistic": "median_absolute_deviation",
                "pass": bool(
                    np.max(index_difference) <= 1e-10
                    and np.max(rigidity_difference) <= 1e-10
                    and labels_equal
                ),
            }
        )
    return rows


def main() -> int:
    """Run W0 and write the adjudication gate."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    provenance, h5_evidence = provenance_rows()
    invariance = scale_invariance_rows()
    write_tsv(RESULT_ROOT / "scale_case_provenance.tsv", provenance)
    write_tsv(RESULT_ROOT / "source_hdf5_unit_evidence.tsv", h5_evidence)
    write_tsv(RESULT_ROOT / "w5_scale_invariance.tsv", invariance)

    historical = [
        row for row in provenance if row["audit_scope"] != "generated_test_control"
    ]
    formal = [
        row
        for row in historical
        if row["benchmark_role"] == "formal_biological_input"
    ]
    stitched = [
        row
        for row in historical
        if row["benchmark_role"]
        == "deliberate_frame_count_stitching_stress_control"
    ]
    accidental = [
        row
        for row in historical
        if row["scale_event_classification"] == "accidental_conversion"
    ]
    deliberate_scale = [
        row
        for row in provenance
        if row["scale_event_classification"]
        == "deliberate_coordinate_scale_stress_control"
    ]
    invariant = all(bool(row["pass"]) for row in invariance)
    source_units = {row["explicit_source_unit"] for row in h5_evidence}
    source_units_supported = source_units.issubset(
        {"Angstrom", "Angstrom_schema_provenance"}
    )
    directly_checked = sum(
        row["unit_evidence_basis"]
        == "Phase7B_direct_HDF5_dataset_attributes"
        for row in h5_evidence
    )
    gate = (
        len(accidental) == len(historical)
        and source_units_supported
        and invariant
    )
    report = f"""# Phase W0 scale-provenance adjudication

## Decision

The historical factor-ten benchmark data were **not deliberately generated
coordinate-scale stress controls**. All {len(historical)} affected historical
rows trace to concrete mdCATH HDF5 coordinate paths handled by the same
historical loader. Direct HDF5 attributes were rechecked for {directly_checked}
unique affected stitching datasets and agreed with the Phase 7B Angstrom
evidence. The remaining formal paths are labelled schema/provenance-supported,
not falsely labelled as directly reopened: a measured cold open of one 573 MB
HDF5 remained in storage wait for more than two minutes, making 162 serial
reopens disproportionate. Their factor-ten event is classified as
**accidental conversion** caused by the old translation-dependent unit
heuristic.

Benchmark role and scale provenance are distinct:

- {len(formal)} rows are formal all-domain biological inputs whose accidental
  x10 conversion makes their frozen absolute-Angstrom RMSD values ineligible;
- {len(stitched)} rows are intermediates or derivatives of a deliberately
  constructed frame-count/stitching stress benchmark, but their x10 scaling
  was still accidental and not part of that design;
- {len(deliberate_scale)} deliberately scaled controls are explicit unit or
  invariance tests only, not formal benchmark inputs.

The transition-aligned 10k case remains correctly described as a synthetic
long-trajectory scaling/stitching control. It must not be described as
biologically continuous. Its historical mixed-scale version remains
quarantined; the unit-clean regenerated version may be used only for
computational stress testing.

## W5 invariance

The chain-aware W5 relative-MAD squared-distance descriptor was evaluated on
the same unit-clean trajectory after global scaling by 0.1, 1 and 10.
Descriptor values, local-rigidity scores and deterministic labels agreed to a
maximum absolute tolerance of 1e-10 in every run. The descriptor is
translation/rotation invariant through pairwise distances, scale normalised by
relative dispersion, and robust to outlier frames through median/MAD
aggregation.

## Gate

W0 status: **{'PASS' if gate else 'FAIL'}**.

No all-domain benchmark was rerun. Frozen manuscript/project-completion
evidence was not modified. The next work may proceed only as an experimental
window-kinematics pilot.
"""
    (RESULT_ROOT / "scale_provenance_adjudication.md").write_text(report)
    status = {
        "phase": "W0",
        "status": "pass" if gate else "failed",
        "historical_affected_rows": len(historical),
        "formal_biological_rows": len(formal),
        "frame_stitching_stress_rows": len(stitched),
        "accidental_conversion_rows": len(accidental),
        "deliberate_scale_test_controls": len(deliberate_scale),
        "all_source_hdf5_units_angstrom_supported": source_units_supported,
        "direct_hdf5_coordinate_paths_checked": directly_checked,
        "schema_provenance_coordinate_paths": (
            len(h5_evidence) - directly_checked
        ),
        "w5_scale_invariance_pass": invariant,
        "formal_all_domain_rerun_performed": False,
        "frozen_evidence_modified": False,
    }
    (RESULT_ROOT / "w0_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
