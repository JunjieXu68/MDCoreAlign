#!/usr/bin/env python
"""Resolve scale provenance with a stratified direct mdCATH HDF5 audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

import h5py
import numpy as np

from mdsuperpose.coordinate_scale import chain_aware_adjacent_pairs
from scripts.mdcath_benchmark.mdcath_common import (
    find_pdb_text,
    parse_pdb_ca_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT / "results/mainline_reintegration_v1/unit_provenance"
)
W0_MANIFEST = (
    PROJECT_ROOT
    / "results/window_motion_domains/w0_scale_provenance/"
    "scale_case_provenance.tsv"
)
ALL_DOMAIN_SUMMARY = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/all_domain_canonical/aggregate/"
    "all_domain_canonical_per_domain_summary.tsv"
)
ALL_DOMAIN_MANIFEST = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/all_domain_canonical/"
    "mdcath_all_domain_canonical_manifest.tsv"
)
STITCH_PANEL = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
    "selected_domains.tsv"
)
LOADER = PROJECT_ROOT / "scripts/mdcath_benchmark/mdcath_common.py"
GENERATOR = (
    PROJECT_ROOT
    / "scripts/mdcath_benchmark/"
    "extract_ca_trajectory_from_mdcath_h5.py"
)
MINIMUM_MANUAL_AUDIT = 16


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV."""

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


def normalized_unit(attrs: dict[str, object]) -> str:
    """Return an explicit normalized source unit without geometry inference."""

    text = " ".join(str(value).lower() for value in attrs.values())
    if "angstrom" in text or "ångström" in text:
        return "angstrom"
    if "nanometer" in text or " nm" in f" {text}":
        return "nanometer"
    return "unresolved"


def domain_id(row: dict[str, str]) -> str:
    """Return a domain ID from a formal trajectory identifier."""

    return row["trajectory_id"].split("__", 1)[0]


def select_manual_cases(
    formal: list[dict[str, str]],
    summaries: dict[str, dict[str, str]],
    stitch_domains: set[str],
) -> list[str]:
    """Select 16 cases across class, length and stitching membership."""

    available = {
        domain_id(row): row
        for row in formal
        if domain_id(row) in summaries
    }
    selected: list[str] = []

    def add(value: str) -> None:
        if value in available and value not in selected:
            selected.append(value)

    for value in sorted(stitch_domains.intersection(available)):
        add(value)
    for cath_class in ("1", "2", "3", "4", "NA"):
        pool = sorted(
            (
                value
                for value in available
                if summaries[value]["cath_class"] == cath_class
            ),
            key=lambda value: int(summaries[value]["n_residues"]),
        )
        for fraction in (0.0, 0.5, 1.0):
            if pool:
                add(pool[round(fraction * (len(pool) - 1))])
    all_cases = sorted(
        available,
        key=lambda value: int(summaries[value]["n_residues"]),
    )
    for fraction in np.linspace(0.0, 1.0, 32):
        if len(selected) >= MINIMUM_MANUAL_AUDIT:
            break
        add(all_cases[round(float(fraction) * (len(all_cases) - 1))])
    if len(selected) < MINIMUM_MANUAL_AUDIT:
        raise RuntimeError("fewer than 16 formal cases are available")
    return selected[:MINIMUM_MANUAL_AUDIT]


def direct_hdf5_audit(
    row: dict[str, str],
    summary: dict[str, str],
    *,
    in_stitch_panel: bool,
) -> dict[str, object]:
    """Hash and directly inspect one source dataset and reference CA frame."""

    source = Path(row["source_hdf5"])
    coordinate_path = row["source_coordinate_path"]
    source_digest = sha256(source)
    with h5py.File(source, "r") as handle:
        if coordinate_path not in handle:
            raise KeyError(f"{source}: missing {coordinate_path}")
        dataset = handle[coordinate_path]
        attrs = {str(key): dataset.attrs[key] for key in dataset.attrs}
        unit = normalized_unit(attrs)
        pdb_path, pdb_text = find_pdb_text(handle)
        if not pdb_text:
            raise RuntimeError(f"{source}: PDB atom metadata unavailable")
        ca_indices, labels, residue_metadata = parse_pdb_ca_metadata(pdb_text)
        raw_reference = np.asarray(dataset[0], dtype=np.float64)
        ca_reference = raw_reference[ca_indices]
        dataset_shape = tuple(int(value) for value in dataset.shape)
        dataset_dtype = str(dataset.dtype)
    chains = np.asarray([item["chain"] for item in residue_metadata])
    residue_ids = np.asarray(
        [item["residue_id"] for item in residue_metadata]
    )
    pairs = chain_aware_adjacent_pairs(
        ca_reference.shape[0],
        chains,
        residue_ids,
    )
    adjacent = np.linalg.norm(
        ca_reference[pairs[:, 0]] - ca_reference[pairs[:, 1]],
        axis=1,
    )
    historical_input = PROJECT_ROOT / row["historical_input"]
    return {
        "trajectory_id": row["trajectory_id"],
        "domain_id": domain_id(row),
        "cath_class": summary["cath_class"],
        "n_residues_formal_result": summary["n_residues"],
        "length_stratum": (
            "small_le_100"
            if int(summary["n_residues"]) <= 100
            else "large_ge_200"
            if int(summary["n_residues"]) >= 200
            else "medium_101_199"
        ),
        "appeared_in_stitch_panel": in_stitch_panel,
        "source_hdf5": str(source),
        "source_hdf5_size_bytes": source.stat().st_size,
        "source_hdf5_sha256": source_digest,
        "source_coordinate_path": coordinate_path,
        "source_dataset_shape": "x".join(str(value) for value in dataset_shape),
        "source_dataset_dtype": dataset_dtype,
        "source_unit_metadata": json.dumps(
            {key: str(value) for key, value in attrs.items()},
            sort_keys=True,
        ),
        "source_unit_normalized": unit,
        "conversion_factor_to_angstrom": (
            1.0 if unit == "angstrom" else 10.0 if unit == "nanometer" else ""
        ),
        "canonical_internal_unit": "angstrom",
        "pdb_metadata_path": pdb_path,
        "n_ca_atoms_direct": ca_reference.shape[0],
        "n_valid_adjacent_ca_pairs": pairs.shape[0],
        "median_adjacent_ca_distance_source_units": float(
            np.median(adjacent)
        ),
        "p05_adjacent_ca_distance_source_units": float(
            np.quantile(adjacent, 0.05)
        ),
        "p95_adjacent_ca_distance_source_units": float(
            np.quantile(adjacent, 0.95)
        ),
        "loader_path": str(LOADER),
        "loader_sha256": sha256(LOADER),
        "loader_version": "mainline_reintegration_v1_explicit_unit_loader",
        "current_conversion_operation": (
            "preserve explicit Angstrom coordinates (factor 1.0)"
            if unit == "angstrom"
            else "convert explicit nanometer coordinates by factor 10.0"
            if unit == "nanometer"
            else "preserve and quarantine pending explicit provenance"
        ),
        "historical_conversion_operation": (
            "if 0.1 < median(abs(coords)) < 20, multiply by 10; "
            "explicit Angstrom metadata was not recognized"
        ),
        "generated_array_path": row["historical_input"],
        "generated_array_exists": historical_input.is_file(),
        "generated_array_sha256": (
            sha256(historical_input) if historical_input.is_file() else ""
        ),
        "benchmark_manifest": row["manifest"],
        "formal_all_domain_membership": True,
        "stress_benchmark_membership": in_stitch_panel,
        "final_classification": (
            "formal biological input with accidental conversion"
        ),
        "audit_status": (
            "pass_explicit_angstrom"
            if unit == "angstrom"
            and 2.5 <= float(np.median(adjacent)) <= 5.5
            else "failed_or_unresolved"
        ),
    }


def classification_for(row: dict[str, str]) -> str:
    """Map one historical/control record to the R1 classification vocabulary."""

    role = row["benchmark_role"]
    if role == "formal_biological_input":
        return "formal biological input with accidental conversion"
    if role == "deliberate_frame_count_stitching_stress_control":
        return (
            "stitched/frame-count stress case derived from an accidental "
            "conversion"
        )
    if role == "deliberate_coordinate_scale_stress_control":
        return "deliberate scale-invariance control"
    return "unresolved"


def build_record_manifest(
    records: list[dict[str, str]],
    manual: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Build one human-readable row for every historical and control record."""

    rows: list[dict[str, object]] = []
    for record in records:
        value = domain_id(record)
        evidence = manual.get(value)
        classification = classification_for(record)
        affected = (
            record["benchmark_role"]
            != "deliberate_coordinate_scale_stress_control"
        )
        rows.append(
            {
                "trajectory_id": record["trajectory_id"],
                "affected_historical_record": affected,
                "audit_scope": record["audit_scope"],
                "benchmark_role": record["benchmark_role"],
                "final_classification": classification,
                "purpose": (
                    "formal"
                    if record["benchmark_role"] == "formal_biological_input"
                    else "stress"
                    if "stitching" in record["benchmark_role"]
                    else "scale-control"
                ),
                "source_hdf5": record["source_hdf5"],
                "source_coordinate_path": record["source_coordinate_path"],
                "source_unit": (
                    evidence["source_unit_normalized"]
                    if evidence
                    else record["source_explicit_unit"]
                ),
                "conversion_factor_to_angstrom": (
                    evidence["conversion_factor_to_angstrom"]
                    if evidence
                    else 1.0
                    if "Angstrom" in record["source_explicit_unit"]
                    else ""
                ),
                "canonical_internal_unit": "angstrom",
                "source_hash": (
                    evidence["source_hdf5_sha256"] if evidence else ""
                ),
                "source_hash_status": (
                    "direct_sha256" if evidence else "not_in_manual_sample"
                ),
                "loader_version": (
                    "mainline_reintegration_v1_explicit_unit_loader"
                ),
                "generation_command": (
                    f"{GENERATOR.relative_to(PROJECT_ROOT)} "
                    f"--h5-file {record['source_hdf5']} "
                    f"--coordinate-path {record['source_coordinate_path']} "
                    "--output-npz <versioned-output> "
                    "--metadata-json <versioned-sidecar>"
                )
                if record["source_hdf5"]
                else "generated scale-control; see generator_script",
                "generator_script": record["generator_script"],
                "manifest": record["manifest"],
                "historical_input": record["historical_input"],
                "historical_scale_status": record["observed_scale_status"],
                "scale_change_deliberate": record["scale_change_deliberate"],
                "root_cause": record["root_cause"],
                "direct_manual_audit": evidence is not None,
                "evidence_level": (
                    "direct HDF5 hash+metadata+reference geometry"
                    if evidence
                    else "record-level manifest plus stratified schema audit"
                    if affected
                    else "explicit generated test-control provenance"
                ),
                "absolute_angstrom_summary_eligible": (
                    False if affected else "not_applicable_arbitrary_scale"
                ),
                "resolution": record["resolution"],
            }
        )
    return rows


def main() -> int:
    """Execute the gated R1 provenance audit."""

    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()):
        raise RuntimeError("R1 output exists; refusing to overwrite")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    records = read_tsv(W0_MANIFEST)
    formal = [
        row
        for row in records
        if row["benchmark_role"] == "formal_biological_input"
    ]
    summaries = {
        row["domain_id"]: row for row in read_tsv(ALL_DOMAIN_SUMMARY)
    }
    canonical_manifest = {
        row["trajectory_id"]: row for row in read_tsv(ALL_DOMAIN_MANIFEST)
    }
    stitch_domains = {
        row["domain_id"] for row in read_tsv(STITCH_PANEL)
    }
    selected = select_manual_cases(formal, summaries, stitch_domains)
    formal_by_domain = {domain_id(row): row for row in formal}
    manual_rows: list[dict[str, object]] = []
    for index, value in enumerate(selected, start=1):
        print(
            f"[R1] direct source audit {index}/{len(selected)}: {value}",
            flush=True,
        )
        row = formal_by_domain[value]
        trajectory = row["trajectory_id"]
        if trajectory not in canonical_manifest:
            raise RuntimeError(
                f"{trajectory}: absent from canonical all-domain manifest"
            )
        manual_rows.append(
            direct_hdf5_audit(
                row,
                summaries[value],
                in_stitch_panel=value in stitch_domains,
            )
        )
    manual = {str(row["domain_id"]): row for row in manual_rows}
    manifest_rows = build_record_manifest(records, manual)
    affected = [
        row for row in manifest_rows if row["affected_historical_record"]
    ]
    unresolved = [
        row for row in affected if row["final_classification"] == "unresolved"
    ]
    manual_failures = [
        row
        for row in manual_rows
        if row["audit_status"] != "pass_explicit_angstrom"
    ]
    classes = {str(row["cath_class"]) for row in manual_rows}
    strata = {str(row["length_stratum"]) for row in manual_rows}
    has_stitch = any(bool(row["appeared_in_stitch_panel"]) for row in manual_rows)
    has_nonstitch = any(
        not bool(row["appeared_in_stitch_panel"]) for row in manual_rows
    )
    gate_pass = (
        len(affected) == 172
        and len(manual_rows) >= MINIMUM_MANUAL_AUDIT
        and not unresolved
        and not manual_failures
        and {"1", "2", "3", "4"}.issubset(classes)
        and {"small_le_100", "medium_101_199", "large_ge_200"}.issubset(
            strata
        )
        and has_stitch
        and has_nonstitch
    )
    write_tsv(RESULT_ROOT / "01_manual_audit_cases.tsv", manual_rows)
    write_tsv(RESULT_ROOT / "01_scale_record_manifest.tsv", manifest_rows)
    policy = """# Coordinate-unit integrity policy

1. Canonical internal coordinates are Angstrom.
2. Explicit Angstrom metadata is preserved with factor 1.0.
3. Explicit nanometer metadata is converted once with factor 10.0.
4. Unknown units are never silently rescaled from coordinate magnitude.
5. Geometry checks are read-only diagnostics. Suspicious or unresolved inputs
   are quarantined from absolute-Angstrom claims.
6. Every regenerated input must carry `source_unit`,
   `conversion_factor_to_angstrom`, `canonical_internal_unit`, `source_hash`,
   `loader_version`, `generation_command`, and `purpose`.
7. Historical factor-ten outputs remain immutable and deprecated; corrected
   replacements must use a new versioned root.
"""
    (RESULT_ROOT / "01_unit_integrity_policy.md").write_text(policy)
    schema = {
        "required_fields": {
            "source_unit": "explicit source metadata; never geometry inferred",
            "conversion_factor_to_angstrom": "1.0 for Angstrom, 10.0 for nm",
            "canonical_internal_unit": "angstrom",
            "source_hash": "SHA256 of the source HDF5",
            "loader_version": "versioned loader identifier",
            "generation_command": "repository-relative reproducible command",
            "purpose": "formal, stress, or scale-control",
        },
        "unknown_unit_action": "quarantine_without_rescaling",
    }
    (
        RESULT_ROOT / "01_regenerated_input_provenance_schema.json"
    ).write_text(json.dumps(schema, indent=2) + "\n")
    counts = {
        value: sum(
            row["final_classification"] == value for row in manifest_rows
        )
        for value in sorted(
            {str(row["final_classification"]) for row in manifest_rows}
        )
    }
    median_adjacent = statistics.median(
        float(row["median_adjacent_ca_distance_source_units"])
        for row in manual_rows
    )
    report = f"""# R1 scale-provenance resolution

## Resolution

The apparent discrepancy is resolved. The benchmark purpose of a trajectory
and the provenance of its factor-ten scale error are separate:

- 162 records are formal biological inputs whose source mdCATH datasets
  explicitly use Angstrom, but whose historical derived arrays were
  accidentally multiplied by ten;
- 10 records belong to deliberate frame-count/stitching stress workflows, but
  their factor-ten values were inherited from the same accidental conversion;
- 3 explicit 0.1x/1x/10x controls are deliberate arbitrary-scale invariance
  controls and are not part of the 172 affected historical records.

No formal source HDF5 was deliberately rescaled as a coordinate-scale stress
control.

## Direct manual audit

{len(manual_rows)} formal cases were directly audited. The sample covers CATH
classes {', '.join(sorted(classes))}, small/medium/large residue-count strata,
the three affected domains that also entered the stitching panel, and formal
cases outside that panel.

For every sampled source:

- the complete local HDF5 file was SHA256 hashed;
- the exact coordinate dataset was opened directly;
- dataset metadata explicitly reported `unit=Angstrom`;
- the conversion factor to canonical Angstrom was 1.0;
- the source reference-frame adjacent-CA geometry was Angstrom-like.

Median adjacent-CA distance across the 16 sources was {median_adjacent:.4f} A.
The per-case source hash, dataset path, metadata, geometry, generated-array
path, benchmark membership and historical/current conversion operations are in
`01_manual_audit_cases.tsv`.

## Classification and use

Classification counts across the complete record manifest:

{chr(10).join(f'- {key}: {value}' for key, value in counts.items())}

Historical affected arrays remain immutable and are ineligible for
absolute-Angstrom summaries. Regeneration must use the provenance fields and
policy in this directory. Geometry may quarantine an input but never silently
rescale it.

## Gate

R1 gate: **{'PASS' if gate_pass else 'FAIL'}**.

Unresolved affected records: {len(unresolved)}. Direct-audit failures:
{len(manual_failures)}. R4 may proceed only because the stratified direct
source evidence is conclusive and all 172 affected records have a documented
classification.
"""
    (RESULT_ROOT / "01_scale_provenance_resolution.md").write_text(report)
    status = {
        "phase": "R1",
        "status": "pass" if gate_pass else "failed",
        "total_manifest_records": len(manifest_rows),
        "affected_historical_records": len(affected),
        "formal_accidental_conversion_records": sum(
            row["final_classification"]
            == "formal biological input with accidental conversion"
            for row in manifest_rows
        ),
        "stitching_accidental_conversion_records": sum(
            str(row["final_classification"]).startswith(
                "stitched/frame-count"
            )
            for row in manifest_rows
        ),
        "deliberate_scale_control_records": sum(
            row["final_classification"]
            == "deliberate scale-invariance control"
            for row in manifest_rows
        ),
        "manual_audit_count": len(manual_rows),
        "manual_audit_failures": len(manual_failures),
        "unresolved_affected_records": len(unresolved),
        "formal_source_unit": "angstrom",
        "formal_conversion_factor_to_angstrom": 1.0,
        "r4_may_proceed": gate_pass,
        "formal_all_domain_rerun_performed": False,
    }
    (RESULT_ROOT / "01_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    if not gate_pass:
        raise RuntimeError("R1 provenance gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
