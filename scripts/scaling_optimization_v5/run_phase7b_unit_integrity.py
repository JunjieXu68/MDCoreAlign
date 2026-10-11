#!/usr/bin/env python
"""Execute Phase 7B coordinate-unit provenance and regeneration only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from mdsuperpose.coordinate_scale import audit_coordinate_scale
from mdsuperpose.representative_weights import (
    estimate_representative_weights,
)
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_refine import (
    compute_frozen_window_signals,
    temporally_stratified_indices,
)
from scripts.mdcath_benchmark.build_transition_aligned_stitched_trajectory import (
    build_transition_aligned,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.scaling_optimization.build_phase2_regression import (
    ROBUST_KWARGS,
    build_cases,
)
from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes
from scripts.scaling_optimization_v4.run_phase7a_rmsd_audit import (
    apply_transforms_independent,
    fit_transforms,
    hard_core,
    independent_evaluation,
    jaccard,
    locked_candidates,
    normalized_weights,
    rotation_angles,
    run_batched,
    run_locked_candidate,
    safe_spearman,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase7b_unit_integrity"
)
LEGACY_LONG_ROOT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
)
ALL_DOMAIN_ROOT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/all_domain_canonical/run_fullframes/"
    "per_trajectory"
)
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}
AFFECTED_DOMAINS = ("1bctA00", "4oelA02", "1te7A00")
REPLICATE_LABELS = ("A", "B", "C", "D", "E")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a table using the union of row fields."""

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


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def labels_to_metadata(
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract chain and residue identifiers from mdCATH labels."""

    chains: list[str] = []
    residues: list[str] = []
    for index, label in enumerate(labels, start=1):
        parts = str(label).split(":")
        chains.append(parts[0] if len(parts) >= 2 else "_")
        residues.append(parts[1] if len(parts) >= 3 else str(index))
    return np.asarray(chains), np.asarray(residues)


def audit_array(
    trajectory_id: str,
    scope: str,
    coords: np.ndarray,
    *,
    labels: list[str] | None = None,
    chain: np.ndarray | None = None,
    residue_ids: np.ndarray | None = None,
    source: str,
    source_unit_status: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Audit one in-memory trajectory and annotate its provenance."""

    if labels is not None:
        chain, residue_ids = labels_to_metadata(labels)
    summary, frames = audit_coordinate_scale(
        coords,
        chain_ids=chain,
        residue_ids=residue_ids,
    )
    row = {
        "trajectory_id": trajectory_id,
        "scope": scope,
        "source": source,
        "source_unit_status": source_unit_status,
        **summary.to_dict(),
    }
    for frame in frames:
        frame.update({"trajectory_id": trajectory_id, "scope": scope})
    return row, frames


def selected_domain_index() -> dict[str, dict[str, str]]:
    """Index the frozen 20-domain synthetic panel."""

    rows = read_tsv(LEGACY_LONG_ROOT / "selected_domains.tsv")
    return {row["domain_id"]: row for row in rows}


def audit_phase7_cases() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Re-audit all 34 Phase 7A case inputs without modifying them."""

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for case in build_cases():
        coords, chain = case.loader()
        values = np.asarray(coords, dtype=np.float64)
        if case.category == "toy":
            summaries.append(
                {
                    "trajectory_id": case.case_id,
                    "scope": "phase7a_golden_panel",
                    "source": case.source,
                    "source_unit_status": "dimensionless_toy",
                    "status": "toy_arbitrary_units",
                    "inferred_internal_unit": "arbitrary",
                    "n_frames": values.shape[0],
                    "n_atoms": values.shape[1],
                    "notes": "Excluded from physical Angstrom summaries.",
                }
            )
            continue
        row, per_frame = audit_array(
            case.case_id,
            "phase7a_golden_panel",
            values,
            chain=np.asarray(chain) if chain is not None else None,
            source=case.source,
            source_unit_status="as_frozen",
        )
        summaries.append(row)
        frames.extend(per_frame)
    return summaries, frames


def audit_legacy_raw_replicates() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    """Audit every cached raw replicate in the 20-domain stitching panel."""

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for path in sorted(
        (LEGACY_LONG_ROOT / "domains").glob(
            "*/raw_replicates/*_ca_coords.npz"
        )
    ):
        with np.load(path, allow_pickle=True) as payload:
            coords = np.asarray(payload["coords"], dtype=np.float64)
            labels = [str(value) for value in payload["labels"].tolist()]
        metadata_path = path.with_name(
            path.name.replace("_ca_coords.npz", "_metadata.json")
        )
        metadata = (
            json.loads(metadata_path.read_text())
            if metadata_path.is_file()
            else {}
        )
        row, per_frame = audit_array(
            f"{path.parents[1].name}_{path.name[0]}",
            "legacy_transition_raw_replicate",
            coords,
            labels=labels,
            source=path.relative_to(PROJECT_ROOT).as_posix(),
            source_unit_status=str(metadata.get("unit_status", "unavailable")),
        )
        summaries.append(row)
        frames.extend(per_frame)
    return summaries, frames


def _pdb_models(
    path: Path, *, first_model_only: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Read a retained CA-only PDB, optionally stopping after model one."""

    models: list[np.ndarray] = []
    current: list[list[float]] = []
    labels: list[str] = []
    first_labels: list[str] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                if line[12:16].strip() != "CA":
                    continue
                current.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
                if not models:
                    labels.append(
                        f"{line[21].strip() or '_'}:"
                        f"{line[22:26].strip()}:"
                        f"{line[17:20].strip() or 'UNK'}"
                    )
            elif line.startswith("ENDMDL") and current:
                models.append(np.asarray(current, dtype=np.float64))
                if len(models) == 1:
                    first_labels = list(labels)
                current = []
                labels = []
                if first_model_only:
                    break
    if current:
        models.append(np.asarray(current, dtype=np.float64))
        if not first_labels:
            first_labels = list(labels)
    if not models:
        raise ValueError(f"no CA models in {path}")
    return np.asarray(models), first_labels


def audit_frozen_all_domain() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    """Audit all retained all-domain trajectory inputs model by model."""

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    summary_paths = sorted(ALL_DOMAIN_ROOT.glob("*/summary.json"))
    for index, summary_path in enumerate(summary_paths, start=1):
        payload = json.loads(summary_path.read_text())
        trajectory_id = str(payload.get("trajectory_id", summary_path.parent.name))
        input_metadata = payload.get("input_metadata", {})
        candidates = (
            summary_path.parent
            / "external_mdlovofit"
            / "trajectory_input.pdb",
            summary_path.parent
            / "external_theseus"
            / "trajectory_input.pdb",
        )
        pdb_path = next((path for path in candidates if path.is_file()), None)
        if pdb_path is None:
            summaries.append(
                {
                    "trajectory_id": trajectory_id,
                    "scope": "frozen_all_domain",
                    "source": "unavailable",
                    "source_unit_status": input_metadata.get(
                        "unit_status", "unavailable"
                    ),
                    "status": "quarantined_missing_retained_input",
                    "notes": "No retained PDB input was available for direct audit.",
                }
            )
            continue
        coords, labels = _pdb_models(pdb_path, first_model_only=True)
        row, _ = audit_array(
            trajectory_id,
            "frozen_all_domain",
            coords,
            labels=labels,
            source=pdb_path.relative_to(PROJECT_ROOT).as_posix(),
            source_unit_status=str(
                input_metadata.get("unit_status", "unavailable")
            ),
        )
        row["n_frames"] = int(input_metadata.get("n_frames", 0) or 0)
        row["frames_directly_audited"] = 1
        row["scale_consistency_basis"] = (
            "retained first frame plus provenance: historical loader applied "
            "one uniform scale operation to the complete trajectory array"
        )
        summaries.append(row)
        if index % 250 == 0:
            print(
                f"[phase7b] audited frozen all-domain {index}/"
                f"{len(summary_paths)}",
                flush=True,
            )
    return summaries, frames


def regenerate_affected_cases() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, tuple[np.ndarray, list[str], list[dict[str, str]]]],
]:
    """Regenerate affected raw and 10k stitched arrays from verified sources."""

    index = selected_domain_index()
    manifest: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    stitched_outputs: dict[
        str, tuple[np.ndarray, list[str], list[dict[str, str]]]
    ] = {}
    for domain_id in AFFECTED_DOMAINS:
        source_h5 = Path(index[domain_id]["h5_path"])
        domain_root = RESULT_ROOT / "regenerated" / domain_id
        raw_root = domain_root / "raw_replicates"
        raw_root.mkdir(parents=True, exist_ok=True)
        raw: dict[
            str, tuple[np.ndarray, list[str], list[dict[str, str]]]
        ] = {}
        for replicate_index, label in enumerate(REPLICATE_LABELS):
            coords, labels, residue_metadata, metadata = extract_ca_trajectory(
                source_h5,
                coordinate_path=f"/{domain_id}/320/{replicate_index}/coords",
                min_ca_atoms=30,
                min_frames=20,
            )
            coords = coords - np.mean(coords, axis=1, keepdims=True)
            audit, _ = audit_array(
                f"regenerated_{domain_id}_{label}",
                "phase7b_regenerated_raw",
                coords,
                labels=labels,
                source=str(source_h5),
                source_unit_status=str(metadata["unit_status"]),
            )
            if audit["status"] != "angstrom_consistent":
                raise RuntimeError(
                    f"regenerated {domain_id}/{label} failed scale guard"
                )
            out_npz = raw_root / f"{label}_ca_coords.npz"
            out_json = raw_root / f"{label}_metadata.json"
            np.savez_compressed(
                out_npz,
                coords=coords.astype(np.float32),
                labels=np.asarray(labels, dtype=object),
                residue_metadata=np.asarray(residue_metadata, dtype=object),
            )
            metadata.update(
                {
                    "coordinate_preprocessing": "per-frame CA centroid removal",
                    "phase7b_scale_guard": audit,
                }
            )
            out_json.write_text(json.dumps(metadata, indent=2) + "\n")
            original = (
                LEGACY_LONG_ROOT
                / "domains"
                / domain_id
                / "raw_replicates"
                / f"{label}_ca_coords.npz"
            )
            manifest.append(
                {
                    "case_id": f"{domain_id}_{label}",
                    "artifact_type": "raw_replicate",
                    "original_path": original.relative_to(PROJECT_ROOT).as_posix(),
                    "original_sha256": sha256(original),
                    "original_status": "quarantined_immutable",
                    "regenerated_path": out_npz.relative_to(PROJECT_ROOT).as_posix(),
                    "regenerated_sha256": sha256(out_npz),
                    "source_h5": str(source_h5),
                    "source_dataset": metadata["coordinate_path"],
                    "source_unit": "Angstrom",
                    "conversion_status": metadata["unit_status"],
                    "scale_guard_status": audit["status"],
                }
            )
            audits.append(audit)
            raw[label] = (coords, labels, residue_metadata)

        lengths = {label: raw[label][0].shape[0] for label in REPLICATE_LABELS}
        sequence: list[str] = []
        total = 0
        sequence_index = 0
        while total < 10000:
            label = REPLICATE_LABELS[sequence_index % len(REPLICATE_LABELS)]
            sequence.append(label)
            total += lengths[label]
            sequence_index += 1
        stitched, transforms, boundaries, boundary_indices = (
            build_transition_aligned(
                domain_id,
                raw,
                sequence,
                10000,
                "canonical_cycle",
                "keep_boundary_first_frame",
            )
        )
        labels = raw["A"][1]
        residue_metadata = raw["A"][2]
        stitched_audit, _ = audit_array(
            f"regenerated_{domain_id}_10000",
            "phase7b_regenerated_stitched",
            stitched,
            labels=labels,
            source="verified_Angstrom_A-E_replicates",
            source_unit_status="verified_angstrom_metadata",
        )
        if stitched_audit["status"] != "angstrom_consistent":
            raise RuntimeError(
                f"regenerated stitched {domain_id} failed scale guard"
            )
        stitched_path = domain_root / "transition_aligned_coords_10000.npz"
        np.savez_compressed(
            stitched_path,
            coords=stitched.astype(np.float32),
            labels=np.asarray(labels, dtype=object),
            residue_metadata=np.asarray(residue_metadata, dtype=object),
            boundary_indices=np.asarray(boundary_indices, dtype=np.int64),
        )
        write_tsv(domain_root / "stitch_transforms.tsv", transforms)
        write_tsv(domain_root / "boundary_diagnostics.tsv", boundaries)
        original_stitched = (
            LEGACY_LONG_ROOT
            / "runs"
            / domain_id
            / "10000"
            / "transition_aligned_coords.npz"
        )
        manifest.append(
            {
                "case_id": f"{domain_id}_10000",
                "artifact_type": "transition_aligned_stitched",
                "original_path": original_stitched.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "original_sha256": sha256(original_stitched),
                "original_status": "quarantined_immutable",
                "regenerated_path": stitched_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "regenerated_sha256": sha256(stitched_path),
                "source_h5": str(source_h5),
                "source_dataset": "A-E 320K coordinate datasets",
                "source_unit": "Angstrom",
                "conversion_status": "verified_angstrom_metadata",
                "scale_guard_status": stitched_audit["status"],
            }
        )
        audits.append(stitched_audit)
        stitched_outputs[domain_id] = (
            stitched,
            labels,
            residue_metadata,
        )
    return manifest, audits, stitched_outputs


def recompute_affected_comparisons(
    regenerated: dict[
        str, tuple[np.ndarray, list[str], list[dict[str, str]]]
    ],
) -> list[dict[str, object]]:
    """Rerun only regenerated Phase 6C-R/7A method comparisons."""

    candidates = locked_candidates()
    rows: list[dict[str, object]] = []
    case_inputs: list[
        tuple[str, np.ndarray, list[str], list[dict[str, str]]]
    ] = []
    for domain_id, (coords, labels, metadata) in regenerated.items():
        indices = np.linspace(
            0, coords.shape[0] - 1, 100, dtype=np.int64
        )
        case_inputs.append(
            (f"mdcath_{domain_id}", coords[indices], labels, metadata)
        )
    coords, labels, metadata = regenerated["4oelA02"]
    case_inputs.append(
        (
            "synthetic_scaling_4oelA02_10000",
            coords,
            labels,
            metadata,
        )
    )
    for case_id, coords, labels, metadata in case_inputs:
        chains, residues = labels_to_metadata(labels)
        reference = coords[0]
        representative_indices = temporally_stratified_indices(
            coords.shape[0], min(2000, coords.shape[0])
        )
        exact = estimate_representative_weights_sequential_lean(
            coords[representative_indices],
            reference,
            sequence_chain_ids=chains,
            **ROBUST_KWARGS,
        )
        exact_weights = np.asarray(exact.mean_weights, dtype=np.float64)
        exact_rotations, exact_translations = fit_transforms(
            coords, reference, exact_weights
        )
        from scripts.scaling_optimization_v4.run_phase7a_rmsd_audit import (
            MethodTransforms,
        )

        sequential = MethodTransforms(
            method="sequential_lean_exact",
            weights=exact_weights,
            rotations=exact_rotations,
            translations=exact_translations,
            source="phase7b_regenerated",
        )
        signals = compute_frozen_window_signals(
            coords, chain_ids=chains
        )
        methods = [sequential]
        methods.extend(
            run_locked_candidate(coords, chains, candidate, signals)
            for candidate in candidates
        )
        methods.append(run_batched(coords, chains))
        sequential_core = hard_core(sequential.weights)
        sequential_eval = independent_evaluation(
            coords,
            reference,
            sequential,
            sequential_core,
            sequential_core,
            None,
        )
        scale_audit, _ = audit_coordinate_scale(
            coords, chain_ids=chains, residue_ids=residues
        )
        for method in methods:
            core = hard_core(method.weights)
            evaluation = (
                sequential_eval
                if method.method == "sequential_lean_exact"
                else independent_evaluation(
                    coords,
                    reference,
                    method,
                    sequential_core,
                    core,
                    None,
                )
            )
            difference = evaluation.full_rmsd - sequential_eval.full_rmsd
            angle = rotation_angles(
                method.rotations, sequential.rotations
            )
            aligned_difference = np.sqrt(
                np.mean(
                    np.sum(
                        (
                            evaluation.aligned
                            - sequential_eval.aligned
                        )
                        ** 2,
                        axis=2,
                    ),
                    axis=1,
                )
            )
            rows.append(
                {
                    "case_id": case_id,
                    "method": method.method,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "coordinate_scale_status": scale_audit.status,
                    "mean_unified_posthoc_rmsd_A": float(
                        np.mean(evaluation.full_rmsd)
                    ),
                    "median_absolute_rmsd_difference_vs_exact_A": float(
                        np.median(np.abs(difference))
                    ),
                    "p95_absolute_rmsd_difference_vs_exact_A": float(
                        np.quantile(np.abs(difference), 0.95)
                    ),
                    "core_jaccard_vs_exact": jaccard(
                        core, sequential_core
                    ),
                    "weight_spearman_vs_exact": safe_spearman(
                        normalized_weights(method.weights),
                        normalized_weights(sequential.weights),
                    ),
                    "rmsf_spearman_vs_exact": safe_spearman(
                        evaluation.rmsf, sequential_eval.rmsf
                    ),
                    "mean_absolute_rmsf_difference_vs_exact_A": float(
                        np.mean(
                            np.abs(
                                evaluation.rmsf - sequential_eval.rmsf
                            )
                        )
                    ),
                    "effective_core_size": float(
                        np.sum(normalized_weights(method.weights))
                    ),
                    "median_rotation_geodesic_degrees": float(
                        np.median(angle)
                    ),
                    "median_transform_coordinate_displacement_A": float(
                        np.median(aligned_difference)
                    ),
                    "mean_weighted_objective_A": float(
                        np.mean(evaluation.weighted_objective)
                    ),
                }
            )
    return rows


def root_cause_rows() -> list[dict[str, object]]:
    """Record the proven root cause for each affected final case."""

    return [
        {
            "case_id": "mdcath_1bctA00",
            "affected_replicates": "A;B;C;D",
            "source_data_unit": "Angstrom",
            "source_geometry_status": "3.836-3.839_A_adjacent_CA",
            "loader_status": "incorrect_factor_10",
            "root_cause": (
                "explicit unit=Angstrom metadata was ignored; translation-"
                "dependent median_abs<20 heuristic multiplied coordinates by 10"
            ),
            "stitching_role": "propagated_preexisting_scale_error",
            "serialization_role": "preserved_error_only",
            "reporting_role": "reported_raw_values_as_Angstrom",
            "resolution": "original_quarantined_and_regenerated_from_HDF5",
        },
        {
            "case_id": "mdcath_4oelA02",
            "affected_replicates": "A",
            "source_data_unit": "Angstrom",
            "source_geometry_status": "3.834_A_adjacent_CA",
            "loader_status": "incorrect_factor_10",
            "root_cause": (
                "explicit unit=Angstrom metadata was ignored; translation-"
                "dependent median_abs<20 heuristic multiplied coordinates by 10"
            ),
            "stitching_role": "propagated_preexisting_scale_error",
            "serialization_role": "preserved_error_only",
            "reporting_role": "reported_raw_values_as_Angstrom",
            "resolution": "original_quarantined_and_regenerated_from_HDF5",
        },
        {
            "case_id": "mdcath_1te7A00",
            "affected_replicates": "A",
            "source_data_unit": "Angstrom",
            "source_geometry_status": "3.832_A_adjacent_CA",
            "loader_status": "incorrect_factor_10",
            "root_cause": (
                "explicit unit=Angstrom metadata was ignored; translation-"
                "dependent median_abs<20 heuristic multiplied coordinates by 10"
            ),
            "stitching_role": "propagated_preexisting_scale_error",
            "serialization_role": "preserved_error_only",
            "reporting_role": "reported_raw values_as_Angstrom",
            "resolution": "original_quarantined_and_regenerated_from_HDF5",
        },
        {
            "case_id": "synthetic_scaling_4oelA02_10000",
            "affected_replicates": "repeated transformed copies of A",
            "source_data_unit": "Angstrom",
            "source_geometry_status": "source_A_was_erroneously_10x",
            "loader_status": "upstream_incorrect_factor_10",
            "root_cause": (
                "transition-aligned stitching consumed the contaminated 4oelA02 "
                "A replicate and mixed it with correctly scaled B-E"
            ),
            "stitching_role": "mixed_incompatible_scale_segments",
            "serialization_role": "preserved_error_only",
            "reporting_role": "frame_pooling_amplified_raw_unit_discrepancy",
            "resolution": "original_quarantined_and_regenerated_from_HDF5",
        },
    ]


def write_documents(
    all_rows: list[dict[str, object]],
    regenerated_manifest: list[dict[str, object]],
    comparisons: list[dict[str, object]],
) -> None:
    """Write provenance, scope, guard and gate documents."""

    frozen_all = [
        row for row in all_rows if row["scope"] == "frozen_all_domain"
    ]
    frozen_factor10 = [
        row
        for row in frozen_all
        if row.get("status") == "uniform_factor_ten_suspect"
    ]
    regenerated = [
        row
        for row in all_rows
        if str(row["scope"]).startswith("phase7b_regenerated")
    ]
    comparison_approx = [
        row
        for row in comparisons
        if row["method"] != "sequential_lean_exact"
    ]
    gate_pass = (
        len(regenerated) == 18
        and all(
            row.get("status") == "angstrom_consistent"
            for row in regenerated
        )
        and all(
            row["coordinate_scale_status"] == "angstrom_consistent"
            for row in comparisons
        )
    )
    provenance = f"""# Phase 7B coordinate-unit provenance

## Proven root cause

The source mdCATH HDF5 datasets for all A-E replicates of `1bctA00`,
`4oelA02`, and `1te7A00` explicitly declare `unit=Angstrom`. Direct source
inspection gives median adjacent CA distances of 3.83-3.84 A in every
replicate. The source data are not mixed-scale.

The contamination entered in
`scripts/mdcath_benchmark/mdcath_common.py::maybe_convert_units_to_angstrom`.
The historical implementation did not recognize explicit Angstrom metadata.
It instead multiplied any coordinate array whose translation-dependent median
absolute coordinate was between 0.1 and 20 by ten. PBC/global placement made
this threshold differ between replicates of the same domain.

The resulting errors were:

- `1bctA00`: A-D multiplied by ten; E unchanged;
- `4oelA02`: A multiplied by ten; B-E unchanged;
- `1te7A00`: A multiplied by ten; B-E unchanged.

Transition-aligned stitching applied rigid transforms but no scaling, so it
propagated and mixed these already incompatible segments. NPZ serialization
preserved the error exactly. Phase 6/7 RMSD reporting then labelled raw
coordinate values as Angstrom; frame pooling made the 10k `4oelA02` derivative
dominate the old aggregate.

## Correction policy

No automatic per-frame rescaling was implemented. Explicit nanometer metadata
is converted by a factor of ten; explicit Angstrom metadata is preserved; an
unknown unit is preserved and must pass the geometry guard. The four historical
derived cases remain immutable and quarantined. Regenerated replacements were
written under `results/scaling_optimization_v5/phase7b_unit_integrity/`.

Phase 7B gate status: **{'PASS' if gate_pass else 'FAIL'}**.
"""
    (RESULT_ROOT / "coordinate_unit_provenance.md").write_text(provenance)

    scope = f"""# Scope of impact

- Phase 6/7 experimental panel: four final cases and six cached raw replicates
  were contaminated. All are explicitly quarantined; regenerated replacements
  pass the Angstrom geometry guard.
- 34-case golden panel: four frozen entries are affected. Their files and
  hashes were not modified. Replacement comparisons are reported separately.
- Frozen all-domain benchmark: {len(frozen_factor10)} of {len(frozen_all)}
  retained trajectory inputs show uniform factor-ten geometry. These results
  are an integrity finding and must not support Angstrom RMSD claims. They were
  not overwritten.
- Project completion: outputs were not modified. Any table derived from the
  affected all-domain rows requires a versioned unit-clean replacement before
  reuse.
- Manuscript figures/summaries: Phase 7A already removed the mixed-scale pooled
  value as a physical claim. Historical all-domain absolute RMSD tables require
  the same caveat.

The regenerated manifest contains {len(regenerated_manifest)} immutable-old to
new artifact mappings. The affected comparison table contains
{len(comparison_approx)} approximate-method rows recalculated in Angstrom.
"""
    (RESULT_ROOT / "scope_of_impact.md").write_text(scope)

    guard = """# Coordinate-scale guard

`mdcorealign audit-coordinate-scale` performs a read-only chain-aware audit.
It excludes cross-chain adjacency and explicit residue-number gaps, reports
per-frame adjacent-CA and fixed-reference-contact distances, detects abrupt
scale changes and factor-ten clusters, and never modifies coordinates.

Permanent mdCATH extraction rules:

1. explicit Angstrom metadata: preserve coordinates;
2. explicit nanometer metadata: convert once to Angstrom;
3. missing metadata: preserve coordinates, report provenance, and require the
   geometry guard;
4. mixed or factor-ten-suspect geometry: quarantine and fail validation;
5. never infer units from absolute coordinate position.

Example:

```bash
mdcorealign audit-coordinate-scale \\
  --input trajectory.npz \\
  --output-dir coordinate_scale_audit
```
"""
    (RESULT_ROOT / "coordinate_scale_guard.md").write_text(guard)


def run_tests() -> dict[str, object]:
    """Run the full test suite with the canonical interpreter."""

    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    output = completed.stdout + completed.stderr
    (RESULT_ROOT / "full_test_results.txt").write_text(output)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": output.strip().splitlines()[-1] if output.strip() else "",
    }


def main() -> int:
    """Run Phase 7B and stop at its gate."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    integrity_before = verify_hashes()
    if integrity_before["status"] != "pass":
        raise RuntimeError("frozen hash integrity failed before Phase 7B")

    summaries, frames = audit_phase7_cases()
    raw_summaries, raw_frames = audit_legacy_raw_replicates()
    summaries.extend(raw_summaries)
    frames.extend(raw_frames)

    all_domain_summaries, all_domain_frames = audit_frozen_all_domain()
    summaries.extend(all_domain_summaries)
    frames.extend(all_domain_frames)

    regenerated_manifest, regenerated_audits, regenerated = (
        regenerate_affected_cases()
    )
    summaries.extend(regenerated_audits)
    comparisons = recompute_affected_comparisons(regenerated)

    write_tsv(RESULT_ROOT / "all_trajectory_scale_audit.tsv", summaries)
    write_tsv(
        RESULT_ROOT / "per_frame_scale_diagnostics.tsv", frames
    )
    write_tsv(
        RESULT_ROOT / "affected_case_root_causes.tsv", root_cause_rows()
    )
    write_tsv(
        RESULT_ROOT / "regenerated_case_manifest.tsv",
        regenerated_manifest,
    )
    write_tsv(
        RESULT_ROOT / "regenerated_affected_comparisons.tsv", comparisons
    )
    write_documents(summaries, regenerated_manifest, comparisons)

    affected_gate = all(
        row["scale_guard_status"] == "angstrom_consistent"
        for row in regenerated_manifest
    )
    tests = run_tests()
    integrity_after = verify_hashes()
    gate = {
        "phase": "7B",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root_cause_documented_count": len(root_cause_rows()),
        "affected_final_case_count": 4,
        "regenerated_artifact_count": len(regenerated_manifest),
        "regenerated_scale_guard_pass": affected_gate,
        "phase8a_input_policy": (
            "fresh extraction from explicit Angstrom HDF5 sources; all "
            "selected cases must pass coordinate-scale guard"
        ),
        "tests": tests,
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "public_default_changed": False,
        "git_metadata_changed": False,
        "project_completion_changed": False,
        "phase7b_stop_go": (
            "pass"
            if affected_gate
            and tests["status"] == "pass"
            and integrity_after["status"] == "pass"
            else "fail"
        ),
    }
    (RESULT_ROOT / "phase7b_status.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    (RESULT_ROOT / "frozen_hash_integrity_report.json").write_text(
        json.dumps(
            {
                "before": integrity_before,
                "after": integrity_after,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["phase7b_stop_go"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
