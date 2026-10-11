#!/usr/bin/env python
"""Audit Phase 6C-R RMSD definitions without changing any solver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np

from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.fast_kabsch import weighted_kabsch_batch
from mdsuperpose.representative_weights import estimate_representative_weights
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_refine import (
    bounded_global_weight_refinement,
    compute_frozen_window_signals,
    solve_window_preconditioned_refinement,
    temporally_stratified_indices,
)
from scripts.scaling_optimization.build_phase2_regression import (
    FORMAL_SYSTEMS,
    ROBUST_KWARGS,
    RegressionCase,
    build_cases,
)
from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes
from scripts.scaling_optimization_v4.run_phase6b import (
    contact_persistence,
    jaccard,
    pair_distance_stability,
    safe_spearman,
)
from scripts.scaling_optimization_v4.run_phase6c_r import (
    Candidate,
    candidate_from_dict,
    make_initial,
    signal_names_for,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT
    / "results/scaling_optimization_v4/phase7a_rmsd_audit"
)
PHASE6_ROOT = PROJECT_ROOT / "results/scaling_optimization_v4/window_refine"
GOLDEN_MANIFEST = (
    PROJECT_ROOT / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
)
SEMANTIC_ROOT = PROJECT_ROOT / "results/scaling_optimization_v3/semantic"
REFERENCE_SENSITIVITY = (
    PROJECT_ROOT
    / "results/md_trajectory_focus/reference_sensitivity/reference_sensitivity.tsv"
)
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
CORE_FRACTION = 0.6
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}
METHOD_ORDER = (
    "sequential_lean_exact",
    "uniform_low_budget",
    "w5_preconditioned",
    "graph_preconditioned",
    "batched_independent",
)


@dataclass(slots=True)
class MethodTransforms:
    """Weights and exact transforms produced by one existing method."""

    method: str
    weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    source: str


@dataclass(slots=True)
class IndependentEvaluation:
    """Metrics recomputed only from coordinates and supplied transforms."""

    aligned: np.ndarray
    full_rmsd: np.ndarray
    common_core_rmsd: np.ndarray
    common_mobile_rmsd: np.ndarray
    method_core_rmsd: np.ndarray
    method_mobile_rmsd: np.ndarray
    hinge_rmsd: np.ndarray | None
    weighted_objective: np.ndarray
    rmsf: np.ndarray


def environment() -> dict[str, str]:
    """Return the canonical fixed-thread environment."""

    values = os.environ.copy()
    values.update(THREAD_ENVIRONMENT)
    return values


def sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows using their union schema."""

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


def load_golden_manifest() -> dict[str, dict[str, str]]:
    """Index immutable golden references."""

    with GOLDEN_MANIFEST.open(newline="") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["status"] == "success"
        }


def hard_core(weights: np.ndarray) -> np.ndarray:
    """Apply the frozen 60% stable-core reporting rule."""

    return select_core_by_target_fraction(
        np.asarray(weights, dtype=np.float64),
        CORE_FRACTION,
    ).core_mask


def apply_transforms_independent(
    coords: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Apply row-vector transforms independently of production helpers."""

    values = np.asarray(coords, dtype=np.float64)
    operators = np.swapaxes(np.asarray(rotations, dtype=np.float64), 1, 2)
    shifts = np.asarray(translations, dtype=np.float64)
    return np.einsum("tni,tij->tnj", values, operators) + shifts[:, None, :]


def rmsd_from_aligned(
    aligned: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ordinary RMSD over an explicit evaluation selection."""

    values = np.asarray(aligned, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    squared = np.sum((values - target[None, :, :]) ** 2, axis=2)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if not np.any(selected):
            return np.full(values.shape[0], np.nan, dtype=np.float64)
        squared = squared[:, selected]
    return np.sqrt(np.mean(squared, axis=1))


def weighted_objective_from_aligned(
    aligned: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute the weighted fitting objective from transformed coordinates."""

    fixed = np.asarray(weights, dtype=np.float64)
    squared = np.sum(
        (np.asarray(aligned) - np.asarray(reference)[None, :, :]) ** 2,
        axis=2,
    )
    return np.sqrt(np.einsum("tn,n->t", squared, fixed) / np.sum(fixed))


def rmsf_from_aligned(aligned: np.ndarray) -> np.ndarray:
    """Compute RMSF without calling the production flexibility module."""

    values = np.asarray(aligned, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=0))


def fit_transforms(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit transforms in chunks while ignoring solver-reported metrics."""

    rotations = np.empty((coords.shape[0], 3, 3), dtype=np.float64)
    translations = np.empty((coords.shape[0], 3), dtype=np.float64)
    for start in range(0, coords.shape[0], chunk_size):
        stop = min(start + chunk_size, coords.shape[0])
        fit = weighted_kabsch_batch(
            coords[start:stop],
            reference,
            weights,
            validate_inputs=False,
            compute_metrics=False,
        )
        rotations[start:stop] = fit.rotations
        translations[start:stop] = fit.translations
    return rotations, translations


def regression(
    predictor: np.ndarray,
    response: np.ndarray,
) -> tuple[float, float, float]:
    """Return correlation, slope and intercept with constant handling."""

    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-15:
        correlation = 1.0 if np.allclose(x, y) else 0.0
        return correlation, float("nan"), float(np.mean(y))
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    return correlation, float(slope), float(intercept)


def rotation_angles(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Compute geodesic rotation disagreement in degrees."""

    relative = np.einsum("tij,tkj->tik", candidate, reference)
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def hinge_mask(case_id: str, n_residues: int) -> np.ndarray | None:
    """Return only the explicit hinge annotation encoded by the toy generator."""

    if case_id != "toy_hinge" or n_residues != 24:
        return None
    mask = np.zeros(n_residues, dtype=bool)
    mask[10:14] = True
    return mask


def coordinate_scale_audit(
    case_id: str,
    category: str,
    coords: np.ndarray,
    chain: np.ndarray | None,
) -> dict[str, object]:
    """Classify coordinate scale from chain-contiguous CA neighbor distances.

    This is an audit diagnostic, not a coordinate conversion. It deliberately
    leaves mixed-scale frozen inputs unchanged.
    """

    values = np.asarray(coords, dtype=np.float64)
    valid_pairs = np.ones(values.shape[1] - 1, dtype=bool)
    if chain is not None:
        labels = np.asarray(chain)
        valid_pairs = labels[1:] == labels[:-1]
    if not np.any(valid_pairs):
        raise ValueError(f"{case_id} has no chain-contiguous neighbor pairs")
    neighbor_distances = np.linalg.norm(
        values[:, 1:] - values[:, :-1], axis=2
    )[:, valid_pairs]
    frame_medians = np.median(neighbor_distances, axis=1)
    angstrom_like = frame_medians < 10.0
    tenfold_like = frame_medians > 20.0
    ambiguous = ~(angstrom_like | tenfold_like)
    if category == "toy":
        status = "synthetic_toy_arbitrary_units"
        physical_factor = float("nan")
    elif np.any(angstrom_like) and np.any(tenfold_like):
        status = "mixed_scale_contaminated"
        physical_factor = float("nan")
    elif np.all(tenfold_like):
        status = "uniform_tenfold_scale_suspect"
        physical_factor = 0.1
    elif np.all(angstrom_like) and 2.5 <= np.median(frame_medians) <= 5.0:
        status = "angstrom_consistent"
        physical_factor = 1.0
    else:
        status = "undetermined"
        physical_factor = float("nan")

    domain_id = case_id.removeprefix("mdcath_")
    if case_id == "synthetic_scaling_4oelA02_10000":
        domain_id = "4oelA02"
    metadata_root = (
        PROJECT_ROOT
        / "results/mdcath_benchmark/"
        "transition_aligned_long_mode_benchmark/domains"
        / domain_id
        / "raw_replicates"
    )
    metadata_statuses: list[str] = []
    metadata_files: list[str] = []
    if metadata_root.is_dir():
        for path in sorted(metadata_root.glob("*_metadata.json")):
            payload = json.loads(path.read_text())
            metadata_statuses.append(str(payload.get("unit_status", "missing")))
            metadata_files.append(path.relative_to(PROJECT_ROOT).as_posix())
    metadata_summary = ";".join(
        f"{value}:{metadata_statuses.count(value)}"
        for value in sorted(set(metadata_statuses))
    )
    return {
        "case_id": case_id,
        "category": category,
        "n_frames": values.shape[0],
        "n_residues": values.shape[1],
        "coordinate_scale_status": status,
        "rmsd_reporting_unit": (
            "angstrom"
            if status == "angstrom_consistent"
            else "raw_coordinate_units"
        ),
        "physical_scale_factor_to_angstrom": physical_factor,
        "median_frame_neighbor_ca_distance": float(
            np.median(frame_medians)
        ),
        "minimum_frame_neighbor_ca_distance": float(np.min(frame_medians)),
        "p05_frame_neighbor_ca_distance": float(
            np.quantile(frame_medians, 0.05)
        ),
        "p95_frame_neighbor_ca_distance": float(
            np.quantile(frame_medians, 0.95)
        ),
        "maximum_frame_neighbor_ca_distance": float(np.max(frame_medians)),
        "angstrom_like_frame_count": int(np.count_nonzero(angstrom_like)),
        "tenfold_like_frame_count": int(np.count_nonzero(tenfold_like)),
        "ambiguous_scale_frame_count": int(np.count_nonzero(ambiguous)),
        "source_unit_statuses": metadata_summary or "unavailable",
        "source_unit_metadata_files": ";".join(metadata_files) or "unavailable",
        "audit_interpretation": {
            "mixed_scale_contaminated": (
                "contains both approximately 3.8 and 38 raw-unit "
                "chain-neighbor CA distances; no single post-hoc conversion "
                "is valid"
            ),
            "angstrom_consistent": (
                "chain-neighbor CA geometry is consistent with Angstrom input"
            ),
            "uniform_tenfold_scale_suspect": (
                "uniform approximately tenfold scale; reported separately"
            ),
            "synthetic_toy_arbitrary_units": (
                "toy geometry is dimensionless and excluded from physical "
                "Angstrom summaries"
            ),
            "undetermined": "coordinate scale could not be classified",
        }[status],
    }


def independent_evaluation(
    coords: np.ndarray,
    reference: np.ndarray,
    method: MethodTransforms,
    common_core: np.ndarray,
    method_core: np.ndarray,
    explicit_hinge: np.ndarray | None,
) -> IndependentEvaluation:
    """Recompute every reported metric from coordinates and transforms."""

    aligned = apply_transforms_independent(
        coords, method.rotations, method.translations
    )
    return IndependentEvaluation(
        aligned=aligned,
        full_rmsd=rmsd_from_aligned(aligned, reference),
        common_core_rmsd=rmsd_from_aligned(
            aligned, reference, common_core
        ),
        common_mobile_rmsd=rmsd_from_aligned(
            aligned, reference, ~common_core
        ),
        method_core_rmsd=rmsd_from_aligned(
            aligned, reference, method_core
        ),
        method_mobile_rmsd=rmsd_from_aligned(
            aligned, reference, ~method_core
        ),
        hinge_rmsd=(
            None
            if explicit_hinge is None
            else rmsd_from_aligned(aligned, reference, explicit_hinge)
        ),
        weighted_objective=weighted_objective_from_aligned(
            aligned, reference, method.weights
        ),
        rmsf=rmsf_from_aligned(aligned),
    )


def load_sequential(
    case_id: str,
    golden_manifest: dict[str, dict[str, str]],
) -> MethodTransforms:
    """Load frozen sequential weights/transforms without trusting its metrics."""

    path = PROJECT_ROOT / golden_manifest[case_id]["reference_npz"]
    with np.load(path, allow_pickle=False) as payload:
        return MethodTransforms(
            method="sequential_lean_exact",
            weights=np.asarray(
                payload["legacy_long_frozen_weights"], dtype=np.float64
            ),
            rotations=np.asarray(
                payload["legacy_long_rotations"], dtype=np.float64
            ),
            translations=np.asarray(
                payload["legacy_long_translations"], dtype=np.float64
            ),
            source=path.relative_to(PROJECT_ROOT).as_posix(),
        )


def locked_candidates() -> list[Candidate]:
    """Map the three locked Phase 6C-R candidates to audit method names."""

    lock = json.loads((PHASE6_ROOT / "locked_candidates.json").read_text())
    return [
        candidate_from_dict(payload) for payload in lock["candidates"]
    ]


def candidate_method_name(candidate: Candidate) -> str:
    """Return stable manuscript-neutral audit names."""

    mapping = {
        "uniform": "uniform_low_budget",
        "w5_relative_mad_soft": "w5_preconditioned",
        "window_graph_soft": "graph_preconditioned",
    }
    return mapping[candidate.initializer]


def run_locked_candidate(
    coords: np.ndarray,
    chain: np.ndarray | None,
    candidate: Candidate,
    signals: dict[str, tuple[object, float]],
) -> MethodTransforms:
    """Recreate a locked candidate and retain only weights/transforms."""

    initial = make_initial(candidate, coords.shape[1], signals or None)
    result = solve_window_preconditioned_refinement(
        coords,
        initial,
        representative_count=candidate.representative_count,
        update_rounds=candidate.update_rounds,
        chain_ids=chain,
    )
    return MethodTransforms(
        method=candidate_method_name(candidate),
        weights=np.asarray(result.final_weights, dtype=np.float64),
        rotations=np.asarray(result.fit.rotations, dtype=np.float64),
        translations=np.asarray(result.fit.translations, dtype=np.float64),
        source=f"locked:{candidate.candidate_id}",
    )


def run_batched(
    coords: np.ndarray,
    chain: np.ndarray | None,
) -> MethodTransforms:
    """Recreate the Phase 3 independently initialized comparison."""

    selected = temporally_stratified_indices(
        coords.shape[0], min(2000, coords.shape[0])
    )
    estimate = estimate_representative_weights(
        coords[selected],
        coords[0],
        batch_size=min(512, selected.size),
        sequence_chain_ids=chain,
        **ROBUST_KWARGS,
    )
    weights = np.asarray(estimate.mean_weights, dtype=np.float64)
    rotations, translations = fit_transforms(coords, coords[0], weights)
    return MethodTransforms(
        method="batched_independent",
        weights=weights,
        rotations=rotations,
        translations=translations,
        source="phase3_batched_independent_recomputed",
    )


def phase6_row_index() -> dict[tuple[str, str], dict[str, str]]:
    """Index Phase 6 rows for independent-metric parity checks."""

    rows = [
        *read_tsv(PHASE6_ROOT / "development_results.tsv"),
        *read_tsv(PHASE6_ROOT / "locked_validation_results.tsv"),
    ]
    locked = {
        "uniform": "uniform_low_budget",
        "w5_relative_mad_soft": "w5_preconditioned",
        "window_graph_soft": "graph_preconditioned",
    }
    return {
        (row["case_id"], locked[row["initializer"]]): row
        for row in rows
        if row["initializer"] in locked
        and row["representative_count"] == "256"
        and row["update_rounds"] == "4"
    }


def mean_or_nan(values: np.ndarray | None) -> float:
    """Return a scalar mean for optional metrics."""

    return float("nan") if values is None else float(np.mean(values))


def normalized_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize a nonzero weight vector to unit maximum."""

    values = np.asarray(weights, dtype=np.float64)
    return values / np.max(values)


def top_weight_correction(
    coords: np.ndarray,
    reference: np.ndarray,
    candidate: MethodTransforms,
    sequential: MethodTransforms,
    sequential_rmsd: np.ndarray,
    count: int,
) -> tuple[float, float]:
    """Measure a counterfactual correction of top weight disagreements."""

    left = normalized_weights(candidate.weights)
    right = normalized_weights(sequential.weights)
    order = np.argsort(np.abs(left - right))[::-1]
    corrected = left.copy()
    corrected[order[: min(count, order.size)]] = right[
        order[: min(count, order.size)]
    ]
    rotations, translations = fit_transforms(coords, reference, corrected)
    aligned = apply_transforms_independent(coords, rotations, translations)
    differences = np.abs(
        rmsd_from_aligned(aligned, reference) - sequential_rmsd
    )
    return float(np.mean(differences)), float(np.median(differences))


def true_backbone_dihedral_variation(
    case_id: str,
    n_residues: int,
) -> tuple[np.ndarray | None, str]:
    """Attempt true phi/psi circular variation for formal MD inputs."""

    if case_id not in FORMAL_SYSTEMS:
        return None, "unavailable_no_matching_backbone_topology"
    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import calc_dihedrals

        config = FORMAL_SYSTEMS[case_id]
        universe = mda.Universe(
            str(config["topology"]), str(config["trajectory"])
        )
        ca = universe.select_atoms(str(config["selection"]))
        if ca.n_atoms != n_residues:
            return None, "unavailable_ca_residue_mapping_mismatch"
        residue_to_ca = {
            int(atom.resindex): index for index, atom in enumerate(ca)
        }
        definitions: list[tuple[int, np.ndarray]] = []
        for residue in ca.residues:
            for selector in (residue.phi_selection, residue.psi_selection):
                atoms = selector()
                if atoms is not None and len(atoms) == 4:
                    definitions.append(
                        (
                            residue_to_ca[int(residue.resindex)],
                            np.asarray(atoms.indices, dtype=int),
                        )
                    )
        if not definitions:
            return None, "unavailable_no_phi_psi_definitions"
        frame_indices = np.unique(
            np.linspace(
                0,
                len(universe.trajectory) - 1,
                min(100, len(universe.trajectory)),
                dtype=int,
            )
        )
        angles = np.empty(
            (frame_indices.size, len(definitions)), dtype=np.float64
        )
        atom_indices = np.asarray(
            [indices for _, indices in definitions], dtype=int
        )
        for output_index, frame_index in enumerate(frame_indices):
            universe.trajectory[int(frame_index)]
            positions = universe.atoms.positions
            angles[output_index] = calc_dihedrals(
                positions[atom_indices[:, 0]],
                positions[atom_indices[:, 1]],
                positions[atom_indices[:, 2]],
                positions[atom_indices[:, 3]],
            )
        circular_variance = 1.0 - np.abs(
            np.mean(np.exp(1j * angles), axis=0)
        )
        per_residue: list[list[float]] = [[] for _ in range(n_residues)]
        for (residue_index, _), value in zip(
            definitions, circular_variance, strict=True
        ):
            per_residue[residue_index].append(float(value))
        output = np.asarray(
            [
                np.mean(values) if values else np.nan
                for values in per_residue
            ],
            dtype=np.float64,
        )
        return output, "available_true_phi_psi_circular_variance"
    except Exception as error:
        return None, f"unavailable_{type(error).__name__}:{error}"


def cross_replicate_metrics(
    candidates: list[Candidate],
) -> dict[str, tuple[float, float, str]]:
    """Recompute R1/R2 reproducibility where a matching case is available."""

    root = (
        PROJECT_ROOT
        / "results/mdcath_benchmark/replicate_stitching/1vq8L01/replicates"
    )
    if not (root / "R1_ca_coords.npz").is_file():
        return {}
    replicates: list[np.ndarray] = []
    for label in ("R1", "R2"):
        with np.load(root / f"{label}_ca_coords.npz", allow_pickle=False) as data:
            replicates.append(np.asarray(data["coords"], dtype=np.float64))
    weights_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in METHOD_ORDER
    }
    for coords in replicates:
        reference = coords[0]
        selected = temporally_stratified_indices(
            coords.shape[0], min(2000, coords.shape[0])
        )
        sequential = estimate_representative_weights_sequential_lean(
            coords[selected],
            reference,
            **ROBUST_KWARGS,
        )
        weights_by_method["sequential_lean_exact"].append(
            np.asarray(sequential.mean_weights, dtype=np.float64)
        )
        batched = estimate_representative_weights(
            coords[selected],
            reference,
            batch_size=min(512, selected.size),
            **ROBUST_KWARGS,
        )
        weights_by_method["batched_independent"].append(
            np.asarray(batched.mean_weights, dtype=np.float64)
        )
        signals = compute_frozen_window_signals(coords)
        for candidate in candidates:
            initial = make_initial(candidate, coords.shape[1], signals)
            representative_indices = temporally_stratified_indices(
                coords.shape[0], candidate.representative_count
            )
            weights, _, _, _ = bounded_global_weight_refinement(
                coords[representative_indices],
                reference,
                initial.weights,
                update_rounds=candidate.update_rounds,
            )
            weights_by_method[candidate_method_name(candidate)].append(weights)
    return {
        method: (
            safe_spearman(values[0], values[1]),
            jaccard(hard_core(values[0]), hard_core(values[1])),
            "recomputed_1vq8L01_R1_vs_R2_320K",
        )
        for method, values in weights_by_method.items()
        if len(values) == 2
    }


def temporal_metrics_index() -> dict[tuple[str, str], tuple[float, float, str]]:
    """Combine predeclared candidate and exact temporal-block diagnostics."""

    output: dict[tuple[str, str], tuple[float, float, str]] = {}
    for row in read_tsv(PHASE6_ROOT / "risk_rule.tsv"):
        mapping = {
            "uniform": "uniform_low_budget",
            "w5_relative_mad_soft": "w5_preconditioned",
            "window_graph_soft": "graph_preconditioned",
        }
        output[(row["case_id"], mapping[row["initializer"]])] = (
            float(row["temporal_block_weight_spearman"]),
            float(row["temporal_block_core_jaccard"]),
            "phase6c_r_first_half_vs_second_half",
        )
    holdout = read_tsv(SEMANTIC_ROOT / "holdout_metrics.tsv")
    method_mapping = {
        "legacy_object": "sequential_lean_exact",
        "batched_independent": "batched_independent",
    }
    for row in holdout:
        if row["split"] != "first_to_second":
            continue
        if row["method"] not in method_mapping:
            continue
        output[(row["case_id"], method_mapping[row["method"]])] = (
            float(row["swap_half_weight_spearman"]),
            float(row["swap_half_core_jaccard"]),
            "phase5_exact_temporal_holdout",
        )
    return output


def audit_cases() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Run the direct 34-case RMSD/transform/held-out audit."""

    golden_manifest = load_golden_manifest()
    candidates = locked_candidates()
    phase6_rows = phase6_row_index()
    temporal = temporal_metrics_index()
    replicate = cross_replicate_metrics(candidates)
    decomposition_rows: list[dict[str, object]] = []
    transform_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    pooled_abs: dict[str, list[np.ndarray]] = {
        method: [] for method in METHOD_ORDER if method != "sequential_lean_exact"
    }
    clean_pooled_abs: dict[str, list[np.ndarray]] = {
        method: [] for method in METHOD_ORDER if method != "sequential_lean_exact"
    }
    pooled_frames_by_case: dict[str, int] = {}
    independent_phase6_metric_errors: list[float] = []
    independent_golden_metric_errors: list[float] = []
    top_correction_reductions: list[float] = []
    common_mobile_contributions: list[float] = []
    for case in build_cases():
        print(f"[phase7a] {case.case_id}", flush=True)
        coords, chain = case.loader()
        coords = np.asarray(coords, dtype=np.float64)
        chain = np.asarray(chain) if chain is not None else None
        scale = coordinate_scale_audit(
            case.case_id, case.category, coords, chain
        )
        scale_rows.append(scale)
        reference = coords[0]
        sequential = load_sequential(case.case_id, golden_manifest)
        sequential_core = hard_core(sequential.weights)
        explicit_hinge = hinge_mask(case.case_id, coords.shape[1])
        sequential_eval = independent_evaluation(
            coords,
            reference,
            sequential,
            sequential_core,
            sequential_core,
            explicit_hinge,
        )
        golden_path = (
            PROJECT_ROOT / golden_manifest[case.case_id]["reference_npz"]
        )
        with np.load(golden_path, allow_pickle=False) as golden_payload:
            independent_golden_metric_errors.append(
                float(
                    np.max(
                        np.abs(
                            sequential_eval.full_rmsd
                            - np.asarray(
                                golden_payload["legacy_long_rmsd"],
                                dtype=np.float64,
                            )
                        )
                    )
                )
            )
        reference_centered = reference - np.mean(
            reference, axis=0, keepdims=True
        )
        reference_radius_of_gyration = float(
            np.sqrt(np.mean(np.sum(reference_centered**2, axis=1)))
        )
        reference_maximum_radius = float(
            np.max(np.linalg.norm(reference_centered, axis=1))
        )
        reference_bounding_box_diagonal = float(
            np.linalg.norm(
                np.max(reference, axis=0) - np.min(reference, axis=0)
            )
        )
        signals = compute_frozen_window_signals(coords, chain_ids=chain)
        methods: list[MethodTransforms] = [sequential]
        methods.extend(
            run_locked_candidate(coords, chain, candidate, signals)
            for candidate in candidates
        )
        methods.append(run_batched(coords, chain))
        dihedral, dihedral_status = true_backbone_dihedral_variation(
            case.case_id, coords.shape[1]
        )
        pooled_frames_by_case[case.case_id] = coords.shape[0]
        for method in methods:
            method_core = hard_core(method.weights)
            evaluation = (
                sequential_eval
                if method.method == "sequential_lean_exact"
                else independent_evaluation(
                    coords,
                    reference,
                    method,
                    sequential_core,
                    method_core,
                    explicit_hinge,
                )
            )
            hard_rotations, hard_translations = fit_transforms(
                coords, reference, method_core.astype(np.float64)
            )
            hard_aligned = apply_transforms_independent(
                coords, hard_rotations, hard_translations
            )
            hard_core_core = rmsd_from_aligned(
                hard_aligned, reference, method_core
            )
            hard_core_full = rmsd_from_aligned(hard_aligned, reference)
            hard_core_mobile = rmsd_from_aligned(
                hard_aligned, reference, ~method_core
            )
            signed = evaluation.full_rmsd - sequential_eval.full_rmsd
            absolute = np.abs(signed)
            nonzero = sequential_eval.full_rmsd > 1e-10
            relative = np.divide(
                signed,
                sequential_eval.full_rmsd,
                out=np.full_like(signed, np.nan),
                where=nonzero,
            )
            mean_relative = (
                float(np.mean(relative[nonzero]))
                if np.any(nonzero)
                else float("nan")
            )
            median_relative = (
                float(np.median(relative[nonzero]))
                if np.any(nonzero)
                else float("nan")
            )
            rmsd_correlation, rmsd_slope, rmsd_intercept = regression(
                sequential_eval.full_rmsd, evaluation.full_rmsd
            )
            _, rmsf_slope, rmsf_intercept = regression(
                sequential_eval.rmsf, evaluation.rmsf
            )
            normalized_method = normalized_weights(method.weights)
            normalized_sequential = normalized_weights(sequential.weights)
            phase6_error = float("nan")
            top1_mean = float("nan")
            top5_mean = float("nan")
            top1_median = float("nan")
            top5_median = float("nan")
            high_leverage_reduction = float("nan")
            stored = phase6_rows.get((case.case_id, method.method))
            if stored is not None:
                phase6_error = abs(
                    float(stored["rmsd_mae_vs_sequential"])
                    - float(np.mean(absolute))
                )
                independent_phase6_metric_errors.append(phase6_error)
            if method.method not in {
                "sequential_lean_exact",
                "batched_independent",
            }:
                top1_mean, top1_median = top_weight_correction(
                    coords,
                    reference,
                    method,
                    sequential,
                    sequential_eval.full_rmsd,
                    1,
                )
                top5_mean, top5_median = top_weight_correction(
                    coords,
                    reference,
                    method,
                    sequential,
                    sequential_eval.full_rmsd,
                    5,
                )
                denominator = max(float(np.mean(absolute)), 1e-15)
                high_leverage_reduction = (
                    float(np.mean(absolute)) - top5_mean
                ) / denominator
                top_correction_reductions.append(high_leverage_reduction)
            scale_factor = np.sum(sequential.weights) / np.sum(method.weights)
            scaled_rotations, scaled_translations = fit_transforms(
                coords, reference, method.weights * scale_factor
            )
            scaled_aligned = apply_transforms_independent(
                coords, scaled_rotations, scaled_translations
            )
            scale_invariance = float(
                np.max(np.abs(scaled_aligned - evaluation.aligned))
            )
            decomposition_rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "method": method.method,
                    "method_source": method.source,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "coordinate_scale_status": scale[
                        "coordinate_scale_status"
                    ],
                    "rmsd_reporting_unit": scale["rmsd_reporting_unit"],
                    "physical_scale_factor_to_angstrom": scale[
                        "physical_scale_factor_to_angstrom"
                    ],
                    "median_frame_neighbor_ca_distance": scale[
                        "median_frame_neighbor_ca_distance"
                    ],
                    "minimum_frame_neighbor_ca_distance": scale[
                        "minimum_frame_neighbor_ca_distance"
                    ],
                    "maximum_frame_neighbor_ca_distance": scale[
                        "maximum_frame_neighbor_ca_distance"
                    ],
                    "reference_radius_of_gyration": (
                        reference_radius_of_gyration
                    ),
                    "reference_maximum_radius": reference_maximum_radius,
                    "reference_bounding_box_diagonal": (
                        reference_bounding_box_diagonal
                    ),
                    "reference_frame_index": 0,
                    "atom_selection": "same_case_loader_CA_selection",
                    "frame_order": "unchanged",
                    "centering_convention": "weighted_centroids_in_fitted_method",
                    "common_fit_evaluation_selection": (
                        "frozen_sequential_60_percent_core"
                    ),
                    "method_core_selection": "method_specific_60_percent_core",
                    "mean_unified_posthoc_rmsd": float(
                        np.mean(evaluation.full_rmsd)
                    ),
                    "mean_signed_rmsd_difference_vs_sequential": float(
                        np.mean(signed)
                    ),
                    "mean_absolute_rmsd_difference_vs_sequential": float(
                        np.mean(absolute)
                    ),
                    "mean_absolute_rmsd_difference_angstrom_when_verifiable": (
                        float(np.mean(absolute))
                        * float(scale["physical_scale_factor_to_angstrom"])
                        if np.isfinite(
                            float(
                                scale[
                                    "physical_scale_factor_to_angstrom"
                                ]
                            )
                        )
                        else float("nan")
                    ),
                    "median_absolute_rmsd_difference_vs_sequential": float(
                        np.median(absolute)
                    ),
                    "p95_absolute_rmsd_difference_vs_sequential": float(
                        np.quantile(absolute, 0.95)
                    ),
                    "mean_relative_rmsd_difference_vs_sequential": (
                        mean_relative
                    ),
                    "median_relative_rmsd_difference_vs_sequential": (
                        median_relative
                    ),
                    "rmsd_correlation_vs_sequential": rmsd_correlation,
                    "rmsd_regression_slope_vs_sequential": rmsd_slope,
                    "rmsd_regression_intercept_vs_sequential": rmsd_intercept,
                    "rmsf_mae_vs_sequential": float(
                        np.mean(np.abs(evaluation.rmsf - sequential_eval.rmsf))
                    ),
                    "rmsf_calibration_slope_vs_sequential": rmsf_slope,
                    "rmsf_calibration_intercept_vs_sequential": rmsf_intercept,
                    "rmsf_spearman_vs_sequential": safe_spearman(
                        evaluation.rmsf, sequential_eval.rmsf
                    ),
                    "core_jaccard_vs_sequential": jaccard(
                        method_core, sequential_core
                    ),
                    "hard_core_size": int(np.count_nonzero(method_core)),
                    "effective_core_size": float(
                        np.sum(method.weights) / np.max(method.weights)
                    ),
                    "effective_core_size_difference_vs_sequential": float(
                        np.sum(normalized_method)
                        - np.sum(normalized_sequential)
                    ),
                    "mean_weight_difference_vs_sequential": float(
                        np.mean(normalized_method - normalized_sequential)
                    ),
                    "mean_absolute_weight_difference_vs_sequential": float(
                        np.mean(
                            np.abs(
                                normalized_method - normalized_sequential
                            )
                        )
                    ),
                    "weight_spearman_vs_sequential": safe_spearman(
                        normalized_method, normalized_sequential
                    ),
                    "maximum_weight_difference_vs_sequential": float(
                        np.max(
                            np.abs(
                                normalized_method - normalized_sequential
                            )
                        )
                    ),
                    "absolute_weight_sum_ratio_vs_sequential": float(
                        np.sum(method.weights) / np.sum(sequential.weights)
                    ),
                    "weight_scale_invariance_max_coordinate_difference": (
                        scale_invariance
                    ),
                    "mean_weighted_fitting_objective": float(
                        np.mean(evaluation.weighted_objective)
                    ),
                    "mean_weighted_objective_difference_vs_sequential": float(
                        np.mean(evaluation.weighted_objective)
                        - np.mean(sequential_eval.weighted_objective)
                    ),
                    "soft_fit_common_core_evaluation_rmsd": float(
                        np.mean(evaluation.common_core_rmsd)
                    ),
                    "soft_fit_common_mobile_evaluation_rmsd": float(
                        np.mean(evaluation.common_mobile_rmsd)
                    ),
                    "soft_fit_method_core_evaluation_rmsd": float(
                        np.mean(evaluation.method_core_rmsd)
                    ),
                    "soft_fit_method_mobile_evaluation_rmsd": float(
                        np.mean(evaluation.method_mobile_rmsd)
                    ),
                    "hard_core_fit_core_evaluation_rmsd": float(
                        np.mean(hard_core_core)
                    ),
                    "hard_core_fit_full_evaluation_rmsd": float(
                        np.mean(hard_core_full)
                    ),
                    "hard_core_fit_mobile_evaluation_rmsd": float(
                        np.mean(hard_core_mobile)
                    ),
                    "interface_hinge_annotation_status": (
                        "toy_generator_known_hinge_residues_11_to_14"
                        if explicit_hinge is not None
                        else "unavailable_no_explicit_annotation"
                    ),
                    "soft_fit_interface_hinge_rmsd": mean_or_nan(
                        evaluation.hinge_rmsd
                    ),
                    "top1_weight_correction_mean_abs_rmsd_difference": (
                        top1_mean
                    ),
                    "top1_weight_correction_median_abs_rmsd_difference": (
                        top1_median
                    ),
                    "top5_weight_correction_mean_abs_rmsd_difference": (
                        top5_mean
                    ),
                    "top5_weight_correction_median_abs_rmsd_difference": (
                        top5_median
                    ),
                    "top5_correction_relative_reduction": (
                        high_leverage_reduction
                    ),
                    "independent_vs_phase6_rmsd_mae_summary_abs_error": (
                        phase6_error
                    ),
                }
            )
            if method.method != "sequential_lean_exact":
                pooled_abs[method.method].append(absolute)
                if scale["coordinate_scale_status"] == "angstrom_consistent":
                    clean_pooled_abs[method.method].append(absolute)
                common_mobile_contributions.append(
                    float(
                        np.mean(evaluation.common_mobile_rmsd)
                        - np.mean(sequential_eval.common_mobile_rmsd)
                    )
                )
                angle = rotation_angles(
                    method.rotations, sequential.rotations
                )
                translation = np.linalg.norm(
                    method.translations - sequential.translations, axis=1
                )
                coordinate_difference = np.sqrt(
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
                core_coordinate_difference = np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                evaluation.aligned[:, sequential_core]
                                - sequential_eval.aligned[:, sequential_core]
                            )
                            ** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                )
                mobile_coordinate_difference = np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                evaluation.aligned[:, ~sequential_core]
                                - sequential_eval.aligned[:, ~sequential_core]
                            )
                            ** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                )
                transform_rows.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "method": method.method,
                        "n_frames": coords.shape[0],
                        "coordinate_scale_status": scale[
                            "coordinate_scale_status"
                        ],
                        "distance_reporting_unit": scale[
                            "rmsd_reporting_unit"
                        ],
                        "rotation_geodesic_median_degrees": float(
                            np.median(angle)
                        ),
                        "rotation_geodesic_p95_degrees": float(
                            np.quantile(angle, 0.95)
                        ),
                        "rotation_geodesic_max_degrees": float(np.max(angle)),
                        "translation_difference_median": float(
                            np.median(translation)
                        ),
                        "translation_difference_p95": float(
                            np.quantile(translation, 0.95)
                        ),
                        "translation_difference_max": float(
                            np.max(translation)
                        ),
                        "same_frame_coordinate_displacement_median": float(
                            np.median(coordinate_difference)
                        ),
                        "same_frame_coordinate_displacement_p95": float(
                            np.quantile(coordinate_difference, 0.95)
                        ),
                        "common_core_coordinate_displacement_median": float(
                            np.median(core_coordinate_difference)
                        ),
                        "common_mobile_coordinate_displacement_median": float(
                            np.median(mobile_coordinate_difference)
                        ),
                        "coordinate_displacement_mobile_core_ratio": float(
                            np.median(mobile_coordinate_difference)
                            / max(
                                float(np.median(core_coordinate_difference)),
                                1e-15,
                            )
                        ),
                    }
                )
            temporal_value = temporal.get((case.case_id, method.method))
            replicate_value = (
                replicate.get(method.method)
                if case.case_id == "mdcath_1vq8L01"
                else None
            )
            finite_dihedral = (
                np.isfinite(dihedral)
                if dihedral is not None
                else np.zeros(coords.shape[1], dtype=bool)
            )
            heldout_rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "method": method.method,
                    "coordinate_scale_status": scale[
                        "coordinate_scale_status"
                    ],
                    "heldout_frames": (
                        coords.shape[0] - max(1, coords.shape[0] // 2)
                    ),
                    "heldout_core_pairwise_distance_variance": (
                        pair_distance_stability(
                            coords[max(1, coords.shape[0] // 2) :],
                            method_core,
                        )
                    ),
                    "heldout_core_contact_persistence": contact_persistence(
                        coords[max(1, coords.shape[0] // 2) :],
                        method_core,
                        chain,
                    ),
                    "temporal_block_weight_spearman": (
                        temporal_value[0]
                        if temporal_value is not None
                        else float("nan")
                    ),
                    "temporal_block_core_jaccard": (
                        temporal_value[1]
                        if temporal_value is not None
                        else float("nan")
                    ),
                    "temporal_block_source": (
                        temporal_value[2]
                        if temporal_value is not None
                        else "unavailable"
                    ),
                    "cross_replicate_weight_spearman": (
                        replicate_value[0]
                        if replicate_value is not None
                        else float("nan")
                    ),
                    "cross_replicate_core_jaccard": (
                        replicate_value[1]
                        if replicate_value is not None
                        else float("nan")
                    ),
                    "cross_replicate_source": (
                        replicate_value[2]
                        if replicate_value is not None
                        else "unavailable_no_matching_replicates"
                    ),
                    "backbone_dihedral_status": dihedral_status,
                    "mean_core_backbone_dihedral_circular_variance": (
                        float(np.nanmean(dihedral[method_core & finite_dihedral]))
                        if dihedral is not None
                        and np.any(method_core & finite_dihedral)
                        else float("nan")
                    ),
                    "mean_mobile_backbone_dihedral_circular_variance": (
                        float(np.nanmean(dihedral[~method_core & finite_dihedral]))
                        if dihedral is not None
                        and np.any(~method_core & finite_dihedral)
                        else float("nan")
                    ),
                }
            )
    pooled_summary = {
        method: {
            "n_frames": int(sum(array.size for array in arrays)),
            "mean_absolute_rmsd_difference": float(
                np.mean(np.concatenate(arrays))
            ),
            "median_absolute_rmsd_difference": float(
                np.median(np.concatenate(arrays))
            ),
            "p95_absolute_rmsd_difference": float(
                np.quantile(np.concatenate(arrays), 0.95)
            ),
        }
        for method, arrays in pooled_abs.items()
    }
    clean_pooled_summary = {
        method: {
            "n_frames": int(sum(array.size for array in arrays)),
            "mean_absolute_rmsd_difference_angstrom": float(
                np.mean(np.concatenate(arrays))
            ),
            "median_absolute_rmsd_difference_angstrom": float(
                np.median(np.concatenate(arrays))
            ),
            "p95_absolute_rmsd_difference_angstrom": float(
                np.quantile(np.concatenate(arrays), 0.95)
            ),
        }
        for method, arrays in clean_pooled_abs.items()
    }
    total_frames = sum(pooled_frames_by_case.values())
    synthetic_frames = pooled_frames_by_case[
        "synthetic_scaling_4oelA02_10000"
    ]
    diagnostics = {
        "pooled_summary": pooled_summary,
        "pooled_summary_unit": "raw_coordinate_units_mixed_scale_panel",
        "clean_angstrom_pooled_summary": clean_pooled_summary,
        "clean_angstrom_case_count": sum(
            row["coordinate_scale_status"] == "angstrom_consistent"
            for row in scale_rows
        ),
        "mixed_scale_case_ids": [
            row["case_id"]
            for row in scale_rows
            if row["coordinate_scale_status"] == "mixed_scale_contaminated"
        ],
        "toy_arbitrary_unit_case_ids": [
            row["case_id"]
            for row in scale_rows
            if row["coordinate_scale_status"]
            == "synthetic_toy_arbitrary_units"
        ],
        "total_panel_frames": total_frames,
        "synthetic_10k_frame_fraction": synthetic_frames / total_frames,
        "maximum_independent_vs_phase6_summary_error": (
            max(independent_phase6_metric_errors)
            if independent_phase6_metric_errors
            else float("nan")
        ),
        "maximum_independent_vs_golden_rmsd_error": max(
            independent_golden_metric_errors
        ),
        "median_top5_weight_correction_relative_reduction": float(
            np.median(top_correction_reductions)
        ),
        "median_common_mobile_rmsd_signed_contribution": float(
            np.median(common_mobile_contributions)
        ),
    }
    return (
        decomposition_rows,
        transform_rows,
        heldout_rows,
        scale_rows,
        diagnostics,
    )


def natural_variability_rows() -> list[dict[str, object]]:
    """Normalize existing exact-method order, block and reference evidence."""

    output: list[dict[str, object]] = []
    orders = read_tsv(SEMANTIC_ROOT / "order_sensitivity.tsv")
    chronological = {
        row["case_id"]: row
        for row in orders
        if row["method"] == "legacy_object"
        and row["representative_order"] == "chronological"
    }
    for row in orders:
        if row["method"] != "legacy_object":
            continue
        if row["representative_order"] not in {
            "chronological",
            "reverse",
            "interleaved",
        }:
            continue
        baseline = chronological[row["case_id"]]
        output.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "variability_axis": "representative_order",
                "exact_method": "sequential_lean_semantics",
                "comparison": row["representative_order"],
                "baseline": "chronological",
                "mean_unified_rmsd": float(row["mean_full_rmsd"]),
                "signed_mean_rmsd_difference": float(
                    row["mean_full_rmsd"]
                )
                - float(baseline["mean_full_rmsd"]),
                "core_jaccard_vs_baseline": float(
                    row["core_jaccard_vs_legacy_chronological"]
                ),
                "weight_spearman_vs_baseline": float(
                    row["weight_spearman_vs_legacy_chronological"]
                ),
                "source_file": (
                    "results/scaling_optimization_v3/semantic/"
                    "order_sensitivity.tsv"
                ),
                "comparability_note": (
                    "same coordinates/reference/selection; representative "
                    "processing order differs"
                ),
            }
        )
    holdout = read_tsv(SEMANTIC_ROOT / "holdout_metrics.tsv")
    for row in holdout:
        if row["method"] != "legacy_object":
            continue
        output.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "variability_axis": "temporal_block",
                "exact_method": "sequential_lean_semantics",
                "comparison": row["split"],
                "baseline": "opposite_temporal_half",
                "mean_unified_rmsd": float(row["mean_full_rmsd"]),
                "signed_mean_rmsd_difference": float("nan"),
                "core_jaccard_vs_baseline": float(
                    row["swap_half_core_jaccard"]
                ),
                "weight_spearman_vs_baseline": float(
                    row["swap_half_weight_spearman"]
                ),
                "source_file": (
                    "results/scaling_optimization_v3/semantic/"
                    "holdout_metrics.tsv"
                ),
                "comparability_note": (
                    "different temporal training/evaluation halves; natural "
                    "stability evidence, not transform-parity evidence"
                ),
            }
        )
    system_map = {
        "I-FABP": "formal_ifabp",
        "ADK": "formal_adk",
        "NhaA": "formal_nhaa",
        "YiiP short": "formal_yiip_short",
    }
    references = read_tsv(REFERENCE_SENSITIVITY)
    first = {
        row["system"]: row
        for row in references
        if row["reference_strategy"] == "first_frame"
    }
    for row in references:
        baseline = first[row["system"]]
        output.append(
            {
                "case_id": system_map[row["system"]],
                "category": "formal_md",
                "variability_axis": "reference_choice",
                "exact_method": "formal_default_soft_pipeline",
                "comparison": row["reference_strategy"],
                "baseline": "first_frame",
                "mean_unified_rmsd": float(row["unified_posthoc_RMSD"]),
                "signed_mean_rmsd_difference": float(
                    row["unified_posthoc_RMSD"]
                )
                - float(baseline["unified_posthoc_RMSD"]),
                "core_jaccard_vs_baseline": float(
                    row["jaccard_core_vs_default_reference"]
                ),
                "weight_spearman_vs_baseline": float(
                    row["spearman_mobility_vs_default_reference"]
                ),
                "source_file": (
                    "results/md_trajectory_focus/reference_sensitivity/"
                    "reference_sensitivity.tsv"
                ),
                "comparability_note": (
                    "reference differs by design; used only to contextualize "
                    "natural reference sensitivity"
                ),
            }
        )
    return output


def summarize_causes(
    decomposition: list[dict[str, object]],
    transforms: list[dict[str, object]],
    heldout: list[dict[str, object]],
    natural: list[dict[str, object]],
    diagnostics: dict[str, object],
) -> dict[str, object]:
    """Classify the measured causes of the pooled discrepancy."""

    approximate = [
        row
        for row in decomposition
        if row["method"]
        in {
            "uniform_low_budget",
            "w5_preconditioned",
            "graph_preconditioned",
        }
    ]
    synthetic = [
        row
        for row in approximate
        if row["case_id"] == "synthetic_scaling_4oelA02_10000"
    ]
    other = [
        row
        for row in approximate
        if row["case_id"] != "synthetic_scaling_4oelA02_10000"
    ]
    clean_angstrom = [
        row
        for row in approximate
        if row["coordinate_scale_status"] == "angstrom_consistent"
    ]
    mixed_scale = [
        row
        for row in approximate
        if row["coordinate_scale_status"] == "mixed_scale_contaminated"
    ]
    transform_approximate = [
        row
        for row in transforms
        if row["method"]
        in {
            "uniform_low_budget",
            "w5_preconditioned",
            "graph_preconditioned",
        }
    ]
    heldout_approximate = [
        row
        for row in heldout
        if row["method"]
        in {
            "uniform_low_budget",
            "w5_preconditioned",
            "graph_preconditioned",
        }
        and row["coordinate_scale_status"] == "angstrom_consistent"
    ]
    heldout_sequential = {
        row["case_id"]: row
        for row in heldout
        if row["method"] == "sequential_lean_exact"
        and row["coordinate_scale_status"] == "angstrom_consistent"
    }
    pair_delta = [
        float(row["heldout_core_pairwise_distance_variance"])
        - float(
            heldout_sequential[row["case_id"]][
                "heldout_core_pairwise_distance_variance"
            ]
        )
        for row in heldout_approximate
        if np.isfinite(float(row["heldout_core_pairwise_distance_variance"]))
        and np.isfinite(
            float(
                heldout_sequential[row["case_id"]][
                    "heldout_core_pairwise_distance_variance"
                ]
            )
        )
    ]
    contact_delta = [
        float(row["heldout_core_contact_persistence"])
        - float(
            heldout_sequential[row["case_id"]][
                "heldout_core_contact_persistence"
            ]
        )
        for row in heldout_approximate
        if np.isfinite(float(row["heldout_core_contact_persistence"]))
        and np.isfinite(
            float(
                heldout_sequential[row["case_id"]][
                    "heldout_core_contact_persistence"
                ]
            )
        )
    ]
    natural_summary: dict[str, dict[str, float | int]] = {}
    for axis in sorted({row["variability_axis"] for row in natural}):
        selected = [
            row for row in natural if row["variability_axis"] == axis
        ]
        rmsd = np.asarray(
            [
                abs(float(row["signed_mean_rmsd_difference"]))
                for row in selected
                if np.isfinite(float(row["signed_mean_rmsd_difference"]))
            ],
            dtype=np.float64,
        )
        core = np.asarray(
            [
                float(row["core_jaccard_vs_baseline"])
                for row in selected
                if np.isfinite(float(row["core_jaccard_vs_baseline"]))
            ],
            dtype=np.float64,
        )
        natural_summary[axis] = {
            "n_rows": len(selected),
            "median_absolute_mean_rmsd_difference": (
                float(np.median(rmsd)) if rmsd.size else float("nan")
            ),
            "maximum_absolute_mean_rmsd_difference": (
                float(np.max(rmsd)) if rmsd.size else float("nan")
            ),
            "median_core_jaccard": float(np.median(core)),
            "minimum_core_jaccard": float(np.min(core)),
        }
    large_cases = {
        row["case_id"]: row
        for row in approximate
        if row["method"] == "w5_preconditioned"
    }
    large_transforms = {
        row["case_id"]: row
        for row in transform_approximate
        if row["method"] == "w5_preconditioned"
    }
    synthetic_case = large_cases["synthetic_scaling_4oelA02_10000"]
    synthetic_transform = large_transforms[
        "synthetic_scaling_4oelA02_10000"
    ]
    full_rmsd_improvement_count = sum(
        float(row["mean_signed_rmsd_difference_vs_sequential"]) < 0.0
        for row in approximate
    )
    weighted_objective_worsening_count = sum(
        float(row["mean_weighted_objective_difference_vs_sequential"]) > 0.0
        for row in approximate
    )
    agreement_summary = {
        "all_approximate_rows": {
            "median_core_jaccard": float(
                np.median(
                    [
                        float(row["core_jaccard_vs_sequential"])
                        for row in approximate
                    ]
                )
            ),
            "median_weight_spearman": float(
                np.median(
                    [
                        float(row["weight_spearman_vs_sequential"])
                        for row in approximate
                    ]
                )
            ),
            "median_rmsf_spearman": float(
                np.median(
                    [
                        float(row["rmsf_spearman_vs_sequential"])
                        for row in approximate
                    ]
                )
            ),
        },
        "clean_angstrom_rows": {
            "median_core_jaccard": float(
                np.median(
                    [
                        float(row["core_jaccard_vs_sequential"])
                        for row in clean_angstrom
                    ]
                )
            ),
            "median_weight_spearman": float(
                np.median(
                    [
                        float(row["weight_spearman_vs_sequential"])
                        for row in clean_angstrom
                    ]
                )
            ),
            "median_rmsf_spearman": float(
                np.median(
                    [
                        float(row["rmsf_spearman_vs_sequential"])
                        for row in clean_angstrom
                    ]
                )
            ),
        },
    }
    return {
        **diagnostics,
        "agreement_summary": agreement_summary,
        "median_clean_angstrom_per_system_mean_abs_rmsd_difference": float(
            np.median(
                [
                    float(
                        row[
                            "mean_absolute_rmsd_difference_"
                            "angstrom_when_verifiable"
                        ]
                    )
                    for row in clean_angstrom
                ]
            )
        ),
        "median_per_system_abs_rmsd_difference_without_synthetic_10k": float(
            np.median(
                [
                    float(row["mean_absolute_rmsd_difference_vs_sequential"])
                    for row in other
                ]
            )
        ),
        "synthetic_10k_mean_abs_rmsd_difference_range": [
            min(
                float(row["mean_absolute_rmsd_difference_vs_sequential"])
                for row in synthetic
            ),
            max(
                float(row["mean_absolute_rmsd_difference_vs_sequential"])
                for row in synthetic
            ),
        ],
        "mixed_scale_mean_abs_rmsd_difference_raw_range": [
            min(
                float(row["mean_absolute_rmsd_difference_vs_sequential"])
                for row in mixed_scale
            ),
            max(
                float(row["mean_absolute_rmsd_difference_vs_sequential"])
                for row in mixed_scale
            ),
        ],
        "median_rotation_disagreement_degrees": float(
            np.median(
                [
                    float(row["rotation_geodesic_median_degrees"])
                    for row in transform_approximate
                ]
            )
        ),
        "median_mobile_core_coordinate_displacement_ratio": float(
            np.median(
                [
                    float(row["coordinate_displacement_mobile_core_ratio"])
                    for row in transform_approximate
                ]
            )
        ),
        "median_heldout_pair_variance_delta_vs_sequential": float(
            np.median(pair_delta)
        ),
        "median_heldout_contact_persistence_delta_vs_sequential": float(
            np.median(contact_delta)
        ),
        "approximate_rows_with_lower_full_set_rmsd": (
            full_rmsd_improvement_count
        ),
        "approximate_rows_total": len(approximate),
        "approximate_rows_with_higher_weighted_objective": (
            weighted_objective_worsening_count
        ),
        "synthetic_10k_w5": {
            "mean_sequential_rmsd": float(
                synthetic_case["mean_unified_posthoc_rmsd"]
                - synthetic_case[
                    "mean_signed_rmsd_difference_vs_sequential"
                ]
            ),
            "mean_w5_rmsd": float(
                synthetic_case["mean_unified_posthoc_rmsd"]
            ),
            "signed_rmsd_difference": float(
                synthetic_case[
                    "mean_signed_rmsd_difference_vs_sequential"
                ]
            ),
            "relative_rmsd_difference": float(
                synthetic_case[
                    "mean_relative_rmsd_difference_vs_sequential"
                ]
            ),
            "weighted_objective_difference": float(
                synthetic_case[
                    "mean_weighted_objective_difference_vs_sequential"
                ]
            ),
            "reference_radius_of_gyration": float(
                synthetic_case["reference_radius_of_gyration"]
            ),
            "reference_bounding_box_diagonal": float(
                synthetic_case["reference_bounding_box_diagonal"]
            ),
            "median_rotation_disagreement_degrees": float(
                synthetic_transform["rotation_geodesic_median_degrees"]
            ),
            "median_coordinate_displacement": float(
                synthetic_transform[
                    "same_frame_coordinate_displacement_median"
                ]
            ),
            "mobile_core_displacement_ratio": float(
                synthetic_transform[
                    "coordinate_displacement_mobile_core_ratio"
                ]
            ),
        },
        "exact_method_natural_variability_summary": natural_summary,
        "causal_assessment": {
            "reference_convention": (
                "not the direct-comparison cause; all audited methods used "
                "frame 0 and identical frame/atom order"
            ),
            "metric_implementation": (
                "excluded by independent application of frozen transforms and "
                "numerical agreement with frozen golden RMSD"
            ),
            "pooling": (
                "major driver because the 10k synthetic case contributes the "
                "majority of pooled frames"
            ),
            "coordinate_scale": (
                "primary reporting confound: four frozen cases contain both "
                "approximately 3.8 and 38 raw-unit neighboring CA distances; "
                "the raw pooled 1.02 value cannot be labelled Angstrom"
            ),
            "absolute_weight_scale": (
                "not a cause; weighted Kabsch is invariant to a global "
                "positive weight rescaling"
            ),
            "high_leverage_residues": (
                "quantified by top-1/top-5 counterfactual correction; not "
                "assumed to be the sole cause"
            ),
            "mobile_domain_orientation": (
                "assessed by common mobile/core coordinate displacement ratio"
            ),
            "global_fit_quality": (
                "must be interpreted with held-out pair variance, contacts, "
                "temporal blocks and cross-replicates, not sequential parity "
                "alone"
            ),
        },
    }


def write_documents(
    causes: dict[str, object],
    decomposition: list[dict[str, object]],
    transforms: list[dict[str, object]],
    heldout: list[dict[str, object]],
) -> None:
    """Write the required audit and revised-interpretation documents."""

    pooled = causes["pooled_summary"]
    clean_pooled = causes["clean_angstrom_pooled_summary"]
    w5 = pooled["w5_preconditioned"]
    uniform = pooled["uniform_low_budget"]
    graph = pooled["graph_preconditioned"]
    clean_w5 = clean_pooled["w5_preconditioned"]
    clean_uniform = clean_pooled["uniform_low_budget"]
    clean_graph = clean_pooled["graph_preconditioned"]
    synthetic = causes["synthetic_10k_w5"]
    natural = causes["exact_method_natural_variability_summary"]
    agreement = causes["agreement_summary"]
    exact_phrase = (
        "W5 was not selected as a stable-core acceleration preconditioner in "
        "its current implementation, but is retained as an experimental "
        "alignment-free local-deformation diagnostic."
    )
    rmsd_audit = f"""# Phase 7A RMSD definition audit

## Scope

This is a metric and interpretation audit only. No algorithm, public default,
golden reference, Git metadata, or `project_completion` output was changed.
All direct comparisons used frame 0 as the reference, the existing CA case
loader, unchanged frame order, and exactly the same input coordinate arrays for
all methods within each case.

## Independent definitions

- **Unified post-hoc RMSD:** ordinary RMSD over all selected CA atoms after
  applying a method's supplied weighted-fit transform.
- **Weighted fitting objective:** the weight-normalized RMSD evaluated with the
  exact weight vector that generated the transform.
- **Common-core/mobile RMSD:** RMSD on the frozen sequential 60% core or its
  complement, giving every method the same evaluation selection.
- **Method-core hard refit:** an additional exact Kabsch fit using the method's
  own 60% hard core, evaluated separately on its core, mobile complement and
  full CA set.
- **RMSF:** recomputed from independently transformed coordinates; it remains
  alignment-frame dependent.

The independent evaluator applies transforms and computes residuals directly
from NumPy coordinate arrays. It does not use production RMSD/RMSF fields.

## Coordinate-scale audit

The Phase 7A consistency check found that four frozen cases contain a mixture
of approximately 3.8 and 38 raw-unit chain-neighbor CA distances:
`{causes['mixed_scale_case_ids'][0]}`,
`{causes['mixed_scale_case_ids'][1]}`,
`{causes['mixed_scale_case_ids'][2]}`, and
`{causes['mixed_scale_case_ids'][3]}`. The cached replicate metadata for the
underlying domains also mixes `assumed_angstrom` and
`guessed_nm_to_angstrom`. These are mixed within trajectories, so no single
post-hoc scale factor can repair them. The three toy systems use arbitrary
synthetic units.

Consequently, the historical pooled value near 1.02 is reproducible in **raw
coordinate units**, but it must not be labelled Angstrom or used as a physical
RMSD-validity threshold. The frozen coordinates and legacy outputs were not
changed. `coordinate_scale_audit.tsv` records the per-frame evidence, and all
affected rows remain in the decomposition for provenance.

## Raw pooled discrepancy

Across the full mixed-scale panel, the independently reproduced raw-coordinate
absolute RMSD differences were:

- uniform: median {uniform['median_absolute_rmsd_difference']:.6f}, P95
  {uniform['p95_absolute_rmsd_difference']:.6f} raw coordinate units;
- W5: median {w5['median_absolute_rmsd_difference']:.6f}, P95
  {w5['p95_absolute_rmsd_difference']:.6f} raw coordinate units;
- graph: median {graph['median_absolute_rmsd_difference']:.6f}, P95
  {graph['p95_absolute_rmsd_difference']:.6f} raw coordinate units.

Across the same per-system approximate-method rows, median core Jaccard, weight
Spearman and RMSF Spearman were
{agreement['all_approximate_rows']['median_core_jaccard']:.6f},
{agreement['all_approximate_rows']['median_weight_spearman']:.6f}, and
{agreement['all_approximate_rows']['median_rmsf_spearman']:.6f}. These
dimensionless selection/ranking agreements remain valid under uniform
coordinate scaling, but do not imply transform identity.

The 10k synthetic case contributed
{100 * causes['synthetic_10k_frame_fraction']:.2f}% of all pooled frames. Its
per-method mean absolute discrepancy ranged from
{causes['synthetic_10k_mean_abs_rmsd_difference_range'][0]:.6f} to
{causes['synthetic_10k_mean_abs_rmsd_difference_range'][1]:.6f} raw units.
It is both frame-dominant and mixed-scale contaminated.

For the {causes['clean_angstrom_case_count']} non-toy cases whose chain-neighbor
geometry is Angstrom-consistent, pooled medians/P95 values were:

- uniform: {clean_uniform['median_absolute_rmsd_difference_angstrom']:.6f} /
  {clean_uniform['p95_absolute_rmsd_difference_angstrom']:.6f} A;
- W5: {clean_w5['median_absolute_rmsd_difference_angstrom']:.6f} /
  {clean_w5['p95_absolute_rmsd_difference_angstrom']:.6f} A;
- graph: {clean_graph['median_absolute_rmsd_difference_angstrom']:.6f} /
  {clean_graph['p95_absolute_rmsd_difference_angstrom']:.6f} A.

The clean-panel median of per-system mean absolute differences was
{causes['median_clean_angstrom_per_system_mean_abs_rmsd_difference']:.6f} A.
This is the defensible physical-scale summary; it does not retroactively repair
or validate the four contaminated cases.

## Determination

The apparent contradiction is primarily caused by pooling a frame-dominant,
mixed-scale 10k trajectory and labelling the resulting raw-coordinate
difference as Angstrom. It is not caused by the independent RMSD implementation,
reference mismatch in the direct comparison, or absolute weight normalization.
The approximate estimators do select nearly the same residues and preserve
RMSF rank while allowing small changes in continuous weights; these changes
produce measurably different rotations/translations without requiring a change
in hard core labels. On the scale-consistent panel the resulting physical RMSD
differences are much smaller. The contaminated cases cannot determine whether
their raw global fit is scientifically better or worse and must be regenerated
with unit-consistent replicates before such a claim.

## Cause assessment

1. **Reference and atom conventions:** excluded as a direct cause because all
   direct methods used identical coordinates, frame 0, atom order and frame
   order. Existing reference-choice experiments remain important natural
   variability context, not an implementation mismatch.
2. **Metric implementation:** independently applying frozen sequential
   transforms reproduces frozen golden RMSD to a maximum error of
   {causes['maximum_independent_vs_golden_rmsd_error']:.3e} raw coordinate
   units, excluding the RMSD implementation as the cause. Recreating the locked low-budget
   candidates differs from the historical Phase 6 per-system summary by at
   most {causes['maximum_independent_vs_phase6_summary_error']:.3e} raw units;
   this is recorded as candidate-regeneration variation, not hidden.
3. **Absolute weight scale:** excluded. Rescaling every weight by a common
   positive factor changed independently transformed coordinates only at
   numerical precision.
4. **High-leverage residues:** top-1/top-5 counterfactual corrections are
   reported per system. The median relative reduction after correcting the five
   largest weight disagreements was
   {causes['median_top5_weight_correction_relative_reduction']:.4f}; this tests
   leverage but does not redefine either method.
5. **Mobile orientation:** the median mobile/core aligned-coordinate
   displacement ratio was
   {causes['median_mobile_core_coordinate_displacement_ratio']:.4f}. The
   transform table should be used to distinguish broad rigid-domain orientation
   changes from core-local disagreement.
   For the 10k synthetic case the ratio was
   {synthetic['mobile_core_displacement_ratio']:.4f}, so its discrepancy is a
   broad global-frame rotation rather than a mobile-only displacement.
6. **Coordinate scale:** this is the primary reporting confound. The affected
   mdCATH trajectories mix incompatible scales within one case. Their absolute
   RMSD, translation and coordinate-displacement values are retained only in
   raw units and excluded from the clean Angstrom summary.
7. **Global fitting within the frozen synthetic input:** the synthetic 10k
   reference has raw-unit radius of gyration
   {synthetic['reference_radius_of_gyration']:.3f} and bounding-box diagonal
   {synthetic['reference_bounding_box_diagonal']:.3f}.
   A median rotation disagreement of
   {synthetic['median_rotation_disagreement_degrees']:.3f} degrees therefore
   produces {synthetic['median_coordinate_displacement']:.3f} raw-unit median
   coordinate displacement. W5 lowers full-set RMSD by
   {abs(synthetic['signed_rmsd_difference']):.3f} raw units
   ({100 * abs(synthetic['relative_rmsd_difference']):.3f}%) but raises its own
   weighted objective relative to sequential by
   {synthetic['weighted_objective_difference']:.3f} raw units. Within this
   frozen input the discrepancy reflects a different stable-core/full-set
   tradeoff, but the contaminated trajectory cannot support a physical
   Angstrom-level scientific conclusion.

## Exact-method natural variability

- Changing chronological sequential processing to reverse/interleaved order
  produced a median absolute mean-RMSD change of
  {natural['representative_order']['median_absolute_mean_rmsd_difference']:.3e}
  source-coordinate units and maximum
  {natural['representative_order']['maximum_absolute_mean_rmsd_difference']:.6f}
  source-coordinate units; minimum core Jaccard was
  {natural['representative_order']['minimum_core_jaccard']:.4f}.
- Formal reference choice produced a median absolute mean-RMSD change of
  {natural['reference_choice']['median_absolute_mean_rmsd_difference']:.4f} A,
  maximum
  {natural['reference_choice']['maximum_absolute_mean_rmsd_difference']:.4f} A,
  and minimum core Jaccard
  {natural['reference_choice']['minimum_core_jaccard']:.4f}.
- Exact temporal halves had median core Jaccard
  {natural['temporal_block']['median_core_jaccard']:.4f} and minimum
  {natural['temporal_block']['minimum_core_jaccard']:.4f}. Their RMSDs are not
  directly subtracted because training/evaluation frames differ.

Held-out pair-distance variance and contact persistence differences versus
sequential on the scale-consistent panel had medians of
{causes['median_heldout_pair_variance_delta_vs_sequential']:.3e} and
{causes['median_heldout_contact_persistence_delta_vs_sequential']:.3e},
respectively. These medians do not prove equivalence, but they provide no
evidence of a panel-wide independent stability collapse.

## Output-specific contracts

- **A, drop-in exact-alignment replacement:** not established because
  transform/RMSD parity is required.
- **B, alternative fast stable-core method:** remains a scientific question;
  independent held-out stability and reproducibility are primary, and
  disagreement with chronological sequential output alone is not failure.
- **C, local-deformation diagnostic:** RMSD parity is not a gate.

{exact_phrase}
"""
    (RESULT_ROOT / "rmsd_definition_audit.md").write_text(rmsd_audit)
    revised = f"""# Revised Phase 6 interpretation

Phase 6C-R used deliberately strict implementation-parity thresholds of
0.05 A median and 0.20 A P95 absolute RMSD difference. These were never general
scientific-validity thresholds. Phase 7A additionally found that four frozen
mdCATH-derived cases mix approximately 3.8 and 38 raw-unit chain-neighbor CA
distances within trajectories. Therefore the historical full-panel pooled
approximately 1.02 value is a reproducible raw-coordinate statistic, not a
valid Angstrom summary.

The low-budget candidates preserve core identity, weight ranking and RMSF
ranking closely, but they are not transform-identical drop-in replacements for
`sequential_lean`. On the {causes['clean_angstrom_case_count']}-case
scale-consistent non-toy panel, W5's pooled median/P95 absolute discrepancy was
{clean_w5['median_absolute_rmsd_difference_angstrom']:.6f}/
{clean_w5['p95_absolute_rmsd_difference_angstrom']:.6f} A. Per-system,
relative, transform and selection-specific decompositions remain required.
The four contaminated cases are preserved for provenance but are not used for
physical-scale judgment.

Three contracts must remain separate:

1. A drop-in exact-alignment replacement requires transform and RMSD parity.
2. An alternative stable-core method should be judged primarily by independent
   held-out structural stability and reproducibility.
3. An alignment-free local-deformation diagnostic does not require RMSD parity
   because it does not claim to emit exact global transforms.

{exact_phrase}

The sliding-window branch is therefore retained. No further window
hyperparameter tuning was performed in Phase 7A.
"""
    (RESULT_ROOT / "revised_phase6_interpretation.md").write_text(revised)


def update_current_phase6_wording() -> None:
    """Narrow live Phase 6 wording while preserving its archived handoff."""

    path = PHASE6_ROOT / "decision.json"
    payload = json.loads(path.read_text())
    payload.pop("w5_abandoned_as_stable_core_acceleration", None)
    payload["w5_stable_core_preconditioner_status"] = (
        "not_selected_in_current_equal_budget_implementation"
    )
    payload["w5_alignment_free_diagnostic_status"] = "retained_experimental"
    payload["phase7a_interpretation"] = (
        "W5 was not selected as a stable-core acceleration preconditioner in "
        "its current implementation, but is retained as an experimental "
        "alignment-free local-deformation diagnostic."
    )
    payload["phase6_handoff_status"] = (
        "historical_archive_preserved; Phase 7A wording supersedes the broad "
        "abandonment interpretation"
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_tests() -> dict[str, object]:
    """Run the full suite in the canonical environment."""

    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output = completed.stdout + completed.stderr
    (RESULT_ROOT / "full_test_results.txt").write_text(output)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": output.strip().splitlines()[-1] if output.strip() else "",
    }


def make_handoff(files: list[Path]) -> tuple[Path, str]:
    """Create a compact archive without coordinates or caches."""

    date = time.strftime("%Y%m%d")
    staging = RESULT_ROOT / "handoff_phase7a"
    staging.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_tsv(staging / "sha256_manifest.tsv", rows)
    (staging / "README.md").write_text(
        "# Phase 7A compact handoff\n\n"
        "This bundle contains only RMSD-definition audit tables, interpretation "
        "documents, the audit script/tests, integrity and test status. It "
        "contains no trajectories, caches, secrets or generated environments. "
        "The archived Phase 6C-R handoff remains unchanged.\n"
    )
    archive = (
        PROJECT_ROOT
        / f"results/scaling_optimization_v4/handoff_phase7a_{date}.tar.gz"
    )
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(staging, arcname="handoff_phase7a")
    digest = sha256(archive)
    archive.with_suffix("").with_suffix(".sha256").write_text(
        f"{digest}  {archive.name}\n"
    )
    return archive, digest


def main() -> int:
    """Execute Phase 7A once and stop."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before_integrity = verify_hashes()
    if before_integrity["status"] != "pass":
        raise RuntimeError("frozen integrity failed before Phase 7A")
    (
        decomposition,
        transforms,
        heldout,
        scale_rows,
        diagnostics,
    ) = audit_cases()
    natural = natural_variability_rows()
    causes = summarize_causes(
        decomposition, transforms, heldout, natural, diagnostics
    )
    write_tsv(
        RESULT_ROOT / "per_system_rmsd_decomposition.tsv", decomposition
    )
    write_tsv(RESULT_ROOT / "transform_disagreement.tsv", transforms)
    write_tsv(
        RESULT_ROOT / "exact_method_natural_variability.tsv", natural
    )
    write_tsv(
        RESULT_ROOT / "heldout_independent_quality.tsv", heldout
    )
    write_tsv(RESULT_ROOT / "coordinate_scale_audit.tsv", scale_rows)
    (RESULT_ROOT / "cause_diagnostics.json").write_text(
        json.dumps(causes, indent=2) + "\n"
    )
    write_documents(causes, decomposition, transforms, heldout)
    update_current_phase6_wording()
    after_integrity = verify_hashes()
    tests = run_tests()
    status = {
        "phase": "7A",
        "scope": "RMSD_definition_and_interpretation_audit_only",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "integrity_before": before_integrity,
        "integrity_after": after_integrity,
        "tests": tests,
        "algorithm_changed": False,
        "public_default_changed": False,
        "git_metadata_changed": False,
        "project_completion_changed": False,
        "golden_references_changed": False,
        "phase6_archived_handoff_changed": False,
        "coordinate_scale_audit": {
            "mixed_scale_case_count": len(
                diagnostics["mixed_scale_case_ids"]
            ),
            "mixed_scale_case_ids": diagnostics["mixed_scale_case_ids"],
            "policy": (
                "retain frozen rows in raw coordinate units; exclude mixed "
                "scale and toy cases from physical Angstrom pooled summaries"
            ),
        },
    }
    (RESULT_ROOT / "phase7a_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    include = [
        RESULT_ROOT / "rmsd_definition_audit.md",
        RESULT_ROOT / "per_system_rmsd_decomposition.tsv",
        RESULT_ROOT / "transform_disagreement.tsv",
        RESULT_ROOT / "exact_method_natural_variability.tsv",
        RESULT_ROOT / "heldout_independent_quality.tsv",
        RESULT_ROOT / "coordinate_scale_audit.tsv",
        RESULT_ROOT / "revised_phase6_interpretation.md",
        RESULT_ROOT / "cause_diagnostics.json",
        RESULT_ROOT / "phase7a_status.json",
        RESULT_ROOT / "full_test_results.txt",
        Path(__file__).resolve(),
        PROJECT_ROOT
        / "tests/scaling_optimization_v4/test_phase7a_rmsd_audit.py",
    ]
    archive, digest = make_handoff(include)
    status["handoff_archive"] = archive.relative_to(PROJECT_ROOT).as_posix()
    status["handoff_sha256"] = digest
    (RESULT_ROOT / "phase7a_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
