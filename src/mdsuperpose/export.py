"""Minimal visualization/export helpers for soft robust and MDLovoFit outputs."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
import time

import numpy as np

from .core_selection import aggregate_weights_by_group, select_core_by_target_fraction
from .mdlovofit import read_multimodel_pdb_with_occupancy
from .trajectory import Trajectory
from .trajectory_mode import solve_trajectory_robust_alignment
from .utils import as_float_array


@dataclass(slots=True)
class VisualizationExportArtifacts:
    """Filesystem outputs and timing information for one export run."""

    method_name: str
    occupancy_pdb_path: Path | None
    bfactor_pdb_path: Path | None
    csv_path: Path | None
    summary_json_path: Path | None
    algorithm_runtime_seconds: float | None
    export_runtime_seconds: float
    render_runtime_seconds: float | None


def _validate_vector(values: np.ndarray, n_atoms: int, *, name: str) -> np.ndarray:
    """Validate a one-dimensional per-atom numeric vector."""

    values = as_float_array(values)
    if values.ndim != 1 or values.shape[0] != n_atoms:
        raise ValueError(f"{name} must have shape ({n_atoms},)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _default_residue_labels(n_atoms: int) -> tuple[str, ...]:
    """Create order-based residue labels when no labels are provided."""

    return tuple(f"RES{index + 1}" for index in range(n_atoms))


def _default_atom_names(n_atoms: int) -> tuple[str, ...]:
    """Create default atom names when no names are provided."""

    return tuple("CA" for _ in range(n_atoms))


def _normalize_optional_labels(
    labels: list[str] | tuple[str, ...] | None,
    *,
    n_atoms: int,
    default_factory,
) -> tuple[str, ...]:
    """Normalize one optional label sequence to the expected atom length."""

    if labels is None:
        return default_factory(n_atoms)
    labels = tuple(str(label) for label in labels)
    if len(labels) != n_atoms:
        raise ValueError(f"labels must have length {n_atoms}")
    return labels


def _parse_residue_label(label: str, fallback_index: int) -> tuple[str, str, int]:
    """Parse a lightweight residue label into PDB-friendly fields."""

    chain = "A"
    residue = label
    if ":" in label:
        chain_part, residue = label.split(":", 1)
        chain = (chain_part or "A")[:1]

    match = re.match(r"([A-Za-z]{1,4})(-?\d+)", residue)
    if match is None:
        return chain, "ALA", fallback_index

    resname = match.group(1).upper()[:3].rjust(3, "A")
    resid = int(match.group(2))
    return chain, resname, resid


def write_single_model_pdb(
    coords: np.ndarray,
    path: str | Path,
    *,
    occupancy: np.ndarray,
    bfactor: np.ndarray,
    residue_labels: list[str] | tuple[str, ...] | None = None,
    atom_names: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Write one visualization-ready single-model PDB."""

    coords = as_float_array(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    n_atoms = coords.shape[0]
    occupancy = _validate_vector(occupancy, n_atoms, name="occupancy")
    bfactor = _validate_vector(bfactor, n_atoms, name="bfactor")
    residue_labels = _normalize_optional_labels(
        residue_labels,
        n_atoms=n_atoms,
        default_factory=_default_residue_labels,
    )
    atom_names = _normalize_optional_labels(
        atom_names,
        n_atoms=n_atoms,
        default_factory=_default_atom_names,
    )

    path = Path(path)
    with path.open("w") as handle:
        for atom_index, (xyz, occ, beta, residue_label, atom_name) in enumerate(
            zip(coords, occupancy, bfactor, residue_labels, atom_names, strict=True),
            start=1,
        ):
            chain, resname, resid = _parse_residue_label(residue_label, atom_index)
            handle.write(
                f"ATOM  {atom_index:5d} {atom_name[:4]:>4} {resname:>3} {chain:1s}{resid:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{beta:6.2f}           C\n"
            )
        handle.write("END\n")


def _group_score_lookup(
    weights: np.ndarray,
    group_ids: np.ndarray | None,
) -> np.ndarray | None:
    """Expand grouped mean weights back to atom resolution when requested."""

    if group_ids is None:
        return None
    group_ids = np.asarray(group_ids)
    if group_ids.shape != weights.shape:
        raise ValueError(f"group_ids must have shape {weights.shape}")
    aggregated = aggregate_weights_by_group(weights, group_ids, reduction="mean")
    lookup = {group_id: group_weight for group_id, group_weight in zip(aggregated.group_ids, aggregated.group_weights, strict=True)}
    return np.asarray([lookup[group_id] for group_id in group_ids], dtype=float)


def _soft_visualization_rows(
    *,
    weights: np.ndarray,
    residuals: np.ndarray,
    core_mask: np.ndarray,
    residue_labels: tuple[str, ...],
    atom_names: tuple[str, ...],
    aggregated_residue_score: np.ndarray | None,
) -> list[dict[str, object]]:
    """Build one CSV-style table for soft robust visualization export."""

    mobility_scores = np.clip(1.0 - weights, 0.0, 1.0)
    rows: list[dict[str, object]] = []
    for atom_index in range(weights.shape[0]):
        rows.append(
            {
                "atom_index": atom_index + 1,
                "atom_name": atom_names[atom_index],
                "residue_index": atom_index + 1,
                "residue_id": residue_labels[atom_index],
                "weight": float(weights[atom_index]),
                "stability_score": float(weights[atom_index]),
                "mobility_score": float(mobility_scores[atom_index]),
                "residual": float(residuals[atom_index]),
                "hard_core_label": int(bool(core_mask[atom_index])),
                "aggregated_residue_score": None
                if aggregated_residue_score is None
                else float(aggregated_residue_score[atom_index]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one visualization table to CSV."""

    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_soft_trajectory_visualization(
    trajectory: Trajectory,
    output_prefix: str | Path,
    *,
    core_fraction: float,
    residue_labels: list[str] | tuple[str, ...] | None = None,
    atom_names: list[str] | tuple[str, ...] | None = None,
    group_ids: np.ndarray | None = None,
    reference_mode: str = "first_frame",
    warm_start: bool = False,
    bfactor_field: str = "mobility_score",
    **robust_kwargs: object,
) -> VisualizationExportArtifacts:
    """Run the soft robust method and export visualization-ready artifacts."""

    trajectory = trajectory if isinstance(trajectory, Trajectory) else Trajectory(coords=trajectory)
    n_atoms = trajectory.n_atoms
    residue_labels = _normalize_optional_labels(
        residue_labels,
        n_atoms=n_atoms,
        default_factory=_default_residue_labels,
    )
    atom_names = _normalize_optional_labels(
        atom_names,
        n_atoms=n_atoms,
        default_factory=_default_atom_names,
    )

    algorithm_start = time.perf_counter()
    trajectory_result = solve_trajectory_robust_alignment(
        trajectory,
        reference_mode=reference_mode,
        warm_start=warm_start,
        **robust_kwargs,
    )
    algorithm_runtime = time.perf_counter() - algorithm_start

    frame_results = trajectory_result.frame_results
    weights = np.mean(np.asarray([frame.result.weights for frame in frame_results], dtype=float), axis=0)
    residuals = np.mean(np.asarray([frame.result.residual_distances for frame in frame_results], dtype=float), axis=0)
    core = select_core_by_target_fraction(weights, core_fraction)
    aggregated_residue_score = _group_score_lookup(weights, group_ids)
    rows = _soft_visualization_rows(
        weights=weights,
        residuals=residuals,
        core_mask=core.core_mask,
        residue_labels=residue_labels,
        atom_names=atom_names,
        aggregated_residue_score=aggregated_residue_score,
    )

    occupancy_values = core.core_mask.astype(float)
    mobility_scores = np.clip(1.0 - weights, 0.0, 1.0)
    stability_scores = np.clip(weights, 0.0, 1.0)
    if bfactor_field == "mobility_score":
        bfactor_values = mobility_scores * 100.0
        bfactor_semantics = "continuous mobility score scaled to [0, 100]; higher values mean more mobile"
    elif bfactor_field == "stability_score":
        bfactor_values = stability_scores * 100.0
        bfactor_semantics = "continuous stability score scaled to [0, 100]; higher values mean more stable"
    else:
        raise ValueError("bfactor_field must be 'mobility_score' or 'stability_score'")

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    occupancy_pdb_path = output_prefix.with_name(f"{output_prefix.name}_soft_core_mask_occupancy.pdb")
    bfactor_pdb_path = output_prefix.with_name(f"{output_prefix.name}_soft_{bfactor_field}_bfactor.pdb")
    csv_path = output_prefix.with_name(f"{output_prefix.name}_soft_scores.csv")
    summary_json_path = output_prefix.with_name(f"{output_prefix.name}_soft_export_summary.json")

    export_start = time.perf_counter()
    write_single_model_pdb(
        trajectory.coords[0],
        occupancy_pdb_path,
        occupancy=occupancy_values,
        bfactor=np.zeros(n_atoms, dtype=float),
        residue_labels=residue_labels,
        atom_names=atom_names,
    )
    write_single_model_pdb(
        trajectory.coords[0],
        bfactor_pdb_path,
        occupancy=occupancy_values,
        bfactor=bfactor_values,
        residue_labels=residue_labels,
        atom_names=atom_names,
    )
    _write_csv(csv_path, rows)
    summary_json_path.write_text(
        json.dumps(
            {
                "method_name": "soft_robust",
                "algorithm_runtime_seconds": algorithm_runtime,
                "export_runtime_seconds": None,
                "render_runtime_seconds": None,
                "reference_mode": reference_mode,
                "warm_start": warm_start,
                "occupancy_semantics": "binary stable-core mask; 1 = selected core, 0 = non-core",
                "bfactor_semantics": bfactor_semantics,
                "bfactor_field": bfactor_field,
                "core_fraction": core.core_fraction,
                "core_size": core.core_size,
                "mean_rmsd": trajectory_result.mean_rmsd,
                "mean_weighted_rmsd": trajectory_result.mean_weighted_rmsd,
                "average_iterations_per_frame": trajectory_result.average_iterations_per_frame,
                "n_frames": trajectory.n_frames,
                "n_atoms": trajectory.n_atoms,
            },
            indent=2,
            sort_keys=True,
        )
    )
    export_runtime = time.perf_counter() - export_start

    payload = json.loads(summary_json_path.read_text())
    payload["export_runtime_seconds"] = export_runtime
    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return VisualizationExportArtifacts(
        method_name="soft_robust",
        occupancy_pdb_path=occupancy_pdb_path,
        bfactor_pdb_path=bfactor_pdb_path,
        csv_path=csv_path,
        summary_json_path=summary_json_path,
        algorithm_runtime_seconds=algorithm_runtime,
        export_runtime_seconds=export_runtime,
        render_runtime_seconds=None,
    )


def _read_multimodel_pdb_bfactors(path: str | Path, n_atoms: int) -> np.ndarray:
    """Read B-factor values from a multi-model PDB."""

    path = Path(path)
    frames: list[list[float]] = []
    current: list[float] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            current.append(float(line[60:66]))
            if len(current) == n_atoms:
                frames.append(current)
                current = []
    if current:
        raise ValueError("aligned PDB ended with an incomplete B-factor frame")
    if not frames:
        raise ValueError(f"no B-factor frames could be parsed from {path}")
    return np.asarray(frames, dtype=float)


def export_external_mdlovofit_visualization(
    aligned_pdb_path: str | Path,
    output_prefix: str | Path,
    *,
    n_atoms: int,
    residue_labels: list[str] | tuple[str, ...] | None = None,
    atom_names: list[str] | tuple[str, ...] | None = None,
    algorithm_runtime_seconds: float | None = None,
    benchmark_overhead_seconds: float | None = None,
) -> VisualizationExportArtifacts:
    """Inspect and export visualization-ready artifacts from external MDLovoFit output."""

    residue_labels = _normalize_optional_labels(
        residue_labels,
        n_atoms=n_atoms,
        default_factory=_default_residue_labels,
    )
    atom_names = _normalize_optional_labels(
        atom_names,
        n_atoms=n_atoms,
        default_factory=_default_atom_names,
    )

    coords, occupancies = read_multimodel_pdb_with_occupancy(aligned_pdb_path, n_atoms=n_atoms)
    bfactors = _read_multimodel_pdb_bfactors(aligned_pdb_path, n_atoms=n_atoms)
    occupancy_first = occupancies[0]
    bfactor_first = bfactors[0]
    occupancy_binary = bool(np.all(np.logical_or(np.isclose(occupancies, 0.0), np.isclose(occupancies, 1.0))))
    occupancy_constant = bool(np.all(np.isclose(occupancies, occupancy_first)))
    bfactor_constant = bool(np.all(np.isclose(bfactors, bfactor_first)))
    if occupancy_binary and occupancy_constant:
        subset_atom_count = int(np.sum(occupancy_first > 0.5))
        subset_atom_fraction = float(subset_atom_count / n_atoms)
    else:
        subset_atom_count = None
        subset_atom_fraction = None

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    occupancy_pdb_path = output_prefix.with_name(f"{output_prefix.name}_mdlovofit_preserved_occupancy.pdb")
    csv_path = output_prefix.with_name(f"{output_prefix.name}_mdlovofit_scores.csv")
    summary_json_path = output_prefix.with_name(f"{output_prefix.name}_mdlovofit_export_summary.json")

    rows: list[dict[str, object]] = []
    for atom_index in range(n_atoms):
        rows.append(
            {
                "atom_index": atom_index + 1,
                "atom_name": atom_names[atom_index],
                "residue_index": atom_index + 1,
                "residue_id": residue_labels[atom_index],
                "raw_occupancy": float(occupancy_first[atom_index]),
                "raw_bfactor": float(bfactor_first[atom_index]),
                "hard_core_label": int(bool(occupancy_first[atom_index] > 0.5)),
            }
        )

    export_start = time.perf_counter()
    write_single_model_pdb(
        coords[0],
        occupancy_pdb_path,
        occupancy=occupancy_first,
        bfactor=bfactor_first,
        residue_labels=residue_labels,
        atom_names=atom_names,
    )
    _write_csv(csv_path, rows)
    summary_json_path.write_text(
        json.dumps(
            {
                "method_name": "external_mdlovofit",
                "algorithm_runtime_seconds": algorithm_runtime_seconds,
                "benchmark_overhead_seconds": benchmark_overhead_seconds,
                "export_runtime_seconds": None,
                "render_runtime_seconds": None,
                "occupancy_semantics": "preserved directly from MDLovoFit aligned output PDB; binary mask interpretation is based on observed output behavior",
                "bfactor_semantics": "preserved directly from MDLovoFit aligned output PDB; no additional semantics assigned by this exporter",
                "n_frames_in_source_pdb": int(coords.shape[0]),
                "n_atoms": n_atoms,
                "occupancy_unique_values": np.unique(occupancies).tolist(),
                "occupancy_mask_binary": occupancy_binary,
                "occupancy_mask_constant_across_frames": occupancy_constant,
                "output_subset_atom_count": subset_atom_count,
                "output_subset_atom_fraction": subset_atom_fraction,
                "bfactor_unique_values": np.unique(bfactors).tolist(),
                "bfactor_constant_across_frames": bfactor_constant,
            },
            indent=2,
            sort_keys=True,
        )
    )
    export_runtime = time.perf_counter() - export_start
    payload = json.loads(summary_json_path.read_text())
    payload["export_runtime_seconds"] = export_runtime
    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return VisualizationExportArtifacts(
        method_name="external_mdlovofit",
        occupancy_pdb_path=occupancy_pdb_path,
        bfactor_pdb_path=None,
        csv_path=csv_path,
        summary_json_path=summary_json_path,
        algorithm_runtime_seconds=algorithm_runtime_seconds,
        export_runtime_seconds=export_runtime,
        render_runtime_seconds=None,
    )
