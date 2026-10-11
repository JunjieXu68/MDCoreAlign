"""Internal no-materialisation benchmark paths.

These helpers intentionally are not exported by the public CLI. They preserve
the registered exact and throughput weight estimators while omitting RMSF,
diagnostics, aligned-coordinate retention, and file serialisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_mean,
)
from mdsuperpose.window_refine import (
    bounded_global_weight_refinement,
    temporally_stratified_indices,
)


@dataclass(slots=True)
class MinimalAlignmentResult:
    """Compact in-memory result for the registered analysis-only contract."""

    mode: str
    backend: str
    frozen_weights: np.ndarray
    core_labels: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    timing: dict[str, float]
    metadata: dict[str, object]

    @property
    def total_runtime_seconds(self) -> float:
        return float(self.timing["total_runtime_seconds"])


def load_ca_multimodel_pdb_minimal(
    path: str | Path,
    *,
    expected_frames: int,
    expected_residues: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a fixed-size CA-only multi-model PDB without Python atom objects."""

    source = Path(path)
    n_frames = int(expected_frames)
    n_residues = int(expected_residues)
    if n_frames < 1 or n_residues < 3:
        raise ValueError("expected_frames and expected_residues are invalid")
    coords = np.empty((n_frames, n_residues, 3), dtype=np.float64)
    chains = np.empty(n_residues, dtype="U1")
    frame_index = 0
    atom_index = 0
    with source.open("rb", buffering=1024 * 1024) as handle:
        for line in handle:
            if line.startswith(b"ATOM"):
                if frame_index >= n_frames or atom_index >= n_residues:
                    raise ValueError("PDB contains more CA coordinates than expected")
                if line[12:16].strip() != b"CA":
                    raise ValueError("benchmark input must be CA-only")
                coords[frame_index, atom_index] = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
                if frame_index == 0:
                    chains[atom_index] = line[21:22].decode("ascii").strip() or "_"
                atom_index += 1
            elif line.startswith(b"ENDMDL"):
                if atom_index != n_residues:
                    raise ValueError(
                        f"model {frame_index + 1} contains {atom_index} CA atoms; "
                        f"expected {n_residues}"
                    )
                frame_index += 1
                atom_index = 0
    if atom_index:
        raise ValueError("PDB ended with an incomplete model")
    if frame_index != n_frames:
        raise ValueError(f"PDB contains {frame_index} models; expected {n_frames}")
    if not np.isfinite(coords).all():
        raise ValueError("PDB contains non-finite coordinates")
    return coords, chains


def _core_labels(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    count = max(1, int(np.floor(weights.size * fraction + 0.5)))
    order = np.lexsort((np.arange(weights.size), -weights))
    labels = np.zeros(weights.size, dtype=bool)
    labels[order[:count]] = True
    return labels


def _compact_cpu_fit(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    values = np.asarray(coords, dtype=np.float64)
    fit = fit_all_frames_fixed_weights(
        values,
        values[0],
        weights,
        chunk_size=chunk_size,
        return_aligned=False,
        compute_rmsd=True,
        compute_weighted_rmsd=True,
        compute_mean_residual=False,
        compute_robust_rmsf=False,
        compute_ordinary_rmsf=False,
        validate_inputs=False,
    )
    assert fit.rmsd is not None
    assert fit.weighted_rmsd is not None
    return fit.rotations, fit.translations, fit.rmsd, fit.weighted_rmsd, {
        "all_frame_fit_seconds": fit.timing["all_frame_weighted_fit"],
        "transform_application_seconds": fit.timing["transform_application"],
        "rmsd_reduction_seconds": fit.timing["RMSD"],
    }


def solve_minimal_cpu(
    coords: np.ndarray,
    *,
    mode: str,
    chain_ids: np.ndarray | None = None,
    representative_count: int | None = None,
    update_rounds: int = 4,
    chunk_size: int = 1000,
) -> MinimalAlignmentResult:
    """Run exact or CPU-throughput analysis without aligned materialisation."""

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[1] < 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (values.shape[1],):
        raise ValueError("chain_ids must match n_residues")
    if mode == "exact":
        count = min(values.shape[0], representative_count or 2000)
        selected = temporally_stratified_indices(values.shape[0], count)
        weight_start = time.perf_counter()
        estimate = estimate_representative_weights_sequential_mean(
            values[selected],
            values[0],
            rule="huber",
            delta=1.0,
            max_iterations=20,
            tolerance=1e-6,
            objective_tolerance=1e-5,
            objective_patience=2,
            sequence_continuity_strength=0.05,
            contact_consistency_strength=0.0,
            sequence_chain_ids=chains,
            record_history=False,
        )
        weights = np.asarray(estimate.mean_weights, dtype=np.float64)
        weight_seconds = time.perf_counter() - weight_start
    elif mode == "throughput":
        count = min(values.shape[0], representative_count or 256)
        selected = temporally_stratified_indices(values.shape[0], count)
        weight_start = time.perf_counter()
        weights, _, _, weight_detail = bounded_global_weight_refinement(
            values[selected],
            values[0],
            np.ones(values.shape[1], dtype=np.float64),
            update_rounds=update_rounds,
            sequence_chain_ids=chains,
        )
        weight_seconds = time.perf_counter() - weight_start
    else:
        raise ValueError("mode must be 'exact' or 'throughput'")
    fit_start = time.perf_counter()
    rotations, translations, rmsd, weighted_rmsd, fit_timing = _compact_cpu_fit(
        values,
        weights,
        chunk_size=chunk_size,
    )
    compact_fit_seconds = time.perf_counter() - fit_start
    total_seconds = time.perf_counter() - total_start
    timing = {
        "weight_estimation_seconds": weight_seconds,
        **fit_timing,
        "compact_fit_and_reduction_seconds": compact_fit_seconds,
        "total_runtime_seconds": total_seconds,
    }
    if mode == "throughput":
        timing.update({f"weight_{key}": value for key, value in weight_detail.items()})
    return MinimalAlignmentResult(
        mode=mode,
        backend="cpu",
        frozen_weights=weights,
        core_labels=_core_labels(weights),
        rotations=rotations,
        translations=translations,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        timing=timing,
        metadata={
            "representative_count": int(selected.size),
            "update_rounds": 20 if mode == "exact" else int(update_rounds),
            "average_iterations_per_frame": (
                estimate.average_iterations_per_frame if mode == "exact" else None
            ),
            "converged_fraction": (
                estimate.converged_fraction if mode == "exact" else None
            ),
            "aligned_coordinates_materialised": False,
            "RMSF_computed": False,
            "public_mode": False,
        },
    )


def solve_minimal_gpu(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    update_rounds: int = 4,
    chunk_size: int = 1000,
) -> MinimalAlignmentResult:
    """Run registered float32/Horn throughput without RMSF or aligned output."""

    from mdcorealign.gpu_backend import (
        _cupy,
        _event_seconds,
        _gpu_weight_estimation,
        _transfer_to_device,
        _weighted_kabsch_gpu,
    )

    total_start = time.perf_counter()
    cp = _cupy()
    values = np.asarray(coords)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[1] < 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    selected = temporally_stratified_indices(
        values.shape[0], min(int(representative_count), values.shape[0])
    )
    weights, weight_timing, risk = _gpu_weight_estimation(
        values,
        values[0],
        selected,
        chains=chains,
        update_rounds=update_rounds,
        dtype=cp.float32,
        rotation_solver="horn",
    )
    target, target_h2d = _transfer_to_device(cp, values[0], cp.float32)
    device_weights, weight_h2d = _transfer_to_device(cp, weights, cp.float32)
    n_frames = values.shape[0]
    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    rmsd = np.empty(n_frames, dtype=np.float64)
    weighted_rmsd = np.empty(n_frames, dtype=np.float64)
    h2d_seconds = target_h2d + weight_h2d
    compute_seconds = 0.0
    d2h_seconds = 0.0
    peak_vram = int(cp.get_default_memory_pool().used_bytes())
    for start in range(0, n_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frames)
        device_chunk, elapsed = _transfer_to_device(
            cp, values[start:stop], cp.float32
        )
        h2d_seconds += elapsed

        def compute(
            device_chunk=device_chunk,
            target=target,
            device_weights=device_weights,
        ):
            rotation, translation, aligned = _weighted_kabsch_gpu(
                cp,
                device_chunk,
                target,
                device_weights,
                rotation_solver="horn",
            )
            delta = aligned - target[None, :, :]
            squared = cp.sum(delta * delta, axis=2)
            full = cp.sqrt(cp.mean(squared, axis=1))
            weighted = cp.sqrt(
                cp.sum(squared * device_weights[None, :], axis=1)
                / cp.sum(device_weights)
            )
            return rotation, translation, full, weighted

        device_output, elapsed = _event_seconds(cp, compute)
        compute_seconds += elapsed
        transfer_start = time.perf_counter()
        host = [cp.asnumpy(item) for item in device_output]
        cp.cuda.Device().synchronize()
        d2h_seconds += time.perf_counter() - transfer_start
        rotations[start:stop] = host[0]
        translations[start:stop] = host[1]
        rmsd[start:stop] = host[2]
        weighted_rmsd[start:stop] = host[3]
        peak_vram = max(peak_vram, int(cp.get_default_memory_pool().used_bytes()))
        del device_chunk, device_output
    del target, device_weights
    cp.get_default_memory_pool().free_all_blocks()
    total_seconds = time.perf_counter() - total_start
    return MinimalAlignmentResult(
        mode="throughput",
        backend="gpu",
        frozen_weights=weights,
        core_labels=_core_labels(weights),
        rotations=rotations,
        translations=translations,
        rmsd=rmsd,
        weighted_rmsd=weighted_rmsd,
        timing={
            "weight_estimation_seconds": sum(weight_timing.values()),
            **weight_timing,
            "all_frame_H2D_seconds": h2d_seconds,
            "all_frame_GPU_compute_seconds": compute_seconds,
            "all_frame_D2H_seconds": d2h_seconds,
            "total_runtime_seconds": total_seconds,
        },
        metadata={
            "representative_count": int(selected.size),
            "update_rounds": int(update_rounds),
            "precision": "float32",
            "rotation_solver": "horn",
            "peak_GPU_memory_bytes": peak_vram,
            "explicit_CUDA_synchronization": True,
            "aligned_coordinates_materialised": False,
            "RMSF_computed": False,
            "public_mode": False,
            **risk,
        },
    )


def solve_minimal_from_pdb(
    path: str | Path,
    *,
    mode: str,
    backend: str,
    expected_frames: int,
    expected_residues: int,
) -> MinimalAlignmentResult:
    """Execute Contract C1 from one pre-generated PDB inside the timed API."""

    total_start = time.perf_counter()
    parse_start = time.perf_counter()
    coords, chains = load_ca_multimodel_pdb_minimal(
        path,
        expected_frames=expected_frames,
        expected_residues=expected_residues,
    )
    parse_seconds = time.perf_counter() - parse_start
    if backend == "cpu":
        result = solve_minimal_cpu(coords, mode=mode, chain_ids=chains)
    elif backend == "gpu" and mode == "throughput":
        result = solve_minimal_gpu(coords, chain_ids=chains)
    else:
        raise ValueError("GPU is available only for throughput mode")
    result.timing["PDB_parsing_seconds"] = parse_seconds
    result.timing["C1_total_runtime_seconds"] = time.perf_counter() - total_start
    return result
