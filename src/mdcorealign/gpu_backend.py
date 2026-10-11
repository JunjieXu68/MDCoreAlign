"""Optional CuPy backend for throughput alignment and W5 diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdsuperpose.flexibility import detect_flexible_segments
from mdsuperpose.window_descriptors import build_local_window_edges
from mdsuperpose.window_kinematics import LocalDeformationResult
from mdsuperpose.window_scan import (
    WindowScanResult,
    aggregate_edge_movement,
    map_movement_to_stability,
    smooth_weights_chain_aware,
    stable_labels_from_weights,
)


@dataclass(slots=True)
class GPUAlignmentResult:
    """GPU throughput output with explicit transfer and compute timings."""

    mode: str
    backend: str
    precision: str
    rotation_solver: str
    selected_frame_indices: np.ndarray
    frozen_weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray | None
    robust_rmsf: np.ndarray
    ordinary_rmsf: np.ndarray
    mean_residual: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    total_runtime_seconds: float
    timing: dict[str, float]
    risk_signals: dict[str, object]
    metadata: dict[str, object]

    @property
    def mean_rmsd(self) -> float:
        return float(np.mean(self.rmsd))

    @property
    def mean_weighted_rmsd(self) -> float:
        return float(np.mean(self.weighted_rmsd))

    @property
    def n_frames(self) -> int:
        """Return the number of fitted frames."""

        return int(self.rmsd.size)

    @property
    def n_atoms(self) -> int:
        """Return the number of fitted atoms or residues."""

        return int(self.frozen_weights.size)

    @property
    def reference_mode(self) -> str:
        """Return the locked throughput reference convention."""

        return "first_frame"

    @property
    def warm_start(self) -> bool:
        """Report that independent throughput updates do not warm start."""

        return False

    @property
    def selection_strategy(self) -> str:
        """Return the representative-frame selection strategy."""

        return "uniform"

    @property
    def max_weight_fit_frames(self) -> int:
        """Return the number of representative frames used."""

        return int(self.selected_frame_indices.size)

    @property
    def chunk_size(self) -> int:
        """Return the configured all-frame chunk size."""

        return int(self.metadata["chunk_size"])

    @property
    def chunks_processed(self) -> int:
        """Return the number of all-frame chunks."""

        return int(np.ceil(self.n_frames / max(1, self.chunk_size)))

    @property
    def average_runtime_per_frame(self) -> float:
        """Return total runtime divided by frame count."""

        return self.total_runtime_seconds / max(1, self.n_frames)

    @property
    def average_iterations_per_frame(self) -> float:
        """Return the fixed representative update budget."""

        return float(self.metadata["update_rounds"])

    @property
    def converged_fraction(self) -> float:
        """Return a compact stability indicator, not strict IRLS convergence."""

        return float(
            float(self.risk_signals["last_two_round_core_jaccard"]) >= 0.90
        )


def _cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "GPU backend requested but no compatible CuPy/CUDA runtime is "
            "available. The installed MDCoreAlign package already contains "
            "the GPU backend. Install a CuPy build compatible with this "
            "machine's CUDA runtime, then run `mdcorealign doctor`."
        ) from exc
    return cp


def gpu_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""

    try:
        cp = _cupy()
        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def gpu_environment() -> dict[str, object]:
    """Return optional-backend and visible-device metadata."""

    cp = _cupy()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    return {
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "device_id": int(device.id),
        "device_name": properties["name"].decode(),
        "compute_capability": (
            f"{properties['major']}.{properties['minor']}"
        ),
        "total_global_memory_bytes": int(properties["totalGlobalMem"]),
    }


def _event_seconds(cp, function, *, stream=None):
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    active_stream = cp.cuda.get_current_stream() if stream is None else stream
    with active_stream:
        start.record(active_stream)
        output = function()
        stop.record(active_stream)
    stop.synchronize()
    return output, float(cp.cuda.get_elapsed_time(start, stop) / 1000.0)


def _pinned_double_buffers(cp, shape, dtype):
    """Allocate two pinned host/device buffers, or return a safe fallback."""

    numpy_dtype = np.dtype(
        np.float32 if dtype == cp.float32 else np.float64
    )
    n_values = int(np.prod(shape))
    n_bytes = n_values * numpy_dtype.itemsize
    try:
        memory = [
            cp.cuda.alloc_pinned_memory(n_bytes),
            cp.cuda.alloc_pinned_memory(n_bytes),
        ]
        host = [
            np.frombuffer(item, dtype=numpy_dtype, count=n_values).reshape(
                shape
            )
            for item in memory
        ]
        device = [
            cp.empty(shape, dtype=dtype),
            cp.empty(shape, dtype=dtype),
        ]
        return memory, host, device
    except Exception:  # pragma: no cover - allocation depends on host/driver.
        return None, None, None


@dataclass(slots=True)
class _PersistentGPUWorkspace:
    """Reusable GPU buffers that never retain a trajectory object reference."""

    cp: Any
    capacity_frames: int = 0
    n_residues: int = 0
    dtype_name: str = ""
    pinned_memory: Any = None
    pinned_host: Any = None
    device_buffers: Any = None
    transfer_stream: Any = None
    compute_stream: Any = None
    covariance_scratch: Any = None
    delta_scratch: Any = None
    squared_scratch: Any = None
    allocation_generation: int = 0

    def ensure(
        self,
        *,
        capacity_frames: int,
        n_residues: int,
        dtype,
    ) -> bool:
        """Ensure reusable buffers exist and return whether allocation was reused."""

        capacity = int(capacity_frames)
        dtype_name = str(dtype)
        reusable = (
            self.device_buffers is not None
            and self.capacity_frames >= capacity
            and self.n_residues == int(n_residues)
            and self.dtype_name == dtype_name
        )
        if reusable:
            return True
        pinned_memory, pinned_host, device_buffers = _pinned_double_buffers(
            self.cp,
            (capacity, int(n_residues), 3),
            dtype,
        )
        self.capacity_frames = capacity
        self.n_residues = int(n_residues)
        self.dtype_name = dtype_name
        self.pinned_memory = pinned_memory
        self.pinned_host = pinned_host
        self.device_buffers = device_buffers
        self.transfer_stream = self.cp.cuda.Stream(non_blocking=True)
        self.compute_stream = self.cp.cuda.Stream(non_blocking=True)
        self.covariance_scratch = self.cp.empty(
            (capacity, 3, 3),
            dtype=dtype,
        )
        self.delta_scratch = self.cp.empty(
            (capacity, int(n_residues), 3),
            dtype=dtype,
        )
        self.squared_scratch = self.cp.empty(
            (capacity, int(n_residues)),
            dtype=dtype,
        )
        self.allocation_generation += 1
        return False

    def clear_trajectory_data(self) -> None:
        """Zero buffers that may contain coordinates before accepting another job."""

        if self.pinned_host is not None:
            for buffer in self.pinned_host:
                buffer.fill(0)
        device_arrays = (
            *(self.device_buffers or ()),
            self.covariance_scratch,
            self.delta_scratch,
            self.squared_scratch,
        )
        for buffer in device_arrays:
            if buffer is not None:
                buffer.fill(0)
        self.cp.cuda.Device().synchronize()


def _row_operators_svd(cp, covariances):
    u, _, vt = cp.linalg.svd(covariances, full_matrices=False)
    determinants = cp.linalg.det(cp.matmul(u, vt))
    corrected = u.copy()
    corrected[:, :, 2] *= cp.where(
        determinants < 0.0,
        -1.0,
        1.0,
    )[:, None]
    return cp.matmul(corrected, vt)


def _row_operators_horn(cp, covariances):
    """Return proper row rotations from Davenport quaternion eigenproblems."""

    h = covariances
    batch = h.shape[0]
    sigma = cp.trace(h, axis1=1, axis2=2)
    symmetric = h + cp.swapaxes(h, 1, 2)
    vector = cp.stack(
        (
            h[:, 1, 2] - h[:, 2, 1],
            h[:, 2, 0] - h[:, 0, 2],
            h[:, 0, 1] - h[:, 1, 0],
        ),
        axis=1,
    )
    matrix = cp.empty((batch, 4, 4), dtype=h.dtype)
    matrix[:, 0, 0] = sigma
    matrix[:, 0, 1:] = vector
    matrix[:, 1:, 0] = vector
    matrix[:, 1:, 1:] = (
        symmetric - sigma[:, None, None] * cp.eye(3, dtype=h.dtype)
    )
    _, eigenvectors = cp.linalg.eigh(matrix)
    q = eigenvectors[:, :, -1]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation = cp.empty((batch, 3, 3), dtype=h.dtype)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return cp.swapaxes(rotation, 1, 2)


def _weighted_kabsch_gpu(
    cp,
    coords,
    reference,
    weights,
    *,
    rotation_solver: str,
    covariance_out=None,
):
    weight_sum = cp.sum(weights)
    reference_center = cp.sum(reference * weights[:, None], axis=0) / weight_sum
    mobile_centers = cp.einsum("bni,n->bi", coords, weights) / weight_sum
    mobile_centered = coords - mobile_centers[:, None, :]
    reference_centered = reference - reference_center
    weighted_reference = (
        weights[:, None] / weight_sum
    ) * reference_centered
    if covariance_out is None:
        covariance = cp.matmul(
            cp.swapaxes(mobile_centered, 1, 2),
            weighted_reference,
        )
    else:
        covariance = covariance_out[: coords.shape[0]]
        cp.matmul(
            cp.swapaxes(mobile_centered, 1, 2),
            weighted_reference,
            out=covariance,
        )
    if rotation_solver in {"svd", "cusolver_svd"}:
        row_operators = _row_operators_svd(cp, covariance)
    elif rotation_solver == "horn":
        row_operators = _row_operators_horn(cp, covariance)
    else:
        raise ValueError(
            "rotation_solver must be 'svd', 'cusolver_svd', or 'horn'"
        )
    rotations = cp.swapaxes(row_operators, 1, 2)
    translations = reference_center[None, :] - cp.einsum(
        "bi,bij->bj",
        mobile_centers,
        row_operators,
    )
    aligned = cp.matmul(coords, row_operators) + translations[:, None, :]
    return rotations, translations, aligned


def _smooth_chain_gpu(cp, frame_weights, chains, strength: float = 0.05):
    if strength <= 0.0 or frame_weights.shape[1] < 2:
        return frame_weights
    n_residues = frame_weights.shape[1]
    valid = np.ones(n_residues - 1, dtype=bool)
    if chains is not None:
        chain_values = np.asarray(chains)
        valid = chain_values[1:] == chain_values[:-1]
    valid_gpu = cp.asarray(valid)
    neighbour_sum = cp.zeros_like(frame_weights)
    neighbour_count = cp.zeros(n_residues, dtype=frame_weights.dtype)
    neighbour_sum[:, 1:] += frame_weights[:, :-1] * valid_gpu[None, :]
    neighbour_sum[:, :-1] += frame_weights[:, 1:] * valid_gpu[None, :]
    neighbour_count[1:] += valid_gpu
    neighbour_count[:-1] += valid_gpu
    neighbour_mean = neighbour_sum / cp.maximum(
        neighbour_count[None, :],
        1.0,
    )
    return cp.where(
        neighbour_count[None, :] > 0,
        (1.0 - strength) * frame_weights + strength * neighbour_mean,
        frame_weights,
    )


def _transfer_to_device(cp, values: np.ndarray, dtype):
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    device = cp.asarray(values, dtype=dtype)
    cp.cuda.Device().synchronize()
    return device, time.perf_counter() - start


def _transfer_to_host(cp, values):
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    host = cp.asnumpy(values)
    cp.cuda.Device().synchronize()
    return host, time.perf_counter() - start


def _gpu_weight_estimation(
    coords: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    *,
    chains: np.ndarray | None,
    update_rounds: int,
    dtype,
    rotation_solver: str,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    cp = _cupy()
    reps, h2d_reps = _transfer_to_device(cp, coords[indices], dtype)
    target, h2d_reference = _transfer_to_device(cp, reference, dtype)
    weights = cp.ones(coords.shape[1], dtype=dtype)
    histories: list[np.ndarray] = [np.ones(coords.shape[1])]
    fit_seconds = 0.0
    update_seconds = 0.0
    for _ in range(int(update_rounds)):
        fit, elapsed = _event_seconds(
            cp,
            lambda reps=reps, target=target, weights=weights: _weighted_kabsch_gpu(
                cp,
                reps,
                target,
                weights,
                rotation_solver=rotation_solver,
            ),
        )
        fit_seconds += elapsed
        _, _, aligned = fit

        def update(aligned=aligned, target=target):
            distances = cp.linalg.norm(
                aligned - target[None, :, :],
                axis=2,
            )
            frame_weights = cp.where(
                distances > delta,
                delta / cp.maximum(distances, 1e-12),
                1.0,
            )
            frame_weights = _smooth_chain_gpu(
                cp,
                frame_weights,
                chains,
                sequence_continuity_strength,
            )
            next_weights = cp.mean(frame_weights, axis=0)
            return next_weights / cp.max(next_weights)

        weights, elapsed = _event_seconds(cp, update)
        update_seconds += elapsed
        histories.append(cp.asnumpy(weights))
    host_weights, d2h = _transfer_to_host(cp, weights)
    del reps, target, weights
    cp.get_default_memory_pool().free_all_blocks()
    core_count = max(1, int(np.floor(coords.shape[1] * 0.6 + 0.5)))
    previous_order = np.lexsort(
        (np.arange(coords.shape[1]), -histories[-2])
    )
    final_order = np.lexsort(
        (np.arange(coords.shape[1]), -histories[-1])
    )
    previous_core = np.zeros(coords.shape[1], dtype=bool)
    final_core = np.zeros(coords.shape[1], dtype=bool)
    previous_core[previous_order[:core_count]] = True
    final_core[final_order[:core_count]] = True
    union = np.count_nonzero(previous_core | final_core)
    return (
        host_weights.astype(np.float64),
        {
            "representative_H2D": h2d_reps + h2d_reference,
            "representative_gpu_fit": fit_seconds,
            "representative_gpu_weight_update": update_seconds,
            "weight_D2H": d2h,
        },
        {
            "final_weight_Linf_change": float(
                np.max(np.abs(histories[-1] - histories[-2]))
            ),
            "last_two_round_core_jaccard": (
                1.0
                if union == 0
                else float(
                    np.count_nonzero(previous_core & final_core) / union
                )
            ),
        },
    )


def _gpu_all_frame_fit(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
    dtype,
    rotation_solver: str,
    store_aligned: bool,
    use_pinned_double_buffer: bool,
    release_memory_pool: bool,
    workspace: _PersistentGPUWorkspace | None = None,
) -> tuple[
    dict[str, np.ndarray | None],
    dict[str, float],
    int,
    dict[str, bool],
]:
    cp = _cupy()
    n_frames, n_residues, _ = coords.shape
    target, reference_h2d = _transfer_to_device(cp, reference, dtype)
    device_weights, weight_h2d = _transfer_to_device(cp, weights, dtype)
    uniform = cp.ones(n_residues, dtype=dtype)
    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    rmsd = np.empty(n_frames, dtype=np.float64)
    weighted_rmsd = np.empty(n_frames, dtype=np.float64)
    aligned_host = (
        np.empty_like(coords, dtype=np.float64) if store_aligned else None
    )
    robust_sum = cp.zeros((n_residues, 3), dtype=cp.float64)
    robust_sumsq = cp.zeros((n_residues, 3), dtype=cp.float64)
    ordinary_sum = cp.zeros((n_residues, 3), dtype=cp.float64)
    ordinary_sumsq = cp.zeros((n_residues, 3), dtype=cp.float64)
    residual_sum = cp.zeros(n_residues, dtype=cp.float64)
    h2d_seconds = reference_h2d + weight_h2d
    compute_seconds = 0.0
    d2h_seconds = 0.0
    peak_used = int(cp.get_default_memory_pool().used_bytes())
    starts = list(range(0, n_frames, int(chunk_size)))
    pinned_memory = pinned_host = device_buffers = None
    transfer_stream = compute_stream = None
    covariance_scratch = delta_scratch = squared_scratch = None
    workspace_reused = False
    transfer_events: list[tuple[object, object]] = []
    if use_pinned_double_buffer:
        capacity = min(int(chunk_size), n_frames)
        if workspace is None:
            pinned_memory, pinned_host, device_buffers = _pinned_double_buffers(
                cp,
                (capacity, n_residues, 3),
                dtype,
            )
        else:
            workspace_reused = workspace.ensure(
                capacity_frames=capacity,
                n_residues=n_residues,
                dtype=dtype,
            )
            pinned_memory = workspace.pinned_memory
            pinned_host = workspace.pinned_host
            device_buffers = workspace.device_buffers
            transfer_stream = workspace.transfer_stream
            compute_stream = workspace.compute_stream
            covariance_scratch = workspace.covariance_scratch
            delta_scratch = workspace.delta_scratch
            squared_scratch = workspace.squared_scratch
    pinned_active = device_buffers is not None
    double_buffer_active = pinned_active and len(starts) > 1
    if pinned_active:
        if transfer_stream is None:
            transfer_stream = cp.cuda.Stream(non_blocking=True)
        if compute_stream is None:
            compute_stream = cp.cuda.Stream(non_blocking=True)

        def prefetch(chunk_index: int):
            start = starts[chunk_index]
            stop = min(start + int(chunk_size), n_frames)
            slot = chunk_index % 2
            count = stop - start
            np.copyto(pinned_host[slot][:count], coords[start:stop])
            transfer_start = cp.cuda.Event()
            transfer_stop = cp.cuda.Event()
            with transfer_stream:
                transfer_start.record(transfer_stream)
                device_buffers[slot][:count].set(
                    pinned_host[slot][:count],
                    stream=transfer_stream,
                )
                transfer_stop.record(transfer_stream)
            transfer_events.append((transfer_start, transfer_stop))
            return device_buffers[slot][:count], transfer_stop

        device_chunk, ready_event = prefetch(0)

    for chunk_index, start in enumerate(starts):
        stop = min(start + int(chunk_size), n_frames)
        if pinned_active:
            compute_stream.wait_event(ready_event)
        else:
            device_chunk, transfer = _transfer_to_device(
                cp,
                coords[start:stop],
                dtype,
            )
            h2d_seconds += transfer

        def compute(
            device_chunk=device_chunk,
            target=target,
            device_weights=device_weights,
            uniform=uniform,
            covariance_scratch=covariance_scratch,
            delta_scratch=delta_scratch,
            squared_scratch=squared_scratch,
        ):
            robust = _weighted_kabsch_gpu(
                cp,
                device_chunk,
                target,
                device_weights,
                rotation_solver=rotation_solver,
                covariance_out=covariance_scratch,
            )
            ordinary = _weighted_kabsch_gpu(
                cp,
                device_chunk,
                target,
                uniform,
                rotation_solver=rotation_solver,
                covariance_out=covariance_scratch,
            )
            robust_rot, robust_trans, robust_aligned = robust
            _, _, ordinary_aligned = ordinary
            if delta_scratch is None or squared_scratch is None:
                delta = robust_aligned - target[None, :, :]
                squared = cp.sum(delta * delta, axis=2)
            else:
                count = robust_aligned.shape[0]
                delta = delta_scratch[:count]
                squared = squared_scratch[:count]
                cp.subtract(
                    robust_aligned,
                    target[None, :, :],
                    out=delta,
                )
                cp.sum(delta * delta, axis=2, out=squared)
            chunk_rmsd = cp.sqrt(cp.mean(squared, axis=1))
            chunk_weighted = cp.sqrt(
                cp.sum(squared * device_weights[None, :], axis=1)
                / cp.sum(device_weights)
            )
            return (
                robust_rot,
                robust_trans,
                robust_aligned,
                ordinary_aligned,
                chunk_rmsd,
                chunk_weighted,
                cp.sum(cp.sqrt(squared), axis=0),
            )

        if pinned_active:
            compute_start = cp.cuda.Event()
            compute_stop = cp.cuda.Event()
            with compute_stream:
                compute_start.record(compute_stream)
                computed = compute()
                compute_stop.record(compute_stream)
            if chunk_index + 1 < len(starts):
                next_device_chunk, next_ready_event = prefetch(
                    chunk_index + 1
                )
            compute_stop.synchronize()
            compute_seconds += float(
                cp.cuda.get_elapsed_time(compute_start, compute_stop)
                / 1000.0
            )
        else:
            computed, elapsed = _event_seconds(cp, compute)
            compute_seconds += elapsed
        (
            device_rotations,
            device_translations,
            robust_aligned,
            ordinary_aligned,
            device_rmsd,
            device_weighted,
            chunk_residual_sum,
        ) = computed

        def reduce_rmsf(
            robust_aligned=robust_aligned,
            ordinary_aligned=ordinary_aligned,
            chunk_residual_sum=chunk_residual_sum,
        ):
            robust_sum[...] += cp.sum(
                robust_aligned,
                axis=0,
                dtype=cp.float64,
            )
            robust_sumsq[...] += cp.sum(
                robust_aligned * robust_aligned,
                axis=0,
                dtype=cp.float64,
            )
            ordinary_sum[...] += cp.sum(
                ordinary_aligned,
                axis=0,
                dtype=cp.float64,
            )
            ordinary_sumsq[...] += cp.sum(
                ordinary_aligned * ordinary_aligned,
                axis=0,
                dtype=cp.float64,
            )
            residual_sum[...] += chunk_residual_sum

        _, elapsed = _event_seconds(
            cp,
            reduce_rmsf,
            stream=compute_stream if pinned_active else None,
        )
        compute_seconds += elapsed
        peak_used = max(
            peak_used,
            int(cp.get_default_memory_pool().used_bytes()),
        )
        transfer_values = (
            device_rotations,
            device_translations,
            device_rmsd,
            device_weighted,
        )
        transfer_start = time.perf_counter()
        host_values = [cp.asnumpy(value) for value in transfer_values]
        if store_aligned:
            host_aligned = cp.asnumpy(robust_aligned)
        cp.cuda.Device().synchronize()
        d2h_seconds += time.perf_counter() - transfer_start
        rotations[start:stop] = host_values[0]
        translations[start:stop] = host_values[1]
        rmsd[start:stop] = host_values[2]
        weighted_rmsd[start:stop] = host_values[3]
        if store_aligned and aligned_host is not None:
            aligned_host[start:stop] = host_aligned
        del (
            device_rotations,
            device_translations,
            robust_aligned,
            ordinary_aligned,
            device_rmsd,
            device_weighted,
            chunk_residual_sum,
        )
        if not pinned_active:
            del device_chunk
        elif chunk_index + 1 < len(starts):
            device_chunk = next_device_chunk
            ready_event = next_ready_event
    for transfer_start, transfer_stop in transfer_events:
        transfer_stop.synchronize()
        h2d_seconds += float(
            cp.cuda.get_elapsed_time(transfer_start, transfer_stop) / 1000.0
        )
    finalize_start = time.perf_counter()
    robust_mean = robust_sum / n_frames
    ordinary_mean = ordinary_sum / n_frames
    robust_variance = cp.maximum(
        robust_sumsq / n_frames - robust_mean * robust_mean,
        0.0,
    )
    ordinary_variance = cp.maximum(
        ordinary_sumsq / n_frames - ordinary_mean * ordinary_mean,
        0.0,
    )
    robust_rmsf = cp.asnumpy(cp.sqrt(cp.sum(robust_variance, axis=1)))
    ordinary_rmsf = cp.asnumpy(cp.sqrt(cp.sum(ordinary_variance, axis=1)))
    mean_residual = cp.asnumpy(residual_sum / n_frames)
    cp.cuda.Device().synchronize()
    d2h_seconds += time.perf_counter() - finalize_start
    del target, device_weights, uniform
    if device_buffers is not None and workspace is None:
        del device_buffers
    if workspace is None:
        del pinned_host, pinned_memory
    if release_memory_pool:
        cp.get_default_memory_pool().free_all_blocks()
    return (
        {
            "rotations": rotations,
            "translations": translations,
            "aligned_coords": aligned_host,
            "robust_rmsf": robust_rmsf,
            "ordinary_rmsf": ordinary_rmsf,
            "mean_residual": mean_residual,
            "rmsd": rmsd,
            "weighted_rmsd": weighted_rmsd,
        },
        {
            "all_frame_H2D": h2d_seconds,
            "all_frame_gpu_compute": compute_seconds,
            "all_frame_D2H": d2h_seconds,
        },
        peak_used,
        {
            "pinned_host_memory": pinned_active,
            "double_buffering": double_buffer_active,
            "persistent_workspace_used": workspace is not None,
            "persistent_workspace_reused": workspace_reused,
            "covariance_scratch_reused": workspace is not None,
            "residual_scratch_reused": workspace is not None,
        },
    )


def solve_fast_global_gpu(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    update_rounds: int = 4,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
    chunk_size: int = 2000,
    precision: str = "float32",
    rotation_solver: str = "svd",
    store_aligned: bool = False,
    use_pinned_double_buffer: bool = True,
    release_memory_pool: bool = True,
    selected_frame_indices: np.ndarray | None = None,
    _workspace: _PersistentGPUWorkspace | None = None,
) -> GPUAlignmentResult:
    """Run fixed-budget shared-weight estimation and all-frame fitting on GPU."""

    total_start = time.perf_counter()
    from mdsuperpose.window_refine import validate_weighting_parameters

    validate_weighting_parameters(delta, sequence_continuity_strength)
    if int(update_rounds) < 1:
        raise ValueError("update_rounds must be at least 1")
    cp = _cupy()
    values = np.asarray(coords)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (values.shape[1],):
        raise ValueError("chain_ids must have shape (n_residues,)")
    if precision == "float64":
        estimate_dtype = cp.float64
        fit_dtype = cp.float64
    elif precision == "float32":
        estimate_dtype = cp.float32
        fit_dtype = cp.float32
    elif precision == "mixed":
        estimate_dtype = cp.float32
        fit_dtype = cp.float64
    else:
        raise ValueError("precision must be 'float64', 'float32', or 'mixed'")
    if selected_frame_indices is None:
        selected = np.unique(
            np.linspace(
                0,
                values.shape[0] - 1,
                min(representative_count, values.shape[0]),
                dtype=np.int64,
            )
        )
    else:
        selected = np.asarray(selected_frame_indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size < 1:
            raise ValueError("selected_frame_indices must be non-empty 1D")
        if selected.min() < 0 or selected.max() >= values.shape[0]:
            raise ValueError("selected_frame_indices are outside frame range")
    weights, weight_timing, risk = _gpu_weight_estimation(
        values,
        values[0],
        selected,
        chains=chains,
        update_rounds=update_rounds,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
        dtype=estimate_dtype,
        rotation_solver=rotation_solver,
    )
    fit, fit_timing, peak_vram, buffer_metadata = _gpu_all_frame_fit(
        values,
        values[0],
        weights,
        chunk_size=chunk_size,
        dtype=fit_dtype,
        rotation_solver=rotation_solver,
        store_aligned=store_aligned,
        use_pinned_double_buffer=use_pinned_double_buffer,
        release_memory_pool=release_memory_pool,
        workspace=_workspace,
    )
    total_seconds = time.perf_counter() - total_start
    return GPUAlignmentResult(
        mode="throughput",
        backend="gpu",
        precision=precision,
        rotation_solver=rotation_solver,
        selected_frame_indices=selected,
        frozen_weights=weights,
        rotations=fit["rotations"],
        translations=fit["translations"],
        aligned_coords=fit["aligned_coords"],
        robust_rmsf=fit["robust_rmsf"],
        ordinary_rmsf=fit["ordinary_rmsf"],
        mean_residual=fit["mean_residual"],
        rmsd=fit["rmsd"],
        weighted_rmsd=fit["weighted_rmsd"],
        total_runtime_seconds=total_seconds,
        timing={
            **weight_timing,
            **fit_timing,
            "GPU_compute_total": (
                weight_timing["representative_gpu_fit"]
                + weight_timing["representative_gpu_weight_update"]
                + fit_timing["all_frame_gpu_compute"]
            ),
            "CPU_GPU_transfer_total": (
                weight_timing["representative_H2D"]
                + weight_timing["weight_D2H"]
                + fit_timing["all_frame_H2D"]
                + fit_timing["all_frame_D2H"]
            ),
            "total_runtime_seconds": total_seconds,
        },
        risk_signals=risk,
        metadata={
            "status": "production",
            "execution_contract": "throughput",
            "fixed_budget_estimator": True,
            "backend": "cupy_cuda12",
            "automatic_exact_fallback": False,
            "peak_GPU_memory_bytes": peak_vram,
            "chunk_size": int(chunk_size),
            "update_rounds": int(update_rounds),
            "effective_parameters": {
                "delta": float(delta), "rule": "huber",
                "sequence_continuity_strength": float(sequence_continuity_strength),
                "warm_start": False, "update_rounds": int(update_rounds),
            },
            "explicit_CUDA_synchronization": True,
            **buffer_metadata,
        },
    )


def scan_w5_gpu(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
    precision: str = "float32",
    frame_chunk_size: int = 5000,
    edge_chunk_size: int = 512,
) -> LocalDeformationResult:
    """Run chain-aware W5 relative-MAD scanning with bounded edge chunks."""

    total_start = time.perf_counter()
    cp = _cupy()
    values = np.asarray(coords)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    chains = (
        np.asarray([""] * values.shape[1], dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids)
    )
    edges = build_local_window_edges(
        values.shape[1],
        window_size=5,
        feature_set="minimal",
        chain_ids=chains,
        residue_positions=residue_positions,
    )
    dtype = cp.float64 if precision == "float64" else cp.float32
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    full_bytes = values.size * np.dtype(
        np.float64 if precision == "float64" else np.float32
    ).itemsize
    keep_full = full_bytes <= int(free_bytes * 0.45)
    h2d_seconds = 0.0
    compute_seconds = 0.0
    edge_movement = np.empty(edges.n_edges, dtype=np.float64)
    edge_centre = np.empty(edges.n_edges, dtype=np.float64)
    edge_scale = np.empty(edges.n_edges, dtype=np.float64)
    device_values = None
    if keep_full:
        device_values, h2d_seconds = _transfer_to_device(cp, values, dtype)
    peak_used = int(cp.get_default_memory_pool().used_bytes())
    for edge_start in range(0, edges.n_edges, int(edge_chunk_size)):
        edge_stop = min(edge_start + int(edge_chunk_size), edges.n_edges)
        source = cp.asarray(edges.source[edge_start:edge_stop])
        target = cp.asarray(edges.target[edge_start:edge_stop])
        if keep_full:
            def distances_full(
                device_values=device_values,
                source=source,
                target=target,
            ):
                delta = (
                    device_values[:, source, :]
                    - device_values[:, target, :]
                )
                return cp.sum(delta * delta, axis=2)

            distances, elapsed = _event_seconds(cp, distances_full)
            compute_seconds += elapsed
        else:
            distances = cp.empty(
                (values.shape[0], edge_stop - edge_start),
                dtype=dtype,
            )
            for frame_start in range(
                0,
                values.shape[0],
                int(frame_chunk_size),
            ):
                frame_stop = min(
                    frame_start + int(frame_chunk_size),
                    values.shape[0],
                )
                device_chunk, elapsed = _transfer_to_device(
                    cp,
                    values[frame_start:frame_stop],
                    dtype,
                )
                h2d_seconds += elapsed

                def chunk_distances(
                    device_chunk=device_chunk,
                    source=source,
                    target=target,
                    distances=distances,
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                ):
                    delta = (
                        device_chunk[:, source, :]
                        - device_chunk[:, target, :]
                    )
                    distances[frame_start:frame_stop] = cp.sum(
                        delta * delta,
                        axis=2,
                    )

                _, elapsed = _event_seconds(cp, chunk_distances)
                compute_seconds += elapsed
                del device_chunk

        def robust_stats(distances=distances):
            centre = cp.median(distances, axis=0)
            scale = 1.4826 * cp.median(
                cp.abs(distances - centre[None, :]),
                axis=0,
            )
            movement = scale / (cp.abs(centre) + 1e-12)
            return movement, centre, scale

        stats, elapsed = _event_seconds(cp, robust_stats)
        compute_seconds += elapsed
        host_stats = [cp.asnumpy(item) for item in stats]
        cp.cuda.Device().synchronize()
        h2d_seconds += 0.0
        edge_movement[edge_start:edge_stop] = host_stats[0]
        edge_centre[edge_start:edge_stop] = host_stats[1]
        edge_scale[edge_start:edge_stop] = host_stats[2]
        peak_used = max(
            peak_used,
            int(cp.get_default_memory_pool().used_bytes()),
        )
        del distances, source, target
    if device_values is not None:
        del device_values
    cp.get_default_memory_pool().free_all_blocks()
    movement, support = aggregate_edge_movement(edge_movement, edges)
    weights, percentiles = map_movement_to_stability(
        movement,
        mapping="rank",
    )
    weights = smooth_weights_chain_aware(
        weights,
        edges.continuity_breaks,
        strength=0.05,
    )
    labels = stable_labels_from_weights(weights, core_fraction=0.6)
    maximum_support = max(1, int(np.max(support)))
    uncertainty = 1.0 - support.astype(np.float64) / maximum_support
    scan = WindowScanResult(
        movement_score=movement,
        movement_percentile=percentiles,
        stability_weights=weights,
        candidate_stable_label=labels,
        support_count=support,
        support_uncertainty=uncertainty,
        edge_movement=edge_movement,
        edge_centre=edge_centre,
        edge_scale=edge_scale,
        edges=edges,
        statistic="relative_mad",
        mapping="rank",
        timing={
            "H2D": h2d_seconds,
            "GPU_compute": compute_seconds,
            "total_scan_only": time.perf_counter() - total_start,
        },
        metadata={
            "output_contract": "scan_only",
            "backend": "gpu",
            "movement_score_semantics": (
                "alignment-free squared-distance instability; not RMSD"
            ),
            "peak_GPU_memory_bytes": peak_used,
            "full_coordinates_resident": keep_full,
            "frame_chunk_size": int(frame_chunk_size),
            "edge_chunk_size": int(edge_chunk_size),
        },
    )
    segments = detect_flexible_segments(
        movement,
        percentile_cutoff=0.8,
        min_segment_length=3,
        merge_gap=1,
        chain=chains,
    )
    return LocalDeformationResult(
        local_deformation_index=movement,
        local_rigidity_score=weights,
        locally_deforming_segments=segments,
        confidence=1.0 - uncertainty,
        temporal_block_reproducibility=float("nan"),
        temporal_block_segment_jaccard=float("nan"),
        block_count=0,
        scan=scan,
        timing={
            "full_local_scan_seconds": scan.timing["total_scan_only"],
            "temporal_block_reproducibility_seconds": 0.0,
            "total_seconds": time.perf_counter() - total_start,
        },
        metadata={
            "status": "experimental",
            "backend": "gpu",
            "output_contract": "alignment_free_local_kinematics",
            "descriptor": "W5_relative_MAD_squared_pair_distance",
            "translation_invariant": True,
            "rotation_invariant": True,
            "scale_normalised": True,
            "score_called_rmsd": False,
            "hard_motion_domains_mature": False,
        },
    )


def solve_contract_gpu(
    coords: np.ndarray,
    *,
    contract: str,
    chain_ids: np.ndarray | None,
    store_aligned: bool,
    optional_soft_graph: bool,
    **kwargs: Any,
):
    """GPU dispatch imported lazily by the contract layer."""

    from .contracts import (
        ContractExecutionResult,
        resolve_contract_parameters,
        run_fixed_cost_sampled_audit,
    )

    kwargs = resolve_contract_parameters(contract, kwargs)
    if contract == "exact":
        raise ValueError(
            "The exact sequential_lean contract is CPU-only; choose "
            "--backend cpu or use throughput/balanced-lite on GPU."
        )
    if contract == "local-deformation":
        local_kwargs = {
            key: kwargs[key]
            for key in (
                "residue_positions",
                "precision",
                "frame_chunk_size",
                "edge_chunk_size",
            )
            if key in kwargs
        }
        local = scan_w5_gpu(
            coords,
            chain_ids=chain_ids,
            **local_kwargs,
        )
        return ContractExecutionResult(
            contract=contract,
            backend_requested="gpu",
            backend_used="gpu",
            alignment=None,
            local_deformation=local,
            motion_graph=None,
            quality_status="local_diagnostic",
            quality_warning=(
                "W5 is alignment-free and is not RMSD or a mature hard-domain "
                "annotation."
            ),
            sampled_audit=None,
            refinement_performed=False,
            total_runtime_seconds=local.timing["total_seconds"],
            timing=local.timing,
            metadata={
                "hard_motion_domains_mature": False,
                "optional_soft_graph_requested_but_not_gpu_ported": (
                    optional_soft_graph
                ),
            },
        )
    alignment_keys = {
        "representative_count",
        "update_rounds",
        "chunk_size",
        "precision",
        "rotation_solver",
        "delta",
        "sequence_continuity_strength",
    }
    allowed = alignment_keys | {"compute_ordinary_rmsf"}
    if contract == "balanced-lite":
        allowed |= {"audit_subset_count", "audit_heldout_count", "allow_limited_refinement",
                    "refinement_representative_count"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"GPU {contract} does not support options: {sorted(unknown)}")
    alignment_kwargs = {
        key: kwargs[key] for key in alignment_keys if key in kwargs
    }
    alignment = solve_fast_global_gpu(
        coords,
        chain_ids=chain_ids,
        store_aligned=store_aligned,
        **alignment_kwargs,
    )
    if not kwargs.get("compute_ordinary_rmsf", True):
        alignment.ordinary_rmsf = None
    if contract == "throughput":
        stable = (
            float(alignment.risk_signals["last_two_round_core_jaccard"])
            >= 0.90
        )
        return ContractExecutionResult(
            contract=contract,
            backend_requested="gpu",
            backend_used="gpu",
            alignment=alignment,
            local_deformation=None,
            motion_graph=None,
            quality_status=(
                "completed"
                if stable
                else "moderate_confidence"
            ),
            quality_warning=(
                None
                if stable
                else "Run balanced-lite or CPU exact verification."
            ),
            sampled_audit=None,
            refinement_performed=False,
            total_runtime_seconds=alignment.total_runtime_seconds,
            timing=alignment.timing,
            metadata={
                "automatic_exact_fallback": False,
                "backend": "gpu",
            },
        )
    if contract != "balanced-lite":
        raise ValueError(f"unsupported GPU contract: {contract}")
    subset_count = int(kwargs.get("audit_subset_count", 32))
    heldout_count = int(kwargs.get("audit_heldout_count", 64))
    level1_seconds = alignment.total_runtime_seconds
    audit = run_fixed_cost_sampled_audit(
        coords,
        alignment.frozen_weights,
        chain_ids=chain_ids,
        subset_count=subset_count,
        heldout_count=heldout_count,
        delta=kwargs["delta"],
        sequence_continuity_strength=kwargs["sequence_continuity_strength"],
    )
    refinement_performed = False
    audit_seconds = audit.runtime_seconds
    refinement_seconds = 0.0
    if (
        bool(kwargs.get("allow_limited_refinement", True))
        and audit.quality_status != "high_confidence"
    ):
        start = time.perf_counter()
        alignment = solve_fast_global_gpu(
            coords,
            chain_ids=chain_ids,
            representative_count=min(
                int(kwargs.get("refinement_representative_count", 512)),
                np.asarray(coords).shape[0],
            ),
            update_rounds=int(kwargs.get("update_rounds", 4)),
            delta=kwargs["delta"],
            sequence_continuity_strength=kwargs["sequence_continuity_strength"],
            chunk_size=int(kwargs.get("chunk_size", 2000)),
            precision=str(kwargs.get("precision", "float32")),
            rotation_solver=str(kwargs.get("rotation_solver", "svd")),
            store_aligned=store_aligned,
        )
        refinement_seconds = time.perf_counter() - start
        refinement_performed = True
        audit = run_fixed_cost_sampled_audit(
            coords,
            alignment.frozen_weights,
            chain_ids=chain_ids,
            subset_count=subset_count,
            heldout_count=heldout_count,
            delta=kwargs["delta"],
            sequence_continuity_strength=kwargs["sequence_continuity_strength"],
        )
        audit_seconds += audit.runtime_seconds
    total = level1_seconds + audit_seconds + refinement_seconds
    if not kwargs.get("compute_ordinary_rmsf", True):
        alignment.ordinary_rmsf = None
    return ContractExecutionResult(
        contract=contract,
        backend_requested="gpu",
        backend_used="gpu",
        alignment=alignment,
        local_deformation=None,
        motion_graph=None,
        quality_status=audit.quality_status,
        quality_warning=audit.warning,
        sampled_audit=audit,
        refinement_performed=refinement_performed,
        total_runtime_seconds=total,
        timing={
            **alignment.timing,
            "level1_GPU_throughput": level1_seconds,
            "sampled_audit_CPU": audit_seconds,
            "limited_GPU_refinement": refinement_seconds,
            "full_exact_fallback": 0.0,
            "total_runtime_seconds": total,
        },
        metadata={
            "automatic_exact_fallback": False,
            "maximum_refinement_stages": 1,
            "sampled_audit_backend": "cpu",
        },
    )


@dataclass(slots=True)
class PersistentGPUWorker:
    """Warm CUDA-context worker for repeated throughput calls."""

    chain_ids: np.ndarray | None
    default_kwargs: dict[str, object]
    _workspace: _PersistentGPUWorkspace
    _representative_index_cache: dict[tuple[int, int], np.ndarray]
    _completed_tasks: int = 0

    @classmethod
    def create(
        cls,
        *,
        chain_ids: np.ndarray | None = None,
        **default_kwargs: object,
    ) -> "PersistentGPUWorker":
        cp = _cupy()
        cp.zeros(1, dtype=cp.float32)
        cp.cuda.Device().synchronize()
        return cls(
            chain_ids=(
                None if chain_ids is None else np.asarray(chain_ids)
            ),
            default_kwargs=dict(default_kwargs),
            _workspace=_PersistentGPUWorkspace(cp),
            _representative_index_cache={},
        )

    def throughput(self, coords: np.ndarray, **kwargs: object):
        options = {**self.default_kwargs, **kwargs}
        options.setdefault("release_memory_pool", False)
        values = np.asarray(coords)
        representative_count = min(
            int(options.get("representative_count", 256)),
            values.shape[0],
        )
        key = (int(values.shape[0]), representative_count)
        selected = self._representative_index_cache.get(key)
        indices_reused = selected is not None
        if selected is None:
            selected = np.unique(
                np.linspace(
                    0,
                    values.shape[0] - 1,
                    representative_count,
                    dtype=np.int64,
                )
            )
            self._representative_index_cache[key] = selected
        result = solve_fast_global_gpu(
            values,
            chain_ids=self.chain_ids,
            selected_frame_indices=selected,
            _workspace=self._workspace,
            **options,
        )
        sanitation_start = time.perf_counter()
        self._workspace.clear_trajectory_data()
        sanitation_seconds = time.perf_counter() - sanitation_start
        result.total_runtime_seconds += sanitation_seconds
        result.timing["post_task_buffer_sanitization"] = sanitation_seconds
        result.timing["total_runtime_seconds"] = result.total_runtime_seconds
        result.metadata["persistent_worker_task"] = True
        result.metadata["trajectory_data_retained_after_task"] = False
        result.metadata["representative_indices_reused"] = indices_reused
        self._completed_tasks += 1
        return result

    @property
    def cache_summary(self) -> dict[str, object]:
        return {
            "CUDA_context_reused": True,
            "CuPy_memory_pool_reused": True,
            "thread_pools_reused": True,
            "pinned_host_buffers_reused": True,
            "device_transfer_buffers_reused": True,
            "CUDA_streams_reused": True,
            "covariance_scratch_reused": True,
            "residual_scratch_reused": True,
            "representative_indices_cached": True,
            "trajectory_data_retained_between_jobs": False,
            "completed_tasks": self._completed_tasks,
            "workspace_allocation_generation": (
                self._workspace.allocation_generation
            ),
            "double_buffering_available_per_call": True,
        }
