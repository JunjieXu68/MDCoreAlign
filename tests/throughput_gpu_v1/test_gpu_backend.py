"""Optional CUDA tests for throughput alignment and W5 diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import (
    PersistentGPUWorker,
    gpu_available,
    scan_w5_gpu,
    solve_fast_global_gpu,
)
from mdsuperpose.window_kinematics import scan_local_kinematics

pytestmark = pytest.mark.skipif(
    not gpu_available(),
    reason="CuPy CUDA device is not available",
)


def _trajectory(
    n_frames: int = 120,
    n_residues: int = 60,
) -> np.ndarray:
    rng = np.random.default_rng(20260731)
    reference = np.cumsum(rng.normal(size=(n_residues, 3)), axis=0)
    coords = np.repeat(reference[None, :, :], n_frames, axis=0)
    coords += rng.normal(scale=0.015, size=coords.shape)
    mobile_start = max(3, n_residues // 3)
    mobile_stop = min(n_residues, mobile_start + max(4, n_residues // 5))
    coords[:, mobile_start:mobile_stop] += rng.normal(
        scale=0.22,
        size=(n_frames, mobile_stop - mobile_start, 3),
    )
    return coords


@pytest.mark.parametrize("rotation_solver", ["svd", "horn"])
def test_gpu_float64_matches_cpu_throughput(rotation_solver: str) -> None:
    coords = _trajectory()
    cpu = solve_fast_global_alignment(
        coords,
        representative_count=48,
        update_rounds=4,
        chunk_size=120,
        store_aligned=False,
        compute_confidence=False,
    )
    gpu = solve_fast_global_gpu(
        coords,
        representative_count=48,
        update_rounds=4,
        chunk_size=120,
        precision="float64",
        rotation_solver=rotation_solver,
    )

    np.testing.assert_allclose(gpu.frozen_weights, cpu.frozen_weights, atol=1e-12)
    np.testing.assert_allclose(gpu.rmsd, cpu.rmsd, atol=1e-11)
    np.testing.assert_allclose(gpu.robust_rmsf, cpu.robust_rmsf, atol=1e-10)
    assert gpu.metadata["explicit_CUDA_synchronization"] is True


@pytest.mark.parametrize("rotation_solver", ["svd", "horn"])
def test_gpu_mixed_precision_preserves_scientific_outputs(
    rotation_solver: str,
) -> None:
    coords = _trajectory()
    reference = solve_fast_global_gpu(
        coords,
        representative_count=48,
        update_rounds=4,
        chunk_size=120,
        precision="float64",
        rotation_solver=rotation_solver,
    )
    mixed = solve_fast_global_gpu(
        coords,
        representative_count=48,
        update_rounds=4,
        chunk_size=120,
        precision="mixed",
        rotation_solver=rotation_solver,
    )

    np.testing.assert_allclose(
        mixed.frozen_weights,
        reference.frozen_weights,
        atol=2e-6,
    )
    np.testing.assert_allclose(mixed.rmsd, reference.rmsd, atol=2e-6)
    np.testing.assert_allclose(
        mixed.robust_rmsf,
        reference.robust_rmsf,
        atol=2e-6,
    )


def test_gpu_w5_matches_cpu_float64() -> None:
    coords = _trajectory()
    chains = np.asarray(["A"] * 30 + ["B"] * 30)
    positions = np.concatenate((np.arange(1, 31), np.arange(1, 31)))
    cpu = scan_local_kinematics(
        coords,
        chain_ids=chains,
        residue_positions=positions,
        evaluate_blocks=False,
    )
    gpu = scan_w5_gpu(
        coords,
        chain_ids=chains,
        residue_positions=positions,
        precision="float64",
    )

    np.testing.assert_allclose(
        gpu.local_deformation_index,
        cpu.local_deformation_index,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gpu.local_rigidity_score,
        cpu.local_rigidity_score,
        atol=1e-12,
    )
    assert gpu.metadata["score_called_rmsd"] is False


def test_persistent_gpu_worker_reports_reuse_limits() -> None:
    worker = PersistentGPUWorker.create(representative_count=32)
    result = worker.throughput(
        _trajectory(60, 30),
        chunk_size=30,
        precision="mixed",
        rotation_solver="horn",
    )

    assert result.rmsd.shape == (60,)
    assert worker.cache_summary["CUDA_context_reused"] is True
    assert worker.cache_summary["pinned_host_buffers_reused"] is True
    assert worker.cache_summary["CUDA_streams_reused"] is True
    assert worker.cache_summary["covariance_scratch_reused"] is True
    assert worker.cache_summary["residual_scratch_reused"] is True
    assert (
        worker.cache_summary["trajectory_data_retained_between_jobs"] is False
    )
    assert worker.cache_summary["double_buffering_available_per_call"] is True
    assert result.metadata["pinned_host_memory"] is True
    assert result.metadata["double_buffering"] is True
    assert result.metadata["persistent_workspace_reused"] is False
    assert result.metadata["trajectory_data_retained_after_task"] is False

    second = worker.throughput(
        _trajectory(60, 30),
        chunk_size=30,
        precision="mixed",
        rotation_solver="horn",
    )
    assert second.metadata["persistent_workspace_reused"] is True
    assert second.metadata["representative_indices_reused"] is True
    assert worker.cache_summary["workspace_allocation_generation"] == 1
