"""Experimental guarded cascade for approximate fast-global alignment.

The exact sequential solver remains the public default. This module implements
the frozen guarded-v1 policy developed on the corrected mdCATH panel. It is
opt-in because the Level-1 certificate detects the predefined catastrophic
tail but does not certify strict output parity for every trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np

from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_kinematics import scan_local_kinematics
from mdsuperpose.window_refine import (
    exact_all_frame_fit,
    temporally_stratified_indices,
)

from .experimental import (
    FastGlobalAlignmentResult,
    _spearman,
    solve_fast_global_alignment,
)

GUARDED_EXPERIMENTAL_WARNING = (
    "Experimental guarded output: the frozen certificate corrected all "
    "predefined catastrophic tails but did not establish strict exact-output "
    "parity for every accepted Level-1 trajectory."
)

# These values were frozen before the G3 held-out Level-2 evaluation.
EFFECTIVE_CORE_FRACTION_CUTOFF = 0.4999966410394734
TEMPORAL_WEIGHT_SPEARMAN_CUTOFF = 0.9
W5_DISAGREEMENT_CUTOFF = 0.7646292738144553


@dataclass(slots=True)
class GuardedFastGlobalResult:
    """Unified result and audit trail from the experimental guarded cascade."""

    accepted_level: int
    mode: str
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
    fallback_reason: str | None
    final_quality_status: str
    metadata: dict[str, object]

    @property
    def mean_rmsd(self) -> float:
        """Return mean full-selection post-fit RMSD."""

        return float(np.mean(self.rmsd))

    @property
    def mean_weighted_rmsd(self) -> float:
        """Return mean fixed-weight post-fit RMSD."""

        return float(np.mean(self.weighted_rmsd))


def _run_sequential_exact(
    coords: np.ndarray,
    chain_ids: np.ndarray | None,
    *,
    representative_count: int,
    chunk_size: int,
    store_aligned: bool,
    selected_indices: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run the exact sequential-weight path and fixed-weight all-frame fit."""

    selected = (
        temporally_stratified_indices(
            coords.shape[0],
            min(representative_count, coords.shape[0]),
        )
        if selected_indices is None
        else np.asarray(selected_indices, dtype=np.int64)
    )
    estimate_start = time.perf_counter()
    estimate = estimate_representative_weights_sequential_lean(
        coords[selected],
        coords[0],
        rule="huber",
        delta=1.0,
        max_iterations=20,
        tolerance=1e-6,
        objective_tolerance=1e-5,
        objective_patience=2,
        sequence_continuity_strength=0.05,
        contact_consistency_strength=0.0,
        sequence_chain_ids=chain_ids,
        record_history=False,
    )
    estimate_seconds = time.perf_counter() - estimate_start
    weights = np.asarray(estimate.mean_weights, dtype=np.float64)
    fit_start = time.perf_counter()
    fit = exact_all_frame_fit(
        coords,
        coords[0],
        weights,
        chunk_size=chunk_size,
    )
    fit_seconds = time.perf_counter() - fit_start
    materialize_start = time.perf_counter()
    aligned = (
        apply_transforms_batch(coords, fit.rotations, fit.translations)
        if store_aligned
        else None
    )
    materialize_seconds = time.perf_counter() - materialize_start
    return (
        {
            "weights": weights,
            "rotations": fit.rotations,
            "translations": fit.translations,
            "aligned": aligned,
            "robust_rmsf": fit.robust_rmsf,
            "ordinary_rmsf": fit.ordinary_rmsf,
            "mean_residual": fit.mean_residual,
            "rmsd": fit.rmsd,
            "weighted_rmsd": fit.weighted_rmsd,
        },
        {
            "level3_weight_estimation": estimate_seconds,
            "level3_all_frame_fit": fit_seconds,
            "level3_aligned_coordinate_materialization": materialize_seconds,
        },
    )


def _from_fast_result(
    result: FastGlobalAlignmentResult,
    *,
    total_runtime_seconds: float,
    timing: dict[str, float],
    risk_signals: dict[str, object],
) -> GuardedFastGlobalResult:
    """Build a guarded Level-1 result without changing numerical arrays."""

    return GuardedFastGlobalResult(
        accepted_level=1,
        mode="guarded-fast-global",
        frozen_weights=result.frozen_weights,
        rotations=result.rotations,
        translations=result.translations,
        aligned_coords=result.aligned_coords,
        robust_rmsf=result.robust_rmsf,
        ordinary_rmsf=cast(np.ndarray, result.ordinary_rmsf),
        mean_residual=result.mean_residual,
        rmsd=result.rmsd,
        weighted_rmsd=result.weighted_rmsd,
        total_runtime_seconds=total_runtime_seconds,
        timing=timing,
        risk_signals=risk_signals,
        fallback_reason=None,
        final_quality_status=(
            "level1_certificate_accepted_not_strict_exact_parity"
        ),
        metadata={
            "status": "experimental",
            "public_default_changed": False,
            "fixed_budget_convergence_status": (
                result.confidence.convergence_status
            ),
            "warning_flags": list(result.confidence.warning_flags),
            "warning": GUARDED_EXPERIMENTAL_WARNING,
        },
    )


def solve_guarded_fast_global_alignment(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    level1_representative_count: int = 256,
    level1_update_rounds: int = 4,
    level2_representative_count: int = 1024,
    level2_update_rounds: int = 4,
    exact_representative_count: int = 2000,
    chunk_size: int = 2000,
    store_aligned: bool = False,
    _representative_index_cache: dict[str, np.ndarray] | None = None,
) -> GuardedFastGlobalResult:
    """Run the frozen three-level guarded-v1 cascade.

    Level 1 is the current low-budget fast-global solver. W5 is evaluated only
    when the cheaper effective-core signal is uncertain. A frozen temporal
    safety route catches the validation tail missed by the cheap certificate.
    Level 2 adds representatives, but guarded-v1 always escalates a routed case
    to exact Level 3 because development data did not support a safe Level-2
    acceptance rule.
    """

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if values.shape[0] < 1 or values.shape[1] < 3:
        raise ValueError("coords must contain at least one frame and 3 residues")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (values.shape[1],):
        raise ValueError("chain_ids must have shape (n_residues,)")

    level1 = solve_fast_global_alignment(
        values,
        chain_ids=chains,
        representative_count=level1_representative_count,
        representative_indices=(
            None
            if _representative_index_cache is None
            else _representative_index_cache["level1"]
        ),
        update_rounds=level1_update_rounds,
        chunk_size=chunk_size,
        store_aligned=store_aligned,
        compute_confidence=True,
        collect_risk_signals=True,
    )
    signals = dict(level1.risk_signals or {})
    effective_core_flag = (
        float(cast(float, signals["effective_core_fraction"]))
        <= EFFECTIVE_CORE_FRACTION_CUTOFF
    )
    temporal_spearman = (
        level1.confidence.temporal_block_weight_spearman
    )
    temporal_warning = (
        temporal_spearman is not None
        and temporal_spearman < TEMPORAL_WEIGHT_SPEARMAN_CUTOFF
    )
    timing = {
        "level1_fast_global": level1.total_runtime_seconds,
        "conditional_W5": 0.0,
        "level2_refinement": 0.0,
        "level3_weight_estimation": 0.0,
        "level3_all_frame_fit": 0.0,
        "level3_aligned_coordinate_materialization": 0.0,
    }

    w5_flag = False
    if effective_core_flag:
        w5_start = time.perf_counter()
        w5 = scan_local_kinematics(
            values,
            chain_ids=chains,
            evaluate_blocks=False,
            descriptor_chunk_size=chunk_size,
        )
        timing["conditional_W5"] = time.perf_counter() - w5_start
        w5_mobility_spearman = _spearman(
            w5.local_deformation_index,
            1.0 - level1.frozen_weights,
        )
        w5_disagreement = 1.0 - w5_mobility_spearman
        w5_flag = w5_disagreement <= W5_DISAGREEMENT_CUTOFF
        signals.update(
            {
                "conditional_W5_evaluated": True,
                "W5_mobility_spearman": w5_mobility_spearman,
                "W5_disagreement": w5_disagreement,
                "conditional_W5_flag": w5_flag,
            }
        )
    else:
        signals.update(
            {
                "conditional_W5_evaluated": False,
                "W5_mobility_spearman": None,
                "W5_disagreement": None,
                "conditional_W5_flag": False,
            }
        )
    route = temporal_warning or (effective_core_flag and w5_flag)
    signals.update(
        {
            "cheap_effective_core_flag": effective_core_flag,
            "temporal_safety_warning": temporal_warning,
            "route_to_level2": route,
        }
    )
    if not route:
        total = time.perf_counter() - total_start
        timing["total_runtime_seconds"] = total
        return _from_fast_result(
            level1,
            total_runtime_seconds=total,
            timing=timing,
            risk_signals=signals,
        )

    level2 = solve_fast_global_alignment(
        values,
        chain_ids=chains,
        representative_count=min(
            level2_representative_count,
            values.shape[0],
        ),
        representative_indices=(
            None
            if _representative_index_cache is None
            else _representative_index_cache["level2"]
        ),
        update_rounds=level2_update_rounds,
        chunk_size=chunk_size,
        store_aligned=False,
        compute_confidence=True,
        collect_risk_signals=True,
    )
    timing["level2_refinement"] = level2.total_runtime_seconds
    signals.update(
        {
            "level1_level2_weight_spearman": _spearman(
                level1.frozen_weights,
                level2.frozen_weights,
            ),
            "level1_level2_RMSF_spearman": _spearman(
                level1.robust_rmsf,
                level2.robust_rmsf,
            ),
            "level2_convergence_status": (
                level2.confidence.convergence_status
            ),
            "level2_returned_as_final": False,
        }
    )

    exact, exact_timing = _run_sequential_exact(
        values,
        chains,
        representative_count=exact_representative_count,
        chunk_size=chunk_size,
        store_aligned=store_aligned,
        selected_indices=(
            None
            if _representative_index_cache is None
            else _representative_index_cache["exact"]
        ),
    )
    timing.update(exact_timing)
    total = time.perf_counter() - total_start
    timing["total_runtime_seconds"] = total
    reasons: list[str] = []
    if temporal_warning:
        reasons.append("low_temporal_block_weight_agreement")
    if effective_core_flag and w5_flag:
        reasons.append("cheap_flag_confirmed_by_conditional_W5")
    reasons.append("level2_not_safely_certifiable_in_guarded_v1")
    return GuardedFastGlobalResult(
        accepted_level=3,
        mode="guarded-fast-global",
        frozen_weights=exact["weights"],
        rotations=exact["rotations"],
        translations=exact["translations"],
        aligned_coords=exact["aligned"],
        robust_rmsf=exact["robust_rmsf"],
        ordinary_rmsf=exact["ordinary_rmsf"],
        mean_residual=exact["mean_residual"],
        rmsd=exact["rmsd"],
        weighted_rmsd=exact["weighted_rmsd"],
        total_runtime_seconds=total,
        timing=timing,
        risk_signals=signals,
        fallback_reason=";".join(reasons),
        final_quality_status="sequential_lean_exact_fallback",
        metadata={
            "status": "experimental",
            "public_default_changed": False,
            "level2_executed": True,
            "exact_fallback_executed": True,
            "warning": GUARDED_EXPERIMENTAL_WARNING,
        },
    )


@dataclass(slots=True)
class GuardedBatchWorker:
    """Persistent same-shape worker for guarded trajectory batches.

    The worker reuses the coordinate input buffer, chain metadata and all
    deterministic representative-index arrays. Numerical Kabsch/SVD temporary
    arrays remain kernel-local and are intentionally not described as reused.
    """

    n_frames: int
    n_residues: int
    chain_ids: np.ndarray
    coordinate_buffer: np.ndarray
    representative_indices: dict[str, np.ndarray]

    @classmethod
    def create(
        cls,
        n_frames: int,
        n_residues: int,
        *,
        chain_ids: np.ndarray | None = None,
        level1_representative_count: int = 256,
        level2_representative_count: int = 1024,
        exact_representative_count: int = 2000,
    ) -> "GuardedBatchWorker":
        """Allocate persistent buffers for one fixed trajectory shape."""

        n_frames = int(n_frames)
        n_residues = int(n_residues)
        if n_frames < 1 or n_residues < 3:
            raise ValueError("worker requires at least one frame and 3 residues")
        chains = (
            np.asarray([""] * n_residues, dtype=object)
            if chain_ids is None
            else np.asarray(chain_ids, dtype=object)
        )
        if chains.shape != (n_residues,):
            raise ValueError("chain_ids must have shape (n_residues,)")
        return cls(
            n_frames=n_frames,
            n_residues=n_residues,
            chain_ids=chains,
            coordinate_buffer=np.empty(
                (n_frames, n_residues, 3),
                dtype=np.float64,
            ),
            representative_indices={
                "level1": temporally_stratified_indices(
                    n_frames,
                    min(level1_representative_count, n_frames),
                ),
                "level2": temporally_stratified_indices(
                    n_frames,
                    min(level2_representative_count, n_frames),
                ),
                "exact": temporally_stratified_indices(
                    n_frames,
                    min(exact_representative_count, n_frames),
                ),
            },
        )

    def align(
        self,
        coords: np.ndarray,
        *,
        store_aligned: bool = False,
    ) -> GuardedFastGlobalResult:
        """Copy one trajectory into the reusable buffer and run guarded-v1."""

        values = np.asarray(coords, dtype=np.float64)
        if values.shape != self.coordinate_buffer.shape:
            raise ValueError("trajectory shape does not match worker")
        np.copyto(self.coordinate_buffer, values)
        return solve_guarded_fast_global_alignment(
            self.coordinate_buffer,
            chain_ids=self.chain_ids,
            level1_representative_count=(
                self.representative_indices["level1"].size
            ),
            level2_representative_count=(
                self.representative_indices["level2"].size
            ),
            exact_representative_count=(
                self.representative_indices["exact"].size
            ),
            store_aligned=store_aligned,
            _representative_index_cache=self.representative_indices,
        )

    @property
    def cache_summary(self) -> dict[str, object]:
        """Describe reused and still-allocated state without overclaiming."""

        return {
            "reused_python_imports": True,
            "reused_thread_pools": True,
            "reused_chain_metadata": True,
            "reused_coordinate_input_buffer": True,
            "reused_representative_index_buffers": True,
            "reused_kabsch_temporary_arrays": False,
            "reused_svd_temporary_arrays": False,
            "zero_allocation_claimed": False,
        }
