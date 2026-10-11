"""Explicit experimental interfaces for fast-global and local diagnostics.

The exact sequential robust solver remains the public default.  This module
provides opt-in contracts whose approximation status is carried in every
result and output row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mdsuperpose.window_descriptors import (
    SparseEdgeSet,
    build_local_window_edges,
    validate_coordinates,
)
from mdsuperpose.window_kinematics import (
    LocalDeformationResult,
    MotionDomainResult,
    scan_local_kinematics,
    scan_motion_domains,
)
from mdsuperpose.window_refine import (
    InitialWeightResult,
    bounded_global_weight_refinement,
    solve_window_preconditioned_refinement,
    temporally_stratified_indices,
)
from mdsuperpose.window_scan import (
    WindowScanResult,
    percentile_ranks,
    scan_sparse_edges,
)

EXPERIMENTAL_WARNING = (
    "Experimental output: validate against exact sequential_lean for the "
    "intended system before scientific interpretation."
)
MOTION_GRAPH_WARNING = (
    "Sparse motion-graph outputs are soft exploratory diagnostics. Hard "
    "partitions are not mature biological-domain annotations."
)


@dataclass(frozen=True, slots=True)
class FastGlobalConfidence:
    """Internal agreement diagnostics for one fixed-budget fast-global run."""

    temporal_block_weight_spearman: float | None
    temporal_block_core_jaccard: float | None
    representative_subset_weight_spearman: float | None
    representative_subset_core_jaccard: float | None
    effective_core_size: float
    final_round_weight_change: float
    representative_converged_fraction: float
    convergence_status: str
    warning_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON/table representation."""

        return {
            "temporal_block_weight_spearman": self.temporal_block_weight_spearman,
            "temporal_block_core_jaccard": self.temporal_block_core_jaccard,
            "representative_subset_weight_spearman": (
                self.representative_subset_weight_spearman
            ),
            "representative_subset_core_jaccard": (
                self.representative_subset_core_jaccard
            ),
            "effective_core_size": self.effective_core_size,
            "final_round_weight_change": self.final_round_weight_change,
            "representative_converged_fraction": (
                self.representative_converged_fraction
            ),
            "convergence_status": self.convergence_status,
            "warning_flags": list(self.warning_flags),
        }


@dataclass(slots=True)
class FastGlobalAlignmentResult:
    """Approximate shared-weight estimation followed by exact all-frame fits."""

    reference_mode: str
    warm_start: bool
    mode: str
    selected_frame_indices: np.ndarray
    selection_strategy: str
    selection_metadata: dict[str, object]
    max_weight_fit_frames: int
    frozen_weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    aligned_coords: np.ndarray | None
    robust_rmsf: np.ndarray
    ordinary_rmsf: np.ndarray | None
    mean_residual: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    total_runtime_seconds: float
    timing: dict[str, float]
    chunk_size: int
    chunks_processed: int
    update_rounds: int
    round_weight_changes: np.ndarray
    representative_converged_counts: np.ndarray
    confidence: FastGlobalConfidence
    metadata: dict[str, object]
    local_deformation: LocalDeformationResult | None = None
    motion_graph: MotionDomainResult | None = None
    risk_signals: dict[str, object] | None = None

    @property
    def n_frames(self) -> int:
        """Return the number of fitted frames."""

        return int(self.rmsd.size)

    @property
    def n_atoms(self) -> int:
        """Return the number of fitted atoms or residues."""

        return int(self.frozen_weights.size)

    @property
    def mean_rmsd(self) -> float:
        """Return mean post-fit full-selection RMSD."""

        return float(np.mean(self.rmsd))

    @property
    def mean_weighted_rmsd(self) -> float:
        """Return mean weighted post-fit RMSD."""

        return float(np.mean(self.weighted_rmsd))

    @property
    def average_runtime_per_frame(self) -> float:
        """Return total runtime divided by frame count."""

        return self.total_runtime_seconds / max(1, self.n_frames)

    @property
    def average_iterations_per_frame(self) -> float:
        """Return the fixed global-update budget, not per-frame IRLS iterations."""

        return float(self.update_rounds)

    @property
    def converged_fraction(self) -> float:
        """Return measured representative-level tolerance fraction."""

        if self.representative_converged_counts.size == 0:
            return 0.0
        return float(
            self.representative_converged_counts[-1]
            / max(1, self.selected_frame_indices.size)
        )


@dataclass(slots=True)
class ExperimentalBatchWorker:
    """Persistent experimental worker for same-shape trajectory batches.

    The worker reuses imported modules/BLAS pools, chain metadata, W5 edges,
    uniform weights and deterministic representative indices. Current NumPy
    Kabsch and descriptor kernels still allocate method-local temporary arrays;
    this class does not claim a zero-allocation execution path.
    """

    n_residues: int
    chain_ids: np.ndarray
    residue_positions: np.ndarray
    w5_edges: SparseEdgeSet
    uniform_weights: np.ndarray
    representative_index_cache: dict[tuple[int, int], np.ndarray]

    @classmethod
    def create(
        cls,
        n_residues: int,
        *,
        chain_ids: np.ndarray | None = None,
        residue_positions: np.ndarray | None = None,
    ) -> "ExperimentalBatchWorker":
        """Create a worker and cache topology-dependent W5 quantities."""

        n_residues = int(n_residues)
        chains = (
            np.asarray([""] * n_residues, dtype=object)
            if chain_ids is None
            else np.asarray(chain_ids, dtype=object)
        )
        positions = (
            np.arange(1, n_residues + 1, dtype=np.int64)
            if residue_positions is None
            else np.asarray(residue_positions, dtype=np.int64)
        )
        if chains.shape != (n_residues,) or positions.shape != (n_residues,):
            raise ValueError(
                "chain_ids and residue_positions must match n_residues"
            )
        edges = build_local_window_edges(
            n_residues,
            window_size=5,
            feature_set="minimal",
            chain_ids=chains,
            residue_positions=positions,
        )
        return cls(
            n_residues=n_residues,
            chain_ids=chains,
            residue_positions=positions,
            w5_edges=edges,
            uniform_weights=np.ones(n_residues, dtype=np.float64),
            representative_index_cache={},
        )

    def representative_indices(
        self, n_frames: int, representative_count: int
    ) -> np.ndarray:
        """Return one cached deterministic representative-index vector."""

        key = (int(n_frames), int(representative_count))
        if key not in self.representative_index_cache:
            self.representative_index_cache[key] = (
                temporally_stratified_indices(*key)
            )
        return self.representative_index_cache[key]

    def scan_w5(
        self,
        coords: np.ndarray,
        *,
        descriptor_chunk_size: int = 2000,
    ) -> WindowScanResult:
        """Run W5 using cached topology edges."""

        values = validate_coordinates(coords)
        if values.shape[1] != self.n_residues:
            raise ValueError("trajectory residue count does not match worker")
        return scan_sparse_edges(
            values,
            self.w5_edges,
            statistic="relative_mad",
            mapping="rank",
            descriptor_chunk_size=descriptor_chunk_size,
        )

    @property
    def cache_summary(self) -> dict[str, object]:
        """Describe exactly which quantities are and are not reused."""

        return {
            "status": "experimental",
            "reused_imports_and_blas_thread_pools": True,
            "reused_chain_metadata": True,
            "reused_w5_edges": True,
            "reused_uniform_weights": True,
            "representative_index_cache_entries": len(
                self.representative_index_cache
            ),
            "reused_trajectory_buffers": False,
            "reused_kernel_temporary_arrays": False,
            "zero_allocation_claimed": False,
        }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute deterministic rank correlation with flat-vector handling."""

    left_rank = percentile_ranks(np.asarray(left, dtype=np.float64))
    right_rank = percentile_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 1e-15 or np.std(right_rank) <= 1e-15:
        return 1.0 if np.allclose(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _core_mask(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """Return a deterministic top-fraction core mask."""

    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return Boolean-set Jaccard with an empty-set convention of one."""

    union = np.count_nonzero(left | right)
    return (
        1.0
        if union == 0
        else float(np.count_nonzero(left & right) / union)
    )


def _estimate_weights(
    coords: np.ndarray,
    indices: np.ndarray,
    *,
    reference: np.ndarray,
    update_rounds: int,
    chain_ids: np.ndarray | None,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
) -> np.ndarray:
    """Estimate one independent fixed-budget shared-weight vector."""

    initial = np.ones(coords.shape[1], dtype=np.float64)
    weights, _, _, _ = bounded_global_weight_refinement(
        coords[indices],
        reference,
        initial,
        update_rounds=update_rounds,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
        sequence_chain_ids=chain_ids,
    )
    return weights


def _confidence_summary(
    coords: np.ndarray,
    selected_indices: np.ndarray,
    final_weights: np.ndarray,
    round_weight_changes: np.ndarray,
    converged_counts: np.ndarray,
    *,
    update_rounds: int,
    chain_ids: np.ndarray | None,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
) -> FastGlobalConfidence:
    """Compare independent temporal and representative subsets at equal budget."""

    warnings: list[str] = []
    temporal_spearman: float | None = None
    temporal_jaccard: float | None = None
    subset_spearman: float | None = None
    subset_jaccard: float | None = None
    reference = coords[0]

    midpoint = coords.shape[0] // 2
    if midpoint >= 2 and coords.shape[0] - midpoint >= 2:
        per_block = max(2, min(selected_indices.size // 2, midpoint))
        first_indices = temporally_stratified_indices(midpoint, per_block)
        second_local = temporally_stratified_indices(
            coords.shape[0] - midpoint, per_block
        )
        first = _estimate_weights(
            coords,
            first_indices,
            reference=reference,
            update_rounds=update_rounds,
            chain_ids=chain_ids,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
        )
        second = _estimate_weights(
            coords,
            second_local + midpoint,
            reference=reference,
            update_rounds=update_rounds,
            chain_ids=chain_ids,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
        )
        temporal_spearman = _spearman(first, second)
        temporal_jaccard = _jaccard(_core_mask(first), _core_mask(second))
    else:
        warnings.append("temporal_block_agreement_unavailable_too_few_frames")

    left_indices = selected_indices[::2]
    right_indices = selected_indices[1::2]
    if left_indices.size >= 2 and right_indices.size >= 2:
        left = _estimate_weights(
            coords,
            left_indices,
            reference=reference,
            update_rounds=update_rounds,
            chain_ids=chain_ids,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
        )
        right = _estimate_weights(
            coords,
            right_indices,
            reference=reference,
            update_rounds=update_rounds,
            chain_ids=chain_ids,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
        )
        subset_spearman = _spearman(left, right)
        subset_jaccard = _jaccard(_core_mask(left), _core_mask(right))
    else:
        warnings.append(
            "representative_subset_agreement_unavailable_too_few_frames"
        )

    final_change = (
        float(round_weight_changes[-1])
        if round_weight_changes.size
        else float("nan")
    )
    representative_fraction = (
        float(converged_counts[-1] / max(1, selected_indices.size))
        if converged_counts.size
        else 0.0
    )
    convergence_status = (
        "fixed_budget_stable"
        if np.isfinite(final_change) and final_change <= 1e-3
        else "fixed_budget_complete_not_converged"
    )
    if convergence_status != "fixed_budget_stable":
        warnings.append("fixed_update_budget_ended_before_weight_stability")
    if temporal_spearman is not None and temporal_spearman < 0.9:
        warnings.append("low_temporal_block_weight_agreement")
    if subset_spearman is not None and subset_spearman < 0.9:
        warnings.append("low_representative_subset_weight_agreement")

    return FastGlobalConfidence(
        temporal_block_weight_spearman=temporal_spearman,
        temporal_block_core_jaccard=temporal_jaccard,
        representative_subset_weight_spearman=subset_spearman,
        representative_subset_core_jaccard=subset_jaccard,
        effective_core_size=float(
            np.sum(final_weights) / max(float(np.max(final_weights)), 1e-12)
        ),
        final_round_weight_change=final_change,
        representative_converged_fraction=representative_fraction,
        convergence_status=convergence_status,
        warning_flags=tuple(warnings),
    )


def _build_online_risk_signals(
    refinement,
    trace: dict[str, np.ndarray],
    chain_ids: np.ndarray | None,
) -> dict[str, object]:
    """Summarize cheap signals available without an exact comparison."""

    weights = np.asarray(refinement.final_weights, dtype=np.float64)
    objectives = np.asarray(
        trace["weighted_objective_history"],
        dtype=np.float64,
    )
    l1_changes = np.asarray(
        trace["round_weight_l1_changes"],
        dtype=np.float64,
    )
    mean_l1_changes = np.asarray(
        trace["round_weight_mean_l1_changes"],
        dtype=np.float64,
    )
    linf_changes = np.asarray(
        trace["round_weight_linf_changes"],
        dtype=np.float64,
    )
    core_jaccards = np.asarray(
        trace["round_core_jaccards"],
        dtype=np.float64,
    )
    core = _core_mask(weights)
    order = np.lexsort((np.arange(weights.size), -weights))
    core_count = int(np.count_nonzero(core))
    cutoff = float(weights[order[core_count - 1]])
    next_weight = (
        float(weights[order[core_count]])
        if core_count < weights.size
        else cutoff
    )
    near_threshold = np.abs(weights - cutoff) <= 0.02
    residual = np.asarray(
        refinement.fit.mean_residual,
        dtype=np.float64,
    )
    frame_rmsd = np.asarray(refinement.fit.rmsd, dtype=np.float64)
    chains = (
        np.asarray([""] * weights.size, dtype=object)
        if chain_ids is None
        else np.asarray(chain_ids, dtype=object)
    )
    transitions = (
        (core[1:] != core[:-1])
        & (chains[1:] == chains[:-1])
    )
    objective_reduction = (
        float((objectives[0] - objectives[-1]) / objectives[0])
        if objectives.size >= 2 and objectives[0] > 0.0
        else 0.0
    )
    last_objective_change = (
        float(objectives[-1] - objectives[-2])
        if objectives.size >= 2
        else 0.0
    )
    return {
        "weighted_objective_history": objectives.tolist(),
        "objective_reduction_fraction": objective_reduction,
        "last_round_objective_change": last_objective_change,
        "last_round_relative_objective_change": (
            abs(last_objective_change) / max(abs(float(objectives[-2])), 1e-12)
            if objectives.size >= 2
            else 0.0
        ),
        "round_weight_l1_changes": l1_changes.tolist(),
        "round_weight_mean_l1_changes": mean_l1_changes.tolist(),
        "round_weight_linf_changes": linf_changes.tolist(),
        "final_weight_l1_change": float(l1_changes[-1]),
        "final_weight_mean_l1_change": float(mean_l1_changes[-1]),
        "final_weight_linf_change": float(linf_changes[-1]),
        "round_core_jaccards": core_jaccards.tolist(),
        "last_two_round_core_jaccard": float(core_jaccards[-1]),
        "core_threshold_weight": cutoff,
        "core_threshold_gap": cutoff - next_weight,
        "near_core_threshold_count": int(np.count_nonzero(near_threshold)),
        "near_core_threshold_fraction": float(np.mean(near_threshold)),
        "mean_residual_median": float(np.median(residual)),
        "mean_residual_p95": float(np.quantile(residual, 0.95)),
        "mean_residual_max": float(np.max(residual)),
        "mean_residual_p95_median_ratio": float(
            np.quantile(residual, 0.95)
            / max(float(np.median(residual)), 1e-12)
        ),
        "frame_rmsd_median": float(np.median(frame_rmsd)),
        "frame_rmsd_p95": float(np.quantile(frame_rmsd, 0.95)),
        "frame_rmsd_p95_median_ratio": float(
            np.quantile(frame_rmsd, 0.95)
            / max(float(np.median(frame_rmsd)), 1e-12)
        ),
        "effective_core_size": float(refinement.effective_core_size),
        "effective_core_fraction": float(
            refinement.effective_core_size / weights.size
        ),
        "core_transition_count": int(np.count_nonzero(transitions)),
        "core_fragmentation_fraction": float(
            np.count_nonzero(transitions) / max(1, weights.size - 1)
        ),
    }


def solve_fast_global_alignment(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    representative_indices: np.ndarray | None = None,
    update_rounds: int = 4,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
    chunk_size: int = 2000,
    store_aligned: bool = True,
    compute_confidence: bool = True,
    collect_risk_signals: bool = False,
    compute_ordinary_rmsf: bool = True,
    representative_precision: str = "float64",
) -> FastGlobalAlignmentResult:
    """Run fixed-budget weights plus exact all-frame fits.

    ``representative_count`` is the representative-frame budget ``K`` and
    ``update_rounds`` is the robust weight-update budget ``M``. Throughput
    approximates weight estimation only; every frame still receives an exact
    weighted rigid fit after the weights are frozen.
    """

    total_start = time.perf_counter()
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    chains = None if chain_ids is None else np.asarray(chain_ids)
    if chains is not None and chains.shape != (values.shape[1],):
        raise ValueError("chain_ids must have shape (n_residues,)")

    initial = InitialWeightResult(
        weights=np.ones(values.shape[1], dtype=np.float64),
        initializer="uniform",
        scan_seconds=0.0,
        metadata={
            "output_contract": "experimental_fast_global",
            "initializer": "uniform",
        },
    )
    refinement_trace: dict[str, np.ndarray] | None = (
        {} if collect_risk_signals else None
    )
    refinement = solve_window_preconditioned_refinement(
        values,
        initial,
        representative_count=representative_count,
        update_rounds=update_rounds,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
        chain_ids=chains,
        chunk_size=chunk_size,
        refinement_trace=refinement_trace,
        representative_indices=representative_indices,
        refinement_precision=representative_precision,
        return_aligned=store_aligned,
        compute_ordinary_rmsf=compute_ordinary_rmsf,
    )
    confidence_start = time.perf_counter()
    if compute_confidence:
        confidence = _confidence_summary(
            values,
            refinement.representative_indices,
            refinement.final_weights,
            refinement.round_weight_changes,
            refinement.converged_representative_counts,
            update_rounds=update_rounds,
            chain_ids=chains,
            delta=delta,
            sequence_continuity_strength=sequence_continuity_strength,
        )
    else:
        confidence = FastGlobalConfidence(
            temporal_block_weight_spearman=None,
            temporal_block_core_jaccard=None,
            representative_subset_weight_spearman=None,
            representative_subset_core_jaccard=None,
            effective_core_size=refinement.effective_core_size,
            final_round_weight_change=float(
                refinement.round_weight_changes[-1]
            ),
            representative_converged_fraction=float(
                refinement.converged_representative_counts[-1]
                / max(1, refinement.representative_indices.size)
            ),
            convergence_status="not_evaluated",
            warning_flags=("confidence_diagnostics_disabled",),
        )
    confidence_seconds = time.perf_counter() - confidence_start
    aligned = refinement.fit.aligned_coords
    materialize_seconds = refinement.fit.timing[
        "aligned_output_materialization"
    ]
    total_seconds = time.perf_counter() - total_start
    risk_signals = (
        _build_online_risk_signals(
            refinement,
            refinement_trace,
            chains,
        )
        if refinement_trace is not None
        else None
    )
    timing = {
        **refinement.timing,
        "confidence_diagnostics": confidence_seconds,
        "aligned_coordinate_materialization": materialize_seconds,
        "total_runtime_seconds": total_seconds,
    }
    return FastGlobalAlignmentResult(
        reference_mode="first_frame",
        warm_start=False,
        mode="fast-global",
        selected_frame_indices=refinement.representative_indices,
        selection_strategy=(
            "uniform_temporally_stratified"
            if representative_indices is None
            else "caller_supplied_locked_indices"
        ),
        selection_metadata={
            "deterministic": True,
            "indices_reused": representative_indices is not None,
        },
        max_weight_fit_frames=int(
            representative_count
            if representative_indices is None
            else np.asarray(representative_indices).size
        ),
        frozen_weights=refinement.final_weights,
        rotations=refinement.fit.rotations,
        translations=refinement.fit.translations,
        aligned_coords=aligned,
        robust_rmsf=refinement.fit.robust_rmsf,
        ordinary_rmsf=refinement.fit.ordinary_rmsf,
        mean_residual=refinement.fit.mean_residual,
        rmsd=refinement.fit.rmsd,
        weighted_rmsd=refinement.fit.weighted_rmsd,
        total_runtime_seconds=total_seconds,
        timing=timing,
        chunk_size=int(chunk_size),
        chunks_processed=int(np.ceil(values.shape[0] / int(chunk_size))),
        update_rounds=int(update_rounds),
        round_weight_changes=refinement.round_weight_changes,
        representative_converged_counts=(
            refinement.converged_representative_counts
        ),
        confidence=confidence,
        metadata={
            "status": "experimental",
            "approximate_weight_estimation": True,
            "all_frame_transforms_exact_for_fixed_weights": True,
            "automatic_fallback": False,
            "transform_identity_with_exact_claimed": False,
            "output_contract": "metrics_transforms_rmsd_rmsf",
            "ordinary_rmsf_computed": compute_ordinary_rmsf,
            "representative_precision": representative_precision,
            "effective_parameters": {
                "delta": float(delta), "rule": "huber",
                "sequence_continuity_strength": float(sequence_continuity_strength),
                "warm_start": False, "update_rounds": int(update_rounds),
            },
            "warning": EXPERIMENTAL_WARNING,
        },
        risk_signals=risk_signals,
    )


def solve_diagnostic_rich_alignment(
    coords: np.ndarray,
    *,
    chain_ids: np.ndarray | None = None,
    representative_count: int = 256,
    update_rounds: int = 4,
    delta: float = 1.0,
    sequence_continuity_strength: float = 0.05,
    chunk_size: int = 2000,
    descriptor_chunk_size: int = 2000,
    block_count: int = 4,
    experimental_motion_graph: bool = False,
    compute_ordinary_rmsf: bool = True,
) -> FastGlobalAlignmentResult:
    """Run fast-global plus W5 and optional soft sparse-graph diagnostics."""

    result = solve_fast_global_alignment(
        coords,
        chain_ids=chain_ids,
        representative_count=representative_count,
        update_rounds=update_rounds,
        delta=delta,
        sequence_continuity_strength=sequence_continuity_strength,
        chunk_size=chunk_size,
        store_aligned=True,
        compute_confidence=True,
        compute_ordinary_rmsf=compute_ordinary_rmsf,
    )
    local = scan_local_kinematics(
        coords,
        chain_ids=chain_ids,
        block_count=block_count,
        descriptor_chunk_size=descriptor_chunk_size,
    )
    graph = (
        scan_motion_domains(
            coords,
            chain_ids=chain_ids,
            block_count=block_count,
            descriptor_chunk_size=descriptor_chunk_size,
        )
        if experimental_motion_graph
        else None
    )
    result.mode = "diagnostic-rich"
    result.local_deformation = local
    result.motion_graph = graph
    result.total_runtime_seconds += local.timing["total_seconds"]
    if graph is not None:
        # The graph recomputes its own W5 scan; expose the measured cost exactly.
        result.total_runtime_seconds += graph.timing["total_seconds"]
    result.timing["local_deformation"] = local.timing["total_seconds"]
    result.timing["motion_graph"] = (
        graph.timing["total_seconds"] if graph is not None else 0.0
    )
    result.timing["total_runtime_seconds"] = result.total_runtime_seconds
    result.metadata = {
        **result.metadata,
        "output_contract": "metrics_transforms_rmsd_rmsf_plus_local_diagnostics",
        "diagnostic_rich_not_fastest_mode": True,
        "motion_graph_warning": MOTION_GRAPH_WARNING if graph is not None else None,
    }
    return result


def local_deformation_rows(
    result: LocalDeformationResult,
    *,
    residue_ids: np.ndarray,
    residue_names: np.ndarray,
    chain_ids: np.ndarray,
    residue_positions: np.ndarray,
    source_provenance: dict[str, object],
) -> list[dict[str, object]]:
    """Build the public alignment-free per-residue output contract."""

    n_residues = result.local_deformation_index.size
    arrays = [
        np.asarray(residue_ids),
        np.asarray(residue_names),
        np.asarray(chain_ids),
        np.asarray(residue_positions),
    ]
    if any(values.shape != (n_residues,) for values in arrays):
        raise ValueError("residue metadata must match the W5 residue count")
    percentiles = percentile_ranks(result.local_deformation_index)
    order = np.lexsort(
        (np.arange(n_residues), -result.local_deformation_index)
    )
    ranks = np.empty(n_residues, dtype=np.int64)
    ranks[order] = np.arange(1, n_residues + 1)
    chain_break_before = np.zeros(n_residues, dtype=bool)
    chain_break_before[1:] = (
        (arrays[2][1:] != arrays[2][:-1])
        | (arrays[3][1:] != arrays[3][:-1] + 1)
    )
    terminal = np.zeros(n_residues, dtype=bool)
    terminal[0] = terminal[-1] = True
    terminal[:-1] |= arrays[2][:-1] != arrays[2][1:]
    terminal[1:] |= arrays[2][1:] != arrays[2][:-1]
    uncertainty = 1.0 - np.clip(result.confidence, 0.0, 1.0)
    return [
        {
            "atom_index": index + 1,
            "residue_id": str(arrays[0][index]),
            "resname": str(arrays[1][index]),
            "chain": str(arrays[2][index]),
            "residue_position": int(arrays[3][index]),
            "local_deformation_index": float(
                result.local_deformation_index[index]
            ),
            "local_rigidity_score": float(
                result.local_rigidity_score[index]
            ),
            "percentile": float(percentiles[index]),
            "rank": int(ranks[index]),
            "locally_deforming_segment_id": int(
                result.locally_deforming_segments[index]
            ),
            "is_terminal": int(terminal[index]),
            "chain_break_before": int(chain_break_before[index]),
            "temporal_block_reproducibility": float(
                result.temporal_block_reproducibility
            ),
            "confidence": float(result.confidence[index]),
            "uncertainty": float(uncertainty[index]),
            "source_unit": source_provenance.get("source_unit"),
            "conversion_factor_to_angstrom": source_provenance.get(
                "conversion_factor_to_angstrom"
            ),
            "canonical_internal_unit": source_provenance.get(
                "canonical_internal_unit", "angstrom"
            ),
        }
        for index in range(n_residues)
    ]


def motion_graph_rows(
    result: MotionDomainResult,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return soft edge and residue diagnostics for the experimental graph."""

    edge_percentiles = percentile_ranks(result.edge_temporal_instability)
    edge_rows = [
        {
            "source_index": int(source) + 1,
            "target_index": int(target) + 1,
            "edge_family": str(family),
            "edge_instability": float(instability),
            "edge_co_motion_affinity": float(1.0 - percentile),
            "experimental_warning": MOTION_GRAPH_WARNING,
        }
        for source, target, family, instability, percentile in zip(
            result.edges.source,
            result.edges.target,
            result.edges.family,
            result.edge_temporal_instability,
            edge_percentiles,
            strict=True,
        )
    ]
    residue_rows = [
        {
            "atom_index": index + 1,
            "hinge_boundary_probability": float(
                result.hinge_boundary_score[index]
            ),
            "local_consensus_confidence": float(
                result.domain_confidence[index]
            ),
            "inter_domain_edge_instability": float(
                result.inter_domain_edge_instability[index]
            ),
            "exploratory_motion_partition_id": int(
                result.motion_domain_id[index]
            ),
            "experimental_warning": MOTION_GRAPH_WARNING,
        }
        for index in range(result.motion_domain_id.size)
    ]
    return edge_rows, residue_rows
