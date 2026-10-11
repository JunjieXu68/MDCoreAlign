"""Long-trajectory optimized robust alignment utilities.

The long mode keeps the scientific stable-core interpretation explicit:

1. estimate soft stable-core weights from representative frames;
2. freeze those weights;
3. apply weighted Kabsch to every frame in chunks;
4. accumulate RMSF and residual summaries without requiring full trajectory
   storage for the analysis statistics.

This is an approximate long-trajectory mode, not a replacement for the default
per-frame iterative solver on short or highly heterogeneous trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .all_frame_engine import fit_all_frames_fixed_weights
from .core_selection import select_core_by_target_fraction
from .fast_kabsch import apply_transforms_batch
from .flexibility import (
    FlexibilityProfile,
    descending_ranks,
    detect_flexible_segments,
    normalize_residue_metadata,
    percentile_scores,
)
from .representative_selection import (
    select_representative_frame_indices as _select_simple_representative_frame_indices,
    select_representative_frames,
)
from .representative_weights import (
    RepresentativeWeightEstimationResult,
    estimate_representative_weights,
)
from .sequential_weights import (
    SequentialMeanWeightEstimationResult,
    SequentialWeightEstimationResult,
    estimate_representative_weights_sequential_mean,
)
from .trajectory import Trajectory
from .trajectory_mode import TrajectoryAlignmentResult, solve_trajectory_robust_alignment
from .utils import ensure_xyz_frames


@dataclass(slots=True)
class StreamingRMSFAccumulator:
    """Numerically stable chunked RMSF accumulator."""

    n_atoms: int
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n_atoms <= 0:
            raise ValueError("n_atoms must be positive")
        self.mean = np.zeros((self.n_atoms, 3), dtype=float)
        self.m2 = np.zeros((self.n_atoms, 3), dtype=float)

    def update(self, coords: np.ndarray) -> None:
        """Add a chunk of aligned coordinates with shape `(n_frames, n_atoms, 3)`."""

        coords = ensure_xyz_frames(coords)
        if coords.shape[1] != self.n_atoms:
            raise ValueError("coords atom count does not match the accumulator")
        chunk_count = int(coords.shape[0])
        if chunk_count == 0:
            return
        assert self.mean is not None
        assert self.m2 is not None
        chunk_mean = np.mean(coords, axis=0)
        centered = coords - chunk_mean[None, :, :]
        chunk_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            self.mean[...] = chunk_mean
            self.m2[...] = chunk_m2
            self.count = chunk_count
            return
        total_count = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean[...] = self.mean + delta * (chunk_count / total_count)
        self.m2[...] = self.m2 + chunk_m2 + delta * delta * (self.count * chunk_count / total_count)
        self.count = total_count

    def rmsf(self) -> np.ndarray:
        """Return per-atom RMSF from all accumulated frames."""

        if self.count <= 0:
            raise ValueError("no coordinates have been accumulated")
        assert self.m2 is not None
        variance = self.m2 / self.count
        return np.sqrt(np.sum(variance, axis=1))


@dataclass(slots=True)
class LongTrajectoryAlignmentResult:
    """Output from long-mode robust trajectory alignment."""

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
    subset_result: (
        TrajectoryAlignmentResult
        | RepresentativeWeightEstimationResult
        | SequentialWeightEstimationResult
        | SequentialMeanWeightEstimationResult
    )
    total_runtime_seconds: float
    timing: dict[str, float]
    chunk_size: int
    chunks_processed: int
    representative_weight_estimator: str = "legacy"

    @property
    def n_frames(self) -> int:
        """Return the number of processed trajectory frames."""

        return int(self.rmsd.shape[0])

    @property
    def n_atoms(self) -> int:
        """Return the number of fitted atoms."""

        return int(self.frozen_weights.shape[0])

    @property
    def average_iterations_per_frame(self) -> float:
        """Return the mean iterations over representative weight-fit frames."""

        return self.subset_result.average_iterations_per_frame

    @property
    def average_runtime_per_frame(self) -> float:
        """Return total long-mode runtime divided by full trajectory frames."""

        if self.n_frames == 0:
            return 0.0
        return float(self.total_runtime_seconds / self.n_frames)

    @property
    def mean_rmsd(self) -> float:
        """Return mean unified post-hoc RMSD over all frames."""

        return float(np.mean(self.rmsd))

    @property
    def mean_weighted_rmsd(self) -> float:
        """Return mean weighted RMSD over all frames."""

        return float(np.mean(self.weighted_rmsd))

    @property
    def converged_fraction(self) -> float:
        """Return convergence fraction on representative weight-fit frames."""

        return self.subset_result.converged_fraction


def select_representative_frame_indices(
    n_frames: int,
    *,
    max_frames: int = 2000,
    strategy: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """Select deterministic representative frames for long-mode weight fitting."""

    return _select_simple_representative_frame_indices(
        n_frames,
        max_frames=max_frames,
        strategy=strategy,
        seed=seed,
    )


def _mean_soft_weights_subset(result: TrajectoryAlignmentResult) -> np.ndarray:
    """Average soft weights from a representative-frame trajectory result."""

    if not result.frame_results:
        raise ValueError("representative-frame result contains no frames")
    return np.mean(np.asarray([frame.result.weights for frame in result.frame_results], dtype=float), axis=0)


def solve_long_trajectory_alignment(
    trajectory: Trajectory | np.ndarray,
    *,
    reference_mode: str = "first_frame",
    mode: str = "long",
    max_weight_fit_frames: int = 2000,
    frame_selection: str = "uniform",
    random_seed: int = 42,
    chunk_size: int = 2000,
    store_aligned: bool = False,
    compute_ordinary_rmsf: bool = True,
    raw_descriptor_size: int = 2048,
    compressed_descriptor_dim: int = 128,
    candidate_pool_size: int = 5000,
    excluded_representative_frame_indices: np.ndarray | None = None,
    representative_weight_estimator: str = "legacy",
    representative_batch_size: int = 2000,
    representative_chain_ids: np.ndarray | None = None,
    representative_warm_refinement_passes: int = 0,
    **robust_kwargs: object,
) -> LongTrajectoryAlignmentResult:
    """Run long-mode robust alignment over a trajectory.

    Long mode currently uses `first_frame` reference semantics. It estimates
    stable-core weights on representative frames, then freezes those weights
    while aligning all frames. This preserves full-trajectory RMSF while
    avoiding iterative robust reweighting on every frame.
    """

    total_start = time.perf_counter()
    input_start = time.perf_counter()
    trajectory = trajectory if isinstance(trajectory, Trajectory) else Trajectory(coords=trajectory)
    coords = ensure_xyz_frames(trajectory.coords)
    input_seconds = time.perf_counter() - input_start
    if reference_mode != "first_frame":
        raise ValueError("long mode currently supports reference_mode='first_frame' only")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    selection_start = time.perf_counter()
    selection = select_representative_frames(
        coords,
        max_frames=max_weight_fit_frames,
        strategy=frame_selection,
        seed=random_seed,
        raw_descriptor_size=raw_descriptor_size,
        compressed_dim=compressed_descriptor_dim,
        candidate_pool_size=candidate_pool_size,
        excluded_frame_indices=excluded_representative_frame_indices,
    )
    selected = selection.indices
    reference = coords[0]
    selection_seconds = time.perf_counter() - selection_start

    weight_start = time.perf_counter()
    if representative_weight_estimator in {"legacy", "legacy_object"}:
        subset = Trajectory(coords=coords[selected])
        subset_result = solve_trajectory_robust_alignment(
            subset,
            reference_mode="first_frame",
            warm_start=True,
            **robust_kwargs,
        )
        frozen_weights = np.clip(_mean_soft_weights_subset(subset_result), 0.0, None)
    elif representative_weight_estimator == "sequential_lean":
        subset_result = estimate_representative_weights_sequential_mean(
            coords[selected],
            reference,
            sequence_chain_ids=representative_chain_ids,
            **robust_kwargs,
        )
        frozen_weights = np.clip(subset_result.mean_weights, 0.0, None)
    elif representative_weight_estimator in {"batched", "batched_independent"}:
        subset_result = estimate_representative_weights(
            coords[selected],
            reference,
            batch_size=representative_batch_size,
            sequence_chain_ids=representative_chain_ids,
            warm_start_refinement_passes=representative_warm_refinement_passes,
            **robust_kwargs,
        )
        frozen_weights = np.clip(subset_result.mean_weights, 0.0, None)
    else:
        raise ValueError(
            "representative_weight_estimator must be 'legacy', "
            "'sequential_lean', or 'batched'"
        )
    if float(np.sum(frozen_weights)) <= 0.0:
        raise RuntimeError("long-mode weight estimation produced all-zero weights")
    weight_seconds = time.perf_counter() - weight_start
    representative_timing = getattr(subset_result, "timing", None) or {}
    selection_metadata = dict(selection.metadata)
    selection_metadata["representative_weight_estimator"] = (
        representative_weight_estimator
    )
    if representative_weight_estimator in {
        "batched",
        "batched_independent",
        "sequential_lean",
    }:
        selection_metadata["representative_call_counts"] = dict(
            subset_result.call_counts
        )
    if representative_weight_estimator in {"batched", "batched_independent"}:
        selection_metadata["representative_batch_size"] = int(
            representative_batch_size
        )
        selection_metadata["representative_warm_refinement_passes"] = int(
            representative_warm_refinement_passes
        )

    all_frame = fit_all_frames_fixed_weights(
        coords,
        reference,
        frozen_weights,
        chunk_size=chunk_size,
        return_aligned=store_aligned,
        compute_rmsd=True,
        compute_weighted_rmsd=True,
        compute_mean_residual=True,
        compute_robust_rmsf=True,
        compute_ordinary_rmsf=compute_ordinary_rmsf,
        validate_inputs=False,
    )
    assert all_frame.rmsd is not None
    assert all_frame.weighted_rmsd is not None
    assert all_frame.robust_rmsf is not None
    assert all_frame.mean_residual is not None
    total_runtime = time.perf_counter() - total_start
    timing = {
        "input_load": input_seconds,
        "initialization": selection_seconds,
        "representative_selection": selection_seconds,
        "iterative_weight_estimation": weight_seconds,
        "representative_batched_kabsch": float(
            representative_timing.get("batched_kabsch", 0.0)
        ),
        "representative_residual_compute": float(
            representative_timing.get("residual_compute", 0.0)
        ),
        "representative_weight_update": float(
            representative_timing.get("weight_update", 0.0)
        ),
        "representative_sequence_regularization": float(
            representative_timing.get(
                "sequence_regularization",
                0.0,
            )
        ),
        "weighted_fit": all_frame.timing["all_frame_weighted_fit"],
        "ordinary_fit": all_frame.timing["all_frame_ordinary_fit"],
        "transform_application": all_frame.timing["transform_application"],
        "residual_update": all_frame.timing["RMSD"],
        "RMSF": all_frame.timing["RMSF"],
        "flexible_segment_detection": 0.0,
        "profile_table_build": 0.0,
        "output_write": 0.0,
        "unified_RMSD_eval": 0.0,
        "total_runtime_seconds": total_runtime,
    }
    return LongTrajectoryAlignmentResult(
        reference_mode=reference_mode,
        warm_start=True,
        mode=mode,
        selected_frame_indices=selected,
        selection_strategy=frame_selection,
        selection_metadata=selection_metadata,
        max_weight_fit_frames=int(max_weight_fit_frames),
        frozen_weights=frozen_weights,
        rotations=all_frame.rotations,
        translations=all_frame.translations,
        aligned_coords=all_frame.aligned_coords,
        robust_rmsf=all_frame.robust_rmsf,
        ordinary_rmsf=all_frame.ordinary_rmsf,
        mean_residual=all_frame.mean_residual,
        rmsd=all_frame.rmsd,
        weighted_rmsd=all_frame.weighted_rmsd,
        subset_result=subset_result,
        total_runtime_seconds=total_runtime,
        timing=timing,
        chunk_size=chunk_size,
        chunks_processed=all_frame.chunks_processed,
        representative_weight_estimator=representative_weight_estimator,
    )


def aligned_coords_from_long_result(result: LongTrajectoryAlignmentResult, coords: np.ndarray) -> np.ndarray:
    """Apply long-mode frame transforms to a coordinate trajectory."""

    coords = ensure_xyz_frames(coords)
    if coords.shape[0] != result.n_frames:
        raise ValueError("coords frame count does not match long-mode result")
    return apply_transforms_batch(coords, result.rotations, result.translations)


def build_long_flexibility_profile(
    result: LongTrajectoryAlignmentResult,
    *,
    residue_labels: list[str] | tuple[str, ...] | None = None,
    flexible_percentile_cutoff: float = 0.8,
    min_segment_length: int = 3,
    merge_gap: int = 0,
    mobility_score_mode: str = "soft_weight",
) -> FlexibilityProfile:
    """Build a flexibility profile from long-mode streaming RMSF outputs."""

    profile_start = time.perf_counter()
    residue_id, resname, chain = normalize_residue_metadata(residue_labels, n_atoms=result.n_atoms)
    stability = np.clip(result.frozen_weights, 0.0, 1.0)
    if mobility_score_mode == "soft_weight":
        mobility_score = np.clip(1.0 - stability, 0.0, 1.0)
    elif mobility_score_mode == "robust_rmsf":
        low = float(np.min(result.robust_rmsf))
        high = float(np.max(result.robust_rmsf))
        mobility_score = np.zeros(result.n_atoms, dtype=float) if np.isclose(low, high) else (result.robust_rmsf - low) / (high - low)
    else:
        raise ValueError("mobility_score_mode must be 'soft_weight' or 'robust_rmsf'")
    segment_start = time.perf_counter()
    segment_ids = detect_flexible_segments(
        mobility_score,
        percentile_cutoff=flexible_percentile_cutoff,
        min_segment_length=min_segment_length,
        merge_gap=merge_gap,
        chain=chain,
    )
    result.timing["flexible_segment_detection"] += time.perf_counter() - segment_start
    percentiles = percentile_scores(mobility_score)
    ranks = descending_ranks(mobility_score)
    result.timing["profile_table_build"] += time.perf_counter() - profile_start
    return FlexibilityProfile(
        residue_id=residue_id,
        resname=resname,
        chain=chain,
        robust_aligned_rmsf=result.robust_rmsf,
        ordinary_aligned_rmsf=result.ordinary_rmsf,
        soft_stability_weight=stability,
        mobility_score=mobility_score,
        flexible_segment_id=segment_ids,
        rank=ranks,
        percentile=percentiles,
    )


def long_mode_core_mask(result: LongTrajectoryAlignmentResult, *, core_fraction: float = 0.6) -> np.ndarray:
    """Return the hard stable-core mask from frozen long-mode weights."""

    return select_core_by_target_fraction(result.frozen_weights, core_fraction).core_mask
