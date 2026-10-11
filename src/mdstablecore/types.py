"""Core data structures for stable-core detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ArrayFloat = np.ndarray
ArrayInt = np.ndarray
ArrayBool = np.ndarray


@dataclass(slots=True)
class TrajectoryData:
    """Container for trajectory coordinates and residue mapping."""

    coords: ArrayFloat
    atom_to_residue: ArrayInt
    residue_ids: list[str]
    atom_names: list[str] | None = None
    frame_ids: ArrayInt | None = None

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[1])

    @property
    def n_residues(self) -> int:
        return len(self.residue_ids)


@dataclass(slots=True)
class ResidueScores:
    """Continuous residue-level scores produced by the workflow."""

    positional: ArrayFloat
    local_geometry: ArrayFloat
    contact_persistence: ArrayFloat
    temporal_consistency: ArrayFloat
    combined: ArrayFloat


@dataclass(slots=True)
class QualityReport:
    """Summary quality metrics for the final result."""

    coarse_refined_core_jaccard: float
    all_atom_rmsd_mean: float
    core_rmsd_mean: float
    subsampling_score_correlation: float
    confidence_mean: float
    selected_quantile: float | None = None
    core_fraction: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MDLovoFitStyleResult:
    """Approximate MDLovoFit-style fraction-based alignment result."""

    fraction: float
    selected_residue_mask: ArrayBool
    residue_rmsd: ArrayFloat
    residue_rmsf: ArrayFloat
    rmsd_low_mean: float
    rmsd_high_mean: float
    rmsd_all_mean: float


@dataclass(slots=True)
class CoreResult:
    """Final stable-core detection output."""

    residue_scores: ResidueScores
    confidence: ArrayFloat
    core_mask: ArrayBool
    uncertain_mask: ArrayBool
    candidate_core_mask: ArrayBool
    coarse_frame_indices: ArrayInt
    refined_reference_index: int
    quality: QualityReport


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the MVP coarse-to-fine pipeline."""

    algorithm: str = "fastcore"
    coarse_fraction: float = 0.2
    min_coarse_frames: int = 20
    max_coarse_frames: int = 200
    candidate_core_quantile: float = 0.65
    final_core_quantile: float = 0.7
    final_core_quantiles: tuple[float, ...] = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
    uncertainty_margin: float = 0.06
    bootstrap_rounds: int = 16
    bootstrap_fraction: float = 0.7
    bootstrap_block_fraction: float = 0.12
    local_window_radius: int = 2
    contact_cutoff: float = 8.0
    temporal_blocks: int = 4
    min_core_run_length: int = 2
    terminal_exclusion: int = 1
    confidence_threshold: float = 0.15
    confidence_weight: float = 0.2
    fastcore_seed_quantile: float = 0.45
    fastcore_quantiles: tuple[float, ...] = (0.38, 0.4, 0.43, 0.45, 0.48, 0.5, 0.53)
    fastcore_target_core_fraction: float = 0.55
    competitive_fractions: tuple[float, ...] = (0.55, 0.65)
    competitive_quantiles: tuple[float, ...] = (0.38, 0.4, 0.43, 0.45, 0.48, 0.5, 0.53)
    competitive_iterations: int = 3
    competitive_refinement_rounds: int = 1
    competitive_target_core_fraction: float = 0.55
    random_seed: int = 7
