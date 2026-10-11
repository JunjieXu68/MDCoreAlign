"""Confidence estimation via frame resampling."""

from __future__ import annotations

import numpy as np

from .core_detection import detect_final_core
from .scoring import score_residues


def _sample_block_indices(
    n_frames: int,
    sample_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample approximately contiguous frame blocks for MD-aware bootstrap."""

    starts = []
    while len(starts) * block_size < sample_size:
        starts.append(int(rng.integers(0, max(1, n_frames - block_size + 1))))

    indices = []
    for start in starts:
        stop = min(n_frames, start + block_size)
        indices.extend(range(start, stop))
        if len(indices) >= sample_size:
            break

    return np.array(sorted(indices[:sample_size]), dtype=int)


def bootstrap_confidence(
    residue_coords: np.ndarray,
    quantile: float,
    rounds: int = 16,
    fraction: float = 0.7,
    block_fraction: float = 0.12,
    local_window_radius: int = 2,
    contact_cutoff: float = 8.0,
    temporal_blocks: int = 4,
    min_run_length: int = 2,
    confidence_threshold: float = 0.0,
    confidence_weight: float = 0.0,
    terminal_exclusion: int = 1,
    random_seed: int = 7,
) -> np.ndarray:
    """Estimate residue confidence from block-bootstrap core inclusion frequency."""

    n_frames = residue_coords.shape[0]
    sample_size = max(4, int(round(n_frames * fraction)))
    block_size = max(2, int(round(n_frames * block_fraction)))
    rng = np.random.default_rng(random_seed)
    inclusion = np.zeros(residue_coords.shape[1], dtype=float)

    for _ in range(rounds):
        sample_indices = _sample_block_indices(
            n_frames=n_frames,
            sample_size=sample_size,
            block_size=block_size,
            rng=rng,
        )
        sampled = residue_coords[sample_indices]
        scores = score_residues(
            sampled,
            local_window_radius=local_window_radius,
            contact_cutoff=contact_cutoff,
            temporal_blocks=temporal_blocks,
        )
        inclusion += detect_final_core(
            scores.combined,
            quantile=quantile,
            confidence=None,
            min_run_length=min_run_length,
            confidence_threshold=confidence_threshold,
            confidence_weight=confidence_weight,
            terminal_exclusion=terminal_exclusion,
        ).astype(float)

    return inclusion / float(rounds)
