"""Stable-core mask detection from continuous residue scores."""

from __future__ import annotations

import numpy as np


def mask_from_quantile(scores: np.ndarray, quantile: float) -> np.ndarray:
    """Create a boolean mask from a score quantile."""

    threshold = float(np.quantile(scores, quantile))
    return scores >= threshold


def smooth_boolean_mask(mask: np.ndarray, min_run_length: int = 2) -> np.ndarray:
    """Remove isolated residues and preserve short contiguous regions."""

    smoothed = mask.copy()
    start = 0
    while start < mask.size:
        value = mask[start]
        end = start + 1
        while end < mask.size and mask[end] == value:
            end += 1
        if value and (end - start) < min_run_length:
            smoothed[start:end] = False
        start = end
    return smoothed


def smooth_residue_signal(values: np.ndarray, radius: int = 1) -> np.ndarray:
    """Smooth residue-wise values with a short moving-average kernel."""

    if radius <= 0:
        return values.copy()
    kernel = np.ones(2 * radius + 1, dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(values, pad_width=radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply_terminal_penalty(values: np.ndarray, terminal_exclusion: int = 1) -> np.ndarray:
    """Reduce terminal residue scores to avoid overcalling flexible ends."""

    penalized = values.copy()
    if terminal_exclusion <= 0 or penalized.size == 0:
        return penalized
    width = min(terminal_exclusion, penalized.size // 2)
    if width > 0:
        penalized[:width] *= 0.8
        penalized[-width:] *= 0.8
    return penalized


def detect_candidate_core(scores: np.ndarray, quantile: float) -> np.ndarray:
    """Build the initial candidate core mask."""

    return smooth_boolean_mask(mask_from_quantile(scores, quantile))


def detect_final_core(
    scores: np.ndarray,
    quantile: float,
    confidence: np.ndarray | None = None,
    min_run_length: int = 2,
    confidence_threshold: float = 0.0,
    confidence_weight: float = 0.0,
    terminal_exclusion: int = 1,
    smoothing_radius: int = 1,
) -> np.ndarray:
    """Build the final refined core mask from score and confidence."""

    signal = scores.copy()
    if confidence is not None:
        signal = (1.0 - confidence_weight) * signal + confidence_weight * confidence
    signal = smooth_residue_signal(signal, radius=smoothing_radius)
    signal = apply_terminal_penalty(signal, terminal_exclusion=terminal_exclusion)
    mask = mask_from_quantile(signal, quantile)
    if confidence is not None and confidence_threshold > 0.0:
        mask &= confidence >= confidence_threshold
    return smooth_boolean_mask(mask, min_run_length=min_run_length)


def uncertain_residues(scores: np.ndarray, quantile: float, margin: float) -> np.ndarray:
    """Mark residues close to the decision threshold as uncertain."""

    threshold = float(np.quantile(scores, quantile))
    return np.abs(scores - threshold) <= margin
