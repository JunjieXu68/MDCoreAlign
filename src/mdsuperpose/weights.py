"""Soft-weight utilities for robust alignment and core estimation.

This module maps residual distances to non-negative atom weights. The current
rules are meant to be used later inside an IRLS-style loop, but they are kept
independent here so they can be tested and benchmarked in isolation.
"""

from __future__ import annotations

import numpy as np

from .utils import ensure_atom_vector


def uniform_weights(n_atoms: int) -> np.ndarray:
    """Return uniform per-atom weights."""

    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    return np.ones(n_atoms, dtype=float)


def clamp_weights(weights: np.ndarray, min_value: float = 0.0, max_value: float = 1.0) -> np.ndarray:
    """Clamp weights to a closed interval."""

    weights = ensure_atom_vector(weights, len(weights), name="weights")
    return np.clip(weights, min_value, max_value)


def _validate_distances(distances: np.ndarray) -> np.ndarray:
    """Validate a one-dimensional vector of residual distances."""

    distances = ensure_atom_vector(distances, len(distances), name="distances")
    if not np.isfinite(distances).all():
        raise ValueError("distances must contain only finite values")
    if np.any(distances < 0.0):
        raise ValueError("distances must be non-negative")
    return distances


def normalize_weights(weights: np.ndarray, mode: str = "sum") -> np.ndarray:
    """Normalize weights by sum or max while preserving relative scale."""

    weights = ensure_atom_vector(weights, len(weights), name="weights")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    if mode == "sum":
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("sum of weights must be positive")
        return weights / total
    if mode == "max":
        maximum = float(np.max(weights))
        if maximum <= 0.0:
            raise ValueError("max of weights must be positive")
        return weights / maximum
    if mode == "none":
        return weights.copy()
    raise ValueError("mode must be one of: 'sum', 'max', 'none'")


def rescale_weights(weights: np.ndarray, min_value: float = 0.0, max_value: float = 1.0) -> np.ndarray:
    """Linearly rescale a non-negative weight vector to a target interval."""

    weights = ensure_atom_vector(weights, len(weights), name="weights")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if max_value < min_value:
        raise ValueError("max_value must be greater than or equal to min_value")

    lower = float(np.min(weights))
    upper = float(np.max(weights))
    if np.isclose(upper, lower):
        return np.full(weights.shape, max_value, dtype=float)
    scaled = (weights - lower) / (upper - lower)
    return min_value + (max_value - min_value) * scaled


def apply_sequence_continuity_regularization(
    weights: np.ndarray,
    strength: float = 0.0,
    passes: int = 1,
) -> np.ndarray:
    """Smooth weights along sequence neighbors with a light local averaging rule.

    For interior atoms, one smoothing pass applies:

    `w_i <- (1 - lambda) w_i + 0.5 * lambda * (w_{i-1} + w_{i+1})`

    Endpoints use their single available neighbor. This is a cheap Jacobi-style
    smoothing step that approximately discourages large differences between
    adjacent sequence positions.
    """

    weights = ensure_atom_vector(weights, len(weights), name="weights")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if passes < 0:
        raise ValueError("passes must be non-negative")
    if weights.shape[0] <= 1 or strength == 0.0 or passes == 0:
        return weights.copy()

    smoothed = weights.copy()
    updated = weights.copy()
    for _ in range(passes):
        updated[:] = smoothed
        updated[0] = (1.0 - strength) * smoothed[0] + strength * smoothed[1]
        updated[-1] = (1.0 - strength) * smoothed[-1] + strength * smoothed[-2]
        if smoothed.shape[0] > 2:
            updated[1:-1] = (
                (1.0 - strength) * smoothed[1:-1]
                + 0.5 * strength * (smoothed[:-2] + smoothed[2:])
            )
        smoothed, updated = updated, smoothed
    return smoothed


def _gaussian_weights_raw(
    distances: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Compute Gaussian weights for validated distances."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return np.exp(-0.5 * (distances / sigma) ** 2)


def gaussian_weights(
    distances: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Compute Gaussian-decay weights from residual distances.

    Behavior:
    - weight is 1 at zero distance
    - decreases smoothly and strictly with distance
    - never becomes exactly zero for finite distance
    """

    distances = _validate_distances(distances)
    return _gaussian_weights_raw(distances, sigma=sigma)


def _huber_weights_raw(
    distances: np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    """Compute Huber weights for validated distances."""

    if delta <= 0.0:
        raise ValueError("delta must be positive")
    weights = np.ones_like(distances)
    mask = distances > delta
    weights[mask] = delta / distances[mask]
    return weights


def huber_weights(
    distances: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Compute Huber-style IRLS weights from residual distances.

    Behavior:
    - weight is 1 for inliers with distance <= delta
    - decays like delta / d for larger distances
    - never becomes negative
    """

    distances = _validate_distances(distances)
    return _huber_weights_raw(distances, delta=delta)


def _tukey_weights_raw(
    distances: np.ndarray,
    *,
    cutoff: float,
) -> np.ndarray:
    """Compute Tukey biweight weights for validated distances."""

    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    scaled = distances / cutoff
    weights = np.zeros_like(distances)
    mask = scaled < 1.0
    weights[mask] = (1.0 - scaled[mask] ** 2) ** 2
    return weights


def tukey_weights(
    distances: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Compute Tukey biweight weights from residual distances.

    Behavior:
    - weight is 1 at zero distance
    - decreases smoothly inside the cutoff
    - becomes exactly zero at and beyond the cutoff
    """

    distances = _validate_distances(distances)
    return _tukey_weights_raw(distances, cutoff=cutoff)


def update_irls_weights(
    distances: np.ndarray,
    rule: str = "gaussian",
    normalize: str = "none",
    validate: bool = True,
    **parameters: float,
) -> np.ndarray:
    """Dispatch to a configured robust weighting rule.

    Supported rules:
    - ``gaussian`` with parameter ``sigma``
    - ``huber`` with parameter ``delta``
    - ``tukey`` with parameter ``cutoff``
    """

    distances = _validate_distances(distances) if validate else np.asarray(distances, dtype=float)
    if rule == "gaussian":
        sigma = float(parameters.get("sigma", 1.0))
        weights = _gaussian_weights_raw(distances, sigma=sigma)
    elif rule == "huber":
        delta = float(parameters.get("delta", 1.0))
        weights = _huber_weights_raw(distances, delta=delta)
    elif rule == "tukey":
        cutoff = float(parameters.get("cutoff", 1.0))
        weights = _tukey_weights_raw(distances, cutoff=cutoff)
    else:
        raise ValueError("rule must be one of: 'gaussian', 'huber', 'tukey'")

    if normalize == "none":
        return weights
    return normalize_weights(weights, mode=normalize)
