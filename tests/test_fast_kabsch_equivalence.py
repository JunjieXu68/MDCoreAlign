"""Regression tests for fast weighted Kabsch primitives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("MDCOREALIGN_TEST_LAYER", "product") != "product" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdcorealign.fast_kabsch import (
    apply_transforms_batch,
    weighted_kabsch_batch,
    weighted_kabsch_single,
)


def reference_weighted_kabsch(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Independent weighted Kabsch implementation for test comparison."""

    total = float(np.sum(weights))
    mobile_center = np.sum((weights / total)[:, None] * mobile, axis=0)
    reference_center = np.sum((weights / total)[:, None] * reference, axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = ((weights / total)[:, None] * mobile_centered).T @ reference_centered
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    correction = np.eye(3)
    correction[2, 2] = -1.0 if np.linalg.det(u @ vt) < 0.0 else 1.0
    row_operator = u @ correction @ vt
    rotation = row_operator.T
    translation = reference_center - mobile_center @ row_operator
    aligned = mobile @ row_operator + translation
    squared = np.sum((aligned - reference) ** 2, axis=1)
    weighted_rmsd = float(np.sqrt(np.sum(weights * squared) / total))
    return rotation, translation, aligned, weighted_rmsd


def random_rotation(theta: float = 0.37) -> np.ndarray:
    """Return a deterministic proper rotation matrix."""

    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_fast_single_matches_independent_reference() -> None:
    """The fast single-frame solver should match an independent Kabsch solve."""

    rng = np.random.default_rng(5)
    mobile = rng.normal(size=(18, 3))
    rotation = random_rotation()
    translation = np.array([0.4, -1.2, 0.7], dtype=float)
    reference = mobile @ rotation.T + translation
    reference[15:] += rng.normal(scale=0.03, size=(3, 3))
    weights = np.linspace(0.2, 2.5, mobile.shape[0])

    result = weighted_kabsch_single(mobile, reference, weights)
    ref_rotation, ref_translation, ref_aligned, ref_weighted_rmsd = reference_weighted_kabsch(
        mobile,
        reference,
        weights,
    )

    np.testing.assert_allclose(result.rotation, ref_rotation, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result.translation, ref_translation, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result.aligned_coords, ref_aligned, atol=1e-10, rtol=1e-10)
    assert abs(result.weighted_rmsd - ref_weighted_rmsd) <= 1e-10
    assert abs(np.linalg.det(result.rotation) - 1.0) <= 1e-10


def test_fast_batch_matches_single_frame_loop() -> None:
    """Batched Kabsch should reproduce repeated single-frame solves."""

    rng = np.random.default_rng(8)
    base = rng.normal(size=(24, 3))
    frames = []
    for offset in np.linspace(-0.2, 0.2, 7):
        rotation = random_rotation(0.2 + offset)
        translation = np.array([offset, 0.3 - offset, -0.1], dtype=float)
        frames.append(base @ rotation.T + translation + rng.normal(scale=0.005, size=base.shape))
    coords = np.asarray(frames)
    reference = base.copy()
    weights = np.linspace(1.0, 3.0, base.shape[0])

    batch = weighted_kabsch_batch(coords, reference, weights)
    singles = [weighted_kabsch_single(frame, reference, weights) for frame in coords]

    np.testing.assert_allclose(batch.rotations, np.asarray([item.rotation for item in singles]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(batch.translations, np.asarray([item.translation for item in singles]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(batch.aligned_coords, np.asarray([item.aligned_coords for item in singles]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(batch.weighted_rmsd, np.asarray([item.weighted_rmsd for item in singles]), atol=1e-10, rtol=1e-10)


def test_apply_transforms_batch_matches_manual_application() -> None:
    """Batched transform application should use the public row-vector convention."""

    rng = np.random.default_rng(11)
    coords = rng.normal(size=(3, 5, 3))
    rotations = np.asarray([random_rotation(0.1), random_rotation(0.2), random_rotation(0.3)])
    translations = rng.normal(size=(3, 3))

    transformed = apply_transforms_batch(coords, rotations, translations)
    manual = np.asarray([coords[i] @ rotations[i].T + translations[i] for i in range(coords.shape[0])])

    np.testing.assert_allclose(transformed, manual, atol=1e-12, rtol=1e-12)


def test_reflection_correction_produces_proper_rotations() -> None:
    """Reflection-like inputs should still return determinant +1 rotations."""

    mobile = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    reference = mobile.copy()
    reference[:, 0] *= -1.0
    weights = np.ones(mobile.shape[0], dtype=float)

    result = weighted_kabsch_single(mobile, reference, weights)

    assert np.linalg.det(result.rotation) > 0.999999


def test_invalid_zero_weights_raise() -> None:
    """All-zero weights must be rejected."""

    coords = np.zeros((4, 3), dtype=float)
    with pytest.raises(ValueError):
        weighted_kabsch_single(coords, coords, np.zeros(4, dtype=float))
    with pytest.raises(ValueError):
        weighted_kabsch_batch(coords[None, :, :], coords, np.zeros(4, dtype=float))
