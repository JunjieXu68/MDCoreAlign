"""Rigid-body invariance and localisation tests for the Phase 6A scan."""

from __future__ import annotations

import numpy as np

from mdsuperpose.window_scan import scan_window_local


def _reference(n_residues: int = 30) -> np.ndarray:
    index = np.arange(n_residues, dtype=float)
    return np.column_stack(
        (
            3.8 * index,
            2.0 * np.sin(index / 3.0),
            1.5 * np.cos(index / 4.0),
        )
    )


def _rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def test_identical_frames_have_zero_movement() -> None:
    coords = np.repeat(_reference()[None, :, :], 20, axis=0)
    result = scan_window_local(coords)
    np.testing.assert_allclose(result.movement_score, 0.0, atol=1e-15)
    np.testing.assert_allclose(result.stability_weights, 1.0, atol=1e-15)


def test_whole_protein_translation_is_invariant() -> None:
    reference = _reference()
    coords = np.asarray(
        [reference + frame * np.asarray((2.0, -0.7, 0.3)) for frame in range(20)]
    )
    result = scan_window_local(coords)
    np.testing.assert_allclose(result.movement_score, 0.0, atol=2e-15)


def test_whole_protein_rotation_is_invariant() -> None:
    reference = _reference()
    coords = np.asarray(
        [
            reference @ _rotation(frame * 0.1).T
            + np.asarray((frame, -0.2 * frame, 0.5))
            for frame in range(20)
        ]
    )
    result = scan_window_local(coords)
    assert float(np.max(result.movement_score)) < 1e-14


def test_local_loop_perturbation_is_localised() -> None:
    rng = np.random.default_rng(42)
    reference = _reference()
    coords = np.repeat(reference[None, :, :], 80, axis=0)
    coords[:, 12:17] += rng.normal(scale=0.8, size=(80, 5, 3))
    result = scan_window_local(coords)
    local = np.mean(result.movement_percentile[10:19])
    distant = np.mean(np.concatenate((result.movement_percentile[:6], result.movement_percentile[23:])))
    assert local > distant + 0.35
    assert np.count_nonzero(result.candidate_stable_label[12:17]) <= 1


def test_single_outlier_and_terminal_tail_are_detected() -> None:
    rng = np.random.default_rng(42)
    reference = _reference()
    coords = np.repeat(reference[None, :, :], 60, axis=0)
    coords[:, 14] += rng.normal(scale=1.2, size=(60, 3))
    coords[:, -5:] += rng.normal(scale=0.7, size=(60, 5, 3))
    result = scan_window_local(coords, feature_set="full")
    assert result.movement_percentile[14] >= 0.8
    assert float(np.mean(result.movement_percentile[-5:])) > 0.65


def test_collinear_geometry_is_finite_and_repeatable() -> None:
    reference = np.column_stack((np.arange(20, dtype=float), np.zeros((20, 2))))
    coords = np.repeat(reference[None, :, :], 12, axis=0)
    first = scan_window_local(coords)
    second = scan_window_local(coords)
    assert np.isfinite(first.movement_score).all()
    np.testing.assert_array_equal(first.stability_weights, second.stability_weights)
    np.testing.assert_array_equal(first.candidate_stable_label, second.candidate_stable_label)
