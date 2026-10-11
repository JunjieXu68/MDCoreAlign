"""Tests for the PDB-safe common rigid centering convention."""

import numpy as np

from scripts.post_b3_optimisation_v1.run_p6_mdlovofit import center_for_pdb


def test_centering_is_one_common_translation() -> None:
    coords = np.arange(72, dtype=float).reshape(3, 8, 3) + 20_000.0
    centered, translation = center_for_pdb(coords)
    assert np.allclose(centered, coords - translation[:, None, :])
    assert np.allclose(centered.mean(axis=1), 0.0)
    assert np.max(np.abs(centered)) < 10_000.0


def test_common_centering_preserves_internal_displacements() -> None:
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(4, 10, 3)) + np.asarray([12_000.0, -15_000.0, 8_000.0])
    centered, _ = center_for_pdb(coords)
    for raw, shifted in zip(coords, centered, strict=True):
        assert np.allclose(
            raw[:, None, :] - raw[None, :, :],
            shifted[:, None, :] - shifted[None, :, :],
        )
