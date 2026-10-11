"""CPU-independent validation of the experimental batched GPU path."""

import numpy as np
import pytest

from mdcorealign.gpu_backend import gpu_available


@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
def test_same_shape_gpu_batch_is_finite_and_shape_safe() -> None:
    from mdcorealign.experimental_gpu_batch import solve_same_shape_gpu_batch

    rng = np.random.default_rng(7)
    coords = rng.normal(size=(2, 12, 8, 3))
    result = solve_same_shape_gpu_batch(
        coords,
        chain_ids=np.asarray(["A"] * 8),
        representative_count=8,
        update_rounds=2,
        run_throughput=True,
        run_w5=True,
    )
    assert result.weights.shape == (2, 8)
    assert result.rmsd.shape == (2, 12)
    assert result.robust_rmsf.shape == (2, 8)
    assert result.local_deformation_index.shape == (2, 8)
    assert np.isfinite(result.rmsd).all()
