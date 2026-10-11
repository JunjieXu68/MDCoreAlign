"""Tests for the MDLovoFit addendum runner's independent evaluator."""

import numpy as np
import pytest

from scripts.database_scale_benchmark_v2.common import read_tsv
from scripts.database_scale_benchmark_v2.orchestrate_mdlovofit_addendum import (
    phase_commands,
)
from scripts.database_scale_benchmark_v2.run_mdlovofit_addendum import (
    independent_frame_rmsd,
    linear_projection,
)
from scripts.database_scale_benchmark_v2.run_real_long_mdlovofit_addendum import (
    MANIFEST as REAL_LONG_MANIFEST,
)
from scripts.database_scale_benchmark_v2.run_real_long_mdlovofit_addendum import (
    _independent_rmsd as real_long_independent_rmsd,
)


def test_independent_frame_rmsd_does_not_refit_coordinates():
    reference = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    aligned = np.asarray([reference, reference + np.asarray([1.0, 0.0, 0.0])])
    per_frame, mean = independent_frame_rmsd(aligned, reference)
    np.testing.assert_allclose(per_frame, [0.0, 1.0])
    assert mean == pytest.approx(0.5)


def test_independent_frame_rmsd_rejects_mismatched_selection():
    with pytest.raises(ValueError, match="incompatible"):
        independent_frame_rmsd(np.zeros((2, 3, 3)), np.zeros((4, 3)))


def test_post_pilot_orchestration_order_is_locked(tmp_path):
    phases = [phase for phase, _ in phase_commands(tmp_path / "python")]
    assert phases == [
        "finalize_pilot",
        "all_domain",
        "finalize_all_domain",
        "deployment",
        "finalize_deployment",
        "resume_internal_B3",
        "finalize_internal_B3",
        "real_long_internal",
        "finalize_real_long_internal",
        "real_long",
        "finalize_real_long",
        "make_comparison_report",
    ]


def test_linear_projection_uses_frozen_domain_shapes_for_raw_manifest():
    success = [
        {
            "domain_id": "12asA00",
            "n_frames": 440,
            "n_residues": 327,
            "complete_end_to_end_seconds": 2.0,
        }
    ]
    raw_manifest = [{"domain_id": "12asA00"}]
    assert linear_projection(success, raw_manifest) == pytest.approx(2.0)


def test_real_long_manifest_is_locked_to_available_prefixes():
    rows = read_tsv(REAL_LONG_MANIFEST)
    assert len({row["system"] for row in rows}) == 15
    assert all(int(row["prefix_frames"]) <= int(row["n_frames"]) for row in rows)
    assert not any(int(row["prefix_frames"]) == 10_000 for row in rows)


def test_real_long_independent_rmsd_does_not_refit():
    reference = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    aligned = np.asarray([reference, reference + [1.0, 0.0, 0.0]])
    mean, median, p95 = real_long_independent_rmsd(aligned, reference)
    assert mean == pytest.approx(0.5)
    assert median == pytest.approx(0.5)
    assert p95 == pytest.approx(0.95)
