"""Tests for transparent backend-auto v2 policy evaluation."""

import numpy as np

from scripts.post_b3_optimisation_v1.run_p3_backend_auto_v2 import evaluate_policy


def _rows():
    return [
        {"workload_residue_frames_observed": "1000", "n_frames": "10", "n_residues": "100", "cpu_loaded_seconds": "0.01", "gpu_loaded_seconds": "0.02"},
        {"workload_residue_frames_observed": "100000", "n_frames": "1000", "n_residues": "100", "cpu_loaded_seconds": "0.10", "gpu_loaded_seconds": "0.02"},
    ]


def test_threshold_policy_counts_choices_and_time() -> None:
    result = evaluate_policy(_rows(), threshold=5000)
    assert result["cpu_choices"] == 1
    assert result["gpu_choices"] == 1
    assert result["total_compute_seconds"] == 0.03
    assert result["incorrect_backend_choices_gt_5pct"] == 0


def test_model_policy_uses_predicted_backend() -> None:
    cpu_model = np.asarray([0.01, 1e-6, 0.0, 0.0])
    gpu_model = np.asarray([0.02, 0.0, 0.0, 0.0])
    result = evaluate_policy(_rows(), threshold=None, cpu_model=cpu_model, gpu_model=gpu_model)
    assert result["cpu_choices"] == 1
    assert result["gpu_choices"] == 1
