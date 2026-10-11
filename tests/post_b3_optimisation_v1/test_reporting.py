"""Regression tests for corrected post-B3 reporting semantics."""

from scripts.post_b3_optimisation_v1.run_p1_p2_reporting import (
    approximation_status,
    geometric_tail,
    registered_reasons,
    sampling_status,
    solver_status,
)


def _exact_row() -> dict[str, str]:
    row = {
        "status": "success",
        "failure_class": "",
        "exact_completed": "True",
        "strict_fixed_budget_status": "fixed_budget_complete_not_converged",
        "representative_subset_weight_spearman": "0.95",
        "representative_subset_core_jaccard": "0.90",
        "temporal_block_weight_spearman": "0.90",
        "temporal_block_core_jaccard": "0.80",
    }
    for method in ("cpu", "gpu"):
        row.update(
            {
                f"{method}_exact_finite_outputs": "True",
                f"{method}_exact_effective_core_fraction": "0.60",
                f"{method}_exact_core_jaccard_vs_exact": "0.96",
                f"{method}_exact_rmsf_spearman_vs_exact": "0.99",
                f"{method}_exact_median_absolute_rmsd_difference_vs_exact_A": "1.2",
            }
        )
    return row


def test_large_rmsd_tail_is_not_registered_catastrophe() -> None:
    row = _exact_row()
    assert not registered_reasons(row, "cpu")
    assert geometric_tail(row, "cpu")
    assert approximation_status(row) == "large_geometric_tail_review"


def test_registered_rmsf_failure_is_separate() -> None:
    row = _exact_row()
    row["cpu_exact_rmsf_spearman_vs_exact"] = "0.85"
    assert registered_reasons(row, "cpu") == ["RMSF_spearman_below_0.90"]
    assert approximation_status(row) == "registered_catastrophic_quality_failure"


def test_independent_solver_and_sampling_status() -> None:
    row = _exact_row()
    assert solver_status(row) == "fixed_budget_complete_not_strictly_converged"
    assert sampling_status(row) == "reproducible_at_registered_internal_thresholds"
    row["temporal_block_core_jaccard"] = "0.5"
    assert sampling_status(row) == "mixed_reproducibility"


def test_short_input_is_unsupported_not_solver_quality() -> None:
    row = _exact_row()
    row["status"] = "failed"
    row["failure_reason"] = "ValueError: too_few_frames: 10 < 20"
    assert solver_status(row) == "unsupported_short_input"
