"""Tests for the prospective B3 pilot gate audit."""

from scripts.database_scale_benchmark_v2.audit_b3_pilot_gates import (
    audit_rows,
    classify_catastrophic,
)

GATES = {
    "valid_input_data_mapping_numerical_failures_max": 0,
    "CPU_GPU_core_jaccard_required": 1.0,
    "catastrophic_exact_relative_failure_fraction_max": 0.001,
    "median_exact_relative_core_jaccard_min": 0.95,
    "median_exact_relative_RMSF_spearman_min": 0.995,
}


def _row(**updates):
    row = {
        "trajectory_id": "case__T320__R1",
        "domain_id": "case",
        "status": "success",
        "failure_class": "",
        "cpu_exact_finite_outputs": "True",
        "cpu_exact_effective_core_fraction": "0.7",
        "cpu_exact_core_jaccard_vs_exact": "0.98",
        "cpu_exact_rmsf_spearman_vs_exact": "0.999",
        "cpu_gpu_core_jaccard": "1.0",
    }
    row.update(updates)
    return row


def test_catastrophic_definition_uses_registered_strict_bounds():
    assert classify_catastrophic(_row(cpu_exact_core_jaccard_vs_exact="0.5")) == []
    assert classify_catastrophic(_row(cpu_exact_rmsf_spearman_vs_exact="0.9")) == []
    reasons = classify_catastrophic(
        _row(
            cpu_exact_core_jaccard_vs_exact="0.49",
            cpu_exact_rmsf_spearman_vs_exact="0.89",
        )
    )
    assert reasons == ["core_jaccard_below_0.50", "RMSF_spearman_below_0.90"]


def test_quality_warning_does_not_block_without_integrity_failure():
    rows = [_row(cpu_exact_rmsf_spearman_vs_exact="0.80")]
    summary, gates, catastrophic, _ = audit_rows(rows, GATES)
    assert len(catastrophic) == 1
    assert not summary["all_prospective_quality_gates_pass"]
    assert summary["full_run_authorized_under_registered_stop_rule"]
    assert any(row["gate"] == "catastrophic_exact_relative_failure_fraction" for row in gates)


def test_integrity_failure_blocks_full_run():
    rows = [_row(status="failed", failure_class="mapping_integrity_failure")]
    summary, _, _, _ = audit_rows(rows, GATES)
    assert not summary["full_run_authorized_under_registered_stop_rule"]
