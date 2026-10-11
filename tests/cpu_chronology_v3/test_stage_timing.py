"""Focused tests for the v3 chronology timing contract."""

from __future__ import annotations

import pytest

from mdsuperpose.stage_timing import StageTiming, all_frame_seconds


def test_valid_nested_timing_rows_are_explicit() -> None:
    timing = StageTiming(
        end_to_end_total_seconds=10.0,
        weight_estimation_total_seconds=5.0,
        candidate_initialisation_seconds=2.0,
        chronological_correction_seconds=2.5,
        all_frame_total_seconds=4.0,
        stable_rmsf_seconds=0.5,
        result_construction_seconds=0.5,
    )
    timing.validate()
    rows = timing.rows(case_id="toy", method="C2", repeat=0)
    by_stage = {row["stage"]: row for row in rows}
    assert by_stage["weight_estimation_total_seconds"]["is_additive"] is False
    assert by_stage["chronological_correction_seconds"]["parent_stage"] == "weight_estimation_total_seconds"
    assert by_stage["stable_rmsf_seconds"]["stage_level"] == 2


def test_exact_does_not_duplicate_weight_time_as_correction() -> None:
    timing = StageTiming(
        end_to_end_total_seconds=3.0,
        weight_estimation_total_seconds=2.0,
        chronological_correction_seconds=2.0,
        all_frame_total_seconds=1.0,
    )
    with pytest.raises(ValueError, match="must not be duplicated"):
        timing.validate(exact_sequential=True)


def test_weight_children_cannot_exceed_parent() -> None:
    timing = StageTiming(
        end_to_end_total_seconds=4.0,
        weight_estimation_total_seconds=2.0,
        candidate_initialisation_seconds=1.5,
        refinement_seconds=1.0,
        all_frame_total_seconds=1.0,
    )
    with pytest.raises(ValueError, match="children exceed"):
        timing.validate()


def test_legacy_all_frame_total_is_mapped_locally() -> None:
    assert all_frame_seconds({"total": 2.5, "RMSF": 0.4}) == (2.5, 0.4)
    with pytest.raises(ValueError, match="nesting"):
        all_frame_seconds({"total": 1.0, "RMSF": 2.0})
