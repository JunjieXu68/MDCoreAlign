"""Focused regression tests for the scientific-evidence upgrade study."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_SCRIPTS = REPO_ROOT / "scripts" / "scientific_evidence_upgrade_v1"
sys.path.insert(0, str(STUDY_SCRIPTS))

from run_se7_heldout_geometry import bootstrap_median, folds  # noqa: E402
from synthetic_truth import (  # noqa: E402
    generate_synthetic_trajectory,
    rotation_geodesic_degrees,
)


def reference_coordinates(n_residues: int = 40) -> np.ndarray:
    """Return a deterministic non-collinear CA-like trace."""

    index = np.arange(n_residues, dtype=np.float64)
    return np.column_stack(
        (
            3.8 * index,
            2.0 * np.sin(index / 3.0),
            1.5 * np.cos(index / 5.0),
        )
    )


def test_synthetic_generator_is_reproducible() -> None:
    kwargs = {
        "scenario": "S4",
        "n_frames": 24,
        "seed": 1701,
        "local_amplitude_A": 0.75,
        "hinge_amplitude_deg": 15.0,
    }
    first = generate_synthetic_trajectory(reference_coordinates(), **kwargs)
    second = generate_synthetic_trajectory(reference_coordinates(), **kwargs)
    np.testing.assert_array_equal(first.observed, second.observed)
    np.testing.assert_array_equal(first.true_stable_mask, second.true_stable_mask)
    np.testing.assert_array_equal(first.true_rotations, second.true_rotations)


def test_pure_rigid_scenario_has_no_internal_deformation() -> None:
    reference = reference_coordinates()
    trajectory = generate_synthetic_trajectory(
        reference,
        scenario="S0",
        n_frames=20,
        seed=1702,
        local_amplitude_A=0.0,
        hinge_amplitude_deg=0.0,
    )
    expected = np.repeat(reference[None, :, :], 20, axis=0)
    np.testing.assert_allclose(trajectory.internal, expected, atol=0.0, rtol=0.0)
    assert np.all(trajectory.true_stable_mask)
    assert not np.any(trajectory.mobile_mask)


def test_rotation_geodesic_identity_and_known_angle() -> None:
    identity = np.eye(3, dtype=np.float64)
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    values = rotation_geodesic_degrees(
        np.stack((identity, rotation)),
        np.stack((identity, identity)),
    )
    np.testing.assert_allclose(values, [0.0, 30.0], atol=1e-10)


def test_blocked_and_interleaved_folds_have_no_leakage() -> None:
    for fold_type in ("contiguous", "interleaved"):
        split = folds(23, fold_type)
        assert len(split) == 4
        test_union: set[int] = set()
        for train, test in split:
            assert np.intersect1d(train, test).size == 0
            assert np.union1d(train, test).size == 23
            test_union.update(int(index) for index in test)
        assert test_union == set(range(23))


def test_bootstrap_median_is_deterministic() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert bootstrap_median(values, "fixed-seed-key") == bootstrap_median(
        values,
        "fixed-seed-key",
    )
    assert bootstrap_median([], "empty") == ("", "")


def test_registered_mdcath_reproduction_stop_is_explicit() -> None:
    summary_path = (
        REPO_ROOT
        / "results/scientific_evidence_upgrade_v1/05_mdcath_rmsf_convention"
        / "audit_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "FAIL_STOP_BEFORE_SE6"
    assert payload["n_cases"] == 100
    assert payload["n_passed"] == 96
    assert payload["maximum_absolute_difference_A"] > 1e-4


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one study table for output-regression checks."""

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_heldout_geometry_panel_is_complete_and_successful() -> None:
    path = (
        REPO_ROOT
        / "results/scientific_evidence_upgrade_v1/07_heldout_geometry"
        / "pair_distance_results.tsv"
    )
    rows = read_tsv(path)
    systems = {(row["source_type"], row["system_id"]) for row in rows}
    assert len(rows) == 20_800
    assert len(systems) == 520
    assert all(row["status"] == "success" for row in rows)


def test_heldout_bootstrap_uses_domain_or_family_units() -> None:
    path = (
        REPO_ROOT
        / "results/scientific_evidence_upgrade_v1/07_heldout_geometry"
        / "paired_method_summary.tsv"
    )
    rows = read_tsv(path)
    assert rows
    assert {row["bootstrap_unit"] for row in rows} == {"domain_or_family"}
    mdcath_absolute = [
        row
        for row in rows
        if row["source_type"] == "mdCATH"
        and row["fold_type"] == "contiguous"
        and row["comparison"] == "MDCoreAlign_throughput"
    ]
    assert mdcath_absolute
    assert {int(row["n_units"]) for row in mdcath_absolute} == {512}


def test_budget_fidelity_production_and_unsupported_m8_are_explicit() -> None:
    root = REPO_ROOT / "results/scientific_evidence_upgrade_v1/08_budget_fidelity"
    rows = read_tsv(root / "validation_results.tsv")
    production = [
        row for row in rows
        if row["configuration"] == "production" and row["status"] == "success"
    ]
    unsupported = [row for row in rows if row["configuration"] == "higher"]
    assert len(production) == 384
    assert np.median([float(row["core_Jaccard_vs_sequential"]) for row in production]) > 0.97
    assert np.median([float(row["RMSF_Spearman_vs_sequential"]) for row in production]) > 0.9998
    assert not any(row["tail_flag"] == "True" for row in production)
    assert len(unsupported) == 384
    assert all(row["status"] == "failed" for row in unsupported)
    assert all("update_rounds must be one of" in row["error"] for row in unsupported)
    assert (root / "unsupported_budget_adjudication.md").is_file()


def test_empirical_complexity_preserves_all_frame_scaling() -> None:
    path = (
        REPO_ROOT
        / "results/scientific_evidence_upgrade_v1/08_budget_fidelity"
        / "empirical_complexity.tsv"
    )
    rows = read_tsv(path)
    all_frame = next(
        row for row in rows
        if row["scaling_axis"] == "frames"
        and row["runtime_component"] == "all_frame_fit_seconds"
    )
    weight = next(
        row for row in rows
        if row["scaling_axis"] == "frames"
        and row["runtime_component"] == "weight_estimation_seconds"
    )
    assert 0.8 < float(all_frame["log_log_slope"]) < 1.2
    assert abs(float(weight["log_log_slope"])) < 0.2


def test_integrated_narrative_and_figure_assets_are_complete() -> None:
    root = REPO_ROOT / "results/scientific_evidence_upgrade_v1/10_integrated_results"
    summary = json.loads((root / "integration_summary.json").read_text(encoding="utf-8"))
    assert summary["narrative"] == "C"
    assert summary["SE6_status"] == "skipped_by_preregistered_stop_rule"
    for figure in range(1, 6):
        matches = list((root / "figures").glob(f"fig{figure}_*.pdf"))
        assert len(matches) == 1 and matches[0].stat().st_size > 0
        png = matches[0].with_suffix(".png")
        assert png.is_file() and png.stat().st_size > 0
