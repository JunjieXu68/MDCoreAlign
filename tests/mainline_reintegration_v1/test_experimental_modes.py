"""Regression coverage for the opt-in reintegrated experimental modes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import mdsuperpose.benchmark as benchmark_module
from mdcorealign.cli import build_parser, run
from mdcorealign.experimental import (
    ExperimentalBatchWorker,
    solve_diagnostic_rich_alignment,
    solve_fast_global_alignment,
)
from mdsuperpose.window_kinematics import scan_local_kinematics


def _coords(n_frames: int = 30, n_residues: int = 24) -> np.ndarray:
    rng = np.random.default_rng(17)
    reference = np.cumsum(rng.normal(size=(n_residues, 3)), axis=0)
    frames = np.repeat(reference[None, :, :], n_frames, axis=0)
    frames[:, 9:15] += rng.normal(
        scale=0.35, size=(n_frames, 6, 3)
    )
    return frames


def test_default_cli_mode_is_production_throughput() -> None:
    args = build_parser().parse_args(["align", "--input", "input.npy"])
    assert args.mode == "throughput"
    assert args.source_unit == "auto"
    legacy = build_parser().parse_args(
        ["align", "--input", "input.npy", "--mode", "default"]
    )
    assert legacy.mode == "default"


def test_w5_chunk_size_and_repeat_output_are_invariant() -> None:
    coords = _coords()
    first = scan_local_kinematics(
        coords, descriptor_chunk_size=7, evaluate_blocks=False
    )
    second = scan_local_kinematics(
        coords, descriptor_chunk_size=13, evaluate_blocks=False
    )
    repeated = scan_local_kinematics(
        coords, descriptor_chunk_size=7, evaluate_blocks=False
    )
    np.testing.assert_array_equal(
        first.local_deformation_index, second.local_deformation_index
    )
    np.testing.assert_array_equal(
        first.local_deformation_index, repeated.local_deformation_index
    )
    np.testing.assert_array_equal(
        first.locally_deforming_segments, second.locally_deforming_segments
    )


def test_w5_global_scale_invariance_is_dimensionless() -> None:
    coords = _coords()
    baseline = scan_local_kinematics(coords, evaluate_blocks=False)
    for scale in (0.1, 1.0, 10.0):
        observed = scan_local_kinematics(
            coords * scale, evaluate_blocks=False
        )
        np.testing.assert_allclose(
            observed.local_deformation_index,
            baseline.local_deformation_index,
            atol=1e-10,
            rtol=1e-10,
        )


def test_fast_global_returns_real_transforms_rmsd_rmsf_and_confidence() -> None:
    result = solve_fast_global_alignment(
        _coords(),
        representative_count=16,
        update_rounds=2,
        chunk_size=11,
    )
    assert result.mode == "fast-global"
    assert result.rotations.shape == (30, 3, 3)
    assert result.translations.shape == (30, 3)
    assert result.aligned_coords is not None
    assert result.rmsd.shape == (30,)
    assert result.robust_rmsf.shape == (24,)
    assert result.metadata["status"] == "experimental"
    assert result.metadata["automatic_fallback"] is False
    fields = result.confidence.to_dict()
    assert fields["temporal_block_weight_spearman"] is not None
    assert fields["representative_subset_core_jaccard"] is not None
    assert fields["convergence_status"] in {
        "fixed_budget_stable",
        "fixed_budget_complete_not_converged",
    }


def test_fast_global_risk_trace_preserves_numerical_outputs() -> None:
    coords = _coords()
    baseline = solve_fast_global_alignment(
        coords,
        representative_count=16,
        update_rounds=4,
        chunk_size=11,
        store_aligned=False,
        compute_confidence=True,
        collect_risk_signals=False,
    )
    traced = solve_fast_global_alignment(
        coords,
        representative_count=16,
        update_rounds=4,
        chunk_size=11,
        store_aligned=False,
        compute_confidence=True,
        collect_risk_signals=True,
    )
    np.testing.assert_array_equal(
        baseline.frozen_weights,
        traced.frozen_weights,
    )
    np.testing.assert_array_equal(baseline.rotations, traced.rotations)
    np.testing.assert_array_equal(
        baseline.translations,
        traced.translations,
    )
    np.testing.assert_array_equal(baseline.rmsd, traced.rmsd)
    np.testing.assert_array_equal(
        baseline.robust_rmsf,
        traced.robust_rmsf,
    )
    assert baseline.risk_signals is None
    assert traced.risk_signals is not None
    assert len(traced.risk_signals["weighted_objective_history"]) == 4
    assert 0.0 <= traced.risk_signals[
        "last_two_round_core_jaccard"
    ] <= 1.0
    assert traced.risk_signals["near_core_threshold_count"] >= 1


def test_diagnostic_rich_composes_fast_global_and_w5() -> None:
    result = solve_diagnostic_rich_alignment(
        _coords(),
        representative_count=16,
        update_rounds=2,
        chunk_size=9,
        descriptor_chunk_size=8,
        block_count=2,
    )
    assert result.mode == "diagnostic-rich"
    assert result.local_deformation is not None
    assert result.motion_graph is None
    assert result.metadata["diagnostic_rich_not_fastest_mode"] is True


def test_local_deformation_cli_writes_schema_and_unit_provenance(
    tmp_path,
) -> None:
    input_path = tmp_path / "coords.npy"
    output_dir = tmp_path / "output"
    np.save(input_path, _coords(n_frames=20))
    status = run(
        [
            "local-deformation",
            "--input",
            str(input_path),
            "--input-format",
            "array",
            "--source-unit",
            "angstrom",
            "--output-dir",
            str(output_dir),
            "--descriptor-chunk-size",
            "6",
            "--block-count",
            "2",
        ]
    )
    assert status == 0
    with (output_dir / "local_deformation.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 24
    assert {
        "residue_id",
        "resname",
        "chain",
        "local_deformation_index",
        "local_rigidity_score",
        "locally_deforming_segment_id",
        "is_terminal",
        "chain_break_before",
        "temporal_block_reproducibility",
        "confidence",
        "source_unit",
        "conversion_factor_to_angstrom",
        "canonical_internal_unit",
    }.issubset(rows[0])
    assert {row["source_unit"] for row in rows} == {"angstrom"}
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["score_is_rmsd"] is False
    assert summary["hard_biological_domains_claimed"] is False


def test_array_auto_preserves_values_and_marks_unit_unknown(tmp_path) -> None:
    input_path = tmp_path / "coords.npy"
    output_dir = tmp_path / "output"
    np.save(input_path, _coords(n_frames=8))
    status = run(
        [
            "local-deformation",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--block-count",
            "2",
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["unit_provenance"]["source_unit"] == "unknown"
    assert summary["unit_provenance"]["conversion_factor_to_angstrom"] == 1.0
    assert summary["unit_provenance"]["canonical_internal_unit"] == "unknown"


def test_fast_global_cli_is_explicit_and_writes_confidence(tmp_path) -> None:
    input_path = tmp_path / "coords.npy"
    output_dir = tmp_path / "output"
    np.save(input_path, _coords(n_frames=12))
    status = run(
        [
            "align",
            "--input",
            str(input_path),
            "--source-unit",
            "angstrom",
            "--mode",
            "fast-global",
            "--fast-representative-count",
            "8",
            "--fast-update-rounds",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["mode"] == "fast-global"
    assert summary["experimental_mode"] is True
    assert summary["converged"] is None
    assert summary["output_contract"] == "metrics_transforms_rmsd_rmsf"
    assert (output_dir / "fast_global_confidence.json").is_file()


def test_experimental_batch_worker_reuses_edges_and_indices() -> None:
    coords = _coords(n_frames=20)
    worker = ExperimentalBatchWorker.create(coords.shape[1])
    first_indices = worker.representative_indices(20, 8)
    second_indices = worker.representative_indices(20, 8)
    assert first_indices is second_indices
    first = worker.scan_w5(coords, descriptor_chunk_size=6)
    second = worker.scan_w5(coords, descriptor_chunk_size=11)
    np.testing.assert_array_equal(first.movement_score, second.movement_score)
    summary = worker.cache_summary
    assert summary["reused_w5_edges"] is True
    assert summary["representative_index_cache_entries"] == 1
    assert summary["reused_kernel_temporary_arrays"] is False


def test_theseus_unified_rmsd_uses_one_common_consensus_orientation(
    monkeypatch,
) -> None:
    coords = _coords(n_frames=8, n_residues=18)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    consensus_frame_coords = coords @ rotation.T + np.asarray(
        [20.0, -7.0, 4.0]
    )
    parsed = SimpleNamespace(
        classical_pairwise_rmsd=1.0,
        maximum_likelihood_rmsd=0.9,
        least_squares_sigma=0.8,
        maximum_likelihood_sigma=0.7,
    )
    run_result = SimpleNamespace(
        aligned_coords=consensus_frame_coords,
        parsed_output=parsed,
        total_runtime_seconds=1.0,
        subprocess_runtime_seconds=0.8,
        input_write_seconds=0.1,
        output_read_seconds=0.1,
        return_code=0,
        stdout="",
        stderr="",
        command=("theseus",),
        aligned_pdb_path=Path("aligned.pdb"),
        output_core_atom_count=None,
        output_core_atom_fraction=None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_theseus_on_coordinates",
        lambda *args, **kwargs: run_result,
    )

    observed = benchmark_module.benchmark_external_theseus(
        coords,
        workdir=Path("."),
    )
    consensus = np.mean(coords, axis=0)
    expected_rotation = benchmark_module.solve_rigid_alignment(
        consensus,
        coords[0],
    )
    expected_oriented = (
        coords @ expected_rotation.rotation.T
        + expected_rotation.translation
    )
    expected_rmsd = float(
        np.mean(
            np.sqrt(
                np.mean(
                    np.sum(
                        (expected_oriented - coords[0][None, :, :]) ** 2,
                        axis=2,
                    ),
                    axis=1,
                )
            )
        )
    )
    assert np.isclose(observed.rmsd, expected_rmsd, atol=1e-12)
    assert observed.extra_metadata["unified_rmsd_orientation"] == (
        "one_common_consensus_to_reference_rigid_transform"
    )
    assert observed.extra_metadata["subprocess_runtime_seconds"] == 0.8
