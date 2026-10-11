"""Product-contract tests for explicit exact, throughput and apply commands."""

from __future__ import annotations

import builtins
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mdcorealign.cli import build_parser, run

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples/toy_md/toy_ca_trajectory.pdb"


def test_public_mode_aliases_and_throughput_default() -> None:
    parser = build_parser()
    assert parser.parse_args(["align", "--input", str(TOY)]).mode == "throughput"
    assert parser.parse_args(["align", "--input", str(TOY), "--mode", "default"]).mode == "default"
    assert parser.parse_args(["align", "--input", str(TOY), "--mode", "exact"]).mode == "exact"
    assert parser.parse_args(["align", "--input", str(TOY), "--mode", "exact"]).exact_representative_count == 2000
    assert parser.parse_args(["align", "--input", str(TOY), "--mode", "throughput"]).mode == "throughput"
    assert parser.parse_args(["apply", "--input", str(TOY), "--transforms", "x.npz"]).command == "apply"


def test_advanced_help_maps_k_and_m_to_the_correct_flags(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["align", "--help-advanced"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--fast-representative-count: representative frames K" in output
    assert "--fast-update-rounds: robust update budget M" in output


def test_update_round_budget_eight_is_tested_and_default_remains_four(capsys) -> None:
    parser = build_parser()
    assert parser.parse_args(["align", "--input", str(TOY)]).fast_update_rounds == 4
    parsed = parser.parse_args(
        ["align", "--input", str(TOY), "--fast-update-rounds", "8"]
    )
    assert parsed.fast_update_rounds == 8
    assert "outside the tested set" not in capsys.readouterr().err


def test_update_round_budget_accepts_positive_untested_values(capsys) -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        ["align", "--input", str(TOY), "--fast-update-rounds", "16"]
    )
    assert parsed.fast_update_rounds == 16
    assert "outside the tested set {1, 2, 4, 8}" in capsys.readouterr().err


def test_update_round_budget_rejects_zero(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(
            ["align", "--input", str(TOY), "--fast-update-rounds", "0"]
        )
    assert exit_info.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_update_round_budget_eight_executes_without_untested_warning(
    tmp_path, capsys
) -> None:
    output = tmp_path / "m8"
    status = run(
        [
            "align",
            "--input",
            str(TOY),
            "--input-format",
            "ca-pdb",
            "--mode",
            "throughput",
            "--backend",
            "cpu",
            "--fast-representative-count",
            "5",
            "--fast-update-rounds",
            "8",
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0
    assert "outside the tested set" not in capsys.readouterr().err
    metadata = json.loads(
        (output / "diagnostics" / "run_metadata.json").read_text()
    )
    assert metadata["update_rounds_M"] == 8


def test_cpu_import_does_not_import_cupy(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("CPU import attempted to load CuPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import mdcorealign

    assert mdcorealign.__version__


def test_throughput_contract_writes_schema_and_apply_roundtrip(tmp_path) -> None:
    analysis = tmp_path / "analysis"
    materialized = tmp_path / "materialized"
    status = run(
        [
            "align",
            "--input",
            str(TOY),
            "--input-format",
            "ca-pdb",
            "--mode",
            "throughput",
            "--backend",
            "cpu",
            "--fast-representative-count",
            "5",
            "--fast-update-rounds",
            "2",
            "--output-dir",
            str(analysis),
        ]
    )
    assert status == 0
    expected = {
        "transforms.npz",
        "frame_metrics.tsv",
        "residue_profile.tsv",
        "flexible_segments.tsv",
        "summary.json",
    }
    assert expected.issubset({path.name for path in analysis.iterdir()})
    assert (analysis / "diagnostics" / "diagnostics.json").is_file()
    assert not (analysis / "aligned_trajectory.pdb").exists()
    with (analysis / "residue_profile.tsv").open(newline="") as handle:
        fields = set(csv.DictReader(handle, delimiter="\t").fieldnames or [])
    assert {
        "residue_id",
        "stable_weight",
        "mobility_score",
        "robust_RMSF",
        "flexible_segment_id",
    }.issubset(fields)
    assert "ordinary_RMSF" not in fields
    summary = json.loads((analysis / "summary.json").read_text())
    assert summary["weights_frozen_before_all_frame_fit"] is True
    assert summary["all_frames_explicitly_fitted"] is True
    assert summary["aligned_coordinates_materialized"] is False
    assert summary["method"] == "fixed-budget continuous-weight alignment"
    assert summary["mode"] == "throughput"
    assert summary["experimental_mode"] is False
    assert summary["converged_fraction"] is None
    assert summary["convergence_interpretation"] == "not_applicable_fixed_budget"
    metadata = json.loads(
        (analysis / "diagnostics" / "run_metadata.json").read_text()
    )
    assert metadata["representative_count_K"] == 5
    assert metadata["update_rounds_M"] == 2

    status = run(
        [
            "apply",
            "--input",
            str(TOY),
            "--input-format",
            "ca-pdb",
            "--transforms",
            str(analysis / "transforms.npz"),
            "--output-dir",
            str(materialized),
        ]
    )
    assert status == 0
    assert (materialized / "aligned_trajectory.pdb").stat().st_size > 0
    apply_summary = json.loads((materialized / "apply_summary.json").read_text())
    assert apply_summary["algorithm_rerun"] is False
    assert apply_summary["fit_label_order_verified"] is True


def test_exact_mode_writes_explicit_transforms_without_materialising(tmp_path) -> None:
    output = tmp_path / "exact"
    assert run(
        [
            "align",
            "--input",
            str(TOY),
            "--input-format",
            "ca-pdb",
            "--mode",
            "exact",
            "--backend",
            "cpu",
            "--exact-representative-count",
            "5",
            "--output-dir",
            str(output),
        ]
    ) == 0
    with np.load(output / "transforms.npz", allow_pickle=False) as archive:
        assert archive["rotations"].shape == (5, 3, 3)
        assert archive["translations"].shape == (5, 3)


def test_explicit_gpu_request_never_silently_falls_back(monkeypatch, tmp_path, capsys) -> None:
    import mdcorealign.gpu_backend as gpu_backend

    def unavailable():
        raise RuntimeError(
            "GPU backend requested but no compatible CuPy/CUDA runtime is "
            "available. The installed MDCoreAlign package already contains "
            "the GPU backend. Install a compatible CuPy build, then run "
            "`mdcorealign doctor`."
        )

    monkeypatch.setattr(gpu_backend, "_cupy", unavailable)
    status = run(
        [
            "align",
            "--input",
            str(TOY),
            "--input-format",
            "ca-pdb",
            "--mode",
            "throughput",
            "--backend",
            "gpu",
            "--output-dir",
            str(tmp_path / "gpu"),
        ]
    )
    assert status == 2
    error = capsys.readouterr().err
    assert "compatible CuPy/CUDA runtime" in error
    assert "already contains the GPU backend" in error
    assert "mdcorealign doctor" in error
    assert "mdcorealign[gpu]" not in error
