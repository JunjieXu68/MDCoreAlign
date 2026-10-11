"""Regression tests for the ATLAS matched-contract summary layer."""

from __future__ import annotations

from scripts.atlas_large_long_v1 import finalize_a4_a5


def test_loaded_w5_keeps_non_alignment_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        finalize_a4_a5,
        "read_tsv",
        lambda _path: [
            {
                "system": "toy",
                "prefix_label": "f00250",
                "method": "CPU_W5",
                "wall_seconds": "0.25",
                "status": "success",
                "is_warmup": "False",
            }
        ],
    )

    rows = finalize_a4_a5._normalise_loaded()

    assert len(rows) == 1
    assert rows[0]["contract"] == (
        "A_loaded_W5_local_deformation_not_alignment_equivalent"
    )
    assert "not an equivalent RMSD workflow" in rows[0]["timing_boundary_caveat"]


def test_group_timing_promotes_stage_and_memory_statistics() -> None:
    rows = []
    for repeat, runtime, compute, rss in (
        (0, 2.0, 1.0, 100.0),
        (1, 4.0, 3.0, 120.0),
    ):
        rows.append(
            {
                "system": "toy",
                "prefix_label": "f00250",
                "n_frames": 250,
                "n_residues": 10,
                "contract": "A_loaded_analysis",
                "method": "GPU_throughput_float32_Horn",
                "repeat_index": repeat,
                "is_warmup": False,
                "status": "success",
                "runtime_seconds": runtime,
                "H2D_seconds": 0.1,
                "GPU_compute_seconds": compute,
                "D2H_seconds": 0.2,
                "peak_RSS_mb_process": rss,
                "peak_VRAM_bytes": 1024 * (repeat + 1),
            }
        )

    summary = finalize_a4_a5._group_timing(rows)[0]

    assert summary["median_seconds"] == 3.0
    assert summary["median_GPU_compute_seconds"] == 2.0
    assert summary["peak_RSS_mb_across_measured_repeats"] == 120.0
    assert summary["peak_VRAM_bytes_across_measured_repeats"] == 2048.0
    assert summary["status"] == "failure_or_incomplete"


def test_batch_oom_is_retained_as_non_biological_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        finalize_a4_a5,
        "read_tsv",
        lambda _path: [
            {
                "system": "large",
                "mode": "GPU_W5_only_batched",
                "batch_size": "8",
                "n_frames_per_trajectory": "10001",
                "n_residues_per_trajectory": "2128",
                "status": "failure",
                "is_warmup": "True",
                "failure_reason": "OutOfMemoryError: allocation failed",
            }
        ],
    )

    failures = finalize_a4_a5._batch_failure_rows()

    assert failures[0]["failure_class"] == "GPU_out_of_memory"
    assert failures[0]["failed_before_measured_repeats"] is True
    assert failures[0]["biological_validation_failure"] is False
