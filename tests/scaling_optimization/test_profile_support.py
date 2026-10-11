"""Tests for optional Phase 1 profiling infrastructure."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.scaling_optimization.profile_support import STAGE_NAMES, MemorySampler, StageRecorder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_stage_recorder_has_required_schema_and_can_be_disabled() -> None:
    enabled = StageRecorder(enabled=True)
    with enabled.stage("rmsd_compute"):
        time.sleep(0.001)
    assert enabled.values["rmsd_compute"] > 0.0
    assert set(STAGE_NAMES).issubset(enabled.to_dict())

    disabled = StageRecorder(enabled=False)
    with disabled.stage("rmsd_compute"):
        pass
    assert disabled.values["rmsd_compute"] == 0.0


def test_memory_sampler_reports_peak_rss_only_when_started() -> None:
    sampler = MemorySampler(interval_seconds=0.001)
    sampler.start()
    _temporary = np.ones((1000, 100), dtype=float)
    sampler.stop()
    result = sampler.to_dict()
    assert result["sample_count"] >= 1
    assert result["peak_rss_bytes"] >= result["initial_rss_bytes"]
    assert result["peak_gpu_allocated_bytes"] is None


def test_child_input_decode_profile_writes_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    input_path = tmp_path / "coords.npz"
    output_path = tmp_path / "run.json"
    np.savez_compressed(input_path, coords=rng.normal(size=(12, 8, 3)).astype(np.float32))
    command = [
        sys.executable,
        "scripts/scaling_optimization/run_profile_matrix.py",
        "--child",
        "--profile-stages",
        "--input-path",
        str(input_path),
        "--dataset-id",
        "toy",
        "--data-semantics",
        "test",
        "--expected-residues",
        "8",
        "--n-frames",
        "12",
        "--config",
        "input_decode_only",
        "--raw-output",
        str(output_path),
        "--output-root",
        str(tmp_path / "out"),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "success"
    assert payload["n_frames"] == 12
    assert payload["n_residues"] == 8
    assert set(STAGE_NAMES).issubset(payload["stages_seconds"])
    assert payload["memory"]["peak_rss_bytes"] > 0
