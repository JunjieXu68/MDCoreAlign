"""Tests for long-trajectory optimized alignment mode."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from mdsuperpose import (
    StreamingRMSFAccumulator,
    build_long_flexibility_profile,
    compute_rmsf,
    select_representative_frame_indices,
    solve_long_trajectory_alignment,
)
from mdsuperpose.fast_kabsch import apply_transforms_batch


def _rigid_toy_trajectory(n_frames: int = 12, n_atoms: int = 8) -> np.ndarray:
    """Build a toy trajectory with one flexible tail atom."""

    rng = np.random.default_rng(123)
    reference = rng.normal(size=(n_atoms, 3))
    frames = []
    for frame_index in range(n_frames):
        theta = 0.03 * frame_index
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        coords = reference @ rotation.T + np.array([0.2 * frame_index, -0.1, 0.05])
        coords[-1] += np.array([0.0, 0.1 * frame_index, 0.0])
        frames.append(coords)
    return np.asarray(frames, dtype=float)


def test_select_representative_frame_indices_are_deterministic() -> None:
    """Representative frame selection should include frame 0 and be deterministic."""

    first = select_representative_frame_indices(100, max_frames=10, strategy="random", seed=42)
    second = select_representative_frame_indices(100, max_frames=10, strategy="random", seed=42)
    assert first[0] == 0
    assert np.array_equal(first, second)
    assert first.shape == (10,)


def test_streaming_rmsf_matches_full_rmsf() -> None:
    """Chunked RMSF accumulation should match conventional full-array RMSF."""

    coords = _rigid_toy_trajectory(n_frames=17, n_atoms=7)
    accumulator = StreamingRMSFAccumulator(coords.shape[1])
    accumulator.update(coords[:5])
    accumulator.update(coords[5:12])
    accumulator.update(coords[12:])
    assert np.allclose(accumulator.rmsf(), compute_rmsf(coords), atol=1e-12)


def test_long_mode_outputs_profile_and_transforms() -> None:
    """Long mode should produce reusable transforms, frozen weights and RMSF."""

    coords = _rigid_toy_trajectory(n_frames=20, n_atoms=9)
    result = solve_long_trajectory_alignment(
        coords,
        max_weight_fit_frames=8,
        chunk_size=6,
        max_iterations=5,
        delta=0.5,
    )
    assert result.selected_frame_indices.shape[0] == 8
    assert result.rotations.shape == (20, 3, 3)
    assert result.translations.shape == (20, 3)
    assert result.robust_rmsf.shape == (9,)
    assert result.ordinary_rmsf.shape == (9,)
    aligned = apply_transforms_batch(coords, result.rotations, result.translations)
    assert aligned.shape == coords.shape
    profile = build_long_flexibility_profile(
        result,
        residue_labels=tuple(f"A:GLY{i + 1}" for i in range(coords.shape[1])),
        min_segment_length=1,
    )
    assert profile.n_residues == coords.shape[1]
    assert np.isfinite(profile.robust_aligned_rmsf).all()


def test_cli_long_mode_writes_long_artifacts(tmp_path) -> None:
    """The user-facing CLI should expose long mode and write diagnostics."""

    coords = _rigid_toy_trajectory(n_frames=10, n_atoms=6)
    input_path = tmp_path / "toy.npy"
    out_dir = tmp_path / "out"
    np.save(input_path, coords)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdcorealign",
            "flexibility",
            "--input",
            str(input_path),
            "--input-format",
            "array",
            "--output-dir",
            str(out_dir),
            "--mode",
            "long",
            "--max-weight-fit-frames",
            "5",
            "--chunk-size",
            "4",
            "--max-iterations",
            "5",
            "--min-segment-length",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Wrote flexibility profile" in completed.stdout
    assert (out_dir / "selected_frame_indices.tsv").exists()
    assert (out_dir / "long_mode_summary.json").exists()
    assert (out_dir / "timing.json").exists()
    assert (out_dir / "flexibility_profile.csv").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["mode"] == "long"
    assert summary["long_mode"] is True
    assert summary["selected_weight_fit_frames"] == 5
