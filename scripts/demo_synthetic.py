"""Synthetic demo for the mdstablecore MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdstablecore.baseline import run_variance_baseline
from mdstablecore.evaluation import evaluate_pipeline
from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.quality_metrics import jaccard_similarity
from mdstablecore.trajectory_io import make_trajectory_data
from mdstablecore.types import PipelineConfig


def build_synthetic_trajectory(
    n_frames: int = 200,
    n_residues: int = 30,
    atoms_per_residue: int = 1,
    random_seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a toy system with a stable core and flexible tails."""

    rng = np.random.default_rng(random_seed)
    base = np.zeros((n_residues * atoms_per_residue, 3), dtype=float)
    atom_to_residue = np.repeat(np.arange(n_residues), atoms_per_residue)

    for residue in range(n_residues):
        base[residue * atoms_per_residue : (residue + 1) * atoms_per_residue, 0] = residue * 3.8

    coords = np.repeat(base[None, :, :], n_frames, axis=0)
    core_residues = np.zeros(n_residues, dtype=bool)
    core_residues[5:22] = True

    for frame in range(n_frames):
        global_shift = rng.normal(scale=0.3, size=(1, 3))
        coords[frame] += global_shift

        for residue in range(n_residues):
            scale = 0.15 if core_residues[residue] else 1.0
            coords[frame, atom_to_residue == residue] += rng.normal(scale=scale, size=(atoms_per_residue, 3))

        hinge = np.sin(frame / 15.0) * 0.8
        coords[frame, atom_to_residue >= 22, 1] += hinge * np.linspace(0.0, 1.0, np.sum(atom_to_residue >= 22))

    return coords, atom_to_residue, core_residues


def main() -> None:
    """Run the synthetic demo and print a short summary."""

    coords, atom_to_residue, expected_core = build_synthetic_trajectory()
    data = make_trajectory_data(coords, atom_to_residue)
    config = PipelineConfig()
    result = run_stable_core_pipeline(data, config=config)
    baseline_scores, baseline_core = run_variance_baseline(data, core_quantile=config.final_core_quantile)
    evaluation = evaluate_pipeline(data, config=config)

    print("Detected core residues:")
    print(np.flatnonzero(result.core_mask))
    print("Expected core residues:")
    print(np.flatnonzero(expected_core))
    print("Baseline core residues:")
    print(np.flatnonzero(baseline_core))
    print("Top 10 residue stability scores:")
    top_indices = np.argsort(result.residue_scores.combined)[::-1][:10]
    for residue in top_indices:
        print(
            f"residue={residue:2d} "
            f"score={result.residue_scores.combined[residue]:.3f} "
            f"confidence={result.confidence[residue]:.3f}"
        )

    pipeline_jaccard = jaccard_similarity(result.core_mask, expected_core)
    baseline_jaccard = jaccard_similarity(baseline_core, expected_core)
    baseline_corr = float(np.corrcoef(baseline_scores, result.residue_scores.combined)[0, 1])

    print(f"Mean confidence: {result.quality.confidence_mean:.3f}")
    print(f"Coarse/refined Jaccard: {result.quality.coarse_refined_core_jaccard:.3f}")
    print(f"Pipeline vs expected Jaccard: {pipeline_jaccard:.3f}")
    print(f"Baseline vs expected Jaccard: {baseline_jaccard:.3f}")
    print(f"Baseline vs pipeline score correlation: {baseline_corr:.3f}")
    print("Evaluation summary:")
    for key in sorted(evaluation):
        print(f"{key}: {evaluation[key]:.3f}")


if __name__ == "__main__":
    main()
