"""Minimal side-by-side benchmark against the real external MDLovoFit binary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import Trajectory, compare_methods_on_trajectory


def build_synthetic_trajectory(
    *,
    n_frames: int = 12,
    n_atoms: int = 10,
    flexible_atoms: int = 3,
    seed: int = 7,
) -> Trajectory:
    """Build a tiny synthetic trajectory with a rigid core and flexible tail."""

    if n_frames < 2:
        raise ValueError("n_frames must be at least 2")
    if n_atoms < 4:
        raise ValueError("n_atoms must be at least 4")
    if not 0 <= flexible_atoms < n_atoms:
        raise ValueError("flexible_atoms must be between 0 and n_atoms - 1")

    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n_atoms)
    reference = np.column_stack([x, np.sin(x), np.cos(x)])
    frames = []
    for frame_index in range(n_frames):
        angle = 0.04 * frame_index
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        translation = np.array([0.05 * frame_index, -0.02 * frame_index, 0.01 * frame_index], dtype=float)
        frame = reference @ rotation.T + translation
        frame += rng.normal(scale=0.01, size=frame.shape)
        if flexible_atoms > 0:
            frame[-flexible_atoms:, 1] += np.linspace(0.0, 1.2, flexible_atoms) * np.sin(0.6 * frame_index)
            frame[-flexible_atoms:, 2] += np.linspace(0.0, 0.8, flexible_atoms) * np.cos(0.4 * frame_index)
        frames.append(frame)
    return Trajectory(coords=np.asarray(frames, dtype=float))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the external benchmark example."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--workdir", type=Path, default=ROOT / "_tmp_mdsuperpose_external")
    return parser.parse_args()


def main() -> None:
    """Run the smallest side-by-side benchmark with the real MDLovoFit binary."""

    args = parse_args()
    trajectory = build_synthetic_trajectory()
    results = compare_methods_on_trajectory(
        trajectory,
        mdlovofit_fraction=args.fraction,
        workdir=args.workdir,
        mdlovofit_executable=args.mdlovofit_executable,
        mdlovofit_config=args.mdlovofit_config,
        robust_rule="huber",
        delta=0.2,
    )

    print("Side-by-side benchmark")
    print(f"frames={trajectory.n_frames} atoms={trajectory.n_atoms}")
    for result in results:
        print(
            f"{result.method_name:28s} "
            f"runtime_s={result.runtime_seconds if result.runtime_seconds is not None else float('nan'):.6f} "
            f"algorithm_s={result.algorithm_runtime_seconds if result.algorithm_runtime_seconds is not None else float('nan'):.6f} "
            f"overhead_s={result.overhead_seconds if result.overhead_seconds is not None else float('nan'):.6f} "
            f"rmsd={result.rmsd if result.rmsd is not None else float('nan'):.6f} "
            f"core_size={result.core_size if result.core_size is not None else 'NA'} "
            f"core_fraction={result.core_fraction if result.core_fraction is not None else 'NA'}"
        )
        if result.method_name == "external_mdlovofit":
            print("  reported_stdout_metrics=" + str(result.extra_metadata))


if __name__ == "__main__":
    main()
