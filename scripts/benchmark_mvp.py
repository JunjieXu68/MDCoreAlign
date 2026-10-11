"""Small runtime benchmark for the mdstablecore MVP."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demo_synthetic import build_synthetic_trajectory
from mdstablecore.baseline import run_variance_baseline
from mdstablecore.pipeline import run_stable_core_pipeline
from mdstablecore.trajectory_io import make_trajectory_data


def main() -> None:
    """Benchmark a few synthetic trajectory sizes."""

    for n_frames in [100, 250, 500]:
        coords, atom_to_residue, _ = build_synthetic_trajectory(n_frames=n_frames)
        data = make_trajectory_data(coords, atom_to_residue)
        start = time.perf_counter()
        result = run_stable_core_pipeline(data)
        pipeline_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        _, baseline_core = run_variance_baseline(data)
        baseline_elapsed = time.perf_counter() - start

        print(
            f"frames={n_frames:4d} "
            f"pipeline_time={pipeline_elapsed:7.3f}s "
            f"baseline_time={baseline_elapsed:7.3f}s "
            f"core={int(result.core_mask.sum()):2d} "
            f"baseline_core={int(baseline_core.sum()):2d} "
            f"conf={result.quality.confidence_mean:.3f}"
        )


if __name__ == "__main__":
    main()
