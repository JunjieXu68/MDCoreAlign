"""Numerical examples for robust distance-to-weight rules."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import gaussian_weights, huber_weights, tukey_weights


def main() -> None:
    """Print and save a numerical example for the current weight rules."""

    RESULTS.mkdir(exist_ok=True)
    distances = np.linspace(0.0, 3.0, num=13, dtype=float)
    gaussian = gaussian_weights(distances, sigma=1.0)
    huber = huber_weights(distances, delta=1.0)
    tukey = tukey_weights(distances, cutoff=2.0)
    output_path = RESULTS / "weight_rules_example.csv"

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance", "gaussian_sigma_1.0", "huber_delta_1.0", "tukey_cutoff_2.0"])
        for row in zip(distances, gaussian, huber, tukey):
            writer.writerow([f"{float(value):.6f}" for value in row])

    print("Weight rule examples")
    print("distance  gaussian  huber  tukey")
    for distance, g_value, h_value, t_value in zip(distances, gaussian, huber, tukey):
        print(f"{distance:8.3f}  {g_value:8.3f}  {h_value:5.3f}  {t_value:5.3f}")
    print()
    print(f"csv={output_path}")


if __name__ == "__main__":
    main()
