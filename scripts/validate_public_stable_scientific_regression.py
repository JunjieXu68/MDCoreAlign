#!/usr/bin/env python3
"""Compare v6 product outputs with the sealed v5 numerical baseline."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / "results/single_full_product_v5/MDCoreAlign_supervisor_test_package_v5_20260829.tar.gz"
)
OUTPUT = ROOT / "results/public_stable_polish_v6/09_validation/scientific_regression.tsv"


def run_mode(source: Path, mode: str, output: Path) -> None:
    """Run one compact mode from a selected source tree."""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source / "src")
    toy = source / "examples/toy_md/toy_ca_trajectory.pdb"
    command = [
        sys.executable,
        "-m",
        "mdcorealign",
        "align",
        "--input",
        str(toy),
        "--input-format",
        "ca-pdb",
        "--mode",
        mode,
        "--backend",
        "cpu",
        "--output-dir",
        str(output),
    ]
    subprocess.run(command, cwd=source, env=environment, check=True, capture_output=True)


def table_column(path: Path, name: str) -> np.ndarray:
    """Read one numeric TSV column."""

    with path.open(newline="") as handle:
        return np.asarray(
            [float(row[name]) for row in csv.DictReader(handle, delimiter="\t")],
            dtype=np.float64,
        )


def comparisons(left: Path, right: Path, mode: str) -> list[dict[str, object]]:
    """Return numerical comparisons for one alignment mode."""

    rows: list[dict[str, object]] = []
    with np.load(left / "transforms.npz") as a, np.load(right / "transforms.npz") as b:
        arrays = {
            "rotations": (a["rotations"], b["rotations"]),
            "translations": (a["translations"], b["translations"]),
        }
        for metric, (old, new) in arrays.items():
            difference = float(np.max(np.abs(old - new)))
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "max_absolute_difference": difference,
                    "status": "pass" if difference <= 1e-12 else "fail",
                }
            )
    table_metrics = {
        "full_set_RMSD": ("frame_metrics.tsv", "full_set_RMSD"),
        "weighted_RMSD": ("frame_metrics.tsv", "weighted_RMSD"),
        "stable_weight": ("residue_profile.tsv", "stable_weight"),
        "robust_RMSF": ("residue_profile.tsv", "robust_RMSF"),
    }
    for metric, (filename, column) in table_metrics.items():
        old = table_column(left / filename, column)
        new = table_column(right / filename, column)
        difference = float(np.max(np.abs(old - new)))
        rows.append(
            {
                "mode": mode,
                "metric": metric,
                "max_absolute_difference": difference,
                "status": "pass" if difference <= 1e-12 else "fail",
            }
        )
    return rows


def main() -> None:
    """Run exact and throughput parity and write the report."""

    with tempfile.TemporaryDirectory(prefix="mdcorealign_v6_regression_") as temporary:
        root = Path(temporary)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(root)
        v5 = root / "MDCoreAlign_supervisor_test_package/source"
        rows: list[dict[str, object]] = []
        for mode in ("throughput", "exact"):
            old_output = root / f"v5_{mode}"
            new_output = root / f"v6_{mode}"
            run_mode(v5, mode, old_output)
            run_mode(ROOT, mode, new_output)
            rows.extend(comparisons(old_output, new_output, mode))
            old_summary = json.loads((old_output / "summary.json").read_text())
            new_summary = json.loads((new_output / "summary.json").read_text())
            assert old_summary["mean_rmsd"] == new_summary["mean_rmsd"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit("scientific regression failed")
    print(f"scientific_regression_rows={len(rows)} status=pass")


if __name__ == "__main__":
    main()
