#!/usr/bin/env python
"""Run locked v3 scientific quality validation without timing repeats."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import threading
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose.chronology_adaptive import AdaptiveRiskRule  # noqa: E402
from mdsuperpose.chronology_study import (  # noqa: E402
    compare_results,
    run_candidate,
    run_exact,
)
from scripts.cpu_chronology_v3.run_self_variability import load_case  # noqa: E402


OUT = ROOT / "results/cpu_chronology_v3"
METHODS = (
    "exact_sequential_f64_compact",
    "throughput_compact_f64",
    "D_block_64_v3",
    "C_correction_2_v3",
    "adaptive_C1_C2_D64_v3",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_rule() -> AdaptiveRiskRule:
    payload = json.loads((OUT / "04_candidates/adaptive_rule.json").read_text())
    return AdaptiveRiskRule(
        **{f"c1_{key}": float(value) for key, value in payload["c1"].items()},
        **{f"c2_{key}": float(value) for key, value in payload["c2"].items()},
    )


def current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


class PeakRSS:
    """Sample process RSS while one method runs, without external packages."""

    def __init__(self) -> None:
        self.peak = current_rss_mib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, current_rss_mib())

    def __enter__(self) -> "PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, current_rss_mib())


def identity_metrics(exact: object) -> dict[str, object]:
    return {
        "stable_weight_max_difference": 0.0,
        "stable_weight_mean_difference": 0.0,
        "weight_spearman": 1.0,
        "core_jaccard": 1.0,
        "rotation_geodesic_median_degrees": 0.0,
        "rotation_geodesic_p95_degrees": 0.0,
        "sampled_aligned_coordinate_difference_median_A": 0.0,
        "sampled_aligned_coordinate_difference_p95_A": 0.0,
        "rmsd_absolute_difference_median_A": 0.0,
        "rmsd_absolute_difference_p95_A": 0.0,
        "rmsd_absolute_difference_max_A": 0.0,
        "stable_core_rmsf_spearman": 1.0,
        "stable_core_rmsf_mae_A": 0.0,
        "non_finite_transform": False,
        "catastrophic_mismatch": False,
        "candidate_runtime_seconds": exact.runtime_seconds,
        "reference_runtime_seconds": exact.runtime_seconds,
        "speedup_vs_exact": 1.0,
    }


def evaluate_case(row: dict[str, str], panel_name: str) -> dict[str, object]:
    started = time.perf_counter()
    coords, chains = load_case(row)
    rule = load_rule()
    quality_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    with PeakRSS() as monitor:
        exact = run_exact(coords, chains)
    base = {
        "case_id": row["case_id"],
        "family_id": row["family_id"],
        "panel": panel_name,
        "n_frames": int(row["n_frames"]),
        "n_residues": int(row["n_residues"]),
        "method": METHODS[0],
        **identity_metrics(exact),
        "peak_rss_mib": monitor.peak,
        "accepted_level": "exact",
    }
    quality_rows.append(base)
    distribution_rows.append(
        {
            "case_id": row["case_id"],
            "panel": panel_name,
            "method": METHODS[0],
            **{f"absolute_rmsd_difference_q{q:g}_A": 0.0 for q in (0, 25, 50, 75, 90, 95, 99, 100)},
        }
    )
    for method in METHODS[1:]:
        with PeakRSS() as monitor:
            candidate = run_candidate(
                coords,
                chains,
                method,
                adaptive_rule=rule if method == "adaptive_C1_C2_D64_v3" else None,
            )
        accepted = str(candidate.metadata.get("accepted_level", "fixed"))
        quality_rows.append(
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "panel": panel_name,
                "n_frames": int(row["n_frames"]),
                "n_residues": int(row["n_residues"]),
                "method": method,
                **compare_results(candidate, exact, coords),
                "peak_rss_mib": monitor.peak,
                "accepted_level": accepted,
            }
        )
        if method == "adaptive_C1_C2_D64_v3":
            path_rows.append(
                {
                    "case_id": row["case_id"],
                    "family_id": row["family_id"],
                    "panel": panel_name,
                    "accepted_level": accepted,
                    "c1_features_json": json.dumps(candidate.metadata.get("c1_features", {}), sort_keys=True),
                    "c2_features_json": json.dumps(candidate.metadata.get("c2_features", {}), sort_keys=True),
                }
            )
        difference = np.abs(candidate.rmsd - exact.rmsd)
        quantiles = np.quantile(difference, np.asarray((0, .25, .5, .75, .9, .95, .99, 1.0)))
        distribution_rows.append(
            {
                "case_id": row["case_id"],
                "panel": panel_name,
                "method": method,
                **{
                    f"absolute_rmsd_difference_q{q:g}_A": float(value)
                    for q, value in zip((0, 25, 50, 75, 90, 95, 99, 100), quantiles, strict=True)
                },
            }
        )
        del candidate
    return {
        "case_id": row["case_id"],
        "quality_rows": quality_rows,
        "path_rows": path_rows,
        "distribution_rows": distribution_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_panel(
    panel_path: Path,
    output_path: Path,
    panel_name: str,
    *,
    workers: int,
) -> None:
    panel = read_tsv(panel_path)
    checkpoint_root = Path(tempfile.gettempdir()) / "mdcorealign_cpu_chronology_v3_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    quality_checkpoint = checkpoint_root / output_path.name
    quality_source = quality_checkpoint if quality_checkpoint.exists() else output_path
    quality = read_tsv(quality_source) if quality_source.exists() else []
    completed = {
        row["case_id"]
        for row in quality
        if row.get("method") == "adaptive_C1_C2_D64_v3"
    }
    quality = [row for row in quality if row["case_id"] in completed]
    paths_file = OUT / "06_validation/adaptive_refinement_paths.tsv"
    paths_checkpoint = checkpoint_root / f"adaptive_refinement_paths_{panel_name}.tsv"
    paths_source = paths_checkpoint if paths_checkpoint.exists() else paths_file
    paths = read_tsv(paths_source) if paths_source.exists() else []
    paths = [row for row in paths if not (row.get("panel") == panel_name and row["case_id"] not in completed)]
    distributions_file = OUT / "06_validation/fresh_long_rmsd_distributions.tsv"
    distributions_checkpoint = checkpoint_root / "fresh_long_rmsd_distributions.tsv"
    distributions_source = distributions_checkpoint if distributions_checkpoint.exists() else distributions_file
    distributions = read_tsv(distributions_source) if distributions_source.exists() else []
    if panel_name == "fresh_long":
        distributions = [row for row in distributions if row["case_id"] in completed]
    pending = [row for row in panel if row["case_id"] not in completed]
    start = time.perf_counter()

    def save(result: dict[str, object], index: int) -> None:
        quality.extend(result["quality_rows"])
        paths.extend(result["path_rows"])
        if panel_name == "fresh_long":
            distributions.extend(result["distribution_rows"])
            write_tsv(distributions_checkpoint, distributions)
        write_tsv(quality_checkpoint, quality)
        write_tsv(paths_checkpoint, paths)
        print(
            f"[V5 {panel_name}] {len(completed) + index}/{len(panel)} "
            f"{result['case_id']} case={float(result['elapsed_seconds']):.1f}s "
            f"wall={time.perf_counter()-start:.1f}s",
            flush=True,
        )

    if workers == 1:
        for index, row in enumerate(pending, start=1):
            save(evaluate_case(row, panel_name), index)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_case, row, panel_name): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                save(future.result(), index)
    write_tsv(output_path, quality)
    write_tsv(paths_file, paths)
    if panel_name == "fresh_long":
        write_tsv(distributions_file, distributions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=("fresh_mdcath", "fresh_long", "historical"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = {
        "fresh_mdcath": (
            OUT / "05_fresh_panels/fresh_mdcath_512.tsv",
            OUT / "06_validation/fresh_mdcath_quality.tsv",
        ),
        "fresh_long": (
            OUT / "05_fresh_panels/fresh_long_panel.tsv",
            OUT / "06_validation/fresh_long_quality.tsv",
        ),
        "historical": (
            OUT / "05_fresh_panels/historical_stress_panel.tsv",
            OUT / "06_validation/historical_stress_quality.tsv",
        ),
    }
    panel_path, output_path = mapping[args.panel]
    if args.limit is not None:
        rows = read_tsv(panel_path)[: args.limit]
        limited = OUT / f"05_fresh_panels/.{args.panel}_limited.tsv"
        write_tsv(limited, rows)
        panel_path = limited
    run_panel(panel_path, output_path, args.panel, workers=args.workers)


if __name__ == "__main__":
    main()
