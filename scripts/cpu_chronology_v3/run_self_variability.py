#!/usr/bin/env python
"""Measure exact self-variability using the registered v3 metric contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory  # noqa: E402
from mdsuperpose.chronology_study import (  # noqa: E402
    CompactStudyResult,
    compare_results,
    representative_indices,
    run_exact,
)
from mdsuperpose.stage_timing import StageTiming  # noqa: E402


OUT = ROOT / "results/cpu_chronology_v3"
PANEL = OUT / "02_self_variability/calibration_panel.tsv"
OUTPUTS = {
    "numerical": OUT / "02_self_variability/numerical_repeatability.tsv",
    "mild_path": OUT / "02_self_variability/mild_path_variability.tsv",
    "contextual": OUT / "02_self_variability/contextual_sensitivity.tsv",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, row: dict[str, object]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_case(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / row["input_path"]
    if path.suffix == ".npy":
        coords = np.load(path, mmap_mode="r")
        chains = None
        metadata_name = path.name.replace("_CA_A_float64.npy", "_residue_metadata.npz")
        metadata = path.with_name(metadata_name)
        if metadata.exists():
            with np.load(metadata, allow_pickle=True) as archive:
                if "chain_ids" in archive:
                    chains = np.asarray(archive["chain_ids"]).astype(str)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            key = "coords" if "coords" in archive.files else archive.files[0]
            coords = np.asarray(archive[key], dtype=np.float64)
            chains = np.asarray(archive["chain_ids"]).astype(str) if "chain_ids" in archive.files else None
            if chains is None and "residue_metadata" in archive.files:
                metadata = list(archive["residue_metadata"])
                chains = np.asarray(
                    [str(item.get("chain", item.get("chain_id", "A"))) if isinstance(item, dict) else "A" for item in metadata]
                )
    elif path.suffix in {".h5", ".hdf5"} or row.get("input_key", "").startswith("/"):
        coords, _, metadata, _ = extract_ca_trajectory(
            path,
            coordinate_path=row["input_key"],
            contiguous_full_atom_read=True,
        )
        coords = np.asarray(coords, dtype=np.float64)
        chains = np.asarray([str(item.get("chain", item.get("chain_id", "A"))) for item in metadata])
    else:
        raise ValueError(f"unsupported input: {path}")
    if coords.shape != (int(row["n_frames"]), int(row["n_residues"]), 3):
        raise ValueError(f"shape mismatch for {row['case_id']}: {coords.shape}")
    if chains is None or chains.shape != (coords.shape[1],):
        chains = np.full(coords.shape[1], "A")
    if not np.isfinite(float(np.sum(coords, dtype=np.float64))):
        raise ValueError(f"non-finite coordinates: {row['case_id']}")
    return coords, chains


def save_snapshot(path: Path, result: CompactStudyResult) -> None:
    np.savez(
        path,
        weights=result.weights,
        rotations=result.rotations,
        translations=result.translations,
        rmsd=result.rmsd,
        weighted_rmsd=result.weighted_rmsd,
        robust_rmsf=result.robust_rmsf,
        mean_residual=result.mean_residual,
        runtime_seconds=np.asarray(result.runtime_seconds),
    )


def load_snapshot(path: Path) -> CompactStudyResult:
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    runtime = float(arrays.pop("runtime_seconds"))
    timing = StageTiming(
        end_to_end_total_seconds=runtime,
        weight_estimation_total_seconds=0.0,
        all_frame_total_seconds=0.0,
        result_construction_seconds=0.0,
    )
    return CompactStudyResult(
        method="exact_sequential_f64_compact_fresh_process",
        runtime_seconds=runtime,
        stage_timing=timing,
        metadata={"fresh_process": True},
        **arrays,
    )


def shifted_indices(indices: np.ndarray, n_frames: int, amount: int) -> np.ndarray:
    return (np.asarray(indices, dtype=int) + int(amount)) % int(n_frames)


def interleaved_blocks(indices: np.ndarray, block_size: int = 64) -> np.ndarray:
    blocks = [indices[first:first + block_size] for first in range(0, indices.size, block_size)]
    return np.concatenate(blocks[::2] + blocks[1::2])


def variants(coords: np.ndarray, chains: np.ndarray) -> list[tuple[str, str, CompactStudyResult]]:
    canonical_indices = representative_indices(coords.shape[0])
    gap = max(1, int(round(coords.shape[0] / canonical_indices.size)))
    output = [
        ("chunk_512", "numerical", run_exact(coords, chains, selected_indices=canonical_indices, chunk_size=512)),
        ("chunk_4096", "numerical", run_exact(coords, chains, selected_indices=canonical_indices, chunk_size=4096)),
        ("cyclic_block_shift", "mild_path", run_exact(coords, chains, selected_indices=np.roll(canonical_indices, -min(64, canonical_indices.size)))),
        ("interleaved_blocks", "mild_path", run_exact(coords, chains, selected_indices=interleaved_blocks(canonical_indices))),
        ("stratified_offset_half", "mild_path", run_exact(coords, chains, selected_indices=shifted_indices(canonical_indices, coords.shape[0], max(1, gap // 2)))),
        ("stratified_offset_full", "mild_path", run_exact(coords, chains, selected_indices=shifted_indices(canonical_indices, coords.shape[0], gap))),
        ("reverse_order", "contextual", run_exact(coords, chains, selected_indices=canonical_indices[::-1])),
        ("reference_middle", "contextual", run_exact(coords, chains, selected_indices=canonical_indices, reference_index=coords.shape[0] // 2)),
        ("reference_last", "contextual", run_exact(coords, chains, selected_indices=canonical_indices, reference_index=coords.shape[0] - 1)),
    ]
    alternate_count = max(32, min(canonical_indices.size - 1, int(round(canonical_indices.size * 0.75))))
    rng = np.random.default_rng(20260821 + coords.shape[0] + coords.shape[1])
    alternate = np.sort(rng.choice(coords.shape[0], size=alternate_count, replace=False))
    output.append(("substantial_subset", "contextual", run_exact(coords, chains, selected_indices=alternate)))
    return output


def comparison_row(
    panel_row: dict[str, str],
    perturbation: str,
    group: str,
    candidate: CompactStudyResult,
    reference: CompactStudyResult,
    coords: np.ndarray,
) -> dict[str, object]:
    return {
        "case_id": panel_row["case_id"],
        "family_id": panel_row["family_id"],
        "panel_role": panel_row["panel_role"],
        "perturbation": perturbation,
        "perturbation_class": group,
        "n_frames": int(panel_row["n_frames"]),
        "n_residues": int(panel_row["n_residues"]),
        **compare_results(candidate, reference, coords),
    }


def fresh_snapshot(panel: Path, case_id: str, output: Path) -> None:
    row = next(row for row in read_tsv(panel) if row["case_id"] == case_id)
    coords, chains = load_case(row)
    save_snapshot(output, run_exact(coords, chains))


def aggregate() -> None:
    summary = []
    for group, path in OUTPUTS.items():
        rows = read_tsv(path)
        def values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "perturbation_class": group,
                "case_perturbation_rows": len(rows),
                "unique_cases": len({row["case_id"] for row in rows}),
                "median_of_case_level_median_rmsd_difference_A": float(np.median(values("rmsd_absolute_difference_median_A"))),
                "p95_across_case_level_p95_rmsd_difference_A": float(np.quantile(values("rmsd_absolute_difference_p95_A"), 0.95)),
                "maximum_case_level_p95_rmsd_difference_A": float(np.max(values("rmsd_absolute_difference_p95_A"))),
                "median_core_jaccard": float(np.median(values("core_jaccard"))),
                "p5_core_jaccard": float(np.quantile(values("core_jaccard"), 0.05)),
                "minimum_core_jaccard": float(np.min(values("core_jaccard"))),
                "median_weight_spearman": float(np.median(values("weight_spearman"))),
                "p5_weight_spearman": float(np.quantile(values("weight_spearman"), 0.05)),
                "median_rmsf_spearman": float(np.median(values("stable_core_rmsf_spearman"))),
                "p5_rmsf_spearman": float(np.quantile(values("stable_core_rmsf_spearman"), 0.05)),
                "median_rmsf_mae_A": float(np.median(values("stable_core_rmsf_mae_A"))),
                "catastrophic_mismatch_count": sum(row["catastrophic_mismatch"] == "True" for row in rows),
            }
        )
    write_tsv(OUT / "02_self_variability/self_variability_summary.tsv", summary)
    by_name = {row["perturbation_class"]: row for row in summary}
    report = f"""# Exact self-variability report

The canonical chronological float64 estimator was compared with numerical,
mild path and contextual perturbations using the same core, transform, RMSD and
RMSF implementation used for candidate evaluation.

| Class | Rows | Median of case medians (A) | P95 across case P95 (A) | Maximum case P95 (A) | Median core Jaccard |
|---|---:|---:|---:|---:|---:|
| Numerical | {by_name['numerical']['case_perturbation_rows']} | {float(by_name['numerical']['median_of_case_level_median_rmsd_difference_A']):.6g} | {float(by_name['numerical']['p95_across_case_level_p95_rmsd_difference_A']):.6g} | {float(by_name['numerical']['maximum_case_level_p95_rmsd_difference_A']):.6g} | {float(by_name['numerical']['median_core_jaccard']):.6f} |
| Mild path | {by_name['mild_path']['case_perturbation_rows']} | {float(by_name['mild_path']['median_of_case_level_median_rmsd_difference_A']):.6g} | {float(by_name['mild_path']['p95_across_case_level_p95_rmsd_difference_A']):.6g} | {float(by_name['mild_path']['maximum_case_level_p95_rmsd_difference_A']):.6g} | {float(by_name['mild_path']['median_core_jaccard']):.6f} |
| Contextual | {by_name['contextual']['case_perturbation_rows']} | {float(by_name['contextual']['median_of_case_level_median_rmsd_difference_A']):.6g} | {float(by_name['contextual']['p95_across_case_level_p95_rmsd_difference_A']):.6g} | {float(by_name['contextual']['maximum_case_level_p95_rmsd_difference_A']):.6g} | {float(by_name['contextual']['median_core_jaccard']):.6f} |

Contextual reference and major subset sensitivity is reported separately and is
not used to inflate the v3 acceptance threshold. Absolute case-level P95 values,
not signed mean changes, define the mild-path calibration envelope.
"""
    (OUT / "02_self_variability/self_variability_report.md").write_text(report)


def run() -> None:
    panel = read_tsv(PANEL)
    completed = set()
    for group, path in OUTPUTS.items():
        if path.exists() and path.stat().st_size:
            completed.update((row["case_id"], row["perturbation"]) for row in read_tsv(path))
    start = time.perf_counter()
    for index, row in enumerate(panel, start=1):
        expected = {"fresh_process_repeat", "chunk_512", "chunk_4096", "cyclic_block_shift", "interleaved_blocks", "stratified_offset_half", "stratified_offset_full", "reverse_order", "reference_middle", "reference_last", "substantial_subset"}
        if all((row["case_id"], name) in completed for name in expected):
            continue
        coords, chains = load_case(row)
        canonical = run_exact(coords, chains)
        if (row["case_id"], "fresh_process_repeat") not in completed:
            with tempfile.TemporaryDirectory(prefix="mdca_v3_fresh_") as temporary:
                snapshot = Path(temporary) / "result.npz"
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "snapshot", "--panel", str(PANEL), "--case-id", row["case_id"], "--output", str(snapshot)],
                    check=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                    stdout=subprocess.DEVNULL,
                )
                repeated = load_snapshot(snapshot)
            append_tsv(OUTPUTS["numerical"], comparison_row(row, "fresh_process_repeat", "numerical", repeated, canonical, coords))
        for name, group, candidate in variants(coords, chains):
            if (row["case_id"], name) in completed:
                continue
            append_tsv(OUTPUTS[group], comparison_row(row, name, group, candidate, canonical, coords))
        elapsed = time.perf_counter() - start
        print(f"[V1] {index}/{len(panel)} {row['case_id']} elapsed={elapsed:.1f}s", flush=True)
    aggregate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--panel", type=Path, required=True)
    snapshot.add_argument("--case-id", required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        fresh_snapshot(args.panel, args.case_id, args.output)
    else:
        run()


if __name__ == "__main__":
    main()
