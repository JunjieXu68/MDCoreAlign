#!/usr/bin/env python3
"""Run G4 matched-output and persistent-batch benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import tempfile
import time

import numpy as np

from mdcorealign.guarded import (
    GuardedBatchWorker,
    solve_guarded_fast_global_alignment,
)
from mdcorealign.pdb_writer import write_multimodel_ca_pdb_streaming
from mdsuperpose.benchmark import benchmark_external_mdlovofit
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.mdlovofit import write_multimodel_pdb
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.scaling_optimization_v5.run_phase8b_runtime_boundaries import (
    run_common_boundary,
    select_canonical_subset,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/guarded_fast_global_v1/g4_matched_contracts"
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
MDLOVOFIT = Path("/usr/local/bin/mdlovofit")
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def internal_alignment(
    method: str,
    coords: np.ndarray,
    chain_ids: np.ndarray,
) -> tuple[object, float]:
    start = time.perf_counter()
    if method == "sequential_lean_exact":
        metrics, fit = run_common_boundary(
            "sequential_lean_exact",
            coords,
            chain_ids,
        )
        result = {
            "rotations": fit.rotations,
            "translations": fit.translations,
            "rmsd": float(metrics["mean_unified_rmsd_A"]),
            "weighted_rmsd": float(np.mean(fit.weighted_rmsd)),
            "accepted_level": 3,
            "fallback_reason": "",
            "final_quality_status": "exact_reference",
        }
    elif method == "guarded_fast_global":
        guarded = solve_guarded_fast_global_alignment(
            coords,
            chain_ids=chain_ids,
            store_aligned=False,
        )
        result = {
            "rotations": guarded.rotations,
            "translations": guarded.translations,
            "rmsd": guarded.mean_rmsd,
            "weighted_rmsd": guarded.mean_weighted_rmsd,
            "accepted_level": guarded.accepted_level,
            "fallback_reason": guarded.fallback_reason or "",
            "final_quality_status": guarded.final_quality_status,
        }
    else:
        raise ValueError(method)
    return result, time.perf_counter() - start


def matched_contracts() -> None:
    """Run contracts A/B on the canonical panel."""

    panel = select_canonical_subset()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="mdca_g4_contracts_",
        dir=OUTPUT,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        for panel_row in panel:
            coords, labels, metadata_rows, metadata = extract_ca_trajectory(
                Path(panel_row["h5_path"]),
                coordinate_path=panel_row["coordinate_path"],
                min_ca_atoms=30,
                min_frames=20,
            )
            coords = np.asarray(coords, dtype=np.float64)
            chains = np.asarray(
                [row.get("chain", "_") for row in metadata_rows],
                dtype=object,
            )
            for method in (
                "sequential_lean_exact",
                "guarded_fast_global",
                "external_mdlovofit",
            ):
                for iteration in range(4):
                    warmup = iteration == 0
                    case_root = (
                        temporary
                        / str(panel_row["domain_id"])
                        / method
                        / str(iteration)
                    )
                    case_root.mkdir(parents=True, exist_ok=True)
                    try:
                        if method == "external_mdlovofit":
                            external = benchmark_external_mdlovofit(
                                coords,
                                fraction=0.6,
                                workdir=case_root,
                                executable=MDLOVOFIT,
                                timeout_seconds=300,
                            )
                            algorithm_seconds = float(
                                external.extra_metadata[
                                    "subprocess_runtime_seconds"
                                ]
                            )
                            full_seconds = external.runtime_seconds
                            materialization_seconds = (
                                external.extra_metadata["input_write_seconds"]
                                + external.extra_metadata["output_read_seconds"]
                            )
                            mean_rmsd = external.rmsd
                            mean_weighted_rmsd = ""
                            accepted_level = ""
                            fallback_reason = ""
                            quality = (
                                "external_parse_success"
                                if external.parsing_success
                                else "external_parse_failure"
                            )
                            status = (
                                "success"
                                if external.parsing_success
                                else "failed"
                            )
                            transform_available = False
                        else:
                            internal, algorithm_seconds = internal_alignment(
                                method,
                                coords,
                                chains,
                            )
                            materialize_start = time.perf_counter()
                            aligned = apply_transforms_batch(
                                coords,
                                internal["rotations"],
                                internal["translations"],
                            )
                            write_multimodel_ca_pdb_streaming(
                                aligned,
                                case_root / "aligned.pdb",
                            )
                            materialization_seconds = (
                                time.perf_counter() - materialize_start
                            )
                            full_seconds = (
                                algorithm_seconds + materialization_seconds
                            )
                            mean_rmsd = internal["rmsd"]
                            mean_weighted_rmsd = internal["weighted_rmsd"]
                            accepted_level = internal["accepted_level"]
                            fallback_reason = internal["fallback_reason"]
                            quality = internal["final_quality_status"]
                            status = "success"
                            transform_available = True
                    except Exception as exc:
                        algorithm_seconds = ""
                        materialization_seconds = ""
                        full_seconds = ""
                        mean_rmsd = ""
                        mean_weighted_rmsd = ""
                        accepted_level = ""
                        fallback_reason = f"{type(exc).__name__}: {exc}"
                        quality = "failed"
                        status = "failed"
                        transform_available = False
                    if warmup:
                        continue
                    common = {
                        "domain_id": panel_row["domain_id"],
                        "cath_class": panel_row["cath_class"],
                        "method": method,
                        "repeat": iteration,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "fit_selection": "CA",
                        "coordinate_unit": "angstrom",
                        "accepted_level": accepted_level,
                        "fallback_reason": fallback_reason,
                        "final_quality_status": quality,
                        "mean_unified_RMSD_A": mean_rmsd,
                        "mean_weighted_RMSD_A": mean_weighted_rmsd,
                        "status": status,
                        "process_peak_RSS_mb": peak_rss_mb(),
                    }
                    rows.append(
                        {
                            **common,
                            "contract": "A_metrics_transforms",
                            "timing_boundary": (
                                "loaded_coordinates_to_metrics_and_transforms"
                            ),
                            "runtime_seconds": algorithm_seconds,
                            "algorithm_runtime_seconds": algorithm_seconds,
                            "materialization_write_seconds": 0.0,
                            "transform_available": transform_available,
                            "external_boundary_caveat": (
                                "subprocess_includes_native_output_generation"
                                if method == "external_mdlovofit"
                                else ""
                            ),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "contract": "B_materialized_aligned_PDB",
                            "timing_boundary": (
                                "loaded_coordinates_to_aligned_CA_PDB"
                            ),
                            "runtime_seconds": full_seconds,
                            "algorithm_runtime_seconds": algorithm_seconds,
                            "materialization_write_seconds": (
                                materialization_seconds
                            ),
                            "transform_available": transform_available,
                            "external_boundary_caveat": "",
                        }
                    )
    write_tsv(OUTPUT / "matched_output_runtime.tsv", rows)


def synthetic_batch() -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(3204)
    reference = np.cumsum(rng.normal(size=(60, 3)), axis=0)
    trajectories: list[np.ndarray] = []
    for case in range(1000):
        coords = np.repeat(reference[None, :, :], 12, axis=0)
        coords += rng.normal(scale=0.01, size=coords.shape)
        coords[:, 24:32] += rng.normal(
            scale=0.08 + 0.002 * (case % 25),
            size=(12, 8, 3),
        )
        trajectories.append(coords)
    return trajectories, np.asarray(["A"] * 60, dtype=object)


def persistent_batch() -> None:
    """Benchmark persistent batches at the requested sizes."""

    trajectories, chains = synthetic_batch()
    rows: list[dict[str, object]] = []
    for batch_size in (1, 10, 100, 1000):
        repeats = 3 if batch_size <= 100 else 1
        batch = trajectories[:batch_size]
        for method in (
            "sequential_lean_exact",
            "guarded_fast_global",
            "external_mdlovofit",
        ):
            for repeat in range(1, repeats + 1):
                worker = (
                    GuardedBatchWorker.create(
                        12,
                        60,
                        chain_ids=chains,
                        level1_representative_count=12,
                        level2_representative_count=12,
                        exact_representative_count=12,
                    )
                    if method == "guarded_fast_global"
                    else None
                )
                fallback_count = 0
                status = "success"
                error = ""
                start = time.perf_counter()
                try:
                    if method == "sequential_lean_exact":
                        for coords in batch:
                            run_common_boundary(
                                "sequential_lean_exact",
                                coords,
                                chains,
                            )
                    elif method == "guarded_fast_global":
                        assert worker is not None
                        for coords in batch:
                            result = worker.align(coords)
                            fallback_count += int(result.accepted_level == 3)
                    else:
                        with tempfile.TemporaryDirectory(
                            prefix="mdca_g4_batch_",
                            dir=OUTPUT,
                        ) as tmpdir:
                            root = Path(tmpdir)
                            for index, coords in enumerate(batch):
                                result = benchmark_external_mdlovofit(
                                    coords,
                                    fraction=0.6,
                                    workdir=root / str(index),
                                    executable=MDLOVOFIT,
                                    timeout_seconds=60,
                                )
                                if not result.parsing_success:
                                    raise RuntimeError(
                                        f"MDLovoFit parse failed at {index}"
                                    )
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "method": method,
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "status": status,
                        "runtime_seconds": elapsed,
                        "warm_per_trajectory_seconds": (
                            elapsed / batch_size
                        ),
                        "trajectories_per_second": batch_size / elapsed,
                        "fallback_count": fallback_count,
                        "fallback_rate": fallback_count / batch_size,
                        "process_peak_RSS_mb": peak_rss_mb(),
                        "persistent_process": True,
                        "reused_coordinate_buffer": (
                            worker is not None
                        ),
                        "reused_representative_index_buffers": (
                            worker is not None
                        ),
                        "reused_kernel_temporary_arrays": False,
                        "error": error,
                    }
                )
    write_tsv(OUTPUT / "persistent_batch_throughput.tsv", rows)
    write_json(
        OUTPUT / "persistent_worker_contract.json",
        GuardedBatchWorker.create(
            12,
            60,
            chain_ids=chains,
            level1_representative_count=12,
            level2_representative_count=12,
            exact_representative_count=12,
        ).cache_summary,
    )


def writer_benchmark() -> None:
    """Compare legacy and streaming writers without changing PDB coordinates."""

    rng = np.random.default_rng(747)
    coords = rng.normal(size=(1000, 153, 3))
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="mdca_g4_writer_",
        dir=OUTPUT,
    ) as tmpdir:
        root = Path(tmpdir)
        for method in ("legacy_per_atom", "streaming_reused_buffer"):
            for repeat in range(1, 6):
                path = root / f"{method}.pdb"
                start = time.perf_counter()
                if method == "legacy_per_atom":
                    write_multimodel_pdb(coords, path)
                else:
                    write_multimodel_ca_pdb_streaming(coords, path)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "method": method,
                        "repeat": repeat,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "runtime_seconds": elapsed,
                        "output_bytes": path.stat().st_size,
                        "coordinate_text_contract": "PDB_8.3_CA_only",
                        "process_peak_RSS_mb": peak_rss_mb(),
                    }
                )
        legacy = (root / "legacy_per_atom.pdb").read_bytes()
        streaming = (root / "streaming_reused_buffer.pdb").read_bytes()
        byte_identical = legacy == streaming
    write_tsv(OUTPUT / "writer_optimization.tsv", rows)
    write_json(
        OUTPUT / "writer_coordinate_integrity.json",
        {
            "legacy_and_streaming_files_byte_identical": byte_identical,
            "coordinates_altered": not byte_identical,
        },
    )


def child(method: str) -> None:
    """Run one fresh-process tiny trajectory call."""

    trajectories, chains = synthetic_batch()
    coords = trajectories[0]
    start = time.perf_counter()
    if method == "sequential_lean_exact":
        run_common_boundary("sequential_lean_exact", coords, chains)
        accepted_level = 3
    elif method == "guarded_fast_global":
        result = solve_guarded_fast_global_alignment(
            coords,
            chain_ids=chains,
        )
        accepted_level = result.accepted_level
    else:
        with tempfile.TemporaryDirectory(prefix="mdca_g4_child_") as tmpdir:
            result = benchmark_external_mdlovofit(
                coords,
                fraction=0.6,
                workdir=Path(tmpdir),
                executable=MDLOVOFIT,
                timeout_seconds=60,
            )
            if not result.parsing_success:
                raise RuntimeError("MDLovoFit parse failure")
        accepted_level = ""
    print(
        json.dumps(
            {
                "method": method,
                "child_compute_seconds": time.perf_counter() - start,
                "accepted_level": accepted_level,
                "peak_RSS_mb": peak_rss_mb(),
            }
        )
    )


def cold_start() -> None:
    rows: list[dict[str, object]] = []
    environment = {**os.environ, **THREAD_ENV}
    for method in (
        "sequential_lean_exact",
        "guarded_fast_global",
        "external_mdlovofit",
    ):
        for repeat in range(1, 4):
            start = time.perf_counter()
            completed = subprocess.run(
                [
                    str(PYTHON),
                    str(Path(__file__).resolve()),
                    "child",
                    "--method",
                    method,
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            wall = time.perf_counter() - start
            payload = json.loads(completed.stdout)
            rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "cold_start_wall_seconds": wall,
                    **payload,
                }
            )
    write_tsv(OUTPUT / "cold_start_latency.tsv", rows)


def _median_summary(
    rows: list[dict[str, str]],
    group_fields: tuple[str, ...],
    runtime_field: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status", "success") != "success":
            continue
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        values = [float(row[runtime_field]) for row in group]
        output.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "n": len(values),
                "median_runtime_seconds": statistics.median(values),
                "q1_runtime_seconds": float(np.quantile(values, 0.25)),
                "q3_runtime_seconds": float(np.quantile(values, 0.75)),
            }
        )
    return output


def finalize() -> None:
    matched = read_tsv(OUTPUT / "matched_output_runtime.tsv")
    batches = read_tsv(OUTPUT / "persistent_batch_throughput.tsv")
    for row in batches:
        if row["method"] == "sequential_lean_exact":
            row["final_quality_status"] = "exact_reference"
        elif row["method"] == "external_mdlovofit":
            row["final_quality_status"] = (
                "external_parse_success"
                if row["status"] == "success"
                else "external_failure"
            )
        else:
            fallback_count = int(row["fallback_count"])
            batch_size = int(row["batch_size"])
            if fallback_count == 0:
                row["final_quality_status"] = (
                    "level1_certificate_accepted_not_strict_exact_parity"
                )
            elif fallback_count == batch_size:
                row["final_quality_status"] = (
                    "all_sequential_lean_exact_fallback"
                )
            else:
                row["final_quality_status"] = (
                    "mixed_level1_and_exact_fallback"
                )
    write_tsv(OUTPUT / "persistent_batch_throughput.tsv", batches)
    writers = read_tsv(OUTPUT / "writer_optimization.tsv")
    cold = read_tsv(OUTPUT / "cold_start_latency.tsv")
    write_tsv(
        OUTPUT / "matched_output_runtime_summary.tsv",
        _median_summary(
            matched,
            ("contract", "method"),
            "runtime_seconds",
        ),
    )
    write_tsv(
        OUTPUT / "persistent_batch_throughput_summary.tsv",
        _median_summary(
            batches,
            ("batch_size", "method"),
            "runtime_seconds",
        ),
    )
    batch_groups: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = {}
    for row in batches:
        if row["status"] == "success":
            batch_groups.setdefault(
                (row["batch_size"], row["method"]),
                [],
            ).append(row)
    batch_detail = []
    for (batch_size, method), group in sorted(batch_groups.items()):
        batch_detail.append(
            {
                "batch_size": batch_size,
                "method": method,
                "n_repeats": len(group),
                "median_total_runtime_seconds": statistics.median(
                    float(row["runtime_seconds"]) for row in group
                ),
                "median_warm_per_trajectory_seconds": statistics.median(
                    float(row["warm_per_trajectory_seconds"])
                    for row in group
                ),
                "median_trajectories_per_second": statistics.median(
                    float(row["trajectories_per_second"])
                    for row in group
                ),
                "max_process_peak_RSS_mb": max(
                    float(row["process_peak_RSS_mb"]) for row in group
                ),
                "median_fallback_rate": statistics.median(
                    float(row["fallback_rate"]) for row in group
                ),
                "final_quality_status": ";".join(
                    sorted(
                        {
                            row["final_quality_status"]
                            for row in group
                        }
                    )
                ),
            }
        )
    write_tsv(
        OUTPUT / "persistent_batch_detailed_summary.tsv",
        batch_detail,
    )
    write_tsv(
        OUTPUT / "writer_optimization_summary.tsv",
        _median_summary(writers, ("method",), "runtime_seconds"),
    )
    write_tsv(
        OUTPUT / "cold_start_latency_summary.tsv",
        _median_summary(cold, ("method",), "cold_start_wall_seconds"),
    )
    matched_summary = read_tsv(
        OUTPUT / "matched_output_runtime_summary.tsv"
    )
    writer_summary = {
        row["method"]: float(row["median_runtime_seconds"])
        for row in read_tsv(
            OUTPUT / "writer_optimization_summary.tsv"
        )
    }
    writer_speedup = (
        writer_summary["legacy_per_atom"]
        / writer_summary["streaming_reused_buffer"]
    )
    report = f"""# G4 matched-output and batch-throughput report

Contracts A (metrics/transforms) and B (materialized aligned CA PDB) were
measured separately on the canonical panel. MDLovoFit cannot expose a pure
numerical-transform boundary through its public executable, so its Contract A
subprocess timing necessarily includes native output generation and is marked
accordingly; no stronger matched-kernel claim is made.

The streaming writer was {writer_speedup:.3f}x the legacy writer on the
1000-frame/153-residue writer case. The generated files were byte-identical,
so coordinates and PDB text fields were not changed.

Persistent batch measurements include batch sizes 1, 10, 100 and 1,000.
The guarded worker reuses imports, thread pools, chain metadata, one coordinate
input buffer and representative-index buffers. NumPy Kabsch/SVD temporary
arrays remain call-local, so this is not a zero-allocation implementation.

Guarded-v1 remains experimental because G3 did not meet the strict held-out
parity contract or fallback-rate target. Results do not support a universal
fastest claim.
"""
    (OUTPUT / "g4_matched_contracts_report.md").write_text(report)
    write_json(
        OUTPUT / "g4_status.json",
        {
            "status": "complete",
            "matched_rows": len(matched),
            "batch_rows": len(batches),
            "writer_rows": len(writers),
            "cold_start_rows": len(cold),
            "writer_speedup": writer_speedup,
            "writer_byte_identity": json.loads(
                (OUTPUT / "writer_coordinate_integrity.json").read_text()
            )["legacy_and_streaming_files_byte_identical"],
            "universal_fastest_claimed": False,
            "zero_allocation_claimed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "matched-contracts",
            "persistent-batch",
            "writer",
            "cold-start",
            "child",
            "finalize",
        ),
    )
    parser.add_argument(
        "--method",
        choices=(
            "sequential_lean_exact",
            "guarded_fast_global",
            "external_mdlovofit",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "matched-contracts":
        matched_contracts()
    elif args.stage == "persistent-batch":
        persistent_batch()
    elif args.stage == "writer":
        writer_benchmark()
    elif args.stage == "cold-start":
        cold_start()
    elif args.stage == "child":
        if args.method is None:
            raise ValueError("--method is required for child stage")
        child(args.method)
    else:
        finalize()


if __name__ == "__main__":
    main()
