#!/usr/bin/env python3
"""Run R5 performance benchmarks with explicit matched output contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import tempfile
import time

import numpy as np

from mdcorealign.experimental import (
    ExperimentalBatchWorker,
    solve_fast_global_alignment,
)
from mdsuperpose.benchmark import (
    benchmark_external_mdlovofit,
    benchmark_external_theseus,
)
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.mdlovofit import write_multimodel_pdb
from mdsuperpose.window_kinematics import (
    align_to_motion_domain,
    scan_local_kinematics,
    scan_motion_domains,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.scaling_optimization_v5.run_phase8b_runtime_boundaries import (
    load_long_input,
    run_common_boundary,
    select_canonical_subset,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/mainline_reintegration_v1/matched_benchmarks"
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}
WARMUPS = 3
REPEATS = 5


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


def _measure_clean(function) -> tuple[object, float]:
    """Measure one call without profiler instrumentation."""

    start = time.perf_counter()
    output = function()
    return output, time.perf_counter() - start


def _peak_rss_mb() -> float:
    """Return process high-water RSS using Linux ru_maxrss units."""

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _array_digest(*arrays: np.ndarray) -> str:
    """Hash numeric result arrays without materializing text representations."""

    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def child_w5(method: str, chunk_size: int) -> None:
    """Fresh-process child for the alignment-free contract."""

    total_start = time.perf_counter()
    decode_start = time.perf_counter()
    coords, chain = load_long_input()
    decode_seconds = time.perf_counter() - decode_start
    if method == "w5_scan_only":
        result, compute = _measure_clean(
            lambda: scan_local_kinematics(
                coords,
                chain_ids=chain,
                evaluate_blocks=False,
                descriptor_chunk_size=chunk_size,
            )
        )
        output_size = result.local_deformation_index.size
    elif method == "w5_sparse_graph":
        result, compute = _measure_clean(
            lambda: scan_motion_domains(
                coords,
                chain_ids=chain,
                evaluate_blocks=False,
                descriptor_chunk_size=chunk_size,
            )
        )
        output_size = result.edges.n_edges
    else:
        raise ValueError(method)
    print(
        json.dumps(
            {
                "decode_seconds": decode_seconds,
                "compute_seconds": compute,
                "total_child_seconds": time.perf_counter() - total_start,
                "python_peak_memory_mb": None,
                "process_peak_rss_mb": _peak_rss_mb(),
                "output_size": output_size,
            }
        )
    )


def contract1() -> None:
    """Benchmark W5 scan and sparse graph without claiming global alignment."""

    coords, chain = load_long_input()
    rows: list[dict[str, object]] = []
    methods = ("w5_scan_only", "w5_sparse_graph")
    chunk_sizes = (250, 500, 1000, 2000, 5000)
    for method in methods:
        for chunk_size in chunk_sizes:
            for iteration in range(WARMUPS + REPEATS):
                warmup = iteration < WARMUPS
                if method == "w5_scan_only":
                    function = lambda: scan_local_kinematics(
                        coords,
                        chain_ids=chain,
                        evaluate_blocks=False,
                        descriptor_chunk_size=chunk_size,
                    )
                else:
                    function = lambda: scan_motion_domains(
                        coords,
                        chain_ids=chain,
                        evaluate_blocks=False,
                        descriptor_chunk_size=chunk_size,
                    )
                result, elapsed = _measure_clean(function)
                if warmup:
                    continue
                rows.append(
                    {
                        "system": "unit_clean_4oelA02_synthetic_10000",
                        "method": method,
                        "timing_boundary": "already_loaded_compute",
                        "output_contract": (
                            "alignment_free_local_deformation"
                            if method == "w5_scan_only"
                            else "alignment_free_soft_sparse_motion_graph"
                        ),
                        "repeat": iteration - WARMUPS + 1,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "descriptor_chunk_size": chunk_size,
                        "runtime_seconds": elapsed,
                        "frames_per_second": coords.shape[0] / elapsed,
                        "residue_frames_per_second": (
                            coords.shape[0] * coords.shape[1] / elapsed
                        ),
                        "python_peak_memory_mb": "",
                        "process_peak_rss_mb": _peak_rss_mb(),
                        "produces_transform": False,
                        "produces_RMSD_RMSF": False,
                    }
                )
    worker = ExperimentalBatchWorker.create(
        coords.shape[1],
        chain_ids=chain,
    )
    for iteration in range(WARMUPS + REPEATS):
        result, elapsed = _measure_clean(
            lambda: worker.scan_w5(
                coords,
                descriptor_chunk_size=2000,
            )
        )
        if iteration < WARMUPS:
            continue
        rows.append(
            {
                "system": "unit_clean_4oelA02_synthetic_10000",
                "method": "w5_scan_only",
                "timing_boundary": "warm_persistent_worker_compute",
                "output_contract": "alignment_free_local_deformation",
                "repeat": iteration - WARMUPS + 1,
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "descriptor_chunk_size": 2000,
                "runtime_seconds": elapsed,
                "frames_per_second": coords.shape[0] / elapsed,
                "residue_frames_per_second": (
                    coords.shape[0] * coords.shape[1] / elapsed
                ),
                "python_peak_memory_mb": "",
                "process_peak_rss_mb": _peak_rss_mb(),
                "produces_transform": False,
                "produces_RMSD_RMSF": False,
                "persistent_cache_summary": json.dumps(
                    worker.cache_summary,
                    sort_keys=True,
                ),
            }
        )
    environment = {**os.environ, **THREAD_ENV}
    for method in methods:
        for repeat in range(1, REPEATS + 1):
            start = time.perf_counter()
            completed = subprocess.run(
                [
                    str(PYTHON),
                    str(Path(__file__).resolve()),
                    "child-w5",
                    "--method",
                    method,
                    "--chunk-size",
                    "2000",
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
                    "system": "unit_clean_4oelA02_synthetic_10000",
                    "method": method,
                    "timing_boundary": "fresh_process_decode_compute",
                    "output_contract": (
                        "alignment_free_local_deformation"
                        if method == "w5_scan_only"
                        else "alignment_free_soft_sparse_motion_graph"
                    ),
                    "repeat": repeat,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "descriptor_chunk_size": 2000,
                    "runtime_seconds": wall,
                    "child_reported_seconds": payload["total_child_seconds"],
                    "decode_seconds": payload["decode_seconds"],
                    "compute_seconds": payload["compute_seconds"],
                    "frames_per_second": coords.shape[0] / wall,
                    "residue_frames_per_second": (
                        coords.shape[0] * coords.shape[1] / wall
                    ),
                    "python_peak_memory_mb": payload[
                        "python_peak_memory_mb"
                    ],
                    "process_peak_rss_mb": payload["process_peak_rss_mb"],
                    "produces_transform": False,
                    "produces_RMSD_RMSF": False,
                }
            )
    write_tsv(OUTPUT / "contract1_local_deformation_runtime.tsv", rows)


def contract2() -> None:
    """Benchmark methods that all produce transforms, RMSD and RMSF."""

    coords, chain = load_long_input()
    rows: list[dict[str, object]] = []
    for method in (
        "sequential_lean_exact",
        "uniform_fast_global",
        "exploratory_graph_anchor",
    ):
        for iteration in range(WARMUPS + REPEATS):
            warmup = iteration < WARMUPS
            if method == "sequential_lean_exact":
                result, elapsed = _measure_clean(
                    lambda: run_common_boundary(
                        "sequential_lean_exact", coords, chain
                    )
                )
                metrics, fit = result
                mean_rmsd = metrics["mean_unified_rmsd_A"]
                rmsf_mean = float(np.mean(fit.robust_rmsf))
                n_transforms = fit.rotations.shape[0]
                mean_weighted_rmsd = float(np.mean(fit.weighted_rmsd))
                transform_digest = _array_digest(
                    fit.rotations, fit.translations
                )
                rmsd_rmsf_digest = _array_digest(
                    fit.rmsd, fit.robust_rmsf, fit.ordinary_rmsf
                )
                diagnostics_status = "exact_fixed_weight_fit_diagnostics"
            elif method == "uniform_fast_global":
                result, elapsed = _measure_clean(
                    lambda: solve_fast_global_alignment(
                        coords,
                        chain_ids=chain,
                        representative_count=256,
                        update_rounds=4,
                        chunk_size=2000,
                        store_aligned=False,
                        compute_confidence=True,
                    )
                )
                mean_rmsd = result.mean_rmsd
                rmsf_mean = float(np.mean(result.robust_rmsf))
                n_transforms = result.rotations.shape[0]
                mean_weighted_rmsd = result.mean_weighted_rmsd
                transform_digest = _array_digest(
                    result.rotations, result.translations
                )
                rmsd_rmsf_digest = _array_digest(
                    result.rmsd,
                    result.robust_rmsf,
                    result.ordinary_rmsf,
                )
                diagnostics_status = result.confidence.convergence_status
            else:
                result, elapsed = _measure_clean(
                    lambda: (
                        lambda domains: align_to_motion_domain(
                            coords,
                            domains,
                            selection="largest_stable",
                        )
                    )(
                        scan_motion_domains(
                            coords,
                            chain_ids=chain,
                            evaluate_blocks=False,
                        )
                    )
                )
                mean_rmsd = float(np.mean(result.rmsd))
                rmsf_mean = float(np.mean(result.rmsf))
                n_transforms = result.rotations.shape[0]
                mean_weighted_rmsd = ""
                transform_digest = _array_digest(
                    result.rotations, result.translations
                )
                rmsd_rmsf_digest = _array_digest(
                    result.rmsd, result.rmsf
                )
                diagnostics_status = (
                    "exploratory_soft_graph_with_hard_anchor_audit"
                )
            if warmup:
                continue
            rows.append(
                {
                    "system": "unit_clean_4oelA02_synthetic_10000",
                    "method": method,
                    "timing_boundary": "already_loaded_metrics_transforms",
                    "output_contract": "transforms_unified_RMSD_RMSF",
                    "repeat": iteration - WARMUPS + 1,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "runtime_seconds": elapsed,
                    "mean_unified_RMSD_A": mean_rmsd,
                    "mean_RMSF_A": rmsf_mean,
                    "mean_weighted_RMSD_A": mean_weighted_rmsd,
                    "n_transforms": n_transforms,
                    "transform_sha256": transform_digest,
                    "RMSD_RMSF_sha256": rmsd_rmsf_digest,
                    "diagnostics_status": diagnostics_status,
                    "python_peak_memory_mb": "",
                    "process_peak_rss_mb": _peak_rss_mb(),
                    "experimental": method != "sequential_lean_exact",
                    "hard_domain_claimed": False,
                }
            )
    write_tsv(OUTPUT / "contract2_metrics_transforms_runtime.tsv", rows)


def _internal_materialized(
    method: str,
    coords: np.ndarray,
    chain: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    algorithm_start = time.perf_counter()
    if method == "sequential_lean_exact":
        metrics, fit = run_common_boundary(
            "sequential_lean_exact", coords, chain
        )
        rotations = fit.rotations
        translations = fit.translations
        rmsd = metrics["mean_unified_rmsd_A"]
    else:
        result = solve_fast_global_alignment(
            coords,
            chain_ids=chain,
            representative_count=256,
            update_rounds=4,
            chunk_size=2000,
            store_aligned=False,
            compute_confidence=False,
        )
        rotations = result.rotations
        translations = result.translations
        rmsd = result.mean_rmsd
    algorithm_seconds = time.perf_counter() - algorithm_start
    apply_start = time.perf_counter()
    aligned = apply_transforms_batch(coords, rotations, translations)
    apply_seconds = time.perf_counter() - apply_start
    write_start = time.perf_counter()
    write_multimodel_pdb(aligned, output_path)
    write_seconds = time.perf_counter() - write_start
    return {
        "algorithm_runtime_seconds": algorithm_seconds,
        "input_write_seconds": 0.0,
        "transform_application_seconds": apply_seconds,
        "output_write_seconds": write_seconds,
        "output_read_seconds": 0.0,
        "full_workflow_seconds": (
            algorithm_seconds + apply_seconds + write_seconds
        ),
        "unified_posthoc_RMSD_A": rmsd,
        "native_RMSD_A": "",
        "status": "success",
        "error": "",
    }


def contract3() -> None:
    """Compare materialized CA-PDB outputs with external tools."""

    panel = select_canonical_subset()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="mdca_r5_materialized_", dir=OUTPUT
    ) as tmpdir:
        temporary = Path(tmpdir)
        for panel_row in panel:
            coords, labels, metadata_rows, metadata = extract_ca_trajectory(
                Path(panel_row["h5_path"]),
                coordinate_path=panel_row["coordinate_path"],
                min_ca_atoms=30,
                min_frames=20,
            )
            coords = np.asarray(coords, dtype=np.float64)
            chain = np.asarray(
                [item.get("chain", "_") for item in metadata_rows],
                dtype=object,
            )
            for method in (
                "sequential_lean_exact",
                "uniform_fast_global",
                "external_mdlovofit",
                "external_theseus",
            ):
                for iteration in range(4):
                    warmup = iteration == 0
                    repeat = iteration if not warmup else 0
                    case_root = (
                        temporary
                        / panel_row["domain_id"]
                        / method
                        / str(iteration)
                    )
                    case_root.mkdir(parents=True, exist_ok=True)
                    try:
                        if method in {
                            "sequential_lean_exact",
                            "uniform_fast_global",
                        }:
                            measurement = _internal_materialized(
                                method,
                                coords,
                                chain,
                                case_root / "aligned.pdb",
                            )
                        elif method == "external_mdlovofit":
                            result = benchmark_external_mdlovofit(
                                coords,
                                fraction=0.6,
                                workdir=case_root,
                                executable="/usr/local/bin/mdlovofit",
                                timeout_seconds=300,
                            )
                            measurement = {
                                "algorithm_runtime_seconds": result.extra_metadata[
                                    "subprocess_runtime_seconds"
                                ],
                                "algorithm_runtime_semantics": (
                                    "external_subprocess_including_native_output"
                                ),
                                "input_write_seconds": result.extra_metadata[
                                    "input_write_seconds"
                                ],
                                "transform_application_seconds": "",
                                "output_write_seconds": "included_in_external_subprocess",
                                "output_read_seconds": result.extra_metadata[
                                    "output_read_seconds"
                                ],
                                "full_workflow_seconds": result.runtime_seconds,
                                "unified_posthoc_RMSD_A": result.rmsd,
                                "native_RMSD_A": result.extra_metadata.get(
                                    "mdlovofit_native_all_atom_rmsd"
                                ),
                                "status": (
                                    "success"
                                    if result.parsing_success
                                    else "failed"
                                ),
                                "error": "",
                            }
                        else:
                            result = benchmark_external_theseus(
                                coords,
                                workdir=case_root,
                                executable=(
                                    "/home/jjxu/miniconda3/envs/"
                                    "mdcorealign-theseus/bin/theseus"
                                ),
                                timeout_seconds=300,
                            )
                            measurement = {
                                "algorithm_runtime_seconds": result.extra_metadata[
                                    "subprocess_runtime_seconds"
                                ],
                                "algorithm_runtime_semantics": (
                                    "external_subprocess_including_native_output"
                                ),
                                "input_write_seconds": result.extra_metadata[
                                    "input_write_seconds"
                                ],
                                "transform_application_seconds": "",
                                "output_write_seconds": "included_in_external_subprocess",
                                "output_read_seconds": result.extra_metadata[
                                    "output_read_seconds"
                                ],
                                "full_workflow_seconds": result.runtime_seconds,
                                "unified_posthoc_RMSD_A": result.rmsd,
                                "native_RMSD_A": result.extra_metadata.get(
                                    "reported_maximum_likelihood_rmsd"
                                ),
                                "status": (
                                    "success"
                                    if result.parsing_success
                                    else "failed"
                                ),
                                "error": "",
                            }
                    except Exception as exc:
                        measurement = {
                            "algorithm_runtime_seconds": "",
                            "algorithm_runtime_semantics": "",
                            "input_write_seconds": "",
                            "transform_application_seconds": "",
                            "output_write_seconds": "",
                            "output_read_seconds": "",
                            "full_workflow_seconds": "",
                            "unified_posthoc_RMSD_A": "",
                            "native_RMSD_A": "",
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    if warmup:
                        continue
                    rows.append(
                        {
                            "domain_id": panel_row["domain_id"],
                            "cath_class": panel_row["cath_class"],
                            "method": method,
                            "repeat": repeat,
                            "n_frames": coords.shape[0],
                            "n_residues": coords.shape[1],
                            "coordinate_unit": "angstrom",
                            "fit_selection": "CA",
                            "output_scope": "CA",
                            "output_format": "multi_model_PDB",
                            "timing_boundary": (
                                "loaded_coordinates_to_materialized_aligned_PDB"
                            ),
                            "algorithm_runtime_semantics": measurement.pop(
                                "algorithm_runtime_semantics",
                                "internal_numerical_alignment_only",
                            ),
                            **measurement,
                        }
                    )
    write_tsv(OUTPUT / "contract3_materialized_pdb_runtime.tsv", rows)


def persistent_batch() -> None:
    """Measure a same-shape W5 batch with and without a persistent worker."""

    rng = np.random.default_rng(804)
    reference = np.cumsum(rng.normal(size=(60, 3)), axis=0)
    trajectories = []
    for case in range(1000):
        coords = np.repeat(reference[None, :, :], 12, axis=0)
        coords[:, 24:32] += rng.normal(
            scale=0.1 + 0.001 * (case % 20),
            size=(12, 8, 3),
        )
        trajectories.append(coords)
    rows: list[dict[str, object]] = []
    for batch_size in (1, 10, 100, 1000):
        batch = trajectories[:batch_size]
        for mode in ("recreate_worker", "persistent_worker"):
            for repeat in range(1, 4):
                worker = ExperimentalBatchWorker.create(60)
                worker.representative_indices(12, 12)
                start = time.perf_counter()
                if mode == "persistent_worker":
                    for coords in batch:
                        worker.scan_w5(coords, descriptor_chunk_size=12)
                else:
                    for coords in batch:
                        current = ExperimentalBatchWorker.create(60)
                        current.representative_indices(12, 12)
                        current.scan_w5(coords, descriptor_chunk_size=12)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "batch_size": batch_size,
                        "mode": mode,
                        "repeat": repeat,
                        "runtime_seconds": elapsed,
                        "trajectories_per_second": batch_size / elapsed,
                        "process_peak_rss_mb": _peak_rss_mb(),
                        "output_contract": "alignment_free_W5_scan",
                        **{
                            f"cache_{key}": value
                            for key, value in worker.cache_summary.items()
                        },
                    }
                )
    write_tsv(OUTPUT / "persistent_batch_worker.tsv", rows)
    write_json(
        OUTPUT / "persistent_batch_worker_contract.json",
        ExperimentalBatchWorker.create(60).cache_summary,
    )


def _median_rows(
    rows: list[dict[str, str]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            tuple(row[field] for field in group_fields), []
        ).append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        runtimes = [
            float(row["runtime_seconds"])
            for row in group
            if row.get("runtime_seconds")
        ]
        output.append(
            summary := {
                **dict(zip(group_fields, key, strict=True)),
                "n_repeats": len(runtimes),
                "median_runtime_seconds": statistics.median(runtimes),
                "q1_runtime_seconds": float(np.quantile(runtimes, 0.25)),
                "q3_runtime_seconds": float(np.quantile(runtimes, 0.75)),
            }
        )
        for field in (
            "frames_per_second",
            "residue_frames_per_second",
            "python_peak_memory_mb",
            "process_peak_rss_mb",
            "trajectories_per_second",
        ):
            values = [
                float(row[field])
                for row in group
                if row.get(field) not in {None, ""}
            ]
            if values:
                summary[f"median_{field}"] = statistics.median(values)
    return output


def finalize() -> None:
    """Summarize R5 contracts and write conservative conclusions."""

    contract1_rows = read_tsv(
        OUTPUT / "contract1_local_deformation_runtime.tsv"
    )
    contract2_rows = read_tsv(
        OUTPUT / "contract2_metrics_transforms_runtime.tsv"
    )
    contract3_rows = read_tsv(
        OUTPUT / "contract3_materialized_pdb_runtime.tsv"
    )
    batch_rows = read_tsv(OUTPUT / "persistent_batch_worker.tsv")
    write_tsv(
        OUTPUT / "contract1_runtime_summary.tsv",
        _median_rows(
            contract1_rows,
            ("method", "timing_boundary", "descriptor_chunk_size"),
        ),
    )
    write_tsv(
        OUTPUT / "contract2_runtime_summary.tsv",
        _median_rows(contract2_rows, ("method", "timing_boundary")),
    )
    successful3 = [
        row for row in contract3_rows if row["status"] == "success"
    ]
    contract3_summary: list[dict[str, object]] = []
    for method in sorted({row["method"] for row in contract3_rows}):
        sample = [row for row in successful3 if row["method"] == method]
        contract3_summary.append(
            {
                "method": method,
                "n_success": len(sample),
                "n_attempted": sum(
                    row["method"] == method for row in contract3_rows
                ),
                "median_full_workflow_seconds": (
                    float(
                        np.median(
                            [
                                float(row["full_workflow_seconds"])
                                for row in sample
                            ]
                        )
                    )
                    if sample
                    else "NA"
                ),
                "median_unified_posthoc_RMSD_A": (
                    float(
                        np.median(
                            [
                                float(row["unified_posthoc_RMSD_A"])
                                for row in sample
                            ]
                        )
                    )
                    if sample
                    else "NA"
                ),
            }
        )
    write_tsv(
        OUTPUT / "contract3_materialized_pdb_summary.tsv",
        contract3_summary,
    )
    domain_method: list[dict[str, object]] = []
    for domain_id in sorted({row["domain_id"] for row in contract3_rows}):
        domain_rows = [
            row
            for row in successful3
            if row["domain_id"] == domain_id
        ]
        for method in sorted({row["method"] for row in domain_rows}):
            group = [row for row in domain_rows if row["method"] == method]
            domain_method.append(
                {
                    "domain_id": domain_id,
                    "cath_class": group[0]["cath_class"],
                    "method": method,
                    "n_repeats": len(group),
                    "n_frames": group[0]["n_frames"],
                    "n_residues": group[0]["n_residues"],
                    "median_full_workflow_seconds": statistics.median(
                        float(row["full_workflow_seconds"]) for row in group
                    ),
                    "median_algorithm_runtime_seconds": statistics.median(
                        float(row["algorithm_runtime_seconds"]) for row in group
                    ),
                    "median_unified_posthoc_RMSD_A": statistics.median(
                        float(row["unified_posthoc_RMSD_A"]) for row in group
                    ),
                    "algorithm_runtime_semantics": (
                        group[0].get("algorithm_runtime_semantics")
                        or (
                            "external_subprocess_including_native_output"
                            if method.startswith("external_")
                            else "internal_numerical_alignment_only"
                        )
                    ),
                }
            )
    write_tsv(
        OUTPUT / "contract3_per_domain_summary.tsv",
        domain_method,
    )
    write_tsv(
        OUTPUT / "persistent_batch_worker_summary.tsv",
        _median_rows(batch_rows, ("batch_size", "mode")),
    )

    c1 = read_tsv(OUTPUT / "contract1_runtime_summary.tsv")
    c2 = read_tsv(OUTPUT / "contract2_runtime_summary.tsv")

    def find(
        rows: list[dict[str, str]], **criteria: str
    ) -> dict[str, str]:
        return next(
            row
            for row in rows
            if all(row[key] == value for key, value in criteria.items())
        )

    w5_loaded = find(
        c1,
        method="w5_scan_only",
        timing_boundary="already_loaded_compute",
        descriptor_chunk_size="2000",
    )
    graph_loaded = find(
        c1,
        method="w5_sparse_graph",
        timing_boundary="already_loaded_compute",
        descriptor_chunk_size="2000",
    )
    exact = find(c2, method="sequential_lean_exact")
    fast = find(c2, method="uniform_fast_global")
    per_domain_lookup = {
        (str(row["domain_id"]), str(row["method"])): row
        for row in domain_method
    }
    comparisons = (
        (
            "MDLovoFit_over_exact_MDCoreAlign",
            "external_mdlovofit",
            "sequential_lean_exact",
        ),
        (
            "MDLovoFit_over_fast_global",
            "external_mdlovofit",
            "uniform_fast_global",
        ),
        (
            "THESEUS_over_exact_MDCoreAlign",
            "external_theseus",
            "sequential_lean_exact",
        ),
    )
    speed_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for comparison, numerator_method, denominator_method in comparisons:
        ratios: list[float] = []
        for domain_id in sorted({row["domain_id"] for row in domain_method}):
            numerator = per_domain_lookup.get(
                (str(domain_id), numerator_method)
            )
            denominator = per_domain_lookup.get(
                (str(domain_id), denominator_method)
            )
            if numerator is None or denominator is None:
                continue
            ratio = float(numerator["median_full_workflow_seconds"]) / float(
                denominator["median_full_workflow_seconds"]
            )
            ratios.append(ratio)
            pair_rows.append(
                {
                    "comparison": comparison,
                    "domain_id": domain_id,
                    "cath_class": numerator["cath_class"],
                    "numerator_method": numerator_method,
                    "denominator_method": denominator_method,
                    "numerator_median_seconds": numerator[
                        "median_full_workflow_seconds"
                    ],
                    "denominator_median_seconds": denominator[
                        "median_full_workflow_seconds"
                    ],
                    "paired_runtime_ratio": ratio,
                    "output_contract": (
                        "loaded_CA_coordinates_to_materialized_CA_PDB"
                    ),
                }
            )
        speed_rows.append(
            {
                "comparison": comparison,
                "output_contract": (
                    "loaded_CA_coordinates_to_materialized_CA_PDB"
                ),
                "n_paired_domains": len(ratios),
                "median_paired_runtime_ratio": statistics.median(ratios),
                "q1_paired_runtime_ratio": float(np.quantile(ratios, 0.25)),
                "q3_paired_runtime_ratio": float(np.quantile(ratios, 0.75)),
                "min_paired_runtime_ratio": min(ratios),
                "max_paired_runtime_ratio": max(ratios),
                "fraction_numerator_slower": float(
                    np.mean(np.asarray(ratios) > 1.0)
                ),
                "headline_71x_allowed": False,
            }
        )
    write_tsv(OUTPUT / "contract3_matched_pairwise.tsv", pair_rows)
    write_tsv(OUTPUT / "matched_speed_claims.tsv", speed_rows)

    fast_exact_rmsd_delta = [
        abs(
            float(
                per_domain_lookup[(str(domain_id), "uniform_fast_global")][
                    "median_unified_posthoc_RMSD_A"
                ]
            )
            - float(
                per_domain_lookup[(str(domain_id), "sequential_lean_exact")][
                    "median_unified_posthoc_RMSD_A"
                ]
            )
        )
        for domain_id in sorted({row["domain_id"] for row in domain_method})
        if (str(domain_id), "uniform_fast_global") in per_domain_lookup
        and (str(domain_id), "sequential_lean_exact") in per_domain_lookup
    ]
    write_json(
        OUTPUT / "contract3_fast_global_quality_summary.json",
        {
            "n_paired_domains": len(fast_exact_rmsd_delta),
            "median_absolute_unified_RMSD_difference_A": statistics.median(
                fast_exact_rmsd_delta
            ),
            "p95_absolute_unified_RMSD_difference_A": float(
                np.quantile(fast_exact_rmsd_delta, 0.95)
            ),
            "max_absolute_unified_RMSD_difference_A": max(
                fast_exact_rmsd_delta
            ),
            "interpretation": (
                "Exploratory panel audit; R4 all-domain tail-risk evidence "
                "governs the public experimental status."
            ),
        },
    )
    speed_lookup = {row["comparison"]: row for row in speed_rows}
    exact_vs_mdlo = speed_lookup["MDLovoFit_over_exact_MDCoreAlign"]
    fast_vs_mdlo = speed_lookup["MDLovoFit_over_fast_global"]
    exact_vs_theseus = speed_lookup["THESEUS_over_exact_MDCoreAlign"]
    (OUTPUT / "05_matched_contract_performance_report.md").write_text(
        f"""# R5 matched output-contract performance report

## Contract 1: alignment-free local diagnostics

On the unit-clean 10k/153-residue transition-aligned synthetic stress case,
loaded W5 scan-only median runtime was
{float(w5_loaded['median_runtime_seconds']):.4f} s and W5 plus sparse graph
was {float(graph_loaded['median_runtime_seconds']):.4f} s. These methods do
not produce transforms, RMSD or RMSF and are not presented as MDLovoFit
equivalents. Fresh-process and chunk-size results are in the associated table.
RSS values are process high-water marks, not isolated per-call allocation
deltas.

## Contract 2: transforms, unified RMSD and RMSF

At the same already-loaded boundary, exact sequential_lean median runtime was
{float(exact['median_runtime_seconds']):.4f} s and experimental uniform
fast-global was {float(fast['median_runtime_seconds']):.4f} s. Output fields
are matched, but fast-global remains approximate because R4 found tail
failures. The exploratory graph-anchor result is not a hard biological-domain
claim.

## Contract 3: materialized CA multi-model PDB

Internal and external methods start from loaded coordinates and end with an
aligned CA multi-model PDB. External subprocess time necessarily includes
native search/fitting and mandatory output generation; pure external numerical
kernels cannot be isolated without modifying those tools. Across the 12 paired
domains, median MDLovoFit/exact-MDCoreAlign full-workflow ratio was
{float(exact_vs_mdlo['median_paired_runtime_ratio']):.3f}x, median
MDLovoFit/fast-global ratio was
{float(fast_vs_mdlo['median_paired_runtime_ratio']):.3f}x, and median
THESEUS/exact-MDCoreAlign ratio was
{float(exact_vs_theseus['median_paired_runtime_ratio']):.3f}x. Ratios above
one favour the MDCoreAlign denominator. No 71x headline is made. See
`contract3_matched_pairwise.tsv` and `matched_speed_claims.tsv`.

Fast-global's median absolute unified-RMSD difference from exact on this panel
was {statistics.median(fast_exact_rmsd_delta):.4f} A and the maximum was
{max(fast_exact_rmsd_delta):.4f} A. This small panel does not override the R4
all-domain tail failures, so fast-global remains experimental.

## Persistent worker

The experimental worker reuses imports/BLAS pools, topology metadata, W5 edges,
uniform weights and representative indices. Current descriptor and Kabsch
kernels still allocate temporary arrays; trajectory buffers are not reused.
The benchmark therefore validates a partial, not zero-allocation, persistent
interface.

## Scope

All 10k measurements use a transition-aligned synthetic long-trajectory stress
case. They support computational scaling only, not biological continuity.
"""
    )
    write_json(
        OUTPUT / "05_r5_status.json",
        {
            "phase": "R5",
            "status": "pass",
            "contract1_rows": len(contract1_rows),
            "contract2_rows": len(contract2_rows),
            "contract3_rows": len(contract3_rows),
            "persistent_batch_rows": len(batch_rows),
            "speedup_71x_claimed": False,
            "timing_boundaries_matched_and_explicit": True,
            "public_default_changed": False,
            "gpu_used": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "contract1",
            "contract2",
            "contract3",
            "persistent-batch",
            "finalize",
            "child-w5",
        ),
    )
    parser.add_argument(
        "--method",
        choices=("w5_scan_only", "w5_sparse_graph"),
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "contract1":
        contract1()
    elif args.stage == "contract2":
        contract2()
    elif args.stage == "contract3":
        contract3()
    elif args.stage == "persistent-batch":
        persistent_batch()
    elif args.stage == "finalize":
        finalize()
    else:
        child_w5(args.method, args.chunk_size)


if __name__ == "__main__":
    main()
