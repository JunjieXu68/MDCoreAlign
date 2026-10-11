#!/usr/bin/env python
"""Run the locked Phase 8B runtime-boundary and startup audit.

This script changes no scientific algorithm. It applies the locked Phase 8A
estimators, then uses one common exact all-frame fitting/evaluation path so
that the internal-method timings have identical output boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from mdsuperpose.mdlovofit import (
    parse_mdlovofit_stdout,
    read_multimodel_pdb_with_occupancy,
    resolve_mdlovofit_executable,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.representative_weights import (
    estimate_representative_weights,
)
from mdsuperpose.sequential_weights import (
    estimate_representative_weights_sequential_lean,
)
from mdsuperpose.window_refine import (
    build_initial_weights,
    bounded_global_weight_refinement,
    compute_frozen_window_signals,
    exact_all_frame_fit,
    temporally_stratified_indices,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)
from scripts.scaling_optimization.build_phase2_regression import ROBUST_KWARGS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v5/phase8b_runtime"
PHASE7B_ROOT = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase7b_unit_integrity"
)
PHASE8A_ROOT = (
    PROJECT_ROOT
    / "results/scaling_optimization_v5/phase8a_broad_validation"
)
LONG_COORDS = (
    PHASE7B_ROOT
    / "regenerated/4oelA02/transition_aligned_coords_10000.npz"
)
LONG_METADATA = (
    PHASE7B_ROOT
    / "regenerated/4oelA02/raw_replicates/A_metadata.json"
)
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
N_WARMUPS = 3
N_REPEATS = 5
METHODS = (
    "sequential_lean_exact",
    "uniform_low_budget",
    "w5_preconditioned",
    "graph_preconditioned",
    "batched_independent",
    "w5_scan_only",
)
GLOBAL_METHODS = METHODS[:-1]
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows using their union schema."""

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


def load_long_input() -> tuple[np.ndarray, np.ndarray]:
    """Load the regenerated, unit-clean 10k canonical trajectory."""

    with np.load(LONG_COORDS, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float64)
    labels = json.loads(LONG_METADATA.read_text())["labels"]
    chain = np.asarray(
        [label.split(":", 1)[0] if ":" in label else "_" for label in labels]
    )
    return coords, chain


def result_digest(weights: np.ndarray, rmsd: np.ndarray | None) -> str:
    """Return a compact deterministic result checksum."""

    digest = hashlib.sha256()
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    if rmsd is not None:
        digest.update(np.asarray(rmsd, dtype=np.float64).tobytes())
    return digest.hexdigest()


def estimate_weights(
    method: str,
    coords: np.ndarray,
    chain: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate one locked weight vector and return stage timings."""

    selected = temporally_stratified_indices(
        coords.shape[0], min(2000, coords.shape[0])
    )
    timing = {
        "scan_seconds": 0.0,
        "representative_selection_seconds": 0.0,
        "core_identification_seconds": 0.0,
    }
    if method == "sequential_lean_exact":
        start = time.perf_counter()
        estimate = estimate_representative_weights_sequential_lean(
            coords[selected],
            coords[0],
            sequence_chain_ids=chain,
            **ROBUST_KWARGS,
        )
        timing["core_identification_seconds"] = time.perf_counter() - start
        return np.asarray(estimate.mean_weights, dtype=np.float64), timing

    if method == "batched_independent":
        start = time.perf_counter()
        estimate = estimate_representative_weights(
            coords[selected],
            coords[0],
            batch_size=min(512, selected.size),
            sequence_chain_ids=chain,
            **ROBUST_KWARGS,
        )
        timing["core_identification_seconds"] = time.perf_counter() - start
        return np.asarray(estimate.mean_weights, dtype=np.float64), timing

    initializer = {
        "uniform_low_budget": "uniform",
        "w5_preconditioned": "w5_relative_mad_soft",
        "graph_preconditioned": "window_graph_soft",
        "w5_scan_only": "w5_relative_mad_soft",
    }[method]
    signals = None
    if initializer != "uniform":
        scan_start = time.perf_counter()
        signals = compute_frozen_window_signals(
            coords,
            chain_ids=chain,
            initializers=(initializer,),
        )
        timing["scan_seconds"] = time.perf_counter() - scan_start
    initial = build_initial_weights(
        coords.shape[1],
        initializer,
        window_signals=signals,
        alpha=None if initializer == "uniform" else 0.5,
    )
    if method == "w5_scan_only":
        timing["core_identification_seconds"] = timing["scan_seconds"]
        return np.asarray(initial.weights, dtype=np.float64), timing

    selection_start = time.perf_counter()
    representatives = temporally_stratified_indices(coords.shape[0], 256)
    timing["representative_selection_seconds"] = (
        time.perf_counter() - selection_start
    )
    refine_start = time.perf_counter()
    weights, _, _, _ = bounded_global_weight_refinement(
        coords[representatives],
        coords[0],
        initial.weights,
        update_rounds=4,
        sequence_chain_ids=chain,
    )
    refinement_seconds = time.perf_counter() - refine_start
    timing["core_identification_seconds"] = (
        timing["scan_seconds"]
        + timing["representative_selection_seconds"]
        + refinement_seconds
    )
    return np.asarray(weights, dtype=np.float64), timing


def run_common_boundary(
    method: str,
    coords: np.ndarray,
    chain: np.ndarray,
) -> tuple[dict[str, object], object | None]:
    """Run a method and the common exact metrics/transforms boundary."""

    total_start = time.perf_counter()
    weights, stages = estimate_weights(method, coords, chain)
    if method == "w5_scan_only":
        return (
            {
                "total_seconds": time.perf_counter() - total_start,
                **stages,
                "alignment_fit_seconds": 0.0,
                "rmsd_seconds": 0.0,
                "rmsf_seconds": 0.0,
                "mean_unified_rmsd_A": "",
                "result_sha256": result_digest(weights, None),
                "output_contract": "alignment_free_local_deformation",
            },
            None,
        )
    fit = exact_all_frame_fit(coords, coords[0], weights, chunk_size=2000)
    total_seconds = time.perf_counter() - total_start
    alignment_seconds = (
        fit.timing["all_frame_weighted_fit"]
        + fit.timing["all_frame_ordinary_fit"]
    )
    return (
        {
            "total_seconds": total_seconds,
            **stages,
            "alignment_fit_seconds": alignment_seconds,
            "rmsd_seconds": fit.timing["RMSD"],
            "rmsf_seconds": fit.timing["RMSF"],
            "mean_unified_rmsd_A": float(np.mean(fit.rmsd)),
            "result_sha256": result_digest(weights, fit.rmsd),
            "output_contract": "metrics_transforms_rmsd_rmsf",
        },
        fit,
    )


def measured_iterations():
    """Yield warmup and measured-repeat identifiers."""

    for index in range(N_WARMUPS):
        yield True, index + 1
    for index in range(N_REPEATS):
        yield False, index + 1


def base_row(
    method: str,
    boundary: str,
    warmup: bool,
    repeat: int,
    coords: np.ndarray,
) -> dict[str, object]:
    """Return common runtime-table fields."""

    return {
        "system": "regenerated_4oelA02_transition_aligned_10000",
        "method": method,
        "boundary": boundary,
        "warmup": warmup,
        "repeat": repeat,
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "coordinate_unit": "angstrom",
    }


def run_in_memory(coords: np.ndarray, chain: np.ndarray) -> list[dict[str, object]]:
    """Measure already-loaded metrics/transforms for every locked mode."""

    rows: list[dict[str, object]] = []
    for method in METHODS:
        for warmup, repeat in measured_iterations():
            metrics, _ = run_common_boundary(method, coords, chain)
            rows.append(
                {
                    **base_row(
                        method,
                        "already_loaded_metrics_transforms",
                        warmup,
                        repeat,
                        coords,
                    ),
                    "decode_seconds": 0.0,
                    **metrics,
                }
            )
    return rows


def run_decode_included(chain: np.ndarray) -> list[dict[str, object]]:
    """Measure compressed-NPZ decode plus common scientific outputs."""

    rows: list[dict[str, object]] = []
    for method in GLOBAL_METHODS:
        for warmup, repeat in measured_iterations():
            total_start = time.perf_counter()
            decode_start = time.perf_counter()
            with np.load(LONG_COORDS, allow_pickle=False) as payload:
                coords = np.asarray(payload["coords"], dtype=np.float64)
            decode_seconds = time.perf_counter() - decode_start
            metrics, _ = run_common_boundary(method, coords, chain)
            metrics["total_seconds"] = time.perf_counter() - total_start
            rows.append(
                {
                    **base_row(
                        method,
                        "npz_decode_included_metrics_transforms",
                        warmup,
                        repeat,
                        coords,
                    ),
                    "decode_seconds": decode_seconds,
                    **metrics,
                }
            )
    return rows


def child_command(method: str) -> list[str]:
    """Return the absolute-interpreter fresh-process command."""

    return [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--child",
        "--method",
        method,
    ]


def run_fresh_process(coords: np.ndarray) -> list[dict[str, object]]:
    """Measure fresh interpreter, imports, NPZ decode and computation."""

    rows: list[dict[str, object]] = []
    environment = {**os.environ, **THREAD_ENVIRONMENT}
    for method in GLOBAL_METHODS:
        for warmup, repeat in measured_iterations():
            start = time.perf_counter()
            completed = subprocess.run(
                child_command(method),
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            elapsed = time.perf_counter() - start
            payload = json.loads(completed.stdout)
            rows.append(
                {
                    **base_row(
                        method,
                        "fresh_process_npz_metrics_transforms",
                        warmup,
                        repeat,
                        coords,
                    ),
                    **payload,
                    "total_seconds": elapsed,
                    "child_reported_seconds": payload["total_seconds"],
                    "process_startup_and_import_seconds": max(
                        0.0, elapsed - float(payload["total_seconds"])
                    ),
                }
            )
    return rows


def run_output_boundaries(
    coords: np.ndarray, chain: np.ndarray
) -> list[dict[str, object]]:
    """Measure transform application and binary/PDB output separately."""

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="mdca_phase8b_output_", dir=RESULT_ROOT
    ) as temporary:
        root = Path(temporary)
        for method in GLOBAL_METHODS:
            metrics, fit = run_common_boundary(method, coords, chain)
            assert fit is not None
            for warmup, repeat in measured_iterations():
                apply_start = time.perf_counter()
                aligned = (
                    np.einsum(
                        "tni,tij->tnj",
                        coords,
                        fit.rotations,
                        optimize=True,
                    )
                    + fit.translations[:, None, :]
                )
                application_seconds = time.perf_counter() - apply_start

                binary_path = root / f"{method}.npy"
                write_start = time.perf_counter()
                np.save(binary_path, aligned, allow_pickle=False)
                binary_seconds = time.perf_counter() - write_start
                rows.append(
                    {
                        **base_row(
                            method,
                            "binary_trajectory_application_write",
                            warmup,
                            repeat,
                            coords,
                        ),
                        "decode_seconds": 0.0,
                        "transform_application_seconds": application_seconds,
                        "binary_write_seconds": binary_seconds,
                        "total_seconds": application_seconds + binary_seconds,
                        "result_sha256": metrics["result_sha256"],
                        "output_contract": "aligned_binary_trajectory",
                    }
                )

                pdb_path = root / f"{method}.pdb"
                pdb_start = time.perf_counter()
                write_multimodel_pdb(aligned, pdb_path)
                pdb_seconds = time.perf_counter() - pdb_start
                rows.append(
                    {
                        **base_row(
                            method,
                            "pdb_write_only",
                            warmup,
                            repeat,
                            coords,
                        ),
                        "pdb_write_seconds": pdb_seconds,
                        "total_seconds": pdb_seconds,
                        "result_sha256": metrics["result_sha256"],
                        "output_contract": "aligned_pdb_serialisation",
                    }
                )
    return rows


def select_canonical_subset() -> list[dict[str, str]]:
    """Choose two locked cases per represented CATH class."""

    manifest = read_tsv(PHASE8A_ROOT / "broad_panel_manifest.tsv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        groups.setdefault(row["cath_class"], []).append(row)
    selected: list[dict[str, str]] = []
    for cath_class in ("1", "2", "3", "4", "6", "NA"):
        candidates = groups[cath_class]
        faster = next(
            (row for row in candidates if row["speed_bin"] == "mdca_faster"),
            candidates[0],
        )
        slower = next(
            (
                row
                for row in reversed(candidates)
                if row["speed_bin"] == "mdca_slower"
                and row["domain_id"] != faster["domain_id"]
            ),
            candidates[-1],
        )
        selected.extend([faster, slower])
    return selected


def run_same_boundary_mdlovofit() -> list[dict[str, object]]:
    """Rerun MDLovoFit and internal modes on a locked 12-domain subset."""

    executable = resolve_mdlovofit_executable()
    rows: list[dict[str, object]] = []
    subset = select_canonical_subset()
    with tempfile.TemporaryDirectory(
        prefix="mdca_phase8b_mdlo_", dir=RESULT_ROOT
    ) as temporary:
        root = Path(temporary)
        for panel in subset:
            coords, labels, _, metadata = extract_ca_trajectory(
                Path(panel["h5_path"]),
                coordinate_path=panel["coordinate_path"],
                min_ca_atoms=30,
                min_frames=20,
            )
            coords = np.asarray(coords, dtype=np.float64)
            chain = np.asarray(
                [
                    label.split(":", 1)[0] if ":" in label else "_"
                    for label in labels
                ]
            )
            case_root = root / panel["domain_id"]
            case_root.mkdir()
            input_path = case_root / "input.pdb"
            write_multimodel_pdb(coords, input_path)

            for method in (
                "sequential_lean_exact",
                "uniform_low_budget",
                "w5_preconditioned",
            ):
                for warmup, repeat in measured_iterations():
                    metrics, _ = run_common_boundary(method, coords, chain)
                    if not warmup:
                        rows.append(
                            {
                                "domain_id": panel["domain_id"],
                                "cath_class": panel["cath_class"],
                                "historical_speed_bin": panel["speed_bin"],
                                "method": method,
                                "repeat": repeat,
                                "n_frames": coords.shape[0],
                                "n_residues": coords.shape[1],
                                "timing_boundary": (
                                    "already_loaded_metrics_transforms_rmsd_rmsf"
                                ),
                                "input_write_seconds": 0.0,
                                "algorithm_or_subprocess_seconds": metrics[
                                    "total_seconds"
                                ],
                                "output_read_seconds": 0.0,
                                "full_workflow_seconds": metrics[
                                    "total_seconds"
                                ],
                                "unified_posthoc_rmsd_A": metrics[
                                    "mean_unified_rmsd_A"
                                ],
                                "native_all_atom_rmsd_A": "",
                                "native_least_mobile_rmsd_A": "",
                                "return_code": 0,
                                "unit_status": metadata["unit_status"],
                            }
                        )

            for warmup, repeat in measured_iterations():
                output_path = case_root / "mdlovofit_aligned.pdb"
                input_write_start = time.perf_counter()
                write_multimodel_pdb(coords, input_path)
                input_write_seconds = time.perf_counter() - input_write_start
                start = time.perf_counter()
                run = run_mdlovofit_on_files(
                    [input_path],
                    fraction=0.6,
                    output_path=output_path,
                    executable=executable,
                    timeout_seconds=300.0,
                )
                subprocess_seconds = time.perf_counter() - start
                read_start = time.perf_counter()
                aligned, _ = read_multimodel_pdb_with_occupancy(
                    output_path, coords.shape[1]
                )
                output_read_seconds = time.perf_counter() - read_start
                squared = np.sum(
                    (aligned - coords[0][None, :, :]) ** 2, axis=2
                )
                unified = float(np.mean(np.sqrt(np.mean(squared, axis=1))))
                if not warmup:
                    parsed = parse_mdlovofit_stdout(run.stdout)
                    rows.append(
                        {
                            "domain_id": panel["domain_id"],
                            "cath_class": panel["cath_class"],
                            "historical_speed_bin": panel["speed_bin"],
                            "method": "mdlovofit_external",
                            "repeat": repeat,
                            "n_frames": coords.shape[0],
                            "n_residues": coords.shape[1],
                            "timing_boundary": (
                                "prewritten_pdb_external_subprocess_plus_output_read"
                            ),
                            "input_write_seconds": input_write_seconds,
                            "algorithm_or_subprocess_seconds": subprocess_seconds,
                            "output_read_seconds": output_read_seconds,
                            "full_workflow_seconds": (
                                input_write_seconds
                                + subprocess_seconds
                                + output_read_seconds
                            ),
                            "unified_posthoc_rmsd_A": unified,
                            "native_all_atom_rmsd_A": (
                                parsed.average_rmsd_all_atoms
                            ),
                            "native_least_mobile_rmsd_A": (
                                parsed.average_rmsd_least_mobile
                            ),
                            "return_code": run.return_code,
                            "unit_status": metadata["unit_status"],
                        }
                    )
    return rows


def quantiles(values: list[float]) -> tuple[float, float, float]:
    """Return median, Q1 and Q3."""

    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def summarize_runtime(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate non-warmup rows by method and boundary."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if str(row["warmup"]).lower() == "true":
            continue
        groups.setdefault(
            (str(row["method"]), str(row["boundary"])), []
        ).append(row)
    output: list[dict[str, object]] = []
    for (method, boundary), group in sorted(groups.items()):
        values = [float(row["total_seconds"]) for row in group]
        median, q1, q3 = quantiles(values)
        output.append(
            {
                "system": group[0]["system"],
                "method": method,
                "boundary": boundary,
                "n_frames": group[0]["n_frames"],
                "n_residues": group[0]["n_residues"],
                "n_warmups": N_WARMUPS,
                "n_measured_repeats": len(group),
                "median_seconds": median,
                "q1_seconds": q1,
                "q3_seconds": q3,
                "iqr_seconds": q3 - q1,
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
                "best_only_reported": False,
                "result_hashes_identical_across_repeats": (
                    len(
                        {
                            str(row.get("result_sha256", ""))
                            for row in group
                        }
                    )
                    == 1
                ),
            }
        )
    return output


def profile_imports() -> list[dict[str, object]]:
    """Measure fresh-process import and CLI startup costs."""

    rows: list[dict[str, object]] = []
    environment = {**os.environ, **THREAD_ENVIRONMENT}
    commands = {
        "python_startup": ["-c", "pass"],
        "numpy": ["-c", "import numpy"],
        "mdsuperpose": ["-c", "import mdsuperpose"],
        "mdcorealign": ["-c", "import mdcorealign"],
        "mdcorealign_cli": ["-c", "import mdcorealign.cli"],
        "mdcorealign_help": ["-m", "mdcorealign.cli", "--help"],
        "scipy": ["-c", "import scipy"],
        "MDAnalysis": ["-c", "import MDAnalysis"],
    }
    for component, arguments in commands.items():
        values: list[float] = []
        statuses: list[str] = []
        errors: list[str] = []
        for _ in range(N_REPEATS):
            start = time.perf_counter()
            completed = subprocess.run(
                [str(PYTHON), *arguments],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            values.append(time.perf_counter() - start)
            statuses.append(
                "available" if completed.returncode == 0 else "unavailable"
            )
            if completed.returncode:
                errors.append(completed.stderr.strip().splitlines()[-1])
        median, q1, q3 = quantiles(values)
        rows.append(
            {
                "component": component,
                "status": (
                    "available"
                    if all(status == "available" for status in statuses)
                    else "unavailable"
                ),
                "n_repeats": N_REPEATS,
                "median_fresh_process_seconds": median,
                "q1_seconds": q1,
                "q3_seconds": q3,
                "increment_over_python_startup_seconds": "",
                "error": "; ".join(sorted(set(errors))),
                "absolute_python": str(PYTHON),
            }
        )
    baseline = next(
        float(row["median_fresh_process_seconds"])
        for row in rows
        if row["component"] == "python_startup"
    )
    for row in rows:
        row["increment_over_python_startup_seconds"] = max(
            0.0, float(row["median_fresh_process_seconds"]) - baseline
        )
    return rows


def median_method_rows(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Aggregate the same-boundary external comparison."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["domain_id"]), str(row["method"])), []
        ).append(row)
    medians: list[dict[str, object]] = []
    for (domain_id, method), group in sorted(groups.items()):
        runtime = statistics.median(
            float(row["algorithm_or_subprocess_seconds"]) for row in group
        )
        full = statistics.median(
            float(row["full_workflow_seconds"]) for row in group
        )
        rmsd = statistics.median(
            float(row["unified_posthoc_rmsd_A"]) for row in group
        )
        medians.append(
            {
                "domain_id": domain_id,
                "cath_class": group[0]["cath_class"],
                "historical_speed_bin": group[0]["historical_speed_bin"],
                "method": method,
                "n_frames": group[0]["n_frames"],
                "n_residues": group[0]["n_residues"],
                "n_repeats": len(group),
                "median_algorithm_or_subprocess_seconds": runtime,
                "median_full_workflow_seconds": full,
                "median_unified_posthoc_rmsd_A": rmsd,
                "native_all_atom_rmsd_A": group[0][
                    "native_all_atom_rmsd_A"
                ],
                "native_least_mobile_rmsd_A": group[0][
                    "native_least_mobile_rmsd_A"
                ],
                "timing_boundary": group[0]["timing_boundary"],
                "median_input_write_seconds": statistics.median(
                    float(row["input_write_seconds"]) for row in group
                ),
                "median_output_read_seconds": statistics.median(
                    float(row["output_read_seconds"]) for row in group
                ),
                "comparison_limitation": (
                    "MDLovoFit subprocess includes native core search, fitting "
                    "and mandatory output writing; these cannot be separated "
                    "without modifying the external executable."
                    if method == "mdlovofit_external"
                    else "Already-loaded internal scientific compute."
                ),
            }
        )
    by_case = {
        (str(row["domain_id"]), str(row["method"])): row for row in medians
    }
    for row in medians:
        if row["method"] == "mdlovofit_external":
            continue
        external = by_case.get(
            (str(row["domain_id"]), "mdlovofit_external")
        )
        row["mdlovofit_over_method_runtime_ratio"] = (
            float(external["median_algorithm_or_subprocess_seconds"])
            / float(row["median_algorithm_or_subprocess_seconds"])
            if external is not None
            else ""
        )
        row["method_minus_mdlovofit_unified_rmsd_A"] = (
            float(row["median_unified_posthoc_rmsd_A"])
            - float(external["median_unified_posthoc_rmsd_A"])
            if external is not None
            else ""
        )
    return medians


def write_docs(
    runtime_summary: list[dict[str, object]],
    imports: list[dict[str, object]],
    comparison: list[dict[str, object]],
) -> None:
    """Write boundary definitions and conservative startup findings."""

    by_key = {
        (str(row["method"]), str(row["boundary"])): row
        for row in runtime_summary
    }
    exact = by_key[
        ("sequential_lean_exact", "already_loaded_metrics_transforms")
    ]
    uniform = by_key[
        ("uniform_low_budget", "already_loaded_metrics_transforms")
    ]
    w5 = by_key[
        ("w5_preconditioned", "already_loaded_metrics_transforms")
    ]
    uniform_fresh = by_key[
        ("uniform_low_budget", "fresh_process_npz_metrics_transforms")
    ]
    python_start = next(
        row for row in imports if row["component"] == "python_startup"
    )
    package_import = next(
        row for row in imports if row["component"] == "mdcorealign"
    )
    mdlo = [
        row for row in comparison if row["method"] == "mdlovofit_external"
    ]
    mdlo_median = float(
        np.median(
            [
                float(row["median_algorithm_or_subprocess_seconds"])
                for row in mdlo
            ]
        )
    )

    (RESULT_ROOT / "canonical_runtime_boundaries.md").write_text(
        f"""# Phase 8B canonical runtime boundaries

All measurements used `{PYTHON}` with two-thread OpenBLAS, OMP, MKL and
NumExpr settings. Three warmups and five measured repeats were used. Medians
and IQRs, rather than best-only values, are reported.

## Internal boundaries

- **Already loaded, metrics/transforms:** coordinates are float64 in memory;
  weight estimation, one common exact all-frame weighted fit, unified RMSD,
  robust RMSF and ordinary-control RMSF are included.
- **NPZ decode included:** compressed NPZ loading/conversion is added to the
  same scientific boundary.
- **Fresh process:** Python startup, imports, NPZ decode and the same scientific
  computation are included.
- **Binary trajectory:** application of precomputed transforms and NumPy
  trajectory writing are measured separately.
- **PDB write:** text serialisation of an already aligned trajectory is measured
  separately and is not hidden inside the scientific-compute result.
- **W5 scan only:** this has no transform and reports a local-deformation
  contract; RMSD/RMSF are intentionally unavailable.

## External MDLovoFit boundary

The input CA-only PDB is prepared before the subprocess boundary. The external
subprocess includes MDLovoFit core search, fitting and mandatory aligned-PDB
writing. Output parsing/read time is reported separately. Internal stages
cannot be separated without changing MDLovoFit, so its subprocess time is not
mislabelled as pure numerical-kernel time. A method-native full workflow is
also retained in the per-repeat table.

The canonical long input is the regenerated, unit-clean 10,000-frame
transition-aligned synthetic trajectory for 4oelA02. It is a computational
scaling input, not a physically continuous MD trajectory.
"""
    )

    (RESULT_ROOT / "startup_optimisation_report.md").write_text(
        f"""# Phase 8B startup-overhead report

## Measured result

- Python startup median: {float(python_start['median_fresh_process_seconds']):.4f} s.
- `import mdcorealign` median: {float(package_import['median_fresh_process_seconds']):.4f} s.
- 10k uniform already-loaded median: {float(uniform['median_seconds']):.4f} s.
- 10k uniform fresh-process median: {float(uniform_fresh['median_seconds']):.4f} s.
- 10k exact already-loaded median: {float(exact['median_seconds']):.4f} s.
- 10k W5 already-loaded median: {float(w5['median_seconds']):.4f} s.
- Canonical 12-case MDLovoFit subprocess median across systems:
  {mdlo_median:.4f} s.

## Safe optimisation decision

No eager-import rewrite was made. The public `mdcorealign` namespace re-exports
the documented API, while NumPy dominates the small measured import cost.
Changing that namespace would add compatibility risk for little measured gain.
The safe high-throughput path is therefore a persistent Python process/API:
load the package once and analyze multiple trajectories, while separately
reporting fresh-process timing. No scientific output, public default, optional
backend or CLI contract was changed.

MDLovoFit timing is newly measured under the same fixed thread environment and
with explicit prepared-input, subprocess and output-read boundaries. Because
MDLovoFit writes its aligned PDB inside the subprocess, an algorithm-kernel-only
comparison is not observable and is not claimed.
"""
    )


def child_main(method: str) -> None:
    """Execute one fresh-process timing and print only JSON."""

    start = time.perf_counter()
    coords, chain = load_long_input()
    decode_seconds = time.perf_counter() - start
    metrics, _ = run_common_boundary(method, coords, chain)
    metrics["decode_seconds"] = decode_seconds
    metrics["total_seconds"] = time.perf_counter() - start
    print(json.dumps(metrics, sort_keys=True))


def main() -> None:
    """Run Phase 8B or one child timing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--method", choices=GLOBAL_METHODS)
    arguments = parser.parse_args()
    if arguments.child:
        if arguments.method is None:
            parser.error("--method is required with --child")
        child_main(arguments.method)
        return

    for key, value in THREAD_ENVIRONMENT.items():
        os.environ[key] = value
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.external_only:
        comparison_raw = run_same_boundary_mdlovofit()
        write_tsv(
            RESULT_ROOT / "same_boundary_mdlovofit_per_repeat.tsv",
            comparison_raw,
        )
        comparison = median_method_rows(comparison_raw)
        write_tsv(
            RESULT_ROOT / "same_boundary_mdlovofit_comparison.tsv",
            comparison,
        )
        summary = read_tsv(RESULT_ROOT / "runtime_summary.tsv")
        imports = read_tsv(RESULT_ROOT / "import_time_breakdown.tsv")
        write_docs(summary, imports, comparison)
        return

    coords, chain = load_long_input()
    rows = run_in_memory(coords, chain)
    rows.extend(run_decode_included(chain))
    rows.extend(run_fresh_process(coords))
    rows.extend(run_output_boundaries(coords, chain))
    write_tsv(RESULT_ROOT / "runtime_per_repeat.tsv", rows)
    summary = summarize_runtime(rows)
    write_tsv(RESULT_ROOT / "runtime_summary.tsv", summary)

    imports = profile_imports()
    write_tsv(RESULT_ROOT / "import_time_breakdown.tsv", imports)

    comparison_raw = run_same_boundary_mdlovofit()
    write_tsv(
        RESULT_ROOT / "same_boundary_mdlovofit_per_repeat.tsv",
        comparison_raw,
    )
    comparison = median_method_rows(comparison_raw)
    write_tsv(
        RESULT_ROOT / "same_boundary_mdlovofit_comparison.tsv",
        comparison,
    )
    write_docs(summary, imports, comparison)
    status = {
        "status": "complete",
        "phase": "8B",
        "canonical_python": str(PYTHON),
        "warmups": N_WARMUPS,
        "measured_repeats": N_REPEATS,
        "canonical_long_case": LONG_COORDS.relative_to(PROJECT_ROOT).as_posix(),
        "canonical_subset_size": len(select_canonical_subset()),
        "mdlovofit_executable": str(resolve_mdlovofit_executable()),
        "scientific_algorithm_changed": False,
        "public_default_changed": False,
        "startup_code_changed": False,
    }
    (RESULT_ROOT / "phase8b_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
