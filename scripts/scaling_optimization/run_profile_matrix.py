#!/usr/bin/env python
"""Run fresh-process Phase 1 stage, memory and scaling profiles.

This harness does not change MDCoreAlign's scientific algorithm. Instrumentation
is active only when this script is invoked with ``--profile-stages`` (or
``MDCOREALIGN_PROFILE_STAGES=1``).
"""

from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import tracemalloc
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mdsuperpose.alignment import apply_transform
from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.flexibility import (
    align_trajectory_ordinary,
    compute_rmsf,
    detect_flexible_segments,
    mean_soft_weights_from_result,
    robust_aligned_coords_from_result,
)
from mdsuperpose.long_mode import (
    aligned_coords_from_long_result,
    build_long_flexibility_profile,
    solve_long_trajectory_alignment,
)
from mdsuperpose.mdlovofit import (
    parse_mdlovofit_stdout,
    read_multimodel_pdb_with_occupancy,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.metrics import mean_rmsd
from mdsuperpose.theseus import (
    parse_theseus_stdout,
    parse_theseus_variances,
    read_multimodel_pdb_coords,
    run_theseus_on_files,
)
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment

from scripts.scaling_optimization.profile_support import (
    MemorySampler,
    StageRecorder,
    array_record,
    write_json,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results/scaling_optimization_v1"
DEFAULT_FRAME_COUNTS = (12, 100, 500, 1000, 5000, 10000)
PRIMARY_CONFIGS = (
    "mdcorealign_default_no_write",
    "mdcorealign_default_write",
    "mdcorealign_long_no_write",
    "mdcorealign_long_write",
    "mdlovofit",
    "theseus",
    "input_decode_only",
    "write_only",
    "mdcorealign_gpu",
)
SECONDARY_CONFIGS = (
    "mdcorealign_default_no_write",
    "mdcorealign_long_no_write",
    "input_decode_only",
)
DATASETS = (
    {
        "dataset_id": "4oelA02_transition_aligned",
        "input_path": (
            "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
            "runs/4oelA02/10000/transition_aligned_coords.npz"
        ),
        "data_semantics": "transition_aligned_synthetic_long_trajectory",
        "expected_residues": 153,
        "primary": True,
    },
    {
        "dataset_id": "4a18O00_real_single_replicate",
        "input_path": (
            "results/mdcath_benchmark/long_continuous_trajectory/"
            "continuous_inputs/4a18O00__T320__R1_ca_coords.npz"
        ),
        "data_semantics": "real_continuous_single_replicate_md",
        "expected_residues": 134,
        "primary": False,
    },
    {
        "dataset_id": "3a2pA00_real_single_replicate",
        "input_path": (
            "results/mdcath_benchmark/long_continuous_trajectory/"
            "continuous_inputs/3a2pA00__T320__R1_ca_coords.npz"
        ),
        "data_semantics": "real_continuous_single_replicate_md",
        "expected_residues": 483,
        "primary": False,
    },
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _display_input_path(path: Path) -> str:
    """Prefer repository-relative input paths without rejecting test fixtures."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external:{path.resolve()}"


def _robust_kwargs() -> dict[str, object]:
    """Return the frozen current benchmark settings."""

    return {
        "rule": "huber",
        "max_iterations": 20,
        "tolerance": 1e-6,
        "objective_tolerance": 1e-5,
        "objective_patience": 2,
        "sequence_continuity_strength": 0.05,
        "contact_consistency_strength": 0.0,
        "record_history": False,
    }


def _load_coords(
    path: Path,
    n_frames: int,
    recorder: StageRecorder,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Load a coordinate prefix while separating open, decode and conversion."""

    with recorder.stage("input_open"):
        payload = np.load(path, allow_pickle=False)
    try:
        with recorder.stage("trajectory_decode"):
            raw = np.asarray(payload["coords"][:n_frames])
    finally:
        payload.close()
    with recorder.stage("dtype_normalisation"):
        coords = np.asarray(raw, dtype=float)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"coords must have shape (frames, atoms, 3), got {coords.shape}")
    if coords.shape[0] != n_frames:
        raise ValueError(f"requested {n_frames} frames but input provided {coords.shape[0]}")
    if not np.isfinite(coords).all():
        raise ValueError("input contains non-finite coordinates")
    records = [
        value
        for value in (
            array_record("decoded_coords", raw),
            array_record("float64_coords", coords),
        )
        if value is not None
    ]
    return coords, records


def _default_alignment(
    coords: np.ndarray,
    recorder: StageRecorder,
    *,
    write_trajectory: bool,
    workdir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Profile the current full per-frame iterative workflow."""

    algorithm_start = time.perf_counter()
    result = solve_trajectory_robust_alignment(
        coords,
        reference_mode="first_frame",
        warm_start=True,
        collect_timing=True,
        **_robust_kwargs(),
    )
    algorithm_seconds = time.perf_counter() - algorithm_start
    internal = result.timing or {}
    alignment_internal = float(internal.get("weighted_fit_sec", 0.0))
    core_internal = sum(
        float(internal.get(key, 0.0))
        for key in ("residual_update_sec", "weight_update_sec", "regularization_sec")
    )
    recorder.add("candidate_core_initialisation", internal.get("initialize_alignment_sec", 0.0))
    recorder.add("alignment_fit", alignment_internal)
    recorder.add("core_update_iterations", core_internal)
    measured_internal = (
        alignment_internal
        + core_internal
        + float(internal.get("convergence_check_sec", 0.0))
        + float(internal.get("prepare_input_sec", 0.0))
    )
    recorder.add("unattributed_algorithm_overhead", max(0.0, algorithm_seconds - measured_internal))

    with recorder.stage("transform_apply"):
        if write_trajectory:
            aligned = np.empty_like(coords, dtype=float)
            for frame, source in zip(result.frame_results, coords, strict=True):
                aligned[frame.frame_index] = apply_transform(source, frame.result)
        else:
            aligned = robust_aligned_coords_from_result(result)

    with recorder.stage("ordinary_alignment_control"):
        ordinary_aligned = align_trajectory_ordinary(coords, reference_frame_index=0)
    with recorder.stage("rmsd_compute"):
        unified_rmsd = mean_rmsd(aligned, coords[0])
    with recorder.stage("rmsf_compute"):
        robust_rmsf = compute_rmsf(aligned)
        ordinary_rmsf = compute_rmsf(ordinary_aligned)
    weights = np.clip(mean_soft_weights_from_result(result), 0.0, 1.0)
    mobility = 1.0 - weights
    with recorder.stage("flexible_segment_calling"):
        segments = detect_flexible_segments(
            mobility,
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
        )
    with recorder.stage("diagnostics"):
        core = select_core_by_target_fraction(weights, 0.6)
        diagnostics = {
            "mean_rmsd": unified_rmsd,
            "mean_weighted_rmsd": result.mean_weighted_rmsd,
            "mean_iterations": result.average_iterations_per_frame,
            "converged_fraction": result.converged_fraction,
            "core_fraction": core.core_fraction,
            "n_flexible_segments": int(np.max(segments)) if segments.size else 0,
        }
    summary_path = workdir / "summary.json"
    with recorder.stage("output_serialisation"):
        write_json(summary_path, diagnostics)
    if write_trajectory:
        with recorder.stage("aligned_trajectory_write"):
            write_multimodel_pdb(aligned, workdir / "aligned_trajectory.pdb")

    frame_array_bytes = int(
        coords.shape[0]
        * (
            coords.shape[1] * 3 * 8
            + coords.shape[1] * 8 * 2
            + 12 * 8
        )
    )
    arrays = [
        array_record("robust_aligned_coords", aligned),
        array_record("ordinary_aligned_coords", ordinary_aligned),
        array_record("robust_rmsf", robust_rmsf),
        array_record("ordinary_rmsf", ordinary_rmsf),
        array_record("mean_soft_weights", weights),
        {
            "name": "default_frame_result_arrays_estimated",
            "shape": [coords.shape[0], coords.shape[1]],
            "dtype": "mixed_float64",
            "nbytes": frame_array_bytes,
            "owns_data": True,
            "is_view": False,
            "base_type": None,
        },
    ]
    return diagnostics, [value for value in arrays if value is not None], {
        "algorithm_runtime_seconds": algorithm_seconds,
        "instrumentation_source": "TrajectoryAlignmentResult.timing plus explicit output stages",
        "stage_limitations": (
            "Per-iteration weighted fits are separated from residual/weight updates. "
            "Final refits and Python orchestration remain in unattributed_algorithm_overhead."
        ),
    }


def _long_alignment(
    coords: np.ndarray,
    recorder: StageRecorder,
    *,
    write_trajectory: bool,
    workdir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Profile current uniform-sampling long-mode without changing its solver."""

    algorithm_start = time.perf_counter()
    result = solve_long_trajectory_alignment(
        coords,
        mode="long",
        max_weight_fit_frames=min(2000, coords.shape[0]),
        frame_selection="uniform",
        random_seed=42,
        chunk_size=min(2000, coords.shape[0]),
        store_aligned=False,
        **_robust_kwargs(),
    )
    algorithm_seconds = time.perf_counter() - algorithm_start
    timing = result.timing
    recorder.add("candidate_core_initialisation", timing.get("representative_selection", 0.0))
    recorder.add("core_update_iterations", timing.get("iterative_weight_estimation", 0.0))
    recorder.add("alignment_fit", timing.get("weighted_fit", 0.0))
    recorder.add("residual_profile_compute", timing.get("residual_update", 0.0))
    recorder.add("rmsf_compute", timing.get("RMSF", 0.0))
    measured = sum(
        float(timing.get(key, 0.0))
        for key in (
            "input_load",
            "representative_selection",
            "iterative_weight_estimation",
            "weighted_fit",
            "residual_update",
            "RMSF",
        )
    )
    recorder.add("unattributed_algorithm_overhead", max(0.0, algorithm_seconds - measured))

    profile_start = time.perf_counter()
    segment_before = float(timing.get("flexible_segment_detection", 0.0))
    profile = build_long_flexibility_profile(result, min_segment_length=3, merge_gap=1)
    profile_seconds = time.perf_counter() - profile_start
    segment_seconds = float(timing.get("flexible_segment_detection", 0.0)) - segment_before
    recorder.add("flexible_segment_calling", max(0.0, segment_seconds))
    recorder.add("output_table_build", max(0.0, profile_seconds - segment_seconds))

    aligned = None
    if write_trajectory:
        with recorder.stage("transform_apply"):
            aligned = aligned_coords_from_long_result(result, coords)
    with recorder.stage("rmsd_compute"):
        unified_rmsd = result.mean_rmsd
    with recorder.stage("diagnostics"):
        core = select_core_by_target_fraction(result.frozen_weights, 0.6)
        diagnostics = {
            "mean_rmsd": unified_rmsd,
            "mean_weighted_rmsd": result.mean_weighted_rmsd,
            "mean_iterations_on_representatives": result.average_iterations_per_frame,
            "converged_fraction_on_representatives": result.converged_fraction,
            "core_fraction": core.core_fraction,
            "n_flexible_segments": (
                int(np.max(profile.flexible_segment_id))
                if profile.flexible_segment_id.size
                else 0
            ),
            "selected_frame_count": int(result.selected_frame_indices.size),
        }
    with recorder.stage("output_serialisation"):
        write_json(workdir / "summary.json", diagnostics)
    if write_trajectory and aligned is not None:
        with recorder.stage("aligned_trajectory_write"):
            write_multimodel_pdb(aligned, workdir / "aligned_trajectory.pdb")

    arrays = [
        array_record("rotations", result.rotations),
        array_record("translations", result.translations),
        array_record("frozen_weights", result.frozen_weights),
        array_record("robust_rmsf", result.robust_rmsf),
        array_record("ordinary_rmsf", result.ordinary_rmsf),
        array_record("mean_residual", result.mean_residual),
        array_record("rmsd", result.rmsd),
        array_record("weighted_rmsd", result.weighted_rmsd),
        array_record("materialized_aligned_coords", aligned),
    ]
    return diagnostics, [value for value in arrays if value is not None], {
        "algorithm_runtime_seconds": algorithm_seconds,
        "instrumentation_source": "LongTrajectoryAlignmentResult.timing plus explicit output stages",
        "stage_limitations": (
            "The current long-mode RMSF timer includes ordinary-control batch fitting and "
            "streaming RMSF accumulation; those subcomponents are not separately exposed."
        ),
    }


def _mdlovofit(
    coords: np.ndarray,
    recorder: StageRecorder,
    *,
    workdir: Path,
    executable: Path,
    timeout_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Profile real external MDLovoFit with I/O separated where observable."""

    input_path = workdir / "trajectory_input.pdb"
    output_path = workdir / "trajectory_aligned.pdb"
    with recorder.stage("external_input_write"):
        write_multimodel_pdb(coords, input_path)
    subprocess_start = time.perf_counter()
    run = run_mdlovofit_on_files(
        [input_path],
        fraction=0.6,
        output_path=output_path,
        executable=executable,
        timeout_seconds=timeout_seconds,
    )
    subprocess_seconds = time.perf_counter() - subprocess_start
    recorder.add("alignment_fit", subprocess_seconds)
    with recorder.stage("external_output_read"):
        aligned, occupancies = read_multimodel_pdb_with_occupancy(output_path, coords.shape[1])
    with recorder.stage("rmsd_compute"):
        unified_rmsd = mean_rmsd(aligned, coords[0])
    with recorder.stage("diagnostics"):
        parsed = parse_mdlovofit_stdout(run.stdout)
        diagnostics = {
            "mean_rmsd": unified_rmsd,
            "native_all_atom_rmsd": parsed.average_rmsd_all_atoms,
            "native_least_mobile_rmsd": parsed.average_rmsd_least_mobile,
            "native_fraction": parsed.fraction_of_best_aligned_atoms,
            "return_code": run.return_code,
        }
    recorder.values["aligned_trajectory_write"] = 0.0
    arrays = [
        array_record("external_aligned_coords", aligned),
        array_record("external_occupancies", occupancies),
    ]
    return diagnostics, [value for value in arrays if value is not None], {
        "algorithm_runtime_seconds": run.subprocess_runtime_seconds,
        "instrumentation_source": "external wrapper boundaries",
        "stage_limitations": (
            "MDLovoFit native core search, fitting and output-file writing occur inside one "
            "subprocess and cannot be separated without modifying the external executable."
        ),
    }


def _theseus(
    coords: np.ndarray,
    recorder: StageRecorder,
    *,
    workdir: Path,
    executable: Path,
    timeout_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Profile real external THESEUS with observable I/O boundaries."""

    input_path = workdir / "trajectory_input.pdb"
    with recorder.stage("external_input_write"):
        write_multimodel_pdb(coords, input_path)
    subprocess_start = time.perf_counter()
    run = run_theseus_on_files(
        [input_path],
        output_root="theseus_profile",
        output_directory=workdir,
        executable=executable,
        timeout_seconds=timeout_seconds,
    )
    subprocess_seconds = time.perf_counter() - subprocess_start
    recorder.add("alignment_fit", subprocess_seconds)
    with recorder.stage("external_output_read"):
        aligned = read_multimodel_pdb_coords(run.aligned_pdb_path, coords.shape[1])
    with recorder.stage("rmsd_compute"):
        unified_rmsd = mean_rmsd(aligned, coords[0])
    with recorder.stage("diagnostics"):
        parsed = parse_theseus_stdout(run.stdout)
        variance_records = parse_theseus_variances(run.variances_path)
        diagnostics = {
            "mean_rmsd": unified_rmsd,
            "native_ml_rmsd": parsed.maximum_likelihood_rmsd,
            "native_ls_rmsd": parsed.classical_pairwise_rmsd,
            "variance_record_count": len(variance_records),
            "return_code": run.return_code,
        }
    recorder.values["aligned_trajectory_write"] = 0.0
    return diagnostics, [array_record("external_aligned_coords", aligned)], {
        "algorithm_runtime_seconds": run.subprocess_runtime_seconds,
        "instrumentation_source": "external wrapper boundaries",
        "stage_limitations": (
            "THESEUS optimisation, transform application and output writing occur inside one "
            "subprocess and cannot be separated without modifying the external executable."
        ),
    }


def _execute_child(args: argparse.Namespace) -> dict[str, object]:
    """Execute one profiled configuration and return its JSON record."""

    enabled = bool(
        args.profile_stages
        or os.environ.get("MDCOREALIGN_PROFILE_STAGES", "").strip().lower() in {"1", "true", "yes"}
    )
    recorder = StageRecorder(enabled=enabled)
    memory = MemorySampler(interval_seconds=args.memory_sample_interval)
    total_start = time.perf_counter()
    memory.start()
    status = "success"
    failure = ""
    diagnostics: dict[str, object] = {}
    arrays: list[dict[str, object]] = []
    details: dict[str, object] = {}
    coords: np.ndarray | None = None
    work_root = args.output_root / "profiling/work"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.config == "mdcorealign_gpu":
            status = "skipped_unavailable"
            failure = (
                "No current GPU execution path exists in src/mdcorealign, src/mdsuperpose, "
                "or src/mdstablecore; CUDA/Nsight availability alone is not a GPU implementation."
            )
            details = {
                "algorithm_runtime_seconds": None,
                "instrumentation_source": "static execution-path audit",
                "stage_limitations": failure,
            }
        else:
            coords, input_arrays = _load_coords(args.input_path, args.n_frames, recorder)
            arrays.extend(input_arrays)
            if coords.shape[1] != args.expected_residues:
                raise ValueError(
                    f"expected {args.expected_residues} residues but loaded {coords.shape[1]}"
                )
            with recorder.stage("preprocess_pbc"):
                pass
            with recorder.stage("atom_selection"):
                pass
            with recorder.stage("reference_selection"):
                reference = coords[0]
            arrays.append(array_record("reference_frame", reference))

            if args.config == "input_decode_only":
                diagnostics = {"decoded_frames": coords.shape[0], "decoded_atoms": coords.shape[1]}
                details = {
                    "algorithm_runtime_seconds": 0.0,
                    "instrumentation_source": "NPZ open/decode control",
                    "stage_limitations": "Input is already CA-selected; topology and PBC stages are not performed.",
                }
            else:
                with tempfile.TemporaryDirectory(
                    prefix=f"{args.run_id}_",
                    dir=work_root,
                ) as tmp:
                    workdir = Path(tmp)
                    if args.config.startswith("mdcorealign_default"):
                        diagnostics, method_arrays, details = _default_alignment(
                            coords,
                            recorder,
                            write_trajectory=config_writes_trajectory(args.config),
                            workdir=workdir,
                        )
                    elif args.config.startswith("mdcorealign_long"):
                        diagnostics, method_arrays, details = _long_alignment(
                            coords,
                            recorder,
                            write_trajectory=config_writes_trajectory(args.config),
                            workdir=workdir,
                        )
                    elif args.config == "mdlovofit":
                        diagnostics, method_arrays, details = _mdlovofit(
                            coords,
                            recorder,
                            workdir=workdir,
                            executable=args.mdlovofit_executable,
                            timeout_seconds=args.external_timeout,
                        )
                    elif args.config == "theseus":
                        diagnostics, method_arrays, details = _theseus(
                            coords,
                            recorder,
                            workdir=workdir,
                            executable=args.theseus_executable,
                            timeout_seconds=args.external_timeout,
                        )
                    elif args.config == "write_only":
                        with recorder.stage("aligned_trajectory_write"):
                            write_multimodel_pdb(coords, workdir / "write_only.pdb")
                        diagnostics = {"written_frames": coords.shape[0]}
                        method_arrays = []
                        details = {
                            "algorithm_runtime_seconds": 0.0,
                            "instrumentation_source": "PDB output control",
                            "stage_limitations": "Measures current CA-only multi-model PDB writer.",
                        }
                    else:
                        raise ValueError(f"unknown config: {args.config}")
                    arrays.extend(value for value in method_arrays if value is not None)
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        failure = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status = "failed"
        failure = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc()
    finally:
        recorder.values["total"] = time.perf_counter() - total_start
        memory.stop()

    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "generated_at": _timestamp(),
        "dataset_id": args.dataset_id,
        "input_path": _display_input_path(args.input_path),
        "data_semantics": args.data_semantics,
        "n_frames": args.n_frames,
        "n_residues": args.expected_residues,
        "config": args.config,
        "method": config_method(args.config),
        "write_aligned_trajectory": config_writes_trajectory(args.config),
        "repeat_id": args.repeat_id,
        "warmup": args.warmup,
        "status": status,
        "failure_or_skip_reason": failure,
        "stage_profiling_enabled": enabled,
        "stages_seconds": recorder.to_dict(),
        "memory": memory.to_dict(),
        "largest_arrays": sorted(
            [value for value in arrays if value is not None],
            key=lambda value: int(value.get("nbytes", 0)),
            reverse=True,
        ),
        "diagnostics": diagnostics,
        "details": details,
        "preprocessing": {
            "topology_load": "not_applicable_preselected_CA_array",
            "atom_selection": "not_applicable_preselected_CA_array",
            "pbc_nojump_centering": "not_performed_input_analyzed_as_provided",
        },
        "gpu": {
            "backend": "unavailable_no_current_execution_path",
            "host_to_device_seconds": None,
            "gpu_compute_seconds": None,
            "device_to_host_seconds": None,
        },
        "thread_settings": {
            key: os.environ.get(key, "unset")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def config_method(config: str) -> str:
    """Map one matrix configuration to a comparable method label."""

    if config.startswith("mdcorealign_default"):
        return "mdcorealign_default"
    if config.startswith("mdcorealign_long"):
        return "mdcorealign_long"
    if config == "mdcorealign_gpu":
        return "mdcorealign_gpu"
    return config


def config_writes_trajectory(config: str) -> bool:
    """Return whether this matrix configuration materializes a trajectory file."""

    return config in {
        "mdcorealign_default_write",
        "mdcorealign_long_write",
        "write_only",
    }


def _child_main(args: argparse.Namespace) -> int:
    """Run one child, optionally under cProfile/tracemalloc."""

    profile: cProfile.Profile | None = None
    if args.cprofile_output:
        profile = cProfile.Profile()
        profile.enable()
    if args.tracemalloc_output:
        tracemalloc.start(25)
    payload = _execute_child(args)
    if profile is not None:
        profile.disable()
        args.cprofile_output.parent.mkdir(parents=True, exist_ok=True)
        profile.dump_stats(args.cprofile_output)
    if args.tracemalloc_output:
        snapshot = tracemalloc.take_snapshot()
        lines = ["# Python allocation trace; NumPy native buffers are better represented by RSS."]
        for stat in snapshot.statistics("lineno")[:50]:
            lines.append(str(stat))
        args.tracemalloc_output.parent.mkdir(parents=True, exist_ok=True)
        args.tracemalloc_output.write_text("\n".join(lines) + "\n")
        tracemalloc.stop()
    write_json(args.raw_output, payload)
    return 0 if payload["status"] in {"success", "skipped_unavailable"} else 1


def _run_fresh_process(
    args: argparse.Namespace,
    *,
    dataset: dict[str, object],
    n_frames: int,
    config: str,
    repeat_id: int,
    warmup: bool,
) -> dict[str, object]:
    """Launch one isolated child and preserve failures/timeouts as JSON."""

    suffix = "warmup" if warmup else f"repeat_{repeat_id}"
    run_id = f"{dataset['dataset_id']}__f{n_frames}__{config}__{suffix}"
    raw_output = args.output_root / "profiling/raw_runs" / f"{run_id}.json"
    if raw_output.exists() and not args.overwrite:
        return json.loads(raw_output.read_text())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--profile-stages",
        "--output-root",
        str(args.output_root),
        "--raw-output",
        str(raw_output),
        "--run-id",
        run_id,
        "--dataset-id",
        str(dataset["dataset_id"]),
        "--input-path",
        str(PROJECT_ROOT / str(dataset["input_path"])),
        "--data-semantics",
        str(dataset["data_semantics"]),
        "--expected-residues",
        str(dataset["expected_residues"]),
        "--n-frames",
        str(n_frames),
        "--config",
        config,
        "--repeat-id",
        str(repeat_id),
        "--external-timeout",
        str(args.external_timeout),
        "--mdlovofit-executable",
        str(args.mdlovofit_executable),
        "--theseus-executable",
        str(args.theseus_executable),
    ]
    if warmup:
        command.append("--warmup")
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
            "MDCOREALIGN_PROFILE_STAGES": "1",
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}{os.pathsep}{PROJECT_ROOT}",
        }
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.process_timeout,
            check=False,
        )
        if raw_output.exists():
            payload = json.loads(raw_output.read_text())
            payload["fresh_process_wall_seconds"] = time.perf_counter() - started
            payload["child_return_code"] = completed.returncode
            payload["child_stdout"] = completed.stdout[-4000:]
            payload["child_stderr"] = completed.stderr[-4000:]
            payload["reproduction_command"] = reproduction_command(
                dataset, n_frames, config, repeat_id, args
            )
            write_json(raw_output, payload)
            return payload
        reason = f"child_return_code={completed.returncode}; stderr={completed.stderr[-2000:]}"
        status = "failed"
    except subprocess.TimeoutExpired as exc:
        reason = f"fresh-process timeout after {args.process_timeout}s: {exc}"
        status = "timeout"
    payload = failure_payload(
        run_id=run_id,
        dataset=dataset,
        n_frames=n_frames,
        config=config,
        repeat_id=repeat_id,
        warmup=warmup,
        status=status,
        reason=reason,
        elapsed=time.perf_counter() - started,
        args=args,
    )
    write_json(raw_output, payload)
    return payload


def reproduction_command(
    dataset: dict[str, object],
    n_frames: int,
    config: str,
    repeat_id: int,
    args: argparse.Namespace,
) -> str:
    """Return a repository-relative exact child command."""

    return (
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "PYTHONPATH=src:. python scripts/scaling_optimization/run_profile_matrix.py "
        f"--child --profile-stages --input-path {dataset['input_path']} "
        f"--dataset-id {dataset['dataset_id']} --data-semantics {dataset['data_semantics']} "
        f"--expected-residues {dataset['expected_residues']} --n-frames {n_frames} "
        f"--config {config} --repeat-id {repeat_id} "
        "--raw-output results/scaling_optimization_v1/profiling/raw_runs/REPRODUCED.json "
        f"--external-timeout {args.external_timeout}"
    )


def failure_payload(
    *,
    run_id: str,
    dataset: dict[str, object],
    n_frames: int,
    config: str,
    repeat_id: int,
    warmup: bool,
    status: str,
    reason: str,
    elapsed: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Create a machine-readable parent-side failure record."""

    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": _timestamp(),
        "dataset_id": dataset["dataset_id"],
        "input_path": dataset["input_path"],
        "data_semantics": dataset["data_semantics"],
        "n_frames": n_frames,
        "n_residues": dataset["expected_residues"],
        "config": config,
        "method": config_method(config),
        "write_aligned_trajectory": config_writes_trajectory(config),
        "repeat_id": repeat_id,
        "warmup": warmup,
        "status": status,
        "failure_or_skip_reason": reason,
        "stage_profiling_enabled": True,
        "stages_seconds": {"total": elapsed},
        "memory": {},
        "largest_arrays": [],
        "diagnostics": {},
        "details": {},
        "reproduction_command": reproduction_command(dataset, n_frames, config, repeat_id, args),
    }


def _run_profiler_reports(args: argparse.Namespace, dataset: dict[str, object]) -> None:
    """Produce deterministic cProfile and allocation reports on 500 frames."""

    profiling_root = args.output_root / "profiling"
    pstats_path = profiling_root / "cprofile_default_500.pstats"
    raw_output = profiling_root / "raw_runs/cprofile_default_500.json"
    trace_output = profiling_root / "python_allocation_trace.txt"
    child_args = argparse.Namespace(
        child=True,
        profile_stages=True,
        output_root=args.output_root,
        raw_output=raw_output,
        run_id="cprofile_default_500",
        dataset_id=dataset["dataset_id"],
        input_path=PROJECT_ROOT / str(dataset["input_path"]),
        data_semantics=dataset["data_semantics"],
        expected_residues=int(dataset["expected_residues"]),
        n_frames=500,
        config="mdcorealign_default_no_write",
        repeat_id=0,
        warmup=False,
        external_timeout=args.external_timeout,
        mdlovofit_executable=args.mdlovofit_executable,
        theseus_executable=args.theseus_executable,
        memory_sample_interval=0.01,
        cprofile_output=pstats_path,
        tracemalloc_output=trace_output,
    )
    _child_main(child_args)
    for sort_key, output_name in (
        ("cumulative", "cprofile_cumulative.txt"),
        ("time", "cprofile_self_time.txt"),
    ):
        with (profiling_root / output_name).open("w") as handle:
            stats = pstats.Stats(str(pstats_path), stream=handle)
            stats.strip_dirs().sort_stats(sort_key).print_stats(100)


def _write_optional_profiler_status(args: argparse.Namespace) -> None:
    """Record optional profiler availability and why GPU profilers were not run."""

    profiling_root = args.output_root / "profiling"
    py_spy = shutil.which("py-spy")
    if py_spy is None:
        (profiling_root / "flamegraph_unavailable.txt").write_text(
            "py-spy is not installed; no sampling flame graph was fabricated.\n"
        )
    (profiling_root / "nsight_systems_summary.txt").write_text(
        "Nsight Systems executable is available at /usr/bin/nsys, but the repository "
        "contains no current GPU execution path. Running Nsight would profile CPU-only "
        "Python and would not answer GPU transfer/launch/compute questions; marked not applicable.\n"
    )
    (profiling_root / "nsight_compute_summary.txt").write_text(
        "Nsight Compute executable is available at /usr/bin/ncu, but there are no "
        "MDCoreAlign CUDA kernels to profile in the current repository; marked not applicable.\n"
    )
    (profiling_root / "gpu_profiling_summary.txt").write_text(
        "GPU hardware and CUDA tooling are present, but no CuPy/PyTorch/JAX/CUDA execution "
        "path was found in src/mdcorealign, src/mdsuperpose or src/mdstablecore. "
        "host_to_device, gpu_compute and device_to_host timings are unavailable.\n"
    )


def _parent_main(args: argparse.Namespace) -> int:
    """Run the requested matrix and preserve every outcome."""

    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_root / "profiling/raw_runs"
    raw_root.mkdir(parents=True, exist_ok=True)
    frames_requested = tuple(
        int(value.strip()) for value in args.frame_counts.split(",") if value.strip()
    )
    configs_filter = set(
        value.strip() for value in args.configs.split(",") if value.strip()
    ) if args.configs else None
    datasets = [DATASETS[0]]
    if args.include_secondary:
        datasets.extend(DATASETS[1:])

    outcomes: list[dict[str, object]] = []
    for dataset in datasets:
        path = PROJECT_ROOT / str(dataset["input_path"])
        if not path.exists():
            for config in PRIMARY_CONFIGS if dataset["primary"] else SECONDARY_CONFIGS:
                payload = failure_payload(
                    run_id=f"{dataset['dataset_id']}__missing",
                    dataset=dataset,
                    n_frames=0,
                    config=config,
                    repeat_id=0,
                    warmup=False,
                    status="failed",
                    reason=f"input missing: {dataset['input_path']}",
                    elapsed=0.0,
                    args=args,
                )
                outcomes.append(payload)
            continue
        with np.load(path, allow_pickle=False) as payload:
            available_frames = int(payload["coords"].shape[0])
        configs = PRIMARY_CONFIGS if dataset["primary"] else SECONDARY_CONFIGS
        if configs_filter is not None:
            configs = tuple(value for value in configs if value in configs_filter)
        available_requested = [value for value in frames_requested if value <= available_frames]
        if args.warmup and available_requested:
            for config in configs:
                outcomes.append(
                    _run_fresh_process(
                        args,
                        dataset=dataset,
                        n_frames=min(12, available_frames),
                        config=config,
                        repeat_id=0,
                        warmup=True,
                    )
                )
        for n_frames in available_requested:
            for config in configs:
                for repeat_id in range(1, args.repeats + 1):
                    payload = _run_fresh_process(
                        args,
                        dataset=dataset,
                        n_frames=n_frames,
                        config=config,
                        repeat_id=repeat_id,
                        warmup=False,
                    )
                    outcomes.append(payload)
                    print(
                        f"[{len(outcomes)}] {dataset['dataset_id']} frames={n_frames} "
                        f"config={config} repeat={repeat_id} status={payload['status']}",
                        flush=True,
                    )

    manifest = {
        "generated_at": _timestamp(),
        "phase": "Phase 1",
        "seed": 42,
        "frame_counts_requested": frames_requested,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cpu_threads": args.cpu_threads,
        "process_timeout": args.process_timeout,
        "external_timeout": args.external_timeout,
        "datasets": datasets,
        "outcome_count": len(outcomes),
        "status_counts": {
            status: sum(row["status"] == status for row in outcomes)
            for status in sorted({str(row["status"]) for row in outcomes})
        },
        "primary_reproduction_command": (
            "PYTHONPATH=src:. python scripts/scaling_optimization/run_profile_matrix.py "
            "--profile-stages --frame-counts 12,100,500,1000,5000,10000 "
            "--repeats 3 --warmup --include-secondary"
        ),
    }
    write_json(args.output_root / "profiling/matrix_manifest.json", manifest)
    if not args.skip_profilers:
        _run_profiler_reports(args, DATASETS[0])
    _write_optional_profiler_status(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build parent and child arguments in one parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-counts", default=",".join(map(str, DEFAULT_FRAME_COUNTS)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--configs", default="")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--process-timeout", type=float, default=600.0)
    parser.add_argument("--external-timeout", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-profilers", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")

    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--raw-output", type=Path, default=Path("profile_run.json"), help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default="manual", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-id", default="manual", help=argparse.SUPPRESS)
    parser.add_argument("--input-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--data-semantics", default="unknown", help=argparse.SUPPRESS)
    parser.add_argument("--expected-residues", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--n-frames", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--config", default="input_decode_only", help=argparse.SUPPRESS)
    parser.add_argument("--repeat-id", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--memory-sample-interval", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--mdlovofit-executable", type=Path, default=Path("/usr/local/bin/mdlovofit"))
    parser.add_argument(
        "--theseus-executable",
        type=Path,
        default=Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus"),
    )
    parser.add_argument("--cprofile-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tracemalloc-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Dispatch parent matrix or one fresh-process child."""

    args = build_parser().parse_args()
    if args.child:
        if args.input_path is None and args.config != "mdcorealign_gpu":
            raise SystemExit("--input-path is required for child runs")
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
