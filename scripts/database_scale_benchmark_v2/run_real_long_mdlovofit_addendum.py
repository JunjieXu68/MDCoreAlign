#!/usr/bin/env python3
"""Run the prospectively locked M3 MDLovoFit real-long prefix panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
from typing import Any

import numpy as np

from mdsuperpose.mdlovofit import (
    build_mdlovofit_command,
    parse_mdlovofit_stdout,
    read_multimodel_pdb_with_occupancy,
)
from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.database_scale_benchmark_v2.common import (
    M_ROOT,
    ROOT,
    read_tsv,
    repository_path,
    sha256,
    write_json,
    write_tsv,
)


PANEL = (
    ROOT
    / "results/gpu_matched_validation_v1/u2_real_panel/real_continuous_panel.tsv"
)
MANIFEST_ROOT = M_ROOT / "manifests"
MANIFEST = MANIFEST_ROOT / "mdlovofit_real_long_prefix_manifest_locked.tsv"
CHECKPOINTS = M_ROOT / "real_long_prefix_checkpoints"
POLICY = MANIFEST_ROOT / "mdlovofit_registered_policy.json"
EXECUTABLE = Path("/usr/local/bin/mdlovofit")
PREFIXES = (250, 500, 1_000, 2_000, 5_000, 10_000)
MEASURED_REPEATS = {250: 5, 500: 5, 1_000: 3, 2_000: 3, 5_000: 1, 10_000: 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "run", "finalize"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _independent_rmsd(
    aligned_coords: np.ndarray, reference: np.ndarray
) -> tuple[float, float, float]:
    differences = np.asarray(aligned_coords, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )[None, :, :]
    per_frame = np.sqrt(np.mean(np.sum(differences * differences, axis=2), axis=1))
    return (
        float(np.mean(per_frame)),
        float(np.median(per_frame)),
        float(np.quantile(per_frame, 0.95)),
    )


def register() -> None:
    policy = json.loads(POLICY.read_text())
    timeout_policy = policy["real_long_timeout_seconds_by_prefix"]
    panel_rows = [row for row in read_tsv(PANEL) if row["status"] == "ready"]
    rows: list[dict[str, object]] = []
    for panel_row in panel_rows:
        available_frames = int(panel_row["n_frames"])
        for prefix in PREFIXES:
            if prefix > available_frames:
                continue
            rows.append(
                {
                    **panel_row,
                    "prefix_frames": prefix,
                    "prefix_label": f"p{prefix:05d}",
                    "timeout_seconds": int(timeout_policy[str(prefix)]),
                    "warmup_repeats": 1,
                    "measured_repeats": MEASURED_REPEATS[prefix],
                    "external_method": "MDLovoFit_20.1.1",
                    "fraction": 0.60,
                    "analysis_contract_includes_mandatory_aligned_PDB_write": True,
                    "input_contract": "same cached canonical CA coordinates",
                    "biological_continuity_claim": panel_row["continuity_class"],
                }
            )
    write_tsv(MANIFEST, rows)
    write_json(
        MANIFEST_ROOT / "mdlovofit_real_long_prefix_registration.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": repository_path(MANIFEST),
            "manifest_sha256": sha256(MANIFEST),
            "systems": len({row["system"] for row in rows}),
            "prefix_settings": len(rows),
            "requested_prefixes": list(PREFIXES),
            "available_prefix_counts": {
                str(prefix): sum(int(row["prefix_frames"]) == prefix for row in rows)
                for prefix in PREFIXES
            },
            "timeout_policy_seconds": timeout_policy,
            "repeat_policy": {
                str(prefix): {"warmups": 1, "measured": MEASURED_REPEATS[prefix]}
                for prefix in PREFIXES
            },
            "post_registration_panel_reduction_allowed": False,
            "ten_thousand_frame_availability": sum(
                int(row["prefix_frames"]) == 10_000 for row in rows
            ),
        },
    )


def _load_cached_coords(task: dict[str, str]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    with np.load(ROOT / task["cache_path"], allow_pickle=False) as archive:
        coords = np.asarray(archive["coords"], dtype=np.float64)
    elapsed = time.perf_counter() - start
    prefix = int(task["prefix_frames"])
    if coords.shape[0] < prefix:
        raise RuntimeError(f"cache has {coords.shape[0]} frames; prefix requires {prefix}")
    if coords.shape[1] != int(task["n_residues"]):
        raise RuntimeError("cached CA residue count does not match the locked manifest")
    values = np.ascontiguousarray(coords[:prefix])
    if not np.isfinite(values).all():
        raise RuntimeError("cached coordinates contain non-finite values")
    return values, elapsed


def _run_repeat(
    task: dict[str, str],
    *,
    repeat_index: int,
    is_warmup: bool,
    workspace: Path,
) -> dict[str, Any]:
    workflow_start = time.perf_counter()
    base: dict[str, Any] = {
        **task,
        "repeat_index": repeat_index,
        "is_warmup": is_warmup,
        "method": "MDLovoFit_20.1.1",
        "mdlovofit_executable": str(EXECUTABLE),
        "timing_contract": "M0_loaded_and_mandatory_materialised_output",
        "successful_aligned_PDB_retained": False,
    }
    input_path = workspace / "input.pdb"
    output_path = workspace / "aligned.pdb"
    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    try:
        coords, decode_seconds = _load_cached_coords(task)
        prep_start = time.perf_counter()
        write_multimodel_ca_pdb_streaming(coords, input_path)
        prep_seconds = time.perf_counter() - prep_start
        command = build_mdlovofit_command(
            EXECUTABLE,
            fraction=0.60,
            output_path=output_path,
            input_paths=[input_path],
        )
        subprocess_start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(task["timeout_seconds"]),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess_seconds = time.perf_counter() - subprocess_start
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return {
                **base,
                "status": "timeout",
                "failure_class": "prospectively_registered_timeout",
                "failure_reason": f"MDLovoFit exceeded {task['timeout_seconds']} seconds",
                "shared_input_decode_seconds": decode_seconds,
                "method_specific_input_preparation_seconds": prep_seconds,
                "method_execution_seconds": subprocess_seconds,
                "coordinate_materialisation_output_write_seconds": "embedded_in_mdlovofit_subprocess_not_separable",
                "output_parse_seconds": 0.0,
                "posthoc_evaluation_seconds": 0.0,
                "loaded_analysis_seconds": prep_seconds + subprocess_seconds,
                "materialised_output_contract_seconds": prep_seconds + subprocess_seconds,
                "complete_end_to_end_seconds": time.perf_counter() - workflow_start,
                "stdout_sha256": _digest(stdout),
                "stderr_sha256": _digest(stderr),
            }
        subprocess_seconds = time.perf_counter() - subprocess_start
        if completed.returncode != 0:
            return {
                **base,
                "status": "failed",
                "failure_class": "external_return_code",
                "failure_reason": f"MDLovoFit return code {completed.returncode}",
                "shared_input_decode_seconds": decode_seconds,
                "method_specific_input_preparation_seconds": prep_seconds,
                "method_execution_seconds": subprocess_seconds,
                "coordinate_materialisation_output_write_seconds": "embedded_in_mdlovofit_subprocess_not_separable",
                "output_parse_seconds": 0.0,
                "posthoc_evaluation_seconds": 0.0,
                "loaded_analysis_seconds": prep_seconds + subprocess_seconds,
                "materialised_output_contract_seconds": prep_seconds + subprocess_seconds,
                "complete_end_to_end_seconds": time.perf_counter() - workflow_start,
                "stdout_sha256": _digest(completed.stdout),
                "stderr_sha256": _digest(completed.stderr),
                "stderr_excerpt": completed.stderr[-2000:],
            }
        if not output_path.is_file():
            raise FileNotFoundError("MDLovoFit completed without an aligned PDB")
        parse_start = time.perf_counter()
        aligned, occupancies = read_multimodel_pdb_with_occupancy(
            output_path, n_atoms=coords.shape[1]
        )
        parse_seconds = time.perf_counter() - parse_start
        evaluation_start = time.perf_counter()
        unified, median_rmsd, p95_rmsd = _independent_rmsd(aligned, coords[0])
        posthoc_seconds = time.perf_counter() - evaluation_start
        parsed = parse_mdlovofit_stdout(completed.stdout)
        loaded_seconds = prep_seconds + subprocess_seconds + parse_seconds + posthoc_seconds
        return {
            **base,
            "status": "success",
            "failure_class": "",
            "failure_reason": "",
            "return_code": completed.returncode,
            "shared_input_decode_seconds": decode_seconds,
            "method_specific_input_preparation_seconds": prep_seconds,
            "method_execution_seconds": subprocess_seconds,
            "coordinate_materialisation_output_write_seconds": "embedded_in_mdlovofit_subprocess_not_separable",
            "output_parse_seconds": parse_seconds,
            "posthoc_evaluation_seconds": posthoc_seconds,
            "loaded_analysis_seconds": loaded_seconds,
            "materialised_output_contract_seconds": loaded_seconds,
            "complete_end_to_end_seconds": decode_seconds + loaded_seconds,
            "unified_posthoc_RMSD_A": unified,
            "median_frame_RMSD_A": median_rmsd,
            "p95_frame_RMSD_A": p95_rmsd,
            "native_all_atom_RMSD_A": parsed.average_rmsd_all_atoms,
            "native_least_mobile_RMSD_A": parsed.average_rmsd_least_mobile,
            "native_most_mobile_RMSD_A": parsed.average_rmsd_most_mobile,
            "native_reported_fraction": parsed.fraction_of_best_aligned_atoms,
            "native_total_atoms": parsed.total_number_of_atoms,
            "native_selection_atoms": parsed.number_of_atoms_of_selection,
            "occupancy_min_observed": float(np.min(occupancies)),
            "occupancy_max_observed": float(np.max(occupancies)),
            "occupancy_semantics_used_for_core_comparison": False,
            "input_PDB_bytes": input_path.stat().st_size,
            "aligned_PDB_bytes": output_path.stat().st_size,
            "stdout_sha256": _digest(completed.stdout),
            "stderr_sha256": _digest(completed.stderr),
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "failure_class": "input_parse_or_implementation_failure",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "complete_end_to_end_seconds": time.perf_counter() - workflow_start,
        }
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def run(limit: int | None, rerun: bool) -> None:
    tasks = read_tsv(MANIFEST)
    if limit is not None:
        tasks = tasks[:limit]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    temp_root = M_ROOT / "temporary_workspaces"
    temp_root.mkdir(parents=True, exist_ok=True)
    total_settings = len(tasks)
    with tempfile.TemporaryDirectory(prefix="real_long_", dir=temp_root) as temporary:
        workspace = Path(temporary)
        for setting_index, task in enumerate(tasks, start=1):
            total_repeats = 1 + int(task["measured_repeats"])
            for repeat_position in range(total_repeats):
                repeat_index = repeat_position - 1
                checkpoint = CHECKPOINTS / (
                    f"{_safe_id(task['system'])}__{task['prefix_label']}__r{repeat_index}.json"
                )
                if checkpoint.is_file() and not rerun:
                    continue
                result = _run_repeat(
                    task,
                    repeat_index=repeat_index,
                    is_warmup=repeat_position == 0,
                    workspace=workspace,
                )
                write_json(checkpoint, result)
            print(
                f"[MDLovoFit real-long] {setting_index}/{total_settings} "
                f"{task['system']} {task['prefix_label']}",
                flush=True,
            )


def _timing_summary(values: list[float], prefix: str) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        f"median_{prefix}": float(np.median(data)),
        f"q1_{prefix}": float(np.quantile(data, 0.25)),
        f"q3_{prefix}": float(np.quantile(data, 0.75)),
        f"iqr_{prefix}": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        f"min_{prefix}": float(np.min(data)),
        f"max_{prefix}": float(np.max(data)),
    }


def finalize() -> None:
    settings = read_tsv(MANIFEST)
    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in settings:
        for repeat_position in range(1 + int(task["measured_repeats"])):
            repeat_index = repeat_position - 1
            checkpoint = CHECKPOINTS / (
                f"{_safe_id(task['system'])}__{task['prefix_label']}__r{repeat_index}.json"
            )
            if not checkpoint.is_file():
                missing.append(checkpoint.name)
            else:
                all_rows.append(json.loads(checkpoint.read_text()))
    if missing:
        raise RuntimeError(f"real-long panel has {len(missing)} missing checkpoints")
    measured = [row for row in all_rows if not row["is_warmup"]]
    write_tsv(M_ROOT / "mdlovofit_real_long_prefixes_raw_repeats.tsv", all_rows)
    summaries: list[dict[str, Any]] = []
    for task in settings:
        group = [
            row
            for row in measured
            if row["system"] == task["system"]
            and row["prefix_label"] == task["prefix_label"]
        ]
        success = [row for row in group if row["status"] == "success"]
        summary: dict[str, Any] = {
            **task,
            "measured_completed": len(group),
            "success": len(success),
            "failed": sum(row["status"] == "failed" for row in group),
            "timeouts": sum(row["status"] == "timeout" for row in group),
            "completion_fraction": len(success) / max(1, len(group)),
        }
        if success:
            for field in (
                "shared_input_decode_seconds",
                "method_specific_input_preparation_seconds",
                "method_execution_seconds",
                "output_parse_seconds",
                "posthoc_evaluation_seconds",
                "loaded_analysis_seconds",
                "materialised_output_contract_seconds",
                "complete_end_to_end_seconds",
            ):
                summary.update(
                    _timing_summary([float(row[field]) for row in success], field)
                )
            summary.update(
                {
                    "median_unified_posthoc_RMSD_A": float(
                        np.median([float(row["unified_posthoc_RMSD_A"]) for row in success])
                    ),
                    "median_native_all_atom_RMSD_A": float(
                        np.median([float(row["native_all_atom_RMSD_A"]) for row in success])
                    ),
                    "median_native_least_mobile_RMSD_A": float(
                        np.median([float(row["native_least_mobile_RMSD_A"]) for row in success])
                    ),
                }
            )
        summaries.append(summary)
    write_tsv(M_ROOT / "mdlovofit_real_long_prefixes.tsv", summaries)
    failures = [row for row in measured if row["status"] != "success"]
    if failures:
        write_tsv(M_ROOT / "mdlovofit_real_long_failures_timeouts.tsv", failures)
    write_json(
        M_ROOT / "mdlovofit_real_long_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "settings": len(settings),
            "systems": len({row["system"] for row in settings}),
            "measured_runs": len(measured),
            "success": sum(row["status"] == "success" for row in measured),
            "failed": sum(row["status"] == "failed" for row in measured),
            "timeouts": sum(row["status"] == "timeout" for row in measured),
            "ten_thousand_frame_settings": sum(
                int(row["prefix_frames"]) == 10_000 for row in settings
            ),
        },
    )


def main() -> None:
    args = parse_args()
    if args.mode == "register":
        register()
    elif args.mode == "run":
        run(args.limit, args.rerun)
    else:
        finalize()


if __name__ == "__main__":
    main()
