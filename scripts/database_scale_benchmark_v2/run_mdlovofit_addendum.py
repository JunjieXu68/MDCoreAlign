#!/usr/bin/env python3
"""Run resumable MDLovoFit addendum panels under the registered M0 contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import traceback
from typing import Any, Iterable

import numpy as np

from mdsuperpose.mdlovofit import (
    build_mdlovofit_command,
    parse_mdlovofit_stdout,
    read_multimodel_pdb_with_occupancy,
)
from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.database_scale_benchmark_v2.common import (
    B2_ROOT,
    M_ROOT,
    ROOT,
    read_tsv,
    write_json,
    write_tsv,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)


MANIFEST_ROOT = M_ROOT / "manifests"
PILOT_CHECKPOINTS = M_ROOT / "pilot_checkpoints"
ALL_DOMAIN_CHECKPOINTS = M_ROOT / "all_domain_checkpoints"
DEPLOYMENT_CHECKPOINTS = M_ROOT / "deployment_domain_checkpoints"
EXTERNAL_EXECUTABLE = Path("/usr/local/bin/mdlovofit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "pilot",
            "finalize-pilot",
            "all-domain",
            "finalize-all-domain",
            "deployment",
            "finalize-deployment",
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def current_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def independent_frame_rmsd(
    aligned_coords: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float]:
    """Evaluate aligned coordinates without performing another fit."""

    aligned = np.asarray(aligned_coords, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if aligned.ndim != 3 or aligned.shape[1:] != target.shape:
        raise ValueError("aligned coordinates and reference have incompatible shapes")
    squared = np.einsum("fij,fij->fi", aligned - target[None, :, :], aligned - target[None, :, :])
    per_frame = np.sqrt(np.mean(squared, axis=1))
    return per_frame, float(np.mean(per_frame))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_coords(task: dict[str, str]) -> tuple[np.ndarray, dict[str, Any], float]:
    start = time.perf_counter()
    coords, _, residue_metadata, metadata = extract_ca_trajectory(
        ROOT / task["source_file"],
        coordinate_path=task["coordinate_path"],
        direct_contiguous_read=True,
    )
    seconds = time.perf_counter() - start
    values = np.asarray(coords, dtype=np.float64)
    if metadata["unit_status"] != "verified_angstrom_metadata":
        raise RuntimeError(f"unit_integrity: {metadata['unit_status']}")
    expected = int(task["expected_n_residues"])
    if values.shape[1] != expected or len(residue_metadata) != expected:
        raise RuntimeError(
            f"mapping_integrity: observed {values.shape[1]} coordinates and "
            f"{len(residue_metadata)} metadata rows; expected {expected}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("numerical_integrity: non-finite coordinates")
    return values, metadata, seconds


def run_one(
    task: dict[str, str],
    *,
    timeout_seconds: float,
    workspace: Path,
) -> dict[str, Any]:
    """Run one trajectory from canonical coordinates and return compact timings."""

    workflow_start = time.perf_counter()
    base: dict[str, Any] = {
        **task,
        "mdlovofit_fraction": 0.60,
        "timeout_seconds": timeout_seconds,
        "mdlovofit_executable": str(EXTERNAL_EXECUTABLE),
        "timing_contract": "M0_loaded_and_mandatory_materialised_output",
        "analysis_contract_includes_mandatory_aligned_PDB_write": True,
        "successful_aligned_PDB_retained": False,
    }
    try:
        coords, metadata, decode_seconds = load_coords(task)
        base.update(
            {
                "status": "running",
                "n_frames": int(coords.shape[0]),
                "n_residues": int(coords.shape[1]),
                "shared_HDF5_decode_seconds": decode_seconds,
                "shared_decode_read_strategy": metadata["read_strategy"],
                "unit_integrity": "pass",
                "mapping_integrity": "pass",
            }
        )
        input_path = workspace / "trajectory_input.pdb"
        output_path = workspace / "trajectory_aligned.pdb"
        output_path.unlink(missing_ok=True)

        prep_start = time.perf_counter()
        write_multimodel_ca_pdb_streaming(coords, input_path)
        input_prep_seconds = time.perf_counter() - prep_start
        input_bytes = input_path.stat().st_size

        command = build_mdlovofit_command(
            EXTERNAL_EXECUTABLE,
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
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess_seconds = time.perf_counter() - subprocess_start
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return {
                **base,
                "status": "timeout",
                "failure_class": "timeout",
                "failure_reason": f"MDLovoFit exceeded {timeout_seconds:g} seconds",
                "method_specific_input_preparation_seconds": input_prep_seconds,
                "mdlovofit_subprocess_including_mandatory_output_write_seconds": subprocess_seconds,
                "output_parse_seconds": 0.0,
                "posthoc_evaluation_seconds": 0.0,
                "loaded_analysis_seconds": input_prep_seconds + subprocess_seconds,
                "materialised_output_contract_seconds": input_prep_seconds + subprocess_seconds,
                "complete_end_to_end_seconds": time.perf_counter() - workflow_start,
                "input_PDB_bytes": input_bytes,
                "aligned_PDB_bytes": output_path.stat().st_size if output_path.exists() else 0,
                "stdout_sha256": _digest(stdout),
                "stderr_sha256": _digest(stderr),
                "peak_RSS_mb": current_rss_mb(),
            }
        subprocess_seconds = time.perf_counter() - subprocess_start
        if completed.returncode != 0:
            return {
                **base,
                "status": "failed",
                "failure_class": "external_return_code",
                "failure_reason": f"MDLovoFit return code {completed.returncode}",
                "method_specific_input_preparation_seconds": input_prep_seconds,
                "mdlovofit_subprocess_including_mandatory_output_write_seconds": subprocess_seconds,
                "output_parse_seconds": 0.0,
                "posthoc_evaluation_seconds": 0.0,
                "loaded_analysis_seconds": input_prep_seconds + subprocess_seconds,
                "materialised_output_contract_seconds": input_prep_seconds + subprocess_seconds,
                "complete_end_to_end_seconds": time.perf_counter() - workflow_start,
                "input_PDB_bytes": input_bytes,
                "aligned_PDB_bytes": output_path.stat().st_size if output_path.exists() else 0,
                "stdout_sha256": _digest(completed.stdout),
                "stderr_sha256": _digest(completed.stderr),
                "stderr_excerpt": completed.stderr[-2000:],
                "peak_RSS_mb": current_rss_mb(),
            }
        if not output_path.is_file():
            raise FileNotFoundError("MDLovoFit completed without the mandatory aligned PDB")

        parse_start = time.perf_counter()
        aligned, occupancies = read_multimodel_pdb_with_occupancy(
            output_path, n_atoms=coords.shape[1]
        )
        output_parse_seconds = time.perf_counter() - parse_start
        evaluation_start = time.perf_counter()
        per_frame_rmsd, unified_rmsd = independent_frame_rmsd(aligned, coords[0])
        posthoc_seconds = time.perf_counter() - evaluation_start
        parsed = parse_mdlovofit_stdout(completed.stdout)
        output_bytes = output_path.stat().st_size
        loaded_seconds = (
            input_prep_seconds
            + subprocess_seconds
            + output_parse_seconds
            + posthoc_seconds
        )
        result = {
            **base,
            "status": "success",
            "failure_class": "",
            "failure_reason": "",
            "return_code": completed.returncode,
            "method_specific_input_preparation_seconds": input_prep_seconds,
            "mdlovofit_subprocess_including_mandatory_output_write_seconds": subprocess_seconds,
            "coordinate_materialisation_output_write_seconds": "embedded_in_mdlovofit_subprocess_not_separable",
            "output_parse_seconds": output_parse_seconds,
            "posthoc_evaluation_seconds": posthoc_seconds,
            "loaded_analysis_seconds": loaded_seconds,
            "materialised_output_contract_seconds": loaded_seconds,
            "complete_end_to_end_seconds": decode_seconds + loaded_seconds,
            "unified_posthoc_RMSD_A": unified_rmsd,
            "median_frame_RMSD_A": float(np.median(per_frame_rmsd)),
            "p95_frame_RMSD_A": float(np.quantile(per_frame_rmsd, 0.95)),
            "native_all_atom_RMSD_A": parsed.average_rmsd_all_atoms,
            "native_least_mobile_RMSD_A": parsed.average_rmsd_least_mobile,
            "native_most_mobile_RMSD_A": parsed.average_rmsd_most_mobile,
            "native_reported_fraction": parsed.fraction_of_best_aligned_atoms,
            "native_total_atoms": parsed.total_number_of_atoms,
            "native_selection_atoms": parsed.number_of_atoms_of_selection,
            "native_frame_RMSD_rows": parsed.frame_rmsd_row_count,
            "occupancy_min_observed": float(np.min(occupancies)),
            "occupancy_max_observed": float(np.max(occupancies)),
            "occupancy_semantics_used_for_core_comparison": False,
            "input_PDB_bytes": input_bytes,
            "aligned_PDB_bytes": output_bytes,
            "stdout_sha256": _digest(completed.stdout),
            "stderr_sha256": _digest(completed.stderr),
            "peak_RSS_mb": current_rss_mb(),
        }
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result
    except Exception as exc:
        text = str(exc).lower()
        failure_class = (
            "unit_integrity_failure"
            if "unit_integrity" in text
            else "mapping_integrity_failure"
            if "mapping_integrity" in text
            else "numerical_or_implementation_failure"
        )
        return {
            **base,
            "status": "failed",
            "failure_class": failure_class,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "complete_end_to_end_seconds": time.perf_counter() - workflow_start,
            "peak_RSS_mb": current_rss_mb(),
        }


def load_manifest(path: Path) -> list[dict[str, str]]:
    return read_tsv(path)


def shared_checkpoint(trajectory_id: str) -> dict[str, Any] | None:
    for root in (PILOT_CHECKPOINTS, ALL_DOMAIN_CHECKPOINTS):
        path = root / f"{trajectory_id}.json"
        if path.is_file():
            return json.loads(path.read_text())
    return None


def run_individual_panel(
    rows: list[dict[str, str]],
    checkpoint_root: Path,
    *,
    limit: int | None,
    rerun: bool,
    label: str,
) -> None:
    pending = [
        row
        for row in rows
        if rerun or not (checkpoint_root / f"{row['trajectory_id']}.json").is_file()
    ]
    if limit is not None:
        pending = pending[:limit]
    temp_root = M_ROOT / "temporary_workspaces"
    temp_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    successes = failures = timeouts = 0
    with tempfile.TemporaryDirectory(prefix=f"{label}_", dir=temp_root) as temporary:
        workspace = Path(temporary)
        for index, task in enumerate(pending, start=1):
            reused = None if rerun else shared_checkpoint(task["trajectory_id"])
            row = reused if reused is not None else run_one(
                task, timeout_seconds=120.0, workspace=workspace
            )
            write_json(checkpoint_root / f"{task['trajectory_id']}.json", row)
            successes += row["status"] == "success"
            failures += row["status"] == "failed"
            timeouts += row["status"] == "timeout"
            if index % 25 == 0 or index == len(pending):
                elapsed = time.perf_counter() - start
                print(
                    f"[{label}] {index}/{len(pending)} new; success={successes}; "
                    f"failed={failures}; timeout={timeouts}; "
                    f"rate={3600 * index / max(elapsed, 1e-9):.1f} trajectories/hour",
                    flush=True,
                )
    write_json(
        M_ROOT / f"latest_{label}_run_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "new_rows": len(pending),
            "success": successes,
            "failed": failures,
            "timeout": timeouts,
            "wall_seconds": time.perf_counter() - start,
        },
    )


def _checkpoint_rows(
    manifest: list[dict[str, str]], checkpoint_root: Path
) -> list[dict[str, Any]]:
    paths = [checkpoint_root / f"{row['trajectory_id']}.json" for row in manifest]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{len(missing):,} checkpoints are missing")
    return [json.loads(path.read_text()) for path in paths]


def _sum(rows: Iterable[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0) or 0.0) for row in rows)


def linear_projection(
    success: list[dict[str, Any]], full_manifest: list[dict[str, str]]
) -> float:
    """Project serial workflow wall time using locked size/frame covariates."""

    d1_by_domain = {
        row["system_id"]: row
        for row in read_tsv(
            ROOT
            / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv"
        )
    }

    def covariates(row: dict[str, Any]) -> tuple[float, float]:
        if row.get("n_frames") and row.get("n_residues"):
            return float(row["n_frames"]), float(row["n_residues"])
        evidence = d1_by_domain[row["domain_id"]]
        return float(evidence["n_frames"]), float(evidence["n_residues"])

    x = np.asarray(
        [
            [1.0, n_frames, n_residues, n_frames * n_residues]
            for row in success
            for n_frames, n_residues in [covariates(row)]
        ],
        dtype=float,
    )
    y = np.asarray([float(row["complete_end_to_end_seconds"]) for row in success])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    target = np.asarray(
        [
            [1.0, n_frames, n_residues, n_frames * n_residues]
            for row in full_manifest
            for n_frames, n_residues in [covariates(row)]
        ],
        dtype=float,
    )
    return float(np.sum(np.clip(target @ coefficients, 0.0, None)))


def finalize_pilot() -> None:
    manifest = load_manifest(MANIFEST_ROOT / "mdlovofit_pilot_2000_locked.tsv")
    rows = _checkpoint_rows(manifest, PILOT_CHECKPOINTS)
    write_tsv(M_ROOT / "mdlovofit_full_database_pilot.tsv", rows)
    success = [row for row in rows if row["status"] == "success"]
    failures = [row for row in rows if row["status"] != "success"]
    full_manifest = [
        row
        for row in read_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv")
        if row["locally_available"].lower() == "true"
    ]
    raw_projection = (
        _sum(rows, "complete_end_to_end_seconds") / len(rows) * len(full_manifest)
    )
    model_projection = linear_projection(success, full_manifest) if success else math.inf
    conservative_projection = max(raw_projection, model_projection)
    pilot_tsv_bytes = (M_ROOT / "mdlovofit_full_database_pilot.tsv").stat().st_size
    compact_projection = int(pilot_tsv_bytes / len(rows) * len(full_manifest))
    peak_temporary = max(
        int(row.get("input_PDB_bytes", 0) or 0) + int(row.get("aligned_PDB_bytes", 0) or 0)
        for row in rows
    )
    failure_fraction = len(failures) / len(rows)
    policy = json.loads(
        (MANIFEST_ROOT / "mdlovofit_registered_policy.json").read_text()
    )["feasibility_decision_rule"]
    pass_wall = conservative_projection <= policy["projected_full_wall_seconds_max"]
    pass_failure = failure_fraction <= policy["failure_plus_timeout_fraction_max"]
    pass_compact = compact_projection <= policy["compact_summary_storage_bytes_max"]
    pass_temp = peak_temporary <= policy["peak_temporary_working_storage_bytes_max"]
    run_full = pass_wall and pass_failure and pass_compact and pass_temp
    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_rows": len(rows),
        "success": len(success),
        "failed": sum(row["status"] == "failed" for row in rows),
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "failure_plus_timeout_fraction": failure_fraction,
        "shared_decode_seconds": _sum(success, "shared_HDF5_decode_seconds"),
        "input_preparation_seconds": _sum(success, "method_specific_input_preparation_seconds"),
        "subprocess_including_mandatory_write_seconds": _sum(success, "mdlovofit_subprocess_including_mandatory_output_write_seconds"),
        "output_parse_seconds": _sum(success, "output_parse_seconds"),
        "posthoc_evaluation_seconds": _sum(success, "posthoc_evaluation_seconds"),
        "loaded_analysis_seconds": _sum(success, "loaded_analysis_seconds"),
        "complete_workflow_seconds": _sum(rows, "complete_end_to_end_seconds"),
        "raw_expansion_projected_full_wall_seconds": raw_projection,
        "size_frame_model_projected_full_wall_seconds": model_projection,
        "conservative_projected_full_wall_seconds": conservative_projection,
        "projected_compact_summary_bytes": compact_projection,
        "projected_persistent_aligned_PDB_bytes_if_retained": int(
            sum(int(row.get("aligned_PDB_bytes", 0) or 0) for row in success)
            / max(1, len(success))
            * len(full_manifest)
        ),
        "peak_temporary_working_storage_bytes": peak_temporary,
        "successful_aligned_PDB_retention_policy": "not_retained_at_database_scale",
        "gate_projected_wall": pass_wall,
        "gate_failure_timeout": pass_failure,
        "gate_compact_storage": pass_compact,
        "gate_peak_temporary_storage": pass_temp,
        "decision": "full_134925" if run_full else "locked_fallback_10000",
        "decision_panel_rows": len(full_manifest) if run_full else 10_000,
        "post_hoc_threshold_or_panel_change": False,
    }
    write_json(M_ROOT / "mdlovofit_deployment_execution_decision.json", decision)
    if failures:
        write_tsv(M_ROOT / "mdlovofit_pilot_failures_timeouts.tsv", failures)
    print(json.dumps(decision, indent=2))


def run_deployment(limit: int | None, rerun: bool) -> None:
    decision = json.loads(
        (M_ROOT / "mdlovofit_deployment_execution_decision.json").read_text()
    )
    manifest_name = (
        B2_ROOT / "b3_all_trajectory_manifest_locked.tsv"
        if decision["decision"] == "full_134925"
        else MANIFEST_ROOT / "mdlovofit_fallback_10000_locked.tsv"
    )
    manifest = [
        row
        for row in read_tsv(manifest_name)
        if row.get("locally_available", "true").lower() == "true"
    ]
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        by_domain.setdefault(row["domain_id"], []).append(row)
    domains = sorted(by_domain)
    pending = [
        domain
        for domain in domains
        if rerun or not (DEPLOYMENT_CHECKPOINTS / f"{domain}.json").is_file()
    ]
    if limit is not None:
        pending = pending[:limit]
    temp_root = M_ROOT / "temporary_workspaces"
    temp_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    success = failed = timeout = 0
    with tempfile.TemporaryDirectory(prefix="deployment_", dir=temp_root) as temporary:
        workspace = Path(temporary)
        for index, domain in enumerate(pending, start=1):
            output = []
            for task in sorted(
                by_domain[domain],
                key=lambda row: (int(row["temperature_K"]), int(row["replicate"])),
            ):
                reused = None if rerun else shared_checkpoint(task["trajectory_id"])
                row = reused if reused is not None else run_one(
                    task, timeout_seconds=120.0, workspace=workspace
                )
                output.append(row)
                success += row["status"] == "success"
                failed += row["status"] == "failed"
                timeout += row["status"] == "timeout"
            write_json(DEPLOYMENT_CHECKPOINTS / f"{domain}.json", output)
            if index % 10 == 0 or index == len(pending):
                elapsed = time.perf_counter() - start
                completed = success + failed + timeout
                print(
                    f"[mdlovofit deployment] domains {index}/{len(pending)}; "
                    f"success={success}; failed={failed}; timeout={timeout}; "
                    f"rate={3600 * completed / max(elapsed, 1e-9):.1f} trajectories/hour",
                    flush=True,
                )


def finalize_deployment() -> None:
    decision = json.loads(
        (M_ROOT / "mdlovofit_deployment_execution_decision.json").read_text()
    )
    manifest_path = (
        B2_ROOT / "b3_all_trajectory_manifest_locked.tsv"
        if decision["decision"] == "full_134925"
        else MANIFEST_ROOT / "mdlovofit_fallback_10000_locked.tsv"
    )
    manifest = [
        row
        for row in read_tsv(manifest_path)
        if row.get("locally_available", "true").lower() == "true"
    ]
    domains = sorted({row["domain_id"] for row in manifest})
    missing = [
        domain for domain in domains if not (DEPLOYMENT_CHECKPOINTS / f"{domain}.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"deployment has {len(missing):,} missing domain checkpoints")
    rows = [
        row
        for domain in domains
        for row in json.loads((DEPLOYMENT_CHECKPOINTS / f"{domain}.json").read_text())
    ]
    if len(rows) != len(manifest):
        raise RuntimeError(f"deployment aggregate has {len(rows):,}/{len(manifest):,} rows")
    write_tsv(M_ROOT / "mdlovofit_external_panel.tsv", rows)
    failures = [row for row in rows if row["status"] != "success"]
    if failures:
        write_tsv(M_ROOT / "mdlovofit_failures_timeouts.tsv", failures)
    summary = {
        "decision": decision["decision"],
        "rows": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "shared_decode_seconds": _sum(rows, "shared_HDF5_decode_seconds"),
        "loaded_analysis_seconds": _sum(rows, "loaded_analysis_seconds"),
        "complete_workflow_seconds": _sum(rows, "complete_end_to_end_seconds"),
    }
    write_json(M_ROOT / "mdlovofit_deployment_status.json", summary)
    print(json.dumps(summary, indent=2))


def finalize_all_domain() -> None:
    manifest = load_manifest(MANIFEST_ROOT / "mdlovofit_all_domain_5397_locked.tsv")
    rows = _checkpoint_rows(manifest, ALL_DOMAIN_CHECKPOINTS)
    write_tsv(M_ROOT / "mdlovofit_all_domain_corrected.tsv", rows)
    failures = [row for row in rows if row["status"] != "success"]
    if failures:
        write_tsv(M_ROOT / "mdlovofit_all_domain_failures_timeouts.tsv", failures)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "success": sum(row["status"] == "success" for row in rows),
                "failed": sum(row["status"] == "failed" for row in rows),
                "timeout": sum(row["status"] == "timeout" for row in rows),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.mode == "pilot":
        run_individual_panel(
            load_manifest(MANIFEST_ROOT / "mdlovofit_pilot_2000_locked.tsv"),
            PILOT_CHECKPOINTS,
            limit=args.limit,
            rerun=args.rerun,
            label="mdlovofit_pilot",
        )
    elif args.mode == "finalize-pilot":
        finalize_pilot()
    elif args.mode == "all-domain":
        run_individual_panel(
            load_manifest(MANIFEST_ROOT / "mdlovofit_all_domain_5397_locked.tsv"),
            ALL_DOMAIN_CHECKPOINTS,
            limit=args.limit,
            rerun=args.rerun,
            label="mdlovofit_all_domain",
        )
    elif args.mode == "finalize-all-domain":
        finalize_all_domain()
    elif args.mode == "deployment":
        run_deployment(args.limit, args.rerun)
    else:
        finalize_deployment()


if __name__ == "__main__":
    main()
