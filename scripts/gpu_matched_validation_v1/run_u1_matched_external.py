#!/usr/bin/env python3
"""Run matched native-interface contracts against MDLovoFit and THESEUS."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import PersistentGPUWorker
from mdsuperpose.alignment import solve_rigid_alignment
from mdsuperpose.flexibility import compute_rmsf
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_with_occupancy,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.theseus import (
    read_multimodel_pdb_coords,
    run_theseus_on_files,
)
from scripts.gpu_matched_validation_v1.common import (
    RESULTS,
    load_coords,
    load_panel,
    read_tsv,
    timing_summary,
    write_tsv,
)


OUTPUT = RESULTS / "u1_matched_contracts"
RAW = OUTPUT / "matched_contract_raw_repeats.tsv"
MDLOVOFIT = Path("/usr/local/bin/mdlovofit")
THESEUS = Path(
    "/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus"
)
WARMUPS = 1
REPEATS = 5


def _unified_metrics(
    aligned: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    rmsd = np.sqrt(
        np.mean(
            np.sum(
                (aligned - reference[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
    )
    return float(np.mean(rmsd)), float(np.mean(compute_rmsf(aligned)))


def _append_rows(rows: list[dict[str, object]]) -> None:
    existing: list[dict[str, object]] = []
    if RAW.exists():
        existing = read_tsv(RAW)
    write_tsv(RAW, [*existing, *rows])


def _complete(
    rows: list[dict[str, str]],
    system: str,
    prefix_label: str,
    contract: str,
    method: str,
) -> bool:
    matching = [
        row
        for row in rows
        if row["system"] == system
        and row["prefix_label"] == prefix_label
        and row["contract"] == contract
        and row["method"] == method
        and row["status"] == "success"
        and row["is_warmup"].lower() == "false"
    ]
    return len(matching) >= REPEATS


def _internal_analysis(
    *,
    system: str,
    coords: np.ndarray,
    chains: np.ndarray,
    workdir: Path,
    existing: list[dict[str, str]],
) -> None:
    prefix_label = "p0250"
    values = coords[: min(250, coords.shape[0])]
    methods = (
        (
            "CPU_exact_sequential_lean",
            lambda: solve_exact_contract(
                values,
                chain_ids=chains,
                representative_count=min(2000, values.shape[0]),
                chunk_size=500,
                store_aligned=False,
            ),
        ),
        (
            "CPU_throughput",
            lambda: solve_fast_global_alignment(
                values,
                chain_ids=chains,
                representative_count=min(256, values.shape[0]),
                update_rounds=4,
                chunk_size=500,
                store_aligned=False,
                compute_confidence=False,
            ),
        ),
    )
    for method, function in methods:
        if _complete(
            existing,
            system,
            prefix_label,
            "A_analysis_metrics_transforms_rmsd_rmsf",
            method,
        ):
            continue
        rows = []
        for index in range(WARMUPS + REPEATS):
            start = time.perf_counter()
            result = function()
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label,
                    "n_frames": values.shape[0],
                    "n_residues": values.shape[1],
                    "contract": "A_analysis_metrics_transforms_rmsd_rmsf",
                    "method": method,
                    "repeat_index": index - WARMUPS,
                    "is_warmup": index < WARMUPS,
                    "status": "success",
                    "total_seconds": elapsed,
                    "algorithm_or_subprocess_seconds": elapsed,
                    "transfer_seconds": 0.0,
                    "output_parse_seconds": 0.0,
                    "posthoc_metrics_seconds": 0.0,
                    "write_seconds": 0.0,
                    "unified_posthoc_rmsd_A": result.mean_rmsd,
                    "mean_robust_rmsf_A": float(
                        np.mean(result.robust_rmsf)
                    ),
                    "native_all_atom_rmsd": "",
                    "native_core_rmsd": "",
                    "native_selected_fraction": "",
                    "notes": "Warm in-memory CA input; exact output contract.",
                }
            )
        _append_rows(rows)
        existing.extend({key: str(value) for key, value in row.items()} for row in rows)

    method = "GPU_throughput_float32_Horn"
    if not _complete(
        existing,
        system,
        prefix_label,
        "A_analysis_metrics_transforms_rmsd_rmsf",
        method,
    ):
        worker = PersistentGPUWorker.create(
            chain_ids=chains,
            representative_count=min(256, values.shape[0]),
            update_rounds=4,
            chunk_size=500,
            precision="float32",
            rotation_solver="horn",
        )
        rows = []
        for index in range(WARMUPS + REPEATS):
            result = worker.throughput(values)
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label,
                    "n_frames": values.shape[0],
                    "n_residues": values.shape[1],
                    "contract": "A_analysis_metrics_transforms_rmsd_rmsf",
                    "method": method,
                    "repeat_index": index - WARMUPS,
                    "is_warmup": index < WARMUPS,
                    "status": "success",
                    "total_seconds": result.total_runtime_seconds,
                    "algorithm_or_subprocess_seconds": (
                        result.total_runtime_seconds
                    ),
                    "transfer_seconds": result.timing.get(
                        "CPU_GPU_transfer_total",
                        0.0,
                    ),
                    "output_parse_seconds": 0.0,
                    "posthoc_metrics_seconds": 0.0,
                    "write_seconds": 0.0,
                    "unified_posthoc_rmsd_A": result.mean_rmsd,
                    "mean_robust_rmsf_A": float(
                        np.mean(result.robust_rmsf)
                    ),
                    "native_all_atom_rmsd": "",
                    "native_core_rmsd": "",
                    "native_selected_fraction": "",
                    "notes": (
                        "Warm in-memory CA input; synchronized GPU and "
                        "post-task buffer sanitation included."
                    ),
                }
            )
        _append_rows(rows)
        existing.extend({key: str(value) for key, value in row.items()} for row in rows)


def _internal_materialized(
    *,
    system: str,
    coords: np.ndarray,
    chains: np.ndarray,
    workdir: Path,
    existing: list[dict[str, str]],
) -> None:
    values = coords[: min(250, coords.shape[0])]
    prefix_label = "p0250"
    methods = (
        (
            "CPU_exact_sequential_lean",
            lambda: solve_exact_contract(
                values,
                chain_ids=chains,
                representative_count=min(2000, values.shape[0]),
                chunk_size=500,
                store_aligned=True,
            ),
        ),
        (
            "CPU_throughput",
            lambda: solve_fast_global_alignment(
                values,
                chain_ids=chains,
                representative_count=min(256, values.shape[0]),
                update_rounds=4,
                chunk_size=500,
                store_aligned=True,
                compute_confidence=False,
            ),
        ),
    )
    for method, function in methods:
        if _complete(
            existing,
            system,
            prefix_label,
            "B_materialised_aligned_PDB",
            method,
        ):
            continue
        rows = []
        for index in range(WARMUPS + REPEATS):
            total_start = time.perf_counter()
            result = function()
            write_start = time.perf_counter()
            write_multimodel_pdb(
                result.aligned_coords,
                workdir / f"{method}_aligned.pdb",
            )
            write_seconds = time.perf_counter() - write_start
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label,
                    "n_frames": values.shape[0],
                    "n_residues": values.shape[1],
                    "contract": "B_materialised_aligned_PDB",
                    "method": method,
                    "repeat_index": index - WARMUPS,
                    "is_warmup": index < WARMUPS,
                    "status": "success",
                    "total_seconds": time.perf_counter() - total_start,
                    "algorithm_or_subprocess_seconds": (
                        result.total_runtime_seconds
                    ),
                    "transfer_seconds": 0.0,
                    "output_parse_seconds": 0.0,
                    "posthoc_metrics_seconds": 0.0,
                    "write_seconds": write_seconds,
                    "unified_posthoc_rmsd_A": result.mean_rmsd,
                    "mean_robust_rmsf_A": float(
                        np.mean(result.robust_rmsf)
                    ),
                    "native_all_atom_rmsd": "",
                    "native_core_rmsd": "",
                    "native_selected_fraction": "",
                    "notes": (
                        "Warm in-memory CA input; full aligned CA PDB "
                        "materialisation included."
                    ),
                }
            )
        _append_rows(rows)
        existing.extend({key: str(value) for key, value in row.items()} for row in rows)

    method = "GPU_throughput_float32_Horn"
    if not _complete(
        existing,
        system,
        prefix_label,
        "B_materialised_aligned_PDB",
        method,
    ):
        worker = PersistentGPUWorker.create(
            chain_ids=chains,
            representative_count=min(256, values.shape[0]),
            update_rounds=4,
            chunk_size=500,
            precision="float32",
            rotation_solver="horn",
            store_aligned=True,
        )
        rows = []
        for index in range(WARMUPS + REPEATS):
            total_start = time.perf_counter()
            result = worker.throughput(values)
            write_start = time.perf_counter()
            write_multimodel_pdb(
                result.aligned_coords,
                workdir / f"{method}_aligned.pdb",
            )
            write_seconds = time.perf_counter() - write_start
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label,
                    "n_frames": values.shape[0],
                    "n_residues": values.shape[1],
                    "contract": "B_materialised_aligned_PDB",
                    "method": method,
                    "repeat_index": index - WARMUPS,
                    "is_warmup": index < WARMUPS,
                    "status": "success",
                    "total_seconds": time.perf_counter() - total_start,
                    "algorithm_or_subprocess_seconds": (
                        result.total_runtime_seconds
                    ),
                    "transfer_seconds": result.timing.get(
                        "CPU_GPU_transfer_total",
                        0.0,
                    ),
                    "output_parse_seconds": 0.0,
                    "posthoc_metrics_seconds": 0.0,
                    "write_seconds": write_seconds,
                    "unified_posthoc_rmsd_A": result.mean_rmsd,
                    "mean_robust_rmsf_A": float(
                        np.mean(result.robust_rmsf)
                    ),
                    "native_all_atom_rmsd": "",
                    "native_core_rmsd": "",
                    "native_selected_fraction": "",
                    "notes": (
                        "Synchronized GPU, host materialisation and aligned "
                        "CA PDB writing included."
                    ),
                }
            )
        _append_rows(rows)


def _run_mdlovofit(
    *,
    system: str,
    prefix_label: str,
    coords: np.ndarray,
    input_path: Path,
    workdir: Path,
    existing: list[dict[str, str]],
) -> None:
    method = "MDLovoFit_20.1.1"
    contract = "A_analysis_metrics_transforms_rmsd_rmsf"
    if _complete(existing, system, prefix_label, contract, method):
        return
    rows: list[dict[str, object]] = []
    raw_dir = OUTPUT / "raw_external_logs" / system / prefix_label
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index in range(WARMUPS + REPEATS):
        output = workdir / f"mdlovofit_{prefix_label}_{index}.pdb"
        try:
            run = run_mdlovofit_on_files(
                [input_path],
                fraction=0.6,
                output_path=output,
                executable=MDLOVOFIT,
                timeout_seconds=1800,
            )
        except Exception as exc:
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "contract": contract,
                    "method": method,
                    "repeat_index": index - WARMUPS,
                    "is_warmup": index < WARMUPS,
                    "status": "failure",
                    "total_seconds": "",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        read_start = time.perf_counter()
        aligned, occupancies = read_multimodel_pdb_with_occupancy(
            output,
            coords.shape[1],
        )
        read_seconds = time.perf_counter() - read_start
        metrics_start = time.perf_counter()
        unified, mean_rmsf = _unified_metrics(aligned, coords[0])
        metrics_seconds = time.perf_counter() - metrics_start
        parsed = run.parsed_output
        common = {
            "system": system,
            "prefix_label": prefix_label,
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "method": method,
            "repeat_index": index - WARMUPS,
            "is_warmup": index < WARMUPS,
            "status": "success",
            "algorithm_or_subprocess_seconds": (
                run.subprocess_runtime_seconds
            ),
            "transfer_seconds": 0.0,
            "output_parse_seconds": read_seconds,
            "posthoc_metrics_seconds": metrics_seconds,
            "unified_posthoc_rmsd_A": unified,
            "mean_robust_rmsf_A": mean_rmsf,
            "native_all_atom_rmsd": parsed.average_rmsd_all_atoms,
            "native_core_rmsd": parsed.average_rmsd_least_mobile,
            "native_selected_fraction": (
                parsed.fraction_of_best_aligned_atoms
            ),
            "occupancy_binary": bool(
                np.all(
                    np.isclose(occupancies, 0.0)
                    | np.isclose(occupancies, 1.0)
                )
            ),
        }
        rows.append(
            {
                **common,
                "contract": contract,
                "total_seconds": (
                    run.subprocess_runtime_seconds
                    + read_seconds
                    + metrics_seconds
                ),
                "write_seconds": (
                    "included_in_external_subprocess_not_separable"
                ),
                "notes": (
                    "Cached input PDB preparation excluded. External "
                    "subprocess, unavoidable aligned-PDB output, parse and "
                    "post-hoc RMSD/RMSF included."
                ),
            }
        )
        if prefix_label == "p0250":
            rows.append(
                {
                    **common,
                    "contract": "B_materialised_aligned_PDB",
                    "total_seconds": run.subprocess_runtime_seconds,
                    "output_parse_seconds": 0.0,
                    "posthoc_metrics_seconds": 0.0,
                    "write_seconds": (
                        "included_in_external_subprocess_not_separable"
                    ),
                    "notes": (
                        "Cached input PDB preparation excluded; subprocess "
                        "and its unavoidable aligned-PDB writing included."
                    ),
                }
            )
        if index == WARMUPS:
            (raw_dir / "mdlovofit_stdout.txt").write_text(run.stdout)
            (raw_dir / "mdlovofit_stderr.txt").write_text(run.stderr)
            (raw_dir / "mdlovofit_command.json").write_text(
                json.dumps(list(run.command), indent=2) + "\n"
            )
        output.unlink(missing_ok=True)
    _append_rows(rows)


def _run_theseus(
    *,
    system: str,
    coords: np.ndarray,
    input_path: Path,
    workdir: Path,
    existing: list[dict[str, str]],
) -> None:
    prefix_label = "p0250"
    method = "THESEUS_3.3.0"
    contract = "A_analysis_metrics_transforms_rmsd_rmsf"
    if _complete(existing, system, prefix_label, contract, method):
        return
    rows: list[dict[str, object]] = []
    raw_dir = OUTPUT / "raw_external_logs" / system / prefix_label
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index in range(WARMUPS + REPEATS):
        run_dir = workdir / f"theseus_{index}"
        try:
            run = run_theseus_on_files(
                [input_path],
                output_root="theseus",
                output_directory=run_dir,
                executable=THESEUS,
                timeout_seconds=600,
            )
        except Exception as exc:
            rows.append(
                {
                    "system": system,
                    "prefix_label": prefix_label,
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "contract": contract,
                    "method": method,
                    "repeat_index": index - WARMUPS,
                    "is_warmup": index < WARMUPS,
                    "status": "failure",
                    "total_seconds": "",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        read_start = time.perf_counter()
        aligned = read_multimodel_pdb_coords(
            run.aligned_pdb_path,
            coords.shape[1],
        )
        read_seconds = time.perf_counter() - read_start
        metrics_start = time.perf_counter()
        consensus = np.mean(aligned, axis=0)
        orientation = solve_rigid_alignment(consensus, coords[0])
        oriented = aligned @ orientation.rotation.T + orientation.translation
        unified, mean_rmsf = _unified_metrics(oriented, coords[0])
        metrics_seconds = time.perf_counter() - metrics_start
        parsed = run.parsed_output
        common = {
            "system": system,
            "prefix_label": prefix_label,
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "method": method,
            "repeat_index": index - WARMUPS,
            "is_warmup": index < WARMUPS,
            "status": "success",
            "algorithm_or_subprocess_seconds": (
                run.subprocess_runtime_seconds
            ),
            "transfer_seconds": 0.0,
            "unified_posthoc_rmsd_A": unified,
            "mean_robust_rmsf_A": mean_rmsf,
            "native_all_atom_rmsd": parsed.maximum_likelihood_rmsd,
            "native_core_rmsd": "",
            "native_selected_fraction": run.output_core_atom_fraction,
        }
        rows.append(
            {
                **common,
                "contract": contract,
                "total_seconds": (
                    run.subprocess_runtime_seconds
                    + read_seconds
                    + metrics_seconds
                ),
                "output_parse_seconds": read_seconds,
                "posthoc_metrics_seconds": metrics_seconds,
                "write_seconds": (
                    "included_in_external_subprocess_not_separable"
                ),
                "notes": (
                    "Cached input PDB preparation excluded. One common "
                    "consensus-to-reference orientation was used for unified "
                    "post-hoc RMSD."
                ),
            }
        )
        rows.append(
            {
                **common,
                "contract": "B_materialised_aligned_PDB",
                "total_seconds": run.subprocess_runtime_seconds,
                "output_parse_seconds": 0.0,
                "posthoc_metrics_seconds": 0.0,
                "write_seconds": (
                    "included_in_external_subprocess_not_separable"
                ),
                "notes": (
                    "Cached input PDB preparation excluded; subprocess and "
                    "native output writing included."
                ),
            }
        )
        if index == WARMUPS:
            (raw_dir / "theseus_stdout.txt").write_text(run.stdout)
            (raw_dir / "theseus_stderr.txt").write_text(run.stderr)
            (raw_dir / "theseus_command.json").write_text(
                json.dumps(list(run.command), indent=2) + "\n"
            )
    _append_rows(rows)


def _summarize() -> None:
    rows = [
        row
        for row in read_tsv(RAW)
        if row["is_warmup"].lower() == "false"
        and row["status"] == "success"
        and not (
            row["contract"] == "B_materialised_aligned_PDB"
            and row["prefix_label"] != "p0250"
        )
    ]
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["system"],
            row["prefix_label"],
            row["contract"],
            row["method"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for (system, prefix, contract, method), group in sorted(groups.items()):
        times = [float(row["total_seconds"]) for row in group]
        summaries.append(
            {
                "system": system,
                "prefix_label": prefix,
                "n_frames": group[0]["n_frames"],
                "n_residues": group[0]["n_residues"],
                "contract": contract,
                "method": method,
                "status": "success",
                "warmups": WARMUPS,
                "measured_repeats": len(group),
                **timing_summary(times),
                "median_unified_posthoc_rmsd_A": float(
                    np.median(
                        [
                            float(row["unified_posthoc_rmsd_A"])
                            for row in group
                        ]
                    )
                ),
                "median_native_all_atom_rmsd": (
                    float(
                        np.median(
                            [
                                float(row["native_all_atom_rmsd"])
                                for row in group
                                if row["native_all_atom_rmsd"]
                                not in {"", "None"}
                            ]
                        )
                    )
                    if any(
                        row["native_all_atom_rmsd"] not in {"", "None"}
                        for row in group
                    )
                    else ""
                ),
                "median_native_core_rmsd": (
                    float(
                        np.median(
                            [
                                float(row["native_core_rmsd"])
                                for row in group
                                if row["native_core_rmsd"]
                                not in {"", "None"}
                            ]
                        )
                    )
                    if any(
                        row["native_core_rmsd"] not in {"", "None"}
                        for row in group
                    )
                    else ""
                ),
                "timing_boundary": group[0]["notes"],
            }
        )
    internal_path = (
        RESULTS
        / "u2_real_panel"
        / "real_panel_internal_matched_timings.tsv"
    )
    if internal_path.exists():
        for row in read_tsv(internal_path):
            if row["contract"] not in {
                "A_analysis_metrics_transforms_rmsd_rmsf",
                "W5_local_deformation_not_alignment_equivalent",
            }:
                continue
            cached_rmsd = ""
            cache_path = (
                RESULTS / "u5_quality" / "result_cache" / f"{row['system']}.npz"
            )
            if cache_path.exists() and row["method"] in {
                "CPU_exact_sequential_lean",
                "CPU_throughput",
                "GPU_throughput_float32_Horn",
            }:
                key = {
                    "CPU_exact_sequential_lean": "exact_rmsd",
                    "CPU_throughput": "cpu_rmsd",
                    "GPU_throughput_float32_Horn": "gpu_rmsd",
                }[row["method"]]
                with np.load(cache_path, allow_pickle=False) as archive:
                    cached_rmsd = float(np.mean(archive[key]))
            summaries.append(
                {
                    "system": row["system"],
                    "prefix_label": "full",
                    "n_frames": row["n_frames"],
                    "n_residues": row["n_residues"],
                    "contract": row["contract"],
                    "method": row["method"],
                    "status": "success",
                    "warmups": row["warmups"],
                    "measured_repeats": row["measured_repeats"],
                    "median_seconds": row["median_seconds"],
                    "q1_seconds": row["q1_seconds"],
                    "q3_seconds": row["q3_seconds"],
                    "iqr_seconds": row["iqr_seconds"],
                    "min_seconds": row["min_seconds"],
                    "max_seconds": row["max_seconds"],
                    "mean_seconds": row["mean_seconds"],
                    "sd_seconds": row["sd_seconds"],
                    "median_unified_posthoc_rmsd_A": cached_rmsd,
                    "median_native_all_atom_rmsd": "",
                    "median_native_core_rmsd": "",
                    "timing_boundary": (
                        "Warm in-memory CA coordinates; decode excluded. "
                        "W5 rows are a non-equivalent local-deformation contract."
                    ),
                }
            )
    write_tsv(OUTPUT / "matched_contract_summary.tsv", summaries)

    native = [
        {
            "system": row["system"],
            "prefix_label": row["prefix_label"],
            "method": row["method"],
            "unified_posthoc_rmsd_A": row[
                "median_unified_posthoc_rmsd_A"
            ],
            "native_all_atom_rmsd": row["median_native_all_atom_rmsd"],
            "native_core_rmsd": row["median_native_core_rmsd"],
            "semantics": (
                "Native fields are tool-reported; unified post-hoc RMSD is "
                "recomputed from aligned coordinates under one evaluator."
            ),
        }
        for row in summaries
        if row["contract"] == "A_analysis_metrics_transforms_rmsd_rmsf"
        and row["method"] in {"MDLovoFit_20.1.1", "THESEUS_3.3.0"}
    ]
    write_tsv(OUTPUT / "external_native_metrics.tsv", native)
    panel = load_panel()
    system_families = {
        row["system"].split("__", 1)[0] for row in panel
    }
    (OUTPUT / "u1_matched_contract_report.md").write_text(
        "\n".join(
            [
                "# U1 matched external-tool comparison",
                "",
                f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
                f"- Warmups per measured setting: `{WARMUPS}`",
                f"- Measured repeats per completed setting: `{REPEATS}`",
                f"- Full-length analysis comparisons include `{len(panel)}` "
                "real, unstitched trajectory entries representing "
                f"`{len(system_families)}` system families for MDLovoFit.",
                "- THESEUS is evaluated on the matched 250-frame prefix because "
                "larger ensemble runs are not uniformly feasible.",
                "- Materialised-output comparisons use the same 250-frame CA "
                "prefix for all completed methods.",
                "- Cached input preparation is excluded. External subprocess "
                "and unavoidable native output writing are included.",
                "- W5 is reported separately as a local-deformation contract "
                "and is not treated as an MDLovoFit-equivalent alignment.",
                "- No 71x or universal-fastest claim is inferred.",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        nargs="*",
        help="Optional system names for an incremental/resume run.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = set(args.systems or ())
    existing = read_tsv(RAW) if RAW.exists() else []

    for row in load_panel():
        system = row["system"]
        if selected and system not in selected:
            continue
        coords, chains = load_coords(row)
        print(f"[U1] {system} {coords.shape}", flush=True)
        with tempfile.TemporaryDirectory(prefix="mdca_u1_") as temporary:
            workdir = Path(temporary)
            prefix_coords = coords[: min(250, coords.shape[0])]
            prefix_input = workdir / "input_p0250.pdb"
            write_multimodel_pdb(prefix_coords, prefix_input)
            _internal_analysis(
                system=system,
                coords=coords,
                chains=chains,
                workdir=workdir,
                existing=existing,
            )
            _internal_materialized(
                system=system,
                coords=coords,
                chains=chains,
                workdir=workdir,
                existing=existing,
            )
            _run_mdlovofit(
                system=system,
                prefix_label="p0250",
                coords=prefix_coords,
                input_path=prefix_input,
                workdir=workdir,
                existing=existing,
            )
            _run_theseus(
                system=system,
                coords=prefix_coords,
                input_path=prefix_input,
                workdir=workdir,
                existing=existing,
            )
            full_input = workdir / "input_full.pdb"
            if coords.shape[0] == prefix_coords.shape[0]:
                full_input = prefix_input
            else:
                write_multimodel_pdb(coords, full_input)
            _run_mdlovofit(
                system=system,
                prefix_label="full",
                coords=coords,
                input_path=full_input,
                workdir=workdir,
                existing=existing,
            )
        existing = read_tsv(RAW)
    _summarize()


if __name__ == "__main__":
    main()
