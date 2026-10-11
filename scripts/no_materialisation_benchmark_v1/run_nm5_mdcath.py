#!/usr/bin/env python3
"""Run the corrected 5,397-domain mdCATH no-materialisation benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

from mdcorealign.benchmark_minimal import solve_minimal_from_pdb
from mdsuperpose.pdb_writer import write_multimodel_ca_pdb_streaming
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import extract_ca_trajectory
from scripts.no_materialisation_benchmark_v1.common import (
    OUT,
    ROOT,
    parse_timing,
    read_tsv,
    run_external,
    stock_command,
    timed_command,
    write_json,
    write_text,
    write_tsv,
)
from scripts.no_materialisation_benchmark_v1.run_nm4_atlas import (
    METHODS,
    _external_row,
    _mdcore_row,
)


DEST = OUT / "03_mdcath_stock_no_output"
CHECKPOINTS = DEST / "checkpoints"
SOURCE = ROOT / "results/database_scale_benchmark_v2/mdlovofit_comparison/mdlovofit_all_domain_corrected_matched.tsv"
SOURCE_CHECKPOINTS = ROOT / "results/database_scale_benchmark_v2/mdlovofit_comparison/all_domain_checkpoints"
TIMEOUT_SECONDS = 120.0


def _frame_bin(value: int) -> str:
    if value < 450:
        return "lt450"
    if value < 500:
        return "450_499"
    return "ge500"


def _lock_subset(panel: list[dict[str, str]], size: int = 200) -> list[dict[str, object]]:
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in panel:
        strata[(row["cath_class"], row["residue_count_bin"], _frame_bin(int(row["n_frames"])))].append(row)
    ordered_strata = sorted(strata)
    for values in strata.values():
        values.sort(key=lambda row: row["domain_id"])
    selected: list[dict[str, str]] = []
    cursor = {key: 0 for key in ordered_strata}
    while len(selected) < size:
        progressed = False
        for key in ordered_strata:
            index = cursor[key]
            values = strata[key]
            if index < len(values):
                selected.append(values[index])
                cursor[key] += 1
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise RuntimeError("panel is smaller than requested locked subset")
    return [
        {
            "selection_index": index,
            "domain_id": row["domain_id"],
            "trajectory_id": row["trajectory_id"],
            "cath_class": row["cath_class"],
            "n_frames": row["n_frames"],
            "n_residues": row["n_residues"],
            "residue_count_bin": row["residue_count_bin"],
            "frame_count_bin": _frame_bin(int(row["n_frames"])),
            "selection_rule": "deterministic_round_robin_over_class_residue_frame_strata_before_NM5_timing",
        }
        for index, row in enumerate(selected)
    ]


def _prepare_input(row: dict[str, str], destination: Path) -> tuple[float, float, int, int, str, int]:
    task_path = SOURCE_CHECKPOINTS / f"{row['trajectory_id']}.json"
    task = json.loads(task_path.read_text())
    decode_start = time.perf_counter()
    coords, _, _, metadata = extract_ca_trajectory(
        ROOT / task["source_file"],
        coordinate_path=task["coordinate_path"],
        direct_contiguous_read=True,
    )
    decode_seconds = time.perf_counter() - decode_start
    values = np.asarray(coords, dtype=np.float64)
    preparation_start = time.perf_counter()
    values -= np.mean(values, axis=1, keepdims=True)
    write_multimodel_ca_pdb_streaming(values, destination)
    preparation_seconds = time.perf_counter() - preparation_start
    return (
        decode_seconds,
        preparation_seconds,
        values.shape[0],
        values.shape[1],
        str(metadata["unit_status"]),
        destination.stat().st_size,
    )


def _item(row: dict[str, str]) -> dict[str, str]:
    return {
        "system": row["domain_id"],
        "prefix_label": row["trajectory_id"],
        "n_frames": row["n_frames"],
        "n_residues": row["n_residues"],
        "centered_input_path": "temporary_pre_generated_centered_CA_PDB_not_retained",
        "centered_input_sha256": "verified_by_native_identity_and_historical_byte_count",
    }


def _run_method(
    method: str,
    row: dict[str, str],
    input_path: Path,
    repeat_index: int,
    work: Path,
) -> dict[str, object]:
    item = _item(row)
    if method == "stock_MDLovoFit_no_output":
        result = run_external(stock_command(input_path), cwd=work, timeout=TIMEOUT_SECONDS)
        return _external_row(method, result, item, repeat_index)
    if method == "timed_MDLovoFit_no_output":
        result = run_external(timed_command(input_path), cwd=work, timeout=TIMEOUT_SECONDS)
        return _external_row(method, result, item, repeat_index)
    if method == "MDCoreAlign_exact_CPU_minimal":
        result = solve_minimal_from_pdb(
            input_path, mode="exact", backend="cpu",
            expected_frames=int(row["n_frames"]), expected_residues=int(row["n_residues"]),
        )
    elif method == "MDCoreAlign_throughput_CPU_minimal":
        result = solve_minimal_from_pdb(
            input_path, mode="throughput", backend="cpu",
            expected_frames=int(row["n_frames"]), expected_residues=int(row["n_residues"]),
        )
    elif method == "MDCoreAlign_throughput_GPU_minimal":
        result = solve_minimal_from_pdb(
            input_path, mode="throughput", backend="gpu",
            expected_frames=int(row["n_frames"]), expected_residues=int(row["n_residues"]),
        )
    else:
        raise ValueError(method)
    return _mdcore_row(method, result, item, repeat_index)


def _checkpoint(row: dict[str, str], root: Path) -> Path:
    return root / f"{row['trajectory_id']}.tsv"


def _method_complete(rows: list[dict[str, str]], method: str, repeats: int) -> bool:
    selected = [row for row in rows if row["method"] == method]
    return len(selected) == repeats and {int(row["repeat_index"]) for row in selected} == set(range(repeats))


def _bootstrap_paired_median(values: np.ndarray, *, seed: int = 20260817, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    medians = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        medians[index] = np.median(values[rng.integers(0, values.size, values.size)])
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def _summaries(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    repeat0 = [row for row in rows if row["repeat_index"] == "0" and row["status"] == "success"]
    by_key = {(row["system"], row["method"]): row for row in repeat0}
    comparisons: list[dict[str, object]] = []
    for method in METHODS[2:]:
        c1_ratios: list[float] = []
        c2_ratios: list[float] = []
        for domain in sorted({row["system"] for row in repeat0}):
            stock = by_key.get((domain, METHODS[0]))
            timed = by_key.get((domain, METHODS[1]))
            candidate = by_key.get((domain, method))
            if not stock or not timed or not candidate:
                continue
            c1_ratios.append(float(stock["C1_PDB_tool_wall_seconds"]) / float(candidate["C1_PDB_tool_wall_seconds"]))
            c2_ratios.append(float(timed["C2_core_compute_seconds"]) / float(candidate["C2_core_compute_seconds"]))
        for contract, ratios in (("C1_same_PDB_tool_workflow", c1_ratios), ("C2_loaded_core_compute", c2_ratios)):
            values = np.asarray(ratios, dtype=np.float64)
            lower, upper = _bootstrap_paired_median(values)
            comparisons.append(
                {
                    "comparison": f"MDLovoFit_over_{method}",
                    "contract": contract,
                    "paired_domains": values.size,
                    "paired_median_speedup": float(np.median(values)),
                    "paired_median_bootstrap_95CI_lower": lower,
                    "paired_median_bootstrap_95CI_upper": upper,
                    "speedup_IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "MDCoreAlign_faster_fraction": float(np.mean(values > 1.0)),
                    "minimum_speedup": float(np.min(values)),
                    "maximum_speedup": float(np.max(values)),
                }
            )
    stages: list[dict[str, object]] = []
    for method in METHODS:
        group = [row for row in repeat0 if row["method"] == method]
        for field in (
            "PDB_parsing_seconds", "weight_or_LOVO_search_seconds",
            "all_frame_final_fit_seconds", "RMSD_reduction_seconds",
            "stdout_formatting_seconds", "coordinate_preparation_seconds",
        ):
            values = np.asarray([float(row[field]) for row in group if row.get(field, "")], dtype=np.float64)
            if values.size:
                stages.append(
                    {
                        "method": method,
                        "stage": field.removesuffix("_seconds"),
                        "domains": values.size,
                        "median_seconds": float(np.median(values)),
                        "IQR_seconds": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                        "sum_seconds": float(np.sum(values)),
                    }
                )
    return comparisons, stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    panel = read_tsv(SOURCE)
    if len(panel) != 5397:
        raise RuntimeError(f"expected 5,397 corrected domains, found {len(panel)}")
    locked_path = DEST / "locked_200_subset_manifest.tsv"
    if not locked_path.exists():
        write_tsv(locked_path, _lock_subset(panel))
    locked = {row["trajectory_id"] for row in read_tsv(locked_path)}
    if len(locked) != 200:
        raise RuntimeError("locked repeated subset is not exactly 200 domains")
    if args.limit:
        panel = panel[: args.limit]
    checkpoint_root = OUT / "07_validation/nm5_smoke" if args.smoke else CHECKPOINTS
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nm5-mdcath-") as temporary:
        work = Path(temporary)
        warmup_path = DEST / "backend_warmup.tsv"
        if not args.smoke and not warmup_path.exists():
            warmup_source = panel[0]
            warmup_input = work / "registered_backend_warmup_centered_CA.pdb"
            decode, prepare, frames, residues, unit, input_bytes = _prepare_input(
                warmup_source, warmup_input
            )
            warmup_rows = []
            for method in METHODS:
                warmup = _run_method(method, warmup_source, warmup_input, -1, work)
                warmup.update(
                    {
                        "shared_HDF5_decode_seconds_outside_timing": decode,
                        "PDB_preparation_seconds_outside_timing": prepare,
                        "temporary_input_PDB_bytes": input_bytes,
                        "unit_status": unit,
                    }
                )
                warmup_rows.append(warmup)
            write_tsv(warmup_path, warmup_rows)
            warmup_input.unlink(missing_ok=True)
        for domain_index, source_row in enumerate(panel):
            checkpoint = _checkpoint(source_row, checkpoint_root)
            current = read_tsv(checkpoint) if checkpoint.exists() else []
            repeats = 5 if source_row["trajectory_id"] in locked and not args.smoke else 1
            if all(_method_complete(current, method, repeats) for method in METHODS):
                continue
            input_path = work / f"{source_row['trajectory_id']}_centered_CA.pdb"
            decode, prepare, frames, residues, unit, input_bytes = _prepare_input(source_row, input_path)
            if frames != int(source_row["n_frames"]) or residues != int(source_row["n_residues"]):
                raise RuntimeError(f"shape mismatch for {source_row['trajectory_id']}")
            historical_bytes = int(json.loads((SOURCE_CHECKPOINTS / f"{source_row['trajectory_id']}.json").read_text())["input_PDB_bytes"])
            input_integrity = input_bytes == historical_bytes and "angstrom" in unit
            order = METHODS[domain_index % len(METHODS):] + METHODS[: domain_index % len(METHODS)]
            for method in order:
                if _method_complete(current, method, repeats):
                    continue
                current = [row for row in current if row["method"] != method]
                for repeat_index in range(repeats):
                    result = _run_method(method, source_row, input_path, repeat_index, work)
                    result.update(
                        {
                            "cath_class": source_row["cath_class"],
                            "residue_count_bin": source_row["residue_count_bin"],
                            "frame_count_bin": _frame_bin(frames),
                            "shared_HDF5_decode_seconds_outside_timing": decode,
                            "PDB_preparation_seconds_outside_timing": prepare,
                            "temporary_input_PDB_bytes": input_bytes,
                            "historical_input_PDB_bytes": historical_bytes,
                            "input_integrity_pass": input_integrity,
                            "native_least_mobile_delta_vs_frozen_with_t_A": (
                                abs(float(result["native_least_mobile_RMSD_A"]) - float(source_row["mdlovofit_native_least_mobile_RMSD_A"]))
                                if result["native_least_mobile_RMSD_A"] != "" else ""
                            ),
                            "native_all_atom_delta_vs_frozen_with_t_A": (
                                abs(float(result["native_all_atom_RMSD_A"]) - float(source_row["mdlovofit_native_all_atom_RMSD_A"]))
                                if result["native_all_atom_RMSD_A"] != "" else ""
                            ),
                        }
                    )
                    if result["native_least_mobile_RMSD_A"] != "":
                        result["native_identity_pass"] = (
                            float(result["native_least_mobile_delta_vs_frozen_with_t_A"]) <= 5e-5
                            and float(result["native_all_atom_delta_vs_frozen_with_t_A"]) <= 5e-5
                        )
                    else:
                        result["native_identity_pass"] = ""
                    current.append(result)
                    if result["status"] != "success":
                        break
                write_tsv(checkpoint, current)
            input_path.unlink(missing_ok=True)
            print(json.dumps({"completed_domain": source_row["trajectory_id"], "index": domain_index + 1, "total": len(panel)}), flush=True)

    if args.smoke:
        write_json(checkpoint_root / "smoke_status.json", {"status": "pass", "domains": len(panel)})
        return
    rows: list[dict[str, str]] = []
    for source_row in panel:
        rows.extend(read_tsv(_checkpoint(source_row, CHECKPOINTS)))
    write_tsv(DEST / "mdcath_no_output_all_domain.tsv", [row for row in rows if row["repeat_index"] == "0"])
    write_tsv(DEST / "mdcath_no_output_repeated_subset.tsv", [row for row in rows if row["system"] + "__T320__R1" in locked])
    summaries, stages = _summaries(rows)
    write_tsv(DEST / "mdcath_no_output_summary.tsv", summaries)
    write_tsv(DEST / "mdcath_stage_decomposition.tsv", stages)
    failures = [
        row for row in rows
        if row["status"] != "success"
        or row["input_integrity_pass"] != "True"
        or row.get("native_identity_pass", "") == "False"
    ]
    write_tsv(DEST / "failures_timeouts.tsv", failures or [{"status": "none", "count": 0}])
    complete = len(panel) == 5397 and all(
        _method_complete(
            [row for row in rows if row["system"] == source_row["domain_id"]],
            method,
            5 if source_row["trajectory_id"] in locked else 1,
        )
        for source_row in panel for method in METHODS
    )
    write_json(DEST / "nm5_status.json", {"status": "pass" if complete else "fail", "domains": len(panel), "complete": complete, "failure_rows": len(failures)})
    write_text(
        DEST / "mdcath_no_output_report.md",
        f"""# Corrected mdCATH All-domain No-output Benchmark

Status: **{'PASS' if complete else 'FAIL'}** for {len(panel):,} corrected
domains. Every domain used one measured run for stock/timed MDLovoFit and the
three compact MDCoreAlign paths; the prospectively locked 200-domain subset used
five repeats. HDF5 decoding and deterministic centred-PDB construction occurred
once per domain outside all method timers. C1 includes each method parsing that
same PDB. C2 uses instrumented LOVO/search plus fixed fit/RMSD and the compact
MDCoreAlign stable-weight plus explicit all-frame fit/RMSD boundaries.
""",
    )
    if not complete:
        raise RuntimeError("NM5 completeness gate failed")


if __name__ == "__main__":
    main()
