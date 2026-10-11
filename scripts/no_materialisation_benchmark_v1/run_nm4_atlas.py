#!/usr/bin/env python3
"""Run the locked 48-setting ATLAS no-materialisation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

import numpy as np

from mdcorealign.benchmark_minimal import solve_minimal_from_pdb
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


DEST = OUT / "02_atlas_stock_no_output"
CHECKPOINTS = DEST / "checkpoints"
METHODS = (
    "stock_MDLovoFit_no_output",
    "timed_MDLovoFit_no_output",
    "MDCoreAlign_exact_CPU_minimal",
    "MDCoreAlign_throughput_CPU_minimal",
    "MDCoreAlign_throughput_GPU_minimal",
)
EXTERNAL_TIMEOUT_SECONDS = 7200


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _iqr(values: list[float]) -> float:
    q = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    return float(q[1] - q[0])


def _core_indices(labels: np.ndarray) -> str:
    return ",".join(str(value) for value in np.flatnonzero(labels))


def _core_hash(labels: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(labels, dtype=np.uint8).tobytes()).hexdigest()


def _external_row(
    method: str,
    run: dict[str, object],
    item: dict[str, str],
    repeat_index: int,
) -> dict[str, object]:
    timing = parse_timing(str(run.get("stderr", "")))
    c2 = ""
    if timing:
        c2 = sum(
            timing.get(name, 0.0)
            for name in ("LOVO_search", "final_fixed_subset_fit", "RMSD_reduction")
        )
    return {
        "system": item["system"],
        "prefix_label": item["prefix_label"],
        "n_frames": int(item["n_frames"]),
        "n_residues": int(item["n_residues"]),
        "residue_frames": int(item["n_frames"]) * int(item["n_residues"]),
        "method": method,
        "repeat_index": repeat_index,
        "is_warmup": repeat_index < 0,
        "status": run["status"],
        "C1_PDB_tool_wall_seconds": run["wall_seconds"],
        "C2_core_compute_seconds": c2,
        "PDB_parsing_seconds": timing.get("PDB_parsing", ""),
        "weight_or_LOVO_search_seconds": timing.get("LOVO_search", ""),
        "all_frame_final_fit_seconds": timing.get("final_fixed_subset_fit", ""),
        "RMSD_reduction_seconds": timing.get("RMSD_reduction", ""),
        "stdout_formatting_seconds": timing.get("stdout_formatting", ""),
        "coordinate_preparation_seconds": timing.get("coordinate_preparation", ""),
        "native_least_mobile_RMSD_A": run.get("native_least_mobile_RMSD_A", ""),
        "native_most_mobile_RMSD_A": run.get("native_most_mobile_RMSD_A", ""),
        "native_all_atom_RMSD_A": run.get("native_all_atom_RMSD_A", ""),
        "mean_unified_RMSD_A": "",
        "mean_weighted_RMSD_A": "",
        "core_indices": "",
        "core_label_sha256": "",
        "input_PDB_path": item["centered_input_path"],
        "input_PDB_sha256": item["centered_input_sha256"],
        "aligned_coordinates_materialised": False,
        "nonessential_output_written_inside_timing": False,
        "return_code": run["return_code"],
        **{f"timed_{key}_seconds": value for key, value in timing.items()},
    }


def _mdcore_row(
    method: str,
    result: object,
    item: dict[str, str],
    repeat_index: int,
) -> dict[str, object]:
    timing = result.timing
    return {
        "system": item["system"],
        "prefix_label": item["prefix_label"],
        "n_frames": int(item["n_frames"]),
        "n_residues": int(item["n_residues"]),
        "residue_frames": int(item["n_frames"]) * int(item["n_residues"]),
        "method": method,
        "repeat_index": repeat_index,
        "is_warmup": repeat_index < 0,
        "status": "success",
        "C1_PDB_tool_wall_seconds": timing["C1_total_runtime_seconds"],
        "C2_core_compute_seconds": result.total_runtime_seconds,
        "PDB_parsing_seconds": timing["PDB_parsing_seconds"],
        "weight_or_LOVO_search_seconds": timing["weight_estimation_seconds"],
        "all_frame_final_fit_seconds": timing.get(
            "all_frame_fit_seconds", timing.get("all_frame_GPU_compute_seconds", "")
        ),
        "RMSD_reduction_seconds": timing.get("rmsd_reduction_seconds", ""),
        "stdout_formatting_seconds": 0.0,
        "coordinate_preparation_seconds": 0.0,
        "native_least_mobile_RMSD_A": "",
        "native_most_mobile_RMSD_A": "",
        "native_all_atom_RMSD_A": "",
        "mean_unified_RMSD_A": float(np.mean(result.rmsd)),
        "mean_weighted_RMSD_A": float(np.mean(result.weighted_rmsd)),
        "core_indices": _core_indices(result.core_labels),
        "core_label_sha256": _core_hash(result.core_labels),
        "input_PDB_path": item["centered_input_path"],
        "input_PDB_sha256": item["centered_input_sha256"],
        "aligned_coordinates_materialised": False,
        "nonessential_output_written_inside_timing": False,
        "return_code": 0,
        **{f"timed_{key}_seconds": value for key, value in timing.items()},
    }


def _run_method(method: str, item: dict[str, str], repeat_index: int, work: Path) -> dict[str, object]:
    input_path = ROOT / item["centered_input_path"]
    if method == "stock_MDLovoFit_no_output":
        return _external_row(
            method,
            run_external(stock_command(input_path), cwd=work, timeout=EXTERNAL_TIMEOUT_SECONDS),
            item,
            repeat_index,
        )
    if method == "timed_MDLovoFit_no_output":
        return _external_row(
            method,
            run_external(timed_command(input_path), cwd=work, timeout=EXTERNAL_TIMEOUT_SECONDS),
            item,
            repeat_index,
        )
    if method == "MDCoreAlign_exact_CPU_minimal":
        result = solve_minimal_from_pdb(
            input_path,
            mode="exact",
            backend="cpu",
            expected_frames=int(item["n_frames"]),
            expected_residues=int(item["n_residues"]),
        )
    elif method == "MDCoreAlign_throughput_CPU_minimal":
        result = solve_minimal_from_pdb(
            input_path,
            mode="throughput",
            backend="cpu",
            expected_frames=int(item["n_frames"]),
            expected_residues=int(item["n_residues"]),
        )
    elif method == "MDCoreAlign_throughput_GPU_minimal":
        result = solve_minimal_from_pdb(
            input_path,
            mode="throughput",
            backend="gpu",
            expected_frames=int(item["n_frames"]),
            expected_residues=int(item["n_residues"]),
        )
    else:
        raise ValueError(method)
    return _mdcore_row(method, result, item, repeat_index)


def _checkpoint_path(item: dict[str, str], root: Path) -> Path:
    return root / f"{item['system']}__{item['prefix_label']}.tsv"


def _complete(rows: list[dict[str, str]], method: str) -> bool:
    selected = [row for row in rows if row["method"] == method]
    return len(selected) == 6 and {int(row["repeat_index"]) for row in selected} == {-1, 0, 1, 2, 3, 4}


def _summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    stages: list[dict[str, object]] = []
    keys = sorted({(row["system"], row["prefix_label"], row["method"]) for row in rows})
    stage_names = (
        "PDB_parsing_seconds",
        "weight_or_LOVO_search_seconds",
        "all_frame_final_fit_seconds",
        "RMSD_reduction_seconds",
        "stdout_formatting_seconds",
        "coordinate_preparation_seconds",
    )
    for system, prefix, method in keys:
        group = [
            row for row in rows
            if row["system"] == system and row["prefix_label"] == prefix
            and row["method"] == method and row["is_warmup"] == "False"
            and row["status"] == "success"
        ]
        if not group:
            continue
        c1 = [float(row["C1_PDB_tool_wall_seconds"]) for row in group]
        c2 = [float(row["C2_core_compute_seconds"]) for row in group if row["C2_core_compute_seconds"]]
        summaries.append(
            {
                "system": system,
                "prefix_label": prefix,
                "n_frames": int(group[0]["n_frames"]),
                "n_residues": int(group[0]["n_residues"]),
                "residue_frames": int(group[0]["residue_frames"]),
                "method": method,
                "successful_measured_repeats": len(group),
                "C1_median_seconds": _median(c1),
                "C1_IQR_seconds": _iqr(c1),
                "C1_min_seconds": min(c1),
                "C1_max_seconds": max(c1),
                "C2_median_seconds": _median(c2) if c2 else "",
                "C2_IQR_seconds": _iqr(c2) if c2 else "",
                "mean_unified_RMSD_A": group[0]["mean_unified_RMSD_A"],
                "mean_weighted_RMSD_A": group[0]["mean_weighted_RMSD_A"],
                "native_least_mobile_RMSD_A": group[0]["native_least_mobile_RMSD_A"],
                "native_all_atom_RMSD_A": group[0]["native_all_atom_RMSD_A"],
            }
        )
        for stage in stage_names:
            values = [float(row[stage]) for row in group if row.get(stage, "") not in ("", None)]
            if values:
                stages.append(
                    {
                        "system": system,
                        "prefix_label": prefix,
                        "method": method,
                        "stage": stage.removesuffix("_seconds"),
                        "median_seconds": _median(values),
                        "IQR_seconds": _iqr(values),
                    }
                )
    return summaries, stages


def _identity(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    historical = read_tsv(ROOT / "results/atlas_large_long_v1/a3_external/external_raw_repeats.tsv")
    frozen_native: dict[tuple[str, str], dict[str, str]] = {}
    for row in historical:
        if row["method"] == "MDLovoFit_20.1.1" and row["status"] == "success":
            frozen_native.setdefault((row["system"], row["prefix_label"]), row)
    output: list[dict[str, object]] = []
    for key in sorted(frozen_native):
        stock = next(
            row for row in rows
            if (row["system"], row["prefix_label"]) == key
            and row["method"] == "stock_MDLovoFit_no_output"
            and row["repeat_index"] == "0"
        )
        old = frozen_native[key]
        least_delta = abs(float(stock["native_least_mobile_RMSD_A"]) - float(old["native_least_mobile_RMSD_A"]))
        all_delta = abs(float(stock["native_all_atom_RMSD_A"]) - float(old["native_all_atom_RMSD_A"]))
        cpu = next(
            row for row in rows
            if (row["system"], row["prefix_label"]) == key
            and row["method"] == "MDCoreAlign_throughput_CPU_minimal"
            and row["repeat_index"] == "0"
        )
        gpu = next(
            row for row in rows
            if (row["system"], row["prefix_label"]) == key
            and row["method"] == "MDCoreAlign_throughput_GPU_minimal"
            and row["repeat_index"] == "0"
        )
        left = {int(value) for value in cpu["core_indices"].split(",") if value}
        right = {int(value) for value in gpu["core_indices"].split(",") if value}
        jaccard = len(left & right) / len(left | right)
        output.append(
            {
                "system": key[0],
                "prefix_label": key[1],
                "stock_no_output_vs_frozen_with_t_least_mobile_delta_A": least_delta,
                "stock_no_output_vs_frozen_with_t_all_atom_delta_A": all_delta,
                "stock_no_output_native_identity_pass": least_delta <= 5e-5 and all_delta <= 5e-5,
                "CPU_GPU_throughput_core_Jaccard": jaccard,
                "CPU_GPU_mean_unified_RMSD_delta_A": abs(float(cpu["mean_unified_RMSD_A"]) - float(gpu["mean_unified_RMSD_A"])),
                "CPU_GPU_implementation_consistency_pass": jaccard == 1.0,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = [row for row in read_tsv(ROOT / "results/atlas_large_long_v1/a3_external/centered_input_manifest.tsv") if row["status"] == "success"]
    if args.limit:
        manifest = manifest[: args.limit]
    checkpoint_root = (OUT / "07_validation/nm4_smoke") if args.smoke else CHECKPOINTS
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nm4-atlas-") as temporary:
        work = Path(temporary)
        for setting_index, item in enumerate(manifest):
            checkpoint = _checkpoint_path(item, checkpoint_root)
            current = read_tsv(checkpoint) if checkpoint.exists() else []
            order = METHODS[setting_index % len(METHODS):] + METHODS[: setting_index % len(METHODS)]
            for method in order:
                if _complete(current, method):
                    continue
                # Incomplete method rows are replaced as one atomic six-run block.
                current = [row for row in current if row["method"] != method]
                for repeat_index in (-1, 0, 1, 2, 3, 4):
                    row = _run_method(method, item, repeat_index, work)
                    current.append(row)
                    if row["status"] != "success":
                        break
                write_tsv(checkpoint, current)
            print(json.dumps({"completed_setting": f"{item['system']}__{item['prefix_label']}", "index": setting_index + 1, "total": len(manifest)}), flush=True)

    if args.smoke:
        write_json(checkpoint_root / "smoke_status.json", {"status": "pass", "settings": len(manifest)})
        return
    all_rows: list[dict[str, str]] = []
    for item in manifest:
        all_rows.extend(read_tsv(_checkpoint_path(item, CHECKPOINTS)))
    write_tsv(DEST / "atlas_no_output_repeats.tsv", all_rows)
    summaries, stages = _summarize(all_rows)
    identities = _identity(all_rows)
    write_tsv(DEST / "atlas_no_output_summary.tsv", summaries)
    write_tsv(DEST / "atlas_stage_decomposition.tsv", stages)
    write_tsv(DEST / "atlas_numerical_identity.tsv", identities)
    complete = len(manifest) == 48 and all(
        _complete(
            [
                row
                for row in all_rows
                if row["system"] == item["system"]
                and row["prefix_label"] == item["prefix_label"]
            ],
            method,
        )
        for item in manifest
        for method in METHODS
    )
    integrity = all(bool(row["stock_no_output_native_identity_pass"]) and bool(row["CPU_GPU_implementation_consistency_pass"]) for row in identities)
    write_text(
        DEST / "atlas_no_output_report.md",
        f"""# ATLAS No-materialisation Benchmark

Status: **{'PASS' if complete and integrity else 'FAIL'}**. The locked panel
contains {len(manifest)} system-prefix settings, each with one warmup and five
measured repeats for the five registered methods. C1 includes parsing the same
centred CA multi-model PDB; C2 excludes parsing and output. No aligned
coordinates, W5, RMSF, diagnostics or benchmark table files were produced
inside the timed regions. Stock no-output native values were checked against
the frozen materialised-output benchmark, and CPU/GPU throughput core labels
were checked for implementation consistency.
""",
    )
    write_json(DEST / "nm4_status.json", {"status": "pass" if complete and integrity else "fail", "settings": len(manifest), "complete": complete, "integrity": integrity})
    if not (complete and integrity):
        raise RuntimeError("NM4 completeness or integrity gate failed")


if __name__ == "__main__":
    main()
