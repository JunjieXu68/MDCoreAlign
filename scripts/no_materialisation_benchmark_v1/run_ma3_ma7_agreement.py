#!/usr/bin/env python3
"""Run the locked, untimed MA3-MA7 method-agreement panels."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from mdcorealign.benchmark_minimal import (
    load_ca_multimodel_pdb_minimal,
    solve_minimal_cpu,
)
from scripts.no_materialisation_benchmark_v1.common import OUT, ROOT, write_json, write_text, write_tsv
from scripts.no_materialisation_benchmark_v1.method_agreement_metrics import (
    alignment_pair_metrics,
    chainwise_jaccard,
    descriptive_category,
    fit_with_mask,
    fit_with_weights,
    heldout_geometry,
    mask_metrics,
    rank_biased_overlap,
    safe_kendall,
    safe_spearman,
    segment_metrics,
    top_k_mask,
)
from scripts.no_materialisation_benchmark_v1.run_nm1_verification import _mdcath_input


MA = OUT / "method_agreement_v2"
MANIFESTS = MA / "00_contracts/panel_manifests"
EXPORTER = MA / "01_mdlovofit_export_validation/bin/mdlovofit_export"
CHECKPOINTS = MA / "08_reporting/work_checkpoints"
PROGRESS = MA / "08_reporting/ma3_ma7_progress.log"
FRACTIONS = (0.40, 0.50, 0.60, 0.70, 0.80)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _count_pdb(path: Path) -> tuple[int, int]:
    frames = 0
    current = 0
    first_count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"ATOM"):
                current += 1
            elif line.startswith(b"ENDMDL"):
                if first_count == 0:
                    first_count = current
                elif current != first_count:
                    raise ValueError(f"inconsistent model size in {path}")
                frames += 1
                current = 0
    if current:
        raise ValueError(f"incomplete final model in {path}")
    return frames, first_count


def _load_task_coords(task: dict[str, object], workspace: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    kind = str(task["kind"])
    if kind == "mdcath":
        path = workspace / f"{task['case_id']}_centered_CA.pdb"
        frames, residues, _ = _mdcath_input(str(task["domain_id"]), path)
    else:
        path = ROOT / str(task["input_PDB"])
        if "n_frames" in task and "n_residues" in task:
            frames, residues = int(task["n_frames"]), int(task["n_residues"])
        else:
            frames, residues = _count_pdb(path)
    coords, chains = load_ca_multimodel_pdb_minimal(
        path,
        expected_frames=frames,
        expected_residues=residues,
    )
    return coords, chains, path


def _read_export(prefix: Path) -> dict[str, object]:
    mask_rows = read_tsv(Path(str(prefix) + "_final_fixed_subset.tsv"))
    frequency_rows = read_tsv(Path(str(prefix) + "_per_frame_final_selection_frequency.tsv"))
    evidence_rows = read_tsv(Path(str(prefix) + "_internal_selection_evidence.tsv"))
    transform_rows = read_tsv(Path(str(prefix) + "_final_transforms.tsv"))
    return {
        "mask": np.asarray([int(row["selected_in_final_fixed_subset"]) == 1 for row in mask_rows]),
        "frequency": np.asarray([float(row["selection_frequency"]) for row in frequency_rows]),
        "evidence": np.asarray([float(row["aggregate_improvement_event_count"]) for row in evidence_rows]),
        "rotations": np.asarray([[[float(row[f"rotation_{i}{j}"]) for j in range(3)] for i in range(3)] for row in transform_rows]),
        "translations": np.asarray([[float(row["translation_x"]), float(row["translation_y"]), float(row["translation_z"])] for row in transform_rows]),
    }


def _run_exporter(input_path: Path, workspace: Path, fraction: float) -> tuple[float, dict[str, object]]:
    prefix = workspace / f"mdlovofit_f{int(round(fraction * 100)):02d}"
    completed = subprocess.run(
        [str(EXPORTER), "-f", f"{fraction:.6f}", "-ma-export", str(prefix), str(input_path)],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=7200,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"exporter failed at fraction {fraction}: {completed.stderr[-2000:]}")
    return fraction, _read_export(prefix)


def _weights_and_native_fits(
    task: dict[str, object],
    coords: np.ndarray,
    chains: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if str(task["kind"]).startswith("atlas"):
        cache = ROOT / "results/atlas_large_long_v1/a3_loaded_analysis/result_cache" / f"{task['system']}__{task['prefix_label']}.npz"
        frozen = np.load(cache)
        exact_weights = np.asarray(frozen["exact_weights"], dtype=np.float64)
        throughput_weights = np.asarray(frozen["cpu_weights"], dtype=np.float64)
        exact_rotations, exact_translations = fit_with_weights(coords, exact_weights)
        throughput_rotations, throughput_translations = fit_with_weights(coords, throughput_weights)
    else:
        exact = solve_minimal_cpu(coords, mode="exact", chain_ids=chains, representative_count=min(coords.shape[0], 2000), chunk_size=1000)
        throughput = solve_minimal_cpu(coords, mode="throughput", chain_ids=chains, representative_count=min(coords.shape[0], 256), update_rounds=4, chunk_size=1000)
        exact_weights = np.asarray(exact.frozen_weights, dtype=np.float64)
        throughput_weights = np.asarray(throughput.frozen_weights, dtype=np.float64)
        exact_rotations, exact_translations = exact.rotations, exact.translations
        throughput_rotations, throughput_translations = throughput.rotations, throughput.translations
    return {
        "exact": {"weights": exact_weights, "rotations": exact_rotations, "translations": exact_translations},
        "throughput": {"weights": throughput_weights, "rotations": throughput_rotations, "translations": throughput_translations},
    }


def _fraction_rows(
    task: dict[str, object],
    fraction: float,
    export: dict[str, object],
    methods: dict[str, dict[str, np.ndarray]],
    coords: np.ndarray,
    chains: np.ndarray,
    detailed: bool,
) -> dict[str, list[dict[str, object]]]:
    common = {
        "panel": task["panel"],
        "case_id": task["case_id"],
        "system": task.get("system", task.get("domain_id", task["case_id"])),
        "prefix_label": task.get("prefix_label", "full"),
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "fraction": fraction,
        "K": math.floor(fraction * coords.shape[1]),
        "chance_Jaccard": fraction / (2.0 - fraction),
    }
    k = int(common["K"])
    mdl_mask = np.asarray(export["mask"], dtype=bool)
    if int(np.count_nonzero(mdl_mask)) != k:
        raise RuntimeError(f"MDLovoFit mask cardinality mismatch for {task['case_id']} fraction {fraction}")
    output: dict[str, list[dict[str, object]]] = {"core": [], "continuous": [], "geometry": [], "heldout": []}
    for method_name, method in methods.items():
        weights = np.asarray(method["weights"], dtype=np.float64)
        candidate_mask = top_k_mask(weights, k)
        binary = mask_metrics(mdl_mask, candidate_mask)
        segments = segment_metrics(mdl_mask, candidate_mask, chains)
        chain_mean, chain_values = chainwise_jaccard(mdl_mask, candidate_mask, chains)
        frequency = np.asarray(export["frequency"], dtype=np.float64)
        evidence = np.asarray(export["evidence"], dtype=np.float64)
        frequency_spearman = safe_spearman(frequency, weights)
        core_row = {
            **common,
            "MDCoreAlign_method": method_name,
            **binary,
            **segments,
            "chainwise_Jaccard_mean": chain_mean,
            "chainwise_Jaccard": chain_values,
            "above_equal_cardinality_chance": float(binary["Jaccard"]) > float(common["chance_Jaccard"]),
        }
        output["core"].append(core_row)
        output["continuous"].append(
            {
                **common,
                "MDCoreAlign_method": method_name,
                "selection_frequency_Spearman": frequency_spearman,
                "selection_frequency_Kendall": safe_kendall(frequency, weights),
                "internal_evidence_Spearman": safe_spearman(evidence, weights),
                "internal_evidence_Kendall": safe_kendall(evidence, weights),
                "selection_frequency_RBO_p0.9": rank_biased_overlap(frequency, weights),
                "internal_evidence_RBO_p0.9": rank_biased_overlap(evidence, weights),
            }
        )
        output["heldout"].append({**common, "method": "MDLovoFit", **heldout_geometry(coords, mdl_mask, seed=101)})
        output["heldout"].append({**common, "method": f"MDCoreAlign_{method_name}", **heldout_geometry(coords, candidate_mask, seed=103)})
        if detailed:
            native_metrics, mdl_rmsf, candidate_rmsf = alignment_pair_metrics(
                coords,
                np.asarray(export["rotations"]),
                np.asarray(export["translations"]),
                method["rotations"],
                method["translations"],
                mdl_mask,
                candidate_mask,
            )
            native_metrics.update(
                {
                    **common,
                    "MDCoreAlign_method": method_name,
                    "comparison_layer": "C_native_method_alignment",
                    "mobile_mask_Jaccard": float(mask_metrics(~mdl_mask, ~candidate_mask)["Jaccard"]),
                }
            )
            native_metrics["descriptive_agreement_category"] = descriptive_category(
                float(binary["Jaccard"]),
                frequency_spearman,
                float(native_metrics["median_aligned_coordinate_difference_A"]),
                float(native_metrics["RMSF_Spearman"]),
            )
            output["geometry"].append(native_metrics)
            hard_rotations, hard_translations = fit_with_mask(coords, candidate_mask)
            hard_metrics, _, _ = alignment_pair_metrics(
                coords,
                np.asarray(export["rotations"]),
                np.asarray(export["translations"]),
                hard_rotations,
                hard_translations,
                mdl_mask,
                candidate_mask,
            )
            hard_metrics.update(
                {
                    **common,
                    "MDCoreAlign_method": method_name,
                    "comparison_layer": "B_matched_hard_subset_refit",
                    "mobile_mask_Jaccard": float(mask_metrics(~mdl_mask, ~candidate_mask)["Jaccard"]),
                    "descriptive_agreement_category": "not_applied_to_layer_B",
                }
            )
            output["geometry"].append(hard_metrics)
    # Deduplicate MDLovoFit held-out rows added once per MDCoreAlign method.
    seen = set()
    output["heldout"] = [row for row in output["heldout"] if not ((key := (row["method"], row["case_id"], row["fraction"])) in seen or seen.add(key))]
    return output


def _checkpoint_path(task: dict[str, object]) -> Path:
    safe = str(task["case_id"]).replace("/", "_")
    return CHECKPOINTS / f"{safe}.json"


def process_task(task: dict[str, object]) -> dict[str, object]:
    checkpoint = _checkpoint_path(task)
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text())
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"ma_agree_{task['case_id']}_") as temporary:
        workspace = Path(temporary)
        coords, chains, input_path = _load_task_coords(task, workspace)
        methods = _weights_and_native_fits(task, coords, chains)
        fractions = FRACTIONS if bool(task["detailed"]) else (0.60,)
        exports: dict[float, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=len(fractions)) as executor:
            futures = [executor.submit(_run_exporter, input_path, workspace, fraction) for fraction in fractions]
            for future in as_completed(futures):
                fraction, export = future.result()
                exports[float(fraction)] = export
        combined = {"core": [], "continuous": [], "geometry": [], "heldout": []}
        for fraction in fractions:
            rows = _fraction_rows(task, fraction, exports[float(fraction)], methods, coords, chains, bool(task["detailed"]))
            for key in combined:
                combined[key].extend(rows[key])
    result = {
        "task": task,
        "elapsed_untimed_seconds": time.time() - started,
        **combined,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, allow_nan=True))
    temporary_path.replace(checkpoint)
    return result


def _tasks() -> list[dict[str, object]]:
    detailed_ids = {row["domain_id"] for row in read_tsv(MANIFESTS / "mdcath_detailed_128.tsv")}
    mdcath = [
        {
            "panel": "mdCATH_primary_512",
            "kind": "mdcath",
            "case_id": row["domain_id"],
            "domain_id": row["domain_id"],
            "n_frames": int(row["n_frames"]),
            "n_residues": int(row["n_residues"]),
            "detailed": row["domain_id"] in detailed_ids,
        }
        for row in read_tsv(MANIFESTS / "mdcath_primary_512.tsv")
    ]
    atlas_full = [
        {
            "panel": "ATLAS_full_8",
            "kind": "atlas_full",
            "case_id": f"{row['system']}__{row['prefix_label']}",
            "system": row["system"],
            "prefix_label": row["prefix_label"],
            "n_frames": int(row["n_frames"]),
            "n_residues": int(row["n_residues"]),
            "input_PDB": row["input_PDB"],
            "detailed": True,
        }
        for row in read_tsv(MANIFESTS / "atlas_full_length_8.tsv")
    ]
    atlas_tails = [
        {
            "panel": "ATLAS_short_tails_6",
            "kind": "atlas_tail",
            "case_id": f"{row['system']}__{row['prefix_label']}",
            "system": row["system"],
            "prefix_label": row["prefix_label"],
            "n_frames": int(row["n_frames"]),
            "n_residues": int(row["n_residues"]),
            "input_PDB": row["input_PDB"],
            "detailed": True,
        }
        for row in read_tsv(MANIFESTS / "atlas_short_prefix_tails_6.tsv")
    ]
    historical = [
        {
            "panel": "historical_four_systems",
            "kind": "historical",
            "case_id": row["system"],
            "system": row["system"],
            "input_PDB": row["input_PDB"],
            "detailed": True,
        }
        for row in read_tsv(MANIFESTS / "historical_four_systems.tsv")
        if row["available"] == "True"
    ]
    # Start the two longest cases early while leaving workers for rapid progress.
    atlas_full.sort(key=lambda row: int(row["n_residues"]), reverse=True)
    return atlas_full[:2] + historical + mdcath + atlas_tails + atlas_full[2:]


def _write_combined(results: list[dict[str, object]]) -> None:
    core = [row for result in results for row in result["core"]]
    continuous = [row for result in results for row in result["continuous"]]
    geometry = [row for result in results for row in result["geometry"]]
    heldout = [row for result in results for row in result["heldout"]]
    write_tsv(MA / "08_reporting/combined_core_agreement.tsv", core)
    write_tsv(MA / "08_reporting/combined_continuous_agreement.tsv", continuous)
    write_tsv(MA / "08_reporting/combined_geometry_agreement.tsv", geometry)
    write_tsv(MA / "08_reporting/combined_heldout_geometry.tsv", heldout)
    write_json(
        MA / "08_reporting/ma3_ma7_raw_status.json",
        {
            "cases": len(results),
            "core_rows": len(core),
            "continuous_rows": len(continuous),
            "geometry_rows": len(geometry),
            "heldout_rows": len(heldout),
            "status": "pass",
        },
    )


def main() -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tasks = _tasks()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    completed_results: list[dict[str, object]] = []
    started = time.time()
    with PROGRESS.open("a", buffering=1) as progress:
        progress.write(f"START\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\ttasks={len(tasks)}\n")
        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_task = {executor.submit(process_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    progress.write(f"FAILED\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{task['case_id']}\t{type(exc).__name__}:{exc}\n")
                    raise
                completed_results.append(result)
                elapsed = time.time() - started
                rate = index / elapsed * 3600.0
                progress.write(
                    f"COMPLETE\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{index}/{len(tasks)}\t{task['case_id']}\t"
                    f"case_seconds={result['elapsed_untimed_seconds']:.3f}\trate_per_hour={rate:.2f}\n"
                )
    # Include any checkpoints returned in completion order and sort deterministically.
    completed_results.sort(key=lambda result: str(result["task"]["case_id"]))
    _write_combined(completed_results)
    write_text(
        MA / "08_reporting/ma3_ma7_raw_run_report.md",
        f"# MA3-MA7 Untimed Extraction Run\n\nCompleted {len(completed_results)}/{len(tasks)} locked cases in {(time.time()-started)/3600.0:.3f} h. All external exporter calls and MDCoreAlign compact extractions were untimed scientific analyses and are excluded from NM speed claims.\n",
    )
    print(json.dumps({"status": "pass", "cases": len(completed_results), "hours": (time.time() - started) / 3600.0}, indent=2))


if __name__ == "__main__":
    main()
