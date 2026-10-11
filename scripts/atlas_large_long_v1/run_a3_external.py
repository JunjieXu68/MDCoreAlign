#!/usr/bin/env python3
"""Run matched MDLovoFit and feasible THESEUS ATLAS prefix benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np
from scipy.stats import spearmanr

from mdsuperpose.alignment import solve_rigid_alignment
from mdsuperpose.flexibility import compute_rmsf
from mdsuperpose.mdlovofit import read_multimodel_pdb_with_occupancy, run_mdlovofit_on_files
from mdsuperpose.theseus import read_multimodel_pdb_coords, run_theseus_on_files
from scripts.atlas_large_long_v1.common import (
    MDLOVOFIT,
    OUT,
    PREFIXES,
    REPEATS,
    ROOT,
    THESEUS,
    WARMUPS,
    prefix_label,
    read_tsv,
    write_json,
    write_tsv,
)


CHECKPOINTS = OUT / "a3_external/checkpoints"
LOGS = OUT / "a3_external/raw_logs"
MDL_TIMEOUT = 1800
THESEUS_TIMEOUT = 600
THESEUS_MAX_POINTS = 2_000_000


def _core(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _flexible(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    count = max(1, int(np.floor(scores.size * fraction + 0.5)))
    order = np.lexsort((np.arange(scores.size), -scores))
    mask = np.zeros(scores.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _orient_and_metrics(aligned: np.ndarray, reference: np.ndarray):
    consensus = np.mean(aligned, axis=0)
    orientation = solve_rigid_alignment(consensus, reference)
    oriented = aligned @ orientation.rotation.T + orientation.translation
    rmsd = np.sqrt(np.mean(np.sum((oriented - reference[None, :, :]) ** 2, axis=2), axis=1))
    rmsf = compute_rmsf(oriented)
    return oriented, rmsd, rmsf


def _load_context(system: str, frames: int):
    manifest = {row["system"]: row for row in read_tsv(OUT / "atlas_manifest.tsv")}[system]
    centered = {
        (row["system"], row["prefix_label"]): row
        for row in read_tsv(OUT / "a3_external/centered_input_manifest.tsv")
    }[(system, prefix_label(frames))]
    coords = np.load(ROOT / manifest["coordinate_cache_path"], mmap_mode="r")[:frames]
    reference = np.asarray(coords[0], dtype=np.float64)
    reference = reference - np.mean(reference, axis=0, keepdims=True)
    cache = np.load(
        OUT / "a3_loaded_analysis/result_cache" / f"{system}__{prefix_label(frames)}.npz",
        allow_pickle=False,
    )
    return manifest, ROOT / centered["centered_input_path"], reference, cache


def _failure_rows(system: str, frames: int, method: str, exc: Exception):
    return [
        {
            "system": system,
            "prefix_label": prefix_label(frames),
            "n_frames": frames,
            "method": method,
            "repeat_index": index,
            "is_warmup": False,
            "status": "timeout" if type(exc).__name__ == "TimeoutExpired" else "failure",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "timeout_seconds": MDL_TIMEOUT if method.startswith("MDLovoFit") else THESEUS_TIMEOUT,
        }
        for index in range(REPEATS)
    ]


def _run_mdlovofit(system: str, frames: int, input_path: Path, reference: np.ndarray, exact) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    quality: dict[str, object] = {}
    log_dir = LOGS / system / prefix_label(frames) / "mdlovofit"
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atlas_mdl_", dir="/tmp") as temporary:
        work = Path(temporary)
        for index in range(WARMUPS + REPEATS):
            output = work / f"aligned_{index}.pdb"
            try:
                run = run_mdlovofit_on_files(
                    [input_path],
                    fraction=0.6,
                    output_path=output,
                    executable=MDLOVOFIT,
                    timeout_seconds=MDL_TIMEOUT,
                )
                read_start = time.perf_counter()
                aligned, occupancies = read_multimodel_pdb_with_occupancy(output, reference.shape[0])
                parse_seconds = time.perf_counter() - read_start
                metrics_start = time.perf_counter()
                _oriented, rmsd, rmsf = _orient_and_metrics(aligned, reference)
                metric_seconds = time.perf_counter() - metrics_start
                parsed = run.parsed_output
                weights = np.mean(occupancies, axis=0)
                core = _core(weights)
                total = run.subprocess_runtime_seconds + parse_seconds + metric_seconds
                rows.extend(
                    [
                        {
                            "system": system,
                            "prefix_label": prefix_label(frames),
                            "n_frames": frames,
                            "n_residues": reference.shape[0],
                            "residue_frames": frames * reference.shape[0],
                            "contract": "A_loaded_analysis_external_with_unavoidable_PDB_output",
                            "method": "MDLovoFit_20.1.1",
                            "repeat_index": index - WARMUPS,
                            "is_warmup": index < WARMUPS,
                            "status": "success",
                            "total_seconds": total,
                            "subprocess_seconds": run.subprocess_runtime_seconds,
                            "output_parse_seconds": parse_seconds,
                            "posthoc_metrics_seconds": metric_seconds,
                            "materialised_output_seconds": run.subprocess_runtime_seconds,
                            "input_PDB_preparation_included": False,
                            "frames_per_second": frames / total,
                            "residue_frames_per_second": frames * reference.shape[0] / total,
                            "peak_RSS_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                            "unified_posthoc_mean_RMSD_A": float(np.mean(rmsd)),
                            "mean_RMSF_A": float(np.mean(rmsf)),
                            "native_all_atom_RMSD_A": parsed.average_rmsd_all_atoms,
                            "native_least_mobile_RMSD_A": parsed.average_rmsd_least_mobile,
                            "native_selected_fraction": parsed.fraction_of_best_aligned_atoms,
                            "core_size_from_mean_occupancy_top60": int(np.count_nonzero(core)),
                            "occupancy_binary": bool(np.all(np.isclose(occupancies, 0.0) | np.isclose(occupancies, 1.0))),
                            "centering_convention": "per_frame_CA_COG",
                        },
                        {
                            "system": system,
                            "prefix_label": prefix_label(frames),
                            "n_frames": frames,
                            "n_residues": reference.shape[0],
                            "residue_frames": frames * reference.shape[0],
                            "contract": "C_materialised_output_external_native_PDB",
                            "method": "MDLovoFit_20.1.1",
                            "repeat_index": index - WARMUPS,
                            "is_warmup": index < WARMUPS,
                            "status": "success",
                            "total_seconds": run.subprocess_runtime_seconds,
                            "subprocess_seconds": run.subprocess_runtime_seconds,
                            "materialised_output_seconds": run.subprocess_runtime_seconds,
                            "output_size_bytes": output.stat().st_size,
                            "input_PDB_preparation_included": False,
                            "notes": "MDLovoFit subprocess includes unavoidable aligned-PDB writing.",
                        },
                    ]
                )
                quality = {
                    "system": system,
                    "prefix_label": prefix_label(frames),
                    "n_frames": frames,
                    "n_residues": reference.shape[0],
                    "method": "MDLovoFit_20.1.1",
                    "core_Jaccard_vs_exact": _jaccard(core, exact["exact_core"]),
                    "weight_Spearman_vs_exact": _spearman(weights, exact["exact_weights"]),
                    "RMSF_Spearman_vs_exact": _spearman(rmsf, exact["exact_rmsf"]),
                    "RMSF_MAE_A_vs_exact": float(np.mean(np.abs(rmsf - exact["exact_rmsf"]))),
                    "flexible_label_agreement_vs_exact": float(np.mean(_flexible(rmsf) == _flexible(exact["exact_rmsf"]))),
                    "median_absolute_RMSD_difference_A_vs_exact": float(np.median(np.abs(rmsd - exact["exact_rmsd"]))),
                    "P95_absolute_RMSD_difference_A_vs_exact": float(np.quantile(np.abs(rmsd - exact["exact_rmsd"]), 0.95)),
                    "registered_catastrophic_quality_failure": bool(not np.isfinite(rmsd).all() or not np.isfinite(rmsf).all() or np.count_nonzero(core) == 0 or _jaccard(core, exact["exact_core"]) < 0.5 or _spearman(rmsf, exact["exact_rmsf"]) < 0.9),
                    "large_geometric_tail_median_RMSD_gt_1A": float(np.median(np.abs(rmsd - exact["exact_rmsd"]))) > 1.0,
                }
                if index == WARMUPS:
                    (log_dir / "stdout.txt").write_text(run.stdout)
                    (log_dir / "stderr.txt").write_text(run.stderr)
                    (log_dir / "command.json").write_text(json.dumps(list(run.command), indent=2) + "\n")
                output.unlink(missing_ok=True)
                del aligned, occupancies, rmsd, rmsf, weights
            except Exception as exc:
                rows.extend(_failure_rows(system, frames, "MDLovoFit_20.1.1", exc))
                quality = {"system": system, "prefix_label": prefix_label(frames), "n_frames": frames, "n_residues": reference.shape[0], "method": "MDLovoFit_20.1.1", "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
                break
    return rows, quality


def _run_theseus(system: str, frames: int, input_path: Path, reference: np.ndarray, exact) -> tuple[list[dict[str, object]], dict[str, object]]:
    points = frames * reference.shape[0]
    if points > THESEUS_MAX_POINTS:
        return [
            {
                "system": system,
                "prefix_label": prefix_label(frames),
                "n_frames": frames,
                "n_residues": reference.shape[0],
                "residue_frames": points,
                "contract": "registered_external_feasibility",
                "method": "THESEUS_3.3.0",
                "status": "skipped_registered_feasibility_rule",
                "feasibility_rule": "frame_residue_points_le_2000000",
            }
        ], {"system": system, "prefix_label": prefix_label(frames), "method": "THESEUS_3.3.0", "status": "skipped_registered_feasibility_rule"}
    rows = []
    quality: dict[str, object] = {}
    log_dir = LOGS / system / prefix_label(frames) / "theseus"
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atlas_theseus_", dir="/tmp") as temporary:
        work = Path(temporary)
        for index in range(WARMUPS + REPEATS):
            run_dir = work / f"run_{index}"
            try:
                run = run_theseus_on_files(
                    [input_path],
                    output_root="theseus",
                    output_directory=run_dir,
                    executable=THESEUS,
                    timeout_seconds=THESEUS_TIMEOUT,
                )
                read_start = time.perf_counter()
                aligned = read_multimodel_pdb_coords(run.aligned_pdb_path, reference.shape[0])
                parse_seconds = time.perf_counter() - read_start
                metrics_start = time.perf_counter()
                _oriented, rmsd, rmsf = _orient_and_metrics(aligned, reference)
                metric_seconds = time.perf_counter() - metrics_start
                variance = np.asarray([record.variance for record in run.variance_records], dtype=float)
                native_core = np.asarray([record.is_core for record in run.variance_records], dtype=bool)
                mapping_available = variance.size == reference.shape[0]
                weights = 1.0 / (variance + 1e-12) if mapping_available else np.asarray([])
                core = native_core if mapping_available and np.any(native_core) else (_core(weights) if mapping_available else None)
                total = run.subprocess_runtime_seconds + parse_seconds + metric_seconds
                rows.extend(
                    [
                        {
                            "system": system,
                            "prefix_label": prefix_label(frames),
                            "n_frames": frames,
                            "n_residues": reference.shape[0],
                            "residue_frames": points,
                            "contract": "A_loaded_analysis_external_with_unavoidable_PDB_output",
                            "method": "THESEUS_3.3.0",
                            "repeat_index": index - WARMUPS,
                            "is_warmup": index < WARMUPS,
                            "status": "success",
                            "total_seconds": total,
                            "subprocess_seconds": run.subprocess_runtime_seconds,
                            "output_parse_seconds": parse_seconds,
                            "posthoc_metrics_seconds": metric_seconds,
                            "materialised_output_seconds": run.subprocess_runtime_seconds,
                            "input_PDB_preparation_included": False,
                            "frames_per_second": frames / total,
                            "residue_frames_per_second": points / total,
                            "peak_RSS_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                            "unified_posthoc_mean_RMSD_A": float(np.mean(rmsd)),
                            "mean_RMSF_A": float(np.mean(rmsf)),
                            "native_maximum_likelihood_RMSD_A": run.parsed_output.maximum_likelihood_rmsd,
                            "native_core_fraction": run.output_core_atom_fraction,
                            "variance_mapping_available": mapping_available,
                        },
                        {
                            "system": system,
                            "prefix_label": prefix_label(frames),
                            "n_frames": frames,
                            "n_residues": reference.shape[0],
                            "residue_frames": points,
                            "contract": "C_materialised_output_external_native_PDB",
                            "method": "THESEUS_3.3.0",
                            "repeat_index": index - WARMUPS,
                            "is_warmup": index < WARMUPS,
                            "status": "success",
                            "total_seconds": run.subprocess_runtime_seconds,
                            "subprocess_seconds": run.subprocess_runtime_seconds,
                            "materialised_output_seconds": run.subprocess_runtime_seconds,
                            "output_size_bytes": run.aligned_pdb_path.stat().st_size,
                            "input_PDB_preparation_included": False,
                        },
                    ]
                )
                quality = {
                    "system": system,
                    "prefix_label": prefix_label(frames),
                    "n_frames": frames,
                    "n_residues": reference.shape[0],
                    "method": "THESEUS_3.3.0",
                    "core_Jaccard_vs_exact": _jaccard(core, exact["exact_core"]) if core is not None else "",
                    "weight_Spearman_vs_exact": _spearman(weights, exact["exact_weights"]) if mapping_available else "",
                    "RMSF_Spearman_vs_exact": _spearman(rmsf, exact["exact_rmsf"]),
                    "RMSF_MAE_A_vs_exact": float(np.mean(np.abs(rmsf - exact["exact_rmsf"]))),
                    "flexible_label_agreement_vs_exact": float(np.mean(_flexible(rmsf) == _flexible(exact["exact_rmsf"]))),
                    "median_absolute_RMSD_difference_A_vs_exact": float(np.median(np.abs(rmsd - exact["exact_rmsd"]))),
                    "P95_absolute_RMSD_difference_A_vs_exact": float(np.quantile(np.abs(rmsd - exact["exact_rmsd"]), 0.95)),
                    "registered_catastrophic_quality_failure": bool(not np.isfinite(rmsd).all() or not np.isfinite(rmsf).all() or _spearman(rmsf, exact["exact_rmsf"]) < 0.9),
                    "large_geometric_tail_median_RMSD_gt_1A": float(np.median(np.abs(rmsd - exact["exact_rmsd"]))) > 1.0,
                }
                if index == WARMUPS:
                    (log_dir / "stdout.txt").write_text(run.stdout)
                    (log_dir / "stderr.txt").write_text(run.stderr)
                    (log_dir / "command.json").write_text(json.dumps(list(run.command), indent=2) + "\n")
                del aligned, rmsd, rmsf
                shutil.rmtree(run_dir, ignore_errors=True)
            except Exception as exc:
                rows.extend(_failure_rows(system, frames, "THESEUS_3.3.0", exc))
                quality = {"system": system, "prefix_label": prefix_label(frames), "n_frames": frames, "n_residues": reference.shape[0], "method": "THESEUS_3.3.0", "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
                break
    return rows, quality


def run(system: str) -> None:
    for frames in PREFIXES:
        checkpoint = CHECKPOINTS / f"{system}__{prefix_label(frames)}.tsv"
        quality_path = CHECKPOINTS / f"{system}__{prefix_label(frames)}__quality.tsv"
        if checkpoint.exists() and quality_path.exists():
            print(f"[A3 external] skip {system} {frames}", flush=True)
            continue
        _manifest, input_path, reference, exact = _load_context(system, frames)
        mdl_rows, mdl_quality = _run_mdlovofit(system, frames, input_path, reference, exact)
        theseus_rows, theseus_quality = _run_theseus(system, frames, input_path, reference, exact)
        write_tsv(checkpoint, [*mdl_rows, *theseus_rows])
        write_tsv(quality_path, [mdl_quality, theseus_quality])
        exact.close()
        print(f"[A3 external] complete {system} {frames}", flush=True)
    write_json(OUT / f"a3_external/{system}_status.json", {"system": system, "status": "complete", "prefixes": list(PREFIXES)})


def finalize() -> None:
    raw = []
    quality = []
    for row in read_tsv(OUT / "atlas_manifest.tsv"):
        for frames in PREFIXES:
            raw.extend(read_tsv(CHECKPOINTS / f"{row['system']}__{prefix_label(frames)}.tsv"))
            quality.extend(read_tsv(CHECKPOINTS / f"{row['system']}__{prefix_label(frames)}__quality.tsv"))
    write_tsv(OUT / "a3_external/external_raw_repeats.tsv", raw)
    write_tsv(OUT / "a3_external/external_quality_per_case.tsv", quality)
    write_json(OUT / "a3_external/external_status.json", {"status": "complete", "rows": len(raw), "quality_rows": len(quality)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "finalize"))
    parser.add_argument("--system")
    args = parser.parse_args()
    if args.mode == "finalize":
        finalize()
        return
    systems = [row["system"] for row in read_tsv(OUT / "atlas_manifest.tsv")]
    selected = systems if args.system is None else [args.system]
    for system in selected:
        run(system)


if __name__ == "__main__":
    main()
