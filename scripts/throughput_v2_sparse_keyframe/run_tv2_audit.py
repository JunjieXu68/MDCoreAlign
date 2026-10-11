#!/usr/bin/env python3
"""Profile the frozen current-throughput compact contract on a locked panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose.alignment import apply_transform, solve_weighted_rigid_alignment  # noqa: E402
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights  # noqa: E402
from mdsuperpose.chronology_study import representative_indices  # noqa: E402
from mdsuperpose.parallel_weights import estimate_shared_weight  # noqa: E402
from scripts.cpu_chronology_v3.run_self_variability import load_case  # noqa: E402


OUT = ROOT / "results/throughput_v2_sparse_keyframe/02_current_throughput_audit"
PANEL = OUT / "calibration_panel.tsv"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_panel() -> list[dict[str, object]]:
    calibration = read_tsv(
        ROOT / "results/cpu_chronology_v3/02_self_variability/calibration_panel.tsv"
    )
    selected_md = [calibration[index] for index in (0, 1, 2, 3)]
    rows: list[dict[str, object]] = []
    for row in selected_md:
        rows.append(
            {
                "case_id": row["case_id"],
                "source_class": "mdCATH_calibration",
                "input_path": row["input_path"],
                "input_key": row["input_key"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "control_target_frames": "",
                "selection_reason": "locked size-spanning mdCATH calibration case",
            }
        )
    atlas = read_tsv(ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
    atlas_by_id = {row["trajectory_id"]: row for row in atlas}
    for case_id in ("ATLAS_1u55_A", "ATLAS_1r6w_A", "ATLAS_1jnr_C", "ATLAS_6sup_A"):
        row = atlas_by_id[case_id]
        rows.append(
            {
                "case_id": case_id,
                "source_class": "ATLAS_R1_real_continuous",
                "input_path": row["coordinate_cache_path"],
                "input_key": "npy_array",
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "control_target_frames": "",
                "selection_reason": "locked ATLAS R1 size-spanning case",
            }
        )
    raw_root = (
        "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
        "domains/4oelA02/raw_replicates"
    )
    for frames in (50_000, 100_000):
        rows.append(
            {
                "case_id": f"transition_control_4oelA02_{frames}",
                "source_class": "transition_aligned_synthetic_scaling_control",
                "input_path": raw_root,
                "input_key": "A-E cycle rebuilt from raw replicate each use",
                "n_frames": frames,
                "n_residues": 153,
                "control_target_frames": frames,
                "selection_reason": "existing 153-residue transition-aligned scaling family",
            }
        )
    write_tsv(PANEL, rows)
    return rows


def load_control(raw_root: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    raw: list[np.ndarray] = []
    for label in "ABCDE":
        with np.load(raw_root / f"{label}_ca_coords.npz", allow_pickle=True) as archive:
            raw.append(np.asarray(archive["coords"], dtype=np.float64))
    n_residues = raw[0].shape[1]
    output = np.empty((n_frames, n_residues, 3), dtype=np.float64)
    first = raw[0]
    copied = min(n_frames, first.shape[0])
    initial = first - np.mean(first[-1], axis=0)[None, None, :]
    output[:copied] = initial[:copied]
    current_last = output[copied - 1]
    position = copied
    segment = 1
    weights = np.ones(n_residues, dtype=np.float64)
    while position < n_frames:
        source = raw[segment % len(raw)]
        fit = solve_weighted_rigid_alignment(source[0], current_last, weights)
        transformed = apply_transform(source, fit)
        stop = min(n_frames, position + transformed.shape[0])
        output[position:stop] = transformed[: stop - position]
        current_last = output[stop - 1]
        position = stop
        segment += 1
    return output, np.full(n_residues, "A", dtype="U1")


def load_profile_case(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    if row["source_class"] == "transition_aligned_synthetic_scaling_control":
        coords, chains = load_control(
            ROOT / row["input_path"],
            int(row["control_target_frames"]),
        )
    else:
        load_row = {
            "case_id": row["case_id"],
            "input_path": row["input_path"],
            "input_key": row["input_key"],
            "n_frames": row["n_frames"],
            "n_residues": row["n_residues"],
        }
        coords, chains = load_case(load_row)
    return coords, chains, time.perf_counter() - started


def profile_case(row: dict[str, str]) -> dict[str, object]:
    coords, chains, decode_seconds = load_profile_case(row)
    reference = np.asarray(coords[0], dtype=np.float64)
    selected = representative_indices(coords.shape[0])
    representatives = np.asarray(coords[selected], dtype=np.float64)

    total_start = time.perf_counter()
    weight_start = time.perf_counter()
    estimate = estimate_shared_weight(representatives, reference, chain_ids=chains)
    weight_seconds = time.perf_counter() - weight_start
    fit_start = time.perf_counter()
    fit = fit_all_frames_fixed_weights(
        coords,
        reference,
        estimate.mean_weights,
        chunk_size=2000,
        return_aligned=False,
        compute_rmsd=True,
        compute_weighted_rmsd=True,
        compute_mean_residual=True,
        compute_robust_rmsf=True,
        compute_ordinary_rmsf=False,
    )
    fit_seconds = time.perf_counter() - fit_start
    algorithm_seconds = time.perf_counter() - total_start
    result_construction = max(0.0, algorithm_seconds - weight_seconds - fit_seconds)
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    timing = fit.timing
    return {
        "case_id": row["case_id"],
        "source_class": row["source_class"],
        "n_frames": int(coords.shape[0]),
        "n_residues": int(coords.shape[1]),
        "representative_count": int(selected.size),
        "decode_seconds": decode_seconds,
        "weight_estimation_seconds": weight_seconds,
        "representative_frame_fitting_seconds": float(
            estimate.timing.get("representative_weighted_fit", 0.0)
        ),
        "representative_residual_seconds": float(
            estimate.timing.get("representative_residual", 0.0)
        ),
        "representative_weight_update_seconds": float(
            estimate.timing.get("representative_weight_update", 0.0)
        ),
        "representative_regularization_seconds": float(
            estimate.timing.get("representative_sequence_regularization", 0.0)
        ),
        "all_frame_total_seconds": fit_seconds,
        "all_frame_covariance_transform_seconds": timing["all_frame_weighted_fit"],
        "transform_application_seconds": timing["transform_application"],
        "rmsd_reduction_seconds": timing["RMSD"],
        "stable_rmsf_seconds": timing["RMSF"],
        "result_construction_seconds": result_construction,
        "algorithm_total_seconds": algorithm_seconds,
        "raw_source_total_seconds": algorithm_seconds + decode_seconds,
        "peak_rss_mib": peak_rss_mib,
        "peak_vram_mib": 0.0,
        "mean_rmsd_A": float(np.mean(fit.rmsd)),
        "mean_weighted_rmsd_A": float(np.mean(fit.weighted_rmsd)),
        "ordinary_rmsf_computed": False,
        "aligned_coordinates_materialised": False,
        "contract": "current_throughput_compact",
    }


def run_worker(case_id: str) -> None:
    rows = read_tsv(PANEL)
    row = next(item for item in rows if item["case_id"] == case_id)
    result = profile_case(row)
    print(json.dumps(result, sort_keys=True), flush=True)


def stage_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    mapping = (
        ("input_decode", "decode_seconds", "decode"),
        ("weight_estimation", "weight_estimation_seconds", "algorithm"),
        ("representative_frame_fitting", "representative_frame_fitting_seconds", "weight_estimation"),
        ("all_frame_covariance_transform", "all_frame_covariance_transform_seconds", "all_frame"),
        ("transform_application", "transform_application_seconds", "all_frame"),
        ("RMSD_reduction", "rmsd_reduction_seconds", "all_residue_statistics"),
        ("stable_core_RMSF", "stable_rmsf_seconds", "all_residue_statistics"),
        ("result_construction", "result_construction_seconds", "algorithm"),
    )
    rows: list[dict[str, object]] = []
    for result in results:
        algorithm_total = float(result["algorithm_total_seconds"])
        raw_total = float(result["raw_source_total_seconds"])
        for stage, key, category in mapping:
            seconds = float(result[key])
            rows.append(
                {
                    "case_id": result["case_id"],
                    "source_class": result["source_class"],
                    "n_frames": result["n_frames"],
                    "n_residues": result["n_residues"],
                    "stage": stage,
                    "category": category,
                    "seconds": seconds,
                    "fraction_of_algorithm": seconds / algorithm_total,
                    "fraction_of_raw_source_total": seconds / raw_total,
                }
            )
    return rows


def write_summary(results: list[dict[str, object]], stages: list[dict[str, object]]) -> None:
    del stages
    by_stratum: dict[tuple[str, str], list[float]] = {}
    for result in results:
        total = float(result["algorithm_total_seconds"])
        categories = {
            "weight_estimation": float(result["weight_estimation_seconds"]),
            "per_frame_fitting": float(result["all_frame_covariance_transform_seconds"]),
            "all_residue_statistics": float(result["transform_application_seconds"])
            + float(result["rmsd_reduction_seconds"])
            + float(result["stable_rmsf_seconds"]),
            "memory_movement_proxy": float(result["transform_application_seconds"]),
            "decode": float(result["decode_seconds"]),
        }
        strata = ("all_cases", str(result["source_class"]))
        for category, seconds in categories.items():
            denominator = (
                float(result["raw_source_total_seconds"])
                if category == "decode"
                else total
            )
            for stratum in strata:
                by_stratum.setdefault((stratum, category), []).append(
                    seconds / denominator
                )
    amdahl: list[dict[str, object]] = []
    for (stratum, category), fractions in by_stratum.items():
        median_fraction = float(np.median(fractions))
        for acceleration in (2.0, 5.0, 10.0, float("inf")):
            remaining = 1.0 - median_fraction
            accelerated = 0.0 if not np.isfinite(acceleration) else median_fraction / acceleration
            amdahl.append(
                {
                    "stratum": stratum,
                    "category": category,
                    "median_runtime_fraction": median_fraction,
                    "assumed_stage_speedup": "infinite" if not np.isfinite(acceleration) else acceleration,
                    "maximum_end_to_end_speedup": 1.0 / (remaining + accelerated),
                }
            )
    write_tsv(OUT / "amdahl_bounds.tsv", amdahl)
    median_rows = [
        {
            "stratum": stratum,
            "category": category,
            "median_fraction": float(np.median(values)),
            "n_cases": len(values),
        }
        for (stratum, category), values in sorted(by_stratum.items())
    ]
    write_tsv(OUT / "bottleneck_fractions.tsv", median_rows)
    lines = [
        "# Current Throughput Bottleneck Report",
        "",
        "The frozen `current_throughput_compact` contract was profiled with one CPU thread, no ordinary RMSF, no aligned-coordinate materialisation, and no file serialisation inside the timed algorithm region.",
        "",
        "## Median Algorithm Fractions by Stratum",
        "",
    ]
    for stratum in (
        "all_cases",
        "mdCATH_calibration",
        "ATLAS_R1_real_continuous",
        "transition_aligned_synthetic_scaling_control",
    ):
        values = sorted(
            (
                (row["category"], float(row["median_fraction"]))
                for row in median_rows
                if row["stratum"] == stratum and row["category"] != "decode"
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        lines.append(f"### {stratum}")
        lines.extend(f"- {name}: {fraction:.1%}" for name, fraction in values)
        lines.append("")
    lines.extend(
        [
            "`memory_movement_proxy` is the measured full-coordinate transform-application time and overlaps the all-residue-statistics category; it is not additive.",
            "",
            "Decode is measured against raw-source total and excluded from algorithmic speedup claims. It is especially variable for compressed HDF5 and cold file-cache reads.",
            "",
            "For the 50k/100k controls, explicit per-frame covariance fitting and all-residue statistics dominate. Sparse landmarks can reduce the first component. Keyframes can reduce explicit fits further, but full per-frame RMSD and full stable RMSF retain an O(TN) lower bound. Profile sampling is therefore required for the most aggressive speed targets, but it belongs to the distinct `v2_profile` contract.",
        ]
    )
    (OUT / "current_bottleneck_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all() -> None:
    rows = build_panel()
    results: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    for index, row in enumerate(rows, start=1):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(row["case_id"])]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"TV2 worker failed for {row['case_id']}: {completed.stderr}"
            )
        payload = completed.stdout.strip().splitlines()[-1]
        results.append(json.loads(payload))
        write_tsv(OUT / "memory_timing.tsv", results)
        print(f"[TV2] {index}/{len(rows)} {row['case_id']}", flush=True)
    stages = stage_rows(results)
    write_tsv(OUT / "stage_timing.tsv", stages)
    write_summary(results, stages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args.worker)
    elif args.summarize_only:
        results = read_tsv(OUT / "memory_timing.tsv")
        write_summary(results, [])
    else:
        run_all()


if __name__ == "__main__":
    main()
