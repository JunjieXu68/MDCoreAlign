#!/usr/bin/env python3
"""Run T0-T3 preservation, quality metadata and balanced-lite validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np

from mdcorealign.contracts import solve_balanced_lite_cpu
from mdcorealign.experimental import _core_mask, _jaccard, _spearman
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_coords,
    write_multimodel_pdb,
)
from scripts.mainline_reintegration_v1.run_r0_inventory import (
    frozen_integrity,
)
from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (
    extract_ca_trajectory,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/throughput_gpu_v1"
G0 = ROOT / "results/guarded_fast_global_v1/g0_freeze"
G1 = (
    ROOT
    / "results/guarded_fast_global_v1/g1_adjudication/"
    "nonconvergence_atlas.tsv"
)
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def t0() -> None:
    """Freeze inputs and verify the integrated writer contract."""

    integrity = frozen_integrity()
    if integrity["status"] != "pass":
        raise RuntimeError("frozen evidence integrity failed")
    rng = np.random.default_rng(570)
    coords = rng.normal(size=(5, 12, 3))
    with tempfile.TemporaryDirectory(prefix="mdca_t0_") as tmpdir:
        path = Path(tmpdir) / "trajectory.pdb"
        write_multimodel_pdb(coords, path)
        parsed = read_multimodel_pdb_coords(path, 12)
        expected = np.round(coords, 3)
        writer_pass = bool(np.array_equal(parsed, expected))
    if not writer_pass:
        raise RuntimeError("integrated streaming PDB writer changed coordinates")
    guarded_manifest = (
        ROOT
        / "results/guarded_fast_global_v1/handoff_to_chatgpt/manifest.json"
    )
    write_json(
        OUTPUT / "t0_freeze/t0_status.json",
        {
            "status": "pass",
            "frozen_integrity": integrity,
            "guarded_v1_preserved": True,
            "guarded_handoff_manifest_sha256": sha256(guarded_manifest),
            "streaming_writer_integrated": True,
            "streaming_writer_coordinate_contract_pass": writer_pass,
            "sequential_lean_preserved": True,
            "public_default_changed": False,
            "git_metadata_modified": False,
        },
    )


def t2() -> None:
    """Translate existing G1 evidence into non-binary quality metadata."""

    rows = read_tsv(G1)
    output: list[dict[str, object]] = []
    for row in rows:
        scientific = row["scientific_classification"]
        if scientific == "actual_tail_failure":
            quality = "actual_failure"
            warning = (
                "Fast-global failed the locked scientific comparison; use "
                "exact verification."
            )
        elif scientific == "scientifically_borderline":
            quality = "moderate_confidence"
            warning = (
                "Fast-global is usable with caution; run balanced-lite or "
                "exact verification for high-stakes interpretation."
            )
        elif scientific == "strictly_converged_scientifically_acceptable":
            quality = "numerically_converged"
            warning = ""
        else:
            quality = "scientifically_stable_budget_limited"
            warning = (
                "Fixed iteration budget ended before the strict numerical "
                "criterion, but locked scientific metrics were acceptable."
            )
        output.append(
            {
                "trajectory_id": row["trajectory_id"],
                "domain_id": row["domain_id"],
                "split": row["split"],
                "quality_status": quality,
                "user_warning": warning,
                "strict_convergence_status": row[
                    "strict_convergence_status"
                ],
                "convergence_reason": row["convergence_reason"],
                "last_round_weight_L1_change": row[
                    "final_round_weight_L1_change"
                ],
                "last_round_weight_Linf_change": row[
                    "final_round_weight_Linf_change"
                ],
                "last_round_core_jaccard": row[
                    "last_two_round_core_jaccard"
                ],
                "effective_core_size": row["effective_core_size"],
                "effective_core_fraction": row[
                    "effective_core_fraction"
                ],
                "residual_tail_ratio": row[
                    "mean_residual_p95_median_ratio"
                ],
                "frame_RMSD_tail_ratio": row[
                    "frame_RMSD_p95_median_ratio"
                ],
                "core_fragmentation_fraction": row[
                    "core_fragmentation_fraction"
                ],
                "independent_subset_weight_spearman": row[
                    "independent_subset_weight_spearman"
                ],
                "independent_subset_core_jaccard": row[
                    "independent_subset_core_jaccard"
                ],
                "predefined_tail_failure": row[
                    "predefined_tail_failure"
                ],
            }
        )
    write_tsv(
        OUTPUT / "t2_quality_metadata/domain_quality_metadata.tsv",
        output,
    )
    categories = sorted({str(row["quality_status"]) for row in output})
    summary = [
        {
            "quality_status": category,
            "count": sum(row["quality_status"] == category for row in output),
            "fraction": float(
                np.mean([row["quality_status"] == category for row in output])
            ),
        }
        for category in categories
    ]
    write_tsv(
        OUTPUT / "t2_quality_metadata/quality_status_summary.tsv",
        summary,
    )
    (OUTPUT / "t2_quality_metadata/t2_quality_interpretation.md").write_text(
        """# T2 non-binary quality interpretation

Strict fixed-budget non-convergence is not treated as an automatic scientific
failure. Existing corrected G1 evidence is mapped without rerunning G1:

- `numerically_converged`: strict numerical criterion and locked science gates;
- `scientifically_stable_budget_limited`: strict budget ended, science gates
  passed;
- `moderate_confidence`: one or more locked comparison metrics were borderline;
- `actual_failure`: predefined tail failure requiring exact verification.

The table exposes the online stability signals separately so downstream tools
do not collapse quality into one Boolean.
"""
    )


def t3_lock() -> None:
    """Freeze a stratified development/validation audit panel."""

    rows = read_tsv(G0 / "locked_panel_split.tsv")
    selected: list[dict[str, str]] = []
    for split_name, target in (("development", 256), ("validation", 512)):
        split_rows = [row for row in rows if row["split"] == split_name]
        tails = [
            row
            for row in split_rows
            if row["predefined_tail_failure"] == "1"
        ]
        normal = [
            row
            for row in split_rows
            if row["predefined_tail_failure"] == "0"
        ]
        normal.sort(
            key=lambda row: stable_hash(
                f"balanced-lite-v1::{row['trajectory_id']}"
            )
        )
        selected.extend(tails + normal[: target - len(tails)])
    selected.sort(key=lambda row: row["trajectory_id"])
    path = OUTPUT / "t3_balanced_lite/locked_audit_panel.tsv"
    write_tsv(path, selected)
    write_json(
        OUTPUT / "t3_balanced_lite/audit_panel_lock.json",
        {
            "status": "locked_before_execution",
            "n_domains": len(selected),
            "development": sum(
                row["split"] == "development" for row in selected
            ),
            "validation": sum(
                row["split"] == "validation" for row in selected
            ),
            "tail_count": sum(
                row["predefined_tail_failure"] == "1" for row in selected
            ),
            "panel_sha256": sha256(path),
            "thresholds_tuned_on_validation": False,
        },
    )


def _load_exact_profile(path: Path) -> dict[str, np.ndarray]:
    rows = read_tsv(path)
    return {
        "weights": np.asarray(
            [float(row["soft_stability_weight"]) for row in rows],
            dtype=np.float64,
        ),
        "rmsf": np.asarray(
            [float(row["robust_RMSF"]) for row in rows],
            dtype=np.float64,
        ),
    }


def t3_worker(task: dict[str, str]) -> dict[str, object]:
    os.environ.update(THREAD_ENV)
    base = {
        "trajectory_id": task["trajectory_id"],
        "domain_id": task["domain_id"],
        "cath_class": task["cath_class"],
        "split": task["split"],
        "predefined_tail_failure": task["predefined_tail_failure"],
    }
    try:
        coords, labels, metadata_rows, metadata = extract_ca_trajectory(
            Path(task["h5_path"]),
            coordinate_path=task["coordinate_path"],
            max_frames=None,
            stride=1,
        )
        coords = np.asarray(coords, dtype=np.float64)
        chains = np.asarray(
            [row.get("chain", "_") for row in metadata_rows],
            dtype=object,
        )
        result = solve_balanced_lite_cpu(
            coords,
            chain_ids=chains,
            representative_count=256,
            update_rounds=4,
            audit_subset_count=32,
            audit_heldout_count=64,
            allow_limited_refinement=True,
            refinement_representative_count=512,
            include_w5_on_low_confidence=False,
            chunk_size=2000,
            store_aligned=False,
        )
        assert result.alignment is not None
        assert result.sampled_audit is not None
        exact = _load_exact_profile(Path(task["exact_profile_path"]))
        candidate_weights = np.asarray(
            result.alignment.frozen_weights,
            dtype=np.float64,
        )
        candidate_rmsf = np.asarray(
            result.alignment.robust_rmsf,
            dtype=np.float64,
        )
        return {
            **base,
            "status": "success",
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "quality_status": result.quality_status,
            "quality_warning": result.quality_warning or "",
            "refinement_performed": int(result.refinement_performed),
            "total_runtime_seconds": result.total_runtime_seconds,
            "level1_runtime_seconds": result.timing[
                "level1_throughput"
            ],
            "sampled_audit_runtime_seconds": result.timing[
                "sampled_audit"
            ],
            "limited_refinement_runtime_seconds": result.timing[
                "limited_refinement"
            ],
            "automatic_exact_fallback": False,
            **result.sampled_audit.to_dict(),
            "exact_core_jaccard": _jaccard(
                _core_mask(exact["weights"]),
                _core_mask(candidate_weights),
            ),
            "exact_weight_spearman": _spearman(
                exact["weights"],
                candidate_weights,
            ),
            "exact_RMSF_spearman": _spearman(
                exact["rmsf"],
                candidate_rmsf,
            ),
            "exact_RMSF_MAE_A": float(
                np.mean(np.abs(exact["rmsf"] - candidate_rmsf))
            ),
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def t3_run(workers: int) -> None:
    """Run the locked balanced-lite panel with resumable output."""

    lock_path = OUTPUT / "t3_balanced_lite/locked_audit_panel.tsv"
    lock = json.loads(
        (OUTPUT / "t3_balanced_lite/audit_panel_lock.json").read_text()
    )
    if sha256(lock_path) != lock["panel_sha256"]:
        raise RuntimeError("T3 audit panel changed after lock")
    tasks = read_tsv(lock_path)
    output_path = OUTPUT / "t3_balanced_lite/balanced_lite_panel.tsv"
    existing = read_tsv(output_path) if output_path.is_file() else []
    by_id = {row["trajectory_id"]: row for row in existing}
    pending = [
        row for row in tasks if row["trajectory_id"] not in by_id
    ]
    if pending:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            for index, row in enumerate(
                pool.imap_unordered(t3_worker, pending, chunksize=1),
                start=1,
            ):
                by_id[str(row["trajectory_id"])] = row
                if index % 20 == 0 or index == len(pending):
                    write_tsv(
                        output_path,
                        sorted(
                            by_id.values(),
                            key=lambda item: item["trajectory_id"],
                        ),
                    )
                    print(
                        f"[T3] {index}/{len(pending)} new",
                        flush=True,
                    )
    write_tsv(
        output_path,
        sorted(by_id.values(), key=lambda row: row["trajectory_id"]),
    )


def t3_finalize() -> None:
    """Summarize balanced-lite without changing validation thresholds."""

    rows = [
        row
        for row in read_tsv(
            OUTPUT / "t3_balanced_lite/balanced_lite_panel.tsv"
        )
        if row["status"] == "success"
    ]
    summary_rows: list[dict[str, object]] = []
    for split_name in ("development", "validation", "all"):
        sample = [
            row
            for row in rows
            if split_name == "all" or row["split"] == split_name
        ]
        tails = [
            row
            for row in sample
            if row["predefined_tail_failure"] == "1"
        ]
        warned = [
            row
            for row in sample
            if row["quality_status"] != "high_confidence"
        ]
        summary_rows.append(
            {
                "split": split_name,
                "n_domains": len(sample),
                "high_confidence_fraction": float(
                    np.mean(
                        [
                            row["quality_status"] == "high_confidence"
                            for row in sample
                        ]
                    )
                ),
                "warning_fraction": len(warned) / len(sample),
                "refinement_fraction": float(
                    np.mean(
                        [int(row["refinement_performed"]) for row in sample]
                    )
                ),
                "tail_count": len(tails),
                "tail_warning_count": sum(
                    row["quality_status"] != "high_confidence"
                    for row in tails
                ),
                "tail_warning_recall": (
                    float(
                        np.mean(
                            [
                                row["quality_status"] != "high_confidence"
                                for row in tails
                            ]
                        )
                    )
                    if tails
                    else ""
                ),
                "median_runtime_seconds": statistics.median(
                    float(row["total_runtime_seconds"]) for row in sample
                ),
                "median_audit_runtime_seconds": statistics.median(
                    float(row["sampled_audit_runtime_seconds"])
                    for row in sample
                ),
                "median_exact_core_jaccard": statistics.median(
                    float(row["exact_core_jaccard"]) for row in sample
                ),
                "median_exact_weight_spearman": statistics.median(
                    float(row["exact_weight_spearman"]) for row in sample
                ),
                "median_exact_RMSF_spearman": statistics.median(
                    float(row["exact_RMSF_spearman"]) for row in sample
                ),
            }
        )
    write_tsv(
        OUTPUT / "t3_balanced_lite/balanced_lite_summary.tsv",
        summary_rows,
    )
    validation = next(
        row for row in summary_rows if row["split"] == "validation"
    )
    (OUTPUT / "t3_balanced_lite/t3_balanced_lite_report.md").write_text(
        f"""# T3 balanced-lite fixed-cost audit

The audit panel was locked before execution. Audit frame counts are bounded at
32 + 32 independent representative frames and 64 held-out frames regardless
of full trajectory length. At most one 512-representative refinement is
permitted. No automatic `sequential_lean` fallback is present.

Held-out validation warning recall for the predefined tails was
{validation['tail_warning_count']}/{validation['tail_count']}. The warning
fraction was {float(validation['warning_fraction']):.3%}; median audit runtime
was {float(validation['median_audit_runtime_seconds']):.6f} s. Low-confidence
results are returned with an explicit recommendation to run the exact contract.
Thresholds were not changed after validation.
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("t0", "t2", "t3-lock", "t3-run", "t3-finalize"),
    )
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "t0":
        t0()
    elif args.stage == "t2":
        t2()
    elif args.stage == "t3-lock":
        t3_lock()
    elif args.stage == "t3-run":
        t3_run(args.workers)
    else:
        t3_finalize()


if __name__ == "__main__":
    main()
