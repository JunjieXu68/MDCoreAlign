#!/usr/bin/env python3
"""Freeze read-only references and prospectively register stream contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.stream_align_v1.common import (
    ATLAS,
    ATLAS_SYSTEMS,
    OUT,
    ROOT,
    SYNTHETIC,
    atlas_manifest,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


def _reference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    key_files = [
        ("atlas_manifest", ATLAS / "atlas_manifest.tsv", "input provenance and cache hashes"),
        ("atlas_timing", ATLAS / "atlas_matched_prefix_results.tsv", "frozen loaded runtimes"),
        ("atlas_quality", ATLAS / "atlas_quality_summary.tsv", "frozen exact-relative quality"),
        ("atlas_performance", ATLAS / "atlas_performance_summary.tsv", "frozen aggregate performance"),
        (
            "synthetic_scaling",
            ROOT / "results/throughput_gpu_v1/t6_scaling/cpu_gpu_scaling_matrix.tsv",
            "frozen synthetic GPU scaling reference",
        ),
        (
            "synthetic_long_mode",
            SYNTHETIC / "speedup_vs_target_frames.tsv",
            "frozen transition-aligned long-mode timing",
        ),
        (
            "tail_atlas",
            ROOT / "results/database_scale_benchmark_v2/b1_tail_adjudication/b1_tail_case_atlas.tsv",
            "locked difficult-case definitions and frozen quality",
        ),
        (
            "tail_panel",
            ROOT / "results/throughput_gpu_v1/t3_balanced_lite/locked_audit_panel.tsv",
            "locked development/validation and 13 tail labels",
        ),
        (
            "manuscript_alias",
            ROOT / "results/manuscript_writing_bundle/mdcorealign_manuscript_draft.md",
            "declared read-only replacement because results/manuscript_v3 is absent",
        ),
    ]
    for role, path, meaning in key_files:
        rows.append(
            {
                "role": role,
                "path": str(path.relative_to(ROOT)),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "sha256": sha256(path) if path.is_file() else "",
                "access": "read_only_never_rerun",
                "meaning": meaning,
            }
        )
    for system in ATLAS_SYSTEMS:
        cache = ATLAS / f"a3_loaded_analysis/result_cache/{system}__f10001.npz"
        rows.append(
            {
                "role": "atlas_frozen_profiles",
                "path": str(cache.relative_to(ROOT)),
                "exists": cache.is_file(),
                "size_bytes": cache.stat().st_size if cache.is_file() else "",
                "sha256": sha256(cache) if cache.is_file() else "",
                "access": "read_only_never_rerun",
                "meaning": "exact/GPU/W5 weights, RMSD and RMSF profiles; transforms are not stored",
            }
        )
    return rows


def _input_rows() -> list[dict[str, object]]:
    manifest = atlas_manifest()
    rows: list[dict[str, object]] = []
    for system in ATLAS_SYSTEMS:
        row = manifest[system]
        cache = ROOT / row["coordinate_cache_path"]
        metadata = ROOT / row["residue_metadata_path"]
        rows.append(
            {
                "panel": "ATLAS_locked_R1",
                "case_id": system,
                "coordinate_path": row["coordinate_cache_path"],
                "coordinate_exists": cache.is_file(),
                "registered_coordinate_sha256": row["coordinate_cache_sha256"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "metadata_path": row["residue_metadata_path"],
                "metadata_exists": metadata.is_file(),
                "new_data_downloaded": False,
            }
        )
    for path in sorted((SYNTHETIC / "runs").glob("*/100000/transition_aligned_coords.npz")):
        rows.append(
            {
                "panel": "transition_aligned_synthetic_long",
                "case_id": path.parents[1].name,
                "coordinate_path": str(path.relative_to(ROOT)),
                "coordinate_exists": True,
                "registered_coordinate_sha256": "not_rehashed_large_existing_control",
                "n_frames": 100000,
                "n_residues": "read_from_archive",
                "metadata_path": "embedded_in_existing_npz",
                "metadata_exists": True,
                "new_data_downloaded": False,
            }
        )
    return rows


def main() -> None:
    if OUT.exists():
        raise RuntimeError("results/stream_align_v1 already exists; refusing to overwrite")
    OUT.mkdir(parents=True)
    references = _reference_rows()
    inputs = _input_rows()
    missing = [row["path"] for row in references if not row["exists"]]
    missing_inputs = [row["coordinate_path"] for row in inputs if not row["coordinate_exists"]]
    write_tsv(OUT / "s0_freeze/frozen_reference_registry.tsv", references)
    write_tsv(OUT / "s0_freeze/existing_input_registry.tsv", inputs)
    write_tsv(
        OUT / "s0_freeze/frozen_integrity_before.tsv",
        [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in references
            if row["exists"]
        ],
    )
    root_status = [
        {"requested_root": "results/atlas_large_long_v1", "status": "present", "replacement": ""},
        {"requested_root": "results/database_scale_benchmark_v2", "status": "present", "replacement": ""},
        {"requested_root": "results/post_b3_optimisation_v1", "status": "present", "replacement": ""},
        {"requested_root": "results/concept_cases_product_v2", "status": "present", "replacement": ""},
        {
            "requested_root": "results/manuscript_v3",
            "status": "named_root_absent_declared_alias_used",
            "replacement": "results/manuscript_writing_bundle plus later frozen revision assets",
        },
    ]
    write_tsv(OUT / "s0_freeze/requested_root_adjudication.tsv", root_status)
    contracts = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "public_defaults_modified": False,
        "baseline_execution_permitted": False,
        "reference_transform_policy": (
            "ATLAS caches do not store transforms. On deterministic sampled audit frames only, "
            "an independent weighted Kabsch evaluator may reconstruct a reference transform from "
            "frozen exact weights. This evaluation is excluded from new-algorithm runtime."
        ),
        "contracts": {
            "stream-scan": {
                "outputs": ["stable_landmarks", "confidence", "keyframes", "coarse_change_profile"],
                "forbidden": ["full_per_frame_transforms", "full_aligned_RMSF"],
            },
            "stream-profile": {
                "outputs": ["stable_landmarks", "keyframes", "sampled_approximate_RMSF", "approximate_RMSD", "uncertainty"],
                "forbidden": ["full_aligned_trajectory", "exact_label"],
            },
            "stream-align": {
                "outputs": ["approximate_per_frame_transforms", "approximate_RMSD", "sampled_streaming_approximate_RMSF", "interval_audit"],
                "forbidden": ["exact_label", "whole_trajectory_exact_fallback"],
            },
        },
        "balanced_targets": {
            "median_RMSF_Spearman_min": 0.995,
            "median_RMSF_MAE_A_max": 0.10,
            "median_RMSD_difference_A_max": 0.10,
            "P95_RMSD_difference_A_max": 0.50,
            "median_rotation_error_degrees_max": 1.0,
            "ATLAS_large_speedup_min": 2.0,
            "synthetic_100k_speedup_min": 5.0,
        },
        "fast_targets": {
            "median_RMSF_Spearman_min": 0.98,
            "median_RMSD_difference_A_max": 0.25,
            "P95_RMSD_difference_A_max": 1.0,
            "ATLAS_large_speedup_min": 5.0,
            "synthetic_100k_speedup_min": 10.0,
        },
        "fixed_risk_thresholds": {
            "rotation_degrees": 0.75,
            "translation_A": 0.15,
            "coordinate_displacement_A": 0.15,
            "landmark_residual": "median_plus_2.5_MAD",
            "signature_curvature": "P95",
        },
    }
    write_json(OUT / "s1_contracts/registered_stream_contracts.json", contracts)
    write_text(
        OUT / "s1_contracts/registered_stream_contracts.md",
        """# Prospectively Registered Stream Contracts

The three modes are separate. `stream-scan` has no full transforms or RMSF;
`stream-profile` reports sampled approximate RMSF/RMSD; `stream-align` reports
interpolated per-frame transforms plus sampled profile and interval audit.
Approximate outputs must never be labelled exact. The numerical gates and risk
thresholds in the adjacent JSON were frozen before ATLAS stream results were
inspected. No baseline execution is permitted.
""",
    )
    status = {
        "phase": "S0_S1",
        "status": "pass_with_declared_manuscript_root_alias" if not missing and not missing_inputs else "blocked",
        "missing_references": missing,
        "missing_inputs": missing_inputs,
        "named_manuscript_v3_root_exists": (ROOT / "results/manuscript_v3").exists(),
        "declared_alias_exists": (ROOT / "results/manuscript_writing_bundle").exists(),
        "new_downloads": 0,
        "baseline_runs": 0,
    }
    write_json(OUT / "s0_freeze/s0_s1_status.json", status)
    write_text(
        OUT / "s0_freeze/s0_s1_status.md",
        f"""# S0-S1 Status

- Status: `{status['status']}`.
- Frozen reference files missing: {len(missing)}.
- Existing coordinate inputs missing: {len(missing_inputs)}.
- The requested `results/manuscript_v3/` name is absent. The existing frozen
  `results/manuscript_writing_bundle/` and later revision assets are registered
  as a declared read-only alias; the original name is not claimed to exist.
- Frozen ATLAS result caches contain profiles and weights but not transforms.
  Sampled reference transforms may therefore be reconstructed only by the
  registered independent evaluator and are excluded from stream runtime.
- Old algorithms/external tools executed: 0.
- New data downloads: 0.
""",
    )
    if missing or missing_inputs:
        raise RuntimeError(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
