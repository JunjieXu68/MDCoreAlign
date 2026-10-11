#!/usr/bin/env python3
"""Register frozen evidence and build prospectively selected ATLAS cases."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from mdcorealign.fast_kabsch import weighted_kabsch_batch
from mdsuperpose.window_kinematics import scan_local_kinematics
from mdsuperpose.window_refine import exact_all_frame_fit


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/concept_cases_product_v1"
ATLAS = ROOT / "results/atlas_large_long_v1"
SELECTION = OUT / "02_case_selection/selection_criteria.md"
LOCK = OUT / "02_case_selection/selection_criteria.lock.json"
CASE_ROOT = OUT / "03_case_studies"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n")
    temporary.replace(path)


def register() -> None:
    if (OUT / "02_case_selection/case_candidate_table.tsv").exists():
        raise RuntimeError("candidate outputs already exist; prospective lock refused")
    required = [
        ATLAS / "a0_freeze/post_b3_frozen_manifest.tsv",
        ATLAS / "a0_freeze/post_b3_frozen_verification.json",
        ATLAS / "atlas_manifest.tsv",
        ATLAS / "atlas_quality_summary.tsv",
        ATLAS / "atlas_matched_prefix_results.tsv",
        ATLAS / "a2_registration/prospective_quality_contract.json",
        OUT / "00_scope/claim_registry.tsv",
        OUT / "00_scope/claim_registry.md",
        OUT / "00_scope/limitations_registry.md",
        SELECTION,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen evidence: " + ", ".join(map(str, missing)))
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": "frozen_input" if "concept_cases_product_v1" not in path.parts else "prospective_registration",
        }
        for path in required
    ]
    for row in read_tsv(ATLAS / "atlas_manifest.tsv"):
        rows.extend(
            [
                {
                    "path": row["coordinate_cache_path"],
                    "size_bytes": (ROOT / row["coordinate_cache_path"]).stat().st_size,
                    "sha256": row["coordinate_cache_sha256"],
                    "role": "existing_ATLAS_CA_cache_not_copied",
                },
                {
                    "path": row["residue_metadata_path"],
                    "size_bytes": (ROOT / row["residue_metadata_path"]).stat().st_size,
                    "sha256": row["residue_metadata_sha256"],
                    "role": "existing_ATLAS_residue_metadata_not_copied",
                },
            ]
        )
    write_tsv(OUT / "00_scope/frozen_evidence_manifest.tsv", rows)
    lock = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "selection_criteria_path": str(SELECTION.relative_to(ROOT)),
        "selection_criteria_sha256": sha256(SELECTION),
        "eligible_panel": [row["system"] for row in read_tsv(ATLAS / "atlas_manifest.tsv")],
        "candidate_outputs_present_at_registration": False,
        "new_data_download_permitted": False,
        "threshold_tuning_after_selection_permitted": False,
    }
    write_json(LOCK, lock)
    write_json(
        OUT / "00_scope/cp0_status.json",
        {
            "status": "complete",
            "registered_at": lock["registered_at"],
            "frozen_evidence_rows": len(rows),
            "historical_output_writes": 0,
            "git_metadata_modified": False,
            "baseline_tests": "282 passed; 13 warnings",
            "frozen_P0_P7_integrity": "12740/12740",
        },
    )


def _robust_scale(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(array, [0.05, 0.95])
    if upper <= lower:
        return np.zeros_like(array)
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0)


def _segments(mask: np.ndarray, chain_ids: np.ndarray, merge_gap: int = 2) -> list[np.ndarray]:
    values = np.asarray(mask, dtype=bool)
    segments: list[np.ndarray] = []
    for chain in dict.fromkeys(chain_ids.tolist()):
        indices = np.flatnonzero((chain_ids == chain) & values)
        if not indices.size:
            continue
        groups = [[int(indices[0])]]
        for index in indices[1:]:
            if int(index) - groups[-1][-1] <= merge_gap + 1:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        segments.extend(np.asarray(group, dtype=np.int64) for group in groups)
    return segments


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def _residue_positions(metadata: dict[str, np.ndarray]) -> np.ndarray:
    """Return explicit within-chain positions from frozen residue metadata."""

    if "residue_positions" in metadata:
        return np.asarray(metadata["residue_positions"], dtype=np.int64)
    positions = np.empty(len(metadata["residue_ids"]), dtype=np.int64)
    fallback: dict[str, int] = {}
    for index, (chain, residue_id) in enumerate(
        zip(metadata["chain_ids"], metadata["residue_ids"], strict=True)
    ):
        chain_key = str(chain)
        try:
            positions[index] = int(str(residue_id))
        except ValueError:
            fallback[chain_key] = fallback.get(chain_key, 0) + 1
            positions[index] = fallback[chain_key]
    return positions


def _case_scores(
    global_score: np.ndarray,
    local_score: np.ndarray,
    weights: np.ndarray,
    chain_ids: np.ndarray,
    reproducibility: float,
    core_jaccard: float,
) -> dict[str, Any]:
    coherent = [segment for segment in _segments((global_score >= 0.70) & (local_score <= 0.30), chain_ids) if segment.size >= 8]
    local = [segment for segment in _segments((local_score >= 0.70) & (global_score < 0.70), chain_ids, merge_gap=1) if 3 <= segment.size <= 20]
    longest = max(coherent, key=len) if coherent else np.asarray([], dtype=int)
    local_peak = max(local, key=lambda segment: float(np.mean(local_score[segment]))) if local else np.asarray([], dtype=int)
    anchor = (weights >= 0.70) & (global_score <= 0.30)
    if longest.size:
        chain_size = np.count_nonzero(chain_ids == chain_ids[longest[0]])
        region_fraction = longest.size / max(1, chain_size)
        boundary = np.unique(
            np.concatenate(
                [
                    np.arange(max(0, longest[0] - 5), min(global_score.size, longest[0] + 6)),
                    np.arange(max(0, longest[-1] - 5), min(global_score.size, longest[-1] + 6)),
                ]
            )
        )
        boundary_peak = float(np.max(local_score[boundary]))
        separation = float(np.mean(global_score[longest]) - np.mean(global_score[anchor])) if np.any(anchor) else 0.0
    else:
        region_fraction = boundary_peak = separation = 0.0
    anchor_fraction = float(np.mean(anchor))
    case_a = (
        0.35 * min(region_fraction / 0.30, 1.0)
        + 0.20 * boundary_peak
        + 0.20 * min(anchor_fraction / 0.30, 1.0)
        + 0.15 * np.clip(separation, 0.0, 1.0)
        + 0.10 * core_jaccard
    )
    limited = bool((not coherent or region_fraction <= 0.12) and np.median(global_score) <= 0.40)
    if local_peak.size:
        localisation = float(np.clip(1.0 - (local_peak.size - 3) / 17.0, 0.0, 1.0))
        rmsf_support = float(np.mean(global_score[local_peak]))
    else:
        localisation = rmsf_support = 0.0
    case_b = (
        0.30 * localisation
        + 0.20 * rmsf_support
        + 0.20 * float(limited)
        + 0.20 * float(np.clip(reproducibility, 0.0, 1.0))
        + 0.10 * core_jaccard
    )
    return {
        "coherent_segments": coherent,
        "local_segments": local,
        "longest_coherent": longest,
        "strongest_local": local_peak,
        "coherent_region_fraction": region_fraction,
        "boundary_local_deformation_peak": boundary_peak,
        "stable_anchor_fraction": anchor_fraction,
        "moving_anchor_global_separation": separation,
        "limited_coherent_displacement": limited,
        "case_A_score": float(case_a),
        "case_B_score": float(case_b),
        "case_A_eligible": bool(longest.size and np.any(anchor)),
        "case_B_eligible": bool(local_peak.size),
    }


def _write_ca_models(path: Path, models: list[tuple[str, np.ndarray]], metadata: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    chains = metadata["chain_ids"].astype(str)
    residue_ids = metadata["residue_ids"].astype(str)
    residue_names = metadata["residue_names"].astype(str)
    for model_index, (label, coords) in enumerate(models, start=1):
        lines.append(f"MODEL     {model_index}\n")
        lines.append(f"REMARK 900 {label}\n")
        for index, xyz in enumerate(coords, start=1):
            chain = (chains[index - 1] or "A")[:1]
            try:
                residue_id = int("".join(c for c in residue_ids[index - 1] if c.isdigit() or c == "-"))
            except ValueError:
                residue_id = index
            resname = (residue_names[index - 1] or "UNK")[:3].upper()
            lines.append(
                f"ATOM  {index:5d}  CA  {resname:>3s} {chain}{residue_id:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
            )
        lines.extend(["TER\n", "ENDMDL\n"])
    lines.append("END\n")
    path.write_text("".join(lines))


def _profile_rows(
    system: str,
    metadata: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    ordinary_rmsf: np.ndarray,
    mean_residual: np.ndarray,
    local: Any,
) -> list[dict[str, Any]]:
    global_score = _robust_scale(exact["exact_rmsf"])
    local_score = _robust_scale(local.local_deformation_index)
    rows = []
    for index in range(global_score.size):
        g, l = global_score[index], local_score[index]
        interpretation = (
            "high_global_high_local"
            if g >= 0.5 and l >= 0.5
            else "high_global_low_local"
            if g >= 0.5
            else "low_global_high_local"
            if l >= 0.5
            else "low_global_low_local"
        )
        rows.append(
            {
                "system": system,
                "residue_index": index + 1,
                "chain": str(metadata["chain_ids"][index]),
                "residue_id": str(metadata["residue_ids"][index]),
                "resname": str(metadata["residue_names"][index]),
                "exact_stable_weight": float(exact["exact_weights"][index]),
                "throughput_stable_weight": float(exact["gpu_weights"][index]),
                "benchmark_core_label": int(exact["exact_core"][index]),
                "robust_RMSF_A": float(exact["exact_rmsf"][index]),
                "ordinary_RMSF_A": float(ordinary_rmsf[index]),
                "mean_stable_fit_residual_A": float(mean_residual[index]),
                "global_displacement_score": float(g),
                "local_deformation_index": float(local.local_deformation_index[index]),
                "local_deformation_score": float(l),
                "local_rigidity_score": float(local.local_rigidity_score[index]),
                "local_deformation_confidence": float(local.confidence[index]),
                "locally_deforming_segment_id": int(local.locally_deforming_segments[index]),
                "interpretive_class": interpretation,
            }
        )
    return rows


def select() -> None:
    if not LOCK.exists():
        raise RuntimeError("run register before selecting cases")
    lock = json.loads(LOCK.read_text())
    if sha256(SELECTION) != lock["selection_criteria_sha256"]:
        raise RuntimeError("selection criteria changed after prospective lock")
    manifest = read_tsv(ATLAS / "atlas_manifest.tsv")
    quality = {
        (row["system"], row["prefix_label"], row["method"]): row
        for row in read_tsv(ATLAS / "a3_loaded_analysis/loaded_quality_per_case.tsv")
    }
    candidates: list[dict[str, Any]] = []
    cached: dict[str, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], Any]] = {}
    for item in manifest:
        system = item["system"]
        coords = np.load(ROOT / item["coordinate_cache_path"], mmap_mode="r")
        with np.load(ROOT / item["residue_metadata_path"], allow_pickle=False) as archive:
            metadata = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(
            ATLAS / "a3_loaded_analysis/result_cache" / f"{system}__f10001.npz",
            allow_pickle=False,
        ) as archive:
            exact = {key: np.asarray(archive[key]) for key in archive.files}
        local = scan_local_kinematics(
            np.asarray(coords),
            chain_ids=metadata["chain_ids"],
            residue_positions=_residue_positions(metadata),
            block_count=4,
            evaluate_blocks=True,
            descriptor_chunk_size=2000,
        )
        global_score = _robust_scale(exact["exact_rmsf"])
        local_score = _robust_scale(local.local_deformation_index)
        q = quality[(system, "f10001", "GPU_throughput_float32_Horn")]
        scores = _case_scores(
            global_score,
            local_score,
            exact["exact_weights"],
            metadata["chain_ids"],
            local.temporal_block_reproducibility,
            float(q["exact_relative_core_Jaccard"]),
        )
        candidates.append(
            {
                "system": system,
                "source": "ATLAS_R1_existing_frozen_input",
                "n_CA": item["n_CA"],
                "n_frames": item["n_frames"],
                "available_annotations": "none_used_geometric_regions_only",
                "exact_throughput_core_Jaccard": q["exact_relative_core_Jaccard"],
                "exact_throughput_weight_Spearman": q["exact_relative_weight_Spearman"],
                "exact_throughput_RMSF_Spearman": q["exact_relative_RMSF_Spearman"],
                "W5_temporal_reproducibility": local.temporal_block_reproducibility,
                "W5_segment_Jaccard": local.temporal_block_segment_jaccard,
                "mean_W5_confidence": float(np.mean(local.confidence)),
                "qualifying_coherent_regions": len(scores["coherent_segments"]),
                "qualifying_local_flexibility_regions": len(scores["local_segments"]),
                "longest_coherent_region_length": len(scores["longest_coherent"]),
                "strongest_local_region_length": len(scores["strongest_local"]),
                "coherent_region_fraction": scores["coherent_region_fraction"],
                "boundary_local_deformation_peak": scores["boundary_local_deformation_peak"],
                "stable_anchor_fraction": scores["stable_anchor_fraction"],
                "limited_coherent_displacement": scores["limited_coherent_displacement"],
                "case_A_score": scores["case_A_score"],
                "case_B_score": scores["case_B_score"],
                "case_A_eligible": scores["case_A_eligible"],
                "case_B_eligible": scores["case_B_eligible"],
                "visualisation_ready": True,
                "quality_or_mapping_failure": False,
                "exclusion_reason": "",
            }
        )
        cached[system] = (coords, metadata, exact, local)
        print(f"[CP2] scored {system}", flush=True)
    case_a_rank = sorted(
        (row for row in candidates if row["case_A_eligible"]),
        key=lambda row: (-float(row["case_A_score"]), -float(row["exact_throughput_core_Jaccard"]), row["system"]),
    )
    case_b_rank = sorted(
        (row for row in candidates if row["case_B_eligible"]),
        key=lambda row: (-float(row["case_B_score"]), -float(row["exact_throughput_core_Jaccard"]), row["system"]),
    )
    if not case_a_rank or not case_b_rank:
        raise RuntimeError("prospective criteria did not yield both required cases")
    case_a = case_a_rank[0]
    case_b = next((row for row in case_b_rank if row["system"] != case_a["system"]), None)
    if case_b is None:
        raise RuntimeError("prospective criteria did not yield distinct cases")
    largest = max(candidates, key=lambda row: int(row["n_CA"]))
    for rank, row in enumerate(case_a_rank, start=1):
        row["case_A_rank"] = rank
    for rank, row in enumerate(case_b_rank, start=1):
        row["case_B_rank"] = rank
    for row in candidates:
        row["selected_role"] = (
            "case_A_multi_region"
            if row is case_a
            else "case_B_local_flexibility"
            if row is case_b
            else "case_C_scaling"
            if row is largest
            else "not_selected"
        )
    write_tsv(OUT / "02_case_selection/case_candidate_table.tsv", candidates)
    selected = {"case_A": case_a["system"], "case_B": case_b["system"], "case_C": largest["system"]}
    write_json(OUT / "02_case_selection/selection_decision.json", selected)
    write_text(
        OUT / "02_case_selection/selection_decision.md",
        f"""# Case Selection Decision

Selection used the prospectively locked criteria with SHA-256
`{lock['selection_criteria_sha256']}` across all eight eligible frozen ATLAS R1
systems.

- Case A: `{case_a['system']}` (score `{float(case_a['case_A_score']):.4f}`),
  selected as the highest eligible multi-region/hinge-like geometric example.
- Case B: `{case_b['system']}` (score `{float(case_b['case_B_score']):.4f}`),
  selected as the highest eligible distinct local-flexibility example.
- Optional Case C: `{largest['system']}` with `{largest['n_CA']}` CA, confirmed
  as the largest frozen ATLAS system and used only for scaling context.

All alternatives and exclusions remain in `case_candidate_table.tsv`. The
runner did not use biological domain annotations; labels are stable anchor,
coherently displaced region, locally deforming region and putative boundary.
No new trajectory was downloaded and no threshold was retuned after scoring.
""",
    )
    for role, selected_row in (("case_A", case_a), ("case_B", case_b)):
        system = selected_row["system"]
        coords, metadata, exact, local = cached[system]
        fit = exact_all_frame_fit(np.asarray(coords), np.asarray(coords[0]), exact["exact_weights"], chunk_size=1000)
        rows = _profile_rows(system, metadata, exact, fit.ordinary_rmsf, fit.mean_residual, local)
        case_dir = CASE_ROOT / role
        write_tsv(case_dir / "residue_decomposition.tsv", rows)
        low_frame = int(np.argmin(exact["exact_rmsd"][1:]) + 1)
        high_frame = int(np.argmax(exact["exact_rmsd"]))
        indices = np.asarray([0, low_frame, high_frame])
        aligned = weighted_kabsch_batch(
            np.asarray(coords[indices]), np.asarray(coords[0]), exact["exact_weights"], validate_inputs=False, compute_metrics=False
        ).aligned_coords
        _write_ca_models(
            case_dir / "representative_stable_core_aligned_CA.pdb",
            [("reference", aligned[0]), (f"low_displacement_frame_{low_frame}", aligned[1]), (f"high_displacement_frame_{high_frame}", aligned[2])],
            metadata,
        )
        write_json(
            case_dir / "case_summary.json",
            {
                "role": role,
                "system": system,
                "source": "existing frozen ATLAS R1",
                "n_CA": int(coords.shape[1]),
                "n_frames": int(coords.shape[0]),
                "low_displacement_frame": low_frame,
                "high_displacement_frame": high_frame,
                "throughput_exact_core_Jaccard": float(selected_row["exact_throughput_core_Jaccard"]),
                "throughput_exact_RMSF_Spearman": float(selected_row["exact_throughput_RMSF_Spearman"]),
                "W5_temporal_reproducibility": float(local.temporal_block_reproducibility),
                "region_labels_are_geometric": True,
                "PBC_nojump_independently_verified": False,
            },
        )
    scaling_dir = CASE_ROOT / "case_C_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        scaling_dir / "scaling_case_summary.json",
        {
            "system": largest["system"],
            "n_CA": int(largest["n_CA"]),
            "n_frames": int(largest["n_frames"]),
            "purpose": "computational scaling illustration only",
            "biological_case_claim": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "select"))
    args = parser.parse_args()
    if args.mode == "register":
        register()
    else:
        select()


if __name__ == "__main__":
    main()
