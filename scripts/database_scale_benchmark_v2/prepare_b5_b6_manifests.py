#!/usr/bin/env python3
"""Prepare conservative local and ATLAS acquisition manifests for B5/B6."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.database_scale_benchmark_v2.common import ROOT, read_tsv, write_json, write_tsv


OUTPUT_ROOT = ROOT / "results" / "database_scale_benchmark_v2"
B5_ROOT = OUTPUT_ROOT / "b5_real_long"
B6_ROOT = OUTPUT_ROOT / "b6_large_structures"
LOCAL_PANEL = ROOT / "results" / "gpu_matched_validation_v1" / "u2_real_panel" / "real_continuous_panel.tsv"

# Sizes were queried prospectively with an HTTP Range request on 2026-08-03.
# No trajectory archive was downloaded during the B3 production run.
LOCKED_ATLAS_CANDIDATES = (
    ("1u55_A", 188, 1.08, 440_548_382, "small_100_to_200"),
    ("1r6w_A", 322, 1.09, 708_550_020, "medium_300_to_500"),
    ("1jnr_C", 643, 1.06, 1_433_560_492, "medium_large_multidomain"),
    ("3nci_A", 903, 2.12, 2_055_433_216, "large_flexible"),
    ("1tg7_A", 971, 0.79, 2_042_908_008, "large_rigid"),
    ("5vz0_A", 1_144, 6.92, 2_520_171_083, "large_high_mobility"),
    ("6qfk_A", 1_357, 1.60, 2_855_955_017, "large_800_to_1500"),
    ("6sup_A", 2_128, 0.93, 4_684_506_283, "very_large_2000_plus"),
)


def local_rows() -> list[dict[str, object]]:
    rows = []
    for source in read_tsv(LOCAL_PANEL):
        frames = int(source["n_frames"])
        rows.append(
            {
                "trajectory_id": source["trajectory_id"],
                "system": source["system"],
                "source_type": source["source_type"],
                "source_status": source["status"],
                "n_frames": frames,
                "n_residues": int(source["n_residues"]),
                "n_chains": int(source["n_chains"]),
                "coordinate_unit": source["coordinate_unit"],
                "topology_path": source["topology_path"],
                "trajectory_path": source["trajectory_path"],
                "selection": source["selection"],
                "single_source_trajectory_file": source["single_source_trajectory_file"],
                "stitched": source["stitched"],
                "eligible_for_B5_10000_frame_target": frames >= 10_000,
                "availability": "locally_ready",
                "notes": source["notes"],
            }
        )
    return rows


def atlas_rows() -> list[dict[str, object]]:
    base = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein"
    return [
        {
            "trajectory_id": f"ATLAS_{identifier}",
            "system": identifier,
            "source_type": "ATLAS_protein_only_archive",
            "source_status": "prospectively_locked_not_downloaded",
            "n_frames": "10000_per_replicate_expected_verify_after_download",
            "n_residues": residues,
            "n_chains": 1,
            "coordinate_unit": "verify_after_download",
            "topology_path": "inside_archive_verify_after_download",
            "trajectory_path": f"{base}/{identifier}",
            "selection": "protein CA after topology correspondence audit",
            "single_source_trajectory_file": "verify_after_download",
            "stitched": False,
            "eligible_for_B5_10000_frame_target": "candidate_pending_input_audit",
            "availability": "remote_candidate_not_downloaded",
            "archive_size_bytes": archive_size,
            "ATLAS_reported_average_RMSF_A": average_rmsf,
            "selection_stratum": stratum,
            "notes": (
                "Official protein-only endpoint documents 10000-frame trajectories. "
                "Replicate count, files, units, topology mapping, PBC status, and exact frame count "
                "must be verified after acquisition."
            ),
        }
        for identifier, residues, average_rmsf, archive_size, stratum in LOCKED_ATLAS_CANDIDATES
    ]


def main() -> None:
    local = local_rows()
    candidates = atlas_rows()
    write_tsv(B5_ROOT / "b5_available_real_long_manifest.tsv", local)
    write_tsv(B5_ROOT / "b5_atlas_acquisition_candidates_locked.tsv", candidates)

    large = [row for row in candidates if int(row["n_residues"]) >= 800]
    large.append(
        {
            "trajectory_id": "NhaA",
            "system": "NhaA",
            "source_type": "local_single_file_trajectory",
            "source_status": "locally_ready_below_target",
            "n_frames": 5001,
            "n_residues": 752,
            "n_chains": 2,
            "coordinate_unit": "angstrom",
            "selection_stratum": "largest_current_local_CA_case_below_800",
            "notes": "Useful multi-chain control, but it does not satisfy the 800-residue target.",
        }
    )
    write_tsv(B6_ROOT / "b6_large_structure_acquisition_manifest.tsv", large)

    local_eligible = sum(bool(row["eligible_for_B5_10000_frame_target"]) for row in local)
    total_bytes = sum(int(row["archive_size_bytes"]) for row in candidates)
    generated = datetime.now(timezone.utc).isoformat()
    status = {
        "generated_at": generated,
        "local_real_continuous_trajectories": len(local),
        "local_trajectories_at_least_10000_frames": local_eligible,
        "local_maximum_frames": max(int(row["n_frames"]) for row in local),
        "local_maximum_residues": max(int(row["n_residues"]) for row in local),
        "locked_ATLAS_candidates": len(candidates),
        "locked_ATLAS_total_archive_bytes": total_bytes,
        "trajectory_archives_downloaded_in_this_step": 0,
        "B5_target_met_before_acquisition": False,
        "B6_800_to_1500_target_met_before_acquisition": False,
        "B6_2000_plus_target_met_before_acquisition": False,
    }
    write_json(B5_ROOT / "b5_b6_inventory_status.json", status)

    plan = f"""# B5 Real-Continuous Availability and Acquisition Plan

Generated: {generated}

## Current local evidence

- The validated local panel contains {len(local)} trajectories from 12 system families.
- No local trajectory reaches 10,000 frames; the maximum is 9,001 frames.
- The largest local CA selection is 752 residues.
- Existing stitched trajectories are excluded from biological validation.

## Locked ATLAS pilot

Eight protein-only ATLAS archives were selected before acquisition to cover 188–2,128
residues and low-to-high ATLAS-reported average RMSF. The official endpoint documents
10,000-frame protein trajectories. The combined remote archive size is
{total_bytes / 1024**3:.2f} GiB. No archive was downloaded while B3 was running.

After B3 completes, download this locked set only. Before benchmarking, verify archive
contents, replicate identities, topology-coordinate correspondence, CA selection, units,
frame order, PBC/nojump status, and exact frame counts. A failed audit excludes the case;
it does not trigger substitution after results are viewed.

## Remaining gaps

- ATLAS supplies a feasible 800–1,500-residue pilot and one 2,128-residue chain, but the
  requested 5–10 independent 2,000+ residue systems are not available in this release.
- The locked ATLAS entries are single-chain records and do not fill the large multi-chain
  complex gap.
- Larger downloads or a separate multi-chain source require a later acquisition decision.

## Sources

- ATLAS API: `https://www.dsimb.inserm.fr/ATLAS/api/docs#/`
- Protein endpoint: `https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{{pdb_chain}}`
- Metadata release: `https://www.dsimb.inserm.fr/ATLAS/api/parsable`
"""
    (B5_ROOT / "b5_missing_coverage_and_acquisition_plan.md").parent.mkdir(
        parents=True, exist_ok=True
    )
    (B5_ROOT / "b5_missing_coverage_and_acquisition_plan.md").write_text(plan)
    print(status)


if __name__ == "__main__":
    main()
