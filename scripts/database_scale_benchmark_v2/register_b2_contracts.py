#!/usr/bin/env python3
"""Prospectively register B2 contracts, gates, and the full mdCATH manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

from scripts.database_scale_benchmark_v2.common import (
    B1_ROOT,
    B2_ROOT,
    ROOT,
    TAIL_DOMAINS,
    read_tsv,
    sha256,
    write_json,
    write_tsv,
)


TEMPERATURES = (320, 348, 379, 413, 450)
REPLICATES = (1, 2, 3, 4, 5)


def stable_order(value: str) -> str:
    """Return a deterministic pseudo-random ordering key."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_manifest() -> list[dict[str, object]]:
    """Join the official 134,950-row manifest to corrected D0 provenance."""

    official_path = ROOT / "results/mdcath_benchmark/manifest/mdcath_trajectory_manifest.tsv"
    official = read_tsv(official_path)
    if len(official) != 134_950:
        raise RuntimeError(f"official manifest has {len(official):,}/134,950 rows")
    registry = {
        row["system_id"]: row
        for row in read_tsv(
            ROOT / "results/database_scale_gpu_v1/d0_freeze/benchmark_registry.tsv"
        )
    }
    rows: list[dict[str, object]] = []
    for source in official:
        domain = source["domain_id"]
        provenance = registry[domain]
        temperature = int(source["temperature_K"])
        replicate = int(source["replica"])
        if temperature not in TEMPERATURES or replicate not in REPLICATES:
            raise RuntimeError(f"unexpected condition: {source['trajectory_id']}")
        locally_available = provenance["local_available"].lower() == "true"
        rows.append(
            {
                "domain_id": domain,
                "trajectory_id": source["trajectory_id"],
                "temperature_K": temperature,
                "replicate": replicate,
                "coordinate_path": source["coordinate_path"],
                "source_file": provenance["source_file"],
                "source_sha256": provenance["source_sha256"],
                "topology_sha256": provenance["topology_sha256"],
                "chain_residue_mapping_sha256": provenance[
                    "chain_residue_mapping_sha256"
                ],
                "cath_class": provenance["cath_class"],
                "cath_architecture": provenance["cath_architecture"],
                "cath_topology": provenance["cath_topology"],
                "cath_homologous_superfamily": provenance[
                    "cath_homologous_superfamily"
                ],
                "expected_n_residues": provenance["n_residues"],
                "canonical_coordinate_unit": "angstrom",
                "continuous_status": "independent_mdCATH_trajectory_not_stitched",
                "locally_available": locally_available,
                "unavailable_reason": (
                    "" if locally_available else provenance["unavailable_reason"]
                ),
                "shared_input_contract": "one decoded CA float64 coordinate array reused by all methods",
                "method_output_contract": "metrics_transforms_RMSD_RMSF_plus_W5_local_diagnostic",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["domain_id"]),
            int(row["temperature_K"]),
            int(row["replicate"]),
        ),
    )


def lock_exact_pilot(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Lock a 2,000-trajectory pilot before B3 performance inspection."""

    available = [row for row in rows if row["locally_available"]]
    tail = [row for row in available if row["domain_id"] in TAIL_DOMAINS]
    chosen = {str(row["trajectory_id"]): row for row in tail}
    strata: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in available:
        key = (row["cath_class"], row["temperature_K"], row["replicate"])
        strata.setdefault(key, []).append(row)
    ordered_strata = sorted(strata)
    cursor = 0
    while len(chosen) < 2_000:
        key = ordered_strata[cursor % len(ordered_strata)]
        candidates = sorted(
            strata[key], key=lambda row: stable_order(str(row["trajectory_id"]))
        )
        index = cursor // len(ordered_strata)
        if index < len(candidates):
            row = candidates[index]
            chosen.setdefault(str(row["trajectory_id"]), row)
        cursor += 1
        if cursor > len(available) * 2:
            raise RuntimeError("unable to fill exact pilot")
    return [
        {
            "trajectory_id": row["trajectory_id"],
            "domain_id": row["domain_id"],
            "temperature_K": row["temperature_K"],
            "replicate": row["replicate"],
            "cath_class": row["cath_class"],
            "selection_reason": (
                "all_D1_tail_conditions"
                if row["domain_id"] in TAIL_DOMAINS
                else "deterministic_CATH_temperature_replicate_stratification"
            ),
        }
        for row in sorted(chosen.values(), key=lambda item: str(item["trajectory_id"]))[:2_000]
    ]


def main() -> None:
    """Write prospective contract and manifest locks."""

    b1 = json.loads((B1_ROOT / "b1_status.json").read_text())
    if b1["status"] != "pass" or not b1["B2_authorized"]:
        raise RuntimeError("B1 integrity gate did not authorize B2")
    contracts = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "registered_before_B3": True,
        "public_default": "sequential_lean",
        "public_default_changed": False,
        "contract_A_exact_global_analysis": {
            "method": "CPU sequential_lean",
            "representatives": "all frames for mdCATH trajectories up to 500 frames",
            "robust_rule": "Huber delta=1.0",
            "maximum_iterations": 20,
            "sequence_continuity_strength": 0.05,
            "contact_consistency_strength": 0.0,
            "outputs": ["weights", "core labels", "transforms", "RMSD", "RMSF", "flexible segments", "diagnostics"],
            "purpose": "reference, audit, and critical-system analysis",
        },
        "contract_B_throughput_global_analysis": {
            "methods": ["CPU throughput", "GPU throughput", "backend auto"],
            "representative_count": 256,
            "global_update_rounds": 4,
            "reference": "first frame",
            "outputs": ["weights", "core labels", "transforms", "post-hoc RMSD", "RMSF", "flexible segments", "quality metadata"],
            "transform_identical_to_exact_required": False,
            "status": "experimental throughput contract",
        },
        "contract_C_local_deformation": {
            "methods": ["CPU W5", "GPU W5"],
            "descriptor": "chain-aware W5 relative-MAD squared pair-distance instability",
            "outputs": ["local_deformation_index", "local_rigidity_score", "deforming segments"],
            "called_RMSD": False,
            "MDLovoFit_equivalent_alignment": False,
            "hard_motion_domains_mature": False,
        },
        "shared_input_boundary": {
            "coordinate_array": "same decoded ordered CA float64 array",
            "unit": "angstrom",
            "reference": "frame 0",
            "frame_order": "source order",
            "decode_time_counted_once": True,
            "storage_format_comparison": False,
        },
    }
    quality_gates = {
        "registered_before_B3": True,
        "post_hoc_threshold_tuning_allowed": False,
        "valid_input_data_mapping_numerical_failures_max": 0,
        "CPU_GPU_core_jaccard_required": 1.0,
        "CPU_GPU_numerical_tie_policy": "retain and report any mismatch; no silent relabeling",
        "catastrophic_exact_relative_failure_fraction_max": 0.001,
        "catastrophic_definition": [
            "non_finite_output",
            "invalid_transform",
            "empty_or_effective_core_fraction_below_0.10",
            "gross_mapping_failure",
            "core_jaccard_below_0.5_on_exact_audit",
            "RMSF_spearman_below_0.9_on_exact_audit",
        ],
        "warning_frequency_must_be_reported": True,
        "median_exact_relative_core_jaccard_min": 0.95,
        "median_exact_relative_RMSF_spearman_min": 0.995,
        "non_catastrophic_warning_blocks_database_run": False,
        "systematic_integrity_failure_blocks_database_run": True,
    }
    auto_policy = {
        "registered_before_B3": True,
        "source_evidence": "results/gpu_matched_validation_v1/u3_scaling/cpu_gpu_crossover_summary.tsv",
        "warm_persistent_GPU_required": True,
        "decision_rule": "GPU if frames_times_residues >= 150000, otherwise CPU",
        "crossover_residue_frames": 150_000,
        "transparent_rule_only": True,
        "all_B3_trajectories_still_run_on_both_CPU_and_GPU_for_policy_validation": True,
    }
    rows = build_manifest()
    pilot = lock_exact_pilot(rows)
    write_json(B2_ROOT / "registered_contracts.json", contracts)
    write_json(B2_ROOT / "prospective_quality_gates.json", quality_gates)
    write_json(B2_ROOT / "backend_auto_policy.json", auto_policy)
    write_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv", rows)
    write_tsv(B2_ROOT / "b3_exact_pilot_2000_locked.tsv", pilot)
    status = {
        "phase": "B2",
        "status": "pass",
        "registered_at": contracts["registered_at"],
        "full_manifest_rows": len(rows),
        "available_trajectory_rows": sum(bool(row["locally_available"]) for row in rows),
        "unavailable_trajectory_rows": sum(not bool(row["locally_available"]) for row in rows),
        "exact_pilot_rows": len(pilot),
        "manifest_sha256": sha256(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv"),
        "pilot_sha256": sha256(B2_ROOT / "b3_exact_pilot_2000_locked.tsv"),
        "B3_pilot_authorized": True,
        "B3_full_authorized": False,
    }
    write_json(B2_ROOT / "b2_status.json", status)
    (B2_ROOT / "b2_contract_registration.md").write_text(
        "# B2 prospective benchmark contracts\n\n"
        "- Registration status: **PASS**.\n"
        "- Registration occurred before B3 execution: **yes**.\n"
        f"- Locked full manifest: **{len(rows):,} trajectories**.\n"
        f"- Locally available: **{status['available_trajectory_rows']:,}**.\n"
        f"- Explicitly unavailable: **{status['unavailable_trajectory_rows']:,}** (25 conditions for `2mh3A00`).\n"
        f"- Exact-cost pilot: **{len(pilot):,} trajectories**.\n"
        "- Auto crossover: **150,000 residue-frames**, warm persistent GPU required.\n"
        "- Non-catastrophic warnings are recorded and do not stop deployment.\n"
        "- B3 pilot is authorized; the full run requires the prospective pilot decision.\n"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
