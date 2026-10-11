#!/usr/bin/env python3
"""Prospectively register the MDLovoFit addendum contracts and panels."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

import numpy as np

from scripts.database_scale_benchmark_v2.common import (
    B2_ROOT,
    M_ROOT,
    ROOT,
    TAIL_DOMAINS,
    read_tsv,
    sha256,
    write_json,
    write_tsv,
)


MDLOVOFIT_FRACTION = 0.60
MDLOVOFIT_TIMEOUT_SECONDS = 120.0
PILOT_SIZE = 2_000
FALLBACK_SIZE = 10_000


def stable_order(value: str) -> str:
    """Return a deterministic pseudo-random ordering key."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def residue_bin(value: int) -> str:
    if value < 100:
        return "lt100"
    if value < 200:
        return "100_199"
    if value < 400:
        return "200_399"
    return "ge400"


def frame_bin(value: int) -> str:
    if value < 450:
        return "lt450"
    if value < 500:
        return "450_499"
    return "ge500"


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def attach_strata(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Attach locked size and flexibility strata from corrected D1 evidence."""

    d1 = {
        row["system_id"]: row
        for row in read_tsv(
            ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv"
        )
    }
    motion_values = np.asarray(
        [
            _float(row, "motion_proxy_noncore_core_RMSF_ratio")
            for row in d1.values()
            if np.isfinite(_float(row, "motion_proxy_noncore_core_RMSF_ratio"))
        ],
        dtype=float,
    )
    q25, q75 = np.quantile(motion_values, [0.25, 0.75])
    attached: list[dict[str, object]] = []
    for source in rows:
        evidence = d1[source["domain_id"]]
        n_residues = int(evidence["n_residues"])
        n_frames = int(evidence["n_frames"])
        motion = _float(evidence, "motion_proxy_noncore_core_RMSF_ratio")
        mobility_bin = (
            "low"
            if motion <= q25
            else "high"
            if motion >= q75
            else "middle"
        )
        attached.append(
            {
                **source,
                "n_residues": n_residues,
                "n_frames": n_frames,
                "residue_count_bin": residue_bin(n_residues),
                "frame_count_bin": frame_bin(n_frames),
                "mobility_proxy": motion,
                "mobility_bin": mobility_bin,
                "mobility_q25": float(q25),
                "mobility_q75": float(q75),
            }
        )
    return attached


def balanced_fill(
    rows: list[dict[str, object]],
    chosen: dict[str, dict[str, object]],
    target: int,
) -> None:
    """Fill a locked panel by cycling through prospective multidimensional strata."""

    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["trajectory_id"] in chosen:
            continue
        key = (
            row["cath_class"],
            row["temperature_K"],
            row["replicate"],
            row["residue_count_bin"],
            row["frame_count_bin"],
            row["mobility_bin"],
        )
        buckets[key].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_order(str(row["trajectory_id"])))
    keys = sorted(buckets, key=lambda key: tuple(map(str, key)))
    positions = defaultdict(int)
    while len(chosen) < target:
        added = 0
        for key in keys:
            position = positions[key]
            if position >= len(buckets[key]):
                continue
            row = buckets[key][position]
            positions[key] += 1
            chosen[str(row["trajectory_id"])] = row
            added += 1
            if len(chosen) >= target:
                break
        if added == 0:
            raise RuntimeError(f"unable to fill locked panel to {target:,} rows")


def compact_panel_row(row: dict[str, object], reason: str) -> dict[str, object]:
    fields = (
        "trajectory_id",
        "domain_id",
        "temperature_K",
        "replicate",
        "cath_class",
        "cath_architecture",
        "cath_topology",
        "cath_homologous_superfamily",
        "coordinate_path",
        "source_file",
        "source_sha256",
        "expected_n_residues",
        "n_residues",
        "n_frames",
        "residue_count_bin",
        "frame_count_bin",
        "mobility_proxy",
        "mobility_bin",
        "canonical_coordinate_unit",
    )
    return {**{field: row[field] for field in fields}, "selection_reason": reason}


def build_locks() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    manifest = [
        row
        for row in read_tsv(B2_ROOT / "b3_all_trajectory_manifest_locked.tsv")
        if row["locally_available"].lower() == "true"
    ]
    rows = attach_strata(manifest)
    all_domain = [
        compact_panel_row(row, "one_corrected_320K_R1_trajectory_per_available_domain")
        for row in rows
        if int(row["temperature_K"]) == 320 and int(row["replicate"]) == 1
    ]
    if len(all_domain) != 5_397:
        raise RuntimeError(f"all-domain lock has {len(all_domain):,}/5,397 rows")

    pilot_chosen = {
        str(row["trajectory_id"]): row
        for row in rows
        if row["domain_id"] in TAIL_DOMAINS
    }
    balanced_fill(rows, pilot_chosen, PILOT_SIZE)
    pilot = [
        compact_panel_row(
            row,
            "all_conditions_for_D1_tail_family"
            if row["domain_id"] in TAIL_DOMAINS
            else "balanced_CATH_temperature_replicate_size_frame_mobility_strata",
        )
        for row in sorted(pilot_chosen.values(), key=lambda item: str(item["trajectory_id"]))
    ]

    fallback_chosen = {
        str(row["trajectory_id"]): row
        for row in rows
        if (int(row["temperature_K"]) == 320 and int(row["replicate"]) == 1)
        or row["domain_id"] in TAIL_DOMAINS
    }
    balanced_fill(rows, fallback_chosen, FALLBACK_SIZE)
    fallback = [
        compact_panel_row(
            row,
            "one_per_available_domain"
            if int(row["temperature_K"]) == 320 and int(row["replicate"]) == 1
            else "all_conditions_for_D1_tail_family"
            if row["domain_id"] in TAIL_DOMAINS
            else "balanced_additional_condition",
        )
        for row in sorted(fallback_chosen.values(), key=lambda item: str(item["trajectory_id"]))
    ]
    return all_domain, pilot, fallback


def main() -> None:
    executable = shutil.which("mdlovofit")
    if executable is None:
        raise FileNotFoundError("MDLovoFit is required by the addendum but was not found")
    version = subprocess.run(
        [executable, "-h"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    all_domain, pilot, fallback = build_locks()
    write_tsv(M_ROOT / "manifests/mdlovofit_all_domain_5397_locked.tsv", all_domain)
    write_tsv(M_ROOT / "manifests/mdlovofit_pilot_2000_locked.tsv", pilot)
    write_tsv(M_ROOT / "manifests/mdlovofit_fallback_10000_locked.tsv", fallback)

    generated = datetime.now(timezone.utc).isoformat()
    policy = {
        "registered_at": generated,
        "registered_before_new_MDLovoFit_results": True,
        "addendum_precedence": True,
        "executable": executable,
        "executable_sha256": sha256(ROOT / executable.removeprefix("/")) if executable.startswith(str(ROOT)) else sha256(__import__("pathlib").Path(executable)),
        "version_banner": next((line.strip() for line in version if "Version" in line), "unavailable"),
        "fraction": MDLOVOFIT_FRACTION,
        "mdcath_timeout_seconds": MDLOVOFIT_TIMEOUT_SECONDS,
        "real_long_timeout_seconds_by_prefix": {
            "250": 120,
            "500": 120,
            "1000": 180,
            "2000": 300,
            "5000": 900,
            "10000": 1800,
        },
        "feasibility_decision_rule": {
            "projected_full_wall_seconds_max": 604800,
            "failure_plus_timeout_fraction_max": 0.01,
            "compact_summary_storage_bytes_max": 10 * 1024**3,
            "peak_temporary_working_storage_bytes_max": 10 * 1024**3,
            "if_pass": "run_all_134925_available_trajectories",
            "if_fail": "run_prospectively_locked_10000_trajectory_panel",
            "post_hoc_panel_reduction_allowed": False,
        },
        "timing_contract": {
            "shared_HDF5_decode": "reported once and excluded from loaded-coordinate method time",
            "method_specific_input_preparation": "canonical CA array to multi-model PDB",
            "MDLovoFit_subprocess": "internal alignment plus mandatory aligned-PDB writing; not separable without modifying MDLovoFit",
            "output_parse": "aligned PDB coordinate and occupancy parsing",
            "posthoc_evaluation": "independent per-frame RMSD to unmodified frame-0 reference; no additional fit",
            "loaded_analysis": "input preparation + mandatory-output subprocess + parse + post-hoc evaluation",
            "materialised_output": "same mandatory-output workflow with aligned PDB produced; output size recorded",
            "persistent_outputs": "compact rows and failure diagnostics only; successful aligned PDBs are not retained at database scale",
        },
        "mask_policy": "no full-scale MDLovoFit core/mobile mask comparison until field semantics are independently verified",
        "historical_R4_role": "validation reference only; does not replace the new M0 timing rerun",
    }
    write_json(M_ROOT / "manifests/mdlovofit_registered_policy.json", policy)
    status = {
        "phase": "M0_registration",
        "status": "pass",
        "registered_at": generated,
        "all_domain_rows": len(all_domain),
        "pilot_rows": len(pilot),
        "fallback_rows": len(fallback),
        "all_domain_manifest_sha256": sha256(M_ROOT / "manifests/mdlovofit_all_domain_5397_locked.tsv"),
        "pilot_manifest_sha256": sha256(M_ROOT / "manifests/mdlovofit_pilot_2000_locked.tsv"),
        "fallback_manifest_sha256": sha256(M_ROOT / "manifests/mdlovofit_fallback_10000_locked.tsv"),
    }
    write_json(M_ROOT / "manifests/m0_registration_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
