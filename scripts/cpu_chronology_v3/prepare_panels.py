#!/usr/bin/env python
"""Prospectively lock calibration and fresh panels for CPU chronology v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3"
D1 = ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv"
V2_PANEL = ROOT / "results/cpu_fast_path_v2/02_real_panel/full_panel.tsv"
V2_CHALLENGE = ROOT / "results/cpu_fast_path_v2/02_real_panel/challenge_panel.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def length_bin(count: int) -> str:
    if count < 100:
        return "lt100"
    if count < 200:
        return "100_199"
    if count < 400:
        return "200_399"
    return "ge400"


def mobility_bin(value: float, cutoffs: tuple[float, float]) -> str:
    if value <= cutoffs[0]:
        return "low"
    if value <= cutoffs[1]:
        return "medium"
    return "high"


def excluded_families() -> set[str]:
    output: set[str] = set()
    for path in (V2_PANEL, V2_CHALLENGE):
        for row in read_tsv(path):
            family = row["family_id"]
            if family.startswith("MDCATH:"):
                output.add(family.split(":", 1)[1])
    return output


def mdcath_pool() -> list[dict[str, object]]:
    excluded = excluded_families()
    rows = []
    mobility_values = []
    for row in read_tsv(D1):
        domain = row["system_id"]
        if domain in excluded or row["status"] != "success":
            continue
        source = ROOT / row["source_file"]
        if not source.exists():
            continue
        if row["mapping_integrity"] != "pass" or row["unit_integrity"] != "pass":
            continue
        mobility = float(row["motion_proxy_noncore_core_RMSF_ratio"])
        mobility_values.append(mobility)
        rows.append(
            {
                "case_id": f"{domain}__T320__R1",
                "family_id": f"MDCATH:{domain}",
                "domain_id": domain,
                "source_class": "mdCATH_320K_R1_real_continuous",
                "input_path": row["source_file"],
                "input_key": row["coordinate_path"],
                "n_frames": int(row["n_frames"]),
                "n_residues": int(row["n_residues"]),
                "cath_class": row["cath_class"],
                "mobility_proxy": mobility,
                "known_tail_status": row["cpu_catastrophic_quality_failure"],
                "coordinate_unit": "angstrom",
                "selection": "embedded_PDB_CA",
                "independent_family": True,
            }
        )
    low, high = np.quantile(np.asarray(mobility_values), [1 / 3, 2 / 3])
    for row in rows:
        row["length_bin"] = length_bin(int(row["n_residues"]))
        row["mobility_bin"] = mobility_bin(float(row["mobility_proxy"]), (float(low), float(high)))
        row["stratum"] = ":".join((str(row["cath_class"]), str(row["length_bin"]), str(row["mobility_bin"])))
        row["selection_hash"] = stable_key(str(row["case_id"]))
    return rows


def stratified_select(pool: list[dict[str, object]], count: int, *, salt: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in pool:
        groups.setdefault(str(row["stratum"]), []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_key(salt + str(row["case_id"])))
    selected: list[dict[str, object]] = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups, key=lambda key: stable_key(salt + key))
    while len(selected) < count:
        progress = False
        for key in keys:
            position = positions[key]
            if position >= len(groups[key]):
                continue
            selected.append(groups[key][position])
            positions[key] += 1
            progress = True
            if len(selected) == count:
                break
        if not progress:
            raise RuntimeError(f"only {len(selected)} eligible mdCATH domains for requested {count}")
    return selected


def legacy_rows() -> list[dict[str, object]]:
    panel = read_tsv(V2_PANEL)
    wanted_local = {"LOCAL_ADK", "LOCAL_NhaA", "LOCAL_I-FABP", "LOCAL_YiiP_90ns"}
    output: list[dict[str, object]] = []
    for row in panel:
        if not (row["case_id"].startswith("ATLAS_") or row["case_id"] in wanted_local):
            continue
        output.append(
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "domain_id": "",
                "source_class": row["source_class"],
                "input_path": row["input_path"],
                "input_key": row["input_key"],
                "n_frames": int(row["n_frames"]),
                "n_residues": int(row["n_residues"]),
                "cath_class": "not_applicable",
                "mobility_proxy": "not_applicable",
                "known_tail_status": "historical_formal",
                "coordinate_unit": row["coordinate_unit"],
                "selection": row["selection"],
                "independent_family": True,
                "length_bin": length_bin(int(row["n_residues"])),
                "mobility_bin": "not_applicable",
                "stratum": "historical_formal",
                "selection_hash": stable_key(row["case_id"]),
            }
        )
    for row in read_tsv(V2_CHALLENGE):
        output.append(
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "domain_id": row["family_id"].split(":", 1)[1],
                "source_class": "v2_registered_challenge_tail",
                "input_path": row["input_path"],
                "input_key": row["input_key"],
                "n_frames": int(row["n_frames"]),
                "n_residues": int(row["n_residues"]),
                "cath_class": "registered_tail",
                "mobility_proxy": "not_applicable",
                "known_tail_status": row["historical_tail_classification"],
                "coordinate_unit": row["coordinate_unit"],
                "selection": row["selection"],
                "independent_family": True,
                "length_bin": length_bin(int(row["n_residues"])),
                "mobility_bin": "registered_tail",
                "stratum": "registered_tail",
                "selection_hash": stable_key(row["case_id"]),
            }
        )
    return output


def lock_calibration() -> None:
    target = OUT / "02_self_variability/calibration_panel.tsv"
    if target.exists():
        raise RuntimeError("calibration panel is already locked")
    selected = stratified_select(mdcath_pool(), 128, salt="cpu_chronology_v3_calibration")
    for row in selected:
        row["panel_role"] = "mdcath_calibration"
    historical = legacy_rows()
    for row in historical:
        row["panel_role"] = (
            "registered_challenge" if row["source_class"] == "v2_registered_challenge_tail" else "historical_formal"
        )
    combined = selected + historical
    if len({str(row["case_id"]) for row in combined}) != len(combined):
        raise RuntimeError("duplicate calibration case")
    write_tsv(target, combined)
    perturbations = [
        ("fresh_process_repeat", "numerical", "same representatives/order/reference in a newly launched Python process"),
        ("chunk_512", "numerical", "same weights; all-frame chunk size 512"),
        ("chunk_4096", "numerical", "same weights; all-frame chunk size 4096"),
        ("cyclic_block_shift", "mild_path", "shift representative order by one 64-frame temporal block"),
        ("interleaved_blocks", "mild_path", "even then odd 64-frame blocks; within-block chronology retained"),
        ("stratified_offset_half", "mild_path", "deterministic half-stride representative offset"),
        ("stratified_offset_full", "mild_path", "second deterministic stratified offset"),
        ("reverse_order", "contextual", "full representative-order reversal"),
        ("reference_middle", "contextual", "middle trajectory frame as reference"),
        ("reference_last", "contextual", "last trajectory frame as reference"),
        ("substantial_subset", "contextual", "deterministic alternate representative subset"),
    ]
    write_tsv(
        OUT / "02_self_variability/perturbation_registry.tsv",
        [{"perturbation": name, "class": group, "definition": definition} for name, group, definition in perturbations],
    )
    report = {
        "locked": True,
        "mdcath_calibration_domains": len(selected),
        "historical_formal": sum(row["panel_role"] == "historical_formal" for row in combined),
        "registered_challenges": sum(row["panel_role"] == "registered_challenge" for row in combined),
        "total_cases": len(combined),
        "panel_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (OUT / "02_self_variability/calibration_panel_lock.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def lock_fresh() -> None:
    gate_hash = OUT / "03_gate_calibration/gate_registry.sha256"
    if not gate_hash.exists():
        raise RuntimeError("v3 gate must be locked before fresh panel selection")
    rule_hash = OUT / "04_candidates/adaptive_rule.sha256"
    if not rule_hash.exists():
        raise RuntimeError("adaptive risk rule must be locked before fresh panel selection")
    target = OUT / "05_fresh_panels/fresh_mdcath_512.tsv"
    if target.exists():
        raise RuntimeError("fresh mdCATH panel is already locked")
    calibration = {
        row["domain_id"] for row in read_tsv(OUT / "02_self_variability/calibration_panel.tsv") if row["domain_id"]
    }
    eligible = [row for row in mdcath_pool() if row["domain_id"] not in calibration]
    selected = stratified_select(eligible, 512, salt="cpu_chronology_v3_fresh_heldout")
    for row in selected:
        row["panel_role"] = "fresh_locked_validation"
        row["gate_registry_sha256"] = gate_hash.read_text().split()[0]
        row["adaptive_rule_sha256"] = rule_hash.read_text().split()[0]
    write_tsv(target, selected)
    report = {
        "locked": True,
        "fresh_domains": len(selected),
        "overlap_calibration": len(calibration & {str(row["domain_id"]) for row in selected}),
        "panel_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "gate_sha256": gate_hash.read_text().split()[0],
        "adaptive_rule_sha256": rule_hash.read_text().split()[0],
    }
    (OUT / "05_fresh_panels/fresh_mdcath_512_lock.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lock-calibration", "lock-fresh"))
    args = parser.parse_args()
    if args.phase == "lock-calibration":
        lock_calibration()
    else:
        lock_fresh()


if __name__ == "__main__":
    main()
