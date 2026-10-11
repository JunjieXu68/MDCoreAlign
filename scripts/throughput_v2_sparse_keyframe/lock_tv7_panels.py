#!/usr/bin/env python3
"""Prospectively lock development, fresh validation, and challenge panels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/throughput_v2_sparse_keyframe/07_fresh_validation"


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(value: str) -> str:
    return hashlib.sha256(f"throughput-v2-development-v1:{value}".encode()).hexdigest()


def normalize_mdcath(row: dict[str, str], role: str) -> dict[str, object]:
    domain_id = row.get("domain_id", "")
    trajectory_id = row.get("trajectory_id", f"{domain_id}__T320__R1")
    source = row.get("source_file", row.get("input_path", ""))
    path = Path(source)
    try:
        input_path = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    except ValueError:
        input_path = str(path)
    return {
        "case_id": trajectory_id,
        "family_id": f"MDCATH:{domain_id}",
        "domain_id": domain_id,
        "source_class": "mdCATH_320K_R1_real_continuous",
        "input_path": input_path,
        "input_key": row.get("coordinate_path", row.get("input_key", "")),
        "n_frames": row.get("n_frames", ""),
        "n_residues": row.get("n_residues", ""),
        "cath_class": row.get("cath_class", "NA"),
        "mobility_proxy": row.get("mobility_proxy", "NA"),
        "residue_count_bin": row.get("residue_count_bin", row.get("length_bin", "NA")),
        "mobility_bin": row.get("mobility_bin", "NA"),
        "panel_role": role,
        "selection_hash": stable_order(trajectory_id),
    }


def atlas_r1_rows() -> list[dict[str, object]]:
    atlas = read_tsv(ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
    selected = {"ATLAS_1u55_A", "ATLAS_1r6w_A", "ATLAS_1jnr_C", "ATLAS_6sup_A"}
    rows = []
    for row in atlas:
        if row["trajectory_id"] not in selected:
            continue
        rows.append(
            {
                "case_id": row["trajectory_id"],
                "family_id": f"ATLAS:{row['system']}",
                "domain_id": row["system"],
                "source_class": "ATLAS_R1_real_continuous_development",
                "input_path": row["coordinate_cache_path"],
                "input_key": "npy_array",
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "cath_class": "NA",
                "mobility_proxy": "NA",
                "residue_count_bin": "ATLAS_size_span",
                "mobility_bin": "NA",
                "panel_role": "development_ATLAS_R1",
                "selection_hash": stable_order(row["trajectory_id"]),
            }
        )
    return sorted(rows, key=lambda item: int(item["n_residues"]))


def choose_development_mdcath() -> list[dict[str, object]]:
    source = read_tsv(
        ROOT
        / "results/database_scale_benchmark_v2/mdlovofit_comparison/"
        "mdlovofit_all_domain_corrected.tsv"
    )
    fresh = {
        row["case_id"]
        for row in read_tsv(
            ROOT / "results/cpu_chronology_v3/05_fresh_panels/fresh_mdcath_512.tsv"
        )
    }
    candidates = [
        row
        for row in source
        if row.get("status") == "success"
        and row.get("trajectory_id") not in fresh
        and (ROOT / row["source_file"]).is_file()
    ]
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in candidates:
        key = (
            row.get("cath_class", "NA"),
            row.get("residue_count_bin", "NA"),
            row.get("mobility_bin", "NA"),
        )
        strata.setdefault(key, []).append(row)
    for values in strata.values():
        values.sort(key=lambda row: stable_order(row["trajectory_id"]))
    selected: list[dict[str, str]] = []
    keys = sorted(strata)
    position = 0
    while len(selected) < 128:
        made_progress = False
        for key in keys:
            if position < len(strata[key]):
                selected.append(strata[key][position])
                made_progress = True
                if len(selected) == 128:
                    break
        if not made_progress:
            raise RuntimeError("insufficient mdCATH development candidates")
        position += 1
    return [normalize_mdcath(row, "development_mdcath_128") for row in selected]


def challenge_rows(all_development: list[dict[str, object]]) -> list[dict[str, object]]:
    historical = read_tsv(
        ROOT / "results/cpu_chronology_v3/05_fresh_panels/historical_stress_panel.tsv"
    )
    rows: list[dict[str, object]] = []
    for row in historical:
        copied: dict[str, object] = dict(row)
        copied["panel_role"] = "registered_current_throughput_tail"
        copied["challenge_reason"] = row.get("known_tail_status", "historical_stress")
        rows.append(copied)
    small = sorted(
        (
            row for row in all_development if row["source_class"].startswith("mdCATH")
        ),
        key=lambda row: int(row["n_residues"]),
    )[:8]
    for row in small:
        copied = dict(row)
        copied["panel_role"] = "challenge_very_small"
        copied["challenge_reason"] = "sparse_landmark_conditioning_risk"
        rows.append(copied)
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        unique[str(row["case_id"])] = row
    return list(unique.values())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    development = atlas_r1_rows() + choose_development_mdcath()
    write_tsv(OUT / "development_panel.tsv", development)
    fresh_mdcath = read_tsv(
        ROOT / "results/cpu_chronology_v3/05_fresh_panels/fresh_mdcath_512.tsv"
    )
    fresh_atlas = read_tsv(
        ROOT / "results/cpu_chronology_v3/05_fresh_panels/fresh_long_panel.tsv"
    )
    write_tsv(OUT / "fresh_mdcath_512.tsv", fresh_mdcath)
    write_tsv(OUT / "fresh_atlas_r2_r3.tsv", fresh_atlas)
    challenges = challenge_rows(development)
    write_tsv(OUT / "challenge_panel.tsv", challenges)
    registry = {
        "development_panel_sha256": sha256(OUT / "development_panel.tsv"),
        "fresh_mdcath_512_sha256": sha256(OUT / "fresh_mdcath_512.tsv"),
        "fresh_atlas_r2_r3_sha256": sha256(OUT / "fresh_atlas_r2_r3.tsv"),
        "challenge_panel_sha256": sha256(OUT / "challenge_panel.tsv"),
        "development_rows": len(development),
        "fresh_mdcath_rows": len(fresh_mdcath),
        "fresh_atlas_rows": len(fresh_atlas),
        "challenge_rows": len(challenges),
        "fresh_results_inspected_for_candidate_tuning": False,
        "selection_seed": "throughput-v2-development-v1",
    }
    (OUT / "panel_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Throughput-v2 Panel Lock Report

- Development: {len(development)} cases, comprising four ATLAS R1 trajectories and 128 deterministic stratified mdCATH domains.
- Fresh mdCATH: {len(fresh_mdcath)} previously locked domains excluded from development.
- Fresh long: {len(fresh_atlas)} ATLAS R2/R3 trajectories from eight families; these are 16 trajectories, not 16 independent protein families.
- Challenge: {len(challenges)} registered historical tails and very-small conditioning-risk systems. Development-derived conditioning failures may be appended with explicit provenance before candidate locks, but fresh failures may not be added for tuning.

The gate registry was frozen before these fresh tables were copied. Fresh result tables were not read to tune candidate thresholds or configurations.
"""
    (OUT / "panel_lock_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
