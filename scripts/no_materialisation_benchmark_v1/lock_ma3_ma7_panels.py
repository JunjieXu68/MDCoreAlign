#!/usr/bin/env python3
"""Prospectively lock the reduced MA3-MA7 analysis panels."""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path

from scripts.no_materialisation_benchmark_v1.common import OUT, ROOT, write_text, write_tsv


MA = OUT / "method_agreement_v2"
MANIFESTS = MA / "00_contracts/panel_manifests"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stratified_select(
    rows: list[dict[str, str]],
    target: int,
    required: set[str],
) -> list[dict[str, str]]:
    by_domain = {row["domain_id"]: row for row in rows}
    selected = [by_domain[domain] for domain in sorted(required) if domain in by_domain]
    chosen = {row["domain_id"] for row in selected}
    groups: dict[tuple[str, str, str], deque[dict[str, str]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda value: value["domain_id"]):
        if row["domain_id"] in chosen:
            continue
        key = (row["cath_class"], row["residue_count_bin"], row["frame_count_bin"])
        groups[key].append(row)
    keys = sorted(groups)
    while len(selected) < target:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < target:
                row = groups[key].popleft()
                selected.append(row)
                chosen.add(row["domain_id"])
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise RuntimeError(f"could only select {len(selected)} of {target} domains")
    return selected


def main() -> None:
    all_rows = read_tsv(OUT / "03_mdcath_stock_no_output/mdcath_no_output_all_domain.tsv")
    domains: list[dict[str, str]] = []
    for row in all_rows:
        if row["method"] != "stock_MDLovoFit_no_output" or row["repeat_index"] != "0":
            continue
        domains.append(
            {
                "domain_id": row["system"],
                "trajectory_id": row["prefix_label"],
                "cath_class": row["cath_class"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "residue_count_bin": row["residue_count_bin"],
                "frame_count_bin": row["frame_count_bin"],
            }
        )
    if len(domains) != 5397:
        raise RuntimeError(f"expected 5,397 corrected domains, found {len(domains)}")

    tail_paths = (
        ROOT / "results/database_scale_benchmark_v2/b1_tail_adjudication/b1_tail_case_atlas.tsv",
        ROOT / "results/guarded_fast_global_v1/g1_adjudication/tail_failure_atlas.tsv",
    )
    tails: set[str] = set()
    for path in tail_paths:
        tails.update(row["domain_id"] for row in read_tsv(path))
    primary = stratified_select(domains, 512, tails)
    detailed = stratified_select(primary, 128, tails)

    primary_rows = [
        {
            "selection_index": index,
            **row,
            "registered_tail": row["domain_id"] in tails,
            "selection_rule": "tail_inclusion_then_deterministic_round_robin_over_CATH_class_residue_bin_frame_bin",
        }
        for index, row in enumerate(primary)
    ]
    detailed_rows = [
        {
            "selection_index": index,
            **row,
            "registered_tail": row["domain_id"] in tails,
            "selection_rule": "tail_inclusion_then_deterministic_round_robin_within_locked_512_panel",
        }
        for index, row in enumerate(detailed)
    ]
    write_tsv(MANIFESTS / "mdcath_primary_512.tsv", primary_rows)
    write_tsv(MANIFESTS / "mdcath_detailed_128.tsv", detailed_rows)
    write_tsv(
        MANIFESTS / "mdcath_registered_tails.tsv",
        [{"domain_id": domain, "included_primary": domain in {row["domain_id"] for row in primary}, "included_detailed": domain in {row["domain_id"] for row in detailed}} for domain in sorted(tails)],
    )

    atlas_manifest = read_tsv(ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
    atlas_full = [
        {
            "system": row["system"],
            "prefix_label": "f10001",
            "n_frames": 10001,
            "n_residues": row["n_residues"],
            "input_PDB": f"results/atlas_large_long_v1/a3_external/centered_inputs/{row['system']}/f10001_centered_CA.pdb",
            "selection_rule": "all_eight_frozen_ATLAS_R1_systems_full_length",
        }
        for row in atlas_manifest
    ]
    write_tsv(MANIFESTS / "atlas_full_length_8.tsv", atlas_full)

    quality = [
        row
        for row in read_tsv(ROOT / "results/atlas_large_long_v1/a3_external/external_quality_per_case.tsv")
        if row["method"] == "MDLovoFit_20.1.1" and row["prefix_label"] != "f10001"
    ]
    short_tails = sorted(quality, key=lambda row: (float(row["core_Jaccard_vs_exact"]), row["system"], row["prefix_label"]))[:6]
    write_tsv(
        MANIFESTS / "atlas_short_prefix_tails_6.tsv",
        [
            {
                "selection_index": index,
                "system": row["system"],
                "prefix_label": row["prefix_label"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "frozen_core_Jaccard_vs_exact": row["core_Jaccard_vs_exact"],
                "input_PDB": f"results/atlas_large_long_v1/a3_external/centered_inputs/{row['system']}/{row['prefix_label']}_centered_CA.pdb",
                "selection_rule": "six_lowest_frozen_MDLovoFit_core_Jaccard_non_full_prefixes",
            }
            for index, row in enumerate(short_tails)
        ],
    )
    historical = [
        ("ifabp", "results/reviewer2_hardening/theseus_rerun/inputs/ifabp_ca_multimodel.pdb"),
        ("adk", "results/reviewer2_hardening/theseus_rerun/inputs/adk_ca_multimodel.pdb"),
        ("nhaa", "results/reviewer2_hardening/theseus_rerun/inputs/nhaa_ca_multimodel.pdb"),
        ("yiip_short", "results/reviewer2_hardening/theseus_rerun/inputs/yiip_short_ca_multimodel.pdb"),
    ]
    write_tsv(
        MANIFESTS / "historical_four_systems.tsv",
        [{"system": system, "input_PDB": path, "available": (ROOT / path).is_file()} for system, path in historical],
    )
    write_text(
        MANIFESTS / "panel_lock_report.md",
        f"""# MA3-MA7 Prospective Panel Lock

- Primary mdCATH panel: 512 domains, including {len(tails)} previously registered tail domains.
- Detailed mdCATH panel: 128 domains, preserving all available registered tails.
- ATLAS full-length panel: {len(atlas_full)} frozen R1 systems at 10,001 frames.
- ATLAS short-prefix tail panel: six cases selected by the lowest frozen MDLovoFit-versus-exact core Jaccard before new exporter results.
- Historical panel: four pre-existing CA multi-model PDB inputs.

The 1,024-domain adaptive expansion is not automatic. It may be triggered only
after the locked 512-domain result by the registered confidence-width,
contradictory-strata, divergent-case-count or mapping-uncertainty rules. A full
5,397-domain method-agreement run requires explicit human approval.
""",
    )
    print({"primary": len(primary), "detailed": len(detailed), "tails": len(tails), "atlas_full": len(atlas_full), "atlas_short_tails": len(short_tails), "historical": len(historical)})


if __name__ == "__main__":
    main()
