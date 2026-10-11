#!/usr/bin/env python3
"""Build the AS4 prior-input exclusion registry from frozen result tables."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "ashar_private_evaluation_v1/03_prior_benchmark_registry"
FIELDS = [
    "source",
    "system_id",
    "pdb_id",
    "chain",
    "domain_id",
    "uniprot_id",
    "trajectory_id",
    "replicate",
    "input_hash",
    "sequence_hash",
    "family_label",
    "role",
    "n_frames",
    "n_residues",
    "input_format",
    "source_manifest",
]


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a CSV or TSV table."""

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def first(row: dict[str, str], *names: str) -> str:
    """Return the first populated field from a row."""

    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value and value.upper() not in {"NA", "NONE", "NAN"}:
            return value
    return ""


def pdb_parts(identifier: str) -> tuple[str, str]:
    """Infer a PDB accession and chain only from a syntactically clear identifier."""

    match = re.search(r"(?i)(?:^|[^0-9a-z])([0-9][0-9a-z]{3})[_-]?([a-z0-9])?", identifier)
    if not match:
        match = re.match(r"(?i)^([0-9][0-9a-z]{3})([a-z0-9])?", identifier)
    if not match:
        return "", ""
    return match.group(1).upper(), (match.group(2) or "").upper()


def normalized_row(**values: str) -> dict[str, str]:
    """Create one registry row with stable field order."""

    row = {field: str(values.get(field, "") or "").strip() for field in FIELDS}
    if not row["pdb_id"]:
        row["pdb_id"], inferred_chain = pdb_parts(
            row["domain_id"] or row["system_id"] or row["trajectory_id"]
        )
        if not row["chain"]:
            row["chain"] = inferred_chain
    return row


def add_generic_table(
    rows: list[dict[str, str]],
    path: Path,
    source: str,
    role: str,
) -> None:
    """Add identifiers from a heterogeneous frozen panel table."""

    if not path.exists():
        return
    relative = path.relative_to(ROOT).as_posix()
    for item in read_table(path):
        system_id = first(item, "case_id", "system", "dataset_name", "trajectory_id", "domain_id")
        trajectory_id = first(item, "trajectory_id", "case_id")
        domain_id = first(item, "domain_id")
        family = first(item, "family_id", "family_label")
        if not family:
            family = f"{source}:{domain_id or system_id}"
        input_hash = first(
            item,
            "input_sha256",
            "coordinate_cache_sha256",
            "registered_coordinate_sha256",
            "source_sha256",
            "trajectory_XTC_sha256",
            "PDB_sha256",
        )
        rows.append(
            normalized_row(
                source=source,
                system_id=system_id,
                domain_id=domain_id,
                trajectory_id=trajectory_id,
                replicate=first(item, "replicate", "selected_replica_label_one_based"),
                input_hash=input_hash,
                sequence_hash=first(item, "sequence_hash", "chain_residue_mapping_sha256"),
                family_label=family,
                role=first(item, "panel_role") or role,
                n_frames=first(item, "n_frames", "expected_n_frames"),
                n_residues=first(item, "n_residues", "n_CA", "expected_n_residues"),
                input_format=first(item, "input_format", "source_class"),
                source_manifest=relative,
            )
        )


def add_mdcath_database(rows: list[dict[str, str]]) -> None:
    """Register every locked full-database mdCATH trajectory."""

    path = RESULTS / "database_scale_benchmark_v2/b2_contracts/b3_all_trajectory_manifest_locked.tsv"
    relative = path.relative_to(ROOT).as_posix()
    for item in read_table(path):
        domain = item["domain_id"]
        pdb_id, chain = pdb_parts(domain)
        rows.append(
            normalized_row(
                source="mdCATH full-database deployment",
                system_id=item["trajectory_id"],
                pdb_id=pdb_id,
                chain=chain,
                domain_id=domain,
                trajectory_id=item["trajectory_id"],
                replicate=item["replicate"],
                input_hash=item["source_sha256"],
                sequence_hash=item["chain_residue_mapping_sha256"],
                family_label=f"mdCATH:{domain}",
                role="full database benchmark input",
                n_residues=item["expected_n_residues"],
                input_format="mdCATH HDF5 coordinate dataset",
                source_manifest=relative,
            )
        )


def add_nmr500(rows: list[dict[str, str]]) -> None:
    """Register all NMR identifiers used by the final 500-case batch."""

    path = RESULTS / "nmr_ensemble_batch_optimized_20260330/nmr_ensemble_batch_runs.csv"
    if not path.exists():
        path = RESULTS / "nmr_ensemble_batch_nocap_20260330/nmr_ensemble_batch_runs.csv"
    seen: set[str] = set()
    relative = path.relative_to(ROOT).as_posix()
    for item in read_table(path):
        identifier = item["dataset_name"].upper()
        if identifier in seen:
            continue
        seen.add(identifier)
        metadata = item.get("preprocessing_steps", "")
        values = dict(re.findall(r"([a-zA-Z0-9_]+)=([^|]+)", metadata))
        rows.append(
            normalized_row(
                source="NMR500 structural-ensemble benchmark",
                system_id=identifier,
                pdb_id=identifier,
                family_label=f"PDB:{identifier}",
                role="NMR500 benchmark input",
                n_frames=values.get("n_models_original", "").strip(),
                n_residues=values.get("n_ca_atoms", "").strip(),
                input_format="experimental multi-model PDB",
                source_manifest=relative,
            )
        )


def add_bundled_examples(rows: list[dict[str, str]]) -> None:
    """Register all compact public examples and release fixtures."""

    source = RESULTS / "public_stable_polish_v6/07_artifacts/release_source"
    for base_name in ("examples", "tests"):
        base = source / base_name
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".pdb",
                ".gro",
                ".xtc",
                ".npy",
                ".npz",
            }:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(ROOT).as_posix()
            identifier = path.parent.name
            rows.append(
                normalized_row(
                    source="public release examples and fixtures",
                    system_id=identifier,
                    trajectory_id=path.name,
                    input_hash=digest,
                    family_label=f"bundled:{identifier}",
                    role=base_name.rstrip("s"),
                    input_format=path.suffix.lower().lstrip("."),
                    source_manifest=relative,
                )
            )


def write_table(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    """Write one deterministic TSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build all AS4 registry outputs."""

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    add_mdcath_database(rows)
    add_nmr500(rows)

    named_tables = [
        ("results/atlas_large_long_v1/atlas_manifest.tsv", "ATLAS R1 large-long panel", "ATLAS R1 benchmark input"),
        ("results/gpu_matched_validation_v1/u2_real_panel/real_continuous_panel.tsv", "formal and real-continuous systems", "real trajectory panel"),
        ("results/stream_align_v1/s0_freeze/existing_input_registry.tsv", "stream-alignment prior registry", "prior trajectory input"),
        ("results/cpu_chronology_v3/02_self_variability/calibration_panel.tsv", "chronology calibration", "chronology development"),
        ("results/cpu_chronology_v3/05_fresh_panels/fresh_long_panel.tsv", "chronology ATLAS R2/R3", "chronology held-out"),
        ("results/cpu_chronology_v3/05_fresh_panels/historical_stress_panel.tsv", "chronology challenge", "chronology challenge"),
        ("results/cpu_fast_path_v2/02_real_panel/full_panel.tsv", "CPU fast-path panel", "development held-out and challenge"),
        ("results/throughput_v2_sparse_keyframe/02_current_throughput_audit/calibration_panel.tsv", "Throughput-v2 calibration", "Throughput-v2 development"),
        ("results/throughput_v2_sparse_keyframe/07_fresh_validation/development_panel.tsv", "Throughput-v2 development", "Throughput-v2 development"),
        ("results/throughput_v2_sparse_keyframe/07_fresh_validation/challenge_panel.tsv", "Throughput-v2 challenge", "Throughput-v2 challenge"),
        ("results/no_materialisation_benchmark_v1/method_agreement_v2/00_contracts/panel_manifests/historical_four_systems.tsv", "method agreement four systems", "method-agreement input"),
        ("results/no_materialisation_benchmark_v1/method_agreement_v2/00_contracts/panel_manifests/atlas_full_length_8.tsv", "method agreement ATLAS", "method-agreement input"),
        ("results/paper_ready_20260326/tables/real_structure_pair_summary.csv", "real structure-pair benchmark", "structure-pair input"),
        ("results/final_multi_system_summary_20260603/final_structure_pair_summary.csv", "formal structure-pair benchmark", "structure-pair input"),
    ]
    for raw, source, role in named_tables:
        add_generic_table(rows, ROOT / raw, source, role)

    panel_root = RESULTS / "no_materialisation_benchmark_v1/method_agreement_v2/00_contracts/panel_manifests"
    for path in sorted(panel_root.glob("*.tsv")):
        if path.name in {"historical_four_systems.tsv", "atlas_full_length_8.tsv"}:
            continue
        add_generic_table(rows, path, "method agreement registered panels", "method-agreement input")

    for system, role in (
        ("I-FABP", "formal MD trajectory"),
        ("ADK", "formal MD trajectory"),
        ("NhaA", "formal MD trajectory"),
        ("YiiP_short", "formal MD trajectory"),
        ("3nci_A", "W5 hinge-like case study"),
        ("6sup_A", "W5 local-flexibility case study"),
    ):
        pdb_id, chain = pdb_parts(system)
        rows.append(
            normalized_row(
                source="historical formal and concept-case registry",
                system_id=system,
                pdb_id=pdb_id,
                chain=chain,
                family_label=f"formal:{system}",
                role=role,
                source_manifest="results/concept_cases_product_v2/00_audit/v1_baseline_manifest.tsv",
            )
        )

    add_bundled_examples(rows)

    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field].lower() for field in FIELDS)
        unique[key] = row
    final_rows = sorted(
        unique.values(),
        key=lambda row: (
            row["source"].lower(),
            row["system_id"].lower(),
            row["trajectory_id"].lower(),
            row["role"].lower(),
        ),
    )
    write_table(OUT / "prior_benchmark_registry.tsv", final_rows, FIELDS)

    hash_rows: list[dict[str, str]] = []
    for row in final_rows:
        for kind, value in (("input", row["input_hash"]), ("sequence", row["sequence_hash"])):
            if value:
                hash_rows.append(
                    {
                        "hash_type": kind,
                        "sha256": value,
                        "system_id": row["system_id"],
                        "trajectory_id": row["trajectory_id"],
                        "source": row["source"],
                    }
                )
    hash_rows = sorted(
        {tuple(row.values()): row for row in hash_rows}.values(),
        key=lambda row: (row["sha256"], row["system_id"]),
    )
    write_table(
        OUT / "prior_input_hashes.tsv",
        hash_rows,
        ["hash_type", "sha256", "system_id", "trajectory_id", "source"],
    )

    family_counts = Counter(row["family_label"] for row in final_rows if row["family_label"])
    family_rows = [
        {
            "family_label": family,
            "registered_input_count": count,
            "scope_note": "Explicitly tracked prior family; not a universal sequence-family claim.",
        }
        for family, count in sorted(family_counts.items())
    ]
    write_table(
        OUT / "prior_system_families.tsv",
        family_rows,
        ["family_label", "registered_input_count", "scope_note"],
    )

    source_counts = Counter(row["source"] for row in final_rows)
    report = [
        "# AS4 Prior-Benchmark Exclusion Registry",
        "",
        f"- Registry rows: {len(final_rows):,}",
        f"- Distinct non-empty input hashes: {len({row['sha256'] for row in hash_rows if row['hash_type'] == 'input'}):,}",
        f"- Explicit family labels: {len(family_rows):,}",
        "- Construction rule: frozen manifests and result tables only; no bulk re-hashing of raw trajectories.",
        "",
        "## Source Coverage",
        "",
    ]
    report.extend(f"- {source}: {count:,} rows" for source, count in sorted(source_counts.items()))
    report.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "Every candidate must be absent by exact identifier and known input hash. "
            "The registry also blocks bundled examples and explicitly tracked system families. "
            "Because mdCATH spans 5,397 domains and 134,950 trajectories, absence from named "
            "families is not interpreted as universal sequence-family novelty. Missing hashes in "
            "legacy NMR and structure-pair result tables remain explicitly blank rather than guessed.",
        ]
    )
    (OUT / "registry_build_report.md").write_text("\n".join(report) + "\n")
    summary = {
        "registry_rows": len(final_rows),
        "input_hash_rows": len(hash_rows),
        "family_rows": len(family_rows),
        "sources": dict(sorted(source_counts.items())),
    }
    (OUT / "registry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
