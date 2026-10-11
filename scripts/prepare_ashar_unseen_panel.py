#!/usr/bin/env python3
"""Acquire metadata and prospectively lock the AS5 unseen validation panel."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ashar_private_evaluation_v1/04_unseen_panel"
REGISTRY = (
    ROOT
    / "results/ashar_private_evaluation_v1/03_prior_benchmark_registry/prior_benchmark_registry.tsv"
)
HASH_REGISTRY = (
    ROOT / "results/ashar_private_evaluation_v1/03_prior_benchmark_registry/prior_input_hashes.tsv"
)
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/rest/v1/core/entry/{identifier}"
PDB_URL = "https://files.rcsb.org/download/{identifier}.pdb"
USER_AGENT = "MDCoreAlign-private-unseen-validation/0.1"
RETRIEVAL_DATE = str(date.today())


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    """Fetch one authoritative RCSB JSON response."""

    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": USER_AGENT}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def request_bytes(url: str) -> bytes:
    """Fetch one compact public PDB file."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def registry_sets() -> tuple[set[str], set[str]]:
    """Build case-insensitive prior identifiers and exact known hashes."""

    identifiers: set[str] = set()
    with REGISTRY.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            for field in (
                "system_id",
                "pdb_id",
                "domain_id",
                "trajectory_id",
                "family_label",
            ):
                value = row.get(field, "").strip().lower()
                if value:
                    identifiers.add(value)
    hashes: set[str] = set()
    with HASH_REGISTRY.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["hash_type"] == "input":
                hashes.add(row["sha256"].lower())
    return identifiers, hashes


def search_recent_protein_nmr(rows: int = 600) -> list[str]:
    """Return recent released protein-containing solution-NMR accessions."""

    query: dict[str, object] = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "SOLUTION NMR",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [
                {
                    "sort_by": "rcsb_accession_info.initial_release_date",
                    "direction": "desc",
                }
            ],
        },
    }
    response = request_json(SEARCH_URL, query)
    return [str(item["identifier"]).upper() for item in response["result_set"]]  # type: ignore[index]


def entry_metadata(identifier: str) -> dict[str, object]:
    """Extract only source metadata used for prospective selection."""

    data = request_json(DATA_URL.format(identifier=identifier))
    info = data.get("rcsb_entry_info", {})
    accession = data.get("rcsb_accession_info", {})
    ensemble = data.get("pdbx_nmr_ensemble", {}) or {}
    struct = data.get("struct", {}) or {}
    return {
        "pdb_id": identifier,
        "title": str(struct.get("title", "")),
        "release_date": str(accession.get("initial_release_date", ""))[:10],
        "deposited_models": int(info.get("deposited_model_count", 0) or 0),
        "submitted_models": int(ensemble.get("conformers_submitted_total_number", 0) or 0),
        "deposited_polymer_residues": int(
            info.get("deposited_modeled_polymer_monomer_count", 0) or 0
        ),
        "protein_entity_count": int(info.get("polymer_entity_count_protein", 0) or 0),
        "polymer_instance_count": int(
            info.get("deposited_polymer_entity_instance_count", 0) or 0
        ),
        "source_metadata_url": DATA_URL.format(identifier=identifier),
    }


def parse_ca_models(data: bytes) -> dict[str, object]:
    """Validate CA correspondence across models without scientific analysis."""

    models: list[list[tuple[str, str, str, str]]] = []
    current: list[tuple[str, str, str, str]] = []
    saw_model = False
    for raw in data.decode("ascii", errors="replace").splitlines():
        record = raw[:6].strip()
        if record == "MODEL":
            saw_model = True
            current = []
        elif record == "ENDMDL":
            if current:
                models.append(current)
            current = []
        elif record == "ATOM" and raw[12:16].strip() == "CA" and raw[16:17] in {" ", "A"}:
            current.append((raw[21:22].strip() or "_", raw[22:26].strip(), raw[26:27], raw[17:20].strip()))
    if not saw_model and current:
        models.append(current)
    if saw_model and current:
        models.append(current)
    if not models:
        raise ValueError("no protein CA atoms found")
    first = models[0]
    consistent = all(model == first for model in models)
    return {
        "n_models_pdb": len(models),
        "n_ca": len(first),
        "n_chains": len({item[0] for item in first}),
        "ca_order_consistent": consistent,
        "chain_ids": ",".join(sorted({item[0] for item in first})),
    }


def matches_slot(row: dict[str, object], slot: str) -> bool:
    """Evaluate one prospectively specified metadata-only slot."""

    n_ca = int(row["n_ca"])
    chains = int(row["n_chains"])
    models = int(row["n_models_pdb"])
    if slot == "small_single_high_model":
        return n_ca < 150 and chains == 1 and models >= 20
    if slot == "small_multichain":
        return n_ca < 150 and chains >= 2 and models >= 10
    if slot == "medium_single":
        return 150 <= n_ca <= 500 and chains == 1 and models >= 10
    if slot == "medium_any":
        return 150 <= n_ca <= 500 and models >= 10
    if slot == "multichain_any":
        return chains >= 2 and models >= 10
    if slot == "high_model_any":
        return models >= 30
    raise ValueError(slot)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write a deterministic TSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Acquire compact candidates and seal a six-case metadata-only lock."""

    OUT.mkdir(parents=True, exist_ok=True)
    input_dir = OUT / "acquired_candidates"
    input_dir.mkdir(parents=True, exist_ok=True)
    prior_ids, prior_hashes = registry_sets()
    identifiers = search_recent_protein_nmr()

    metadata: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(entry_metadata, identifier): identifier for identifier in identifiers}
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                item = future.result()
            except Exception as error:
                metadata.append({"pdb_id": identifier, "metadata_error": str(error)})
                continue
            item["exact_identifier_prior_match"] = str(identifier.lower() in prior_ids)
            metadata.append(item)
    metadata.sort(key=lambda item: (str(item.get("release_date", "")), str(item["pdb_id"])), reverse=True)

    eligible_metadata = [
        item
        for item in metadata
        if not item.get("metadata_error")
        and item["exact_identifier_prior_match"] == "False"
        and int(item["deposited_models"]) >= 10
        and 30 <= int(item["deposited_polymer_residues"]) <= 600
    ]

    candidate_rows: list[dict[str, object]] = []
    acquisition_rows: list[dict[str, object]] = []
    slots = [
        ("small_single_high_model", "multi-model PDB"),
        ("small_multichain", "PDB + XTC"),
        ("medium_single", "GRO + XTC"),
        ("medium_any", "multi-model PDB"),
        ("multichain_any", "PDB + XTC"),
        ("high_model_any", "multi-model PDB"),
    ]
    locked: list[dict[str, object]] = []
    reserves: list[dict[str, object]] = []
    attempted = 0
    for item in eligible_metadata:
        if attempted >= 100:
            break
        attempted += 1
        identifier = str(item["pdb_id"])
        url = PDB_URL.format(identifier=identifier)
        row = dict(item)
        try:
            data = request_bytes(url)
            digest = sha256_bytes(data)
            parsed = parse_ca_models(data)
            row.update(parsed)
            row["input_sha256"] = digest
            row["hash_prior_match"] = str(digest.lower() in prior_hashes)
            row["download_status"] = "success"
            row["file_size_bytes"] = len(data)
            path = input_dir / f"{identifier}.pdb"
            path.write_bytes(data)
            row["local_path"] = path.relative_to(ROOT).as_posix()
            row["eligible"] = str(
                bool(parsed["ca_order_consistent"])
                and int(parsed["n_models_pdb"]) >= 10
                and int(parsed["n_ca"]) >= 30
                and digest.lower() not in prior_hashes
            )
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            row.update(
                {
                    "input_sha256": "",
                    "hash_prior_match": "unknown",
                    "download_status": "failed",
                    "download_error": str(error),
                    "eligible": "False",
                }
            )
        candidate_rows.append(row)
        acquisition_rows.append(
            {
                "pdb_id": identifier,
                "source_url": url,
                "source_metadata_url": row.get("source_metadata_url", ""),
                "retrieval_date": RETRIEVAL_DATE,
                "original_filename": f"{identifier}.pdb",
                "sha256": row.get("input_sha256", ""),
                "usage_licence_note": "RCSB PDB public archive; wwPDB usage policy applies; accession provenance retained.",
                "frame_model_count": row.get("n_models_pdb", ""),
                "ca_residue_count": row.get("n_ca", ""),
                "chain_count": row.get("n_chains", ""),
                "topology_trajectory_pairing": "single experimental multi-model PDB; derived topology/trajectory representations created only after lock",
                "download_status": row.get("download_status", ""),
                "local_path": row.get("local_path", ""),
            }
        )

        if row.get("eligible") != "True":
            continue
        selected_ids = {str(selected["pdb_id"]) for selected in locked}
        for slot, representation in slots:
            if any(selected.get("selection_slot") == slot for selected in locked):
                continue
            if identifier in selected_ids:
                break
            if matches_slot(row, slot):
                chosen = dict(row)
                chosen["selection_slot"] = slot
                chosen["planned_input_representation"] = representation
                chosen["selection_basis"] = "release metadata, size, model count, chain count, consistency, registry absence"
                locked.append(chosen)
                break
        if len(locked) == len(slots) and len(reserves) < 2 and identifier not in {
            str(selected["pdb_id"]) for selected in locked
        }:
            reserve = dict(row)
            reserve["selection_slot"] = f"reserve_{len(reserves) + 1}"
            reserve["planned_input_representation"] = "multi-model PDB"
            reserve["selection_basis"] = "next eligible metadata-only candidate after all primary slots were filled"
            reserves.append(reserve)
        if len(locked) == len(slots) and len(reserves) == 2:
            break

    if len(locked) != 6:
        raise SystemExit(f"Could not fill all six prospective slots; filled {len(locked)}")

    common_fields = [
        "pdb_id",
        "title",
        "release_date",
        "selection_slot",
        "planned_input_representation",
        "deposited_models",
        "n_models_pdb",
        "deposited_polymer_residues",
        "n_ca",
        "n_chains",
        "chain_ids",
        "ca_order_consistent",
        "input_sha256",
        "exact_identifier_prior_match",
        "hash_prior_match",
        "family_overlap_status",
        "source_metadata_url",
        "local_path",
        "selection_basis",
    ]
    for row in locked + reserves:
        row["family_overlap_status"] = "no explicit prior same-PDB family; universal sequence-family novelty not claimed"
    candidate_fields = sorted({key for row in candidate_rows for key in row})
    write_tsv(OUT / "candidate_pool.tsv", candidate_rows, candidate_fields)
    write_tsv(OUT / "locked_unseen_panel.tsv", locked, common_fields)
    write_tsv(OUT / "reserve_panel.tsv", reserves, common_fields)
    write_tsv(
        OUT / "acquisition_manifest.tsv",
        acquisition_rows,
        [
            "pdb_id",
            "source_url",
            "source_metadata_url",
            "retrieval_date",
            "original_filename",
            "sha256",
            "usage_licence_note",
            "frame_model_count",
            "ca_residue_count",
            "chain_count",
            "topology_trajectory_pairing",
            "download_status",
            "local_path",
        ],
    )
    locked_path = OUT / "locked_unseen_panel.tsv"
    lock_digest = hashlib.sha256(locked_path.read_bytes()).hexdigest()
    (OUT / "panel_lock.sha256").write_text(f"{lock_digest}  locked_unseen_panel.tsv\n")
    categories = ", ".join(
        f"{row['pdb_id']} ({row['selection_slot']}; {row['planned_input_representation']})"
        for row in locked
    )
    report = f"""# AS5 Prospective Unseen-Panel Lock

- Lock date: {RETRIEVAL_DATE}
- Lock SHA-256: `{lock_digest}`
- Primary cases: 6
- Reserve cases: {len(reserves)}
- Candidate downloads attempted before lock: {len(acquisition_rows)}
- Selection inputs: RCSB source metadata, model/CA/chain counts, CA correspondence,
  intended format, and prior-registry identifier/hash status only.
- Scientific outputs inspected before lock: none.

## Locked Cases

{categories}

Every case is absent by exact PDB identifier and known coordinate-file hash from
the AS4 registry. The panel covers experimental multi-model PDB, derived PDB+XTC,
and derived GRO+XTC representations; small, medium, multi-chain, and high-model-count
ensembles are represented. These are experimental multi-conformation ensembles,
not time-ordered MD trajectories. Absence from explicitly tracked prior families is
documented, but universal sequence-family novelty is not claimed because the prior
mdCATH benchmark is broad.
"""
    (OUT / "panel_lock_report.md").write_text(report)


if __name__ == "__main__":
    main()
