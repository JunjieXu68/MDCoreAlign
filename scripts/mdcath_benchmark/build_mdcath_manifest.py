#!/usr/bin/env python
"""Build a deterministic mdCATH trajectory manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import (
    HF_REPO_ID,
    LICENSE,
    MANIFEST_COLUMNS,
    MANIFEST_PATH,
    REPLICAS,
    TEMPERATURES_K,
    default_search_roots,
    ensure_result_dirs,
    find_coordinate_datasets,
    find_local_h5_files,
    import_h5py,
    list_hf_h5_files,
    now_utc,
    parse_domain_id_from_h5_path,
    parse_temperature_replica_from_path,
    trajectory_id,
    write_json,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument("--local-root", type=Path, action="append", help="Optional local mdCATH root.")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH, help="Output manifest TSV.")
    parser.add_argument("--limit-domains", type=int, help="Limit number of domains for testing.")
    parser.add_argument(
        "--source",
        choices=("auto", "hf", "local"),
        default="auto",
        help="Domain-file source for manifest construction.",
    )
    return parser.parse_args()


def rows_from_hf(repo_id: str, limit_domains: int | None) -> list[dict[str, object]]:
    """Build manifest rows from the remote file list only."""

    h5_files = list_hf_h5_files(repo_id)
    if limit_domains is not None:
        h5_files = h5_files[:limit_domains]
    rows: list[dict[str, object]] = []
    for h5_file in h5_files:
        domain_id = parse_domain_id_from_h5_path(h5_file)
        for temperature in TEMPERATURES_K:
            for replica in REPLICAS:
                rows.append(
                    base_manifest_row(
                        domain_id=domain_id,
                        temperature=temperature,
                        replica=replica,
                        h5_file=h5_file,
                        hf_path_or_url=f"https://huggingface.co/datasets/{repo_id}/resolve/main/{h5_file}",
                        metadata_source="huggingface_file_listing",
                        coordinate_path=f"/{domain_id}/{temperature}/{replica - 1}/coords",
                        notes="coordinate path assigned from inspected mdCATH schema: /domain/temperature/zero_based_replica/coords",
                    )
                )
    return rows


def rows_from_local(local_roots: list[Path], limit_domains: int | None) -> list[dict[str, object]]:
    """Build manifest rows from local HDF5 files and inspect coordinate paths when possible."""

    h5_files = find_local_h5_files(local_roots)
    if limit_domains is not None:
        h5_files = h5_files[:limit_domains]
    rows: list[dict[str, object]] = []
    h5py = import_h5py()
    for h5_file in h5_files:
        domain_id = parse_domain_id_from_h5_path(h5_file)
        coordinate_candidates: list[dict[str, object]] = []
        try:
            with h5py.File(h5_file, "r") as handle:
                coordinate_candidates = find_coordinate_datasets(handle)
        except Exception:
            coordinate_candidates = []
        if coordinate_candidates:
            for candidate in coordinate_candidates:
                temperature, replica = parse_temperature_replica_from_path(str(candidate["path"]))
                if temperature is None or replica is None:
                    continue
                rows.append(
                    base_manifest_row(
                        domain_id=domain_id,
                        temperature=temperature,
                        replica=replica,
                        h5_file=str(h5_file),
                        local_cache_path=str(h5_file),
                        metadata_source="local_h5_schema_inspection",
                        coordinate_path=str(candidate["path"]),
                        notes="coordinate path inferred from local HDF5 schema",
                    )
                )
        if not any(row["domain_id"] == domain_id for row in rows):
            for temperature in TEMPERATURES_K:
                for replica in REPLICAS:
                    rows.append(
                        base_manifest_row(
                            domain_id=domain_id,
                            temperature=temperature,
                            replica=replica,
                            h5_file=str(h5_file),
                            local_cache_path=str(h5_file),
                            metadata_source="local_file_listing",
                            coordinate_path="schema_pending",
                            notes="local HDF5 found but trajectory coordinate path not inferred",
                        )
                    )
    return rows


def base_manifest_row(
    *,
    domain_id: str,
    temperature: int,
    replica: int,
    h5_file: str,
    metadata_source: str,
    coordinate_path: str,
    notes: str,
    local_cache_path: str = "",
    hf_path_or_url: str = "",
) -> dict[str, object]:
    """Create one manifest row with stable defaults."""

    is_canonical = temperature == 320 and replica == 1
    return {
        "domain_id": domain_id,
        "cath_class": "NA",
        "cath_architecture": "NA",
        "cath_topology": "NA",
        "cath_homologous_superfamily": "NA",
        "cath_s35_or_s20_if_available": "NA",
        "domain_length": "NA",
        "n_atoms": "NA",
        "n_residues": "NA",
        "temperature_K": temperature,
        "replica": replica,
        "trajectory_id": trajectory_id(domain_id, temperature, replica),
        "expected_n_frames": "NA",
        "coordinate_path": coordinate_path,
        "metadata_source": metadata_source,
        "h5_file": h5_file,
        "local_cache_path": local_cache_path,
        "hf_path_or_url": hf_path_or_url,
        "license": LICENSE,
        "mdCATH_version_or_snapshot": "huggingface_main_current",
        "include_in_domain_canonical_benchmark": int(is_canonical),
        "include_in_full_trajectory_benchmark": 1,
        "include_in_external_baseline_subset": int(is_canonical),
        "notes": notes,
    }


def main() -> None:
    """Build the manifest."""

    args = parse_args()
    ensure_result_dirs()
    local_roots = args.local_root if args.local_root else default_search_roots()
    rows: list[dict[str, object]]
    source_used = args.source
    if args.source in {"auto", "local"}:
        rows = rows_from_local(local_roots, args.limit_domains)
        if rows or args.source == "local":
            source_used = "local"
        else:
            rows = rows_from_hf(args.repo_id, args.limit_domains)
            source_used = "hf"
    else:
        rows = rows_from_hf(args.repo_id, args.limit_domains)
    write_tsv(args.output, rows, MANIFEST_COLUMNS)
    write_json(
        args.output.with_suffix(".json"),
        {
            "generated_at": now_utc(),
            "repo_id": args.repo_id,
            "source_used": source_used,
            "n_rows": len(rows),
            "n_domains": len({str(row["domain_id"]) for row in rows}),
            "output": str(args.output),
            "notes": "Manifest construction does not imply data download or trajectory processing.",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
