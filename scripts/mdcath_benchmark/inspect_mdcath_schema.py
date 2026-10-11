#!/usr/bin/env python
"""Inspect mdCATH availability and HDF5 schema without bulk download."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import (
    HF_REPO_ID,
    LICENSE,
    SCHEMA_REPORT_PATH,
    check_optional_dependencies,
    default_search_roots,
    ensure_result_dirs,
    find_coordinate_datasets,
    find_local_h5_files,
    find_pdb_text,
    find_string_datasets,
    import_h5py,
    list_hf_h5_files,
    now_utc,
    parse_domain_id_from_h5_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-file", type=Path, help="Optional local mdCATH HDF5 file to inspect.")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument("--max-hf-files", type=int, default=20, help="Number of remote files to preview.")
    parser.add_argument("--output", type=Path, default=SCHEMA_REPORT_PATH, help="Markdown report path.")
    return parser.parse_args()


def inspect_h5_file(path: Path) -> dict[str, Any]:
    """Inspect one local HDF5 file."""

    h5py = import_h5py()
    with h5py.File(path, "r") as handle:
        coordinate_candidates = find_coordinate_datasets(handle)
        pdb_path, pdb_text = find_pdb_text(handle)
        atom_metadata = find_string_datasets(handle, ("atom", "residue", "chain", "name", "pdb", "topology"))
        root_attrs = {str(key): str(value) for key, value in handle.attrs.items()}
    return {
        "h5_file": str(path),
        "domain_id": parse_domain_id_from_h5_path(path),
        "root_attrs": root_attrs,
        "coordinate_candidates": coordinate_candidates[:25],
        "n_coordinate_candidates": len(coordinate_candidates),
        "pdb_text_path": pdb_path,
        "pdb_text_available": pdb_text is not None,
        "pdb_text_preview": pdb_text[:1000] if pdb_text else None,
        "metadata_candidates": atom_metadata[:50],
        "n_metadata_candidates": len(atom_metadata),
    }


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Build the schema inspection report body and metadata."""

    ensure_result_dirs()
    deps = check_optional_dependencies()
    local_files = [args.h5_file] if args.h5_file else find_local_h5_files(default_search_roots())
    local_files = [path for path in local_files if path and path.exists()]

    hf_files: list[str] = []
    hf_error = None
    if deps.huggingface_hub_available:
        try:
            hf_files = list_hf_h5_files(args.repo_id)
        except Exception as exc:  # pragma: no cover - network-dependent
            hf_error = str(exc)
    else:
        hf_error = deps.huggingface_hub_error

    h5_inspection = None
    h5_error = None
    if local_files:
        try:
            h5_inspection = inspect_h5_file(local_files[0])
        except Exception as exc:
            h5_error = str(exc)

    metadata = {
        "generated_at": now_utc(),
        "repo_id": args.repo_id,
        "license": LICENSE,
        "environment": {
            "MDCATH_ROOT": os.environ.get("MDCATH_ROOT", ""),
            "MDCATH_CACHE": os.environ.get("MDCATH_CACHE", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
        },
        "dependencies": deps.__dict__,
        "local_h5_count": len(local_files),
        "local_h5_preview": [str(path) for path in local_files[:10]],
        "hf_h5_count": len(hf_files),
        "hf_h5_preview": hf_files[: args.max_hf_files],
        "hf_error": hf_error,
        "h5_inspection": h5_inspection,
        "h5_error": h5_error,
    }

    lines = [
        "# mdCATH HDF5 Schema Inspection",
        "",
        f"Generated: `{metadata['generated_at']}`",
        f"Hugging Face repo: `{args.repo_id}`",
        f"License: `{LICENSE}`",
        "",
        "## Dependency Status",
        "",
        f"- h5py available: `{deps.h5py_available}`",
        f"- huggingface_hub available: `{deps.huggingface_hub_available}`",
        f"- h5py error: `{deps.h5py_error or 'None'}`",
        f"- huggingface_hub error: `{deps.huggingface_hub_error or 'None'}`",
        "",
        "## Local Data Search",
        "",
        f"- Local mdCATH HDF5 files found: `{len(local_files)}`",
    ]
    if local_files:
        lines.extend(f"- `{path}`" for path in local_files[:10])
    else:
        lines.append("- No local mdCATH HDF5 files were found in the conservative search roots.")
    lines.extend(
        [
            "",
            "## Remote Repository Listing",
            "",
            f"- Remote HDF5 domain files listed: `{len(hf_files)}`",
            f"- Remote listing error: `{hf_error or 'None'}`",
        ]
    )
    lines.extend(f"- `{path}`" for path in hf_files[: args.max_hf_files])
    lines.extend(["", "## Inspected Local HDF5 File", ""])
    if h5_inspection:
        lines.extend(
            [
                f"- File: `{h5_inspection['h5_file']}`",
                f"- Domain ID: `{h5_inspection['domain_id']}`",
                f"- Coordinate candidate count: `{h5_inspection['n_coordinate_candidates']}`",
                f"- PDB/topology text available: `{h5_inspection['pdb_text_available']}`",
                f"- PDB/topology text path: `{h5_inspection['pdb_text_path'] or 'NA'}`",
                "",
                "### Coordinate Candidates",
                "",
            ]
        )
        for item in h5_inspection["coordinate_candidates"]:
            lines.append(f"- `{item['path']}` shape=`{item['shape']}` dtype=`{item['dtype']}` attrs=`{item['attrs']}`")
        lines.extend(["", "### Metadata Candidates", ""])
        for item in h5_inspection["metadata_candidates"][:20]:
            lines.append(f"- `{item['path']}` shape=`{item['shape']}` dtype=`{item['dtype']}` preview=`{item['preview']}`")
    else:
        lines.append("- No local HDF5 file was inspected.")
        if h5_error:
            lines.append(f"- HDF5 inspection error: `{h5_error}`")
    lines.extend(
        [
            "",
            "## Verification Note",
            "",
            "This report does not download the full mdCATH dataset. If no local HDF5 file is inspected, "
            "coordinate paths, CA metadata paths, and units remain schema-pending until at least one domain "
            "file is cached and inspected.",
            "",
        ]
    )
    return "\n".join(lines), metadata


def main() -> None:
    """Run schema inspection."""

    args = parse_args()
    report, metadata = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    write_json(args.output.with_suffix(".json"), metadata)
    print(args.output)


if __name__ == "__main__":
    main()
