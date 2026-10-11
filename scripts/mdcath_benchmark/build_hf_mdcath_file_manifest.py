#!/usr/bin/env python
"""Build a Hugging Face file manifest for mdCATH without downloading HDF5 data."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import HF_REPO_ID, RESULT_ROOT, write_tsv


DATASET_DOMAIN_RE = re.compile(r"mdcath_dataset_([^/]+)\.h5$")
DOMAIN_RE = re.compile(r"([0-9][A-Za-z0-9]{3}[A-Za-z0-9][0-9]{2,3})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "manifest/hf_mdcath_file_manifest.tsv")
    parser.add_argument("--expand", action="store_true", help="Use slower expanded HF metadata with file sizes/hashes.")
    return parser.parse_args()


def _entry_attr(entry: object, names: tuple[str, ...], default: object = "") -> object:
    for name in names:
        if hasattr(entry, name):
            value = getattr(entry, name)
            if value is not None:
                return value
    return default


def list_entries(repo_id: str, revision: str, *, expand: bool = False) -> tuple[list[dict[str, object]], str]:
    """List repository entries with metadata when available."""

    from huggingface_hub import HfApi, list_repo_files

    rows: list[dict[str, object]] = []
    source = "list_repo_files_fast"
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    if not expand:
        for path in list_repo_files(repo_id, repo_type="dataset", revision=revision):
            rows.append(build_row(repo_id, revision, path, "", None, None))
        return rows, source
    source = "list_repo_tree_expand"
    try:
        tree = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
        )
        for entry in tree:
            path = str(_entry_attr(entry, ("path",), ""))
            if not path:
                continue
            size = _entry_attr(entry, ("size",), "")
            lfs = _entry_attr(entry, ("lfs",), None)
            xet = _entry_attr(entry, ("xet",), None)
            rows.append(build_row(repo_id, revision, path, size, lfs, xet))
    except Exception:
        source = "list_repo_files_fallback"
        for path in list_repo_files(repo_id, repo_type="dataset", revision=revision):
            rows.append(build_row(repo_id, revision, path, "", None, None))
    return rows, source


def build_row(
    repo_id: str,
    revision: str,
    path: str,
    size: object,
    lfs: object | None,
    xet: object | None,
) -> dict[str, object]:
    filename = Path(path).name
    extension = "".join(Path(filename).suffixes).lower()
    dataset_match = DATASET_DOMAIN_RE.search(path)
    match = dataset_match or DOMAIN_RE.search(path)
    inferred_domain = match.group(1) if match else "NA"
    lfs_sha = ""
    if isinstance(lfs, dict):
        lfs_sha = str(lfs.get("sha256") or lfs.get("oid") or "")
    elif lfs is not None and hasattr(lfs, "sha256"):
        lfs_sha = str(getattr(lfs, "sha256"))
    xet_hash = ""
    if isinstance(xet, dict):
        xet_hash = str(xet.get("hash") or "")
    elif xet is not None and hasattr(xet, "hash"):
        xet_hash = str(getattr(xet, "hash"))
    is_h5 = int(extension in {".h5", ".hdf5"})
    return {
        "repo_id": repo_id,
        "revision_or_commit": revision,
        "path": path,
        "filename": filename,
        "file_extension": extension or "NA",
        "size_bytes": size if size not in (None, "") else "NA",
        "lfs_sha256_if_available": lfs_sha or "NA",
        "xet_hash_if_available": xet_hash or "NA",
        "is_h5_or_hdf5": is_h5,
        "inferred_domain_id": inferred_domain,
        "inferred_temperature_if_in_path": "NA",
        "inferred_replica_if_in_path": "NA",
        "parse_status": "domain_id_inferred" if match else "no_domain_id",
        "notes": "",
    }


def main() -> None:
    args = parse_args()
    rows, source = list_entries(args.repo_id, args.revision, expand=args.expand)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "repo_id",
        "revision_or_commit",
        "path",
        "filename",
        "file_extension",
        "size_bytes",
        "lfs_sha256_if_available",
        "xet_hash_if_available",
        "is_h5_or_hdf5",
        "inferred_domain_id",
        "inferred_temperature_if_in_path",
        "inferred_replica_if_in_path",
        "parse_status",
        "notes",
    ]
    write_tsv(args.output, sorted(rows, key=lambda row: str(row["path"])), columns)
    h5_rows = [row for row in rows if row["is_h5_or_hdf5"] == 1]
    domains = {row["inferred_domain_id"] for row in h5_rows if row["inferred_domain_id"] != "NA"}
    sizes = []
    for row in rows:
        try:
            sizes.append(int(row["size_bytes"]))
        except Exception:
            pass
    metadata_like = [
        row for row in rows
        if str(row["file_extension"]).lower() in {".csv", ".tsv", ".json", ".yaml", ".yml", ".txt", ".md", ".parquet"}
    ]
    summary = [
        {
            "n_total_repo_entries": len(rows),
            "n_h5_files": len(h5_rows),
            "n_inferred_domains": len(domains),
            "total_size_bytes_from_manifest": sum(sizes) if sizes else "NA",
            "n_metadata_like_files": len(metadata_like),
            "n_unparsed_h5_files": sum(1 for row in h5_rows if row["inferred_domain_id"] == "NA"),
            "example_paths": ";".join(str(row["path"]) for row in rows[:10]),
            "listing_source": source,
        }
    ]
    write_tsv(args.output.with_name("hf_mdcath_file_manifest_summary.tsv"), summary)
    print(args.output)


if __name__ == "__main__":
    main()
