#!/usr/bin/env python
"""Download or locate one mdCATH domain HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import (
    HF_REPO_ID,
    cache_size_gb,
    default_search_roots,
    download_hf_file,
    ensure_result_dirs,
    find_local_h5_files,
    parse_domain_id_from_h5_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", required=True, help="mdCATH domain ID, e.g. 1a39A00.")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument("--cache-dir", type=Path, default=Path("results/mdcath_benchmark/cache"))
    parser.add_argument("--no-download", action="store_true", help="Only search local cache; do not download.")
    parser.add_argument("--max-cache-gb", type=float, default=None, help="Warn if cache exceeds this size.")
    return parser.parse_args()


def find_domain_local(domain_id: str, roots: list[Path]) -> Path | None:
    """Find a cached HDF5 file for one domain."""

    expected = f"mdcath_dataset_{domain_id}.h5"
    for path in find_local_h5_files(roots):
        if path.name == expected or parse_domain_id_from_h5_path(path) == domain_id:
            return path
    return None


def main() -> None:
    """Locate or download one domain file."""

    args = parse_args()
    ensure_result_dirs()
    roots = [args.cache_dir, *default_search_roots()]
    local = find_domain_local(args.domain_id, roots)
    status = "found_local"
    error = None
    if local is None and not args.no_download:
        repo_file = f"data/mdcath_dataset_{args.domain_id}.h5"
        try:
            local = download_hf_file(repo_file, cache_dir=args.cache_dir, repo_id=args.repo_id)
            status = "downloaded_or_cached"
        except Exception as exc:
            status = "download_failed"
            error = str(exc)
    elif local is None:
        status = "not_found_no_download"

    payload = {
        "domain_id": args.domain_id,
        "status": status,
        "local_path": str(local) if local else None,
        "cache_dir": str(args.cache_dir),
        "cache_size_gb": cache_size_gb(args.cache_dir),
        "max_cache_gb": args.max_cache_gb,
        "error": error,
    }
    write_json(Path("results/mdcath_benchmark/logs") / f"cache_{args.domain_id}.json", payload)
    if local:
        print(local)
    else:
        raise SystemExit(f"{status}: {error or 'domain file unavailable'}")


if __name__ == "__main__":
    main()
