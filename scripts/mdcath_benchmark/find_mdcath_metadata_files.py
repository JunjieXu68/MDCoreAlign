#!/usr/bin/env python
"""Identify and download small mdCATH metadata files from the HF file manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import HF_REPO_ID, RESULT_ROOT, read_tsv, write_tsv


METADATA_EXTENSIONS = {".csv", ".tsv", ".json", ".yaml", ".yml", ".txt", ".md", ".parquet"}
METADATA_KEYWORDS = ("metadata", "cath", "domain", "split", "manifest", "index", "readme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-manifest", type=Path, default=RESULT_ROOT / "manifest/hf_mdcath_file_manifest.tsv")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "metadata_files")
    parser.add_argument("--report", type=Path, default=Path("docs/mdcath_metadata_file_inventory.md"))
    return parser.parse_args()


def reason_for(row: dict[str, str]) -> str | None:
    path = row["path"].lower()
    ext = row["file_extension"].lower()
    reasons = []
    is_h5 = row.get("is_h5_or_hdf5") == "1"
    filename = row.get("filename", "").lower()
    if ext in METADATA_EXTENSIONS:
        reasons.append(f"extension:{ext}")
    for keyword in METADATA_KEYWORDS:
        if keyword in path and not (is_h5 and keyword == "cath"):
            reasons.append(f"keyword:{keyword}")
    if is_h5 and any(k in filename for k in ("source", "metadata", "manifest", "index")):
        reasons.append("h5_metadata_candidate_not_downloaded")
    return ";".join(reasons) if reasons else None


def inspect_tabular(path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return "NA", "no", "no"
    suffix = path.suffix.lower()
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return "NA", "unknown", "unknown"
    first = next((line for line in text.splitlines() if line.strip() and not line.startswith("#")), "")
    if suffix in {".tsv", ".txt"}:
        sep = "\t" if "\t" in first else None
    elif suffix == ".csv":
        sep = ","
    else:
        sep = None
    columns = first.split(sep) if sep else []
    joined = " ".join(columns).lower()
    contains_domain = "domain" in joined or any("domain" in line.lower() for line in text.splitlines()[:20])
    contains_cath = any(key in joined for key in ("class", "architecture", "topology", "homolog"))
    return ",".join(columns[:30]) if columns else "NA", "yes" if contains_domain else "no", "yes" if contains_cath else "no"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(args.file_manifest)
    selected = []
    from huggingface_hub import hf_hub_download

    for row in rows:
        reason = reason_for(row)
        if not reason:
            continue
        output_row = dict(row)
        output_row["why_selected"] = reason
        output_row["downloaded"] = "no"
        output_row["local_path"] = "NA"
        output_row["columns_if_tabular"] = "NA"
        output_row["contains_domain_id"] = "unknown"
        output_row["contains_cath_hierarchy"] = "unknown"
        if row["file_extension"].lower() in METADATA_EXTENSIONS and row.get("is_h5_or_hdf5") != "1":
            try:
                local = hf_hub_download(
                    repo_id=row["repo_id"],
                    repo_type="dataset",
                    filename=row["path"],
                    local_dir=str(args.output_dir),
                )
                output_row["downloaded"] = "yes"
                output_row["local_path"] = str(local)
                cols, has_domain, has_cath = inspect_tabular(Path(local))
                output_row["columns_if_tabular"] = cols
                output_row["contains_domain_id"] = has_domain
                output_row["contains_cath_hierarchy"] = has_cath
            except Exception as exc:
                output_row["notes"] = f"download_failed:{type(exc).__name__}:{exc}"
        selected.append(output_row)
    table_path = args.output_dir / "metadata_file_inventory.tsv"
    columns = [
        "path", "filename", "file_extension", "size_bytes", "why_selected", "downloaded",
        "local_path", "columns_if_tabular", "contains_domain_id", "contains_cath_hierarchy", "notes",
    ]
    write_tsv(table_path, selected, columns)
    lines = [
        "# mdCATH Metadata File Inventory",
        "",
        "Only small metadata-like files were downloaded. HDF5 files were not downloaded in this step.",
        "",
        "| file path | size | why selected | downloaded | columns | contains domain_id | contains CATH hierarchy |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['path']}` | {row.get('size_bytes','NA')} | {row.get('why_selected','')} | "
            f"{row.get('downloaded','no')} | `{row.get('columns_if_tabular','NA')}` | "
            f"{row.get('contains_domain_id','unknown')} | {row.get('contains_cath_hierarchy','unknown')} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(table_path)


if __name__ == "__main__":
    main()
