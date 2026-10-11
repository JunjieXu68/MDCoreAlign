#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/mdcorealign_vt02_1_polish"
ARCHIVE="$OUT/MDCoreAlign_VT02_1_20260831.tar.gz"
LOG="$OUT/validation"
CANONICAL_PYTHON="${VT02_1_PYTHON:-$(command -v python)}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mdcorealign_vt02_1_archive_validation.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$LOG"
test -f "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$WORK"
EXTRACTED="$WORK/MDCoreAlign_VT02_1"
test -d "$EXTRACTED"

"$CANONICAL_PYTHON" -m venv --system-site-packages "$WORK/venv"
VENV_PYTHON="$WORK/venv/bin/python"
WHEEL="$EXTRACTED/dist/mdcorealign-0.1.0rc3-py3-none-any.whl"
test -f "$WHEEL"

cd /tmp
"$VENV_PYTHON" -m pip install --no-deps --force-reinstall "$WHEEL" > "$LOG/archive_only_install.txt" 2>&1
"$VENV_PYTHON" -m pip check > "$LOG/pip_check.txt" 2>&1
"$WORK/venv/bin/mdcorealign" --help > "$LOG/public_help.txt" 2>&1
"$WORK/venv/bin/mdcorealign" apply --help > "$LOG/apply_help.txt" 2>&1
"$WORK/venv/bin/mdcorealign" doctor --json > "$LOG/doctor.json" 2>&1
PATH="$WORK/venv/bin:$PATH" PYTHON="$VENV_PYTHON" MDCOREALIGN="$WORK/venv/bin/mdcorealign" \
  bash "$EXTRACTED/run_smoke_test.sh" > "$LOG/archive_smoke_output.txt" 2>&1

RETAINED="$LOG/retained_full_atom"
mkdir -p "$RETAINED"
"$WORK/venv/bin/mdcorealign" align \
  --input "$EXTRACTED/examples/full_protein/full_atom_trajectory.pdb" \
  --input-format ca-pdb --mode throughput --backend cpu \
  --fast-representative-count 5 --output-dir "$RETAINED/result" \
  > "$LOG/simple_cli_tests.txt" 2>&1
for format in pdb cif xtc; do
  "$WORK/venv/bin/mdcorealign" apply "$RETAINED/result" \
    -o "$RETAINED/aligned.$format" --output-dir "$RETAINED/apply_$format" \
    >> "$LOG/simple_cli_tests.txt" 2>&1
done
cp "$RETAINED/result/summary.json" "$LOG/summary_after_full_atom.json"
cp "$RETAINED/result/diagnostics/source_manifest.json" "$LOG/source_manifest_after_full_atom.json"
cp "$RETAINED/apply_pdb/apply_summary.json" "$LOG/pdb_export_summary.json"
cp "$RETAINED/apply_cif/apply_summary.json" "$LOG/mmcif_export_summary.json"
cp "$RETAINED/apply_xtc/apply_summary.json" "$LOG/xtc_export_summary.json"

"$VENV_PYTHON" - "$LOG/environment_manifest.tsv" <<'PY'
from __future__ import annotations
import csv, importlib.metadata, platform, sys
from pathlib import Path
target = Path(sys.argv[1])
rows = [
    {"component": "environment", "version": "mdcorealign-vt02-1"},
    {"component": "python", "version": platform.python_version()},
]
for name in ("mdcorealign", "numpy", "scipy", "MDAnalysis", "gemmi", "cupy"):
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    rows.append({"component": name, "version": version})
with target.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("component", "version"), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY

cd "$EXTRACTED"
"$VENV_PYTHON" -m pytest -q -p no:cacheprovider tests > "$LOG/shipped_pytest_output.txt" 2>&1
"$CANONICAL_PYTHON" -m ruff check src/mdcorealign > "$LOG/archive_ruff_output.txt" 2>&1
"$CANONICAL_PYTHON" -m mypy --python-version 3.12 --follow-imports=skip \
  --ignore-missing-imports src/mdcorealign/__init__.py > "$LOG/archive_mypy_output.txt" 2>&1
"$VENV_PYTHON" -m compileall -q src
printf 'compileall: PASS\n' > "$LOG/archive_compileall_output.txt"
"$CANONICAL_PYTHON" -m twine check dist/* > "$LOG/twine_check.txt" 2>&1
"$CANONICAL_PYTHON" -m build --outdir "$WORK/rebuilt_dist" . > "$LOG/archive_rebuild_output.txt" 2>&1

mkdir -p "$WORK/clean_scan"
tar -xzf "$ARCHIVE" -C "$WORK/clean_scan"
SCAN_PACKAGE="$WORK/clean_scan/MDCoreAlign_VT02_1"

cd "$ROOT"
"$CANONICAL_PYTHON" - "$ROOT" "$SCAN_PACKAGE" "$ARCHIVE" "$LOG" <<'PY'
from __future__ import annotations

import csv
import hashlib
import re
import sys
import tarfile
from pathlib import Path

root, package, archive, log = map(Path, sys.argv[1:])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


markdown = sorted(path.relative_to(package).as_posix() for path in package.rglob("*.md"))
assert markdown == ["DEVELOPER.md", "README.md", "VALIDATION.md"], markdown

with (package / "validation/sha256_manifest.tsv").open(newline="") as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))
assert manifest
manifest_rows = []
for row in manifest:
    path = package / row["path"]
    status = path.is_file() and digest(path) == row["sha256"] and path.stat().st_size == int(row["size_bytes"])
    manifest_rows.append({"path": row["path"], "status": "PASS" if status else "FAIL"})
assert all(row["status"] == "PASS" for row in manifest_rows)
write_tsv(log / "archive_extract_manifest.tsv", manifest_rows)

with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
assert members and all(Path(member.name).parts[0] == "MDCoreAlign_VT02_1" for member in members)
assert not any(member.size > 100 * 1024 * 1024 for member in members)

text_suffixes = {"", ".cff", ".json", ".md", ".py", ".sh", ".toml", ".tsv", ".txt", ".yml", ".yaml"}
cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
local_path = re.compile(r"/(?:home|storage)/[A-Za-z0-9_.-]+/")
secret = re.compile(r"hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]{8,}", re.I)
person = re.compile(r"\b(?:Ashar|Malik|Junjie|Xu|advisor|supervisor|private-alpha)\b|mdcorealign-vt01|0\.1\.0rc1", re.I)
scan_rows = []
language_rows = []
for path in sorted(package.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in text_suffixes:
        continue
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    relative = path.relative_to(package).as_posix()
    issues = []
    if local_path.search(text):
        issues.append("absolute_local_path")
    if secret.search(text):
        issues.append("secret_pattern")
    if person.search(text):
        issues.append("person_specific_content")
    scan_rows.append({"path": relative, "issues": "|".join(issues) or "none", "status": "FAIL" if issues else "PASS"})
    language_rows.append({"path": relative, "cjk_count": len(cjk.findall(text)), "status": "FAIL" if cjk.search(text) else "PASS"})
assert all(row["status"] == "PASS" for row in scan_rows), [row for row in scan_rows if row["status"] != "PASS"]
assert all(row["status"] == "PASS" for row in language_rows)
write_tsv(log / "path_secret_scan.tsv", scan_rows)
write_tsv(log / "language_scan.tsv", language_rows)

parity_rows = []
for source_root in (root / "src/mdcorealign", root / "src/mdsuperpose"):
    for source in sorted(source_root.rglob("*.py")):
        target = package / source.relative_to(root)
        status = target.is_file() and digest(source) == digest(target)
        parity_rows.append({"path": source.relative_to(root).as_posix(), "status": "PASS" if status else "FAIL"})
assert all(row["status"] == "PASS" for row in parity_rows)
write_tsv(log / "source_hash_parity.tsv", parity_rows)

forbidden_suffixes = {".h5", ".hdf5", ".dcd", ".tpr", ".npy", ".npz"}
assert not any(path.suffix.lower() in forbidden_suffixes for path in package.rglob("*") if path.is_file())
assert not any("--plain" in path.read_text() for path in (package / "README.md", package / "run_smoke_test.sh"))
assert "--plain" not in (log / "apply_help.txt").read_text()

summary_data = __import__("json").loads((log / "summary_after_full_atom.json").read_text())
manifest_data = __import__("json").loads((log / "source_manifest_after_full_atom.json").read_text())
assert summary_data["schema_version"] == "1.1.0"
assert summary_data["software_version"] == "0.1.0rc3"
assert summary_data["n_fit_atoms"] == summary_data["n_analysis_coordinate_atoms"] == 6
assert summary_data["default_export_selection"] == "protein"
assert summary_data["n_exportable_atoms"] == 24
assert "apply_selection" not in summary_data and "n_apply_atoms" not in summary_data
assert manifest_data["source_result_schema_version"] == "1.1.0"
for field in ("n_fit_atoms", "n_analysis_coordinate_atoms", "default_export_selection", "n_exportable_atoms", "source_is_ca_only"):
    assert summary_data[field] == manifest_data[field]
for name in ("pdb", "mmcif", "xtc"):
    export = __import__("json").loads((log / f"{name}_export_summary.json").read_text())
    for field in ("export_selection", "n_exported_atoms", "output_format", "annotation_mapping", "source_result_schema_version"):
        assert field in export
    assert export["n_exported_atoms"] == 24

summary = f"""# Archive-only validation

- Archive: `{archive.name}`
- Extracted root: `MDCoreAlign_VT02_1/`
- Package version: `0.1.0rc3`
- Manifest entries verified: {len(manifest_rows)}
- Shipped wheel installed from the extracted archive: PASS
- Location-independent execution from `/tmp`: PASS
- CPU, exact, W5, ordinary-RMSF, PDB, mmCIF and XTC smoke: PASS
- Explicit GPU execution and CPU/GPU parity: PASS
- Shipped tests: PASS
- Build, Twine, pip check, Ruff, public API mypy and compileall: PASS
- Exactly three Markdown files: PASS
- English/CJK, person-specific content, path and secret scans: PASS
- Source hash parity: PASS
- Extraction and SHA-256 manifest verification: PASS
"""
(log / "archive_only_validation.md").write_text(summary)
print("VT02.1 independent archive validation: PASS")
PY

echo "VT02.1 release validation complete"
