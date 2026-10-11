#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/mdcorealign_vt02_2_pip_first"
ARCHIVE="$OUT/06_release/MDCoreAlign_VT02_2_20260901.tar.gz"
LOG="$OUT/03_pip_validation"
CANONICAL_PYTHON="${VT02_2_PYTHON:-$(command -v python3)}"
VALIDATION_PYTHON="${VT02_2_VALIDATION_PYTHON:-$CANONICAL_PYTHON}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mdcorealign_vt02_2_archive_validation.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$LOG"
test -f "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$WORK"
EXTRACTED="$WORK/MDCoreAlign_VT02_2"
WHEEL="$EXTRACTED/dist/mdcorealign-0.1.0rc4-py3-none-any.whl"
SDIST="$EXTRACTED/dist/mdcorealign-0.1.0rc4.tar.gz"
test -d "$EXTRACTED"
test -f "$WHEEL"
test -f "$SDIST"

cd /tmp
"$VALIDATION_PYTHON" -m pip install --no-deps --force-reinstall "$WHEEL" \
  > "$LOG/archive_only_install.txt" 2>&1
"$VALIDATION_PYTHON" -m pip check > "$LOG/pip_check.txt" 2>&1
"$VALIDATION_PYTHON" -m mdcorealign doctor --json > "$LOG/doctor.json" 2>&1
PATH="$(dirname "$VALIDATION_PYTHON"):$PATH" PYTHON="$VALIDATION_PYTHON" \
  MDCOREALIGN="$(dirname "$VALIDATION_PYTHON")/mdcorealign" \
  bash "$EXTRACTED/run_smoke_test.sh" > "$LOG/archive_smoke_output.txt" 2>&1

cd "$EXTRACTED"
"$VALIDATION_PYTHON" -m pytest -q -p no:cacheprovider tests \
  > "$LOG/shipped_pytest_output.txt" 2>&1
"$CANONICAL_PYTHON" -m ruff check src/mdcorealign tests/vt02_2_pip_first \
  > "$LOG/archive_ruff_output.txt" 2>&1
"$CANONICAL_PYTHON" -m mypy --python-version 3.12 --follow-imports=skip \
  --ignore-missing-imports src/mdcorealign \
  > "$LOG/archive_mypy_output.txt" 2>&1
"$VALIDATION_PYTHON" -m compileall -q src
printf 'compileall: PASS\n' > "$LOG/archive_compileall_output.txt"
"$CANONICAL_PYTHON" -m twine check dist/* > "$LOG/twine_check.txt" 2>&1
"$CANONICAL_PYTHON" -m build --outdir "$WORK/rebuilt_dist" . \
  > "$LOG/archive_rebuild_output.txt" 2>&1

cd "$ROOT"
"$CANONICAL_PYTHON" - "$ROOT" "$EXTRACTED" "$ARCHIVE" "$LOG" <<'PY'
from __future__ import annotations

import csv
import email
import hashlib
import re
import sys
import tarfile
import zipfile
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
    status = (
        path.is_file()
        and digest(path) == row["sha256"]
        and path.stat().st_size == int(row["size_bytes"])
    )
    manifest_rows.append({"path": row["path"], "status": "PASS" if status else "FAIL"})
assert all(row["status"] == "PASS" for row in manifest_rows)
write_tsv(log / "archive_extract_manifest.tsv", manifest_rows)

with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
assert members and all(Path(member.name).parts[0] == "MDCoreAlign_VT02_2" for member in members)
assert not any(member.size > 100 * 1024 * 1024 for member in members)

wheel = package / "dist/mdcorealign-0.1.0rc4-py3-none-any.whl"
with zipfile.ZipFile(wheel) as handle:
    metadata_name = next(name for name in handle.namelist() if name.endswith(".dist-info/METADATA"))
    metadata = email.message_from_bytes(handle.read(metadata_name))
requires = metadata.get_all("Requires-Dist", [])
lower = [value.lower() for value in requires]
for name in ("numpy", "scipy", "mdanalysis", "gemmi"):
    assert any(value.startswith(name) for value in lower), (name, requires)
assert not any(value.startswith("cupy") and "extra ==" not in value for value in lower)
assert sum(value.startswith("cupy-cuda12x") for value in lower) == 1
assert any("extra == 'cuda12'" in value or 'extra == "cuda12"' in value for value in lower)

text_suffixes = {
    "", ".cff", ".json", ".md", ".py", ".sh", ".toml", ".tsv", ".txt", ".yml", ".yaml"
}
cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
local_path = re.compile(r"/(?:home|storage)/[A-Za-z0-9_.-]+/")
secret = re.compile(
    r"hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]{8,}",
    re.I,
)
stale = re.compile(r"MDCoreAlign VT0(?:1|2\.1)|0\.1\.0rc[123]|mdcorealign-vt0(?:1|2-1)", re.I)
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
    if stale.search(text):
        issues.append("stale_release_identity")
    scan_rows.append(
        {"path": relative, "issues": "|".join(issues) or "none", "status": "FAIL" if issues else "PASS"}
    )
    language_rows.append(
        {"path": relative, "cjk_count": len(cjk.findall(text)), "status": "FAIL" if cjk.search(text) else "PASS"}
    )
assert all(row["status"] == "PASS" for row in scan_rows), [
    row for row in scan_rows if row["status"] != "PASS"
]
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
assert "python -m venv" in (package / "README.md").read_text()
readme = (package / "README.md").read_text()
assert readme.find("### Pip and venv (primary)") < readme.find("### Conda (optional)")
assert "--conda" in (package / "install.sh").read_text()

summary = f"""# Archive-only validation

- Archive: `{archive.name}`
- Extracted root: `MDCoreAlign_VT02_2/`
- Package version: `0.1.0rc4`
- Manifest entries verified: {len(manifest_rows)}
- Shipped wheel installed from the extracted archive: PASS
- Location-independent execution from `/tmp`: PASS
- CPU, exact, W5, ordinary-RMSF, PDB, mmCIF and XTC smoke: PASS
- Shipped tests: PASS
- Rebuild, Twine, pip check, Ruff, package-wide mypy and compileall: PASS
- Exactly three Markdown files: PASS
- Wheel dependency and CUDA-12-extra metadata: PASS
- English/CJK, stale identity, path and secret scans: PASS
- Source hash parity: PASS
- Extraction and SHA-256 manifest verification: PASS
"""
(log / "archive_only_validation.md").write_text(summary)
print("VT02.2 independent archive validation: PASS")
PY

echo "VT02.2 release validation complete"
