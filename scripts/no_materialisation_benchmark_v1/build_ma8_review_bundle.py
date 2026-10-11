#!/usr/bin/env python3
"""Build and validate the mandatory MA8 method-agreement review archive."""

from __future__ import annotations

import csv
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/no_materialisation_benchmark_v1"
MA = BASE / "method_agreement_v2"
HANDOFF = MA / "handoff_to_chatgpt"
ARCHIVE = BASE / f"method_agreement_v2_review_bundle_{datetime.now():%Y%m%d}.tar.gz"
VALIDATION = MA / "08_reporting/validation"

TEXT_SUFFIXES = {".c", ".csv", ".diff", ".f90", ".h", ".json", ".log", ".md", ".patch", ".py", ".toml", ".tsv", ".txt"}
EXCLUDED_SUFFIXES = {".dcd", ".h5", ".hdf5", ".npy", ".npz", ".pdb", ".tpr", ".xtc"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sanitise_text(text: str) -> str:
    replacements = {
        str(ROOT): "<repository-root>",
        "/storage/junjie/projects/mdlovofit": "<repository-root>",
        "/storage/junjie/": "<storage-root>/",
        "/home/jjxu/": "<user-home>/",
        "/home/jjxu": "<user-home>",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        destination.write_text(sanitise_text(source.read_text(errors="replace")))
    else:
        shutil.copy2(source, destination)


def copy_results(stage: Path) -> None:
    required = {
        MA / "08_reporting/final_method_agreement_report.md": stage / "final_method_agreement_report.md",
        MA / "08_reporting/phase_status.tsv": stage / "phase_status.tsv",
        MA / "08_reporting/key_metrics.tsv": stage / "key_metrics.tsv",
        MA / "00_contracts/method_agreement_contract.md": stage / "method_agreement_contract.md",
        MA / "00_contracts/fraction_registry.tsv": stage / "fraction_registry.tsv",
        MA / "00_contracts/metric_dictionary.tsv": stage / "metric_dictionary.tsv",
        MA / "00_contracts/agreement_category_dictionary.md": stage / "agreement_category_dictionary.md",
        MA / "01_mdlovofit_export_validation/mdlovofit_subset_semantics.md": stage / "mdlovofit_subset_semantics.md",
        MA / "01_mdlovofit_export_validation/export_validation_report.md": stage / "export_validation_report.md",
        MA / "01_mdlovofit_export_validation/exporter_self_consistency.tsv": stage / "exporter_self_consistency.tsv",
        MA / "01_mdlovofit_export_validation/subset_vs_occupancy.tsv": stage / "subset_vs_occupancy.tsv",
        MA / "01_mdlovofit_export_validation/transform_reconstruction.tsv": stage / "transform_reconstruction.tsv",
        MA / "02_legacy_four_systems/four_system_core_agreement.tsv": stage / "four_system_core_agreement.tsv",
        MA / "02_legacy_four_systems/four_system_alignment_agreement.tsv": stage / "four_system_alignment_agreement.tsv",
        MA / "03_atlas_60pct/atlas_agreement_summary.tsv": stage / "atlas_agreement_summary.tsv",
        MA / "04_mdcath_60pct/mdcath_agreement_summary.tsv": stage / "mdcath_agreement_summary.tsv",
        MA / "05_fraction_sensitivity/fraction_sensitivity_summary.tsv": stage / "fraction_sensitivity_summary.tsv",
        MA / "06_alignment_outputs/transform_agreement.tsv": stage / "transform_agreement.tsv",
        MA / "06_alignment_outputs/aligned_coordinate_agreement.tsv": stage / "aligned_coordinate_agreement.tsv",
        MA / "06_alignment_outputs/rmsd_agreement.tsv": stage / "rmsd_agreement.tsv",
        MA / "06_alignment_outputs/rmsf_agreement.tsv": stage / "rmsf_agreement.tsv",
        MA / "07_independent_stability/independent_stability_summary.tsv": stage / "independent_stability_summary.tsv",
        MA / "07_independent_stability/disagreement_case_atlas.md": stage / "disagreement_case_atlas.md",
        MA / "08_reporting/manuscript_wording_recommendations.md": stage / "manuscript_wording_recommendations.md",
        MA / "08_reporting/claim_evidence_matrix.tsv": stage / "claim_evidence_matrix.tsv",
        MA / "08_reporting/limitations.md": stage / "limitations.md",
        MA / "08_reporting/adaptive_expansion_decision.json": stage / "adaptive_expansion_decision.json",
        VALIDATION / "focused_test_output.log": stage / "validation/focused_test_output.log",
        VALIDATION / "full_test_output.log": stage / "validation/full_test_output.log",
    }
    for source, destination in required.items():
        copy_file(source, destination)

    for directory in ("00_contracts", "01_mdlovofit_export_validation", "02_legacy_four_systems", "03_atlas_60pct", "04_mdcath_60pct", "05_fraction_sensitivity", "06_alignment_outputs", "07_independent_stability"):
        for source in sorted((MA / directory).rglob("*")):
            if not source.is_file() or source.suffix.lower() in EXCLUDED_SUFFIXES or source.name == "mdlovofit_export":
                continue
            copy_file(source, stage / "complete_results" / source.relative_to(MA))
    for source in sorted((MA / "08_reporting/figures").glob("*")):
        copy_file(source, stage / "figures" / source.name)


def copy_sources(stage: Path) -> None:
    sources = [
        ROOT / "scripts/no_materialisation_benchmark_v1/build_method_agreement_exporter.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_ma0_ma2.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/lock_ma3_ma7_panels.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/method_agreement_metrics.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_ma3_ma7_agreement.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/finalize_ma3_ma7.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/build_ma8_review_bundle.py",
        ROOT / "tests/no_materialisation_v1/test_method_agreement_metrics.py",
    ]
    exporter_sources = sorted((MA / "01_mdlovofit_export_validation/source").glob("*"))
    sources.extend(path for path in exporter_sources if path.is_file())
    patch_lines: list[str] = []
    for source in sources:
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path("exporter_source") / source.name
        copy_file(source, stage / "source" / relative)
        content = sanitise_text(source.read_text(errors="replace")).splitlines(keepends=True)
        patch_lines.extend(difflib.unified_diff([], content, fromfile=f"a/{relative.as_posix()}", tofile=f"b/{relative.as_posix()}"))
    (stage / "unified_source_diff.patch").write_text("".join(patch_lines))


def write_entry_files(stage: Path) -> None:
    phase_path = stage / "phase_status.tsv"
    phases = read_tsv(phase_path)
    for row in phases:
        if row["phase"] == "MA8":
            row["status"] = "PASS"
            row["evidence"] = "sha256_manifest.tsv and archive validation"
    write_tsv(phase_path, phases)
    (stage / "known_limitations.md").write_text((stage / "limitations.md").read_text())
    (stage / "00_README_FIRST.md").write_text(
        "# MA8 Method-Agreement Review Handoff\n\n"
        "Start with `final_method_agreement_report.md`, `key_metrics.tsv`, `phase_status.tsv` and `known_limitations.md`.\n\n"
        "This archive contains an untimed method-agreement analysis. Exact identity is required only between the diagnostic exporter and stock MDLovoFit `-t` output for the same input. MDCoreAlign and MDLovoFit are not expected to return identical cores or transforms.\n\n"
        "The locked scope contains 512 primary mdCATH domains, 128 detailed mdCATH domains, eight ATLAS full-length systems, six ATLAS short-prefix tails and four historical systems. The full 5,397-domain agreement run was not performed.\n\n"
        "The 60% fraction uses `K = floor(0.60 * N)` for equal-cardinality method comparison and is not a universal biological core size. Timing evidence is outside this archive and was not rerun.\n\n"
        "Validation commands:\n\n"
        "```bash\n"
        "PYTHONPATH=src .venv-gpu-cu12/bin/python -m pytest -q tests/no_materialisation_v1/test_method_agreement_exporter.py\n"
        "PYTHONPATH=src .venv-gpu-cu12/bin/python -m pytest -q\n"
        ".venv-gpu-cu12/bin/python -m scripts.no_materialisation_benchmark_v1.finalize_ma3_ma7\n"
        ".venv-gpu-cu12/bin/python -m scripts.no_materialisation_benchmark_v1.build_ma8_review_bundle\n"
        "```\n"
    )


def safety_scan(stage: Path) -> list[dict[str, str]]:
    patterns = {
        "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "credential_assignment": re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]+"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
        "workspace_absolute_path": re.compile(r"/storage/junjie/"),
        "user_home_path": re.compile(r"/home/jjxu/"),
    }
    findings = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"file": path.relative_to(stage).as_posix(), "finding": label})
    return findings


def make_manifests(stage: Path) -> None:
    rows = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            continue
        rows.append({"relative_path": path.relative_to(stage).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    rows.extend(
        [
            {"relative_path": "file_manifest.tsv", "size_bytes": "self", "sha256": "see_sha256_manifest"},
            {"relative_path": "sha256_manifest.tsv", "size_bytes": "self", "sha256": "self_excluded_by_definition"},
        ]
    )
    write_tsv(stage / "file_manifest.tsv", rows)
    hashes = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.tsv":
            hashes.append({"relative_path": path.relative_to(stage).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    write_tsv(stage / "sha256_manifest.tsv", hashes)


def validate_archive(path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="ma8_verify_") as temporary:
        target = Path(temporary)
        with tarfile.open(path, "r:gz") as handle:
            members = handle.getmembers()
            if any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members):
                raise RuntimeError("unsafe archive member")
            handle.extractall(target)
        extracted = target / "handoff_to_chatgpt"
        if {item.name for item in target.iterdir()} != {"handoff_to_chatgpt"}:
            raise RuntimeError("archive must contain one handoff_to_chatgpt root")
        hashes = read_tsv(extracted / "sha256_manifest.tsv")
        for row in hashes:
            candidate = extracted / row["relative_path"]
            if not candidate.is_file() or sha256(candidate) != row["sha256"]:
                raise RuntimeError(f"hash mismatch: {row['relative_path']}")
        actual = {item.relative_to(extracted).as_posix() for item in extracted.rglob("*") if item.is_file()}
        expected = {row["relative_path"] for row in hashes} | {"sha256_manifest.tsv"}
        if actual != expected:
            raise RuntimeError(f"archive file-set mismatch: missing={expected-actual}, extra={actual-expected}")
        return {"archive_file_count": len(actual), "manifest_rows_verified": len(hashes), "single_root": "handoff_to_chatgpt"}


def main() -> None:
    status = json.loads((MA / "08_reporting/ma3_ma7_final_status.json").read_text())
    if status.get("status") != "pass":
        raise RuntimeError("MA3-MA7 finalization is not complete")
    with tempfile.TemporaryDirectory(prefix="ma8_stage_", dir=BASE) as temporary:
        stage = Path(temporary) / "handoff_to_chatgpt"
        stage.mkdir()
        copy_results(stage)
        copy_sources(stage)
        write_entry_files(stage)
        excluded = [path for path in stage.rglob("*") if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES]
        oversized = [path for path in stage.rglob("*") if path.is_file() and path.stat().st_size > 100 * 1024 * 1024]
        findings = safety_scan(stage)
        if excluded or oversized or findings:
            raise RuntimeError(f"bundle safety scan failed: excluded={excluded}, oversized={oversized}, findings={findings}")
        make_manifests(stage)
        if HANDOFF.exists():
            if any(HANDOFF.iterdir()):
                raise RuntimeError(f"refusing to overwrite non-empty handoff: {HANDOFF}")
            HANDOFF.rmdir()
        shutil.copytree(stage, HANDOFF)
    if ARCHIVE.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {ARCHIVE}")
    with tarfile.open(ARCHIVE, "w:gz") as handle:
        handle.add(HANDOFF, arcname="handoff_to_chatgpt")
    validation = validate_archive(ARCHIVE)
    validation.update(
        {
            "archive": ARCHIVE.relative_to(ROOT).as_posix(),
            "archive_sha256": sha256(ARCHIVE),
            "archive_size_bytes": ARCHIVE.stat().st_size,
            "absolute_path_scan": "pass",
            "excluded_format_scan": "pass",
            "manifest_verification": "pass",
            "secret_scan": "pass",
            "status": "pass",
        }
    )
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "ma8_archive_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    phases = read_tsv(MA / "08_reporting/phase_status.tsv")
    for row in phases:
        if row["phase"] == "MA8":
            row["status"] = "PASS"
            row["evidence"] = ARCHIVE.relative_to(ROOT).as_posix()
    write_tsv(MA / "08_reporting/phase_status.tsv", phases)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
