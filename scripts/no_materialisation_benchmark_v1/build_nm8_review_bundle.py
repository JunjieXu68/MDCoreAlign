#!/usr/bin/env python3
"""Build and validate the mandatory NM8 review handoff archive."""

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
OUT = ROOT / "results/no_materialisation_benchmark_v1"
HANDOFF = OUT / "handoff_to_chatgpt"
ARCHIVE = OUT / f"no_materialisation_benchmark_v1_review_bundle_{datetime.now():%Y%m%d}.tar.gz"

TEXT_SUFFIXES = {
    ".csv",
    ".diff",
    ".json",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
}
EXCLUDED_SUFFIXES = {".dcd", ".h5", ".hdf5", ".npy", ".npz", ".tpr", ".xtc"}


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
        raise ValueError(f"refusing to write empty TSV: {path}")
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
        "/home/jjxu": "<user-home>",
        "/home/jjxu/": "<user-home>/",
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


def copy_required(stage: Path) -> None:
    mappings = {
        OUT / "06_analysis/final_no_materialisation_decision_report.md": stage / "final_no_materialisation_decision_report.md",
        OUT / "00_freeze/contract_registry.md": stage / "contract_registry.md",
        OUT / "00_freeze/timing_boundary_dictionary.tsv": stage / "timing_boundary_dictionary.tsv",
        OUT / "00_freeze/environment_manifest.tsv": stage / "environment_manifest.tsv",
        OUT / "00_freeze/superseding_nm_ma_scope_20260819.md": stage / "superseding_nm_ma_scope_20260819.md",
        OUT / "01_stock_verification/no_output_verification.tsv": stage / "no_output_verification.tsv",
        OUT / "01_stock_verification/no_output_verification_report.md": stage / "no_output_verification_report.md",
        OUT / "02_atlas_stock_no_output/atlas_no_output_summary.tsv": stage / "atlas_no_output_summary.tsv",
        OUT / "02_atlas_stock_no_output/atlas_stage_decomposition.tsv": stage / "atlas_stage_decomposition.tsv",
        OUT / "02_atlas_stock_no_output/atlas_numerical_identity.tsv": stage / "atlas_numerical_identity.tsv",
        OUT / "03_mdcath_stock_no_output/mdcath_no_output_summary.tsv": stage / "mdlovofit_all_domain_no_output_summary.tsv",
        OUT / "06_analysis/mdlovofit_stage_decomposition.tsv": stage / "mdlovofit_stage_decomposition.tsv",
        OUT / "06_analysis/core_compute_speedup.tsv": stage / "core_compute_speedup.tsv",
        OUT / "06_analysis/contract_comparison_table.tsv": stage / "contract_comparison_table.tsv",
        OUT / "06_analysis/atlas_contract_speedup_detail.tsv": stage / "atlas_contract_speedup_detail.tsv",
        OUT / "06_analysis/speedup_claim_recommendations.tsv": stage / "speedup_claim_recommendations.tsv",
        OUT / "06_analysis/manuscript_wording_recommendations.md": stage / "manuscript_wording_recommendations.md",
        OUT / "06_analysis/limitations.md": stage / "limitations.md",
        OUT / "06_analysis/quality_timing_integrity_audit.md": stage / "quality_timing_integrity_audit.md",
        OUT / "06_analysis/failures_timeouts.tsv": stage / "failures_timeouts.tsv",
        OUT / "06_analysis/warning_failure_audit.tsv": stage / "warning_failure_audit.tsv",
        OUT / "04_instrumented_kernel/instrumentation_validation.md": stage / "instrumentation_validation.md",
        OUT / "04_instrumented_kernel/instrumentation_patch.diff": stage / "instrumentation_patch.diff",
        OUT / "04_instrumented_kernel/numerical_identity.tsv": stage / "instrumented_numerical_identity.tsv",
        OUT / "07_validation/focused_test_output.log": stage / "validation/focused_test_output.log",
        OUT / "07_validation/focused_test_initial_environment_failure.log": stage / "validation/focused_test_initial_environment_failure.log",
        OUT / "07_validation/full_test_output.log": stage / "validation/full_test_output.log",
    }
    for source, destination in mappings.items():
        copy_file(source, destination)

    for source in sorted((OUT / "06_analysis/figures").glob("*")):
        copy_file(source, stage / "figures" / source.name)

    sources = [
        ROOT / "scripts/no_materialisation_benchmark_v1/common.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_nm0_registry.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_nm1_verification.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_nm2_parity.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_nm3_instrumentation_validation.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_nm4_atlas.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/run_nm5_mdcath.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/finalize_nm6_nm7.py",
        ROOT / "scripts/no_materialisation_benchmark_v1/build_nm8_review_bundle.py",
        ROOT / "tests/no_materialisation_v1/test_benchmark_minimal.py",
        ROOT / "src/mdcorealign/benchmark_minimal.py",
    ]
    for source in sources:
        copy_file(source, stage / "source" / source.relative_to(ROOT))

    patch_lines: list[str] = []
    for source in sources:
        relative = source.relative_to(ROOT).as_posix()
        content = sanitise_text(source.read_text(errors="replace")).splitlines(keepends=True)
        patch_lines.extend(
            difflib.unified_diff(
                [],
                content,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    (stage / "unified_source_diff.patch").write_text("".join(patch_lines))


def build_status_and_metrics(stage: Path) -> None:
    phases = [
        {"phase": "NM0", "status": "PASS", "evidence": "contract_registry.md; environment_manifest.tsv", "notes": "Frozen before timing."},
        {"phase": "NM1", "status": "PASS", "evidence": "no_output_verification.tsv", "notes": "5/5 stock no-output identity cases."},
        {"phase": "NM2", "status": "PASS", "evidence": "source/tests/no_materialisation_v1/test_benchmark_minimal.py", "notes": "Compact MDCoreAlign path parity."},
        {"phase": "NM3", "status": "PASS", "evidence": "instrumentation_validation.md", "notes": "5/5 stdout-byte and numerical identity cases."},
        {"phase": "NM4", "status": "PASS", "evidence": "atlas_no_output_summary.tsv", "notes": "48/48 ATLAS settings."},
        {"phase": "NM5", "status": "PASS", "evidence": "mdlovofit_all_domain_no_output_summary.tsv", "notes": "5,397/5,397 corrected mdCATH domains."},
        {"phase": "NM6", "status": "SKIPPED_BY_SCOPE", "evidence": "superseding_nm_ma_scope_20260819.md", "notes": "Optional deployment extension not executed."},
        {"phase": "NM7", "status": "PASS", "evidence": "final_no_materialisation_decision_report.md", "notes": "C1/C2 interpretation complete."},
        {"phase": "NM8", "status": "PASS", "evidence": "sha256_manifest.tsv", "notes": "Archive builder validates extraction and hashes before success."},
    ]
    write_tsv(stage / "phase_status.tsv", phases)

    contracts = read_tsv(OUT / "06_analysis/contract_comparison_table.tsv")
    metrics: list[dict[str, object]] = []
    for row in contracts:
        metrics.append(
            {
                "panel": row["panel"],
                "contract": row["contract"],
                "comparison": row["comparison"],
                "paired_n": row["paired_domains"],
                "median_speedup": row["paired_median_speedup"],
                "bootstrap_95CI_lower": row["paired_median_bootstrap_95CI_lower"],
                "bootstrap_95CI_upper": row["paired_median_bootstrap_95CI_upper"],
                "faster_fraction": row["MDCoreAlign_faster_fraction"],
            }
        )
    write_tsv(stage / "key_metrics.tsv", metrics)

    (stage / "known_limitations.md").write_text(
        "# Known Limitations\n\n"
        "- C1 and C2 are different timing contracts and their speedups must not be combined.\n"
        "- The previous approximately 70-fold result included unequal materialisation and is not an algorithm-only result.\n"
        "- The instrumented binary is used only for C2 stage timing; stock MDLovoFit is the C1 comparator.\n"
        "- The compact NM paths omit RMSF, diagnostics and aligned-coordinate writing.\n"
        "- The 82 warning rows are successful runs with strict historical native-value identity warnings, not failures or timeouts.\n"
        "- Runtime results are hardware- and workload-specific and do not support a universal-fastest claim.\n"
    )

    (stage / "00_README_FIRST.md").write_text(
        "# NM8 Review Handoff\n\n"
        "This archive contains the completed no-materialisation timing evidence for NM0-NM8. Start with `final_no_materialisation_decision_report.md`, `phase_status.tsv`, and `key_metrics.tsv`.\n\n"
        "## Canonical environment\n\n"
        "Python 3.11.4, NumPy 1.26.0, CuPy 13.6.0, MDLovoFit 20.1.1, and two CPU threads for MKL/OpenBLAS/OMP/NumExpr. The stock MDLovoFit executable SHA-256 is recorded in `environment_manifest.tsv`.\n\n"
        "## Contracts\n\n"
        "C1 is a stock, same-PDB, no-aligned-output tool workflow comparison. C2 is a timing-instrumented loaded core-compute comparison. The larger C2 ratios are not full-workflow speedups.\n\n"
        "## Reproducible validation commands\n\n"
        "```bash\n"
        "PYTHONPATH=src .venv-gpu-cu12/bin/python -m pytest -q tests/no_materialisation_v1/test_benchmark_minimal.py\n"
        "PYTHONPATH=src .venv-gpu-cu12/bin/python -m pytest -q\n"
        ".venv-gpu-cu12/bin/python -m scripts.no_materialisation_benchmark_v1.finalize_nm6_nm7\n"
        ".venv-gpu-cu12/bin/python -m scripts.no_materialisation_benchmark_v1.build_nm8_review_bundle\n"
        "```\n\n"
        "The complete suite passed 307 tests with 13 Matplotlib/Pyparsing deprecation warnings. The initial focused-test environment failure is retained separately because the installed package was stale; the canonical src-layout command passed.\n\n"
        "`sha256_manifest.tsv` intentionally excludes its own recursive hash. The SHA-256 of the complete archive protects that file. Raw trajectories, coordinate caches, environments and secrets are excluded.\n"
    )


def scan_stage(stage: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    secret_patterns = {
        "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "credential_assignment": re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]+"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    }
    path_patterns = {
        "workspace_absolute_path": re.compile(r"/storage/junjie/"),
        "user_home_path": re.compile(r"/home/jjxu/"),
    }
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        for label, pattern in {**secret_patterns, **path_patterns}.items():
            if pattern.search(text):
                findings.append({"file": path.relative_to(stage).as_posix(), "finding": label})
    return findings


def make_manifests(stage: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            continue
        relative = path.relative_to(stage).as_posix()
        section = relative.split("/", 1)[0] if "/" in relative else "entry"
        rows.append(
            {
                "relative_path": relative,
                "section": section,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    rows.extend(
        [
            {"relative_path": "file_manifest.tsv", "section": "entry", "size_bytes": "self", "sha256": "see_sha256_manifest"},
            {"relative_path": "sha256_manifest.tsv", "section": "entry", "size_bytes": "self", "sha256": "self_excluded_by_definition"},
        ]
    )
    write_tsv(stage / "file_manifest.tsv", rows)

    hash_rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.tsv":
            hash_rows.append(
                {
                    "relative_path": path.relative_to(stage).as_posix(),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_tsv(stage / "sha256_manifest.tsv", hash_rows)


def validate_archive(archive: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="nm8_verify_") as temporary:
        target = Path(temporary)
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            if any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members):
                raise RuntimeError("archive contains an unsafe member path")
            handle.extractall(target)
        roots = {path.name for path in target.iterdir()}
        if roots != {"handoff_to_chatgpt"}:
            raise RuntimeError(f"unexpected archive roots: {roots}")
        extracted = target / "handoff_to_chatgpt"
        hashes = read_tsv(extracted / "sha256_manifest.tsv")
        for row in hashes:
            path = extracted / row["relative_path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"manifest verification failed: {row['relative_path']}")
        actual = {path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()}
        expected = {row["relative_path"] for row in hashes} | {"sha256_manifest.tsv"}
        if actual != expected:
            raise RuntimeError(f"archive file-set mismatch: missing={expected-actual}, extra={actual-expected}")
        return {
            "archive_file_count": len(actual),
            "manifest_rows_verified": len(hashes),
            "single_root": "handoff_to_chatgpt",
        }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nm8_stage_", dir=OUT) as temporary:
        stage = Path(temporary) / "handoff_to_chatgpt"
        stage.mkdir()
        copy_required(stage)
        build_status_and_metrics(stage)

        excluded = [path for path in stage.rglob("*") if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES]
        oversized = [path for path in stage.rglob("*") if path.is_file() and path.stat().st_size > 100 * 1024 * 1024]
        findings = scan_stage(stage)
        if excluded or oversized or findings:
            raise RuntimeError(
                f"bundle safety scan failed: excluded={excluded}, oversized={oversized}, findings={findings}"
            )

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
    (OUT / "07_validation/nm8_archive_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "07_validation/nm8_status.json").write_text(
        json.dumps(
            {
                "archive_sha256": validation["archive_sha256"],
                "file_count": validation["archive_file_count"],
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
