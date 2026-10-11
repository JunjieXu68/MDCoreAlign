#!/usr/bin/env python3
"""Assemble and independently validate the mandatory AT10 review archive."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    OUT,
    ROOT,
    read_tsv,
    sha256,
    write_json,
    write_text,
    write_tsv,
)

UTC = timezone.utc
VALIDATION_LOGS = {"focused_pytest.txt", "full_pytest.txt", "ruff.txt", "compileall.txt"}


def unused_path(path: Path) -> Path:
    """Keep any earlier review artifact intact when assembling another bundle."""

    if not path.exists():
        return path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}_{stamp}{suffix}")


def verify_validation() -> None:
    """Require recorded success for every AT10 check, including the full suite."""

    source = OUT / "09_report/validation_results.json"
    results = json.loads(source.read_text())
    checks = results["checks"]
    if (
        {row["log"] for row in checks} != VALIDATION_LOGS
        or len(checks) != len(VALIDATION_LOGS)
        or any(row["returncode"] != 0 for row in checks)
    ):
        raise ValueError(f"AT10 checks are incomplete or failed: {source}")
    for filename in VALIDATION_LOGS:
        path = OUT / "09_report/validation" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing AT10 validation evidence: {path}")


def verify_frozen_contracts() -> None:
    """Read the original panel and preregistration hashes without rewriting them."""

    for manifest in (
        OUT / "01_panel/atlas_48_setting_manifest.sha256",
        OUT / "02_preregistration/preregistration.sha256",
    ):
        for line in manifest.read_text().splitlines():
            expected, filename = line.split(maxsplit=1)
            source = manifest.parent / filename
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"frozen contract hash mismatch: {source}")


def copy_file(source: Path, destination: Path) -> None:
    """Copy one required artifact and fail clearly if it is missing."""

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def phase_status() -> list[dict[str, str]]:
    """Build compact phase status from required artifacts."""

    checks = {
        "AT0": OUT / "00_freeze/software_manifest.tsv",
        "AT1": OUT / "01_panel/atlas_48_setting_manifest.tsv",
        "AT2": OUT / "02_preregistration/preregistration.sha256",
        "AT3": OUT / "03_thread_affinity_audit/thread_contract_decision.md",
        "AT4": OUT / "04_preflight/preflight_decision.md",
        "AT5": OUT / "05_timing/timing_completion_report.md",
        "AT6": OUT / "06_core_extraction/core_extraction_report.md",
        "AT7": OUT / "07_analysis/panel_speedup_summary.tsv",
        "AT8": OUT / "08_figures/figure3_core_agreement.pdf",
        "AT9": OUT / "09_report/final_atlas_single_thread_report.md",
        "AT10": OUT / "09_report/validation_summary.md",
    }
    return [
        {
            "phase": phase,
            "status": "complete" if path.is_file() else "missing",
            "evidence": str(path.relative_to(OUT)),
        }
        for phase, path in checks.items()
    ]


def key_metrics() -> list[dict[str, str]]:
    """Flatten final runtime and agreement summaries."""

    rows = []
    for source in (
        OUT / "07_analysis/panel_speedup_summary.tsv",
        OUT / "07_analysis/core_agreement_summary.tsv",
    ):
        values = read_tsv(source)[0]
        rows.extend(
            {
                "metric": key,
                "value": value,
                "source": str(source.relative_to(OUT)),
            }
            for key, value in values.items()
        )
    return rows


def build_handoff() -> Path:
    """Populate the self-contained handoff tree from frozen result files."""

    handoff = unused_path(OUT / "handoff_to_chatgpt")
    handoff.mkdir(parents=True)
    required = {
        "final_atlas_single_thread_report.md": OUT
        / "09_report/final_atlas_single_thread_report.md",
        "atlas_48_setting_manifest.tsv": OUT / "01_panel/atlas_48_setting_manifest.tsv",
        "atlas_48_setting_manifest.sha256": OUT / "01_panel/atlas_48_setting_manifest.sha256",
        "benchmark_contract.md": OUT / "02_preregistration/benchmark_contract.md",
        "statistical_plan.md": OUT / "02_preregistration/statistical_plan.md",
        "core_agreement_contract.md": OUT / "02_preregistration/core_agreement_contract.md",
        "preregistration.sha256": OUT / "02_preregistration/preregistration.sha256",
        "command_contract.tsv": OUT / "02_preregistration/command_contract.tsv",
        "run_order.tsv": OUT / "02_preregistration/run_order.tsv",
        "software_manifest.tsv": OUT / "00_freeze/software_manifest.tsv",
        "executable_hashes.tsv": OUT / "00_freeze/executable_hashes.tsv",
        "hardware_manifest.tsv": OUT / "00_freeze/hardware_manifest.tsv",
        "environment_manifest.tsv": OUT / "00_freeze/environment_manifest.tsv",
        "pip_freeze.txt": OUT / "00_freeze/pip_freeze.txt",
        "numpy_blas_config.txt": OUT / "00_freeze/numpy_blas_config.txt",
        "mdlovofit_help.txt": OUT / "00_freeze/mdlovofit_help.txt",
        "thread_environment.tsv": OUT / "03_thread_affinity_audit/thread_environment.tsv",
        "affinity_contract.tsv": OUT / "03_thread_affinity_audit/affinity_contract.tsv",
        "threadpoolctl.json": OUT / "03_thread_affinity_audit/threadpoolctl.json",
        "mdlovofit_thread_trace.tsv": OUT / "03_thread_affinity_audit/mdlovofit_thread_trace.tsv",
        "thread_contract_decision.md": OUT / "03_thread_affinity_audit/thread_contract_decision.md",
        "cpu_topology.tsv": OUT / "03_thread_affinity_audit/cpu_topology.tsv",
        "machine_state.tsv": OUT / "03_thread_affinity_audit/machine_state.tsv",
        "process_tree.txt": OUT / "03_thread_affinity_audit/process_tree.txt",
        "preflight_decision.md": OUT / "04_preflight/preflight_decision.md",
        "preflight_results.tsv": OUT / "04_preflight/preflight_results.tsv",
        "raw_timing_records.tsv": OUT / "05_timing/raw_timing_records.tsv",
        "failures_timeouts.tsv": OUT / "05_timing/failures_timeouts.tsv",
        "timing_completion_report.md": OUT / "05_timing/timing_completion_report.md",
        "timing_summary_per_setting.tsv": OUT / "07_analysis/timing_summary_per_setting.tsv",
        "timing_summary_per_system.tsv": OUT / "07_analysis/timing_summary_per_system.tsv",
        "timing_summary_per_prefix.tsv": OUT / "07_analysis/timing_summary_per_prefix.tsv",
        "panel_speedup_summary.tsv": OUT / "07_analysis/panel_speedup_summary.tsv",
        "repeat_variability.tsv": OUT / "07_analysis/repeat_variability.tsv",
        "order_effects.tsv": OUT / "07_analysis/order_effects.tsv",
        "mdlovofit_self_consistency.tsv": OUT / "06_core_extraction/mdlovofit_self_consistency.tsv",
        "core_extraction_report.md": OUT / "06_core_extraction/core_extraction_report.md",
        "core_mapping_audit.tsv": OUT / "06_core_extraction/core_mapping_audit.tsv",
        "mdcorealign_weights.tsv": OUT / "06_core_extraction/mdcorealign_weights.tsv",
        "mdcorealign_cores.tsv": OUT / "06_core_extraction/mdcorealign_cores.tsv",
        "mdlovofit_cores.tsv": OUT / "06_core_extraction/mdlovofit_cores.tsv",
        "core_agreement_per_setting.tsv": OUT / "06_core_extraction/core_agreement_per_setting.tsv",
        "core_agreement_summary.tsv": OUT / "07_analysis/core_agreement_summary.tsv",
        "speed_agreement_associations.tsv": OUT / "07_analysis/speed_agreement_associations.tsv",
        "bootstrap_results.tsv": OUT / "07_analysis/bootstrap_results.tsv",
        "manuscript_numbers.tsv": OUT / "09_report/manuscript_numbers.tsv",
        "claim_evidence_matrix.tsv": OUT / "09_report/claim_evidence_matrix.tsv",
        "methods_text.md": OUT / "09_report/methods_text.md",
        "results_text.md": OUT / "09_report/results_text.md",
        "limitations_text.md": OUT / "09_report/limitations_text.md",
        "validation_summary.md": OUT / "09_report/validation_summary.md",
        "validation_results.json": OUT / "09_report/validation_results.json",
    }
    for destination, source in required.items():
        copy_file(source, handoff / destination)
    for source in (OUT / "08_figures").rglob("*"):
        if source.is_file():
            copy_file(source, handoff / "figures" / source.relative_to(OUT / "08_figures"))
    failed_audit = OUT / "03_thread_affinity_audit/attempt_1_failed"
    for source in failed_audit.rglob("*"):
        if source.is_file():
            copy_file(source, handoff / "attempt_1_failed" / source.relative_to(failed_audit))
    for directory in ("parser_recovery_audit", "export_records", "export_attempts", "prefetch_audit"):
        audit = OUT / "06_core_extraction" / directory
        for source in audit.rglob("*"):
            if (
                source.is_file()
                and source.suffix.lower()
                in {".md", ".txt", ".tsv", ".csv", ".json", ".jsonl", ".sha256", ".py"}
                and source.stat().st_size <= 8 * 1024 * 1024
            ):
                copy_file(
                    source,
                    handoff / "core_extraction_audit" / directory / source.relative_to(audit),
                )
    for source in (ROOT / "scripts/atlas_strict_single_thread_cpu_v1").glob("*.py"):
        copy_file(source, handoff / "new_benchmark_code" / source.name)
    copy_file(
        ROOT / "tests/test_atlas_strict_single_thread_cpu_v1.py",
        handoff / "tests/test_atlas_strict_single_thread_cpu_v1.py",
    )
    copy_file(
        ROOT / "tests/test_atlas_untimed_export_recovery.py",
        handoff / "tests/test_atlas_untimed_export_recovery.py",
    )
    for source in (OUT / "09_report/validation").glob("*"):
        if source.is_file():
            copy_file(source, handoff / "tests" / source.name)
    write_tsv(handoff / "phase_status.tsv", phase_status())
    write_tsv(handoff / "key_metrics.tsv", key_metrics())
    write_text(
        handoff / "00_README_FIRST.md",
        """# MDCoreAlign strict single-thread ATLAS review handoff

This directory contains the complete evidence needed to review the locked
48-setting CPU timing and untimed 60% core-agreement study. Formal runtime uses
the common centred CA-only PDB, one logical CPU, verified one-thread numerical
pools, no aligned-coordinate materialisation, one warm-up, five repeats, warm
cache, and a prospectively counterbalanced order. Core extraction is separate
and untimed. Neither method's core is treated as biological ground truth.

Start with `final_atlas_single_thread_report.md`, then inspect the raw timing,
thread decision, panel manifest, bootstrap table, and core-agreement table.
""",
    )
    files = sorted(path for path in handoff.rglob("*") if path.is_file())
    manifest_rows = [
        {
            "path": str(path.relative_to(handoff)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
        if path.name != "sha256_manifest.tsv"
    ]
    write_tsv(handoff / "sha256_manifest.tsv", manifest_rows)
    return handoff


def archive_and_validate(handoff: Path) -> dict[str, object]:
    """Create the dated archive and verify it in an independent extraction."""

    archive = unused_path(
        OUT / f"atlas_strict_single_thread_cpu_v1_review_bundle_{date.today():%Y%m%d}.tar.gz"
    )
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(handoff, arcname="handoff_to_chatgpt", recursive=True)
    with tempfile.TemporaryDirectory(prefix="mdca_at10_verify_") as temporary:
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(temporary, filter="data")
        extracted = Path(temporary) / "handoff_to_chatgpt"
        rows = read_tsv(extracted / "sha256_manifest.tsv")
        expected = {row["path"] for row in rows}
        actual = {
            str(path.relative_to(extracted))
            for path in extracted.rglob("*")
            if path.is_file() and path.name != "sha256_manifest.tsv"
        }
        if len(expected) != len(rows) or expected != actual:
            raise ValueError("archive file set differs from its SHA-256 manifest")
        mismatches = []
        for row in rows:
            path = extracted / row["path"]
            if (
                not path.is_file()
                or path.stat().st_size != int(row["size_bytes"])
                or sha256(path) != row["sha256"]
            ):
                mismatches.append(row["path"])
        if mismatches:
            raise ValueError(f"archive manifest verification failed: {mismatches}")
        extracted_count = len([path for path in extracted.rglob("*") if path.is_file()])
    return {
        "archive_path": str(archive.relative_to(ROOT)),
        "archive_size_bytes": archive.stat().st_size,
        "archive_file_count": extracted_count,
        "archive_sha256": sha256(archive),
        "manifest_verification": "pass",
        "extraction_verification": "pass",
        "generated_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    statuses = phase_status()
    missing = [row for row in statuses if row["status"] != "complete"]
    if missing:
        raise ValueError(f"cannot package incomplete study: {missing}")
    verify_validation()
    verify_frozen_contracts()
    handoff = build_handoff()
    result = archive_and_validate(handoff)
    result["handoff_path"] = str(handoff.relative_to(ROOT))
    write_json(unused_path(OUT / "09_report/review_archive_metadata.json"), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
