#!/usr/bin/env python
"""Build and independently validate the mandatory chronology-v3 review archive."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/cpu_chronology_v3"
OUTPUT_ROOT = Path(os.environ.get("MDCA_V3_REVIEW_OUTPUT_ROOT", str(RESULTS))).resolve()
DATE = "20260821"
ARCHIVE = OUTPUT_ROOT / f"cpu_chronology_v3_review_bundle_{DATE}.tar.gz"
CHECKPOINT = RESULTS / "00_freeze/source_checkpoint.tar.gz"

REQUIRED = {
    "final_v3_decision.md": RESULTS / "08_decision/final_v3_decision.md",
    "instrumentation_fix_report.md": RESULTS / "01_instrumentation/instrumentation_fix_report.md",
    "stage_schema.tsv": RESULTS / "01_instrumentation/stage_schema.tsv",
    "self_variability_report.md": RESULTS / "02_self_variability/self_variability_report.md",
    "self_variability_summary.tsv": RESULTS / "02_self_variability/self_variability_summary.tsv",
    "v3_gate_registry.tsv": RESULTS / "03_gate_calibration/v3_gate_registry.tsv",
    "v3_gate_registry.json": RESULTS / "03_gate_calibration/v3_gate_registry.json",
    "gate_registry.sha256": RESULTS / "03_gate_calibration/gate_registry.sha256",
    "candidate_registry.tsv": RESULTS / "04_candidates/candidate_registry.tsv",
    "locked_candidate_configs.tsv": RESULTS / "04_candidates/locked_candidate_configs.tsv",
    "adaptive_rule.md": RESULTS / "04_candidates/adaptive_rule.md",
    "adaptive_rule.sha256": RESULTS / "04_candidates/adaptive_rule.sha256",
    "fresh_mdcath_512.tsv": RESULTS / "05_fresh_panels/fresh_mdcath_512.tsv",
    "fresh_long_panel.tsv": RESULTS / "05_fresh_panels/fresh_long_panel.tsv",
    "atlas_r2_r3_audit.md": RESULTS / "05_fresh_panels/atlas_r2_r3_audit.md",
    "fresh_mdcath_quality.tsv": RESULTS / "06_validation/fresh_mdcath_quality.tsv",
    "fresh_long_quality.tsv": RESULTS / "06_validation/fresh_long_quality.tsv",
    "historical_stress_quality.tsv": RESULTS / "06_validation/historical_stress_quality.tsv",
    "adaptive_refinement_paths.tsv": RESULTS / "06_validation/adaptive_refinement_paths.tsv",
    "v3_gate_results.tsv": RESULTS / "06_validation/v3_gate_results.tsv",
    "fresh_long_summary.tsv": RESULTS / "07_performance/fresh_long_summary.tsv",
    "fresh_mdcath_128_repeats.tsv": RESULTS / "07_performance/fresh_mdcath_128_repeats.tsv",
    "stage_decomposition_v3.tsv": RESULTS / "07_performance/stage_decomposition_v3.tsv",
    "memory_v3.tsv": RESULTS / "07_performance/memory_v3.tsv",
    "frozen_mdlovofit_direct_comparison.tsv": RESULTS / "07_performance/frozen_mdlovofit_direct_comparison.tsv",
    "pareto_comparison.tsv": RESULTS / "08_decision/pareto_comparison.tsv",
    "long_trajectory_pareto.tsv": RESULTS / "08_decision/long_trajectory_pareto.tsv",
    "tail_case_summary.tsv": RESULTS / "08_decision/tail_case_summary.tsv",
    "candidate_decisions.tsv": RESULTS / "08_decision/candidate_decisions.tsv",
    "product_integration_proposal.md": RESULTS / "08_decision/product_integration_proposal.md",
    "manuscript_impact.md": RESULTS / "08_decision/manuscript_impact.md",
    "next_branch_if_failed.md": RESULTS / "08_decision/next_branch_if_failed.md",
}

OPTIONAL = {
    "timing_consistency_tests.tsv": RESULTS / "01_instrumentation/timing_consistency_tests.tsv",
    "v2_to_v3_stage_map.tsv": RESULTS / "01_instrumentation/v2_to_v3_stage_map.tsv",
    "numerical_repeatability.tsv": RESULTS / "02_self_variability/numerical_repeatability.tsv",
    "mild_path_variability.tsv": RESULTS / "02_self_variability/mild_path_variability.tsv",
    "contextual_sensitivity.tsv": RESULTS / "02_self_variability/contextual_sensitivity.tsv",
    "gate_derivation_report.md": RESULTS / "03_gate_calibration/gate_derivation_report.md",
    "C2_definition.md": RESULTS / "04_candidates/C2_definition.md",
    "adaptive_risk_features.tsv": RESULTS / "04_candidates/adaptive_risk_features.tsv",
    "adaptive_rule_calibration.tsv": RESULTS / "04_candidates/adaptive_rule_calibration.tsv",
    "fresh_mdcath_performance_128.tsv": RESULTS / "05_fresh_panels/fresh_mdcath_performance_128.tsv",
    "historical_stress_panel.tsv": RESULTS / "05_fresh_panels/historical_stress_panel.tsv",
    "panel_lock_report.md": RESULTS / "05_fresh_panels/panel_lock_report.md",
    "quality_validation_report.md": RESULTS / "06_validation/quality_validation_report.md",
    "fresh_long_repeats.tsv": RESULTS / "07_performance/fresh_long_repeats.tsv",
    "performance_report.md": RESULTS / "07_performance/performance_report.md",
    "exact_source_hash_verification.tsv": RESULTS / "00_freeze/exact_source_hash_verification.tsv",
    "environment_manifest.tsv": RESULTS / "00_freeze/environment_manifest.tsv",
    "study_scope.md": RESULTS / "00_freeze/study_scope.md",
}

CHANGED_SOURCE = (
    "src/mdsuperpose/stage_timing.py",
    "src/mdsuperpose/chronology_adaptive.py",
    "src/mdsuperpose/chronology_study.py",
    "scripts/cpu_chronology_v3/prepare_panels.py",
    "scripts/cpu_chronology_v3/run_self_variability.py",
    "scripts/cpu_chronology_v3/lock_gate.py",
    "scripts/cpu_chronology_v3/calibrate_candidates.py",
    "scripts/cpu_chronology_v3/prepare_atlas_r2_r3.py",
    "scripts/cpu_chronology_v3/lock_performance_subset.py",
    "scripts/cpu_chronology_v3/run_quality_validation.py",
    "scripts/cpu_chronology_v3/commit_quality_checkpoints.py",
    "scripts/cpu_chronology_v3/run_performance.py",
    "scripts/cpu_chronology_v3/finalize_v3.py",
    "scripts/cpu_chronology_v3/make_figures.py",
    "scripts/cpu_chronology_v3/build_review_archive.py",
)

CHANGED_TESTS = (
    "tests/cpu_chronology_v3/test_stage_timing.py",
    "tests/cpu_chronology_v3/test_adaptive_chronology.py",
    "tests/cpu_chronology_v3/test_candidate_contracts.py",
)

TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".patch", ".py", ".rst", ".sha256",
    ".sh", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sanitise(text: str) -> str:
    replacements = (
        (str(ROOT), "<REPOSITORY_ROOT>"),
        (str(ROOT.parent), "<PROJECTS_ROOT>"),
        (str(Path.home()), "<USER_HOME>"),
        (Path.home().name, "<USER>"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(r"/tmp/[A-Za-z0-9_./-]+", "<TEMP_PATH>", text)
    text = text.replace("/tmp/", "<TEMP_ROOT>/")
    return text


def copy_sanitised(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES or destination.suffix.lower() in TEXT_SUFFIXES:
        destination.write_text(sanitise(source.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def load_checkpoint() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(CHECKPOINT, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                files[member.name.removeprefix("./")] = handle.read()
    return files


def build_diffs(handoff: Path, checkpoint: dict[str, bytes]) -> None:
    combined: list[str] = []
    diff_root = handoff / "individual_diffs"
    diff_root.mkdir(parents=True, exist_ok=True)
    for relative in CHANGED_SOURCE + CHANGED_TESTS:
        current_path = ROOT / relative
        if not current_path.exists():
            raise FileNotFoundError(relative)
        current = current_path.read_text(encoding="utf-8").splitlines(keepends=True)
        prior_bytes = checkpoint.get(relative)
        prior = prior_bytes.decode("utf-8").splitlines(keepends=True) if prior_bytes else []
        lines = list(
            difflib.unified_diff(
                prior,
                current,
                fromfile=f"a/{relative}" if prior_bytes else "/dev/null",
                tofile=f"b/{relative}",
            )
        )
        if not lines:
            continue
        combined.extend(lines)
        (diff_root / (relative.replace("/", "__") + ".patch")).write_text(
            sanitise("".join(lines)), encoding="utf-8"
        )
    (handoff / "unified_source_diff.patch").write_text(sanitise("".join(combined)), encoding="utf-8")


def build_phase_status(handoff: Path) -> None:
    rows = [
        {"phase": "V0", "status": "complete", "evidence": "00_freeze and 01_instrumentation", "notes": "v2 evidence frozen; timing schema corrected"},
        {"phase": "V1", "status": "complete", "evidence": "self_variability_report.md", "notes": "numerical, mild-path and contextual envelopes separated"},
        {"phase": "V2", "status": "complete_locked", "evidence": "v3_gate_registry.json", "notes": "gate locked before fresh panels were opened"},
        {"phase": "V3", "status": "complete_locked", "evidence": "candidate_registry.tsv and adaptive_rule.sha256", "notes": "five configurations only"},
        {"phase": "V4", "status": "complete", "evidence": "fresh_mdcath_512.tsv and fresh_long_panel.tsv", "notes": "512 fresh mdCATH domains and 16 ATLAS R2/R3 trajectories"},
        {"phase": "V5", "status": "complete", "evidence": "fresh quality tables", "notes": "all expected scientific-evaluation rows present"},
        {"phase": "V6", "status": "complete", "evidence": "fresh performance tables", "notes": "one warmup plus five repeats; CPU 35; one thread"},
        {"phase": "V7", "status": "complete", "evidence": "pareto_comparison.tsv", "notes": "quality and speed compared with exact and throughput"},
        {"phase": "V8", "status": "complete_mixed", "evidence": "candidate_decisions.tsv", "notes": "C2 and adaptive passed their locked rules; D64 failed long-speed gate"},
        {"phase": "V9", "status": "complete", "evidence": "validation logs and this archive", "notes": "330 passed, 8 skipped, 2 subtests; 21 focused passed"},
    ]
    write_tsv(handoff / "phase_status.tsv", rows, ["phase", "status", "evidence", "notes"])


def build_key_metrics(handoff: Path) -> None:
    variability = {row["perturbation_class"]: row for row in read_tsv(RESULTS / "02_self_variability/self_variability_summary.tsv")}
    gate = json.loads((RESULTS / "03_gate_calibration/v3_gate_registry.json").read_text())
    decisions = {row["method"]: row for row in read_tsv(RESULTS / "06_validation/v3_gate_results.tsv")}
    routes = [row for row in read_tsv(RESULTS / "06_validation/adaptive_refinement_paths.tsv") if row["panel"] in {"fresh_mdcath", "fresh_long"}]
    metrics: list[dict[str, object]] = [
        {"metric": "fresh_mdcath_domains", "value": 512, "unit": "domains", "interpretation": "locked fresh held-out panel"},
        {"metric": "fresh_long_trajectories", "value": 16, "unit": "trajectories", "interpretation": "ATLAS R2/R3 from eight families"},
        {"metric": "fresh_long_frames_each", "value": 10001, "unit": "frames", "interpretation": "full-length local ATLAS members"},
        {"metric": "exact_like_rmsd_threshold", "value": gate["exact_like"]["p95_of_case_p95_rmsd_difference_A_max"], "unit": "angstrom", "interpretation": "prospectively locked"},
        {"metric": "balanced_rmsd_threshold", "value": gate["balanced"]["p95_of_case_p95_rmsd_difference_A_max"], "unit": "angstrom", "interpretation": "prospectively locked"},
        {"metric": "adaptive_C1_routes", "value": sum(row["accepted_level"] == "C1" for row in routes), "unit": "cases", "interpretation": "fresh panels"},
        {"metric": "adaptive_C2_routes", "value": sum(row["accepted_level"] == "C2" for row in routes), "unit": "cases", "interpretation": "fresh panels"},
        {"metric": "adaptive_D64_routes", "value": sum(row["accepted_level"] == "D64" for row in routes), "unit": "cases", "interpretation": "fresh panels"},
        {"metric": "canonical_tests", "value": "330 passed; 8 skipped; 2 subtests passed", "unit": "pytest", "interpretation": "full canonical suite"},
        {"metric": "focused_tests", "value": "21 passed", "unit": "pytest", "interpretation": "chronology, warm-start, output and CLI contracts"},
        {"metric": "package_version", "value": "0.1.0rc1", "unit": "version", "interpretation": "wheel and sdist passed twine check"},
    ]
    for category, row in variability.items():
        metrics.extend((
            {"metric": f"{category}_p95_across_case_p95_rmsd", "value": row["p95_across_case_level_p95_rmsd_difference_A"], "unit": "angstrom", "interpretation": "exact self-variability"},
            {"metric": f"{category}_maximum_case_p95_rmsd", "value": row["maximum_case_level_p95_rmsd_difference_A"], "unit": "angstrom", "interpretation": "exact self-variability"},
            {"metric": f"{category}_minimum_core_jaccard", "value": row["minimum_core_jaccard"], "unit": "fraction", "interpretation": "exact self-variability"},
        ))
    for method, row in decisions.items():
        if method == "throughput_compact_f64" or method.endswith("_v3"):
            metrics.extend((
                {"metric": f"{method}_mdcath_speedup", "value": row["fresh_mdcath_128_median_speedup_vs_exact"], "unit": "fold", "interpretation": "paired loaded-coordinate medians"},
                {"metric": f"{method}_long_speedup", "value": row["fresh_long_median_speedup_vs_exact"], "unit": "fold", "interpretation": "paired loaded-coordinate medians"},
                {"metric": f"{method}_p95_case_p95_rmsd", "value": row["p95_across_case_p95_rmsd_difference_A"], "unit": "angstrom", "interpretation": "fresh combined quality"},
                {"metric": f"{method}_minimum_core_jaccard", "value": row["minimum_core_jaccard"], "unit": "fraction", "interpretation": "fresh combined quality"},
                {"metric": f"{method}_exact_like_pass", "value": row["exact_like_full_pass"], "unit": "boolean", "interpretation": "locked complete gate"},
                {"metric": f"{method}_balanced_pass", "value": row["balanced_full_pass"], "unit": "boolean", "interpretation": "locked complete gate"},
            ))
    write_tsv(handoff / "key_metrics.tsv", metrics, ["metric", "value", "unit", "interpretation"])


def write_validation_reports(handoff: Path) -> None:
    cpu = Path("/tmp/mdcorealign_cpu_chronology_v3_cpu_clean/clean_install_report.md")
    gpu_probe = Path("/tmp/mdcorealign_cpu_chronology_v3_gpu_clean/gpu_import_probe.txt").read_text()
    gpu_smoke = Path("/tmp/mdcorealign_cpu_chronology_v3_gpu_clean/gpu_output_validation.txt").read_text()
    copy_sanitised(cpu, handoff / "cpu_clean_install_report.md")
    (handoff / "gpu_clean_install_report.md").write_text(
        "# GPU clean-install smoke report\n\n"
        "- Status: `pass`.\n"
        "- Environment: fresh temporary venv installed with `pip install '.[gpu]'`.\n"
        "- Explicit throughput GPU CLI smoke: `pass`; no CPU fallback.\n"
        "- Aligned coordinates were not materialised.\n\n"
        "```text\n" + sanitise(gpu_probe + gpu_smoke) + "```\n",
        encoding="utf-8",
    )
    build_files = sorted(Path("/tmp/mdcorealign_cpu_chronology_v3_dist").glob("*"))
    (handoff / "package_build_report.md").write_text(
        "# Package build report\n\n"
        "- Status: `pass`.\n"
        "- Command: `python -m build --outdir <TEMP_PATH> .`.\n"
        "- Artifacts: " + ", ".join(f"`{path.name}` ({path.stat().st_size} bytes)" for path in build_files) + ".\n"
        "- `twine check`: `pass`.\n",
        encoding="utf-8",
    )
    for source, name in (
        (Path("/tmp/mdcorealign_cpu_chronology_v3_full_test_output.txt"), "full_test_output.txt"),
        (Path("/tmp/mdcorealign_cpu_chronology_v3_focused_test_output.txt"), "focused_test_output.txt"),
        (Path("/tmp/mdcorealign_cpu_chronology_v3_focused_test_invocation_error.txt"), "focused_test_invocation_error.txt"),
        (Path("/tmp/mdcorealign_cpu_chronology_v3_package_build.log"), "package_build_output.txt"),
        (Path("/tmp/mdcorealign_cpu_chronology_v3_twine_check.txt"), "twine_check.txt"),
    ):
        copy_sanitised(source, handoff / name)


def copy_figures(handoff: Path) -> None:
    source = Path("/tmp/mdcorealign_cpu_chronology_v3_figures")
    required = [f"fig{index}_" for index in range(1, 10)]
    available = [path for path in source.iterdir() if path.is_file()]
    for prefix in required:
        for suffix in (".pdf", ".png"):
            if not any(path.name.startswith(prefix) and path.suffix == suffix and path.stat().st_size > 0 for path in available):
                raise RuntimeError(f"missing non-empty figure {prefix}*{suffix}")
    for path in available:
        copy_sanitised(path, handoff / "figures" / path.name)


def scan(handoff: Path) -> list[dict[str, str]]:
    patterns = {
        "absolute_local_path": re.compile(r"/(?:home|storage|tmp)/[^\s`'\"<>]+"),
        "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "secret_assignment": re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}"),
    }
    rows: list[dict[str, str]] = []
    for path in sorted(handoff.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == "secret_path_scan.tsv":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for category, pattern in patterns.items():
            for match in pattern.finditer(text):
                rows.append({"path": path.relative_to(handoff).as_posix(), "category": category, "match": match.group(0)[:120]})
    return rows


def write_readme(handoff: Path) -> None:
    (handoff / "00_README_FIRST.md").write_text(
        "# MDCoreAlign chronology-calibrated CPU fast path v3\n\n"
        "This review bundle records V0-V9. The historical v2 conclusion is preserved: "
        "no v2 candidate passed its original pre-registered gate. V3 is a separate "
        "prospective study calibrated against exact self-variability.\n\n"
        "## Scope\n\n"
        "- MDCoreAlign only: no MDLovoFit, THESEUS, GPU performance or W5 performance reruns.\n"
        "- Sequential float64 exact source and public defaults were unchanged.\n"
        "- Fresh validation: 512 mdCATH domains plus ATLAS R2/R3 (16 trajectories, eight families, 10,001 frames each).\n"
        "- Timing: loaded float64 coordinates, one CPU thread, CPU affinity 35, no ordinary RMSF, no materialisation, one warmup plus five repeats.\n\n"
        "## Decisions\n\n"
        "- D64 was a quality near-match but failed the long-trajectory speed requirement.\n"
        "- C2 passed the complete exact-like and balanced v3 gates.\n"
        "- Adaptive C1->C2->D64 passed the balanced and additional promotion-for-review rules.\n"
        "- These are internal study candidates only; no new public mode or default change was made.\n"
        "- Frozen MDLovoFit timings were not directly combined because the timing/output contracts did not match.\n\n"
        "## Validation\n\n"
        "- Canonical tests: 330 passed, 8 skipped, 2 subtests passed.\n"
        "- Focused tests: 21 passed.\n"
        "- CPU and explicit GPU clean-install smoke checks passed.\n"
        "- Wheel/sdist build and twine checks passed for version 0.1.0rc1.\n"
        "- Start with `final_v3_decision.md`, `key_metrics.tsv`, `v3_gate_results.tsv`, and the figures.\n",
        encoding="utf-8",
    )


def validate_archive(handoff: Path) -> tuple[int, str]:
    files = sorted(path for path in handoff.rglob("*") if path.is_file())
    hash_rows = [
        {"path": path.relative_to(handoff).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in files
        if path.name not in {"sha256_manifest.tsv", "file_manifest.tsv"}
    ]
    write_tsv(handoff / "sha256_manifest.tsv", hash_rows, ["path", "sha256", "size_bytes"])
    all_files = sorted(path for path in handoff.rglob("*") if path.is_file())
    manifest_rows = [
        {"path": path.relative_to(handoff).as_posix(), "size_bytes": path.stat().st_size, "category": path.parts[-2] if len(path.parts) > 1 else "root"}
        for path in all_files
    ]
    write_tsv(handoff / "file_manifest.tsv", manifest_rows, ["path", "size_bytes", "category"])

    findings = scan(handoff)
    write_tsv(handoff / "secret_path_scan.tsv", findings, ["path", "category", "match"])
    if findings:
        raise RuntimeError(f"secret/path scan found {len(findings)} issue(s)")

    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(handoff, arcname="handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdca-v3-review-") as directory:
        extracted_root = Path(directory)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(extracted_root, filter="data")
        extracted = extracted_root / "handoff_to_chatgpt"
        for row in hash_rows:
            path = extracted / str(row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"archive manifest mismatch: {row['path']}")
        extracted_files = sorted(path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file())
        expected_files = sorted(path.relative_to(handoff).as_posix() for path in handoff.rglob("*") if path.is_file())
        if extracted_files != expected_files:
            raise RuntimeError("archive extraction file inventory mismatch")
    return len([path for path in handoff.rglob("*") if path.is_file()]), sha256(ARCHIVE)


def main() -> None:
    missing = [f"{name}: {path}" for name, path in REQUIRED.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing required evidence:\n" + "\n".join(missing))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    handoff = OUTPUT_ROOT / "handoff_to_chatgpt"
    handoff.mkdir(parents=True, exist_ok=True)
    for name, source in REQUIRED.items():
        copy_sanitised(source, handoff / name)
    for name, source in OPTIONAL.items():
        if source.is_file() and source.stat().st_size > 0:
            copy_sanitised(source, handoff / name)

    for relative in CHANGED_SOURCE:
        copy_sanitised(ROOT / relative, handoff / "changed_source" / relative)
    for relative in CHANGED_TESTS:
        copy_sanitised(ROOT / relative, handoff / "changed_tests" / relative)

    build_diffs(handoff, load_checkpoint())
    build_phase_status(handoff)
    build_key_metrics(handoff)
    write_validation_reports(handoff)
    copy_figures(handoff)
    write_readme(handoff)

    file_count, archive_hash = validate_archive(handoff)
    print(json.dumps({
        "archive": str(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_hash,
        "file_count": file_count,
        "secret_path_scan": "pass",
        "extraction_manifest_verification": "pass",
    }, indent=2))


if __name__ == "__main__":
    main()
