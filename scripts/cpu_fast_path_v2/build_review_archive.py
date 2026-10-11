#!/usr/bin/env python
"""Build and independently validate the mandatory F8 review archive."""

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
RESULTS = ROOT / "results/cpu_fast_path_v2"
DATE = "20260821"
ARCHIVE = RESULTS / f"cpu_fast_path_v2_review_bundle_{DATE}.tar.gz"
SOURCE_CHECKPOINT = RESULTS / "00_freeze/source_checkpoint.tar.gz"

REQUIRED_COPIES = {
    "final_fast_path_decision.md": RESULTS / "07_decision/final_fast_path_decision.md",
    "default_output_contract.md": RESULTS / "01_output_contract/default_output_contract.md",
    "full_panel.tsv": RESULTS / "02_real_panel/full_panel.tsv",
    "development_panel.tsv": RESULTS / "02_real_panel/development_panel.tsv",
    "heldout_panel.tsv": RESULTS / "02_real_panel/heldout_panel.tsv",
    "challenge_panel.tsv": RESULTS / "02_real_panel/challenge_panel.tsv",
    "precision_contracts.tsv": RESULTS / "03_precision_compact/precision_contracts.tsv",
    "precision_decision.md": RESULTS / "03_precision_compact/final_precision_decision.md",
    "candidate_registry.tsv": RESULTS / "04_parallel_weights/candidate_registry.tsv",
    "locked_candidate_configs.tsv": RESULTS / "04_parallel_weights/locked_candidate_configs.tsv",
    "heldout_results.tsv": RESULTS / "04_parallel_weights/heldout_results.tsv",
    "challenge_results.tsv": RESULTS / "04_parallel_weights/challenge_results.tsv",
    "exact_like_summary.tsv": RESULTS / "05_quality/exact_like_summary.tsv",
    "balanced_summary.tsv": RESULTS / "05_quality/balanced_summary.tsv",
    "self_variability_comparison.tsv": RESULTS / "05_quality/self_variability_comparison.tsv",
    "real_system_summary.tsv": RESULTS / "06_performance/real_system_summary.tsv",
    "prefix_scaling.tsv": RESULTS / "06_performance/prefix_scaling.tsv",
    "thread_scaling.tsv": RESULTS / "06_performance/thread_scaling.tsv",
    "stage_decomposition.tsv": RESULTS / "06_performance/stage_decomposition.tsv",
    "memory.tsv": RESULTS / "06_performance/memory.tsv",
    "tail_atlas.tsv": RESULTS / "05_quality/tail_atlas.tsv",
    "accepted_changes.tsv": RESULTS / "07_decision/accepted_changes.tsv",
    "rejected_candidates.tsv": RESULTS / "07_decision/rejected_candidates.tsv",
    "user_interface_impact.md": RESULTS / "07_decision/user_interface_impact.md",
    "manuscript_impact.md": RESULTS / "07_decision/manuscript_impact.md",
    "full_test_output.txt": RESULTS / "handoff_to_chatgpt/full_test_output.txt",
    "focused_test_output.txt": RESULTS / "handoff_to_chatgpt/focused_test_output.txt",
    "cpu_clean_install_report.md": RESULTS / "handoff_to_chatgpt/cpu_clean_install_report.md",
    "gpu_clean_install_report.md": RESULTS / "handoff_to_chatgpt/gpu_clean_install_report.md",
    "package_build_report.md": RESULTS / "handoff_to_chatgpt/package_build_report.md",
    "twine_check.txt": RESULTS / "handoff_to_chatgpt/twine_check.txt",
}

OPTIONAL_COPIES = {
    "backward_compatibility.md": RESULTS / "01_output_contract/backward_compatibility.md",
    "schema_change.tsv": RESULTS / "01_output_contract/schema_change.tsv",
    "panel_selection_report.md": RESULTS / "02_real_panel/panel_selection_report.md",
    "precision_development_results.tsv": RESULTS / "03_precision_compact/development_results.tsv",
    "precision_heldout_quality.tsv": RESULTS / "03_precision_compact/heldout_quality.tsv",
    "precision_heldout_runtime.tsv": RESULTS / "03_precision_compact/heldout_runtime.tsv",
    "precision_challenge_cases.tsv": RESULTS / "03_precision_compact/challenge_cases.tsv",
    "algorithm_descriptions.md": RESULTS / "04_parallel_weights/algorithm_descriptions.md",
    "development_ablation.tsv": RESULTS / "04_parallel_weights/development_ablation.tsv",
    "refinement_rates.tsv": RESULTS / "04_parallel_weights/refinement_rates.tsv",
    "quality_gate_registry.tsv": RESULTS / "05_quality/quality_gate_registry.tsv",
    "failures_warnings.tsv": RESULTS / "06_performance/failures_warnings.tsv",
    "default_output_decision.md": RESULTS / "07_decision/default_output_decision.md",
    "parallel_estimator_decision.md": RESULTS / "07_decision/parallel_estimator_decision.md",
}

CHANGED_SOURCE = (
    "src/mdcorealign/cli.py",
    "src/mdcorealign/experimental.py",
    "src/mdsuperpose/flexibility.py",
    "src/mdsuperpose/parallel_weights.py",
    "scripts/cpu_fast_path_v2/run_fast_path_study.py",
    "scripts/cpu_fast_path_v2/run_thread_scaling.py",
    "scripts/cpu_fast_path_v2/summarize_locked_quality.py",
    "scripts/cpu_fast_path_v2/finalize_fast_path.py",
    "scripts/cpu_fast_path_v2/build_review_archive.py",
    "README.md",
    "docs/cli_reference.md",
    "docs/output_formats.md",
    "docs/python_api.md",
    "docs/rmsf_flexible_segments.md",
    "examples/real_md_small/README.md",
)

CHANGED_TESTS = (
    "tests/cpu_fast_path_v2/test_parallel_weights.py",
    "tests/test_product_contract_cli.py",
)

TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".patch", ".py", ".rst", ".sh", ".toml",
    ".tsv", ".txt", ".yaml", ".yml",
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


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sanitise_text(text: str) -> str:
    home = Path.home()
    user_name = home.name
    temporary_prefix = str(Path(os.sep) / "tmp" / "mdca_")
    replacements = (
        (str(ROOT), "<REPOSITORY_ROOT>"),
        (str(ROOT.parent), "<PROJECTS_ROOT>"),
        (str(home), "<USER_HOME>"),
        ("/home/<USER>", "<USER_HOME>"),
        (user_name, "<USER>"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(re.escape(temporary_prefix) + r"[^\s`'\"<>]+", "<TEMP_PATH>", text)
    return text


def copy_sanitised(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        destination.write_text(sanitise_text(source.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def load_checkpoint() -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(SOURCE_CHECKPOINT, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                members[member.name.removeprefix("./")] = extracted.read()
    return members


def build_diffs(handoff: Path, checkpoint: dict[str, bytes]) -> None:
    unified: list[str] = []
    diff_root = handoff / "individual_diffs"
    diff_root.mkdir()
    for relative in CHANGED_SOURCE + CHANGED_TESTS:
        current = (ROOT / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        old_bytes = checkpoint.get(relative)
        old = old_bytes.decode("utf-8").splitlines(keepends=True) if old_bytes is not None else []
        from_name = f"a/{relative}" if old_bytes is not None else "/dev/null"
        to_name = f"b/{relative}"
        lines = list(difflib.unified_diff(old, current, fromfile=from_name, tofile=to_name))
        if not lines:
            continue
        unified.extend(lines)
        safe_name = relative.replace("/", "__") + ".patch"
        (diff_root / safe_name).write_text("".join(lines), encoding="utf-8")
    (handoff / "unified_source_diff.patch").write_text("".join(unified), encoding="utf-8")


def build_key_metrics(handoff: Path) -> None:
    performance = read_tsv(RESULTS / "06_performance/real_system_summary.tsv")
    by_method_runtime: dict[str, list[float]] = {}
    by_method_speedup: dict[str, list[float]] = {}
    for row in performance:
        by_method_runtime.setdefault(row["method"], []).append(float(row["median_loaded_runtime_seconds"]))
        by_method_speedup.setdefault(row["method"], []).append(float(row["speedup_vs_exact"]))

    quality = {row["configuration"]: row for row in read_tsv(RESULTS / "05_quality/balanced_summary.tsv")}
    metrics: list[dict[str, object]] = [
        {"metric": "real_system_families", "value": 20, "unit": "families", "interpretation": "8 development and 12 locked held-out families"},
        {"metric": "formal_measured_runs", "value": 1230, "unit": "runs", "interpretation": "one-thread loaded-coordinate timings after warmup"},
        {"metric": "canonical_tests", "value": "322 passed; 8 skipped; 2 subtests passed", "unit": "pytest", "interpretation": "final full canonical suite"},
        {"metric": "ordinary_rmsf_default", "value": "opt-in", "unit": "contract", "interpretation": "explicit --ordinary-rmsf; no placeholder values"},
        {"metric": "mixed_precision_decision", "value": "not promoted", "unit": "decision", "interpretation": "development speedup below 1.15x gate"},
    ]
    for method in sorted(by_method_runtime):
        metrics.append({
            "metric": f"full_length_median_runtime_{method}",
            "value": f"{np.median(by_method_runtime[method]):.9f}",
            "unit": "seconds",
            "interpretation": "median of per-system medians across 20 families",
        })
        metrics.append({
            "metric": f"full_length_median_speedup_{method}",
            "value": f"{np.median(by_method_speedup[method]):.6f}",
            "unit": "fold_vs_exact",
            "interpretation": "median system-level speedup",
        })
    for method, row in quality.items():
        metrics.extend((
            {"metric": f"combined_median_core_jaccard_{method}", "value": row["median_core_jaccard"], "unit": "fraction", "interpretation": "held-out plus registered challenge panel"},
            {"metric": f"combined_p95_rmsd_difference_{method}", "value": row["p95_rmsd_absolute_difference_A"], "unit": "angstrom", "interpretation": "held-out plus registered challenge panel"},
            {"metric": f"catastrophic_count_{method}", "value": row["catastrophic_count"], "unit": "cases", "interpretation": "registered quality contract"},
            {"metric": f"balanced_pass_{method}", "value": row["balanced_pass"], "unit": "boolean", "interpretation": "prospectively registered balanced gate"},
        ))
    write_tsv(handoff / "key_metrics.tsv", ["metric", "value", "unit", "interpretation"], metrics)


def build_phase_status(handoff: Path) -> None:
    rows = [
        {"phase": "F0", "status": "complete", "evidence": "source checkpoint and environment freeze", "notes": "timed external work was allowed to finish before source edits"},
        {"phase": "F1", "status": "complete", "evidence": "output contract, implementation, documentation and tests", "notes": "ordinary RMSF is opt-in in the release-candidate CLI"},
        {"phase": "F2", "status": "complete", "evidence": "20-family real panel plus 14 registered challenge tails", "notes": "no stitched or synthetic cases in the primary panel"},
        {"phase": "F3", "status": "complete_negative", "evidence": "matched compact precision development benchmark", "notes": "no mixed candidate met the 1.15x speed gate"},
        {"phase": "F4", "status": "complete", "evidence": "five internal estimator families evaluated", "notes": "candidate names remain internal"},
        {"phase": "F5", "status": "complete_negative", "evidence": "held-out and challenge quality gates", "notes": "no candidate passed exact-like or balanced targets"},
        {"phase": "F6", "status": "complete", "evidence": "1,230 formal timings and thread-scaling subset", "notes": "one-thread remains the primary timing contract"},
        {"phase": "F7", "status": "complete", "evidence": "product and algorithm decision records", "notes": "chronological float64 exact remains unchanged"},
        {"phase": "F8", "status": "complete", "evidence": "tests, clean installs, build, figures and validated archive", "notes": "no external benchmark rerun"},
    ]
    write_tsv(handoff / "phase_status.tsv", ["phase", "status", "evidence", "notes"], rows)


def write_readme(handoff: Path, exact_hash_match: bool) -> None:
    text = f"""# MDCoreAlign CPU Fast-Path Study v2 Review Bundle

This self-contained review bundle records F0-F8 without raw trajectories,
coordinate caches, aligned structures, third-party benchmark outputs, virtual
environments, Git internals or secrets.

## Registered scope

- MDCoreAlign configurations only; no MDLovoFit, THESEUS, GPU timing or W5 timing.
- Twenty independent real MD families: eight development and twelve locked held-out.
- Fourteen registered challenge-tail cases evaluated separately.
- Primary timing: one CPU thread, loaded coordinates, one warmup and five repeats,
  compact output, no ordinary RMSF, no coordinate materialisation and no file output.
- The chronological float64 exact estimator is the unchanged reference.

## Main decisions

- Ordinary/full-set RMSF is opt-in through `--ordinary-rmsf` for the proposed
  release-candidate CLI. Old schemas remain readable and omitted values remain null.
- Float64 throughput is retained because no mixed-precision option met the 1.15x
  development speed gate.
- No parallel candidate passed the full prospectively registered exact-like or
  balanced contract. Candidate speedups are therefore study results, not promoted
  public algorithms.
- Exact reference source hash unchanged from F0: `{str(exact_hash_match).lower()}`.

## Validation

- Full suite: 322 passed, 8 skipped, 2 subtests passed.
- Focused suite: 16 passed.
- CPU-only clean install, GPU clean-install smoke, package build and twine checks passed.
- All eight required figures are present as PDF and PNG.
- The archive was independently extracted and checked against its detached
  `sha256_manifest.tsv`. The SHA manifest itself is excluded from its own entries,
  as a file cannot contain its final cryptographic hash without self-reference.

Start with `final_fast_path_decision.md`, `key_metrics.tsv`, the two quality
summaries and `real_system_summary.tsv`. Candidate-level raw held-out and challenge
results are included for independent review.
"""
    (handoff / "00_README_FIRST.md").write_text(text, encoding="utf-8")


def scan_text_files(handoff: Path) -> list[dict[str, str]]:
    local_roots = [str(Path(os.sep) / name) + os.sep for name in ("home", "storage", "tmp")]
    user_name = Path.home().name
    patterns = {
        "absolute_local_path": re.compile(
            "(?:" + "|".join(re.escape(root) for root in local_roots) + r")[^\s`'\"<>]+"
        ),
        "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "generic_secret_assignment": re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}"),
        "username": re.compile(r"(?<![A-Za-z0-9_])" + re.escape(user_name) + r"(?![A-Za-z0-9_])"),
    }
    findings: list[dict[str, str]] = []
    for path in sorted(handoff.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                findings.append({
                    "path": path.relative_to(handoff).as_posix(),
                    "pattern": label,
                    "match": match.group(0)[:120],
                })
    return findings


def build_manifests(handoff: Path) -> tuple[int, list[dict[str, str]]]:
    evidence_files = [path for path in sorted(handoff.rglob("*")) if path.is_file()]
    manifest_rows = [
        {
            "relative_path": path.relative_to(handoff).as_posix(),
            "size_bytes": path.stat().st_size,
            "category": path.relative_to(handoff).parts[0] if len(path.relative_to(handoff).parts) > 1 else "root",
        }
        for path in evidence_files
    ]
    manifest_rows.extend((
        {"relative_path": "file_manifest.tsv", "size_bytes": "self_describing", "category": "root"},
        {"relative_path": "sha256_manifest.tsv", "size_bytes": "self_describing", "category": "root"},
    ))
    write_tsv(handoff / "file_manifest.tsv", ["relative_path", "size_bytes", "category"], manifest_rows)
    files_to_hash = [
        path for path in sorted(handoff.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.tsv"
    ]
    hash_rows = [
        {"relative_path": path.relative_to(handoff).as_posix(), "sha256": sha256(path)}
        for path in files_to_hash
    ]
    write_tsv(handoff / "sha256_manifest.tsv", ["relative_path", "sha256"], hash_rows)
    return len(files_to_hash) + 1, hash_rows


def validate_required(handoff: Path) -> None:
    required = set(REQUIRED_COPIES) | {
        "00_README_FIRST.md", "phase_status.tsv", "key_metrics.tsv",
        "file_manifest.tsv", "sha256_manifest.tsv", "unified_source_diff.patch",
    }
    missing = sorted(name for name in required if not (handoff / name).is_file())
    for directory in ("changed_source", "changed_tests", "individual_diffs", "figures"):
        if not (handoff / directory).is_dir() or not any((handoff / directory).rglob("*")):
            missing.append(directory + "/")
    expected_figures = {
        f"figures/fig{number}_{stem}.{suffix}"
        for number, stem in (
            (1, "quality_speed_pareto"),
            (2, "real_system_runtime_distributions"),
            (3, "speedup_vs_frames_residues"),
            (4, "weight_core_rmsf_agreement"),
            (5, "stage_decomposition"),
            (6, "ordinary_rmsf_contract"),
            (7, "thread_scaling"),
            (8, "challenge_tail_cases"),
        )
        for suffix in ("pdf", "png")
    }
    missing.extend(sorted(name for name in expected_figures if not (handoff / name).is_file()))
    if missing:
        raise RuntimeError("Missing F8 review items: " + ", ".join(missing))


def populate(handoff: Path) -> bool:
    checkpoint = load_checkpoint()
    for destination, source in REQUIRED_COPIES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        copy_sanitised(source, handoff / destination)
    for destination, source in OPTIONAL_COPIES.items():
        if source.is_file():
            copy_sanitised(source, handoff / destination)

    shutil.copytree(RESULTS / "handoff_to_chatgpt/figures", handoff / "figures")
    for relative in CHANGED_SOURCE:
        copy_sanitised(ROOT / relative, handoff / "changed_source" / relative)
    for relative in CHANGED_TESTS:
        copy_sanitised(ROOT / relative, handoff / "changed_tests" / relative)
    build_diffs(handoff, checkpoint)

    frozen_manifest = {row["path"]: row["sha256"] for row in read_tsv(RESULTS / "00_freeze/source_manifest.tsv")}
    exact_path = ROOT / "src/mdsuperpose/sequential_weights.py"
    exact_hash_match = sha256(exact_path) == frozen_manifest["src/mdsuperpose/sequential_weights.py"]
    build_key_metrics(handoff)
    build_phase_status(handoff)
    write_readme(handoff, exact_hash_match)
    return exact_hash_match


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mdca_f8_stage_") as temporary:
        handoff = Path(temporary) / "handoff_to_chatgpt"
        handoff.mkdir()
        exact_hash_match = populate(handoff)

        pre_manifest_findings = scan_text_files(handoff)
        if pre_manifest_findings:
            raise RuntimeError("Archive text scan failed: " + json.dumps(pre_manifest_findings[:10], indent=2))

        build_manifests(handoff)
        validate_required(handoff)
        findings = scan_text_files(handoff)
        if findings:
            raise RuntimeError("Final archive text scan failed: " + json.dumps(findings[:10], indent=2))

        with tarfile.open(ARCHIVE, "w:gz") as archive:
            archive.add(handoff, arcname="handoff_to_chatgpt", recursive=True)

        with tempfile.TemporaryDirectory(prefix="mdca_f8_extract_") as extract_directory:
            with tarfile.open(ARCHIVE, "r:gz") as archive:
                archive.extractall(extract_directory, filter="data")
            extracted = Path(extract_directory) / "handoff_to_chatgpt"
            validate_required(extracted)
            hash_rows = read_tsv(extracted / "sha256_manifest.tsv")
            mismatches = [
                row["relative_path"]
                for row in hash_rows
                if not (extracted / row["relative_path"]).is_file()
                or sha256(extracted / row["relative_path"]) != row["sha256"]
            ]
            extracted_findings = scan_text_files(extracted)
            if mismatches or extracted_findings:
                raise RuntimeError(
                    f"Extracted validation failed: mismatches={mismatches}, findings={extracted_findings[:10]}"
                )
            file_count = sum(path.is_file() for path in extracted.rglob("*"))
            maximum_size = max(path.stat().st_size for path in extracted.rglob("*") if path.is_file())

        canonical = RESULTS / "handoff_to_chatgpt"
        for path in handoff.iterdir():
            destination = canonical / path.name
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination)

    report = {
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "included_file_count": file_count,
        "maximum_included_file_size_bytes": maximum_size,
        "required_sections_present": True,
        "secret_and_absolute_path_scan_passed": True,
        "extraction_passed": True,
        "manifest_verification_passed": True,
        "exact_reference_hash_matches_f0": exact_hash_match,
        "sha256_manifest_self_entry": "excluded_by_self-reference_definition",
    }
    (RESULTS / "archive_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
