"""Audit repository state and freeze a pre-manuscript result manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_ready_common import ROOT, column_count, read_table, row_count, sha256_file, write_json  # noqa: E402


KEY_RESULT_FILES = [
    "results/final_multi_system_summary_20260603/final_real_trajectory_summary.csv",
    "results/final_multi_system_summary_20260603/final_structure_pair_summary.csv",
    "results/final_multi_system_summary_20260603/final_nmr_ensemble_summary.csv",
    "results/final_multi_system_summary_20260603/final_external_biology_summary.csv",
    "results/final_multi_system_summary_20260603/final_baseline_breadth_hardtrim_summary.csv",
    "results/higher_tier_external_baseline_theseus_formal/combined_theseus_formal_summary.csv",
    "results/higher_tier_external_baseline_theseus_structure_pairs_formal/combined_theseus_structure_pair_formal_summary.csv",
    "results/flexibility/flexibility_summary_four_systems.tsv",
    "results/flexibility/flexibility_statistics_four_systems.tsv",
    "results/flexibility/figure_data/fig_rmsf_profiles.tsv",
    "results/flexibility/figure_data/fig_mobility_vs_rmsf.tsv",
    "results/flexibility/figure_data/fig_core_noncore_rmsf.tsv",
    "results/flexibility/figures/fig_rmsf_profiles_four_systems.pdf",
    "results/flexibility/figures/fig_rmsf_profiles_four_systems.png",
    "results/flexibility/figures/fig_mobility_vs_robust_rmsf.pdf",
    "results/flexibility/figures/fig_core_vs_noncore_rmsf.pdf",
    "results/final_statistics/trajectory_benchmark_statistics.tsv",
    "results/final_statistics/pair_benchmark_statistics.tsv",
    "results/final_statistics/nmr500_benchmark_statistics.tsv",
    "results/final_statistics/flexibility_statistics_final.tsv",
    "results/final_statistics/biology_support_statistics.tsv",
    "results/final_statistics/final_statistics_summary.md",
    "results/ablation/regularization/regularization_ablation_four_systems.tsv",
    "results/ablation/regularization/regularization_ablation_summary.md",
    "results/diagnostics/alignment_traces_four_systems.tsv",
    "results/diagnostics/runtime_profiles_four_systems.json",
    "results/diagnostics/diagnostics_summary.md",
    "results/sensitivity/sensitivity_summary.tsv",
    "results/sensitivity/default_parameter_justification.md",
    "results/paper_ready/audit/paper_outputs_audit.tsv",
    "results/paper_ready/test_report.txt",
]

STALE_STATUS_FILES = [
    "README.md",
    "docs/current_main_results_summary.md",
    "docs/current_main_results_summary.txt",
    "docs/rmsf_flexible_segments.md",
    "results/flexibility/figures/figure_captions_draft.md",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper_ready")
    parser.add_argument("--skip-tests", action="store_true", help="Record test status as skipped instead of running unittest.")
    return parser.parse_args()


def git_info() -> dict[str, object]:
    """Return Git commit metadata when available."""

    top = run_command(["git", "rev-parse", "--show-toplevel"], cwd=ROOT)
    head = run_command(["git", "rev-parse", "HEAD"], cwd=ROOT)
    return {
        "repository_toplevel": top["stdout"].strip() if top["returncode"] == 0 else None,
        "commit": head["stdout"].strip() if head["returncode"] == 0 else None,
        "commit_available": head["returncode"] == 0,
        "warnings": [] if head["returncode"] == 0 else [head["stderr"].strip() or "git HEAD unavailable"],
    }


def dependency_versions() -> dict[str, object]:
    """Return versions for core and optional dependencies."""

    packages = ["numpy", "scipy", "pandas", "matplotlib", "MDAnalysis", "mdtraj"]
    versions: dict[str, object] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def package_version() -> str | None:
    """Return installed package version if available."""

    for name in ("mdsuperpose", "mdlovofit", "mdcorealign"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def external_tool_status() -> dict[str, object]:
    """Detect external baseline tools without requiring them."""

    tools = {
        "MDLovoFit": [os.environ.get("MDLOVOFIT_EXECUTABLE"), "mdlovofit"],
        "THESEUS": [os.environ.get("THESEUS_EXECUTABLE"), "theseus"],
    }
    status: dict[str, object] = {}
    for label, candidates in tools.items():
        executable = next((candidate for candidate in candidates if candidate and shutil.which(candidate)), None)
        if executable is None:
            status[label] = {"available": False, "executable": None, "version_output": None}
            continue
        command = [executable, "--version"]
        version = run_command(command, cwd=ROOT, timeout=5)
        if version["returncode"] != 0:
            version = run_command([executable, "-h"], cwd=ROOT, timeout=5)
        status[label] = {
            "available": True,
            "executable": executable,
            "version_output": (version["stdout"] or version["stderr"]).strip()[:1000],
        }
    return status


def run_tests(skip: bool) -> dict[str, object]:
    """Run the unittest suite and parse the test count."""

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    if skip:
        return {"command": " ".join(command), "status": "skipped", "passed": None, "tests_run": None, "stdout": "", "stderr": ""}
    result = run_command(command, cwd=ROOT, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    combined = f"{result['stdout']}\n{result['stderr']}"
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    return {
        "command": " ".join(command),
        "status": "passed" if result["returncode"] == 0 else "failed",
        "passed": result["returncode"] == 0,
        "tests_run": int(match.group(1)) if match else None,
        "stdout": result["stdout"][-4000:],
        "stderr": result["stderr"][-4000:],
    }


def file_manifest(paths: list[str]) -> tuple[list[dict[str, object]], list[str]]:
    """Build file metadata records and return missing-file warnings."""

    records: list[dict[str, object]] = []
    warnings: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            warnings.append(f"missing required key result file: {relative}")
            records.append({"path": relative, "exists": False})
            continue
        stat = path.stat()
        records.append(
            {
                "path": relative,
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
                "row_count": row_count(path),
                "column_count": column_count(path),
            }
        )
    return records, warnings


def stale_status_warnings(paths: list[str]) -> list[str]:
    """Detect stale test-count mentions in key documentation files."""

    warnings: list[str] = []
    stale_pattern = re.compile(r"\b(70|76|80)\s+(?:unit\s+)?tests?\b|`(70|76|80)`\s+(?:unit\s+)?tests?", re.I)
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            warnings.append(f"status file missing: {relative}")
            continue
        text = path.read_text(errors="replace")
        for match in stale_pattern.finditer(text):
            warnings.append(f"stale test-count mention in {relative}: {match.group(0)!r}")
    return warnings


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 20,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run a subprocess and capture text output."""

    try:
        completed = subprocess.run(command, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True)
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def build_markdown(manifest: dict[str, object]) -> str:
    """Build a compact human-readable manifest report."""

    files = manifest["key_result_files"]
    missing = [record for record in files if not record.get("exists")]
    warnings = manifest["warnings"]
    lines = [
        "# Project State Manifest",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Python: `{manifest['python_version']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Package version: `{manifest['package_version'] or 'not installed as package'}`",
        f"- Git commit: `{manifest['git']['commit'] or 'unavailable'}`",
        f"- Test status: `{manifest['test_status']['status']}`; tests run: `{manifest['test_status']['tests_run']}`",
        f"- Key result files checked: `{len(files)}`",
        f"- Missing required files: `{len(missing)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Key Result Files", "", "| File | Size bytes | Rows | Columns | SHA256 |", "| --- | ---: | ---: | ---: | --- |"])
    for record in files:
        lines.append(
            "| {path} | {size} | {rows} | {cols} | {sha} |".format(
                path=record["path"],
                size=record.get("size_bytes", ""),
                rows=record.get("row_count", ""),
                cols=record.get("column_count", ""),
                sha=str(record.get("sha256", ""))[:16],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the project-state audit."""

    args = parse_args()
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    file_records, file_warnings = file_manifest(KEY_RESULT_FILES)
    warnings = []
    warnings.extend(file_warnings)
    warnings.extend(stale_status_warnings(STALE_STATUS_FILES))

    manifest = {
        "generated_at": generated_at,
        "python_version": sys.version.replace("\n", " "),
        "package_version": package_version(),
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "git": git_info(),
        "dependencies": dependency_versions(),
        "external_tools": external_tool_status(),
        "test_status": run_tests(args.skip_tests),
        "key_result_files": file_records,
        "status_files_checked": STALE_STATUS_FILES,
        "warnings": warnings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "project_state_manifest.json"
    md_path = args.output_dir / "project_state_manifest.md"
    write_json(json_path, manifest)
    md_path.write_text(build_markdown(manifest))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Audit summary: missing_files={len(file_warnings)} warnings={len(warnings)} test_status={manifest['test_status']['status']}")
    return 1 if file_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
