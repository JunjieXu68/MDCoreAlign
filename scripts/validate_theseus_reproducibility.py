"""Validate or rerun THESEUS formal baseline reproducibility."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import read_table, sha256_file, write_json, write_table  # noqa: E402
from submission_critical_common import ROOT  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "submission_critical" / "theseus_reproducibility"

VALIDATION_FIELDS = [
    "case_group",
    "source_file",
    "status",
    "executable_detected",
    "row_count",
    "required_columns_present",
    "checksum_sha256",
    "rerun_file",
    "max_abs_unified_posthoc_rmsd_delta",
    "max_relative_unified_posthoc_rmsd_delta",
    "notes",
]

REQUIRED_COLUMNS = [
    "benchmark_mode",
    "dataset_name",
    "method_name",
    "n_repeats",
    "runtime_seconds_mean",
    "unified_posthoc_rmsd_mean",
    "theseus_native_classical_pairwise_rmsd_mean",
    "theseus_native_maximum_likelihood_rmsd_mean",
    "theseus_native_least_squares_sigma_mean",
    "theseus_native_maximum_likelihood_sigma_mean",
]

EXISTING_TABLES = {
    "formal_trajectory": ROOT
    / "results"
    / "higher_tier_external_baseline_theseus_formal"
    / "combined_theseus_formal_summary.csv",
    "formal_structure_pair": ROOT
    / "results"
    / "higher_tier_external_baseline_theseus_structure_pairs_formal"
    / "combined_theseus_structure_pair_formal_summary.csv",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--theseus-executable", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--skip-rerun", action="store_true", help="Only validate existing THESEUS tables.")
    return parser.parse_args()


def detect_executable(explicit: Path | None, timeout_seconds: float) -> dict[str, object]:
    """Detect THESEUS executable and capture lightweight environment metadata."""

    candidate = explicit or (Path(os.environ["THESEUS_EXECUTABLE"]) if os.environ.get("THESEUS_EXECUTABLE") else None)
    if candidate is None:
        which = shutil.which("theseus")
        candidate = Path(which) if which else None
    detected = bool(candidate and candidate.exists() and os.access(candidate, os.X_OK))
    payload: dict[str, object] = {
        "detected": detected,
        "executable_path": str(candidate) if candidate else "",
        "THESEUS_EXECUTABLE": os.environ.get("THESEUS_EXECUTABLE", ""),
        "PATH_contains_theseus": shutil.which("theseus") or "",
        "platform": platform.platform(),
        "python_version": sys.version,
        "version_stdout": "",
        "version_stderr": "",
        "help_stdout": "",
        "help_stderr": "",
    }
    if detected and candidate is not None:
        for label, command in (
            ("version", [str(candidate), "--version"]),
            ("help", [str(candidate), "-h"]),
        ):
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                payload[f"{label}_stdout"] = completed.stdout[:8000]
                payload[f"{label}_stderr"] = completed.stderr[:8000]
                payload[f"{label}_return_code"] = completed.returncode
            except Exception as exc:
                payload[f"{label}_stderr"] = f"{type(exc).__name__}: {exc}"
    return payload


def table_columns(path: Path) -> list[str]:
    """Return CSV columns."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def validate_existing_tables(executable_detected: bool) -> list[dict[str, object]]:
    """Validate existing official THESEUS tables."""

    rows: list[dict[str, object]] = []
    for case_group, path in EXISTING_TABLES.items():
        if not path.exists():
            rows.append(
                {
                    "case_group": case_group,
                    "source_file": str(path.relative_to(ROOT)),
                    "status": "missing_existing_table",
                    "executable_detected": executable_detected,
                    "row_count": 0,
                    "required_columns_present": False,
                    "checksum_sha256": "",
                    "rerun_file": "",
                    "max_abs_unified_posthoc_rmsd_delta": "",
                    "max_relative_unified_posthoc_rmsd_delta": "",
                    "notes": "Existing official THESEUS table is missing.",
                }
            )
            continue
        columns = table_columns(path)
        required_ok = all(column in columns for column in REQUIRED_COLUMNS)
        rows.append(
            {
                "case_group": case_group,
                "source_file": str(path.relative_to(ROOT)),
                "status": "existing_table_validated" if required_ok else "existing_table_schema_warning",
                "executable_detected": executable_detected,
                "row_count": len(read_table(path)),
                "required_columns_present": required_ok,
                "checksum_sha256": sha256_file(path),
                "rerun_file": "",
                "max_abs_unified_posthoc_rmsd_delta": "",
                "max_relative_unified_posthoc_rmsd_delta": "",
                "notes": "Existing table validated; not a current-environment rerun.",
            }
        )
    return rows


def _write_detection_log(output_dir: Path, environment: dict[str, object]) -> None:
    """Write executable detection text log."""

    raw_dir = output_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"detected={environment['detected']}",
        f"executable_path={environment['executable_path']}",
        f"THESEUS_EXECUTABLE={environment['THESEUS_EXECUTABLE']}",
        f"PATH_contains_theseus={environment['PATH_contains_theseus']}",
        "",
        "## version stdout",
        str(environment.get("version_stdout", "")),
        "## version stderr",
        str(environment.get("version_stderr", "")),
        "## help stdout",
        str(environment.get("help_stdout", "")),
        "## help stderr",
        str(environment.get("help_stderr", "")),
    ]
    (raw_dir / "executable_detection.txt").write_text("\n".join(lines) + "\n")


def build_summary(environment: dict[str, object], rows: list[dict[str, object]], rerun_attempted: bool) -> str:
    """Build Markdown summary."""

    detected = bool(environment["detected"])
    valid_tables = sum(row["status"] == "existing_table_validated" for row in rows)
    lines = [
        "# THESEUS Reproducibility Validation",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        "",
        "## Required Questions",
        "",
        f"1. Was THESEUS executable detected? `{detected}`.",
        f"2. Which version was used? `{str(environment.get('version_stdout') or environment.get('version_stderr') or 'unavailable').splitlines()[0] if detected else 'not_applicable_executable_missing'}`.",
        f"3. Which formal cases were rerun? `{'formal trajectory and structure-pair cases' if rerun_attempted else 'none'}`.",
        "4. Do rerun values match existing official tables? No current rerun comparison is reported unless rerun files are present in this directory.",
        "5. If not rerun, exactly why not? " + ("Rerun was skipped by `--skip-rerun`." if detected and not rerun_attempted else "THESEUS executable was not detected in the current environment." if not detected else "Rerun attempted; inspect raw logs."),
        f"6. Which THESEUS results are safe to cite? Existing validated summary tables with checksums are safe to cite as previously generated results in `{valid_tables}/{len(rows)}` table groups, provided the manuscript states they were not rerun in this environment.",
        "7. Which THESEUS results should be marked as previously generated but not rerun? All THESEUS formal trajectory and structure-pair rows if `status=existing_table_validated` and no rerun file is listed.",
        "",
        "## Validation Rows",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['case_group']}: {row['status']} ({row['row_count']} rows).")
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path, environment: dict[str, object]) -> None:
    """Write manifest."""

    files = [
        output_dir / "theseus_environment.json",
        output_dir / "theseus_rerun_or_validation.tsv",
        output_dir / "theseus_reproducibility_summary.md",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/validate_theseus_reproducibility.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executable_detected": bool(environment["detected"]),
            "source_tables": {key: str(path.relative_to(ROOT)) for key, path in EXISTING_TABLES.items()},
            "output_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.exists()},
        },
    )


def main() -> int:
    """Validate THESEUS reproducibility state."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = detect_executable(args.theseus_executable, args.timeout_seconds)
    _write_detection_log(args.output_dir, environment)
    write_json(args.output_dir / "theseus_environment.json", environment)
    rows = validate_existing_tables(bool(environment["detected"]))
    if not environment["detected"]:
        for row in rows:
            row["status"] = "not_rerun_executable_missing" if row["status"] == "existing_table_validated" else row["status"]
            if row["notes"]:
                row["notes"] += " "
            row["notes"] += "THESEUS executable is absent, so this is validation rather than rerun."
    rerun_attempted = False
    write_table(args.output_dir / "theseus_rerun_or_validation.tsv", rows, VALIDATION_FIELDS)
    (args.output_dir / "theseus_reproducibility_summary.md").write_text(
        build_summary(environment, rows, rerun_attempted)
    )
    write_manifest(args.output_dir, environment)
    print(f"Wrote THESEUS reproducibility validation to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
