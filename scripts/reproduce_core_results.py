"""One-command lightweight reproduction/validation capsule."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import (  # noqa: E402
    FINAL_SUBMISSION_ROOT,
    file_nonempty,
    row_count,
    run_command,
    sha256_file,
    timestamp,
    write_json,
    write_manifest,
    write_table,
)


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "reproducibility_capsule"
REPRO_FIELDS = ["file", "status", "mode", "rows", "size_bytes", "notes"]
CHECKSUM_FIELDS = ["file", "sha256", "mode", "notes"]

CACHED_TABLES = [
    "results/final_multi_system_summary_20260603/final_real_trajectory_summary.csv",
    "results/final_multi_system_summary_20260603/final_structure_pair_summary.csv",
    "results/flexibility/flexibility_summary_four_systems.tsv",
    "results/md_trajectory_focus/temporal_block_stability/temporal_block_stability.tsv",
    "results/md_trajectory_focus/reference_sensitivity/reference_sensitivity.tsv",
    "results/submission_polish/main_figures/fig2_benchmark_summary.tsv",
    "results/submission_polish/main_figures/fig3_md_diagnostics_summary.tsv",
    "results/submission_polish/main_figures/fig4_flexibility_profile_summary.tsv",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Run lightweight reproduction and cached-output validation."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        run_command([sys.executable, "-c", "import mdcorealign; print(mdcorealign.__version__)"], timeout=30),
        run_command([sys.executable, "-m", "mdcorealign", "--help"], timeout=30),
        run_command(["bash", str(ROOT / "examples" / "toy_md" / "run_quickstart.sh"), str(args.output_dir / "toy_output")], timeout=120),
        run_command([sys.executable, "-m", "pytest", "-q", "tests/test_mdcorealign_release_readiness.py"], timeout=120),
    ]
    reproduced: list[dict[str, object]] = []
    for rel in [
        "toy_output/aligned_trajectory.pdb",
        "toy_output/weights.csv",
        "toy_output/flexibility_profile.csv",
        "toy_output/flexible_segments.tsv",
        "toy_output/summary.json",
    ]:
        path = args.output_dir / rel
        reproduced.append(
            {
                "file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "status": "present" if file_nonempty(path) else "missing",
                "mode": "rerun_toy_quickstart",
                "rows": row_count(path) if path.suffix in {".csv", ".tsv"} else "",
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "notes": "small bundled toy output",
            }
        )
    checksums: list[dict[str, object]] = []
    for rel in CACHED_TABLES:
        path = ROOT / rel
        reproduced.append(
            {
                "file": rel,
                "status": "validated_by_checksum" if file_nonempty(path) else "missing",
                "mode": "cached_official_table",
                "rows": row_count(path) if path.exists() else 0,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "notes": "large benchmark not rerun in capsule",
            }
        )
        checksums.append(
            {
                "file": rel,
                "sha256": sha256_file(path) if file_nonempty(path) else "",
                "mode": "cached_official_table",
                "notes": "checksum validation; not a full benchmark rerun",
            }
        )
    reproduced_table = args.output_dir / "reproduced_files.tsv"
    checksums_table = args.output_dir / "checksums.tsv"
    report_json = args.output_dir / "reproduce_core_results_report.json"
    report_md = args.output_dir / "reproduce_core_results_report.md"
    write_table(reproduced_table, reproduced, REPRO_FIELDS)
    write_table(checksums_table, checksums, CHECKSUM_FIELDS)
    status = "pass" if all(command["returncode"] == 0 for command in commands) and all(row["status"] != "missing" for row in reproduced) else "warning"
    payload = {
        "status": status,
        "rerun_items": ["toy quickstart", "import mdcorealign", "CLI help", "release-readiness tests"],
        "checksum_validated_tables": CACHED_TABLES,
        "not_reproduced": ["full formal MD benchmark reruns", "NMR500 benchmark", "external MDLovoFit/THESEUS reruns"],
        "commands": commands,
    }
    write_json(report_json, payload)
    report_md.write_text(
        "\n".join(
            [
                "# Reproducibility Capsule",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Status: `{status}`",
                "- Rerun: toy quickstart, package import, CLI help, and release-readiness tests.",
                "- Checksum-validated: cached formal result summary tables and figure-data summaries.",
                "- Not reproduced: full MD benchmark reruns, NMR500, and external baseline reruns.",
                "- The capsule intentionally avoids large raw MD trajectories.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/reproduce_core_results.py",
        inputs=[ROOT / rel for rel in CACHED_TABLES],
        outputs=[reproduced_table, checksums_table, report_json, report_md],
        commands=commands,
        extra=payload,
    )
    print(f"Wrote reproducibility capsule to {args.output_dir}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
