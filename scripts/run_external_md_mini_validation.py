"""Optional external public MD mini-validation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_quality_common import FINAL_ROOT, timestamp, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_ROOT / "external_md_mini_validation"
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "external_md_mini_validation.yaml"

FIELDS = [
    "system",
    "source_database",
    "trajectory_file",
    "topology_file",
    "selection",
    "n_frames",
    "n_residues",
    "runtime_algorithm_seconds",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "n_flexible_segments",
    "converged",
    "n_iterations",
    "preprocessing_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--download", action="store_true", help="Reserved; downloads are not performed unless implemented explicitly")
    return parser.parse_args()


def main() -> int:
    """Write not-applicable output unless a local config is provided."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = args.output_dir / "external_md_mini_validation.tsv"
    summary = args.output_dir / "external_md_mini_validation_summary.md"
    rows: list[dict[str, object]] = []
    status = "not_applicable"
    if args.config.exists():
        rows.append(
            {
                "system": "unavailable",
                "source_database": "config_present_but_runner_not_enabled",
                "trajectory_file": "",
                "topology_file": "",
                "selection": "",
                "preprocessing_status": "not_run",
                "notes": "Config file exists, but this conservative release candidate does not auto-run uncontrolled external datasets.",
            }
        )
        status = "config_present_not_run"
    write_table(table, rows, FIELDS)
    summary.write_text(
        "\n".join(
            [
                "# Optional External MD Mini-Validation",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Status: `{status}`",
                f"- Config path: `{args.config}`",
                "",
                "## Required Answers",
                "",
                f"1. How many external trajectories were analyzed? `{len(rows) if status != 'not_applicable' else 0}`.",
                "2. What source database did they come from? Not applicable without a local config and explicit user-provided files.",
                "3. Did the workflow run successfully? Not run; this optional analysis is non-blocking.",
                "4. Were core/non-core RMSF separation and mobility-RMSF association observed? Not evaluated.",
                "5. Were there preprocessing or format issues? Not evaluated.",
                "6. Should this analysis be main text or supplementary? Supplementary only if later enabled with controlled local data.",
                "",
                "No websites were scraped and no downloads were attempted.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/run_external_md_mini_validation.py",
        inputs=[args.config],
        outputs=[table, summary],
        extra={"status": status, "download_attempted": bool(args.download and False), "random_seed": None},
    )
    print(f"Wrote external MD mini-validation status to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
