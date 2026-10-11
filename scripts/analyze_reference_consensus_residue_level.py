"""Attempt residue-level reference consensus when per-reference profiles exist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import FINAL_SUBMISSION_ROOT, timestamp, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "reference_consensus_residue_level"
RESIDUE_FIELDS = [
    "system",
    "chain",
    "residue_id",
    "resname",
    "n_reference_strategies",
    "core_frequency",
    "mean_mobility_score",
    "sd_mobility_score",
    "min_mobility_score",
    "max_mobility_score",
    "reference_sensitive",
    "reference_sensitive_reason",
    "consensus_core_label",
    "consensus_mobile_label",
    "default_reference_core_label",
    "default_reference_mobility_score",
]
SYSTEM_FIELDS = [
    "system",
    "status",
    "n_residues",
    "n_reference_strategies",
    "n_consensus_core",
    "n_consensus_mobile",
    "n_reference_sensitive_residues",
    "fraction_reference_sensitive",
    "notes",
]
SYSTEMS = ["I-FABP", "ADK", "NhaA", "YiiP short"]
REFERENCE_STRATEGIES = ["first_frame", "middle_frame", "last_frame", "ordinary_mean_structure", "robust_mean_structure"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=ROOT / "results" / "md_trajectory_focus" / "reference_sensitivity" / "per_reference_profiles",
    )
    return parser.parse_args()


def main() -> int:
    """Write residue-consensus outputs or explicit unavailable status."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_files = sorted(args.profiles_dir.glob("*.tsv")) if args.profiles_dir.exists() else []
    residue_rows: list[dict[str, object]] = []
    system_rows: list[dict[str, object]] = []
    if not profile_files:
        for system in SYSTEMS:
            system_rows.append(
                {
                    "system": system,
                    "status": "unavailable",
                    "n_reference_strategies": 0,
                    "notes": "Per-reference residue-level weights/mobility profiles were not available; no residue-level consensus was inferred by array index.",
                }
            )
    else:
        # Reserved path for future frozen per-reference residue profiles.
        for system in SYSTEMS:
            system_rows.append(
                {
                    "system": system,
                    "status": "unavailable_parser_not_enabled",
                    "n_reference_strategies": len(REFERENCE_STRATEGIES),
                    "notes": "Profile files were detected, but no frozen schema parser is implemented in this release-candidate pass.",
                }
            )
    residue_table = args.output_dir / "reference_consensus_residues.tsv"
    system_table = args.output_dir / "reference_consensus_system_summary.tsv"
    summary = args.output_dir / "reference_consensus_residue_summary.md"
    write_table(residue_table, residue_rows, RESIDUE_FIELDS)
    write_table(system_table, system_rows, SYSTEM_FIELDS)
    summary.write_text(
        "\n".join(
            [
                "# Residue-Level Reference Consensus",
                "",
                f"- Generated at: `{timestamp()}`",
                "- Status: `unavailable`",
                "- Residue-level per-reference weights or mobility profiles were not present in the frozen formal outputs.",
                "- No residue-level labels were inferred by array index.",
                "- Needed data: one residue-level mobility/core table per system and reference strategy with stable chain/residue identifiers.",
                "- No reference-independence claim should be made.",
                "- If generated later, this belongs in Supplementary unless promoted explicitly by the author.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/analyze_reference_consensus_residue_level.py",
        inputs=profile_files,
        outputs=[residue_table, system_table, summary],
        extra={
            "status": "unavailable" if not profile_files else "unavailable_parser_not_enabled",
            "consensus_core_threshold": 0.8,
            "consensus_mobile_threshold": 0.2,
            "reference_sensitive_definition": "core label changes OR mobility SD >= system-specific 75th percentile",
            "reference_strategies": REFERENCE_STRATEGIES,
        },
    )
    print(f"Wrote residue-level reference consensus status to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
