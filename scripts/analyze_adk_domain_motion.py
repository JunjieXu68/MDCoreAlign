"""Optional ADK domain-level interpretation using local canonical annotation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import read_table  # noqa: E402
from md_trajectory_focus_common import MD_FOCUS_ROOT, common_manifest, timestamp, write_table  # noqa: E402


OUTPUT_DIR = MD_FOCUS_ROOT / "adk_domain_motion"

FIELDS = [
    "domain",
    "n_residues",
    "mean_mobility_score",
    "mean_robust_RMSF",
    "mean_ordinary_RMSF",
    "fraction_core_residues",
    "mean_residual",
    "domain_core_enrichment",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_rows(root: Path) -> list[dict[str, object]]:
    """Build domain rows or an unavailable row if local annotation is missing."""

    domain_path = root / "results" / "higher_tier_external_biology" / "adk" / "adk_equilibrium_external_biology_residues.csv"
    flex_path = root / "results" / "flexibility" / "adk_flexibility_profile.csv"
    if not domain_path.exists() or not flex_path.exists():
        return [
            {
                "domain": "unavailable",
                "n_residues": "",
                "mean_mobility_score": "",
                "mean_robust_RMSF": "",
                "mean_ordinary_RMSF": "",
                "fraction_core_residues": "",
                "mean_residual": "",
                "domain_core_enrichment": "",
                "notes": "Local canonical ADK domain annotation was not available; analysis not run.",
            }
        ]
    domain_rows = [
        row
        for row in read_table(domain_path)
        if row.get("method_name") == "soft_weight_seq_contact_regularized"
    ]
    flex_rows = read_table(flex_path)
    if len(domain_rows) != len(flex_rows):
        return [
            {
                "domain": "unavailable",
                "n_residues": "",
                "mean_mobility_score": "",
                "mean_robust_RMSF": "",
                "mean_ordinary_RMSF": "",
                "fraction_core_residues": "",
                "mean_residual": "",
                "domain_core_enrichment": "",
                "notes": "Domain and flexibility row counts did not match; analysis not run.",
            }
        ]
    all_core_fraction = float(np.mean([int(row["hard_core_label"]) for row in domain_rows]))
    rows = []
    for domain in sorted({row["domain_label"] for row in domain_rows}):
        indices = [index for index, row in enumerate(domain_rows) if row["domain_label"] == domain]
        core_fraction = float(np.mean([int(domain_rows[index]["hard_core_label"]) for index in indices]))
        rows.append(
            {
                "domain": domain,
                "n_residues": len(indices),
                "mean_mobility_score": float(np.mean([float(domain_rows[index]["mobility_score"]) for index in indices])),
                "mean_robust_RMSF": float(np.mean([float(flex_rows[index]["robust_aligned_RMSF"]) for index in indices])),
                "mean_ordinary_RMSF": float(np.mean([float(flex_rows[index]["ordinary_aligned_RMSF"]) for index in indices])),
                "fraction_core_residues": core_fraction,
                "mean_residual": float(np.mean([float(domain_rows[index]["residual"]) for index in indices])),
                "domain_core_enrichment": core_fraction / all_core_fraction if all_core_fraction else "",
                "notes": "Domain labels from local canonical ADK external-biology table; RMSF from formal ADK flexibility profile.",
            }
        )
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    if rows and rows[0]["domain"] == "unavailable":
        answer = "ADK domain annotation unavailable locally; no biological interpretation is made."
    else:
        by_domain = {row["domain"]: row for row in rows}
        core = by_domain.get("CORE")
        lid = by_domain.get("LID")
        nmp = by_domain.get("NMP")
        answer = (
            f"CORE mean mobility={core['mean_mobility_score'] if core else 'NA'}, "
            f"LID={lid['mean_mobility_score'] if lid else 'NA'}, "
            f"NMP={nmp['mean_mobility_score'] if nmp else 'NA'}."
        )
    lines = [
        "# ADK Domain Motion Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        f"1. Is the ADK CORE domain less mobile than LID/NMP? {answer}",
        "2. Are flexible segments enriched in LID/NMP-like regions? Inspect mean mobility and fraction core residues by domain in `adk_domain_motion_summary.tsv`.",
        "3. Does this support stable-core alignment interpretation? It supports interpretation only if CORE has higher core enrichment and lower mobility than mobile domains.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run optional ADK domain analysis."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(SCRIPT_DIR.parent)
    write_table(args.output_dir / "adk_domain_motion_summary.tsv", rows, FIELDS)
    (args.output_dir / "adk_domain_motion_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_adk_domain_motion.py",
        output_dir=args.output_dir,
        systems=["adk_equilibrium"],
        output_files=[
            args.output_dir / "adk_domain_motion_summary.tsv",
            args.output_dir / "adk_domain_motion_summary.md",
        ],
    )
    print(f"Wrote ADK domain motion outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
