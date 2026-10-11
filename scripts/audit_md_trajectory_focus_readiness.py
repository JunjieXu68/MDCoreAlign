"""Audit readiness of the MD-trajectory-focused validation package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import read_table, sha256_file, write_json, write_table  # noqa: E402
from md_trajectory_focus_common import MD_FOCUS_ROOT, ROOT  # noqa: E402


OUTPUT_DIR = MD_FOCUS_ROOT / "readiness_audit"
FIELDS = ["component", "status", "evidence", "notes"]

REQUIRED = [
    ("temporal block stability", MD_FOCUS_ROOT / "temporal_block_stability" / "temporal_block_stability.tsv", "blocker"),
    ("stride sensitivity", MD_FOCUS_ROOT / "stride_sensitivity" / "stride_sensitivity.tsv", "blocker"),
    ("trajectory length dependence", MD_FOCUS_ROOT / "trajectory_length_dependence" / "trajectory_length_dependence.tsv", "blocker"),
    ("reference sensitivity", MD_FOCUS_ROOT / "reference_sensitivity" / "reference_sensitivity.tsv", "warning"),
    ("warm-start ablation", MD_FOCUS_ROOT / "warm_start_ablation" / "warm_start_ablation.tsv", "warning"),
    ("preprocessing manifest", MD_FOCUS_ROOT / "preprocessing_manifest" / "md_preprocessing_manifest.tsv", "blocker"),
    ("per-frame residuals", MD_FOCUS_ROOT / "per_frame_residuals" / "per_frame_residuals.tsv", "blocker"),
]

OPTIONAL = [
    ("ADK domain motion", MD_FOCUS_ROOT / "adk_domain_motion" / "adk_domain_motion_summary.tsv"),
    ("PCA alignment effect", MD_FOCUS_ROOT / "pca_alignment_effect" / "pca_alignment_effect.tsv"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def non_empty_table(path: Path) -> bool:
    """Return whether a table exists and has at least one row."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return len(read_table(path, delimiter="\t")) > 0
    except Exception:
        return False


def audit_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build checklist rows and blockers."""

    rows = []
    blockers = []
    for component, path, missing_status in REQUIRED:
        ok = non_empty_table(path)
        status = "pass" if ok else missing_status
        row = {
            "component": component,
            "status": status,
            "evidence": str(path.relative_to(ROOT)),
            "notes": "required MD trajectory focus analysis" if ok else "missing or empty required analysis",
        }
        rows.append(row)
        if status == "blocker":
            blockers.append(row)
    for component, path in OPTIONAL:
        ok = non_empty_table(path)
        rows.append(
            {
                "component": component,
                "status": "pass" if ok else "not_applicable",
                "evidence": str(path.relative_to(ROOT)),
                "notes": "optional interpretation analysis" if ok else "optional analysis not available",
            }
        )
    return rows, blockers


def recommendation(rows: list[dict[str, object]], blockers: list[dict[str, object]]) -> str:
    """Return readiness recommendation."""

    if blockers:
        return "C. Not ready; MD trajectory validation gaps remain"
    warnings = sum(row["status"] == "warning" for row in rows)
    if warnings:
        return "B. Almost ready; only optional interpretation analyses missing"
    return "A. MD-trajectory-focused manuscript ready"


def build_summary(rows: list[dict[str, object]], blockers: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    rec = recommendation(rows, blockers)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("pass", "warning", "blocker", "not_applicable")}
    lines = [
        "# MD Trajectory Focus Readiness Audit",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        f"- Recommendation: `{rec}`.",
        f"- Status counts: `{counts}`.",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- {row['component']}: {row['notes']}" for row in blockers)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Main vs Supplementary",
            "",
            "- Main text candidates: temporal block stability, stride/length/reference sensitivity summaries, preprocessing manifest, per-frame residual behavior.",
            "- Supplementary candidates: ADK domain motion, PCA alignment effect, full per-system TSV tables.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path) -> None:
    """Write manifest."""

    files = [
        output_dir / "md_trajectory_focus_checklist.tsv",
        output_dir / "md_trajectory_focus_summary.md",
        output_dir / "blockers.json",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/audit_md_trajectory_focus_readiness.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.exists()},
        },
    )


def main() -> int:
    """Run readiness audit."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, blockers = audit_rows()
    write_table(args.output_dir / "md_trajectory_focus_checklist.tsv", rows, FIELDS)
    (args.output_dir / "md_trajectory_focus_summary.md").write_text(build_summary(rows, blockers))
    write_json(args.output_dir / "blockers.json", {"blockers": blockers, "recommendation": recommendation(rows, blockers)})
    write_manifest(args.output_dir)
    print(f"Wrote MD trajectory focus readiness audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
