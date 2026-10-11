"""Conservative additional YiiP biology-proxy analysis from local files only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import fmt, read_table, sha256_file, spearman, write_json, write_table  # noqa: E402
from submission_critical_common import ROOT  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "submission_critical" / "yiip_biology_proxy"

FIELDS = [
    "dataset_name",
    "proxy_name",
    "status",
    "n_pairs_or_residues",
    "metric_name",
    "metric_value",
    "interpretation",
    "source_file",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def chain_symmetry_rows() -> list[dict[str, object]]:
    """Measure A/B chain consistency for duplicated YiiP chains."""

    profile_path = ROOT / "results" / "flexibility" / "yiip_short_flexibility_profile.csv"
    rows = read_table(profile_path)
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["chain"], row["resname"] + row["residue_id"])
        by_key[key] = row
    paired = []
    for row in rows:
        if row["chain"] != "A":
            continue
        key_b = ("B", row["resname"] + row["residue_id"])
        if key_b in by_key:
            paired.append((row, by_key[key_b]))
    metrics = []
    for field in ("mobility_score", "soft_stability_weight", "robust_aligned_RMSF"):
        x = [float(a[field]) for a, _ in paired]
        y = [float(b[field]) for _, b in paired]
        metrics.append(
            {
                "dataset_name": "yiip_equilibrium_short",
                "proxy_name": "A_B_chain_symmetry_consistency",
                "status": "weak_support",
                "n_pairs_or_residues": len(paired),
                "metric_name": f"spearman_A_vs_B_{field}",
                "metric_value": spearman(x, y),
                "interpretation": "Higher positive correlation indicates internally consistent chain-level mobility/stability patterns, not direct functional validation.",
                "source_file": str(profile_path.relative_to(ROOT)),
                "notes": "Uses local formal flexibility profile only; no curated YiiP functional annotation was available.",
            }
        )
    core_agreement = float(np.mean([int(a["core_label"]) == int(b["core_label"]) for a, b in paired])) if paired else None
    metrics.append(
        {
            "dataset_name": "yiip_equilibrium_short",
            "proxy_name": "A_B_chain_symmetry_consistency",
            "status": "weak_support",
            "n_pairs_or_residues": len(paired),
            "metric_name": "A_B_core_label_agreement",
            "metric_value": core_agreement,
            "interpretation": "Agreement of hard core labels between homologous chains is a conservative structural sanity check.",
            "source_file": str(profile_path.relative_to(ROOT)),
            "notes": "Not a substitute for external biological annotation.",
        }
    )
    return metrics


def interface_proxy_rows() -> list[dict[str, object]]:
    """Summarize existing interface-density proxy conservatively."""

    summary_path = ROOT / "results" / "higher_tier_external_biology" / "yiip" / "yiip_equilibrium_short_external_biology_summary.csv"
    if not summary_path.exists():
        return [
            {
                "dataset_name": "yiip_equilibrium_short",
                "proxy_name": "interface_density_existing_proxy",
                "status": "inconclusive",
                "n_pairs_or_residues": "",
                "metric_name": "missing",
                "metric_value": "",
                "interpretation": "No existing local interface-density table was found.",
                "source_file": str(summary_path.relative_to(ROOT)),
                "notes": "YiiP remains biology-proxy limited.",
            }
        ]
    out = []
    for row in read_table(summary_path):
        if row.get("method_name") != "soft_weight_seq_contact_regularized":
            continue
        spearman_value = row.get("spearman_stability_vs_inter_subunit_contact_density", "")
        enrichment = row.get("high_interface_core_enrichment", "")
        out.append(
            {
                "dataset_name": "yiip_equilibrium_short",
                "proxy_name": "interface_density_existing_proxy",
                "status": "weak_support",
                "n_pairs_or_residues": row.get("n_residues", ""),
                "metric_name": "spearman_stability_vs_inter_subunit_contact_density",
                "metric_value": spearman_value,
                "interpretation": "Existing interface-density proxy is weak and should not be framed as strong biological validation.",
                "source_file": str(summary_path.relative_to(ROOT)),
                "notes": f"high_interface_core_enrichment={enrichment}; local proxy only.",
            }
        )
    return out


def classify(rows: list[dict[str, object]]) -> str:
    """Classify overall YiiP proxy status conservatively."""

    values = []
    for row in rows:
        try:
            if row["metric_value"] != "":
                values.append(abs(float(row["metric_value"])))
        except (TypeError, ValueError):
            pass
    if not values:
        return "inconclusive"
    if max(values) >= 0.3:
        return "weak_support"
    return "inconclusive"


def build_summary(rows: list[dict[str, object]], status: str) -> str:
    """Build Markdown summary."""

    lines = [
        "# YiiP Additional Biology Proxy",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        f"- Overall classification: `{status}`.",
        "- No curated transmembrane-region, loop/periplasmic/cytoplasmic, or functional-residue annotation file was found locally.",
        "- Therefore this analysis remains a conservative proxy/sanity check, not strong external biological validation.",
        "",
        "## Metrics",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['proxy_name']} / {row['metric_name']}: {fmt(row['metric_value'])} ({row['status']}).")
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path, status: str) -> None:
    """Write manifest."""

    files = [
        output_dir / "yiip_proxy_summary.tsv",
        output_dir / "yiip_proxy_summary.md",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/analyze_yiip_additional_biology_proxy.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "output_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.exists()},
        },
    )


def main() -> int:
    """Run YiiP proxy analysis."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [*chain_symmetry_rows(), *interface_proxy_rows()]
    status = classify(rows)
    write_table(args.output_dir / "yiip_proxy_summary.tsv", rows, FIELDS)
    (args.output_dir / "yiip_proxy_summary.md").write_text(build_summary(rows, status))
    write_manifest(args.output_dir, status)
    print(f"Wrote YiiP proxy analysis to {args.output_dir} ({status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
