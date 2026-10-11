"""Run true lambda_sequence/lambda_contact sensitivity reruns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import fmt, sha256_file, write_json  # noqa: E402
from submission_critical_common import (  # noqa: E402
    DEFAULT_CONTACT_LAMBDA,
    DEFAULT_SEQUENCE_LAMBDA,
    FORMAL_SYSTEMS,
    ROOT,
    load_formal_system,
    run_soft,
    write_table,
)


OUTPUT_DIR = ROOT / "results" / "submission_critical" / "lambda_sensitivity_rerun"

LAMBDA_SEQUENCE = [
    ("zero", 0.0),
    ("low", 0.3),
    ("default", DEFAULT_SEQUENCE_LAMBDA),
    ("high", 0.9),
]
LAMBDA_CONTACT = [
    ("zero", 0.0),
    ("low", 0.25),
    ("default", DEFAULT_CONTACT_LAMBDA),
    ("high", 0.75),
]

FIELDS = [
    "system",
    "lambda_sequence_label",
    "lambda_sequence_value",
    "lambda_contact_label",
    "lambda_contact_value",
    "is_default_setting",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "runtime_algorithm_seconds",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "core_fragmentation_score",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "n_flexible_segments",
    "converged",
    "n_iterations",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def parameter_grid() -> list[dict[str, object]]:
    """Return the fixed 4x4 lambda grid."""

    rows: list[dict[str, object]] = []
    for seq, contact in product(LAMBDA_SEQUENCE, LAMBDA_CONTACT):
        rows.append(
            {
                "lambda_sequence_label": seq[0],
                "lambda_sequence_value": seq[1],
                "lambda_contact_label": contact[0],
                "lambda_contact_value": contact[1],
                "is_default_setting": seq[0] == "default" and contact[0] == "default",
            }
        )
    return rows


def run_rerun(systems: list[str]) -> list[dict[str, object]]:
    """Run the true lambda sensitivity grid."""

    rows: list[dict[str, object]] = []
    grid = parameter_grid()
    for system_key in systems:
        system = load_formal_system(system_key)
        for params in grid:
            notes = "formal_lambda_rerun_completed"
            try:
                metrics, _, _ = run_soft(
                    system,
                    lambda_sequence=float(params["lambda_sequence_value"]),
                    lambda_contact=float(params["lambda_contact_value"]),
                )
            except Exception as exc:
                metrics = {
                    "unified_posthoc_RMSD": "",
                    "weighted_RMSD": "",
                    "runtime_algorithm_seconds": "",
                    "effective_core_size": "",
                    "n_core_residues": "",
                    "n_core_segments": "",
                    "core_fragmentation_score": "",
                    "mean_core_RMSF": "",
                    "mean_noncore_RMSF": "",
                    "noncore_core_RMSF_ratio": "",
                    "spearman_robust_RMSF_vs_mobility": "",
                    "n_flexible_segments": "",
                    "converged": "unavailable",
                    "n_iterations": "",
                }
                notes = f"unavailable: {type(exc).__name__}: {exc}"
            row = {
                "system": system.label,
                **params,
                "is_default_setting": "yes" if params["is_default_setting"] else "no",
                **metrics,
                "notes": notes,
            }
            rows.append(row)
            print(
                "[lambda] "
                f"{system.label} seq={params['lambda_sequence_label']} "
                f"contact={params['lambda_contact_label']}: {notes}"
            )
    return rows


def _values(rows: list[dict[str, object]], field: str) -> list[float]:
    """Extract finite float values."""

    out = []
    for row in rows:
        try:
            text = str(row.get(field, "")).strip()
            if text:
                out.append(float(text))
        except ValueError:
            pass
    return out


def _range_text(rows: list[dict[str, object]], field: str) -> str:
    """Return compact min-max text."""

    values = _values(rows, field)
    if not values:
        return "unavailable"
    return f"{fmt(float(np.min(values)))} to {fmt(float(np.max(values)))}"


def build_justification(rows: list[dict[str, object]]) -> str:
    """Build default-parameter justification Markdown."""

    by_system: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_system.setdefault(str(row["system"]), []).append(row)
    default_rows = [row for row in rows if row["is_default_setting"] == "yes"]
    lines = [
        "# True Lambda Sensitivity Rerun: Default Parameter Justification",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        "- This is a true rerun of the alignment procedure for each lambda pair, not a post-hoc profile relabeling.",
        "- Defaults were not tuned per system.",
        "",
        "## Required Questions",
        "",
        "1. Are conclusions stable around the default lambda values? Inspect default rows against low/high rows in `lambda_sensitivity_rerun.tsv`; stability should be judged by RMSD guardrails and continuity/RMSF metrics.",
        f"2. Which metric is most sensitive to lambda_sequence? Across all rows, core segment count spans `{_range_text(rows, 'n_core_segments')}` and unified RMSD spans `{_range_text(rows, 'unified_posthoc_RMSD')}`.",
        f"3. Which metric is most sensitive to lambda_contact? The same grid enables contact-specific inspection; no per-system best lambda is selected here.",
        "4. Do lambda values mainly affect geometry, core continuity, or RMSF interpretation? The grid records all three; geometry should be treated as a guardrail unless changes are large and consistent.",
        "5. Is there evidence that per-system tuning is needed? This script does not tune per system; any such claim would require a separate locked validation design.",
        "6. Should the default values remain unchanged for the manuscript? Yes by default unless the rerun table shows clear geometry degradation or unstable core metrics around the default.",
        "7. Which results should go to main text and which should go to Supplementary? Main text can cite a compact default-vs-neighbor stability statement; the full 64-row grid belongs in Supplementary.",
        "",
        "## Default Rows",
        "",
    ]
    for row in default_rows:
        lines.append(
            f"- {row['system']}: RMSD={fmt(row['unified_posthoc_RMSD'])}, "
            f"weighted_RMSD={fmt(row['weighted_RMSD'])}, "
            f"core_segments={row['n_core_segments']}, "
            f"noncore/core RMSF ratio={fmt(row['noncore_core_RMSF_ratio'])}."
        )
    return "\n".join(lines) + "\n"


def plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return plot-ready sensitivity rows."""

    return rows


def write_manifest(output_dir: Path, systems: list[str]) -> None:
    """Write manifest."""

    files = [
        output_dir / "lambda_sensitivity_rerun.tsv",
        output_dir / "lambda_sensitivity_plot_data.tsv",
        output_dir / "default_parameter_justification.md",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/rerun_lambda_sensitivity_formal_systems.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "systems": systems,
            "lambda_sequence_grid": LAMBDA_SEQUENCE,
            "lambda_contact_grid": LAMBDA_CONTACT,
            "row_count_expected": len(systems) * len(parameter_grid()),
            "output_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.exists()},
        },
    )


def main() -> int:
    """Run true lambda sensitivity."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_rerun(args.systems)
    write_table(args.output_dir / "lambda_sensitivity_rerun.tsv", rows, FIELDS)
    write_table(args.output_dir / "lambda_sensitivity_plot_data.tsv", plot_rows(rows), FIELDS)
    (args.output_dir / "default_parameter_justification.md").write_text(build_justification(rows))
    write_manifest(args.output_dir, args.systems)
    print(f"Wrote lambda sensitivity rerun outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
