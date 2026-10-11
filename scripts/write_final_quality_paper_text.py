"""Write conservative final-quality manuscript text snippets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_quality_common import FINAL_ROOT, read_table, timestamp, write_manifest  # noqa: E402


OUTPUT_DIR = FINAL_ROOT / "paper_text_updates"
PREPROCESSING_DIR = FINAL_ROOT / "preprocessing_verification"
TEMPORAL_DIR = FINAL_ROOT / "temporal_convergence"
REFERENCE_DIR = FINAL_ROOT / "reference_consensus"
USER_DIR = FINAL_ROOT / "user_readiness"
EXTERNAL_DIR = FINAL_ROOT / "external_md_mini_validation"
SELECTION_DIR = FINAL_ROOT / "atom_selection_sensitivity"


FILES = [
    "methods_preprocessing_verification.md",
    "methods_reference_consensus.md",
    "methods_clean_install_reproducibility.md",
    "results_preprocessing_verification.md",
    "results_temporal_convergence_classification.md",
    "results_reference_consensus.md",
    "results_user_readiness.md",
    "optional_results_external_md_validation.md",
    "optional_results_atom_selection_sensitivity.md",
    "limitations_final_quality_update.md",
    "availability_and_reproducibility_final.md",
    "supplementary_allocation_final_quality.md",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _safe_rows(path: Path) -> list[dict[str, str]]:
    """Read a table if it exists."""

    return read_table(path) if path.exists() else []


def _count_temporal_classes() -> dict[str, int]:
    """Return temporal convergence class counts."""

    rows = _safe_rows(TEMPORAL_DIR / "temporal_convergence_classification.tsv")
    return dict(Counter(row.get("classification", "unknown") for row in rows))


def _preprocessing_summary() -> tuple[int, int, int]:
    """Return count of preprocessing rows, jump rows, and recenter rows."""

    rows = _safe_rows(PREPROCESSING_DIR / "pbc_centering_verification.tsv")
    recentered = _safe_rows(PREPROCESSING_DIR / "recentered_comparison.tsv")
    jump_count = sum(str(row.get("large_jump_detected", "")).lower() == "true" for row in rows)
    recentered_count = sum(str(row.get("recentered_rerun_performed", "")).lower() == "true" for row in recentered)
    return len(rows), jump_count, recentered_count


def _reference_range() -> tuple[str, str]:
    """Return compact reference sensitivity range text."""

    rows = _safe_rows(REFERENCE_DIR / "reference_consensus_system_summary.tsv")
    if not rows:
        return "unavailable", "unavailable"
    jaccards = [row.get("lowest_pairwise_core_jaccard", "") for row in rows if row.get("lowest_pairwise_core_jaccard", "")]
    spearmans = [
        row.get("lowest_pairwise_mobility_spearman", "")
        for row in rows
        if row.get("lowest_pairwise_mobility_spearman", "")
    ]
    return (min(jaccards) if jaccards else "unavailable", min(spearmans) if spearmans else "unavailable")


def build_texts() -> dict[str, str]:
    """Build all final-quality text snippets."""

    n_pre, n_jump, n_recenter = _preprocessing_summary()
    classes = _count_temporal_classes()
    min_jaccard, min_spearman = _reference_range()
    clean_exists = (USER_DIR / "clean_install_report.md").exists()
    build_exists = (USER_DIR / "package_build_report.md").exists()
    toy_exists = (USER_DIR / "toy_quickstart_report.md").exists()
    generated = timestamp()

    return {
        "methods_preprocessing_verification.md": "\n".join(
            [
                "# Methods: MD Preprocessing Verification Addendum",
                "",
                f"Generated: `{generated}`.",
                "",
                "For the final-quality review pass, the formal MD trajectory inputs were inspected with lightweight coordinate sanity checks implemented in `scripts/verify_md_preprocessing_pbc_centering.py`. The checks evaluated residue/chain metadata continuity, frame-to-frame selected-CA displacement, center-of-geometry drift, and availability of box-vector-like metadata where present.",
                "",
                "These checks are intentionally conservative. They do not replace upstream MD preprocessing and they do not prove that PBC unwrapping, nojump processing, or centering was performed correctly before the coordinates were provided to MDCoreAlign.",
                "",
                "Required manuscript limitation: Formal inputs were analyzed as provided; PBC/nojump and centering status could not be independently verified from local input metadata.",
                "",
                "A diagnostic recentering comparison removes only per-frame center of geometry and is used as a sensitivity check, not as a PBC/nojump repair.",
                "",
                "Primary files: `results/final_quality_upgrade/preprocessing_verification/pbc_centering_verification.tsv` and `results/final_quality_upgrade/preprocessing_verification/recentered_comparison.tsv`.",
            ]
        ),
        "methods_reference_consensus.md": "\n".join(
            [
                "# Methods: Reference-Consensus Diagnostics",
                "",
                f"Generated: `{generated}`.",
                "",
                "Reference sensitivity was summarized from the existing formal reference-strategy diagnostics. The current final-quality pass does not claim reference independence. Instead, stable-core and mobility interpretations are reported as reference-aware quantities.",
                "",
                "Residue-level reference-consensus labels were not inferred because per-reference residue-level profiles were not available in the frozen formal outputs. The system-level table therefore reports pairwise agreement ranges relative to the default first-frame reference and reserves residue-level consensus as a future or supplementary rerun target.",
                "",
                "Primary files: `results/final_quality_upgrade/reference_consensus/reference_consensus_system_summary.tsv` and `results/final_quality_upgrade/reference_consensus/reference_consensus_summary.md`.",
            ]
        ),
        "methods_clean_install_reproducibility.md": "\n".join(
            [
                "# Methods: Clean-Install Reproducibility",
                "",
                f"Generated: `{generated}`.",
                "",
                "Package-level reproducibility was checked with source-install, command-line help, toy quickstart, and package build validation scripts. Core MDCoreAlign commands are designed to run without installed MDLovoFit or THESEUS executables; those external tools remain optional baselines.",
                "",
                "The clean-install check creates an isolated environment, installs the package from the source checkout, verifies `import mdcorealign`, runs all documented CLI help commands, executes the bundled toy workflow, and runs the release-readiness tests.",
                "",
                "Primary files: `results/final_quality_upgrade/user_readiness/clean_install_report.md`, `results/final_quality_upgrade/user_readiness/toy_quickstart_report.md`, and `results/final_quality_upgrade/user_readiness/package_build_report.md`.",
            ]
        ),
        "results_preprocessing_verification.md": "\n".join(
            [
                "# Results: Preprocessing Verification",
                "",
                f"Generated: `{generated}`.",
                "",
                f"The final-quality preprocessing table contains `{n_pre}` formal MD systems. Large frame-to-frame selected-CA jumps were flagged in `{n_jump}` systems by the lightweight threshold check. Diagnostic recentering comparisons were successfully produced for `{n_recenter}` systems when the local data and dependencies allowed.",
                "",
                "These diagnostics support transparent reporting of input-coordinate sanity checks, but they do not establish upstream PBC/nojump provenance. The manuscript should state that the formal inputs were analyzed as provided and that PBC/nojump and centering status could not be independently verified from local input metadata.",
            ]
        ),
        "results_temporal_convergence_classification.md": "\n".join(
            [
                "# Results: Temporal Convergence Classification",
                "",
                f"Generated: `{generated}`.",
                "",
                f"Temporal non-full block rows classified in this pass: `{sum(classes.values())}`. Class counts: `{classes}`.",
                "",
                "Rows classified as `max_iter_reached_but_stable` should be described as conservative max-iteration stopping rather than numerical failure. Rows classified as `high_reference_sensitivity` should be kept as diagnostics of time-window or reference sensitivity. Rows classified as `true_failure` or `weight_collapse`, if present in a later rerun, should be excluded from primary conclusions.",
                "",
                "Primary file: `results/final_quality_upgrade/temporal_convergence/temporal_convergence_classification.tsv`.",
            ]
        ),
        "results_reference_consensus.md": "\n".join(
            [
                "# Results: Reference Sensitivity and Consensus",
                "",
                f"Generated: `{generated}`.",
                "",
                f"System-level reference diagnostics gave a lowest available core-Jaccard value of `{min_jaccard}` and a lowest available mobility-Spearman value of `{min_spearman}` relative to the default reference among the frozen formal rows.",
                "",
                "These values should be interpreted as evidence that reference choice can affect stable-core and mobility summaries. The manuscript should not claim reference independence. If residue-level consensus cores are desired, they should be generated by a dedicated per-reference residue-profile rerun.",
            ]
        ),
        "results_user_readiness.md": "\n".join(
            [
                "# Results: User Readiness",
                "",
                f"Generated: `{generated}`.",
                "",
                f"Clean-install report present: `{clean_exists}`. Toy quickstart report present: `{toy_exists}`. Package build and twine-check report present: `{build_exists}`.",
                "",
                "The intended user-facing workflow is: install the package, run `mdcorealign --help`, execute the toy quickstart, and inspect the generated aligned trajectory, weights, flexibility profile, flexible segments, and summary JSON.",
            ]
        ),
        "optional_results_external_md_validation.md": "\n".join(
            [
                "# Optional Results: External MD Mini-Validation",
                "",
                f"Generated: `{generated}`.",
                "",
                "No uncontrolled public trajectory downloads or website scraping were performed in the final-quality release-candidate pass. If external MD mini-validation is added later, it should use explicit local input files, recorded preprocessing status, and the same algorithm-runtime fairness rules as the formal benchmark systems.",
                "",
                "Primary file: `results/final_quality_upgrade/external_md_mini_validation/external_md_mini_validation.tsv`.",
            ]
        ),
        "optional_results_atom_selection_sensitivity.md": "\n".join(
            [
                "# Optional Results: Atom-Selection Sensitivity",
                "",
                f"Generated: `{generated}`.",
                "",
                "The formal release-candidate workflow remains CA-only. Backbone sensitivity was not inferred from array index alone because a validated residue-level backbone-to-CA aggregation and matching protocol is not frozen.",
                "",
                "CA-only should therefore be described as the primary formal representation. Backbone or all-heavy-atom sensitivity belongs in supplementary/future validation unless a dedicated validated rerun is added.",
            ]
        ),
        "limitations_final_quality_update.md": "\n".join(
            [
                "# Limitations: Final Quality Update",
                "",
                f"Generated: `{generated}`.",
                "",
                "- MDCoreAlign does not prove upstream PBC unwrapping, nojump processing, or centering from generic local coordinate files.",
                "- Stable-core and mobility outputs are reference-aware. Do not claim reference independence.",
                "- Warm start is an efficiency option and should not be described as scientifically necessary.",
                "- THESEUS results should not be described as rerun locally when only existing tables were validated.",
                "- YiiP should not be overclaimed as externally biologically validated; it remains a challenging membrane-protein case with limited external biological-proxy support.",
                "- RMSF is alignment-frame dependent; ordinary-aligned RMSF and robust-aligned RMSF are two alignment-conditioned flexibility profiles.",
                "- NMR500 is supplementary structural-ensemble evidence if the manuscript is MD-trajectory focused.",
                "- CA-only is the frozen formal representation unless a dedicated atom-selection sensitivity rerun is added.",
            ]
        ),
        "availability_and_reproducibility_final.md": "\n".join(
            [
                "# Availability and Reproducibility",
                "",
                f"Generated: `{generated}`.",
                "",
                "The release-candidate package exposes the `mdcorealign` Python package and the `mdcorealign` console command. The repository includes a toy CA-only trajectory, a one-command quickstart, installation documentation, CLI documentation, Python API notes, output-format documentation, preprocessing guidance, troubleshooting notes, license, citation metadata, changelog, and GitHub Actions configuration.",
                "",
                "The package build is checked with `python -m build` and `twine check dist/*`. TestPyPI upload remains a release-management step after the GitHub remote and project URLs are finalized.",
            ]
        ),
        "supplementary_allocation_final_quality.md": "\n".join(
            [
                "# Supplementary Allocation: Final Quality Update",
                "",
                f"Generated: `{generated}`.",
                "",
                "Recommended main-text material: four-system MD trajectory benchmark summary, flexibility-profile summary, preprocessing limitation statement, temporal convergence classification summary, and user-facing package quickstart.",
                "",
                "Recommended supplementary material: full preprocessing verification tables, temporal block classifications, reference-sensitivity system-level table, optional external MD mini-validation status, optional atom-selection sensitivity status, NMR500 structural-ensemble evidence, THESEUS validation caveat, and clean-install/package-build reports.",
            ]
        ),
    }


def main() -> int:
    """Write final-quality paper text snippets."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts = build_texts()
    output_paths = []
    for name in FILES:
        path = args.output_dir / name
        path.write_text(texts[name] + "\n")
        output_paths.append(path)
    write_manifest(
        args.output_dir,
        script="scripts/write_final_quality_paper_text.py",
        inputs=[
            PREPROCESSING_DIR / "pbc_centering_verification.tsv",
            TEMPORAL_DIR / "temporal_convergence_classification.tsv",
            REFERENCE_DIR / "reference_consensus_system_summary.tsv",
            USER_DIR / "clean_install_report.md",
            USER_DIR / "package_build_report.md",
            USER_DIR / "toy_quickstart_report.md",
            EXTERNAL_DIR / "external_md_mini_validation.tsv",
            SELECTION_DIR / "atom_selection_sensitivity.tsv",
        ],
        outputs=output_paths,
        extra={"text_files": FILES},
    )
    print(f"Wrote final-quality paper text snippets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
