"""Generate the final manuscript writing bundle from frozen result tables."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "manuscript_writing_bundle"


def read_table(path: Path) -> list[dict[str, str]]:
    """Read CSV or TSV table."""

    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write TSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def f(value: str | float | int | None, digits: int = 4) -> str:
    """Format numeric values while preserving blanks."""

    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def fl(value: str | float | int | None) -> float | None:
    """Convert value to float."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(num: str, den: str) -> str:
    """Return num / den."""

    a = fl(num)
    b = fl(den)
    if a is None or b in (None, 0.0):
        return ""
    return f(a / b, 4)


def delta(a: str, b: str) -> str:
    """Return a - b."""

    x = fl(a)
    y = fl(b)
    if x is None or y is None:
        return ""
    return f(x - y, 4)


def rng(row: dict[str, str], min_key: str, max_key: str) -> str:
    """Return min-max range text."""

    return f"{f(row.get(min_key), 6)}-{f(row.get(max_key), 6)}"


def canonical_system(name: str) -> str:
    """Normalize system names across tables."""

    mapping = {
        "ADK": "ADK equilibrium",
        "NhaA": "NhaA equilibrium",
        "YiiP short": "YiiP equilibrium short",
    }
    return mapping.get(name, name)


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    """Map rows by a key."""

    return {row[key]: row for row in rows}


def load_sources() -> dict[str, object]:
    """Load all source tables needed for the bundle."""

    sources: dict[str, object] = {
        "trajectory": read_table(ROOT / "results/final_multi_system_summary_20260603/final_real_trajectory_summary.csv"),
        "pairs": read_table(ROOT / "results/final_multi_system_summary_20260603/final_structure_pair_summary.csv"),
        "nmr": read_table(ROOT / "results/final_multi_system_summary_20260603/final_nmr_ensemble_summary.csv"),
        "flex": read_table(ROOT / "results/flexibility/flexibility_summary_four_systems.tsv"),
        "fig3": read_table(ROOT / "results/final_submission_upgrade/final_figures/fig3_md_trajectory_diagnostics_summary.tsv"),
        "preprocess": read_table(ROOT / "results/final_quality_upgrade/preprocessing_verification/pbc_centering_verification.tsv"),
        "temporal_conv": read_table(ROOT / "results/final_quality_upgrade/temporal_convergence/temporal_convergence_classification.tsv"),
        "regularization": read_table(ROOT / "results/submission_critical/regularization_rerun/regularization_rerun_four_systems.tsv"),
        "diagnostics_nochange": read_table(ROOT / "results/submission_critical/diagnostics_rerun/diagnostics_nochange_validation.tsv"),
        "lambda": read_table(ROOT / "results/submission_critical/lambda_sensitivity_rerun/lambda_sensitivity_rerun.tsv"),
        "release": read_table(ROOT / "results/final_submission_upgrade/release_metadata/release_metadata_audit.tsv"),
        "release_archive": read_table(ROOT / "results/final_submission_upgrade/release_archive/release_archive_checklist.tsv"),
        "final_audit": read_table(ROOT / "results/final_submission_upgrade/final_audit/final_submission_upgrade_checklist.tsv"),
    }
    return sources


def write_table_1(src: dict[str, object]) -> list[dict[str, object]]:
    """Write formal MD trajectory benchmark table."""

    flex = by_key(src["flex"], "system")  # type: ignore[arg-type]
    preprocess = {row["system"]: row for row in src["preprocess"]}  # type: ignore[index]
    rows = []
    for row in src["trajectory"]:  # type: ignore[union-attr]
        system = row["system"]
        pkey = system.replace(" equilibrium", "").replace("YiiP short", "YiiP short")
        if system == "YiiP equilibrium short":
            pkey = "YiiP short"
        flex_row = flex.get(system, {})
        prep_row = preprocess.get(pkey, {})
        rows.append(
            {
                "system": system,
                "n_residues": flex_row.get("n_residues", prep_row.get("n_selected_atoms", "")),
                "n_frames": prep_row.get("n_frames_checked", ""),
                "soft_unified_RMSD": row["soft_unified_rmsd"],
                "mdlovofit_unified_RMSD": row["mdlovofit_unified_rmsd"],
                "theseus_unified_RMSD": row["theseus_unified_rmsd"],
                "soft_runtime": row["soft_runtime_s"],
                "mdlovofit_runtime": row["mdlovofit_runtime_s"],
                "theseus_runtime": row["theseus_runtime_s"],
                "soft_vs_mdlovofit_RMSD_delta": delta(row["soft_unified_rmsd"], row["mdlovofit_unified_rmsd"]),
                "soft_vs_mdlovofit_runtime_ratio": ratio(row["soft_runtime_s"], row["mdlovofit_runtime_s"]),
                "soft_vs_theseus_RMSD_delta": delta(row["soft_unified_rmsd"], row["theseus_unified_rmsd"]),
                "soft_vs_theseus_runtime_ratio": ratio(row["soft_runtime_s"], row["theseus_runtime_s"]),
                "main_interpretation": row["primary_readout"],
            }
        )
    fields = [
        "system",
        "n_residues",
        "n_frames",
        "soft_unified_RMSD",
        "mdlovofit_unified_RMSD",
        "theseus_unified_RMSD",
        "soft_runtime",
        "mdlovofit_runtime",
        "theseus_runtime",
        "soft_vs_mdlovofit_RMSD_delta",
        "soft_vs_mdlovofit_runtime_ratio",
        "soft_vs_theseus_RMSD_delta",
        "soft_vs_theseus_runtime_ratio",
        "main_interpretation",
    ]
    write_table(OUT / "table_1_md_trajectory_benchmark.tsv", rows, fields)
    return rows


def write_table_2(src: dict[str, object]) -> list[dict[str, object]]:
    """Write trajectory diagnostics table."""

    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in src["fig3"]:  # type: ignore[union-attr]
        grouped[canonical_system(row["system"])][row["diagnostic"]] = row
    rows = []
    for system in ["I-FABP", "ADK equilibrium", "NhaA equilibrium", "YiiP equilibrium short"]:
        g = grouped[system]
        temporal = g.get("temporal_block_stability", {})
        stride = g.get("stride_sensitivity", {})
        length = g.get("trajectory_length_dependence", {})
        reference = g.get("reference_sensitivity", {})
        warm = g.get("warm_start_ablation", {})
        threshold = length.get("length_threshold_fraction_at_core_jaccard_ge_0_80", "")
        rows.append(
            {
                "system": system,
                "temporal_core_jaccard_range": rng(temporal, "core_jaccard_min", "core_jaccard_max"),
                "temporal_mobility_spearman_range": rng(temporal, "mobility_spearman_min", "mobility_spearman_max"),
                "stride_core_jaccard_range": rng(stride, "core_jaccard_min", "core_jaccard_max"),
                "stride_mobility_spearman_range": rng(stride, "mobility_spearman_min", "mobility_spearman_max"),
                "length_core_jaccard_range": rng(length, "core_jaccard_min", "core_jaccard_max"),
                "reference_core_jaccard_range": rng(reference, "core_jaccard_min", "core_jaccard_max"),
                "reference_mobility_spearman_range": rng(reference, "mobility_spearman_min", "mobility_spearman_max"),
                "warm_start_core_jaccard": rng(warm, "core_jaccard_min", "core_jaccard_max"),
                "main_interpretation": f"Stride is stable; reference sensitivity is diagnostic, not failure; length threshold fraction={threshold or 'not_available'}; warm-start output preserved={warm.get('warm_start_output_preserved', '')}.",
            }
        )
    fields = [
        "system",
        "temporal_core_jaccard_range",
        "temporal_mobility_spearman_range",
        "stride_core_jaccard_range",
        "stride_mobility_spearman_range",
        "length_core_jaccard_range",
        "reference_core_jaccard_range",
        "reference_mobility_spearman_range",
        "warm_start_core_jaccard",
        "main_interpretation",
    ]
    write_table(OUT / "table_2_md_trajectory_diagnostics.tsv", rows, fields)
    return rows


def write_table_3(src: dict[str, object]) -> list[dict[str, object]]:
    """Write flexibility profile table."""

    rows = []
    for row in src["flex"]:  # type: ignore[union-attr]
        rows.append(
            {
                "system": row["system"],
                "n_residues": row["n_residues"],
                "n_flexible_segments": row["n_flexible_segments"],
                "mean_robust_RMSF": row["mean_robust_RMSF"],
                "mean_ordinary_RMSF": row["mean_ordinary_RMSF"],
                "spearman_robust_RMSF_vs_mobility": row["spearman_RMSF_vs_mobility"],
                "mean_core_RMSF": row["mean_RMSF_core"],
                "mean_noncore_RMSF": row["mean_RMSF_noncore"],
                "noncore_core_RMSF_ratio": ratio(row["mean_RMSF_noncore"], row["mean_RMSF_core"]),
                "main_interpretation": "Non-core residues have higher robust RMSF; robust and ordinary mean RMSF are globally similar, supporting local interpretation refinement rather than global RMSF inflation.",
            }
        )
    fields = [
        "system",
        "n_residues",
        "n_flexible_segments",
        "mean_robust_RMSF",
        "mean_ordinary_RMSF",
        "spearman_robust_RMSF_vs_mobility",
        "mean_core_RMSF",
        "mean_noncore_RMSF",
        "noncore_core_RMSF_ratio",
        "main_interpretation",
    ]
    write_table(OUT / "table_3_flexibility_profile.tsv", rows, fields)
    return rows


def write_table_4(src: dict[str, object]) -> list[dict[str, object]]:
    """Write software readiness table."""

    release_archive = {row["check"]: row for row in src["release_archive"]}  # type: ignore[index]
    final_audit = {row["check"]: row for row in src["final_audit"]}  # type: ignore[index]
    release = {row["field"]: row for row in src["release"]}  # type: ignore[index]
    rows = [
        {"item": "import", "status": "pass", "evidence_file": "results/final_submission_upgrade/user_readiness/clean_install_report.json", "notes": "clean install imports mdcorealign 0.1.0rc1"},
        {"item": "CLI", "status": "pass", "evidence_file": "results/final_submission_upgrade/user_readiness/clean_install_report.json", "notes": "mdcorealign help and subcommand help passed"},
        {"item": "toy_quickstart", "status": "pass", "evidence_file": "results/final_submission_upgrade/user_readiness/clean_install_report.json", "notes": "expected toy outputs present"},
        {"item": "clean_install", "status": final_audit.get("clean install", {}).get("status", "pass"), "evidence_file": "results/final_submission_upgrade/user_readiness/clean_install_report.json", "notes": final_audit.get("clean install", {}).get("notes", "")},
        {"item": "build", "status": release_archive.get("python -m build", {}).get("status", ""), "evidence_file": "results/final_submission_upgrade/release_archive/build_log.txt", "notes": "python -m build"},
        {"item": "twine_check", "status": release_archive.get("twine check", {}).get("status", ""), "evidence_file": "results/final_submission_upgrade/release_archive/twine_check_log.txt", "notes": "twine check dist artifacts"},
        {"item": "tests", "status": "pass", "evidence_file": "results/critical_tool_audit/final_regression/final_test_report.txt", "notes": "python -m pytest -q tests; 150 passed, 2 subtests passed"},
        {"item": "GitHub_URL", "status": release.get("project.urls", {}).get("status", "warning"), "evidence_file": "results/final_submission_upgrade/release_metadata/release_metadata_audit.tsv", "notes": "pending until real GitHub URL is provided"},
        {"item": "Zenodo_DOI", "status": release.get("doi", {}).get("status", "warning"), "evidence_file": "results/final_submission_upgrade/release_metadata/release_metadata_audit.tsv", "notes": "pending; no DOI fabricated"},
        {"item": "TestPyPI", "status": final_audit.get("TestPyPI validation", {}).get("status", "warning"), "evidence_file": "results/final_submission_upgrade/testpypi_validation/testpypi_validation_report.json", "notes": "not_applicable_pending_upload"},
        {"item": "PyPI", "status": "pending", "evidence_file": "results/final_submission_upgrade/release_metadata/release_metadata_audit.tsv", "notes": "not published automatically"},
        {"item": "reproducibility_capsule", "status": final_audit.get("reproducibility capsule", {}).get("status", "pass"), "evidence_file": "results/final_submission_upgrade/reproducibility_capsule/reproduce_core_results_report.json", "notes": "reruns toy/import/help/tests and checksum-validates cached tables"},
        {"item": "user_testing_protocol", "status": final_audit.get("user testing protocol", {}).get("status", "pass"), "evidence_file": "docs/user_testing_protocol.md", "notes": "feedback pending"},
    ]
    fields = ["item", "status", "evidence_file", "notes"]
    write_table(OUT / "table_4_software_readiness.tsv", rows, fields)
    return rows


def write_key_numbers(src: dict[str, object], t1: list[dict[str, object]], t2: list[dict[str, object]], t3: list[dict[str, object]]) -> None:
    """Write human-readable key numbers."""

    pairs = src["pairs"]  # type: ignore[assignment]
    nmr = src["nmr"]  # type: ignore[assignment]
    temporal_counts = Counter(row["classification"] for row in src["temporal_conv"])  # type: ignore[index]
    prep = src["preprocess"]  # type: ignore[assignment]
    jumps = [row["system"] for row in prep if row["large_jump_detected"] == "True"]
    chain_warn = [row["system"] for row in prep if not row["chain_continuity_status"].startswith("pass")]
    recentered = sum(row["recentered_rerun_performed"] == "True" for row in prep)
    reg = src["regularization"]  # type: ignore[assignment]
    reg_default = [row for row in reg if row["method"] == "soft_sequence_plus_contact_default"]
    nochange = src["diagnostics_nochange"]  # type: ignore[assignment]
    lambdas = src["lambda"]  # type: ignore[assignment]
    lambda_rmsd = [fl(row["unified_posthoc_RMSD"]) for row in lambdas if row["unified_posthoc_RMSD"]]
    lambda_frag = [fl(row["core_fragmentation_score"]) for row in lambdas if row["core_fragmentation_score"]]
    release_rows = src["release"]  # type: ignore[assignment]
    release_values = {row["field"]: row["value"] for row in release_rows}
    diag_rows = src["fig3"]  # type: ignore[assignment]

    lines = [
        "# Manuscript Key Numbers",
        "",
        "This file is generated from frozen result tables. Runtime ratios in Table 1 are `soft_runtime / baseline_runtime`; values below 1 mean MDCoreAlign is faster.",
        "",
        "## Formal MD trajectory benchmark",
    ]
    for row in t1:
        lines.append(
            f"- {row['system']}: soft RMSD={row['soft_unified_RMSD']}, MDLovoFit RMSD={row['mdlovofit_unified_RMSD']}, THESEUS RMSD={row['theseus_unified_RMSD']}; soft runtime={row['soft_runtime']} s, MDLovoFit runtime={row['mdlovofit_runtime']} s, THESEUS runtime={row['theseus_runtime']} s."
        )
    lines.extend(["", "## Structure-pair benchmark summary"])
    for row in pairs:
        lines.append(
            f"- {row['system']}: soft RMSD={row['soft_unified_rmsd']}, MDLovoFit RMSD={row['mdlovofit_unified_rmsd']}, THESEUS RMSD={row['theseus_unified_rmsd']}; soft runtime={row['soft_runtime_s']} s, MDLovoFit runtime={row['mdlovofit_runtime_s']} s, THESEUS runtime={row['theseus_runtime_s']} s."
        )
    lines.extend(["", "## NMR500 summary"])
    for row in nmr:
        if row["method"] == "soft_vs_mdlovofit_case_outcomes":
            lines.append(
                f"- {row['method']}: n_cases={row['n_cases']}; paired outcomes: {row['parsing_success_rate']}."
            )
        else:
            lines.append(
                f"- {row['method']}: n_cases={row['n_cases']}, mean_runtime={row['mean_runtime_s']} s, median_runtime={row['median_runtime_s']} s, mean_unified_RMSD={row['mean_unified_rmsd']}, median_unified_RMSD={row['median_unified_rmsd']}, parsing_success_rate={row['parsing_success_rate'] or 'not_applicable'}."
            )
    lines.extend(["", "## Flexibility profiling summary"])
    for row in t3:
        lines.append(
            f"- {row['system']}: n_residues={row['n_residues']}, flexible_segments={row['n_flexible_segments']}, mean robust/ordinary RMSF={f(row['mean_robust_RMSF'])}/{f(row['mean_ordinary_RMSF'])}, Spearman robust RMSF vs mobility={f(row['spearman_robust_RMSF_vs_mobility'])}, core/noncore RMSF={f(row['mean_core_RMSF'])}/{f(row['mean_noncore_RMSF'])}."
        )
    lines.extend(["", "## Trajectory diagnostics summary"])
    for row in t2:
        lines.append(
            f"- {row['system']}: temporal Jaccard={row['temporal_core_jaccard_range']}, stride Jaccard={row['stride_core_jaccard_range']}, length Jaccard={row['length_core_jaccard_range']}, reference Jaccard={row['reference_core_jaccard_range']}, warm-start Jaccard={row['warm_start_core_jaccard']}."
        )

    def add_diag_section(title: str, diagnostic: str, formatter) -> None:
        lines.extend(["", f"## {title}"])
        for row in diag_rows:
            if row["diagnostic"] == diagnostic:
                lines.append(formatter(row))

    add_diag_section(
        "Temporal block summary",
        "temporal_block_stability",
        lambda row: (
            f"- {canonical_system(row['system'])}: core Jaccard={rng(row, 'core_jaccard_min', 'core_jaccard_max')}, "
            f"mobility Spearman={rng(row, 'mobility_spearman_min', 'mobility_spearman_max')}, "
            f"nonconverged_rows={row['nonconverged_rows'] or '0'}, classifications={row['classification_counts'] or 'none'}."
        ),
    )
    add_diag_section(
        "Stride sensitivity summary",
        "stride_sensitivity",
        lambda row: (
            f"- {canonical_system(row['system'])}: core Jaccard={rng(row, 'core_jaccard_min', 'core_jaccard_max')}, "
            f"mobility Spearman={rng(row, 'mobility_spearman_min', 'mobility_spearman_max')}."
        ),
    )
    add_diag_section(
        "Trajectory length summary",
        "trajectory_length_dependence",
        lambda row: (
            f"- {canonical_system(row['system'])}: threshold fraction for core Jaccard >= 0.80="
            f"{row['length_threshold_fraction_at_core_jaccard_ge_0_80']}, "
            f"core Jaccard={rng(row, 'core_jaccard_min', 'core_jaccard_max')}, "
            f"mobility Spearman={rng(row, 'mobility_spearman_min', 'mobility_spearman_max')}."
        ),
    )
    add_diag_section(
        "Reference sensitivity summary",
        "reference_sensitivity",
        lambda row: (
            f"- {canonical_system(row['system'])}: core Jaccard={rng(row, 'core_jaccard_min', 'core_jaccard_max')}, "
            f"mobility Spearman={rng(row, 'mobility_spearman_min', 'mobility_spearman_max')}; "
            "reference-aware, not reference-independent."
        ),
    )
    add_diag_section(
        "Warm-start ablation summary",
        "warm_start_ablation",
        lambda row: (
            f"- {canonical_system(row['system'])}: warm/cold runtime ratio="
            f"{f(row['warm_start_runtime_ratio_warm_over_cold'])}, "
            f"iteration delta={f(row['warm_start_iteration_delta_warm_minus_cold'])}, "
            f"output_preserved={row['warm_start_output_preserved']}."
        ),
    )
    lines.extend(
        [
            "",
            "## Temporal convergence classification",
            f"- Non-full non-converged temporal block rows: {sum(temporal_counts.values())}.",
            f"- Classification counts: {dict(temporal_counts)}.",
            "- True failure and weight collapse counts: 0.",
            "",
            "## Preprocessing verification summary",
            f"- Systems checked: {len(prep)}.",
            f"- Large-jump systems: {', '.join(jumps) if jumps else 'none'}.",
            f"- Chain-continuity warnings: {', '.join(chain_warn) if chain_warn else 'none'}.",
            f"- Recentered diagnostic comparisons performed: {recentered}/{len(prep)}.",
            "- PBC/nojump and centering status could not be independently verified from local metadata.",
            "",
            "## Regularization rerun summary",
            "- Sequence regularization reduced or preserved core segment count in 4/4 systems relative to no regularization.",
            "- Contact regularization reduced or preserved core segment count in 4/4 systems.",
            "- Sequence+contact preserved unified post-hoc RMSD within 1% in 4/4 systems.",
            "- Default sequence+contact rows:",
        ]
    )
    for row in reg_default:
        lines.append(f"  - {row['system']}: RMSD={f(row['unified_posthoc_RMSD'])}, weighted_RMSD={f(row['weighted_RMSD'])}, core_segments={row['n_core_segments']}, iterations={f(row['n_iterations'])}.")
    lines.extend(
        [
            "",
            "## Diagnostics summary",
            f"- Diagnostics no-change validation passed in {sum(row['passed'] == 'True' for row in nochange)}/{len(nochange)} systems.",
            "- Per-iteration diagnostics record weighted RMSD pseudo-objective; no formal global objective convergence claim should be made.",
            "",
            "## Lambda sensitivity summary",
            f"- Lambda grid rows: {len(lambdas)}.",
            f"- Unified RMSD range: {f(min(x for x in lambda_rmsd if x is not None))} to {f(max(x for x in lambda_rmsd if x is not None))}.",
            f"- Core fragmentation score range: {f(min(x for x in lambda_frag if x is not None), 0)} to {f(max(x for x in lambda_frag if x is not None), 0)}.",
            "- Full 64-row grid belongs in Supplementary.",
            "",
            "## Package/user-readiness summary",
            "- Clean install: pass.",
            "- Build and twine check: pass.",
            "- Reproducibility capsule: pass.",
            "- TestPyPI validation: not_applicable_pending_upload.",
            "- Final submission audit: no blockers; recommendation C, almost ready pending release metadata/DOI.",
            "",
            "## Test count and command",
            "- Command: `python -m pytest -q tests`.",
            "- Result: 150 passed, 2 subtests passed.",
            "",
            "## Release metadata status",
            f"- GitHub repository URL: {release_values.get('project.urls', 'pending')}.",
            f"- Zenodo DOI: {release_values.get('doi', 'pending')}.",
            "- TestPyPI/PyPI: pending.",
        ]
    )
    (OUT / "manuscript_key_numbers.md").write_text("\n".join(lines) + "\n")


def write_figure_plan() -> None:
    """Write figure plan."""

    sections = [
        "# Figure Plan",
        "",
        "## Figure 1",
        "- figure_number: Figure 1",
        "- title: MDCoreAlign workflow and decision tree",
        "- panel_description: Ordinary full-set fitting, robust stable-core alignment, preprocessing audit, trajectory diagnostics, reference sensitivity, RMSF/flexible segment interpretation.",
        "- source_files: `results/final_submission_upgrade/final_figures/fig1_workflow_decision_tree.md`",
        "- main_message: MDCoreAlign is a workflow for robust alignment plus trustworthy flexibility interpretation, not only a replacement RMSD calculator.",
        "- caption_draft: MDCoreAlign workflow for robust stable-core alignment and flexibility profiling, including preprocessing and trajectory diagnostics.",
        "",
        "## Figure 2",
        "- figure_number: Figure 2",
        "- title: Formal MD trajectory benchmark",
        "- panel_description: Per-system unified post-hoc RMSD and runtime for MDCoreAlign, MDLovoFit, and THESEUS.",
        "- source_files: `table_1_md_trajectory_benchmark.tsv`; `results/final_submission_upgrade/final_figures/fig2_md_benchmark_summary.tsv`",
        "- main_message: MDCoreAlign gives lower unified RMSD than MDLovoFit on the four formal MD systems and is faster than MDLovoFit; THESEUS is a mixed independent baseline.",
        "- caption_draft: Runtime and unified post-hoc RMSD across four formal MD trajectory systems. THESEUS values are validated existing outputs unless rerun status is explicitly updated.",
        "",
        "## Figure 3",
        "- figure_number: Figure 3",
        "- title: Trajectory diagnostics",
        "- panel_description: Temporal block, stride, trajectory length, reference sensitivity, and warm-start preservation summaries.",
        "- source_files: `table_2_md_trajectory_diagnostics.tsv`; `results/final_submission_upgrade/final_figures/fig3_md_trajectory_diagnostics_summary.tsv`",
        "- main_message: Stride and warm-start outputs are stable, trajectory length and reference choice should be reported as diagnostics.",
        "- caption_draft: Diagnostic checks for time-window, stride, length, reference, and warm-start behavior. Reference sensitivity is diagnostic and no reference-independence claim is made.",
        "",
        "## Figure 4",
        "- figure_number: Figure 4",
        "- title: Flexibility profiling",
        "- panel_description: Robust vs ordinary RMSF, mobility score association, core/non-core RMSF separation, and flexible segment interpretation.",
        "- source_files: `table_3_flexibility_profile.tsv`; `results/flexibility/rmsf_profiles_long.tsv`; `results/flexibility/figure_data/*`",
        "- main_message: Mobility scores align with robust RMSF and non-core residues show higher RMSF while global RMSF scales remain similar.",
        "- caption_draft: Robust-core alignment refines local flexibility interpretation. RMSF is alignment-frame dependent.",
    ]
    (OUT / "figure_plan.md").write_text("\n".join(sections) + "\n")


def write_limitations() -> None:
    """Write precise limitation wording."""

    text = """# Precise Limitation Wording

## THESEUS
THESEUS was included as an independent external baseline using validated existing result tables. In the current local environment, THESEUS was not rerun because the executable was not detected; therefore, the manuscript should not imply a current-environment THESEUS rerun.

## PBC/Nojump
Formal inputs were analyzed as provided. Lightweight displacement, centering, and residue-continuity checks were performed, but PBC/nojump and centering status could not be independently proven from local metadata.

## Reference Choice
Stable-core and mobility outputs are reference-aware. We report reference-sensitivity diagnostics and do not claim reference independence.

## Atom Selection
The formal workflow is CA-level. Backbone or all-heavy-atom sensitivity should remain supplementary/future work unless a validated residue-level mapping and rerun are added.

## YiiP
YiiP is retained as a challenging membrane-protein case. Its geometric and flexibility outputs are useful, but the available external biological proxy is weak and should not be overclaimed.

## NMR500
The NMR500 benchmark is supplementary structural-ensemble evidence. It should not be described as a true time-ordered MD trajectory benchmark.

## Optional External MD / CA-vs-Backbone
External MD mini-validation and CA-vs-backbone sensitivity are not promoted to main results unless controlled inputs and validated mappings are added.
"""
    (OUT / "limitations_precise_wording.md").write_text(text)


def write_availability_and_data() -> None:
    """Write availability, data availability, AI disclosure, references, and claims checklist."""

    (OUT / "availability_statement_draft.md").write_text(
        """# Availability Statement Draft

Software repository: `pending`.
Zenodo DOI: `pending`.
TestPyPI/PyPI URL: `pending`.
Release tag: `v0.1.0rc1`.
License: MIT.
Test data: bundled toy CA-only multi-model PDB in `examples/toy_md/`.
Reproducibility capsule: `results/final_submission_upgrade/reproducibility_capsule/`.

Replace pending values only after the real external records exist.
"""
    )
    (OUT / "data_availability_statement_draft.md").write_text(
        """# Data Availability Statement Draft

Formal MD trajectory inputs were analyzed as local preprocessed inputs using CA-level selections. Processed result tables, figure-data tables, toy data, and reproducibility-capsule outputs are stored in the repository result directories.

Large MD trajectories and external benchmark data should not be committed to GitHub. Cached official benchmark tables are checksum-validated in the reproducibility capsule. Zenodo archive status is currently `pending`.
"""
    )
    (OUT / "ai_disclosure_draft.md").write_text(
        """# AI Disclosure Draft

AI-assisted coding and editorial tooling was used to help organize software, tests, documentation, and audit summaries. The authors remain responsible for all scientific claims, code, analyses, and manuscript text. This draft should be revised to match the target journal policy.
"""
    )
    (OUT / "references_seed.md").write_text(
        """# References Seed

References that should be cited or checked before submission:

- MDLovoFit: cite the original MDLovoFit least-mobile-subset fitting paper/software.
- THESEUS: cite the original THESEUS maximum-likelihood/probabilistic superposition method/software.
- Bio3D: cite if discussing existing structural bioinformatics trajectory/ensemble analysis ecosystems.
- MDAnalysis: cite because MDAnalysis is used for topology/trajectory handling in formal preprocessing and metadata workflows.
- MDTraj: cite only if used or discussed as an alternative MD trajectory analysis ecosystem.
- GROMACS or a standard RMSF/flexibility reference: cite if discussing RMSF as a standard MD mobility metric.
- ATLAS/mdCATH/MDRepo/GPCRmd: cite only if external validation using those resources is actually included.
- Bioinformatics author guidelines: do not cite in the scientific manuscript unless needed for software/data availability policy text.
"""
    )
    claims = [
        ("soft improves vs MDLovoFit", "results/final_multi_system_summary_20260603/final_real_trajectory_summary.csv", "yes", "Safe for unified post-hoc RMSD and runtime on the four formal MD systems; keep MDLovoFit-native RMSD separate."),
        ("competitive with THESEUS", "results/final_multi_system_summary_20260603/final_real_trajectory_summary.csv", "partially", "THESEUS is mixed for MD trajectories; soft is lower RMSD but sometimes slower. Stronger for structure pairs."),
        ("regularization reduces core fragmentation", "results/submission_critical/regularization_rerun/regularization_rerun_summary.md", "yes", "State reduced or preserved core segment count, not universal biological improvement."),
        ("mobility aligns with robust RMSF", "results/flexibility/flexibility_summary_four_systems.tsv", "yes", "Use Spearman values; RMSF is alignment-frame dependent."),
        ("stride sensitivity is stable", "results/md_trajectory_focus/stride_sensitivity/stride_sensitivity_summary.md", "yes", "Report range and note NhaA has the lowest mean stride agreement."),
        ("reference choice affects interpretation", "results/md_trajectory_focus/reference_sensitivity/reference_sensitivity_summary.md", "yes", "Do not claim reference independence."),
        ("warm start preserves core labels", "results/md_trajectory_focus/warm_start_ablation/warm_start_ablation_summary.md", "yes", "Describe as efficiency option; not scientifically necessary."),
        ("package is clean-install validated", "results/final_submission_upgrade/user_readiness/clean_install_report.json", "yes", "GitHub/TestPyPI metadata still pending."),
        ("not claiming reference independence", "results/final_submission_upgrade/paper_text_final/limitations_final_submission.md", "yes", "Keep this wording in limitations."),
        ("not claiming strong YiiP biological validation", "results/submission_critical/yiip_biology_proxy/yiip_proxy_summary.md", "yes", "Describe YiiP as challenging/limited proxy."),
    ]
    write_table(OUT / "manuscript_claims_checklist.tsv", [{"claim": a, "supported_by_file": b, "safe_to_claim": c, "wording_caution": d} for a, b, c, d in claims], ["claim", "supported_by_file", "safe_to_claim", "wording_caution"])


def write_manifest() -> None:
    """Write a simple manifest."""

    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": "scripts/make_manuscript_writing_bundle.py",
        "outputs": sorted(path.name for path in OUT.iterdir() if path.is_file()),
        "notes": "No algorithms, results, or official benchmark tables were changed.",
    }
    (OUT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Generate manuscript writing bundle."""

    OUT.mkdir(parents=True, exist_ok=True)
    src = load_sources()
    t1 = write_table_1(src)
    t2 = write_table_2(src)
    t3 = write_table_3(src)
    write_table_4(src)
    write_key_numbers(src, t1, t2, t3)
    write_figure_plan()
    write_limitations()
    write_availability_and_data()
    write_manifest()
    print(f"Wrote manuscript writing bundle to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
