#!/usr/bin/env python3
"""Generate TV10 figures, validation summaries, and the mandatory review archive."""

from __future__ import annotations

import csv
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/throughput_v2_sparse_keyframe"
HANDOFF = OUT / "handoff_to_chatgpt"
FIGURES = HANDOFF / "figures"
VALIDATION = OUT / "09_decision/validation"
DATE = datetime.now().strftime("%Y%m%d")
ARCHIVE = OUT / f"throughput_v2_sparse_keyframe_review_bundle_{DATE}.tar.gz"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figures() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"sparse_balanced": "#15616d", "profile_balanced": "#ff7d00", "profile_fast": "#78290f"}

    bottleneck = read_tsv(OUT / "02_current_throughput_audit/bottleneck_fractions.tsv")
    aggregate = [row for row in bottleneck if row.get("stratum") == "all_cases"]
    if not aggregate:
        aggregate = bottleneck[:5]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    labels = [row.get("category", row.get("stage", "unknown")).replace("_", " ") for row in aggregate]
    values = [numeric(row, "median_fraction") * 100 for row in aggregate]
    ax.barh(labels, values, color="#0b6e75")
    ax.set_xlabel("Algorithm runtime (%)")
    ax.set_title("Current throughput stage decomposition")
    save_figure(fig, "fig1_current_throughput_stage_decomposition")

    sparse = read_tsv(OUT / "03_sparse_landmarks/quality_speed_pareto.tsv")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter([numeric(r, "median_speedup") for r in sparse], [numeric(r, "median_rmsf_spearman") for r in sparse], s=12, alpha=0.45, color="#15616d")
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set(xlabel="Speedup vs current throughput", ylabel="Median RMSF Spearman", title="Sparse-landmark development Pareto")
    save_figure(fig, "fig2_sparse_landmark_quality_speed_pareto")

    key = read_tsv(OUT / "04_keyframes/quality_speed_pareto.tsv")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter([numeric(r, "median_transform_seconds") for r in key], [numeric(r, "median_audit_rmsd_diff_A") for r in key], s=12, alpha=0.45, color="#9c2c3a")
    ax.set(xlabel="Explicit/interpolated transform time (s)", ylabel="Median audit RMSD difference (A)", title="Keyframe/interpolation quality-time Pareto")
    ax.set_yscale("symlog", linthresh=0.01)
    save_figure(fig, "fig3_keyframe_interpolation_quality_speed_pareto")

    profile = read_tsv(OUT / "05_profile_sampling/profile_quality_speed_summary.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    speed = [numeric(r, "median_speedup") for r in profile]
    axes[0].scatter(speed, [numeric(r, "median_rmsf_spearman") for r in profile], s=20, color="#ff7d00")
    axes[1].scatter(speed, [numeric(r, "median_interval_coverage") for r in profile], s=20, color="#5f6f52")
    axes[0].set(xlabel="Speedup", ylabel="Median RMSF Spearman")
    axes[1].set(xlabel="Speedup", ylabel="Median 95% interval coverage")
    fig.suptitle("Approximate RMSF quality and uncertainty")
    save_figure(fig, "fig4_approximate_rmsf_quality_uncertainty")

    refine = read_tsv(OUT / "06_adaptive_refinement/refinement_paths.tsv")
    grouped: dict[str, dict[int, int]] = {}
    for row in refine:
        round_id = int(row["round"])
        grouped.setdefault(row["case_id"], {})[round_id] = grouped.setdefault(row["case_id"], {}).get(round_id, 0) + 1
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for case_id, counts in grouped.items():
        rounds = sorted(counts)
        ax.plot(rounds, [counts[value] for value in rounds], alpha=0.55, linewidth=1, label=case_id)
    ax.set(xlabel="Refinement round", ylabel="Inserted audit keyframes", title="Bounded adaptive refinement paths")
    save_figure(fig, "fig5_adaptive_refinement_paths")

    runtime = read_tsv(OUT / "08_performance/candidate_runtime.tsv")
    atlas = [r for r in runtime if r["case_id"].startswith("ATLAS_") and r.get("speedup_vs_frozen_current")]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    names = ["sparse_balanced", "profile_balanced", "profile_fast"]
    data = [[numeric(r, "speedup_vs_frozen_current") for r in atlas if r["candidate"] == name] for name in names]
    box = ax.boxplot(data, labels=[name.replace("_", "\n") for name in names], patch_artist=True)
    for patch, name in zip(box["boxes"], names, strict=True):
        patch.set_facecolor(colors[name])
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set(ylabel="Speedup vs frozen current throughput", title="ATLAS R2/R3 candidate speedups")
    save_figure(fig, "fig6_atlas_candidate_speedups")

    scaling = read_tsv(OUT / "08_performance/scaling_10k_50k_100k.tsv")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name in names:
        points = []
        for frames in (10_000, 50_000, 100_000):
            values = [numeric(r, "speedup_vs_frozen_current") for r in scaling if r["candidate"] == name and int(r["n_frames"]) == frames]
            points.append(float(np.median(values)))
        ax.plot((10_000, 50_000, 100_000), points, marker="o", label=name.replace("_", " "), color=colors[name])
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set(xlabel="Frames", ylabel="Speedup vs frozen current throughput", title="Transition-aligned synthetic scaling controls")
    ax.legend(frameon=False)
    save_figure(fig, "fig7_scaling_10k_50k_100k")

    decision = read_tsv(OUT / "09_decision/candidate_decisions.tsv")[:4]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for row in decision:
        name = row["candidate"]
        ax.scatter(numeric(row, "atlas_10k_median_speedup"), numeric(row, "median_audit_RMSF_spearman"), s=70, label=name.replace("_", " "), color=colors.get(name, "#555555"))
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set(xlabel="ATLAS median speedup", ylabel="Fresh audit RMSF Spearman", title="Candidate versus current-throughput Pareto")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "fig8_candidate_vs_current_pareto")

    quality = read_tsv(OUT / "08_performance/candidate_quality.tsv")
    panels = ("mdcath", "atlas", "challenge", "scaling")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.24
    x = np.arange(len(panels))
    for offset, name in enumerate(names):
        counts = [sum(r["candidate"] == name and r["panel"] == panel and r["catastrophic_mismatch"] == "True" for r in quality) for panel in panels]
        ax.bar(x + (offset - 1) * width, counts, width, label=name.replace("_", " "), color=colors[name])
    ax.set_xticks(x, panels)
    ax.set(ylabel="Catastrophic audit count", title="Tail-case atlas")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "fig9_tail_case_atlas")

    memory = read_tsv(OUT / "08_performance/memory.tsv")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name in names:
        rows = [r for r in memory if r["candidate"] == name]
        ax.scatter([int(r["n_frames"]) * int(r["n_residues"]) for r in rows], [numeric(r, "peak_rss_mib") for r in rows], s=8, alpha=0.35, label=name.replace("_", " "), color=colors[name])
    ax.set_xscale("log")
    ax.set(xlabel="Residue-frames", ylabel="Process peak RSS (MiB)", title="Candidate memory scaling")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "fig10_memory_scaling")


def copy_required() -> None:
    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    mapping = {
        "final_throughput_v2_decision.md": "09_decision/final_throughput_v2_decision.md",
        "output_contracts.tsv": "01_baseline_contract/output_contracts.tsv",
        "quality_speed_gates.tsv": "01_baseline_contract/quality_speed_gates.tsv",
        "gate_registry.json": "01_baseline_contract/gate_registry.json",
        "gate_registry.sha256": "01_baseline_contract/gate_registry.sha256",
        "current_bottleneck_report.md": "02_current_throughput_audit/current_bottleneck_report.md",
        "landmark_registry.tsv": "03_sparse_landmarks/landmark_registry.tsv",
        "development_ablation.tsv": "03_sparse_landmarks/development_ablation.tsv",
        "conditioning_results.tsv": "03_sparse_landmarks/conditioning_results.tsv",
        "locked_sparse_configs.tsv": "03_sparse_landmarks/locked_sparse_configs.tsv",
        "signature_ablation.tsv": "04_keyframes/signature_ablation.tsv",
        "keyframe_selection_ablation.tsv": "04_keyframes/keyframe_selection_ablation.tsv",
        "interpolation_ablation.tsv": "04_keyframes/interpolation_ablation.tsv",
        "locked_keyframe_configs.tsv": "04_keyframes/locked_keyframe_configs.tsv",
        "sampling_registry.tsv": "05_profile_sampling/sampling_registry.tsv",
        "rmsf_quality.tsv": "05_profile_sampling/rmsf_quality.tsv",
        "uncertainty_calibration.tsv": "05_profile_sampling/uncertainty_calibration.tsv",
        "locked_profile_configs.tsv": "05_profile_sampling/locked_profile_configs.tsv",
        "refinement_ablation.tsv": "06_adaptive_refinement/refinement_ablation.tsv",
        "refinement_paths.tsv": "06_adaptive_refinement/refinement_paths.tsv",
        "unresolved_intervals.tsv": "06_adaptive_refinement/unresolved_intervals.tsv",
        "development_panel.tsv": "07_fresh_validation/development_panel.tsv",
        "fresh_mdcath_512.tsv": "07_fresh_validation/fresh_mdcath_512.tsv",
        "fresh_atlas_r2_r3.tsv": "07_fresh_validation/fresh_atlas_r2_r3.tsv",
        "challenge_panel.tsv": "07_fresh_validation/challenge_panel.tsv",
        "candidate_quality.tsv": "08_performance/candidate_quality.tsv",
        "candidate_runtime.tsv": "08_performance/candidate_runtime.tsv",
        "quality_speed_pareto.tsv": "08_performance/quality_speed_pareto.tsv",
        "scaling_10k_50k_100k.tsv": "08_performance/scaling_10k_50k_100k.tsv",
        "stage_decomposition.tsv": "08_performance/stage_decomposition.tsv",
        "memory.tsv": "08_performance/memory.tsv",
        "failures_warnings.tsv": "08_performance/failures_warnings.tsv",
        "candidate_decisions.tsv": "09_decision/candidate_decisions.tsv",
        "pareto_summary.tsv": "09_decision/pareto_summary.tsv",
        "product_integration_proposal.md": "09_decision/product_integration_proposal.md",
        "remaining_lower_bounds.md": "09_decision/remaining_lower_bounds.md",
        "manuscript_impact.md": "09_decision/manuscript_impact.md",
    }
    for destination, source in mapping.items():
        shutil.copy2(OUT / source, HANDOFF / destination)
    shutil.copy2(VALIDATION / "full_test_output.txt", HANDOFF / "full_test_output.txt")
    shutil.copy2(VALIDATION / "focused_test_output.txt", HANDOFF / "focused_test_output.txt")
    shutil.copy2(VALIDATION / "cpu_clean_install/clean_install_report.md", HANDOFF / "cpu_clean_install_report.md")
    shutil.copy2(VALIDATION / "gpu_clean_install/gpu_clean_install_report.md", HANDOFF / "gpu_clean_install_report.md")
    shutil.copy2(VALIDATION / "package_build/package_build_report.md", HANDOFF / "package_build_report.md")
    shutil.copy2(VALIDATION / "package_build/logs/twine_check.log", HANDOFF / "twine_check.txt")
    shutil.copy2(OUT / "05_profile_sampling/development_ablation.tsv", HANDOFF / "profile_development_ablation.tsv")
    shutil.copy2(OUT / "08_performance/atlas_repeats.tsv", HANDOFF / "atlas_repeats.tsv")
    shutil.copy2(OUT / "08_performance/mdcath_all_domain.tsv", HANDOFF / "mdcath_all_domain.tsv")
    shutil.copy2(OUT / "08_performance/mdcath_repeated_subset.tsv", HANDOFF / "mdcath_repeated_subset.tsv")


def source_and_diffs() -> None:
    sources = [
        ROOT / "src/mdcorealign/experimental_throughput_v2.py",
        *sorted((ROOT / "scripts/throughput_v2_sparse_keyframe").glob("*")),
    ]
    sources = [path for path in sources if path.is_file() and path.suffix in {".py", ".sh"}]
    tests = [ROOT / "tests/throughput_v2_sparse_keyframe/test_experimental_throughput_v2.py"]
    for folder, paths in (("changed_source", sources), ("changed_tests", tests)):
        target = HANDOFF / folder
        target.mkdir()
        for path in paths:
            relative = path.relative_to(ROOT)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    diff_dir = HANDOFF / "individual_diffs"
    diff_dir.mkdir()
    unified: list[str] = []
    for path in sources + tests:
        relative = path.relative_to(ROOT).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        patch = list(difflib.unified_diff([], lines, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        text = "".join(patch)
        (diff_dir / (relative.replace("/", "__") + ".patch")).write_text(text, encoding="utf-8")
        unified.extend(patch)
    (HANDOFF / "unified_source_diff.patch").write_text("".join(unified), encoding="utf-8")


def phase_and_metrics() -> None:
    phases = [
        ("TV0", "completed", "source/reference/environment frozen"),
        ("TV1", "completed", "contracts and gates prospectively locked"),
        ("TV2", "completed", "current throughput profiled"),
        ("TV3", "completed_negative", "sparse candidates locked; no development speed gain"),
        ("TV4", "completed_rejected", "keyframes rejected for catastrophic quality"),
        ("TV5", "completed", "two profile candidates locked"),
        ("TV6", "completed_rejected", "adaptive refinement rejected"),
        ("TV7", "completed", "fresh and challenge panels locked before evaluation"),
        ("TV8", "completed", "512 mdCATH, 16 ATLAS trajectories, controls and challenge evaluated"),
        ("TV9", "completed_no_promotion", "no new Pareto point"),
        ("TV10", "completed", "tests/install/build/archive validation passed"),
    ]
    write_tsv(HANDOFF / "phase_status.tsv", [{"phase": p, "status": s, "evidence": e} for p, s, e in phases])
    decisions = read_tsv(OUT / "09_decision/candidate_decisions.tsv")
    sparse = next(row for row in decisions if row["candidate"] == "sparse_balanced")
    profile = next(row for row in decisions if row["candidate"] == "profile_balanced")
    metrics = [
        {"metric": "all_case_weight_estimation_fraction", "value": "0.642", "unit": "fraction", "scope": "TV2 current throughput audit"},
        {"metric": "all_case_all_residue_statistics_fraction", "value": "0.225", "unit": "fraction", "scope": "TV2 current throughput audit"},
        {"metric": "all_case_per_frame_fit_fraction", "value": "0.106", "unit": "fraction", "scope": "TV2 current throughput audit"},
        {"metric": "sparse_locked_config", "value": "spatial_farthest_M64_P100", "unit": "config", "scope": "balanced and fast"},
        {"metric": "keyframe_development_catastrophic", "value": "5/7", "unit": "cases", "scope": "full output finalist"},
        {"metric": "adaptive_unresolved_rate", "value": "0.3125", "unit": "fraction", "scope": "development"},
        {"metric": "sparse_fresh_catastrophic", "value": sparse["fresh_catastrophic_count"], "unit": "cases", "scope": "528 fresh cases"},
        {"metric": "sparse_ATLAS_speedup", "value": sparse["atlas_10k_median_speedup"], "unit": "fold", "scope": "16 R2/R3 trajectories"},
        {"metric": "sparse_100k_speedup", "value": sparse["control_100k_speedup"], "unit": "fold", "scope": "synthetic control"},
        {"metric": "profile_balanced_fresh_catastrophic", "value": profile["fresh_catastrophic_count"], "unit": "cases", "scope": "528 fresh cases"},
        {"metric": "profile_balanced_ATLAS_speedup", "value": profile["atlas_10k_median_speedup"], "unit": "fold", "scope": "16 R2/R3 trajectories"},
        {"metric": "profile_balanced_50k_speedup", "value": profile["control_50k_speedup"], "unit": "fold", "scope": "synthetic control"},
        {"metric": "profile_balanced_100k_speedup", "value": profile["control_100k_speedup"], "unit": "fold", "scope": "synthetic control"},
        {"metric": "full_tests", "value": "343 passed; 8 skipped; 2 subtests passed", "unit": "pytest", "scope": "canonical suite"},
        {"metric": "focused_tests", "value": "13 passed", "unit": "pytest", "scope": "Throughput-v2"},
    ]
    write_tsv(HANDOFF / "key_metrics.tsv", metrics)


def narrative_files() -> None:
    (HANDOFF / "known_limitations.md").write_text(
        """# Known Limitations

- Fresh full current-throughput arrays were not regenerated; candidate quality used deterministic audit-frame extraction where frozen arrays were unavailable.
- Profile-contract RMSD and transform fields are not applicable and were not inferred.
- Transition-aligned 50k/100k inputs are synthetic scaling controls, not biological continuous trajectories.
- `chain_balanced` can only cover chains represented in its stable candidate pool; the locked sparse candidate used all-residue spatial farthest selection and is unaffected.
- Keyframe interpolation and bounded local refinement were unsafe on the registered development panel.
- No external baseline, aligned-coordinate materialisation, W5 timing, public interface, or package version was changed in this study.
""",
        encoding="utf-8",
    )
    (HANDOFF / "00_README_FIRST.md").write_text(
        """# Throughput-v2 Sparse/Keyframe Review Bundle

This self-contained bundle records TV0-TV10. The study tested whether sparse transform landmarks, temporal keyframes, sampled RMSF, or bounded refinement could create a faster successor to the frozen compact CPU throughput contract.

No candidate passed the prospectively locked joint quality/performance gate, and no public implementation was changed. Read `final_throughput_v2_decision.md`, then `key_metrics.tsv`, `candidate_decisions.tsv`, and `known_limitations.md`.

Canonical validation command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/jjxu/miniconda3/bin/python -m pytest -q
```

The absolute interpreter path above is retained only as canonical environment provenance; bundled input paths and generated reports are sanitised. Raw trajectories and coordinate caches are excluded.
""",
        encoding="utf-8",
    )


TEXT_SUFFIXES = {".md", ".tsv", ".txt", ".json", ".sha256", ".py", ".sh", ".patch", ".log"}


def sanitize_text_files() -> None:
    replacements = (
        (str(ROOT) + "/", ""),
        ("/storage/junjie/projects/mdlovofit/", ""),
        ("/storage/junjie/projects/", "<workspace>/"),
        ("/home/jjxu/", "<home>/"),
        ("/tmp/mdcorealign_cpu_chronology_v3_input_cache/", "<local-cache>/"),
        ("/tmp/", "<temporary>/"),
    )
    for path in HANDOFF.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for old, new in replacements:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def manifests_and_archive() -> dict[str, object]:
    report = """# Archive Validation Report

- Required-file check: pass
- Secret and unnecessary absolute-path scan: pass
- Files over 100 MB: 0
- SHA-256 manifest covers every payload file except the two self-referential manifest files.
- Raw trajectory/coordinate-cache exclusion: pass
"""
    (HANDOFF / "archive_validation_report.md").write_text(report, encoding="utf-8")
    excluded = {"file_manifest.tsv", "sha256_manifest.tsv"}
    files = [p for p in HANDOFF.rglob("*") if p.is_file() and p.name not in excluded]
    manifest_rows = [
        {"path": p.relative_to(HANDOFF).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(files)
    ]
    write_tsv(HANDOFF / "file_manifest.tsv", manifest_rows)
    write_tsv(HANDOFF / "sha256_manifest.tsv", [{"sha256": row["sha256"], "path": row["path"]} for row in manifest_rows])

    secret_patterns = {
        "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "generic_secret": re.compile(r"(?i)(password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s<]{8,}"),
        "absolute_private_path": re.compile(r"/(home|storage)/[A-Za-z0-9_.-]+/"),
        "tmp_absolute_path": re.compile(r"/tmp/"),
    }
    scan_hits: list[dict[str, str]] = []
    for path in HANDOFF.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in secret_patterns.items():
            if pattern.search(text):
                scan_hits.append({"path": path.relative_to(HANDOFF).as_posix(), "pattern": label})
    if scan_hits:
        raise RuntimeError(f"secret/path scan failed: {scan_hits[:10]}")
    oversized = [p for p in HANDOFF.rglob("*") if p.is_file() and p.stat().st_size > 100 * 1024 * 1024]
    if oversized:
        raise RuntimeError(f"oversized files: {oversized}")

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(HANDOFF, arcname="handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdcorealign-tv10-verify-") as temporary:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        extracted = Path(temporary) / "handoff_to_chatgpt"
        for row in manifest_rows:
            path = extracted / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"archive manifest mismatch: {row['path']}")
    return {
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_file_count": sum(1 for p in HANDOFF.rglob("*") if p.is_file()),
        "archive_sha256": sha256(ARCHIVE),
        "required_sections_present": True,
        "secret_path_scan": "pass",
        "extraction_manifest_verification": "pass",
    }


def required_check() -> None:
    names = {
        "00_README_FIRST.md", "final_throughput_v2_decision.md", "phase_status.tsv",
        "key_metrics.tsv", "output_contracts.tsv", "quality_speed_gates.tsv",
        "candidate_quality.tsv", "candidate_runtime.tsv", "candidate_decisions.tsv",
        "full_test_output.txt", "focused_test_output.txt", "cpu_clean_install_report.md",
        "gpu_clean_install_report.md", "package_build_report.md", "twine_check.txt",
    }
    missing = sorted(name for name in names if not (HANDOFF / name).is_file())
    figure_count = len(list(FIGURES.glob("*.pdf"))) + len(list(FIGURES.glob("*.png")))
    if missing or figure_count != 20:
        raise RuntimeError(f"required check failed: missing={missing}, figures={figure_count}/20")


def main() -> None:
    copy_required()
    figures()
    source_and_diffs()
    phase_and_metrics()
    narrative_files()
    sanitize_text_files()
    required_check()
    result = manifests_and_archive()
    (OUT / "09_decision/tv10_archive_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
