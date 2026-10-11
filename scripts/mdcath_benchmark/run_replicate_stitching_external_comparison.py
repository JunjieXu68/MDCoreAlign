#!/usr/bin/env python
"""Add MDLovoFit and THESEUS comparisons to replicate-stitching diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdsuperpose.flexibility import robust_aligned_coords_from_result
from mdsuperpose.mdlovofit import (
    parse_mdlovofit_stdout,
    read_multimodel_pdb_coords,
    read_multimodel_pdb_with_occupancy,
    run_mdlovofit_on_files,
    write_multimodel_pdb,
)
from mdsuperpose.metrics import mean_rmsd
from mdsuperpose.theseus import parse_theseus_stdout, parse_theseus_variances
from mdsuperpose.trajectory_mode import solve_trajectory_robust_alignment
from scripts.mdcath_benchmark.mdcath_common import write_json, write_tsv


DOMAINS = ("2rdeA02", "1b4aA01", "1vq8L01")
PREFIXES = ("R1", "R1_R2", "R1_R2_R3", "R1_R2_R3_R4", "R1_R2_R3_R4_R5")
MDLOVOFIT_EXE = Path("/usr/local/bin/mdlovofit")
THESEUS_EXE = Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus")

INPUT_COLUMNS = [
    "domain_id",
    "prefix",
    "included_replicates",
    "n_frames",
    "n_residues",
    "input_path",
    "input_format",
    "sha256",
    "residue_mapping_ok",
    "notes",
]

TOOL_STATUS_COLUMNS = [
    "tool",
    "executable_path",
    "executable_exists",
    "help_command_works",
    "version_or_header",
    "final_status",
    "notes",
]

EXTERNAL_RUN_COLUMNS = [
    "domain_id",
    "prefix",
    "included_replicates",
    "n_replicates",
    "n_frames",
    "n_residues",
    "status",
    "runtime_sec",
    "command",
    "stdout_path",
    "stderr_path",
    "output_path",
    "failure_reason",
    "notes",
]

WARM_REPEATS_COLUMNS = [
    "domain_id",
    "prefix",
    "method",
    "repeat_id",
    "n_replicates",
    "n_frames",
    "n_residues",
    "runtime_scope",
    "runtime_sec",
    "status",
    "notes",
]

WARM_SUMMARY_COLUMNS = [
    "domain_id",
    "prefix",
    "method",
    "n_repeats",
    "median_runtime_sec",
    "iqr_runtime_sec",
    "n_frames",
    "n_residues",
    "runtime_per_frame",
    "runtime_relative_to_R1",
    "frame_count_relative_to_R1",
    "observed_over_expected_linear",
    "notes",
]

RMSD_COLUMNS = [
    "domain_id",
    "prefix",
    "included_replicates",
    "n_replicates",
    "n_frames",
    "n_residues",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "mdcorealign_lower_RMSD_than_mdlovofit",
    "mdcorealign_lower_RMSD_than_theseus",
    "mdlovofit_status",
    "theseus_status",
    "notes",
]

SCALING_MODEL_COLUMNS = [
    "domain_id",
    "method",
    "n_prefixes",
    "R2_linear",
    "R2_quadratic",
    "AIC_linear",
    "AIC_quadratic",
    "estimated_scaling_preference",
    "runtime_per_frame_stability",
    "notes",
]

TOOL_COMPARISON_COLUMNS = [
    "domain_id",
    "prefix",
    "included_replicates",
    "n_replicates",
    "n_frames",
    "n_residues",
    "mdcorealign_median_runtime_sec",
    "mdlovofit_median_runtime_sec",
    "theseus_median_runtime_sec",
    "mdcorealign_vs_mdlovofit_speedup",
    "mdcorealign_vs_theseus_speedup",
    "mdcorealign_faster_than_mdlovofit",
    "mdcorealign_faster_than_theseus",
    "mdcorealign_unified_RMSD",
    "mdlovofit_unified_RMSD",
    "theseus_unified_RMSD",
    "mdcorealign_lower_RMSD_than_mdlovofit",
    "mdcorealign_lower_RMSD_than_theseus",
    "boundary_to_within_jump_ratio",
    "notes",
]

MDLOVO_CORE_COLUMNS = [
    "domain_id",
    "prefix",
    "n_frames",
    "n_residues",
    "mdlovofit_core_fraction",
    "mdlovofit_core_residues",
    "mdlovofit_mobile_residues",
    "extraction_status",
    "failure_reason",
]

CORE_AGREEMENT_COLUMNS = [
    "domain_id",
    "prefix",
    "core_jaccard",
    "mobile_jaccard",
    "mdcorealign_core_fraction",
    "mdlovofit_core_fraction",
    "core_fraction_delta",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/mdcath_benchmark/replicate_stitching"),
    )
    parser.add_argument("--mdlovofit-exe", type=Path, default=MDLOVOFIT_EXE)
    parser.add_argument("--theseus-exe", type=Path, default=THESEUS_EXE)
    parser.add_argument(
        "--domain-ids",
        default=None,
        help="Optional comma-separated domain IDs. Defaults to the original 3-domain diagnostic panel.",
    )
    parser.add_argument(
        "--prefixes",
        default=None,
        help="Optional comma-separated stitched prefixes. Defaults to R1 through R1_R2_R3_R4_R5.",
    )
    parser.add_argument(
        "--docs-prefix",
        default="replicate_stitching",
        help="Prefix for docs written by this wrapper.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def split_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated CLI value with a tuple default."""

    if value is None:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV table if present."""

    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fnum(value: object) -> float | None:
    """Parse finite float."""

    try:
        if value in ("NA", "", None, "not_run"):
            return None
        converted = float(value)
    except Exception:
        return None
    return converted if math.isfinite(converted) else None


def sha256_file(path: Path) -> str:
    """Return SHA256 for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_replicates(prefix: str) -> list[str]:
    """Split a stitched prefix into replicate labels."""

    return prefix.split("_")


def load_stitched(root: Path, domain_id: str, prefix: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Load an existing stitched NPZ trajectory."""

    path = root / domain_id / "stitched" / f"stitched_{prefix}.npz"
    data = np.load(path, allow_pickle=True)
    coords = np.asarray(data["coords"], dtype=float)
    labels = [str(item) for item in data["labels"].tolist()]
    metadata = json.loads(str(data["metadata"].item())) if "metadata" in data.files else {}
    return coords, labels, metadata


def mdcorealign_summary_path(root: Path, domain_id: str, prefix: str) -> Path:
    """Return existing stitched-prefix MDCoreAlign summary path."""

    return root / domain_id / "mdcorealign_runs" / f"stitched_prefix_{prefix}" / "summary.json"


def mdcorealign_aligned_path(root: Path, domain_id: str, prefix: str) -> Path:
    """Return existing stitched-prefix MDCoreAlign aligned arrays path."""

    summary = json.loads(mdcorealign_summary_path(root, domain_id, prefix).read_text())
    return Path(summary["aligned_path"])


def check_command(command: list[str], timeout: float = 20.0) -> tuple[bool, str]:
    """Run a tool status command and return success plus short output."""

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    text = (completed.stdout + "\n" + completed.stderr).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return completed.returncode == 0 or bool(lines), "\n".join(lines[:8])


def write_tool_status(root: Path, mdlovofit_exe: Path, theseus_exe: Path, docs_prefix: str) -> None:
    """Inspect external executable availability and write status docs."""

    rows = []
    for tool, exe, commands in (
        ("MDLovoFit", mdlovofit_exe, [[str(mdlovofit_exe), "-h"], [str(mdlovofit_exe), "--help"]]),
        ("THESEUS", theseus_exe, [[str(theseus_exe), "-h"], [str(theseus_exe), "--version"]]),
    ):
        exists = exe.exists() and os.access(exe, os.X_OK)
        command_outputs = []
        works = False
        if exists:
            for command in commands:
                ok, output = check_command(command)
                works = works or ok
                command_outputs.append(f"$ {' '.join(command)}\n{output}")
        rows.append(
            {
                "tool": tool,
                "executable_path": exe,
                "executable_exists": int(exists),
                "help_command_works": int(works),
                "version_or_header": command_outputs[-1] if command_outputs else "NA",
                "final_status": "available" if exists and works else "unavailable",
                "notes": "Help/version command inspected; nonzero help exits with useful text are accepted.",
            }
        )
    write_tsv(root / "aggregate" / "external_tool_status.tsv", rows, TOOL_STATUS_COLUMNS)
    Path(f"docs/{docs_prefix}_external_tool_status.md").write_text(
        "# Replicate-Stitching External Tool Status\n\n"
        + "\n".join(
            f"## {row['tool']}\n\n"
            f"- Executable: `{row['executable_path']}`\n"
            f"- Exists/executable: {bool(row['executable_exists'])}\n"
            f"- Help/version command works: {bool(row['help_command_works'])}\n"
            f"- Final status: {row['final_status']}\n\n"
            "```text\n"
            f"{row['version_or_header']}\n"
            "```\n"
            for row in rows
        )
    )


def export_external_inputs(root: Path) -> list[dict[str, Any]]:
    """Export cached stitched trajectories as CA-only multi-model PDB inputs."""

    rows = []
    external_inputs = root / "external_inputs"
    external_inputs.mkdir(parents=True, exist_ok=True)
    for domain_id in DOMAINS:
        for prefix in PREFIXES:
            coords, labels, _metadata = load_stitched(root, domain_id, prefix)
            out = external_inputs / f"{domain_id}_{prefix}_ca_multimodel.pdb"
            write_multimodel_pdb(coords, out)
            rows.append(
                {
                    "domain_id": domain_id,
                    "prefix": prefix,
                    "included_replicates": ",".join(included_replicates(prefix)),
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "input_path": out,
                    "input_format": "CA-only multi-model PDB",
                    "sha256": sha256_file(out),
                    "residue_mapping_ok": int(len(labels) == coords.shape[1]),
                    "notes": "Coordinates copied from cached stitched NPZ; no stride, no frame cap, no redownload.",
                }
            )
    write_tsv(external_inputs / "input_manifest.tsv", rows, INPUT_COLUMNS)
    return rows


def run_mdcorealign_repeat(coords: np.ndarray) -> tuple[str, float, str]:
    """Run one warm-cache MDCoreAlign repeat on cached coordinates."""

    start = time.perf_counter()
    try:
        result = solve_trajectory_robust_alignment(
            coords,
            reference_mode="first_frame",
            warm_start=True,
            rule="huber",
            max_iterations=20,
            tolerance=1e-6,
            objective_tolerance=1e-5,
            objective_patience=2,
            sequence_continuity_strength=0.05,
            contact_consistency_strength=0.0,
            record_history=False,
        )
        _ = robust_aligned_coords_from_result(result)
        return "success", time.perf_counter() - start, "algorithm_only_cached_coords"
    except Exception as exc:
        return "failed", time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def run_mdlovofit_repeat(
    input_path: Path,
    coords: np.ndarray,
    outdir: Path,
    executable: Path,
    timeout: float,
) -> dict[str, Any]:
    """Run one MDLovoFit repeat from an existing input PDB."""

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "trajectory_aligned.pdb"
    try:
        run = run_mdlovofit_on_files(
            [input_path],
            fraction=0.6,
            output_path=output_path,
            executable=executable,
            timeout_seconds=timeout,
        )
        stdout_path = outdir / "stdout.txt"
        stderr_path = outdir / "stderr.txt"
        stdout_path.write_text(run.stdout)
        stderr_path.write_text(run.stderr)
        aligned, occupancy = read_multimodel_pdb_with_occupancy(output_path, n_atoms=coords.shape[1])
        rmsd = mean_rmsd(aligned, coords[0])
        return {
            "status": "success",
            "runtime": run.subprocess_runtime_seconds,
            "command": " ".join(run.command),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "output_path": output_path,
            "failure_reason": "NA",
            "rmsd": rmsd,
            "aligned": aligned,
            "occupancy": occupancy,
            "parsed": run.parsed_output,
            "notes": f"sha256_output={sha256_file(output_path)}",
        }
    except Exception as exc:
        stdout_path = outdir / "stdout.txt"
        stderr_path = outdir / "stderr.txt"
        stdout_path.write_text("")
        stderr_path.write_text(str(exc))
        return {
            "status": "failed",
            "runtime": "NA",
            "command": f"{executable} -f 0.600000 -t {output_path} {input_path}",
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "output_path": output_path,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "rmsd": "NA",
            "aligned": None,
            "occupancy": None,
            "parsed": None,
            "notes": "NA",
        }


def theseus_output_path(outdir: Path, input_path: Path) -> Path | None:
    """Find the THESEUS aligned PDB after direct -a0 execution."""

    candidates = sorted(outdir.glob("*_sup.pdb"))
    if candidates:
        return candidates[0]
    stem = input_path.stem
    candidate = outdir / f"{stem}_sup.pdb"
    return candidate if candidate.exists() else None


def run_theseus_repeat(
    input_path: Path,
    coords: np.ndarray,
    outdir: Path,
    executable: Path,
    timeout: float,
) -> dict[str, Any]:
    """Run one THESEUS repeat using the requested `theseus -a0 input.pdb` command."""

    outdir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "-a0", str(input_path.resolve())]
    env = dict(os.environ)
    candidate_lib = executable.parent.parent / "lib"
    if candidate_lib.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{candidate_lib}{os.pathsep}{existing}" if existing else str(candidate_lib)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=outdir,
            env=env,
            timeout=timeout,
            check=False,
        )
        runtime = time.perf_counter() - start
        stdout_path = outdir / "stdout.txt"
        stderr_path = outdir / "stderr.txt"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"THESEUS return code {completed.returncode}")
        aligned_path = theseus_output_path(outdir, input_path)
        if aligned_path is None:
            raise FileNotFoundError("No THESEUS *_sup.pdb output found")
        aligned = read_multimodel_pdb_coords(aligned_path, n_atoms=coords.shape[1])
        rmsd = mean_rmsd(aligned, coords[0])
        variance_paths = sorted(outdir.glob("*_variances.txt"))
        variance_records = parse_theseus_variances(variance_paths[0]) if variance_paths else tuple()
        parsed = parse_theseus_stdout(completed.stdout)
        return {
            "status": "success",
            "runtime": runtime,
            "command": " ".join(command),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "output_path": aligned_path,
            "failure_reason": "NA",
            "rmsd": rmsd,
            "aligned": aligned,
            "parsed": parsed,
            "variance_records": variance_records,
            "notes": f"sha256_output={sha256_file(aligned_path)}",
        }
    except Exception as exc:
        runtime = time.perf_counter() - start
        stdout_path = outdir / "stdout.txt"
        stderr_path = outdir / "stderr.txt"
        stdout_path.write_text(stdout_path.read_text() if stdout_path.exists() else "")
        stderr_path.write_text((stderr_path.read_text() if stderr_path.exists() else "") + f"\n{type(exc).__name__}: {exc}\n")
        return {
            "status": "failed",
            "runtime": runtime,
            "command": " ".join(command),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "output_path": "NA",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "rmsd": "NA",
            "aligned": None,
            "parsed": None,
            "variance_records": tuple(),
            "notes": "NA",
        }


def percentile_iqr(values: list[float]) -> float | str:
    """Return interquartile range."""

    if not values:
        return "NA"
    q75, q25 = np.percentile(np.asarray(values, dtype=float), [75, 25])
    return float(q75 - q25)


def median(values: list[float]) -> float | str:
    """Return median or NA."""

    return float(np.median(np.asarray(values, dtype=float))) if values else "NA"


def jaccard(a: np.ndarray, b: np.ndarray) -> float | str:
    """Return Jaccard index for two masks."""

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b)
    if not np.any(union):
        return "NA"
    return float(np.sum(np.logical_and(a, b)) / np.sum(union))


def fit_scaling(domain_id: str, method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit simple linear/quadratic scaling models for one domain/method."""

    rows = [row for row in rows if row["domain_id"] == domain_id and row["method"] == method and row["status"] == "success"]
    by_prefix: dict[str, list[float]] = {}
    frames_by_prefix: dict[str, float] = {}
    for row in rows:
        value = fnum(row["runtime_sec"])
        if value is None:
            continue
        by_prefix.setdefault(row["prefix"], []).append(value)
        frames_by_prefix[row["prefix"]] = float(row["n_frames"])
    xs = []
    ys = []
    for prefix in PREFIXES:
        if prefix in by_prefix:
            xs.append(frames_by_prefix[prefix])
            ys.append(float(np.median(by_prefix[prefix])))
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if xs_arr.size < 3:
        return {
            "domain_id": domain_id,
            "method": method,
            "n_prefixes": xs_arr.size,
            "R2_linear": "NA",
            "R2_quadratic": "NA",
            "AIC_linear": "NA",
            "AIC_quadratic": "NA",
            "estimated_scaling_preference": "unclear",
            "runtime_per_frame_stability": "NA",
            "notes": "insufficient successful prefixes",
        }

    def fit(design: np.ndarray) -> tuple[float, float]:
        beta, *_ = np.linalg.lstsq(design, ys_arr, rcond=None)
        pred = design @ beta
        rss = float(np.sum((ys_arr - pred) ** 2))
        tss = float(np.sum((ys_arr - np.mean(ys_arr)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0.0 else 1.0
        n = ys_arr.size
        k = design.shape[1]
        aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
        return r2, aic

    linear = np.column_stack([np.ones_like(xs_arr), xs_arr])
    quadratic = np.column_stack([np.ones_like(xs_arr), xs_arr, xs_arr * xs_arr])
    r2_lin, aic_lin = fit(linear)
    r2_quad, aic_quad = fit(quadratic)
    per_frame = ys_arr / xs_arr
    stability = float((np.max(per_frame) - np.min(per_frame)) / np.median(per_frame)) if np.median(per_frame) else float("nan")
    preference = "linear"
    if aic_quad + 2.0 < aic_lin and stability > 0.5:
        preference = "possible_quadratic"
    elif aic_quad + 2.0 < aic_lin and stability > 0.25:
        preference = "possible_superlinear"
    elif abs(aic_quad - aic_lin) < 2.0:
        preference = "unclear"
    return {
        "domain_id": domain_id,
        "method": method,
        "n_prefixes": xs_arr.size,
        "R2_linear": r2_lin,
        "R2_quadratic": r2_quad,
        "AIC_linear": aic_lin,
        "AIC_quadratic": aic_quad,
        "estimated_scaling_preference": preference,
        "runtime_per_frame_stability": stability,
        "notes": "Conservative five-prefix diagnostic; do not over-interpret.",
    }


def update_report(
    root: Path,
    tool_summary: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    rmsd_rows: list[dict[str, Any]],
    docs_prefix: str,
) -> None:
    """Append/update external baseline section in the diagnostic report."""

    report_path = Path(f"docs/{docs_prefix}_diagnostic_report.md")
    base = report_path.read_text() if report_path.exists() else "# Replicate-Stitching Diagnostic Report\n"
    marker = "\n## 11. External Baseline Behaviour on Stitched Replicate Trajectories\n"
    if marker in base:
        base = base.split(marker)[0].rstrip() + "\n"
    mdlovo_success = sum(row["mdlovofit_unified_RMSD"] != "NA" for row in rmsd_rows)
    theseus_success = sum(row["theseus_unified_RMSD"] != "NA" for row in rmsd_rows)
    mdlo_faster = sum(int(row["mdcorealign_faster_than_mdlovofit"]) for row in tool_summary if row["mdcorealign_faster_than_mdlovofit"] != "NA")
    th_faster = sum(int(row["mdcorealign_faster_than_theseus"]) for row in tool_summary if row["mdcorealign_faster_than_theseus"] != "NA")
    n_mdlo = sum(row["mdlovofit_median_runtime_sec"] != "NA" for row in tool_summary)
    n_th = sum(row["theseus_median_runtime_sec"] != "NA" for row in tool_summary)
    scaling_pref = {}
    for method in ("mdcorealign", "mdlovofit", "theseus"):
        prefs = [row["estimated_scaling_preference"] for row in scaling_rows if row["method"] == method]
        scaling_pref[method] = ", ".join(f"{pref}:{prefs.count(pref)}" for pref in sorted(set(prefs)))
    report_path.write_text(
        base.rstrip()
        + marker
        + "\n"
        "MDLovoFit and THESEUS were run on the same cached stitched-prefix CA-only inputs used by MDCoreAlign. Timing was measured on cached inputs and excluded Hugging Face download time and HDF5 extraction time. The external input PDBs were exported once before timing repeats.\n\n"
        f"- MDLovoFit successful stitched-prefix RMSD evaluations: {mdlovo_success}/{len(rmsd_rows)}.\n"
        f"- THESEUS successful stitched-prefix RMSD evaluations: {theseus_success}/{len(rmsd_rows)}.\n"
        f"- MDCoreAlign was faster than MDLovoFit in {mdlo_faster}/{n_mdlo} median warm-cache prefix comparisons.\n"
        f"- MDCoreAlign was faster than THESEUS in {th_faster}/{n_th} median warm-cache prefix comparisons.\n"
        f"- Scaling labels: MDCoreAlign ({scaling_pref.get('mdcorealign', 'NA')}), MDLovoFit ({scaling_pref.get('mdlovofit', 'NA')}), THESEUS ({scaling_pref.get('theseus', 'NA')}).\n\n"
        "Unified post-hoc RMSD was used only as a geometric audit under a shared reference convention. It should not be interpreted as the sole measure of stable/mobile biological interpretation. THESEUS is a statistical-superposition reference rather than a workflow-equivalent stable/mobile method. Stitched replicates remain a diagnostic stress test because replicate-boundary jumps are large and do not represent physically continuous MD transitions.\n"
    )


def make_external_plots(
    root: Path,
    warm_summary: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    """Regenerate the replicate-stitching scaling figure with external baselines."""

    import matplotlib.pyplot as plt

    figdir = root / "figures"
    figdata = root / "figure_data"
    figdir.mkdir(parents=True, exist_ok=True)
    figdata.mkdir(parents=True, exist_ok=True)
    write_tsv(figdata / "external_runtime_summary.tsv", warm_summary, WARM_SUMMARY_COLUMNS)
    write_tsv(figdata / "external_tool_comparison_summary.tsv", comparison_rows, TOOL_COMPARISON_COLUMNS)

    methods = ("mdcorealign", "mdlovofit", "theseus")
    colors = {"mdcorealign": "#4477AA", "mdlovofit": "#CC6677", "theseus": "#228833"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for domain_id in DOMAINS:
        for method in methods:
            rows = [
                row for row in warm_summary
                if row["domain_id"] == domain_id and row["method"] == method and fnum(row["median_runtime_sec"]) is not None
            ]
            rows = sorted(rows, key=lambda row: float(row["n_frames"]))
            if not rows:
                continue
            frames = [float(row["n_frames"]) for row in rows]
            runtimes = [float(row["median_runtime_sec"]) for row in rows]
            per_frame = [float(row["runtime_per_frame"]) for row in rows]
            observed = [float(row["observed_over_expected_linear"]) for row in rows]
            label = f"{domain_id} {method}"
            axes[0].plot(frames, runtimes, marker="o", color=colors[method], alpha=0.45, label=label)
            axes[1].plot(frames, per_frame, marker="o", color=colors[method], alpha=0.45)
            axes[2].plot(frames, observed, marker="o", color=colors[method], alpha=0.45)
    for method in methods:
        axes[0].plot([], [], marker="o", color=colors[method], label=method)
    axes[0].set_title("A. Runtime vs frames")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Median runtime (s)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("B. Runtime per frame")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("s/frame")
    axes[2].set_title("C. Observed / expected linear runtime")
    axes[2].set_xlabel("Frames")
    axes[2].set_ylabel("Ratio")
    for domain_id in DOMAINS:
        rows = [row for row in comparison_rows if row["domain_id"] == domain_id]
        rows = sorted(rows, key=lambda row: float(row["n_frames"]))
        frames = [float(row["n_frames"]) for row in rows]
        for method, key in (
            ("mdcorealign", "mdcorealign_unified_RMSD"),
            ("mdlovofit", "mdlovofit_unified_RMSD"),
            ("theseus", "theseus_unified_RMSD"),
        ):
            ys = [fnum(row[key]) for row in rows]
            if all(value is not None for value in ys):
                axes[3].plot(frames, ys, marker="o", color=colors[method], alpha=0.45)
        delta_pairs = [
            (float(row["n_frames"]), fnum(row["mdcorealign_median_runtime_sec"]), fnum(row["mdlovofit_median_runtime_sec"]))
            for row in rows
        ]
        delta_frames = [frame for frame, mdca, mdlo in delta_pairs if mdca is not None and mdlo is not None]
        deltas = [mdca - mdlo for _frame, mdca, mdlo in delta_pairs if mdca is not None and mdlo is not None]
        axes[4].plot(delta_frames, deltas, marker="o", alpha=0.5, label=domain_id)
        boundary = [fnum(row["boundary_to_within_jump_ratio"]) for row in rows]
        axes[5].plot(frames, boundary, marker="o", alpha=0.5, label=domain_id)
    axes[3].set_title("D. Unified RMSD")
    axes[3].set_xlabel("Frames")
    axes[3].set_ylabel("RMSD")
    axes[4].set_title("E. MDCoreAlign - MDLovoFit runtime")
    axes[4].set_xlabel("Frames")
    axes[4].set_ylabel("Seconds")
    axes[4].axhline(0.0, color="black", linewidth=0.8)
    axes[4].legend(fontsize=8)
    axes[5].set_title("F. Boundary jump ratio")
    axes[5].set_xlabel("Frames")
    axes[5].set_ylabel("Boundary / within")
    axes[5].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "replicate_stitching_scaling.pdf")
    fig.savefig(figdir / "replicate_stitching_scaling.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the external-tool stitched comparison."""

    global DOMAINS, PREFIXES
    args = parse_args()
    DOMAINS = split_csv(args.domain_ids, DOMAINS)
    PREFIXES = split_csv(args.prefixes, PREFIXES)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    (args.root / "aggregate").mkdir(parents=True, exist_ok=True)
    write_tool_status(args.root, args.mdlovofit_exe, args.theseus_exe, args.docs_prefix)
    input_rows = export_external_inputs(args.root)
    input_by_key = {(row["domain_id"], row["prefix"]): row for row in input_rows}
    stitching_rows = {
        (row["domain_id"], row["stitch_name"]): row
        for domain_id in DOMAINS
        for row in read_tsv(args.root / domain_id / "stitching_manifest.tsv")
    }

    warm_rows: list[dict[str, Any]] = []
    mdlo_run_rows: list[dict[str, Any]] = []
    theseus_run_rows: list[dict[str, Any]] = []
    rmsd_records: dict[tuple[str, str], dict[str, Any]] = {}
    mdlo_core_rows: list[dict[str, Any]] = []
    core_agreement_rows: list[dict[str, Any]] = []

    for domain_id in DOMAINS:
        for prefix in PREFIXES:
            input_row = input_by_key[(domain_id, prefix)]
            input_path = Path(input_row["input_path"])
            coords, _labels, _metadata = load_stitched(args.root, domain_id, prefix)
            n_reps = len(included_replicates(prefix))
            n_frames = int(coords.shape[0])
            n_residues = int(coords.shape[1])
            mdca_art = np.load(mdcorealign_aligned_path(args.root, domain_id, prefix))
            mdca_aligned = np.asarray(mdca_art["aligned_coords"], dtype=float)
            mdca_core = np.asarray(mdca_art["core_mask"], dtype=bool)
            mdca_rmsd = mean_rmsd(mdca_aligned, coords[0])

            method_results: dict[str, list[float]] = {"mdcorealign": [], "mdlovofit": [], "theseus": []}
            mdlo_first: dict[str, Any] | None = None
            theseus_first: dict[str, Any] | None = None
            for repeat_id in range(1, args.repeats + 1):
                status, runtime, notes = run_mdcorealign_repeat(coords)
                if status == "success":
                    method_results["mdcorealign"].append(runtime)
                warm_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "method": "mdcorealign",
                        "repeat_id": repeat_id,
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "runtime_scope": "algorithm_only_cached_coords",
                        "runtime_sec": runtime,
                        "status": status,
                        "notes": notes,
                    }
                )

                mdlo = run_mdlovofit_repeat(
                    input_path,
                    coords,
                    args.root / "mdlovofit_runs" / domain_id / prefix / f"repeat_{repeat_id}",
                    args.mdlovofit_exe,
                    args.timeout_seconds,
                )
                if mdlo["status"] == "success":
                    method_results["mdlovofit"].append(float(mdlo["runtime"]))
                    if mdlo_first is None:
                        mdlo_first = mdlo
                warm_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "method": "mdlovofit",
                        "repeat_id": repeat_id,
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "runtime_scope": "external_subprocess_cached_input",
                        "runtime_sec": mdlo["runtime"],
                        "status": mdlo["status"],
                        "notes": mdlo["failure_reason"] if mdlo["status"] != "success" else "cached input; no HDF5/download/export time",
                    }
                )

                theseus = run_theseus_repeat(
                    input_path,
                    coords,
                    args.root / "theseus_runs" / domain_id / prefix / f"repeat_{repeat_id}",
                    args.theseus_exe,
                    args.timeout_seconds,
                )
                if theseus["status"] == "success":
                    method_results["theseus"].append(float(theseus["runtime"]))
                    if theseus_first is None:
                        theseus_first = theseus
                warm_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "method": "theseus",
                        "repeat_id": repeat_id,
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "runtime_scope": "external_subprocess_cached_input",
                        "runtime_sec": theseus["runtime"],
                        "status": theseus["status"],
                        "notes": theseus["failure_reason"] if theseus["status"] != "success" else "cached input; no HDF5/download/export time",
                    }
                )

            if mdlo_first is not None:
                mdlo_run_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "included_replicates": input_row["included_replicates"],
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "status": mdlo_first["status"],
                        "runtime_sec": mdlo_first["runtime"],
                        "command": mdlo_first["command"],
                        "stdout_path": mdlo_first["stdout_path"],
                        "stderr_path": mdlo_first["stderr_path"],
                        "output_path": mdlo_first["output_path"],
                        "failure_reason": mdlo_first["failure_reason"],
                        "notes": mdlo_first["notes"],
                    }
                )
            else:
                mdlo_run_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "included_replicates": input_row["included_replicates"],
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "status": "failed",
                        "runtime_sec": "NA",
                        "command": "NA",
                        "stdout_path": "NA",
                        "stderr_path": "NA",
                        "output_path": "NA",
                        "failure_reason": "all repeats failed",
                        "notes": "NA",
                    }
                )

            if theseus_first is not None:
                theseus_run_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "included_replicates": input_row["included_replicates"],
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "status": theseus_first["status"],
                        "runtime_sec": theseus_first["runtime"],
                        "command": theseus_first["command"],
                        "stdout_path": theseus_first["stdout_path"],
                        "stderr_path": theseus_first["stderr_path"],
                        "output_path": theseus_first["output_path"],
                        "failure_reason": theseus_first["failure_reason"],
                        "notes": theseus_first["notes"],
                    }
                )
            else:
                theseus_run_rows.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "included_replicates": input_row["included_replicates"],
                        "n_replicates": n_reps,
                        "n_frames": n_frames,
                        "n_residues": n_residues,
                        "status": "failed",
                        "runtime_sec": "NA",
                        "command": "NA",
                        "stdout_path": "NA",
                        "stderr_path": "NA",
                        "output_path": "NA",
                        "failure_reason": "all repeats failed",
                        "notes": "NA",
                    }
                )

            mdlo_rmsd = mdlo_first["rmsd"] if mdlo_first is not None else "NA"
            theseus_rmsd = theseus_first["rmsd"] if theseus_first is not None else "NA"
            rmsd_records[(domain_id, prefix)] = {
                "domain_id": domain_id,
                "prefix": prefix,
                "included_replicates": input_row["included_replicates"],
                "n_replicates": n_reps,
                "n_frames": n_frames,
                "n_residues": n_residues,
                "mdcorealign_unified_RMSD": mdca_rmsd,
                "mdlovofit_unified_RMSD": mdlo_rmsd,
                "theseus_unified_RMSD": theseus_rmsd,
                "mdcorealign_lower_RMSD_than_mdlovofit": int(fnum(mdlo_rmsd) is not None and mdca_rmsd < float(mdlo_rmsd)),
                "mdcorealign_lower_RMSD_than_theseus": int(fnum(theseus_rmsd) is not None and mdca_rmsd < float(theseus_rmsd)),
                "mdlovofit_status": mdlo_first["status"] if mdlo_first is not None else "failed",
                "theseus_status": theseus_first["status"] if theseus_first is not None else "failed",
                "notes": "Unified RMSD is a geometric audit to frame 0.",
            }

            if mdlo_first is not None and mdlo_first["occupancy"] is not None:
                occupancy = np.asarray(mdlo_first["occupancy"], dtype=float)
                binary = np.all(np.logical_or(np.isclose(occupancy, 0.0), np.isclose(occupancy, 1.0)))
                constant = np.all(np.isclose(occupancy, occupancy[0]))
                if binary and constant:
                    mask = occupancy[0] > 0.5
                    core_res = [str(i + 1) for i, keep in enumerate(mask) if keep]
                    mobile_res = [str(i + 1) for i, keep in enumerate(mask) if not keep]
                    mdlo_fraction = float(np.mean(mask))
                    mdlo_core_rows.append(
                        {
                            "domain_id": domain_id,
                            "prefix": prefix,
                            "n_frames": n_frames,
                            "n_residues": n_residues,
                            "mdlovofit_core_fraction": mdlo_fraction,
                            "mdlovofit_core_residues": ",".join(core_res),
                            "mdlovofit_mobile_residues": ",".join(mobile_res),
                            "extraction_status": "success",
                            "failure_reason": "NA",
                        }
                    )
                    core_agreement_rows.append(
                        {
                            "domain_id": domain_id,
                            "prefix": prefix,
                            "core_jaccard": jaccard(mdca_core, mask),
                            "mobile_jaccard": jaccard(~mdca_core, ~mask),
                            "mdcorealign_core_fraction": float(np.mean(mdca_core)),
                            "mdlovofit_core_fraction": mdlo_fraction,
                            "core_fraction_delta": float(np.mean(mdca_core) - mdlo_fraction),
                            "notes": "MDLovoFit mask extracted from binary constant occupancy field.",
                        }
                    )
                else:
                    mdlo_core_rows.append(
                        {
                            "domain_id": domain_id,
                            "prefix": prefix,
                            "n_frames": n_frames,
                            "n_residues": n_residues,
                            "mdlovofit_core_fraction": "NA",
                            "mdlovofit_core_residues": "NA",
                            "mdlovofit_mobile_residues": "NA",
                            "extraction_status": "failed",
                            "failure_reason": "occupancy was not binary and constant across frames",
                        }
                    )

    aggregate = args.root / "aggregate"
    write_tsv(
        aggregate / "mdlovofit_stitched_runs.tsv",
        [
            {
                "domain_id": row["domain_id"],
                "prefix": row["prefix"],
                "included_replicates": row["included_replicates"],
                "n_replicates": row["n_replicates"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "mdlovofit_status": row["status"],
                "mdlovofit_runtime_sec": row["runtime_sec"],
                "mdlovofit_command": row["command"],
                "stdout_path": row["stdout_path"],
                "stderr_path": row["stderr_path"],
                "output_path": row["output_path"],
                "failure_reason": row["failure_reason"],
                "notes": row["notes"],
            }
            for row in mdlo_run_rows
        ],
        [
            "domain_id",
            "prefix",
            "included_replicates",
            "n_replicates",
            "n_frames",
            "n_residues",
            "mdlovofit_status",
            "mdlovofit_runtime_sec",
            "mdlovofit_command",
            "stdout_path",
            "stderr_path",
            "output_path",
            "failure_reason",
            "notes",
        ],
    )
    write_tsv(
        aggregate / "theseus_stitched_runs.tsv",
        [
            {
                "domain_id": row["domain_id"],
                "prefix": row["prefix"],
                "included_replicates": row["included_replicates"],
                "n_replicates": row["n_replicates"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "theseus_status": row["status"],
                "theseus_runtime_sec": row["runtime_sec"],
                "theseus_command": row["command"],
                "stdout_path": row["stdout_path"],
                "stderr_path": row["stderr_path"],
                "output_path": row["output_path"],
                "failure_reason": row["failure_reason"],
                "notes": row["notes"],
            }
            for row in theseus_run_rows
        ],
        [
            "domain_id",
            "prefix",
            "included_replicates",
            "n_replicates",
            "n_frames",
            "n_residues",
            "theseus_status",
            "theseus_runtime_sec",
            "theseus_command",
            "stdout_path",
            "stderr_path",
            "output_path",
            "failure_reason",
            "notes",
        ],
    )
    write_tsv(aggregate / "stitched_external_tool_warmcache_repeats.tsv", warm_rows, WARM_REPEATS_COLUMNS)

    warm_summary = []
    for domain_id in DOMAINS:
        for method in ("mdcorealign", "mdlovofit", "theseus"):
            r1_values = [
                float(row["runtime_sec"]) for row in warm_rows
                if row["domain_id"] == domain_id and row["prefix"] == "R1" and row["method"] == method and row["status"] == "success"
            ]
            r1_median = float(np.median(r1_values)) if r1_values else None
            r1_frames = None
            for prefix in PREFIXES:
                rows = [
                    row for row in warm_rows
                    if row["domain_id"] == domain_id and row["prefix"] == prefix and row["method"] == method and row["status"] == "success"
                ]
                values = [float(row["runtime_sec"]) for row in rows]
                input_row = input_by_key[(domain_id, prefix)]
                frames = float(input_row["n_frames"])
                if prefix == "R1":
                    r1_frames = frames
                med = median(values)
                rel = float(med / r1_median) if values and r1_median else "NA"
                frame_rel = float(frames / r1_frames) if r1_frames else "NA"
                observed = float(rel / frame_rel) if isinstance(rel, float) and isinstance(frame_rel, float) and frame_rel else "NA"
                warm_summary.append(
                    {
                        "domain_id": domain_id,
                        "prefix": prefix,
                        "method": method,
                        "n_repeats": len(values),
                        "median_runtime_sec": med,
                        "iqr_runtime_sec": percentile_iqr(values),
                        "n_frames": int(frames),
                        "n_residues": input_row["n_residues"],
                        "runtime_per_frame": float(med / frames) if isinstance(med, float) else "NA",
                        "runtime_relative_to_R1": rel,
                        "frame_count_relative_to_R1": frame_rel,
                        "observed_over_expected_linear": observed,
                        "notes": "Warm-cache repeats on cached stitched coordinates/PDB; no download or HDF5 extraction.",
                    }
                )
    write_tsv(aggregate / "stitched_external_tool_warmcache_summary.tsv", warm_summary, WARM_SUMMARY_COLUMNS)

    rmsd_rows = [rmsd_records[(domain_id, prefix)] for domain_id in DOMAINS for prefix in PREFIXES]
    write_tsv(aggregate / "stitched_prefix_unified_rmsd_comparison.tsv", rmsd_rows, RMSD_COLUMNS)
    write_tsv(aggregate / "mdlovofit_stitched_core_mobile_profiles.tsv", mdlo_core_rows, MDLOVO_CORE_COLUMNS)
    write_tsv(aggregate / "mdcorealign_mdlovofit_stitched_core_agreement.tsv", core_agreement_rows, CORE_AGREEMENT_COLUMNS)

    scaling_rows = [
        fit_scaling(domain_id, method, warm_rows)
        for domain_id in DOMAINS
        for method in ("mdcorealign", "mdlovofit", "theseus")
    ]
    write_tsv(aggregate / "tool_scaling_model_summary.tsv", scaling_rows, SCALING_MODEL_COLUMNS)

    comparison_rows = []
    warm_lookup = {(row["domain_id"], row["prefix"], row["method"]): row for row in warm_summary}
    for domain_id in DOMAINS:
        for prefix in PREFIXES:
            input_row = input_by_key[(domain_id, prefix)]
            mdca = warm_lookup[(domain_id, prefix, "mdcorealign")]
            mdlo = warm_lookup[(domain_id, prefix, "mdlovofit")]
            thes = warm_lookup[(domain_id, prefix, "theseus")]
            rmsd = rmsd_records[(domain_id, prefix)]
            boundary = stitching_rows.get((domain_id, prefix), {}).get("boundary_to_within_jump_ratio", "NA")
            mdca_rt = fnum(mdca["median_runtime_sec"])
            mdlo_rt = fnum(mdlo["median_runtime_sec"])
            th_rt = fnum(thes["median_runtime_sec"])
            comparison_rows.append(
                {
                    "domain_id": domain_id,
                    "prefix": prefix,
                    "included_replicates": input_row["included_replicates"],
                    "n_replicates": len(included_replicates(prefix)),
                    "n_frames": input_row["n_frames"],
                    "n_residues": input_row["n_residues"],
                    "mdcorealign_median_runtime_sec": mdca["median_runtime_sec"],
                    "mdlovofit_median_runtime_sec": mdlo["median_runtime_sec"],
                    "theseus_median_runtime_sec": thes["median_runtime_sec"],
                    "mdcorealign_vs_mdlovofit_speedup": float(mdlo_rt / mdca_rt) if mdca_rt and mdlo_rt else "NA",
                    "mdcorealign_vs_theseus_speedup": float(th_rt / mdca_rt) if mdca_rt and th_rt else "NA",
                    "mdcorealign_faster_than_mdlovofit": int(mdca_rt is not None and mdlo_rt is not None and mdca_rt < mdlo_rt),
                    "mdcorealign_faster_than_theseus": int(mdca_rt is not None and th_rt is not None and mdca_rt < th_rt),
                    "mdcorealign_unified_RMSD": rmsd["mdcorealign_unified_RMSD"],
                    "mdlovofit_unified_RMSD": rmsd["mdlovofit_unified_RMSD"],
                    "theseus_unified_RMSD": rmsd["theseus_unified_RMSD"],
                    "mdcorealign_lower_RMSD_than_mdlovofit": rmsd["mdcorealign_lower_RMSD_than_mdlovofit"],
                    "mdcorealign_lower_RMSD_than_theseus": rmsd["mdcorealign_lower_RMSD_than_theseus"],
                    "boundary_to_within_jump_ratio": boundary,
                    "notes": "Median warm-cache runtime and unified RMSD comparison.",
                }
            )
    write_tsv(aggregate / "stitched_tool_comparison_summary.tsv", comparison_rows, TOOL_COMPARISON_COLUMNS)
    make_external_plots(args.root, warm_summary, comparison_rows)
    update_report(args.root, comparison_rows, scaling_rows, rmsd_rows, args.docs_prefix)
    write_json(
        aggregate / "external_comparison_manifest.json",
        {
            "domains": DOMAINS,
            "prefixes": PREFIXES,
            "repeats": args.repeats,
            "mdlovofit_executable": str(args.mdlovofit_exe),
            "theseus_executable": str(args.theseus_exe),
            "thread_environment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "note": "External comparison uses cached stitched CA-only inputs; no mdCATH download or HDF5 extraction included.",
        },
    )
    print(args.root / "aggregate" / "stitched_tool_comparison_summary.tsv")


if __name__ == "__main__":
    main()
