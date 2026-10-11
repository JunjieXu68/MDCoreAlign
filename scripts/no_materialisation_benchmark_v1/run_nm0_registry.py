#!/usr/bin/env python3
"""Freeze source evidence and prospectively register NM benchmark contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
import subprocess

from scripts.no_materialisation_benchmark_v1.common import (
    OUT,
    ROOT,
    STOCK,
    SUBDIRS,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


FROZEN = (
    "results/atlas_large_long_v1/a3_external/centered_input_manifest.tsv",
    "results/atlas_large_long_v1/a3_external/external_raw_repeats.tsv",
    "results/atlas_large_long_v1/atlas_external_baseline_summary.tsv",
    "results/atlas_large_long_v1/atlas_manifest.tsv",
    "results/database_scale_benchmark_v2/mdlovofit_comparison/mdlovofit_all_domain_corrected_matched.tsv",
    "results/database_scale_benchmark_v2/mdlovofit_comparison/mdlovofit_external_panel.tsv",
    "results/database_scale_benchmark_v2/mdlovofit_comparison/mdlovofit_matched_contract_summary.tsv",
    "results/database_scale_benchmark_v2/b1_tail_adjudication/b1_tail_case_atlas.tsv",
    "scripts/database_scale_benchmark_v2/run_mdlovofit_addendum.py",
    "scripts/atlas_large_long_v1/run_a3_internal_loaded.py",
    "src/mdcorealign/contracts.py",
    "src/mdcorealign/experimental.py",
    "src/mdcorealign/gpu_backend.py",
)


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output root: {OUT}")
    for name in SUBDIRS:
        (OUT / name).mkdir(parents=True, exist_ok=False)
    missing = [relative for relative in FROZEN if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen references are missing: {missing}")
    rows = []
    for relative in FROZEN:
        path = ROOT / relative
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "access": "read_only_reference",
            }
        )
    write_tsv(OUT / "00_freeze/frozen_source_manifest.tsv", rows)
    historical_roots = (
        "results/atlas_large_long_v1",
        "results/database_scale_benchmark_v2",
        "results/post_b3_optimisation_v1",
        "results/concept_cases_product_v2",
        "results/manuscript_v3",
        "results/project_completion",
    )
    write_tsv(
        OUT / "00_freeze/historical_root_audit.tsv",
        [
            {
                "path": relative,
                "exists": (ROOT / relative).exists(),
                "policy": "preserve_no_overwrite",
                "note": (
                    "named root absent before NM0; no replacement inferred"
                    if not (ROOT / relative).exists()
                    else "present and frozen"
                ),
            }
            for relative in historical_roots
        ],
    )
    environment = [
        {"item": "generated_at", "value": datetime.now(timezone.utc).isoformat()},
        {"item": "hostname", "value": platform.node()},
        {"item": "platform", "value": platform.platform()},
        {"item": "python", "value": command_output([str(ROOT / '.venv-gpu-cu12/bin/python'), '--version'])},
        {"item": "numpy", "value": command_output([str(ROOT / '.venv-gpu-cu12/bin/python'), '-c', 'import numpy; print(numpy.__version__)'])},
        {"item": "cupy", "value": command_output([str(ROOT / '.venv-gpu-cu12/bin/python'), '-c', 'import cupy; print(cupy.__version__)'])},
        {"item": "mdlovofit_path", "value": str(STOCK)},
        {"item": "mdlovofit_sha256", "value": sha256(STOCK)},
        {"item": "mdlovofit_version", "value": "20.1.1"},
        {"item": "gfortran", "value": command_output(['gfortran', '--version']).splitlines()[0]},
        {"item": "cpu_threads", "value": "MKL/OPENBLAS/OMP/NUMEXPR=2"},
        {"item": "git_metadata_policy", "value": "read_only_no_initialisation"},
        {"item": "new_downloads", "value": "0"},
    ]
    write_tsv(OUT / "00_freeze/environment_manifest.tsv", environment)
    write_tsv(
        OUT / "00_freeze/timing_boundary_dictionary.tsv",
        [
            {"contract": "C1", "method": "stock MDLovoFit", "includes": "same centred CA PDB parse; LOVO search; fixed-subset fit; RMSD reduction; stdout", "excludes": "aligned PDB; RMSF; posthoc output parsing"},
            {"contract": "C1", "method": "MDCoreAlign minimal", "includes": "same centred CA PDB parse; weight estimation; explicit all-frame fit; RMSD reduction", "excludes": "aligned coordinates; RMSF; diagnostics; files"},
            {"contract": "C2", "method": "instrumented MDLovoFit", "includes": "LOVO search; fixed-subset fit; RMSD reduction", "excludes": "PDB parse; process startup; stdout; coordinate output"},
            {"contract": "C2", "method": "MDCoreAlign minimal", "includes": "loaded-coordinate weight estimation; explicit all-frame fit; RMSD reduction", "excludes": "PDB parse; RMSF; diagnostics; files"},
            {"contract": "C3", "method": "optional node throughput", "includes": "persistent GPU or independent stock processes from PDB", "excludes": "aligned-coordinate output"},
        ],
    )
    write_text(
        OUT / "00_freeze/contract_registry.md",
        """# Prospectively Registered Contracts

The primary C1 comparison starts both tools from the identical pre-generated,
centred, CA-only multi-model PDB and excludes aligned-coordinate output. Stock
MDLovoFit is unmodified and is called without `-t` or `-rmsf`. MDCoreAlign
parses that PDB inside C1, returns compact weights/transforms/RMSDs in memory,
and writes nothing inside the timed region.

C2 starts from loaded coordinates. MDLovoFit C2 uses a separately named,
timing-only instrumented binary and includes LOVO search, final fixed-subset
fit and RMSD reduction. MDCoreAlign C2 includes weight estimation, explicit
all-frame weighted fitting and RMSD reduction. C3 is optional and separate.

Scientific parameters are frozen at fraction 0.60, default `ntrial=100`,
default `iref=1`, no `-perframe`, no `-rmsf`. The only stock command change is
removal of `-t <aligned.pdb>`. No algorithm or public default is changed.

ATLAS uses the 48 existing centred PDBs. Historical mdCATH temporary PDBs were
not retained. They will be deterministically reconstructed before timing from
the frozen coordinates with the original streaming writer, checked against
historical input byte counts, shared by all methods, and then removed.
""",
    )
    write_text(
        OUT / "00_freeze/prior_command_audit.md",
        """# Prior Command Audit

The frozen wrappers construct `mdlovofit -f 0.600000 -t OUTPUT INPUT`.
MDLovoFit 20.1.1 source initialises `ntrial=100`, `iref=1`,
`printperframe=.false.`, `print_rmsf=.false.` and `trajfile=\"NONE\"`.
No frozen matched-PDB command adds `-perframe`, `-rmsf`, `-ntrial` or `-iref`.
The registered no-output command is therefore
`mdlovofit -f 0.600000 INPUT`; removal of `-t OUTPUT` is its only change.

The earlier approximately 70-fold workflow result is not registered as an
algorithm-only comparison because MDLovoFit materialised an aligned PDB while
compact MDCoreAlign did not.
""",
    )
    write_json(
        OUT / "00_freeze/nm0_status.json",
        {
            "status": "pass",
            "frozen_files": len(rows),
            "stock_sha256": sha256(STOCK),
            "new_downloads": 0,
            "manuscript_v3_named_root_exists": (ROOT / "results/manuscript_v3").exists(),
            "contracts_registered_before_timing": True,
        },
    )


if __name__ == "__main__":
    main()
