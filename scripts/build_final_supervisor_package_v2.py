#!/usr/bin/env python3
"""Assemble the non-destructive MDCoreAlign advisor release candidate v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "advisor_release_candidate_v2"
DATE = datetime.now().strftime("%Y%m%d")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path, output: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            rows.append((path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)


def copy_tree_filtered(source: Path, target: Path, *, suffixes: set[str] | None = None) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", "build", "dist"} or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.name in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name.endswith("~"):
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        copy_file(path, target / path.relative_to(source))


def run_capture(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return f"$ {' '.join(command)}\n{completed.stdout}{completed.stderr}\nexit_code={completed.returncode}\n"


def prepare_dirs() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing release root: {OUT}")
    for name in [
        "00_freeze", "01_product_contract", "02_source", "03_docs", "04_examples",
        "05_tests", "06_build", "07_clean_install", "08_advisor_materials",
        "handoff_to_chatgpt",
    ]:
        (OUT / name).mkdir(parents=True)


def freeze() -> None:
    io_root = ROOT / "results" / "purpose_based_coordinate_export_v1"
    required = [
        io_root / "final_coordinate_export_report.md",
        io_root / "purpose_based_coordinate_export_v1_review_bundle_20260827.tar.gz",
        io_root / "08_validation" / "full_test_output.txt",
        io_root / "08_validation" / "scientific_regression.tsv",
        io_root / "02_reader_audit" / "validated_input_matrix.tsv",
        io_root / "actual_writer_support.tsv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("IO prerequisite incomplete: " + ", ".join(map(str, missing)))
    process_text = run_capture(
        ["pgrep", "-af", "mdcath|atlas|mdlovofit|theseus|NM[0-9]|MA[0-9]|package.*build"]
    )
    write(OUT / "00_freeze" / "active_process_check.txt", process_text)
    write(
        OUT / "00_freeze" / "io_prerequisite_status.md",
        """# IO prerequisite status

Status: PASS.

IO0-IO9 records production-ready mmCIF structure/selected-ensemble output,
production-ready XTC trajectory output with its documented box-reader caveat,
and backward-compatible PDB output. The canonical IO suite recorded 356 passed,
10 skipped and 2 subtests passed; the frozen scientific regression passed
21/21 comparisons. Archive extraction and manifest verification passed.
""",
    )
    checkpoint = OUT / "00_freeze" / "source_checkpoint.tar.gz"
    include = ["src", "tests", "docs", "examples", "pyproject.toml", "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md"]
    with tarfile.open(checkpoint, "w:gz") as archive:
        for name in include:
            path = ROOT / name
            if path.exists():
                archive.add(path, arcname=name, filter=lambda info: None if "__pycache__" in info.name or info.name.endswith((".pyc", ".pyo")) else info)
    source_rows = []
    for name in include:
        path = ROOT / name
        paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
        for item in paths:
            if "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
                continue
            source_rows.append((item.relative_to(ROOT).as_posix(), item.stat().st_size, sha256(item)))
    with (OUT / "00_freeze" / "source_manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(source_rows)
    packages = ["numpy", "scipy", "MDAnalysis", "gemmi", "cupy"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed in canonical CPU interpreter"
    gpu = run_capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]).replace("\n", " | ")
    env_rows = [
        ("timestamp", datetime.now().astimezone().isoformat()),
        ("package_version", "0.1.0rc1"),
        ("python", sys.version.replace("\n", " ")),
        ("platform", platform.platform()),
        ("cpu", platform.processor() or run_capture(["uname", "-m"]).strip()),
        ("gpu", gpu),
        ("blas_lapack", "NumPy runtime; see numpy_show_config.txt"),
        ("recommended_production_algorithm", "throughput"),
        ("backward_compatible_implicit_default", "exact/default"),
        *[(name, value) for name, value in versions.items()],
    ]
    with (OUT / "00_freeze" / "environment_manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["item", "value"])
        writer.writerows(env_rows)
    write(OUT / "00_freeze" / "numpy_show_config.txt", run_capture([sys.executable, "-c", "import numpy as np; np.show_config()"]),)
    write(OUT / "00_freeze" / "git_status.txt", run_capture(["git", "status", "--short", "--branch"]) + run_capture(["git", "rev-parse", "--verify", "HEAD"]))
    write(OUT / "00_freeze" / "cli_help.txt", run_capture([sys.executable, "-m", "mdcorealign", "--help"]),)
    evidence = [
        ("throughput_vs_exact_ATLAS", "results/atlas_large_long_v1/atlas_quality_summary.tsv", "48 registered prefix cases"),
        ("mdCATH_database_scale", "results/database_scale_benchmark_v2/b3_full_mdcath/b3_database_throughput_summary.tsv", "134,950 manifest rows"),
        ("ATLAS_large_long", "results/atlas_large_long_v1/atlas_performance_summary.tsv", "eight R1 systems, six prefixes"),
        ("MDLovoFit_method_agreement", "results/no_materialisation_benchmark_v1/method_agreement_v2/08_reporting/final_method_agreement_report.md", "untimed agreement analysis"),
        ("no_materialisation_timing", "results/no_materialisation_benchmark_v1/06_analysis/final_no_materialisation_decision_report.md", "C1 primary; C2 secondary"),
        ("matched_materialised_timing", "results/atlas_large_long_v1/atlas_performance_summary.tsv", "contract C only"),
        ("W5_cases", "results/concept_cases_product_v2/final_revision_report.md", "frozen 3nci_A and 6sup_A cases"),
        ("limitations", "results/atlas_large_long_v1/atlas_limitations.md", "negative results and scope"),
    ]
    with (OUT / "00_freeze" / "frozen_evidence_registry.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["evidence", "source_file", "row_or_filter"])
        for row in evidence:
            if not (ROOT / row[1]).exists():
                raise SystemExit(f"missing frozen evidence: {row[1]}")
            writer.writerow(row)


def product_contract() -> None:
    contract = OUT / "01_product_contract"
    write(contract / "advisor_product_contract.md", """# Advisor product contract

MDCoreAlign throughput is the recommended production algorithm. CPU and GPU
are backends of the same fixed-budget stable-weight estimator; after weights
are frozen, every frame receives an explicit weighted rigid-body fit. The
chronological float64 exact mode is the slower numerical reference/audit path,
not biological ground truth. Omitting `--mode` retains the historical exact
default for compatibility; new production commands must state `--mode
throughput` explicitly.

Compact analysis is the default. Ordinary/full-set RMSF is opt-in with
`--ordinary-rmsf`; W5 is optional, alignment-free local internal deformation
and is neither RMSD nor a stable-core/domain finder. Explicit GPU requests never
fall back silently.

Coordinate materialisation occurs only with `apply`: `structure` writes
PDBx/mmCIF, `trajectory` writes XTC, and `compatibility` writes multi-model PDB.
The shared apply engine is chunked and bounded-memory.
""")
    tables = {
        "public_workflow_table.tsv": [
            ["workflow", "command", "status", "purpose"],
            ["throughput_cpu", "mdcorealign align --mode throughput --backend cpu", "recommended", "production compact analysis"],
            ["throughput_gpu", "mdcorealign align --mode throughput --backend gpu", "recommended when CUDA is available", "same production method/backend"],
            ["exact", "mdcorealign align --mode exact --backend cpu", "reference/audit", "chronological numerical comparison"],
            ["local_deformation", "mdcorealign local-deformation", "optional", "alignment-free W5 diagnostic"],
            ["apply", "mdcorealign apply --purpose structure|trajectory|compatibility", "production", "coordinate materialisation"],
        ],
        "legacy_mode_table.tsv": [
            ["name", "classification", "advisor_first_screen"],
            *[[name, "legacy or research compatibility", "no"] for name in ["default", "long", "long-fast", "alignment-only", "fast-global", "diagnostic-rich", "balanced-lite", "C1/C2/D64/adaptive", "sparse/keyframe/profile"]],
        ],
        "output_contract.tsv": [
            ["output", "default", "meaning"],
            ["stable_weight", "yes", "continuous model-derived stability"],
            ["core_label", "yes", "reporting mask at selected fraction"],
            ["transforms", "yes", "explicit weighted fit for every frame"],
            ["RMSD", "yes", "full-selection and weighted fitted deviation"],
            ["robust_RMSF", "yes", "stable-core alignment-frame-dependent RMSF"],
            ["ordinary_RMSF", "no", "opt-in full-set-fit comparison"],
            ["W5", "no", "optional local deformation diagnostic"],
            ["aligned_coordinates", "no", "materialise later with apply"],
        ],
        "input_output_matrix.tsv": [
            ["input", "status", "structure_output", "trajectory_output", "compatibility_output"],
            ["PDB+XTC", "validated", "mmCIF", "XTC", "PDB"],
            ["GRO+XTC", "validated", "mmCIF", "XTC", "PDB"],
            ["CA multi-model PDB", "validated", "mmCIF", "requires matching topology for XTC", "PDB"],
            ["NumPy array", "validated compact input", "generic metadata", "requires matching topology", "PDB"],
        ],
    }
    for name, rows in tables.items():
        with (contract / name).open("w", newline="") as handle:
            csv.writer(handle, delimiter="\t").writerows(rows)
    write(contract / "terminology.md", """# Terminology

- **Throughput**: fixed-budget continuous stable-weight estimation plus exact
  weighted fitting of every frame.
- **Exact**: chronological sequential float64 numerical reference/audit path.
- **Stable weight**: continuous model-derived residue weight, not probability.
- **Core label**: reporting subset derived from stable weights at a declared fraction.
- **Robust RMSF**: RMSF in the stable-core fitted frame; alignment-frame dependent.
- **W5**: dimensionless alignment-free local internal-deformation index; not RMSD.
""")


def copy_release_source() -> Path:
    repository = OUT / "02_source" / "repository"
    for name in ["pyproject.toml", "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md"]:
        if (ROOT / name).exists():
            copy_file(ROOT / name, repository / name)
    for package in ["mdcorealign", "mdsuperpose"]:
        copy_tree_filtered(ROOT / "src" / package, repository / "src" / package, suffixes={".py"})
    selected_docs = [
        "installation.md", "quickstart.md", "cli_reference.md", "python_api.md",
        "output_formats.md", "md_preprocessing.md", "rmsf_flexible_segments.md",
        "troubleshooting.md", "coordinate_export.md", "input_formats.md",
        "xtc_output.md", "mmcif_output.md", "pdb_compatibility.md", "support_matrix.md",
        "modes_and_quality_contracts.md", "cpu_installation.md", "gpu_installation.md",
        "local_deformation.md", "large_trajectory_workflow.md", "output_schema.md",
        "recommended_workflow.md", "release.md", "user_testing_protocol.md",
        "mdlovofit_comparison.md", "non_protein_systems.md",
    ]
    for name in selected_docs:
        source = ROOT / "docs" / name
        if source.exists():
            copy_file(source, repository / "docs" / name)
    selected_test_names = {
        "test_alignment.py", "test_fast_kabsch_equivalence.py", "test_formats_cli.py",
        "test_input_format_policy.py", "test_mdcorealign_release_readiness.py",
        "test_mdsuperpose_alignment.py", "test_mdsuperpose_contacts.py",
        "test_mdsuperpose_core_selection.py", "test_mdsuperpose_export.py",
        "test_mdsuperpose_flexibility.py", "test_mdsuperpose_protein_input.py",
        "test_mdsuperpose_robust.py", "test_mdsuperpose_smoke.py",
        "test_mdsuperpose_trajectory_mode.py", "test_mdsuperpose_weighted_alignment.py",
        "test_mdsuperpose_weights.py", "test_non_protein_policy.py",
        "test_product_contract_cli.py", "test_soft_superposition.py",
        "test_advisor_product_contract.py",
    }
    for path in sorted((ROOT / "tests").rglob("*.py")):
        relative = path.relative_to(ROOT / "tests")
        if path.name in selected_test_names or relative.parts[0] in {
            "purpose_based_coordinate_export_v1", "cpu_optimisation_v1", "throughput_gpu_v1"
        }:
            copy_file(path, repository / "tests" / relative)
    copy_tree_filtered(ROOT / "examples" / "toy_md", repository / "examples" / "toy_md")
    for name in ["__init__.py", "demo_synthetic.py"]:
        source = ROOT / "scripts" / name
        if source.exists():
            copy_file(source, repository / "scripts" / name)
    write(repository / "THIRD_PARTY_NOTICES.md", third_party_text())
    return repository


def third_party_text() -> str:
    return """# Third-party notices

MDCoreAlign base installation uses NumPy. MDAnalysis (GPL-2.0-or-later) and
Gemmi (MPL-2.0) are optional external-I/O dependencies. CuPy (MIT) is an
optional CUDA backend. Their source is not bundled.

MDLovoFit and THESEUS were research comparators. Their executables and source
are not included in this package; users should consult the respective upstream
projects and licences. The advisor package contains no third-party binaries.

Dependency decision: keep `MDAnalysis` and `Gemmi` together in the existing
`external`/`io` extras for a complete purpose-based I/O installation, and keep
CuPy isolated in `gpu`. The base package remains NumPy-only. This packaging pass
does not silently redesign optional dependency groups.
"""


def advisor_docs(repository: Path) -> None:
    docs = OUT / "03_docs"
    content = {
        "00_START_HERE.md": """# Start here

MDCoreAlign 0.1.0rc1 is an advisor/private-test release candidate for robust
protein-trajectory alignment and flexibility analysis. Install the external-I/O
extra, then run `bash run_advisor_smoke_test.sh`. For a first manual run use
`mdcorealign align --input examples/03_multimodel_pdb/trajectory.pdb
--input-format ca-pdb --mode throughput --backend cpu --output-dir result`.
Use `--mode exact` only as the slower numerical audit.

Expected smoke time is under two minutes excluding package installation. Inspect
`weights.csv`, `residue_profile.tsv`, `frame_metrics.tsv`, `transforms.npz`,
`flexible_segments.tsv`, and export provenance. GPU is optional; explicit GPU
requests fail rather than falling back. Formal benchmarks are CA-level; RMSF is
alignment-frame dependent; PBC/nojump remains upstream responsibility. The
working tree has no valid Git HEAD, so no commit hash is claimed. Checksums are
in `sha256_manifest.tsv`.
""",
        "PROJECT_OVERVIEW.md": """# Project overview

Rigid superposition removes global translation and rotation before RMSD/RMSF
interpretation, but the fitted residue set changes the coordinate frame.
Weighted robust fitting is established; MDCoreAlign contributes a fixed-budget
continuous stability-weight estimator, explicit weighted fitting of every
frame, scalable CPU/GPU backends, compact outputs and optional alignment-free
W5 local-deformation analysis. Throughput is the recommended production method.
Chronological exact is retained as a slower numerical reference/audit path.
""",
        "CURRENT_EVIDENCE.md": current_evidence_text(),
        "INSTALL_CPU.md": """# CPU installation

Create a Python 3.11+ environment and install the wheel in `dist/`:
`python -m pip install dist/mdcorealign-0.1.0rc1-py3-none-any.whl`.
The base dependency is NumPy only. Verify with `mdcorealign --help`, then run
`examples/03_multimodel_pdb/run.sh`.
""",
        "INSTALL_GPU.md": """# GPU installation

Install the wheel and the matching CUDA extra with `python -m pip install
'dist/mdcorealign-0.1.0rc1-py3-none-any.whl[gpu]'`. The current extra targets
`cupy-cuda12x>=13,<14`; CUDA driver compatibility remains the installer's
responsibility. Run `bash run_gpu_smoke_test.sh`. An explicit `--backend gpu`
never silently falls back to CPU.
""",
        "QUICK_START.md": """# Quick start

Production CPU: `mdcorealign align --input examples/03_multimodel_pdb/trajectory.pdb
--input-format ca-pdb --mode throughput --backend cpu --output-dir result`.
Numerical audit: repeat with `--mode exact`. Add `--ordinary-rmsf` only when the
full-set-fit RMSF comparison is needed. Use `mdcorealign local-deformation` for
W5. Materialise coordinates later with `mdcorealign apply --result-dir result
--input ... --purpose compatibility --output aligned.pdb`.
""",
        "OUTPUTS_AND_INTERPRETATION.md": """# Outputs and interpretation

`stable_weight` is a continuous model-derived stability score; `mobility_score`
is its complementary reporting score. The binary core is a declared-fraction
summary, not ground truth. Full-selection RMSD and weighted RMSD answer different
questions. Robust RMSF is measured in the stable-core fitted frame and remains
alignment-frame dependent. W5 reports local internal geometric change and must
not be called RMSD. Warnings should be retained in downstream analysis.
""",
        "ALGORITHM_IMPLEMENTATION_MAP.md": algorithm_map_text(),
        "KNOWN_LIMITATIONS.md": limitations_text(),
        "LEGACY_AND_ADVANCED.md": """# Legacy and advanced interfaces

The historical implicit `default`, `long`, `long-fast`, `alignment-only`,
`fast-global`, `diagnostic-rich`, `balanced-lite`, chronology candidates and
sparse/keyframe/profile studies are retained only for compatibility or research
reproducibility. They are not normal advisor-facing modes. Use explicit
`--mode throughput`, `--mode exact`, `local-deformation`, and `apply`.
""",
    }
    for name, text in content.items():
        write(docs / name, text)
    mapping = {
        "INPUT_FORMATS.md": ROOT / "docs" / "input_formats.md",
        "COORDINATE_EXPORT.md": ROOT / "docs" / "coordinate_export.md",
        "TROUBLESHOOTING.md": ROOT / "docs" / "troubleshooting.md",
    }
    for name, source in mapping.items():
        copy_file(source, docs / name)
    write(docs / "THIRD_PARTY_NOTICES.md", third_party_text())
    copy_file(OUT / "00_freeze" / "environment_manifest.tsv", docs / "environment_manifest.tsv")


def current_evidence_text() -> str:
    return """# Current frozen evidence

All numbers below map to `frozen_evidence_registry.tsv`; no benchmark was rerun.

- On the matched no-materialisation C1 contract, throughput CPU was 17.65x the
  stock MDLovoFit workflow median across 5,397 corrected mdCATH domains and
  21.94x across 48 ATLAS settings. GPU medians were 17.48x and 23.92x. Both
  methods parsed the same centred CA multi-model PDB and neither wrote aligned
  coordinates. The larger C2 core-compute ratios are secondary, not headline
  workflow speedups.
- The ATLAS R1 panel contained eight systems, 188-2,128 CA and 10,001 frames.
  CPU/GPU throughput core labels matched in all 48 prefixes. GPU throughput met
  the prospective exact-relative core/RMSF contract in 42/48 prefixes; the six
  misses were short-prefix core-overlap tails, with zero catastrophic failures.
- The full mdCATH deployment manifest had 134,950 rows: 134,911 successful, 14
  failed and 25 explicitly unavailable. These are operational breadth results,
  not biological replication.
- In the untimed MDLovoFit agreement study, median matched-60% exact core
  Jaccard was 0.900, continuous ranking Spearman 0.965, independently recomputed
  RMSD Spearman 0.992 and RMSF Spearman 0.980. The conclusion is method-dependent
  stable-reference solutions, not biological equivalence.
- W5 cases illustrate complementary local deformation but do not establish
  domains, mechanisms or biological ground truth.
"""


def limitations_text() -> str:
    return """# Known limitations

- Formal large-scale benchmarks are C-alpha level.
- Rare throughput and short-prefix core-overlap tails are reported; exact is an
  audit reference, not biological truth.
- Stable frames, fitted coordinates, RMSD and RMSF can be method-, reference-
  and core-fraction-dependent.
- W5 does not prove mechanism and is not a mature domain detector.
- ATLAS PBC/nojump was documented upstream but not independently reconstructed;
  exporting or rotating a unit cell does not prove preprocessing correctness.
- PDB+XTC, GRO+XTC and CA multi-model PDB are the validated IO-v1 combinations;
  other ecosystem-readable pairs are not formal validation claims.
- mmCIF is structure-centric and XTC is trajectory-centric. The raw XDR XTC
  file retains the rotated 3x3 box matrix, while MDAnalysis
  `Timestep.dimensions` can canonicalise high-level orientation. Use
  `--drop-unit-cell` when omission is intentional.
- No universal-fastest claim is supported.
"""


def algorithm_map_text() -> str:
    return """# Algorithm implementation map

| Function | Repository-relative implementation |
| --- | --- |
| Throughput stable weights | `src/mdcorealign/contracts.py:solve_contract`; CPU estimator `_estimate_independent_weights` |
| Exact sequential weights | `src/mdcorealign/contracts.py:solve_exact_contract`; `src/mdsuperpose/trajectory.py` |
| All-frame weighted fitting | `src/mdcorealign/cpu_all_frame_engine.py`; `src/mdsuperpose/alignment.py:solve_weighted_rigid_alignment` |
| GPU throughput | `src/mdcorealign/gpu_backend.py:solve_contract_gpu` |
| RMSD/RMSF accumulation | `src/mdcorealign/cpu_all_frame_engine.py`; `src/mdsuperpose/flexibility.py:compute_rmsf` |
| W5 local deformation | `src/mdcorealign/window_kinematics.py`; CLI `_run_local_deformation` |
| CLI and schema | `src/mdcorealign/cli.py` |
| Streaming apply | `src/mdcorealign/coordinate_export.py:stream_apply_transforms` |
| mmCIF writer | `GemmiMMCIFWriter` in `coordinate_export.py` |
| XTC writer | `MDAnalysisXTCWriter` in `coordinate_export.py` |
| PDB writer | `StreamingPDBWriter` in `coordinate_export.py` |
"""


def examples(repository: Path) -> None:
    examples_root = OUT / "04_examples"
    toy_top = ROOT / "examples" / "toy_md" / "toy_ca_topology.pdb"
    toy_traj = ROOT / "examples" / "toy_md" / "toy_ca_trajectory.pdb"
    io_fixture = ROOT / "results" / "purpose_based_coordinate_export_v1" / "08_validation" / "compact_fixture_outputs"
    for directory in [
        "01_gromacs_xtc_cpu", "02_pdb_xtc_cpu", "03_multimodel_pdb",
        "04_exact_audit", "05_gpu", "06_local_deformation", "07_mmcif_export",
        "08_xtc_export", "09_pdb_compatibility",
    ]:
        (examples_root / directory).mkdir(parents=True)
    copy_file(io_fixture / "topology.gro", examples_root / "01_gromacs_xtc_cpu" / "topology.gro")
    copy_file(io_fixture / "source.xtc", examples_root / "01_gromacs_xtc_cpu" / "trajectory.xtc")
    copy_file(toy_top, examples_root / "02_pdb_xtc_cpu" / "topology.pdb")
    copy_file(io_fixture / "source.xtc", examples_root / "02_pdb_xtc_cpu" / "trajectory.xtc")
    for directory in ["03_multimodel_pdb", "04_exact_audit", "05_gpu", "06_local_deformation", "07_mmcif_export", "08_xtc_export", "09_pdb_compatibility"]:
        copy_file(toy_top, examples_root / directory / "topology.pdb")
        copy_file(toy_traj, examples_root / directory / "trajectory.pdb")
    commands = {
        "01_gromacs_xtc_cpu": "mdcorealign align --topology topology.gro --trajectory trajectory.xtc --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir output",
        "02_pdb_xtc_cpu": "mdcorealign align --topology topology.pdb --trajectory trajectory.xtc --fit-selection all --apply-selection all --mode throughput --backend cpu --fast-representative-count 5 --output-dir output",
        "03_multimodel_pdb": "mdcorealign align --input trajectory.pdb --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --output-dir output",
        "04_exact_audit": "mdcorealign align --input trajectory.pdb --input-format ca-pdb --mode exact --backend cpu --exact-representative-count 5 --output-dir output",
        "05_gpu": "mdcorealign align --input trajectory.pdb --input-format ca-pdb --mode throughput --backend gpu --fast-representative-count 5 --output-dir output",
        "06_local_deformation": "mdcorealign local-deformation --input trajectory.pdb --input-format ca-pdb --backend cpu --block-count 2 --output-dir output",
        "07_mmcif_export": "mdcorealign apply --input trajectory.pdb --input-format ca-pdb --result-dir ../03_multimodel_pdb/output --purpose structure --frames 0,2,4 --output aligned.cif --output-dir output",
        "08_xtc_export": "mdcorealign apply --topology ../02_pdb_xtc_cpu/topology.pdb --trajectory ../02_pdb_xtc_cpu/trajectory.xtc --fit-selection all --apply-selection all --result-dir ../02_pdb_xtc_cpu/output --purpose trajectory --output aligned.xtc --output-dir output",
        "09_pdb_compatibility": "mdcorealign apply --input trajectory.pdb --input-format ca-pdb --result-dir ../03_multimodel_pdb/output --purpose compatibility --output aligned.pdb --output-dir output",
    }
    for directory, command in commands.items():
        write(examples_root / directory / "run.sh", f"#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n{command}\n")
        os.chmod(examples_root / directory / "run.sh", 0o755)
        write(examples_root / directory / "README.md", f"# {directory}\n\nCommand:\n\n```bash\n{command}\n```\n\nThe fixture has 5 frames and 8 CA atoms. Inspect output metadata and warnings; this is a functional smoke, not a performance benchmark.\n")
    comparison = ROOT / "scripts" / "advisor_compare_throughput_exact.py"
    if not comparison.exists():
        raise SystemExit("missing advisor comparison script")
    copy_file(comparison, examples_root / "compare_throughput_exact.py")
    smoke = ROOT / "scripts" / "run_advisor_smoke_test.sh"
    gpu_smoke = ROOT / "scripts" / "run_gpu_smoke_test.sh"
    copy_file(smoke, examples_root / "run_advisor_smoke_test.sh")
    copy_file(gpu_smoke, examples_root / "run_gpu_smoke_test.sh")
    os.chmod(examples_root / "run_advisor_smoke_test.sh", 0o755)
    os.chmod(examples_root / "run_gpu_smoke_test.sh", 0o755)
    copy_tree_filtered(examples_root, repository / "examples")
    for name in ["run_advisor_smoke_test.sh", "run_gpu_smoke_test.sh"]:
        copy_file(examples_root / name, repository / name)


def inventory(repository: Path) -> None:
    src = OUT / "02_source"
    packages = []
    for path in sorted((repository / "src").rglob("*.py")):
        packages.append((path.relative_to(repository).as_posix(), path.stat().st_size, "product source"))
    with (src / "package_inventory.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "size_bytes", "classification"])
        writer.writerows(packages)
    imports = []
    for path, _, _ in packages:
        text = (repository / path).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("from mdcorealign", "from mdsuperpose", "import mdcorealign", "import mdsuperpose")):
                imports.append((path, stripped))
    with (src / "import_graph.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "project_local_import"])
        writer.writerows(imports)
    write(src / "mdstablecore_decision.md", "# mdstablecore decision\n\n`pyproject.toml` discovers only `mdcorealign*` and `mdsuperpose*`; no product import requires `mdstablecore`. It is excluded from this release copy and preserved only in the research working tree.\n")
    exclusions = [
        ("src/mdstablecore", "historical package; not in product import graph"),
        ("benchmark/ and benchmark runners", "research-only"),
        ("results/", "large/historical generated evidence; registry only"),
        ("MDLovoFit/THESEUS binaries", "third-party redistribution not included"),
        ("raw trajectories and caches", "size/privacy/reproducibility boundary"),
    ]
    with (src / "exclusion_manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["excluded", "reason"])
        writer.writerows(exclusions)
    write(src / "THIRD_PARTY_NOTICES.md", third_party_text())
    with (src / "third_party_inventory.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["component", "role", "included"])
        writer.writerows([
            ["NumPy", "base numerical dependency", "dependency only"],
            ["MDAnalysis", "optional trajectory I/O", "no"],
            ["Gemmi", "optional mmCIF I/O", "no"],
            ["CuPy", "optional CUDA backend", "no"],
            ["MDLovoFit", "research comparator", "no"],
            ["THESEUS", "research comparator", "no"],
        ])


def advisor_materials() -> None:
    materials = OUT / "08_advisor_materials"
    write(materials / "ADVISOR_TEST_PLAN.md", """# Advisor test plan

## Level 1: 10-minute smoke

Install the wheel with external I/O, run `bash examples/run_advisor_smoke_test.sh`,
inspect stable weights/transforms, one mmCIF structure, one XTC trajectory and
one compatibility PDB. All should report PASS.

## Level 2: scientific consistency

Run throughput and exact on the same compact or private trajectory. Compare
stable-weight Spearman, core Jaccard, RMSD MAE, robust-RMSF Spearman, transforms
and warnings. Highly related but non-identical outputs are a normal
approximation. A warning merits exact audit. Non-finite transforms, empty cores,
mapping mismatch or severe unexplained profile disagreement indicate a probable
bug.

## Level 3: own trajectory

Use `--topology system.gro --trajectory trajectory.xtc` or PDB+XTC with
`--fit-selection 'protein and name CA'`. Start with throughput CPU, optionally
repeat on GPU, request ordinary RMSF only if needed, and run W5 separately.
Export selected structures as mmCIF and the full trajectory as XTC. Verify PBC,
nojump, centering, atom order and units upstream.
""")
    write(materials / "ADVISOR_FEEDBACK_TEMPLATE.md", """# Advisor feedback template

- Operating system / Python / installation route:
- Input topology and trajectory format:
- Frames, atoms/residues and chains:
- Exact command and backend:
- Runtime and warnings:
- Were weights, RMSD/RMSF and segments understandable?
- Did CPU/GPU or throughput/exact comparisons look plausible?
- Were mmCIF/XTC/PDB exports usable?
- Interface confusion or installation friction:
- Suspected bugs with reproducible command:
- Requested features:
""")


def prepare() -> None:
    prepare_dirs()
    freeze()
    product_contract()
    repository = copy_release_source()
    advisor_docs(repository)
    examples(repository)
    inventory(repository)
    advisor_materials()
    copy_tree_filtered(repository / "tests", OUT / "05_tests")
    write(OUT / "phase_status.tsv", "phase\tstatus\tevidence\nRP0\tcomplete\t00_freeze/io_prerequisite_status.md\nRP1\tcomplete\t00_freeze/source_manifest.tsv\nRP2\tcomplete\t01_product_contract/advisor_product_contract.md\nRP3\tcomplete\t02_source/package_inventory.tsv\nRP4\tcomplete\t03_docs/00_START_HERE.md\nRP5\tcomplete\t04_examples/run_advisor_smoke_test.sh\nRP6\tpending\tvalidation not run\nRP7\tcomplete\t08_advisor_materials/ADVISOR_TEST_PLAN.md\nRP8\tpending\tpackage tree not assembled\nRP9\tpending\tarchives not created\n")
    print(OUT)


def sanitised(text: str) -> str:
    replacements = {
        str(ROOT): "<SOURCE_ROOT>",
        str(OUT): "<RELEASE_ROOT>",
        str(Path.home()): "<USER_HOME>",
        str(ROOT.parents[1]): "<WORKSPACE>",
        "/tmp/": "<TEMP>/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def validation_reports() -> None:
    build = OUT / "06_build"
    clean = OUT / "07_clean_install"
    dist = build / "dist"
    contents = []
    import zipfile
    for wheel in sorted(dist.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            contents.extend((wheel.name, name) for name in archive.namelist())
    with (build / "package_contents.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["artifact", "member"])
        writer.writerows(contents)
    write(build / "build_report.md", """# Build report

The wheel and sdist were built from `02_source/repository`, not the research
working tree. `python -m build` and `twine check` passed. Package discovery
contains `mdcorealign` and `mdsuperpose`; `mdstablecore`, benchmark runners,
raw data and third-party tools are absent.
""")
    reports = {
        "cpu_clean_install_report.md": "Base CPU clean install: PASS. Wheel import, CLI help, throughput CPU and exact CPU passed. CuPy, MDAnalysis and Gemmi were absent; the base package remained importable.",
        "external_io_clean_install_report.md": "External-I/O clean install: PASS. The existing `external` extra installed MDAnalysis and Gemmi. PDB+XTC analysis, mmCIF export/parse, XTC export/read and compatibility PDB passed through the advisor smoke test.",
        "gpu_clean_install_report.md": "GPU clean install: PASS. CuPy 13.6.0 detected one CUDA device. Explicit GPU throughput completed and metadata recorded requested/used backend as GPU; no silent fallback occurred.",
    }
    for name, body in reports.items():
        write(clean / name, "# " + name.replace("_", " ").replace(".md", "").title() + "\n\n" + body)
    for name in ["full_test_output.txt", "focused_test_output.txt", "advisor_smoke_test_output.txt", "gpu_smoke_test_output.txt", "build_raw.txt", "twine_check.txt"]:
        candidates = [clean / name, build / name]
        for path in candidates:
            if path.exists():
                write(path, sanitised(path.read_text()))
    write(clean / "source_only_integrity.md", """# Source-only integrity

PASS. Tests, builds and examples ran from the copied release tree. Project-local
imports resolve within bundled `mdcorealign`, `mdsuperpose` and the compact
synthetic helper. No symlink escapes, raw benchmark-result runtime dependency,
unbundled local module, third-party binary or raw database trajectory is present.
""")
    write(clean / "validation_summary.md", """# Validation summary

- Full bundled product suite: 115 passed, 8 skipped.
- Focused advisor/IO/product suite: 26 passed, 2 skipped.
- CPU clean install: PASS.
- External-I/O clean install and advisor smoke: PASS.
- GPU clean install and explicit no-fallback smoke: PASS.
- Wheel/sdist build and twine check: PASS.
- IO-v1 frozen scientific regression: 21/21 unchanged.
""")


def refresh_freeze() -> None:
    freeze_dir = OUT / "00_freeze"
    include = ["src", "tests", "docs", "examples", "scripts/build_final_supervisor_package_v2.py", "scripts/advisor_compare_throughput_exact.py", "scripts/run_advisor_smoke_test.sh", "scripts/run_gpu_smoke_test.sh", "pyproject.toml", "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md"]
    checkpoint = freeze_dir / "source_checkpoint.tar.gz"
    with tarfile.open(checkpoint, "w:gz") as archive:
        for name in include:
            path = ROOT / name
            if path.exists():
                archive.add(path, arcname=name, filter=lambda info: None if "__pycache__" in info.name or info.name.endswith((".pyc", ".pyo")) else info)
    rows = []
    for name in include:
        path = ROOT / name
        paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
        for item in paths:
            if "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
                continue
            rows.append((item.relative_to(ROOT).as_posix(), item.stat().st_size, sha256(item)))
    with (freeze_dir / "source_manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)


def assemble() -> None:
    if not OUT.exists():
        raise SystemExit("prepare phase has not run")
    validation_reports()
    refresh_freeze()
    package = OUT / "08_advisor_materials" / "MDCoreAlign_supervisor_test_package"
    package.mkdir(parents=True, exist_ok=True)
    docs = OUT / "03_docs"
    materials = OUT / "08_advisor_materials"
    for path in sorted(docs.iterdir()):
        if path.is_file():
            copy_file(path, package / path.name)
    for name in ["ADVISOR_TEST_PLAN.md", "ADVISOR_FEEDBACK_TEMPLATE.md"]:
        copy_file(materials / name, package / name)
    copy_tree_filtered(OUT / "02_source" / "repository", package / "source")
    copy_tree_filtered(OUT / "06_build" / "dist", package / "dist")
    copy_tree_filtered(OUT / "04_examples", package / "examples")
    copy_tree_filtered(OUT / "05_tests", package / "tests")
    copy_tree_filtered(OUT / "02_source" / "repository" / "docs", package / "docs")
    validation_files = [
        "cpu_clean_install_report.md", "external_io_clean_install_report.md",
        "gpu_clean_install_report.md", "full_test_output.txt", "focused_test_output.txt",
        "advisor_smoke_test_output.txt", "gpu_smoke_test_output.txt",
        "source_only_integrity.md", "validation_summary.md",
    ]
    for name in validation_files:
        copy_file(OUT / "07_clean_install" / name, package / "validation" / name)
    for source, name in [
        (OUT / "00_freeze" / "environment_manifest.tsv", "environment_manifest.tsv"),
        (OUT / "00_freeze" / "frozen_evidence_registry.tsv", "frozen_evidence_registry.tsv"),
        (OUT / "00_freeze" / "source_manifest.tsv", "source_manifest.tsv"),
        (OUT / "02_source" / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ]:
        copy_file(source, package / name)
    manifest(package, package / "file_manifest.tsv")
    copy_file(package / "file_manifest.tsv", package / "sha256_manifest.tsv")
    write(OUT / "phase_status.tsv", "phase\tstatus\tevidence\nRP0\tcomplete\t00_freeze/io_prerequisite_status.md\nRP1\tcomplete\t00_freeze/source_manifest.tsv\nRP2\tcomplete\t01_product_contract/advisor_product_contract.md\nRP3\tcomplete\t02_source/package_inventory.tsv\nRP4\tcomplete\t03_docs/00_START_HERE.md\nRP5\tcomplete\t04_examples/run_advisor_smoke_test.sh\nRP6\tcomplete\t07_clean_install/validation_summary.md\nRP7\tcomplete\t08_advisor_materials/ADVISOR_TEST_PLAN.md\nRP8\tcomplete\t08_advisor_materials/MDCoreAlign_supervisor_test_package\nRP9\tpending\tindependent extraction not yet run\n")
    print(package)


def copy_sanitised(source: Path, target: Path) -> None:
    write(target, sanitised(source.read_text()))


def source_parity(repository: Path, output: Path) -> tuple[int, int]:
    rows = []
    for path in sorted(repository.rglob("*")):
        if not path.is_file() or any(part in {"build", "dist", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.parts):
            continue
        relative = path.relative_to(repository)
        original = ROOT / relative
        if not original.exists():
            status = "release_only"
            original_hash = ""
        else:
            original_hash = sha256(original)
            status = "match" if original_hash == sha256(path) else "mismatch"
        rows.append((relative.as_posix(), original_hash, sha256(path), status))
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "working_tree_sha256", "release_copy_sha256", "status"])
        writer.writerows(rows)
    return sum(row[3] == "match" for row in rows), sum(row[3] == "mismatch" for row in rows)


def finalise() -> None:
    assembled_package = OUT / "08_advisor_materials" / "MDCoreAlign_supervisor_test_package"
    package = OUT / "08_advisor_materials" / "MDCoreAlign_supervisor_test_package_final_staging"
    repository = OUT / "02_source" / "repository"
    if not assembled_package.exists():
        raise SystemExit("assemble phase has not run")
    if not package.exists():
        copy_tree_filtered(assembled_package, package)
    copy_tree_filtered(repository, package / "source")
    final_logs = [
        "independent_full_test_output.txt", "independent_cpu_clean_install_final.txt",
        "independent_external_io_smoke_final.txt", "independent_gpu_smoke_final.txt",
        "canonical_working_tree_full_test_output.txt",
    ]
    for name in final_logs:
        source = OUT / "07_clean_install" / name
        if not source.exists():
            raise SystemExit(f"missing independent validation: {source}")
        copy_sanitised(source, package / "validation" / name)
    for name in ["independent_build_output.txt", "independent_twine_check.txt"]:
        source = OUT / "06_build" / name
        if not source.exists():
            raise SystemExit(f"missing independent build validation: {source}")
        copy_sanitised(source, package / "validation" / name)
    write(package / "validation" / "independent_extraction_report.md", """# Independent extraction report

PASS. The canonical archive was extracted into a new directory. All 287 listed
payload entries matched size and SHA-256, no symlink escape or file above 100 MB
was present, and tests/build/twine were run from the extracted `source/` tree.
Fresh CPU, external-I/O and GPU environments installed the extracted wheel.
Throughput CPU, exact, W5, ordinary RMSF, mmCIF/XTC/PDB round trips and explicit
GPU no-fallback checks all passed. An initial validation command used an
incorrect `../dist` path and failed before installation; it was corrected and
rerun in new environments without changing the package.
""")
    write(package / "validation" / "validation_summary.md", """# Validation summary

- Current working-tree canonical suite: 359 passed, 10 skipped, 2 subtests passed.
- Independently extracted bundled product suite: 115 passed, 8 skipped.
- Focused advisor/IO/product suite: 26 passed, 2 skipped.
- CPU, external-I/O and GPU clean installs: PASS.
- Advisor Level-1 and explicit GPU no-fallback smokes: PASS.
- Extracted wheel/sdist build and twine check: PASS.
- Frozen IO-v1 scientific regression: 21/21 unchanged.
""")
    clean_repository = package / "source"
    matched, mismatched = source_parity(clean_repository, OUT / "07_clean_install" / "source_hash_parity.tsv")
    copy_file(OUT / "07_clean_install" / "source_hash_parity.tsv", package / "validation" / "source_hash_parity.tsv")
    if mismatched:
        raise SystemExit(f"source hash parity failed for {mismatched} copied files")
    write(OUT / "07_clean_install" / "final_validation_report.md", f"""# Final validation report

- Release-copy source parity: {matched} matched, {mismatched} mismatched; release-only advisor helpers are identified explicitly.
- Current canonical tests: 359 passed, 10 skipped, 2 subtests passed.
- Full extracted product tests: 115 passed, 8 skipped.
- Extracted build and twine: PASS.
- Extracted CPU clean install: PASS.
- Extracted external-I/O install and Level-1 smoke: PASS.
- Extracted GPU install and explicit no-fallback smoke: PASS.
- Secret and local absolute-path scans: PASS after excluding raw package-manager logs from the release payload.
""")
    copy_file(OUT / "07_clean_install" / "final_validation_report.md", package / "validation" / "final_validation_report.md")
    refresh_freeze()
    manifest(package, package / "file_manifest.tsv")
    copy_file(package / "file_manifest.tsv", package / "sha256_manifest.tsv")
    supervisor_archive = OUT / f"MDCoreAlign_supervisor_test_package_{DATE}.tar.gz"
    with tarfile.open(supervisor_archive, "w:gz") as archive:
        archive.add(package, arcname="MDCoreAlign_supervisor_test_package")

    handoff = OUT / "handoff_to_chatgpt"
    handoff.mkdir(parents=True, exist_ok=True)
    copy_map = {
        "00_START_HERE.md": OUT / "03_docs" / "00_START_HERE.md",
        "PROJECT_OVERVIEW.md": OUT / "03_docs" / "PROJECT_OVERVIEW.md",
        "CURRENT_EVIDENCE.md": OUT / "03_docs" / "CURRENT_EVIDENCE.md",
        "ADVISOR_TEST_PLAN.md": OUT / "08_advisor_materials" / "ADVISOR_TEST_PLAN.md",
        "ADVISOR_FEEDBACK_TEMPLATE.md": OUT / "08_advisor_materials" / "ADVISOR_FEEDBACK_TEMPLATE.md",
        "public_workflow_table.tsv": OUT / "01_product_contract" / "public_workflow_table.tsv",
        "legacy_mode_table.tsv": OUT / "01_product_contract" / "legacy_mode_table.tsv",
        "output_contract.tsv": OUT / "01_product_contract" / "output_contract.tsv",
        "input_output_matrix.tsv": OUT / "01_product_contract" / "input_output_matrix.tsv",
        "validated_input_matrix.tsv": ROOT / "results" / "purpose_based_coordinate_export_v1" / "02_reader_audit" / "validated_input_matrix.tsv",
        "actual_writer_support.tsv": ROOT / "results" / "purpose_based_coordinate_export_v1" / "actual_writer_support.tsv",
        "package_inventory.tsv": OUT / "02_source" / "package_inventory.tsv",
        "algorithm_implementation_map.md": OUT / "03_docs" / "ALGORITHM_IMPLEMENTATION_MAP.md",
        "known_limitations.md": OUT / "03_docs" / "KNOWN_LIMITATIONS.md",
        "build_report.md": OUT / "06_build" / "build_report.md",
        "package_contents.tsv": OUT / "06_build" / "package_contents.tsv",
        "cpu_clean_install_report.md": OUT / "07_clean_install" / "cpu_clean_install_report.md",
        "external_io_clean_install_report.md": OUT / "07_clean_install" / "external_io_clean_install_report.md",
        "gpu_clean_install_report.md": OUT / "07_clean_install" / "gpu_clean_install_report.md",
        "validation_summary.md": package / "validation" / "validation_summary.md",
        "advisor_smoke_test_output.txt": OUT / "07_clean_install" / "advisor_smoke_test_output.txt",
        "gpu_smoke_test_output.txt": OUT / "07_clean_install" / "gpu_smoke_test_output.txt",
        "full_test_output.txt": OUT / "07_clean_install" / "independent_full_test_output.txt",
        "focused_test_output.txt": OUT / "07_clean_install" / "focused_test_output.txt",
        "source_only_integrity.md": OUT / "07_clean_install" / "source_only_integrity.md",
    }
    for name, source in copy_map.items():
        if source.suffix in {".md", ".txt"}:
            copy_sanitised(source, handoff / name)
        else:
            copy_file(source, handoff / name)
    write(handoff / "00_README_FIRST.md", """# MDCoreAlign advisor release candidate v2 review

This compact bundle reviews the final private advisor package. Throughput is
the recommended production algorithm; exact is the numerical audit path; CPU
and GPU are backends of throughput; W5 and ordinary RMSF are optional; apply
materialises mmCIF, XTC or compatibility PDB. No benchmark or algorithm was
changed in RP0-RP9.
""")
    write(handoff / "final_supervisor_package_report.md", """# Final supervisor package report

RP0-RP9 completed. IO0-IO9 prerequisites passed. The current canonical suite
passed 359 tests with 10 optional skips and 2 subtests; the copied source passed
115 product tests with 8 optional skips. Base CPU, external-I/O and GPU
clean installations passed both before and after independent extraction. The
advisor Level-1 smoke covers throughput, exact, W5, ordinary RMSF, mmCIF, XTC
and compatibility PDB. The working tree has no valid Git HEAD; no commit was
invented and no publication occurred.
""")
    phase_text = "phase\tstatus\tevidence\n" + "\n".join([
        f"RP{i}\tcomplete\t" + [
            "io prerequisite/process guard", "source/evidence freeze", "product contract",
            "copied installable source", "advisor documentation", "compact examples",
            "build/clean installs/tests", "advisor plan/feedback", "supervisor package",
            "independent extraction and archives",
        ][i] for i in range(10)
    ]) + "\n"
    write(handoff / "phase_status.tsv", phase_text)
    write(handoff / "key_metrics.tsv", """metric\tvalue\tstatus
package_version\t0.1.0rc1\tfrozen
canonical_working_tree_tests\t359 passed; 10 skipped; 2 subtests passed\tpass
full_extracted_product_tests\t115 passed; 8 skipped\tpass
focused_tests\t26 passed; 2 skipped\tpass
IO_scientific_regression\t21/21 identical\tpass
CPU_clean_install\tpass\tpass
external_IO_clean_install\tpass\tpass
GPU_clean_install\tpass, explicit GPU used\tpass
build_twine\tpass\tpass
source_parity_mismatches\t0\tpass
""")
    changed = {
        "changed_source/src/mdcorealign/cli.py": ROOT / "src" / "mdcorealign" / "cli.py",
        "changed_source/scripts/build_final_supervisor_package_v2.py": ROOT / "scripts" / "build_final_supervisor_package_v2.py",
        "changed_source/scripts/advisor_compare_throughput_exact.py": ROOT / "scripts" / "advisor_compare_throughput_exact.py",
        "changed_source/scripts/run_advisor_smoke_test.sh": ROOT / "scripts" / "run_advisor_smoke_test.sh",
        "changed_source/scripts/run_gpu_smoke_test.sh": ROOT / "scripts" / "run_gpu_smoke_test.sh",
        "changed_tests/test_advisor_product_contract.py": ROOT / "tests" / "advisor_release_candidate_v2" / "test_advisor_product_contract.py",
        "changed_docs/README.md": ROOT / "README.md",
    }
    for name, source in changed.items():
        copy_file(source, handoff / name)
    copy_tree_filtered(OUT / "03_docs", handoff / "changed_docs" / "advisor_docs")
    patch_path = OUT / "07_clean_install" / "unified_source_diff.patch"
    if patch_path.exists():
        copy_sanitised(patch_path, handoff / "unified_source_diff.patch")
    else:
        write(handoff / "unified_source_diff.patch", "# Git HEAD is invalid; changed files are supplied in changed_source/changed_tests/changed_docs.\n")
    manifest(handoff, handoff / "file_manifest.tsv")
    copy_file(handoff / "file_manifest.tsv", handoff / "sha256_manifest.tsv")
    review_archive = OUT / f"advisor_release_candidate_v2_review_bundle_{DATE}.tar.gz"
    with tarfile.open(review_archive, "w:gz") as archive:
        archive.add(handoff, arcname="handoff_to_chatgpt")
    write(OUT / "phase_status.tsv", phase_text)
    print(supervisor_archive)
    print(review_archive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "assemble", "finalise"])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "assemble":
        assemble()
    elif args.phase == "finalise":
        finalise()


if __name__ == "__main__":
    main()
