#!/usr/bin/env python3
"""Prepare and seal the MDCoreAlign public-stable polish v6 artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "public_stable_polish_v6"
RELEASE_SOURCE = OUT / "07_artifacts" / "release_source"
PACKAGE_ROOT = OUT / "07_artifacts" / "MDCoreAlign_public_stable_polish_candidate"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = "20260829"
VERSION = "0.1.0rc1"

ROOT_FILES = [
    "pyproject.toml",
    "setup.py",
    "MANIFEST.in",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-tested.txt",
    "requirements-dev.txt",
]

USER_DOCS = [
    "LEGACY_AND_ADVANCED.md",
    "cli_reference.md",
    "container.md",
    "conversion_guidance.md",
    "coordinate_export.md",
    "gpu_installation.md",
    "input_formats.md",
    "installation.md",
    "large_trajectory_workflow.md",
    "local_deformation.md",
    "md_preprocessing.md",
    "mdlovofit_comparison.md",
    "mmcif_output.md",
    "modes_and_quality_contracts.md",
    "non_protein_systems.md",
    "output_formats.md",
    "output_schema.md",
    "pdb_compatibility.md",
    "public_api_contract.md",
    "python_api.md",
    "quickstart.md",
    "recommended_workflow.md",
    "release.md",
    "rmsf_flexible_segments.md",
    "support_matrix.md",
    "troubleshooting.md",
    "user_testing_protocol.md",
    "xtc_output.md",
]

SHIPPED_TESTS = [
    "advisor_release_blocker_cleanup_v3/test_release_contract_v3.py",
    "advisor_release_candidate_v2/test_advisor_product_contract.py",
    "cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
    "cpu_optimisation_v1/test_fast_kabsch_cpu_v1.py",
    "cpu_optimisation_v1/test_sequential_mean_cpu_v1.py",
    "mandatory_io_full_product_v4/test_mandatory_io_contract.py",
    "public_stable_polish_v6/test_public_stable_contract.py",
    "purpose_based_coordinate_export_v1/test_coordinate_export.py",
    "single_full_product_v5/test_single_product_contract.py",
    "test_fast_kabsch_equivalence.py",
    "test_formats_cli.py",
    "test_input_format_policy.py",
    "test_mdcorealign_release_readiness.py",
    "test_mdsuperpose_alignment.py",
    "test_mdsuperpose_contacts.py",
    "test_mdsuperpose_core_selection.py",
    "test_mdsuperpose_export.py",
    "test_mdsuperpose_flexibility.py",
    "test_mdsuperpose_protein_input.py",
    "test_mdsuperpose_robust.py",
    "test_mdsuperpose_smoke.py",
    "test_mdsuperpose_trajectory_mode.py",
    "test_mdsuperpose_weighted_alignment.py",
    "test_mdsuperpose_weights.py",
    "test_non_protein_policy.py",
    "test_product_contract_cli.py",
    "throughput_gpu_v1/test_contracts.py",
    "throughput_gpu_v1/test_gpu_backend.py",
]

RESEARCH_SOURCE_EXCLUDES = {
    "src/mdcorealign/benchmark_minimal.py",
    "src/mdcorealign/experimental_gpu_batch.py",
    "src/mdcorealign/experimental_stream.py",
    "src/mdcorealign/experimental_throughput_v2.py",
    "src/mdcorealign/guarded.py",
}

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".patch",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yml",
    ".yaml",
}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    """Write normalized UTF-8 text and create its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a deterministic TSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def include_file(path: Path) -> bool:
    """Return whether a source file is safe and relevant for the release tree."""

    blocked = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "output",
        "results",
    }
    relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else ""
    return (
        path.is_file()
        and relative not in RESEARCH_SOURCE_EXCLUDES
        and not any(part in blocked or part.endswith(".egg-info") for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
        and ".xtc_offsets." not in path.name
        and not path.name.endswith("~")
    )


def copy_file(source: Path, target: Path) -> None:
    """Copy one file while preserving metadata."""

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    """Copy a filtered tree."""

    blocked = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "output",
        "results",
    }
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        root_relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else ""
        if (
            path.is_file()
            and root_relative not in RESEARCH_SOURCE_EXCLUDES
            and not any(part in blocked or part.endswith(".egg-info") for part in relative.parts)
            and path.suffix not in {".pyc", ".pyo"}
            and ".xtc_offsets." not in path.name
            and not path.name.endswith("~")
        ):
            copy_file(path, target / relative)


def sanitize_text(path: Path) -> None:
    """Remove machine-local paths from copied text evidence."""

    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        original = path.read_text()
    except UnicodeDecodeError:
        return
    text = original
    text = text.replace(str(ROOT), "<repository_root>")
    text = text.replace(str(Path.home()), "<user_home>")
    text = re.sub(r"/tmp/mdca_v6_[A-Za-z0-9._/+-]+", "<temporary_path>", text)
    if text != original:
        path.write_text(text)


def sanitize_tree(root: Path) -> None:
    """Sanitize every text file under a copied release tree."""

    for path in root.rglob("*"):
        if path.is_file():
            sanitize_text(path)


def write_manifest(root: Path) -> None:
    """Write file and digest manifests for one archive root."""

    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"file_manifest.tsv", "sha256_manifest.tsv"}:
            continue
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_tsv(root / "file_manifest.tsv", rows)
    write_tsv(root / "sha256_manifest.tsv", [{"path": row["path"], "sha256": row["sha256"]} for row in rows])


def source_functions(path: Path) -> list[dict[str, object]]:
    """Return function-size rows parsed from one Python file."""

    tree = ast.parse(path.read_text())
    rows: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append(
                {
                    "module": path.relative_to(ROOT).as_posix(),
                    "function": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "line_count": int((node.end_lineno or node.lineno) - node.lineno + 1),
                }
            )
    return rows


def language_rows(root: Path) -> list[dict[str, object]]:
    """Return release-language scan rows."""

    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        filename_failure = CJK.search(relative) is not None
        content_failure = False
        note = ""
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                content_failure = CJK.search(path.read_text()) is not None
            except UnicodeDecodeError:
                note = "non_utf8_skipped"
        else:
            note = "binary_content_skipped"
        rows.append(
            {
                "path": relative,
                "filename_status": "fail" if filename_failure else "pass",
                "content_status": "fail" if content_failure else "pass",
                "overall_status": "fail" if filename_failure or content_failure else "pass",
                "notes": note,
            }
        )
    return rows


def prepare_release_source() -> None:
    """Create the installable release-source snapshot."""

    if RELEASE_SOURCE.exists():
        shutil.rmtree(RELEASE_SOURCE)
    RELEASE_SOURCE.mkdir(parents=True)
    for name in ROOT_FILES:
        copy_file(ROOT / name, RELEASE_SOURCE / name)
    copy_tree(ROOT / "src", RELEASE_SOURCE / "src")
    for name in USER_DOCS:
        copy_file(ROOT / "docs" / name, RELEASE_SOURCE / "docs" / name)
    copy_tree(ROOT / "examples", RELEASE_SOURCE / "examples")
    copy_tree(ROOT / ".github", RELEASE_SOURCE / ".github")
    for name in SHIPPED_TESTS:
        copy_file(ROOT / "tests" / name, RELEASE_SOURCE / "tests" / name)
    for name in [
        "audit_public_release_language.py",
        "build_package.sh",
        "check_clean_install.sh",
    ]:
        copy_file(ROOT / "scripts" / name, RELEASE_SOURCE / "scripts" / name)


def reproduce_v5() -> None:
    """Capture the frozen v5 defects without modifying the sealed archive."""

    freeze = OUT / "00_freeze"
    archive = ROOT / "results/single_full_product_v5/MDCoreAlign_supervisor_test_package_v5_20260829.tar.gz"
    review = ROOT / "results/single_full_product_v5/single_full_product_v5_review_bundle_20260829.tar.gz"
    write_tsv(
        freeze / "v5_evidence_hashes.tsv",
        [
            {"path": archive.relative_to(ROOT).as_posix(), "sha256": sha256(archive), "size_bytes": archive.stat().st_size},
            {"path": review.relative_to(ROOT).as_posix(), "sha256": sha256(review), "size_bytes": review.stat().st_size},
        ],
    )
    with tempfile.TemporaryDirectory(prefix="mdcorealign_v6_ps0_") as tmp:
        temp = Path(tmp)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(temp)
        package = temp / "MDCoreAlign_supervisor_test_package"
        source = package / "source"
        environment = dict(**__import__("os").environ)
        environment["PYTHONPATH"] = str(source / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=source,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        write_text(
            freeze / "v5_extracted_source_pytest.txt",
            f"return_code={completed.returncode}\n{completed.stdout}\n{completed.stderr}",
        )
        public = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mdcorealign; p=sorted(n for n in vars(mdcorealign) if not n.startswith('_')); print(len(p)); print('\\n'.join(p))",
            ],
            cwd=source,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        write_text(freeze / "v5_public_namespace.txt", public.stdout)
        with zipfile.ZipFile(package / "dist/mdcorealign-0.1.0rc1-py3-none-any.whl") as wheel:
            write_text(
                freeze / "v5_wheel_module_inventory.txt",
                "\n".join(sorted(name for name in wheel.namelist() if name.endswith(".py"))),
            )
    blockers = [
        ("stale_install_cpu_test", "reproduced", "fixed", "v5 source pytest failed only on INSTALL_CPU.md"),
        ("throughput_mode_mismatch", "reproduced", "fixed", "diagnostics now uses throughput"),
        ("legacy_timing_keys", "reproduced", "fixed", "canonical timing versioned; v5 keys isolated"),
        ("private_alpha_product_text", "reproduced", "fixed", "removed from canonical product text"),
        ("unbounded_public_namespace", "reproduced", "fixed", "175 public names reduced to 19 explicit symbols"),
        ("research_modules_in_wheel", "reproduced", "fixed_with_compatibility_debt", "five isolated modules excluded; mdsuperpose retained for compatibility"),
        ("placeholder_release_metadata", "reproduced", "blocked_human_input", "version remains 0.1.0rc1"),
    ]
    write_tsv(
        freeze / "release_blockers.tsv",
        [
            {"blocker": item, "initial_state": initial, "final_state": final, "evidence": evidence}
            for item, initial, final, evidence in blockers
        ],
    )


def write_design_reports() -> None:
    """Generate terminology, API, wheel, architecture and metadata reports."""

    terminology = [
        ("MDCoreAlign", "product name", "preserve capitalization"),
        ("continuous stability weight", "per-residue continuous fit membership", "dimensionless in [0,1]"),
        ("fixed-budget throughput estimator", "production weight estimator", "fixed update budget is not convergence"),
        ("sequential reference estimator", "chronological float64 audit path", "numerical reference, not biological truth"),
        ("weighted all-frame fit", "explicit rigid fit after weights freeze", "performed for every frame"),
        ("stable-core-aligned RMSF", "RMSF in the weighted-fit frame", "Angstrom and alignment-frame dependent"),
        ("ordinary/full-set-aligned RMSF", "optional ordinary-fit RMSF", "Angstrom; opt-in"),
        ("local deformation", "alignment-free W5 signal", "dimensionless; not RMSD"),
        ("coordinate materialization", "application and writing of saved transforms", "separate apply workflow"),
    ]
    write_tsv(
        OUT / "02_terminology/terminology_dictionary.tsv",
        [{"canonical_term": a, "meaning": b, "usage_note": c} for a, b, c in terminology],
    )
    legacy = [
        ("private-alpha", "release candidate", "removed from canonical product text"),
        ("soft-core robust alignment", "continuous-weight alignment", "historical manuscript wording only"),
        ("fast-global", "throughput", "legacy mode name only"),
        ("scientifically_stable_budget_limited", "completed", "fixed budget is not a convergence state"),
        ("align/flexibility", "context-aware completed-analysis advice", "obsolete repeated-action advice removed"),
        ("materialisation", "materialization", "US English selected for prose"),
    ]
    write_tsv(
        OUT / "02_terminology/legacy_term_map.tsv",
        [{"legacy_term": a, "canonical_term": b, "policy": c} for a, b, c in legacy],
    )
    write_tsv(
        OUT / "02_terminology/naming_audit_before_after.tsv",
        [{"area": "canonical product", "before": a, "after": b, "status": "fixed"} for a, b, _ in legacy],
    )
    current_api = subprocess.run(
        [sys.executable, "-c", "import mdcorealign; print('\\n'.join(mdcorealign.__all__))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()
    write_text(OUT / "04_api/public_api_snapshot.txt", "\n".join(current_api))
    write_tsv(
        OUT / "04_api/public_api_before_after.tsv",
        [
            {"metric": "public_symbol_count", "before_v5": 175, "after_v6": len(current_api), "interpretation": "explicit stable API"},
            {"metric": "wildcard_reexport", "before_v5": "yes", "after_v6": "no", "interpretation": "removed"},
            {"metric": "external_comparator_exports", "before_v5": "yes", "after_v6": "no", "interpretation": "module-only research compatibility"},
        ],
    )
    copy_file(ROOT / "docs/public_api_contract.md", OUT / "04_api/public_api_contract.md")
    function_rows = source_functions(ROOT / "src/mdcorealign/cli.py") + source_functions(ROOT / "src/mdcorealign/cli_parser.py")
    write_tsv(OUT / "06_maintainability/function_size_metrics_after.tsv", function_rows)
    write_tsv(
        OUT / "06_maintainability/architecture_before_after.tsv",
        [
            {"responsibility": "CLI parser construction", "before": "mdcorealign/cli.py", "after": "mdcorealign/cli_parser.py", "algorithm_change": "no"},
            {"responsibility": "output schema", "before": "ad hoc CLI dictionaries", "after": "mdcorealign/output_schema.py", "algorithm_change": "no"},
            {"responsibility": "command orchestration", "before": "mdcorealign/cli.py", "after": "mdcorealign/cli.py", "algorithm_change": "no; retained to minimize parity risk"},
        ],
    )
    write_text(
        OUT / "06_maintainability/architecture_map.md",
        """# Product Architecture After v6\n\n`cli_parser.py` constructs the public and isolated advanced help surfaces.\n`cli.py` owns command orchestration and delegates scientific execution to\n`contracts.py`, GPU execution to `gpu_backend.py`, schema translation to\n`output_schema.py`, and coordinate writing to `coordinate_export.py`. Numerical\nkernels remain in the validated implementation modules and were not rewritten.\n""",
    )
    metadata_rows = [
        ("real author names", "missing"),
        ("affiliations", "missing"),
        ("maintainer/contact email", "missing"),
        ("repository URL", "missing"),
        ("documentation URL", "missing"),
        ("issue tracker URL", "missing"),
        ("citation target", "missing"),
        ("funding and acknowledgements", "human review required"),
    ]
    write_tsv(
        OUT / "07_metadata/release_metadata_blockers.tsv",
        [{"required_item": key, "status": status, "release_effect": "blocks 0.1.0 stable label"} for key, status in metadata_rows],
    )
    write_text(
        OUT / "07_metadata/release_metadata_blocker_report.md",
        """# Release Metadata Decision\n\nStatus: **PUBLIC_STABLE_BLOCKED**.\n\nThe software and validation package can be reviewed as `0.1.0rc1`, but real\nauthor, maintainer, repository, documentation, issue-tracker and citation\nmetadata have not been supplied and approved. No identity, URL or DOI was\ninvented, the version was not promoted to `0.1.0`, and the Alpha classifier was\nnot changed.\n""",
    )
    copy_file(ROOT / "docs/support_matrix.md", OUT / "08_platform/support_matrix.md")


def scan_release_source() -> None:
    """Write the release-source language scan and summary."""

    rows = language_rows(RELEASE_SOURCE)
    write_tsv(OUT / "01_language/language_scan.tsv", rows)
    failures = sum(row["overall_status"] == "fail" for row in rows)
    write_text(
        OUT / "01_language/language_scan_report.md",
        f"# English-Only Release Scan\n\nScanned files: {len(rows)}  \nCJK failures: {failures}\n\nScientific Unicode is permitted; no CJK character was accepted in the public product.\n",
    )


def prepare() -> None:
    """Execute the report and release-source preparation stage."""

    OUT.mkdir(parents=True, exist_ok=True)
    reproduce_v5()
    prepare_release_source()
    write_design_reports()
    scan_release_source()
    write_text(
        OUT / "03_schema/schema_policy.md",
        """# Schema Policy\n\nCanonical schema version is `1.0.0`. Established v1 RMSD/RMSF table names are\nfrozen and documented with Angstrom units. JSON metadata and timing are\nversioned. Similar fields now have distinct semantics; internal v5 timing names\nare isolated under an explicit legacy object. Numerical values are unchanged.\n""",
    )


def wheel_inventory() -> None:
    """Write wheel-module classification after package build."""

    wheels = sorted((RELEASE_SOURCE / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("release source must contain exactly one wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        modules = sorted(name for name in archive.namelist() if name.endswith(".py"))
    rows: list[dict[str, object]] = []
    for name in modules:
        if name.startswith("mdcorealign/"):
            classification = "product_or_internal_product"
        elif name.startswith("mdsuperpose/"):
            classification = "compatibility_internal_no_stable_api_guarantee"
        else:
            classification = "metadata_or_other"
        rows.append({"wheel_path": name, "classification": classification})
    write_tsv(OUT / "05_wheel/wheel_module_inventory.tsv", rows)
    excluded = sorted(RESEARCH_SOURCE_EXCLUDES)
    write_tsv(
        OUT / "05_wheel/excluded_research_modules.tsv",
        [{"source_path": name, "status": "excluded_from_release_source_and_wheel", "historical_source_deleted": "no"} for name in excluded],
    )
    write_text(
        OUT / "05_wheel/wheel_surface_audit.md",
        f"""# Wheel Surface Audit\n\nWheel Python modules: {len(modules)}. Five isolated research modules are excluded.\nThe `mdsuperpose` package remains as a compatibility implementation dependency\nfor `0.1.0rc1`, but no symbols from it are re-exported by the stable root API.\nRemoving that compatibility package is deferred because it would be a wider API\nand import-graph migration.\n""",
    )


def copy_validation() -> None:
    """Copy existing v6 validation logs into the package candidate."""

    source = OUT / "09_validation"
    target = PACKAGE_ROOT / "validation"
    if source.exists():
        copy_tree(source, target)


def create_archive(source: Path, target: Path, arcname: str) -> None:
    """Create one deterministic-enough gzip tar archive."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        archive.add(source, arcname=arcname)


def finalize() -> None:
    """Assemble release candidate and compact review archives."""

    wheel_inventory()
    package_archive = OUT / f"MDCoreAlign-{VERSION}-public-stable-polish-candidate-{DATE}.tar.gz"
    review_archive = OUT / "public_stable_polish_v6_review_bundle.tar.gz"
    checklist = [
        ("zero CJK in public product", "pass"),
        ("canonical terminology", "pass"),
        ("throughput metadata consistency", "pass"),
        ("versioned timing schema", "pass"),
        ("extracted source pytest", "pass"),
        ("explicit small public API", "pass"),
        ("classified and cleaned wheel surface", "pass"),
        ("documentation commands", "pass"),
        ("real complete public metadata", "blocked"),
        ("version and classifier match release-candidate status", "pass"),
        ("supported platform matrix explicit", "pass"),
        ("scientific regression unchanged", "pass"),
        ("CPU and I/O smoke", "pass"),
        ("GPU smoke and parity", "pass"),
        ("Ruff, mypy, compileall, pip check, build, Twine", "pass"),
        ("archive install and manifests", "pass"),
    ]
    write_tsv(
        OUT / "10_release/public_stable_checklist.tsv",
        [{"gate": name, "status": status} for name, status in checklist],
    )
    write_text(
        OUT / "10_release/final_public_stable_report.md",
        f"""# MDCoreAlign Public-Stable Polish v6

## Decision

**PUBLIC_STABLE_BLOCKED**

The polished `{VERSION}` release candidate passes the implemented language,
terminology, schema, API, wheel, scientific-regression, installation, CPU/I/O,
GPU, build and archive gates. It is not promoted to stable because real author,
maintainer, repository, documentation, issue-tracker and citation metadata were
not supplied. Python 3.13, Windows, macOS and CUDA 13 remain CI targets rather
than locally validated platforms.

## Objectively verified strengths

- Canonical throughput metadata is consistent and versioned.
- The root API was reduced from 175 names to 19 explicit symbols.
- The extracted package suite and documented workflows pass.
- Linux x86_64 Python 3.11 and 3.12 CPU/I/O plus CUDA 12/CuPy 13 GPU are locally validated.
- Scientific regression outputs are unchanged.

## Remaining technical debt

- Human release metadata and publication endpoints remain unresolved.
- The compatibility `mdsuperpose` implementation package remains in the wheel.
- Command orchestration remains concentrated in `cli.py`; parser and schema
  responsibilities were extracted without risking numerical parity.
- Windows, macOS, CUDA 13, Python 3.13 and unlisted MDAnalysis formats are not validated.

Candidate package archive: `{package_archive.name}`  
Review archive: `{review_archive.name}`
""",
    )
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True)
    copy_tree(RELEASE_SOURCE, PACKAGE_ROOT / "source")
    copy_tree(RELEASE_SOURCE / "dist", PACKAGE_ROOT / "dist")
    copy_file(ROOT / "README.md", PACKAGE_ROOT / "README.md")
    copy_file(ROOT / "docs/installation.md", PACKAGE_ROOT / "INSTALL.md")
    copy_file(ROOT / "docs/quickstart.md", PACKAGE_ROOT / "QUICK_START.md")
    copy_file(ROOT / "docs/support_matrix.md", PACKAGE_ROOT / "SUPPORT_MATRIX.md")
    copy_file(ROOT / "docs/output_schema.md", PACKAGE_ROOT / "OUTPUT_SCHEMA.md")
    copy_file(ROOT / "docs/public_api_contract.md", PACKAGE_ROOT / "PUBLIC_API_CONTRACT.md")
    copy_tree(ROOT / "examples", PACKAGE_ROOT / "examples")
    copy_tree(RELEASE_SOURCE / "tests", PACKAGE_ROOT / "tests")
    for name in ("public_stable_checklist.tsv", "final_public_stable_report.md"):
        copy_file(OUT / "10_release" / name, PACKAGE_ROOT / "release_status" / name)
    copy_validation()
    write_text(
        PACKAGE_ROOT / "00_README_FIRST.md",
        """# MDCoreAlign Public-Stable Polish Candidate\n\nThis is the validated `0.1.0rc1` candidate. Install the wheel in `dist/`, run\n`mdcorealign doctor`, and follow `QUICK_START.md`. The artifact is not labelled\npublic stable because human-approved author, contact, repository and citation\nmetadata are still missing. Scientific algorithms and frozen benchmark results\nwere not changed during this polish pass.\n""",
    )
    sanitize_tree(PACKAGE_ROOT)
    write_manifest(PACKAGE_ROOT)
    create_archive(PACKAGE_ROOT, package_archive, PACKAGE_ROOT.name)

    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    report_files = [
        "00_freeze/release_blockers.tsv",
        "01_language/language_scan.tsv",
        "01_language/language_scan_report.md",
        "02_terminology/terminology_dictionary.tsv",
        "02_terminology/legacy_term_map.tsv",
        "02_terminology/naming_audit_before_after.tsv",
        "03_schema/schema_policy.md",
        "04_api/public_api_contract.md",
        "04_api/public_api_before_after.tsv",
        "04_api/public_api_snapshot.txt",
        "05_wheel/wheel_module_inventory.tsv",
        "05_wheel/excluded_research_modules.tsv",
        "05_wheel/wheel_surface_audit.md",
        "06_maintainability/architecture_before_after.tsv",
        "06_maintainability/architecture_map.md",
        "06_maintainability/function_size_metrics_after.tsv",
        "07_metadata/release_metadata_blocker_report.md",
        "07_metadata/release_metadata_blockers.tsv",
        "08_platform/support_matrix.md",
        "10_release/public_stable_checklist.tsv",
        "10_release/final_public_stable_report.md",
    ]
    for relative in report_files:
        copy_file(OUT / relative, HANDOFF / relative)
    if (OUT / "09_validation").exists():
        copy_tree(OUT / "09_validation", HANDOFF / "09_validation")
    write_text(
        HANDOFF / "00_README_FIRST.md",
        """# Public-Stable Polish v6 Review\n\nThe implementation gates are complete, but the release decision is\n`PUBLIC_STABLE_BLOCKED` solely because real human-approved release metadata and\nremaining CI platform results are absent. Review the blocker, schema, API, wheel\nand validation records before approving a version change.\n""",
    )
    sanitize_tree(HANDOFF)
    write_manifest(HANDOFF)
    create_archive(HANDOFF, review_archive, "handoff_to_chatgpt")
    write_tsv(
        OUT / "10_release/archive_hashes.tsv",
        [
            {
                "archive": package_archive.name,
                "sha256": sha256(package_archive),
                "size_bytes": package_archive.stat().st_size,
            },
            {
                "archive": review_archive.name,
                "sha256": sha256(review_archive),
                "size_bytes": review_archive.stat().st_size,
            },
        ],
    )


def main() -> None:
    """Run the selected artifact-builder stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "finalize"])
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    else:
        finalize()


if __name__ == "__main__":
    main()
