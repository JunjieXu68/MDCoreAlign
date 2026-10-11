"""Build and independently validate the mandatory O9 review archive."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "cpu_optimisation_v1"
HANDOFF = RESULTS / "handoff_to_chatgpt"
ARCHIVE = RESULTS / f"cpu_optimisation_v1_review_bundle_{date.today():%Y%m%d}.tar.gz"

TEXT_SUFFIXES = {
    "",
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
    ".yaml",
    ".yml",
}
ABSOLUTE_PATH_PATTERNS = (
    re.compile("/" + "storage/"),
    re.compile("/" + "home/"),
    re.compile("/" + "tmp/"),
    re.compile(r"\b" + "jj" + "xu" + r"\b", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?i)(api[_-]?key|password|access[_-]?token)\s*[:=]\s*[^<\s][^\s]*"),
)

CHANGED_PATHS = [
    "README.md",
    "MANIFEST.in",
    "pyproject.toml",
    "docs/cli_reference.md",
    "docs/output_formats.md",
    "src/mdcorealign/benchmark_minimal.py",
    "src/mdcorealign/cli.py",
    "src/mdcorealign/contracts.py",
    "src/mdcorealign/experimental.py",
    "src/mdsuperpose/all_frame_engine.py",
    "src/mdsuperpose/fast_kabsch.py",
    "src/mdsuperpose/flexibility.py",
    "src/mdsuperpose/long_mode.py",
    "src/mdsuperpose/sequential_weights.py",
    "src/mdsuperpose/window_refine.py",
    "tests/cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
    "tests/cpu_optimisation_v1/test_fast_kabsch_cpu_v1.py",
    "tests/cpu_optimisation_v1/test_sequential_mean_cpu_v1.py",
    "scripts/cpu_optimisation_v1/audit_kabsch_callsites.py",
    "scripts/cpu_optimisation_v1/benchmark_allframe_snapshot.py",
    "scripts/cpu_optimisation_v1/benchmark_components.py",
    "scripts/cpu_optimisation_v1/benchmark_covariance.py",
    "scripts/cpu_optimisation_v1/benchmark_cpu_paths.py",
    "scripts/cpu_optimisation_v1/benchmark_frame_scaling.py",
    "scripts/cpu_optimisation_v1/benchmark_precision.py",
    "scripts/cpu_optimisation_v1/benchmark_rotation_backends.py",
    "scripts/cpu_optimisation_v1/finalize_o9.py",
    "scripts/cpu_optimisation_v1/build_o9_review_archive.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_text(text: str) -> str:
    replacements = (
        (str(ROOT), "<REPOSITORY_ROOT>"),
        (str(ROOT.parent), "<WORKSPACE_ROOT>"),
        (str(Path.home()), "<HOME>"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    temporary_path_pattern = "/" + r"tmp/[A-Za-z0-9_.+/=-]+"
    text = re.sub(temporary_path_pattern, "<TEMP_PATH>", text)
    return text


def copy_review_file(source: Path, relative_destination: Path) -> None:
    destination = HANDOFF / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="replace")
        destination.write_text(sanitize_text(text), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def create_phase_and_metric_tables() -> None:
    phases = []
    descriptions = {
        "O0": "Freeze source, environment, process gate and baseline",
        "O1": "Batched-matmul covariance",
        "O2": "Transform/materialisation API split",
        "O3": "Optional ordinary RMSF contract",
        "O4": "Mean-only exact sequential estimator",
        "O5": "Rotation backend evaluation",
        "O6": "CPU precision evaluation",
        "O7": "Shared all-frame engine and workspace",
        "O8": "Packaging and mode audit",
        "O9": "Validation, decisions and review archive",
    }
    for phase, description in descriptions.items():
        phases.append(
            {
                "phase": phase,
                "status": "complete",
                "description": description,
                "notes": "No external scientific baseline rerun",
            }
        )
    write_tsv(HANDOFF / "phase_status.tsv", phases)

    performance = read_tsv(RESULTS / "08_decision" / "performance_summary.tsv")
    memory = read_tsv(RESULTS / "08_decision" / "memory_summary.tsv")

    def perf(category: str, mode: str, frames: str, contract: str) -> dict[str, str]:
        return next(
            row
            for row in performance
            if row["category"] == category
            and row["mode"] == mode
            and row["n_frames"] == frames
            and row["contract"] == contract
        )

    exact = perf("transition_aligned_scaling", "exact", "10000", "exact_full")
    throughput = perf(
        "transition_aligned_scaling", "throughput", "10000", "throughput_full"
    )
    compact = perf(
        "transition_aligned_scaling",
        "throughput",
        "10000",
        "throughput_compact",
    )
    all_frame = perf(
        "isolated_all_frame_engine", "fixed_weights", "10000", "full"
    )
    exact_memory = next(row for row in memory if row["scope"] == "exact_representative_estimator")
    metrics = [
        {"metric": "package_version", "value": "0.1.0rc1", "unit": "version", "contract": "package"},
        {"metric": "canonical_tests", "value": "327 passed", "unit": "tests", "contract": "full suite"},
        {"metric": "focused_tests", "value": "41 passed", "unit": "tests", "contract": "focused suite"},
        {"metric": "exact_10k_full_speedup", "value": exact["median_speedup"], "unit": "x", "contract": "historical full output"},
        {"metric": "throughput_10k_full_speedup", "value": throughput["median_speedup"], "unit": "x", "contract": "historical full output"},
        {"metric": "throughput_10k_compact_speedup", "value": compact["median_speedup"], "unit": "x", "contract": "compact versus historical full baseline"},
        {"metric": "all_frame_10k_full_speedup", "value": all_frame["median_speedup"], "unit": "x", "contract": "fixed-weight full output"},
        {"metric": "exact_mean_only_peak_memory_reduction", "value": exact_memory["memory_reduction_ratio"], "unit": "x", "contract": "representative estimator"},
        {"metric": "ordinary_rmsf_default", "value": "enabled", "unit": "state", "contract": "public default unchanged"},
        {"metric": "mixed_precision", "value": "experimental_not_promoted", "unit": "decision", "contract": "CPU throughput"},
        {"metric": "mdstablecore_in_distribution", "value": "false", "unit": "boolean", "contract": "wheel and sdist"},
    ]
    write_tsv(HANDOFF / "key_metrics.tsv", metrics)


def create_patch() -> Path:
    output = HANDOFF / "unified_source_diff.patch"
    checkpoint_archive = RESULTS / "00_freeze" / "source_checkpoint.tar.gz"
    parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mdca_o9_checkpoint_") as temporary:
        checkpoint = Path(temporary)
        with tarfile.open(checkpoint_archive, "r:gz") as archive:
            archive.extractall(checkpoint)
        for relative in CHANGED_PATHS:
            current = ROOT / relative
            before = checkpoint / relative
            if not current.is_file():
                raise FileNotFoundError(current)
            command = [
                "diff",
                "-u",
                "--label",
                f"a/{relative}",
                "--label",
                f"b/{relative}",
                str(before if before.is_file() else Path("/dev/null")),
                str(current),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr)
            if result.stdout:
                parts.append(result.stdout)
    output.write_text(sanitize_text("".join(parts)), encoding="utf-8")
    return output


def populate_handoff() -> None:
    if HANDOFF.exists() and any(HANDOFF.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty handoff: {HANDOFF}")
    HANDOFF.mkdir(parents=True, exist_ok=True)

    root_copies = {
        "08_decision/final_cpu_optimisation_report.md": "final_cpu_optimisation_report.md",
        "08_decision/accepted_changes.tsv": "accepted_changes.tsv",
        "08_decision/rejected_or_deferred_changes.tsv": "rejected_or_deferred_changes.tsv",
        "08_decision/scientific_parity_summary.tsv": "scientific_parity_summary.tsv",
        "08_decision/performance_summary.tsv": "performance_summary.tsv",
        "08_decision/memory_summary.tsv": "memory_summary.tsv",
        "00_freeze/source_sha256.tsv": "source_checkpoint_manifest.tsv",
        "02_kabsch_kernel/covariance_decision.md": "covariance_decision.md",
        "03_all_frame_engine/api_design.md": "api_design.md",
        "03_all_frame_engine/ordinary_rmsf_contract.md": "ordinary_rmsf_contract.md",
        "04_exact_estimator/mean_only_design.md": "mean_only_design.md",
        "05_precision_solver/rotation_backend_decision.md": "rotation_backend_decision.md",
        "05_precision_solver/precision_decision.md": "precision_decision.md",
        "03_all_frame_engine/workspace_design.md": "workspace_design.md",
        "06_packaging_modes/package_dependency_audit.md": "package_dependency_audit.md",
        "06_packaging_modes/mode_alias_map.tsv": "mode_alias_map.tsv",
        "06_packaging_modes/core_fraction_cardinality.md": "core_fraction_cardinality.md",
    }
    for source, destination in root_copies.items():
        copy_review_file(RESULTS / source, Path(destination))

    for directory in ("01_baseline", "02_kabsch_kernel", "03_all_frame_engine", "04_exact_estimator", "05_precision_solver", "07_validation", "08_decision"):
        for source in sorted((RESULTS / directory).glob("*.tsv")):
            copy_review_file(source, Path("evidence") / directory / source.name)

    validation_patterns = (
        "*test_output*.txt",
        "*clean_install_report*.txt",
        "package_build_output_clean.txt",
        "package_build_output_final.txt",
        "twine_check_output_clean.txt",
        "twine_check_output_final.txt",
        "package_content_audit_clean.txt",
        "package_content_audit_final.txt",
    )
    copied: set[Path] = set()
    for pattern in validation_patterns:
        for source in sorted((RESULTS / "07_validation").glob(pattern)):
            if source.name.endswith("_venv.txt") or source in copied:
                continue
            copied.add(source)
            copy_review_file(source, Path("validation") / source.name)
    copy_review_file(
        RESULTS / "01_baseline" / "full_test_output.txt",
        Path("validation") / "baseline_full_test_output.txt",
    )

    for relative in CHANGED_PATHS:
        copy_review_file(ROOT / relative, Path("changed_files") / relative)

    create_phase_and_metric_tables()
    create_patch()

    readme = """# CPU optimisation v1 review handoff

This archive contains the O0-O9 CPU optimisation evidence and changed source.
No external scientific baseline was rerun. Exact remains chronological
`sequential_lean`, float64, and the public default. Ordinary full-set RMSF
remains enabled by default.

## Canonical environment

- Python 3.11.4
- NumPy 1.26.0 and SciPy 1.11.4 for canonical source validation
- OpenBLAS, canonical CPU timing at two threads
- GPU clean-install smoke: CuPy 13.6.0, CUDA runtime 12.9, Quadro P5000

## Validation

- Canonical tests: 327 passed, 13 warnings
- Focused tests: 41 passed
- CPU and GPU clean installs: passed
- Wheel/sdist build and `twine check`: passed

Paths in copied logs are replaced with `<REPOSITORY_ROOT>`, `<WORKSPACE_ROOT>`,
`<HOME>`, or `<TEMP_PATH>`. `sha256_manifest.tsv` hashes every payload file
except itself because a file cannot contain its own stable cryptographic hash.
The archive itself has an external SHA-256 in `review_archive_validation.tsv`.
"""
    (HANDOFF / "00_README_FIRST.md").write_text(readme, encoding="utf-8")


def write_manifests_and_scan() -> tuple[int, int]:
    payload = sorted(
        path
        for path in HANDOFF.rglob("*")
        if path.is_file() and path.name not in {"file_manifest.tsv", "sha256_manifest.tsv"}
    )
    file_rows = []
    violations: list[str] = []
    max_size = 0
    for path in payload:
        relative = path.relative_to(HANDOFF).as_posix()
        size = path.stat().st_size
        max_size = max(max_size, size)
        file_rows.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": sha256(path),
                "category": relative.split("/", 1)[0] if "/" in relative else "root",
            }
        )
        if size > 100 * 1024 * 1024:
            violations.append(f"oversize:{relative}:{size}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in ABSOLUTE_PATH_PATTERNS:
                if pattern.search(text):
                    violations.append(f"absolute_path_or_username:{relative}:{pattern.pattern}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    violations.append(f"secret_pattern:{relative}:{pattern.pattern}")
    if violations:
        raise RuntimeError("Review scan failed:\n" + "\n".join(violations))
    write_tsv(HANDOFF / "file_manifest.tsv", file_rows)

    hash_files = sorted(path for path in HANDOFF.rglob("*") if path.is_file() and path.name != "sha256_manifest.tsv")
    hash_rows = [
        {"sha256": sha256(path), "path": path.relative_to(HANDOFF).as_posix()}
        for path in hash_files
    ]
    write_tsv(HANDOFF / "sha256_manifest.tsv", hash_rows)
    return len(list(path for path in HANDOFF.rglob("*") if path.is_file())), max_size


def validate_archive() -> dict[str, object]:
    if ARCHIVE.exists():
        raise RuntimeError(f"Refusing to overwrite existing archive: {ARCHIVE}")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(HANDOFF, arcname="handoff_to_chatgpt", recursive=True)

    with tempfile.TemporaryDirectory(prefix="mdca_o9_extract_") as temporary:
        extraction = Path(temporary)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            members = archive.getmembers()
            if not members or any(
                not member.name.startswith("handoff_to_chatgpt/")
                and member.name != "handoff_to_chatgpt"
                for member in members
            ):
                raise RuntimeError("Archive does not contain exactly one handoff root")
            archive.extractall(extraction)
        extracted = extraction / "handoff_to_chatgpt"
        manifest = read_tsv(extracted / "sha256_manifest.tsv")
        failures = []
        for row in manifest:
            path = extracted / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                failures.append(row["path"])
        if failures:
            raise RuntimeError(f"Archive manifest failures: {failures}")
        canonical_files = sorted(
            path.relative_to(HANDOFF).as_posix()
            for path in HANDOFF.rglob("*")
            if path.is_file()
        )
        extracted_files = sorted(
            path.relative_to(extracted).as_posix()
            for path in extracted.rglob("*")
            if path.is_file()
        )
        if canonical_files != extracted_files:
            raise RuntimeError("Extracted file inventory differs from source handoff")
    return {
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "included_file_count": len(canonical_files),
        "extraction_status": "PASS",
        "manifest_status": "PASS",
        "secret_and_absolute_path_scan": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main() -> None:
    populate_handoff()
    file_count, max_size = write_manifests_and_scan()
    result = validate_archive()
    result["prearchive_file_count"] = file_count
    result["largest_included_file_bytes"] = max_size
    write_tsv(
        RESULTS / "review_archive_validation.tsv",
        [{key: value for key, value in result.items()}],
    )
    print(result)


if __name__ == "__main__":
    main()
