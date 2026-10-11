#!/usr/bin/env python3
"""Prepare the non-destructive MDCoreAlign source-code handoff snapshot."""

from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/source_code_handoff_v1"
SNAPSHOT = OUT / "MDCoreAlign_source_code_handoff"

TOP_LEVEL_FILES = (
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "SECURITY.md",
    "environment.yml",
    "pyproject.toml",
    "release_checklist.md",
)

USER_DOCS = (
    "docs/installation.md",
    "docs/quickstart.md",
    "docs/cli_reference.md",
    "docs/python_api.md",
    "docs/output_formats.md",
    "docs/output_schema.md",
    "docs/md_preprocessing.md",
    "docs/rmsf_flexible_segments.md",
    "docs/troubleshooting.md",
    "docs/support_matrix.md",
    "docs/recommended_workflow.md",
    "docs/local_deformation.md",
    "docs/modes_and_quality_contracts.md",
    "docs/gpu_installation.md",
    "docs/cpu_installation.md",
    "docs/release.md",
    "docs/user_testing_protocol.md",
    "docs/input_formats.md",
    "docs/conversion_guidance.md",
    "docs/container.md",
    "docs/large_trajectory_workflow.md",
    "docs/non_protein_systems.md",
    "docs/mdlovofit_comparison.md",
)

SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".toml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(values)


def copy_exact(relative: Path, parity: list[dict[str, object]]) -> None:
    source = ROOT / relative
    destination = SNAPSHOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    snapshot_hash = sha256(destination)
    parity.append(
        {
            "relative_path": relative.as_posix(),
            "source_sha256": source_hash,
            "snapshot_sha256": snapshot_hash,
            "match": source_hash == snapshot_hash,
            "size_bytes": source.stat().st_size,
        }
    )


def package_files() -> list[Path]:
    output = []
    for package in ("mdcorealign", "mdsuperpose", "mdstablecore"):
        output.extend(sorted((ROOT / "src" / package).glob("*.py")))
    return output


def test_files() -> list[Path]:
    return sorted(path for path in (ROOT / "tests").rglob("*.py") if "__pycache__" not in path.parts)


def script_files() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "scripts").rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES and "__pycache__" not in path.parts
    )


def benchmark_files() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "benchmark").rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and "__pycache__" not in path.parts
    )


def example_files() -> list[Path]:
    output = []
    for path in sorted((ROOT / "examples").rglob("*")):
        if not path.is_file() or "output" in path.parts or "__pycache__" in path.parts:
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        output.append(path)
    return output


def github_files() -> list[Path]:
    return sorted(path for path in (ROOT / ".github").rglob("*") if path.is_file())


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT / "src").with_suffix("")
    return ".".join(relative.parts)


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    arguments = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults_offset = len(positional) - len(node.args.defaults)
    for index, argument in enumerate(positional):
        name = argument.arg
        if index >= defaults_offset:
            name += "=..."
        arguments.append(name)
    if node.args.vararg:
        arguments.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        arguments.append("*")
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        arguments.append(argument.arg + ("=..." if default is not None else ""))
    if node.args.kwarg:
        arguments.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(arguments)})"


def source_indices(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    functions: list[dict[str, object]] = []
    modules: list[dict[str, object]] = []
    edges: set[tuple[str, str]] = set()
    known_modules = {module_name(path) for path in paths}
    for path in paths:
        module = module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        modules.append(
            {
                "module": module,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "package": module.split(".")[0],
                "module_docstring": (ast.get_docstring(tree) or "").split("\n", 1)[0],
                "source_lines": len(path.read_text().splitlines()),
            }
        )
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    {
                        "module": module,
                        "symbol": node.name,
                        "kind": "function",
                        "line": node.lineno,
                        "visibility": "private" if node.name.startswith("_") else "public",
                        "signature": signature(node),
                        "summary": (ast.get_docstring(node) or "").split("\n", 1)[0],
                    }
                )
            elif isinstance(node, ast.ClassDef):
                functions.append(
                    {
                        "module": module,
                        "symbol": node.name,
                        "kind": "class",
                        "line": node.lineno,
                        "visibility": "private" if node.name.startswith("_") else "public",
                        "signature": node.name,
                        "summary": (ast.get_docstring(node) or "").split("\n", 1)[0],
                    }
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append(
                            {
                                "module": module,
                                "symbol": f"{node.name}.{child.name}",
                                "kind": "method",
                                "line": child.lineno,
                                "visibility": "private" if child.name.startswith("_") else "public",
                                "signature": signature(child),
                                "summary": (ast.get_docstring(child) or "").split("\n", 1)[0],
                            }
                        )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if target in known_modules or any(target.startswith(f"{prefix}.") for prefix in ("mdcorealign", "mdsuperpose", "mdstablecore")):
                        edges.add((module, target))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base_parts = module.split(".")[:-node.level]
                    target = ".".join(base_parts + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if target and any(target == prefix or target.startswith(f"{prefix}.") for prefix in ("mdcorealign", "mdsuperpose", "mdstablecore")):
                    edges.add((module, target))
    edge_rows = [{"caller_module": caller, "imported_module": target} for caller, target in sorted(edges)]
    return functions, modules, edge_rows


def write_architecture(functions: list[dict[str, object]], modules: list[dict[str, object]], edges: list[dict[str, object]]) -> None:
    write_tsv(SNAPSHOT / "core_function_index.tsv", functions)
    write_tsv(SNAPSHOT / "docs/source_architecture/module_index.tsv", modules)
    write_tsv(SNAPSHOT / "docs/source_architecture/module_import_graph.tsv", edges)
    package_counts = {package: sum(row["package"] == package for row in modules) for package in ("mdcorealign", "mdsuperpose", "mdstablecore")}
    guide = f"""# MDCoreAlign Source Code Guide

## Scope

This archive is a source-code and validation handoff for MDCoreAlign
`0.1.0rc1`. Repository-relative module paths are preserved. Scientific
algorithms, numerical defaults and public behaviour are copied byte-for-byte
from the source repository.

## Package layout

- `src/mdcorealign/`: user-facing package, CLI, exact/throughput contracts,
  optional CUDA execution, guarded/experimental interfaces and PDB streaming.
- `src/mdsuperpose/`: robust alignment algorithms, representative selection,
  all-frame weighted Kabsch fitting, RMSD/RMSF, W5 local deformation,
  trajectory handling and external-tool adapters.
- `src/mdstablecore/`: earlier internal stable-core scoring pipeline retained
  because it is part of the importable research source tree.

Module counts are: mdcorealign={package_counts['mdcorealign']},
mdsuperpose={package_counts['mdsuperpose']}, and
mdstablecore={package_counts['mdstablecore']}.

## Primary workflows

```text
mdcorealign align --mode exact
  -> mdcorealign.cli
  -> mdcorealign.contracts.solve_contract
  -> mdsuperpose.sequential_weights (exact stable-weight estimation)
  -> mdsuperpose.fast_kabsch (explicit weighted fit for every frame)
  -> compact transforms, RMSD, weighted RMSD and RMSF outputs

mdcorealign align --mode throughput --backend cpu
  -> mdcorealign.experimental.solve_fast_global_alignment
  -> representative-frame stable-weight estimation
  -> explicit weighted fit for every frame after weights are frozen

mdcorealign align --mode throughput --backend gpu
  -> mdcorealign.gpu_backend.solve_fast_global_gpu
  -> optional CuPy backend with explicit transfer/compute timing
  -> no silent CPU fallback for an explicit GPU request

mdcorealign local-deformation
  -> mdsuperpose.window_kinematics.scan_local_kinematics
  -> alignment-free, scale-normalised W5 local-deformation outputs
  -> this score is not called RMSD

mdcorealign apply
  -> saved transforms
  -> mdsuperpose.fast_kabsch.apply_transforms_batch
  -> implemented PDB writers in mdcorealign.pdb_writer/mdsuperpose.pdb_writer
```

Throughput approximates stable-weight estimation only. Every frame still
receives an explicit weighted rigid-body fit after weights are frozen.

## Source indexes

- `core_function_index.tsv`: function/class/method name, module, signature,
  source line, visibility and first docstring line.
- `docs/source_architecture/module_index.tsv`: all installable modules.
- `docs/source_architecture/module_import_graph.tsv`: internal import edges.
- `docs/source_architecture/CALL_GRAPH.md`: reviewer-oriented workflow graph.

## Validation

See `validation/` for build, twine, clean-install, CPU/GPU, full-test, CLI,
example and archive-validation records. `source_hash_parity.tsv` demonstrates
byte-for-byte parity for every copied source/configuration file.

## Boundaries

Raw trajectories, database HDF5 files, coordinate caches, virtual
environments, Git internals and historical result trees are excluded.
MDLovoFit and THESEUS source/binaries are not redistributed. Their optional
adapters remain in `src/mdsuperpose/`.
"""
    (SNAPSHOT / "SOURCE_CODE_GUIDE.md").write_text(guide)
    (SNAPSHOT / "docs/source_architecture/CALL_GRAPH.md").write_text(
        """# High-Level Call Graph

```mermaid
flowchart TD
  CLI[mdcorealign.cli] --> CONTRACT[mdcorealign.contracts]
  CONTRACT --> EXACT[mdsuperpose.sequential_weights]
  CONTRACT --> FAST[mdcorealign.experimental]
  CONTRACT --> GPU[mdcorealign.gpu_backend]
  EXACT --> KABSCH[mdsuperpose.fast_kabsch]
  FAST --> REP[mdsuperpose.representative_selection]
  FAST --> KABSCH
  GPU --> KABSCH
  KABSCH --> RMSD[transforms and RMSD]
  KABSCH --> RMSF[mdsuperpose.flexibility / long_mode]
  CLI --> W5[mdsuperpose.window_kinematics]
  CLI --> APPLY[apply saved transforms]
  APPLY --> WRITER[mdcorealign.pdb_writer / mdsuperpose.pdb_writer]
```

Exact and throughput differ in stable-weight estimation. The final all-frame
weighted rigid fit is explicit in both workflows. W5 is an independent local
deformation diagnostic rather than a substitute for global alignment.
"""
    )


def write_notices() -> None:
    (SNAPSHOT / "THIRD_PARTY_NOTICES.md").write_text(
        """# Third-Party Notices

MDCoreAlign depends on NumPy and optionally uses CuPy, MDAnalysis, SciPy,
h5py, huggingface_hub and Matplotlib under their respective upstream licenses.

MDLovoFit and THESEUS are external comparison tools. This archive does not
contain their source code or binaries because redistribution rights were not
independently verified for this handoff. MDCoreAlign's Python adapters merely
locate and invoke separately installed executables. No instrumented MDLovoFit
source or binary is included.
"""
    )


def environment_record() -> None:
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "package_version": "0.1.0rc1",
        "python_executable_for_snapshot_creation": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_head_available": git_head.returncode == 0,
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else "unavailable_no_valid_HEAD",
        "scientific_algorithms_modified": False,
        "public_defaults_modified": False,
    }
    (SNAPSHOT / "validation").mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "validation/source_snapshot_environment.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing handoff root: {OUT}")
    SNAPSHOT.mkdir(parents=True)
    parity: list[dict[str, object]] = []
    for name in TOP_LEVEL_FILES:
        copy_exact(Path(name), parity)
    package_paths = package_files()
    for source in package_paths:
        copy_exact(source.relative_to(ROOT), parity)
    for source in (
        test_files()
        + script_files()
        + benchmark_files()
        + example_files()
        + github_files()
    ):
        copy_exact(source.relative_to(ROOT), parity)
    for name in USER_DOCS:
        copy_exact(Path(name), parity)
    if not all(bool(row["match"]) for row in parity):
        raise RuntimeError("source copy hash parity failed")
    write_tsv(SNAPSHOT / "validation/source_hash_parity.tsv", parity)
    functions, modules, edges = source_indices(package_paths)
    write_architecture(functions, modules, edges)
    write_notices()
    environment_record()
    (SNAPSHOT / "validation/SRC0_SRC3_status.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "copied_files": len(parity),
                "installable_modules": len(modules),
                "indexed_symbols": len(functions),
                "internal_import_edges": len(edges),
                "source_hash_parity": "pass",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"snapshot": SNAPSHOT.relative_to(ROOT).as_posix(), "copied_files": len(parity), "modules": len(modules), "symbols": len(functions)}, indent=2))


if __name__ == "__main__":
    main()
