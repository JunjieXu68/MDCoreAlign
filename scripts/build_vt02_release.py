#!/usr/bin/env python3
"""Assemble the pip-first MDCoreAlign VT02.2 user archive."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mdcorealign_vt02_2_pip_first"
PACKAGE_ROOT = OUT / "06_release"
PACKAGE = OUT / "02_packaging" / "package_staging" / "MDCoreAlign_VT02_2"
VALIDATION = OUT / "02_packaging" / "validation_staging"
DATE = "20260901"
VERSION = "0.1.0rc4"
PYTHON = Path(os.environ.get("VT02_2_PYTHON", sys.executable))

TEST_FILES = (
    "vt02_2_pip_first/test_dependency_metadata.py",
    "purpose_based_coordinate_export_v1/test_coordinate_export.py",
    "vt02_full_protein_export/test_vt02_export.py",
    "throughput_gpu_v1/test_contracts.py",
    "throughput_gpu_v1/test_gpu_backend.py",
    "cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
    "cpu_optimisation_v1/test_fast_kabsch_cpu_v1.py",
    "cpu_optimisation_v1/test_sequential_mean_cpu_v1.py",
    "test_fast_kabsch_equivalence.py",
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
    "test_product_contract_cli.py",
)


def write_text(path: Path, text: str, *, executable: bool = False) -> None:
    """Write normalized UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")
    if executable:
        path.chmod(0o755)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, target: Path) -> None:
    """Copy source files while excluding generated caches."""

    blocked = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file():
            continue
        if any(part in blocked or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or ".xtc_offsets." in path.name:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resid: int,
    xyz: np.ndarray,
) -> str:
    """Format one full-atom PDB record for the compact example."""

    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{0.55:6.2f}{12.50:6.2f}          {atom_name[0]:>2s}\n"
    )


def write_full_atom_example(path: Path, *, n_frames: int = 5) -> None:
    """Write a two-chain, four-atoms-per-residue multi-model example."""

    residues = ("ALA", "GLY", "SER", "VAL", "LEU", "THR")
    atoms = ("N", "CA", "C", "O")
    lines: list[str] = []
    for frame in range(n_frames):
        lines.append(f"MODEL     {frame + 1:4d}\n")
        serial = 0
        for residue_index, resname in enumerate(residues, start=1):
            chain = "A" if residue_index <= 3 else "B"
            theta = residue_index * 0.73
            base = np.asarray(
                [residue_index * 2.2, np.sin(theta) * 3.0, np.cos(theta) * 2.0]
            )
            displacement = np.asarray(
                [0.0, 0.0, (0.35 if residue_index >= 5 else 0.02) * frame]
            )
            for atom_index, atom_name in enumerate(atoms):
                serial += 1
                offset = np.asarray(
                    [0.35 * atom_index, 0.17 * atom_index, -0.11 * atom_index]
                )
                lines.append(
                    atom_line(
                        serial,
                        atom_name,
                        resname,
                        chain,
                        residue_index,
                        base + displacement + offset,
                    )
                )
        lines.extend(("TER\n", "ENDMDL\n"))
    lines.append("END\n")
    write_text(path, "".join(lines))


def readme_text() -> str:
    """Reuse the validated public README with package-local wording."""

    text = (ROOT / "README.md").read_text()
    return text.replace(
        "but are documented under `docs/LEGACY_AND_ADVANCED.md` rather than the first-screen interface.",
        "but are intentionally omitted from the first-screen interface.",
    )


def validation_text() -> str:
    """Return the sealed VT02.2 validation summary."""

    return """# VT02.2 Validation

## Environment

VT02.2 was validated from independently extracted artifacts with isolated Python 3.11 and 3.12 pip environments. The base wheel resolves NumPy, SciPy, MDAnalysis and Gemmi without Conda. A separate clean pip environment validates the same wheel with the `cuda12` extra on the certified NVIDIA/CUDA-12 host. The optional Conda route is recorded separately.

## Results

Canonical, shipped, pip, GPU, Conda and sdist outcomes are recorded under `validation/`. Frozen previous-release versus VT02.2 scientific regression covers stable weights, core labels, transforms, RMSD, RMSF, W5 and flexible segments and must have zero numerical difference.

The full-protein fixture uses 6 fitting CA atoms and exports 24 protein atoms over 5 frames. PDB, mmCIF and XTC round trips preserve the complete protein and transformed coordinates. The CA-only fixture exports 8 CA atoms, emits a warning and invents no atoms. CPU/GPU parity uses absolute tolerance 1e-6 for continuous values and identity for labels.

The release gate includes focused Ruff, a public-API mypy check, compileall, build, Twine, pip check, archive-only installation, unrelated-directory smoke execution, English/CJK scanning, person-specific-content scanning, path/secret scanning, archive extraction and SHA-256 manifest verification. Evidence is under `validation/`.

No external scientific benchmark was run for VT02.2. This release changes installation metadata and packaging only.
"""


def developer_text() -> str:
    """Return the compact architecture and contributor guide."""

    return """# VT02.2 Developer Guide

`src/mdcorealign/` contains the public CLI, compact result schema, source provenance, CPU/GPU throughput, W5 interface and streaming coordinate export. `src/mdsuperpose/` contains numerical alignment, robust weighting, trajectory, RMSF, core-selection and local-kinematics components.

The core product flow is:

```text
align -> stable-weight estimation -> frozen weights -> explicit weighted fit for every frame
      -> compact result + versioned source_manifest.json

apply -> verified source recovery -> saved transforms -> bounded-memory chunked writer
      -> annotated PDB, annotated mmCIF, or XTC plus annotated topology
```

`apply` must consume saved transforms and must not rerun the scientific algorithm. Residue annotations are mapped by chain, residue name, residue identifier and insertion code, never by raw array index.

Alignment summaries use output schema `1.1.0`. Canonical fields are `fit_selection`, `n_fit_atoms`, `analysis_coordinate_selection`, `n_analysis_coordinate_atoms`, `default_export_selection`, `n_exportable_atoms` and `source_is_ca_only`. Historical names may appear only inside `legacy_metadata`. Export summaries use `export_selection`, `n_exported_atoms`, `output_format`, `annotation_mapping` and `source_result_schema_version`.

Development checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src/mdcorealign tests/vt02_full_protein_export
python -m mypy --python-version 3.12 --follow-imports=skip src/mdcorealign/__init__.py
python -m build
python -m twine check dist/*
```

The public default remains throughput CPU. Sequential exact remains the reference/audit path. GPU use is explicit. There is no normal-user plain-coordinate PDB mode. Export-only changes must preserve weights, labels, transforms, RMSD, RMSF, W5 and thresholds.
"""


def environment_text() -> str:
    """Return the validated Python/CUDA Conda specification."""

    return """name: mdcorealign-vt02-2
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.12
  - pip
  - numpy=2.5.2
  - scipy=1.16.3
  - mdanalysis=2.10.0
  - gemmi=0.7.5
  - cuda-version=12.9
  - cupy=13.6.0
"""


def install_script() -> str:
    """Return the location-independent pip-first installation helper."""

    return r"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
WHEEL="$ROOT/dist/mdcorealign-0.1.0rc4-py3-none-any.whl"
VENV="$ROOT/.venv"
CUDA12=0
USE_CONDA=0
RECREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
      VENV="$2"; shift 2 ;;
    --cuda12) CUDA12=1; shift ;;
    --conda) USE_CONDA=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    -h|--help)
      echo "Usage: bash install.sh [--venv PATH] [--cuda12] [--conda] [--recreate]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$WHEEL" ]]; then
  echo "Bundled wheel is missing: $WHEEL" >&2
  exit 1
fi

if [[ "$USE_CONDA" -eq 1 ]]; then
  command -v conda >/dev/null || { echo "--conda requires the conda command" >&2; exit 1; }
  if conda env list | awk '{print $1}' | grep -Fxq mdcorealign-vt02-2; then
    [[ "$RECREATE" -eq 1 ]] || { echo "Conda environment mdcorealign-vt02-2 already exists; use --recreate" >&2; exit 1; }
    conda env remove -n mdcorealign-vt02-2 -y
  fi
  conda env create -f "$ROOT/environment.yml"
  conda run -n mdcorealign-vt02-2 python -m pip install --no-deps "$WHEEL"
  conda run -n mdcorealign-vt02-2 python -m pip check
  conda run -n mdcorealign-vt02-2 mdcorealign doctor
  conda run -n mdcorealign-vt02-2 env PYTHON=python MDCOREALIGN=mdcorealign bash "$ROOT/run_smoke_test.sh"
  echo "MDCoreAlign VT02.2 optional Conda installation: PASS"
  exit 0
fi

if [[ -e "$VENV" ]]; then
  [[ "$RECREATE" -eq 1 ]] || { echo "Target exists: $VENV; use --recreate" >&2; exit 1; }
  [[ -f "$VENV/pyvenv.cfg" ]] || { echo "Refusing to replace non-venv target: $VENV" >&2; exit 1; }
  rm -rf "$VENV"
fi
"$PYTHON" -m venv "$VENV"
VENV_PYTHON="$VENV/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
if [[ "$CUDA12" -eq 1 ]]; then
  "$VENV_PYTHON" -m pip install "$WHEEL[cuda12]"
else
  "$VENV_PYTHON" -m pip install "$WHEEL"
fi
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" -m mdcorealign doctor
PATH="$VENV/bin:$PATH" PYTHON="$VENV_PYTHON" MDCOREALIGN="$VENV/bin/mdcorealign" \
  bash "$ROOT/run_smoke_test.sh"
echo "MDCoreAlign VT02.2 pip/venv installation: PASS"
"""


def smoke_script() -> str:
    """Return the strict archive-only VT02.2 smoke workflow."""

    return r"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
CLI="${MDCOREALIGN:-mdcorealign}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mdcorealign_vt02_2_smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

"$CLI" align --input "$ROOT/examples/full_protein/full_atom_trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --fast-representative-count 5 --output-dir result
test ! -e result/aligned.pdb
test -f result/diagnostics/source_manifest.json
"$CLI" apply result -o aligned.pdb
"$CLI" apply result -o aligned.cif
"$CLI" apply result -o aligned.xtc
"$CLI" align --input "$ROOT/examples/full_protein/full_atom_trajectory.pdb" --input-format ca-pdb --mode exact --backend cpu --fast-representative-count 5 --output-dir exact
"$CLI" align --input "$ROOT/examples/full_protein/full_atom_trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --ordinary-rmsf --fast-representative-count 5 --output-dir ordinary
"$CLI" local-deformation --input "$ROOT/examples/full_protein/full_atom_trajectory.pdb" --input-format ca-pdb --output-dir local

"$PYTHON" - "$WORK" <<'PY'
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import gemmi
import MDAnalysis as mda
root=Path(sys.argv[1])
with (root/'result/residue_profile.tsv').open(newline='') as handle:
    profile=list(csv.DictReader(handle,delimiter='\t'))
assert len(profile)==6
pdb_lines=[line for line in (root/'aligned.pdb').read_text().splitlines() if line.startswith('ATOM')]
assert len(pdb_lines)==24
assert {float(line[54:60]) for line in pdb_lines} <= {0.0,1.0}
assert 'OCCUPANCY = REPORTING CORE LABEL' in (root/'aligned.pdb').read_text()
block=gemmi.cif.read(str(root/'aligned.cif')).sole_block()
assert len(list(block.find_values('_atom_site.id')))==24
assert len(list(block.find_values('_mdcorealign_residue_profile.residue_id')))==6
u=mda.Universe(str(root/'aligned.topology.pdb'),str(root/'aligned.xtc'))
assert u.atoms.n_atoms==24 and len(u.trajectory)==5
side=json.loads((root/'aligned.xtc.mdcorealign.json').read_text())
assert side['atom_count']==24 and side['frame_count']==5
with (root/'ordinary/residue_profile.tsv').open(newline='') as handle:
    ordinary_fields=next(csv.reader(handle,delimiter='\t'))
assert 'ordinary_RMSF' in ordinary_fields
assert (root/'local/local_deformation.tsv').is_file()
print('CPU and I/O smoke: PASS')
PY

if "$PYTHON" -c 'import cupy as cp; raise SystemExit(0 if cp.cuda.runtime.getDeviceCount() else 1)' >/dev/null 2>&1; then
  "$CLI" align --input "$ROOT/examples/full_protein/full_atom_trajectory.pdb" --input-format ca-pdb --mode throughput --backend gpu --fast-representative-count 5 --output-dir gpu
  "$PYTHON" - "$WORK" <<'PY'
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle,delimiter='\t'))
cpu=rows(root/'result/residue_profile.tsv')
gpu=rows(root/'gpu/residue_profile.tsv')
for field in ('stable_weight','robust_RMSF','mobility_score','mean_residual'):
    np.testing.assert_allclose([float(row[field]) for row in cpu],[float(row[field]) for row in gpu],atol=1e-6,rtol=0)
assert [row['core_label'] for row in cpu]==[row['core_label'] for row in gpu]
print('GPU execution and CPU/GPU parity: PASS')
PY
else
  echo "GPU smoke: SKIPPED (no compatible CuPy/CUDA device)"
fi
echo "MDCoreAlign VT02.2 archive smoke: PASS"
"""


def copy_tests() -> None:
    """Copy the registered shipped-test subset."""

    for relative in TEST_FILES:
        source = ROOT / "tests" / relative
        destination = PACKAGE / "tests" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_validation_evidence() -> None:
    """Copy compact validation evidence into the user package."""

    (PACKAGE / "validation").mkdir(parents=True, exist_ok=True)
    for name in (
        "canonical_pytest_output.txt",
        "io_focused_pytest_output.txt",
        "scientific_regression.tsv",
        "cpu_gpu_parity.tsv",
        "full_protein_atom_count_tests.tsv",
        "ca_only_warning_test.txt",
        "ruff_output.txt",
        "mypy_output.txt",
        "compileall_output.txt",
    ):
        source = VALIDATION / name
        if source.is_file():
            shutil.copy2(source, PACKAGE / "validation" / name)
    for source, target in (
        (VALIDATION / "pdb_annotation_tests.tsv", "pdb_annotation_tests.tsv"),
        (VALIDATION / "mmcif_annotation_tests.tsv", "mmcif_annotation_tests.tsv"),
        (VALIDATION / "xtc_export_tests.tsv", "xtc_export_tests.tsv"),
        (VALIDATION / "simple_cli_tests.txt", "simple_cli_tests.txt"),
        (VALIDATION / "environment_manifest.tsv", "environment_manifest.tsv"),
    ):
        destination = PACKAGE / "validation" / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def sanitize_copied_evidence() -> None:
    """Remove machine-specific paths from copied release evidence only."""

    replacements = {
        str(ROOT): "<repository-root>",
        "/storage/junjie/projects": "<workspace-root>",
        "/home/jjxu": "<home>",
        "mdcorealign-vt01": "mdcorealign-validation",
    }
    text_suffixes = {".txt", ".tsv", ".json", ".yml", ".yaml", ".cff", ".toml", ".py", ".sh", ".md"}
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for original, replacement in replacements.items():
            updated = updated.replace(original, replacement)
        if updated != text:
            path.write_text(updated)


def build_distributions() -> None:
    """Build the wheel and source distribution from the assembled snapshot."""

    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "build",
            "--outdir",
            str(PACKAGE / "dist"),
            str(PACKAGE),
        ],
        check=True,
        cwd=ROOT,
    )
    expected = {
        f"mdcorealign-{VERSION}-py3-none-any.whl",
        f"mdcorealign-{VERSION}.tar.gz",
    }
    observed = {path.name for path in (PACKAGE / "dist").iterdir() if path.is_file()}
    if observed != expected:
        raise SystemExit(f"distribution mismatch: {sorted(observed)}")


def clean_build_artifacts() -> None:
    """Remove build-only directories from the assembled user package."""

    for path in (PACKAGE / "build", PACKAGE / "src/mdcorealign.egg-info"):
        if path.exists():
            shutil.rmtree(path)


def write_manifests() -> None:
    """Write deterministic payload manifests for independent verification."""

    manifest_paths = {
        PACKAGE / "validation/file_manifest.tsv",
        PACKAGE / "validation/sha256_manifest.tsv",
    }
    files = [
        path
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file()
        and path not in manifest_paths
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    rows = [
        {
            "path": path.relative_to(PACKAGE).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    for target in manifest_paths:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def archive_package() -> Path:
    """Create the date-stamped user archive with one canonical root."""

    archive = PACKAGE_ROOT / f"MDCoreAlign_VT02_2_{DATE}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(PACKAGE.rglob("*")):
            relative = path.relative_to(PACKAGE)
            if "__pycache__" in relative.parts or any(
                part.endswith(".egg-info") for part in relative.parts
            ):
                continue
            handle.add(
                path,
                arcname=Path("MDCoreAlign_VT02_2") / relative,
                recursive=False,
            )
    return archive


def main() -> int:
    """Assemble, build and archive VT02.2 without changing science."""

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    for source_name in ("mdcorealign", "mdsuperpose"):
        copy_tree(ROOT / "src" / source_name, PACKAGE / "src" / source_name)
    copy_tests()
    for filename in ("pyproject.toml", "LICENSE", "CITATION.cff"):
        shutil.copy2(ROOT / filename, PACKAGE / filename)
    write_text(PACKAGE / "README.md", readme_text())
    write_text(PACKAGE / "VALIDATION.md", validation_text())
    write_text(PACKAGE / "DEVELOPER.md", developer_text())
    write_text(PACKAGE / "environment.yml", environment_text())
    write_text(PACKAGE / "install.sh", install_script(), executable=True)
    write_text(PACKAGE / "run_smoke_test.sh", smoke_script(), executable=True)
    write_full_atom_example(PACKAGE / "examples/full_protein/full_atom_trajectory.pdb")
    for toy_name in ("toy_ca_topology.pdb", "toy_ca_trajectory.pdb"):
        source_toy = ROOT / "examples/toy_md" / toy_name
        destination_toy = PACKAGE / "examples/toy_md" / toy_name
        destination_toy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_toy, destination_toy)
    copy_validation_evidence()
    sanitize_copied_evidence()
    build_distributions()
    clean_build_artifacts()
    write_manifests()
    archive = archive_package()
    markdown = sorted(
        path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.md")
    )
    if markdown != ["DEVELOPER.md", "README.md", "VALIDATION.md"]:
        raise SystemExit(f"unexpected Markdown files: {markdown}")
    print(f"archive={archive}")
    print(f"sha256={sha256(archive)}")
    print(f"size_bytes={archive.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
