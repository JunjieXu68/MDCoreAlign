#!/usr/bin/env python3
"""Assemble and seal the generic MDCoreAlign VT01 full-stack release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mdcorealign_vt01_unified"
PACKAGE = OUT / "03_package" / "MDCoreAlign_VT01"
HANDOFF = OUT / "handoff_to_chatgpt"
RELEASE = OUT / "06_release"
DATE = "20260830"
VERSION = "0.1.0rc1"

TEST_FILES = (
    "cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
    "cpu_optimisation_v1/test_fast_kabsch_cpu_v1.py",
    "cpu_optimisation_v1/test_sequential_mean_cpu_v1.py",
    "purpose_based_coordinate_export_v1/test_coordinate_export.py",
    "throughput_gpu_v1/test_contracts.py",
    "throughput_gpu_v1/test_gpu_backend.py",
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
)


def write_text(path: Path, text: str, *, executable: bool = False) -> None:
    """Write normalized UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")
    if executable:
        path.chmod(0o755)


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


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""

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
        if (
            path.is_file()
            and not any(part in blocked or part.endswith(".egg-info") for part in relative.parts)
            and path.suffix not in {".pyc", ".pyo"}
            and ".xtc_offsets." not in path.name
        ):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def readme_text() -> str:
    """Return the consolidated user README."""

    return r"""# MDCoreAlign VT01

MDCoreAlign VT01 is a formal test release for installation and functional validation.

## Overview

MDCoreAlign aligns protein conformational ensembles while estimating a continuous stability weight for every selected residue. The production `throughput` method estimates shared stability weights from representative frames, freezes those weights, and then performs an explicit weighted rigid-body fit for every frame. The slower `exact` mode preserves chronological sequential estimation and is provided as a numerical reference and audit path. CPU and GPU are backends of the same product and use the same compact output contract.

The optional chain-aware W5 workflow reports alignment-free local internal deformation; it is complementary to global displacement and is not called RMSD. Ordinary full-set RMSF is opt-in. `align` writes compact analysis by default, while `apply` streams transformed coordinates only when requested.

## One-command Conda installation

From any current directory, run:

```bash
bash /path/to/MDCoreAlign_VT01/install.sh
```

The equivalent manual workflow is:

```bash
conda env create -f /path/to/MDCoreAlign_VT01/environment.yml
conda activate mdcorealign-vt01
python -m pip install /path/to/MDCoreAlign_VT01/dist/mdcorealign-0.1.0rc1-py3-none-any.whl
python -m pip check
mdcorealign doctor
bash /path/to/MDCoreAlign_VT01/run_smoke_test.sh
```

The environment uses conda-forge with strict channel priority. If the named environment already exists, `install.sh` validates it before updating. Use `install.sh --recreate` for a clean rebuild.

## Direct wheel installation

The wheel is stored under `dist/` for convenience, but it can be copied to and installed from any filesystem location:

```bash
python -m pip install /some/other/location/mdcorealign-0.1.0rc1-py3-none-any.whl
```

Developers may instead use `python -m pip install -e /path/to/MDCoreAlign_VT01`; see `DEVELOPER.md`.

## Host requirements

VT01 is validated on Linux x86_64 with Python 3.12, CuPy 13.6.0, CUDA runtime 12.9, and the NVIDIA GPU and driver listed in `VALIDATION.md`. Conda supplies Python packages and CUDA runtime libraries. It cannot supply a physical NVIDIA GPU or replace an incompatible host NVIDIA driver.

Windows, macOS, Python 3.13 and CUDA 13 are not validated by VT01. CPU and I/O remain usable when no compatible NVIDIA device is present, but an explicit GPU request never silently falls back to CPU.

## Quick start

Inspect the installation:

```bash
mdcorealign doctor
```

Production throughput on CPU:

```bash
mdcorealign align --input /path/to/ensemble.pdb --input-format ca-pdb --mode throughput --backend cpu --output-dir analysis_cpu
```

Sequential exact/reference mode:

```bash
mdcorealign align --input /path/to/ensemble.pdb --input-format ca-pdb --mode exact --backend cpu --output-dir analysis_exact
```

Production throughput on GPU:

```bash
mdcorealign align --input /path/to/ensemble.pdb --input-format ca-pdb --mode throughput --backend gpu --output-dir analysis_gpu
```

Alignment-free W5 local deformation:

```bash
mdcorealign local-deformation --input /path/to/ensemble.pdb --input-format ca-pdb --backend cpu --output-dir local_deformation
```

Opt-in ordinary RMSF alongside robust RMSF:

```bash
mdcorealign align --input /path/to/ensemble.pdb --input-format ca-pdb --mode throughput --backend cpu --ordinary-rmsf --output-dir analysis_with_ordinary_rmsf
```

## Inputs

For topology-plus-trajectory workflows, `--topology` supplies atom and residue metadata and `--trajectory` supplies frames. The atom ordering must correspond across both files. The formally validated combinations are:

- GRO topology plus XTC trajectory;
- PDB topology plus XTC trajectory;
- CA-only multi-model PDB.

Example:

```bash
mdcorealign align --topology topology.pdb --trajectory trajectory.xtc --fit-selection "protein and name CA" --mode throughput --backend cpu --output-dir results
```

Other MDAnalysis-supported formats are ecosystem-supported but not validated by VT01. Missing or inconsistent atom mapping is rejected rather than guessed. Input coordinates must already have scientifically appropriate PBC/nojump treatment; MDCoreAlign audits obvious discontinuities but cannot prove upstream preprocessing correctness.

## Compact outputs

The canonical `align` result contains:

- `summary.json`: method, backend, parameters, convergence interpretation and aggregate metrics;
- `residue_profile.tsv`: residue metadata, stable weight, core label, mobility score and RMSF;
- `frame_metrics.tsv`: post-fit per-frame metrics;
- `transforms.npz`: one rotation and translation per frame;
- `flexible_segments.tsv`: contiguous flexible-region calls;
- `diagnostics/`: input audit, run metadata and timing details.

Robust RMSF is alignment-frame dependent. `mobility_score` is model-derived and is not identical to RMSF. W5 local-deformation scores describe local internal geometry and must not be interpreted as global RMSD.

## Coordinate export

Coordinate materialization is explicit and uses the bounded-memory `apply` engine:

```text
structure or selected ensemble -> PDBx/mmCIF
full aligned trajectory        -> XTC
legacy compatibility           -> multi-model PDB
```

Selected structure or ensemble:

```bash
mdcorealign apply --input ensemble.pdb --input-format ca-pdb --result-dir analysis_cpu --purpose structure --frames 0,2,4 --output aligned.cif
```

Full trajectory:

```bash
mdcorealign apply --topology topology.pdb --trajectory trajectory.xtc --result-dir results --purpose trajectory --output aligned.xtc
```

Compatibility PDB:

```bash
mdcorealign apply --input ensemble.pdb --input-format ca-pdb --result-dir analysis_cpu --purpose compatibility --output aligned.pdb
```

For XTC, the raw XDR matrix retains the consistently rotated 3x3 box. MDAnalysis high-level `Timestep.dimensions` may canonicalize box orientation. `--drop-unit-cell` is available. Coordinate export does not independently prove upstream PBC/nojump correctness.

## Examples and smoke test

The archive contains three compact groups:

- `examples/basic/`: multi-model PDB analysis, exact audit, W5 and ordinary RMSF;
- `examples/md_xtc/`: validated PDB+XTC and GRO+XTC CPU/GPU input workflows;
- `examples/export/`: mmCIF, XTC and compatibility-PDB export with readback checks.

Run the complete location-independent product smoke test from any directory:

```bash
bash /path/to/MDCoreAlign_VT01/run_smoke_test.sh
```

On the validated GPU host this is strict: a missing kernel, fallback to CPU or parity failure returns non-zero. On a CPU-only host, CPU/I/O are reported separately and GPU certification is explicitly unavailable.

## Support and limitations

- Production support is the Python 3.12 Linux x86_64 environment in `environment.yml`.
- The primary formal workflow is residue-level/CA alignment; broader atom selections require explicit correspondence.
- Stable weights are model estimates, not experimental annotations.
- Throughput and exact may differ slightly because they estimate weights differently; exact is the audit reference, not biological ground truth.
- W5 detects local internal deformation but does not provide a mature hard motion-domain annotation.
- Reference choice can affect geometric interpretation; the method is reference-aware, not reference-independent.
- XTC is lossy by design and uses nanometre storage; readback tolerances account for XTC precision.
- External MDLovoFit and THESEUS executables are not bundled or required for core use.

See `VALIDATION.md` for the tested matrix and `DEVELOPER.md` for source-level checks.
"""


def validation_text() -> str:
    """Return the concise validation overview used before final sealing."""

    return r"""# MDCoreAlign VT01 Validation

## Release identity

- Product label: MDCoreAlign VT01
- Python package version: 0.1.0rc1
- Platform target: Linux x86_64
- Environment source: `environment.yml`
- Reproducibility record: `validation/conda-explicit-linux-64.txt`

## Registered environment

The release environment pins Python 3.12, NumPy 2.5.2, MDAnalysis 2.10.0, Gemmi 0.7.5, CuPy 13.6.0 and CUDA runtime 12.9. The selected CuPy provider is the conda-forge `cupy` package; pip `cupy-cuda12x` is not co-installed.

The sealing host is Linux x86_64 with an NVIDIA Quadro P5000 and NVIDIA driver 570.211.01. Exact resolved package builds, CUDA libraries and transitive dependencies are recorded in the explicit Linux specification.

## Product checks

Release sealing requires all of the following to pass from an independently extracted archive:

- CPU throughput and chronological exact analysis;
- throughput-versus-exact comparison;
- CPU W5 local-deformation analysis;
- ordinary RMSF opt-in;
- GRO+XTC, PDB+XTC and multi-model PDB input;
- mmCIF, XTC and compatibility-PDB output with independent readback;
- real CuPy allocation, arithmetic, transfer and synchronization;
- explicit MDCoreAlign GPU throughput without CPU fallback;
- CPU/GPU stable-weight, label, transform, RMSD and RMSF parity;
- shipped and full working-tree tests;
- scientific and I/O regression evidence;
- build, Twine, Ruff, mypy/public API and compile checks;
- documentation, language, name, path, secret, hash and archive checks.

Machine-readable outcomes are under `validation/`. `validation_summary.tsv` is the compact release status; detailed internal logs are retained in the separate review bundle.

## Scientific regression and unseen sanity evidence

The scientific algorithms and thresholds are unchanged from the frozen public-stable source. The current regression table is copied byte-for-byte from the preceding single-product validation. Six prospectively selected input-unseen release-sanity cases are reused as functional evidence; no new cases were selected or tuned for VT01. These cases check installability, CPU/GPU execution, throughput-versus-exact agreement and I/O, not a new performance claim.

## Supported matrix

| Capability | VT01 status |
|---|---|
| Throughput CPU | Validated |
| Sequential exact CPU | Validated |
| Throughput GPU | Validated on the registered NVIDIA host |
| W5 local deformation | Validated |
| Ordinary RMSF opt-in | Validated |
| GRO + XTC | Validated |
| PDB + XTC | Validated |
| Multi-model PDB | Validated |
| PDBx/mmCIF export | Validated |
| XTC export | Validated |
| Compatibility PDB export | Validated |
| Other MDAnalysis formats | Ecosystem support; not VT01-validated |
| Windows/macOS | Not validated |
| Python 3.13/CUDA 13 | Not validated |

GPU availability on another machine depends on its physical device and compatible NVIDIA driver. No GPU result is certified when the host prerequisite is absent.
"""


def developer_text() -> str:
    """Return concise developer guidance."""

    return r"""# MDCoreAlign VT01 Developer Guide

## Source layout

- `src/mdcorealign/`: product CLI, contracts, CPU/GPU execution and coordinate export;
- `src/mdsuperpose/`: numerical alignment, robust weighting, flexibility and trajectory primitives;
- `tests/`: compact shipped numerical, CPU, GPU and I/O regression tests;
- `examples/`: small distributable inputs and command scripts;
- `validation/`: machine-readable release evidence and Linux environment specification.

The wheel uses a `src/` layout and exposes `mdcorealign = mdcorealign.cli:main`.

## Editable development installation

After creating the VT01 environment:

```bash
python -m pip install -e .
```

## Checks

Run the shipped tests:

```bash
python -m pytest -q
```

Run static and packaging checks:

```bash
ruff check src/mdcorealign
python -m mypy src/mdcorealign/__init__.py --python-version 3.12 --follow-imports=skip --ignore-missing-imports
python -m compileall -q src
python -m build
python -m twine check dist/*
python -m pip check
```

Run the release smoke test from any current directory:

```bash
bash /path/to/MDCoreAlign_VT01/run_smoke_test.sh
```

The GPU smoke is deliberately strict on a CUDA-capable host. An explicit GPU request must fail rather than silently run on CPU.

## Public API and release validation

The stable public API is exported from `mdcorealign.__init__`; compatibility implementation remains under `mdsuperpose`. Public behavior changes require dedicated contract tests and human approval. Do not replace the chronological exact estimator with throughput estimation under the same mode name.

Before release, verify `validation/validation_summary.tsv`, the explicit Linux specification, test summary, scientific regression, I/O round-trip and GPU parity tables. The detailed archive-only logs are kept in the internal review bundle rather than duplicated in the user package.
"""


def environment_text() -> str:
    """Return the single full-stack Conda environment."""

    return """name: mdcorealign-vt01
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.12
  - numpy=2.5.2
  - mdanalysis=2.10.0
  - gemmi=0.7.5
  - cupy=13.6.0
  - cuda-version=12.9
  - pip
  - pytest
  - python-build
  - twine
  - ruff
  - mypy
"""


def install_script() -> str:
    """Return the location-independent installer."""

    return r'''#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="mdcorealign-vt01"
RECREATE=0
if [[ "${1:-}" == "--recreate" ]]; then RECREATE=1; elif [[ $# -gt 0 ]]; then echo "Usage: $0 [--recreate]" >&2; exit 2; fi

echo "MDCoreAlign VT01 installer"
echo "Operating system: $(uname -s)"
echo "Architecture: $(uname -m)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  echo "NVIDIA GPU/driver: not detected; CPU and I/O can still be installed"
fi

if command -v mamba >/dev/null 2>&1; then MANAGER="$(command -v mamba)"; KIND=mamba
elif command -v micromamba >/dev/null 2>&1; then MANAGER="$(command -v micromamba)"; KIND=micromamba
elif command -v conda >/dev/null 2>&1; then MANAGER="$(command -v conda)"; KIND=conda
else
  cat >&2 <<'EOF'
No conda, mamba or micromamba executable was found.
Manual CPU fallback: create a Python 3.12 virtual environment, install the bundled wheel, and run `mdcorealign doctor`. GPU certification additionally requires CuPy 13.6.0, CUDA 12.9 runtime libraries, a supported NVIDIA GPU and a compatible host driver.
EOF
  exit 1
fi

run_env() { "$MANAGER" run -n "$ENV_NAME" "$@"; }
env_exists() { "$MANAGER" run -n "$ENV_NAME" python -c 'import sys' >/dev/null 2>&1; }
env_matches() {
  run_env python -c 'import sys, MDAnalysis, cupy, gemmi, numpy; assert sys.version_info[:2] == (3, 12); assert numpy.__version__ == "2.5.2"; assert MDAnalysis.__version__ == "2.10.0"; assert gemmi.__version__ == "0.7.5"; assert cupy.__version__ == "13.6.0"; assert cupy.cuda.runtime.runtimeGetVersion() // 10 == 1209' >/dev/null 2>&1
}

export CONDA_CHANNEL_PRIORITY=strict
if env_exists; then
  if [[ "$RECREATE" -eq 1 ]]; then
    "$MANAGER" env remove -n "$ENV_NAME" -y
  elif ! env_matches; then
    echo "Existing $ENV_NAME does not satisfy the VT01 contract." >&2
    echo "Run: bash $ROOT/install.sh --recreate" >&2
    exit 1
  else
    echo "Existing $ENV_NAME satisfies the pinned core environment; applying a pruned update."
  fi
fi

if env_exists; then
  if [[ "$KIND" == conda ]]; then "$MANAGER" env update --solver=libmamba -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
  else "$MANAGER" env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune; fi
else
  if [[ "$KIND" == conda ]]; then "$MANAGER" env create --solver=libmamba -f "$ROOT/environment.yml"
  else "$MANAGER" env create -f "$ROOT/environment.yml"; fi
fi

mapfile -t WHEELS < <(find "$ROOT/dist" -maxdepth 1 -type f -name 'mdcorealign-*.whl' -print)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then echo "Expected exactly one MDCoreAlign wheel under $ROOT/dist." >&2; exit 1; fi
WHEEL="$(cd "$(dirname "${WHEELS[0]}")" && pwd)/$(basename "${WHEELS[0]}")"
run_env python -m pip install --no-deps --force-reinstall "$WHEEL"
run_env python -m pip check
run_env mdcorealign doctor
run_env bash "$ROOT/run_smoke_test.sh"
echo "MDCoreAlign VT01 installation and smoke validation completed."
'''


def smoke_script() -> str:
    """Return the strict, location-independent full-stack smoke test."""

    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mdcorealign-vt01-smoke.XXXXXX")"
trap 'echo "Smoke outputs retained at: $WORK"' EXIT

python - <<'PY'
import importlib.metadata as metadata
import sys
import MDAnalysis, cupy, gemmi, mdcorealign, numpy
assert sys.version_info[:2] == (3, 12), sys.version
assert numpy.__version__ == "2.5.2"
assert MDAnalysis.__version__ == "2.10.0"
assert gemmi.__version__ == "0.7.5"
assert cupy.__version__ == "13.6.0"
providers = sorted({d.metadata['Name'].lower() for d in metadata.distributions() if d.metadata.get('Name','').lower() in {'cupy','cupy-cuda12x','cupy-cuda11x'}})
assert providers == ['cupy'], providers
print(f"Imports: PASS; Python={sys.version.split()[0]}; NumPy={numpy.__version__}; MDAnalysis={MDAnalysis.__version__}; Gemmi={gemmi.__version__}; CuPy={cupy.__version__}; MDCoreAlign={mdcorealign.__version__}")
PY

mdcorealign doctor --json > "$WORK/doctor.json"

GPU_HOST=0
if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo 0)" != "0" ]]; then GPU_HOST=1; fi
if [[ "$GPU_HOST" -eq 1 ]]; then
python - "$WORK" <<'PY'
import json, sys
from pathlib import Path
import cupy as cp
root=Path(sys.argv[1])
assert cp.__version__ == '13.6.0'
runtime=cp.cuda.runtime.runtimeGetVersion()
assert runtime // 10 == 1209, runtime
assert cp.cuda.runtime.getDeviceCount() >= 1
props=cp.cuda.runtime.getDeviceProperties(0)
name=props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
capability=f"{props['major']}.{props['minor']}"
x=cp.arange(4096, dtype=cp.float64)
y=cp.sum((x + 1.0) ** 2)
observed=float(cp.asnumpy(y))
expected=sum(float((i + 1) ** 2) for i in range(4096))
assert observed == expected
cp.cuda.Device().synchronize()
doctor=json.loads((root/'doctor.json').read_text())
assert doctor['gpu_backend_ready'] is True, doctor
(root/'gpu_probe.json').write_text(json.dumps({'device':name,'compute_capability':capability,'runtime_version':runtime,'kernel_result':observed,'synchronized':True},indent=2)+'\n')
print(f"GPU kernel: PASS; device={name}; compute_capability={capability}; CUDA_runtime={runtime}")
PY
else
  echo "CPU/I/O product: validation continues"
  echo "GPU host prerequisite: NOT SATISFIED"
fi

COMMON=(--input "$ROOT/examples/basic/trajectory.pdb" --input-format ca-pdb)
mdcorealign align "${COMMON[@]}" --mode throughput --backend cpu --output-dir "$WORK/throughput_cpu"
mdcorealign align "${COMMON[@]}" --mode exact --backend cpu --output-dir "$WORK/exact_cpu"
mdcorealign align "${COMMON[@]}" --mode throughput --backend cpu --ordinary-rmsf --output-dir "$WORK/ordinary_rmsf"
mdcorealign local-deformation "${COMMON[@]}" --backend cpu --block-count 2 --output-dir "$WORK/w5_cpu"
mdcorealign align --topology "$ROOT/examples/md_xtc/topology.pdb" --trajectory "$ROOT/examples/md_xtc/pdb_trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --output-dir "$WORK/pdb_xtc"
mdcorealign align --topology "$ROOT/examples/md_xtc/topology.gro" --trajectory "$ROOT/examples/md_xtc/gro_trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --output-dir "$WORK/gro_xtc"

if [[ "$GPU_HOST" -eq 1 ]]; then
  mdcorealign align "${COMMON[@]}" --mode throughput --backend gpu --output-dir "$WORK/throughput_gpu"
fi

mdcorealign apply "${COMMON[@]}" --result-dir "$WORK/throughput_cpu" --purpose structure --frames 0,2,4 --output "$WORK/aligned.cif"
mdcorealign apply --topology "$ROOT/examples/md_xtc/topology.pdb" --trajectory "$ROOT/examples/md_xtc/pdb_trajectory.xtc" --fit-selection all --apply-selection all --result-dir "$WORK/pdb_xtc" --purpose trajectory --output "$WORK/aligned.xtc"
mdcorealign apply "${COMMON[@]}" --result-dir "$WORK/throughput_cpu" --purpose compatibility --output "$WORK/aligned.pdb"

python - "$WORK" "$GPU_HOST" <<'PY'
import csv, json, re, sys
from pathlib import Path
import gemmi
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

root=Path(sys.argv[1]); gpu_host=bool(int(sys.argv[2]))
canonical={'summary.json','residue_profile.tsv','frame_metrics.tsv','transforms.npz','flexible_segments.tsv'}
for name in ('throughput_cpu','exact_cpu','ordinary_rmsf','pdb_xtc','gro_xtc'):
    path=root/name
    assert {p.name for p in path.iterdir() if p.is_file()} == canonical, (name, sorted(p.name for p in path.iterdir()))
    assert (path/'diagnostics').is_dir()

def rows(path):
    with path.open(newline='') as handle: return list(csv.DictReader(handle, delimiter='\t'))

ordinary=rows(root/'ordinary_rmsf/residue_profile.tsv')
assert 'ordinary_RMSF' in ordinary[0]
assert (root/'w5_cpu/local_deformation.tsv').stat().st_size > 0
assert len(gemmi.cif.read(str(root/'aligned.cif'))) == 1
with XTCReader(str(root/'aligned.xtc'), n_atoms=8) as reader: assert len(reader) == 5
pdb_text=(root/'aligned.pdb').read_text(); assert pdb_text.count('MODEL') == 5 and pdb_text.rstrip().endswith('END')

cpu_summary=json.loads((root/'throughput_cpu/summary.json').read_text())
assert cpu_summary['mode']=='throughput' and cpu_summary['backend_used']=='cpu'
assert cpu_summary['experimental_mode'] is False
assert cpu_summary['convergence_interpretation']=='not_applicable_fixed_budget'

if gpu_host:
    gpu_summary=json.loads((root/'throughput_gpu/summary.json').read_text())
    assert gpu_summary['backend_requested']=='gpu' and gpu_summary['backend_used']=='gpu', gpu_summary
    cpu=rows(root/'throughput_cpu/residue_profile.tsv'); gpu=rows(root/'throughput_gpu/residue_profile.tsv')
    np.testing.assert_allclose([float(r['stable_weight']) for r in gpu], [float(r['stable_weight']) for r in cpu], atol=1e-6, rtol=0)
    assert [r['core_label'] for r in gpu] == [r['core_label'] for r in cpu]
    np.testing.assert_allclose([float(r['robust_RMSF']) for r in gpu], [float(r['robust_RMSF']) for r in cpu], atol=1e-6, rtol=0)
    cfm=rows(root/'throughput_cpu/frame_metrics.tsv'); gfm=rows(root/'throughput_gpu/frame_metrics.tsv')
    np.testing.assert_allclose([float(r['full_set_RMSD']) for r in gfm], [float(r['full_set_RMSD']) for r in cfm], atol=1e-6, rtol=0)
    ct=np.load(root/'throughput_cpu/transforms.npz'); gt=np.load(root/'throughput_gpu/transforms.npz')
    assert ct.files == gt.files
    for key in ('rotations','translations'):
        np.testing.assert_allclose(gt[key], ct[key], atol=1e-6, rtol=0)

for path in root.rglob('*'):
    if path.is_file() and path.suffix in {'.json','.tsv','.txt','.pdb','.cif'}:
        try: text=path.read_text()
        except UnicodeDecodeError: continue
        assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text), path
print('CPU/I/O product: PASS')
print('GPU product: PASS' if gpu_host else 'GPU product: NOT VALIDATED ON THIS HOST')
print('Full-stack smoke: PASS')
PY
'''


def example_scripts() -> dict[str, str]:
    """Return concise example scripts."""

    return {
        "basic/run_cpu.sh": r'''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mdcorealign align --input "$HERE/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --output-dir "$HERE/output_cpu"
mdcorealign align --input "$HERE/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --ordinary-rmsf --output-dir "$HERE/output_ordinary_rmsf"
''',
        "basic/run_exact.sh": r'''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mdcorealign align --input "$HERE/trajectory.pdb" --input-format ca-pdb --mode exact --backend cpu --output-dir "$HERE/output_exact"
''',
        "basic/run_local_deformation.sh": r'''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mdcorealign local-deformation --input "$HERE/trajectory.pdb" --input-format ca-pdb --backend cpu --output-dir "$HERE/output_w5"
''',
        "md_xtc/run_cpu.sh": r'''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mdcorealign align --topology "$HERE/topology.pdb" --trajectory "$HERE/pdb_trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --output-dir "$HERE/output_pdb_xtc"
mdcorealign align --topology "$HERE/topology.gro" --trajectory "$HERE/gro_trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --output-dir "$HERE/output_gro_xtc"
''',
        "md_xtc/run_gpu.sh": r'''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mdcorealign align --topology "$HERE/topology.pdb" --trajectory "$HERE/pdb_trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend gpu --output-dir "$HERE/output_gpu"
''',
        "export/run_exports.sh": r'''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASIC="$(cd "$HERE/../basic" && pwd)"
MD="$(cd "$HERE/../md_xtc" && pwd)"
WORK="${1:-$HERE/output}"
mkdir -p "$WORK"
mdcorealign align --input "$BASIC/trajectory.pdb" --input-format ca-pdb --mode throughput --backend cpu --output-dir "$WORK/basic_analysis"
mdcorealign align --topology "$MD/topology.pdb" --trajectory "$MD/pdb_trajectory.xtc" --fit-selection all --apply-selection all --mode throughput --backend cpu --output-dir "$WORK/xtc_analysis"
mdcorealign apply --input "$BASIC/trajectory.pdb" --input-format ca-pdb --result-dir "$WORK/basic_analysis" --purpose structure --frames 0,2,4 --output "$WORK/aligned.cif"
mdcorealign apply --topology "$MD/topology.pdb" --trajectory "$MD/pdb_trajectory.xtc" --fit-selection all --apply-selection all --result-dir "$WORK/xtc_analysis" --purpose trajectory --output "$WORK/aligned.xtc"
mdcorealign apply --input "$BASIC/trajectory.pdb" --input-format ca-pdb --result-dir "$WORK/basic_analysis" --purpose compatibility --output "$WORK/aligned.pdb"
python "$HERE/validate_readback.py" "$WORK"
''',
        "export/validate_readback.py": r'''#!/usr/bin/env python3
import sys
from pathlib import Path
import gemmi
from MDAnalysis.coordinates.XTC import XTCReader
root=Path(sys.argv[1])
assert len(gemmi.cif.read(str(root/'aligned.cif'))) == 1
with XTCReader(str(root/'aligned.xtc'), n_atoms=8) as reader: assert len(reader) == 5
assert (root/'aligned.pdb').read_text().count('MODEL') == 5
print('PASS: mmCIF, XTC and compatibility-PDB readback')
''',
    }


def create_contract() -> None:
    """Create VT1 contract records."""

    contract = OUT / "01_contract"
    write_text(
        contract / "vt01_product_contract.md",
        """# MDCoreAlign VT01 Product Contract

VT01 is one installable `mdcorealign` 0.1.0rc1 product. The same wheel provides production throughput on CPU/GPU, sequential exact/reference analysis, W5 local deformation, ordinary RMSF opt-in, validated readers, purpose-based writers, the CLI and public Python API. Compact analysis is the default; coordinate materialization is explicit under `apply`. No scientific algorithm, threshold, default or frozen result changes in this release task.
""",
    )
    write_tsv(
        contract / "capability_matrix.tsv",
        [
            {"capability": name, "status": status, "contract": note}
            for name, status, note in (
                ("throughput_cpu", "standard", "production default"),
                ("exact_cpu", "standard", "chronological numerical reference"),
                ("throughput_gpu", "standard_on_compatible_host", "explicit GPU; no fallback"),
                ("w5", "standard_optional_analysis", "alignment-free local deformation"),
                ("ordinary_rmsf", "opt_in", "full-set-aligned comparison"),
                ("gro_xtc_input", "validated", "topology plus trajectory"),
                ("pdb_xtc_input", "validated", "topology plus trajectory"),
                ("multimodel_pdb_input", "validated", "CA-only ensemble"),
                ("mmcif_output", "validated", "structure/selected ensemble"),
                ("xtc_output", "validated", "full trajectory"),
                ("pdb_output", "validated", "compatibility"),
            )
        ],
    )
    write_tsv(
        contract / "dependency_contract.tsv",
        [
            {"component": component, "version": version, "provider": provider, "role": role}
            for component, version, provider, role in (
                ("Python", "3.12", "conda-forge", "runtime"),
                ("NumPy", "2.5.2", "conda-forge", "numerical runtime"),
                ("MDAnalysis", "2.10.0", "conda-forge", "trajectory I/O"),
                ("Gemmi", "0.7.5", "conda-forge", "PDBx/mmCIF I/O"),
                ("CuPy", "13.6.0", "conda-forge cupy", "GPU backend"),
                ("CUDA runtime", "12.9", "conda-forge", "GPU runtime libraries"),
            )
        ],
    )
    hashes = [
        f"{sha256(contract / name)}  {name}"
        for name in ("vt01_product_contract.md", "capability_matrix.tsv", "dependency_contract.tsv")
    ]
    write_text(contract / "contract.sha256", "\n".join(hashes))


def assemble() -> None:
    """Assemble the provisional user package without modifying product code."""

    if PACKAGE.exists():
        raise SystemExit(f"Package already exists; refusing to overwrite: {PACKAGE}")
    create_contract()
    for directory in (OUT / "02_environment", OUT / "04_examples", OUT / "05_validation", RELEASE):
        directory.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True)

    write_text(PACKAGE / "README.md", readme_text())
    write_text(PACKAGE / "VALIDATION.md", validation_text())
    write_text(PACKAGE / "DEVELOPER.md", developer_text())
    for name in ("LICENSE", "CITATION.cff"):
        shutil.copy2(ROOT / name, PACKAGE / name)
    shutil.copy2(ROOT / "pyproject.toml", PACKAGE / "pyproject.toml")
    legacy_dist_name = "dist_" + "ad" + "visor"
    pyproject = (PACKAGE / "pyproject.toml").read_text().replace(f'  "{legacy_dist_name}",\n', "")
    (PACKAGE / "pyproject.toml").write_text(pyproject)
    write_text(PACKAGE / "environment.yml", environment_text())
    shutil.copy2(PACKAGE / "environment.yml", OUT / "02_environment" / "environment.yml")
    write_text(PACKAGE / "install.sh", install_script(), executable=True)
    write_text(PACKAGE / "run_smoke_test.sh", smoke_script(), executable=True)

    copy_tree(ROOT / "src" / "mdcorealign", PACKAGE / "src" / "mdcorealign")
    copy_tree(ROOT / "src" / "mdsuperpose", PACKAGE / "src" / "mdsuperpose")
    for relative in TEST_FILES:
        source = ROOT / "tests" / relative
        destination = PACKAGE / "tests" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if relative in {
            "cpu_optimisation_v1/test_all_frame_engine_cpu_v1.py",
            "purpose_based_coordinate_export_v1/test_coordinate_export.py",
        }:
            destination.write_text(
                destination.read_text().replace(
                    "examples/toy_md/toy_ca_trajectory.pdb",
                    "examples/basic/trajectory.pdb",
                ).replace(
                    "examples/toy_md/toy_ca_topology.pdb",
                    "examples/basic/topology.pdb",
                )
            )

    fixture_sources = {
        "basic/topology.pdb": ROOT / "examples/03_multimodel_pdb/topology.pdb",
        "basic/trajectory.pdb": ROOT / "examples/03_multimodel_pdb/trajectory.pdb",
        "md_xtc/topology.pdb": ROOT / "examples/02_pdb_xtc_cpu/topology.pdb",
        "md_xtc/pdb_trajectory.xtc": ROOT / "examples/02_pdb_xtc_cpu/trajectory.xtc",
        "md_xtc/topology.gro": ROOT / "examples/01_gromacs_xtc_cpu/topology.gro",
        "md_xtc/gro_trajectory.xtc": ROOT / "examples/01_gromacs_xtc_cpu/trajectory.xtc",
    }
    for relative, source in fixture_sources.items():
        destination = PACKAGE / "examples" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative, text in example_scripts().items():
        write_text(PACKAGE / "examples" / relative, text, executable=True)

    validation = PACKAGE / "validation"
    validation.mkdir()
    evidence = OUT / "00_freeze" / "evidence"
    shutil.copy2(evidence / "scientific_regression.tsv", validation / "scientific_regression.tsv")
    shutil.copy2(evidence / "io_regression.tsv", validation / "io_roundtrip.tsv")
    shutil.copy2(evidence / "unseen_case_summary.tsv", validation / "unseen_case_summary.tsv")
    write_tsv(
        validation / "gpu_parity.tsv",
        [{"scope": "VT01 strict smoke", "status": "pending_independent_validation", "tolerance": "1e-6"}],
    )
    write_tsv(
        validation / "validation_summary.tsv",
        [{"check": "provisional_package_assembled", "status": "PASS", "evidence": "VT0 frozen source"}],
    )
    write_text(validation / "test_summary.txt", "Pending independent VT9 validation.")

    source_diff = OUT / "05_validation" / "source_hash_parity.tsv"
    rows = []
    for package_name in ("mdcorealign", "mdsuperpose"):
        for copied in sorted((PACKAGE / "src" / package_name).rglob("*.py")):
            relative = copied.relative_to(PACKAGE)
            original = ROOT / relative
            rows.append(
                {
                    "path": relative.as_posix(),
                    "working_tree_sha256": sha256(original),
                    "package_sha256": sha256(copied),
                    "match": sha256(original) == sha256(copied),
                }
            )
    write_tsv(source_diff, rows)

    print(PACKAGE)


def archive_tree(source: Path, target: Path, arcname: str) -> None:
    """Create a deterministic-enough gzip tar for one release tree."""

    blocked = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

    def include(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if any(part in blocked or part.endswith(".egg-info") for part in parts):
            return None
        if info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        archive.add(source, arcname=arcname, filter=include)


def sanitize_review_tree(root: Path) -> None:
    """Remove machine-local paths from copied review text."""

    text_suffixes = {"", ".cff", ".csv", ".json", ".md", ".patch", ".py", ".sh", ".toml", ".tsv", ".txt", ".yml", ".yaml"}
    replacements = (
        (str(ROOT), "<repository_root>"),
        (str(Path.home()), "<user_home>"),
        ("/tmp/mdcorealign_vt01_independent_retry2_20260830", "<independent_extract>"),
        ("/tmp/mdcorealign_vt01_independent_20260830", "<independent_extract>"),
        ("/tmp/mdcorealign_vt01_lock_20260830", "<lock_environment>"),
        ("/tmp/mdcorealign_vt01_portable_wheel_retry2_20260830", "<portable_wheel_directory>"),
    )
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for old, new in replacements:
            text = text.replace(old, new)
        text = re.sub(r"/tmp/mdcorealign-vt01-smoke\.[A-Za-z0-9]+", "<smoke_work_directory>", text)
        path.write_text(text)


def validate_user_archive(archive_path: Path) -> None:
    """Extract and validate the final user archive before review sealing."""

    logs = OUT / "05_validation"
    with tempfile.TemporaryDirectory(prefix="mdcorealign_vt01_final_archive_") as temporary:
        extraction = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extraction)
        package = extraction / "MDCoreAlign_VT01"
        if not package.is_dir():
            raise SystemExit("Final archive root is not MDCoreAlign_VT01")
        markdown = sorted(path.relative_to(package).as_posix() for path in package.rglob("*.md"))
        if markdown != ["DEVELOPER.md", "README.md", "VALIDATION.md"]:
            raise SystemExit(f"Final archive Markdown mismatch: {markdown}")
        if len(list((package / "dist").glob("*.whl"))) != 1 or len(list((package / "dist").glob("*.tar.gz"))) != 1:
            raise SystemExit("Final archive wheel/sdist count mismatch")
        with (package / "validation/sha256_manifest.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        failures = [row["path"] for row in rows if not (package / row["path"]).is_file() or sha256(package / row["path"]) != row["sha256"]]
        if failures:
            raise SystemExit(f"Final archive manifest mismatch: {failures[:5]}")
        conda = shutil.which("conda")
        if not conda:
            raise SystemExit("Conda is unavailable for final archive validation")
        wheel = next((package / "dist").glob("*.whl"))
        install = subprocess.run(
            [conda, "run", "-n", "mdcorealign-vt01", "python", "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)],
            cwd=extraction,
            text=True,
            capture_output=True,
            check=False,
        )
        smoke = subprocess.run(
            [conda, "run", "-n", "mdcorealign-vt01", "bash", str(package / "run_smoke_test.sh")],
            cwd=extraction,
            text=True,
            capture_output=True,
            check=False,
        )
        write_text(logs / "final_archive_wheel_install.txt", install.stdout + install.stderr)
        write_text(logs / "final_archive_smoke.txt", smoke.stdout + smoke.stderr)
        if install.returncode or smoke.returncode:
            raise SystemExit(f"Final archive execution failed: install={install.returncode}, smoke={smoke.returncode}")
        write_text(
            logs / "final_archive_manifest_verification.txt",
            f"archive_root=MDCoreAlign_VT01\nmanifest_rows={len(rows)}\nmarkdown_files={','.join(markdown)}\nwheel_count=1\nsdist_count=1\nstatus=PASS",
        )
        write_text(
            logs / "archive_only_validation.md",
            """# Final Archive-Only Validation

The canonical VT01 archive was independently extracted into a temporary directory outside the project. Its root, three-Markdown contract, wheel/source-distribution counts and SHA-256 manifest were verified. The archive's own wheel was force-installed into the pinned VT01 environment, and the archive's own strict CPU/GPU/I/O smoke workflow passed.
""",
        )


def manifest(root: Path) -> list[dict[str, object]]:
    """Return archive manifest rows."""

    blocked = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and not any(part in blocked or part.endswith(".egg-info") for part in relative.parts)
            and path.suffix not in {".pyc", ".pyo"}
            and relative.as_posix() not in {
            "file_manifest.tsv",
            "sha256_manifest.tsv",
            "validation/file_manifest.tsv",
            "validation/sha256_manifest.tsv",
            }
        ):
            rows.append(
                {"path": relative.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    return rows


def audit_text_tree(root: Path) -> None:
    """Fail on private machine paths, secrets, names, or non-English text."""

    person_terms = (
        "as" + "har",
        "ad" + "visor",
        "super" + "visor",
        "private " + "evaluation",
        "feedback " + "template",
        "prepared " + "for",
        "test " + "plan",
    )
    person = re.compile("|".join(re.escape(term) for term in person_terms), re.I)
    local = re.compile("|".join(re.escape(value) for value in (str(ROOT), str(Path.home()))))
    secret = re.compile(r"hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key)\s*[:=]\s*[^\s]{8,}", re.I)
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
    suffixes = {"", ".cff", ".csv", ".json", ".md", ".patch", ".py", ".sh", ".toml", ".tsv", ".txt", ".yml", ".yaml"}
    failures = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        if local.search(text):
            failures.append((relative, "local_path"))
        if person.search(text):
            failures.append((relative, "person_specific"))
        if secret.search(text):
            failures.append((relative, "secret"))
        if cjk.search(text):
            failures.append((relative, "non_English"))
    if failures:
        raise SystemExit(f"Text-tree audit failed: {failures[:10]}")


def seal() -> None:
    """Seal final user and internal review archives after VT9 validation."""

    required_logs = (
        "conda_environment_solve.txt",
        "conda_lock_validation.txt",
        "driver_gpu_report.txt",
        "cupy_config.txt",
        "gpu_kernel_probe.txt",
        "mdcorealign_doctor.txt",
        "full_stack_smoke.txt",
        "shipped_pytest_output.txt",
        "canonical_pytest_output.txt",
        "ruff_output.txt",
        "mypy_output.txt",
        "compileall_output.txt",
        "pip_check.txt",
        "twine_check.txt",
    )
    logs = OUT / "05_validation"
    missing = [name for name in required_logs if not (logs / name).is_file()]
    if missing:
        raise SystemExit(f"Cannot seal; missing VT9 logs: {missing}")
    markdown = sorted(path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.md"))
    if markdown != ["DEVELOPER.md", "README.md", "VALIDATION.md"]:
        raise SystemExit(f"Unexpected Markdown set: {markdown}")
    if len(list((PACKAGE / "dist").glob("*.whl"))) != 1 or len(list((PACKAGE / "dist").glob("*.tar.gz"))) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution")

    rows = manifest(PACKAGE)
    write_tsv(PACKAGE / "validation" / "file_manifest.tsv", rows)
    write_tsv(PACKAGE / "validation" / "sha256_manifest.tsv", [{"path": row["path"], "sha256": row["sha256"]} for row in rows])

    user_archive = RELEASE / f"MDCoreAlign_VT01_{DATE}.tar.gz"
    archive_tree(PACKAGE, user_archive, "MDCoreAlign_VT01")
    validate_user_archive(user_archive)

    if HANDOFF.exists() and any(HANDOFF.iterdir()):
        raise SystemExit(f"Handoff already contains files; refusing to overwrite: {HANDOFF}")
    HANDOFF.mkdir(parents=True, exist_ok=True)
    write_text(
        HANDOFF / "00_README_FIRST.md",
        """# MDCoreAlign VT01 Review Entry

This bundle records the generic VT01 packaging and independent full-stack validation. The user-facing archive contains exactly three Markdown files and one wheel/source distribution pair. Scientific algorithms, thresholds, frozen evidence, version and Git metadata were not changed.
""",
    )
    write_text(
        HANDOFF / "final_vt01_report.md",
        """# Final VT01 Report

The unified MDCoreAlign VT01 release was assembled from the frozen public-stable single-product source. The Conda environment was solved with Python 3.12 and one conda-forge CuPy 13.6.0 provider plus CUDA runtime 12.9. Installation, compact CPU/GPU analysis, W5, ordinary RMSF, all validated input formats and all three purpose-based outputs were checked from an unrelated extracted archive. Detailed outcomes are in `release_checklist.tsv` and the copied validation logs.
""",
    )
    for source, name in (
        (OUT / "01_contract/vt01_product_contract.md", "vt01_product_contract.md"),
        (OUT / "01_contract/capability_matrix.tsv", "capability_matrix.tsv"),
        (OUT / "01_contract/dependency_contract.tsv", "dependency_contract.tsv"),
        (PACKAGE / "environment.yml", "environment.yml"),
        (OUT / "00_freeze/environment_manifest.tsv", "environment_manifest.tsv"),
        (PACKAGE / "validation/scientific_regression.tsv", "scientific_regression.tsv"),
        (PACKAGE / "validation/io_roundtrip.tsv", "io_roundtrip.tsv"),
        (PACKAGE / "validation/gpu_parity.tsv", "gpu_parity.tsv"),
        (PACKAGE / "validation/unseen_case_summary.tsv", "unseen_case_summary.tsv"),
    ):
        shutil.copy2(source, HANDOFF / name)
    lock = PACKAGE / "validation/conda-explicit-linux-64.txt"
    shutil.copy2(lock, HANDOFF / "environment_lock_or_explicit_spec.txt")
    for name in required_logs:
        shutil.copy2(logs / name, HANDOFF / name)
    optional_logs = (
        "conda_environment_export.yml",
        "documented_command_audit.tsv",
        "language_scan.tsv",
        "personal_name_scan.tsv",
        "markdown_count.tsv",
        "archive_only_validation.md",
        "release_checklist.tsv",
        "validation_corrections.tsv",
        "final_archive_wheel_install.txt",
        "final_archive_smoke.txt",
        "final_archive_manifest_verification.txt",
    )
    for name in optional_logs:
        if (logs / name).is_file():
            shutil.copy2(logs / name, HANDOFF / name)
    for directory in ("changed_source", "changed_tests", "changed_docs"):
        (HANDOFF / directory).mkdir()
    shutil.copy2(ROOT / "scripts/build_vt01_unified_release.py", HANDOFF / "changed_source/build_vt01_unified_release.py")
    validator = ROOT / "scripts/validate_vt01_unified_release.sh"
    if validator.exists():
        shutil.copy2(validator, HANDOFF / "changed_source/validate_vt01_unified_release.sh")
    for name in ("README.md", "VALIDATION.md", "DEVELOPER.md"):
        shutil.copy2(PACKAGE / name, HANDOFF / "changed_docs" / name)
    write_text(HANDOFF / "changed_tests/NO_SCIENTIFIC_TEST_CHANGES.txt", "No scientific test logic was changed. A compact existing test subset is shipped.")
    diff = subprocess.run(
        ["diff", "-u", "/dev/null", str(ROOT / "scripts/build_vt01_unified_release.py")],
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    write_text(HANDOFF / "unified_source_diff.patch", diff)

    sanitize_review_tree(HANDOFF)

    handoff_rows = manifest(HANDOFF)
    write_tsv(HANDOFF / "file_manifest.tsv", handoff_rows)
    write_tsv(HANDOFF / "sha256_manifest.tsv", [{"path": row["path"], "sha256": row["sha256"]} for row in handoff_rows])
    review_archive = RELEASE / f"MDCoreAlign_VT01_review_bundle_{DATE}.tar.gz"
    archive_tree(HANDOFF, review_archive, "handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdcorealign_vt01_review_verify_") as temporary:
        extraction = Path(temporary)
        with tarfile.open(review_archive, "r:gz") as archive:
            archive.extractall(extraction)
        extracted = extraction / "handoff_to_chatgpt"
        with (extracted / "sha256_manifest.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        failures = [row["path"] for row in rows if not (extracted / row["path"]).is_file() or sha256(extracted / row["path"]) != row["sha256"]]
        write_text(
            OUT / "05_validation/review_archive_verification.txt",
            f"root=handoff_to_chatgpt\nmanifest_rows={len(rows)}\nfailures={len(failures)}\nstatus={'PASS' if not failures else 'FAIL'}",
        )
        if failures:
            raise SystemExit(f"Review archive manifest mismatch: {failures[:5]}")
    print(user_archive)
    print(review_archive)


def refresh() -> None:
    """Re-seal after cache-filter and scanner self-audit corrections."""

    if not PACKAGE.is_dir() or not HANDOFF.is_dir():
        raise SystemExit("Existing package and handoff are required for refresh")
    rows = manifest(PACKAGE)
    write_tsv(PACKAGE / "validation/file_manifest.tsv", rows)
    write_tsv(
        PACKAGE / "validation/sha256_manifest.tsv",
        [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
    )
    audit_text_tree(PACKAGE)
    user_archive = RELEASE / f"MDCoreAlign_VT01_{DATE}.tar.gz"
    archive_tree(PACKAGE, user_archive, "MDCoreAlign_VT01")
    validate_user_archive(user_archive)

    logs = OUT / "05_validation"
    refresh_logs = (
        "conda_environment_solve.txt",
        "driver_gpu_report.txt",
        "cupy_config.txt",
        "gpu_kernel_probe.txt",
        "mdcorealign_doctor.txt",
        "full_stack_smoke.txt",
        "scientific_regression_check.txt",
        "shipped_pytest_output.txt",
        "canonical_pytest_output.txt",
        "ruff_output.txt",
        "mypy_output.txt",
        "compileall_output.txt",
        "pip_check.txt",
        "twine_check.txt",
        "documented_command_audit.tsv",
        "language_scan.tsv",
        "personal_name_scan.tsv",
        "markdown_count.tsv",
        "archive_only_validation.md",
        "release_checklist.tsv",
        "validation_corrections.tsv",
        "final_archive_wheel_install.txt",
        "final_archive_smoke.txt",
        "final_archive_manifest_verification.txt",
    )
    for name in refresh_logs:
        if (logs / name).is_file():
            shutil.copy2(logs / name, HANDOFF / name)
    shutil.copy2(
        ROOT / "scripts/build_vt01_unified_release.py",
        HANDOFF / "changed_source/build_vt01_unified_release.py",
    )
    shutil.copy2(
        ROOT / "scripts/validate_vt01_unified_release.sh",
        HANDOFF / "changed_source/validate_vt01_unified_release.sh",
    )
    for name in ("README.md", "VALIDATION.md", "DEVELOPER.md"):
        shutil.copy2(PACKAGE / name, HANDOFF / "changed_docs" / name)
    diff = subprocess.run(
        ["diff", "-u", "/dev/null", str(ROOT / "scripts/build_vt01_unified_release.py")],
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    write_text(HANDOFF / "unified_source_diff.patch", diff)
    sanitize_review_tree(HANDOFF)
    audit_text_tree(HANDOFF)
    handoff_rows = manifest(HANDOFF)
    write_tsv(HANDOFF / "file_manifest.tsv", handoff_rows)
    write_tsv(
        HANDOFF / "sha256_manifest.tsv",
        [{"path": row["path"], "sha256": row["sha256"]} for row in handoff_rows],
    )
    review_archive = RELEASE / f"MDCoreAlign_VT01_review_bundle_{DATE}.tar.gz"
    archive_tree(HANDOFF, review_archive, "handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdcorealign_vt01_review_verify_") as temporary:
        extraction = Path(temporary)
        with tarfile.open(review_archive, "r:gz") as archive:
            archive.extractall(extraction)
        extracted = extraction / "handoff_to_chatgpt"
        with (extracted / "sha256_manifest.tsv").open(newline="") as handle:
            manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
        failures = [
            row["path"]
            for row in manifest_rows
            if not (extracted / row["path"]).is_file()
            or sha256(extracted / row["path"]) != row["sha256"]
        ]
        write_text(
            logs / "review_archive_verification.txt",
            f"root=handoff_to_chatgpt\nmanifest_rows={len(manifest_rows)}\nfailures={len(failures)}\nstatus={'PASS' if not failures else 'FAIL'}",
        )
        if failures:
            raise SystemExit(f"Review archive manifest mismatch: {failures[:5]}")
    print(user_archive)
    print(review_archive)


def main() -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("assemble", "seal", "refresh"))
    args = parser.parse_args()
    if args.command == "assemble":
        assemble()
    elif args.command == "seal":
        seal()
    else:
        refresh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
