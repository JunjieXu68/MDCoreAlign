# MDCoreAlign

MDCoreAlign performs stable-residue-weighted alignment and flexibility analysis
for protein conformational ensembles. It separates stable-weight estimation
from the exact weighted rigid fit applied to every frame. The production
throughput workflow approximates only weight estimation; it does not skip or
approximate the per-frame rigid fit.

## Five Public Workflows

```bash
# 1. Production analysis on CPU (throughput is the default mode).
mdcorealign align --topology system.gro --trajectory trajectory.xtc --output-dir result

# Explicit form of the same production CPU workflow.
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --mode throughput --backend cpu --output-dir result_explicit

# 2. The same throughput method on an explicitly requested CUDA GPU.
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --mode throughput --backend gpu --output-dir result_gpu

# 3. Slower chronological float64 numerical reference/audit.
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --mode exact --backend cpu --output-dir result_exact

# 4. Complementary alignment-free W5 local-deformation analysis.
mdcorealign local-deformation --topology system.gro \
  --trajectory trajectory.xtc --output-dir local_result

# 5. Optional visualization-ready coordinate materialization after analysis.
mdcorealign apply result -o aligned.pdb
mdcorealign apply result -o aligned.cif
mdcorealign apply result -o aligned.xtc
```

| Workflow | Scientific purpose | GPU | Primary output |
| --- | --- | --- | --- |
| `align` (default) | Production stable-weight estimation and exact all-frame fitting | Optional | Compact metrics, transforms and residue profile |
| `align --mode exact` | Chronological numerical reference/audit | No | Matched compact outputs |
| `local-deformation` | Alignment-free local internal-deformation diagnostic | Optional | Dimensionless W5 profile and segments |
| `apply RESULT -o aligned.pdb` | Annotated complete-protein structure | No | PDB |
| `apply RESULT -o aligned.cif` | Annotated complete-protein structure | No | PDBx/mmCIF |
| `apply RESULT -o aligned.xtc` | Complete aligned protein trajectory | No | XTC plus annotated topology |

An explicit GPU request never silently falls back to CPU. CPU execution does
not import or require CuPy. Historical research modes remain parseable for
reproducibility but are documented under `docs/LEGACY_AND_ADVANCED.md` rather
than the first-screen interface.

## Installation

### Pip and venv (primary)

Conda is optional. Use Python 3.11 or newer with venv and pip. The recommended
installation uses the official repository and the fixed release tag:

```bash
git clone https://github.com/JunjieXu68/MDCoreAlign.git
cd MDCoreAlign
git checkout v1.0.0
python -m venv .venv
. .venv/bin/activate
python -m pip install .
mdcorealign doctor
```

This installs NumPy, SciPy, MDAnalysis and Gemmi for the complete CPU and
coordinate-I/O product. On a compatible CUDA-12 system, install the `cuda12`
extra from the same checked-out tag:

```bash
python -m pip install '.[cuda12]'
mdcorealign doctor
```

CPU and GPU use the same MDCoreAlign package; CuPy is not a base dependency.
The historical `[io]` and `[external]` extras remain no-op compatibility aliases.

For an installation without a retained source checkout, use the same fixed tag:

```bash
python -m pip install 'git+https://github.com/JunjieXu68/MDCoreAlign.git@v1.0.0'
```

Matching wheel and source archives are listed on the
[1.0.0 GitHub Release](https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0).
These instructions use GitHub; they do not require or claim a PyPI publication.

### Conda (optional)

From the same checked-out source directory:

```bash
conda env create -f environment.yml
conda activate mdcorealign
python -m pip install --no-deps .
```

Developers working from the source directory can use `python -m pip install
".[dev]"`. MDAnalysis supplies topology/trajectory reading and XTC writing;
Gemmi supplies validated PDBx/mmCIF writing and parsing. Neither changes the
alignment mathematics. MDLovoFit and THESEUS are separate research comparators
and are not required by MDCoreAlign.

## Quick Start

The wheel installs the CLI and runtime modules. Example files are provided in
the matching source checkout or release sdist, not inside the wheel. From that
source directory, with the installed CLI on PATH, run the CA-only toy analysis:

```bash
bash examples/toy_md/run_quickstart.sh toy_output
```

Open `toy_output/summary.json` first. The compact result contains:

```text
summary.json
residue_profile.tsv
frame_metrics.tsv
transforms.npz
flexible_segments.tsv
diagnostics/
```

Coordinates are intentionally not written by `align`. The result stores source
hashes and selections so normal export does not require those arguments again:

```bash
mdcorealign apply toy_output -o toy_aligned.pdb
```

The toy source contains only C-alpha atoms, so its export remains C-alpha-only
and emits a warning. MDCoreAlign never invents missing atoms.

## Core Options

```bash
mdcorealign align --help
mdcorealign align --help-advanced
mdcorealign apply --help
mdcorealign formats
```

`--ordinary-rmsf` opts into the full-set ordinary-fit RMSF comparison.
`--local-deformation` adds W5 output to `diagnostics/`. `--legacy-artifacts`
is available only for scripts that require deprecated flat files such as
`weights.csv`; it is not part of the default layout.

The throughput weight-estimation budget uses the following unambiguous symbols:

| Symbol | CLI option | Python keyword | Default |
| --- | --- | --- | --- |
| `K` representative frames | `--fast-representative-count` | `representative_count` | 256 |
| `M` robust update rounds | `--fast-update-rounds` | `update_rounds` | 4 |

Both resolved values are recorded as `representative_count_K` and
`update_rounds_M` in `diagnostics/run_metadata.json`.

Use `--fast-update-rounds 8` (API: `update_rounds=8`) to request eight
throughput updates on CPU or GPU. The default remains four updates. Additional
updates can leave converged numerical results unchanged; output differences
are not a test of the number of updates executed.

`--delta` is a positive finite robust-loss scale in Angstrom. Throughput and
exact retain their existing 1 Angstrom default; legacy sequential mode retains
0.2 Angstrom. Requested and effective settings are recorded in run metadata.
Explicit incompatible robust-rule, iteration-limit and warm-start controls
fail before fitting. See
[the parameter contract](docs/parameter_contract.md) for supported controls.

## Input Formats

| Concept | Default |
| --- | --- |
| Analysis atoms | Protein C-alpha atoms |
| Default coordinate export | Complete protein |
| CA-only source | All available C-alpha atoms; missing atoms are not invented |

Formally validated combinations are:

- PDB topology + XTC trajectory;
- GRO topology + XTC trajectory;
- CA-only multi-model PDB.

NumPy arrays with shape `(frames, atoms, 3)` are supported for programmatic and
small test workflows. Fixed atom correspondence is required. Missing residues,
atom reordering, and changing selections must be resolved before analysis.
Other MDAnalysis-readable pairs may work but are not formally validated.

Protein C-alpha fitting is the formal workflow. Use `--fit-selection` to state
the selection explicitly when metadata are available.

## Output Interpretation

- `residue_profile.tsv` contains continuous `stable_weight`, its complementary
  `mobility_score`, the derived fixed-fraction `core_label`, robust RMSF,
  residuals and flexible-segment labels.
- `frame_metrics.tsv` contains full-set and stable-weighted RMSD after explicit
  per-frame weighted fitting.
- `transforms.npz` contains one rotation and translation for every frame.
- `flexible_segments.tsv` summarizes contiguous high-mobility regions.
- `diagnostics/` contains timing, input audit, chain mapping, run metadata and
  generated representative-frame diagnostics. Its `source_manifest.json`
  records source paths, SHA-256 hashes, selections, frame count and atom-order
  fingerprints for safe later export.

`mobility_score = 1 - stable_weight`. It is model-derived and is not identical
to RMSF. Robust RMSF is alignment-frame dependent. Ordinary RMSF is absent
unless `--ordinary-rmsf` is requested.

## Coordinate Export

Use `apply`, not `align`, for coordinate materialization. The output suffix
selects the format and the saved result recovers the original input and
selections:

```bash
# Complete-protein annotated frame 0 -> PDB.
mdcorealign apply result -o aligned.pdb

# Complete-protein annotated frame 0 -> PDBx/mmCIF.
mdcorealign apply result -o aligned.cif

# Complete aligned protein trajectory plus annotated topology -> XTC.
mdcorealign apply result -o aligned.xtc
```

For PDB and mmCIF, occupancy is the reporting core label (`1.00` core,
`0.00` non-core) and B-factor/`B_iso_or_equiv` is stable-core-aligned robust
RMSF in angstrom. These are visualization annotations, not experimental
occupancies or crystallographic temperature factors. If chain-restricted fitting
leaves exported residues without analysis values, the export instead preserves
source occupancy/B-factor values throughout and warns explicitly. mmCIF additionally stores
the complete versioned residue profile. XTC cannot store these fields, so
`apply` creates `aligned.topology.pdb`, or `aligned.topology.cif` when PDB
cannot safely represent the topology, plus `aligned.xtc.mdcorealign.json`.

The default fit selection is `protein and name CA`; when full protein atoms are
present, the same saved transforms are applied to `protein` by default. If
source files have moved, provide verified advanced overrides:

```bash
mdcorealign apply result -o aligned.xtc \
  --topology /new/path/system.gro \
  --trajectory /new/path/trajectory.xtc
```

For XTC, the raw XDR file retains the consistently rotated 3x3 box matrix;
MDAnalysis high-level `Timestep.dimensions` may canonicalize the box
orientation. `--drop-unit-cell` remains available. Coordinate export does not
independently prove upstream PBC/nojump correctness.

## Python API

```python
from mdcorealign.contracts import solve_contract

execution = solve_contract(
    coords,                    # shape: (frames, residues, 3), Angstrom
    contract="throughput",
    backend="cpu",
    store_aligned=False,
)
weights = execution.alignment.frozen_weights
rotations = execution.alignment.rotations
translations = execution.alignment.translations
```

## Limitations

- MDCoreAlign does not replace upstream unwrap/nojump and centering.
- Input diagnostics cannot prove correct PBC processing from generic files.
- Results are reference-aware, not reference-independent.
- Exact is a numerical reference, not biological ground truth.
- Throughput and exact can estimate different stable weights; both perform an
  explicit weighted rigid fit for every frame after weights are frozen.
- W5 is a dimensionless local-deformation diagnostic, not RMSD or a mature hard
  biological motion-domain annotation.
- Flexible-segment labels are analysis aids and require biological context.
- Formal validation is currently centred on protein C-alpha workflows.
- Analysis can materialize the selected coordinates in memory. Compute chunking
  does not guarantee bounded memory for the complete input-to-output workflow.
- Solver timing fields exclude input reading, result serialization and a later
  `apply` command; they are not complete CLI wall-clock timings.

## Evidence And Citation

Frozen comparisons support related but non-identical, method-dependent stable
reference solutions for MDCoreAlign and MDLovoFit. Runtime claims must use
matched inputs and output contracts; no universal-fastest claim is made.

See [CITATION.cff](CITATION.cff) for version 1.0.0 citation metadata and the
[GitHub Release](https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0)
for the tag, source identity, assets and published validation record. The
archival DOI, when assigned, is recorded on that linked GitHub Release page.

Version 1.0.0 retains the numerical implementation and default scientific
parameters of the maintained rc8 runtime. Version and installation-documentation
updates do not constitute a new reproduction of historical scientific results.
The preserved VT02.2 (`0.1.0rc4`) package predates the current M8 support and
interface fixes. Historical analyses require their recorded code, inputs,
parameters and artifacts; software acceptance tests alone do not reproduce them.
See [CI testing boundaries](docs/ci-testing.md) for the separate product,
source-research and historical-artifact checks.
See [historical reproduction status](docs/reproduction-status.md) for the
verified scope and limitations of the retained scientific evidence.
