# Troubleshooting

## GPU Requested But Unavailable

The installed MDCoreAlign wheel already contains the GPU backend. Install a
CuPy build compatible with the CUDA driver, for example `pip install
"cupy-cuda12x>=13,<14"` on the validated CUDA 12 stack, then run
`mdcorealign doctor`. An explicit GPU request never falls back to CPU; use
`--backend cpu` deliberately. CUDA 13 is not validated by this release.

## Apply Rejects Saved Transforms

Reuse the original trajectory frame ordering, topology, fit selection and
residue ordering. `apply` rejects mismatched frame counts, fit-selection sizes
and residue labels rather than applying an unsafe transform archive.

## Standard Dependency Missing

PDBx/mmCIF structure export uses Gemmi and XTC trajectory export uses
MDAnalysis. Both are included by the standard wheel. If either import is
missing, the environment is incomplete; reinstall with `python -m pip install
--force-reinstall <wheel-path>` and run `python -m pip check`.

## Purpose And Format Conflict

Use `structure` with mmCIF (or explicit PDB), `trajectory` with XTC, and
`compatibility` with PDB. A full trajectory requested as structure text output
must use explicit frame selection and may require
`--allow-large-text-output`; XTC is preferred.

## XTC Has No Topology

XTC stores coordinates and trajectory metadata but not atom identities. Keep
and supply the exact topology matching `--apply-selection`. The export writes
`export_atom_metadata.tsv` and provenance, but those tables do not replace a
topology for MD software.

## `mdcorealign: command not found`

Install the package first:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Fit Selection Returned Zero Atoms

Example message:

```text
fit-selection 'protein and name CA' returned zero atoms
```

Actions:

- Run `mdcorealign audit-preprocessing` on the input first.
- Check whether the PDB contains `ATOM` records and C-alpha atom names.
- Try `--fit-selection "protein"` if the input is protein-only but atom names
  are nonstandard.
- Keep the formal manuscript default as `protein and name CA` unless there is a
  documented reason to use a broader selection.

## Apply Selection Returned Zero Atoms

Example message:

```text
apply-selection 'protein' returned zero atoms
```

Actions:

- Use `--apply-selection "protein and name CA"` for CA-only examples.
- Use `--apply-selection "protein"` only when all-atom protein records are present.
- Use `--apply-selection "all"` only when HETATM records should be transformed
  for visualization or downstream inspection.

## Input shape errors

Array inputs must have shape `(n_frames, n_atoms, 3)`. CA-only PDB inputs must
have the same CA atom order and residue labels in every model.

## Format Detection Failed

Example message:

```text
Format detection failed. Check file extension or use --topology-format / --trajectory-format.
```

Actions:

- Check file extensions.
- Prefer explicit `--topology` and `--trajectory` for real MD data.
- Add `--topology-format` and `--trajectory-format` if extensions are unusual.
- Convert to a conservative PDB/XTC, GRO/XTC, PSF/DCD, PRMTOP/NC, CA-only PDB,
  or NumPy workflow if MDAnalysis cannot read the files.

## Missing Topology

Example message:

```text
A reliable topology is required for residue-level stable-core and RMSF analysis.
```

Actions:

- Provide `--topology topology.pdb --trajectory trajectory.xtc`.
- Do not use trajectory-only files for residue-level RMSF/stable-core analysis.
- If you have a small multi-model PDB, use `--input ensemble.pdb --input-format ca-pdb`.

## Atom Correspondence Or Topology Mismatch

Example message:

```text
topology/trajectory atom identity changed across PDB models
```

Actions:

- Confirm every model/frame has the same atom order.
- Remove models with missing residues before running MDCoreAlign.
- Do not concatenate unrelated structures unless atom correspondence is explicit.
- Prefer a CA-only multi-model PDB for the compact single-file CLI workflow.

## Missing Residue Or Chain Metadata

MDCoreAlign can run with missing chain IDs, but the biological interpretation is
weaker. Inspect `chain_mapping_report.tsv`.

Actions:

- Add chain IDs upstream if possible.
- Keep a residue metadata table with the original source.
- Do not merge flexible segments across chains manually; chain boundaries are
  respected by the segment detector.

## NaN Or Infinite Coordinates

Example message:

```text
input coordinates contain NaN or infinite values
```

Actions:

- Regenerate the trajectory subset.
- Check for failed coordinate conversion or corrupted frames.
- Remove invalid frames before alignment.

## Large Frame-To-Frame Jumps

Large jumps in `input_audit.tsv` can indicate PBC imaging, chain breaks, or a
badly concatenated trajectory.

Actions:

- Inspect `large_frame_to_frame_displacement_A`.
- Re-run upstream nojump/unwrap/center preprocessing if needed.
- Do not claim PBC/nojump status from MDCoreAlign alone.

## Missing MDLovoFit or THESEUS

Core commands do not require MDLovoFit or THESEUS. Missing external tools only
affect optional baseline-comparison workflows.

## Strange RMSF values

RMSF depends on the alignment frame. Check:

- atom correspondence;
- reference mode;
- whether the trajectory was unwrapped/nojumped upstream;
- whether mobile domains are expected.

## PBC/nojump uncertainty

The core CLI cannot prove upstream periodic-boundary preprocessing. Use your MD
engine or analysis package to unwrap and center trajectories before running
MDCoreAlign.

## Output Directory Problems

If an output directory already contains files, MDCoreAlign may overwrite files
with the same names. For release-candidate testing, use a fresh output directory for
each command.

## Trajectory Reader Unavailable

MDAnalysis-backed topology/trajectory I/O is part of the standard installation.
If it cannot be imported, reinstall the wheel and run `python -m pip check`.
PDB+XTC, GRO+XTC, and CA multi-model PDB are covered by the current compact I/O
validation matrix. Other backend-readable combinations should be round-trip
checked before production use.
