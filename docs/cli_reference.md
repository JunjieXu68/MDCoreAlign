# CLI Reference

## `align`

`align` writes compact stable-core analysis outputs and per-frame transforms.
It does not materialize coordinates by default.

```bash
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --output-dir result
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --mode exact --backend cpu --output-dir exact_result
mdcorealign align --help
mdcorealign align --help-advanced
```

First-screen options are `--input`, `--topology`, `--trajectory`,
`--input-format`, `--output-dir`, `--fit-selection`, `--mode
throughput|exact`, `--backend cpu|gpu`, `--ordinary-rmsf`,
`--local-deformation`, `--legacy-artifacts`, and `--dry-run`.

Throughput is the production default. It estimates one fixed stable-weight
vector with a bounded compute budget, then explicitly fits every frame. Exact
retains chronological float64 warm-start estimation as a numerical reference.
An explicit GPU request never silently falls back to CPU.

`--legacy-artifacts` writes deprecated flat compatibility files; it is not
part of the normal output contract. Historical research controls remain
accepted and are described in `LEGACY_AND_ADVANCED.md`.

## `doctor`

```bash
mdcorealign doctor
mdcorealign doctor --json
```

`doctor` reports MDCoreAlign, Python, NumPy, MDAnalysis, Gemmi, CuPy/CUDA and
detected-GPU status without installing packages or changing the environment.
It recommends CPU when a compatible GPU runtime is unavailable.

## `local-deformation`

```bash
mdcorealign local-deformation --topology system.gro \
  --trajectory trajectory.xtc --output-dir local_result
```

This alignment-free W5 workflow reports a dimensionless local-deformation
index, local rigidity, segments and block reproducibility. It does not report
RMSD or mature hard biological motion domains.

## `apply`

`apply` reuses saved transforms and never reruns stable-weight estimation.

```bash
mdcorealign apply result -o aligned.pdb
mdcorealign apply result -o aligned.cif
mdcorealign apply result -o aligned.xtc
mdcorealign apply --help-advanced
```

The suffix selects the output contract. PDB/mmCIF default to frame 0 and carry
core-label occupancy plus robust-RMSF B-factor annotations. XTC defaults to all
frames and creates an annotated companion topology. The result source manifest
recovers and validates paths, hashes, selections, frame count and fit-atom
order. Advanced source relocation, frame selection and streaming controls are
isolated under `--help-advanced`.

## Compatibility Commands

`flexibility`, `diagnostics`, `audit-preprocessing`, `audit-coordinate-scale`,
and `formats` remain available. New analyses should use `align` and inspect its
compact profile and diagnostics before invoking specialised compatibility
commands.
