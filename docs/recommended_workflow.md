# Recommended Workflow

## 1. Prepare And Audit

Unwrap/nojump and center periodic trajectories upstream. Then run:

```bash
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --fit-selection "protein and name CA" --output-dir result --dry-run
```

Inspect `result/diagnostics/input_audit.tsv` and
`result/diagnostics/chain_mapping_report.tsv`. These checks do not prove
correct upstream PBC processing.

## 2. Run Compact Production Analysis

```bash
mdcorealign align --topology system.gro --trajectory trajectory.xtc \
  --fit-selection "protein and name CA" --output-dir result
```

Throughput CPU is the default. Open `summary.json`, then
`residue_profile.tsv`. Every frame receives an explicit weighted rigid fit
after the shared stable weights are frozen.

## 3. Optional Audits

Use `--mode exact --backend cpu` for a chronological numerical reference,
`--ordinary-rmsf` for the ordinary-fit RMSF comparison, or
`--local-deformation` for W5 output under `diagnostics/`. RMSF is
alignment-frame dependent; W5 is dimensionless and is not RMSD.

## 4. Coordinate Export When Needed

```bash
mdcorealign apply result -o aligned.pdb
mdcorealign apply result -o aligned.cif
mdcorealign apply result -o aligned.xtc
```

PDB and mmCIF are annotated for visualization. XTC writes the full protein
trajectory and a matching annotated topology. Source paths, hashes and
selections are recovered from the result; use advanced overrides only after a
source file has moved.
