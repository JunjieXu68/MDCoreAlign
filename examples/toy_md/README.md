# Toy MD Quickstart

This directory contains a tiny CA-only multi-model PDB trajectory with eight
residues and five frames. Residues 4-5 move more strongly than the rest of the
chain, making it useful for smoke-testing robust alignment and flexible segment
output.

Run from a fresh install:

```bash
bash examples/toy_md/run_quickstart.sh
```

The default compact outputs are written to `examples/toy_md/output/`:

- `residue_profile.tsv`
- `frame_metrics.tsv`
- `transforms.npz`
- `flexible_segments.tsv`
- `summary.json`
- `diagnostics/`

No aligned coordinates are written by the analysis command. Use
`mdcorealign apply --purpose compatibility` to create a multi-model PDB.

The toy files are intentionally small and are not intended as scientific
benchmark evidence.
