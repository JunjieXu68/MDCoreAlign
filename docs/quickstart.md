# Quickstart

From the matching source checkout or extracted sdist, with the installed CLI
on PATH, run the compact analysis. The wheel alone does not contain example
files:

```bash
bash examples/toy_md/run_quickstart.sh toy_output
```

Open `toy_output/summary.json` first. The result root contains five
primary files plus `diagnostics/`; it does not contain aligned coordinates.

```bash
mdcorealign align \
  --input examples/toy_md/toy_ca_trajectory.pdb \
  --input-format ca-pdb \
  --output-dir toy_output
```

Omitting `--mode` selects production throughput on CPU. Use `--mode exact` for
the slower chronological numerical audit, `--backend gpu` for an explicit GPU
throughput run, and `--ordinary-rmsf` only when the ordinary-fit comparison is
needed.

Run `mdcorealign doctor` to inspect mandatory dependency versions and whether
the included GPU backend is ready in the current CuPy/CUDA environment.

Coordinate materialization is a separate step:

```bash
mdcorealign apply toy_output -o toy_aligned.pdb
mdcorealign apply toy_output -o toy_aligned.cif
mdcorealign apply toy_output -o toy_aligned.xtc
```

The toy input is CA-only, so exports remain CA-only and emit a warning; missing
atoms are never invented. The toy input is a smoke test, not scientific
benchmark evidence.
