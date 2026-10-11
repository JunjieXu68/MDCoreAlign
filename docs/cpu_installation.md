# CPU Installation

```bash
python -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install dist/mdcorealign-0.1.0rc6-py3-none-any.whl
mdcorealign align --help
```

The standard wheel installs NumPy, SciPy, MDAnalysis and Gemmi and provides the full
validated CPU product: array and multi-model PDB input, PDB/GRO + XTC input,
throughput CPU, exact CPU, W5, compact analysis, and mmCIF/XTC/PDB export. CuPy
is not imported or required for CPU execution.
