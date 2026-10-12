# CPU Installation

Use Python 3.11 or newer and the official fixed release tag:

```bash
git clone https://github.com/JunjieXu68/MDCoreAlign.git
cd MDCoreAlign
git checkout v1.0.0
python -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install .
mdcorealign doctor
mdcorealign align --help
```

The base package installs NumPy, SciPy, MDAnalysis and Gemmi. It provides array
and multi-model PDB input, PDB/GRO + XTC input, throughput CPU, exact CPU, W5,
compact analysis, and mmCIF/XTC/PDB export. CuPy is not imported or required for
CPU execution.

For fixed-tag Git installation without a retained checkout, matching binary
and source assets, and validation scope, see [Installation](installation.md)
and the [1.0.0 Release](https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0).
