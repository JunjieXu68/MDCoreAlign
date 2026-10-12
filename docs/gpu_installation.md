# GPU Installation

GPU is a backend in the same MDCoreAlign product, not a separate edition.
The `cuda12` extra installs CuPy 13 for CUDA 12. The CuPy build must match the
computer's driver/runtime; see the
[1.0.0 Release](https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0)
for recorded validation scope. CPU CI does not establish real-GPU validation.

Use the official fixed-tag checkout to install the package and run its example:

```bash
git clone https://github.com/JunjieXu68/MDCoreAlign.git
cd MDCoreAlign
git checkout v1.0.0
python -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install '.[cuda12]'
mdcorealign doctor
mdcorealign align --input examples/toy_md/toy_ca_trajectory.pdb \
  --input-format ca-pdb --backend gpu --output-dir gpu_result
```

This installs the complete MDCoreAlign package and its optional CUDA dependency.
GPU uses the same throughput method. An explicit GPU request never silently
falls back to CPU. CUDA 13 is not claimed as validated. Backend auto remains
experimental and is not a first-screen workflow.

The wheel contains runtime modules rather than example inputs; use examples
from the matching checkout or extracted source archive. See
[Installation](installation.md) for the CPU route and release-asset identity.
