# GPU Installation

GPU is a backend in the same MDCoreAlign product, not a separate edition.
Install the standard package first, then install the CuPy build matching this
computer. CUDA 12 with CuPy 13 is the validated configuration for this release.

Run the example from the matching source checkout or extracted sdist; the
wheel does not contain the `examples/` directory.

```bash
python -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install dist/mdcorealign-0.1.0rc6-py3-none-any.whl
python -m pip install "cupy-cuda12x>=13,<14"
mdcorealign doctor
mdcorealign align --input examples/toy_md/toy_ca_trajectory.pdb \
  --input-format ca-pdb --backend gpu --output-dir gpu_result
```

The CuPy build must match the local CUDA driver/runtime. GPU is a backend of
the same throughput method. An explicit GPU request never silently falls back
to CPU. The first command installs the complete MDCoreAlign product; the
second enables its included GPU backend. CUDA 13 is not claimed as validated.
Backend auto remains experimental and is not a first-screen workflow.
