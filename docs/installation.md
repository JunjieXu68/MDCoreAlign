# Installation

Use the candidate wheel identified by the accompanying `RELEASE_READINESS.md`.
This document does not claim that the candidate is published on PyPI.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install /path/to/mdcorealign-0.1.0rc6-py3-none-any.whl
python -m pip check
mdcorealign --help
mdcorealign doctor
```

The base product depends on NumPy, SciPy, MDAnalysis and Gemmi. It provides CPU
analysis and coordinate I/O. GPU execution uses the same MDCoreAlign wheel with
the optional CUDA 12 dependency:

```bash
python -m pip install "/path/to/mdcorealign-0.1.0rc6-py3-none-any.whl[cuda12]"
mdcorealign doctor
```

CuPy and the host driver must be compatible. CUDA 13 and platforms not listed
in the acceptance report are unverified. An explicit GPU request does not fall
back to CPU. Historical `[io]` and `[external]` extras remain no-op aliases.
MDLovoFit and THESEUS are separate research tools, not product dependencies.

The wheel contains runtime modules and the CLI. For example inputs and tests,
extract the matching source distribution or use the matching source checkout;
run its examples with the installed CLI on PATH. Source test tiers and fixture
requirements are described in [testing.md](testing.md).

Normal pip installation obtains dependencies from the configured package
index. The public release does not bundle third-party binary dependencies.
The separately documented personal offline backup covers only its recorded
packages, platform and tested restoration environment; it is not an offsite
backup or a general offline installer for every platform.
