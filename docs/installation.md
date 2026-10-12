# Installation

Use Python 3.11 or newer. The recommended route uses the official GitHub
repository at the fixed `v1.0.0` tag; it does not require or claim a PyPI
publication of MDCoreAlign.

```bash
git clone https://github.com/JunjieXu68/MDCoreAlign.git
cd MDCoreAlign
git checkout v1.0.0
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m pip check
mdcorealign --help
mdcorealign doctor
```

The base product depends on NumPy, SciPy, MDAnalysis and Gemmi. It provides CPU
analysis and coordinate I/O. GPU execution uses the same package with the
optional CUDA 12 dependency. From that same checkout:

```bash
python -m pip install '.[cuda12]'
mdcorealign doctor
```

CuPy and the host driver must be compatible. CUDA 13 and platforms not listed
in the release validation record are unverified. An explicit GPU request does
not fall back to CPU. Historical `[io]` and `[external]` extras remain no-op
aliases. MDLovoFit and THESEUS are separate research tools, not product
dependencies.

A fixed-tag installation without a retained source checkout is also available:

```bash
python -m pip install 'git+https://github.com/JunjieXu68/MDCoreAlign.git@v1.0.0'
```

Alternatively, use the matching wheel or source archive listed on the
[1.0.0 GitHub Release](https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0).
Verify its asset identity against that release record. The wheel contains
runtime modules and the CLI. Example inputs and tests are provided in the
matching checkout/source archive; run those examples with the installed CLI on
PATH. See [CI testing boundaries](ci-testing.md) for installed-product,
source-research and historical-artifact checks.

Normal pip installation obtains dependencies from the configured package
index. The public release does not bundle third-party binary dependencies.
The separately documented personal offline backup covers only its recorded
packages, platform and tested restoration environment; it is not an offsite
backup or a general offline installer for every platform.
