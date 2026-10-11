# Software Regression and Research-Archive QA

Install the developer dependencies in an isolated full source checkout, then run:

```bash
python -m pytest -q -m "not archive_qa"
```

The developer extra includes h5py for the source-only HDF5 helper regressions.
This is the portable software suite, including the public parameter, input,
export and packaging regressions. GPU tests require a working optional CUDA
runtime; unavailable-device skips must be reported separately from passes.
For the packaging assertion, point to the artifact actually being evaluated:

```bash
MDCOREALIGN_WHEEL_UNDER_TEST=/absolute/path/to/candidate.whl \
  python -m pytest -q tests/public_stable_polish_v6/test_public_stable_contract.py
```

The historical audit's 54 artifact-dependent failures are listed individually
in `tests/archive_qa_nodes.json`, with their required fixture locations. Two
additional presentation checks are in the same tier: a manuscript-figure
source-consistency check whose private generator is retained locally, and a
historical figure-table check whose original glob loop would otherwise perform
no assertions in a checkout without results. Their original assertions remain
intact. These checks read preserved scientific results or manuscript assets;
they do not regenerate them. The collection hook
only assigns the `archive_qa` marker; it does not skip tests. Without `-m`, pytest
still attempts both tiers. Run the archived checks from the original project
or a checkout supplied with the listed `results/` fixtures:

```bash
python -m pytest -q -m archive_qa
```

Missing historical fixtures are NOT RUN/MISSING FIXTURE (or a recorded failed
attempt), never a passing software regression. Do not rerun molecular dynamics
or the large benchmark merely to populate them. Some source-only research
helper tests use `scripts/` and development modules intentionally excluded from
the wheel; installed-product tests must instead import the installed public
package without a source-tree `PYTHONPATH`.

Candidate acceptance records must name the full source commit, exact wheel
SHA256, interpreter/import paths, test selection, command, and pass/fail/skip
counts. A source test, a CPU test, an optional-GPU test and an installed-wheel
test are separate scopes. No test count from a prior candidate is evidence for
the current one. Local acceptance does not establish a green remote CI run.

The release sdist supplies portable inputs and public regression files, but its
excluded research modules are intentionally absent. Use the full Git checkout
for the complete development suite; for installed-product acceptance from an
extracted sdist run `tests/fix_and_release` and `tests/m8_delivery` with the
installed package, from outside the development source tree.
